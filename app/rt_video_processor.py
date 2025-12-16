import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase
import threading
import queue
import logging

from lane_detection.detection import DetectionSystem
from lane_detection.utils import get_logger, setup_logging

logger = get_logger(__name__)

setup_logging(
    log_level=logging.DEBUG,
    log_to_file=False,
    console_output=True
)

class RTVideoProcessor(VideoProcessorBase):

    def __init__(self, file_path:str, roi:np.ndarray, configs:dict, view_style:str):

        # System configs
        self.file_path = file_path
        self.roi = roi
        self.configs = configs
        self.view_style = view_style
        self.system = DetectionSystem(self.file_path, self.roi, **self.configs)
        self.frame_names = self.system._configure_output(
            view_style=self.view_style, 
            file_out_name=None,
            fourcc=None,
            method="final",
            print_controls=False
        )
        h, w = self.system.studio.source.height, self.system.studio.source.width
        self.place_holder = np.zeros((h, w, 3), np.uint8)
        
        # Thread-Safe Queue and State
        self.result_queue = queue.Queue(maxsize=1)
        self.last_frame = None
        self._stop_event = threading.Event()

        # Background worker
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()

        logger.debug("Initialized Processor")

    def _run(self):
        '''Background processor that allows processing to occur at CPU allowable speed'''
        logger.debug("Worker Thread Starting")

        while not self._stop_event.is_set():

            ret, frame = self.system.studio.return_frame()

            if not ret:
                logger.debug("Reached end of file")
                break

            try:
                thresh, feature_map = self.system.generator.generate(frame)
                masked = self.system.mask.inverse_mask(feature_map)
                lane_pts = self.system.selector.select(masked)

                lane_lines = []

                for i in range(2):
                    pts = lane_pts[i]
                    if pts.size == 0:
                        continue
                    
                    if i == 0:
                        detector = self.system.detector1
                        evaluator = self.system.evaluator1
                    else:
                        detector = self.system.detector2
                        evaluator = self.system.evaluator2

                    line = self.system.detect_line(pts, detector)
                    if i == 0:
                        lane_lines.append(np.flipud(line))
                    else:
                        lane_lines.append(line)
                    self.system.evaluate_model(detector, evaluator)
                        
                frame_lst = [frame, thresh, feature_map, masked]
                final = self.system.studio.gen_view(frame_lst, self.frame_names, lane_lines, self.view_style)

                if not self.result_queue.empty():
                    try: 
                        self.result_queue.get_nowait()
                    except queue.Empty: 
                        pass
                self.result_queue.put(final, block=False)
            
            except Exception:
                self.result_queue.put(frame)


    def recv(self):
        '''Item grabber; grabs newest item from queue'''
        try:
            new_frame = self.result_queue.get(timeout=0.1)
        except queue.Empty:
            if self.last_frame is None:
                return self.place_holder
            else:
                return self.last_frame
        self.last_frame = new_frame
        return self.last_frame
    
    def on_ended(self):
        '''Clean-up when streamer ends'''
        self._stop_event.set()
        self.worker_thread.join(timeout=1.0)