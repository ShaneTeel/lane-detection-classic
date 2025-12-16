import numpy as np

from lane_detection.detection import DetectionSystem
from lane_detection.utils import get_logger

logger = get_logger(__name__)

class StreamlitDetector():

    def __init__(self, file_path:str, roi:np.ndarray, configs:dict, view_style:str):

        # System configs
        self.system = DetectionSystem(file_path, roi, **configs)
        self.view_style = view_style
        self.frame_names = self.system._configure_output(
            view_style=view_style, 
            file_out_name=None,
            fourcc=None,
            method="final",
            print_controls=False
        )

        logger.debug("Initialized Detector")

    def process_frame(self, frame: np.ndarray):
        '''Frame processor that orchestrates lane line detection'''

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
        return self.system.studio.gen_view(frame_lst, self.frame_names, lane_lines, self.view_style)
    
    def return_frame(self):
        return self.system.studio.return_frame()
    
    def write_frame(self, frame):
        self.system.studio.write_frames(frame)