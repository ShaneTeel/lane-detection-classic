import cv2
import imageio

from lane_detection.utils import get_logger

logger = get_logger(__name__)

class StreamlitWriter():

    def __init__(self, file_out_path, fps):
        self.file_out_path = file_out_path
        self.fps = fps
        self.writer = None
        self._initialize_writer()

        logger.debug("Initialized Writer")
    
    def write_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.writer.append_data(frame_rgb)

    def _initialize_writer(self):
        self.writer = imageio.get_writer(
            self.file_out_path,
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8
        )
    
    def release(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None