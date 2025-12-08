import cv2
from lane_detection.utils import get_logger

logger = get_logger(__name__)

class Custodian():

    def __init__(self, source, writer):
        self.source = source
        self.writer = writer

    def _clean_up(self):
        if self.source.cap is not None:
            self.source.cap.release()
            self.source.cap = None
        if self.writer.writer is not None:
            self.writer.writer.release()
            self.writer.writer = None

    def __del__(self):
        self._clean_up()
        cv2.destroyAllWindows()
        logger.info("Clean up complete; resources destroyed.")
    