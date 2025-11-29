import cv2
import os

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
        if self.source.temp_file_path is not None:
            os.unlink(self.source.temp_file_path)
        cv2.destroyAllWindows()