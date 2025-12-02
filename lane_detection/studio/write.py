import cv2
from .source import Reader

class Writer():
    def __init__(self, source:Reader):
        self.ext = source.ext
        self.width = source.width
        self.height = source.height
        self.fps = source.fps
        self.writer = None
    
    def save_object(self, frame):
        self.writer.write(frame)

    def _initialize_writer(self, file_out_name):
        file_out_name += self.ext
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(file_out_name, fourcc, self.fps, (self.width, self.height))