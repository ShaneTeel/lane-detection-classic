import cv2
import io
import base64
from time import time
from .source import Reader

class Writer():
    def __init__(self, source:Reader):
        self.file_out_name = "../media/out/" + source.name + "-processed"
        self.ext = source.ext
        self.width = source.width
        self.height = source.height
        self.fps = source.fps
        self.writer = None
    
    def save_object(self, frame):
        self.writer.write(frame)

    # Gen file download link (app-specific)
    def get_download_link(self, file, file_name, text):
        buffered = io.BytesIO()
        file.save(buffered, format='mp4')
        file_str = base64.b64encode(buffered.getvalue()).decode()
        href = f"<a href='data:file/txt;base64,{file_str}' download='{file_name}'>{text}</a>"
        return href

    def _initialize_writer(self):
        self.file_out_name += self.ext
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.file_out_name, fourcc, self.fps, (self.width, self.height))