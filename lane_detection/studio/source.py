import cv2
import os
import tempfile
import shutil
from streamlit.runtime.uploaded_file_manager import UploadedFile

class Reader():

    def __init__(self, source):
        self.source = source
        self.source_type = None
        self.name = None
        self.temp_file_path = None
        self.ext = None
        self.cap = None
        self.width = None
        self.height = None
        self.fps = None
        self.frame_count = None

        self._initialize_source()
    
    def _initialize_source(self):
        if isinstance(self.source, int):
            self._initialize_camera()
        elif isinstance(self.source, str):
            if self._is_image_file():
                self._initialize_image()
            else:
                self._initialize_video()
        elif isinstance(self.source, UploadedFile):
            self._initialize_file_upload()
        else:
            raise ValueError(f"Invalid source type: {type(self.source)}. Expected str or int.")

    def _is_image_file(self):
        '''ADD'''
        valid_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        _, ext = os.path.splitext(os.path.basename(self.source))
        return ext in valid_suffix
    
    def _initialize_image(self):
        '''ADD'''
        self.source_type = 'image'
        self.image = cv2.imread(self.source)
        if self.image is None:
            raise ValueError(f"Error: Failed to read image from {self.source}")
        
        self.height, self.width = self.image.shape[:2]
        
        _, self.ext = os.path.splitext(os.path.basename(self.source))
        if self.name is None:
            self.name, _ = os.path.splitext(os.path.basename(self.source))

        print(f"Successfully read image {self.name}: {self.source} ({self.height}x{self.width})")

    def _initialize_video(self):
        '''ADD'''
        self.source_type = 'video'
        self.cap = cv2.VideoCapture(self.source)


        if not self.cap.isOpened():
            raise ValueError(f"Error: Failed to open video file {self.source}")
        
        else:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            _, self.ext = os.path.splitext(os.path.basename(self.source))
            if self.name is None:
                self.name, _ = os.path.splitext(os.path.basename(self.source))
            
            print(f"Successfully loaded video: {self.source} ({self.width}x{self.height}, {self.fps} FPS, {self.frame_count} frames)")

    def _initialize_camera(self):
        '''ADD'''
        self.source_type = 'camera'
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Failed to open camera file {self.source}")
        else:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            if self.name is None:
                self.name = 'camera_' + str(self.source)
            
            print(f"Successfully opened camera: {self.source} ({self.width}x{self.height}, {self.fps:.1f} FPS)")

    def _initialize_file_upload(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            try:
                shutil.copyfileobj(self.source.file, temp_file)
                self.temp_file_path = temp_file.name
            finally:
                self.source.file.close()
        self.cap = cv2.VideoCapture(self.temp_file_path)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Failed to open {self.name} video.")

        else:       
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.name, self.ext = os.path.splitext(os.path.basename(self.source.filename))