import cv2
import os
import tempfile
from lane_detection.utils import get_logger

logger = get_logger(__name__)

class Reader():

    def __init__(self, source):
        self.source = source
        self.source_type = None
        self.name = None
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

        logger.info(f"Successfully read image {self.name}: {self.source} ({self.height}x{self.width})")

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
            
            logger.info(f"Successfully loaded video: {self.source} ({self.width}x{self.height}, {self.fps} FPS, {self.frame_count} frames)")

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
            
            logger.info(f"Successfully opened camera: {self.source} ({self.width}x{self.height}, {self.fps:.1f} FPS)")