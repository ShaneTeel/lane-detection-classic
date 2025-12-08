    
import cv2
import numpy as np
from lane_detection.utils import get_logger

logger = get_logger(__name__)

class Illustrator:
    '''Superimposes shapes/lines on an image'''

    def __init__(self, stroke_color:tuple = (0, 0, 0), fill_color:tuple = (0, 255, 0)):
        
        self.stroke_color = self._hex_to_bgr(stroke_color)
        self.fill_color = self._hex_to_bgr(fill_color)
    
    def gen_composite(self, frame, lines, stroke:bool=True, fill:bool=True):
        if not stroke and not fill:
            raise ValueError("ERROR: One of `stroke` or `fill` must be `True`. Both cannot be `False`.")
        canvas = np.zeros_like(frame)
        valid_lines = []
        for line in lines:
            if line.size != 0:
                valid_lines.append(line)
            if len(valid_lines) == 0:
                logger.warning("No lines found, skipping")
                return cv2.addWeighted(frame, 0.8, canvas, 0.5, 0.0)
        if stroke:
            for line in valid_lines:
                self._draw_lines(canvas, [line])
        if fill:
            if len(valid_lines) != 2:
                logger.warning("Left or right lane lines not found, skipping fill")
                return cv2.addWeighted(frame, 0.8, canvas, 0.5, 0.0)
            self._draw_fill(valid_lines, canvas)
        return cv2.addWeighted(frame, 0.8, canvas, 0.5, 0.0)

    def _draw_lines(self, img, line):
        if line is None:
            logger.warning("No lines, skipping.")
            return
        else:
            cv2.polylines(img, line, isClosed=False, color=self.stroke_color, thickness=3, lineType=cv2.LINE_AA)

    def _draw_fill(self, lines, frame):
        if len(lines[0]) != len(lines[1]):
            return 
        poly = np.concatenate(lines, dtype=np.int32).reshape(1, -1, 2)
        cv2.fillPoly(img=frame, pts=poly, color=self.fill_color)

    def _draw_banner_text(self, frame, text):
        frame = self._channel_checker(frame)
        h, w = frame.shape[:2]
        banner_height = int(0.08 * h)
        cv2.rectangle(frame, (0, 0), (w, banner_height), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

        cv2.putText(frame, text, (int(w // 2) - 80, 10 + (banner_height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        return frame

    def _hex_to_bgr(self, color):
        if isinstance(color, tuple) and len(color) == 3:
            if len(color) == 3:
                return color
            else:
                return color[:3]
        
        if color.startswith("#"):
            hex_color = color[1:7]

            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)

            return (b, g, r)

    def _channel_checker(self, frame):
        if len(frame.shape) < 3:
            frame = cv2.merge([frame, frame, frame])
            return frame
        else:
            return frame