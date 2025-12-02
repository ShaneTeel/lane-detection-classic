import cv2
import numpy as np
from numpy.typing import NDArray

class Render:

    def render_mosaic(self, frames:list, max_width:int):

        top = self._render_diptych(frames[:2])
        bottom = self._render_diptych(frames[2:])

        mosaic = np.vstack([top, bottom])
        h, w = mosaic.shape[:2]
        cv2.line(mosaic, (0, h // 2), (w, h // 2), (0, 255, 255), 1, cv2.LINE_AA)
        return self._resize_frame(mosaic, max_width)

    def render_inset(self, composite:np.ndarray, frames:list):
        triptych = self._render_triptych(frames)
        _, cw = composite.shape[:2]
        
        triptych = self._resize_frame(triptych, cw)
        th, _ = triptych.shape[:2]
    
        composite[0:th, :] = triptych
        return composite

    def _render_diptych(self, frames:list):
        diptych = np.hstack(frames)
        h, w = diptych.shape[:2]
        cv2.line(diptych, (w // 2, 0), (w // 2, h), (0, 255, 255), 1, cv2.LINE_AA)
        return diptych
    
    def _render_triptych(self, frames:list):
        triptych = np.hstack(frames)
        h, w = triptych.shape[:2]
        
        cv2.line(triptych, (w // 3, 0), (w // 3, h), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(triptych, (w - (w // 3), 0), (w - (w // 3), h), (0, 255, 255), 1, cv2.LINE_AA)
        return triptych
    
    def _resize_frame(self, frame:NDArray, max_width:int):
        h, w = frame.shape[:2]
        if w > max_width:
            ratio = max_width / w
            w = max_width
            h = int(ratio * h)
            return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        return frame