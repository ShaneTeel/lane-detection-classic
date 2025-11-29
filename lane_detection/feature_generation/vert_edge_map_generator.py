import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Literal
   
class VerticalEdgeMapGenerator():
    '''
    Description
    -----------
    Image preprocessing class that generates a feature map for use in lane line detection.

    Parameters
    ----------
    threshold : float, default 150.0
        Argument passed `cv2.threshold()`'s `thresh` argument. Pixel intensities >= threshold param will be set to 255.
        
    ksize : int, {3, 5, 7, 9, 11, 15}, default 3
        The kernel size passed to both `cv2.GaussianBlur()` and `cv2.Sobel()`.

    Public Methods
    --------------
    `.generate()`
    '''
        
    def __init__(self, threshold:float=150.0, ksize:Literal[3, 5, 7, 9, 11, 13, 15]=3) -> None:
        '''
        Parameters
        ----------
        threshold : float, default 150.0
            Argument passed `cv2.threshold()`'s `thresh` argument. Pixel intensities >= threshold param will be set to 255.
            
        ksize : int, {3, 5, 7, 9, 11, 13, 15}, default 3
            The kernel size passed to both `cv2.GaussianBlur()` and `cv2.Sobel()`.
        '''
        self.threshold = threshold
        self.ksize = ksize
        
    def generate(self, frame:NDArray) -> tuple[NDArray, NDArray]:
        '''
        Description
        -----------
        Generates a vertical-based edge map. The frame/image is first converted from BGR to HSL format. 
        The L-channel is then passed to `cv2.threshold()`. The resulting binary image is then passed to `cv2.GaussianBlur()` 
        and `cv2.Sobel()`, respectively. Lastly, edge map pixel intensities are normalized back to a 0-255 pixel intensity range. 

        Parameters
        ----------
        frame : NDArray
            source frame/image that will be used to generate a vertical-based edge map.

        Returns
        -------
        binary image : NDArray
            result of `cv2.threshold()`.
        edge map : NDArray
            result of `cv2.Sobel()` + pixel normalization
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        thresh = self._threshold(frame, self.threshold)
        edge_map = self._detect_edges(thresh, self.ksize)
        return thresh, edge_map
    
    def _threshold(self, frame:NDArray, threshold:float=150.0):
        '''
        Description
        -----------
        Performs a BGR to HSL color conversion on a frame/image, 
        then generates a binary image from the L-channel using `cv2.threshold()`. 

        Parameters
        ----------
        frame : NDArray
            source frame/image that will be used to generate a binary image.

        threshold : float
            Value used to determine whether a pixel intensity within the frame/image will be converted to 0 or 255 
            
        Returns
        -------
        binary image : NDArray
            result of `cv2.threshold()`.
        '''
        _, L_thresh = cv2.threshold(frame[:, :, 1], threshold, 255.0, cv2.THRESH_BINARY)
        _, S_thresh = cv2.threshold(frame[:, :, 2], threshold, 255.0, cv2.THRESH_BINARY)
        return cv2.bitwise_or(L_thresh, S_thresh)

    def _detect_edges(self, frame:NDArray, ksize:Literal[3, 5, 7, 9, 11, 13, 15]=3):
        '''
        Description
        -----------
        Applies `cv2.GaussianBlur()` and `cv2.Sobel()` to a binary image, respectively. 
        The edge map produced by `cv2.Sobel() is then normalized back to a 0-255 pixel intensity range. 

        Parameters
        ----------
        frame : NDArray
            binary frame/image that will be used to generate an edge map.
        
        ksize : int, {3, 5, 7, 9, 11, 15}, default 3
            height/width of kernel used to convolve image with `cv2.GaussianBlur()` and `cv2.Sobel()`.
            
        Returns
        -------
        edge map : NDArray
            result of `cv2.GaussianBlur()` --> `cv2.Sobel()`.
        '''
        kernel = (ksize, ksize)
        frame = cv2.GaussianBlur(frame, kernel, 0)

        sobel_X = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_X = np.absolute(sobel_X)

        return np.uint8(sobel_X / np.max(sobel_X) * 255)