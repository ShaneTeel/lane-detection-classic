import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Literal
from lane_detection.utils.constants import (
    THRESH_WHITE_LOWER, THRESH_WHITE_UPPER,
    THRESH_YELLOW_LOWER, THRESH_YELLOW_UPPER
)
   
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
        
    def __init__(self, ksize:Literal[3, 5, 7, 9, 11, 13, 15]=3) -> None:
        '''
        Parameters
        ----------
        threshold : float, default 150.0
            Argument passed `cv2.threshold()`'s `thresh` argument. Pixel intensities >= threshold param will be set to 255.
            
        ksize : int, {3, 5, 7, 9, 11, 13, 15}, default 3
            The kernel size passed to both `cv2.GaussianBlur()` and `cv2.Sobel()`.
        '''
        self.white_lower = THRESH_WHITE_LOWER
        self.white_upper = THRESH_WHITE_UPPER
        self.yellow_lower = THRESH_YELLOW_LOWER
        self.yellow_upper = THRESH_YELLOW_UPPER
        
        self.ksize = ksize
        
    def generate(self, frame:NDArray) -> tuple[NDArray, NDArray]:
        '''
        Description
        -----------
        Generates a vertical-based edge map. The frame/image is first converted from BGR to HSL format. 
        Two thresh masks are created, targeting white and yellow pixels. The thresh masks are then merged and applied to the original image. 
        The HSL-Thresh masked image is then passed to `cv2.GaussianBlur()` and `cv2.Sobel()`, respectively. 
        Lastly, edge map pixel intensities are normalized back to a 0-255 pixel intensity range. 

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
        thresh = self._thresh_mask(frame, self.white_lower, self.white_upper, self.yellow_lower, self.yellow_upper)
        edge_map = self._detect_edges(thresh, self.ksize)
        return thresh, edge_map
    
    def _thresh_mask(self, frame:NDArray, white_lower:NDArray, white_upper:NDArray, yellow_lower:NDArray, yellow_upper:NDArray):
        '''
        Description
        -----------
        Performs a BGR to HSL color conversion on a frame/image, 
        then generates a masks from the L- and S-channels using `cv2.inRange()`. The two masks are merged via a `cv2.bitwise_or()`.
        Lastly, the merged HLS-mask is applied to the original frame with `cv2.bitwise_and()`. 

        Parameters
        ----------
        frame : NDArray
            source frame/image that will be used to generate the mask.

        white_lower : NDArray
            Vector of size == 3 that represents the lower bands of Hue, Lightness, and Saturation used to detect the "white" portions of the image. Passed as `lowerb` argument to `cv2.inRange()`. 
        white_upper : NDArray
            Vector of size == 3 that represents the upper bands of Hue, Lightness, and Saturation used to detect the "white" portions of the image. Passed as `upperb` argument to `cv2.inRange()`. 
        white_lower : NDArray
            Vector of size == 3 that represents the lower bands of Hue, Lightness, and Saturation used to detect the "yellow" portions of the image. Passed as `lowerb` argument to `cv2.inRange()`. 
        white_lower : NDArray
            Vector of size == 3 that represents the upper bands of Hue, Lightness, and Saturation used to detect the "yellow" portions of the image. Passed as `upperb` argument to `cv2.inRange()`. 
            
        Returns
        -------
        HSL-Thresh Mask : NDArray

        '''
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, white_lower, white_upper)
        yellow = cv2.inRange(hls, yellow_lower, yellow_upper)
        mask = cv2.bitwise_or(white, yellow)
        return cv2.bitwise_and(frame, frame, mask=mask)

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = (ksize, ksize)
        frame = cv2.GaussianBlur(frame, kernel, 0)

        sobel_X = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=ksize)
        sobel_X = np.absolute(sobel_X)

        return np.uint8(sobel_X / np.max(sobel_X) * 255)