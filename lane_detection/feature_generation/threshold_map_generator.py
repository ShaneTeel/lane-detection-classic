import cv2
from numpy.typing import NDArray
from typing import Literal
from lane_detection.utils.constants import (
    THRESH_WHITE_LOWER, THRESH_WHITE_UPPER,
    THRESH_YELLOW_LOWER, THRESH_YELLOW_UPPER
)
   
class ThresholdMapGenerator():
    '''
    Description
    -----------
    Image preprocessing class that generates a feature map for use in lane line detection.

    Parameters
    ----------
    threshold : float, default 150.0
        Argument passed `cv2.threshold()`'s `thresh` argument. Pixel intensities >= threshold param will be set to 255.

    large_ksize : int, {11, 13, 15, 17, 19, 21}, default 15
        The kernel size used during the first morphological operation performed against the frame/image.
        The kernel is used to close large gaps between non-zero pixels.

    small_ksize : int, {3, 5, 7, 9, 11, 13, 15}, default 3
        The kernel size used during the second morphological operation performed against the frame/image.
        The kernel is used to close small gaps between non-zero pixels. It is also used to dilate the image.

    Public Methods
    --------------
    `.generate()`
    '''
    def __init__(self, large_ksize:Literal[11, 13, 15, 17, 19, 21] = 15, small_ksize:Literal[3, 5, 7, 9, 11, 13, 15] = 15):
        '''
        Parameters
        ----------
        threshold : float, default 150.0
            Argument passed `cv2.threshold()`'s `thresh` argument. Pixel intensities >= threshold param will be set to 255.

        large_ksize : int, {11, 13, 15, 17, 19, 21}, default 15
            The kernel size used during the first morphological operation performed against the frame/image.
            The kernel is used to close large gaps between non-zero pixels.

        small_ksize : int, {3, 5, 7, 9, 11, 13, 15}, default 3
            The kernel size used during the second morphological operation performed against the frame/image.
            The kernel is used to close small gaps between non-zero pixels. It is also used to dilate the image.            
        '''
        self.white_lower = THRESH_WHITE_LOWER
        self.white_upper = THRESH_WHITE_UPPER
        self.yellow_lower = THRESH_YELLOW_LOWER
        self.yellow_upper = THRESH_YELLOW_UPPER

        self.small_ksize = small_ksize
        self.large_ksize = large_ksize
        
    def generate(self, frame:NDArray):
        '''
        Description
        -----------
        Generates a threshold-based feature map. The frame/image is first converted from BGR to HSL format. 
        The L-channel is then passed to `cv2.threshold()`. The resulting binary image is then passed to `cv2.morphologyEx()`, twice:
        Once to close large gaps between non-zero pixels and a second to close small gaps between non-zero pixels. 
        Lastly, the morphed binary image is then dilated via `cv2.dilate()` to increase the size of the non-zero regions of the frame/image. 

        Parameters
        ----------
        frame : NDArray
            source frame/image that will be used to generate a vertical-based edge map.

        Returns
        -------
        binary image : NDArray
            result of `cv2.threshold()`.
        feature map : NDArray
            result of `cv2.morphologyEx()` + `cv2.dilate()`
        '''
        thresh = self._thresh_mask(frame, self.white_lower, self.white_upper, self.yellow_lower, self.yellow_upper)
        dense_mask = self._morph_frame(thresh, self.large_ksize, self.small_ksize)
        return thresh, dense_mask

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

    def _morph_frame(self, frame:NDArray, large_ksize:Literal[11, 13, 15, 17, 19, 21] = 15, small_ksize:Literal[3, 5, 7, 9, 11, 13, 15]=3):
        '''
        Description
        -----------
        Converts an image from BGR to grayscale, then applies `cv2.morphologyEx()` x2 and `cv2.dilate()` to a thresh mask, respectively. 

        Parameters
        ----------
        frame : NDArray
            binary frame/image that will be used to generate a feature map.
        
        large_ksize : int, {11, 13, 15, 17, 19, 21}, default 15
            The kernel size used during the first morphological operation performed against the frame/image.
            The kernel is used to close large gaps between non-zero pixels.

        small_ksize : int, {3, 5, 7, 9, 11, 13, 15}, default 3
            The kernel size used during the second morphological operation performed against the frame/image.
            The kernel is used to close small gaps between non-zero pixels. It is also used to dilate the image.
    
        Returns
        -------
        dense_mask : NDArray
            result of `cv2.morphologyEx()` x2 --> `cv2.dilate()`.
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (large_ksize, large_ksize))
        closed = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, morph_kernel)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_ksize, small_ksize))
        filled = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_small)

        dense_mask = cv2.dilate(filled, kernel_small, iterations=1)
        return dense_mask