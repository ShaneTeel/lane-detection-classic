import cv2
from numpy.typing import NDArray
from typing import Literal
   
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
    def __init__(self, threshold:float=150.0, large_ksize:Literal[11, 13, 15, 17, 19, 21] = 15, small_ksize:Literal[3, 5, 7, 9, 11, 13, 15] = 15):
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
        self.threshold = threshold
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        thresh = self._threshold(frame, self.threshold)
        dense_mask = self._morph_frame(thresh, self.large_ksize, self.small_ksize)
        return thresh, dense_mask
    
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

    def _morph_frame(self, frame:NDArray, large_ksize:Literal[11, 13, 15, 17, 19, 21] = 15, small_ksize:Literal[3, 5, 7, 9, 11, 13, 15]=3):
        '''
        Description
        -----------
        Applies `cv2.morphologyEx()` x2 and `cv2.dilate()` to a binary image, respectively. 

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
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (large_ksize, large_ksize))
        closed = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, morph_kernel)

        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_ksize, small_ksize))
        filled = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_small)

        dense_mask = cv2.dilate(filled, kernel_small, iterations=1)
        return dense_mask