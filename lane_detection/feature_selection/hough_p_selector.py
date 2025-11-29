import cv2
import numpy as np
from numpy.typing import NDArray

class HoughPSelector():
    '''
    Description
    -----------
    Image preprocessing class that selects features from a feature map using `cv2.HoughLinesP()` for use in lane line detection.
    By default, the arguments for rho and theta are set to 1 pixel and 1 degree (in radians), respectively.

    Parameters
    ----------
    All params passed to `cv2.HoughLinesP()`; see OpenCV's `Hough Line Transform`_ documentation page for further information.

    .. _Hough Line Transform: https://docs.opencv.org/4.x/d3/de6/tutorial_js_houghlines.html
 
    min_votes : int, default 50
        The minimum number of votes needed in the accumulator 
        array for a candidate line segment to be considered a valid line. 

    min_length : int, default 15
        Minimum line length for a candidate line to be considered a valid line.

    max_gap : int, default 10
        Minimum distance between points on along the same candidate line to link the points as part of the same line.

    Public Methods
    --------------
    `.select()`
    '''
    def __init__(self, min_votes:int=50, min_length:int=15, max_gap:int=10):
        '''
        Parameters
        ----------
        All params passed to `cv2.HoughLinesP()`; see OpenCV's `Hough Line Transform`_ documentation page for further information.

        .. _Hough Line Transform: https://docs.opencv.org/4.x/d3/de6/tutorial_js_houghlines.html
 
        min_votes : int, default 50
            The minimum number of votes needed in the accumulator 
            array for a candidate line segment to be considered a valid line. 

        min_length : int, default 15
            Minimum line length for a candidate line to be considered a valid line.

        max_gap : int, default 10
            Minimum distance between points on along the same candidate line to link the points as part of the same line.
        '''
        self.min_votes = min_votes
        self.min_length = min_length
        self.max_gap = max_gap

    def select(self, feature_map:NDArray):
        '''
        Description
        -----------
        Randomly samples non-zero pixel coordinates from a feature map using `cv2.HoughLinesP()` and classifies them as 'left' or 'right'
        based on x-value. By default, the arguments for rho and theta are set to 1 pixel and 1 degree (in radians), respectively.

        Parameters
        ----------
        feature_map : NDArray
            feature map (binary image or edge map) containing non-zero pixels.

        Returns
        -------
        lane_pts : NDArray
            a list of pts representing distinct lane lines.
        '''
        x_mid = feature_map.shape[1] // 2
        lines = self._point_extraction(feature_map, self.min_votes, self.min_length, self.max_gap)
        if lines is None:
            return [np.array([]).reshape(-1, 2), np.array([]).reshape(-1, 2)]
        kps = self._point_splitting(lines, x_mid)
        return kps

    def _point_extraction(self, feature_map:NDArray, min_votes:int, min_length:int, max_gap:int):
        '''
        Description
        -----------
        Implements `cv2.HoughLinesP()` to randomly sample non-zero pixels form a binary image
        to detect line segments.

        Parameters
        ----------
        feature_map : NDArray
            feature map (binary image or edge map) containing non-zero pixels.

        Returns
        -------
        lines : NDArray
            an array of pts representing line segments for non-zero pixels extracted from feature map.
        '''
        lines = cv2.HoughLinesP(feature_map, 1, np.pi / 180, min_votes, min_length, max_gap)
        if lines is None:
            return None
        return lines

    def _point_splitting(self, lines:list, x_mid:int):
        '''
        Description
        -----------
        Splits line segments extracted from `._point_extraction()` based on `x_mid` value.

        Parameters
        ----------
        lines : NDArray
            line segments generated from `cv2.HoughLinesP()` / `._point_extraction()`.
        x_mid : int
            Demarcation point for `pts` x-values. Those <= `x_mid` are classified as a seperate line that those with x-values >= `x_mid`

        Returns
        -------
        classified : list
            list of NDArrays, with each NDArray representing a distinct lane line from the feature map.
        '''
        left = []
        right = []
        for line in lines:
            X1, y1, X2, y2 = line.flatten()
            if X1 <= x_mid and X2 <= x_mid:
                left.append([X1, y1])
                left.append([X2, y2])
            if X1 >= x_mid and X2 >= x_mid:
                right.append([X1, y1])
                right.append([X2, y2])
        return [np.array(left).reshape(-1, 2), np.array(right).reshape(-1, 2)]