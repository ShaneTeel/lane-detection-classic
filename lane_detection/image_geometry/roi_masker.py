import cv2
import numpy as np
from numpy.typing import NDArray

class ROIMasker():
    '''
    Description
    -----------
    Image preprocessing class that uses a Region of Interest (ROI) to apply inverse masking to a frame/image.
    The mask is intended to reduce noise in preparation for feature extraction / selection.

    Parameters
    ----------
    roi : NDArray
        Region of interest that will be perserved after masking is performed. 

    Public Methods
    --------------
    `.inverse_mask()`
    `.get_src_pts()`
    `.src_y_min_max()`
    '''
    def __init__(self, roi:NDArray):
        '''
        Parameters
        ----------
        roi : NDArray
            Region of interest that will be perserved after masking is performed.
        
        '''
        self.src_pts = self._roi_validation(roi)
        self.src_y_min, self.src_y_max = self._get_min_max(self.src_pts, 1)

    def inverse_mask(self, frame:NDArray):
        '''
        Description
        -----------
        Applies inverse masking to a frame/image.
            1. Generates a blank canvas of the same shape as the frame/image passed.
            2. Modifies the canvas with `cv2.fillPoly()` using the ROI passed during object initialization.
            3. Performs `cv2.bitwise_and()` on the frame/image and the modified canvas to retain non-zero pixel values
            only in the area represented by the ROI.

        Parameters
        ----------
        frame : NDArray
            Frame/image that the mask is applied to. 

        Public Methods
        --------------
        masked : NDArray
            Frame/image with non-zero pixels only appearing in the area represented by the ROI.
        '''
        canvas = np.zeros_like(frame)
        if len(frame.shape) > 2:
            num_channels = frame.shape[2]
            roi_color = (255,) * num_channels
        else:
            roi_color = 255

        cv2.fillPoly(img=canvas, pts=self.src_pts.astype(np.int32), color=roi_color)
        masked = cv2.bitwise_and(src1=frame, src2=canvas)
        return masked
    
    def _roi_validation(self, roi:NDArray):
        '''
        Description
        -----------
        Private method called upon during object initialization to validate the shape, enforce a point order, and convert the roi dtype to `np.float32`.

        The ROI point order and shape is as follows:
            [[
            [Bottom Left]
            [Bottom Right]
            [Top Right]
            [Top Left]
            ]]

        Parameters
        ----------
        roi : NDArray
            Proposed region of interest.

        Returns
        --------------
        src_pts : NDArray
            ROI validated, reshaped, and type casted.
        '''
        if roi.shape != (1, 4, 2):
            try:
                roi = roi.reshape(1, 4, 2)
            except Exception as e:
                raise ValueError(e)

        points = roi[0]
        bottom = points[points[:, 1] > points[:, 1].mean()]
        top = points[points[:, 1] <= points[:, 1].mean()]

        bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]
        top_left, top_right = top[np.argsort(top[:, 0])]

        return np.array([[bottom_left],
                         [bottom_right],
                         [top_right],
                         [top_left]], 
                         dtype=np.float32).reshape(1, 4, 2)
    
    def _get_min_max(self, roi:NDArray, col:int = 1):
        '''
        Description
        -----------
        Identifies the min and max values of the ROI along a specific column.
        Computed at initialization. 

        Candidate `start` / `stop` points passed to the `SingleLaneLineDetector().fit_predict()` 
        during lane line detection.

        Parameters
        ----------
        roi : NDArray
            array of shape (1, 4, 2).

        Returns
        --------------
        min : np.float32
            The min value of the ROI for a specific column
        max : np.float32
            The max value of the ROI for a specific column
        '''
        return roi[0, :, col].min(), roi[0, :, col].max()
    
    def get_src_pts(self):
        '''
        Description
        -----------
        Public method that returns the validated roi for use by `BEVProjector()` 
        when computing the `dst_pts` needed for `cv2.perspectiveTransform()`.

        Returns
        --------------
        src_pts : NDArray
            Array of shape=(1, 4, 2), type=np.float32, and ordered as bottom left, bottom right, top right, and top left.
        '''
        return self.src_pts

    def src_y_min_max(self):
        '''
        Description
        -----------
        Public facing method that returns the already computed min and max values of the ROI's y-coordinates. 

        Candidate `start` / `stop` points passed to the `SingleLaneLineDetector().fit_predict()` 
        during lane line detection.

        Returns
        --------------
        min : np.float32
            The min value of the ROI for a specific column
        max : np.float32
            The max value of the ROI for a specific column
        '''
        return self.src_y_min, self.src_y_max