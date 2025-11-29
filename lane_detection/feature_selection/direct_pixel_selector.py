import numpy as np
from numpy.typing import NDArray
    
class DirectPixelSelector():
    '''
    Description
    -----------
    Image preprocessing class that selects features from a feature map using a holistic pixel extraction approach for use in lane line detection.

    Parameters
    ----------
    n_std : float, default 2.0
        Used to filter pixel coordinates by their x-value; 
        pixel coordinates with an x-values beyond `n_std` of the median-x are removed 
        from the array of pixel coordinates for that proposed lane line.

    Public Methods
    --------------
    `.select()`
    '''
    def __init__(self, n_std:float=2.0):
        '''
        Parameters
        ----------
        n_std : float, default 2.0
            Used to filter pixel coordinates by their x-value; 
            pixel coordinates with an x-values beyond `n_std` of the median-x are removed 
            from the array of pixel coordinates for that proposed lane line.
        '''
        self.n_std = n_std
        
    def select(self, feature_map:NDArray):
        '''
        Description
        -----------
        Extracts non-zero pixel coordinates from a feature map and classifies them as 'left' or 'right'
        based on x-value. 

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
        pts = self._point_extraction(feature_map)
        lane_pts = self._point_splitting(pts, x_mid)
        lane_pts = self._X_point_filtering(lane_pts, self.n_std)
        return lane_pts
    
    def _point_extraction(self, feature_map:NDArray):
        '''
        Description
        -----------
        Extracts non-zero pixel coordinates from a feature map.

        Parameters
        ----------
        feature_map : NDArray
            feature map (binary image or edge map) containing non-zero pixels.

        Returns
        -------
        pts : NDArray
            an array of pts representing image coordinates for non-zero pixels extracted from feature map.
        '''
        pts = np.where(feature_map != 0)
        if pts is None:
            return np.array([])
        return np.column_stack((pts[1], pts[0]))
    
    def _point_splitting(self, pts:NDArray, x_mid:int):
        '''
        Description
        -----------
        Splits points extracted from `._point_extraction()` based on x_mid value.

        Parameters
        ----------
        pts : NDArray
            feature map (binary image or edge map) containing non-zero pixels.
        x_mid : int
            Demarcation point for `pts` x-values. Those <= `x_mid` are classified as a seperate line that those with x-values >= `x_mid`

        Returns
        -------
        classified : list
            list of NDArrays, with each NDArray representing a distinct lane line from the feature map.
        '''
        if len(pts) == 0:
            return [np.array([]), np.array([])]

        left = pts[pts[:, 0] < x_mid]
        right = pts[pts[:, 0] >= x_mid]
        classified = [left, right]
        return classified

    def _X_point_filtering(self, lanes:list, n_std:float=2.0):
        '''
        Description
        -----------
        Filters points classified by `._point_splitting()` based on `n_std` value.

        Parameters
        ----------
        lanes : list
            list of NDArrays, with each NDArray representing a distinct lane line from the feature map.
        n_std : int
            Used to filter pixel coordinates by their x-value; 
            pixel coordinates with an x-values beyond `n_std` of the median-x are removed 
            from the array of pixel coordinates for that proposed lane line.

        Returns
        -------
        filtered : list
            list of NDArrays, with each NDArray representing a filtered lane line from the feature map.
        '''
        filtered = []
        for lane in lanes:
            if lane is None:
                filtered.append(lane)
            else:
                X = lane[:, 0]

                X_center = np.median(X)
                X_std = np.std(X)
                X_mask = np.abs(X - X_center) < (n_std * X_std)

                filtered.append(lane[X_mask])

        return filtered
