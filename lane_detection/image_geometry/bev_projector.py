import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Literal


class BEVProjector():
    '''
    Description
    -----------
    Image preprocessing class that performs homography on an array of points that represent pixel coordinates in image space.
    The points are projected into BEV (bird's eye view) space so that the points representing lane lines appear parallel. 
    This is done to improve fitting. 

    Parameters
    ----------
    src_pts : NDArray
        Region of interest that will be perserved after masking is performed.
    
    forward_range : float, default = 40.0
        The forward range of visibility, in meters; how far ahead to look (length of lane lines)
    
    lateral_range : float, default = 7.0
        The lateral range of visibility, in meters; how far left/right to look (number of lane lines)

    resolution : float, default = 0.05
        Meters per pixel in BEV space (0.05m = 5cm per pixel)

    Public Methods
    --------------
    `.project()`
    `.dst_y_min_max()`
    '''
    def __init__(self, src_pts:NDArray, forward_range: float = 40.0, lateral_range: float = 7.0, resolution: float = 0.05):
        '''
        Parameters
        ----------
        src_pts : NDArray
            Region of interest that will be perserved after masking is performed.
        
        forward_range : float, default = 40.0
            The forward range of visibility, in meters; how far ahead to look (length of lane lines)
        
        lateral_range : float, default = 7.0
            The lateral range of visibility, in meters; how far left/right to look (number of lane lines)

        resolution : float, default = 0.05
            Meters per pixel in BEV space (0.05m = 5cm per pixel)

        At Initialization
        -----------------
        BEV dimensions are computed:
            - height = int(self.forward_range / self.resolution)
            - width = int(self.forward_range / self.resolution)
        
        BEV destination points are computed

        Homography matrix / Inverse Homography matrix are computed
        '''
        self.src_pts = src_pts
        self.forward_range = forward_range
        self.lateral_range = lateral_range
        self.resolution = resolution
        self.bev_height = int(self.forward_range / self.resolution)
        self.bev_width = int(self.lateral_range / self.resolution)
        self.dst_pts = self._calc_dst_pts()
        self.H = self._calc_H_mat()
        self.H_I = np.linalg.inv(self.H)

    def project(self, pts:NDArray, direction:Literal["forward", "backward"]):
        """
        Transform points between camera and bird's-eye view coordinate systems.
        
        Parameters
        ----------
        pts : NDArray, shape (n_points, 2)
            Points to transform in (x, y) pixel coordinates
        direction : {"forward", "backward"}
            "forward" = camera -> BEV, "backward" = BEV -> camera
            
        Returns
        -------
        transformed_pts : NDArray, shape (n_points, 2)
            Points in target coordinate system
            
        Notes
        -----
        Uses homography transformation via `cv2.perspectiveTransform()`.
        Forward transform makes parallel lanes appear truly parallel.
        Backward transform projects detected lanes back to camera view.
        """
        if len(pts) == 0:
            return pts
        
        pts = np.array([pts], dtype=np.float32)
        if pts.ndim == 2:
            pts = pts.reshape(1, -1, 2)
        
        m = self.H if direction == "forward" else self.H_I

        pts = cv2.perspectiveTransform(pts, m)
        return pts.reshape(-1, 2).astype(np.int32)
    
    def _calc_H_mat(self):
        """
        Compute homography matrix using Direct Linear Transformation (DLT).
        
        Solves for 3x3 homography mapping source ROI trapezoid to rectangular
        BEV destination. Constructs 9x9 system of linear equations from
        point correspondences.
        
        Returns
        -------
        H : NDArray, shape (3, 3)
            Homography matrix normalized so H[2,2] = 1
            
        Notes
        -----
        Each point correspondence contributes 2 equations to the system.
        With 4 point pairs, we get 8 equations for 8 unknowns (9th fixed to 1).
        """
        A = np.zeros((9, 9), dtype=np.float32)
        A[8, 8] = 1

        xi_yi = self.src_pts[:, :, :].reshape(-1, 2)
        ui_vi = self.dst_pts[:, :, :].reshape(-1, 2)
        DOF = list(range(0, 8, 2))

        for dof, (xi, yi), (ui, vi) in zip(DOF, xi_yi, ui_vi):
            A[dof,:] = np.array([-xi, -yi, -1, 0, 0, 0, xi * ui, yi * ui, ui])
            A[dof+1,:] = np.array([0, 0, 0, -xi, -yi, -1, xi * vi, yi * vi, vi])

        b = np.array([0]*8 + [1], dtype=np.float32)

        H = np.linalg.solve(A, b).reshape(3, 3)

        return H / H[2, 2]
            
    def _calc_dst_pts(self):
        bev = np.array([[[0, self.bev_height],
                         [self.bev_width, self.bev_height],
                         [self.bev_width, 0],
                         [0, 0]]],
                         dtype=np.float32)
        return bev.reshape(1, 4, 2)
    
    def dst_y_min_max(self):
        return 0, self.bev_height