import numpy as np
from numpy.typing import NDArray
from typing import Literal
    
class KalmanFilter():
    """
    Apply Kalman filtering to smooth coefficient estimates.
    
    Parameters
    ----------
    coeffs : NDArray, shape (n_coeffs,)
        New measurement (polynomial coefficients from current frame)
    y_range : float
        Vertical span of fitted points (affects measurement uncertainty)
    n_inliers : int
        Number of inlier points in fit
    inlier_ratio : float
        Fraction of points classified as inliers (0-1)
        
    Returns
    -------
    filtered_coeffs : NDArray, shape (n_coeffs,)
        Smoothed coefficient estimates
        
    Notes
    -----
    Prediction step: propagate state forward using motion model
    Update step: incorporate new measurement with adaptive noise
    Poor fits (low inlier_ratio) receive high measurement noise → trust prediction more
    """
    def __init__(self, fps:int, coeffs:NDArray, P_primer:float=0.5, process_noise:Literal["low", "medium", "high"]="low"):
        '''
        Parameters
        ----------
        coeffs : NDArray, shape (n_coeffs,)
            New measurement (polynomial coefficients from current frame)
        y_range : float
            Vertical span of fitted points (affects measurement uncertainty)
        n_inliers : int
            Number of inlier points in fit
        inlier_ratio : float
            Fraction of points classified as inliers (0-1)
        '''
        if fps <= 0:
            raise ValueError(f"ERROR: 'fps' must be positive, non-zero; got {fps}")
        if len(coeffs) < 2:
            raise ValueError(f"Need at least 2 coeffs, got {len(coeffs)}")

        self.dt = 1 / fps
        self.x = self._initialize_current_state(coeffs)
        self.poly_size = len(coeffs)
        self.P_primer = P_primer
        self.P = self._initialize_P(P_primer)
        self.I = np.eye(len(self.x))
        self.Q = self._initialize_Q(process_noise)
        self.F, self.H = self._initialize_F_H()
        self.F_T = self.F.T
        self.H_T = self.H.T

    def filter_coeffs(self, coeffs:NDArray, y_range:int, n_inliers:int, inlier_ratio:float):
        """
        Apply Kalman filtering to smooth coefficient estimates.
        
        Parameters
        ----------
        coeffs : NDArray, shape (n_coeffs,)
            New measurement (polynomial coefficients from current frame)
        y_range : float
            Vertical span of fitted points (affects measurement uncertainty)
        n_inliers : int
            Number of inlier points in fit
        inlier_ratio : float
            Fraction of points classified as inliers (0-1)
            
        Returns
        -------
        filtered_coeffs : NDArray, shape (n_coeffs,)
            Smoothed coefficient estimates
            
        Notes
        -----
        Prediction step: propagate state forward using motion model
        Update step: incorporate new measurement with adaptive noise
        Poor fits (low inlier_ratio) receive high measurement noise → trust prediction more
        """
        if coeffs.size == 0:
            return self._get_coeffs()
          
        self._predict(coeffs)
                
        self._update(coeffs, y_range, n_inliers, inlier_ratio)
        
        return self._get_coeffs()

    def _predict(self, coeffs:NDArray):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F_T + self.Q

        if np.trace(self.P) > 500.0:
            print("WARNING: Kalman filter diverging, performing reset")
            self._reset(coeffs)

    def _update(self, coeffs:NDArray, y_range:float, n_inliers:float, inlier_ratio:float):
        R = self._compute_R(y_range, n_inliers, inlier_ratio)
        z = coeffs.reshape(-1, 1)
        innovation = z - self.H @ self.x

        innovation_magnitude = np.linalg.norm(innovation)

        if innovation_magnitude > 5.0:
            print(f'WARNING: Large innovation {innovation_magnitude:.2f}, measurement may be outlier')
            return

        S = self.H @ self.P @ self.H_T + R

        try:
            K = self.P @ self.H_T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("WARNING: `S` is singular, using pseudo-inverse")
            K = self.P @ self.H_T @ np.linalg.pinv(S)

        self.x = self.x + K @ innovation
        IKH = self.I - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def _get_coeffs(self):
        return self.x[:len(self.x)//2].flatten()
    
    def _reset(self, coeffs:NDArray):
        
        self.x = self._initialize_current_state(coeffs)
        self.P = self._initialize_P(self.P_primer)
    
    def _initialize_current_state(self, coeffs:NDArray):
        top = coeffs.reshape(-1, 1)
        bottom = np.zeros_like(top)
        return np.block([
            [top],
            [bottom]
        ]).astype(float)
    
    def _initialize_Q(self, process_noise:Literal["low", "medium", "high"]):
        noise_dict = {"low": [0.1, 0.5], "medium": [0.55, 1.25], "high": [1.0, 2.0]}
        pos_noise, vel_noise = noise_dict[process_noise]
        
        left = np.full(self.poly_size, pos_noise)
        right = np.full(self.poly_size, vel_noise)
        
        return np.diag(np.hstack([left, right]))

    def _initialize_F_H(self):
        I = np.eye(len(self.x) // 2, dtype=float)
        dt_I = self.dt * I
        zeros = np.zeros_like(I, dtype=float)
        F = np.block([
            [I, dt_I],
            [zeros, I]
        ])
        H = np.block([
            [I, zeros]
        ])
        return F, H

    def _compute_R(self, y_range:float, n_inliers:int, inlier_ratio:float):
        """
        Compute adaptive measurement noise covariance matrix.
        
        R increases when:
        - Fit range is small (less information)
        - Inlier count is low (less reliable)
        - Inlier ratio is poor (likely bad detection)
        - For higher-order coefficients (polynomial principle)
        
        Parameters
        ----------
        y_range : float
            Vertical extent of fitted points in scaled space
        n_inliers : int
            Number of inlier points
        inlier_ratio : float
            Fraction of points that are inliers
            
        Returns
        -------
        R : NDArray, shape (n_coeffs, n_coeffs)
            Diagonal measurement noise covariance matrix
        """
        # Get variables for R_diag
        poly_factor = np.arange(1, self.poly_size + 1)

        range_factor = np.sqrt(200.0 / max(y_range, 50))
       
        n_factor = np.sqrt(100.0 / max(n_inliers, 10))

        base_noise = self._get_base_noise(inlier_ratio)
        
        R_diag = base_noise * poly_factor * range_factor * n_factor
        return np.diag(R_diag)

    def _get_base_noise(self, inlier_ratio:float):
        if inlier_ratio > 0.8:
            return 1.0
        if inlier_ratio > 0.5:
            return 2.0
        return 10.0
    
    def _initialize_P(self, P_primer:float=0.5):
        vel_P = P_primer * 10.0
        pos_cov = np.eye(self.poly_size) * P_primer
        vel_cov = np.eye(self.poly_size) * vel_P
        zeros_block = np.zeros((self.poly_size, self.poly_size))
        return np.block([
            [pos_cov, zeros_block],
            [zeros_block, vel_cov]
        ])