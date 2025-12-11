import numpy as np
from numpy.typing import NDArray
from typing import Literal, Union
import logging

from lane_detection.scalers import MinMaxScaler, StandardScaler
from .models import OLSRegression, RANSACRegression, KalmanFilter
from lane_detection.utils import get_logger

logger = get_logger(__name__)

class SingleLaneLineDetector():
    """
    Detects and tracks a single lane line using polynomial regression and Kalman filtering.
    
    Combines feature scaling, regression (OLS or RANSAC), and temporal filtering
    to generate smooth, stable lane line predictions across video frames.
    
    Parameters
    ----------
    scaler_type : {"min_max", "z_score"}
        Feature scaling method
    estimator_type : {"OLS", "RANSAC"}
        Regression algorithm
    degree : {1, 2, 3}, default=2
        Polynomial degree for lane fitting
    confidence : float, default=0.99
        RANSAC confidence level (only used if estimator_type="RANSAC")
    min_inliers : float, default=0.8
        RANSAC minimum inlier ratio (only used if estimator_type="RANSAC")
    max_error : int, default=10
        RANSAC error threshold in pixels (only used if estimator_type="RANSAC")
    P_primer : float, default=0.5
        Initial Kalman filter covariance (higher = less certain initial state)
    process_noise : {"low", "medium", "high"}, default="low"
        Kalman filter process noise level
    fps : int
        Video frame rate for Kalman filter time step calculation
        
    Attributes
    ----------
    name : str
        Detector identifier
    kalman : KalmanFilter
        Temporal smoothing filter (initialized after first fit)
    estimator : OLSRegression or RANSACRegression
        Fitted regression model
    scaler : MinMaxScaler or StandardScaler
        Fitted feature scaler
        
    Notes
    -----
    The detector operates in scaled space for numerical stability, then transforms
    predictions back to pixel coordinates for visualization.
    """
    def __init__(self, scaler_type:Literal["min_max", "z_score"], estimator_type:Literal["OLS", "RANSAC"], degree:Literal[1, 2, 3]=2, confidence:float=0.99, min_inliers:float=0.8, max_error:int=10, P_primer:float=0.5, process_noise:Literal["low", "medium", "high"]="low", fps:int=None):
        '''
        Parameters
        ----------
        scaler_type : {"min_max", "z_score"}
            Feature scaling method
        estimator_type : {"OLS", "RANSAC"}
            Regression algorithm
        degree : {1, 2, 3}, default=2
            Polynomial degree for lane fitting
        confidence : float, default=0.99
            RANSAC confidence level (only used if estimator_type="RANSAC")
        min_inliers : float, default=0.8
            RANSAC minimum inlier ratio (only used if estimator_type="RANSAC")
        max_error : int, default=10
            RANSAC error threshold in pixels (only used if estimator_type="RANSAC")
        P_primer : float, default=0.5
            Initial Kalman filter covariance (higher = less certain initial state)
        process_noise : {"low", "medium", "high"}, default="low"
            Kalman filter process noise level
        fps : int
            Video frame rate for Kalman filter time step calculation
        '''
        self.estimator_type = estimator_type
        self.estimator = None
        self.kalman = None
        self.process_noise = process_noise
        self.P_primer = P_primer
        self.fps = fps
        self.degree = degree
        self.confidence = confidence
        self.min_inliers = min_inliers
        self.max_error = max_error
        self.scaler_type = scaler_type
        self.scaler = None
        self.name = f"Kalman Filtered {estimator_type} Regression"

        logging.debug(f"Initialized SingleLaneLineDetector: Name={self.name}, with "
                      f"Scaler Type={self.scaler_type}.")

    def detect(self, lane:NDArray, start:Union[float, int], stop:Union[float, int]):
        """
        Detect lane line from point cloud and generate smooth polynomial.
        
        Parameters
        ----------
        lane : NDArray, shape (n_points, 2)
            Extracted lane points in (x, y) pixel coordinates
        start : float or int
            Starting y-coordinate for line generation
        stop : float or int
            Ending y-coordinate for line generation
            
        Returns
        -------
        line : NDArray, shape (100, 2)
            Smooth lane line points in (x, y) pixel coordinates
            
        Notes
        -----
        Pipeline: scale points → fit polynomial → apply Kalman filter → 
        generate smooth line → inverse transform to pixel space
        """
        if len(lane) < self.degree + 1:
            logger.warning("Lane line does not have enough points to perform fit.")
            return np.array([])

        # Scale and fit points
        X, y = self.fit_transform(lane)
        coeffs = self.fit(X, y)

        # Generate clean X and predict y; inverse transform to normal space
        start, stop = self.scaler.transform_by_X(start), self.scaler.transform_by_X(stop)
        X_lin, y_pred = self.predict(coeffs, start, stop)
        return self.generate_poly_line(X_lin, y_pred)
    
    def generate_poly_line(self, X:NDArray, y:NDArray):
        X_fin, y_fin = self.inverse_transform(X, y)
        return np.array([y_fin, X_fin], dtype=np.int32).T
        

    def fit_transform(self, lane:NDArray):

        X = lane[:, 1]
        y = lane[:, 0]

        self.scaler = self._get_scaler(self.scaler_type)

        return self.scaler.fit_transform(X, y)
    
    def fit(self, X:NDArray, y:NDArray):
        
        self.estimator = self._get_estimator()
        coeffs = self.estimator.fit(X, y)
        if self.kalman is None:
            self.kalman = KalmanFilter(self.fps, coeffs, self.P_primer, self.process_noise)
        y_range = self.scaler.y_range()
        n_inliers, inlier_ratio = self.estimator.get_inlier_info()
        self.coeffs = self.kalman.filter_coeffs(coeffs, y_range, n_inliers, inlier_ratio) 
        return self.coeffs
    
    def predict(self, coeffs:NDArray, start, stop):
        return self.estimator.predict(coeffs, start, stop)
    
    def inverse_transform(self, X:NDArray, y:NDArray):
        return self.scaler.inverse_transform(X, y)
    
    def get_fitted(self):
        if self.estimator is None:
            return np.array([]), np.array([])
        return self.estimator.get_fitted()
    
    def generate_evaluation_prediction(self):
        X, y = self.get_fitted()
        if X.size == 0 or y.size == 0:
            print("here")
            return np.array([]).reshape(-1, 2), np.array([]).reshape(-1, 2)
        y_pred_scaled = self.estimator.poly_val(self.coeffs, X)
        X_true, y_true = self.scaler.inverse_transform(X, y)
        _, y_pred = self.scaler.inverse_transform(X, y_pred_scaled)
        return np.column_stack([y_true, X_true]), np.column_stack([y_pred, X_true])

    def _get_estimator(self):
        if self.estimator_type == "OLS":
            return OLSRegression(degree=self.degree)
        
        if self.scaler is None:
            max_error = self.max_error
            logger.warning("Performing fit without scaling max_error")
        else:
            y_range = self.scaler.y_range()
            max_error = np.abs(self.max_error / y_range)
        return RANSACRegression(self.degree, self.confidence, self.min_inliers, max_error)
    
    def _get_scaler(self, scaler_type:Literal["min_max", "z_score"]):
            return MinMaxScaler() if scaler_type == "min_max" else StandardScaler()

    def get_name(self):
        return self.name