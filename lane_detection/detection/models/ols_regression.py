import numpy as np
from numpy.typing import NDArray
from lane_detection.utils import get_logger

logger = get_logger(__name__)

class OLSRegression:
    """
    Ordinary Least Squares polynomial regression for lane line fitting.
    
    Fits polynomials by minimizing sum of squared residuals. Fast but
    sensitive to outliers.
    
    Parameters
    ----------
    degree : int, default=2
        Polynomial degree (1=linear, 2=quadratic, 3=cubic)
        
    Attributes
    ----------
    fitted_X : NDArray
        X values from last fit
    fitted_y : NDArray
        y values from last fit
    n_inliers : int
        Number of points used (equals total points for OLS)
    inlier_ratio : float
        Always 1.0 for OLS (no outlier rejection)
        
    Notes
    -----
    Solves normal equations: (X.T @ X)^(-1) @ (X.T @ y)
    Uses condition number check to detect ill-conditioned systems.
    """
    
    def __init__(self, degree:int = 2):
        '''
        Parameters
        ----------
        degree : int, default=2
            Polynomial degree (1=linear, 2=quadratic, 3=cubic)
        '''
        self.degree = degree
        self.poly_size = self.degree + 1
        self.fitted = False
        self.name = "OLS Regression"

    def fit_predict(self, X:NDArray, y:NDArray, start:float | int, stop: float | int):
        coeffs = self.fit(X, y)
        return self.predict(coeffs, start, stop)
    
    def fit(self, X:NDArray, y:NDArray):
        """
        Fit polynomial coefficients to data using least squares.
        
        Parameters
        ----------
        X : NDArray, shape (n_samples,)
            Independent variable (typically y-coordinates in scaled space)
        y : NDArray, shape (n_samples,)
            Dependent variable (typically x-coordinates in scaled space)
            
        Returns
        -------
        coeffs : NDArray, shape (degree+1,)
            Polynomial coefficients [c0, c1, ..., c_degree]
            Returns None if system is ill-conditioned
            
        Notes
        -----
        Stores fitted data for later evaluation. Coefficients represent
        polynomial: y = c0 + c1*X + c2*X^2 + ... + c_degree*X^degree
        """
        # Generate X matrix
        X_mat = self._gen_X_design(X)

        self.fitted_X, self.fitted_y = X, y

        self.inlier_ratio = 1.0
        self.n_inliers = len(X)
        
        # Estimate best coeffs
        self.fitted = True
        
        return self._calc_coeffs(X_mat, y)
    
    def predict(self, coeffs:NDArray, start, stop):
        # Get start, stop for np.linspace
        
        # Generate 100 points in scaled space
        X_lin = np.linspace(start, stop, 100)
        
        # Estimate respective y-values in scaled space
        y_pred = self.poly_val(coeffs, X_lin)

        return X_lin, y_pred
        
    def _calc_coeffs(self, X:NDArray, y:NDArray):
        '''(X.T * X)**-1 * (X.T & y)'''
        if X is None or y is None:
            raise ValueError(f"Error: 'X' ({X}) or 'y' ({y}) == 'NoneType'")
        else:
            XT = X.T
            A = XT @ X
            
            if np.linalg.cond(A) > 1e10:
                return None
            b = XT @ y
            return np.linalg.solve(A, b)
        
    def poly_val(self, coeffs:NDArray, X:NDArray):
        result = coeffs[-1]
        for coef in reversed(coeffs[:-1]):
            result = result * X + coef
        return result
    
    def _gen_X_design(self, X:NDArray):
        X_mat = [np.ones_like(X)]
        for i in range(1, self.poly_size):
            X_mat.append(X**i)
        X = np.column_stack(X_mat)

        return X

    def get_fitted(self):
        if not self.fitted:
            raise RuntimeError("ERROR: Must perform fit before calling")
        return self.fitted_X, self.fitted_y
    
    def get_inlier_info(self):
        if not self.fitted:
            raise RuntimeError("ERROR: Must perform fit before calling")
        return self.n_inliers, self.inlier_ratio