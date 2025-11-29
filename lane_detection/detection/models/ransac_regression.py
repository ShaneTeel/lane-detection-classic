import numpy as np
from numpy.typing import NDArray
from .ols_regression import OLSRegression

class RANSACRegression():
    """
    RANSAC-based polynomial regression robust to outliers.
    
    Iteratively samples point subsets, fits models, and identifies the model
    with maximum inlier support. Robust to up to 50% outlier contamination.
    
    Parameters
    ----------
    degree : int, default=2
        Polynomial degree
    confidence : float, default=0.99
        Probability of finding outlier-free sample
    min_inliers : float, default=0.8
        Minimum inlier ratio for consensus (0-1 for fraction, ≥1 for count)
    max_error : int, default=10
        Inlier threshold in scaled space units
        
    Attributes
    ----------
    best_coeffs : NDArray
        Coefficients with maximum inlier support
    best_inlier_count : int
        Number of inliers for best model
    inlier_ratio : float
        Fraction of points classified as inliers
        
    Notes
    -----
    Adaptively determines iteration count based on observed inlier ratio.
    Falls back to OLS if insufficient points or no consensus reached.
    """

    def __init__(self, degree:int = 2, confidence:float=0.99, min_inliers:float = 0.8, max_error:int = 10):
        '''
        Parameters
        ----------
        degree : int, default=2
            Polynomial degree
        confidence : float, default=0.99
            Probability of finding outlier-free sample
        min_inliers : float, default=0.8
            Minimum inlier ratio for consensus (0-1 for fraction, ≥1 for count)
        max_error : int, default=10
            Inlier threshold in scaled space units
        '''
        self.estimator = OLSRegression(degree)
        self.poly_size = self.estimator.poly_size
        self.degree = self.estimator.degree
        self.max_iter = 1000
        self.P = confidence
        self.min_inliers = min_inliers
        self.max_error = max_error
        self.inlier_ratio = None
        self.best_inliers = None
        self.best_inlier_count = 0
        self.best_coeffs = None
        self.fitted = False
        self.name = "RANSAC Regression"

    def fit_predict(self, X:NDArray, y:NDArray, start:float | int, stop: float | int):
        coeffs = self.fit(X, y)
        return self.predict(coeffs, start, stop)

    def fit(self, X:NDArray, y:NDArray):
        """
        Fit polynomial using RANSAC outlier rejection.
        
        Parameters
        ----------
        X : NDArray, shape (n_samples,)
            Independent variable
        y : NDArray, shape (n_samples,)
            Dependent variable
            
        Returns
        -------
        coeffs : NDArray, shape (degree+1,)
            Polynomial coefficients for best inlier set
            
        Notes
        -----
        Algorithm:
        1. Randomly sample minimal point set
        2. Fit polynomial to sample
        3. Count inliers (points within max_error)
        4. Update best model if more inliers found
        5. Adapt iteration count based on inlier ratio
        6. Refit final model on all inliers
        """
        population = len(X)

        consensus = self._calc_consensus(population)

        # If there are too few points to perform RANSAC, fall back to OLS
        sample_size = min(max(self.poly_size, 1), population)

        if population < sample_size:
            self.fitted = True
            return self.estimator.fit(X, y)

        # Use the minimal sample size necessary for the model (poly_size)
        N_required = float("inf")
        N_completed = 0

        while N_completed < N_required and N_completed < self.max_iter:
            sample_X, sample_y = self._rand_sampling(X, y, population, sample_size)

            # Fit polynomial to samples
            ret, coeffs = self._perform_sample_fit(sample_X, sample_y)
            
            if not ret:
                print(f"WARNING: Unable to perform fit, skipping iteration {N_completed}")
                N_completed += 1
                continue

            # Evaluate sample fit on all points (population, scaled X)
            self._evaluate_sample_fit(coeffs, X, y)
            
            self.inlier_ratio = self.best_inlier_count / population if population > 0 else 0.0
            if self.inlier_ratio == 1:
                break

            if self.inlier_ratio > 0.0:
                u_s = self.inlier_ratio ** self.poly_size
                if u_s != 1:
                    try:
                        N_required = np.log(1 - self.P) / np.log(1 - u_s)
                    except Exception as e:
                        raise ValueError(e)                    

            N_completed += 1

        self.fitted = True
        return self._get_best_coeffs(consensus, X, y)

    def predict(self, coeffs:NDArray, start, stop):
        return self.estimator.predict(coeffs, start, stop)
    
    def poly_val(self, coeffs:NDArray, X:NDArray):
        return self.estimator.poly_val(coeffs, X)

    def _perform_sample_fit(self, sample_X:NDArray, sample_y:NDArray):
        # Fit polynomial to samples
        try:
            coeffs = self.estimator.fit(sample_X, sample_y)

            if isinstance(coeffs, np.ndarray) or len(coeffs) == self.poly_size:
                return True, coeffs
            else:
                print(f"WARNING: calcualted coeffs not of correct type ({np.ndarray}) or correct size ({self.poly_size})")
                return False, None
            
        except (np.linalg.LinAlgError, TypeError, ValueError) as e:
            print(f"WARNING: polyfit error - {e}")
            return False, None

    def _evaluate_sample_fit(self, coeffs:NDArray, X:NDArray, y:NDArray):
        y_pred = self.poly_val(coeffs, X)

        # Use absolute error
        sample_errors = np.abs(y - y_pred)

        inlier_mask = sample_errors <= self.max_error
        inlier_count = np.sum(inlier_mask)

        # Best coeffs check
        if inlier_count > self.best_inlier_count:
            self.best_inlier_count = inlier_count
            self.best_inliers = inlier_mask
            self.best_coeffs = coeffs.copy()
    
    def _rand_sampling(self, X:NDArray, y:NDArray, population:int, sample_size:int):
        # Random sampling
        sample_idx = np.random.choice(population, size=sample_size, replace=False)
        sample_X = X[sample_idx]
        sample_y = y[sample_idx]
        return sample_X, sample_y
    
    def _calc_consensus(self, population:int):
        # Determine consensus count. Support fraction (0<min_inliers<1) or absolute count (>=1).        
        if self.min_inliers is None:
            return int(np.ceil(population * 0.5))
        else:
            try:
                if self.min_inliers < 1:
                    return int(np.ceil(population * self.min_inliers))
                else:
                    return int(self.min_inliers)
            except TypeError:
                # fallback to 50% if misconfigured
                return int(np.ceil(population * 0.5))
            
    def _get_best_coeffs(self, consensus:NDArray, X:NDArray, y:NDArray):
        if self.best_inliers is not None and self.best_inlier_count >= consensus:
            inlier_X = X[self.best_inliers]
            inlier_y = y[self.best_inliers]

            if len(inlier_X) >= self.poly_size:
                try:
                    ransac_coeffs = self.estimator.fit(inlier_X, inlier_y)
                    if isinstance(ransac_coeffs, np.ndarray) and len(ransac_coeffs) == self.poly_size:
                        return ransac_coeffs
                except (np.linalg.LinAlgError, TypeError, ValueError) as e:
                    print(f"WARNING: Failed to refit best coeffs due to {e}; returning best_coeffs without refit")
                else:
                    return self.best_coeffs # Return best coeffs without refit 
                       
        if self.best_inliers is None or self.best_inlier_count < consensus:
            print(f"NO CONSENSUS! Best inlier's account for {self.inlier_ratio * 100}% of total population, but args required {self.min_inliers * 100}%. Falling back to full-data OLS.")
        # Leading Coefficient check: If leading coefficient is a negative value, fit all data
        if self.degree == 2 and self.best_coeffs is not None:
            if self.best_coeffs[-1] < 0:
                print(f"WARNING: Suspecious parabola a = {self.best_coeffs[-1]}")
                return self.estimator.fit(X, y)

        # FAIL SAFE: Fit all data (ordinary least squares)
        try:
            print("FAIL SAFE")
            last_resort = self.estimator.fit(X, y)
            if isinstance(last_resort, np.ndarray) and len(last_resort) == self.poly_size:
                return last_resort
        except:
            print("FAIL SAFE FAILED")
            return None

    def get_fitted(self):
        if not self.fitted:
            raise RuntimeError("ERROR: Must perform fit before calling")
        return self.estimator.get_fitted()
    
    def get_inlier_info(self):
        if not self.fitted:
            raise RuntimeError("ERROR: Must perform fit before calling")
        
        return self.best_inlier_count, self.inlier_ratio