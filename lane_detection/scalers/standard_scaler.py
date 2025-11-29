import numpy as np
from numpy.typing import NDArray

class StandardScaler():
    '''
    Description
    -----------
    Preprocessing step that transforms features by scaling each feature to a given range.

    Public Methods
    --------------
    `.fit()`
    `.transform()`
    `.fit_transform()`
    `.inverse_transform()`
    `.transform_by_X()`
    `.transform_by_y()`
    `.y_range()`
    '''

    def __init__(self):
        self._is_fitted = False

    def fit(self, X:NDArray, y:NDArray):
        '''
        Description
        -----------
        Computes the mean and std for each argument and returns `self` for method chaining purposes.

        Parameters
        ----------
        X : NDArray
            Array of n_samples used to compute mean and std
        y : NDArray
            Array of n_samples used to compute mean and std

        Returns
        -------
        `StandardScaler()`
        '''     
        self.X_mean, self.X_std = np.mean(X), np.std(X)
        self.X_max, self.X_min = X.max(), X.min()

        self.y_mean, self.y_std = np.mean(y), np.std(y)
        self.y_max, self.y_min = y.max(), y.min()
        
        self._is_fitted = True
        return self

    def transform(self, X:NDArray, y:NDArray):
        '''
        Description
        -----------
        Scales each argument to a given range based on the mean and std computed during `.fit()`.

        Parameters
        ----------
        X : NDArray
            Array of n_samples to be transformed into scaled space.
        y : NDArray
            Array of n_samples to be transformed into scaled space.

        Returns
        -------
        X_scaled : NDArray
            X argument scaled.
        y_scaled : NDArray
            y argument scaled.
        '''
        self._fit_check()

        X = self.transform_by_X(X)
        y = self.transform_by_y(y)

        return X, y
    
    def fit_transform(self, X:NDArray, y:NDArray):
        '''
        Description
        -----------
        Computes the mean and std for each argument and scales them to a given range based on the computed values.

        Parameters
        ----------
        X : NDArray
            Array of n_samples to be fit and transformed into scaled space.
        y : NDArray
            Array of n_samples to be fit and transformed into scaled space.

        Returns
        -------
        X_scaled : NDArray
            X argument scaled.
        y_scaled : NDArray
            y argument scaled.
        '''
        
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, X_scaled:NDArray, y_scaled:NDArray):
        '''
        Description
        -----------
        Transforms each argument back to their original, un-scaled space.

        Parameters
        ----------
        X_scaled : NDArray
            Array of n_samples to be inverse scaled.
        y_scaled : NDArray
            Array of n_samples to be inverse scaled.

        Returns
        -------
        X : NDArray
            X argument un-scaled.
        y : NDArray
            y argument un-scaled.

        Raises
        ------
        RuntimeError
            Raised when any public method is called before `.fit()` or `.fit_transform()`.
        '''
        self._fit_check()

        X = self._inverse_transform_X(X_scaled)
        y = self._inverse_transform_y(y_scaled)
        return X, y
    
    def _inverse_transform_X(self, X_scaled:NDArray):
        '''
        Description
        -----------
        Transforms X_scaled back to its' original, un-scaled space.

        Parameters
        ----------
        X_scaled : NDArray
            Array of n_samples to be inverse scaled.

        Returns
        -------
        X : NDArray
            X argument un-scaled.

        Raises
        ------
        RuntimeError
            Raised when any public method is called before `.fit()` or `.fit_transform()`.
        '''    
        self._fit_check()

        return X_scaled * self.X_std + self.X_mean if self.X_std > 0 else X_scaled
    
    def _inverse_transform_y(self, y_scaled:NDArray):
        '''
        Description
        -----------
        Transforms y_scaled back to its' original, un-scaled space.

        Parameters
        ----------
        y_scaled : NDArray
            Array of n_samples to be inverse scaled.

        Returns
        -------
        y : NDArray
            y argument un-scaled.

        Raises
        ------
        RuntimeError
            Raised when any public method is called before `.fit()` or `.fit_transform()`.
        ''' 
        self._fit_check()

        return y_scaled * self.y_std + self.y_mean if self.y_std > 0 else y_scaled


    def transform_by_X(self, val:NDArray | int | float):
        '''
        Description
        -----------
        Transforms a value by fitted `X` mean and std.

        Parameters
        ----------
        val : NDArray | int | float
            value to be scaled.

        Returns
        -------
        val_scaled : NDArray | int | float
            value post-scaling.
        Raises
        ------
        RuntimeError
            Raised when any public method is called before `.fit()` or `.fit_transform()`.
        '''   
        self._fit_check()

        return (val - self.X_mean) / self.X_std if self.X_std != 0 else val

    def transform_by_y(self, val:NDArray | int | float):
        '''
        Description
        -----------
        Transforms a value by fitted `y` mean and std.

        Parameters
        ----------
        val : NDArray | int | float
            value to be scaled.

        Returns
        -------
        val_scaled : NDArray | int | float
            value post-scaling.
        Raises
        ------
        RuntimeError
            Raised when any public method is called before `.fit()` or `.fit_transform()`.
        '''   
        self._fit_check()

        return (val - self.y_mean) / self.y_std if self.y_std != 0 else val
    
    def X_range(self):
        '''
        Description
        -----------
        Computes the range of the fitted `X` variable.

        Returns
        -------
        X_range : int | float
            scaler-value representing the range of `X`.
        Raises
        ------
        RuntimeError
            Raised when any public method is called before `.fit()` or `.fit_transform()`.
        '''
        self._fit_check()

        return self.X_max - self.X_min

    def y_range(self):
        '''
        Description
        -----------
        Computes the range of the fitted `y` variable.

        Returns
        -------
        y_range : int | float
            scaler-value representing the range of `y`.
        Raises
        ------
        RuntimeError
            Raised when any public method is called before `.fit()` or `.fit_transform()`.
        '''
        self._fit_check()

        return self.y_max - self.y_min
    
    def _fit_check(self):
        '''
        Description
        -----------
        Utility function used to raise a `RuntimeError()` whenever a method is called before `.fit()` or `.fit_transform()`.
        '''
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before performing this operation.")