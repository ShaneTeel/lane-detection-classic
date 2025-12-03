import numpy as np
import logging

from .regression_metrics import (
    r2_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_average
)

logger = logging.getLogger(__name__)

class RegressionEvaluator():
    '''
    Description
    -----------
    Evaluator class that computes metrics for individual lane lines.

    Parameters
    ----------
    name : str, default None
        The name of the evaluator; typically derived from the name of the estimator used to predict the lane line.

    Public Methods
    --------------
    `.compute_metrics()`
        Takes an `NDArray` representing `y_true` and one representing `y_pred` and computes metrics for `R2`, `MSE`, `RMSE`, and `MAE`.
        This method returns no values.
    
    `.return_metrics()`
        Takes no arguments, but returns a `dict` of key (metric name) and value (avg metric).
    '''
    def __init__(self, name:str=None):
        '''
        Parameters
        ----------
        name : str, default None
            The name of the evaluator; typically derived from the name of the estimator used to predict the lane line.
        '''
        self.r2 = []
        self.mse = []
        self.rmse = []
        self.mae = []
        self.name = "Regression" if name is None else name

    def compute_metrics(self, y_true, y_pred):
        '''
        Description
        -----------
        Evaluates `R2`, `MSE`, `RMSE`, and `MAE` based on y_true, y_pred generated from lane line detection.

        Parameters
        ----------
        name : str, default None
            The name of the evaluator; typically derived from the name of the estimator used to predict the lane line.
        
        Returns
        -------
        None
        '''
        n = len(y_true)
        if n == 0:
            logger.warning("`y_true` contains no points; skipping line.")
            return
        
        self.r2.append(r2_score(y_true, y_pred))
        self.mse.append(mean_squared_error(y_true, y_pred, n))
        self.rmse.append(root_mean_squared_error(y_true, y_pred, n))
        self.mae.append(mean_absolute_average(y_true, y_pred))
        
    def return_metrics(self):
        '''
        Description
        -----------
        Returns averages of `R2`, `MSE`, `RMSE`, and `MAE` measured scores.
        
        Returns
        -------
        metrics : dict
            `dict` of averages of `R2`, `MSE`, `RMSE`, and `MAE` measured scores.
        '''
        return {"R2": np.mean(self.r2), "MSE": np.mean(self.mse), "RMSE": np.mean(self.rmse), "MAE": np.mean(self.mae)}