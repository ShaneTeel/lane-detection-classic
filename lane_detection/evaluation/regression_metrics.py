import numpy as np

def _residual_sum_of_squares(y_true, y_pred):
    '''Computes Residual Sum of Squares'''
    return np.sum((y_true - y_pred)**2)

def _total_sum_of_squares(y_true):
    '''Computes Total Sum of Squares'''
    y_mean = np.mean(y_true)
    return np.sum((y_true - y_mean)**2)

def r2_score(y_true, y_pred):
    '''Computes R2 Score'''
    rss = _residual_sum_of_squares(y_true, y_pred)
    tss = _total_sum_of_squares(y_true)
    r2 = 1 - rss / tss
    return r2

def mean_squared_error(y_true, y_pred, n):
    '''Computes Mean Squared Error Score'''
    rss = _residual_sum_of_squares(y_true, y_pred)
    return rss / n

def root_mean_squared_error(y_true, y_pred, n):
    '''Computes Root Mean Squared Error Score'''
    mse = mean_squared_error(y_true, y_pred, n)
    return mse**0.5

def mean_absolute_average(y_true, y_pred):
    '''Computes Mean Absolute Average Score'''
    return np.mean(abs(y_true - y_pred))