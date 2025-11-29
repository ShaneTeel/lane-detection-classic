import itertools
from typing import Literal

from .detection_system import systems_factory

class GridSearch():

    """
    Exhaustive hyperparameter search for lane detection system configuration.
    
    Tests all combinations of provided parameters and identifies the configuration
    yielding optimal performance on the specified metric.
    
    Parameters
    ----------
    source : str
        Path to video/image file for evaluation
    generator : {"edge", "thresh"}
        Feature map generation method
    selector : {"direct", "hough"}
        Point selection strategy
    estimator : {"ols", "ransac"}
        Regression model type
    param_grid : dict
        Parameter combinations to search. Must include "roi" key
    metric : {"r2", "mse", "rmse", "mae"}, default="r2"
        Optimization metric (higher is better for R², lower for errors)
        
    Attributes
    ----------
    best_params : dict
        Parameter configuration achieving best metric score
    best_score : float
        Optimal metric value achieved
    best_system : DetectionSystem
        Fitted system using best parameters (if refit=True)
    results : list
        Metric scores for all tested configurations
        
    Examples
    --------
    >>> param_grid = {
    ...     "roi": [roi],
    ...     "degree": [1, 2, 3],
    ...     "threshold": [100, 150, 200],
    ...     "use_bev": [True, False]
    ... }
    >>> gs = GridSearch("video.mp4", "edge", "hough", "ols", param_grid, "r2")
    >>> best_params, best_score, all_scores = gs.search_grid(refit=True)
    """
    
    def __init__(self, source:str, generator:Literal["edge", "thresh"], selector:Literal["direct", "hough"], estimator:Literal["ols", "ransac"], param_grid:dict, metric:Literal["r2", "mse", "rmse", "mae"]="r2"):

        self.src = source
        self.generator, self.selector, self.estimator = generator, selector, estimator
        self.combos, self.keys = self._initialize_configs(param_grid)
        self.metric_name = metric
        self.best_params = None
        self.final_configs = None
        self.best_score = None
        self.best_system = None
        self.results = []

    def search_grid(self, refit:bool=True):
        """
        Execute grid search over all parameter combinations.
        
        Parameters
        ----------
        refit : bool, default=True
            Whether to refit best system after identifying optimal parameters
            
        Returns
        -------
        final_configs : dict
            Complete configuration dictionary with best parameters
        best_score : float
            Optimal metric value achieved
        results : list
            All metric scores in order tested
            
        Notes
        -----
        For R² metric, higher scores are better. For error metrics (MSE, RMSE, MAE),
        lower scores are better. Grid search automatically handles optimization direction.
        """
        for combo in self.combos:
            params = dict(zip(self.keys, combo))
            system = self._initialize_system(params)
            _ = system.run(view_style=None)


            met1, met2 = system._return_score(self.metric_name)
            avg_score = sum([met1, met2]) / 2
            self.results.append(avg_score)

            if self.best_score is None:
                self._update_state(avg_score, params, system)
                continue

            if self.metric_name == "r2" and avg_score > self.best_score:
                self._update_state(avg_score, params, system)
                continue

            if self.metric_name != "r2" and avg_score < self.best_score:
                self._update_state(avg_score, params, system)
                continue

        default_configs = system.get_default_configs()
        self.final_configs = self._merge_configs(default_configs, self.best_params)
        if refit:
            self.best_system = self._initialize_system(self.final_configs)
        return self.final_configs, self.best_score, self.results
        
    def _initialize_configs(self, param_grid:dict):
        combinations = list(itertools.product(*param_grid.values()))
        keys = list(param_grid.keys())
        if "roi" not in keys:
            raise KeyError("ERROR: Argument for 'param_grid' must include 'roi' param.")
        
        return combinations, keys
        
    def _merge_configs(self, default_configs:dict, user_configs:dict):
        def _recursive_update(default_dict:dict, new_dict:dict):
            for key, val in new_dict.items():
                if isinstance(val, dict) and key in default_dict and isinstance(default_dict[key], dict):
                    _recursive_update(default_dict[key], val)
                else:
                    default_dict[key] = val
            return default_dict
        return _recursive_update(default_configs, user_configs)

    def _initialize_system(self, params:dict):
        return systems_factory(self.src, self.generator, self.selector, self.estimator, params=params)

    def _update_state(self, score, params, system):
        self.best_score = score
        self.best_params = params
        self.best_system = system