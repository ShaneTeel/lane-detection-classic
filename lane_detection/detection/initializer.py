# Imports
from pydantic import BaseModel, Field, create_model, ConfigDict
from typing import Literal, Union
import numpy as np
from numpy.typing import NDArray

from lane_detection.studio import StudioManager
from lane_detection.feature_generation import VerticalEdgeMapGenerator, ThresholdMapGenerator
from lane_detection.feature_selection import HoughPSelector, DirectPixelSelector
from lane_detection.image_geometry import BEVProjector, ROIMasker
from .single_lane_line_detector import SingleLaneLineDetector
from lane_detection.evaluation import RegressionEvaluator

class Initializer():

    def __init__(self, generator:Literal["thresh", "edge"], selector:Literal["direct", "hough"], estimator:Literal["ols", "ransac"], **kwargs):
        user_selection = {"generator": generator, "selector": selector, "estimator": estimator}
        
        self.generator = generator
        self.selector = selector
        self.estimator = estimator
        self.configs = self._initialize_configs(user_selection, **kwargs)

    def initialize_studio(self, source:Union[str, int, StudioManager]):
        if isinstance(source, StudioManager):
            return source
        studio = StudioManager(source)
        self.fps, _, _ = studio.get_metadata()
        return studio
    
    def initialize_geometry(self, roi:NDArray | ROIMasker):
        masker = roi if isinstance(roi, ROIMasker) else ROIMasker(roi)
        if not self.configs.use_bev:
            return masker, None
        src_pts = masker.get_src_pts()
        return masker, BEVProjector(src_pts=src_pts, forward_range=self.configs.forward_range, lateral_range=self.configs.lateral_range, resolution=self.configs.resolution)

    def initialize_generator(self):
        if "edge" in self.generator:
            return VerticalEdgeMapGenerator(self.configs.ksize)
        elif "thresh" in self.generator:
            return ThresholdMapGenerator(self.configs.large_ksize, self.configs.small_ksize)

    def initialize_selector(self):
        if "direct" in self.selector:
            return DirectPixelSelector(n_std=self.configs.n_std)
        
        if "hough" in self.selector:
            return HoughPSelector(min_votes=self.configs.min_votes, min_length=self.configs.min_length, max_gap=self.configs.max_gap)

    def initialize_detectors(self):
        base_params = {
            "scaler_type": self.configs.scaler_type,
            "estimator_type": "RANSAC" if "ransac" in self.estimator else "OLS",
            "degree": self.configs.degree,
            "P_primer": self.configs.P_primer,
            "process_noise": self.configs.process_noise,
            "fps": self.fps
        }
        if "ransac" in self.estimator:
            base_params.update({
                "confidence": self.configs.confidence,
                "min_inliers": self.configs.min_inliers,
                "max_error": self.configs.max_error
            })
        return SingleLaneLineDetector(**base_params), SingleLaneLineDetector(**base_params)
    
    def initialize_evaluators(self, name):
        return RegressionEvaluator(name), RegressionEvaluator(name)

    def _initialize_configs(self, user_selection:dict, **kwargs):
        final = {}

        for option, choice in user_selection.items():
            config_obj = CONFIG_MAP[option][choice](**kwargs)
            final.update(config_obj.model_dump())

        config_obj = CONFIG_MAP["bev"](**kwargs)
        final.update(config_obj.model_dump())

        FinalConfigs = create_model(
            "FinalConfigs", **{k: (type(v), v) for k, v in final.items()},
        )
        return FinalConfigs()

class BEVConfigs(BaseModel):
    use_bev:bool = False
    forward_range:float = 40.0
    lateral_range:float = 7.0 
    resolution:float = Field(default=0.03, ge=0.01)

class EdgeMapConfigs(BaseModel):
    ksize:Literal[3, 5, 7, 9, 11, 13, 15] = 3

class ThreshMapConfigs(BaseModel):
    small_ksize:Literal[3, 5, 7, 9, 11, 13, 15] = 3
    large_ksize:Literal[11, 13, 15, 17, 19, 21] = 15

class HoughConfigs(BaseModel):
    min_votes:int = Field(default=75, ge=1)
    min_length:int = Field(default=25, ge=1)
    max_gap:int = Field(default=10, ge=1)

class DirectPixelConfigs(BaseModel):
    n_std:float = Field(default=2.0, ge=0.5)

class OLSConfigs(BaseModel):
    scaler_type:Literal["min_max", "z_score"] = "min_max"
    degree:Literal[1, 2, 3] = 2
    P_primer:float = Field(default=0.5, ge=0.0, le=0.99)
    process_noise:Literal["low", "medium", "high"] = "low"

class RANSACConfigs(OLSConfigs):
    confidence:float = Field(default=0.99, ge=0.0, le=1.0)
    min_inliers:float = Field(default=0.8, ge=0.0, le=1.0)
    max_error:int = Field(default=10, ge=0)
    
CONFIG_MAP = {
    "generator": {"edge": EdgeMapConfigs, "thresh": ThreshMapConfigs},
    "selector": {"direct": DirectPixelConfigs, "hough": HoughConfigs},
    "estimator": {"ols": OLSConfigs, "ransac": RANSACConfigs},
    "bev": BEVConfigs
    }