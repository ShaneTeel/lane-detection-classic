# Visualization
import cv2 
import numpy as np

# Type Hints
from numpy.typing import NDArray
from typing import Literal, Union

# Package-internal module
from .initializer import Initializer
from .single_lane_line_detector import SingleLaneLineDetector
from lane_detection.evaluation import RegressionEvaluator

import logging

logger = logging.getLogger(__name__)

def systems_factory(source:str, generator:Literal["edge", "thresh"], selector:Literal["direct", "hough"], estimator:Literal["ols", "ransac"], params:dict):
    
    if "roi" not in params.keys():
        raise KeyError("ERROR: argument passed for params must inlcude key labeled 'roi'.")
    
    return DetectionSystem(source=source, generator=generator, selector=selector, estimator=estimator, **params)

class DetectionSystem():
    """
    Orchestrates the complete lane detection pipeline including feature generation, 
    selection, regression modeling, Kalman filtering, and visualization.
    
    Parameters
    ----------
    source : str or int
        Path to video/image file or camera index (0 for default webcam)
    roi : NDArray, shape (1, 4, 2)
        Region of interest defining the area to analyze for lanes
    generator : {"edge", "thresh"}
        Feature map generation method
    selector : {"direct", "hough"}
        Point selection strategy from feature maps
    estimator : {"ols", "ransac"}
        Regression model for fitting lane polynomials
    **kwargs : dict
        Additional configuration parameters passed to component initializers
    """

    _BOLD = "\033[1m"
    _ITALICS = "\033[3m"
    _UNDERLINE = "\033[4m"
    _END = "\033[0m"

    def __init__(self, source:Union[str, int], roi:NDArray, generator:Literal["edge", "thresh"], selector:Literal["direct", "hough"], estimator:Literal["ols", "ransac"], **kwargs):        
        '''
        Parameters
        ----------
        source : str or int
            Path to video/image file or camera index (0 for default webcam)
        roi : NDArray, shape (1, 4, 2)
            Region of interest defining the area to analyze for lanes
        generator : {"edge", "thresh"}
            Feature map generation method
        selector : {"direct", "hough"}
            Point selection strategy from feature maps
        estimator : {"ols", "ransac"}
            Regression model for fitting lane polynomials
        **kwargs : dict
            Additional configuration parameters passed to component initializers
        '''    
        logger.debug("Initializing detection system")

        self.initializer = Initializer(generator, selector, estimator, **kwargs)
        
        self.studio = self.initializer.initialize_studio(source)        
        self.mask, self.bev = self.initializer.initialize_geometry(roi)
        self.generator = self.initializer.initialize_generator()
        self.selector = self.initializer.initialize_selector()
        self.detector1, self.detector2 = self.initializer.initialize_detectors()
        self.evaluator1, self.evaluator2 = self.initializer.initialize_evaluators(self.detector1.get_name())
        self.name = f"{self.studio.get_name()} {self.detector1.get_name()}"
        self.start, self.stop = self.get_linspace_start_stop()
        self.exit = False

    def preview(self, view_style:Literal["original", "masked", "diptych"]="diptych"):
        """
        Display ROI masking without running full detection pipeline.
        
        Useful for validating ROI selection before processing.
        
        Parameters
        ----------
        view_style : {"original", "masked", "diptych"}, default="diptych"
            Preview display mode
        """
        frame_names = self._configure_output(view_style, False, method="preview")
        
        while True and not self.exit:
            ret, frame = self.studio.return_frame()
            if not ret:
                break
            else:
                masked = self.mask.inverse_mask(frame)
                self._generate_output(view_style, [frame, masked], frame_names, None, False, False, False)

    def run(self, view_style:Union[Literal["inset", "mosaic", "composite"], None]="composite", stroke:bool=False, fill:bool=True, file_out_name:str=None):
        """
        Execute lane detection pipeline on entire video/image source.
        
        Processes each frame through feature generation, point selection,
        polynomial fitting, Kalman filtering, and optional visualization.
        
        Parameters
        ----------
        view_style : {"inset", "mosaic", "composite", None}, default="composite"
            Visualization layout style. None disables visualization
        stroke : bool, default=False
            Draw lane line boundaries
        fill : bool, default=True
            Fill region between detected lanes
        save : bool, default=False
            Save processed video to disk
            
        Returns
        -------
        report : str
            Formatted evaluation metrics (R², MSE, RMSE, MAE) for left/right lanes
            
        Notes
        -----
        Metrics are computed by comparing fitted polynomials against extracted points.
        When using BEV projection, comparison occurs in camera space after inverse transform.
        """
        
        frame_names = self._configure_output(view_style, file_out_name, method="final")

        while True and not self.exit:
            ret, frame = self.studio.return_frame()

            if not ret:
                break

            else:
                if self.studio.source_type() == "image":
                    self.exit = True
                thresh, feature_map = self.generator.generate(frame)
                masked = self.mask.inverse_mask(feature_map)
                lane_pts = self.selector.select(masked)
                
                lane_lines = []
                for i in range(2):
                    detector = self.detector1 if i == 0 else self.detector2
                    evaluator = self.evaluator1 if i == 0 else self.evaluator2
                    if lane_pts[i].size == 0:
                        continue

                    pts = lane_pts[i]

                    line = self.detect_line(pts, detector)
                    lane_lines.append(np.flipud(line)) if i == 0 else lane_lines.append(line)
                    self.evaluate_model(detector, evaluator)

                frame_lst = [frame, thresh, feature_map, masked]

                if view_style is not None:
                    self._generate_output(view_style, frame_lst, frame_names, lane_lines, stroke, fill)
                
        return self._generate_report()
    
    def detect_line(self, pts:NDArray, detector:SingleLaneLineDetector):
        if self.bev is not None:
            pts = self.bev.project(pts, "forward")    
        line = detector.detect(pts, self.start, self.stop)
        return self.bev.project(line, "backward") if self.bev is not None else line

    def evaluate_model(self, detector:SingleLaneLineDetector, evaluator:RegressionEvaluator):
        y_true, y_pred = detector.generate_evaluation_prediction()
        if self.bev is not None:
            y_true = self.bev.project(y_true, "backward")[:, 0]
            y_pred = self.bev.project(y_pred, "backward")[:, 0]
        
        evaluator.compute_metrics(y_true, y_pred)
    
    def _generate_report(self):
        metrics1 = self.evaluator1.return_metrics()
        metrics2 = self.evaluator2.return_metrics()
        report = f"\n{self._BOLD}{self._UNDERLINE}{self._ITALICS}{self.name} Report{self._END}\n\n"
        report += f"{self._BOLD}{self._ITALICS}Metrics      Left     Right       Avg{self._END}\n"
        for key in metrics1.keys():
            met1 = metrics1[key]
            met2 = metrics2[key]
            comb = (met1 + met2) / 2
            report += f"{key:>7}{met1:>10.4f}{met2:>10.4f}{comb:>10.4f}\n"
        return report
    
    def _return_score(self, score_type:Literal["R2", "MSE", "RMSE", "MAE"]="R2"):
        met1 = self.evaluator1.return_metrics()[score_type.upper()]
        met2 = self.evaluator2.return_metrics()[score_type.upper()]
        return met1, met2
    
    def _configure_output(self, view_style:str=None, file_out_name:str=None, method:Literal["preview", "final"]="final"):
        if view_style is not None:      
            if self.studio.source_type() != "image":
                self.studio.print_menu()
            if file_out_name is not None:
                self.studio.create_writer(file_out_name)
            return self.studio.get_frame_names(view_style.lower(), method)

    def _generate_output(self, view_style, frame_lst:list, frame_names:list, lane_lines:NDArray=None, stroke:bool=True, fill:bool=False):
        final = self.studio.gen_view(frame_lst, frame_names, lane_lines, view_style, stroke, fill)
        cv2.namedWindow(self.name)
        cv2.imshow(self.name, final)
        if self.studio.control_playback():
            self.exit = True
        if self.studio.writer_check():
            self.studio.write_frames(final)

    def get_default_configs(self):
        return dict(self.initializer.configs)

    def get_linspace_start_stop(self):
        if self.bev is not None:
            return self.bev.dst_y_min_max()
        return self.mask.src_y_min_max()