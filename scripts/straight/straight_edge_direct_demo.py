import numpy as np
import logging

from lane_detection.detection import DetectionSystem
from lane_detection.utils import setup_logging, get_logger

setup_logging(log_level=logging.WARNING,
              log_to_file=True,
              console_output=True)

logger = get_logger(__name__)

def demo_video(src:str, roi:np.ndarray, **kwargs):
    logger.info("="*60)
    logger.info(f"Starting Lane Line Detection Applicaiton for {src}")
    logger.info("="*60)
    
    system = DetectionSystem(
        source=src,
        roi=roi, 
        **kwargs
    )
    eval = system.run("inset", stroke=False, fill=True)

    print(eval)

if __name__=="__main__":
    src = "media/in/lane1-straight.mp4"

    STRAIGHT_LANE_VID_ROI = np.array([[[100, 540], 
                                       [900, 540], 
                                       [550, 340], 
                                       [420, 340]]], dtype=np.int32)

    STRAIGHT_EDGE_DIRECT_KWARGS = {
        "generator": "edge",
        "ksize": 5,
        "selector": "direct",
        "n_std": 6.0,
        "estimator": "ols",
        "degree": 2,
        "scaler_type": "z_score",
        "P_primer": 0.5,
        "process_noise": "low",
        "use_bev": True,
        "forward_range": 20.0,
        "lateral_range": 7.0,
        "resolution": 0.05
    }

    demo_video(src, STRAIGHT_LANE_VID_ROI, **STRAIGHT_EDGE_DIRECT_KWARGS)