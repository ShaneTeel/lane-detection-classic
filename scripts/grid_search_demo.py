import numpy as np
import logging

from lane_detection.detection import GridSearch
from lane_detection.utils import setup_logging, get_logger

setup_logging(log_level=logging.WARNING,
              log_dir="./logs",
              log_to_file=True,
              console_output=True)

logger = get_logger(__name__)

def demo_grid_search(src:str, params):
    logger.info("="*60)
    logger.info(f"Starting Grid Search Applicaiton for {src}")
    logger.info("="*60)

    gs = GridSearch(source=src, generator="edge", selector="hough", estimator="ols", metric="r2", param_grid=grid)

    params, scores, _ = gs.search_grid(refit=True)

    print(f"Best R2 Scores: {scores}")
    print(f"Best Params: {params}")

    system = gs.best_system

    eval = system.run("inset", stroke=True, fill=False)

    print(eval)

if __name__=="__main__":
    src = "media/in/readme_test_img.jpg"
    
    STRAIGHT_LANE_VID_ROI = np.array([[[100, 540], 
                                       [900, 540], 
                                       [550, 340], 
                                       [420, 340]]], dtype=np.int32)
    grid = {
        "roi": [STRAIGHT_LANE_VID_ROI], 
        "scaler_type": ["z_score", "min_max"], 
        "ksize": [3, 5, 7, 11],
        "max_gap": [10, 15],
        "use_bev": [True, False],
        "degree": [1, 2]
    }
    demo_grid_search(src, grid)