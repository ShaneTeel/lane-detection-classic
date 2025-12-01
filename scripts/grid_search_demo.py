import numpy as np
from lane_detection.detection import GridSearch

def demo_grid_search():
    src = "media/in/lane1-straight.jpg"

    roi = np.array([[[100, 540], 
                     [900, 540], 
                     [530, 330], 
                     [440, 330]]], dtype=np.int32)

    grid = {
        "roi": [roi], 
        "scaler_type": ["z_score", "min_max"], 
        "ksize": [3, 5, 7, 11],
        "max_gap": [10, 15],
        "use_bev": [True, False],
        "degree": [1, 2]
    }

    gs = GridSearch(source=src, generator="edge", selector="hough", estimator="ols", metric="r2", param_grid=grid)

    params, scores, _ = gs.search_grid(refit=True)

    print(f"Best R2 Scores: {scores}")
    print(f"Best Params: {params}")

    system = gs.best_system

    eval = system.run("inset", stroke=True, fill=False)

    print(eval)

if __name__=="__main__":
    demo_grid_search()