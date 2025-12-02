import numpy as np
from lane_detection.detection import DetectionSystem

def demo_video(src:str, roi:np.ndarray, **kwargs):
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
    

    STRAIGHT_THRESH_DIRECT_KWARGS = {
        "generator": "thresh",
        "small_ksize": 5,
        "large_ksize": 15,
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
    
    demo_video(src, STRAIGHT_LANE_VID_ROI, **STRAIGHT_THRESH_DIRECT_KWARGS)