import numpy as np
from lane_detection.detection import DetectionSystem

def demo_video():
    src = "media/in/lane1-straight.mp4"

    roi = np.array([[[100, 540], 
                     [900, 540], 
                     [525, 325], 
                     [445, 325]]], dtype=np.int32)

    system = DetectionSystem(
        source=src, 
        roi=roi, 
        generator="edge",
        selector='hough', 
        estimator='ols',
        degree=2,
        scaler_type="z_score",
        P_primer=0.9,
        process_noise="low",
        threshold=150,
        ksize=5,
        small_ksize=5,
        large_ksize=15,
        n_std=2.0,
        use_bev=True,
        forward_range=20.0,
        lateral_range=20.0,
        resolution=0.05
    )

    eval = system.run("inset", stroke=False, fill=True)

    print(eval)

if __name__=="__main__":
    demo_video()