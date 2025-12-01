import numpy as np
from lane_detection.detection import DetectionSystem

def demo_video():
    src = "media/in/lane1-curved.mp4"

    roi = np.array([[[180, 665], 
                     [1100, 665], 
                     [725, 450], 
                     [575, 450]]], dtype=np.int32)

    system = DetectionSystem(
        source=src, 
        roi=roi, 
        generator="edge",
        selector='hough', 
        estimator='ols',
        degree=2,
        scaler_type="z_score",
        P_primer=0.99,
        process_noise="high",
        min_inliers=0.6,
        max_error=5,
        ksize=3,
        small_ksize=5,
        large_ksize=15,
        n_std=2.0,
        use_bev=True,
        forward_range=40.0,
        lateral_range=7.0,
        resolution=0.05
    )
    print(system.studio.source.width)
    print(system.studio.source.height)
    eval = system.run("inset", stroke=False, fill=True)

    print(eval)

if __name__=="__main__":
    demo_video()