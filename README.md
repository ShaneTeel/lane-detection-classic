# Lane Detection

Traditional computer vision lane detection using edge detection, polynomial fitting, and Kalman filtering. Built to demonstrate understanding of CV pipelines without deep learning. 

![Demo](media/out/lane1-straight-processed-vid.gif)

See full video here: [lane1-straight-processed-vid](https://youtu.be/QkCduVmN1P8)

## What It Does

Detects lane lines in road videos using:
- Converts image/frame from BGR to HSL
- Performs thresholding using both the Saturation and Lightness channels, then merges results
- Sobel edge detection on thresholding results OR performs morphology to generate a threshold-based feature map
- Inverse ROI masking to reduce noise prior to fitting
- Manual OLS / manual RANSAC polynomial fitting (outlier rejection)
- Manually-calculated Kalman filtering for temporal smoothing
- Optional bird's-eye view transformation; occurs post image preprocessing, but prior to polynomial fitting

## Quick Start
```bash
git clones https://github.com/ShaneTeel/lane-detection-classic.git
cd lane-detection-classic
pip install -e .
python scripts/demo_video.py
```
### For Single Video / Image processing
Define your ROI and run:
```python
from lane_detection.detection import DetectionSystem
import numpy as np

roi = np.array([[[100, 540], [900, 540], [525, 325], [445, 325]]])

system = DetectionSystem(
    source=<"filepath to video goes here">,
    roi=roi,
    generator="edge",
    selector="hough",
    estimator="ransac"
)

report = system.run("composite", fill=True)

print(report)
```

### For Grid Search Processing
Define your ROI and run:


## How It Works

1. **Feature Generation** - Create edge map or threshold map
2. **ROI Masking** - Focus on road area
3. **Point Selection** - Extract lane points (direct or Hough)
4. **Polynomial Fit** - OLS or RANSAC regression
5. **Kalman Filter** - Smooth across frames
6. **Visualization** - Draw detected lanes

## Project Structure
```
lane_detection/
├── detection/           # Main pipeline
│   ├── models/          # OLS, RANSAC, Kalman
│   └── ...
├── feature_generation/  # Edge/threshold maps
├── feature_selection/   # Point extraction
├── scalers/             # MinMax, StandardScaler
├── image_geometry/      # ROI mask, BEV projection
└── studio/              # Visualization
```

## Limitations

- Struggles with heavy shadows, pixel intensity changes mid-road (i.e., transitions from asphalt to concrete mid-lane)
- Doesn't handle construction zones well
- RANSAC + BEV is slow on high-res video
- Requires manual ROI selection

## Why Not Deep Learning?

This project demonstrates:
- Understanding of classical CV techniques
- Ability to build modular systems
- Knowledge of robust regression and filtering

This project is not meant to challenge modern approaches. It is used as a learning exercise to advance author's understanding of the following:
- Classic computer vision models
- Basic preprocessing steps
- Manual application of Kalman, Homography, and Regression.

## To-Do
1. Add logger
2. Add notebooks outline the application of each step (e.g., feature generation, feature selection, regression, etc.)
3. Add unit tests for crtiical modules