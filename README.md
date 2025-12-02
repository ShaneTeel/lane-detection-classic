# Lane Detection
## Description / Overview
A classic lane line detection system that employs the following techniques:

- HSL-channel Masking
- Edge Detection
- Inverse ROI masking
- Probabilistic Hough Lines Transform or pixel-wise point extraction 
- Homography (Bird's Eye View) using a manually-calculated H-matrix
- Regression using a manually-built OLS and RANSAC
- Temporal smoothing using a manually-calculated Kalman filter

Built to demonstrate understanding of CV pipelines without deep learning and without the use of camera calibration data. 

## Demo

![Demo](media/out/readme/curved-edge-direct-demo.gif)

See full video here: [Curved Road Lane Line Detection w/ Edge Map](https://youtu.be/AOmAQo3oTFU)

## Quick Start
### Install Package
```bash
git clone https://github.com/ShaneTeel/lane-detection-classic.git
cd lane-detection-classic
pip install -e .
```
### Run Demo Scripts
**Straight Lane Video**
```
python scripts/straight/straight_edge_direct_demo.py
```
**Curved Lane Video**
```
python scripts/curved/curved_edge_direct_demo.py
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
    selector="direct",
    estimator="ols"
)

report = system.run("composite", stroke=False, fill=True)

print(report)
```
## Methodology

**Feature Generation**

**ROI Masking**

**Feature Selection / Extraction** 

**BEV Projection** (Optional)

**Polynomial Fit**

**Kalman Filter**

**Visualization**

## Project Structure
```
lane_detection/
|-- detection/           # Main pipeline
│   |-- models/          # OLS, RANSAC, Kalman
|-- feature_generation/  # Edge/threshold maps
|-- feature_selection/   # Point extraction
|-- scalers/             # MinMax, StandardScaler
|-- image_geometry/      # ROI mask, BEV projection
|-- studio/              # Visualization
```

## Trade-Offs
**RANSAC vs OLS**
- RANSAC struggles with curved roads. Additionally, as the polynomial degree increases, the minimum sample size needed typically results in an unstable fit. Lastly, RANSAC can reduce computational speed.
- OLS is not very resistent to outliers and requires quality feature generation / selection to ensure proper application.

**Hough vs Direct**
- Probabilistic Hough Lines Transform performs struggles with curved roads. If BEV Transform were applied prior to `cv2.HoughLinesP()`, this issue is likely mitigated, but requires camera parameters (not included in this exercise).
- The direct approach is much less resilient to outliers and requires special attention to the `n_std` argument to ensure outliers are filtered out appropriately.

**Thresh vs Edge**
- The thresh map amplifies both good pixel coordinates and bad pixel coordinates, but is useful when the actual lane lines are faded / worn as an edge detection approach would return too few points to produce a good fit.
- The vertical edge map rejects noise resulting from horizontal lines, but can produce too few points to produce the right fit when the actual lane lines are faded / worn. 

**BEV** (Optional)
- BEV projection aids in generating polylines that conform to the actual lane line locations, but will reduce computation speed.
- Deviates from modern approaches that perform BEV projection prior to feature selection.
- Requires camera parameters (not included in this exercise) to improve effectiveness.
- Does not un-distort the image due to a lack of camera parameters / calibration calculation.

## Limitations

- Struggles with heavy road-noise (i.e., overpasses, road construction change (asphalt --> concrete))
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
- Add notebooks to outline the application of each step (e.g., feature generation, feature selection, regression, etc.)
- Add unit tests for critical modules (e.g., Kalman, RANSAC, OLS)