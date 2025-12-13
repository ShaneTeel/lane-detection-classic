# Lane-Line Detection

A classical approach to lane line detection featuring:
- Manual Kalman-filtering
- RANSAC/OLS Polynomial Regression
- Homography-calculation and projection
- Streamlit app for demonstration

## Table-of-Contents
- [Demo / Examples](#demo--examples)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Methodology](#methodlogy)
- [Trade-Offs](#trade-offs)
- [Classical vs. Deep Learning](#classical-vs-deep-learning)
- [To-Do](#to-do)

## Demo / Examples
### Curved Road Example w/ Results
**Visual Output**

![Curved Example](media/out/readme/curved-edge-direct-demo.gif)

**Performance Evaluation**


```html
Regression
|--------------------------------------------|
| Metric   |   Left    |   Right  |   Avg.   |
|--------------------------------------------|
| R2       |  0.9806   |  0.9859  |  0.9832  |
| RMSE     |  10.1696  |  8.5702  |  9.3699  |
| MAE      |  8.4617   |  7.0495  |  7.7556  |
|--------------------------------------------|
```

See full video here: [Curved Road Lane Line Detection w/ Edge Map](https://youtu.be/AOmAQo3oTFU)

[Return to TOC](#table-of-contents)

## Key Features
### From-Scratch Implementation
- Kalman filter with adapative measurements for noise
- RANSAC with dynamic iteration calculation
- Homography via Direct Linear Transformation
    - No camera calibration or parameters required

### Production Engineering
- Modular architecture with interchangeable steps
- Pydantic paramter configuration validation
- Comprehensive logging
- Grid search hyperparamter optimization

### Flexible Pipeline
- Customizable approaches (edge/thresh, direct/hough, ols/ransac)
- Optional bird's eye view projection of extracted features
- Real-time video processing with temporal smoothing

### Performance
- **R2 Score**: 0.94-0.99, configuration dependent
- **Tested on**:
    - Straight roads
    - Curved roads
    - Worn lane lines
    - Variable lighting

[Return to TOC](#table-of-contents)

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
[Return to TOC](#table-of-contents)


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
[Return to TOC](#table-of-contents)

## Methodology

### Pipeline Overview
```mermaid 
---
title: Lane Line Detection Stages
id: eb8afec7-e8d6-443b-9df4-357dccd01d6c
---
flowchart LR;
    A([Read Image / Video]) --> B;
    subgraph Feature Generation;
        B[HSL-Masking] --> |Generator A| CA;
        CA["Threshold + Morphology<br>(Close --> Dilate)"] --> D;
        B[HSL-Masking] --> |Generator B| CB;
        CB["Vertical Edge Detection<br>(Sobel-X)"] --> D
        end
    D[Inverse ROI-Masking] --> |Extractor A| EA;
    D[Inverse ROI-Masking] --> |Extractor B| EB;
    subgraph Feature Transformation;
        EA[Probabilistic Hough Lines Transform] --> F;
        EB[Direct Pixel-Wise Extraction] --> F;
        F{BEV?} --> |Yes| G;
        F{BEV?} --> |No| H;
        G[Perspective Transform] --> H;
        end
    H[Feature Scaling] --> |Estimator A| IA;
    H[Feature Scaling] --> |Estimator B| IB
    subgraph Dynamic Linear Modeling
        IA["Outlier-Rejection Curve Fitting<br>(RANSAC)"] --> J;
        IB["Outlier-Sensitive Curve Fitting<br>(OLS)"] --> J;
        J["Temporal Lane Tracking<br>(Kalman-Filter)"] --> K;
        end
    K[Extrapolated Lane-Line Prediction] --> L;
    L([Visualization]);
```

[Return to TOC](#table-of-contents)

## Trade-Offs
**Feature Generation**

*Thresh*
- Amplifies both good pixel coordinates and bad pixel coordinates.
- Useful when the actual lane lines are faded / worn.

*Edge*
- Rejects noise resulting from horizontal lines
- Can generate too few points; not enough features to generate the right fit. 

**Feature Selection**

*Hough*
- Struggles with curved roads. 
- If BEV Transform were applied prior to `cv2.HoughLinesP()`, this issue is likely mitigated, but requires camera parameters (not included in this exercise).

*Direct*
- Much less resilient to outliers
- Requires special attention to the `n_std` argument to ensure outliers are filtered out appropriately.

**Estimators**

*RANSAC*
- Struggles with curved roads. 
- As polynomial degree increases, the minimum sample size needed results in an unstable fit. 
- Can reduce computational speed.

*OLS*
- Not very resistent to outliers.
- Requires a more deliberate feature generation / selection to ensure proper outliers filtering.

**BEV** (Optional)
- Aids in generating polylines that conform to the actual lane line locations.
- Reduces computational speed.
- Requires camera parameters (not included in this exercise) to improve use.

### Limitations

- Struggles with heavy road-noise (i.e., overpasses, road construction change (asphalt --> concrete))
- Requires manual ROI selection

[Return to TOC](#table-of-contents)

## Classical vs. Deep Learning
This project demonstrates **fundamental understanding** of classical computer vision techniques.

### Benefits of Classical CV

**The User Can Learn**:
- Coordinate transformation (homography)
- State estimation and filtering (Kalman)
- Robust regression techniques (outlier-rejection w/ RANSAC)
- Production system design (modularity, testing, logging, etc.)

**When to Use Classical**:
- System is resource constrained
- Interpretebility is critical
- Edge deployments, or more to the point, when a GPU is not needed
- Edge cases that challenge failure response

**Coming Soon!**: Comparitive analysis with YOLOv8 implementation
- Following implementation, author will update with a more thorough examination of the two approaches (classic vs deep learning).


[Return to TOC](#table-of-contents)

## To-Do
- Add unit tests for critical modules (e.g., Kalman, RANSAC, OLS, BEV/Homography).

[Return to TOC](#table-of-contents)