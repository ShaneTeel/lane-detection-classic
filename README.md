# Lane-Line Detection Demo App

Interactive demo for the [Classic Lane Line Detection](https://github.com/ShaneTeel/lane-detection-classic/tree/main) project

## Try it Live

[Launch Demo](https://lane-detection-classic-v1.streamlit.app/)

## Table-of-Contents
- [Key Features](#key-features)
- [Guided Walk-Through](#guided-walk-through)
- [Configuration Options](#configuration-options)
- [Local Setup](#local-setup)
- [Troubleshooting](#troubleshooting)

## Key Features
- **Interactive ROI selection** - Used to define a region of interest within which to apply the detection
- **Configurable pipeline** - Adjust paramers for feature generation, selection, and modeling
- **Multiple View Styles** - Choose from three viewing options that allow users to inspect the results of their configuration at each stage of the pipeline
- **Real-Time Visualization** - Allows for immediate inspection of results

## Guided Walk-Through

### Video Demonstration

### Step-by-Step Instructions
1. **Upload a Video File**
2. **Select ROI** - Move cursor over image and click 4 points to define a region of interest
3. **Configure System** - Adjust paramters in sidebar
4. **Run Detection** - Click `Run Detection` to process video

## Configuration Options

### Feature Generation
- **Edge Detection** - Sobel-based vertical edge detection (generates an edge-map as a feature map)
- **Thresholding** - Performs morphological operations on HSL-channel masked image

### Feature Extraction / Selection
- **Hough** - Probabilistic Hough Lines Transform via `cv2.HoughLinesP()`
- **Direct** - Pixel-wise extraction with outlier filtering

### Feature Transformation
- **Scaler** - Z-Score or Min-Max scaling
- **BEV (Homography)** - Optional bird's eye view projection

### Dynamic Linear Modeling
- **Estimator** - OLS or RANSAC regression
- **Degree** - 1, 2, or 3 degree polynomial
- **Kalman Filter** - Temporal lane tracker / smoother with adaptive noise

## Local Setup
- Python 3.10+

### App Installation Instructions
**All commands below start from the repositories root directory**

```bash
python -m pip install -r requirements.txt

streamlit run ./app/demo.py
```

## Troubleshooting
### Video Failed to Upload
- **Max File Size** - 200MB
- **Unsupported File Type** - MP4, MKV, MOV, AVI

### Detection Fails
- **ROI Primacy** - Ensure the ROI caputures the location of the actual lane lines
- **System Configs**
    - Adjust current configs to be either more or less restrictive
    - Use a different generator, selector, or estimator

### Slow Processing
- **RANSAC** - Assists in filtering outliers, but can slow processing speed significantly. If ROI, generator, and selector are configured appropriately, RANSAC may not be necessary
- **BEV** - Disable for improved processing spped
- **Edge Detection** - Edge detection performs fewer convolutions than the thresholding feature generator.  