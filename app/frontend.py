import cv2
import numpy as np
import streamlit as st
import tempfile
from streamlit_image_coordinates import streamlit_image_coordinates as img_xy
from PIL import Image, ImageDraw

from lane_detection.detection import DetectionSystem
from lane_detection.image_geometry import ROIMasker
from lane_detection.studio import StudioManager

st.set_page_config(layout="wide")

# Helper Funcs
def add_point():
    if len(st.session_state['click_points']) == 4:
        return
    else:
        raw = st.session_state['point']
        point = raw['x'], raw['y']
        st.session_state['click_points'].append(point)

# Destructor Attributes
if 'reset' not in st.session_state:
    st.session_state['reset'] = True
if 'container_lst' not in st.session_state:
    st.session_state['container_lst'] = []

# Source Attributes
if 'file' not in st.session_state:
    st.session_state['file'] = None
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None

# ROI Attributes
if 'roi_window' not in st.session_state:
    st.session_state['roi_window'] = None
if 'roi_frame' not in st.session_state:
    st.session_state['roi_frame'] = None
if 'point' not in st.session_state:
    st.session_state['points'] = None
if 'click_points' not in st.session_state:
    st.session_state['click_points'] = []
if 'poly_img' not in st.session_state:
    st.session_state['poly_img'] = None

# Set title
st.title("Classic Lane-Line Detection Demo")

# Create file upload button
st.header("Source Management")
cols0 = st.columns(2)
with cols0[0]:
    st.subheader("Video Upload")
    uploaded_file = st.file_uploader("Choose a source file:", type=['mp4', 'mov', 'mkv', 'avi'], key=f"file_uploader_{st.session_state['reset']}")
    
with cols0[1]:
    st.subheader("Video Release")
    st.write("Select Before Exiting")
    release = st.button("Release", help="Release all capture objects and empty all containers", type='primary')

st.divider()

# File Upload Block 
if uploaded_file is not None and uploaded_file != st.session_state["uploaded_file"]:
    st.spinner("Reading file...")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp:
            temp.write(uploaded_file.read())
            st.session_state["file"] = temp.name
        studio = StudioManager(st.session_state["file"])
        ret, frame = studio.return_frame()
        st.session_state["roi_frame"] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

with st.sidebar:
    run = st.button("Run Detection", type="secondary")
    st.title("System Configuration")
    st.subheader("Feature Generation")

    # Feature Generation Input
    st.markdown("#### Feature Generation")
    feature_gen = st.radio("Select Feature Generator", ["edge", "thresh"], horizontal=True, index=0)
    if feature_gen == "edge":
        ksize = st.select_slider("Blur & Canny Kernel Size", [3, 5, 7, 9, 11, 13, 15], value=5)
    else:
        thresh_cols = st.columns(2)
        with thresh_cols[0]:
            small_ksize = st.select_slider("Close Kernel Size", [3, 5, 7, 9, 11, 13, 15], value=5)
        with thresh_cols[1]:
            large_ksize = st.select_slider("Dilate Kernel Size", [11, 13, 15, 17, 19, 21], value=15)

    # Feature Selection Input
    st.markdown("#### Feature Selection")
    feature_sel = st.radio("Select Feature Extractor", ["hough", "direct"], horizontal=True, index=1)
    if feature_sel == "hough":
        hough_cols = st.columns(3)
        h, w = st.session_state["roi_frame"].size if st.session_state["roi_frame"] is not None else 0
        diag = (w**2 + h**2)**0.5
        area = w * h
        with hough_cols[0]:
            min_votes = st.number_input("Min. Votes", min_value=1, max_value=area, value=50)
        with hough_cols[1]:
            min_length = st.number_input("Min. Line Length", min_value=1, max_value=int(diag), value=10)
        with hough_cols[2]:
            max_gap = st.number_input("Max. Line Gap", min_value=0, max_value=int(diag), value=20)
    else:
        n_std = st.number_input("Number of Standard Deviations to Filter", min_value=0.5, max_value=10.0, value=2.0)

    # Modeling Configs
    st.markdown("#### Dynamic Linear Modeling")
    st.markdown("##### Transformation Options")
    scaler = st.radio("Select Scaler", ["min_max", "z_score"], horizontal=True, index=1)
    use_bev = st.radio("Use BEV?", [True, False], horizontal=True, index=0)
    if use_bev:
        bev_cols = st.columns(3)
        with bev_cols[0]:
            forward_range = st.number_input("Forward", min_value=0.0, max_value=100.0, value=40.0)
        with bev_cols[1]:
            lateral_range = st.number_input("Lateral", min_value=0.0, max_value=100.0, value=7.0)
        with bev_cols[2]:
            resolution = st.number_input("GSD", min_value=0.01, max_value=1.0, value=0.03)

    st.markdown("##### Estimator Options")
    estimator = st.radio(label="Select Estimator", options=["ols", "ransac"], horizontal=True, index=0)
    degree = st.radio("Select Degree", [1, 2, 3], horizontal=True, index=1)
    if estimator == "ransac":
        ransac_cols = st.columns(3)
        with ransac_cols[0]:
            confidence = st.number_input("Probability", min_value=0.0, max_value=1.0, value=0.5)
        with ransac_cols[1]:
            min_inliers = st.number_input("Min. Inliers", min_value=0.0, max_value=1.0, value=0.8)
        with ransac_cols[2]:
            max_error = st.number_input("Max Error", min_value=0, value=10)

    st.markdown("##### Kalman Options")
    P_primer = st.number_input("Initial Confidence", min_value=0.0, max_value=0.99, value=0.5)
    process_noise = st.radio("Process Noise (Environment)", ["low", "medium", "high"], horizontal=True)

    system_configs = {
        "generator": feature_gen,
        "selector": feature_sel,
        "scaler_type": scaler,
        "estimator": estimator,
        "degree": degree,
        "P_primer": P_primer,
        "process_noise": process_noise,
        "use_bev": use_bev
    }

    if feature_gen == "edge":
        system_configs.update({"ksize": ksize})
    else:
        system_configs.update({"small_ksize": small_ksize, "large_ksize": large_ksize})
    if feature_sel == "direct":
        system_configs.update({"n_std": n_std})
    else:
        system_configs.update({"min_votes": min_votes, "min_length": min_length, "max_gap": max_gap})
    if estimator == "RANSAC":
        system_configs.update({"confidence": confidence, "min_inliers": min_inliers, "max_error": max_error})
    if use_bev:
        system_configs.update({"forward_range": forward_range, "lateral_range": lateral_range, "resolution": resolution})

    st.session_state["configs"] = system_configs

st.set_page_config(initial_sidebar_state="expanded")

if st.session_state['file'] is not None:
    st.subheader("Media Viewer")
    st.markdown("**Move cursor over image and right-click at four different points on the image.**")
    st.write(" ")

    viewer_cols = st.columns([2, 3])
    with viewer_cols[0]:
        st.markdown("##### ROI Options")
        reset = st.button("Reset", type='primary')
    with viewer_cols[1]:
        st.markdown("##### View Options")
        view_options = ['inset', 'mosaic', "composite"]
        run_cols = st.columns(2)
        with run_cols[0]:
            view_selection = st.segmented_control("Render Options", view_options, label_visibility="collapsed")

    # Create ROI Window
    if st.session_state['roi_window'] is None:
        st.session_state['roi_window'] = st.empty()
        st.session_state['container_lst'].append(st.session_state['roi_window'])
    
    # Create writeable ROI Frame
    if st.session_state.get("roi_frame") is not None:
        try:
            if st.session_state['poly_img'] is None:
                poly_img = st.session_state.get("roi_frame")
                st.session_state['poly_img'] = poly_img.copy()
            
            img_draw = st.session_state.get("poly_img")
            draw = ImageDraw.Draw(img_draw)

            if len(st.session_state["click_points"]) > 0:
                for pt in st.session_state["click_points"]:
                    draw.ellipse([pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5], (255, 255, 0))

            if len(st.session_state["click_points"]) == 4:
                points = np.array(st.session_state["click_points"]).reshape(-1, 4, 2)
                mask = ROIMasker(points)
                poly_lst = mask.src_pts.tolist()
                poly = [(x, y) for point in poly_lst for x, y in point]
                draw.polygon(poly, outline=(255, 255, 0), width=5)

            value = img_xy(img_draw, key='point', on_click=add_point, cursor='crosshair')

            if reset:
                st.session_state['click_points'].clear()
                st.session_state['poly_img'] = st.session_state['roi_frame'].copy()
                st.rerun()

        except Exception as e:
            st.error(f"Error preparing image coordinates: {str(e)}")

    if release:
        st.session_state['reset'] = not st.session_state['reset']
        st.session_state['file'] = None
        st.session_state["poly_img"] = None
        st.session_state["click_points"].clear()
        if studio:
            studio.clean._clean_up()
        for container in st.session_state['container_lst']:
            container.empty()
        st.rerun()