import cv2
import os
import numpy as np
import pandas as pd
import time
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates as img_xy
from PIL import Image, ImageDraw
import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")

# Helper Funcs
def add_point():
    if len(st.session_state['click_points']) == 4:
        return
    else:
        raw = st.session_state['point']
        point = raw['x'], raw['y']
        st.session_state['click_points'].append(point)

# Page Layout
st.set_page_config(layout="wide")

# Destructor Attributes
if 'reset' not in st.session_state:
    st.session_state['reset'] = True
if 'container_lst' not in st.session_state:
    st.session_state['container_lst'] = []

# Source Attributes
if 'file' not in st.session_state:
    st.session_state['file'] = None

# ROI Attributes
if 'roi_window' not in st.session_state:
    st.session_state['roi_window'] = None
if 'roi_frame' not in st.session_state:
    st.session_state['roi_frame'] = None
if 'point' not in st.session_state:
    st.session_state['points'] = None
if 'click_points' not in st.session_state:
    st.session_state['click_points'] = []
if 'roi_poly' not in st.session_state:
    st.session_state['roi_poly'] = None
if 'roi_rerun' not in st.session_state:
    st.session_state['roi_rerun'] = False
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
if uploaded_file is not None and uploaded_file != st.session_state['file']:
    st.spinner("Initializing capture object...")
    st.session_state['file'] = uploaded_file
    file = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    try:
        response = requests.post(f"{BACKEND_URL}/initialize", files=file)
        response.raise_for_status()
        arr = np.frombuffer(response.content, np.uint8)
        if arr is not None:
            frame = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if frame is not None and frame.size > 0:
                rgb_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.session_state['roi_frame'] = rgb_img
            else:
                st.error(f"Error: Image return from server is none type object or of shape 0.")
        else:
            st.error(f"Error: server failed to read frame from source.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to intialization service: {str(e)}.")
        st.warning(f"Make sure backend service is running at {BACKEND_URL}.")

# Model Configuration Block
if st.session_state['file'] is not None:
    st.header("Step 1: Configure System")
    cols1 = st.columns(2, border=True)

    # ROI Sub-Block
    with cols1[0]:
        cols1P = st.columns([9, 2])

        with cols1P[0]:
            st.subheader("Select a Region of Interest* (ROI)")
            st.markdown("**Move cursor over image and right-click at four different points on the image.**")
        with cols1P[1]:
            st.write(" ")
            reset = st.button("Reset", type='primary')

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

                value = img_xy(img_draw, key='point', on_click=add_point, cursor='crosshair')

                if len(st.session_state['click_points']) == 4:
                    points = st.session_state.get('click_points')
                    roi_payload = {"points": points}
                    try:
                        response = requests.post(f"{BACKEND_URL}/roi", json=roi_payload)
                        response.raise_for_status()
                        poly_lst = response.json().get("poly")

                        orig_poly = [(x, y) for point in poly_lst for x, y in point]
                        st.session_state['roi_poly'] = orig_poly
                        draw.polygon(orig_poly, outline=(255, 255, 0), width=5)
                        if not st.session_state['roi_rerun']:
                            st.session_state['roi_rerun'] = True
                            st.rerun()

                        st.markdown("**Selected ROI**")
                        if st.session_state['roi_poly'] is not None:
                            names = ["Bottom-Left", "Bottom-Right", "Top-Left", "Top-Right"]
                            roi_poly = {name: point for name, point in zip(names, orig_poly)}
                            df = pd.DataFrame(roi_poly)
                            df.index = ["x", "y"]
                            st.table(df, border='horizontal')
                        
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error connecting to Original ROI service: {str(e)}")
                        st.warning(f"Make sure backend service is running at {BACKEND_URL}.")

                if reset:
                    st.session_state['click_points'].clear()
                    st.session_state['roi_poly'] = None
                    st.session_state['poly_img'] = st.session_state['roi_frame'].copy()
                    if st.session_state['roi_rerun']:
                        st.session_state['roi_rerun'] = False
                        st.rerun()

            except Exception as e:
                st.error(f"Error preparing image coordinates: {str(e)}")

    # Parameter Block
    with cols1[1]:
        st.subheader("Set Parameters")
        feature_cols = st.columns(2)

        # Feature Generation Input
        with feature_cols[0]:
            st.markdown("##### Feature Generation")
            feature_gen = st.radio("Select Feature Generator", ["edge", "thresh"], horizontal=True, index=0)
            if feature_gen == "edge":
                ksize = st.select_slider("Blur & Canny Kernel Size", [3, 5, 7, 9, 11, 13, 15], value=5)
            else:
                cols1Ab = st.columns(2)
                with cols1Ab[0]:
                    small_ksize = st.select_slider("Close Kernel Size", [3, 5, 7, 9, 11, 13, 15], value=5)
                with cols1Ab[1]:
                    large_ksize = st.select_slider("Dilate Kernel Size", [11, 13, 15, 17, 19, 21], value=15)

        # Feature Selection Input
        with feature_cols[1]:
            st.markdown("##### Feature Selection")
            feature_sel = st.radio("Select Feature Extractor", ["hough", "direct"], horizontal=True, index=1)
            if feature_sel == "hough":
                hough_cols = st.columns(3)
                h, w = st.session_state["roi_frame"].size
                diag = (w**2 + h**2)**0.5
                area = w * h
                with hough_cols[0]:
                    min_votes = st.number_input("Min. Votes", min_value=1, max_value=area, value=50)
                with hough_cols[1]:
                    min_length = st.number_input("Min. Line Length", min_value=1, max_value=int(diag), value=10)
                with hough_cols[2]:
                    max_gap = st.number_input("Max Line Gap", min_value=0, max_value=int(diag), value=20)
            else:
                n_std = st.number_input("Number of Standard Deviations to Filter", min_value=0.5, max_value=10.0, value=2.0)        
        st.markdown("##### Dynamic Linear Modeling")
        model_cols = st.columns(3)
        with model_cols[0]:
            st.caption("Transformation Options")
            scaler = st.radio("Select Scaler", ["min_max", "z_score"], horizontal=True, index=1)
            use_bev = st.radio("Use BEV?", [True, False], horizontal=True, index=0)
            if use_bev:
                bev_cols = st.columns(3)
                with bev_cols[0]:
                    forward_range = st.number_input("Forward", min_value=0.0, max_value=100.0, value=40.0)
                with bev_cols[1]:
                    lateral_range = st.number_input("Lateral", min_value=0.0, max_value=100.0, value=7.0)
                with bev_cols[2]:
                    resolution = st.number_input("GSD", min_value=0.01, max_value=1.0, value=0.3)

        with model_cols[1]:
            st.caption("Estimator Options")
            estimator = st.radio(label="Select Estimator", options=["ols", "ransac"], horizontal=True, index=0)
            degree = st.radio("Select Degree", [1, 2, 3], horizontal=True, index=2)
            if estimator == "ransac":
                ransac_cols = st.columns(3)
                with ransac_cols[0]:
                    confidence = st.number_input("Probability", min_value=0.0, max_value=1.0, value=0.5)
                with ransac_cols[1]:
                    min_inliers = st.number_input("Min. Inliers", min_value=0.0, max_value=1.0, value=0.8)
                with ransac_cols[2]:
                    max_error = st.number_input("Max Error", min_value=0, value=10)

        with model_cols[2]:
            st.caption("Kalman Options")
            P_primer = st.number_input("Initial Confidence", min_value=0.0, max_value=0.99, value=0.5)
            process_noise = st.radio("Process Noise (Environment)", ["low", "medium", "high"], horizontal=True)
        st.text("")
        configure = st.button("Configure", type='secondary')

    if configure:
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
        with st.spinner("Configuring system..."):
            try:
                with requests.post(f"{BACKEND_URL}/configure", json=system_configs) as r:
                    r.raise_for_status()

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to processing service: {str(e)}.")
                st.warning(f"Make sure backend service is running at {BACKEND_URL}.")

    view_options = ['inset', 'mosaic', "composite"]
    st.divider()
    st.header("Step 2: Run & Evaluate")
    cols2 = st.columns(2, border=True)
    with cols2[0]:
        cols2A = st.columns(2)
        with cols2A[0]:
            st.subheader("Visual Inspection")
        with cols2A[1]:
            view_selection = st.segmented_control("Render Options", view_options)
        cols2B = st.columns(2)
        with cols2B[0]:
            run = st.button("Run Detection")
        with cols2B[1]:
            stop = st.button("Stop", type='primary')
        if run and not view_selection:
            st.error("You must select a viewing option prior to processing video.")
        elif run and view_selection:
            stream_window = st.empty()
            stream_url = f"{BACKEND_URL}/stream_video?timestamp={time.time()}&style={view_selection.replace(' ', '%20')}"
            w, h = st.session_state['roi_frame'].size
            stream_window.markdown(
                f"""
                <img src="{stream_url}" style="width: 100%; height: 100%;" />
                """,
                unsafe_allow_html=True,
                width="content"
            )
        if stop:
            try:
                r = requests.post(f"{BACKEND_URL}/stop_video", json=False)
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to streaming service: {str(e)}.")
                st.warning(f"Make sure backend service is running at {BACKEND_URL}.")  

            st.write(" ")

    with cols2[1]:
        st.subheader("Evaluation Report")
    

    if release:
        st.session_state['reset'] = not st.session_state['reset']
        st.session_state['file'] = None
        st.session_state['cap'].release()
        for container in st.session_state['container_lst']:
            container.empty()
        st.rerun()