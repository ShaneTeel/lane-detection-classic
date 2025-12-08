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
st.set_page_config(layout='wide')

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
    st.header("Step 1: Configure Model")
    cols1 = st.columns(2, border=True)

    # ROI Sub-Block
    with cols1[0]:
        cols1P = st.columns([9, 1])

        with cols1P[0]:
            st.subheader("Select a Region of Interest* (ROI)")
            st.markdown("**Move cursor over image and right-click at four different points on the image.**")
            st.write("**The ROI is the area of the frame that the Lane Detection model is run against.*")
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
                    w, h = poly_img.size
                    st.session_state['poly_img'] = poly_img.copy()
                
                img_draw = st.session_state.get("poly_img")

                draw = ImageDraw.Draw(img_draw)

                value = img_xy(img_draw, key='point', on_click=add_point, cursor='crosshair')

                if len(st.session_state['click_points']) == 4:
                    points = st.session_state.get('click_points')
                    roi_payload = {"points": points, "method": "original"}
                    try:
                        response = requests.post(f"{BACKEND_URL}/roi", json=roi_payload)
                        response.raise_for_status()
                        poly_lst = response.json().get("poly")
                        orig_poly = [(x, y) for x, y in [point for point in poly_lst]]
                        st.session_state['roi_poly'] = orig_poly
                        draw.polygon(orig_poly, outline=(255, 255, 0), width=5)
                        if not st.session_state['roi_rerun']:
                            st.session_state['roi_rerun'] = True
                            st.rerun()
                        
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

        # ROI Selection
        cols1A = st.columns(2)
        with cols1A[0]:
            st.markdown("**Selected ROI**")
            if st.session_state['roi_poly'] is not None:
                names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
                roi_poly = st.session_state.get("roi_poly")
                roi_poly = {name: point for name, point in zip(names, points)}
                df = pd.DataFrame(roi_poly)
                df.index = ["x", "y"]
                st.table(df, border='horizontal')

        # Thresholding Input (cv2.inRange())
        with cols1A[1]:
            st.markdown("**Thresholding**")
            lower_bounds, upper_bounds = st.slider("Lower / Upper (Inclusive)", min_value=0, max_value=255, value=(150, 255))

        # Canny Input (cv2.Canny())
        st.markdown("**Canny Edge Detection**")
        cols1B = st.columns(2)
        with cols1B[0]:
            blur_first = st.selectbox("Gaussian Blur", ["Before Canny", "After Canny", "No Blur"], help="Whether to perform Gaussian Blur before edge detection, after edge detection, or not at all.")
        with cols1B[1]:
            canny_low, canny_high = st.slider("Weak / Sure Edge (Inclusive)", min_value=0, max_value=300, value=(50, 150))

        # Hough Input (cv2.HoughLinesP())
        st.markdown("**Probabilistic Hough Line Transform**")
        cols1C = st.columns(5)
        with cols1C[0]:
            w, h, = st.session_state['roi_frame'].size
            diag = (w**2 + h**2)**0.5
            area = w * h
            rho = st.number_input("Rho (ρ)", min_value=0.1, max_value=diag, value=1.0)
        with cols1C[1]:
            theta = st.number_input("Theta (θ)", min_value=0, max_value=180, value=180, help="Value will be divided by π once passed to model.")
        with cols1C[2]:
            min_votes = st.number_input("Threshold", min_value=1, max_value=area, value=50)
        with cols1C[3]:
            min_line_length = st.number_input("Min. Line Length", min_value=1, max_value=int(diag), value=10)
        with cols1C[4]:
            max_line_gap = st.number_input("Max Line Gap", min_value=0, max_value=int(diag), value=20)
        
        st.markdown("**Composite Styling***")
        cols1D = st.columns(4)
        with cols1D[0]:
            stroke_bool = st.checkbox("Draw Lane Lines (Stroke)", value=True)
        with cols1D[1]:
            if stroke_bool:
                stroke_color = st.color_picker("Stroke Color", value="#FF0000")
        with cols1D[2]:
            fill_bool = st.checkbox("Draw Lane Area (Fill)", value=True)
        with cols1D[3]:
            if fill_bool:
                fill_color = st.color_picker("Fill Color", value="#00FF00")
        cols1E = st.columns(3)
        with cols1E[0]:
            st.write("*Style options do not affect the algorithm.")
        with cols1E[2]:
            configure = st.button("Configure System", type='secondary')
    if configure:

        processor_configs = {
                "in_range": {
                    "lower_bounds": lower_bounds, 
                    "upper_bounds": upper_bounds
                },
                "canny": {
                    "canny_low": canny_low, 
                    "canny_high": canny_high, 
                    "blur_first": blur_first
                },
                "hough": {
                    "rho": rho, 
                    "theta": np.pi / theta, 
                    "thresh": min_votes, 
                    "min_length": min_line_length, 
                    "max_gap": max_line_gap
                },
                "composite": {
            
                    "stroke": True if stroke_bool else False, 
                    "stroke_color": stroke_color if stroke_bool else "#FFFFFF",
                    "fill": True if fill_bool else False,
                    "fill_color": fill_color if fill_bool else "#FFFFFF"
                }
            }
        st.session_state["processor_configs"] = processor_configs
        with st.spinner("Configuring processor..."):
            try:
                with requests.post(f"{BACKEND_URL}/configure", json=processor_configs) as r:
                    r.raise_for_status()

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to processing service: {str(e)}.")
                st.warning(f"Make sure backend service is running at {BACKEND_URL}.")

    view_options = ['Step-by-Step', 'Composite Only']
    st.divider()
    st.header("Step 2: Inspect & Evaluate")
    cols2 = st.columns(2, border=True)
    with cols2[0]:
        cols2A = st.columns(2)
        with cols2A[0]:
            st.subheader("Visual Inspection")
        with cols2A[1]:
            view_selection = st.segmented_control("Render Options", view_options)
        cols2B = st.columns(2)
        with cols2B[0]:
            run = st.button("Process Video")
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