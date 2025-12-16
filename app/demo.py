# Display / Image operations
import cv2
from PIL import Image, ImageDraw
import numpy as np

# App design
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates as img_xy
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Read / Write
import tempfile
import queue
import time

# Required lane detection modules
from lane_detection.image_geometry import ROIMasker
from lane_detection.studio import StudioManager

# Streamlit specific Detection System wrapper
from rt_video_processor import RTVideoProcessor

# Destructor Attributes
if 'reset' not in st.session_state:
    st.session_state['reset'] = True
if 'container_lst' not in st.session_state:
    st.session_state['container_lst'] = []

# Source Attributes
if 'file_in' not in st.session_state:
    st.session_state['file_in'] = None
if 'file_out' not in st.session_state:
    st.session_state['file_out'] = None
if "uploaded_file" not in st.session_state:
    st.session_state["uploaded_file"] = None
if "studio" not in st.session_state:
    st.session_state["studio"] = None

# ROI Attributes
if 'view_window' not in st.session_state:
    st.session_state['view_window'] = None
if 'roi_frame' not in st.session_state:
    st.session_state['roi_frame'] = None
if 'point' not in st.session_state:
    st.session_state['points'] = None
if 'click_points' not in st.session_state:
    st.session_state['click_points'] = []
if 'poly_img' not in st.session_state:
    st.session_state['poly_img'] = None
if 'roi' not in st.session_state:
    st.session_state['roi'] = None

# Run Attributes
if "run" not in st.session_state:
    st.session_state["run"] = False
if "processor" not in st.session_state:
    st.session_state["processor"] = None
if "frame_data" not in st.session_state:
    st.session_state["frame_data"] = None

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Helper Funcs
def add_point():
    '''
    Takes a user mouse click over an image
    and returns an (x, y) tuple added to a session state list
    '''
    if len(st.session_state['click_points']) == 4:
        return
    else:
        raw = st.session_state['point']
        point = raw['x'], raw['y']
        st.session_state['click_points'].append(point)

# Set title
st.title("Lane-Line Detection Demo")

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
if uploaded_file is not None:
    st.spinner("Reading file...")
    if st.session_state["file_in"] is None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_in:
                temp_in.write(uploaded_file.read())
                st.session_state["file_in"] = temp_in.name
            studio = StudioManager(st.session_state["file_in"])
            st.session_state["studio"] = studio
            ret, frame = studio.return_frame()
            st.session_state["roi_frame"] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

with st.sidebar:
    run = st.button("Run Detection", type="secondary")
    if run:
        st.session_state["run"] = True
    st.title("System Configuration")
    st.subheader("Feature Engineering")

    # Feature Generation Input
    st.markdown("#### Generation")
    feature_gen = st.radio("Select a method for generating a feature map", ["edge", "thresh"], horizontal=True, index=0)
    if feature_gen == "edge":
        ksize = st.select_slider("Blur & Canny Kernel Size", [3, 5, 7, 9, 11, 13, 15], value=5)
    else:
        thresh_cols = st.columns(2)
        with thresh_cols[0]:
            small_ksize = st.select_slider("Close Kernel Size", [3, 5, 7, 9, 11, 13, 15], value=5)
        with thresh_cols[1]:
            large_ksize = st.select_slider("Dilate Kernel Size", [11, 13, 15, 17, 19, 21], value=15)

    # Feature Extraction Input
    st.markdown("#### Extraction")
    feature_sel = st.radio("Select a method for feature map extraction", ["hough", "direct"], horizontal=True, index=1)
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
        n_std = st.number_input("Number of Standard Deviations to Filter", min_value=0.5, max_value=10.0, value=6.0)

    st.markdown("#### Transformation")
    scaler = st.radio("Select Scaler", ["min_max", "z_score"], horizontal=True, index=1)
    use_bev = st.radio("Use BEV?", [True, False], horizontal=True, index=0)
    if use_bev:
        bev_cols = st.columns(3)
        with bev_cols[0]:
            forward_range = st.number_input("Forward", min_value=0.0, max_value=100.0, value=20.0)
        with bev_cols[1]:
            lateral_range = st.number_input("Lateral", min_value=0.0, max_value=100.0, value=7.0)
        with bev_cols[2]:
            resolution = st.number_input("GSD", min_value=0.01, max_value=1.0, value=0.03)

    # Modeling Configs
    st.subheader("Dynamic Linear Modeling")
    st.markdown("#### Estimator Options")
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

    st.markdown("#### Kalman Options")
    P_primer = st.number_input("Initial Confidence", min_value=0.0, max_value=0.99, value=0.5)
    process_noise = st.radio("Process Noise (Environment)", ["low", "medium", "high"], horizontal=True, index=1)

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
    if estimator == "ransac":
        system_configs.update({"confidence": confidence, "min_inliers": min_inliers, "max_error": max_error})
    if use_bev:
        system_configs.update({"forward_range": forward_range, "lateral_range": lateral_range, "resolution": resolution})

    st.session_state["configs"] = system_configs

if st.session_state['file_in'] is not None and not release:
    st.subheader("Media Viewer")
    st.markdown("**Move cursor over image and right-click at four different points on the image.**")
    st.write(" ")

    viewer_cols = st.columns([1, 3])
    with viewer_cols[0]:
        st.markdown("##### Viewer Options")
        reset = st.button("Reset", type='primary')
        if reset:
            st.session_state['run'] = False
            st.session_state["roi"] = None
            st.session_state['click_points'].clear()
            st.session_state['poly_img'] = st.session_state['roi_frame'].copy()
            st.session_state['processor'] = None
            st.rerun()

    with viewer_cols[1]:
        st.markdown("#####")
        view_options = ['inset', 'mosaic', "composite"]
        run_cols = st.columns(2)
        with run_cols[0]:
            view_selection = st.segmented_control("Render Options", view_options, label_visibility="collapsed", default=view_options[2])
        with run_cols[1]:
            play = st.button("Play", type="secondary")
    

    # Create Viewing Window
    if st.session_state['view_window'] is None:
        st.session_state['view_window'] = st.empty()
        st.session_state['container_lst'].append(st.session_state['view_window'])
    
    # Create writeable ROI Frame
    if st.session_state.get("roi_frame") is not None:
        try:
            if st.session_state['poly_img'] is None:
                poly_img = st.session_state.get("roi_frame")
                st.session_state['poly_img'] = poly_img.copy()
            # with st.session_state["view_window"]:
            img_draw = st.session_state.get("poly_img")
            draw = ImageDraw.Draw(img_draw)

            if len(st.session_state["click_points"]) > 0:
                for pt in st.session_state["click_points"]:
                    draw.ellipse([pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5], (255, 0, 0))

            if len(st.session_state["click_points"]) == 4:
                points = np.array(st.session_state["click_points"]).reshape(-1, 4, 2)
                st.session_state["roi"] = points
                mask = ROIMasker(points)
                poly_lst = mask.src_pts.tolist()
                poly = [(x, y) for point in poly_lst for x, y in point]
                draw.polygon(poly, outline=(255, 0, 0), width=5)

            value = img_xy(img_draw, key='point', on_click=add_point, cursor='crosshair')


        except Exception as e:
            st.error(f"Error preparing image coordinates: {str(e)}")
            
if st.session_state["run"]:
    if st.session_state["roi"] is None:
        st.error("Error, user must select ROI before running detection.")
    st.session_state["click_points"].clear()
    src = st.session_state["file_in"]
    roi = st.session_state["roi"]
    kwargs = st.session_state["configs"]
    if st.session_state["processor"] is None:
        st.session_state["processor"] = RTVideoProcessor(src, roi, kwargs, view_selection)
    
    if st.session_state["processor"] is not None:
        processor = st.session_state["processor"]
    try:
        frame_data = processor.recv()
        st.session_state["frame_data"] = frame_data
        with st.session_state["view_window"].container():
            st.image(frame_data, channels="BGR")
        # webrtc_streamer(
        #     key="RTVideoProcessor-v1",
        #     video_processor_factory=lambda: RTVideoProcessor(src, roi, kwargs, view_selection),
        #     mode=WebRtcMode.RECVONLY,
        #     media_stream_constraints={"video": True, "audio": False},
        #     rtc_configuration={"iceServers": []}, 
        #     async_processing=True,
        #     video_html_attrs={
        #         "style": {"width": "100%"},
        #         "controls": True,
        #         "autoPlay": True,
        #     }
        # )
        time.sleep(0.01)
        st.rerun()
    except queue.Empty:
        if st.session_state["frame_data"] is not None:
            st.success("Processing complete!")
        else:
            st.info("Processing frame...")
        st.session_state["run"] = False


    if play:
        if not st.session_state["processed"]:
            st.error("Cannot play until video is finished processing")
        with st.session_state["view_window"]:
            st.video(st.session_state["file_out"])

if release:
    st.session_state['reset'] = not st.session_state['reset']
    temp_in.close()
    st.session_state['file_in'] = None
    st.session_state["file_out"] = None
    st.session_state["uploaded_file"] = None
    st.session_state["processed"] = False
    st.session_state["view_window"] = None
    st.session_state['points'] = None
    st.session_state["poly_img"] = None
    st.session_state["roi"] = None
    st.session_state["click_points"].clear()
    if st.session_state["studio"] is not None:
        st.session_state["studio"].clean._clean_up()
    for container in st.session_state['container_lst']:
        container.empty()
    st.rerun()