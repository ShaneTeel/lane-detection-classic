# Display / Image operations
import streamlit as st

# Streamlit specific Detection System wrapper
from elements import *

# Destructor Attributes
if 'reset' not in st.session_state:
    st.session_state['reset'] = True
if "release" not in st.session_state:
    st.session_state["release"] = False
if 'container_lst' not in st.session_state:
    st.session_state['container_lst'] = []

# Source Attributes
if 'file_in' not in st.session_state:
    st.session_state['file_in'] = None
if 'file_out' not in st.session_state:
    st.session_state['file_out'] = None
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
if "configs" not in st.session_state:
    st.session_state["configs"] = None
if "run" not in st.session_state:
    st.session_state["run"] = False
if "detector" not in st.session_state:
    st.session_state["detector"] = None
if "processed" not in st.session_state:
    st.session_state["processed"] = False
if "play" not in st.session_state:
    st.session_state["play"] = False

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# Set title
st.title("Lane-Line Detection Demo")

uploaded_file = show_source_management()

st.divider()

# File Upload Block 
if uploaded_file is not None and uploaded_file != st.session_state["file_in"]:
    if st.session_state["file_in"] is None:
        show_file_upload(uploaded_file)
    else:
        show_release()
        show_file_upload(uploaded_file)

if st.session_state['file_in'] is not None and not st.session_state["release"]:
    show_media_preview()

with st.sidebar:
    show_sidebar_configs()
            
if st.session_state["run"]:
    show_video_processing()

if st.session_state["play"]:
    with st.session_state["view_window"]:
        st.video(st.session_state["file_out"])

if st.session_state["release"]:
    show_release()