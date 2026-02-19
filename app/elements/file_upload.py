import streamlit as st
import tempfile
import cv2
from PIL import Image

from lane_detection.studio import StudioManager


def show_file_upload(file:str):
    st.session_state["file_in"] = file

    if st.session_state["file_out"] is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_out:
            st.session_state["file_out"] = temp_out.name
            temp_out.close()

    if st.session_state["studio"] is not None:
        st.session_state["studio"].clean._clean_up()

    studio = StudioManager(st.session_state["file_in"])
    st.session_state["studio"] = studio
    ret, frame = studio.return_frame()
    st.session_state["roi_frame"] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))