import streamlit as st
import os

def show_release():
    st.session_state['reset'] = not st.session_state['reset']
    for container in st.session_state['container_lst']:
        container.empty()
    st.session_state['container_lst'].clear()

    st.session_state["file_in"] = None

    if st.session_state["file_out"] and os.path.exists(st.session_state["file_out"]):
        os.remove(st.session_state['file_out'])
        st.session_state["file_out"] = None

    if st.session_state["studio"] is not None:
        st.session_state["studio"].clean._clean_up()

    st.session_state['points'] = None
    st.session_state["click_points"].clear()
    st.session_state["poly_img"] = None
    st.session_state["roi"] = None
    st.session_state["view_window"] = None
    st.session_state['roi_frame'] = None

    st.session_state["configs"] = None
    st.session_state["run"] = False
    st.session_state["detector"] = None
    st.session_state["processed"] = False
    st.session_state["play"] = False
    st.session_state["release"] = False
    
    st.rerun()