import streamlit as st
import numpy as np
from PIL import ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates as img_xy

from lane_detection.image_geometry import ROIMasker

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

def show_media_preview():
    st.subheader("Media Viewer")
    st.markdown("**Move cursor over image and right-click at four different points on the image.**")
    st.write(" ")

    st.markdown("##### Viewer Options")
    viewer_cols = st.columns(3, vertical_alignment="center")
    with viewer_cols[0]:
        reset = st.button("Reset", type='secondary')
        if reset:
            st.session_state['run'] = False
            st.session_state["roi"] = None
            st.session_state['click_points'].clear()
            st.session_state['poly_img'] = st.session_state['roi_frame'].copy()
            st.session_state['detector'] = None
            st.session_state["processed"] = False
            st.session_state["play"] = False
            st.rerun()

    with viewer_cols[1]:
        if st.session_state["roi"] is not None:
            run = st.button("Run Detection", type="secondary")
            if run:
                st.session_state["run"] = True
        else:
            run = st.button("Run Detection", type="secondary", disabled=True, help="Select ROI First.")

    with viewer_cols[2]:
        if st.session_state["processed"]:
            play = st.button("Play", type="secondary")
            if play:
                st.session_state["play"] = True
        else:
            play = st.button("Play", type="secondary", disabled=True, help="Process video first.")

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
                st.session_state["click_points"].clear()
                st.rerun()

            value = img_xy(img_draw, key='point', on_click=add_point, cursor='crosshair')

        except Exception as e:
            st.error(f"Error preparing image coordinates: {str(e)}")