import streamlit as st

def show_sidebar_configs():
    st.title("System Configuration")
    st.subheader("Feature Engineering")

    # Feature Generation Input
    st.markdown("#### Generation")
    feature_gen = st.radio("Select a method for generating a feature map", ["edge", "thresh"], horizontal=True, index=0)
    if feature_gen == "edge":
        ksize = st.select_slider("Blur & Sobel-X Kernel Size", [3, 5, 7, 9, 11, 13, 15], value=5)
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

    # Styling Configs
    st.subheader("Rendering Style")
    st.markdown("#### Slect a video rendering option")
    view_selection = st.radio("Render Options", ['inset', 'mosaic', "composite"], label_visibility="collapsed", horizontal=True, index=2)

    st.caption("Render options only affect the final frame design, not the algorithm applied to detect lane lines.")

    system_configs = {
        "generator": feature_gen,
        "selector": feature_sel,
        "estimator": estimator,
        "scaler_type": scaler,
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

    st.session_state["configs"] = {"system": system_configs, "view_selection": view_selection}