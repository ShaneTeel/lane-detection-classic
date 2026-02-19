import streamlit as st
import time

from .streamlit_detector import StreamlitDetector

def show_video_processing():
    src = st.session_state["file_in"]
    roi = st.session_state["roi"]

    config_dict = st.session_state["configs"]
    kwargs = config_dict["system"]
    view_selection = config_dict["view_selection"]

    if st.session_state["detector"] is None:
        detector = StreamlitDetector(file_path=src, roi=roi, file_out_name=st.session_state["file_out"], view_style=view_selection, configs=kwargs)
        st.session_state["detector"] = detector

    progress_bar = st.progress(0, text="Video processing in progress...")
    total_frames = st.session_state["studio"].source.frame_count
    frame_idx = 0
    try:
        while True:
            ret, frame = detector.return_frame()
            if not ret:
                st.session_state["run"] = False
                st.session_state["processed"] = True
                st.session_state["detector"].system.studio.clean._clean_up()
                st.session_state["detector"].writer.release()
                time.sleep(0.5)
                break

            final = detector.process_frame(frame)
            detector.write_frame(final)
            frame_idx += 1

            percent_complete = int((frame_idx / total_frames * 100))
            progress_bar.progress(percent_complete, text=f"Processed {frame_idx}/{total_frames} ({percent_complete}%).")

        st.rerun()
    
    except Exception as e:
        st.error(f"Processing error: {e}")
    
    finally:
        if st.session_state["detector"]:
            st.session_state["detector"].writer.release()
            st.session_state["detector"].system.studio.clean._clean_up()
