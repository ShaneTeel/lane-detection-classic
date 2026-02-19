import streamlit as st

def show_source_management():
    # Create file upload button
    st.header("Source Management")
    cols0 = st.columns(2)
    with cols0[0]:
        st.subheader("Video Selector")
        vid_options = ["lane1-curved.mp4", "lane1-straight.mp4"]
        # uploaded_file = st.file_uploader("Choose a source file:", type=['mp4', 'mov', 'mkv', 'avi'], key=f"file_uploader_{st.session_state['reset']}")
        uploaded_file = f"./media/{st.selectbox('Select Video', options=vid_options)}"

    with cols0[1]:
        st.subheader("Video Release")
        st.write("Select Before Exiting")
        release = st.button("Release", help="Release all capture objects and empty all containers", type='primary')
        if release:
            st.session_state["release"] = True
            
    return uploaded_file