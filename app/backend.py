from fastapi import FastAPI, HTTPException, Response, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
import base64
import asyncio
import tempfile
import shutil
from lane_detection.studio import StudioManager
from lane_detection.image_geometry import ROIMasker
from lane_detection.detection import DetectionSystem

app = FastAPI()

class AppState:

    def __init__(self):
        self.mask = None
        self.system = None
        self.studio = None
        self.play_flag = None

    def add_item(self, key, value):
        self.data[key] = value

    def get_attr(self, key):
        return self.data.get(key)
    
state = AppState()

@app.get("/")
def read_root():
    return {"message": "CannyHoughP Lane-Detection API", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/initialize")
async def initialize_source(file: UploadFile = File(...)):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        try:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        finally:
            await file.close()

    try:
        studio = StudioManager(temp_file_path)
        state.studio = studio
        ret, frame = studio.return_frame()
        if not ret:
            error = f"Error: Could not read frame from {studio.source.name}"
            raise HTTPException(status_code=500, detail=error)
        else:
            frame = studio.render._resize_frame(frame, 750)
            ret, im = cv2.imencode(".jpeg", frame)
            return Response(im.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roi")
def define_roi(request: dict):
    points = np.array(request.get("points"))

    mask = ROIMasker(points)
    
    state.mask = mask

    return {"poly": mask.src_pts.astype(int).tolist()}

@app.post("/configure")
def configure_system(request: dict):
    try:
        system = DetectionSystem(state.studio, state.mask, **request)
        state.system = system
        
    except Exception as e:
        error = f"Error occured while processing frames: {str(e)}"
        raise HTTPException(status_code=500, detail=error)

async def run_detection(style: str, state: AppState = state):
    state.play_flag = True
    system = state.system
    system.studio.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_names = system._configure_output(style, None, method="final")

    while state.play_flag:
        ret1, raw = system.studio.return_frame()
        if not ret1:
            stop_video()

        raw_resized = system.studio.render._resize_frame(raw, 750)
        thresh, feature_map = system.generator.generate(raw_resized)
        masked = system.mask.inverse_mask(feature_map)
        lane_pts = system.selector.select(masked)
        lane_lines = []
        for i in range(2):
            detector = system.detector1 if i == 0 else system.detector2
            evaluator = system.evaluator1 if i == 0 else system.evaluator2
            if lane_pts[i].size == 0:
                continue

            pts = lane_pts[i]

            line = system.detect_line(pts, detector)
            lane_lines.append(np.flipud(line)) if i == 0 else lane_lines.append(line)
            system.evaluate_model(detector, evaluator)
        frame_lst = [raw_resized, thresh, feature_map, masked]
        final = system.studio.gen_view(frame_lst, frame_names, lane_lines, style, stroke=False, fill=True)
        ret2, buffer = cv2.imencode(".jpg", final)
        
        if not ret2:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        await asyncio.sleep(0)

@app.get("/stream_video")
def stream_video(style: str):
    try:
        return StreamingResponse(
            content=run_detection(style),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
        
    except Exception as e:
        error = f"Error occured while processing frames: {str(e)}"
        raise HTTPException(status_code=500, detail=error)
    
@app.post("/stop_video")
def stop_video():
    try:
        state.play_flag = False
        return {'status': 'stopped'}
        
    except Exception as e:
        error = f"Error occured while processing frames: {str(e)}"
        raise HTTPException(status_code=500, detail=error)
    

# Gen file download link (app-specific)
def get_download_link(self, file, file_name, text):
    buffered = io.BytesIO()
    file.save(buffered, format='mp4')
    file_str = base64.b64encode(buffered.getvalue()).decode()
    href = f"<a href='data:file/txt;base64,{file_str}' download='{file_name}'>{text}</a>"
    return href