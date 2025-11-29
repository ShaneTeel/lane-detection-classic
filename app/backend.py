from fastapi import FastAPI, HTTPException, Response, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import asyncio
from scripts import Render, Read, Write, CannyRANSAC, CannyHoughP

app = FastAPI()

class AppState:

    def __init__(self):
        self.roi = None
        self.processor = None
        self.source = None
        self.render = Render()
        self.writer = None
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
def create_source(file: UploadFile = File(...)):
    try:
        source = Read(file)
        state.source = source

        writer = Write(source.name, '.mp4', source.width, source.height, source.fps)
        state.writer = writer

        ret, frame = source.return_frame()
        if not ret:
            error = f"Error: Could not read frame from {source.name}"
            raise HTTPException(status_code=500, detail=error)
        else:
            ret, im = cv2.imencode(".jpeg", frame)
            return Response(im.tobytes(), media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/roi")
def define_roi(request: dict):
    points = request.get("points")
    method = request.get("method")

    top = min(*[y for _, y in [point for point in points]])
    bottom = max(*[y for _, y in [point for point in points]])
    mid_y = sum([top, bottom]) // 2

    left = min(*[x for x, _ in [point for point in points]])
    right = max(*[x for x, _ in [point for point in points]])
    mid_x = sum([left, right]) // 2

    roi = [1, 2, 3, 4]
    
    for point in points:
        x, y = point
        if x < mid_x and y < mid_y:
            roi[0] = point if method == 'original' else (x, top)
        elif x > mid_x and y < mid_y:
            roi[1] = point if method == 'original' else (x, top)
        elif x > mid_x and y > mid_y:
            roi[2] = point if method == 'original' else (x, bottom)
        elif x < mid_x and y > mid_y:
            roi[3] = point if method == 'original' else (x, bottom)

    state.roi = roi

    return {"poly": roi}

@app.post("/configure")
def configure_processor(request: dict):
    try:
        processor = CannyHoughP(state.roi, request)
        state.processor = processor
        
    except Exception as e:
        error = f"Error occured while processing frames: {str(e)}"
        raise HTTPException(status_code=500, detail=error)

async def render_frame(style: str, state: AppState = state):
    state.play_flag = True
    state.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_names = ["Threshold", "Edge Map", "Hough Lines", "Final Composite"]

    while state.play_flag:
        ret1, raw = state.source.return_frame()
        if not ret1:
            state.source.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        thresh, edge, hough, composite = state.processor.run(raw)
        if style == 'Step-by-Step':
            frame = state.render.render_mosaic([thresh, edge, hough, composite], frame_names)
        else:
            frame = composite
        
        ret2, buffer = cv2.imencode(".jpg", frame)
        if not ret2:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        await asyncio.sleep(0)

@app.get("/stream_video")
def stream_video(style: str):
    try:
        return StreamingResponse(
            content=render_frame(style),
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
    
