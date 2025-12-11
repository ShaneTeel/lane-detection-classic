
from lane_detection.studio import StudioManager
from lane_detection.image_geometry import ROIMasker
from lane_detection.detection import DetectionSystem

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
            break

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
    

