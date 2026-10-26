"""Browser UI for the actual MediaPipe / Transformer pipeline. No simulated predictions."""
from __future__ import annotations
import base64
import importlib.util
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(os.environ.get('MODEL_PATH', str(ROOT / 'checkpoints/transformer_70_30_best.h5')))
LABELS_PATH = Path(os.environ.get('LABELS_PATH', str(ROOT / 'checkpoints/class_names.json')))
LETTERS = ['alif','ba','ta','tsa','jim','ha','kho','dal','dzal','ra','zai','sin','syin','shad','dhad','tha','zha','ain','ghain','fa','qaf','kaf','lam','mim','nun','waw','haa','ya']
DEFAULT_LABELS = [f'{letter}_{vowel}' for letter in LETTERS for vowel in ['fathah','kasrah','dhammah','sukun']]
WINDOW, MAX_BYTES = 20, 24 * 1024 * 1024
lock = threading.RLock()
sessions = {}
model = None
model_error = None
app = FastAPI(title='Hijaiyah Live', docs_url='/api/docs')
app.mount('/assets', StaticFiles(directory=ROOT / 'web'), name='assets')

@app.get('/')
def home():
    return FileResponse(ROOT / 'web/index.html')

@app.get('/api/status')
def status():
    return {'checkpoint_found':CHECKPOINT.is_file(), 'model_loaded':model is not None,
            'vision_available':importlib.util.find_spec('mediapipe') is not None,
            'sequence_length':WINDOW, 'features':75, 'classes':112, 'checkpoint_name':CHECKPOINT.name}

async def read_upload(upload, limit=MAX_BYTES):
    raw = await upload.read(limit + 1)
    if not raw: raise HTTPException(400, 'Choose a non-empty image or video.')
    if len(raw) > limit: raise HTTPException(413, 'File too large. The limit is 24 MB.')
    return raw

def get_session(key):
    if not re.fullmatch(r'[a-zA-Z0-9-]{1,64}', key): raise HTTPException(400, 'Invalid camera session.')
    now = time.monotonic()
    for old in list(sessions):
        if now - sessions[old]['seen'] > 300: sessions.pop(old)['vision'].close()
    if key not in sessions:
        if len(sessions) >= 16: raise HTTPException(429, 'Too many camera sessions.')
        try:
            import mediapipe as mp
            vision = mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=.5, min_tracking_confidence=.5)
        except (ImportError, AttributeError) as exc:
            raise HTTPException(503, 'Install the vision requirements with MediaPipe 0.10.21 to enable landmark extraction.') from exc
        sessions[key] = {'vision':vision, 'sequence':[], 'seen':now}
    sessions[key]['seen'] = now
    return sessions[key]

def infer(sequence, hand_found):
    global model, model_error
    if not hand_found: return None, 'Place your right hand inside the camera view.'
    if len(sequence) < WINDOW: return None, f'Collecting a gesture sequence: {len(sequence)}/{WINDOW} frames.'
    if not CHECKPOINT.is_file(): return None, 'Landmarks are ready. Add the trained checkpoint to enable recognition.'
    try:
        import numpy as np
        from tensorflow.keras.models import load_model
        from src.preprocessing import normalize_features
        if model is None: model = load_model(CHECKPOINT, compile=False)
        labels = json.loads(LABELS_PATH.read_text()) if LABELS_PATH.is_file() else DEFAULT_LABELS
        if not isinstance(labels, list) or len(labels)!=112 or not all(isinstance(x,str) for x in labels):
            raise ValueError('class_names.json must list 112 names in training order.')
        scores = model.predict(normalize_features(np.asarray(sequence,dtype=np.float32))[None,...], verbose=0)[0]
        if len(scores)!=len(labels) or not np.isfinite(scores).all(): raise ValueError('Checkpoint output and labels do not match.')
        indices = np.argsort(scores)[-3:][::-1]
        top = [{'label':labels[int(i)],'confidence':float(scores[i])} for i in indices]
        return {**top[0], 'alternatives':top, 'stable':top[0]['confidence']>=.7}, 'Recognition complete.'
    except Exception as exc:
        model_error = str(exc)
        raise HTTPException(503, 'Checkpoint load failed. Check TensorFlow and the training label order.') from exc

def process(rgb, session):
    import cv2
    import mediapipe as mp
    from src.feature_extraction import extract_keypoints
    h,w = rgb.shape[:2]
    if max(h,w)>960: rgb=cv2.resize(rgb,(int(w*960/max(h,w)),int(h*960/max(h,w))))
    results = session['vision'].process(rgb)
    hand = results.right_hand_landmarks is not None
    session['sequence']=(session['sequence']+[extract_keypoints(results)])[-WINDOW:]
    annotated=rgb.copy()
    if hand: mp.solutions.drawing_utils.draw_landmarks(annotated,results.right_hand_landmarks,mp.solutions.holistic.HAND_CONNECTIONS)
    ok,jpeg=cv2.imencode('.jpg',cv2.cvtColor(annotated,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,80])
    prediction,message=infer(session['sequence'],hand)
    return {'frame':'data:image/jpeg;base64,'+base64.b64encode(jpeg).decode() if ok else None,
            'hand_detected':hand,'landmarks':21 if hand else 0,'frames':len(session['sequence']),
            'prediction':prediction,'message':message}

@app.post('/api/frame')
async def camera_frame(file:UploadFile=File(...), session:str=Form(...)):
    import cv2
    import numpy as np
    raw=await read_upload(file,3*1024*1024)
    image=cv2.imdecode(np.frombuffer(raw,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise HTTPException(415,'The frame is not a valid image.')
    with lock: return process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),get_session(session))

@app.post('/api/video')
async def video_file(file:UploadFile=File(...)):
    raw=await read_upload(file)
    suffix=Path(file.filename or 'video.mp4').suffix.lower()
    if suffix not in {'.mp4','.webm','.mov','.avi'}: raise HTTPException(415,'Use MP4, WebM, MOV, or AVI.')
    import cv2
    with tempfile.TemporaryDirectory(prefix='hijaiyah-') as directory:
        path=Path(directory)/('input'+suffix);path.write_bytes(raw)
        capture=cv2.VideoCapture(str(path))
        try:
            with lock:
                key='video-'+os.urandom(8).hex();session=get_session(key);result=None
                try:
                    for _ in range(WINDOW):
                        ok,image=capture.read()
                        if not ok: break
                        result=process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),session)
                    if result is None: raise HTTPException(415,'This video could not be decoded.')
                    return result
                finally: sessions.pop(key)['vision'].close()
        finally: capture.release()

@app.delete('/api/session/{key}')
def reset(key:str):
    with lock:
        if key in sessions: sessions.pop(key)['vision'].close()
    return {'reset':True}
