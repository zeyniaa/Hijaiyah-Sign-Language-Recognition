"""Gradio webcam interface for the real Hijaiyah sign recognition pipeline.
Run from the repository root with: python app.py
The UI never fabricates predictions: a trained checkpoint must be present.
"""
from pathlib import Path
import threading
import cv2
import gradio as gr
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from src.feature_extraction import extract_keypoints
from src.preprocessing import normalize_features

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "transformer_70_30_best.h5"
LETTERS = ["alif", "ba", "ta", "tsa", "jim", "ha", "kho", "dal", "dzal", "ra", "zai",
           "sin", "syin", "shad", "dhad", "tha", "zha", "ain", "ghain", "fa", "qaf",
           "kaf", "lam", "mim", "nun", "waw", "haa", "ya"]
HARAKAT = ["fathah", "kasrah", "dhammah", "sukun"]
CLASS_NAMES = [f"{letter}_{harakat}" for letter in LETTERS for harakat in HARAKAT]
WINDOW_SIZE = 20
THRESHOLD = 0.70
model = load_model(CHECKPOINT) if CHECKPOINT.is_file() else None
model_lock = threading.Lock()
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)
drawing = mp.solutions.drawing_utils
connections = mp.solutions.holistic.HAND_CONNECTIONS

def status_text():
    if model is None:
        return ("### Checkpoint belum tersedia\nUI webcam dan ekstraksi MediaPipe siap. "
                "Untuk prediksi asli, letakkan transformer_70_30_best.h5 di folder checkpoints.")
    return "### Model siap\nTransformer checkpoint berhasil dimuat. Tunjukkan satu gerakan tangan ke kamera."

def process_frame(frame, sequence):
    sequence = list(sequence or [])
    if frame is None:
        return None, "Menunggu kamera…", "Aktifkan kamera untuk mulai.", sequence
    rgb = np.asarray(frame, dtype=np.uint8)
    annotated = rgb.copy()
    result = holistic.process(rgb)
    keypoints = extract_keypoints(result)
    sequence.append(keypoints)
    sequence = sequence[-WINDOW_SIZE:]
    if result.right_hand_landmarks:
        drawing.draw_landmarks(annotated, result.right_hand_landmarks, connections)
    if model is None:
        return annotated, "Checkpoint belum tersedia", "Prediksi dinonaktifkan; tidak ada hasil contoh yang ditampilkan.", sequence
    if len(sequence) < WINDOW_SIZE:
        return annotated, "Mengumpulkan frame", f"{len(sequence):02d}/{WINDOW_SIZE} frame untuk satu urutan gerakan.", sequence
    batch = normalize_features(np.asarray(sequence, dtype=np.float32))[None, ...]
    with model_lock:
        probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    label = CLASS_NAMES[index] if index < len(CLASS_NAMES) else f"class_{index}"
    if confidence < THRESHOLD:
        return annotated, "Belum yakin", f"Keyakinan {confidence:.1%}; tahan gerakan lebih stabil.", sequence
    return annotated, label.replace("_", " ").title(), f"Keyakinan model {confidence:.1%} · Transformer · jendela 20 frame.", sequence

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto !important}
#hero {padding: 18px 22px; border-radius: 18px; background: linear-gradient(120deg,#f7eeee,#f4eff9); margin-bottom: 18px}
#hero h1 {margin: 0 0 6px; color: #282336}
#hero p {margin: 0; color: #776d80}
"""
with gr.Blocks(title="Hijaiyah Sign Recognition · Live", css=CSS, theme=gr.themes.Soft(primary_hue="rose", secondary_hue="purple")) as demo:
    gr.Markdown("# Hijaiyah Sign Recognition\n**Live camera · MediaPipe Holistic · Transformer**\n\nEach prediction uses a 20-frame sequence from the real model pipeline.", elem_id="hero")
    with gr.Row():
        with gr.Column(scale=3):
            camera = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Live camera input")
            prediction = gr.Textbox(label="Model prediction", value="Aktifkan kamera untuk mulai.", interactive=False)
        with gr.Column(scale=2):
            annotated = gr.Image(label="MediaPipe hand landmarks", interactive=False)
            progress = gr.Markdown(status_text())
            sequence = gr.State([])
            gr.Markdown("**How it works**  \nCamera frame → 75 keypoints → 20-frame window → Transformer (112 classes).")
    camera.stream(process_frame, inputs=[camera, sequence], outputs=[annotated, prediction, progress, sequence], stream_every=0.15, concurrency_limit=1)
    clear = gr.Button("Reset sequence")
    clear.click(lambda: ([], "Aktifkan kamera untuk mulai."), outputs=[sequence, prediction])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
