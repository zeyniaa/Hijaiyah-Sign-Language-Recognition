"""
Real-Time Webcam Inference Script for Hijaiyah Sign Language Recognition.
Captures live frames, extracts 75 keypoints via MediaPipe Holistic,
maintains a 20-frame sliding window, and predicts the sign class in real time.
"""

import sys
import argparse
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model

from src.feature_extraction import extract_keypoints
from src.preprocessing import normalize_features

# Standard 112 Hijaiyah Classes (28 letters x 4 harakat: fathah, kasrah, dhammah, sukun)
LETTERS = [
    "alif", "ba", "ta", "tsa", "jim", "ha", "kho", "dal", "dzal", "ra", "zai",
    "sin", "syin", "shad", "dhad", "tha", "zha", "ain", "ghain", "fa", "qaf",
    "kaf", "lam", "mim", "nun", "waw", "haa", "ya"
]
HARAKAT = ["fathah", "kasrah", "dhammah", "sukun"]
CLASS_NAMES = [f"{letter}_{h}" for letter in LETTERS for h in HARAKAT]


def parse_args():
    parser = argparse.ArgumentParser(description="Live Real-time Sign Language Prediction")
    parser.add_argument("--model-path", type=str, default="checkpoints/transformer_70_30_best.h5", help="Path to trained model")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default 0)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold to display prediction")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.model_path}...")
    try:
        model = load_model(args.model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Could not load model ({e}). Real-time demo requires a valid .h5 checkpoint.")
        sys.exit(1)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot access camera at index {args.camera}.")
        sys.exit(1)

    sequence_buffer = []
    current_prediction = "Waiting for gesture..."
    confidence_score = 0.0

    print("\nStarting live webcam inference. Press 'q' in the camera window to exit.")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)

            # MediaPipe detection
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Draw right hand landmarks
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

            # Extract 75 keypoints
            keypoints = extract_keypoints(results)
            sequence_buffer.append(keypoints)

            # Keep sliding window of 20 frames
            if len(sequence_buffer) > 20:
                sequence_buffer.pop(0)

            # Perform prediction when 20 frames are buffered
            if len(sequence_buffer) == 20:
                input_tensor = np.array(sequence_buffer, dtype=np.float32)
                normalized_tensor = normalize_features(input_tensor)
                input_batch = np.expand_dims(normalized_tensor, axis=0)  # (1, 20, 75)

                prediction_probs = model.predict(input_batch, verbose=0)[0]
                best_class_idx = np.argmax(prediction_probs)
                confidence = float(prediction_probs[best_class_idx])

                if confidence >= args.threshold:
                    class_name = CLASS_NAMES[best_class_idx] if best_class_idx < len(CLASS_NAMES) else f"Class {best_class_idx}"
                    current_prediction = f"{class_name.upper()}"
                    confidence_score = confidence
                else:
                    current_prediction = "Low Confidence"
                    confidence_score = confidence

            # Render UI overlay
            cv2.rectangle(frame, (0, 0), (640, 50), (20, 20, 30), -1)
            display_text = f"Sign: {current_prediction} ({confidence_score * 100:.1f}%)"
            cv2.putText(
                frame,
                display_text,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 128) if confidence_score >= args.threshold else (180, 180, 180),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Hijaiyah Sign Language Live Inference (LPMQ)", frame)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
