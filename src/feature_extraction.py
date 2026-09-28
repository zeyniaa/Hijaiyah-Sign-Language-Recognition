"""
MediaPipe Holistic Feature Extraction Pipeline.
Extracts 75 spatial & visibility keypoints per video frame:
- 21 Right Hand Landmarks (x, y, z) = 63 features
- 3 Pose Landmarks: Right Shoulder(12), Right Elbow(14), Right Wrist(16) (x, y, z, visibility) = 12 features
Total = 75 numerical features per frame.
"""

import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic


def extract_keypoints(results) -> np.ndarray:
    """
    Extracts 75 keypoints from MediaPipe Holistic detection results.

    Parameters:
    - results: MediaPipe Holistic detection output.

    Returns:
    - np.ndarray: 1D array of shape (75,) with float32 values.
    """
    row = []

    # 1. Right Hand Landmarks (21 points x 3 coordinates = 63 features)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])
    else:
        # Zero imputation if hand is temporarily outside frame or occluded
        row.extend([0.0] * 63)

    # 2. Right Arm Pose Landmarks (3 keypoints: Shoulder 12, Elbow 14, Wrist 16 x 4 values = 12 features)
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for idx in [12, 14, 16]:
            lm = landmarks[idx]
            row.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        row.extend([0.0] * 12)

    return np.array(row, dtype=np.float32)


def process_video(video_path: str, max_frames: int = 20) -> np.ndarray:
    """
    Processes a single video file, extracts frame keypoints, and returns a sequence tensor.

    Parameters:
    - video_path: str, path to input video file.
    - max_frames: int, maximum number of frames to capture (default 20).

    Returns:
    - np.ndarray: Sequence of shape (frames, 75).
    """
    cap = cv2.VideoCapture(video_path)
    sequence = []

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened() and len(sequence) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

    cap.release()
    return np.array(sequence, dtype=np.float32)
