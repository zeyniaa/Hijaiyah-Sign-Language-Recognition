"""
Data Preprocessing Pipeline:
- Selective Min-Max Normalization (XYZ spatial coordinates scaled to [0,1], Visibility preserved)
- Post-Padding & Post-Truncating to fixed sequence length (20 frames)
- Stratified Train-Test Splitting (80:20 or 70:30)
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences


def normalize_features(X_raw: np.ndarray) -> np.ndarray:
    """
    Applies selective Min-Max scaling to spatial coordinates (x, y, z)
    while preserving native visibility scores in [0, 1].

    Parameters:
    - X_raw: np.ndarray of shape (N, T, 75) or (T, 75).

    Returns:
    - np.ndarray: Normalized tensor of the same shape.
    """
    is_3d = X_raw.ndim == 3
    if not is_3d:
        X_raw = np.expand_dims(X_raw, axis=0)

    N, T, F = X_raw.shape
    # Flatten time and sample dimensions for feature scaling: (N * T, 75)
    flattened = X_raw.reshape(-1, F)

    # Spatial features: 63 hand coords + 9 pose coords (indices 63, 64, 65, 67, 68, 69, 71, 72, 73)
    # Visibility indices: 66 (shoulder_v), 70 (elbow_v), 74 (wrist_v)
    v_indices = [66, 70, 74]
    xyz_indices = [i for i in range(F) if i not in v_indices]

    scaler = MinMaxScaler(feature_range=(0, 1))
    flattened_xyz = scaler.fit_transform(flattened[:, xyz_indices])

    # Reassemble normalized array
    normalized = np.empty_like(flattened, dtype=np.float32)
    normalized[:, xyz_indices] = flattened_xyz
    normalized[:, v_indices] = flattened[:, v_indices]

    result = normalized.reshape(N, T, F)
    return result if is_3d else result[0]


def pad_sequence_data(sequences: list, max_len: int = 20) -> np.ndarray:
    """
    Applies post-padding with zero-vectors and post-truncating to enforce
    a uniform sequence length of 20 frames.

    Parameters:
    - sequences: list of 2D np.ndarray, each of shape (frame_count, 75)
    - max_len: int, target temporal length (default 20).

    Returns:
    - np.ndarray of shape (N, 20, 75).
    """
    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding="post",
        truncating="post",
        dtype="float32",
        value=0.0
    )
    return padded


def split_dataset(X: np.ndarray, y: np.ndarray, test_size: float = 0.20, random_state: int = 42):
    """
    Performs stratified random sampling based on sequence labels to ensure
    identical class balance across training and testing partitions.

    Parameters:
    - X: np.ndarray of shape (N, 20, 75)
    - y: np.ndarray of shape (N, num_classes) or (N,)
    - test_size: float (0.20 for 80:20 scenario, 0.30 for 70:30 scenario)
    - random_state: int, random seed (default 42)

    Returns:
    - X_train, X_test, y_train, y_test
    """
    y_stratify = np.argmax(y, axis=1) if y.ndim > 1 else y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y_stratify
    )
    return X_train, X_test, y_train, y_test
