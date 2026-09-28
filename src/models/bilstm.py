"""
Bidirectional LSTM (Bi-LSTM) Baseline Model for Hijaiyah Sign Language Recognition.
Baseline 1 achieving 83.22% Accuracy (Split 70:30).
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam


def create_bilstm_model(input_shape=(20, 75), num_classes=112, learning_rate=0.0001):
    """
    Creates and compiles the Bidirectional LSTM (Bi-LSTM) baseline model.

    Parameters:
    - input_shape: tuple, shape of input sequence (timesteps, features) -> default (20, 75)
    - num_classes: int, number of target classes -> default 112
    - learning_rate: float, Adam optimizer learning rate -> default 0.0001

    Returns:
    - Compiled Keras Model instance.
    """
    model = Sequential(name="Bi_LSTM_Baseline")

    # Layer 1: Bidirectional LSTM (processes forward & backward temporal context)
    model.add(Bidirectional(
        LSTM(units=64, return_sequences=False),
        input_shape=input_shape,
        name="bilstm_layer"
    ))

    # Layer 2: Dropout (prevents overfitting)
    model.add(Dropout(0.3, name="dropout_1"))

    # Layer 3: Dense feature transformation
    model.add(Dense(64, activation="relu", name="dense_features"))

    # Layer 4: Output Layer
    model.add(Dense(num_classes, activation="softmax", name="output_layer"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    return model
