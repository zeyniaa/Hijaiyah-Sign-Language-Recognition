"""
Hybrid Stacked LSTM-GRU Baseline Model for Hijaiyah Sign Language Recognition.
Baseline 2 achieving 69.13% Accuracy (Split 70:30).
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam


def create_hybrid_model(input_shape=(20, 75), num_classes=112, learning_rate=0.0001):
    """
    Creates and compiles the Stacked Hybrid LSTM-GRU baseline model.

    Parameters:
    - input_shape: tuple, shape of input sequence (timesteps, features) -> default (20, 75)
    - num_classes: int, number of target classes -> default 112
    - learning_rate: float, Adam optimizer learning rate -> default 0.0001

    Returns:
    - Compiled Keras Model instance.
    """
    model = Sequential(name="Hybrid_Stacked_LSTM_GRU")

    # Layer 1: LSTM (Bottom Layer, preserves temporal sequence)
    model.add(LSTM(
        units=64,
        return_sequences=True,
        input_shape=input_shape,
        name="lstm_bottom_layer"
    ))

    # Layer 2: GRU (Top Layer, summarizes temporal sequence into 64-dim vector)
    model.add(GRU(
        units=64,
        return_sequences=False,
        name="gru_top_layer"
    ))

    # Layer 3: Regularization
    model.add(Dropout(0.3, name="dropout_layer"))

    # Layer 4: Output Layer
    model.add(Dense(num_classes, activation="softmax", name="output_layer"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    return model
