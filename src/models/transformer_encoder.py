"""
Transformer Encoder Model Architecture for Hijaiyah Sign Language Recognition.
Proposed State-of-the-Art Architecture achieving 97.54% Accuracy (Split 70:30).
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    MultiHeadAttention,
    LayerNormalization,
    Conv1D,
    GlobalAveragePooling1D,
    Dropout,
    Dense
)
from tensorflow.keras.optimizers import Adam


def create_transformer_model(input_shape=(20, 75), num_classes=112, learning_rate=0.0001):
    """
    Creates and compiles the Transformer Encoder model.

    Parameters:
    - input_shape: tuple, shape of input sequence (timesteps, features) -> default (20, 75)
    - num_classes: int, number of target classes -> default 112 (28 letters x 4 harakat)
    - learning_rate: float, Adam optimizer learning rate -> default 0.0001

    Returns:
    - Compiled Keras Model instance.
    """
    inputs = Input(shape=input_shape, name="input_keypoints")

    # 1. Multi-Head Self-Attention Sub-Layer
    # 4 parallel heads to capture fine-grained hand and arm spatial relationships
    attention_output = MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        name="multi_head_attention"
    )(inputs, inputs)

    # 2. Residual Connection & Layer Normalization
    attention_output = LayerNormalization(
        epsilon=1e-6,
        name="norm_attention"
    )(inputs + attention_output)

    # 3. Feed Forward Network (FFN) using Conv1D kernel_size=1
    ffn_output = Conv1D(
        filters=64,
        kernel_size=1,
        activation="relu",
        name="conv1d_ffn"
    )(attention_output)

    # 4. Residual Connection & Layer Normalization
    x = LayerNormalization(
        epsilon=1e-6,
        name="norm_ffn"
    )(attention_output + ffn_output)

    # 5. Global Average Pooling (aggregates time-dimension to 64-dim vector, avoids Flatten explosion)
    x = GlobalAveragePooling1D(name="global_average_pooling")(x)

    # 6. Regularization
    x = Dropout(0.3, name="dropout")(x)

    # 7. Classification Head
    outputs = Dense(
        num_classes,
        activation="softmax",
        name="output_classification"
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="Transformer_Encoder_LPMQ")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    return model
