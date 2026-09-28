"""
Training Pipeline for Hijaiyah Sign Language Deep Learning Models.
Supports Transformer Encoder (SOTA), Bi-LSTM, and Hybrid Stacked LSTM-GRU.
"""

import os
import argparse
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.models.transformer_encoder import create_transformer_model
from src.models.bilstm import create_bilstm_model
from src.models.hybrid_lstm_gru import create_hybrid_model
from src.preprocessing import split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hijaiyah Sign Language Recognition Models")
    parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["transformer", "bilstm", "hybrid"],
        help="Model architecture to train (default: transformer)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="70:30",
        choices=["70:30", "80:20"],
        help="Train-test split ratio (default: 70:30)"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Adam learning rate (default: 0.0001)")
    parser.add_argument("--data-dir", type=str, default="data/sample", help="Directory containing .npy data files")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for saved models")
    return parser.parse_args()


def load_data(data_dir: str):
    """
    Loads preprocessed feature and label arrays.
    Falls back to synthetic tensor if sample files are not yet created.
    """
    x_path = os.path.join(data_dir, "X_padded.npy")
    y_path = os.path.join(data_dir, "y_encoded.npy")

    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"Loaded dataset from {data_dir}: X={X.shape}, y={y.shape}")
    else:
        print(f"Data not found in {data_dir}. Generating synthetic demonstration tensor (112 classes)...")
        np.random.seed(42)
        num_samples = 112 * 4  # 4 samples per class for demonstration
        X = np.random.uniform(0.0, 1.0, (num_samples, 20, 75)).astype(np.float32)
        y_indices = np.repeat(np.arange(112), 4)
        y = np.zeros((num_samples, 112), dtype=np.float32)
        for i, idx in enumerate(y_indices):
            y[i, idx] = 1.0

    return X, y


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Hijaiyah Sign Language Deep Learning Training Pipeline")
    print(f"Model       : {args.model.upper()}")
    print(f"Split Ratio : {args.split}")
    print(f"Batch Size  : {args.batch_size}")
    print(f"Max Epochs  : {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)

    # 1. Load Data & Perform Stratified Split
    X, y = load_data(args.data_dir)
    test_size = 0.30 if args.split == "70:30" else 0.20
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=test_size)
    print(f"Data split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

    # 2. Build Model
    input_shape = (20, 75)
    num_classes = y.shape[1]

    if args.model == "transformer":
        model = create_transformer_model(input_shape, num_classes, learning_rate=args.lr)
    elif args.model == "bilstm":
        model = create_bilstm_model(input_shape, num_classes, learning_rate=args.lr)
    else:
        model = create_hybrid_model(input_shape, num_classes, learning_rate=args.lr)

    model.summary()

    # 3. Callbacks Setup
    model_checkpoint_path = os.path.join(
        args.output_dir,
        f"{args.model}_{args.split.replace(':', '_')}_best.h5"
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_checkpoint_path,
            monitor="val_categorical_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )
    ]

    # 4. Train Model
    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Evaluate on Unseen Test Set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("\n" + "=" * 40)
    print(f"Final Evaluation on Test Set ({args.split}):")
    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {accuracy * 100:.2f}%")
    print(f"Saved Checkpoint : {model_checkpoint_path}")
    print("=" * 40)


if __name__ == "__main__":
    main()
