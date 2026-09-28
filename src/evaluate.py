"""
Evaluation and Confusion Matrix Analysis Script.
Computes Weighted Average Accuracy, Precision, Recall, and F1-Score matching Thesis Chapter V.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hijaiyah Sign Language Model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved .h5 model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/sample", help="Directory containing test data")
    parser.add_argument("--output-img", type=str, default="docs/confusion_matrix.png", help="Path to save confusion matrix image")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found.")
        return

    # Load Model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)

    # Load Test Data
    x_test_path = os.path.join(args.data_dir, "X_test.npy")
    y_test_path = os.path.join(args.data_dir, "y_test.npy")

    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print(f"Test files not found in {args.data_dir}. Generating synthetic test data for evaluation...")
        np.random.seed(42)
        X_test = np.random.uniform(0.0, 1.0, (112 * 2, 20, 75)).astype(np.float32)
        y_test_indices = np.repeat(np.arange(112), 2)
        y_test = np.zeros((112 * 2, 112), dtype=np.float32)
        for i, idx in enumerate(y_test_indices):
            y_test[i, idx] = 1.0
    else:
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)

    # Predictions
    print("Running model inference on test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

    # Classification Metrics (Weighted Average)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    weighted_avg = report.get("weighted avg", {})

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS (WEIGHTED AVERAGE):")
    print(f"Accuracy  : {report.get('accuracy', 0) * 100:.2f}%")
    print(f"Precision : {weighted_avg.get('precision', 0) * 100:.2f}%")
    print(f"Recall    : {weighted_avg.get('recall', 0) * 100:.2f}%")
    print(f"F1-Score  : {weighted_avg.get('f1-score', 0) * 100:.2f}%")
    print("=" * 50)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(args.output_img), exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Confusion Matrix - Hijaiyah Sign Language Recognition (112 Classes)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(args.output_img, dpi=300)
    plt.close()
    print(f"Confusion Matrix saved to: {args.output_img}")


if __name__ == "__main__":
    main()
