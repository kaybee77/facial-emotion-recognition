"""
Evaluate a trained FER2013 model on the held-out 15% test set.

Recreates the exact same stratified 70/15/15 split as train_fer2013.py (same
seed), loads a saved .keras model, and reports accuracy, balanced accuracy,
macro-F1, the confusion matrix, and a per-class report. Useful for recovering
test metrics when a training run was interrupted after the model was saved.

Usage:
    python evaluate.py --model model_classweights.keras
    python evaluate.py --model model_classweights.keras --cm-out confusion_matrix.png
"""

import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

from train_fer2013 import load_fer2013, stratified_70_15_15, EMOTION_NAMES
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a FER2013 model on the test split")
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", default="fer2013.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cm-out", default="", help="Optional path to save confusion-matrix PNG")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}")

    feats, y, y_int, le = load_fer2013(args.csv)
    _, _, _, _, X_test, y_test = stratified_70_15_15(feats, y, y_int, args.seed)
    names = [EMOTION_NAMES.get(int(c), int(c)) for c in le.classes_]

    model = keras.models.load_model(args.model)
    y_pred = model.predict(X_test, batch_size=args.batch_size, verbose=0).argmax(1)
    y_true = y_test.argmax(1)

    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n==== {args.model} ====")
    print(f"accuracy: {acc:.4f} | balanced accuracy: {bal:.4f} | macro-F1: {mf1:.4f}")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    if args.cm_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=names)\
            .plot(cmap="Blues", xticks_rotation=45)
        plt.tight_layout(); plt.savefig(args.cm_out, dpi=150)
        print(f"Saved {args.cm_out}")


if __name__ == "__main__":
    main()
