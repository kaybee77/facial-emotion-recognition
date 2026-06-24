"""
Classify facial emotion in new images using the trained VGG-19 FER2013 model.

Loads best_model_fer.keras (produced by train_fer2013.py) and predicts the
emotion for a single image or every image in a folder. Preprocessing matches
training exactly: convert to grayscale, resize to 48x48, expand to 3 channels,
scale to [0, 1].

Optionally detects+crops the face first (--detect) using OpenCV's Haar cascade,
which improves accuracy on real photos where the face isn't already centered.

Usage:
    python predict_fer.py --input some_face.jpg
    python predict_fer.py --input ./photos --topk 3 --detect
    python predict_fer.py --input ./photos --csv results.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import cv2
from tensorflow import keras

IMG_SIZE = 48
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
DEFAULT_CLASSES = ["anger", "disgust", "fear", "happiness",
                   "sadness", "surprise", "neutral"]


def load_class_names(path):
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return DEFAULT_CLASSES


def list_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
    raise SystemExit(f"Input not found: {input_path}")


def _face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def preprocess(path, detect, cascade):
    """Return a (48, 48, 3) float32 array in [0, 1], or None if unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if detect and cascade is not None:
        faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(48, 48))
        if len(faces) > 0:
            # use the largest detected face
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            img = img[y:y + h, x:x + w]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) / 255.0
    return img


def main():
    ap = argparse.ArgumentParser(description="Predict facial emotion with the trained FER2013 VGG-19 model")
    ap.add_argument("--model", default="best_model_fer.keras")
    ap.add_argument("--input", required=True, help="Image file or folder of images")
    ap.add_argument("--classes", default="fer_classes.json",
                    help="JSON list of class names in label order")
    ap.add_argument("--topk", type=int, default=1, help="Show top-K predictions")
    ap.add_argument("--detect", action="store_true",
                    help="Detect & crop the face first (recommended for real photos)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--csv", default="", help="Optional path to write results as CSV")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model} (run train_fer2013.py first)")

    model = keras.models.load_model(args.model)
    class_names = load_class_names(args.classes)
    topk = max(1, min(args.topk, len(class_names)))
    cascade = _face_cascade() if args.detect else None

    paths = list_images(args.input)
    batch, kept = [], []
    for p in paths:
        arr = preprocess(p, args.detect, cascade)
        if arr is None:
            print(f"{p.name:40s} -> (could not read, skipped)")
            continue
        batch.append(arr)
        kept.append(p)
    if not batch:
        raise SystemExit("No readable images found.")

    print(f"Loaded {args.model} | {len(class_names)} classes | {len(kept)} image(s)"
          + (" | face-detect ON" if args.detect else "") + "\n")

    probs = model.predict(np.stack(batch, 0), batch_size=args.batch_size, verbose=0)

    rows = []
    for p, pr in zip(kept, probs):
        order = np.argsort(pr)[::-1][:topk]
        top = [(class_names[i], float(pr[i])) for i in order]
        label, conf = top[0]
        extra = "  ".join(f"{n}:{c:.3f}" for n, c in top[1:])
        print(f"{p.name:40s} -> {label:12s} ({conf:.1%})" + (f"   | {extra}" if extra else ""))
        rows.append({"file": str(p), "prediction": label, "confidence": f"{conf:.4f}",
                     **{f"top{j+1}": f"{n}:{c:.4f}" for j, (n, c) in enumerate(top)}})

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
