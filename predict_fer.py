"""
Classify facial emotion in new images using the trained VGG-19 FER2013 model.

Loads model_classweights.keras when available and predicts the emotion for a
single image or every image in a folder. Preprocessing converts to grayscale,
crops detected faces, resizes to 48x48, expands to 3 channels, scales to [0, 1],
and averages a few stable test-time augmentation variants.

Optionally detects+crops the face first (--detect) using MediaPipe when
available, with OpenCV Haar as a fallback.

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

import face_detect
import fer_preprocess

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
DEFAULT_MODEL = "model_classweights.keras"
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


def preprocess(path, detect):
    """Return grayscale face/full-image crop plus whether a face was detected."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    detected = False
    if detect:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = face_detect.detect_faces(rgb, min_size=40)
        if faces:
            # use the largest detected face
            box = max(faces, key=lambda r: r[2] * r[3])
            gray = fer_preprocess.crop_with_padding(gray, box)
            detected = True
    return gray, detected


def main():
    ap = argparse.ArgumentParser(description="Predict facial emotion with the trained FER2013 VGG-19 model")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--input", required=True, help="Image file or folder of images")
    ap.add_argument("--classes", default="fer_classes.json",
                    help="JSON list of class names in label order")
    ap.add_argument("--topk", type=int, default=1, help="Show top-K predictions")
    ap.add_argument("--detect", action="store_true",
                    help="Detect & crop the face first (recommended for real photos)")
    ap.add_argument("--no-tta", action="store_true",
                    help="Disable upload-friendly test-time augmentation")
    ap.add_argument("--smile-prior", action="store_true",
                    help="Opt-in heuristic that nudges clear smiles toward happiness")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--csv", default="", help="Optional path to write results as CSV")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model} (run train_fer2013.py first)")

    model = keras.models.load_model(args.model)
    class_names = load_class_names(args.classes)
    topk = max(1, min(args.topk, len(class_names)))
    if args.detect:
        print(f"Face detector: {face_detect.backend()}")

    paths = list_images(args.input)
    kept, crops, detected_flags = [], [], []
    for p in paths:
        item = preprocess(p, args.detect)
        if item is None:
            print(f"{p.name:40s} -> (could not read, skipped)")
            continue
        crop, detected = item
        kept.append(p)
        crops.append(crop)
        detected_flags.append(detected)
    if not crops:
        raise SystemExit("No readable images found.")

    print(f"Loaded {args.model} | {len(class_names)} classes | {len(kept)} image(s)"
          + (" | face-detect ON" if args.detect else "")
          + (" | TTA ON" if not args.no_tta else " | TTA OFF")
          + (" | smile-prior ON" if args.smile_prior else "") + "\n")

    if args.no_tta:
        batch = np.stack([fer_preprocess.preprocess_face(c) for c in crops], axis=0)
        probs = model.predict(batch, batch_size=args.batch_size, verbose=0)
    else:
        probs = np.stack([
            fer_preprocess.predict_face_probs(model, crop, use_tta=True)
            for crop in crops
        ], axis=0)

    rows = []
    for p, pr, crop, detected in zip(kept, probs, crops, detected_flags):
        if detected and args.smile_prior:
            pr = fer_preprocess.apply_smile_prior(pr, class_names, crop)
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
