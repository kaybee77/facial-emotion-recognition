"""
Run the trained Keras classifier on new images.

Loads a model saved by train_classifier_keras.py and predicts the class (with
confidence) for a single image or every image in a folder. This is the Keras
equivalent of MATLAB's `classify(net, img)`.

Usage:
    python predict.py --model best_model.keras --input some_image.jpg
    python predict.py --model best_model.keras --input ./images_to_classify --topk 3
    python predict.py --model best_model.keras --input ./images --csv results.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from tensorflow import keras

INPUT_SIZE = 224
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}

# The trained model already contains its preprocessing layers (Resizing /
# backbone preprocess_input were baked into train_classifier_keras.py), so here
# we only need to load the raw RGB image at the right size.


def list_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
    raise SystemExit(f"Input not found: {input_path}")


def load_image(path):
    img = keras.utils.load_img(path, target_size=(INPUT_SIZE, INPUT_SIZE))
    return keras.utils.img_to_array(img)  # float32 HxWx3, range 0-255


def infer_class_names(model, provided):
    if provided:
        return [c.strip() for c in provided.split(",")]
    n = model.output_shape[-1]
    return [f"class_{i}" for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description="Classify images with a trained Keras model")
    ap.add_argument("--model", default="best_model.keras", help="Path to saved .keras model")
    ap.add_argument("--input", required=True, help="Image file or folder of images")
    ap.add_argument("--classes", default="",
                    help="Optional comma-separated class names (in label order)")
    ap.add_argument("--topk", type=int, default=1, help="Show top-K predictions")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--csv", default="", help="Optional path to write results as CSV")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}")

    model = keras.models.load_model(args.model)
    class_names = infer_class_names(model, args.classes)
    topk = max(1, min(args.topk, len(class_names)))

    paths = list_images(args.input)
    if not paths:
        raise SystemExit("No images found.")
    print(f"Loaded {args.model} | {len(class_names)} classes | {len(paths)} image(s)\n")

    batch = np.stack([load_image(p) for p in paths], axis=0)
    probs = model.predict(batch, batch_size=args.batch_size, verbose=0)

    rows = []
    for path, p in zip(paths, probs):
        order = np.argsort(p)[::-1][:topk]
        top = [(class_names[i], float(p[i])) for i in order]
        label, conf = top[0]
        extra = "  ".join(f"{n}:{c:.3f}" for n, c in top[1:])
        line = f"{path.name:40s} -> {label:20s} ({conf:.3f})"
        print(line + (f"   | {extra}" if extra else ""))
        rows.append({"file": str(path), "prediction": label, "confidence": f"{conf:.4f}",
                     **{f"top{j+1}": f"{n}:{c:.4f}" for j, (n, c) in enumerate(top)}})

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
