"""
Real-time facial-emotion recognition from a webcam (or a video file).

Uses the trained VGG-19 FER2013 model (best_model_fer.keras). For each frame it
detects faces with OpenCV's Haar cascade, classifies each face's emotion, and
draws a bounding box + label + confidence. Preprocessing matches training
exactly (grayscale -> 48x48 -> 3 channels -> /255).

Controls:
    q or Esc  - quit
    s         - save a snapshot of the current annotated frame

Usage:
    python webcam_fer.py                       # default camera (index 0)
    python webcam_fer.py --camera 1            # a different camera
    python webcam_fer.py --video clip.mp4      # run on a video file instead
    python webcam_fer.py --every 3             # classify every 3rd frame (faster)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import cv2
from tensorflow import keras

IMG_SIZE = 48
DEFAULT_CLASSES = ["anger", "disgust", "fear", "happiness",
                   "sadness", "surprise", "neutral"]
# BGR colors per emotion for the boxes/labels
COLORS = {
    "anger": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
    "happiness": (0, 215, 255), "sadness": (255, 0, 0),
    "surprise": (0, 255, 255), "neutral": (200, 200, 200),
}


def load_class_names(path):
    if path and Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return DEFAULT_CLASSES


def preprocess_face(gray_roi):
    """grayscale ROI -> (1, 48, 48, 3) float32 in [0, 1]."""
    face = cv2.resize(gray_roi, (IMG_SIZE, IMG_SIZE)).astype("float32")
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB) / 255.0
    return np.expand_dims(face, 0)


def main():
    ap = argparse.ArgumentParser(description="Real-time FER with the trained VGG-19 model")
    ap.add_argument("--model", default="best_model_fer.keras")
    ap.add_argument("--classes", default="fer_classes.json")
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument("--video", default="", help="Run on a video file instead of a camera")
    ap.add_argument("--every", type=int, default=1,
                    help="Classify every Nth frame (reuse last result between)")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="Hide labels below this confidence")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model} (run train_fer2013.py first)")

    print("Loading model ...")
    model = keras.models.load_model(args.model)
    class_names = load_class_names(args.classes)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {'video '+args.video if args.video else 'camera '+str(args.camera)}")

    print("Running. Press 'q' or Esc to quit, 's' to save a snapshot.")
    frame_idx, last_faces, fps_t, fps = 0, [], time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx % max(1, args.every) == 0:
            detections = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            results = []
            for (x, y, w, h) in detections:
                probs = model.predict(preprocess_face(gray[y:y + h, x:x + w]),
                                      verbose=0)[0]
                k = int(np.argmax(probs))
                results.append((x, y, w, h, class_names[k], float(probs[k])))
            last_faces = results

        # draw (reuse last_faces on skipped frames)
        for (x, y, w, h, label, conf) in last_faces:
            if conf < args.min_conf:
                continue
            color = COLORS.get(label, (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, -1)
            cv2.putText(frame, text, (x + 3, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - fps_t, 1e-6))
        fps_t = now
        cv2.putText(frame, f"{fps:4.1f} FPS  |  faces: {len(last_faces)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

        cv2.imshow("Facial Emotion Recognition  (q=quit, s=snapshot)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or Esc
            break
        if key == ord("s"):
            fn = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(fn, frame)
            print(f"Saved {fn}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
