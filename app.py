"""
Facial Emotion Recognition - exhibition web UI (Gradio).

A lightweight, polished demo for showing the trained VGG-19 FER2013 model to an
audience. Visitors can either use their webcam or upload a photo; the app detects
faces, classifies each one's emotion, draws labeled boxes, and shows a confidence
bar chart for the main face.

Run:
    pip install gradio          # one-time
    python app.py               # opens http://127.0.0.1:7860 in your browser

Options (environment variables):
    FER_SHARE=1   -> create a temporary public share link (gradio.live)
    FER_PORT=7860 -> change the local port
"""

import json
import os
from pathlib import Path

import numpy as np
import cv2
import gradio as gr
from tensorflow import keras

MODEL_PATH = "best_model_fer.keras"
CLASSES_PATH = "fer_classes.json"
IMG_SIZE = 48
DEFAULT_CLASSES = ["anger", "disgust", "fear", "happiness",
                   "sadness", "surprise", "neutral"]
# Friendly emoji per emotion for the exhibition vibe.
EMOJI = {"anger": "😠", "disgust": "🤢", "fear": "😨", "happiness": "😄",
         "sadness": "😢", "surprise": "😲", "neutral": "😐"}
BOX_BGR = {"anger": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
           "happiness": (0, 215, 255), "sadness": (255, 0, 0),
           "surprise": (0, 255, 255), "neutral": (200, 200, 200)}

# ---- load model + classes + face detector once ----
if not Path(MODEL_PATH).exists():
    raise SystemExit(f"Model not found: {MODEL_PATH} (run train_fer2013.py first)")
print("Loading model ...")
MODEL = keras.models.load_model(MODEL_PATH)
CLASSES = json.load(open(CLASSES_PATH)) if Path(CLASSES_PATH).exists() else DEFAULT_CLASSES
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _classify(gray_roi):
    """grayscale ROI -> dict {emotion: prob}."""
    face = cv2.resize(gray_roi, (IMG_SIZE, IMG_SIZE)).astype("float32")
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB) / 255.0
    probs = MODEL.predict(np.expand_dims(face, 0), verbose=0)[0]
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


def predict(image):
    """image: RGB numpy array from webcam/upload.
    Returns (annotated RGB image, label->confidence dict)."""
    if image is None:
        return None, {}
    rgb = image.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(60, 60))

    if len(faces) == 0:
        # No face found -> classify the whole frame so the demo still responds.
        scores = _classify(gray)
        return rgb, scores

    main_scores, main_area = {}, -1
    for (x, y, w, h) in faces:
        scores = _classify(gray[y:y + h, x:x + w])
        top = max(scores, key=scores.get)
        color = BOX_BGR.get(top, (0, 255, 0))
        cv2.rectangle(rgb, (x, y), (x + w, y + h), color, 3)
        label = f"{top} {scores[top]:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(rgb, (x, y - th - 10), (x + tw + 8, y), color, -1)
        cv2.putText(rgb, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        if w * h > main_area:  # keep the largest face for the bar chart
            main_area, main_scores = w * h, scores

    # Prettify the labels shown in the bar chart with emoji.
    pretty = {f"{EMOJI.get(k, '')} {k}": v for k, v in main_scores.items()}
    return rgb, pretty


# --------------------------- UI ---------------------------
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet")

with gr.Blocks(title="Facial Emotion Recognition") as demo:
    gr.Markdown(
        """
        # 🎭 Facial Emotion Recognition
        ### VGG-19 deep learning model · trained on FER2013 · 7 emotions
        Use your **webcam** or **upload a photo**, then press **Analyze**.
        The model detects each face and predicts its emotion.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(sources=["webcam", "upload"], type="numpy",
                           label="Webcam / Upload", height=380)
            btn = gr.Button("✨ Analyze Emotion", variant="primary", size="lg")
            gr.Markdown("*Tip: face the camera with good lighting for best results.*")
        with gr.Column(scale=1):
            out_img = gr.Image(label="Detected faces", height=380)
            out_lbl = gr.Label(num_top_classes=7, label="Emotion confidence")

    btn.click(predict, inputs=inp, outputs=[out_img, out_lbl])
    # Also analyze automatically when a new image/snapshot arrives.
    inp.change(predict, inputs=inp, outputs=[out_img, out_lbl])

    gr.Markdown(
        "<sub>Emotions: 😠 anger · 🤢 disgust · 😨 fear · 😄 happiness · "
        "😢 sadness · 😲 surprise · 😐 neutral</sub>"
    )


if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("FER_PORT", 7860)),
        share=bool(os.environ.get("FER_SHARE")),
        inbrowser=True,
    )
