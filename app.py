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
import tempfile
import zipfile
from pathlib import Path

import cv2
import gradio as gr
from tensorflow import keras

import face_detect
import fer_preprocess

MODEL_PATH = os.environ.get(
    "FER_MODEL",
    "model_classweights.keras",
)
USE_SMILE_PRIOR = bool(os.environ.get("FER_SMILE_PRIOR"))
CLASSES_PATH = "fer_classes.json"
EXAMPLES_DIR = Path("examples")
DEFAULT_CLASSES = ["anger", "disgust", "fear", "happiness",
                   "sadness", "surprise", "neutral"]
BOX_BGR = {"anger": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
           "happiness": (0, 215, 255), "sadness": (255, 0, 0),
           "surprise": (0, 255, 255), "neutral": (200, 200, 200)}

# ---- load model + classes + face detector once ----
if not Path(MODEL_PATH).exists():
    raise SystemExit(f"Model not found: {MODEL_PATH} (run train_fer2013.py first)")
print("Loading model ...")


def _drop_none_quantization_config(value):
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        for child in value.values():
            _drop_none_quantization_config(child)
    elif isinstance(value, list):
        for child in value:
            _drop_none_quantization_config(child)


def _load_model(path):
    try:
        return keras.models.load_model(path)
    except Exception as exc:
        if "quantization_config" not in str(exc):
            raise

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            patched_path = tmp.name

        with zipfile.ZipFile(path, "r") as src, zipfile.ZipFile(patched_path, "w") as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "config.json":
                    config = json.loads(data.decode("utf-8"))
                    _drop_none_quantization_config(config)
                    data = json.dumps(config).encode("utf-8")
                dst.writestr(item, data)

        return keras.models.load_model(patched_path)


MODEL = _load_model(MODEL_PATH)
CLASSES = json.load(open(CLASSES_PATH)) if Path(CLASSES_PATH).exists() else DEFAULT_CLASSES
print(f"Face detector: {face_detect.backend()}")


def _scores_from_probs(probs):
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}


def _classify(gray_roi):
    """grayscale ROI -> dict {emotion: prob}."""
    probs = fer_preprocess.predict_face_probs(MODEL, gray_roi, use_tta=True)
    return _scores_from_probs(probs)


def predict(image):
    """image: RGB numpy array from webcam/upload.
    Returns (annotated RGB image, label->confidence dict)."""
    if image is None:
        return None, {}
    rgb = image.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    min_face_size = max(40, int(min(rgb.shape[:2]) * 0.12))
    faces = face_detect.detect_faces(
        rgb,
        min_size=min_face_size,
        min_score=0.65,
        min_rel_area=0.01,
    )

    if len(faces) == 0:
        return rgb, {}

    face_crops = [fer_preprocess.crop_with_padding(gray, box) for box in faces]
    main_scores, main_area = {}, -1
    for (x, y, w, h), crop in zip(faces, face_crops):
        probs = fer_preprocess.predict_face_probs(MODEL, crop, use_tta=True)
        if USE_SMILE_PRIOR:
            probs = fer_preprocess.apply_smile_prior(probs, CLASSES, crop)
        scores = _scores_from_probs(probs)
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

    return rgb, main_scores


def _example_images():
    if not EXAMPLES_DIR.exists():
        return []
    return [
        [str(path)]
        for path in sorted(EXAMPLES_DIR.iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]


# --------------------------- UI ---------------------------
THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet")

with gr.Blocks(title="Facial Emotion Recognition", theme=THEME) as demo:
    gr.Markdown(
        """
        # Facial Emotion Recognition
        ### VGG-19 deep learning model trained on FER2013
        Use your **webcam** or **upload a photo**, then press **Analyze**.
        The model detects each face and predicts its emotion.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(sources=["webcam", "upload"], type="numpy",
                           label="Webcam / Upload", height=380)
            btn = gr.Button("Analyze Emotion", variant="primary", size="lg")
            gr.Markdown("*Tip: face the camera with good lighting for best results.*")
        with gr.Column(scale=1):
            out_img = gr.Image(label="Detected faces", height=380)
            out_lbl = gr.Label(num_top_classes=7, label="Emotion confidence")

    gr.Examples(
        examples=_example_images(),
        inputs=inp,
        outputs=[out_img, out_lbl],
        fn=predict,
        cache_examples=False,
    )

    btn.click(predict, inputs=inp, outputs=[out_img, out_lbl])
    # Also analyze automatically when a new image/snapshot arrives.
    inp.change(predict, inputs=inp, outputs=[out_img, out_lbl])

    gr.Markdown(
        "<sub>Emotions: anger, disgust, fear, happiness, sadness, surprise, neutral.</sub>"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("FER_PORT", 7860)),
        share=bool(os.environ.get("FER_SHARE")),
        inbrowser=True,
    )
