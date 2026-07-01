---
title: Facial Emotion Recognition
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Facial Emotion Recognition

A Gradio demo for facial emotion recognition. The app accepts webcam snapshots
or uploaded photos, detects faces, classifies each face, draws labeled boxes on
the image, and shows confidence scores for the largest detected face.

The model recognizes 7 emotions: **anger, disgust, fear, happiness, sadness,
surprise, neutral**.

## Project Layout

| File | Purpose |
|------|---------|
| `app.py` | Gradio web UI for webcam/upload emotion recognition |
| `face_detect.py` | Shared MediaPipe face detection with OpenCV Haar fallback |
| `fer_preprocess.py` | Face cropping, lighting normalization, and test-time augmentation |
| `fer_classes.json` | Emotion class labels used by the model |
| `blaze_face_short_range.tflite` | MediaPipe face detector model |
| `model_classweights.keras` | Trained emotion-recognition model |
| `Dockerfile` | Container build used by Hugging Face Spaces |
| `requirements.txt` | Runtime Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

> TensorFlow is CPU-only on native Windows for versions >= 2.11. Use WSL2 or a
> Linux machine with CUDA for GPU acceleration.

## Run Locally

```bash
python app.py
```

The app starts on `http://127.0.0.1:7860` by default. Set `FER_PORT` to use a
different port:

```bash
FER_PORT=7861 python app.py
```

To create a temporary public Gradio share link:

```bash
FER_SHARE=1 python app.py
```

## Model Artifacts

This runtime demo expects these files in the project root:

- `model_classweights.keras`
- `fer_classes.json`
- `blaze_face_short_range.tflite`

`fer2013.csv` is only needed for training experiments and is not required to run
the web app.

## Docker

Build and run the app with Docker:

```bash
docker build -t facial-emotion-recognition .
docker run --rm -p 7860:7860 facial-emotion-recognition
```

## Limitations

This is a portfolio/demo project, not a psychological assessment tool. Facial
emotion recognition is sensitive to lighting, pose, camera quality, face crop,
dataset bias, and expression ambiguity. It works best on clear, front-facing
faces and should be treated as an approximate computer-vision demo rather than a
reliable measure of a person's real emotional state.

## Tech Stack

TensorFlow / Keras, OpenCV, MediaPipe, Gradio.

## License

MIT
