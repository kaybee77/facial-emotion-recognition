"""Shared FER image preprocessing helpers."""

import cv2
import numpy as np

IMG_SIZE = 48
_SMILE_CASCADE = None


def crop_with_padding(gray_image, box, pad=0.25):
    """Crop a detected face as a square with context around the box."""
    x, y, w, h = box
    h_img, w_img = gray_image.shape[:2]
    side = int(max(w, h) * (1.0 + 2.0 * pad))
    cx = x + w // 2
    cy = y + h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w_img, x0 + side)
    y1 = min(h_img, y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    return gray_image[y0:y1, x0:x1]


def normalize_lighting(gray_roi):
    """Improve very dark or low-contrast webcam crops without overprocessing."""
    if gray_roi.size == 0:
        return gray_roi
    if gray_roi.mean() >= 95 and gray_roi.std() >= 35:
        return gray_roi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(gray_roi)


def preprocess_face(gray_roi):
    """grayscale ROI -> (48, 48, 3) float32 in [0, 1]."""
    gray_roi = normalize_lighting(gray_roi)
    face = cv2.resize(gray_roi, (IMG_SIZE, IMG_SIZE)).astype("float32")
    return cv2.cvtColor(face, cv2.COLOR_GRAY2RGB) / 255.0


def _preprocess_variant(gray_roi, enhance):
    if enhance:
        gray_roi = normalize_lighting(gray_roi)
    face = cv2.resize(gray_roi, (IMG_SIZE, IMG_SIZE)).astype("float32")
    return cv2.cvtColor(face, cv2.COLOR_GRAY2RGB) / 255.0


def preprocess_face_variants(gray_roi):
    """Return upload-friendly test-time augmentation variants for one face."""
    base = _preprocess_variant(gray_roi, enhance=False)
    enhanced = _preprocess_variant(gray_roi, enhance=True)
    variants = [base, np.flip(base, axis=1)]

    # Include the enhanced crop only when it meaningfully changes the pixels.
    if np.mean(np.abs(base - enhanced)) > 0.01:
        variants.extend([enhanced, np.flip(enhanced, axis=1)])

    return np.stack(variants, axis=0).astype("float32")


def predict_face_probs(model, gray_roi, use_tta=True):
    """Predict one grayscale face crop, optionally averaging TTA variants."""
    if not use_tta:
        batch = np.expand_dims(preprocess_face(gray_roi), axis=0)
    else:
        batch = preprocess_face_variants(gray_roi)
    probs = np.asarray(model(batch, training=False))
    return probs.mean(axis=0)


def _smile_cascade():
    global _SMILE_CASCADE
    if _SMILE_CASCADE is None:
        _SMILE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml")
    return _SMILE_CASCADE


def smile_confidence(gray_roi):
    """Return a conservative 0..1 smile signal for live webcam crops."""
    if gray_roi.size == 0:
        return 0.0
    roi = normalize_lighting(gray_roi)
    h, w = roi.shape[:2]
    min_w = max(18, int(w * 0.18))
    min_h = max(10, int(h * 0.08))
    smiles = _smile_cascade().detectMultiScale(
        roi, scaleFactor=1.7, minNeighbors=18, minSize=(min_w, min_h))
    if len(smiles) == 0:
        return 0.0
    largest = max(sw * sh for (_, _, sw, sh) in smiles)
    area_ratio = largest / float(max(1, w * h))
    return float(min(1.0, 0.45 + area_ratio * 8.0))


def apply_smile_prior(probs, class_names, gray_roi):
    """Nudge predictions toward happiness when the live crop has a clear smile."""
    if "happiness" not in class_names:
        return probs
    smile = smile_confidence(gray_roi)
    if smile <= 0:
        return probs
    adjusted = np.asarray(probs, dtype="float32").copy()
    happy_idx = class_names.index("happiness")
    adjusted[happy_idx] += 1.35 * smile
    if smile >= 0.35:
        best_other = float(np.delete(adjusted, happy_idx).max())
        adjusted[happy_idx] = max(float(adjusted[happy_idx]), best_other + 0.08)
    adjusted /= max(float(adjusted.sum()), 1e-6)
    return adjusted
