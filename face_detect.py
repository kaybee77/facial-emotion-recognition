"""
Robust face detection shared by the FER scripts.

Uses Google MediaPipe Face Detection (Tasks API) when available — accurate on
real-world photos and webcam frames, with far fewer false positives than Haar.
Falls back to OpenCV's Haar cascade if MediaPipe or its model file is missing.

API:
    detect_faces(rgb_image, min_size=40) -> list of (x, y, w, h) boxes
    backend() -> "mediapipe" or "haar"

The input image must be RGB (uint8, HxWx3). Callers working in BGR/grayscale
should convert to RGB before calling, then crop their grayscale image with the
returned boxes.
"""

import atexit
from pathlib import Path

import cv2
import numpy as np

# MediaPipe Tasks face-detector model (downloaded once, ~230 KB).
_MODEL_PATH = str(Path(__file__).with_name("blaze_face_short_range.tflite"))

_DETECTOR = None
_USE_MP = None
_MP = None  # mediapipe module handle (for building mp.Image)


def _init():
    global _DETECTOR, _USE_MP, _MP
    if _USE_MP is not None:
        return
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        if not Path(_MODEL_PATH).exists():
            raise FileNotFoundError(_MODEL_PATH)
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
            min_detection_confidence=0.5)
        _DETECTOR = vision.FaceDetector.create_from_options(options)
        _MP = mp
        _USE_MP = True
        atexit.register(_cleanup)  # close cleanly to avoid a __del__ error at exit
    except Exception:
        _DETECTOR = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        _USE_MP = False


def _cleanup():
    """Close the MediaPipe detector cleanly at exit.

    MediaPipe's FaceDetector.__del__ re-runs close() during interpreter shutdown,
    which raises on the already-shutdown executor. We close once here (atexit runs
    before MediaPipe's own teardown) and then no-op the class __del__ so shutdown
    won't try to close it again.
    """
    global _DETECTOR
    if _USE_MP and _DETECTOR is not None:
        try:
            _DETECTOR.close()
        except Exception:
            pass
        try:
            type(_DETECTOR).__del__ = lambda self: None
        except Exception:
            pass
        _DETECTOR = None


def backend():
    _init()
    return "mediapipe" if _USE_MP else "haar"


def _area(box):
    return max(0, box[2]) * max(0, box[3])


def _intersection(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    w = max(0, min(ax1, bx1) - max(ax0, bx0))
    h = max(0, min(ay1, by1) - max(ay0, by0))
    return w * h


def _dedupe_nested_boxes(boxes):
    """Keep separate faces, suppress duplicate/nested false detections."""
    kept = []
    for box in sorted(boxes, key=_area, reverse=True):
        box_area = max(1, _area(box))
        duplicate = False
        for prev in kept:
            overlap = _intersection(box, prev)
            prev_area = max(1, _area(prev))
            iou = overlap / float(box_area + prev_area - overlap)
            contained = overlap / float(box_area)
            if iou > 0.45 or contained > 0.65:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def _detection_score(det):
    try:
        if det.categories:
            return float(det.categories[0].score)
    except Exception:
        pass
    return 1.0


def detect_faces(rgb_image, min_size=40, min_score=0.5, min_rel_area=0.0):
    """Return a list of (x, y, w, h) face boxes, clamped to image bounds."""
    _init()
    h_img, w_img = rgb_image.shape[:2]
    img_area = max(1, h_img * w_img)
    boxes = []

    if _USE_MP:
        img = np.ascontiguousarray(rgb_image, dtype=np.uint8)
        mp_image = _MP.Image(image_format=_MP.ImageFormat.SRGB, data=img)
        result = _DETECTOR.detect(mp_image)
        for det in result.detections:
            if _detection_score(det) < min_score:
                continue
            bb = det.bounding_box  # pixel coordinates
            x0, y0 = max(0, bb.origin_x), max(0, bb.origin_y)
            x1 = min(w_img, bb.origin_x + bb.width)
            y1 = min(h_img, bb.origin_y + bb.height)
            box = (x0, y0, x1 - x0, y1 - y0)
            if box[2] >= min_size and box[3] >= min_size and _area(box) / img_area >= min_rel_area:
                boxes.append(box)
    else:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        dets = _DETECTOR.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6,
            minSize=(min_size, min_size))
        boxes = [
            (int(x), int(y), int(w), int(h))
            for (x, y, w, h) in dets
            if (int(w) * int(h)) / img_area >= min_rel_area
        ]

    return _dedupe_nested_boxes(boxes)
