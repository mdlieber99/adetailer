"""Mediapipe Tasks (`mediapipe.tasks.python.vision`) implementation.

Used when the legacy `mp.solutions` API is unavailable, which is the case from
mediapipe 0.10.30 onwards (and therefore on Python 3.13, where only those
versions have wheels).  The output of every function here is shaped exactly
like the corresponding `adetailer.mediapipe` legacy function, so downstream
code cannot tell the two apart.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from rich import print  # noqa: A004  Shadowing built-in 'print'

from adetailer.common import (
    PredictOutput,
    create_bbox_from_mask,
    create_mask_from_bbox,
)
from adetailer.mediapipe_data import (
    FACEMESH_FACE_OVAL,
    FACEMESH_LEFT_EYE,
    FACEMESH_RIGHT_EYE,
)

if TYPE_CHECKING:
    import mediapipe as mp

FACE_DETECTOR_MODEL = "blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL = "face_landmarker.task"

# Canonical Google-hosted assets, with a huggingface mirror as a second try.
MODEL_URLS: dict[str, tuple[str, ...]] = {
    FACE_DETECTOR_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
        "https://huggingface.co/spaces/learnmlf/FollowYourEmoji/resolve/main/media_pipe/mp_models/blaze_face_short_range.tflite",
    ),
    FACE_LANDMARKER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "https://huggingface.co/lithiumice/models_hub/resolve/main/face_landmarker.task",
    ),
}

MAX_NUM_FACES = 20


def model_dir() -> Path:
    """`<extension>/models/mediapipe`, overridable for tests / custom setups."""
    env = os.environ.get("ADETAILER_MEDIAPIPE_MODEL_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "models" / "mediapipe"


def _download(url: str, path: Path) -> str:
    """Download `url` to `path`; return "" on success or the error text."""
    tmp = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return f"    {url}\n      {type(e).__name__}: {e}"
    return ""


def get_model_path(name: str) -> Path:
    """Return the cached model file, downloading it on first use."""
    path = model_dir() / name
    if path.is_file() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in MODEL_URLS[name]:
        error = _download(url, path)
        if not error:
            return path
        errors.append(error)

    tried = "\n".join(errors)
    msg = (
        f"[-] ADetailer: failed to download the mediapipe model {name!r}\n"
        f"  target path: {path}\n"
        f"  tried:\n{tried}\n"
        f"  Download the file manually and place it at the target path."
    )
    # `mediapipe_predict` swallows exceptions, so make sure this is visible.
    print(msg)
    raise RuntimeError(msg)


def _base_options(name: str) -> Any:
    from mediapipe.tasks.python import BaseOptions

    return BaseOptions(model_asset_path=str(get_model_path(name)))


def _to_mp_image(arr: np.ndarray) -> mp.Image:
    import mediapipe as mp

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(arr, dtype=np.uint8),
    )


def mediapipe_face_detection(
    model_type: int, image: Image.Image, confidence: float = 0.3
) -> PredictOutput[float]:
    from mediapipe.tasks.python import vision

    # Mediapipe Tasks only publishes the short-range BlazeFace model; there is
    # no Tasks equivalent of the legacy full-range model (model_selection=1),
    # so "mediapipe_face_full" is approximated with the short-range detector.
    del model_type

    arr = np.array(image)

    options = vision.FaceDetectorOptions(
        base_options=_base_options(FACE_DETECTOR_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=confidence,
    )
    with vision.FaceDetector.create_from_options(options) as face_detector:
        pred = face_detector.detect(_to_mp_image(arr))

    if not pred.detections:
        return PredictOutput()

    preview_array = arr.copy()

    bboxes = []
    confidences = []
    for detection in pred.detections:
        _draw_detection(preview_array, detection)

        box = detection.bounding_box
        x1 = float(box.origin_x)
        y1 = float(box.origin_y)
        x2 = x1 + box.width
        y2 = y1 + box.height

        # The legacy protobuf `detection.score` is a repeated field, so its
        # entries are length-1 sequences, not floats. Keep that shape.
        confidences.append([detection.categories[0].score])
        bboxes.append([x1, y1, x2, y2])

    masks = create_mask_from_bbox(bboxes, image.size)
    preview = Image.fromarray(preview_array)

    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def _landmark_points(landmarks: list[Any], size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.array([[land.x * w, land.y * h] for land in landmarks], dtype=int)


def _detect_landmarks(arr: np.ndarray, confidence: float) -> Any:
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=_base_options(FACE_LANDMARKER_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=MAX_NUM_FACES,
        min_face_detection_confidence=confidence,
    )
    with vision.FaceLandmarker.create_from_options(options) as face_landmarker:
        return face_landmarker.detect(_to_mp_image(arr))


def mediapipe_face_mesh(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    arr = np.array(image)
    pred = _detect_landmarks(arr, confidence)

    if not pred.face_landmarks:
        return PredictOutput()

    preview = arr.copy()
    masks = []
    confidences = []

    for landmarks in pred.face_landmarks:
        points = _landmark_points(landmarks, image.size)
        _draw_face_mesh(preview, points)

        outline = cv2.convexHull(points).reshape(-1).tolist()

        mask = Image.new("L", image.size, "black")
        draw = ImageDraw.Draw(mask)
        draw.polygon(outline, fill="white")
        masks.append(mask)
        confidences.append(1.0)  # Confidence is unknown

    bboxes = create_bbox_from_mask(masks, image.size)
    preview = Image.fromarray(preview)
    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def mediapipe_face_mesh_eyes_only(
    image: Image.Image, confidence: float = 0.3
) -> PredictOutput[int]:
    from adetailer.mediapipe import draw_preview

    left_idx = np.array(list(FACEMESH_LEFT_EYE)).flatten()
    right_idx = np.array(list(FACEMESH_RIGHT_EYE)).flatten()

    arr = np.array(image)
    pred = _detect_landmarks(arr, confidence)

    if not pred.face_landmarks:
        return PredictOutput()

    preview = image.copy()
    masks = []
    confidences = []

    for landmarks in pred.face_landmarks:
        points = _landmark_points(landmarks, image.size)
        left_eyes = points[left_idx]
        right_eyes = points[right_idx]
        left_outline = cv2.convexHull(left_eyes).reshape(-1).tolist()
        right_outline = cv2.convexHull(right_eyes).reshape(-1).tolist()

        mask = Image.new("L", image.size, "black")
        draw = ImageDraw.Draw(mask)
        for outline in (left_outline, right_outline):
            draw.polygon(outline, fill="white")
        masks.append(mask)
        confidences.append(1.0)  # Confidence is unknown

    bboxes = create_bbox_from_mask(masks, image.size)
    preview = draw_preview(preview, bboxes, masks)
    return PredictOutput(
        bboxes=bboxes, masks=masks, confidences=confidences, preview=preview
    )


def _draw_detection(preview: np.ndarray, detection: Any) -> None:
    """Stand-in for `mp.solutions.drawing_utils.draw_detection`."""
    box = detection.bounding_box
    cv2.rectangle(
        preview,
        (box.origin_x, box.origin_y),
        (box.origin_x + box.width, box.origin_y + box.height),
        (0, 255, 0),
        2,
    )
    for keypoint in detection.keypoints:
        h, w = preview.shape[:2]
        center = (int(keypoint.x * w), int(keypoint.y * h))
        cv2.circle(preview, center, 2, (255, 0, 0), 2)


def _draw_face_mesh(preview: np.ndarray, points: np.ndarray) -> None:
    """Stand-in for the legacy tesselation drawing: oval outline + landmarks."""
    for x, y in points:
        cv2.circle(preview, (int(x), int(y)), 1, (192, 192, 192), -1)
    for start, end in FACEMESH_FACE_OVAL:
        p1 = points[start]
        p2 = points[end]
        cv2.line(
            preview, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 1
        )
