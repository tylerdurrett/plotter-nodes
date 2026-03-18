"""Face landmark detection using MediaPipe FaceLandmarker."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from portrait_map_lab.models import LandmarkResult

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_MODEL_FILENAME = "face_landmarker.task"


def _get_model_path() -> Path:
    """Return path to the FaceLandmarker model file, downloading if necessary.

    The model is cached in ``<project_root>/models/face_landmarker.task``.
    Set the ``FACE_LANDMARKER_MODEL_PATH`` environment variable to override
    the default location.
    """
    override = os.environ.get("FACE_LANDMARKER_MODEL_PATH")
    if override:
        path = Path(override)
        if not path.exists():
            raise RuntimeError(f"Model file not found at override path: {path}")
        return path

    path = _MODEL_DIR / _MODEL_FILENAME
    if path.exists():
        return path

    logger.info("Downloading FaceLandmarker model to %s ...", path)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, path)
    except Exception as exc:
        # Clean up partial download
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download FaceLandmarker model: {exc}") from exc
    logger.info("Model downloaded successfully.")
    return path


def detect_landmarks(image: np.ndarray) -> LandmarkResult:
    """Detect face landmarks in a BGR image using MediaPipe FaceLandmarker.

    Parameters
    ----------
    image:
        BGR uint8 image array (as loaded by ``cv2.imread``).

    Returns
    -------
    LandmarkResult
        Detected landmarks with pixel coordinates, image shape, and confidence.

    Raises
    ------
    ValueError
        If no face is detected in the image.
    """
    h, w = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    model_path = _get_model_path()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face detected in the image.")

    if len(result.face_landmarks) > 1:
        logger.warning(
            "Multiple faces detected (%d); using the first face.",
            len(result.face_landmarks),
        )

    face_lms = result.face_landmarks[0]

    landmarks = np.array(
        [[lm.x * w, lm.y * h] for lm in face_lms],
        dtype=np.float64,
    )

    presences = [lm.presence for lm in face_lms if lm.presence is not None]
    confidence = float(np.mean(presences)) if presences else 1.0

    if confidence < 0.5:
        logger.warning("Low landmark confidence: %.3f", confidence)

    return LandmarkResult(
        landmarks=landmarks,
        image_shape=(h, w),
        confidence=confidence,
    )
