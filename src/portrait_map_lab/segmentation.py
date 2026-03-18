"""Image segmentation using MediaPipe selfie multiclass model.

This module provides semantic segmentation of portrait images into six
classes (background, hair, body skin, face skin, clothes, accessories)
and extraction of contour polygons from segmentation masks.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_multiclass_256x256/float32/latest/"
    "selfie_multiclass_256x256.tflite"
)
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
_MODEL_FILENAME = "selfie_multiclass_256x256.tflite"

# Segmentation class indices
SEGMENTATION_BACKGROUND = 0
SEGMENTATION_HAIR = 1
SEGMENTATION_BODY_SKIN = 2
SEGMENTATION_FACE_SKIN = 3
SEGMENTATION_CLOTHES = 4
SEGMENTATION_ACCESSORIES = 5


def _get_segmentation_model_path() -> Path:
    """Return path to the segmentation model file, downloading if necessary.

    The model is cached in ``<project_root>/models/selfie_multiclass_256x256.tflite``.
    Set the ``SEGMENTATION_MODEL_PATH`` environment variable to override
    the default location.
    """
    override = os.environ.get("SEGMENTATION_MODEL_PATH")
    if override:
        path = Path(override)
        if not path.exists():
            raise RuntimeError(f"Model file not found at override path: {path}")
        return path

    path = _MODEL_DIR / _MODEL_FILENAME
    if path.exists():
        return path

    logger.info("Downloading segmentation model to %s ...", path)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(_MODEL_URL, path)
    except Exception as exc:
        # Clean up partial download
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download segmentation model: {exc}") from exc
    logger.info("Model downloaded successfully.")
    return path


def segment_image(image: np.ndarray) -> np.ndarray:
    """Segment a BGR image using MediaPipe multiclass selfie segmenter.

    Parameters
    ----------
    image
        BGR uint8 image array (as loaded by ``cv2.imread``).

    Returns
    -------
    np.ndarray
        Category mask at original image resolution, uint8, shape (H, W).
        Each pixel value is a class index:
        0=background, 1=hair, 2=body skin, 3=face skin,
        4=clothes, 5=accessories.
    """
    h, w = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    model_path = _get_segmentation_model_path()
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_category_mask=True,
    )

    with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp_image)

    # category_mask is an mp.Image; extract as numpy and copy
    # (the underlying buffer is owned by MediaPipe and may be freed)
    category_mask = result.category_mask.numpy_view().copy()

    # Upscale from model resolution (256x256) to original image size
    if category_mask.shape != (h, w):
        category_mask = cv2.resize(category_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return category_mask.astype(np.uint8)


def extract_segmentation_polygon(
    category_mask: np.ndarray,
    classes: list[int],
    epsilon_factor: float = 0.005,
) -> np.ndarray:
    """Extract contour polygon from segmentation mask for given classes.

    Combines the specified class masks into a single binary mask,
    applies morphological closing to fill small holes, then extracts
    the largest contour as a polygon.

    Parameters
    ----------
    category_mask
        Full-resolution category mask from :func:`segment_image`.
    classes
        List of class indices to combine (e.g. ``[3]`` for face skin,
        ``[1, 3, 5]`` for full head).
    epsilon_factor
        Simplification factor for ``cv2.approxPolyDP``.
        ``epsilon = epsilon_factor * arc_length``.
        Set to 0 to disable simplification.

    Returns
    -------
    np.ndarray
        Contour polygon as Nx2 array of pixel coordinates (float64),
        compatible with downstream rasterize/distance functions.

    Raises
    ------
    ValueError
        If no contour is found for the specified classes.
    """
    # Build combined binary mask
    binary_mask = np.zeros(category_mask.shape, dtype=np.uint8)
    for cls in classes:
        binary_mask[category_mask == cls] = 255

    # Morphological close to fill small gaps and smooth boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError(
            f"No contour found for segmentation classes {classes}. "
            "The image may not contain a detectable face."
        )

    # Take largest contour by area
    contour = max(contours, key=cv2.contourArea)

    # Simplify if requested
    if epsilon_factor > 0:
        arc_length = cv2.arcLength(contour, closed=True)
        epsilon = epsilon_factor * arc_length
        contour = cv2.approxPolyDP(contour, epsilon, closed=True)

    # Reshape from (N, 1, 2) to (N, 2) and convert to float64
    polygon = contour[:, 0, :].astype(np.float64)

    return polygon


__all__ = [
    "segment_image",
    "extract_segmentation_polygon",
    "SEGMENTATION_BACKGROUND",
    "SEGMENTATION_HAIR",
    "SEGMENTATION_BODY_SKIN",
    "SEGMENTATION_FACE_SKIN",
    "SEGMENTATION_CLOTHES",
    "SEGMENTATION_ACCESSORIES",
]
