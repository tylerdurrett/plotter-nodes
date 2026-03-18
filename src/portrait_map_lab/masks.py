"""Binary mask generation from facial regions."""

from __future__ import annotations

import cv2
import numpy as np

from portrait_map_lab.face_regions import get_region_polygons
from portrait_map_lab.models import LandmarkResult, RegionDefinition


def rasterize_mask(polygon: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon to a binary mask.

    Parameters
    ----------
    polygon
        Nx2 array of polygon vertex coordinates.
    image_shape
        Output mask dimensions as (height, width).

    Returns
    -------
    np.ndarray
        Binary mask as uint8 array with values 0 or 255.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    if len(polygon) > 0:
        points = polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], 255)
    return mask


def build_region_masks(
    landmarks: LandmarkResult, regions: list[RegionDefinition]
) -> dict[str, np.ndarray]:
    """Build binary masks for all facial regions.

    Parameters
    ----------
    landmarks
        Face landmark detection result.
    regions
        List of region definitions specifying landmark indices.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from region name to binary mask. Includes a special
        "combined_eyes" key with the union of left and right eye masks.
    """
    polygons = get_region_polygons(landmarks, regions)
    masks = {}

    for name, polygon in polygons.items():
        masks[name] = rasterize_mask(polygon, landmarks.image_shape)

    # Add combined eyes mask if both eyes present
    if "left_eye" in masks and "right_eye" in masks:
        masks["combined_eyes"] = cv2.bitwise_or(masks["left_eye"], masks["right_eye"])

    return masks
