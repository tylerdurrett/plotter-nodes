"""Face contour distance field computation.

This module provides functions to extract face oval contours from MediaPipe
landmarks, rasterize them as masks, compute signed distance fields, and
prepare directional distance fields for influence mapping.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from portrait_map_lab.distance_fields import compute_distance_field
from portrait_map_lab.models import LandmarkResult


def get_face_oval_polygon(landmarks: LandmarkResult) -> np.ndarray:
    """Extract the face boundary as the convex hull of all landmarks.

    Uses the convex hull of all 478 landmark points, which gives the
    tightest outer boundary that encloses every detected face landmark.
    This captures the full visible face including cheeks, which the
    FACEMESH_FACE_OVAL topology can miss due to 3D→2D projection.

    Parameters
    ----------
    landmarks
        Face landmark detection result containing 478 landmark points.

    Returns
    -------
    np.ndarray
        Face boundary polygon as Nx2 array of pixel coordinates (float64),
        ordered clockwise.
    """
    points = landmarks.landmarks.astype(np.float64)
    hull = ConvexHull(points)
    # ConvexHull vertices are counterclockwise; reverse for clockwise
    hull_points = points[hull.vertices[::-1]]
    return hull_points


def rasterize_contour_mask(
    polygon: np.ndarray, image_shape: tuple[int, int], thickness: int = 1
) -> np.ndarray:
    """Rasterize a polygon outline as a binary mask.

    Draws the polygon as a closed polyline with the specified thickness.

    Parameters
    ----------
    polygon
        Nx2 array of polygon vertices in pixel coordinates.
    image_shape
        Output mask shape as (height, width).
    thickness
        Line thickness in pixels. Default is 1.

    Returns
    -------
    np.ndarray
        Binary mask (uint8) with values 0 or 255.
        Shape is image_shape.
    """
    # Create empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to integer coordinates for cv2
    pts = np.round(polygon).astype(np.int32)

    # Draw closed polyline
    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=thickness)

    return mask


def rasterize_filled_mask(polygon: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a filled polygon as a binary mask.

    Fills the interior of the polygon.

    Parameters
    ----------
    polygon
        Nx2 array of polygon vertices in pixel coordinates.
    image_shape
        Output mask shape as (height, width).

    Returns
    -------
    np.ndarray
        Binary mask (uint8) with values 0 or 255.
        Shape is image_shape.
    """
    # Create empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to integer coordinates for cv2
    pts = np.round(polygon).astype(np.int32)

    # Fill polygon
    cv2.fillPoly(mask, [pts], color=255)

    return mask


def compute_signed_distance(contour_mask: np.ndarray, filled_mask: np.ndarray) -> np.ndarray:
    """Compute signed distance field from contour and filled masks.

    Computes Euclidean distance from the contour, with negative values
    inside the filled region and positive values outside.

    Parameters
    ----------
    contour_mask
        Binary mask of the contour outline (0 or 255).
    filled_mask
        Binary mask of the filled interior (0 or 255).

    Returns
    -------
    np.ndarray
        Signed distance field (float64) where:
        - Pixels on the contour have distance ~0.0
        - Interior pixels (inside filled mask) have negative distances
        - Exterior pixels (outside filled mask) have positive distances
    """
    # Compute unsigned distance from contour
    unsigned_distance = compute_distance_field(contour_mask)

    # Create signed distance field
    signed_distance = unsigned_distance.copy()

    # Apply sign based on filled mask
    # Inside the filled region: negative distance
    # Outside the filled region: positive distance
    interior_mask = filled_mask > 0
    signed_distance[interior_mask] = -signed_distance[interior_mask]

    # Ensure contour pixels are exactly 0
    # This handles any rasterization differences between polylines and fillPoly
    contour_pixels = contour_mask > 0
    signed_distance[contour_pixels] = 0.0

    return signed_distance


def prepare_directional_distance(
    signed_distance: np.ndarray,
    mode: str = "inward",
    clamp_value: float = 9999.0,
    band_width: float | None = None,
) -> np.ndarray:
    """Convert signed distance to directional unsigned distance.

    Prepares the distance field for influence mapping by selecting
    and transforming distances based on the specified direction mode.

    Parameters
    ----------
    signed_distance
        Signed distance field from compute_signed_distance.
    mode
        Direction mode for distance selection:
        - "inward": Keep interior distances, clamp exterior
        - "outward": Keep exterior distances, clamp interior
        - "both": Use absolute distance everywhere
        - "band": Use absolute distance, clamp beyond band_width
    clamp_value
        Value to use for clamped regions. Default is 9999.0.
    band_width
        Maximum distance for "band" mode. Required if mode="band".

    Returns
    -------
    np.ndarray
        Unsigned distance field (float64) prepared for influence mapping.

    Raises
    ------
    ValueError
        If mode is not recognized or band_width is missing for "band" mode.
    """
    if mode == "inward":
        # Keep interior distances (negative values), clamp exterior
        result = np.where(signed_distance < 0, np.abs(signed_distance), clamp_value)
    elif mode == "outward":
        # Keep exterior distances (positive values), clamp interior
        result = np.where(signed_distance > 0, signed_distance, clamp_value)
    elif mode == "both":
        # Use absolute distance everywhere
        result = np.abs(signed_distance)
    elif mode == "band":
        if band_width is None:
            raise ValueError("band_width is required for 'band' mode")
        # Use absolute distance, clamp beyond band_width
        abs_dist = np.abs(signed_distance)
        result = np.where(abs_dist <= band_width, abs_dist, clamp_value)
    else:
        raise ValueError(f"Unknown direction mode: {mode}")

    return result


__all__ = [
    "get_face_oval_polygon",
    "rasterize_contour_mask",
    "rasterize_filled_mask",
    "compute_signed_distance",
    "prepare_directional_distance",
]
