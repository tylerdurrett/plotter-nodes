"""Luminance extraction and tonal target computation for density mapping."""

from __future__ import annotations

import cv2
import numpy as np

from portrait_map_lab.models import LuminanceConfig

__all__ = ["extract_luminance", "apply_clahe", "compute_tonal_target"]


def extract_luminance(image: np.ndarray) -> np.ndarray:
    """Extract grayscale luminance from a BGR or grayscale image.

    Parameters
    ----------
    image
        Input image as BGR uint8 (H, W, 3) or grayscale uint8 (H, W).

    Returns
    -------
    np.ndarray
        Grayscale luminance as float64 array with values in [0, 1].
    """
    # Handle grayscale input
    if image.ndim == 2:
        gray = image
    # Handle BGR input
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {image.shape}")

    # Convert to float64 and normalize to [0, 1]
    return gray.astype(np.float64) / 255.0


def apply_clahe(luminance: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization to luminance.

    CLAHE enhances local contrast while limiting amplification to avoid noise.

    Parameters
    ----------
    luminance
        Grayscale luminance as float64 array with values in [0, 1].
    clip_limit
        Threshold for contrast limiting. Higher values allow more contrast.
    tile_size
        Size of grid tiles for histogram equalization. Image is divided into
        tile_size x tile_size regions.

    Returns
    -------
    np.ndarray
        Enhanced luminance as float64 array with values in [0, 1].
    """
    # Convert to uint8 for CLAHE processing
    luminance_uint8 = (luminance * 255).astype(np.uint8)

    # Create and apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced_uint8 = clahe.apply(luminance_uint8)

    # Convert back to float64 [0, 1]
    return enhanced_uint8.astype(np.float64) / 255.0


def compute_tonal_target(
    image: np.ndarray, config: LuminanceConfig | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tonal target from image luminance.

    Extracts luminance, applies CLAHE enhancement, and computes inverted
    tonal target where dark areas produce high density values.

    Parameters
    ----------
    image
        Input image as BGR uint8 (H, W, 3) or grayscale uint8 (H, W).
    config
        Configuration for luminance processing. If None, uses defaults.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (luminance, clahe_luminance, tonal_target), all as
        float64 arrays with values in [0, 1]. The tonal_target is
        inverted so dark areas have high values.
    """
    if config is None:
        config = LuminanceConfig()

    # Extract luminance
    luminance = extract_luminance(image)

    # Apply CLAHE enhancement
    clahe_luminance = apply_clahe(luminance, config.clip_limit, config.tile_size)

    # Compute inverted tonal target (dark areas → high density)
    tonal_target = 1.0 - clahe_luminance

    return luminance, clahe_luminance, tonal_target
