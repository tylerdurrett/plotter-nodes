"""Complexity map computation for measuring local image complexity."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from portrait_map_lab.models import ComplexityConfig, ComplexityResult

__all__ = [
    "compute_gradient_energy",
    "compute_laplacian_energy",
    "compute_multiscale_gradient_energy",
    "normalize_map",
    "apply_mask",
    "compute_complexity_map",
]


def compute_gradient_energy(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Compute gradient energy using Sobel gradients.

    Measures local image complexity based on gradient magnitude,
    smoothed with a Gaussian filter to reduce noise.

    Parameters
    ----------
    gray
        Grayscale input image as float64 array with values in [0, 1].
    sigma
        Standard deviation for Gaussian smoothing of the energy map.

    Returns
    -------
    np.ndarray
        Raw gradient energy map as float64 array (not normalized).
    """
    # Compute Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    magnitude = np.sqrt(gx * gx + gy * gy)

    # Smooth with Gaussian if sigma > 0
    if sigma > 0:
        magnitude = gaussian_filter(magnitude, sigma=sigma)

    return magnitude


def compute_laplacian_energy(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Compute Laplacian energy for detecting fine details.

    Uses the Laplacian operator to detect edges and texture,
    taking the absolute value and smoothing the result.

    Parameters
    ----------
    gray
        Grayscale input image as float64 array with values in [0, 1].
    sigma
        Standard deviation for Gaussian smoothing of the energy map.

    Returns
    -------
    np.ndarray
        Raw Laplacian energy map as float64 array (not normalized).
    """
    # Compute Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Take absolute value for energy
    energy = np.abs(laplacian)

    # Smooth with Gaussian if sigma > 0
    if sigma > 0:
        energy = gaussian_filter(energy, sigma=sigma)

    return energy


def compute_multiscale_gradient_energy(
    gray: np.ndarray, scales: list[float], weights: list[float]
) -> np.ndarray:
    """Compute weighted sum of gradient energies at multiple scales.

    Captures both fine and coarse details by computing gradient energy
    at different scales and combining them with specified weights.

    Parameters
    ----------
    gray
        Grayscale input image as float64 array with values in [0, 1].
    scales
        List of sigma values for computing gradient energy at different scales.
    weights
        List of weights for combining the multi-scale energies.

    Returns
    -------
    np.ndarray
        Raw multiscale gradient energy map as float64 array (not normalized).

    Raises
    ------
    ValueError
        If scales and weights have different lengths.
    """
    if len(scales) != len(weights):
        raise ValueError(
            f"scales and weights must have same length, got {len(scales)} and {len(weights)}"
        )

    # Initialize result array
    result = np.zeros_like(gray, dtype=np.float64)

    # Compute weighted sum of energies at each scale
    for scale_sigma, weight in zip(scales, weights):
        energy = compute_gradient_energy(gray, sigma=scale_sigma)
        result += weight * energy

    return result


def normalize_map(raw: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Normalize a map to [0, 1] using percentile-based scaling.

    Uses the specified percentile as the normalization ceiling,
    which provides robustness against outliers.

    Parameters
    ----------
    raw
        Raw input array to normalize.
    percentile
        Percentile value to use as the normalization ceiling.
        Use 100.0 for max normalization.

    Returns
    -------
    np.ndarray
        Normalized array with values in [0, 1] as float64.
    """
    # Handle empty or all-zero input
    if raw.size == 0:
        return raw

    # Compute the normalization ceiling
    ceiling = np.percentile(raw, percentile)

    # Handle case where ceiling is zero or negative
    if ceiling <= 0:
        return np.zeros_like(raw, dtype=np.float64)

    # Normalize and clip to [0, 1]
    normalized = raw / ceiling
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(np.float64)


def apply_mask(map_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary mask to a map array.

    Zeros out regions where the mask is 0, preserves values where
    the mask is non-zero. Handles both uint8 (0/255) and float (0/1) masks.

    Parameters
    ----------
    map_array
        Map array to mask, typically normalized to [0, 1].
    mask
        Binary mask array, either uint8 with values 0/255 or float with values 0.0/1.0.

    Returns
    -------
    np.ndarray
        Masked array with same dtype as input map_array.
    """
    # Convert mask to float64 binary (0 or 1)
    if mask.dtype == np.uint8:
        # Handle uint8 mask (0 or 255)
        mask_binary = (mask > 0).astype(np.float64)
    else:
        # Handle float mask (0.0 or 1.0)
        mask_binary = (mask > 0.5).astype(np.float64)

    # Apply mask element-wise
    result = map_array * mask_binary

    return result


def compute_complexity_map(
    image: np.ndarray, config: ComplexityConfig | None = None, mask: np.ndarray | None = None
) -> ComplexityResult:
    """Compute complexity map for an image.

    Main entry point for complexity computation. Converts the image to
    grayscale, computes the specified complexity metric, normalizes the
    result, and optionally applies a mask.

    Parameters
    ----------
    image
        Input image, either BGR uint8 or grayscale float64.
    config
        Configuration for complexity computation. If None, uses defaults.
    mask
        Optional binary mask to apply to the complexity map.

    Returns
    -------
    ComplexityResult
        Result containing raw and normalized complexity maps.

    Raises
    ------
    ValueError
        If an unknown metric is specified in the config.
    """
    # Use default config if not provided
    if config is None:
        config = ComplexityConfig()

    # Convert to grayscale float64 [0, 1]
    if len(image.shape) == 3:
        # BGR to grayscale - cv2.cvtColor requires uint8
        if image.dtype != np.uint8:
            # Convert to uint8 for cvtColor
            if image.max() <= 1.0:
                # Assume [0, 1] range
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                # Assume [0, 255] range
                img_uint8 = image.astype(np.uint8)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float64) / 255.0
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = gray.astype(np.float64) / 255.0
    else:
        gray = image.copy()
        # Ensure float64 in [0, 1]
        if gray.dtype == np.uint8:
            gray = gray.astype(np.float64) / 255.0
        else:
            gray = gray.astype(np.float64)
            # Normalize to [0, 1] if needed
            if gray.max() > 1.0:
                gray = gray / 255.0
            # Clip to ensure [0, 1] range
            gray = np.clip(gray, 0.0, 1.0)

    # Dispatch to appropriate metric function
    if config.metric == "gradient":
        raw_complexity = compute_gradient_energy(gray, sigma=config.sigma)
    elif config.metric == "laplacian":
        raw_complexity = compute_laplacian_energy(gray, sigma=config.sigma)
    elif config.metric == "multiscale_gradient":
        raw_complexity = compute_multiscale_gradient_energy(
            gray, scales=config.scales, weights=config.scale_weights
        )
    else:
        raise ValueError(f"Unknown complexity metric: {config.metric}")

    # Normalize to [0, 1]
    complexity = normalize_map(raw_complexity, percentile=config.normalize_percentile)

    # Apply mask if provided
    if mask is not None:
        complexity = apply_mask(complexity, mask)
        # Also mask the raw complexity for consistency
        raw_complexity = apply_mask(raw_complexity, mask)

    return ComplexityResult(
        raw_complexity=raw_complexity, complexity=complexity, metric=config.metric
    )
