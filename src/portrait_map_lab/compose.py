"""Map composition utilities for density target construction."""

from __future__ import annotations

import numpy as np

__all__ = ["compose_maps", "build_density_target"]


def compose_maps(map_a: np.ndarray, map_b: np.ndarray, mode: str = "multiply") -> np.ndarray:
    """Compose two maps using various blend modes.

    Applies different blending operations to combine two maps, useful for
    creating density targets from multiple input sources.

    Parameters
    ----------
    map_a
        First map as float array with values in [0, 1].
    map_b
        Second map as float array with values in [0, 1]. Must have same shape as map_a.
    mode
        Blend mode to use. Options are:
        - "multiply": Element-wise product (default)
        - "screen": Inverse of multiply, 1 - (1-a)*(1-b)
        - "max": Element-wise maximum
        - "weighted": Weighted sum normalized to [0, 1]

    Returns
    -------
    np.ndarray
        Composed map as float64 array with values clipped to [0, 1].

    Raises
    ------
    ValueError
        If mode is not recognized or if map shapes don't match.
    """
    # Validate shapes match
    if map_a.shape != map_b.shape:
        raise ValueError(
            f"Map shapes must match. Got map_a.shape={map_a.shape}, map_b.shape={map_b.shape}"
        )

    # Convert to float64 for computation
    a = map_a.astype(np.float64)
    b = map_b.astype(np.float64)

    # Apply blend mode
    if mode == "multiply":
        result = a * b
    elif mode == "screen":
        result = 1.0 - (1.0 - a) * (1.0 - b)
    elif mode == "max":
        result = np.maximum(a, b)
    elif mode == "weighted":
        # Weighted sum with equal weights, normalized
        # This matches the normalization approach in combine.py
        result = (a + b) / 2.0
    else:
        raise ValueError(
            f"Unknown blend mode: {mode}. "
            f"Valid modes are: multiply, screen, max, weighted"
        )

    # Ensure output is in [0, 1]
    return np.clip(result, 0.0, 1.0)


def build_density_target(
    tonal_target: np.ndarray,
    importance: np.ndarray,
    mode: str = "multiply",
    gamma: float = 1.0
) -> np.ndarray:
    """Build density target by composing tonal and importance maps.

    Combines a tonal target (from luminance) with an importance map
    (from features/contours) using the specified blend mode, then applies
    gamma correction for fine-tuning the density distribution.

    Parameters
    ----------
    tonal_target
        Tonal density target from luminance, with values in [0, 1].
        Dark image areas should have high values.
    importance
        Structural importance map with values in [0, 1].
        Important features should have high values.
    mode
        Blend mode for composition. See compose_maps for options.
    gamma
        Gamma correction exponent. Values < 1.0 brighten (increase density),
        values > 1.0 darken (decrease density), 1.0 is identity.

    Returns
    -------
    np.ndarray
        Density target as float64 array with values in [0, 1].
    """
    # Compose the maps using the specified mode
    composed = compose_maps(tonal_target, importance, mode=mode)

    # Apply gamma correction
    if gamma != 1.0:
        # Avoid numerical issues with zero values when gamma < 1
        composed = np.power(composed, gamma)

    # Ensure output is in [0, 1] (gamma might introduce small numerical errors)
    return np.clip(composed, 0.0, 1.0)
