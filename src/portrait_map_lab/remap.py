"""Distance-to-influence remapping with configurable curves."""

from __future__ import annotations

import numpy as np

from portrait_map_lab.models import RemapConfig


def remap_influence(distance_field: np.ndarray, config: RemapConfig) -> np.ndarray:
    """Remap distance field to influence map using specified curve.

    Applies a falloff curve to convert pixel distances into influence values,
    where influence is maximal (1.0) at the mask boundary and decreases
    with distance.

    Parameters
    ----------
    distance_field
        Distance field as float array with pixel-unit distances.
    config
        Remapping configuration specifying curve type and parameters.

    Returns
    -------
    np.ndarray
        Influence map as float64 array with values in [0.0, 1.0].

    Raises
    ------
    ValueError
        If config.curve is not a recognized curve type.
    """
    # Clamp distances to maximum configured distance
    d = np.minimum(distance_field, config.clamp_distance)

    # Apply curve based on type
    if config.curve == "linear":
        # Linear falloff: max(0, 1 - d/radius)
        influence = np.maximum(0.0, 1.0 - d / config.radius)

    elif config.curve == "gaussian":
        # Gaussian falloff: exp(-(d²/(2σ²)))
        influence = np.exp(-(d**2) / (2 * config.sigma**2))

    elif config.curve == "exponential":
        # Exponential falloff: exp(-d/τ)
        influence = np.exp(-d / config.tau)

    else:
        raise ValueError(
            f"Unknown curve type: {config.curve}. Supported: 'linear', 'gaussian', 'exponential'"
        )

    # Ensure output is float64 and properly bounded
    influence = influence.astype(np.float64)
    return np.clip(influence, 0.0, 1.0)
