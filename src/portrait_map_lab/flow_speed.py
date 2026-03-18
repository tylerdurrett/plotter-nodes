"""Flow speed computation from complexity maps."""

from __future__ import annotations

import numpy as np

from portrait_map_lab.models import FlowSpeedConfig

__all__ = [
    "compute_flow_speed",
]


def compute_flow_speed(
    complexity: np.ndarray, config: FlowSpeedConfig | None = None
) -> np.ndarray:
    """Compute flow speed from complexity map.

    Derives particle flow speed from local complexity using linear inverse
    mapping, where higher complexity produces slower speeds.

    Parameters
    ----------
    complexity
        Normalized complexity map as float64 array with values in [0, 1].
        Higher values indicate more complex/detailed regions.
    config
        Flow speed configuration. If None, uses default configuration
        with speed_min=0.3 and speed_max=1.0.

    Returns
    -------
    np.ndarray
        Flow speed map as float64 array with values in [speed_min, speed_max].
        Same shape as input complexity map.

    Notes
    -----
    The speed computation uses linear inverse mapping:
        speed = speed_max - complexity * (speed_max - speed_min)

    This ensures:
    - complexity=0 (smooth) -> speed_max (fast)
    - complexity=1 (detailed) -> speed_min (slow)
    - Linear interpolation for intermediate values
    """
    if config is None:
        config = FlowSpeedConfig()

    # Compute speed with linear inverse mapping
    # speed = speed_max - complexity * (speed_max - speed_min)
    speed_range = config.speed_max - config.speed_min
    speed = config.speed_max - complexity * speed_range

    # Clip to valid range (handles any numerical edge cases)
    speed = np.clip(speed, config.speed_min, config.speed_max)

    # Ensure output is float64
    return speed.astype(np.float64)
