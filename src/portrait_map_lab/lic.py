"""Line Integral Convolution (LIC) visualization for flow fields."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from portrait_map_lab.models import LICConfig

__all__ = ["compute_lic"]


def compute_lic(
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    config: LICConfig | None = None,
) -> np.ndarray:
    """Compute Line Integral Convolution visualization of a flow field.

    LIC creates a texture that visualizes flow patterns by averaging noise
    values along streamlines. The result highlights the directional structure
    of the flow field.

    Parameters
    ----------
    flow_x
        X-component of the flow field (unit vectors).
    flow_y
        Y-component of the flow field (unit vectors).
    config
        Configuration for LIC computation. If None, uses default configuration.

    Returns
    -------
    np.ndarray
        LIC texture as float64 array with values in [0, 1].
    """
    if config is None:
        config = LICConfig()

    h, w = flow_x.shape

    # Generate white noise with fixed seed for determinism
    rng = np.random.RandomState(config.seed)
    noise = rng.random((h, w))

    # Initialize coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float64)

    # Initialize accumulator for averaged noise values
    accumulator = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.float64)

    # Trace streamlines forward and backward
    for direction in [1, -1]:  # Forward then backward
        # Reset coordinates for this direction
        current_x = x_coords.copy()
        current_y = y_coords.copy()

        # Trace for specified number of steps
        for _ in range(config.length):
            # Sample noise at current positions
            if config.use_bilinear:
                # Use bilinear interpolation
                sampled = ndimage.map_coordinates(
                    noise,
                    [current_y, current_x],
                    order=1,  # Linear interpolation
                    mode='nearest',  # Handle boundaries
                    prefilter=False,
                )
            else:
                # Use nearest-neighbor sampling
                x_int = np.clip(np.round(current_x).astype(int), 0, w - 1)
                y_int = np.clip(np.round(current_y).astype(int), 0, h - 1)
                sampled = noise[y_int, x_int]

            # Accumulate sampled values
            accumulator += sampled
            count += 1

            # Sample flow at current positions
            if config.use_bilinear:
                flow_x_sampled = ndimage.map_coordinates(
                    flow_x,
                    [current_y, current_x],
                    order=1,
                    mode='nearest',
                    prefilter=False,
                )
                flow_y_sampled = ndimage.map_coordinates(
                    flow_y,
                    [current_y, current_x],
                    order=1,
                    mode='nearest',
                    prefilter=False,
                )
            else:
                x_int = np.clip(np.round(current_x).astype(int), 0, w - 1)
                y_int = np.clip(np.round(current_y).astype(int), 0, h - 1)
                flow_x_sampled = flow_x[y_int, x_int]
                flow_y_sampled = flow_y[y_int, x_int]

            # Advance positions along flow direction
            current_x += direction * config.step * flow_x_sampled
            current_y += direction * config.step * flow_y_sampled

            # Clip to image bounds
            current_x = np.clip(current_x, 0, w - 1)
            current_y = np.clip(current_y, 0, h - 1)

    # Average the accumulated values
    # Avoid division by zero (though count should never be 0)
    result = np.divide(accumulator, count, out=np.zeros_like(accumulator), where=count > 0)

    # Normalize to [0, 1] range
    min_val = np.min(result)
    max_val = np.max(result)
    if max_val > min_val:
        result = (result - min_val) / (max_val - min_val)
    else:
        # Uniform field - return mid-gray
        result = np.full_like(result, 0.5)

    return result.astype(np.float64)
