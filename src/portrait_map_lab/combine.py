"""Weighted combination of multiple influence maps."""

from __future__ import annotations

import numpy as np


def combine_maps(maps: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Combine multiple influence maps using weighted sum.

    Takes a dictionary of influence maps and corresponding weights,
    computes a weighted sum, and normalizes the result to [0.0, 1.0].

    Parameters
    ----------
    maps
        Dictionary mapping region names to influence maps.
        All maps must have the same shape.
    weights
        Dictionary mapping region names to weight values.
        Keys must match the keys in maps.

    Returns
    -------
    np.ndarray
        Combined influence map as float64 array with values in [0.0, 1.0].

    Raises
    ------
    ValueError
        If weight keys don't match map keys, or if maps have different shapes.
    """
    # Validate that keys match
    if set(weights.keys()) != set(maps.keys()):
        missing_in_weights = set(maps.keys()) - set(weights.keys())
        missing_in_maps = set(weights.keys()) - set(maps.keys())
        msg = "Weight keys must match map keys."
        if missing_in_weights:
            msg += f" Missing in weights: {missing_in_weights}."
        if missing_in_maps:
            msg += f" Missing in maps: {missing_in_maps}."
        raise ValueError(msg)

    # Handle empty case
    if not maps:
        raise ValueError("Cannot combine empty maps dictionary")

    # Get reference shape from first map
    shapes = [m.shape for m in maps.values()]
    reference_shape = shapes[0]

    # Validate all maps have same shape
    if not all(shape == reference_shape for shape in shapes):
        raise ValueError(f"All maps must have the same shape. Got shapes: {set(shapes)}")

    # Initialize result array
    result = np.zeros(reference_shape, dtype=np.float64)

    # Compute weighted sum
    total_weight = 0.0
    for name, influence_map in maps.items():
        weight = weights[name]
        if weight != 0:  # Skip zero weights for efficiency
            result += weight * influence_map.astype(np.float64)
            total_weight += abs(weight)  # Use absolute value for normalization

    # Normalize by total weight if non-zero
    if total_weight > 0:
        result /= total_weight

    # Ensure output is in [0.0, 1.0]
    return np.clip(result, 0.0, 1.0)
