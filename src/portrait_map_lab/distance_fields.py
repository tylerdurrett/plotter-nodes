"""Distance field computation from binary masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_distance_field(mask: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance field from a binary mask.

    Calculates the distance from each pixel to the nearest non-masked
    region. Pixels inside the mask have distance 0.0.

    Parameters
    ----------
    mask
        Binary mask as uint8 array (0 or 255 values).

    Returns
    -------
    np.ndarray
        Distance field as float64 array with pixel-unit distances.
    """
    # Invert mask: distance from non-masked regions
    inverted = ~(mask > 0)

    # Apply Euclidean distance transform
    distance_field = distance_transform_edt(inverted)

    return distance_field.astype(np.float64)
