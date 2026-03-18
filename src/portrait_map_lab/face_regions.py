"""Face region definitions and polygon extraction."""

from __future__ import annotations

import numpy as np

from portrait_map_lab.models import LandmarkResult, RegionDefinition, _default_regions

DEFAULT_REGIONS: list[RegionDefinition] = _default_regions()


def get_region_polygons(
    landmarks: LandmarkResult, regions: list[RegionDefinition]
) -> dict[str, np.ndarray]:
    """Extract pixel coordinate polygons for each facial region.

    Parameters
    ----------
    landmarks
        Face landmark detection result.
    regions
        List of region definitions specifying landmark indices.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from region name to Nx2 polygon array of pixel coordinates.
    """
    polygons = {}
    for region in regions:
        polygon = landmarks.landmarks[region.landmark_indices]
        polygons[region.name] = polygon.copy()
    return polygons
