"""Core dataclasses for the portrait map lab pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ContourConfig",
    "ContourResult",
    "LandmarkResult",
    "PipelineConfig",
    "PipelineResult",
    "RegionDefinition",
    "RemapConfig",
]


@dataclass(frozen=True, slots=True)
class LandmarkResult:
    """Result of face landmark detection."""

    landmarks: np.ndarray  # Nx2 or Nx3 pixel coordinates
    image_shape: tuple[int, int]  # (height, width)
    confidence: float


@dataclass(frozen=True, slots=True)
class RegionDefinition:
    """Definition of a facial region by landmark indices."""

    name: str
    landmark_indices: list[int]


@dataclass(slots=True)
class RemapConfig:
    """Configuration for distance-to-influence remapping."""

    curve: str = "gaussian"
    radius: float = 150.0
    sigma: float = 80.0
    tau: float = 60.0
    clamp_distance: float = 300.0


@dataclass(slots=True)
class ContourConfig:
    """Configuration for face contour distance pipeline."""

    remap: RemapConfig = field(default_factory=RemapConfig)
    direction: str = "inward"
    band_width: float | None = None
    contour_thickness: int = 1
    output_dir: str = "output"


def _default_regions() -> list[RegionDefinition]:
    """Return default eye and mouth region definitions with MediaPipe Face Mesh indices."""
    return [
        RegionDefinition(
            name="left_eye",
            landmark_indices=[
                263,
                249,
                390,
                373,
                374,
                380,
                381,
                382,
                362,
                398,
                384,
                385,
                386,
                387,
                388,
                466,
            ],
        ),
        RegionDefinition(
            name="right_eye",
            landmark_indices=[
                33,
                7,
                163,
                144,
                145,
                153,
                154,
                155,
                133,
                173,
                157,
                158,
                159,
                160,
                161,
                246,
            ],
        ),
        RegionDefinition(
            name="mouth",
            landmark_indices=[
                61,
                146,
                91,
                181,
                84,
                17,
                314,
                405,
                321,
                375,
                291,
                409,
                270,
                269,
                267,
                0,
                37,
                39,
                40,
                185,
            ],
        ),
    ]


@dataclass(slots=True)
class PipelineConfig:
    """Configuration for the full processing pipeline."""

    regions: list[RegionDefinition] = field(default_factory=_default_regions)
    remap: RemapConfig = field(default_factory=RemapConfig)
    weights: dict[str, float] = field(default_factory=lambda: {"eyes": 0.6, "mouth": 0.4})
    output_dir: str = "output"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Complete result of the feature distance pipeline."""

    landmarks: LandmarkResult
    masks: dict[str, np.ndarray]
    distance_fields: dict[str, np.ndarray]
    influence_maps: dict[str, np.ndarray]
    combined: np.ndarray


@dataclass(frozen=True, slots=True)
class ContourResult:
    """Result of face contour distance pipeline."""

    landmarks: LandmarkResult
    contour_polygon: np.ndarray
    contour_mask: np.ndarray
    filled_mask: np.ndarray
    signed_distance: np.ndarray
    directional_distance: np.ndarray
    influence_map: np.ndarray
