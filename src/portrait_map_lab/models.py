"""Core dataclasses for the portrait map lab pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ComposeConfig",
    "ComposedResult",
    "ContourConfig",
    "ContourResult",
    "DensityResult",
    "ETFConfig",
    "ETFResult",
    "ExportManifest",
    "ExportMapEntry",
    "FlowConfig",
    "FlowResult",
    "LandmarkResult",
    "LICConfig",
    "LuminanceConfig",
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


@dataclass(slots=True)
class LuminanceConfig:
    """Configuration for luminance extraction and CLAHE processing."""

    clip_limit: float = 2.0
    tile_size: int = 8


@dataclass(slots=True)
class ComposeConfig:
    """Configuration for density map composition."""

    luminance: LuminanceConfig = field(default_factory=LuminanceConfig)
    feature_weight: float = 0.6
    contour_weight: float = 0.4
    tonal_blend_mode: str = "multiply"
    tonal_weight: float = 1.0
    importance_weight: float = 1.0
    gamma: float = 1.0


@dataclass(slots=True)
class ETFConfig:
    """Configuration for Edge Tangent Field computation."""

    blur_sigma: float = 1.5
    structure_sigma: float = 5.0
    refine_sigma: float = 3.0
    refine_iterations: int = 2
    sobel_ksize: int = 3


@dataclass(slots=True)
class FlowConfig:
    """Configuration for flow field computation."""

    etf: ETFConfig = field(default_factory=ETFConfig)
    contour_smooth_sigma: float = 1.0
    blend_mode: str = "coherence"
    coherence_power: float = 2.0
    fallback_threshold: float = 0.1


@dataclass(slots=True)
class LICConfig:
    """Configuration for Line Integral Convolution visualization."""

    length: int = 30
    step: float = 1.0
    seed: int = 42
    use_bilinear: bool = True


@dataclass(frozen=True, slots=True)
class DensityResult:
    """Result of density target computation pipeline."""

    luminance: np.ndarray
    clahe_luminance: np.ndarray
    tonal_target: np.ndarray
    importance: np.ndarray
    density_target: np.ndarray


@dataclass(frozen=True, slots=True)
class ETFResult:
    """Result of Edge Tangent Field computation."""

    tangent_x: np.ndarray
    tangent_y: np.ndarray
    coherence: np.ndarray
    gradient_magnitude: np.ndarray


@dataclass(frozen=True, slots=True)
class FlowResult:
    """Result of flow field computation pipeline."""

    etf: ETFResult
    contour_flow_x: np.ndarray
    contour_flow_y: np.ndarray
    blend_weight: np.ndarray
    flow_x: np.ndarray
    flow_y: np.ndarray


@dataclass(frozen=True, slots=True)
class ComposedResult:
    """Complete result of all pipelines combined."""

    feature_result: PipelineResult
    contour_result: ContourResult
    density_result: DensityResult
    flow_result: FlowResult
    lic_image: np.ndarray


@dataclass(frozen=True, slots=True)
class ExportMapEntry:
    """Metadata for one exported binary map."""

    filename: str
    key: str
    dtype: str
    shape: tuple[int, int]
    value_range: tuple[float, float]
    description: str


@dataclass(frozen=True, slots=True)
class ExportManifest:
    """Manifest describing a complete export bundle."""

    version: int
    source_image: str
    width: int
    height: int
    created_at: str
    maps: tuple[ExportMapEntry, ...]
