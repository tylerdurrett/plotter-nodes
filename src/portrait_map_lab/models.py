"""Core dataclasses for the portrait map lab pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ComposeConfig",
    "ComposedResult",
    "ComplexityConfig",
    "ComplexityResult",
    "ContourConfig",
    "ContourResult",
    "DensityResult",
    "ETFConfig",
    "ETFResult",
    "ExportManifest",
    "ExportMapEntry",
    "FlowConfig",
    "FlowResult",
    "FlowSpeedConfig",
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
class ComplexityConfig:
    """Configuration for complexity map computation."""

    metric: str = "gradient"  # gradient, laplacian, or multiscale_gradient
    sigma: float = 3.0  # Gaussian smoothing sigma for single-scale metrics
    scales: list[float] = field(default_factory=lambda: [1.0, 3.0, 8.0])  # multiscale sigmas
    scale_weights: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])  # weights
    normalize_percentile: float = 99.0  # percentile for normalization (100.0 = max)
    output_dir: str = "output"


@dataclass(frozen=True, slots=True)
class ComplexityResult:
    """Result of complexity map computation."""

    raw_complexity: np.ndarray  # unnormalized metric output
    complexity: np.ndarray  # normalized [0, 1] complexity map
    metric: str  # which metric was used


@dataclass(slots=True)
class FlowSpeedConfig:
    """Configuration for flow speed derivation from complexity."""

    speed_min: float = 0.3  # speed in most complex areas
    speed_max: float = 1.0  # speed in smooth areas


@dataclass(slots=True)
class ContourConfig:
    """Configuration for face contour distance pipeline."""

    contour_method: str = "landmarks"
    remap: RemapConfig = field(default_factory=RemapConfig)
    direction: str = "inward"
    band_width: float | None = None
    contour_thickness: int = 1
    epsilon_factor: float = 0.005
    smooth_contour: bool = True
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

    landmarks: LandmarkResult | None
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
    flow_speed: np.ndarray | None = None  # particle speed scalar derived from complexity


@dataclass(frozen=True, slots=True)
class ComposedResult:
    """Complete result of all pipelines combined."""

    feature_result: PipelineResult
    contour_result: ContourResult
    density_result: DensityResult
    flow_result: FlowResult
    lic_image: np.ndarray
    complexity_result: ComplexityResult | None = None


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
