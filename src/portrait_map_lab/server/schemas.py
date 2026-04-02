"""Pydantic request/response schemas for the Map Generation API.

Config schemas mirror the dataclass hierarchy in ``models.py`` with all fields
optional so clients only need to send overrides.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from portrait_map_lab.export import _INTERMEDIATE_MAP_DEFINITIONS, _MAP_DEFINITIONS
from portrait_map_lab.models import (
    ComplexityConfig,
    ComposeConfig,
    ContourConfig,
    FlowConfig,
    FlowSpeedConfig,
    PipelineConfig,
)

__all__ = [
    "ComplexityConfigSchema",
    "ContourConfigSchema",
    "DensityConfigSchema",
    "ETFConfigSchema",
    "FeaturesConfigSchema",
    "FlowConfigSchema",
    "FlowSpeedConfigSchema",
    "GenerateConfigSchema",
    "GenerateRequest",
    "GenerateResponse",
    "LuminanceConfigSchema",
    "PreviewInfo",
    "SessionInfo",
    "MAP_KEY_INFOS",
    "MapKeyInfo",
    "RemapConfigSchema",
    "VALID_MAP_KEYS",
    "build_compose_config",
    "build_complexity_config",
    "build_contour_config",
    "build_flow_config",
    "build_flow_speed_config",
    "build_pipeline_config",
]

# Canonical set of valid map keys, derived from export definitions.
VALID_MAP_KEYS: frozenset[str] = frozenset(key for key, *_ in _MAP_DEFINITIONS)

# Canonical set of valid intermediate map keys (v2 manifests).
VALID_INTERMEDIATE_KEYS: frozenset[str] = frozenset(
    key for key, *_ in _INTERMEDIATE_MAP_DEFINITIONS
)


# ---------------------------------------------------------------------------
# Config sub-schemas (all fields optional — override-only semantics)
# ---------------------------------------------------------------------------


class RemapConfigSchema(BaseModel):
    """Override schema for :class:`~portrait_map_lab.models.RemapConfig`."""

    curve: str | None = None
    radius: float | None = None
    sigma: float | None = None
    tau: float | None = None
    clamp_distance: float | None = None


class LuminanceConfigSchema(BaseModel):
    """Override schema for :class:`~portrait_map_lab.models.LuminanceConfig`."""

    clip_limit: float | None = None
    tile_size: int | None = None


class ETFConfigSchema(BaseModel):
    """Override schema for :class:`~portrait_map_lab.models.ETFConfig`."""

    blur_sigma: float | None = None
    structure_sigma: float | None = None
    refine_sigma: float | None = None
    refine_iterations: int | None = None
    sobel_ksize: int | None = None


class FeaturesConfigSchema(BaseModel):
    """Override schema for :class:`~portrait_map_lab.models.PipelineConfig`.

    Omits ``regions`` and ``output_dir`` (internal to CLI).
    """

    weights: dict[str, float] | None = None
    remap: RemapConfigSchema | None = None


class ContourConfigSchema(BaseModel):
    """Override schema for :class:`~portrait_map_lab.models.ContourConfig`.

    Omits ``output_dir`` (internal to CLI).
    """

    contour_method: str | None = None
    remap: RemapConfigSchema | None = None
    direction: str | None = None
    band_width: float | None = None
    contour_thickness: int | None = None
    epsilon_factor: float | None = None
    smooth_contour: bool | None = None


class DensityConfigSchema(BaseModel):
    """Override schema for density composition (``ComposeConfig`` in models)."""

    luminance: LuminanceConfigSchema | None = None
    feature_weight: float | None = None
    contour_weight: float | None = None
    tonal_blend_mode: str | None = None
    tonal_weight: float | None = None
    importance_weight: float | None = None
    gamma: float | None = None


class ComplexityConfigSchema(BaseModel):
    """Override schema for :class:`~portrait_map_lab.models.ComplexityConfig`.

    Omits ``output_dir`` (internal to CLI).
    """

    metric: str | None = None
    sigma: float | None = None
    scales: list[float] | None = None
    scale_weights: list[float] | None = None
    normalize_percentile: float | None = None


class FlowConfigSchema(BaseModel):
    """Override schema for :class:`~portrait_map_lab.models.FlowConfig`."""

    etf: ETFConfigSchema | None = None
    contour_smooth_sigma: float | None = None
    blend_mode: str | None = None
    coherence_power: float | None = None
    fallback_threshold: float | None = None


class FlowSpeedConfigSchema(BaseModel):
    """Override schema for :class:`~portrait_map_lab.models.FlowSpeedConfig`."""

    speed_min: float | None = None
    speed_max: float | None = None


# ---------------------------------------------------------------------------
# Top-level config grouping
# ---------------------------------------------------------------------------


class GenerateConfigSchema(BaseModel):
    """Top-level configuration wrapper grouping all pipeline overrides."""

    features: FeaturesConfigSchema | None = None
    contour: ContourConfigSchema | None = None
    density: DensityConfigSchema | None = None
    complexity: ComplexityConfigSchema | None = None
    flow: FlowConfigSchema | None = None
    flow_speed: FlowSpeedConfigSchema | None = None


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request body for ``POST /api/generate``."""

    image_path: str | None = None
    maps: list[str] | None = None
    persist: str | None = Field(None, pattern=r"^[a-zA-Z0-9_-]+$")
    config: GenerateConfigSchema | None = None
    mode: str | None = Field(None, pattern=r"^(intermediates)$")

    @field_validator("maps")
    @classmethod
    def validate_map_keys(cls, v: list[str] | None) -> list[str] | None:
        """Reject unrecognised map keys."""
        if v is None:
            return v
        invalid = set(v) - VALID_MAP_KEYS
        if invalid:
            raise ValueError(
                f"Invalid map key(s): {sorted(invalid)}. "
                f"Valid keys: {sorted(VALID_MAP_KEYS)}"
            )
        return v


class PreviewInfo(BaseModel):
    """Metadata for a single preview PNG file."""

    category: str
    name: str
    url: str


class GenerateResponse(BaseModel):
    """Response body for ``POST /api/generate``."""

    session_id: str
    manifest: dict
    base_url: str
    previews: list[PreviewInfo] = []


class SessionInfo(BaseModel):
    """Metadata for a cached session returned by ``GET /api/sessions``."""

    session_id: str
    source_image: str
    created_at: str
    map_keys: list[str]
    persistent: bool = False
    previews: list[PreviewInfo] = []


class MapKeyInfo(BaseModel):
    """Metadata about a single available map key."""

    key: str
    value_range: list[float]
    description: str


# Pre-built list of all map key metadata, computed once at import time.
MAP_KEY_INFOS: list[MapKeyInfo] = [
    MapKeyInfo(
        key=key,
        value_range=list(value_range),
        description=description,
    )
    for key, _attr_path, value_range, description in _MAP_DEFINITIONS
]


# ---------------------------------------------------------------------------
# Config merge helpers — Pydantic schema → pipeline dataclass
# ---------------------------------------------------------------------------


def _merge_onto(schema: BaseModel, target: object) -> None:
    """Set non-None scalar fields from a Pydantic schema onto a dataclass."""
    for name, value in schema:
        if value is not None and not isinstance(value, BaseModel):
            setattr(target, name, value)


def _merge_sub(schema: BaseModel | None, target: object) -> object:
    """Merge optional sub-config overrides onto a default dataclass instance."""
    if schema is not None:
        _merge_onto(schema, target)
    return target


def build_pipeline_config(
    schema: FeaturesConfigSchema | None,
) -> PipelineConfig | None:
    """Build a ``PipelineConfig`` from optional overrides.

    Returns ``None`` when *schema* is ``None`` (use pipeline defaults).
    """
    if schema is None:
        return None
    cfg = PipelineConfig()
    _merge_onto(schema, cfg)
    if schema.remap is not None:
        _merge_sub(schema.remap, cfg.remap)
    return cfg


def build_contour_config(
    schema: ContourConfigSchema | None,
) -> ContourConfig | None:
    """Build a ``ContourConfig`` from optional overrides."""
    if schema is None:
        return None
    cfg = ContourConfig()
    _merge_onto(schema, cfg)
    if schema.remap is not None:
        _merge_sub(schema.remap, cfg.remap)
    return cfg


def build_compose_config(
    schema: DensityConfigSchema | None,
) -> ComposeConfig | None:
    """Build a ``ComposeConfig`` from optional overrides."""
    if schema is None:
        return None
    cfg = ComposeConfig()
    _merge_onto(schema, cfg)
    if schema.luminance is not None:
        _merge_sub(schema.luminance, cfg.luminance)
    return cfg


def build_complexity_config(
    schema: ComplexityConfigSchema | None,
) -> ComplexityConfig | None:
    """Build a ``ComplexityConfig`` from optional overrides."""
    if schema is None:
        return None
    cfg = ComplexityConfig()
    _merge_onto(schema, cfg)
    return cfg


def build_flow_config(
    schema: FlowConfigSchema | None,
) -> FlowConfig | None:
    """Build a ``FlowConfig`` from optional overrides."""
    if schema is None:
        return None
    cfg = FlowConfig()
    _merge_onto(schema, cfg)
    if schema.etf is not None:
        _merge_sub(schema.etf, cfg.etf)
    return cfg


def build_flow_speed_config(
    schema: FlowSpeedConfigSchema | None,
) -> FlowSpeedConfig | None:
    """Build a ``FlowSpeedConfig`` from optional overrides."""
    if schema is None:
        return None
    cfg = FlowSpeedConfig()
    _merge_onto(schema, cfg)
    return cfg
