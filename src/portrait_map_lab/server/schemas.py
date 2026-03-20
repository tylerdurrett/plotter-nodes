"""Pydantic request/response schemas for the Map Generation API.

Config schemas mirror the dataclass hierarchy in ``models.py`` with all fields
optional so clients only need to send overrides.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from portrait_map_lab.export import _MAP_DEFINITIONS

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
    "MAP_KEY_INFOS",
    "MapKeyInfo",
    "RemapConfigSchema",
    "VALID_MAP_KEYS",
]

# Canonical set of valid map keys, derived from export definitions.
VALID_MAP_KEYS: frozenset[str] = frozenset(key for key, *_ in _MAP_DEFINITIONS)


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
    persist: str | None = None
    config: GenerateConfigSchema | None = None

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


class GenerateResponse(BaseModel):
    """Response body for ``POST /api/generate``."""

    session_id: str
    manifest: dict
    base_url: str


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
