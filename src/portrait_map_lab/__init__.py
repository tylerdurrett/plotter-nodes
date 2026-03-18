"""Portrait Map Lab — feature distance mapping pipeline for pen plotter artwork."""

from __future__ import annotations

from portrait_map_lab.combine import combine_maps
from portrait_map_lab.compose import build_density_target, compose_maps
from portrait_map_lab.distance_fields import compute_distance_field
from portrait_map_lab.etf import compute_etf, compute_structure_tensor, extract_tangent_field, refine_tangent_field
from portrait_map_lab.face_contour import (
    FACE_OVAL_INDICES,
    compute_signed_distance,
    get_face_oval_polygon,
    prepare_directional_distance,
    rasterize_contour_mask,
    rasterize_filled_mask,
)
from portrait_map_lab.face_regions import DEFAULT_REGIONS, get_region_polygons
from portrait_map_lab.flow_fields import (
    align_tangent_field,
    blend_flow_fields,
    compute_blend_weight,
    compute_contour_flow,
)
from portrait_map_lab.landmarks import detect_landmarks
from portrait_map_lab.lic import compute_lic
from portrait_map_lab.luminance import apply_clahe, compute_tonal_target, extract_luminance
from portrait_map_lab.masks import build_region_masks
from portrait_map_lab.models import (
    ComposeConfig,
    ComposedResult,
    ContourConfig,
    ContourResult,
    DensityResult,
    ETFConfig,
    ETFResult,
    FlowConfig,
    FlowResult,
    LICConfig,
    LandmarkResult,
    LuminanceConfig,
    PipelineConfig,
    PipelineResult,
    RegionDefinition,
    RemapConfig,
)
from portrait_map_lab.pipelines import (
    run_all_pipelines,
    run_contour_pipeline,
    run_density_pipeline,
    run_feature_distance_pipeline,
    run_flow_pipeline,
    save_all_outputs,
    save_contour_outputs,
    save_density_outputs,
    save_flow_outputs,
    save_pipeline_outputs,
)
from portrait_map_lab.remap import remap_influence
from portrait_map_lab.storage import ensure_output_dir, load_image, save_array, save_image
from portrait_map_lab.viz import (
    colorize_map,
    draw_contour,
    draw_landmarks,
    make_contact_sheet,
    overlay_lic,
    visualize_flow_field,
)

__version__ = "0.1.0"

__all__ = [
    # Main pipeline functions
    "run_all_pipelines",
    "run_feature_distance_pipeline",
    "run_contour_pipeline",
    "run_density_pipeline",
    "run_flow_pipeline",
    "save_all_outputs",
    "save_pipeline_outputs",
    "save_contour_outputs",
    "save_density_outputs",
    "save_flow_outputs",
    # Core data models
    "ComposeConfig",
    "ComposedResult",
    "ContourConfig",
    "ContourResult",
    "DensityResult",
    "ETFConfig",
    "ETFResult",
    "FlowConfig",
    "FlowResult",
    "LICConfig",
    "LandmarkResult",
    "LuminanceConfig",
    "PipelineConfig",
    "PipelineResult",
    "RegionDefinition",
    "RemapConfig",
    # Face contour functions
    "FACE_OVAL_INDICES",
    "get_face_oval_polygon",
    "rasterize_contour_mask",
    "rasterize_filled_mask",
    "compute_signed_distance",
    "prepare_directional_distance",
    # Luminance and composition
    "extract_luminance",
    "apply_clahe",
    "compute_tonal_target",
    "compose_maps",
    "build_density_target",
    # ETF functions
    "compute_structure_tensor",
    "extract_tangent_field",
    "refine_tangent_field",
    "compute_etf",
    # Flow field functions
    "compute_contour_flow",
    "align_tangent_field",
    "compute_blend_weight",
    "blend_flow_fields",
    # LIC visualization
    "compute_lic",
    # Processing functions
    "detect_landmarks",
    "get_region_polygons",
    "build_region_masks",
    "compute_distance_field",
    "remap_influence",
    "combine_maps",
    # Storage utilities
    "load_image",
    "save_image",
    "save_array",
    "ensure_output_dir",
    # Visualization utilities
    "draw_landmarks",
    "draw_contour",
    "colorize_map",
    "make_contact_sheet",
    "visualize_flow_field",
    "overlay_lic",
    # Constants
    "DEFAULT_REGIONS",
]
