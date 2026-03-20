"""Portrait Map Lab — feature distance mapping pipeline for pen plotter artwork."""

from __future__ import annotations

from portrait_map_lab.combine import combine_maps
from portrait_map_lab.complexity_map import (
    apply_mask,
    compute_complexity_map,
    compute_gradient_energy,
    compute_laplacian_energy,
    compute_multiscale_gradient_energy,
    normalize_map,
)
from portrait_map_lab.compose import build_density_target, compose_maps
from portrait_map_lab.distance_fields import compute_distance_field
from portrait_map_lab.etf import (
    compute_etf,
    compute_structure_tensor,
    extract_tangent_field,
    refine_tangent_field,
)
from portrait_map_lab.export import (
    ExportBundle,
    build_export_bundle,
    export_composed_result,
    save_export_bundle,
)
from portrait_map_lab.face_contour import (
    average_signed_distances,
    compute_sdf_from_polygon,
    compute_signed_distance,
    derive_contour_from_sdf,
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
from portrait_map_lab.flow_speed import compute_flow_speed
from portrait_map_lab.landmarks import detect_landmarks
from portrait_map_lab.lic import compute_lic
from portrait_map_lab.luminance import apply_clahe, compute_tonal_target, extract_luminance
from portrait_map_lab.masks import build_region_masks
from portrait_map_lab.models import (
    ComplexityConfig,
    ComplexityResult,
    ComposeConfig,
    ComposedResult,
    ContourConfig,
    ContourResult,
    DensityResult,
    ETFConfig,
    ETFResult,
    ExportManifest,
    ExportMapEntry,
    FlowConfig,
    FlowResult,
    FlowSpeedConfig,
    LandmarkResult,
    LICConfig,
    LuminanceConfig,
    PipelineConfig,
    PipelineResult,
    RegionDefinition,
    RemapConfig,
)
from portrait_map_lab.pipelines import (
    run_all_pipelines,
    run_complexity_pipeline,
    run_contour_pipeline,
    run_contour_pipeline_with_landmarks,
    run_density_pipeline,
    run_feature_distance_pipeline,
    run_feature_pipeline_with_landmarks,
    run_flow_pipeline,
    save_all_outputs,
    save_complexity_outputs,
    save_contour_outputs,
    save_density_outputs,
    save_flow_outputs,
    save_pipeline_outputs,
)
from portrait_map_lab.remap import remap_influence
from portrait_map_lab.segmentation import (
    SEGMENTATION_ACCESSORIES,
    SEGMENTATION_BACKGROUND,
    SEGMENTATION_BODY_SKIN,
    SEGMENTATION_CLOTHES,
    SEGMENTATION_FACE_SKIN,
    SEGMENTATION_HAIR,
    extract_segmentation_polygon,
    segment_image,
)
from portrait_map_lab.storage import (
    ensure_output_dir,
    load_image,
    save_array,
    save_binary_map,
    save_image,
)
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
    "run_feature_pipeline_with_landmarks",
    "run_contour_pipeline",
    "run_contour_pipeline_with_landmarks",
    "run_density_pipeline",
    "run_complexity_pipeline",
    "run_flow_pipeline",
    "save_all_outputs",
    "save_pipeline_outputs",
    "save_contour_outputs",
    "save_density_outputs",
    "save_complexity_outputs",
    "save_flow_outputs",
    # Export functions
    "build_export_bundle",
    "save_export_bundle",
    "export_composed_result",
    "ExportBundle",
    # Core data models
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
    "LICConfig",
    "LandmarkResult",
    "LuminanceConfig",
    "PipelineConfig",
    "PipelineResult",
    "RegionDefinition",
    "RemapConfig",
    # Face contour functions
    "get_face_oval_polygon",
    "rasterize_contour_mask",
    "rasterize_filled_mask",
    "compute_signed_distance",
    "prepare_directional_distance",
    "compute_sdf_from_polygon",
    "average_signed_distances",
    "derive_contour_from_sdf",
    # Luminance and composition
    "extract_luminance",
    "apply_clahe",
    "compute_tonal_target",
    "compose_maps",
    "build_density_target",
    # Complexity map functions
    "compute_complexity_map",
    "compute_gradient_energy",
    "compute_laplacian_energy",
    "compute_multiscale_gradient_energy",
    "normalize_map",
    "apply_mask",
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
    # Flow speed
    "compute_flow_speed",
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
    "save_binary_map",
    "ensure_output_dir",
    # Visualization utilities
    "draw_landmarks",
    "draw_contour",
    "colorize_map",
    "make_contact_sheet",
    "visualize_flow_field",
    "overlay_lic",
    # Segmentation functions
    "segment_image",
    "extract_segmentation_polygon",
    "SEGMENTATION_BACKGROUND",
    "SEGMENTATION_HAIR",
    "SEGMENTATION_BODY_SKIN",
    "SEGMENTATION_FACE_SKIN",
    "SEGMENTATION_CLOTHES",
    "SEGMENTATION_ACCESSORIES",
    # Constants
    "DEFAULT_REGIONS",
]
