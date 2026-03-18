"""Portrait Map Lab — feature distance mapping pipeline for pen plotter artwork."""

from __future__ import annotations

from portrait_map_lab.combine import combine_maps
from portrait_map_lab.distance_fields import compute_distance_field
from portrait_map_lab.face_contour import (
    FACE_OVAL_INDICES,
    compute_signed_distance,
    get_face_oval_polygon,
    prepare_directional_distance,
    rasterize_contour_mask,
    rasterize_filled_mask,
)
from portrait_map_lab.face_regions import DEFAULT_REGIONS, get_region_polygons
from portrait_map_lab.landmarks import detect_landmarks
from portrait_map_lab.masks import build_region_masks
from portrait_map_lab.models import (
    ContourConfig,
    ContourResult,
    LandmarkResult,
    PipelineConfig,
    PipelineResult,
    RegionDefinition,
    RemapConfig,
)
from portrait_map_lab.pipelines import (
    run_contour_pipeline,
    run_feature_distance_pipeline,
    save_contour_outputs,
    save_pipeline_outputs,
)
from portrait_map_lab.remap import remap_influence
from portrait_map_lab.storage import ensure_output_dir, load_image, save_array, save_image
from portrait_map_lab.viz import colorize_map, draw_contour, draw_landmarks, make_contact_sheet

__version__ = "0.1.0"

__all__ = [
    # Main pipeline functions
    "run_feature_distance_pipeline",
    "run_contour_pipeline",
    "save_pipeline_outputs",
    "save_contour_outputs",
    # Core data models
    "ContourConfig",
    "ContourResult",
    "LandmarkResult",
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
    # Constants
    "DEFAULT_REGIONS",
]
