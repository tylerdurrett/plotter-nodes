"""Pipeline compositions for end-to-end portrait processing workflows."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from portrait_map_lab.combine import combine_maps
from portrait_map_lab.complexity_map import compute_complexity_map
from portrait_map_lab.compose import build_density_target
from portrait_map_lab.distance_fields import compute_distance_field
from portrait_map_lab.etf import compute_etf
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
from portrait_map_lab.flow_fields import (
    align_tangent_field,
    blend_flow_fields,
    compute_blend_weight,
    compute_contour_flow,
)
from portrait_map_lab.flow_speed import compute_flow_speed
from portrait_map_lab.landmarks import detect_landmarks
from portrait_map_lab.lic import compute_lic
from portrait_map_lab.luminance import compute_tonal_target
from portrait_map_lab.masks import build_region_masks
from portrait_map_lab.models import (
    ComplexityConfig,
    ComplexityResult,
    ComposeConfig,
    ComposedResult,
    ContourConfig,
    ContourResult,
    DensityResult,
    FlowConfig,
    FlowResult,
    FlowSpeedConfig,
    LandmarkResult,
    LICConfig,
    PipelineConfig,
    PipelineResult,
)
from portrait_map_lab.remap import remap_influence
from portrait_map_lab.segmentation import (
    SEGMENTATION_ACCESSORIES,
    SEGMENTATION_FACE_SKIN,
    SEGMENTATION_HAIR,
    extract_segmentation_polygon,
    segment_image,
)
from portrait_map_lab.storage import save_array, save_image
from portrait_map_lab.viz import (
    colorize_map,
    draw_contour,
    draw_landmarks,
    make_contact_sheet,
    overlay_lic,
    visualize_flow_field,
)

logger = logging.getLogger(__name__)


def run_feature_distance_pipeline(
    image: np.ndarray, config: PipelineConfig | None = None
) -> PipelineResult:
    """Run the complete feature distance pipeline on a portrait image.

    Parameters
    ----------
    image
        BGR uint8 image array (as loaded by cv2.imread).
    config
        Pipeline configuration. If None, uses default configuration.

    Returns
    -------
    PipelineResult
        Complete pipeline result with landmarks, masks, distance fields,
        influence maps, and combined output.

    Raises
    ------
    ValueError
        If no face is detected in the image.
    """
    if config is None:
        config = PipelineConfig()

    logger.info("Starting feature distance pipeline...")

    # Step 1: Detect landmarks
    logger.info("Detecting face landmarks...")
    landmarks = detect_landmarks(image)

    # Step 2: Build region masks
    logger.info("Building region masks...")
    masks = build_region_masks(landmarks, config.regions)

    # Step 3: Compute distance fields
    # We compute distance fields for combined_eyes and mouth
    logger.info("Computing distance fields...")
    distance_fields = {}

    if "combined_eyes" in masks:
        distance_fields["eyes"] = compute_distance_field(masks["combined_eyes"])
    else:
        # Fallback: if no combined_eyes, try to use left_eye or right_eye
        if "left_eye" in masks:
            distance_fields["eyes"] = compute_distance_field(masks["left_eye"])
        elif "right_eye" in masks:
            distance_fields["eyes"] = compute_distance_field(masks["right_eye"])
        else:
            logger.warning("No eye masks found, creating empty distance field")
            distance_fields["eyes"] = (
                np.ones_like(masks.get("mouth", np.zeros(landmarks.image_shape)), dtype=np.float64)
                * 1000.0
            )

    if "mouth" in masks:
        distance_fields["mouth"] = compute_distance_field(masks["mouth"])
    else:
        logger.warning("No mouth mask found, creating empty distance field")
        distance_fields["mouth"] = np.ones(landmarks.image_shape, dtype=np.float64) * 1000.0

    # Step 4: Remap distance fields to influence maps
    logger.info("Remapping distance fields to influence maps...")
    influence_maps = {}
    for name, field in distance_fields.items():
        influence_maps[name] = remap_influence(field, config.remap)

    # Step 5: Combine influence maps with weights
    logger.info("Combining influence maps...")
    combined = combine_maps(influence_maps, config.weights)

    logger.info("Pipeline completed successfully")

    return PipelineResult(
        landmarks=landmarks,
        masks=masks,
        distance_fields=distance_fields,
        influence_maps=influence_maps,
        combined=combined,
    )


def save_pipeline_outputs(result: PipelineResult, image: np.ndarray, output_dir: Path) -> None:
    """Save all pipeline outputs to the specified directory.

    Creates the following output structure:
    ```
    output_dir/
        landmarks.png
        mask_left_eye.png
        mask_right_eye.png
        mask_mouth.png
        mask_combined_eyes.png  # if present
        distance_eyes_raw.npy
        distance_eyes_heatmap.png
        distance_mouth_raw.npy
        distance_mouth_heatmap.png
        influence_eyes.png
        influence_mouth.png
        combined_importance.png
        contact_sheet.png
    ```

    Parameters
    ----------
    result
        Complete pipeline result to save.
    image
        Original input image (for landmark overlay).
    output_dir
        Directory to save outputs to (will be created if needed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving pipeline outputs to %s", output_dir)

    # Save landmark overlay
    landmarks_img = draw_landmarks(image, result.landmarks)
    save_image(landmarks_img, output_dir / "landmarks.png")

    # Save masks
    for name, mask in result.masks.items():
        save_image(mask, output_dir / f"mask_{name}.png")

    # Save distance fields (raw and heatmap)
    for name, field in result.distance_fields.items():
        # Save raw numpy array
        save_array(field, output_dir / f"distance_{name}_raw.npy")

        # Normalize and save heatmap visualization
        # Clamp to reasonable range for visualization
        field_norm = np.clip(field, 0, 300) / 300.0
        heatmap = colorize_map(field_norm, colormap="viridis")
        save_image(heatmap, output_dir / f"distance_{name}_heatmap.png")

    # Save influence maps
    for name, influence in result.influence_maps.items():
        influence_img = colorize_map(influence, colormap="inferno")
        save_image(influence_img, output_dir / f"influence_{name}.png")

    # Save combined importance map
    combined_img = colorize_map(result.combined, colormap="hot")
    save_image(combined_img, output_dir / "combined_importance.png")

    # Create and save contact sheet
    contact_images = {
        "Original": image,
        "Landmarks": landmarks_img,
    }

    # Add masks
    for name in ["left_eye", "right_eye", "mouth", "combined_eyes"]:
        if name in result.masks:
            # Convert binary mask to 3-channel for display
            mask_display = cv2.cvtColor(result.masks[name], cv2.COLOR_GRAY2BGR)
            contact_images[f"Mask: {name}"] = mask_display

    # Add distance heatmaps
    for name, field in result.distance_fields.items():
        field_norm = np.clip(field, 0, 300) / 300.0
        heatmap = colorize_map(field_norm, colormap="viridis")
        contact_images[f"Distance: {name}"] = heatmap

    # Add influence maps
    for name, influence in result.influence_maps.items():
        influence_img = colorize_map(influence, colormap="inferno")
        contact_images[f"Influence: {name}"] = influence_img

    # Add combined result
    contact_images["Combined"] = combined_img

    contact_sheet = make_contact_sheet(contact_images, columns=4)
    save_image(contact_sheet, output_dir / "contact_sheet.png")

    logger.info("All outputs saved successfully")


def run_contour_pipeline(image: np.ndarray, config: ContourConfig | None = None) -> ContourResult:
    """Run the face contour distance pipeline on a portrait image.

    Parameters
    ----------
    image
        BGR uint8 image array (as loaded by cv2.imread).
    config
        Contour pipeline configuration. If None, uses default configuration.

    Returns
    -------
    ContourResult
        Complete pipeline result with landmarks, contour polygon, masks,
        distance fields, and influence map.

    Raises
    ------
    ValueError
        If no face is detected in the image.
    """
    if config is None:
        config = ContourConfig()

    logger.info("Starting contour distance pipeline (method: %s)...", config.contour_method)

    # Step 1: Extract contour polygon based on method
    if config.contour_method == "landmarks":
        logger.info("Detecting face landmarks...")
        landmarks = detect_landmarks(image)
        logger.info("Extracting face oval polygon...")
        contour_polygon = get_face_oval_polygon(landmarks)
        image_shape = landmarks.image_shape
    elif config.contour_method in ("segmentation_face", "segmentation_head"):
        landmarks = None
        logger.info("Running segmentation...")
        category_mask = segment_image(image)
        classes = _get_segmentation_classes(config.contour_method)
        logger.info("Extracting contour polygon from segmentation (classes: %s)...", classes)
        contour_polygon = extract_segmentation_polygon(
            category_mask, classes, epsilon_factor=config.epsilon_factor
        )
        image_shape = image.shape[:2]
    elif config.contour_method == "average":
        landmarks = detect_landmarks(image)
        return _compute_average_contour(landmarks, image, config)
    else:
        raise ValueError(f"Unknown contour method: {config.contour_method}")

    # Step 2: Rasterize masks and compute SDF
    logger.info("Rasterizing contour mask...")
    contour_mask = rasterize_contour_mask(
        contour_polygon, image_shape, thickness=config.contour_thickness
    )
    logger.info("Rasterizing filled mask...")
    filled_mask = rasterize_filled_mask(contour_polygon, image_shape)
    logger.info("Computing signed distance field...")
    signed_distance = compute_signed_distance(contour_mask, filled_mask)

    # Step 3: Optional smoothing — re-derive polygon/masks from smoothed SDF
    if config.smooth_contour:
        smooth_sigma = max(image_shape) * 0.01
        logger.info("Smoothing contour (sigma=%.1f)...", smooth_sigma)
        contour_polygon, contour_mask, filled_mask = derive_contour_from_sdf(
            signed_distance,
            thickness=config.contour_thickness,
            epsilon_factor=config.epsilon_factor,
            smooth_sigma=smooth_sigma,
        )
        # Recompute SDF from smoothed contour for consistency
        signed_distance = compute_signed_distance(contour_mask, filled_mask)

    # Step 4: Prepare directional distance
    logger.info("Preparing directional distance (mode: %s)...", config.direction)
    directional_distance = prepare_directional_distance(
        signed_distance, mode=config.direction, band_width=config.band_width
    )

    # Step 6: Remap to influence map
    logger.info("Remapping distance to influence...")
    influence_map = remap_influence(directional_distance, config.remap)

    logger.info("Contour pipeline completed successfully")

    return ContourResult(
        landmarks=landmarks,
        contour_polygon=contour_polygon,
        contour_mask=contour_mask,
        filled_mask=filled_mask,
        signed_distance=signed_distance,
        directional_distance=directional_distance,
        influence_map=influence_map,
    )


def _get_segmentation_classes(method: str) -> list[int]:
    """Return segmentation class indices for the given contour method."""
    if method == "segmentation_face":
        return [SEGMENTATION_FACE_SKIN]
    elif method == "segmentation_head":
        return [SEGMENTATION_HAIR, SEGMENTATION_FACE_SKIN, SEGMENTATION_ACCESSORIES]
    raise ValueError(f"Unknown segmentation method: {method}")


def _compute_average_contour(
    landmarks: LandmarkResult,
    image: np.ndarray,
    config: ContourConfig,
) -> ContourResult:
    """Compute contour by averaging SDFs from all three methods.

    Runs landmarks, segmentation_face, and segmentation_head, computes
    each SDF, averages them, and derives the contour from the blended SDF.
    """
    image_shape = landmarks.image_shape
    logger.info("Computing average of all three contour methods...")

    # Method 1: landmarks
    logger.info("  [1/3] Landmarks contour...")
    poly_lm = get_face_oval_polygon(landmarks)
    sdf_lm = compute_sdf_from_polygon(poly_lm, image_shape, config.contour_thickness)

    # Method 2 & 3: segmentation (run model once)
    logger.info("  [2/3] Segmentation face contour...")
    category_mask = segment_image(image)
    poly_face = extract_segmentation_polygon(
        category_mask,
        _get_segmentation_classes("segmentation_face"),
        epsilon_factor=config.epsilon_factor,
    )
    sdf_face = compute_sdf_from_polygon(poly_face, image_shape, config.contour_thickness)

    logger.info("  [3/3] Segmentation head contour...")
    poly_head = extract_segmentation_polygon(
        category_mask,
        _get_segmentation_classes("segmentation_head"),
        epsilon_factor=config.epsilon_factor,
    )
    sdf_head = compute_sdf_from_polygon(poly_head, image_shape, config.contour_thickness)

    # Average and derive outputs — clamp SDFs so no method dominates far from its boundary,
    # then smooth for rounder, more inclusive contours
    logger.info("  Averaging and smoothing signed distance fields...")
    sdf_clamp = max(image_shape) * 0.05
    signed_distance = average_signed_distances([sdf_lm, sdf_face, sdf_head], clamp=sdf_clamp)
    # Smooth sigma scales with image size for consistent results across resolutions
    smooth_sigma = max(image_shape) * 0.01
    contour_polygon, contour_mask, filled_mask = derive_contour_from_sdf(
        signed_distance,
        thickness=config.contour_thickness,
        epsilon_factor=0.001,
        smooth_sigma=smooth_sigma,
    )

    logger.info("Preparing directional distance (mode: %s)...", config.direction)
    directional_distance = prepare_directional_distance(
        signed_distance, mode=config.direction, band_width=config.band_width
    )
    logger.info("Remapping distance to influence...")
    influence_map = remap_influence(directional_distance, config.remap)
    logger.info("Contour pipeline completed successfully")

    return ContourResult(
        landmarks=landmarks,
        contour_polygon=contour_polygon,
        contour_mask=contour_mask,
        filled_mask=filled_mask,
        signed_distance=signed_distance,
        directional_distance=directional_distance,
        influence_map=influence_map,
    )


def save_contour_outputs(result: ContourResult, image: np.ndarray, output_dir: Path) -> None:
    """Save all contour pipeline outputs to the specified directory.

    Creates the following output structure:
    ```
    output_dir/
        contour_overlay.png
        contour_mask.png
        filled_mask.png
        signed_distance_raw.npy
        signed_distance_heatmap.png
        directional_distance_raw.npy
        directional_distance_heatmap.png
        contour_influence.png
        contact_sheet.png
    ```

    Parameters
    ----------
    result
        Complete contour pipeline result to save.
    image
        Original input image (for contour overlay).
    output_dir
        Directory to save outputs to (will be created if needed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving contour pipeline outputs to %s", output_dir)

    # Save contour overlay
    contour_overlay = draw_contour(image, result.contour_polygon)
    save_image(contour_overlay, output_dir / "contour_overlay.png")

    # Save masks
    save_image(result.contour_mask, output_dir / "contour_mask.png")
    save_image(result.filled_mask, output_dir / "filled_mask.png")

    # Save signed distance field (raw and heatmap with diverging colormap)
    save_array(result.signed_distance, output_dir / "signed_distance_raw.npy")

    # For signed distance, use symmetric normalization for diverging colormap
    max_abs_dist = np.max(np.abs(result.signed_distance))
    if max_abs_dist > 0:
        # Normalize to [-1, 1] then shift to [0, 1] for colormap
        signed_norm = (result.signed_distance / max_abs_dist + 1.0) / 2.0
    else:
        signed_norm = np.full_like(result.signed_distance, 0.5)
    signed_heatmap = colorize_map(signed_norm, colormap="RdBu")
    save_image(signed_heatmap, output_dir / "signed_distance_heatmap.png")

    # Save directional distance field (raw and heatmap)
    save_array(result.directional_distance, output_dir / "directional_distance_raw.npy")

    # Normalize directional distance for visualization
    # Clamp to reasonable range and normalize
    dir_norm = np.clip(result.directional_distance, 0, 300) / 300.0
    dir_heatmap = colorize_map(dir_norm, colormap="viridis")
    save_image(dir_heatmap, output_dir / "directional_distance_heatmap.png")

    # Save influence map
    influence_img = colorize_map(result.influence_map, colormap="inferno")
    save_image(influence_img, output_dir / "contour_influence.png")

    # Create and save contact sheet
    contact_images = {
        "Original": image,
        "Contour Overlay": contour_overlay,
        "Contour Mask": cv2.cvtColor(result.contour_mask, cv2.COLOR_GRAY2BGR),
        "Filled Mask": cv2.cvtColor(result.filled_mask, cv2.COLOR_GRAY2BGR),
        "Signed Distance": signed_heatmap,
        "Directional Distance": dir_heatmap,
        "Influence Map": influence_img,
    }

    contact_sheet = make_contact_sheet(contact_images, columns=4)
    save_image(contact_sheet, output_dir / "contact_sheet.png")

    logger.info("All contour outputs saved successfully")


def run_density_pipeline(
    image: np.ndarray,
    feature_result: PipelineResult,
    contour_result: ContourResult,
    config: ComposeConfig | None = None,
) -> DensityResult:
    """Run the density target composition pipeline.

    Combines tonal targets from luminance with importance maps from features
    and contours to create a composed density target for particle placement.

    Parameters
    ----------
    image
        BGR uint8 image array (as loaded by cv2.imread).
    feature_result
        Pre-computed feature pipeline result containing feature importance maps.
    contour_result
        Pre-computed contour pipeline result containing contour importance map.
    config
        Configuration for density composition. If None, uses default configuration.

    Returns
    -------
    DensityResult
        Complete density pipeline result with luminance, CLAHE, tonal target,
        combined importance map, and final density target.

    Raises
    ------
    ValueError
        If input arrays have mismatched shapes.
    """
    if config is None:
        config = ComposeConfig()

    logger.info("Starting density pipeline...")

    # Step 1: Compute tonal targets from luminance
    logger.info("Computing tonal targets from luminance...")
    luminance, clahe_luminance, tonal_target = compute_tonal_target(image, config.luminance)

    # Step 2: Combine feature and contour importance maps
    logger.info(
        "Combining importance maps (features: %.2f, contour: %.2f)...",
        config.feature_weight,
        config.contour_weight,
    )

    # Create maps and weights dictionaries for combine_maps
    importance_maps = {
        "features": feature_result.combined,
        "contour": contour_result.influence_map,
    }
    importance_weights = {
        "features": config.feature_weight,
        "contour": config.contour_weight,
    }

    # Combine the importance maps with specified weights
    importance = combine_maps(importance_maps, importance_weights)

    # Step 3: Build final density target
    logger.info(
        "Building density target (mode: %s, gamma: %.2f)...", config.tonal_blend_mode, config.gamma
    )
    density_target = build_density_target(
        tonal_target=tonal_target,
        importance=importance,
        mode=config.tonal_blend_mode,
        gamma=config.gamma,
    )

    logger.info("Density pipeline completed successfully")

    return DensityResult(
        luminance=luminance,
        clahe_luminance=clahe_luminance,
        tonal_target=tonal_target,
        importance=importance,
        density_target=density_target,
    )


def save_density_outputs(
    result: DensityResult,
    output_dir: Path,
    image: np.ndarray | None = None,
) -> None:
    """Save all density pipeline outputs to the specified directory.

    Creates the following output structure:
    ```
    output_dir/density/
        luminance.png
        clahe_luminance.png
        tonal_target.png
        importance.png
        density_target.png
        density_target_raw.npy
        contact_sheet.png
    ```

    Parameters
    ----------
    result
        Complete density pipeline result to save.
    output_dir
        Directory to save outputs to (will be created if needed).
    image
        Original input image for the contact sheet. Optional.
    """
    # Create density subdirectory
    density_dir = Path(output_dir) / "density"
    density_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving density pipeline outputs to %s", density_dir)

    # Save grayscale luminance images
    luminance_uint8 = (result.luminance * 255).astype(np.uint8)
    save_image(luminance_uint8, density_dir / "luminance.png")

    clahe_luminance_uint8 = (result.clahe_luminance * 255).astype(np.uint8)
    save_image(clahe_luminance_uint8, density_dir / "clahe_luminance.png")

    # Save colorized tonal target (hot colormap)
    tonal_target_img = colorize_map(result.tonal_target, colormap="hot")
    save_image(tonal_target_img, density_dir / "tonal_target.png")

    # Save colorized importance map (inferno colormap)
    importance_img = colorize_map(result.importance, colormap="inferno")
    save_image(importance_img, density_dir / "importance.png")

    # Save colorized density target (hot colormap)
    density_target_img = colorize_map(result.density_target, colormap="hot")
    save_image(density_target_img, density_dir / "density_target.png")

    # Save raw density target array
    save_array(result.density_target, density_dir / "density_target_raw.npy")

    # Create and save contact sheet
    contact_images = {}

    # Add original image if provided
    if image is not None:
        contact_images["Original"] = image

    # Add luminance images (convert to BGR for display)
    contact_images["Luminance"] = cv2.cvtColor(luminance_uint8, cv2.COLOR_GRAY2BGR)
    contact_images["CLAHE Luminance"] = cv2.cvtColor(clahe_luminance_uint8, cv2.COLOR_GRAY2BGR)

    # Add colorized maps
    contact_images["Tonal Target"] = tonal_target_img
    contact_images["Importance Map"] = importance_img
    contact_images["Density Target"] = density_target_img

    contact_sheet = make_contact_sheet(contact_images, columns=3)
    save_image(contact_sheet, density_dir / "contact_sheet.png")

    logger.info("All density outputs saved successfully")


def run_complexity_pipeline(
    image: np.ndarray,
    config: ComplexityConfig | None = None,
    mask: np.ndarray | None = None,
) -> ComplexityResult:
    """Run the complexity map computation pipeline.

    Computes local image complexity using gradient, Laplacian, or multiscale metrics.
    The complexity map can be used for flow speed modulation.

    Parameters
    ----------
    image
        BGR uint8 image array (as loaded by cv2.imread).
    config
        Configuration for complexity computation. If None, uses default configuration.
    mask
        Optional mask to restrict complexity computation to specific regions.

    Returns
    -------
    ComplexityResult
        Result containing raw and normalized complexity maps.
    """
    if config is None:
        config = ComplexityConfig()

    logger.info("Starting complexity pipeline (metric: %s)...", config.metric)

    # Compute complexity map
    result = compute_complexity_map(image, config, mask)

    logger.info("Complexity pipeline completed successfully")

    return result


def save_complexity_outputs(
    result: ComplexityResult,
    output_dir: Path,
    image: np.ndarray | None = None,
) -> None:
    """Save all complexity pipeline outputs to the specified directory.

    Creates the following output structure:
    ```
    output_dir/complexity/
        <metric>_energy.png        # Raw complexity heatmap
        <metric>_energy_raw.npy    # Raw complexity array
        complexity.png             # Normalized complexity heatmap
        complexity_raw.npy         # Normalized complexity array
        contact_sheet.png          # Combined visualization
    ```

    Parameters
    ----------
    result
        Complete complexity pipeline result to save.
    output_dir
        Directory to save outputs to (will be created if needed).
    image
        Original input image for the contact sheet. Optional.
    """
    # Create complexity subdirectory
    complexity_dir = Path(output_dir) / "complexity"
    complexity_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving complexity outputs to %s", complexity_dir)

    # Save raw complexity heatmap (colorized with viridis)
    raw_img = colorize_map(
        result.raw_complexity / result.raw_complexity.max()
        if result.raw_complexity.max() > 0
        else result.raw_complexity,
        colormap="viridis",
    )
    save_image(raw_img, complexity_dir / f"{result.metric}_energy.png")

    # Save raw complexity array
    save_array(result.raw_complexity, complexity_dir / f"{result.metric}_energy_raw.npy")

    # Save normalized complexity heatmap (colorized with inferno)
    complexity_img = colorize_map(result.complexity, colormap="inferno")
    save_image(complexity_img, complexity_dir / "complexity.png")

    # Save normalized complexity array
    save_array(result.complexity, complexity_dir / "complexity_raw.npy")

    # Build contact sheet
    contact_images = {}

    if image is not None:
        # Resize input image to match complexity map dimensions if needed
        if image.shape[:2] != result.complexity.shape:
            contact_images["Input"] = cv2.resize(
                image, (result.complexity.shape[1], result.complexity.shape[0])
            )
        else:
            contact_images["Input"] = image

    contact_images[f"{result.metric.title()} Energy"] = raw_img
    contact_images["Normalized Complexity"] = complexity_img

    contact_sheet = make_contact_sheet(contact_images, columns=3)
    save_image(contact_sheet, complexity_dir / "contact_sheet.png")

    logger.info("All complexity outputs saved successfully")


def run_flow_pipeline(
    image: np.ndarray,
    contour_result: ContourResult,
    config: FlowConfig | None = None,
    complexity_result: ComplexityResult | None = None,
    speed_config: FlowSpeedConfig | None = None,
) -> FlowResult:
    """Run the flow field computation pipeline.

    Combines Edge Tangent Fields (ETF) from image gradients with contour-based
    flow to create a blended flow field for particle trajectory guidance.
    Optionally computes flow speed from complexity map.

    Parameters
    ----------
    image
        BGR uint8 image array (as loaded by cv2.imread).
    contour_result
        Pre-computed contour pipeline result containing signed distance field.
    config
        Configuration for flow field computation. If None, uses default configuration.
    complexity_result
        Optional pre-computed complexity result for flow speed modulation.
    speed_config
        Configuration for flow speed derivation. If None, uses default configuration.

    Returns
    -------
    FlowResult
        Complete flow pipeline result with ETF, contour flow, blend weights,
        final blended flow field, and optional flow speed.

    Raises
    ------
    ValueError
        If input arrays have mismatched shapes.
    """
    if config is None:
        config = FlowConfig()

    logger.info("Starting flow field pipeline...")

    # Step 1: Compute Edge Tangent Field (ETF)
    logger.info("Computing Edge Tangent Field...")
    etf_result = compute_etf(image, config.etf)

    # Step 2: Compute contour flow from signed distance field
    logger.info("Computing contour flow from signed distance field...")
    contour_flow_x, contour_flow_y = compute_contour_flow(
        contour_result.signed_distance,
        smooth_sigma=config.contour_smooth_sigma,
    )

    # Step 3: Align ETF to contour flow to resolve 180° ambiguity
    logger.info("Aligning ETF tangent field to contour flow...")
    aligned_tx, aligned_ty = align_tangent_field(
        etf_result.tangent_x,
        etf_result.tangent_y,
        contour_flow_x,
        contour_flow_y,
    )

    # Step 4: Compute coherence-based blend weights
    logger.info("Computing blend weights from coherence (power=%.1f)...", config.coherence_power)
    blend_weight = compute_blend_weight(etf_result.coherence, config)

    # Step 5: Blend ETF and contour flow
    logger.info(
        "Blending flow fields (mode: %s, threshold: %.2f)...",
        config.blend_mode,
        config.fallback_threshold,
    )
    flow_x, flow_y = blend_flow_fields(
        aligned_tx,
        aligned_ty,
        contour_flow_x,
        contour_flow_y,
        blend_weight,
        fallback_threshold=config.fallback_threshold,
    )

    # Step 6: Compute flow speed from complexity if provided
    flow_speed = None
    if complexity_result is not None:
        logger.info("Computing flow speed from complexity map...")
        flow_speed = compute_flow_speed(complexity_result.complexity, speed_config)
        logger.info("Flow speed computed (min=%.2f, max=%.2f)", flow_speed.min(), flow_speed.max())

    logger.info("Flow field pipeline completed successfully")

    return FlowResult(
        etf=etf_result,
        contour_flow_x=contour_flow_x,
        contour_flow_y=contour_flow_y,
        blend_weight=blend_weight,
        flow_x=flow_x,
        flow_y=flow_y,
        flow_speed=flow_speed,
    )


def save_flow_outputs(
    result: FlowResult,
    output_dir: Path,
    image: np.ndarray | None = None,
) -> None:
    """Save all flow pipeline outputs to the specified directory.

    Creates the following output structure:
    ```
    output_dir/flow/
        etf_coherence.png
        etf_quiver.png
        contour_flow_quiver.png
        blend_weight.png
        flow_lic.png
        flow_lic_overlay.png
        flow_quiver.png
        flow_x_raw.npy
        flow_y_raw.npy
        contact_sheet.png
    ```

    Parameters
    ----------
    result
        Complete flow pipeline result to save.
    output_dir
        Directory to save outputs to (will be created if needed).
    image
        Original input image for the contact sheet. Optional.
    """
    # Create flow subdirectory
    flow_dir = Path(output_dir) / "flow"
    flow_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving flow outputs to %s", flow_dir)

    # Save ETF coherence heatmap
    coherence_img = colorize_map(result.etf.coherence, colormap="viridis")
    save_image(coherence_img, flow_dir / "etf_coherence.png")

    # Save blend weight heatmap
    weight_img = colorize_map(result.blend_weight, colormap="viridis")
    save_image(weight_img, flow_dir / "blend_weight.png")

    # Generate and save quiver plots
    logger.info("Generating flow field visualizations...")

    # ETF quiver plot
    etf_quiver = visualize_flow_field(
        result.etf.tangent_x,
        result.etf.tangent_y,
        image=image,
        step=20,
        scale=15.0,
        color=(255, 0, 0),  # Blue for ETF
    )
    save_image(etf_quiver, flow_dir / "etf_quiver.png")

    # Contour flow quiver plot
    contour_quiver = visualize_flow_field(
        result.contour_flow_x,
        result.contour_flow_y,
        image=image,
        step=20,
        scale=15.0,
        color=(0, 255, 255),  # Yellow for contour flow
    )
    save_image(contour_quiver, flow_dir / "contour_flow_quiver.png")

    # Blended flow quiver plot
    flow_quiver = visualize_flow_field(
        result.flow_x,
        result.flow_y,
        image=image,
        step=20,
        scale=15.0,
        color=(0, 255, 0),  # Green for blended flow
    )
    save_image(flow_quiver, flow_dir / "flow_quiver.png")

    # Generate LIC visualization
    logger.info("Computing LIC visualization...")
    lic_image = compute_lic(result.flow_x, result.flow_y)

    # Save LIC as grayscale
    lic_uint8 = (lic_image * 255).astype(np.uint8)
    save_image(lic_uint8, flow_dir / "flow_lic.png")

    # Save LIC overlay if we have an input image
    if image is not None:
        lic_overlay = overlay_lic(lic_image, image, alpha=0.5)
        save_image(lic_overlay, flow_dir / "flow_lic_overlay.png")

    # Save raw flow field arrays for downstream use
    save_array(result.flow_x, flow_dir / "flow_x_raw.npy")
    save_array(result.flow_y, flow_dir / "flow_y_raw.npy")

    # Save flow speed if computed
    if result.flow_speed is not None:
        # Save flow speed heatmap
        speed_img = colorize_map(result.flow_speed, colormap="plasma")
        save_image(speed_img, flow_dir / "flow_speed.png")

        # Save raw flow speed array
        save_array(result.flow_speed, flow_dir / "flow_speed_raw.npy")
        logger.info("Saved flow speed outputs")

    # Build enhanced contact sheet with all visualizations
    contact_images = {}
    if image is not None:
        contact_images["Original"] = image

    contact_images["ETF Coherence"] = coherence_img
    contact_images["ETF Quiver"] = etf_quiver
    contact_images["Contour Flow"] = contour_quiver
    contact_images["Blend Weight"] = weight_img
    contact_images["Flow Quiver"] = flow_quiver
    contact_images["Flow LIC"] = cv2.cvtColor(lic_uint8, cv2.COLOR_GRAY2BGR)

    if image is not None:
        contact_images["LIC Overlay"] = lic_overlay

    # Add flow speed to contact sheet if available
    if result.flow_speed is not None:
        contact_images["Flow Speed"] = speed_img

    contact_sheet = make_contact_sheet(contact_images, columns=4)
    save_image(contact_sheet, flow_dir / "contact_sheet.png")

    logger.info("All flow outputs saved successfully")


def _run_feature_pipeline_with_landmarks(
    landmarks: LandmarkResult,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run feature pipeline with pre-computed landmarks.

    Internal helper to avoid duplicate landmark detection.
    """
    if config is None:
        config = PipelineConfig()

    # Build region masks
    logger.info("Building region masks...")
    masks = build_region_masks(landmarks, config.regions)

    # Compute distance fields
    logger.info("Computing distance fields...")
    distance_fields = {}

    if "combined_eyes" in masks:
        distance_fields["eyes"] = compute_distance_field(masks["combined_eyes"])
    else:
        # Fallback: if no combined_eyes, try to use left_eye or right_eye
        if "left_eye" in masks:
            distance_fields["eyes"] = compute_distance_field(masks["left_eye"])
        elif "right_eye" in masks:
            distance_fields["eyes"] = compute_distance_field(masks["right_eye"])
        else:
            logger.warning("No eye masks found, creating empty distance field")
            distance_fields["eyes"] = (
                np.ones_like(masks.get("mouth", np.zeros(landmarks.image_shape)), dtype=np.float64)
                * 1000.0
            )

    if "mouth" in masks:
        distance_fields["mouth"] = compute_distance_field(masks["mouth"])
    else:
        logger.warning("No mouth mask found, creating empty distance field")
        distance_fields["mouth"] = np.ones(landmarks.image_shape, dtype=np.float64) * 1000.0

    # Remap distance fields to influence maps
    logger.info("Remapping distance fields to influence maps...")
    influence_maps = {}
    for name, field in distance_fields.items():
        influence_maps[name] = remap_influence(field, config.remap)

    # Combine influence maps with weights
    logger.info("Combining influence maps...")
    combined = combine_maps(influence_maps, config.weights)

    return PipelineResult(
        landmarks=landmarks,
        masks=masks,
        distance_fields=distance_fields,
        influence_maps=influence_maps,
        combined=combined,
    )


def _run_contour_pipeline_with_landmarks(
    landmarks: LandmarkResult,
    config: ContourConfig | None = None,
    image: np.ndarray | None = None,
) -> ContourResult:
    """Run contour pipeline with pre-computed landmarks.

    Internal helper to avoid duplicate landmark detection.
    For segmentation methods, the *image* parameter is required.
    """
    if config is None:
        config = ContourConfig()

    if config.contour_method == "landmarks":
        # Use pre-computed landmarks
        logger.info("Extracting face oval polygon...")
        contour_polygon = get_face_oval_polygon(landmarks)
        result_landmarks = landmarks
    elif config.contour_method in ("segmentation_face", "segmentation_head"):
        if image is None:
            raise ValueError("image is required for segmentation contour methods")
        result_landmarks = None
        logger.info("Running segmentation...")
        category_mask = segment_image(image)
        classes = _get_segmentation_classes(config.contour_method)
        logger.info("Extracting contour polygon from segmentation (classes: %s)...", classes)
        contour_polygon = extract_segmentation_polygon(
            category_mask, classes, epsilon_factor=config.epsilon_factor
        )
    elif config.contour_method == "average":
        if image is None:
            raise ValueError("image is required for average contour method")
        return _compute_average_contour(landmarks, image, config)
    else:
        raise ValueError(f"Unknown contour method: {config.contour_method}")

    image_shape = landmarks.image_shape

    # Rasterize masks and compute SDF
    logger.info("Rasterizing contour mask...")
    contour_mask = rasterize_contour_mask(
        contour_polygon, image_shape, thickness=config.contour_thickness
    )
    logger.info("Rasterizing filled mask...")
    filled_mask = rasterize_filled_mask(contour_polygon, image_shape)
    logger.info("Computing signed distance field...")
    signed_distance = compute_signed_distance(contour_mask, filled_mask)

    # Optional smoothing — re-derive polygon/masks from smoothed SDF
    if config.smooth_contour:
        smooth_sigma = max(image_shape) * 0.01
        logger.info("Smoothing contour (sigma=%.1f)...", smooth_sigma)
        contour_polygon, contour_mask, filled_mask = derive_contour_from_sdf(
            signed_distance,
            thickness=config.contour_thickness,
            epsilon_factor=config.epsilon_factor,
            smooth_sigma=smooth_sigma,
        )
        signed_distance = compute_signed_distance(contour_mask, filled_mask)

    # Prepare directional distance
    logger.info("Preparing directional distance (mode: %s)...", config.direction)
    directional_distance = prepare_directional_distance(
        signed_distance, mode=config.direction, band_width=config.band_width
    )

    # Remap to influence map
    logger.info("Remapping distance to influence...")
    influence_map = remap_influence(directional_distance, config.remap)

    return ContourResult(
        landmarks=result_landmarks,
        contour_polygon=contour_polygon,
        contour_mask=contour_mask,
        filled_mask=filled_mask,
        signed_distance=signed_distance,
        directional_distance=directional_distance,
        influence_map=influence_map,
    )


def run_all_pipelines(
    image: np.ndarray,
    feature_config: PipelineConfig | None = None,
    contour_config: ContourConfig | None = None,
    compose_config: ComposeConfig | None = None,
    flow_config: FlowConfig | None = None,
    lic_config: LICConfig | None = None,
    complexity_config: ComplexityConfig | None = None,
    speed_config: FlowSpeedConfig | None = None,
) -> ComposedResult:
    """Run all pipelines in sequence, sharing landmarks for efficiency.

    This is the main entry point for the complete portrait processing workflow.
    Landmarks are detected once and shared across all pipeline stages.
    When complexity is enabled, it's computed after contour and used for flow speed.

    Parameters
    ----------
    image
        BGR uint8 image array (as loaded by cv2.imread).
    feature_config
        Configuration for feature distance pipeline. If None, uses defaults.
    contour_config
        Configuration for contour pipeline. If None, uses defaults.
    compose_config
        Configuration for density composition. If None, uses defaults.
    flow_config
        Configuration for flow field computation. If None, uses defaults.
    lic_config
        Configuration for LIC visualization. If None, uses defaults.
    complexity_config
        Optional configuration for complexity computation. If None, complexity is skipped.
    speed_config
        Optional configuration for flow speed derivation. If None, uses defaults when
        complexity is enabled.

    Returns
    -------
    ComposedResult
        Complete pipeline result containing all intermediate results:
        - feature_result: Feature distance pipeline results
        - contour_result: Contour pipeline results
        - density_result: Density composition results
        - complexity_result: Optional complexity computation results
        - flow_result: Flow field computation results (with optional speed)
        - lic_image: Line Integral Convolution visualization

    Raises
    ------
    ValueError
        If no face is detected in the image.
    """
    logger.info("Starting complete pipeline processing...")

    # Step 1: Detect landmarks once for all pipelines
    logger.info("Detecting face landmarks (shared across all pipelines)...")
    landmarks = detect_landmarks(image)
    logger.info("Landmarks detected with confidence: %.2f%%", landmarks.confidence * 100)

    # Step 2: Run feature pipeline with shared landmarks
    logger.info("\n--- Stage 1: Feature Distance Pipeline ---")
    feature_result = _run_feature_pipeline_with_landmarks(landmarks, feature_config)
    logger.info("Feature pipeline completed")

    # Step 3: Run contour pipeline with shared landmarks
    logger.info("\n--- Stage 2: Contour Pipeline ---")
    contour_result = _run_contour_pipeline_with_landmarks(landmarks, contour_config, image=image)
    logger.info("Contour pipeline completed")

    # Step 4: Run density pipeline using results from features and contour
    logger.info("\n--- Stage 3: Density Composition Pipeline ---")
    density_result = run_density_pipeline(image, feature_result, contour_result, compose_config)
    logger.info("Density pipeline completed")

    # Step 5: Optionally run complexity pipeline (unmasked — full image)
    complexity_result = None
    if complexity_config is not None:
        logger.info("\n--- Stage 4: Complexity Map Pipeline ---")
        complexity_result = run_complexity_pipeline(image, complexity_config)
        logger.info("Complexity pipeline completed")

    # Step 6: Run flow pipeline using contour result and optional complexity
    logger.info("\n--- Stage %d: Flow Field Pipeline ---", 5 if complexity_result else 4)
    flow_result = run_flow_pipeline(
        image, contour_result, flow_config, complexity_result, speed_config
    )
    logger.info("Flow pipeline completed")

    # Step 7: Compute LIC visualization from flow field
    logger.info("\n--- Stage %d: LIC Visualization ---", 6 if complexity_result else 5)
    logger.info("Computing Line Integral Convolution...")
    lic_image = compute_lic(flow_result.flow_x, flow_result.flow_y, lic_config)
    logger.info("LIC visualization completed")

    logger.info("\n=== All pipelines completed successfully ===")

    # Create and return composed result
    return ComposedResult(
        feature_result=feature_result,
        contour_result=contour_result,
        density_result=density_result,
        flow_result=flow_result,
        lic_image=lic_image,
        complexity_result=complexity_result,
    )


def save_all_outputs(
    result: ComposedResult,
    output_dir: Path,
    image: np.ndarray,
) -> None:
    """Save all pipeline outputs to their respective subdirectories.

    Creates the following directory structure:
    ```
    output_dir/
        features/
            landmarks.png, masks, distance fields, influence maps, etc.
        contour/
            contour overlay, masks, distance fields, influence map, etc.
        density/
            luminance, CLAHE, tonal target, importance, density target, etc.
        complexity/  (optional)
            <metric>_energy.png, complexity.png, raw arrays, etc.
        flow/
            ETF coherence, flow fields, LIC visualization, flow speed, etc.
    ```

    Parameters
    ----------
    result
        Complete composed pipeline result to save.
    output_dir
        Base directory to save outputs (subdirectories will be created).
    image
        Original input image for visualizations.
    """
    output_dir = Path(output_dir)
    logger.info("Saving all pipeline outputs to %s", output_dir)

    # Save feature pipeline outputs
    features_dir = output_dir / "features"
    logger.info("Saving feature outputs to %s", features_dir)
    save_pipeline_outputs(result.feature_result, image, features_dir)

    # Save contour pipeline outputs
    contour_dir = output_dir / "contour"
    logger.info("Saving contour outputs to %s", contour_dir)
    save_contour_outputs(result.contour_result, image, contour_dir)

    # Save density pipeline outputs
    logger.info("Saving density outputs...")
    save_density_outputs(result.density_result, output_dir, image)

    # Save complexity pipeline outputs if present
    if result.complexity_result is not None:
        logger.info("Saving complexity outputs...")
        save_complexity_outputs(result.complexity_result, output_dir, image)

    # Save flow pipeline outputs
    logger.info("Saving flow outputs...")
    save_flow_outputs(result.flow_result, output_dir, image)

    logger.info("All outputs saved successfully to %s", output_dir)
