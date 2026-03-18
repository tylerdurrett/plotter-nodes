"""Pipeline compositions for end-to-end portrait processing workflows."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from portrait_map_lab.combine import combine_maps
from portrait_map_lab.compose import build_density_target
from portrait_map_lab.distance_fields import compute_distance_field
from portrait_map_lab.etf import compute_etf
from portrait_map_lab.face_contour import (
    compute_signed_distance,
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
from portrait_map_lab.landmarks import detect_landmarks
from portrait_map_lab.luminance import compute_tonal_target
from portrait_map_lab.masks import build_region_masks
from portrait_map_lab.models import (
    ComposeConfig,
    ContourConfig,
    ContourResult,
    DensityResult,
    FlowConfig,
    FlowResult,
    PipelineConfig,
    PipelineResult,
)
from portrait_map_lab.remap import remap_influence
from portrait_map_lab.storage import save_array, save_image
from portrait_map_lab.viz import colorize_map, draw_contour, draw_landmarks, make_contact_sheet

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

    logger.info("Starting contour distance pipeline...")

    # Step 1: Detect landmarks
    logger.info("Detecting face landmarks...")
    landmarks = detect_landmarks(image)

    # Step 2: Extract face oval polygon
    logger.info("Extracting face oval polygon...")
    contour_polygon = get_face_oval_polygon(landmarks)

    # Step 3: Rasterize contour mask
    logger.info("Rasterizing contour mask...")
    contour_mask = rasterize_contour_mask(
        contour_polygon, landmarks.image_shape, thickness=config.contour_thickness
    )

    # Step 4: Rasterize filled mask
    logger.info("Rasterizing filled mask...")
    filled_mask = rasterize_filled_mask(contour_polygon, landmarks.image_shape)

    # Step 5: Compute signed distance field
    logger.info("Computing signed distance field...")
    signed_distance = compute_signed_distance(contour_mask, filled_mask)

    # Step 6: Prepare directional distance
    logger.info("Preparing directional distance (mode: %s)...", config.direction)
    directional_distance = prepare_directional_distance(
        signed_distance, mode=config.direction, band_width=config.band_width
    )

    # Step 7: Remap to influence map
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
    logger.info("Combining importance maps (features: %.2f, contour: %.2f)...",
                config.feature_weight, config.contour_weight)

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
    logger.info("Building density target (mode: %s, gamma: %.2f)...",
                config.tonal_blend_mode, config.gamma)
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


def run_flow_pipeline(
    image: np.ndarray,
    contour_result: ContourResult,
    config: FlowConfig | None = None,
) -> FlowResult:
    """Run the flow field computation pipeline.

    Combines Edge Tangent Fields (ETF) from image gradients with contour-based
    flow to create a blended flow field for particle trajectory guidance.

    Parameters
    ----------
    image
        BGR uint8 image array (as loaded by cv2.imread).
    contour_result
        Pre-computed contour pipeline result containing signed distance field.
    config
        Configuration for flow field computation. If None, uses default configuration.

    Returns
    -------
    FlowResult
        Complete flow pipeline result with ETF, contour flow, blend weights,
        and final blended flow field.

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
    logger.info("Blending flow fields (mode: %s, threshold: %.2f)...",
                config.blend_mode, config.fallback_threshold)
    flow_x, flow_y = blend_flow_fields(
        aligned_tx,
        aligned_ty,
        contour_flow_x,
        contour_flow_y,
        blend_weight,
        fallback_threshold=config.fallback_threshold,
    )

    logger.info("Flow field pipeline completed successfully")

    return FlowResult(
        etf=etf_result,
        contour_flow_x=contour_flow_x,
        contour_flow_y=contour_flow_y,
        blend_weight=blend_weight,
        flow_x=flow_x,
        flow_y=flow_y,
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
        etf_quiver.png  (deferred to Phase 5)
        contour_flow_quiver.png  (deferred to Phase 5)
        blend_weight.png
        flow_lic.png  (deferred to Phase 5)
        flow_lic_overlay.png  (deferred to Phase 5)
        flow_quiver.png  (deferred to Phase 5)
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

    # Save raw flow field arrays for downstream use
    save_array(result.flow_x, flow_dir / "flow_x_raw.npy")
    save_array(result.flow_y, flow_dir / "flow_y_raw.npy")

    # Build contact sheet (quiver and LIC overlays deferred to Phase 5)
    contact_images = {}
    if image is not None:
        contact_images["Original"] = image

    contact_images["ETF Coherence"] = coherence_img
    contact_images["Blend Weight"] = weight_img

    # For now, visualize flow magnitude as placeholder
    flow_magnitude = np.sqrt(result.flow_x**2 + result.flow_y**2)
    flow_mag_img = colorize_map(flow_magnitude, colormap="hot")
    contact_images["Flow Magnitude"] = flow_mag_img

    contact_sheet = make_contact_sheet(contact_images, columns=3)
    save_image(contact_sheet, flow_dir / "contact_sheet.png")

    logger.info("All flow outputs saved successfully")
