"""Pipeline compositions for end-to-end portrait processing workflows."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from portrait_map_lab.combine import combine_maps
from portrait_map_lab.distance_fields import compute_distance_field
from portrait_map_lab.landmarks import detect_landmarks
from portrait_map_lab.masks import build_region_masks
from portrait_map_lab.models import PipelineConfig, PipelineResult
from portrait_map_lab.remap import remap_influence
from portrait_map_lab.storage import save_array, save_image
from portrait_map_lab.viz import colorize_map, draw_landmarks, make_contact_sheet

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
