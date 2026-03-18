#!/usr/bin/env python3
"""Command-line interface for running the portrait map lab pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from portrait_map_lab.models import PipelineConfig, RemapConfig
from portrait_map_lab.pipelines import run_feature_distance_pipeline, save_pipeline_outputs
from portrait_map_lab.storage import ensure_output_dir, load_image


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the portrait map lab feature distance pipeline on an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional argument
    parser.add_argument("image_path", type=Path, help="Path to the input portrait image")

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output directory for results",
    )

    # Weight configuration
    parser.add_argument(
        "--eye-weight",
        type=float,
        default=0.6,
        help="Weight for eye influence in the combined map",
    )
    parser.add_argument(
        "--mouth-weight",
        type=float,
        default=0.4,
        help="Weight for mouth influence in the combined map",
    )

    # Remap configuration
    parser.add_argument(
        "--curve",
        type=str,
        choices=["linear", "gaussian", "exponential"],
        default="gaussian",
        help="Type of distance-to-influence remapping curve",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=150.0,
        help="Radius parameter for linear curve (distance at which influence reaches 0)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=80.0,
        help="Sigma parameter for gaussian curve (controls spread)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=60.0,
        help="Tau parameter for exponential curve (controls decay rate)",
    )
    parser.add_argument(
        "--clamp-distance",
        type=float,
        default=300.0,
        help="Maximum distance to consider (pixels beyond this are clamped)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI script.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Parse arguments
    args = parse_args()

    # Validate image path
    if not args.image_path.exists():
        logger.error("Image file not found: %s", args.image_path)
        return 1

    try:
        # Load the image
        logger.info("Loading image: %s", args.image_path)
        image = load_image(args.image_path)
        height, width = image.shape[:2]
        logger.info("Image dimensions: %dx%d", width, height)

        # Extract image name for output subdirectory
        image_name = args.image_path.stem

        # Create output directory with pipeline subdirectory
        output_dir = ensure_output_dir(args.output_dir, image_name, pipeline="features")
        logger.info("Output directory: %s", output_dir)

        # Build pipeline configuration from CLI arguments
        remap_config = RemapConfig(
            curve=args.curve,
            radius=args.radius,
            sigma=args.sigma,
            tau=args.tau,
            clamp_distance=args.clamp_distance,
        )

        weights = {
            "eyes": args.eye_weight,
            "mouth": args.mouth_weight,
        }

        config = PipelineConfig(
            remap=remap_config,
            weights=weights,
            output_dir=str(output_dir),
        )

        # Run the pipeline
        logger.info("Running feature distance pipeline...")
        result = run_feature_distance_pipeline(image, config)

        # Save outputs
        logger.info("Saving pipeline outputs...")
        save_pipeline_outputs(result, image, output_dir)

        # Print summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Input image: {args.image_path}")
        print(f"Image dimensions: {width}x{height}")
        print(f"Landmarks detected: {len(result.landmarks.landmarks)}")
        print(f"Face detection confidence: {result.landmarks.confidence:.2%}")
        print(f"Output directory: {output_dir}")
        print("\nFiles created:")
        print("  - landmarks.png (landmark overlay)")
        for mask_name in result.masks.keys():
            print(f"  - mask_{mask_name}.png")
        for field_name in result.distance_fields.keys():
            print(f"  - distance_{field_name}_raw.npy")
            print(f"  - distance_{field_name}_heatmap.png")
        for influence_name in result.influence_maps.keys():
            print(f"  - influence_{influence_name}.png")
        print("  - combined_importance.png")
        print("  - contact_sheet.png (all visualizations)")
        print("\nConfiguration used:")
        print(f"  - Curve type: {args.curve}")
        print(f"  - Eye weight: {args.eye_weight}")
        print(f"  - Mouth weight: {args.mouth_weight}")
        if args.curve == "linear":
            print(f"  - Radius: {args.radius}")
        elif args.curve == "gaussian":
            print(f"  - Sigma: {args.sigma}")
        elif args.curve == "exponential":
            print(f"  - Tau: {args.tau}")
        print(f"  - Clamp distance: {args.clamp_distance}")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except ValueError as e:
        logger.error("Processing error: %s", e)
        if "No face detected" in str(e):
            logger.info(
                "Tip: Make sure the image contains a clear, front-facing portrait "
                "with visible facial features."
            )
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
