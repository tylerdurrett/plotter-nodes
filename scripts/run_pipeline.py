#!/usr/bin/env python3
"""Command-line interface for running the portrait map lab pipelines."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from portrait_map_lab.models import ContourConfig, PipelineConfig, RemapConfig
from portrait_map_lab.pipelines import (
    run_contour_pipeline,
    run_feature_distance_pipeline,
    save_contour_outputs,
    save_pipeline_outputs,
)
from portrait_map_lab.storage import ensure_output_dir, load_image


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared arguments to a parser.

    Parameters
    ----------
    parser
        Parser to add arguments to.
    """
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output directory for results",
    )

    # Remap configuration (shared between pipelines)
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run portrait map lab pipelines on an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        title="pipelines",
        description="Available pipeline types",
        dest="pipeline",
        required=True,
        help="Pipeline to run",
    )

    # Features subcommand (existing eye/mouth pipeline)
    features_parser = subparsers.add_parser(
        "features",
        help="Run feature distance pipeline (eyes and mouth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    features_parser.add_argument("image_path", type=Path, help="Path to the input portrait image")
    add_shared_arguments(features_parser)
    features_parser.add_argument(
        "--eye-weight",
        type=float,
        default=0.6,
        help="Weight for eye influence in the combined map",
    )
    features_parser.add_argument(
        "--mouth-weight",
        type=float,
        default=0.4,
        help="Weight for mouth influence in the combined map",
    )

    # Contour subcommand (new face contour pipeline)
    contour_parser = subparsers.add_parser(
        "contour",
        help="Run face contour distance pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    contour_parser.add_argument("image_path", type=Path, help="Path to the input portrait image")
    add_shared_arguments(contour_parser)
    contour_parser.add_argument(
        "--direction",
        type=str,
        choices=["inward", "outward", "both", "band"],
        default="inward",
        help="Direction mode for distance computation",
    )
    contour_parser.add_argument(
        "--band-width",
        type=float,
        default=None,
        help="Width of the band for 'band' direction mode",
    )
    contour_parser.add_argument(
        "--contour-thickness",
        type=int,
        default=1,
        help="Thickness of the contour line in pixels",
    )

    # All subcommand (runs both pipelines)
    all_parser = subparsers.add_parser(
        "all",
        help="Run all available pipelines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    all_parser.add_argument("image_path", type=Path, help="Path to the input portrait image")
    add_shared_arguments(all_parser)

    return parser.parse_args()


def build_remap_config(args: argparse.Namespace) -> RemapConfig:
    """Build RemapConfig from command-line arguments.

    Parameters
    ----------
    args
        Parsed command-line arguments.

    Returns
    -------
    RemapConfig
        Remap configuration.
    """
    return RemapConfig(
        curve=args.curve,
        radius=args.radius,
        sigma=args.sigma,
        tau=args.tau,
        clamp_distance=args.clamp_distance,
    )


def handle_features(
    args: argparse.Namespace, image: Path, image_name: str, logger: logging.Logger
) -> int:
    """Handle the features subcommand.

    Parameters
    ----------
    args
        Parsed command-line arguments.
    image
        Loaded input image.
    image_name
        Name of the image file (without extension).
    logger
        Logger instance.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    height, width = image.shape[:2]

    # Create output directory with pipeline subdirectory
    output_dir = ensure_output_dir(args.output_dir, image_name, pipeline="features")
    logger.info("Output directory: %s", output_dir)

    # Build pipeline configuration
    remap_config = build_remap_config(args)
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
    print("FEATURE PIPELINE COMPLETED SUCCESSFULLY")
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


def handle_contour(
    args: argparse.Namespace, image: Path, image_name: str, logger: logging.Logger
) -> int:
    """Handle the contour subcommand.

    Parameters
    ----------
    args
        Parsed command-line arguments.
    image
        Loaded input image.
    image_name
        Name of the image file (without extension).
    logger
        Logger instance.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    height, width = image.shape[:2]

    # Create output directory with pipeline subdirectory
    output_dir = ensure_output_dir(args.output_dir, image_name, pipeline="contour")
    logger.info("Output directory: %s", output_dir)

    # Build contour configuration
    remap_config = build_remap_config(args)
    config = ContourConfig(
        remap=remap_config,
        direction=args.direction,
        band_width=args.band_width,
        contour_thickness=args.contour_thickness,
        output_dir=str(output_dir),
    )

    # Run the pipeline
    logger.info("Running face contour pipeline...")
    result = run_contour_pipeline(image, config)

    # Save outputs
    logger.info("Saving contour outputs...")
    save_contour_outputs(result, image, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("CONTOUR PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Input image: {args.image_path}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Landmarks detected: {len(result.landmarks.landmarks)}")
    print(f"Face detection confidence: {result.landmarks.confidence:.2%}")
    print(f"Output directory: {output_dir}")
    print("\nFiles created:")
    print("  - contour_overlay.png (face oval overlay)")
    print("  - contour_mask.png")
    print("  - filled_mask.png")
    print("  - signed_distance_raw.npy")
    print("  - signed_distance_heatmap.png")
    print("  - directional_distance_raw.npy")
    print("  - directional_distance_heatmap.png")
    print("  - contour_influence.png")
    print("  - contact_sheet.png (all visualizations)")
    print("\nConfiguration used:")
    print(f"  - Direction mode: {args.direction}")
    if args.direction == "band" and args.band_width:
        print(f"  - Band width: {args.band_width}")
    print(f"  - Contour thickness: {args.contour_thickness}")
    print(f"  - Curve type: {args.curve}")
    if args.curve == "linear":
        print(f"  - Radius: {args.radius}")
    elif args.curve == "gaussian":
        print(f"  - Sigma: {args.sigma}")
    elif args.curve == "exponential":
        print(f"  - Tau: {args.tau}")
    print(f"  - Clamp distance: {args.clamp_distance}")
    print("=" * 60)

    return 0


def handle_all(
    args: argparse.Namespace, image: Path, image_name: str, logger: logging.Logger
) -> int:
    """Handle the all subcommand (runs both pipelines).

    Parameters
    ----------
    args
        Parsed command-line arguments.
    image
        Loaded input image.
    image_name
        Name of the image file (without extension).
    logger
        Logger instance.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error).
    """
    height, width = image.shape[:2]
    logger.info("Running all pipelines...")

    # Run features pipeline with defaults
    logger.info("\n--- Running feature distance pipeline ---")
    features_output_dir = ensure_output_dir(args.output_dir, image_name, pipeline="features")
    logger.info("Features output directory: %s", features_output_dir)

    remap_config = build_remap_config(args)
    features_config = PipelineConfig(
        remap=remap_config,
        weights={"eyes": 0.6, "mouth": 0.4},  # defaults
        output_dir=str(features_output_dir),
    )

    features_result = run_feature_distance_pipeline(image, features_config)
    save_pipeline_outputs(features_result, image, features_output_dir)

    # Run contour pipeline with defaults
    logger.info("\n--- Running face contour pipeline ---")
    contour_output_dir = ensure_output_dir(args.output_dir, image_name, pipeline="contour")
    logger.info("Contour output directory: %s", contour_output_dir)

    contour_config = ContourConfig(
        remap=remap_config,
        direction="inward",  # default
        band_width=None,
        contour_thickness=1,  # default
        output_dir=str(contour_output_dir),
    )

    contour_result = run_contour_pipeline(image, contour_config)
    save_contour_outputs(contour_result, image, contour_output_dir)

    # Print combined summary
    print("\n" + "=" * 60)
    print("ALL PIPELINES COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Input image: {args.image_path}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Landmarks detected: {len(features_result.landmarks.landmarks)}")
    print(f"Face detection confidence: {features_result.landmarks.confidence:.2%}")
    print("\nFeature Pipeline:")
    print(f"  Output directory: {features_output_dir}")
    print("  Files created:")
    print("    - landmarks.png, masks, distance fields, influence maps")
    print("    - combined_importance.png, contact_sheet.png")
    print("\nContour Pipeline:")
    print(f"  Output directory: {contour_output_dir}")
    print("  Files created:")
    print("    - contour_overlay.png, masks, signed/directional distance")
    print("    - contour_influence.png, contact_sheet.png")
    print("\nShared configuration:")
    print(f"  - Curve type: {args.curve}")
    if args.curve == "linear":
        print(f"  - Radius: {args.radius}")
    elif args.curve == "gaussian":
        print(f"  - Sigma: {args.sigma}")
    elif args.curve == "exponential":
        print(f"  - Tau: {args.tau}")
    print(f"  - Clamp distance: {args.clamp_distance}")
    print("=" * 60)

    return 0


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

        # Route to appropriate handler based on subcommand
        if args.pipeline == "features":
            return handle_features(args, image, image_name, logger)
        elif args.pipeline == "contour":
            return handle_contour(args, image, image_name, logger)
        elif args.pipeline == "all":
            return handle_all(args, image, image_name, logger)
        else:
            logger.error("Unknown pipeline: %s", args.pipeline)
            return 1

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
