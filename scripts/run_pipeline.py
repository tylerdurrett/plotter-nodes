#!/usr/bin/env python3
"""Command-line interface for running the portrait map lab pipelines."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2

from portrait_map_lab.models import (
    ComplexityConfig,
    ComposeConfig,
    ContourConfig,
    ETFConfig,
    FlowConfig,
    FlowSpeedConfig,
    LICConfig,
    LuminanceConfig,
    PipelineConfig,
    RemapConfig,
)
from portrait_map_lab.pipelines import (
    run_all_pipelines,
    run_complexity_pipeline,
    run_contour_pipeline,
    run_density_pipeline,
    run_feature_distance_pipeline,
    run_flow_pipeline,
    save_all_outputs,
    save_complexity_outputs,
    save_contour_outputs,
    save_density_outputs,
    save_flow_outputs,
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
    contour_parser.add_argument(
        "--contour-method",
        type=str,
        choices=["landmarks", "segmentation_face", "segmentation_head", "average"],
        default="landmarks",
        help="Method for contour extraction: landmarks (convex hull), "
        "segmentation_face (face skin mask), segmentation_head (hair+face+accessories), "
        "average (blend of all three methods)",
    )
    contour_parser.add_argument(
        "--epsilon-factor",
        type=float,
        default=0.005,
        help="Contour simplification factor for segmentation methods (0 to disable)",
    )
    contour_parser.add_argument(
        "--no-smooth",
        action="store_true",
        default=False,
        help="Disable contour smoothing (produces sharper, more angular contours)",
    )

    # Density subcommand (density composition pipeline)
    density_parser = subparsers.add_parser(
        "density",
        help="Run density composition pipeline (features + contour + density)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    density_parser.add_argument("image_path", type=Path, help="Path to the input portrait image")
    add_shared_arguments(density_parser)
    # Luminance arguments
    density_parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit for contrast enhancement",
    )
    density_parser.add_argument(
        "--tile-size",
        type=int,
        default=8,
        help="CLAHE tile grid size for local enhancement",
    )
    # Composition arguments
    density_parser.add_argument(
        "--tonal-blend",
        type=str,
        choices=["multiply", "screen", "max", "weighted"],
        default="multiply",
        help="Blend mode for combining tonal target with importance",
    )
    density_parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction for final density target",
    )
    density_parser.add_argument(
        "--feature-weight",
        type=float,
        default=0.6,
        help="Weight for feature importance in combined map",
    )
    density_parser.add_argument(
        "--contour-weight",
        type=float,
        default=0.4,
        help="Weight for contour importance in combined map",
    )

    # Flow subcommand (flow field pipeline)
    flow_parser = subparsers.add_parser(
        "flow",
        help="Run flow field pipeline (contour + ETF + flow + LIC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    flow_parser.add_argument("image_path", type=Path, help="Path to the input portrait image")
    add_shared_arguments(flow_parser)
    # ETF arguments
    flow_parser.add_argument(
        "--blur-sigma",
        type=float,
        default=1.5,
        help="Gaussian blur sigma for ETF input",
    )
    flow_parser.add_argument(
        "--structure-sigma",
        type=float,
        default=5.0,
        help="Gaussian sigma for structure tensor smoothing",
    )
    flow_parser.add_argument(
        "--refine-sigma",
        type=float,
        default=3.0,
        help="Gaussian sigma for tangent field refinement",
    )
    flow_parser.add_argument(
        "--refine-iterations",
        type=int,
        default=2,
        help="Number of refinement iterations for ETF",
    )
    flow_parser.add_argument(
        "--sobel-ksize",
        type=int,
        default=3,
        help="Sobel kernel size for gradient computation",
    )
    # Flow blending arguments
    flow_parser.add_argument(
        "--contour-smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for contour flow smoothing",
    )
    flow_parser.add_argument(
        "--coherence-power",
        type=float,
        default=2.0,
        help="Power factor for coherence-based blend weight",
    )
    flow_parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.1,
        help="Magnitude threshold for flow fallback",
    )
    # LIC arguments
    flow_parser.add_argument(
        "--lic-length",
        type=int,
        default=30,
        help="Length of LIC streamline integration",
    )
    flow_parser.add_argument(
        "--lic-step",
        type=float,
        default=1.0,
        help="Step size for LIC integration",
    )
    flow_parser.add_argument(
        "--lic-seed",
        type=int,
        default=42,
        help="Random seed for LIC noise texture",
    )
    # Complexity arguments (optional - only used when --metric is provided)
    flow_parser.add_argument(
        "--metric",
        type=str,
        choices=["gradient", "laplacian", "multiscale_gradient"],
        default=None,
        help="Optional complexity metric to compute for flow speed modulation",
    )
    flow_parser.add_argument(
        "--complexity-sigma",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma for single-scale complexity metrics",
    )
    flow_parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 8.0],
        help="Sigma values for multiscale complexity metric",
    )
    flow_parser.add_argument(
        "--scale-weights",
        type=float,
        nargs="+",
        default=[0.5, 0.3, 0.2],
        help="Weights for each scale in multiscale complexity metric",
    )
    flow_parser.add_argument(
        "--normalize-percentile",
        type=float,
        default=99.0,
        help="Percentile for complexity normalization (100.0 = max normalization)",
    )
    flow_parser.add_argument(
        "--speed-min",
        type=float,
        default=0.3,
        help="Flow speed in most complex areas (when complexity is enabled)",
    )
    flow_parser.add_argument(
        "--speed-max",
        type=float,
        default=1.0,
        help="Flow speed in smooth areas (when complexity is enabled)",
    )

    # Complexity subcommand (complexity map pipeline)
    complexity_parser = subparsers.add_parser(
        "complexity",
        help="Run complexity map pipeline for flow speed modulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    complexity_parser.add_argument(
        "image_path", type=Path, help="Path to the input portrait image"
    )
    add_shared_arguments(complexity_parser)
    complexity_parser.add_argument(
        "--metric",
        type=str,
        choices=["gradient", "laplacian", "multiscale_gradient"],
        default="gradient",
        help="Complexity metric to compute",
    )
    complexity_parser.add_argument(
        "--complexity-sigma",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma for single-scale metrics",
    )
    complexity_parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 8.0],
        help="Sigma values for multiscale metric",
    )
    complexity_parser.add_argument(
        "--scale-weights",
        type=float,
        nargs="+",
        default=[0.5, 0.3, 0.2],
        help="Weights for each scale in multiscale metric",
    )
    complexity_parser.add_argument(
        "--normalize-percentile",
        type=float,
        default=99.0,
        help="Percentile for normalization (100.0 = max normalization)",
    )
    complexity_parser.add_argument(
        "--mask-image",
        type=Path,
        default=None,
        help="Optional mask image to restrict complexity computation",
    )

    # All subcommand (runs all pipelines)
    all_parser = subparsers.add_parser(
        "all",
        help="Run all available pipelines with shared landmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    all_parser.add_argument("image_path", type=Path, help="Path to the input portrait image")
    add_shared_arguments(all_parser)
    # Add all density and flow arguments to all subcommand
    # Feature weights
    all_parser.add_argument(
        "--eye-weight",
        type=float,
        default=0.6,
        help="Weight for eye influence in the combined map",
    )
    all_parser.add_argument(
        "--mouth-weight",
        type=float,
        default=0.4,
        help="Weight for mouth influence in the combined map",
    )
    # Contour arguments
    all_parser.add_argument(
        "--direction",
        type=str,
        choices=["inward", "outward", "both", "band"],
        default="inward",
        help="Direction mode for distance computation",
    )
    all_parser.add_argument(
        "--band-width",
        type=float,
        default=None,
        help="Width of the band for 'band' direction mode",
    )
    all_parser.add_argument(
        "--contour-thickness",
        type=int,
        default=1,
        help="Thickness of the contour line in pixels",
    )
    all_parser.add_argument(
        "--contour-method",
        type=str,
        choices=["landmarks", "segmentation_face", "segmentation_head", "average"],
        default="landmarks",
        help="Method for contour extraction: landmarks (convex hull), "
        "segmentation_face (face skin mask), segmentation_head (hair+face+accessories), "
        "average (blend of all three methods)",
    )
    all_parser.add_argument(
        "--epsilon-factor",
        type=float,
        default=0.005,
        help="Contour simplification factor for segmentation methods (0 to disable)",
    )
    all_parser.add_argument(
        "--no-smooth",
        action="store_true",
        default=False,
        help="Disable contour smoothing (produces sharper, more angular contours)",
    )
    # Density arguments
    all_parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.0,
        help="CLAHE clip limit for contrast enhancement",
    )
    all_parser.add_argument(
        "--tile-size",
        type=int,
        default=8,
        help="CLAHE tile grid size for local enhancement",
    )
    all_parser.add_argument(
        "--tonal-blend",
        type=str,
        choices=["multiply", "screen", "max", "weighted"],
        default="multiply",
        help="Blend mode for combining tonal target with importance",
    )
    all_parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction for final density target",
    )
    all_parser.add_argument(
        "--feature-weight",
        type=float,
        default=0.6,
        help="Weight for feature importance in combined map",
    )
    all_parser.add_argument(
        "--contour-weight",
        type=float,
        default=0.4,
        help="Weight for contour importance in combined map",
    )
    # Flow arguments
    all_parser.add_argument(
        "--blur-sigma",
        type=float,
        default=1.5,
        help="Gaussian blur sigma for ETF input",
    )
    all_parser.add_argument(
        "--structure-sigma",
        type=float,
        default=5.0,
        help="Gaussian sigma for structure tensor smoothing",
    )
    all_parser.add_argument(
        "--refine-sigma",
        type=float,
        default=3.0,
        help="Gaussian sigma for tangent field refinement",
    )
    all_parser.add_argument(
        "--refine-iterations",
        type=int,
        default=2,
        help="Number of refinement iterations for ETF",
    )
    all_parser.add_argument(
        "--coherence-power",
        type=float,
        default=2.0,
        help="Power factor for coherence-based blend weight",
    )
    all_parser.add_argument(
        "--fallback-threshold",
        type=float,
        default=0.1,
        help="Magnitude threshold for flow fallback",
    )
    all_parser.add_argument(
        "--lic-length",
        type=int,
        default=30,
        help="Length of LIC streamline integration",
    )
    all_parser.add_argument(
        "--lic-seed",
        type=int,
        default=42,
        help="Random seed for LIC noise texture",
    )
    # Complexity arguments (optional - only used when --metric is provided)
    all_parser.add_argument(
        "--metric",
        type=str,
        choices=["gradient", "laplacian", "multiscale_gradient"],
        default=None,
        help="Optional complexity metric to compute for flow speed modulation",
    )
    all_parser.add_argument(
        "--complexity-sigma",
        type=float,
        default=3.0,
        help="Gaussian smoothing sigma for single-scale complexity metrics",
    )
    all_parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 8.0],
        help="Sigma values for multiscale complexity metric",
    )
    all_parser.add_argument(
        "--scale-weights",
        type=float,
        nargs="+",
        default=[0.5, 0.3, 0.2],
        help="Weights for each scale in multiscale complexity metric",
    )
    all_parser.add_argument(
        "--normalize-percentile",
        type=float,
        default=99.0,
        help="Percentile for complexity normalization (100.0 = max normalization)",
    )
    all_parser.add_argument(
        "--speed-min",
        type=float,
        default=0.3,
        help="Flow speed in most complex areas (when complexity is enabled)",
    )
    all_parser.add_argument(
        "--speed-max",
        type=float,
        default=1.0,
        help="Flow speed in smooth areas (when complexity is enabled)",
    )
    # Export bundle
    all_parser.add_argument(
        "--export",
        action="store_true",
        help="Generate export bundle (float32 .bin + manifest.json) for TypeScript",
    )

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


def build_complexity_config(args: argparse.Namespace) -> ComplexityConfig | None:
    """Build ComplexityConfig from command-line arguments.

    Parameters
    ----------
    args
        Parsed command-line arguments.

    Returns
    -------
    ComplexityConfig | None
        Complexity configuration if metric is specified, None otherwise.
    """
    if not hasattr(args, "metric") or args.metric is None:
        return None

    return ComplexityConfig(
        metric=args.metric,
        sigma=args.complexity_sigma,
        scales=args.scales,
        scale_weights=args.scale_weights,
        normalize_percentile=args.normalize_percentile,
        output_dir=str(args.output_dir),
    )


def build_flow_speed_config(args: argparse.Namespace) -> FlowSpeedConfig | None:
    """Build FlowSpeedConfig from command-line arguments.

    Parameters
    ----------
    args
        Parsed command-line arguments.

    Returns
    -------
    FlowSpeedConfig | None
        Flow speed configuration if speed args are present, None otherwise.
    """
    if not hasattr(args, "speed_min"):
        return None

    return FlowSpeedConfig(
        speed_min=args.speed_min,
        speed_max=args.speed_max,
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
        contour_method=args.contour_method,
        remap=remap_config,
        direction=args.direction,
        band_width=args.band_width,
        contour_thickness=args.contour_thickness,
        epsilon_factor=args.epsilon_factor,
        smooth_contour=not args.no_smooth,
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
    if result.landmarks is not None:
        print(f"Landmarks detected: {len(result.landmarks.landmarks)}")
        print(f"Face detection confidence: {result.landmarks.confidence:.2%}")
    else:
        print(f"Contour method: {args.contour_method}")
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


def handle_density(
    args: argparse.Namespace, image: Path, image_name: str, logger: logging.Logger
) -> int:
    """Handle the density subcommand.

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
    logger.info("Running density composition pipeline...")

    # Create output directory
    output_dir = ensure_output_dir(args.output_dir, image_name, pipeline="density")
    logger.info("Output directory: %s", output_dir)

    # Build configurations
    remap_config = build_remap_config(args)

    # Run features pipeline to get feature importance
    logger.info("Running feature distance pipeline...")
    features_config = PipelineConfig(
        remap=remap_config,
        weights={"eyes": 0.6, "mouth": 0.4},  # defaults
        output_dir=str(output_dir),
    )
    features_result = run_feature_distance_pipeline(image, features_config)

    # Run contour pipeline to get contour importance
    logger.info("Running face contour pipeline...")
    contour_config = ContourConfig(
        remap=remap_config,
        direction="inward",  # default
        band_width=None,
        contour_thickness=1,  # default
        output_dir=str(output_dir),
    )
    contour_result = run_contour_pipeline(image, contour_config)

    # Build density composition config
    luminance_config = LuminanceConfig(
        clip_limit=args.clip_limit,
        tile_size=args.tile_size,
    )
    compose_config = ComposeConfig(
        luminance=luminance_config,
        feature_weight=args.feature_weight,
        contour_weight=args.contour_weight,
        tonal_blend_mode=args.tonal_blend,
        gamma=args.gamma,
    )

    # Run density pipeline
    logger.info("Running density composition...")
    density_result = run_density_pipeline(image, features_result, contour_result, compose_config)

    # Save density outputs
    logger.info("Saving density outputs...")
    parent_dir = output_dir.parent
    save_density_outputs(density_result, parent_dir, image)

    # Print summary
    print("\n" + "=" * 60)
    print("DENSITY PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Input image: {args.image_path}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Output directory: {parent_dir / 'density'}")
    print("\nFiles created:")
    print("  - luminance.png")
    print("  - clahe_luminance.png")
    print("  - tonal_target.png")
    print("  - importance.png")
    print("  - density_target.png")
    print("  - density_target_raw.npy")
    print("  - contact_sheet.png")
    print("\nConfiguration used:")
    print(f"  - CLAHE clip limit: {args.clip_limit}")
    print(f"  - CLAHE tile size: {args.tile_size}")
    print(f"  - Tonal blend mode: {args.tonal_blend}")
    print(f"  - Gamma: {args.gamma}")
    print(f"  - Feature weight: {args.feature_weight}")
    print(f"  - Contour weight: {args.contour_weight}")
    print("=" * 60)

    return 0


def handle_flow(
    args: argparse.Namespace, image: Path, image_name: str, logger: logging.Logger
) -> int:
    """Handle the flow subcommand.

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
    logger.info("Running flow field pipeline...")

    # Create output directory
    output_dir = ensure_output_dir(args.output_dir, image_name, pipeline="flow")
    logger.info("Output directory: %s", output_dir)

    # Build configurations
    remap_config = build_remap_config(args)

    # Run contour pipeline to get signed distance field
    logger.info("Running face contour pipeline...")
    contour_config = ContourConfig(
        remap=remap_config,
        direction="inward",  # default
        band_width=None,
        contour_thickness=1,  # default
        output_dir=str(output_dir),
    )
    contour_result = run_contour_pipeline(image, contour_config)

    # Optionally run complexity pipeline if metric is specified
    complexity_result = None
    speed_config = None
    if args.metric is not None:
        logger.info("Running complexity pipeline for flow speed modulation...")
        complexity_config = build_complexity_config(args)
        complexity_result = run_complexity_pipeline(image, complexity_config)
        speed_config = build_flow_speed_config(args)
        # Save complexity outputs in parent dir
        save_complexity_outputs(complexity_result, output_dir.parent, image)

    # Build flow config
    etf_config = ETFConfig(
        blur_sigma=args.blur_sigma,
        structure_sigma=args.structure_sigma,
        refine_sigma=args.refine_sigma,
        refine_iterations=args.refine_iterations,
        sobel_ksize=args.sobel_ksize,
    )
    flow_config = FlowConfig(
        etf=etf_config,
        contour_smooth_sigma=args.contour_smooth_sigma,
        blend_mode="coherence",  # default
        coherence_power=args.coherence_power,
        fallback_threshold=args.fallback_threshold,
    )

    # Run flow pipeline with optional complexity for speed
    logger.info("Running flow field computation...")
    flow_result = run_flow_pipeline(
        image, contour_result, flow_config, complexity_result, speed_config
    )

    # Save flow outputs
    logger.info("Saving flow outputs...")
    parent_dir = output_dir.parent
    save_flow_outputs(flow_result, parent_dir, image)

    # Print summary
    print("\n" + "=" * 60)
    print("FLOW PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Input image: {args.image_path}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Output directory: {parent_dir / 'flow'}")
    print("\nFiles created:")
    print("  - etf_coherence.png")
    print("  - etf_quiver.png")
    print("  - contour_flow_quiver.png")
    print("  - blend_weight.png")
    print("  - flow_lic.png")
    print("  - flow_lic_overlay.png")
    print("  - flow_quiver.png")
    print("  - flow_x_raw.npy")
    print("  - flow_y_raw.npy")
    print("  - contact_sheet.png")
    print("\nConfiguration used:")
    print(f"  - ETF blur sigma: {args.blur_sigma}")
    print(f"  - ETF structure sigma: {args.structure_sigma}")
    print(f"  - ETF refine sigma: {args.refine_sigma}")
    print(f"  - ETF refine iterations: {args.refine_iterations}")
    print(f"  - Coherence power: {args.coherence_power}")
    print(f"  - Fallback threshold: {args.fallback_threshold}")
    print(f"  - LIC length: {args.lic_length}")
    print(f"  - LIC seed: {args.lic_seed}")
    if complexity_result is not None:
        print("\nComplexity-based flow speed:")
        print(f"  - Metric: {args.metric}")
        print(f"  - Speed range: {args.speed_min:.1f} to {args.speed_max:.1f}")
        print("  - flow_speed.png and flow_speed_raw.npy saved")
    print("=" * 60)

    return 0


def handle_complexity(
    args: argparse.Namespace, image: Path, image_name: str, logger: logging.Logger
) -> int:
    """Handle the complexity subcommand.

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
    logger.info("Running complexity map pipeline...")

    # Create output directory
    output_dir = ensure_output_dir(args.output_dir, image_name, pipeline="complexity")
    logger.info("Output directory: %s", output_dir)

    # Build complexity configuration
    config = ComplexityConfig(
        metric=args.metric,
        sigma=args.complexity_sigma,
        scales=args.scales,
        scale_weights=args.scale_weights,
        normalize_percentile=args.normalize_percentile,
        output_dir=str(output_dir),
    )

    # Load mask image if provided
    mask = None
    if args.mask_image is not None and args.mask_image.exists():
        logger.info("Loading mask image: %s", args.mask_image)
        mask = load_image(args.mask_image)
        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Run complexity pipeline
    logger.info("Computing %s complexity map...", config.metric)
    result = run_complexity_pipeline(image, config, mask)

    # Save complexity outputs
    logger.info("Saving complexity outputs...")
    parent_dir = output_dir.parent
    save_complexity_outputs(result, parent_dir, image)

    # Print summary
    print("\n" + "=" * 60)
    print("COMPLEXITY PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Input image: {args.image_path}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Output directory: {parent_dir / 'complexity'}")
    print("\nFiles created:")
    print(f"  - {args.metric}_energy.png (raw complexity heatmap)")
    print(f"  - {args.metric}_energy_raw.npy (raw complexity array)")
    print("  - complexity.png (normalized complexity heatmap)")
    print("  - complexity_raw.npy (normalized complexity array)")
    print("  - contact_sheet.png (all visualizations)")
    print("\nConfiguration used:")
    print(f"  - Metric: {args.metric}")
    if args.metric in ["gradient", "laplacian"]:
        print(f"  - Sigma: {args.complexity_sigma}")
    elif args.metric == "multiscale_gradient":
        print(f"  - Scales: {args.scales}")
        print(f"  - Scale weights: {args.scale_weights}")
    print(f"  - Normalize percentile: {args.normalize_percentile}")
    if args.mask_image:
        print(f"  - Mask image: {args.mask_image}")
    print("\nComplexity statistics:")
    print(f"  - Min complexity: {result.complexity.min():.3f}")
    print(f"  - Max complexity: {result.complexity.max():.3f}")
    print(f"  - Mean complexity: {result.complexity.mean():.3f}")
    print("=" * 60)

    return 0


def handle_all(
    args: argparse.Namespace, image: Path, image_name: str, logger: logging.Logger
) -> int:
    """Handle the all subcommand (runs all pipelines with shared landmarks).

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
    logger.info("Running all pipelines with shared landmarks...")

    # Create base output directory
    base_output_dir = Path(args.output_dir) / image_name
    base_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Base output directory: %s", base_output_dir)

    # Build all configurations
    remap_config = build_remap_config(args)

    # Feature config
    features_config = PipelineConfig(
        remap=remap_config,
        weights={"eyes": args.eye_weight, "mouth": args.mouth_weight},
        output_dir=str(base_output_dir / "features"),
    )

    # Contour config
    contour_config = ContourConfig(
        contour_method=args.contour_method,
        remap=remap_config,
        direction=args.direction,
        band_width=args.band_width,
        contour_thickness=args.contour_thickness,
        epsilon_factor=args.epsilon_factor,
        smooth_contour=not args.no_smooth,
        output_dir=str(base_output_dir / "contour"),
    )

    # Density composition config
    luminance_config = LuminanceConfig(
        clip_limit=args.clip_limit,
        tile_size=args.tile_size,
    )
    compose_config = ComposeConfig(
        luminance=luminance_config,
        feature_weight=args.feature_weight,
        contour_weight=args.contour_weight,
        tonal_blend_mode=args.tonal_blend,
        gamma=args.gamma,
    )

    # Flow config
    etf_config = ETFConfig(
        blur_sigma=args.blur_sigma,
        structure_sigma=args.structure_sigma,
        refine_sigma=args.refine_sigma,
        refine_iterations=args.refine_iterations,
        sobel_ksize=3,  # default (not exposed in all subcommand to avoid clutter)
    )
    flow_config = FlowConfig(
        etf=etf_config,
        contour_smooth_sigma=1.0,  # default (not exposed in all subcommand)
        blend_mode="coherence",  # default
        coherence_power=args.coherence_power,
        fallback_threshold=args.fallback_threshold,
    )

    # LIC config
    lic_config = LICConfig(
        length=args.lic_length,
        step=1.0,  # default (not exposed in all subcommand)
        seed=args.lic_seed,
        use_bilinear=True,  # default
    )

    # Build complexity and speed configs if metric is specified
    complexity_config = build_complexity_config(args)
    speed_config = build_flow_speed_config(args) if complexity_config else None

    # Run all pipelines with shared landmarks
    logger.info("Running complete pipeline with shared landmarks...")
    result = run_all_pipelines(
        image,
        features_config,
        contour_config,
        compose_config,
        flow_config,
        lic_config,
        complexity_config,
        speed_config,
    )

    # Save all outputs
    logger.info("Saving all outputs...")
    save_all_outputs(result, base_output_dir, image)

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("ALL PIPELINES COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Input image: {args.image_path}")
    print(f"Image dimensions: {width}x{height}")
    print(f"Landmarks detected: {len(result.feature_result.landmarks.landmarks)}")
    print(f"Face detection confidence: {result.feature_result.landmarks.confidence:.2%}")
    print(f"Base output directory: {base_output_dir}")
    print("\n--- Stage 1: Feature Pipeline ---")
    print(f"  Output directory: {base_output_dir / 'features'}")
    print("  Files created: landmarks, masks, distance fields, influence maps")
    print(f"  Eye weight: {args.eye_weight}, Mouth weight: {args.mouth_weight}")
    print("\n--- Stage 2: Contour Pipeline ---")
    print(f"  Output directory: {base_output_dir / 'contour'}")
    print("  Files created: contour overlay, masks, signed/directional distance")
    print(f"  Method: {args.contour_method}")
    print(f"  Direction: {args.direction}, Thickness: {args.contour_thickness}")
    print("\n--- Stage 3: Density Composition ---")
    print(f"  Output directory: {base_output_dir / 'density'}")
    print("  Files created: luminance, CLAHE, tonal target, density maps")
    print(f"  CLAHE: clip={args.clip_limit}, tile={args.tile_size}")
    print(f"  Blend: {args.tonal_blend}, Gamma: {args.gamma}")
    print(f"  Weights: features={args.feature_weight}, contour={args.contour_weight}")
    print("\n--- Stage 4: Flow Fields ---")
    print(f"  Output directory: {base_output_dir / 'flow'}")
    print("  Files created: ETF, flow fields, LIC visualization")
    print(f"  ETF: blur={args.blur_sigma}, structure={args.structure_sigma}")
    print(f"  Refinement: sigma={args.refine_sigma}, iter={args.refine_iterations}")
    print(f"  Blending: coherence^{args.coherence_power}, fallback={args.fallback_threshold}")
    print(f"  LIC: length={args.lic_length}, seed={args.lic_seed}")
    if complexity_config is not None:
        print("\n--- Stage 5: Complexity Map ---")
        print(f"  Output directory: {base_output_dir / 'complexity'}")
        print("  Files created: complexity maps, flow speed modulation")
        print(f"  Metric: {args.metric}")
        if args.metric in ["gradient", "laplacian"]:
            print(f"  Sigma: {args.complexity_sigma}")
        elif args.metric == "multiscale_gradient":
            print(f"  Scales: {args.scales}")
        print(f"  Speed range: {args.speed_min:.1f} to {args.speed_max:.1f}")
    print("\nShared configuration:")
    print(f"  - Remap curve: {args.curve}")
    if args.curve == "linear":
        print(f"  - Radius: {args.radius}")
    elif args.curve == "gaussian":
        print(f"  - Sigma: {args.sigma}")
    elif args.curve == "exponential":
        print(f"  - Tau: {args.tau}")
    print(f"  - Clamp distance: {args.clamp_distance}")
    print("\n✓ Landmarks detected once and shared across all pipelines")
    stages = 5 if complexity_config else 4
    print(f"✓ All {stages} pipeline stages completed successfully")

    # Export bundle if requested
    if getattr(args, "export", False):
        from portrait_map_lab.export import build_export_bundle, save_export_bundle

        logger.info("Building export bundle...")
        bundle = build_export_bundle(result, image_name, png_source_dir=base_output_dir)
        export_dir = save_export_bundle(bundle, base_output_dir)

        print("\n--- Export Bundle ---")
        print(f"  Export directory: {export_dir}")
        print(f"  Manifest: {export_dir / 'manifest.json'}")
        print(f"  Binary maps: {len(bundle.binary_maps)} files")
        for entry in bundle.manifest.maps:
            size_kb = len(bundle.binary_maps[entry.key]) / 1024
            print(f"    {entry.filename} ({size_kb:.0f} KB) — {entry.description}")
        print(f"  Preview PNGs: {len(bundle.png_files)} files")
        print("✓ Export bundle created for TypeScript consumption")

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

        # Extract image name for output subdirectory, prefixed with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_name = f"{timestamp}_{args.image_path.stem}"

        # Route to appropriate handler based on subcommand
        if args.pipeline == "features":
            return handle_features(args, image, image_name, logger)
        elif args.pipeline == "contour":
            return handle_contour(args, image, image_name, logger)
        elif args.pipeline == "density":
            return handle_density(args, image, image_name, logger)
        elif args.pipeline == "flow":
            return handle_flow(args, image, image_name, logger)
        elif args.pipeline == "complexity":
            return handle_complexity(args, image, image_name, logger)
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
