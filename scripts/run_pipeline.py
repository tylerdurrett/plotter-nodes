#!/usr/bin/env python3
"""Command-line interface for running the portrait map lab pipelines."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from portrait_map_lab.models import (
    ComposeConfig,
    ContourConfig,
    ETFConfig,
    FlowConfig,
    LICConfig,
    LuminanceConfig,
    PipelineConfig,
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
    density_result = run_density_pipeline(
        image, features_result, contour_result, compose_config
    )

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

    # Run flow pipeline (LIC is computed inside run_flow_pipeline)
    logger.info("Running flow field computation...")
    flow_result = run_flow_pipeline(image, contour_result, flow_config)

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
        remap=remap_config,
        direction=args.direction,
        band_width=args.band_width,
        contour_thickness=args.contour_thickness,
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

    # Run all pipelines with shared landmarks
    logger.info("Running complete pipeline with shared landmarks...")
    result = run_all_pipelines(
        image,
        features_config,
        contour_config,
        compose_config,
        flow_config,
        lic_config,
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
    print("✓ All 4 pipeline stages completed successfully")
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
        elif args.pipeline == "density":
            return handle_density(args, image, image_name, logger)
        elif args.pipeline == "flow":
            return handle_flow(args, image, image_name, logger)
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
