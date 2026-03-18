"""Tests for the pipeline module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from portrait_map_lab.models import (
    ComposeConfig,
    ComposedResult,
    ContourConfig,
    ContourResult,
    DensityResult,
    ETFConfig,
    FlowConfig,
    FlowResult,
    LICConfig,
    LuminanceConfig,
    PipelineConfig,
    PipelineResult,
    RemapConfig,
)
from portrait_map_lab.pipelines import (
    run_all_pipelines,
    run_contour_pipeline,
    run_density_pipeline,
    run_feature_distance_pipeline,
    run_flow_pipeline,
    save_all_outputs,
    save_density_outputs,
    save_flow_outputs,
    save_pipeline_outputs,
)


def test_pipeline_returns_complete_result(test_image):
    """Test that pipeline returns a PipelineResult with all fields populated."""
    result = run_feature_distance_pipeline(test_image)

    assert isinstance(result, PipelineResult)
    assert result.landmarks is not None
    assert result.masks is not None and len(result.masks) > 0
    assert result.distance_fields is not None and len(result.distance_fields) > 0
    assert result.influence_maps is not None and len(result.influence_maps) > 0
    assert result.combined is not None


def test_pipeline_shapes_match(test_image):
    """Test that all output arrays have correct shapes."""
    h, w = test_image.shape[:2]
    result = run_feature_distance_pipeline(test_image)

    # Check landmark shape
    assert result.landmarks.image_shape == (h, w)
    assert result.landmarks.landmarks.shape[0] > 0  # Has landmarks
    assert result.landmarks.landmarks.shape[1] == 2  # 2D coordinates

    # Check mask shapes
    for mask in result.masks.values():
        assert mask.shape == (h, w)
        assert mask.dtype == np.uint8

    # Check distance field shapes
    for field in result.distance_fields.values():
        assert field.shape == (h, w)
        assert field.dtype == np.float64

    # Check influence map shapes
    for influence in result.influence_maps.values():
        assert influence.shape == (h, w)
        assert influence.dtype == np.float64

    # Check combined shape
    assert result.combined.shape == (h, w)
    assert result.combined.dtype == np.float64


def test_combined_map_normalized(test_image):
    """Test that the combined map is normalized to [0.0, 1.0]."""
    result = run_feature_distance_pipeline(test_image)

    assert result.combined.min() >= 0.0
    assert result.combined.max() <= 1.0


def test_pipeline_with_default_config(test_image):
    """Test that pipeline runs successfully with default config."""
    # Should not raise any exceptions
    result = run_feature_distance_pipeline(test_image)
    assert result is not None


def test_pipeline_with_custom_config(test_image):
    """Test pipeline with custom configuration."""
    config = PipelineConfig(
        remap=RemapConfig(curve="linear", radius=100.0),
        weights={"eyes": 0.7, "mouth": 0.3},
        output_dir="custom_output",
    )

    result = run_feature_distance_pipeline(test_image, config)
    assert result is not None

    # Verify custom weights were used (influence maps should exist for weighted keys)
    assert "eyes" in result.influence_maps
    assert "mouth" in result.influence_maps


def test_pipeline_creates_expected_masks(test_image):
    """Test that pipeline creates expected mask keys."""
    result = run_feature_distance_pipeline(test_image)

    # Should have at least some of these masks
    possible_masks = {"left_eye", "right_eye", "mouth", "combined_eyes"}
    assert len(set(result.masks.keys()) & possible_masks) > 0

    # If both eye masks exist, combined_eyes should exist
    if "left_eye" in result.masks and "right_eye" in result.masks:
        assert "combined_eyes" in result.masks


def test_pipeline_creates_distance_fields(test_image):
    """Test that pipeline creates distance fields for eyes and mouth."""
    result = run_feature_distance_pipeline(test_image)

    # Should have distance fields for eyes and mouth
    assert "eyes" in result.distance_fields
    assert "mouth" in result.distance_fields

    # Distance fields should have reasonable values
    for field in result.distance_fields.values():
        assert np.isfinite(field).all()  # No NaN or inf
        assert field.min() >= 0.0  # Distance is non-negative


def test_pipeline_creates_influence_maps(test_image):
    """Test that pipeline creates influence maps with correct properties."""
    result = run_feature_distance_pipeline(test_image)

    # Should have influence maps for eyes and mouth
    assert "eyes" in result.influence_maps
    assert "mouth" in result.influence_maps

    # Influence maps should be normalized
    for influence in result.influence_maps.values():
        assert influence.min() >= 0.0
        assert influence.max() <= 1.0
        assert np.isfinite(influence).all()


def test_save_pipeline_outputs_creates_files(test_image):
    """Test that save_pipeline_outputs creates all expected files."""
    result = run_feature_distance_pipeline(test_image)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        save_pipeline_outputs(result, test_image, output_dir)

        # Check that output directory was created
        assert output_dir.exists()

        # Check for expected files
        expected_files = [
            "landmarks.png",
            "distance_eyes_raw.npy",
            "distance_eyes_heatmap.png",
            "distance_mouth_raw.npy",
            "distance_mouth_heatmap.png",
            "influence_eyes.png",
            "influence_mouth.png",
            "combined_importance.png",
            "contact_sheet.png",
        ]

        for filename in expected_files:
            filepath = output_dir / filename
            assert filepath.exists(), f"Expected file {filename} not found"

        # Check for mask files (at least some should exist)
        mask_files = list(output_dir.glob("mask_*.png"))
        assert len(mask_files) > 0, "No mask files found"

        # Verify .npy files can be loaded
        eyes_dist = np.load(output_dir / "distance_eyes_raw.npy")
        assert eyes_dist.shape == test_image.shape[:2]
        mouth_dist = np.load(output_dir / "distance_mouth_raw.npy")
        assert mouth_dist.shape == test_image.shape[:2]


def test_save_pipeline_outputs_handles_missing_masks(test_image):
    """Test that save function handles cases where some masks might be missing."""
    # Run pipeline normally
    result = run_feature_distance_pipeline(test_image)

    # Remove a mask to simulate missing region
    if "left_eye" in result.masks:
        del result.masks["left_eye"]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        # Should not raise exception even with missing mask
        save_pipeline_outputs(result, test_image, output_dir)
        assert output_dir.exists()
        assert (output_dir / "contact_sheet.png").exists()


def test_pipeline_confidence_score(test_image):
    """Test that landmark confidence is reasonable."""
    result = run_feature_distance_pipeline(test_image)
    assert 0.0 <= result.landmarks.confidence <= 1.0


def test_pipeline_with_blank_image():
    """Test that pipeline handles blank image appropriately."""
    # Create a blank white image
    blank_image = np.ones((480, 640, 3), dtype=np.uint8) * 255

    with pytest.raises(ValueError, match="No face detected"):
        run_feature_distance_pipeline(blank_image)


# Density Pipeline Tests


def test_density_pipeline_complete_result(test_image):
    """Test that density pipeline returns a DensityResult with all fields populated."""
    # First run feature and contour pipelines to get inputs
    feature_result = run_feature_distance_pipeline(test_image)
    contour_result = run_contour_pipeline(test_image)

    # Run density pipeline
    result = run_density_pipeline(test_image, feature_result, contour_result)

    # Check result type and fields
    assert isinstance(result, DensityResult)
    assert result.luminance is not None
    assert result.clahe_luminance is not None
    assert result.tonal_target is not None
    assert result.importance is not None
    assert result.density_target is not None


def test_density_pipeline_shapes(test_image):
    """Test that all density pipeline arrays have correct shapes."""
    h, w = test_image.shape[:2]

    # Run prerequisite pipelines
    feature_result = run_feature_distance_pipeline(test_image)
    contour_result = run_contour_pipeline(test_image)

    # Run density pipeline
    result = run_density_pipeline(test_image, feature_result, contour_result)

    # Check all array shapes match image dimensions
    assert result.luminance.shape == (h, w)
    assert result.clahe_luminance.shape == (h, w)
    assert result.tonal_target.shape == (h, w)
    assert result.importance.shape == (h, w)
    assert result.density_target.shape == (h, w)

    # Check all arrays are float64
    assert result.luminance.dtype == np.float64
    assert result.clahe_luminance.dtype == np.float64
    assert result.tonal_target.dtype == np.float64
    assert result.importance.dtype == np.float64
    assert result.density_target.dtype == np.float64


def test_density_pipeline_value_ranges(test_image):
    """Test that all density pipeline outputs are in [0, 1] range."""
    # Run prerequisite pipelines
    feature_result = run_feature_distance_pipeline(test_image)
    contour_result = run_contour_pipeline(test_image)

    # Run density pipeline
    result = run_density_pipeline(test_image, feature_result, contour_result)

    # Check all values are in [0, 1]
    assert result.luminance.min() >= 0.0 and result.luminance.max() <= 1.0
    assert result.clahe_luminance.min() >= 0.0 and result.clahe_luminance.max() <= 1.0
    assert result.tonal_target.min() >= 0.0 and result.tonal_target.max() <= 1.0
    assert result.importance.min() >= 0.0 and result.importance.max() <= 1.0
    assert result.density_target.min() >= 0.0 and result.density_target.max() <= 1.0

    # Check no NaN or inf values
    assert np.isfinite(result.luminance).all()
    assert np.isfinite(result.clahe_luminance).all()
    assert np.isfinite(result.tonal_target).all()
    assert np.isfinite(result.importance).all()
    assert np.isfinite(result.density_target).all()


def test_save_density_outputs_creates_files(test_image):
    """Test that save_density_outputs creates all expected files."""
    # Run prerequisite pipelines
    feature_result = run_feature_distance_pipeline(test_image)
    contour_result = run_contour_pipeline(test_image)

    # Run density pipeline
    result = run_density_pipeline(test_image, feature_result, contour_result)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        save_density_outputs(result, output_dir, image=test_image)

        # Check that density subdirectory was created
        density_dir = output_dir / "density"
        assert density_dir.exists()

        # Check for expected files
        expected_files = [
            "luminance.png",
            "clahe_luminance.png",
            "tonal_target.png",
            "importance.png",
            "density_target.png",
            "density_target_raw.npy",
            "contact_sheet.png",
        ]

        for filename in expected_files:
            filepath = density_dir / filename
            assert filepath.exists(), f"Expected file {filename} not found"

        # Verify .npy file can be loaded and has correct shape
        density_array = np.load(density_dir / "density_target_raw.npy")
        assert density_array.shape == test_image.shape[:2]
        assert density_array.dtype == np.float64


def test_density_pipeline_with_custom_config(test_image):
    """Test density pipeline with custom configuration."""
    # Run prerequisite pipelines
    feature_result = run_feature_distance_pipeline(test_image)
    contour_result = run_contour_pipeline(test_image)

    # Create custom config
    config = ComposeConfig(
        luminance=LuminanceConfig(clip_limit=3.0, tile_size=16),
        feature_weight=0.8,
        contour_weight=0.2,
        tonal_blend_mode="screen",
        gamma=0.8,
    )

    # Run density pipeline with custom config
    result = run_density_pipeline(test_image, feature_result, contour_result, config)

    # Should complete successfully
    assert result is not None
    assert isinstance(result, DensityResult)

    # Result should still be valid
    assert result.density_target.min() >= 0.0
    assert result.density_target.max() <= 1.0


def test_density_pipeline_blend_modes(test_image):
    """Test that different blend modes produce different results."""
    # Run prerequisite pipelines
    feature_result = run_feature_distance_pipeline(test_image)
    contour_result = run_contour_pipeline(test_image)

    # Test different blend modes
    modes = ["multiply", "screen", "max", "weighted"]
    results = {}

    for mode in modes:
        config = ComposeConfig(tonal_blend_mode=mode)
        result = run_density_pipeline(test_image, feature_result, contour_result, config)
        results[mode] = result.density_target

    # Results should be different for different modes
    # (but not necessarily all different from each other)
    unique_results = []
    for mode, array in results.items():
        is_unique = True
        for existing in unique_results:
            if np.allclose(array, existing):
                is_unique = False
                break
        if is_unique:
            unique_results.append(array)

    # At least 2 different results expected
    assert len(unique_results) >= 2, "Blend modes should produce different results"


def test_save_density_outputs_without_image(test_image):
    """Test save_density_outputs works without providing original image."""
    # Run prerequisite pipelines
    feature_result = run_feature_distance_pipeline(test_image)
    contour_result = run_contour_pipeline(test_image)

    # Run density pipeline
    result = run_density_pipeline(test_image, feature_result, contour_result)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        # Save without providing image
        save_density_outputs(result, output_dir, image=None)

        # Should still create outputs
        density_dir = output_dir / "density"
        assert density_dir.exists()
        assert (density_dir / "contact_sheet.png").exists()


def test_flow_pipeline_returns_complete_result(test_image):
    """Test that flow pipeline returns a FlowResult with all fields populated."""
    # First need a contour result
    contour_result = run_contour_pipeline(test_image)

    # Run flow pipeline
    result = run_flow_pipeline(test_image, contour_result)

    assert isinstance(result, FlowResult)
    assert result.etf is not None
    assert result.contour_flow_x is not None
    assert result.contour_flow_y is not None
    assert result.blend_weight is not None
    assert result.flow_x is not None
    assert result.flow_y is not None


def test_flow_pipeline_array_shapes(test_image):
    """Test that all flow arrays have matching shapes."""
    h, w = test_image.shape[:2]

    # Get contour result first
    contour_result = run_contour_pipeline(test_image)

    # Run flow pipeline
    result = run_flow_pipeline(test_image, contour_result)

    # All arrays should match image dimensions
    assert result.etf.tangent_x.shape == (h, w)
    assert result.etf.tangent_y.shape == (h, w)
    assert result.etf.coherence.shape == (h, w)
    assert result.contour_flow_x.shape == (h, w)
    assert result.contour_flow_y.shape == (h, w)
    assert result.blend_weight.shape == (h, w)
    assert result.flow_x.shape == (h, w)
    assert result.flow_y.shape == (h, w)


def test_flow_pipeline_unit_vectors(test_image):
    """Test that flow vectors are unit length."""
    contour_result = run_contour_pipeline(test_image)
    result = run_flow_pipeline(test_image, contour_result)

    # Check ETF vectors
    etf_mag = np.sqrt(result.etf.tangent_x**2 + result.etf.tangent_y**2)
    assert np.allclose(etf_mag, 1.0, atol=1e-6)

    # Check contour flow vectors — zero-gradient pixels (from SDF smoothing)
    # produce zero-magnitude vectors, which is correct; only check non-degenerate pixels
    contour_mag = np.sqrt(result.contour_flow_x**2 + result.contour_flow_y**2)
    nonzero = contour_mag > 0.5
    assert nonzero.any(), "Expected some non-zero contour flow vectors"
    assert np.allclose(contour_mag[nonzero], 1.0, atol=1e-6)

    # Check final flow vectors
    flow_mag = np.sqrt(result.flow_x**2 + result.flow_y**2)
    assert np.allclose(flow_mag, 1.0, atol=1e-6)


def test_flow_pipeline_with_config(test_image):
    """Test flow pipeline with custom configuration."""
    contour_result = run_contour_pipeline(test_image)

    config = FlowConfig(
        contour_smooth_sigma=2.0,
        coherence_power=3.0,
        fallback_threshold=0.2,
    )

    result = run_flow_pipeline(test_image, contour_result, config)

    assert isinstance(result, FlowResult)
    # Result should be complete
    assert result.flow_x is not None
    assert result.flow_y is not None


def test_save_flow_outputs(test_image):
    """Test saving flow pipeline outputs."""
    contour_result = run_contour_pipeline(test_image)
    result = run_flow_pipeline(test_image, contour_result)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        save_flow_outputs(result, output_dir, image=test_image)

        # Check that files were created
        flow_dir = output_dir / "flow"
        assert flow_dir.exists()
        assert (flow_dir / "etf_coherence.png").exists()
        assert (flow_dir / "blend_weight.png").exists()
        assert (flow_dir / "flow_x_raw.npy").exists()
        assert (flow_dir / "flow_y_raw.npy").exists()
        assert (flow_dir / "contact_sheet.png").exists()

        # Load and check saved arrays
        flow_x = np.load(flow_dir / "flow_x_raw.npy")
        flow_y = np.load(flow_dir / "flow_y_raw.npy")

        assert flow_x.shape == result.flow_x.shape
        assert flow_y.shape == result.flow_y.shape
        assert np.allclose(flow_x, result.flow_x)
        assert np.allclose(flow_y, result.flow_y)


def test_save_flow_outputs_without_image(test_image):
    """Test saving flow outputs without providing original image."""
    contour_result = run_contour_pipeline(test_image)
    result = run_flow_pipeline(test_image, contour_result)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        # Save without providing image
        save_flow_outputs(result, output_dir, image=None)

        # Should still create outputs
        flow_dir = output_dir / "flow"
        assert flow_dir.exists()


def test_run_all_pipelines(test_image):
    """Test running all pipelines with shared landmarks."""
    # Run the complete pipeline
    result = run_all_pipelines(test_image)

    # Check that result has the expected structure
    assert result is not None
    assert result.feature_result is not None
    assert result.contour_result is not None
    assert result.density_result is not None
    assert result.flow_result is not None
    assert result.lic_image is not None

    # Check that all results have correct types
    assert isinstance(result.feature_result, PipelineResult)
    assert isinstance(result.contour_result, ContourResult)
    assert isinstance(result.density_result, DensityResult)
    assert isinstance(result.flow_result, FlowResult)
    assert isinstance(result, ComposedResult)

    # Check that LIC image has correct properties
    assert isinstance(result.lic_image, np.ndarray)
    assert result.lic_image.dtype == np.float64
    assert result.lic_image.shape == test_image.shape[:2]
    assert result.lic_image.min() >= 0.0
    assert result.lic_image.max() <= 1.0


def test_all_pipeline_shared_landmarks(test_image):
    """Test that landmarks are shared across all pipeline results."""
    result = run_all_pipelines(test_image)

    # Extract landmarks from each result
    feature_landmarks = result.feature_result.landmarks
    contour_landmarks = result.contour_result.landmarks

    # Check that landmarks are the same object (shared reference)
    assert feature_landmarks is contour_landmarks

    # Check that landmarks have the same values
    assert feature_landmarks.confidence == contour_landmarks.confidence
    assert np.array_equal(feature_landmarks.landmarks, contour_landmarks.landmarks)
    assert feature_landmarks.image_shape == contour_landmarks.image_shape


def test_all_pipeline_shapes(test_image):
    """Test that all arrays in the composed result have matching shapes."""
    h, w = test_image.shape[:2]
    result = run_all_pipelines(test_image)

    # Check feature result shapes
    assert result.feature_result.combined.shape == (h, w)
    for mask in result.feature_result.masks.values():
        assert mask.shape == (h, w)

    # Check contour result shapes
    assert result.contour_result.signed_distance.shape == (h, w)
    assert result.contour_result.influence_map.shape == (h, w)

    # Check density result shapes
    assert result.density_result.luminance.shape == (h, w)
    assert result.density_result.density_target.shape == (h, w)

    # Check flow result shapes
    assert result.flow_result.flow_x.shape == (h, w)
    assert result.flow_result.flow_y.shape == (h, w)

    # Check LIC image shape
    assert result.lic_image.shape == (h, w)


def test_all_pipeline_ranges(test_image):
    """Test that all normalized values are in expected ranges."""
    result = run_all_pipelines(test_image)

    # Check normalized maps are in [0, 1]
    assert result.feature_result.combined.min() >= 0.0
    assert result.feature_result.combined.max() <= 1.0

    assert result.contour_result.influence_map.min() >= 0.0
    assert result.contour_result.influence_map.max() <= 1.0

    assert result.density_result.luminance.min() >= 0.0
    assert result.density_result.luminance.max() <= 1.0
    assert result.density_result.density_target.min() >= 0.0
    assert result.density_result.density_target.max() <= 1.0

    # Check flow fields are unit vectors
    flow_mag = np.sqrt(result.flow_result.flow_x**2 + result.flow_result.flow_y**2)
    assert np.allclose(flow_mag, 1.0, atol=1e-6)

    # Check LIC is in [0, 1]
    assert result.lic_image.min() >= 0.0
    assert result.lic_image.max() <= 1.0


def test_save_all_outputs(test_image):
    """Test saving all pipeline outputs to subdirectories."""
    result = run_all_pipelines(test_image)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_output"
        save_all_outputs(result, output_dir, test_image)

        # Check that all subdirectories were created
        assert (output_dir / "features").exists()
        assert (output_dir / "contour").exists()
        assert (output_dir / "density").exists()
        assert (output_dir / "flow").exists()

        # Check for key files in each subdirectory
        # Features
        assert (output_dir / "features" / "landmarks.png").exists()
        assert (output_dir / "features" / "combined_importance.png").exists()
        assert (output_dir / "features" / "contact_sheet.png").exists()

        # Contour
        assert (output_dir / "contour" / "contour_overlay.png").exists()
        assert (output_dir / "contour" / "signed_distance_raw.npy").exists()
        assert (output_dir / "contour" / "contact_sheet.png").exists()

        # Density
        assert (output_dir / "density" / "luminance.png").exists()
        assert (output_dir / "density" / "density_target.png").exists()
        assert (output_dir / "density" / "density_target_raw.npy").exists()
        assert (output_dir / "density" / "contact_sheet.png").exists()

        # Flow
        assert (output_dir / "flow" / "etf_coherence.png").exists()
        assert (output_dir / "flow" / "flow_x_raw.npy").exists()
        assert (output_dir / "flow" / "flow_y_raw.npy").exists()
        assert (output_dir / "flow" / "flow_lic.png").exists()
        assert (output_dir / "flow" / "contact_sheet.png").exists()


def test_all_pipeline_with_custom_configs(test_image):
    """Test running all pipelines with custom configurations."""
    # Create custom configs
    feature_config = PipelineConfig(weights={"eyes": 0.7, "mouth": 0.3})
    contour_config = ContourConfig(direction="outward")
    compose_config = ComposeConfig(
        luminance=LuminanceConfig(clip_limit=3.0), tonal_blend_mode="screen", gamma=1.5
    )
    flow_config = FlowConfig(etf=ETFConfig(refine_iterations=3), coherence_power=3.0)
    lic_config = LICConfig(length=50, seed=123)

    # Run with custom configs
    result = run_all_pipelines(
        test_image,
        feature_config=feature_config,
        contour_config=contour_config,
        compose_config=compose_config,
        flow_config=flow_config,
        lic_config=lic_config,
    )

    # Check that result is valid
    assert result is not None
    assert result.feature_result is not None
    assert result.density_result is not None
    assert result.flow_result is not None
    assert result.lic_image is not None
