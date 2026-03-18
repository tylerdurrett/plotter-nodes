"""Tests for the pipeline module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from portrait_map_lab.models import PipelineConfig, PipelineResult, RemapConfig
from portrait_map_lab.pipelines import run_feature_distance_pipeline, save_pipeline_outputs


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
