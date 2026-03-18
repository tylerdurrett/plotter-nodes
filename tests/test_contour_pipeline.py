"""Integration tests for the face contour pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from portrait_map_lab.models import ContourConfig, RemapConfig
from portrait_map_lab.pipelines import run_contour_pipeline, save_contour_outputs


class TestContourPipeline:
    """Test the complete contour pipeline."""

    def test_run_contour_pipeline_default_config(self, test_image):
        """Test running the pipeline with default configuration."""
        # Run pipeline with defaults
        result = run_contour_pipeline(test_image)

        # Verify result structure
        assert result.landmarks is not None
        assert result.contour_polygon.ndim == 2
        assert result.contour_polygon.shape[1] == 2
        assert result.contour_polygon.shape[0] >= 3  # convex hull
        assert result.contour_mask.shape == result.landmarks.image_shape
        assert result.filled_mask.shape == result.landmarks.image_shape
        assert result.signed_distance.shape == result.landmarks.image_shape
        assert result.directional_distance.shape == result.landmarks.image_shape
        assert result.influence_map.shape == result.landmarks.image_shape

        # Verify data types
        assert result.contour_polygon.dtype == np.float64
        assert result.contour_mask.dtype == np.uint8
        assert result.filled_mask.dtype == np.uint8
        assert result.signed_distance.dtype == np.float64
        assert result.directional_distance.dtype == np.float64
        assert result.influence_map.dtype == np.float64

    def test_run_contour_pipeline_custom_config(self, test_image):
        """Test running the pipeline with custom configuration."""
        # Create custom config
        config = ContourConfig(
            remap=RemapConfig(curve="exponential", tau=100.0),
            direction="outward",
            contour_thickness=3,
        )

        # Run pipeline
        result = run_contour_pipeline(test_image, config)

        # Verify result exists and has correct shape
        assert result.contour_polygon.ndim == 2
        assert result.contour_polygon.shape[1] == 2
        assert result.influence_map.shape == result.landmarks.image_shape

    def test_all_direction_modes(self, test_image):
        """Test all direction modes produce valid output."""
        modes = ["inward", "outward", "both", "band"]

        for mode in modes:
            if mode == "band":
                config = ContourConfig(direction=mode, band_width=50.0)
            else:
                config = ContourConfig(direction=mode)

            result = run_contour_pipeline(test_image, config)

            # Verify directional distance is valid
            assert result.directional_distance.dtype == np.float64
            assert not np.isnan(result.directional_distance).any()
            assert result.directional_distance.min() >= 0  # Should be unsigned after prepare

            # Verify influence map is normalized
            assert result.influence_map.min() >= 0.0
            assert result.influence_map.max() <= 1.0

    def test_signed_distance_has_both_signs(self, test_image):
        """Test that signed distance field has both negative and positive values."""
        result = run_contour_pipeline(test_image)

        # Signed distance should have both negative (inside) and positive (outside) values
        assert result.signed_distance.min() < 0
        assert result.signed_distance.max() > 0

        # Values on the contour should be near zero
        contour_pixels = result.contour_mask > 0
        contour_distances = result.signed_distance[contour_pixels]
        assert np.allclose(contour_distances, 0.0, atol=1e-10)

    def test_influence_map_normalized(self, test_image):
        """Test that influence map values are properly normalized."""
        result = run_contour_pipeline(test_image)

        # Influence should be in [0, 1] range
        assert result.influence_map.min() >= 0.0
        assert result.influence_map.max() <= 1.0

        # Should have some variation (not all same value)
        assert not np.allclose(result.influence_map, result.influence_map.mean())

    def test_save_contour_outputs_creates_files(self, test_image):
        """Test that save_contour_outputs creates all expected files."""
        # Run pipeline
        result = run_contour_pipeline(test_image)

        # Save outputs to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_output"
            save_contour_outputs(result, test_image, output_dir)

            # Check all expected files exist
            expected_files = [
                "contour_overlay.png",
                "contour_mask.png",
                "filled_mask.png",
                "signed_distance_raw.npy",
                "signed_distance_heatmap.png",
                "directional_distance_raw.npy",
                "directional_distance_heatmap.png",
                "contour_influence.png",
                "contact_sheet.png",
            ]

            for filename in expected_files:
                filepath = output_dir / filename
                assert filepath.exists(), f"Expected file {filename} not found"

            # Verify .npy files are loadable and have correct shape
            signed_loaded = np.load(output_dir / "signed_distance_raw.npy")
            assert signed_loaded.shape == result.signed_distance.shape

            directional_loaded = np.load(output_dir / "directional_distance_raw.npy")
            assert directional_loaded.shape == result.directional_distance.shape

    def test_contour_thickness_affects_mask(self, test_image):
        """Test that contour thickness parameter affects the mask."""
        # Run with thin contour
        config_thin = ContourConfig(contour_thickness=1)
        result_thin = run_contour_pipeline(test_image, config_thin)

        # Run with thick contour
        config_thick = ContourConfig(contour_thickness=5)
        result_thick = run_contour_pipeline(test_image, config_thick)

        # Thick contour should have more nonzero pixels
        thin_pixels = np.count_nonzero(result_thin.contour_mask)
        thick_pixels = np.count_nonzero(result_thick.contour_mask)
        assert thick_pixels > thin_pixels

    def test_band_mode_requires_band_width(self, test_image):
        """Test that band mode raises error without band_width."""
        config = ContourConfig(direction="band", band_width=None)

        with pytest.raises(ValueError, match="band_width is required"):
            run_contour_pipeline(test_image, config)

    def test_pipeline_with_different_remap_curves(self, test_image):
        """Test pipeline with different remap curve types."""
        curves = ["linear", "gaussian", "exponential"]

        for curve in curves:
            config = ContourConfig(remap=RemapConfig(curve=curve))
            result = run_contour_pipeline(test_image, config)

            # Verify influence map is valid for each curve type
            assert result.influence_map.min() >= 0.0
            assert result.influence_map.max() <= 1.0
            assert not np.isnan(result.influence_map).any()
