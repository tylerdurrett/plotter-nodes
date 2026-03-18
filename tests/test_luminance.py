"""Tests for portrait_map_lab.luminance module."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_map_lab.luminance import apply_clahe, compute_tonal_target, extract_luminance
from portrait_map_lab.models import LuminanceConfig


class TestExtractLuminance:
    def test_bgr_input(self):
        """Test luminance extraction from BGR image."""
        # Create a BGR image with known colors
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[:50, :, 0] = 255  # Blue in top half
        bgr_image[50:, :, 1] = 128  # Green in bottom half

        luminance = extract_luminance(bgr_image)

        assert luminance.shape == (100, 100)
        assert luminance.dtype == np.float64
        assert 0.0 <= luminance.min() <= luminance.max() <= 1.0

    def test_grayscale_input(self):
        """Test that grayscale input passes through correctly."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        luminance = extract_luminance(gray_image)

        assert luminance.shape == gray_image.shape
        assert luminance.dtype == np.float64
        assert np.allclose(luminance, gray_image / 255.0)

    def test_output_range(self):
        """Test that output is always in [0, 1]."""
        # Test with black image
        black_image = np.zeros((50, 50, 3), dtype=np.uint8)
        black_luminance = extract_luminance(black_image)
        assert black_luminance.min() == 0.0
        assert black_luminance.max() == 0.0

        # Test with white image
        white_image = np.full((50, 50, 3), 255, dtype=np.uint8)
        white_luminance = extract_luminance(white_image)
        assert white_luminance.min() == 1.0
        assert white_luminance.max() == 1.0

    def test_invalid_input(self):
        """Test that invalid input raises ValueError."""
        # 1D array
        with pytest.raises(ValueError, match="Expected 2D or 3D array"):
            extract_luminance(np.array([1, 2, 3]))

        # 4D array
        with pytest.raises(ValueError, match="Expected 2D or 3D array"):
            extract_luminance(np.zeros((10, 10, 3, 2)))


class TestApplyClahe:
    def test_output_range(self):
        """Test that CLAHE output is in [0, 1]."""
        luminance = np.random.rand(100, 100)
        enhanced = apply_clahe(luminance)

        assert enhanced.dtype == np.float64
        assert 0.0 <= enhanced.min() <= enhanced.max() <= 1.0

    def test_uniform_image(self):
        """Test that CLAHE on uniform image produces consistent output."""
        # Uniform gray image will be remapped by CLAHE but should remain uniform
        uniform = np.full((100, 100), 0.5, dtype=np.float64)
        enhanced = apply_clahe(uniform)

        # Check that output is still uniform (all values nearly the same)
        assert enhanced.std() < 0.01  # Very low standard deviation
        # Check that it's still in valid range
        assert 0.0 <= enhanced.min() <= enhanced.max() <= 1.0

    def test_shape_preservation(self):
        """Test that CLAHE preserves image shape."""
        for shape in [(50, 50), (100, 200), (317, 241)]:
            luminance = np.random.rand(*shape)
            enhanced = apply_clahe(luminance)
            assert enhanced.shape == shape

    def test_different_parameters(self):
        """Test CLAHE with different clip_limit and tile_size values."""
        luminance = np.random.rand(100, 100)

        # Test different clip limits
        enhanced1 = apply_clahe(luminance, clip_limit=1.0)
        enhanced2 = apply_clahe(luminance, clip_limit=4.0)
        assert enhanced1.shape == enhanced2.shape == luminance.shape

        # Test different tile sizes
        enhanced3 = apply_clahe(luminance, tile_size=4)
        enhanced4 = apply_clahe(luminance, tile_size=16)
        assert enhanced3.shape == enhanced4.shape == luminance.shape

    def test_contrast_enhancement(self):
        """Test that CLAHE actually enhances contrast."""
        # Create low-contrast image
        low_contrast = np.random.rand(100, 100) * 0.2 + 0.4  # Values in [0.4, 0.6]
        enhanced = apply_clahe(low_contrast, clip_limit=3.0)

        # Enhanced image should have wider value range
        original_range = low_contrast.max() - low_contrast.min()
        enhanced_range = enhanced.max() - enhanced.min()
        assert enhanced_range > original_range * 1.5  # At least 1.5x range increase


class TestComputeTonalTarget:
    def test_output_shapes(self):
        """Test that all outputs have correct shape."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        luminance, clahe_luminance, tonal_target = compute_tonal_target(image)

        assert luminance.shape == (100, 100)
        assert clahe_luminance.shape == (100, 100)
        assert tonal_target.shape == (100, 100)

    def test_output_dtypes(self):
        """Test that all outputs are float64."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        luminance, clahe_luminance, tonal_target = compute_tonal_target(image)

        assert luminance.dtype == np.float64
        assert clahe_luminance.dtype == np.float64
        assert tonal_target.dtype == np.float64

    def test_output_ranges(self):
        """Test that all outputs are in [0, 1]."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        luminance, clahe_luminance, tonal_target = compute_tonal_target(image)

        assert 0.0 <= luminance.min() <= luminance.max() <= 1.0
        assert 0.0 <= clahe_luminance.min() <= clahe_luminance.max() <= 1.0
        assert 0.0 <= tonal_target.min() <= tonal_target.max() <= 1.0

    def test_tonal_inversion(self):
        """Test that tonal target is inverted (bright → low, dark → high)."""
        # Create image with clear bright and dark regions
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :] = 255  # Top half white
        image[50:, :] = 0  # Bottom half black

        luminance, clahe_luminance, tonal_target = compute_tonal_target(image)

        # Bright regions should have low tonal values
        bright_tonal = tonal_target[:50, :].mean()
        # Dark regions should have high tonal values
        dark_tonal = tonal_target[50:, :].mean()

        assert dark_tonal > bright_tonal
        # More specifically, should be approximately inverted
        assert bright_tonal < 0.1  # Near 0 for white
        assert dark_tonal > 0.9  # Near 1 for black

    def test_black_image(self):
        """Test that black image produces tonal target near 1.0."""
        black_image = np.zeros((50, 50, 3), dtype=np.uint8)
        _, _, tonal_target = compute_tonal_target(black_image)

        # Black should produce high density (near 1.0)
        assert tonal_target.mean() > 0.95
        assert tonal_target.min() > 0.95

    def test_white_image(self):
        """Test that white image produces tonal target near 0.0."""
        white_image = np.full((50, 50, 3), 255, dtype=np.uint8)
        _, _, tonal_target = compute_tonal_target(white_image)

        # White should produce low density (near 0.0)
        assert tonal_target.mean() < 0.05
        assert tonal_target.max() < 0.05

    def test_none_config(self):
        """Test that None config uses defaults."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # With None config
        result_none = compute_tonal_target(image, config=None)

        # With default config
        default_config = LuminanceConfig()
        result_default = compute_tonal_target(image, config=default_config)

        # Results should be identical
        for r_none, r_default in zip(result_none, result_default):
            assert np.allclose(r_none, r_default)

    def test_custom_config(self):
        """Test that custom config parameters are used."""
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Test with different configs
        config1 = LuminanceConfig(clip_limit=1.0, tile_size=4)
        config2 = LuminanceConfig(clip_limit=4.0, tile_size=16)

        result1 = compute_tonal_target(image, config=config1)
        result2 = compute_tonal_target(image, config=config2)

        # Luminance should be the same (no CLAHE yet)
        assert np.allclose(result1[0], result2[0])

        # CLAHE results should differ due to different parameters
        # (can't guarantee exact difference, but should not be identical)
        assert not np.allclose(result1[1], result2[1], atol=0.001)

    def test_grayscale_input(self):
        """Test that grayscale input works correctly."""
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        luminance, clahe_luminance, tonal_target = compute_tonal_target(gray_image)

        assert luminance.shape == (100, 100)
        assert clahe_luminance.shape == (100, 100)
        assert tonal_target.shape == (100, 100)
        assert np.allclose(tonal_target, 1.0 - clahe_luminance)
