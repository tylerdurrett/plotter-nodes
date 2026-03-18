"""Tests for portrait_map_lab.compose module."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_map_lab.compose import build_density_target, compose_maps


class TestComposeMaps:
    """Tests for compose_maps function."""

    def test_multiply_mode(self):
        """Test multiply blend mode."""
        # Test basic multiplication
        map_a = np.array([[1.0, 0.5], [0.0, 0.25]])
        map_b = np.array([[0.5, 1.0], [1.0, 0.5]])

        result = compose_maps(map_a, map_b, mode="multiply")

        expected = np.array([[0.5, 0.5], [0.0, 0.125]])
        assert np.allclose(result, expected)
        assert result.dtype == np.float64

    def test_multiply_identity(self):
        """Test that multiply with 1.0 is identity."""
        map_a = np.random.rand(50, 50)
        map_b = np.ones((50, 50))

        result = compose_maps(map_a, map_b, mode="multiply")

        assert np.allclose(result, map_a)

    def test_multiply_zero(self):
        """Test that multiply with 0.0 gives zero."""
        map_a = np.random.rand(50, 50)
        map_b = np.zeros((50, 50))

        result = compose_maps(map_a, map_b, mode="multiply")

        assert np.allclose(result, 0.0)

    def test_screen_mode(self):
        """Test screen blend mode."""
        # Screen blend: 1 - (1-a)*(1-b)
        map_a = np.array([[0.0, 0.5], [1.0, 0.25]])
        map_b = np.array([[0.0, 0.5], [0.5, 0.5]])

        result = compose_maps(map_a, map_b, mode="screen")

        # Expected calculations:
        # [0,0]: 1 - (1-0)*(1-0) = 1 - 1 = 0
        # [0,1]: 1 - (1-0.5)*(1-0.5) = 1 - 0.25 = 0.75
        # [1,0]: 1 - (1-1)*(1-0.5) = 1 - 0 = 1
        # [1,1]: 1 - (1-0.25)*(1-0.5) = 1 - 0.375 = 0.625
        expected = np.array([[0.0, 0.75], [1.0, 0.625]])
        assert np.allclose(result, expected)

    def test_screen_black(self):
        """Test that screen with black (0) is identity."""
        map_a = np.random.rand(50, 50)
        map_b = np.zeros((50, 50))

        result = compose_maps(map_a, map_b, mode="screen")

        # screen(a, 0) = 1 - (1-a)*(1-0) = 1 - (1-a) = a
        assert np.allclose(result, map_a)

    def test_screen_white(self):
        """Test that screen with white (1) gives white."""
        map_a = np.random.rand(50, 50)
        map_b = np.ones((50, 50))

        result = compose_maps(map_a, map_b, mode="screen")

        # screen(a, 1) = 1 - (1-a)*(1-1) = 1 - 0 = 1
        assert np.allclose(result, 1.0)

    def test_max_mode(self):
        """Test max blend mode."""
        map_a = np.array([[0.2, 0.8], [0.5, 0.3]])
        map_b = np.array([[0.7, 0.4], [0.5, 0.9]])

        result = compose_maps(map_a, map_b, mode="max")

        expected = np.array([[0.7, 0.8], [0.5, 0.9]])
        assert np.allclose(result, expected)

    def test_max_preserves_higher(self):
        """Test that max always preserves the higher value."""
        map_a = np.random.rand(100, 100)
        map_b = np.random.rand(100, 100)

        result = compose_maps(map_a, map_b, mode="max")

        # Check that result is always the maximum
        assert np.all(result >= map_a)
        assert np.all(result >= map_b)
        assert np.allclose(result, np.maximum(map_a, map_b))

    def test_weighted_mode(self):
        """Test weighted blend mode."""
        map_a = np.array([[0.0, 1.0], [0.4, 0.6]])
        map_b = np.array([[1.0, 0.0], [0.6, 0.4]])

        result = compose_maps(map_a, map_b, mode="weighted")

        # Weighted with equal weights = average
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert np.allclose(result, expected)

    def test_output_range(self):
        """Test that output is always in [0, 1]."""
        # Test with random inputs for all modes
        map_a = np.random.rand(50, 50)
        map_b = np.random.rand(50, 50)

        for mode in ["multiply", "screen", "max", "weighted"]:
            result = compose_maps(map_a, map_b, mode=mode)
            assert result.min() >= 0.0
            assert result.max() <= 1.0
            assert result.dtype == np.float64

    def test_clipping(self):
        """Test that values outside [0, 1] are clipped."""
        # Create maps with values outside valid range
        map_a = np.array([[-0.5, 0.5], [1.5, 0.8]])
        map_b = np.array([[0.5, 1.2], [0.3, -0.1]])

        # Multiply mode might produce values outside range with invalid inputs
        result = compose_maps(map_a, map_b, mode="multiply")

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_shape_mismatch_error(self):
        """Test that mismatched shapes raise ValueError."""
        map_a = np.zeros((50, 50))
        map_b = np.zeros((60, 60))

        with pytest.raises(ValueError, match="Map shapes must match"):
            compose_maps(map_a, map_b)

    def test_invalid_mode_error(self):
        """Test that invalid mode raises ValueError."""
        map_a = np.zeros((50, 50))
        map_b = np.zeros((50, 50))

        with pytest.raises(ValueError, match="Unknown blend mode"):
            compose_maps(map_a, map_b, mode="invalid")

    def test_different_dtypes(self):
        """Test that different input dtypes are handled correctly."""
        map_a = np.random.rand(50, 50).astype(np.float32)
        map_b = np.random.rand(50, 50).astype(np.float16)

        result = compose_maps(map_a, map_b, mode="multiply")

        assert result.dtype == np.float64
        assert result.shape == map_a.shape


class TestBuildDensityTarget:
    """Tests for build_density_target function."""

    def test_basic_composition(self):
        """Test basic density target composition."""
        tonal = np.array([[0.8, 0.6], [0.4, 0.2]])
        importance = np.array([[1.0, 0.5], [0.5, 1.0]])

        result = build_density_target(tonal, importance, mode="multiply", gamma=1.0)

        expected = np.array([[0.8, 0.3], [0.2, 0.2]])
        assert np.allclose(result, expected)
        assert result.dtype == np.float64

    def test_gamma_brightening(self):
        """Test that gamma < 1.0 brightens (increases density)."""
        tonal = np.full((50, 50), 0.5)
        importance = np.full((50, 50), 0.5)

        # Multiply gives 0.25, gamma=0.5 gives 0.25^0.5 = 0.5
        result = build_density_target(tonal, importance, mode="multiply", gamma=0.5)

        assert np.allclose(result, 0.5)

        # Verify brightening
        without_gamma = build_density_target(tonal, importance, mode="multiply", gamma=1.0)
        assert result.mean() > without_gamma.mean()

    def test_gamma_darkening(self):
        """Test that gamma > 1.0 darkens (decreases density)."""
        tonal = np.full((50, 50), 0.5)
        importance = np.full((50, 50), 0.8)

        # Multiply gives 0.4, gamma=2.0 gives 0.4^2 = 0.16
        result = build_density_target(tonal, importance, mode="multiply", gamma=2.0)

        assert np.allclose(result, 0.16)

        # Verify darkening
        without_gamma = build_density_target(tonal, importance, mode="multiply", gamma=1.0)
        assert result.mean() < without_gamma.mean()

    def test_gamma_identity(self):
        """Test that gamma=1.0 is identity."""
        tonal = np.random.rand(50, 50)
        importance = np.random.rand(50, 50)

        result_gamma1 = build_density_target(tonal, importance, mode="multiply", gamma=1.0)
        result_composed = compose_maps(tonal, importance, mode="multiply")

        assert np.allclose(result_gamma1, result_composed)

    def test_different_blend_modes(self):
        """Test that all blend modes work in build_density_target."""
        tonal = np.random.rand(30, 30)
        importance = np.random.rand(30, 30)

        for mode in ["multiply", "screen", "max", "weighted"]:
            result = build_density_target(tonal, importance, mode=mode, gamma=1.0)
            assert result.shape == tonal.shape
            assert result.dtype == np.float64
            assert result.min() >= 0.0
            assert result.max() <= 1.0

    def test_output_range_with_gamma(self):
        """Test that output is in [0, 1] even with extreme gamma values."""
        tonal = np.random.rand(50, 50)
        importance = np.random.rand(50, 50)

        # Test with very small gamma
        result_small = build_density_target(tonal, importance, gamma=0.1)
        assert result_small.min() >= 0.0
        assert result_small.max() <= 1.0

        # Test with very large gamma
        result_large = build_density_target(tonal, importance, gamma=10.0)
        assert result_large.min() >= 0.0
        assert result_large.max() <= 1.0

    def test_zero_handling_with_gamma(self):
        """Test that zeros are handled properly with gamma < 1."""
        tonal = np.array([[0.0, 0.5], [1.0, 0.0]])
        importance = np.array([[1.0, 1.0], [0.0, 0.5]])

        # With gamma < 1, 0^gamma should still be 0
        result = build_density_target(tonal, importance, mode="multiply", gamma=0.5)

        # Check that zeros remain zeros
        assert result[0, 0] == 0.0  # 0*1 = 0, 0^0.5 = 0
        assert result[1, 1] == 0.0  # 0*0.5 = 0, 0^0.5 = 0
        assert result[1, 0] == 0.0  # 1*0 = 0, 0^0.5 = 0

    def test_full_range_usage(self):
        """Test that full [0, 1] range can be achieved."""
        # Create tonal and importance that produce 0 and 1 values
        tonal = np.array([[0.0, 1.0], [1.0, 1.0]])
        importance = np.array([[1.0, 1.0], [0.0, 1.0]])

        result = build_density_target(tonal, importance, mode="multiply", gamma=1.0)

        assert result.min() == 0.0  # Should have exact 0
        assert result.max() == 1.0  # Should have exact 1

    def test_screen_mode_bright_preservation(self):
        """Test that screen mode preserves bright areas."""
        # Dark tonal (high density) and high importance
        tonal = np.full((50, 50), 0.9)  # Dark image → high tonal values
        importance = np.full((50, 50), 0.8)  # High importance

        # Screen mode should keep high values high
        result = build_density_target(tonal, importance, mode="screen", gamma=1.0)

        # screen(0.9, 0.8) = 1 - (1-0.9)*(1-0.8) = 1 - 0.1*0.2 = 0.98
        assert np.allclose(result, 0.98)
        assert result.mean() > 0.9  # Higher than input average

    def test_realistic_portrait_values(self):
        """Test with realistic portrait processing values."""
        # Simulate realistic tonal target (inverted luminance)
        # Dark hair/shadows: 0.8-1.0, Skin: 0.3-0.5, Highlights: 0.0-0.2
        tonal = np.random.rand(100, 100) * 0.7 + 0.2

        # Simulate importance map (features highlighted)
        # Features: 0.7-1.0, Background: 0.0-0.3
        importance = np.random.rand(100, 100) * 0.8 + 0.1

        # Typical multiply composition with slight brightening
        result = build_density_target(tonal, importance, mode="multiply", gamma=0.8)

        assert result.shape == tonal.shape
        assert result.dtype == np.float64
        assert 0.0 <= result.min() <= result.max() <= 1.0
        # Result should have good contrast
        assert result.std() > 0.1
