"""Tests for portrait_map_lab.complexity_map module."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_map_lab.complexity_map import (
    apply_mask,
    compute_complexity_map,
    compute_gradient_energy,
    compute_laplacian_energy,
    compute_multiscale_gradient_energy,
    normalize_map,
)
from portrait_map_lab.models import ComplexityConfig, ComplexityResult


class TestComputeGradientEnergy:
    """Tests for compute_gradient_energy function."""

    def test_flat_image(self):
        """Test that flat image produces near-zero energy."""
        # Create uniform gray image
        gray = np.ones((100, 100), dtype=np.float64) * 0.5
        energy = compute_gradient_energy(gray, sigma=3.0)

        # Should be near zero everywhere
        assert np.all(energy < 1e-6)

    def test_sharp_edge(self):
        """Test that sharp edge produces high energy at edge."""
        # Create image with vertical edge in the middle
        gray = np.zeros((100, 100), dtype=np.float64)
        gray[:, 50:] = 1.0

        energy = compute_gradient_energy(gray, sigma=1.0)

        # Peak energy should be around the edge (column 50)
        edge_energy = energy[:, 48:52].mean()
        background_energy = energy[:, :20].mean()

        assert edge_energy > background_energy * 10  # Edge much stronger than background

    def test_output_shape_and_dtype(self):
        """Test output shape matches input and dtype is float64."""
        gray = np.random.rand(50, 75).astype(np.float64)
        energy = compute_gradient_energy(gray, sigma=3.0)

        assert energy.shape == gray.shape
        assert energy.dtype == np.float64

    def test_non_negative(self):
        """Test that all energy values are non-negative."""
        gray = np.random.rand(50, 50).astype(np.float64)
        energy = compute_gradient_energy(gray, sigma=3.0)

        assert np.all(energy >= 0)

    def test_sigma_effect(self):
        """Test that larger sigma produces smoother output."""
        # Create image with noise
        np.random.seed(42)
        gray = np.random.rand(50, 50).astype(np.float64)

        energy_small = compute_gradient_energy(gray, sigma=1.0)
        energy_large = compute_gradient_energy(gray, sigma=5.0)

        # Large sigma should have lower variance (smoother)
        assert np.var(energy_large) < np.var(energy_small)


class TestComputeLaplacianEnergy:
    """Tests for compute_laplacian_energy function."""

    def test_flat_image(self):
        """Test that flat image produces near-zero energy."""
        gray = np.ones((100, 100), dtype=np.float64) * 0.5
        energy = compute_laplacian_energy(gray, sigma=3.0)

        assert np.all(energy < 1e-6)

    def test_checkerboard_pattern(self):
        """Test that fine checkerboard pattern produces high energy."""
        # Create checkerboard pattern
        gray = np.zeros((100, 100), dtype=np.float64)
        gray[::2, ::2] = 1.0
        gray[1::2, 1::2] = 1.0

        energy = compute_laplacian_energy(gray, sigma=1.0)

        # Checkerboard should have uniformly high energy
        assert energy.mean() > 0.1

    def test_sharp_edge(self):
        """Test that sharp edge produces energy at the edge."""
        gray = np.zeros((100, 100), dtype=np.float64)
        gray[:, 50:] = 1.0

        energy = compute_laplacian_energy(gray, sigma=1.0)

        # Edge should have high energy
        edge_energy = energy[:, 48:52].mean()
        background_energy = energy[:, :20].mean()

        assert edge_energy > background_energy * 5

    def test_output_dtype(self):
        """Test that output is float64 and non-negative."""
        gray = np.random.rand(50, 50).astype(np.float64)
        energy = compute_laplacian_energy(gray, sigma=3.0)

        assert energy.dtype == np.float64
        assert np.all(energy >= 0)


class TestComputeMultiscaleGradientEnergy:
    """Tests for compute_multiscale_gradient_energy function."""

    def test_default_scales_weights(self):
        """Test with default scales and weights."""
        gray = np.random.rand(50, 50).astype(np.float64)
        scales = [1.0, 3.0, 8.0]
        weights = [0.5, 0.3, 0.2]

        energy = compute_multiscale_gradient_energy(gray, scales, weights)

        assert energy.shape == gray.shape
        assert energy.dtype == np.float64
        assert np.all(energy >= 0)

    def test_mismatched_lengths(self):
        """Test that mismatched scales/weights raises ValueError."""
        gray = np.random.rand(50, 50).astype(np.float64)
        scales = [1.0]
        weights = [1.0, 2.0]

        with pytest.raises(ValueError, match="scales and weights must have same length"):
            compute_multiscale_gradient_energy(gray, scales, weights)

    def test_multiple_scales_contribution(self):
        """Test that result has contributions from multiple scales."""
        # Create image with features at different scales
        gray = np.zeros((100, 100), dtype=np.float64)
        # Add fine detail
        gray[::4, ::4] = 0.5
        # Add coarse edge
        gray[:, 50:] += 0.5

        # Single scale (fine)
        energy_fine = compute_multiscale_gradient_energy(gray, [1.0], [1.0])
        # Single scale (coarse)
        energy_coarse = compute_multiscale_gradient_energy(gray, [8.0], [1.0])
        # Multi-scale
        energy_multi = compute_multiscale_gradient_energy(
            gray, [1.0, 8.0], [0.5, 0.5]
        )

        # Multi-scale should be approximately the weighted average
        # Allow for some tolerance due to numerical precision
        expected = 0.5 * energy_fine.mean() + 0.5 * energy_coarse.mean()
        assert abs(energy_multi.mean() - expected) < 0.01

    def test_output_properties(self):
        """Test output is float64 and non-negative."""
        gray = np.random.rand(60, 80).astype(np.float64)
        energy = compute_multiscale_gradient_energy(
            gray, [2.0, 4.0], [0.6, 0.4]
        )

        assert energy.dtype == np.float64
        assert np.all(energy >= 0)


class TestNormalizeMap:
    """Tests for normalize_map function."""

    def test_percentile_100_is_max(self):
        """Test that percentile=100 divides by max."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        normalized = normalize_map(arr, percentile=100.0)

        # Max is 4.0, so result should be arr/4.0
        expected = arr / 4.0
        assert np.allclose(normalized, expected)

    def test_percentile_99_clips_top(self):
        """Test that percentile=99 clips top 1% of values."""
        # Create array with outlier
        arr = np.ones((100, 100), dtype=np.float64)
        arr[0, 0] = 100.0  # Single outlier

        normalized = normalize_map(arr, percentile=99.0)

        # Most values should be around 1.0 (since 99th percentile is ~1.0)
        assert np.median(normalized) > 0.9
        # Outlier should be clipped to 1.0
        assert normalized[0, 0] == 1.0

    def test_zero_input(self):
        """Test that zero input returns zeros without divide-by-zero."""
        arr = np.zeros((50, 50), dtype=np.float64)
        normalized = normalize_map(arr, percentile=99.0)

        assert np.all(normalized == 0.0)
        assert not np.any(np.isnan(normalized))

    def test_output_range(self):
        """Test that output is always in [0, 1]."""
        arr = np.random.rand(50, 50) * 10  # Random values up to 10
        normalized = normalize_map(arr, percentile=95.0)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0


class TestApplyMask:
    """Tests for apply_mask function."""

    def test_zeros_masked_regions(self):
        """Test that masked regions are zeroed out."""
        map_array = np.ones((50, 50), dtype=np.float64)
        mask = np.zeros((50, 50), dtype=np.float64)
        mask[10:40, 10:40] = 1.0

        masked = apply_mask(map_array, mask)

        # Outside mask should be zero
        assert np.all(masked[:10, :] == 0.0)
        assert np.all(masked[40:, :] == 0.0)
        # Inside mask should be preserved
        assert np.all(masked[10:40, 10:40] == 1.0)

    def test_preserves_unmasked(self):
        """Test that unmasked values are preserved."""
        map_array = np.random.rand(50, 50).astype(np.float64)
        mask = np.ones((50, 50), dtype=np.float64)

        masked = apply_mask(map_array, mask)

        assert np.allclose(masked, map_array)

    def test_uint8_mask(self):
        """Test handling of uint8 masks (0/255)."""
        map_array = np.ones((30, 30), dtype=np.float64)
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        masked = apply_mask(map_array, mask)

        # Check masked regions
        assert np.all(masked[:10, :] == 0.0)
        assert np.all(masked[10:20, 10:20] == 1.0)

    def test_float_mask(self):
        """Test handling of float masks (0.0/1.0)."""
        map_array = np.ones((30, 30), dtype=np.float64) * 0.5
        mask = np.zeros((30, 30), dtype=np.float64)
        mask[5:25, 5:25] = 1.0

        masked = apply_mask(map_array, mask)

        assert np.all(masked[:5, :] == 0.0)
        assert np.all(masked[5:25, 5:25] == 0.5)


class TestComputeComplexityMap:
    """Tests for compute_complexity_map main entry point."""

    def test_default_config(self):
        """Test with default configuration."""
        # Create test image
        image = np.random.rand(100, 100, 3).astype(np.float64)
        result = compute_complexity_map(image)

        assert isinstance(result, ComplexityResult)
        assert result.complexity.shape == (100, 100)
        assert result.raw_complexity.shape == (100, 100)
        assert result.metric == "gradient"
        assert result.complexity.min() >= 0.0
        assert result.complexity.max() <= 1.0

    def test_gradient_metric(self):
        """Test gradient metric dispatch."""
        image = np.random.rand(50, 50).astype(np.float64)
        config = ComplexityConfig(metric="gradient", sigma=2.0)

        result = compute_complexity_map(image, config)

        assert result.metric == "gradient"
        assert result.complexity.dtype == np.float64

    def test_laplacian_metric(self):
        """Test laplacian metric dispatch."""
        image = np.random.rand(50, 50).astype(np.float64)
        config = ComplexityConfig(metric="laplacian", sigma=2.0)

        result = compute_complexity_map(image, config)

        assert result.metric == "laplacian"

    def test_multiscale_gradient_metric(self):
        """Test multiscale_gradient metric dispatch."""
        image = np.random.rand(50, 50).astype(np.float64)
        config = ComplexityConfig(
            metric="multiscale_gradient",
            scales=[1.0, 2.0],
            scale_weights=[0.6, 0.4]
        )

        result = compute_complexity_map(image, config)

        assert result.metric == "multiscale_gradient"

    def test_unknown_metric(self):
        """Test that unknown metric raises ValueError."""
        image = np.random.rand(50, 50).astype(np.float64)
        config = ComplexityConfig(metric="unknown")

        with pytest.raises(ValueError, match="Unknown complexity metric"):
            compute_complexity_map(image, config)

    def test_with_mask(self):
        """Test complexity computation with mask."""
        image = np.ones((50, 50), dtype=np.float64)
        mask = np.zeros((50, 50), dtype=np.float64)
        mask[10:40, 10:40] = 1.0

        result = compute_complexity_map(image, mask=mask)

        # Outside mask should be zero
        assert np.all(result.complexity[:10, :] == 0.0)
        # Both raw and normalized should be masked
        assert np.all(result.raw_complexity[:10, :] == 0.0)

    def test_bgr_input(self):
        """Test handling of BGR color image input."""
        # Create BGR image (3 channels)
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)

        result = compute_complexity_map(image)

        assert result.complexity.shape == (50, 50)
        assert result.complexity.dtype == np.float64

    def test_grayscale_uint8_input(self):
        """Test handling of grayscale uint8 input."""
        image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)

        result = compute_complexity_map(image)

        assert result.complexity.shape == (50, 50)
        assert result.complexity.min() >= 0.0
        assert result.complexity.max() <= 1.0

    def test_result_fields(self):
        """Test that result has correct types and shapes."""
        image = np.random.rand(60, 80).astype(np.float64)
        config = ComplexityConfig(metric="gradient", normalize_percentile=95.0)

        result = compute_complexity_map(image, config)

        assert isinstance(result, ComplexityResult)
        assert isinstance(result.raw_complexity, np.ndarray)
        assert isinstance(result.complexity, np.ndarray)
        assert isinstance(result.metric, str)
        assert result.raw_complexity.shape == (60, 80)
        assert result.complexity.shape == (60, 80)
        assert result.metric == "gradient"
