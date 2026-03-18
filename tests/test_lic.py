"""Tests for portrait_map_lab.lic module."""

from __future__ import annotations

import numpy as np

from portrait_map_lab.lic import compute_lic
from portrait_map_lab.models import LICConfig


class TestComputeLIC:
    """Tests for compute_lic function."""

    def test_output_shape(self):
        """Test that output shape matches input flow field shape."""
        h, w = 100, 150
        flow_x = np.ones((h, w), dtype=np.float64)
        flow_y = np.zeros((h, w), dtype=np.float64)

        result = compute_lic(flow_x, flow_y)

        assert result.shape == (h, w)

    def test_output_dtype(self):
        """Test that output dtype is float64."""
        flow_x = np.ones((50, 50), dtype=np.float64)
        flow_y = np.zeros((50, 50), dtype=np.float64)

        result = compute_lic(flow_x, flow_y)

        assert result.dtype == np.float64

    def test_output_range(self):
        """Test that output values are in [0, 1] range."""
        flow_x = np.random.randn(60, 60).astype(np.float64)
        flow_y = np.random.randn(60, 60).astype(np.float64)
        # Normalize to unit vectors
        mag = np.sqrt(flow_x**2 + flow_y**2)
        flow_x = flow_x / np.maximum(mag, 1e-10)
        flow_y = flow_y / np.maximum(mag, 1e-10)

        result = compute_lic(flow_x, flow_y)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_deterministic(self):
        """Test that same seed produces same output."""
        flow_x = np.ones((40, 40), dtype=np.float64)
        flow_y = np.zeros((40, 40), dtype=np.float64)
        config = LICConfig(seed=42)

        result1 = compute_lic(flow_x, flow_y, config)
        result2 = compute_lic(flow_x, flow_y, config)

        assert np.allclose(result1, result2)

    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different outputs."""
        flow_x = np.ones((40, 40), dtype=np.float64)
        flow_y = np.zeros((40, 40), dtype=np.float64)

        config1 = LICConfig(seed=42)
        config2 = LICConfig(seed=123)

        result1 = compute_lic(flow_x, flow_y, config1)
        result2 = compute_lic(flow_x, flow_y, config2)

        # Results should be different but both valid
        assert not np.allclose(result1, result2)
        assert np.all(result1 >= 0.0) and np.all(result1 <= 1.0)
        assert np.all(result2 >= 0.0) and np.all(result2 <= 1.0)

    def test_uniform_flow_horizontal_streaks(self):
        """Test that uniform rightward flow creates horizontal streaks."""
        h, w = 50, 100
        # Uniform flow pointing right
        flow_x = np.ones((h, w), dtype=np.float64)
        flow_y = np.zeros((h, w), dtype=np.float64)

        config = LICConfig(length=20, seed=42)
        result = compute_lic(flow_x, flow_y, config)

        # Check that horizontal variance is low (streaky along rows)
        # and vertical variance is higher (variation across rows)
        for row in range(10, h - 10):  # Check central rows
            row_variance = np.var(result[row, :])
            # Each row should have low variance (streak)
            assert row_variance < 0.05, f"Row {row} has high variance: {row_variance}"

        # Variance across different rows should be higher
        col_sample = result[:, w // 2]  # Sample middle column
        col_variance = np.var(col_sample)
        assert col_variance > 0.005, f"Column variance too low: {col_variance}"

    def test_uniform_flow_vertical_streaks(self):
        """Test that uniform downward flow creates vertical streaks."""
        h, w = 100, 50
        # Uniform flow pointing down
        flow_x = np.zeros((h, w), dtype=np.float64)
        flow_y = np.ones((h, w), dtype=np.float64)

        config = LICConfig(length=20, seed=42)
        result = compute_lic(flow_x, flow_y, config)

        # Check that vertical variance is low (streaky along columns)
        # and horizontal variance is higher (variation across columns)
        for col in range(10, w - 10):  # Check central columns
            col_variance = np.var(result[:, col])
            # Each column should have low variance (streak)
            assert col_variance < 0.05, f"Column {col} has high variance: {col_variance}"

        # Variance across different columns should be higher
        row_sample = result[h // 2, :]  # Sample middle row
        row_variance = np.var(row_sample)
        assert row_variance > 0.005, f"Row variance too low: {row_variance}"

    def test_bilinear_vs_nearest(self):
        """Test that both bilinear and nearest-neighbor modes produce valid output."""
        flow_x = np.ones((60, 60), dtype=np.float64)
        flow_y = np.zeros((60, 60), dtype=np.float64)

        config_bilinear = LICConfig(use_bilinear=True, seed=42)
        config_nearest = LICConfig(use_bilinear=False, seed=42)

        result_bilinear = compute_lic(flow_x, flow_y, config_bilinear)
        result_nearest = compute_lic(flow_x, flow_y, config_nearest)

        # Both should produce valid outputs
        assert result_bilinear.shape == (60, 60)
        assert result_nearest.shape == (60, 60)
        assert np.all(result_bilinear >= 0.0) and np.all(result_bilinear <= 1.0)
        assert np.all(result_nearest >= 0.0) and np.all(result_nearest <= 1.0)

        # They should be highly correlated (same noise pattern, different interpolation)
        correlation = np.corrcoef(result_bilinear.flatten(), result_nearest.flatten())[0, 1]
        assert correlation > 0.9  # Very similar patterns due to same seed
        # Mean values should be very close
        assert np.abs(np.mean(result_bilinear) - np.mean(result_nearest)) < 0.01

    def test_config_none(self):
        """Test that default config (None) works correctly."""
        flow_x = np.ones((40, 40), dtype=np.float64)
        flow_y = np.zeros((40, 40), dtype=np.float64)

        # Should use default config without error
        result = compute_lic(flow_x, flow_y, config=None)

        assert result.shape == (40, 40)
        assert result.dtype == np.float64
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_zero_flow(self):
        """Test that zero flow field is handled gracefully."""
        flow_x = np.zeros((30, 30), dtype=np.float64)
        flow_y = np.zeros((30, 30), dtype=np.float64)

        result = compute_lic(flow_x, flow_y)

        # Should return valid output without errors
        assert result.shape == (30, 30)
        assert result.dtype == np.float64
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

        # With zero flow, result should be relatively uniform
        # (just averaged noise at each point)
        assert np.std(result) < 0.3

    def test_circular_flow(self):
        """Test LIC with circular flow field."""
        # Create a circular flow pattern
        h, w = 80, 80
        y, x = np.mgrid[0:h, 0:w].astype(np.float64)
        cx, cy = w / 2, h / 2
        dx = x - cx
        dy = y - cy

        # Tangent to circles centered at (cx, cy)
        flow_x = -dy
        flow_y = dx

        # Normalize to unit vectors
        mag = np.sqrt(flow_x**2 + flow_y**2)
        mag = np.maximum(mag, 1e-10)  # Avoid division by zero
        flow_x = flow_x / mag
        flow_y = flow_y / mag

        config = LICConfig(length=15, seed=42)
        result = compute_lic(flow_x, flow_y, config)

        # Should produce valid output
        assert result.shape == (h, w)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

        # Check that we have variation (not uniform)
        assert np.std(result) > 0.05

    def test_small_step_size(self):
        """Test LIC with small step size."""
        flow_x = np.ones((40, 40), dtype=np.float64)
        flow_y = np.zeros((40, 40), dtype=np.float64)

        config = LICConfig(step=0.5, length=20, seed=42)
        result = compute_lic(flow_x, flow_y, config)

        assert result.shape == (40, 40)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_large_step_size(self):
        """Test LIC with large step size."""
        flow_x = np.ones((40, 40), dtype=np.float64)
        flow_y = np.zeros((40, 40), dtype=np.float64)

        config = LICConfig(step=2.0, length=10, seed=42)
        result = compute_lic(flow_x, flow_y, config)

        assert result.shape == (40, 40)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_short_length(self):
        """Test LIC with short integration length."""
        flow_x = np.ones((40, 40), dtype=np.float64)
        flow_y = np.zeros((40, 40), dtype=np.float64)

        config = LICConfig(length=5, seed=42)
        result = compute_lic(flow_x, flow_y, config)

        assert result.shape == (40, 40)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_long_length(self):
        """Test LIC with long integration length."""
        flow_x = np.ones((40, 40), dtype=np.float64)
        flow_y = np.zeros((40, 40), dtype=np.float64)

        config = LICConfig(length=50, seed=42)
        result = compute_lic(flow_x, flow_y, config)

        assert result.shape == (40, 40)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
