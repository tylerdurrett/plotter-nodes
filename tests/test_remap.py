"""Tests for portrait_map_lab.remap module."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_map_lab.models import RemapConfig
from portrait_map_lab.remap import remap_influence


class TestRemapInfluence:
    def test_output_values_in_range(self):
        """Output values should be in [0.0, 1.0]."""
        distance_field = np.random.uniform(0, 500, (100, 100))
        config = RemapConfig(curve="gaussian")

        influence = remap_influence(distance_field, config)

        assert np.all(influence >= 0.0)
        assert np.all(influence <= 1.0)

    def test_influence_one_at_zero_distance_all_curves(self):
        """Influence should be 1.0 at distance 0.0 for all curve types."""
        distance_field = np.array([[0.0, 10.0], [50.0, 100.0]])

        for curve in ["linear", "gaussian", "exponential"]:
            config = RemapConfig(curve=curve)
            influence = remap_influence(distance_field, config)

            # Check that distance 0.0 maps to influence 1.0
            assert influence[0, 0] == 1.0, f"Failed for curve: {curve}"

    def test_influence_decreases_with_distance(self):
        """Influence should decrease as distance increases."""
        distances = np.array([0.0, 10.0, 30.0, 50.0, 100.0, 200.0]).reshape(2, 3)

        for curve in ["linear", "gaussian", "exponential"]:
            config = RemapConfig(curve=curve)
            influence = remap_influence(distances, config)

            # Flatten to make comparison easier
            flat_dist = distances.flatten()
            flat_inf = influence.flatten()

            # Sort by distance
            sorted_idx = np.argsort(flat_dist)
            sorted_dist = flat_dist[sorted_idx]
            sorted_inf = flat_inf[sorted_idx]

            # Check monotonic decrease
            for i in range(len(sorted_inf) - 1):
                if sorted_dist[i] < sorted_dist[i + 1]:
                    assert sorted_inf[i] >= sorted_inf[i + 1], (
                        f"Influence not decreasing for curve {curve}: "
                        f"d={sorted_dist[i]:.1f} -> {sorted_dist[i + 1]:.1f}, "
                        f"inf={sorted_inf[i]:.3f} -> {sorted_inf[i + 1]:.3f}"
                    )

    def test_linear_curve_reaches_zero_at_radius(self):
        """Linear curve should reach 0.0 at d >= radius."""
        config = RemapConfig(curve="linear", radius=100.0)

        # Test distances at and beyond radius
        distances = np.array([0.0, 50.0, 99.0, 100.0, 150.0, 200.0])
        influence = remap_influence(distances, config)

        # Check values at specific distances
        assert influence[0] == 1.0  # At d=0
        assert 0.0 < influence[1] < 1.0  # At d=50 (midpoint)
        assert influence[3] == 0.0  # At d=100 (radius)
        assert influence[4] == 0.0  # At d=150 (beyond radius)
        assert influence[5] == 0.0  # At d=200 (beyond radius)

    def test_invalid_curve_raises_error(self):
        """Invalid curve name should raise ValueError."""
        distance_field = np.zeros((10, 10))
        config = RemapConfig(curve="invalid_curve")

        with pytest.raises(ValueError, match="Unknown curve type: invalid_curve"):
            remap_influence(distance_field, config)

    def test_distance_clamping(self):
        """Distances should be clamped at config.clamp_distance."""
        # Create distances that exceed clamp_distance
        distances = np.array([0.0, 100.0, 300.0, 500.0, 1000.0])
        config = RemapConfig(curve="exponential", clamp_distance=300.0, tau=60.0)

        influence = remap_influence(distances, config)

        # Distances at/beyond clamp should produce same influence
        assert influence[2] == influence[3]  # 300 and 500 clamped to 300
        assert influence[2] == influence[4]  # 300 and 1000 clamped to 300

    def test_gaussian_curve_parameters(self):
        """Gaussian curve should respect sigma parameter."""
        distances = np.array([0.0, 40.0, 80.0, 120.0])

        # Smaller sigma = faster falloff
        config_narrow = RemapConfig(curve="gaussian", sigma=40.0)
        influence_narrow = remap_influence(distances, config_narrow)

        # Larger sigma = slower falloff
        config_wide = RemapConfig(curve="gaussian", sigma=120.0)
        influence_wide = remap_influence(distances, config_wide)

        # At same distance > 0, wider sigma should have more influence
        assert influence_wide[1] > influence_narrow[1]  # At d=40
        assert influence_wide[2] > influence_narrow[2]  # At d=80

    def test_exponential_curve_parameters(self):
        """Exponential curve should respect tau parameter."""
        distances = np.array([0.0, 30.0, 60.0, 120.0])

        # Smaller tau = faster falloff
        config_fast = RemapConfig(curve="exponential", tau=30.0)
        influence_fast = remap_influence(distances, config_fast)

        # Larger tau = slower falloff
        config_slow = RemapConfig(curve="exponential", tau=90.0)
        influence_slow = remap_influence(distances, config_slow)

        # At same distance > 0, larger tau should have more influence
        assert influence_slow[1] > influence_fast[1]  # At d=30
        assert influence_slow[2] > influence_fast[2]  # At d=60

    def test_output_dtype_is_float64(self):
        """Output should be float64 for precision."""
        distance_field = np.array([0.0, 50.0, 100.0], dtype=np.float32)
        config = RemapConfig()

        influence = remap_influence(distance_field, config)

        assert influence.dtype == np.float64

    def test_2d_array_shape_preserved(self):
        """2D input shape should be preserved in output."""
        distance_field = np.random.uniform(0, 200, (50, 75))
        config = RemapConfig(curve="gaussian")

        influence = remap_influence(distance_field, config)

        assert influence.shape == distance_field.shape
