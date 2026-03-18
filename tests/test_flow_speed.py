"""Tests for portrait_map_lab.flow_speed module."""

from __future__ import annotations

import numpy as np

from portrait_map_lab.flow_speed import compute_flow_speed
from portrait_map_lab.models import FlowSpeedConfig


class TestComputeFlowSpeed:
    """Tests for compute_flow_speed function."""

    def test_zero_complexity_produces_max_speed(self):
        """Test that zero complexity everywhere produces speed_max everywhere."""
        # Create zero complexity map
        complexity = np.zeros((50, 50), dtype=np.float64)

        # Default config has speed_max=1.0
        speed = compute_flow_speed(complexity)

        # All values should equal speed_max
        assert np.allclose(speed, 1.0)
        assert np.isclose(speed.min(), 1.0)
        assert np.isclose(speed.max(), 1.0)

    def test_full_complexity_produces_min_speed(self):
        """Test that full complexity (1.0) everywhere produces speed_min everywhere."""
        # Create full complexity map
        complexity = np.ones((50, 50), dtype=np.float64)

        # Default config has speed_min=0.3
        speed = compute_flow_speed(complexity)

        # All values should equal speed_min
        assert np.allclose(speed, 0.3)
        assert np.isclose(speed.min(), 0.3)
        assert np.isclose(speed.max(), 0.3)

    def test_mid_complexity_produces_midpoint_speed(self):
        """Test that 0.5 complexity produces midpoint speed."""
        # Create mid-level complexity map
        complexity = np.full((30, 40), 0.5, dtype=np.float64)

        # Default: speed_min=0.3, speed_max=1.0, midpoint=0.65
        speed = compute_flow_speed(complexity)

        # Expected: 1.0 - 0.5 * (1.0 - 0.3) = 1.0 - 0.35 = 0.65
        expected = 0.65
        assert np.allclose(speed, expected)

    def test_output_shape_matches_input(self):
        """Test that output shape matches input complexity map."""
        complexity = np.random.rand(100, 80).astype(np.float64)
        speed = compute_flow_speed(complexity)

        assert speed.shape == complexity.shape
        assert speed.shape == (100, 80)

    def test_output_dtype_is_float64(self):
        """Test that output dtype is always float64."""
        # Test with different input dtypes
        complexity_f32 = np.random.rand(20, 20).astype(np.float32)
        speed = compute_flow_speed(complexity_f32)
        assert speed.dtype == np.float64

        complexity_f64 = np.random.rand(20, 20).astype(np.float64)
        speed = compute_flow_speed(complexity_f64)
        assert speed.dtype == np.float64

    def test_values_within_configured_range(self):
        """Test that all output values are within [speed_min, speed_max]."""
        # Test with random complexity values
        complexity = np.random.rand(50, 60).astype(np.float64)

        # Default config
        speed = compute_flow_speed(complexity)
        assert speed.min() >= 0.3  # speed_min
        assert speed.max() <= 1.0  # speed_max

        # Custom config
        config = FlowSpeedConfig(speed_min=0.1, speed_max=0.8)
        speed = compute_flow_speed(complexity, config)
        assert speed.min() >= 0.1
        assert speed.max() <= 0.8

    def test_custom_speed_range(self):
        """Test that custom speed_min/speed_max work correctly."""
        complexity = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)

        config = FlowSpeedConfig(speed_min=0.2, speed_max=0.9)
        speed = compute_flow_speed(complexity, config)

        # Check specific values
        assert np.isclose(speed[0, 0], 0.9)  # complexity=0 -> speed_max
        assert np.isclose(speed[0, 1], 0.55)  # complexity=0.5 -> midpoint
        assert np.isclose(speed[0, 2], 0.2)  # complexity=1 -> speed_min

    def test_linear_inverse_relationship(self):
        """Test that speed has linear inverse relationship with complexity."""
        # Create gradient of complexity values
        complexity = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float64)

        config = FlowSpeedConfig(speed_min=0.3, speed_max=1.0)
        speed = compute_flow_speed(complexity, config)

        # Speed should decrease linearly as complexity increases
        speed_flat = speed.flatten()
        complexity_flat = complexity.flatten()

        # Verify linear relationship: speed = speed_max - complexity * (speed_max - speed_min)
        expected_speed = 1.0 - complexity_flat * 0.7
        assert np.allclose(speed_flat, expected_speed)

    def test_handles_edge_cases(self):
        """Test that function handles edge cases gracefully."""
        # Test with values slightly outside [0, 1] due to numerical errors
        complexity = np.array([[-0.001, 0.5, 1.001]], dtype=np.float64)

        speed = compute_flow_speed(complexity)

        # Should clip to valid range
        assert speed.min() >= 0.3
        assert speed.max() <= 1.0

    def test_none_config_uses_defaults(self):
        """Test that None config uses default FlowSpeedConfig."""
        complexity = np.array([[0.0, 1.0]], dtype=np.float64)

        # Using None should be same as default config
        speed_none = compute_flow_speed(complexity, None)
        speed_default = compute_flow_speed(complexity, FlowSpeedConfig())

        assert np.allclose(speed_none, speed_default)
        # Check actual default values
        assert np.isclose(speed_none[0, 0], 1.0)  # speed_max
        assert np.isclose(speed_none[0, 1], 0.3)  # speed_min

    def test_preserves_spatial_patterns(self):
        """Test that spatial patterns are preserved (inverted) in speed map."""
        # Create a complexity map with a clear pattern
        x = np.linspace(-1, 1, 50)
        y = np.linspace(-1, 1, 50)
        X, Y = np.meshgrid(x, y)
        # Gaussian blob - high complexity in center
        complexity = np.exp(-(X**2 + Y**2) / 0.2)
        complexity = complexity / complexity.max()  # Normalize to [0, 1]

        speed = compute_flow_speed(complexity)

        # Speed should be low where complexity is high (center)
        center_idx = 25
        assert speed[center_idx, center_idx] < speed[0, 0]
        assert speed[center_idx, center_idx] < speed[-1, -1]

        # Speed at edges should be close to max
        assert speed[0, 0] > 0.9
        assert speed[-1, -1] > 0.9
