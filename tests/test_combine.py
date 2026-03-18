"""Tests for portrait_map_lab.combine module."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_map_lab.combine import combine_maps


class TestCombineMaps:
    def test_output_normalized_to_unit_range(self):
        """Output should be normalized to [0.0, 1.0]."""
        # Create maps with values that would sum > 1 without normalization
        maps = {
            "region1": np.array([[0.8, 0.9], [0.7, 0.6]]),
            "region2": np.array([[0.7, 0.8], [0.9, 0.5]]),
        }
        weights = {"region1": 1.0, "region2": 1.0}

        result = combine_maps(maps, weights)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_equal_weights_identical_maps_returns_same(self):
        """Equal weights on identical maps should return the same map."""
        # Create identical maps
        base_map = np.array([[0.1, 0.5], [0.8, 0.3]])
        maps = {
            "map1": base_map.copy(),
            "map2": base_map.copy(),
            "map3": base_map.copy(),
        }
        weights = {"map1": 1.0, "map2": 1.0, "map3": 1.0}

        result = combine_maps(maps, weights)

        np.testing.assert_array_almost_equal(result, base_map)

    def test_zero_weight_excludes_map(self):
        """Zero weight for a map should exclude it from result."""
        maps = {
            "included": np.array([[1.0, 0.5], [0.3, 0.8]]),
            "excluded": np.array([[0.0, 0.0], [1.0, 1.0]]),
        }
        weights = {"included": 1.0, "excluded": 0.0}

        result = combine_maps(maps, weights)

        # Result should equal the included map only
        np.testing.assert_array_almost_equal(result, maps["included"])

    def test_mismatched_keys_raises_error(self):
        """Mismatched keys between maps and weights should raise ValueError."""
        maps = {"region1": np.zeros((10, 10)), "region2": np.ones((10, 10))}

        # Missing key in weights
        weights_missing = {"region1": 1.0}
        with pytest.raises(ValueError, match="Weight keys must match map keys"):
            combine_maps(maps, weights_missing)

        # Extra key in weights
        weights_extra = {"region1": 1.0, "region2": 0.5, "region3": 0.3}
        with pytest.raises(ValueError, match="Weight keys must match map keys"):
            combine_maps(maps, weights_extra)

        # Completely different keys
        weights_different = {"other1": 1.0, "other2": 0.5}
        with pytest.raises(ValueError, match="Weight keys must match map keys"):
            combine_maps(maps, weights_different)

    def test_different_shaped_maps_raises_error(self):
        """Maps with different shapes should raise ValueError."""
        maps = {
            "map1": np.zeros((10, 10)),
            "map2": np.zeros((10, 15)),  # Different shape
        }
        weights = {"map1": 1.0, "map2": 1.0}

        with pytest.raises(ValueError, match="All maps must have the same shape"):
            combine_maps(maps, weights)

    def test_weighted_combination(self):
        """Weighted combination should respect weight ratios."""
        # Create simple test maps
        maps = {
            "high_weight": np.array([[1.0, 1.0], [1.0, 1.0]]),
            "low_weight": np.array([[0.0, 0.0], [0.0, 0.0]]),
        }
        weights = {"high_weight": 0.8, "low_weight": 0.2}

        result = combine_maps(maps, weights)

        # Result should be 0.8 * 1.0 + 0.2 * 0.0 normalized by total weight (1.0)
        expected = np.array([[0.8, 0.8], [0.8, 0.8]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_negative_weights_handled_correctly(self):
        """Negative weights should be handled in normalization."""
        maps = {
            "positive": np.array([[1.0, 0.5]]),
            "negative": np.array([[0.5, 1.0]]),
        }
        weights = {"positive": 1.0, "negative": -0.5}

        result = combine_maps(maps, weights)

        # Should compute weighted sum then normalize by sum of absolute weights
        # (1.0 * [1.0, 0.5] + -0.5 * [0.5, 1.0]) / 1.5
        # = ([1.0, 0.5] + [-0.25, -0.5]) / 1.5
        # = [0.75, 0.0] / 1.5
        # = [0.5, 0.0]
        expected = np.array([[0.5, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_map_with_unit_weight(self):
        """Single map with weight 1.0 should return the same map."""
        original = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        maps = {"solo": original}
        weights = {"solo": 1.0}

        result = combine_maps(maps, weights)

        np.testing.assert_array_almost_equal(result, original)

    def test_output_dtype_is_float64(self):
        """Output should be float64 for precision."""
        maps = {
            "map1": np.array([[0.5]], dtype=np.float32),
            "map2": np.array([[0.5]], dtype=np.float32),
        }
        weights = {"map1": 1.0, "map2": 1.0}

        result = combine_maps(maps, weights)

        assert result.dtype == np.float64

    def test_empty_maps_raises_error(self):
        """Empty maps dictionary should raise ValueError."""
        maps = {}
        weights = {}

        with pytest.raises(ValueError, match="Cannot combine empty maps dictionary"):
            combine_maps(maps, weights)

    def test_all_zero_weights_returns_zeros(self):
        """All zero weights should return array of zeros."""
        maps = {
            "map1": np.array([[0.5, 0.7]]),
            "map2": np.array([[0.3, 0.9]]),
        }
        weights = {"map1": 0.0, "map2": 0.0}

        result = combine_maps(maps, weights)

        np.testing.assert_array_equal(result, np.zeros((1, 2)))

    def test_practical_eye_mouth_combination(self):
        """Test practical scenario of combining eye and mouth maps."""
        # Simulate typical use case
        maps = {
            "eyes": np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]),
            "mouth": np.array([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]]),
        }
        weights = {"eyes": 0.6, "mouth": 0.4}

        result = combine_maps(maps, weights)

        # Verify weighted combination
        # Total weight = 0.6 + 0.4 = 1.0
        expected = (0.6 * maps["eyes"] + 0.4 * maps["mouth"]) / 1.0
        np.testing.assert_array_almost_equal(result, expected)

        # Verify output properties
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert result.dtype == np.float64
