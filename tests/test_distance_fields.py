"""Tests for portrait_map_lab.distance_fields module."""

from __future__ import annotations

import numpy as np

from portrait_map_lab.distance_fields import compute_distance_field


class TestComputeDistanceField:
    def test_output_shape_matches_input(self):
        """Distance field should have same shape as input mask."""
        mask = np.zeros((100, 150), dtype=np.uint8)
        mask[40:60, 50:100] = 255  # Rectangle in center

        distance_field = compute_distance_field(mask)

        assert distance_field.shape == mask.shape

    def test_output_dtype_is_float64(self):
        """Distance field should be float64 for precision."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255

        distance_field = compute_distance_field(mask)

        assert distance_field.dtype == np.float64

    def test_pixels_inside_mask_have_zero_distance(self):
        """All pixels within the mask region should have distance 0.0."""
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[30:50, 30:50] = 255  # 20x20 square

        distance_field = compute_distance_field(mask)

        # Check all masked pixels have distance 0
        assert np.all(distance_field[mask > 0] == 0.0)

    def test_distance_increases_from_boundary(self):
        """Distance should increase moving away from mask boundary."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # 20x20 square in center

        distance_field = compute_distance_field(mask)

        # Sample points at increasing distances from center
        center = 50
        # Point inside mask
        assert distance_field[center, center] == 0.0
        # Points progressively further from mask
        assert distance_field[30, 50] > distance_field[35, 50]
        assert distance_field[20, 50] > distance_field[30, 50]
        assert distance_field[10, 50] > distance_field[20, 50]

    def test_empty_mask_produces_max_distances(self):
        """Empty mask (no masked regions) should produce maximum distances everywhere."""
        mask = np.zeros((50, 50), dtype=np.uint8)

        distance_field = compute_distance_field(mask)

        # All pixels should have positive distance since nothing is masked
        assert np.all(distance_field > 0.0)
        # Center should have maximum distance
        center_dist = distance_field[25, 25]
        assert center_dist > distance_field[0, 0]  # Greater than corner

    def test_full_mask_all_zeros(self):
        """Full mask (all pixels masked) should produce zero distances everywhere."""
        mask = np.ones((60, 80), dtype=np.uint8) * 255

        distance_field = compute_distance_field(mask)

        # All points should have distance 0 since everything is masked
        assert np.all(distance_field == 0.0)
