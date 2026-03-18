"""Tests for portrait_map_lab.masks module."""

from __future__ import annotations

import numpy as np

from portrait_map_lab.face_regions import DEFAULT_REGIONS
from portrait_map_lab.masks import build_region_masks, rasterize_mask
from portrait_map_lab.models import LandmarkResult, RegionDefinition


class TestRasterizeMask:
    def test_output_dtype_and_shape(self):
        polygon = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.float64)
        mask = rasterize_mask(polygon, (100, 100))

        assert mask.dtype == np.uint8
        assert mask.shape == (100, 100)

    def test_contains_only_0_and_255(self):
        polygon = np.array([[5, 5], [25, 5], [25, 25], [5, 25]], dtype=np.float64)
        mask = rasterize_mask(polygon, (50, 50))

        unique = np.unique(mask)
        assert set(unique) <= {0, 255}

    def test_known_polygon_has_nonzero_area(self):
        # Create a square polygon
        polygon = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], dtype=np.float64)
        mask = rasterize_mask(polygon, (60, 60))

        # Check that mask has nonzero pixels
        assert np.sum(mask > 0) > 0
        # Verify approximate area (should be around 20x20 = 400 pixels)
        filled_pixels = np.sum(mask == 255)
        assert 350 < filled_pixels < 450  # Allow some tolerance for rasterization

    def test_empty_polygon_produces_empty_mask(self):
        polygon = np.array([], dtype=np.float64).reshape(0, 2)
        mask = rasterize_mask(polygon, (100, 100))

        assert np.all(mask == 0)


class TestBuildRegionMasks:
    def test_returns_expected_keys(self):
        # Create synthetic landmarks
        landmarks = LandmarkResult(
            landmarks=np.random.rand(478, 2) * 100,
            image_shape=(100, 100),
            confidence=0.95,
        )

        masks = build_region_masks(landmarks, DEFAULT_REGIONS)

        assert "left_eye" in masks
        assert "right_eye" in masks
        assert "mouth" in masks
        assert "combined_eyes" in masks

    def test_mask_dimensions_match_image_shape(self):
        h, w = 200, 300
        landmarks = LandmarkResult(
            landmarks=np.random.rand(478, 2) * [w, h],
            image_shape=(h, w),
            confidence=0.95,
        )

        masks = build_region_masks(landmarks, DEFAULT_REGIONS)

        for name, mask in masks.items():
            assert mask.shape == (h, w), f"Mask {name} has wrong shape"

    def test_all_masks_are_uint8(self):
        landmarks = LandmarkResult(
            landmarks=np.random.rand(478, 2) * 100,
            image_shape=(100, 100),
            confidence=0.95,
        )

        masks = build_region_masks(landmarks, DEFAULT_REGIONS)

        for name, mask in masks.items():
            assert mask.dtype == np.uint8, f"Mask {name} has wrong dtype"

    def test_combined_eyes_is_union(self):
        # Create landmarks that form non-overlapping regions
        landmarks_array = np.zeros((478, 2), dtype=np.float64)
        # Left eye region - small square on left
        landmarks_array[[263, 249, 390, 373]] = [[10, 10], [20, 10], [20, 20], [10, 20]]
        # Right eye region - small square on right
        landmarks_array[[33, 7, 163, 144]] = [[30, 10], [40, 10], [40, 20], [30, 20]]

        landmarks = LandmarkResult(
            landmarks=landmarks_array,
            image_shape=(50, 50),
            confidence=1.0,
        )

        # Use simplified regions for testing
        test_regions = [
            RegionDefinition("left_eye", [263, 249, 390, 373]),
            RegionDefinition("right_eye", [33, 7, 163, 144]),
        ]

        masks = build_region_masks(landmarks, test_regions)

        # Verify combined_eyes is the bitwise OR
        expected_combined = np.bitwise_or(masks["left_eye"], masks["right_eye"])
        np.testing.assert_array_equal(masks["combined_eyes"], expected_combined)

    def test_no_combined_eyes_without_both_eyes(self):
        landmarks = LandmarkResult(
            landmarks=np.random.rand(478, 2) * 100,
            image_shape=(100, 100),
            confidence=0.95,
        )

        # Only mouth region
        regions = [RegionDefinition("mouth", [0, 1, 2])]
        masks = build_region_masks(landmarks, regions)

        assert "combined_eyes" not in masks
