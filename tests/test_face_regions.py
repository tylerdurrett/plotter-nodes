"""Tests for portrait_map_lab.face_regions module."""

from __future__ import annotations

import numpy as np

from portrait_map_lab.face_regions import DEFAULT_REGIONS, get_region_polygons
from portrait_map_lab.models import LandmarkResult, RegionDefinition


class TestDefaultRegions:
    def test_contains_expected_names(self):
        names = {r.name for r in DEFAULT_REGIONS}
        assert names == {"left_eye", "right_eye", "mouth"}

    def test_all_have_landmark_indices(self):
        for region in DEFAULT_REGIONS:
            assert len(region.landmark_indices) > 0
            assert all(isinstance(idx, int) for idx in region.landmark_indices)


class TestGetRegionPolygons:
    def test_returns_correct_shapes(self):
        landmarks = LandmarkResult(
            landmarks=np.random.rand(478, 2) * 100,
            image_shape=(480, 640),
            confidence=0.95,
        )
        regions = [
            RegionDefinition("test1", [0, 1, 2]),
            RegionDefinition("test2", [10, 11, 12, 13]),
        ]

        polygons = get_region_polygons(landmarks, regions)

        assert "test1" in polygons
        assert "test2" in polygons
        assert polygons["test1"].shape == (3, 2)
        assert polygons["test2"].shape == (4, 2)
        assert polygons["test1"].dtype == np.float64
        assert polygons["test2"].dtype == np.float64

    def test_coordinates_within_bounds(self):
        h, w = 480, 640
        landmarks = LandmarkResult(
            landmarks=np.random.rand(478, 2) * [w, h],
            image_shape=(h, w),
            confidence=0.95,
        )

        polygons = get_region_polygons(landmarks, DEFAULT_REGIONS)

        for name, polygon in polygons.items():
            assert np.all(polygon[:, 0] >= 0), f"Region {name} has negative x"
            assert np.all(polygon[:, 0] <= w), f"Region {name} exceeds width"
            assert np.all(polygon[:, 1] >= 0), f"Region {name} has negative y"
            assert np.all(polygon[:, 1] <= h), f"Region {name} exceeds height"

    def test_polygon_extraction_preserves_values(self):
        landmarks_array = np.arange(478 * 2, dtype=np.float64).reshape(478, 2)
        landmarks = LandmarkResult(
            landmarks=landmarks_array,
            image_shape=(100, 100),
            confidence=1.0,
        )
        regions = [RegionDefinition("test", [0, 10, 20])]

        polygons = get_region_polygons(landmarks, regions)

        expected = landmarks_array[[0, 10, 20]]
        np.testing.assert_array_equal(polygons["test"], expected)
