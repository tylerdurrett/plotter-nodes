"""Tests for portrait_map_lab.models dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_map_lab.models import (
    ContourConfig,
    ContourResult,
    LandmarkResult,
    PipelineConfig,
    PipelineResult,
    RegionDefinition,
    RemapConfig,
)


class TestLandmarkResult:
    def test_creation(self):
        landmarks = np.zeros((478, 2), dtype=np.float64)
        result = LandmarkResult(landmarks=landmarks, image_shape=(480, 640), confidence=0.95)
        assert result.landmarks.shape == (478, 2)
        assert result.image_shape == (480, 640)
        assert result.confidence == 0.95

    def test_frozen(self):
        result = LandmarkResult(
            landmarks=np.zeros((10, 2)), image_shape=(100, 100), confidence=0.5
        )
        with pytest.raises(AttributeError):
            result.confidence = 0.9  # type: ignore[misc]


class TestRegionDefinition:
    def test_creation(self):
        region = RegionDefinition(name="test_region", landmark_indices=[1, 2, 3])
        assert region.name == "test_region"
        assert region.landmark_indices == [1, 2, 3]

    def test_frozen(self):
        region = RegionDefinition(name="test", landmark_indices=[1])
        with pytest.raises(AttributeError):
            region.name = "changed"  # type: ignore[misc]


class TestRemapConfig:
    def test_defaults(self):
        config = RemapConfig()
        assert config.curve == "gaussian"
        assert config.radius == 150.0
        assert config.sigma == 80.0
        assert config.tau == 60.0
        assert config.clamp_distance == 300.0

    def test_mutable(self):
        config = RemapConfig()
        config.sigma = 100.0
        assert config.sigma == 100.0


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert len(config.regions) == 3
        region_names = {r.name for r in config.regions}
        assert region_names == {"left_eye", "right_eye", "mouth"}
        assert isinstance(config.remap, RemapConfig)
        assert config.weights == {"eyes": 0.6, "mouth": 0.4}
        assert config.output_dir == "output"

    def test_regions_have_valid_indices(self):
        config = PipelineConfig()
        for region in config.regions:
            assert len(region.landmark_indices) > 0
            assert all(idx >= 0 for idx in region.landmark_indices)

    def test_mutable(self):
        config = PipelineConfig()
        config.output_dir = "custom_output"
        assert config.output_dir == "custom_output"

    def test_independent_defaults(self):
        a = PipelineConfig()
        b = PipelineConfig()
        a.weights["eyes"] = 0.9
        assert b.weights["eyes"] == 0.6


class TestPipelineResult:
    def test_creation(self):
        landmarks = LandmarkResult(
            landmarks=np.zeros((478, 2)), image_shape=(100, 100), confidence=0.9
        )
        result = PipelineResult(
            landmarks=landmarks,
            masks={"eyes": np.zeros((100, 100), dtype=np.uint8)},
            distance_fields={"eyes": np.zeros((100, 100), dtype=np.float64)},
            influence_maps={"eyes": np.zeros((100, 100), dtype=np.float64)},
            combined=np.zeros((100, 100), dtype=np.float64),
        )
        assert result.landmarks.image_shape == (100, 100)
        assert "eyes" in result.masks
        assert result.combined.shape == (100, 100)

    def test_frozen(self):
        landmarks = LandmarkResult(
            landmarks=np.zeros((10, 2)), image_shape=(50, 50), confidence=0.8
        )
        result = PipelineResult(
            landmarks=landmarks,
            masks={},
            distance_fields={},
            influence_maps={},
            combined=np.zeros((50, 50)),
        )
        with pytest.raises(AttributeError):
            result.combined = np.ones((50, 50))  # type: ignore[misc]


class TestContourConfig:
    def test_defaults(self):
        config = ContourConfig()
        assert isinstance(config.remap, RemapConfig)
        assert config.direction == "inward"
        assert config.band_width is None
        assert config.contour_thickness == 1
        assert config.output_dir == "output"

    def test_mutable(self):
        config = ContourConfig()
        config.direction = "outward"
        assert config.direction == "outward"
        config.band_width = 50.0
        assert config.band_width == 50.0

    def test_independent_defaults(self):
        a = ContourConfig()
        b = ContourConfig()
        a.remap.sigma = 100.0
        assert b.remap.sigma == 80.0  # Should still be default value


class TestContourResult:
    def test_creation(self):
        landmarks = LandmarkResult(
            landmarks=np.zeros((478, 2)), image_shape=(100, 100), confidence=0.9
        )
        result = ContourResult(
            landmarks=landmarks,
            contour_polygon=np.zeros((36, 2), dtype=np.float64),
            contour_mask=np.zeros((100, 100), dtype=np.uint8),
            filled_mask=np.zeros((100, 100), dtype=np.uint8),
            signed_distance=np.zeros((100, 100), dtype=np.float64),
            directional_distance=np.zeros((100, 100), dtype=np.float64),
            influence_map=np.zeros((100, 100), dtype=np.float64),
        )
        assert result.landmarks.image_shape == (100, 100)
        assert result.contour_polygon.shape == (36, 2)
        assert result.contour_mask.shape == (100, 100)
        assert result.filled_mask.shape == (100, 100)
        assert result.signed_distance.shape == (100, 100)
        assert result.directional_distance.shape == (100, 100)
        assert result.influence_map.shape == (100, 100)

    def test_frozen(self):
        landmarks = LandmarkResult(
            landmarks=np.zeros((10, 2)), image_shape=(50, 50), confidence=0.8
        )
        result = ContourResult(
            landmarks=landmarks,
            contour_polygon=np.zeros((36, 2)),
            contour_mask=np.zeros((50, 50)),
            filled_mask=np.zeros((50, 50)),
            signed_distance=np.zeros((50, 50)),
            directional_distance=np.zeros((50, 50)),
            influence_map=np.zeros((50, 50)),
        )
        with pytest.raises(AttributeError):
            result.influence_map = np.ones((50, 50))  # type: ignore[misc]
