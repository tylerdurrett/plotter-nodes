"""Tests for portrait_map_lab.models dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_map_lab.models import (
    ComposeConfig,
    ComposedResult,
    ContourConfig,
    ContourResult,
    DensityResult,
    ETFConfig,
    ETFResult,
    FlowConfig,
    FlowResult,
    LandmarkResult,
    LICConfig,
    LuminanceConfig,
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


class TestLuminanceConfig:
    def test_defaults(self):
        config = LuminanceConfig()
        assert config.clip_limit == 2.0
        assert config.tile_size == 8

    def test_mutable(self):
        config = LuminanceConfig()
        config.clip_limit = 3.0
        assert config.clip_limit == 3.0
        config.tile_size = 16
        assert config.tile_size == 16


class TestComposeConfig:
    def test_defaults(self):
        config = ComposeConfig()
        assert isinstance(config.luminance, LuminanceConfig)
        assert config.luminance.clip_limit == 2.0
        assert config.feature_weight == 0.6
        assert config.contour_weight == 0.4
        assert config.tonal_blend_mode == "multiply"
        assert config.tonal_weight == 1.0
        assert config.importance_weight == 1.0
        assert config.gamma == 1.0

    def test_mutable(self):
        config = ComposeConfig()
        config.gamma = 2.0
        assert config.gamma == 2.0
        config.feature_weight = 0.8
        assert config.feature_weight == 0.8

    def test_independent_defaults(self):
        a = ComposeConfig()
        b = ComposeConfig()
        a.luminance.clip_limit = 5.0
        assert b.luminance.clip_limit == 2.0  # Should still be default value


class TestETFConfig:
    def test_defaults(self):
        config = ETFConfig()
        assert config.blur_sigma == 1.5
        assert config.structure_sigma == 5.0
        assert config.refine_sigma == 3.0
        assert config.refine_iterations == 2
        assert config.sobel_ksize == 3

    def test_mutable(self):
        config = ETFConfig()
        config.blur_sigma = 2.0
        assert config.blur_sigma == 2.0
        config.refine_iterations = 3
        assert config.refine_iterations == 3


class TestFlowConfig:
    def test_defaults(self):
        config = FlowConfig()
        assert isinstance(config.etf, ETFConfig)
        assert config.etf.blur_sigma == 1.5
        assert config.contour_smooth_sigma == 1.0
        assert config.blend_mode == "coherence"
        assert config.coherence_power == 2.0
        assert config.fallback_threshold == 0.1

    def test_mutable(self):
        config = FlowConfig()
        config.coherence_power = 3.0
        assert config.coherence_power == 3.0
        config.blend_mode = "linear"
        assert config.blend_mode == "linear"

    def test_independent_defaults(self):
        a = FlowConfig()
        b = FlowConfig()
        a.etf.blur_sigma = 3.0
        assert b.etf.blur_sigma == 1.5  # Should still be default value


class TestLICConfig:
    def test_defaults(self):
        config = LICConfig()
        assert config.length == 30
        assert config.step == 1.0
        assert config.seed == 42
        assert config.use_bilinear is True

    def test_mutable(self):
        config = LICConfig()
        config.length = 50
        assert config.length == 50
        config.use_bilinear = False
        assert config.use_bilinear is False


class TestDensityResult:
    def test_creation(self):
        result = DensityResult(
            luminance=np.zeros((100, 100), dtype=np.float64),
            clahe_luminance=np.zeros((100, 100), dtype=np.float64),
            tonal_target=np.zeros((100, 100), dtype=np.float64),
            importance=np.zeros((100, 100), dtype=np.float64),
            density_target=np.zeros((100, 100), dtype=np.float64),
        )
        assert result.luminance.shape == (100, 100)
        assert result.clahe_luminance.shape == (100, 100)
        assert result.tonal_target.shape == (100, 100)
        assert result.importance.shape == (100, 100)
        assert result.density_target.shape == (100, 100)

    def test_frozen(self):
        result = DensityResult(
            luminance=np.zeros((50, 50)),
            clahe_luminance=np.zeros((50, 50)),
            tonal_target=np.zeros((50, 50)),
            importance=np.zeros((50, 50)),
            density_target=np.zeros((50, 50)),
        )
        with pytest.raises(AttributeError):
            result.luminance = np.ones((50, 50))  # type: ignore[misc]


class TestETFResult:
    def test_creation(self):
        result = ETFResult(
            tangent_x=np.zeros((100, 100), dtype=np.float64),
            tangent_y=np.zeros((100, 100), dtype=np.float64),
            coherence=np.zeros((100, 100), dtype=np.float64),
            gradient_magnitude=np.zeros((100, 100), dtype=np.float64),
        )
        assert result.tangent_x.shape == (100, 100)
        assert result.tangent_y.shape == (100, 100)
        assert result.coherence.shape == (100, 100)
        assert result.gradient_magnitude.shape == (100, 100)

    def test_frozen(self):
        result = ETFResult(
            tangent_x=np.zeros((50, 50)),
            tangent_y=np.zeros((50, 50)),
            coherence=np.zeros((50, 50)),
            gradient_magnitude=np.zeros((50, 50)),
        )
        with pytest.raises(AttributeError):
            result.tangent_x = np.ones((50, 50))  # type: ignore[misc]


class TestFlowResult:
    def test_creation(self):
        etf_result = ETFResult(
            tangent_x=np.zeros((100, 100)),
            tangent_y=np.zeros((100, 100)),
            coherence=np.zeros((100, 100)),
            gradient_magnitude=np.zeros((100, 100)),
        )
        result = FlowResult(
            etf=etf_result,
            contour_flow_x=np.zeros((100, 100), dtype=np.float64),
            contour_flow_y=np.zeros((100, 100), dtype=np.float64),
            blend_weight=np.zeros((100, 100), dtype=np.float64),
            flow_x=np.zeros((100, 100), dtype=np.float64),
            flow_y=np.zeros((100, 100), dtype=np.float64),
        )
        assert result.etf.tangent_x.shape == (100, 100)
        assert result.contour_flow_x.shape == (100, 100)
        assert result.contour_flow_y.shape == (100, 100)
        assert result.blend_weight.shape == (100, 100)
        assert result.flow_x.shape == (100, 100)
        assert result.flow_y.shape == (100, 100)

    def test_frozen(self):
        etf_result = ETFResult(
            tangent_x=np.zeros((50, 50)),
            tangent_y=np.zeros((50, 50)),
            coherence=np.zeros((50, 50)),
            gradient_magnitude=np.zeros((50, 50)),
        )
        result = FlowResult(
            etf=etf_result,
            contour_flow_x=np.zeros((50, 50)),
            contour_flow_y=np.zeros((50, 50)),
            blend_weight=np.zeros((50, 50)),
            flow_x=np.zeros((50, 50)),
            flow_y=np.zeros((50, 50)),
        )
        with pytest.raises(AttributeError):
            result.flow_x = np.ones((50, 50))  # type: ignore[misc]


class TestComposedResult:
    def test_creation(self):
        landmarks = LandmarkResult(
            landmarks=np.zeros((478, 2)), image_shape=(100, 100), confidence=0.9
        )
        feature_result = PipelineResult(
            landmarks=landmarks,
            masks={},
            distance_fields={},
            influence_maps={},
            combined=np.zeros((100, 100)),
        )
        contour_result = ContourResult(
            landmarks=landmarks,
            contour_polygon=np.zeros((36, 2)),
            contour_mask=np.zeros((100, 100)),
            filled_mask=np.zeros((100, 100)),
            signed_distance=np.zeros((100, 100)),
            directional_distance=np.zeros((100, 100)),
            influence_map=np.zeros((100, 100)),
        )
        density_result = DensityResult(
            luminance=np.zeros((100, 100)),
            clahe_luminance=np.zeros((100, 100)),
            tonal_target=np.zeros((100, 100)),
            importance=np.zeros((100, 100)),
            density_target=np.zeros((100, 100)),
        )
        etf_result = ETFResult(
            tangent_x=np.zeros((100, 100)),
            tangent_y=np.zeros((100, 100)),
            coherence=np.zeros((100, 100)),
            gradient_magnitude=np.zeros((100, 100)),
        )
        flow_result = FlowResult(
            etf=etf_result,
            contour_flow_x=np.zeros((100, 100)),
            contour_flow_y=np.zeros((100, 100)),
            blend_weight=np.zeros((100, 100)),
            flow_x=np.zeros((100, 100)),
            flow_y=np.zeros((100, 100)),
        )
        result = ComposedResult(
            feature_result=feature_result,
            contour_result=contour_result,
            density_result=density_result,
            flow_result=flow_result,
            lic_image=np.zeros((100, 100), dtype=np.float64),
        )
        assert result.feature_result.landmarks.image_shape == (100, 100)
        assert result.contour_result.signed_distance.shape == (100, 100)
        assert result.density_result.density_target.shape == (100, 100)
        assert result.flow_result.flow_x.shape == (100, 100)
        assert result.lic_image.shape == (100, 100)

    def test_frozen(self):
        landmarks = LandmarkResult(
            landmarks=np.zeros((10, 2)), image_shape=(50, 50), confidence=0.8
        )
        feature_result = PipelineResult(
            landmarks=landmarks,
            masks={},
            distance_fields={},
            influence_maps={},
            combined=np.zeros((50, 50)),
        )
        contour_result = ContourResult(
            landmarks=landmarks,
            contour_polygon=np.zeros((36, 2)),
            contour_mask=np.zeros((50, 50)),
            filled_mask=np.zeros((50, 50)),
            signed_distance=np.zeros((50, 50)),
            directional_distance=np.zeros((50, 50)),
            influence_map=np.zeros((50, 50)),
        )
        density_result = DensityResult(
            luminance=np.zeros((50, 50)),
            clahe_luminance=np.zeros((50, 50)),
            tonal_target=np.zeros((50, 50)),
            importance=np.zeros((50, 50)),
            density_target=np.zeros((50, 50)),
        )
        etf_result = ETFResult(
            tangent_x=np.zeros((50, 50)),
            tangent_y=np.zeros((50, 50)),
            coherence=np.zeros((50, 50)),
            gradient_magnitude=np.zeros((50, 50)),
        )
        flow_result = FlowResult(
            etf=etf_result,
            contour_flow_x=np.zeros((50, 50)),
            contour_flow_y=np.zeros((50, 50)),
            blend_weight=np.zeros((50, 50)),
            flow_x=np.zeros((50, 50)),
            flow_y=np.zeros((50, 50)),
        )
        result = ComposedResult(
            feature_result=feature_result,
            contour_result=contour_result,
            density_result=density_result,
            flow_result=flow_result,
            lic_image=np.zeros((50, 50)),
        )
        with pytest.raises(AttributeError):
            result.lic_image = np.ones((50, 50))  # type: ignore[misc]
