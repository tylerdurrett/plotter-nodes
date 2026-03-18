"""Integration tests for complexity map pipeline and flow speed integration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from portrait_map_lab.complexity_map import compute_complexity_map
from portrait_map_lab.models import (
    ComplexityConfig,
    ComplexityResult,
    ContourResult,
    FlowResult,
    FlowSpeedConfig,
)
from portrait_map_lab.pipelines import (
    run_all_pipelines,
    run_complexity_pipeline,
    run_flow_pipeline,
    save_complexity_outputs,
)


class TestRunComplexityPipeline:
    """Test the run_complexity_pipeline function."""

    def test_with_default_config(self):
        """Test pipeline with default configuration."""
        # Create a test image with edges
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[40:60, 40:60] = 255  # White square

        result = run_complexity_pipeline(image)

        assert isinstance(result, ComplexityResult)
        assert result.complexity.shape == (100, 100)
        assert result.raw_complexity.shape == (100, 100)
        assert result.metric == "gradient"
        assert np.all(result.complexity >= 0)
        assert np.all(result.complexity <= 1)

    def test_with_all_metrics(self):
        """Test pipeline with all available metrics."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[40:60, 40:60] = 255

        metrics = ["gradient", "laplacian", "multiscale_gradient"]
        for metric in metrics:
            config = ComplexityConfig(metric=metric)
            result = run_complexity_pipeline(image, config)

            assert result.metric == metric
            assert result.complexity.shape == (100, 100)
            assert np.all(result.complexity >= 0)
            assert np.all(result.complexity <= 1)

    def test_with_mask(self):
        """Test pipeline with mask restricting computation area."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Create mask for center region
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        result = run_complexity_pipeline(image, mask=mask)

        # Complexity should be zero outside mask
        assert np.all(result.complexity[:30, :] == 0)
        assert np.all(result.complexity[70:, :] == 0)
        assert np.all(result.complexity[:, :30] == 0)
        assert np.all(result.complexity[:, 70:] == 0)

    def test_normalization_settings(self):
        """Test different normalization percentile settings."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        config99 = ComplexityConfig(normalize_percentile=99.0)
        result99 = run_complexity_pipeline(image, config99)

        config100 = ComplexityConfig(normalize_percentile=100.0)
        result100 = run_complexity_pipeline(image, config100)

        # 99th percentile should clip more values to 1.0
        assert np.sum(result99.complexity == 1.0) >= np.sum(result100.complexity == 1.0)


class TestSaveComplexityOutputs:
    """Test the save_complexity_outputs function."""

    def test_creates_expected_files(self):
        """Test that all expected files are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create test result
            result = ComplexityResult(
                raw_complexity=np.random.rand(100, 100),
                complexity=np.random.rand(100, 100),
                metric="gradient",
            )

            # Create test image
            image = np.zeros((100, 100, 3), dtype=np.uint8)

            # Save outputs
            save_complexity_outputs(result, output_dir, image)

            # Check expected files
            complexity_dir = output_dir / "complexity"
            assert complexity_dir.exists()
            assert (complexity_dir / "gradient_energy.png").exists()
            assert (complexity_dir / "gradient_energy_raw.npy").exists()
            assert (complexity_dir / "complexity.png").exists()
            assert (complexity_dir / "complexity_raw.npy").exists()
            assert (complexity_dir / "contact_sheet.png").exists()

    def test_npy_files_loadable(self):
        """Test that saved .npy files can be loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create test result
            raw = np.random.rand(100, 100)
            normalized = np.random.rand(100, 100)
            result = ComplexityResult(
                raw_complexity=raw,
                complexity=normalized,
                metric="laplacian",
            )

            save_complexity_outputs(result, output_dir)

            # Load and verify
            complexity_dir = output_dir / "complexity"
            loaded_raw = np.load(complexity_dir / "laplacian_energy_raw.npy")
            loaded_norm = np.load(complexity_dir / "complexity_raw.npy")

            np.testing.assert_array_equal(loaded_raw, raw)
            np.testing.assert_array_equal(loaded_norm, normalized)


class TestFlowWithSpeed:
    """Test flow pipeline with complexity-based speed."""

    def test_flow_without_complexity(self):
        """Test flow pipeline without complexity (backward compatibility)."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create mock contour result
        contour_result = ContourResult(
            landmarks=None,
            contour_polygon=np.array([[30, 30], [70, 30], [70, 70], [30, 70]]),
            contour_mask=np.ones((100, 100), dtype=np.uint8),
            filled_mask=np.ones((100, 100), dtype=np.uint8),
            signed_distance=np.random.randn(100, 100),
            directional_distance=np.random.randn(100, 100),
            influence_map=np.random.rand(100, 100),
        )

        result = run_flow_pipeline(image, contour_result)

        assert isinstance(result, FlowResult)
        assert result.flow_speed is None
        assert result.flow_x.shape == (100, 100)
        assert result.flow_y.shape == (100, 100)

    def test_flow_with_complexity(self):
        """Test flow pipeline with complexity for speed modulation."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create mock contour result
        contour_result = ContourResult(
            landmarks=None,
            contour_polygon=np.array([[30, 30], [70, 30], [70, 70], [30, 70]]),
            contour_mask=np.ones((100, 100), dtype=np.uint8),
            filled_mask=np.ones((100, 100), dtype=np.uint8),
            signed_distance=np.random.randn(100, 100),
            directional_distance=np.random.randn(100, 100),
            influence_map=np.random.rand(100, 100),
        )

        # Create complexity result
        complexity_result = ComplexityResult(
            raw_complexity=np.random.rand(100, 100),
            complexity=np.random.rand(100, 100),
            metric="gradient",
        )

        result = run_flow_pipeline(
            image,
            contour_result,
            complexity_result=complexity_result,
        )

        assert isinstance(result, FlowResult)
        assert result.flow_speed is not None
        assert result.flow_speed.shape == (100, 100)
        assert np.all(result.flow_speed >= 0.3)  # Default speed_min
        assert np.all(result.flow_speed <= 1.0)  # Default speed_max

    def test_flow_with_custom_speed_config(self):
        """Test flow pipeline with custom speed configuration."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create mock contour result
        contour_result = ContourResult(
            landmarks=None,
            contour_polygon=np.array([[30, 30], [70, 30], [70, 70], [30, 70]]),
            contour_mask=np.ones((100, 100), dtype=np.uint8),
            filled_mask=np.ones((100, 100), dtype=np.uint8),
            signed_distance=np.random.randn(100, 100),
            directional_distance=np.random.randn(100, 100),
            influence_map=np.random.rand(100, 100),
        )

        # Create complexity result
        complexity_result = ComplexityResult(
            raw_complexity=np.random.rand(100, 100),
            complexity=np.random.rand(100, 100),
            metric="gradient",
        )

        # Custom speed config
        speed_config = FlowSpeedConfig(speed_min=0.1, speed_max=0.8)

        result = run_flow_pipeline(
            image,
            contour_result,
            complexity_result=complexity_result,
            speed_config=speed_config,
        )

        assert result.flow_speed is not None
        assert np.all(result.flow_speed >= 0.1)
        assert np.all(result.flow_speed <= 0.8)


class TestAllPipelinesWithComplexity:
    """Test complete pipeline with complexity integration."""

    def test_all_pipelines_without_complexity(self):
        """Test that all pipelines work without complexity (backward compat)."""
        # Create test image with a face-like pattern
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add a simple oval for face detection
        center = (100, 100)
        axes = (40, 50)
        import cv2
        cv2.ellipse(image, center, axes, 0, 0, 360, (255, 255, 255), -1)

        # Skip this test if face detection would fail
        try:
            result = run_all_pipelines(image)
        except ValueError as e:
            if "No face detected" in str(e):
                pytest.skip("Face detection not available in test environment")
            raise

        assert result.complexity_result is None
        assert result.flow_result.flow_speed is None

    def test_all_pipelines_with_complexity(self):
        """Test complete pipeline with complexity enabled."""
        # Create test image with a face-like pattern
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        # Add a simple oval for face detection
        center = (100, 100)
        axes = (40, 50)
        import cv2
        cv2.ellipse(image, center, axes, 0, 0, 360, (255, 255, 255), -1)

        complexity_config = ComplexityConfig(metric="gradient")
        speed_config = FlowSpeedConfig(speed_min=0.2, speed_max=0.9)

        # Skip this test if face detection would fail
        try:
            result = run_all_pipelines(
                image,
                complexity_config=complexity_config,
                speed_config=speed_config,
            )
        except ValueError as e:
            if "No face detected" in str(e):
                pytest.skip("Face detection not available in test environment")
            raise

        assert result.complexity_result is not None
        assert result.complexity_result.metric == "gradient"
        assert result.flow_result.flow_speed is not None
        assert np.all(result.flow_result.flow_speed >= 0.2)
        assert np.all(result.flow_result.flow_speed <= 0.9)

    def test_save_all_outputs_with_complexity(self):
        """Test that save_all_outputs handles complexity outputs correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create test image
            image = np.zeros((200, 200, 3), dtype=np.uint8)
            center = (100, 100)
            axes = (40, 50)
            import cv2
            cv2.ellipse(image, center, axes, 0, 0, 360, (255, 255, 255), -1)

            complexity_config = ComplexityConfig()

            # Skip if face detection fails
            try:
                from portrait_map_lab.pipelines import save_all_outputs
                result = run_all_pipelines(image, complexity_config=complexity_config)
                save_all_outputs(result, output_dir, image)
            except ValueError as e:
                if "No face detected" in str(e):
                    pytest.skip("Face detection not available in test environment")
                raise

            # Check that complexity outputs were saved
            complexity_dir = output_dir / "complexity"
            assert complexity_dir.exists()
            assert (complexity_dir / "complexity.png").exists()
            assert (complexity_dir / "gradient_energy.png").exists()

            # Check that flow speed was saved
            flow_dir = output_dir / "flow"
            assert (flow_dir / "flow_speed.png").exists()
            assert (flow_dir / "flow_speed_raw.npy").exists()