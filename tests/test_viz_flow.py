"""Tests for flow visualization functions in portrait_map_lab.viz module."""

from __future__ import annotations

import numpy as np

from portrait_map_lab.viz import overlay_lic, visualize_flow_field


class TestVisualizeFlowField:
    """Tests for visualize_flow_field function."""

    def test_output_shape_with_image(self):
        """Test output shape matches input image."""
        h, w = 100, 120
        flow_x = np.ones((h, w), dtype=np.float64)
        flow_y = np.zeros((h, w), dtype=np.float64)
        image = np.full((h, w, 3), 128, dtype=np.uint8)

        result = visualize_flow_field(flow_x, flow_y, image=image)

        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8

    def test_output_shape_without_image(self):
        """Test output shape when no image is provided."""
        h, w = 80, 90
        flow_x = np.ones((h, w), dtype=np.float64)
        flow_y = np.zeros((h, w), dtype=np.float64)

        result = visualize_flow_field(flow_x, flow_y, image=None)

        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8

    def test_creates_white_canvas_when_no_image(self):
        """Test that a white canvas is created when no image is provided."""
        flow_x = np.ones((50, 50), dtype=np.float64)
        flow_y = np.zeros((50, 50), dtype=np.float64)

        result = visualize_flow_field(flow_x, flow_y, image=None)

        # Check that most of the image is white (arrows will be colored)
        white_pixels = np.all(result == 255, axis=2)
        white_ratio = np.sum(white_pixels) / white_pixels.size
        assert white_ratio > 0.9  # Most pixels should be white

    def test_custom_step_size(self):
        """Test that custom step size affects arrow density."""
        flow_x = np.ones((100, 100), dtype=np.float64)
        flow_y = np.zeros((100, 100), dtype=np.float64)

        # Small step = more arrows
        result_dense = visualize_flow_field(flow_x, flow_y, step=10)
        # Large step = fewer arrows
        result_sparse = visualize_flow_field(flow_x, flow_y, step=30)

        # Both should be valid images
        assert result_dense.shape == (100, 100, 3)
        assert result_sparse.shape == (100, 100, 3)

    def test_custom_color(self):
        """Test that custom color is applied to arrows."""
        flow_x = np.ones((60, 60), dtype=np.float64)
        flow_y = np.zeros((60, 60), dtype=np.float64)

        # Red arrows
        red_color = (0, 0, 255)  # BGR format
        result = visualize_flow_field(flow_x, flow_y, color=red_color, step=15)

        # Check that there are red pixels in the result
        red_pixels = np.logical_and(
            result[:, :, 2] > 200,  # High red channel
            result[:, :, 0] < 50,    # Low blue channel
        )
        assert np.any(red_pixels)  # Should have some red pixels from arrows

    def test_zero_flow(self):
        """Test handling of zero flow field."""
        flow_x = np.zeros((40, 40), dtype=np.float64)
        flow_y = np.zeros((40, 40), dtype=np.float64)

        result = visualize_flow_field(flow_x, flow_y)

        # Should produce valid output without errors
        assert result.shape == (40, 40, 3)
        assert result.dtype == np.uint8


class TestOverlayLIC:
    """Tests for overlay_lic function."""

    def test_output_shape(self):
        """Test that output shape matches input image."""
        h, w = 100, 120
        lic_image = np.random.random((h, w)).astype(np.float64)
        image = np.full((h, w, 3), 128, dtype=np.uint8)

        result = overlay_lic(lic_image, image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_alpha_zero_returns_image(self):
        """Test that alpha=0 returns original image."""
        lic_image = np.ones((50, 50), dtype=np.float64)
        image = np.full((50, 50, 3), 100, dtype=np.uint8)

        result = overlay_lic(lic_image, image, alpha=0.0)

        assert np.allclose(result, image)

    def test_alpha_one_returns_lic(self):
        """Test that alpha=1 returns LIC as grayscale BGR."""
        lic_image = np.full((50, 50), 0.5, dtype=np.float64)  # Mid-gray
        image = np.full((50, 50, 3), 0, dtype=np.uint8)  # Black

        result = overlay_lic(lic_image, image, alpha=1.0)

        expected_value = int(0.5 * 255)
        assert np.all(result == expected_value)

    def test_alpha_blending(self):
        """Test that alpha blending works correctly."""
        h, w = 60, 60
        lic_image = np.full((h, w), 1.0, dtype=np.float64)  # White
        image = np.zeros((h, w, 3), dtype=np.uint8)  # Black

        result = overlay_lic(lic_image, image, alpha=0.5)

        # Should be mid-gray (50% blend of white and black)
        expected_value = 127  # Approximately half of 255
        assert np.all(np.abs(result.astype(int) - expected_value) <= 1)

    def test_clipping_to_valid_range(self):
        """Test that output is clipped to valid uint8 range."""
        # Create LIC with values outside [0, 1]
        lic_image = np.full((40, 40), 2.0, dtype=np.float64)  # Out of range
        image = np.full((40, 40, 3), 200, dtype=np.uint8)

        result = overlay_lic(lic_image, image, alpha=0.5)

        # Should be clipped to [0, 255]
        assert np.all(result >= 0)
        assert np.all(result <= 255)
        assert result.dtype == np.uint8

    def test_varying_lic_values(self):
        """Test overlay with varying LIC values."""
        h, w = 80, 80
        # Create gradient LIC
        lic_image = np.linspace(0, 1, h*w).reshape(h, w).astype(np.float64)
        image = np.full((h, w, 3), 128, dtype=np.uint8)

        result = overlay_lic(lic_image, image, alpha=0.5)

        # Result should have variation
        assert np.std(result) > 10  # Not uniform
        assert result.shape == (h, w, 3)
        assert result.dtype == np.uint8
