"""Tests for portrait_map_lab.etf module."""

from __future__ import annotations

import numpy as np
import pytest

from portrait_map_lab.etf import (
    compute_etf,
    compute_structure_tensor,
    extract_tangent_field,
    refine_tangent_field,
)
from portrait_map_lab.models import ETFConfig, ETFResult


class TestComputeStructureTensor:
    """Tests for compute_structure_tensor function."""

    def test_output_shapes(self):
        """Test that structure tensor components have correct shapes."""
        gray = np.random.rand(100, 100).astype(np.float64)
        Jxx, Jxy, Jyy = compute_structure_tensor(
            gray, blur_sigma=1.0, structure_sigma=2.0, sobel_ksize=3
        )

        assert Jxx.shape == gray.shape
        assert Jxy.shape == gray.shape
        assert Jyy.shape == gray.shape

    def test_output_dtype(self):
        """Test that structure tensor components are float64."""
        gray = np.random.rand(50, 50).astype(np.float64)
        Jxx, Jxy, Jyy = compute_structure_tensor(
            gray, blur_sigma=1.0, structure_sigma=2.0, sobel_ksize=3
        )

        assert Jxx.dtype == np.float64
        assert Jxy.dtype == np.float64
        assert Jyy.dtype == np.float64

    def test_uniform_image(self):
        """Test that uniform image produces near-zero gradients."""
        gray = np.ones((50, 50), dtype=np.float64) * 0.5
        Jxx, Jxy, Jyy = compute_structure_tensor(
            gray, blur_sigma=1.0, structure_sigma=2.0, sobel_ksize=3
        )

        # All tensor components should be near zero for uniform image
        assert np.allclose(Jxx, 0.0, atol=1e-10)
        assert np.allclose(Jxy, 0.0, atol=1e-10)
        assert np.allclose(Jyy, 0.0, atol=1e-10)

    def test_zero_sigma(self):
        """Test that zero sigma values skip smoothing."""
        gray = np.random.rand(30, 30).astype(np.float64)
        Jxx, Jxy, Jyy = compute_structure_tensor(
            gray, blur_sigma=0.0, structure_sigma=0.0, sobel_ksize=3
        )

        # Should still produce valid output
        assert Jxx.shape == gray.shape
        assert not np.isnan(Jxx).any()


class TestExtractTangentField:
    """Tests for extract_tangent_field function."""

    def test_unit_length(self):
        """Test that output tangent vectors have unit length."""
        # Create random structure tensor components
        shape = (50, 50)
        Jxx = np.random.rand(*shape)
        Jxy = np.random.rand(*shape)
        Jyy = np.random.rand(*shape)

        tx, ty, coherence = extract_tangent_field(Jxx, Jxy, Jyy)

        # Check unit length
        magnitude = np.sqrt(tx * tx + ty * ty)
        assert np.allclose(magnitude, 1.0, atol=1e-6)

    def test_coherence_range(self):
        """Test that coherence values are in [0, 1]."""
        shape = (50, 50)
        Jxx = np.random.rand(*shape)
        Jxy = np.random.rand(*shape)
        Jyy = np.random.rand(*shape)

        tx, ty, coherence = extract_tangent_field(Jxx, Jxy, Jyy)

        assert coherence.min() >= 0.0
        assert coherence.max() <= 1.0

    def test_zero_tensor(self):
        """Test handling of zero structure tensor."""
        shape = (30, 30)
        Jxx = np.zeros(shape)
        Jxy = np.zeros(shape)
        Jyy = np.zeros(shape)

        tx, ty, coherence = extract_tangent_field(Jxx, Jxy, Jyy)

        # Should handle gracefully without NaN or Inf
        assert not np.isnan(tx).any()
        assert not np.isnan(ty).any()
        assert not np.isnan(coherence).any()
        assert not np.isinf(tx).any()
        assert not np.isinf(ty).any()
        assert not np.isinf(coherence).any()

        # Coherence should be near zero for zero tensor
        assert coherence.max() < 0.01


class TestRefineTangentField:
    """Tests for refine_tangent_field function."""

    def test_preserves_unit_length(self):
        """Test that refinement preserves unit length."""
        # Create random unit vectors
        angles = np.random.rand(40, 40) * 2 * np.pi
        tx = np.cos(angles)
        ty = np.sin(angles)

        tx_refined, ty_refined = refine_tangent_field(tx, ty, sigma=2.0, iterations=3)

        # Check unit length after refinement
        magnitude = np.sqrt(tx_refined * tx_refined + ty_refined * ty_refined)
        assert np.allclose(magnitude, 1.0, atol=1e-6)

    def test_zero_iterations(self):
        """Test that zero iterations returns input unchanged."""
        tx = np.random.rand(30, 30)
        ty = np.random.rand(30, 30)

        # Normalize input
        mag = np.sqrt(tx * tx + ty * ty)
        tx = tx / np.maximum(mag, 1e-10)
        ty = ty / np.maximum(mag, 1e-10)

        tx_refined, ty_refined = refine_tangent_field(tx, ty, sigma=2.0, iterations=0)

        assert np.allclose(tx_refined, tx)
        assert np.allclose(ty_refined, ty)

    def test_zero_sigma(self):
        """Test that zero sigma performs only renormalization."""
        # Create non-unit vectors
        tx = np.random.rand(30, 30) * 2
        ty = np.random.rand(30, 30) * 2

        tx_refined, ty_refined = refine_tangent_field(tx, ty, sigma=0.0, iterations=1)

        # Should be normalized but not smoothed
        magnitude = np.sqrt(tx_refined * tx_refined + ty_refined * ty_refined)
        assert np.allclose(magnitude, 1.0, atol=1e-6)


class TestComputeETF:
    """Tests for compute_etf main function."""

    def test_bgr_input(self):
        """Test ETF computation with BGR input."""
        bgr_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        config = ETFConfig(refine_iterations=1)

        result = compute_etf(bgr_image, config)

        assert isinstance(result, ETFResult)
        assert result.tangent_x.shape == (100, 100)
        assert result.tangent_y.shape == (100, 100)
        assert result.coherence.shape == (100, 100)
        assert result.gradient_magnitude.shape == (100, 100)

    def test_grayscale_input(self):
        """Test ETF computation with grayscale input."""
        gray_image = np.random.rand(80, 80).astype(np.float64)
        result = compute_etf(gray_image)

        assert isinstance(result, ETFResult)
        assert result.tangent_x.shape == gray_image.shape
        assert result.tangent_y.shape == gray_image.shape

    def test_uint8_grayscale_input(self):
        """Test ETF computation with uint8 grayscale input."""
        gray_image = np.random.randint(0, 256, (60, 60), dtype=np.uint8)
        result = compute_etf(gray_image)

        assert isinstance(result, ETFResult)
        assert result.tangent_x.shape == gray_image.shape

    def test_default_config(self):
        """Test that default config produces reasonable results."""
        image = np.random.rand(50, 50)
        result = compute_etf(image)

        # Check unit vectors
        magnitude = np.sqrt(result.tangent_x**2 + result.tangent_y**2)
        assert np.allclose(magnitude, 1.0, atol=1e-6)

        # Check coherence range
        assert result.coherence.min() >= 0.0
        assert result.coherence.max() <= 1.0

    def test_uniform_image_low_coherence(self):
        """Test that uniform image produces low coherence."""
        # Create uniform gray image
        uniform_image = np.ones((50, 50), dtype=np.float64) * 0.5
        result = compute_etf(uniform_image)

        # Coherence should be very low (no edges)
        assert result.coherence.mean() < 0.1
        assert result.coherence.max() < 0.2

    def test_vertical_edge_tangent_direction(self):
        """Test that vertical edge produces approximately vertical tangent vectors."""
        # Create image with vertical edge
        image = np.zeros((100, 100), dtype=np.float64)
        image[:, :50] = 1.0  # Left half white, right half black

        config = ETFConfig(blur_sigma=0.5, structure_sigma=2.0)
        result = compute_etf(image, config)

        # Near the edge (around column 50), tangent should be mostly vertical
        # Check a vertical strip around the edge
        edge_region = result.tangent_y[:, 48:52]
        edge_tangent_x = result.tangent_x[:, 48:52]

        # Most tangent vectors should be more vertical than horizontal
        vertical_dominant = np.abs(edge_region) > np.abs(edge_tangent_x)
        assert vertical_dominant.sum() > 0.7 * vertical_dominant.size

    def test_horizontal_edge_tangent_direction(self):
        """Test that horizontal edge produces approximately horizontal tangent vectors."""
        # Create image with horizontal edge
        image = np.zeros((100, 100), dtype=np.float64)
        image[:50, :] = 1.0  # Top half white, bottom half black

        config = ETFConfig(blur_sigma=0.5, structure_sigma=2.0)
        result = compute_etf(image, config)

        # Near the edge (around row 50), tangent should be mostly horizontal
        # Check a wider strip around the edge to account for smoothing
        edge_region_x = result.tangent_x[45:55, 20:80]
        edge_region_y = result.tangent_y[45:55, 20:80]

        # Also check coherence to focus on actual edge regions
        edge_coherence = result.coherence[45:55, 20:80]

        # Only consider points with significant coherence (actual edges)
        significant_mask = edge_coherence > 0.1

        if significant_mask.any():
            # Most tangent vectors at significant points should be more horizontal than vertical
            horizontal_dominant = np.abs(edge_region_x[significant_mask]) > np.abs(
                edge_region_y[significant_mask]
            )
            assert horizontal_dominant.sum() > 0.6 * horizontal_dominant.size
        else:
            # If no significant edges detected, that's also a test failure
            assert False, "No significant edges detected in horizontal edge test"

    def test_high_values_normalization(self):
        """Test that high-valued grayscale images are properly normalized."""
        # Create image with values > 1
        image = np.random.rand(50, 50) * 100
        result = compute_etf(image)

        # Should still produce valid unit vectors
        magnitude = np.sqrt(result.tangent_x**2 + result.tangent_y**2)
        assert np.allclose(magnitude, 1.0, atol=1e-6)

    def test_invalid_input_shape(self):
        """Test that invalid input shape raises ValueError."""
        invalid_image = np.random.rand(10, 10, 10, 10)  # 4D array
        with pytest.raises(ValueError, match="Expected 2D or 3D array"):
            compute_etf(invalid_image)
