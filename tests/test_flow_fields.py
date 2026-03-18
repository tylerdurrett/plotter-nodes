"""Tests for portrait_map_lab.flow_fields module."""

from __future__ import annotations

import numpy as np

from portrait_map_lab.flow_fields import (
    align_tangent_field,
    blend_flow_fields,
    compute_blend_weight,
    compute_contour_flow,
)
from portrait_map_lab.models import FlowConfig


class TestComputeContourFlow:
    """Tests for compute_contour_flow function."""

    def test_unit_length(self):
        """Test that output flow vectors are unit length."""
        # Create a simple signed distance field
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        signed_distance = np.sqrt(X**2 + Y**2) - 3.0  # Circle of radius 3

        flow_x, flow_y = compute_contour_flow(signed_distance, smooth_sigma=0.0)

        # Compute magnitude
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # All vectors should be unit length (within numerical tolerance)
        assert np.allclose(magnitude, 1.0, atol=1e-6)

    def test_perpendicular_to_gradient(self):
        """Test that flow is perpendicular to the distance gradient."""
        # Create a signed distance field with clear gradient
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        signed_distance = X  # Linear gradient in x direction

        flow_x, flow_y = compute_contour_flow(signed_distance, smooth_sigma=0.0)

        # Gradient should be (1, 0) everywhere
        # Flow should be perpendicular: (0, 1)
        # Allow some tolerance near boundaries
        center_slice = slice(5, -5)
        assert np.allclose(flow_x[center_slice, center_slice], 0.0, atol=0.1)
        assert np.allclose(flow_y[center_slice, center_slice], 1.0, atol=0.1)

    def test_output_shape(self):
        """Test that output shape matches input."""
        signed_distance = np.random.randn(100, 80).astype(np.float64)
        flow_x, flow_y = compute_contour_flow(signed_distance)

        assert flow_x.shape == signed_distance.shape
        assert flow_y.shape == signed_distance.shape

    def test_smoothing_preserves_unit_length(self):
        """Test that smoothing still produces unit vectors."""
        signed_distance = np.random.randn(50, 50).astype(np.float64)
        flow_x, flow_y = compute_contour_flow(signed_distance, smooth_sigma=2.0)

        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        assert np.allclose(magnitude, 1.0, atol=1e-6)


class TestAlignTangentField:
    """Tests for align_tangent_field function."""

    def test_flips_opposing_vectors(self):
        """Test that opposing vectors get flipped."""
        # Create tangent field pointing right
        tx = np.ones((10, 10), dtype=np.float64)
        ty = np.zeros((10, 10), dtype=np.float64)

        # Create reference field
        ref_x = np.ones((10, 10), dtype=np.float64)
        ref_y = np.zeros((10, 10), dtype=np.float64)

        # Make half the reference field point left (opposite)
        ref_x[:, 5:] = -1.0

        tx_aligned, ty_aligned = align_tangent_field(tx, ty, ref_x, ref_y)

        # First half should be unchanged (already aligned)
        assert np.allclose(tx_aligned[:, :5], 1.0)
        assert np.allclose(ty_aligned[:, :5], 0.0)

        # Second half should be flipped (was opposing)
        assert np.allclose(tx_aligned[:, 5:], -1.0)
        assert np.allclose(ty_aligned[:, 5:], 0.0)

    def test_preserves_aligned_vectors(self):
        """Test that already-aligned vectors are unchanged."""
        # Create identical fields
        tx = np.random.randn(20, 20).astype(np.float64)
        ty = np.random.randn(20, 20).astype(np.float64)

        # Normalize
        mag = np.sqrt(tx**2 + ty**2)
        tx = tx / np.maximum(mag, 1e-10)
        ty = ty / np.maximum(mag, 1e-10)

        ref_x = tx.copy()
        ref_y = ty.copy()

        tx_aligned, ty_aligned = align_tangent_field(tx, ty, ref_x, ref_y)

        # Should be unchanged
        assert np.allclose(tx_aligned, tx)
        assert np.allclose(ty_aligned, ty)

    def test_dot_product_threshold(self):
        """Test that flip occurs exactly at dot product < 0."""
        # Create vectors at various angles
        angles = np.linspace(0, 2*np.pi, 20)
        tx = np.cos(angles).reshape(-1, 1)
        ty = np.sin(angles).reshape(-1, 1)

        # Reference pointing right
        ref_x = np.ones_like(tx)
        ref_y = np.zeros_like(ty)

        tx_aligned, ty_aligned = align_tangent_field(tx, ty, ref_x, ref_y)

        # Check that all aligned vectors have positive dot product with reference
        dot_product = tx_aligned * ref_x + ty_aligned * ref_y
        assert np.all(dot_product >= 0)


class TestComputeBlendWeight:
    """Tests for compute_blend_weight function."""

    def test_output_range(self):
        """Test that output is in [0, 1]."""
        coherence = np.random.rand(30, 30).astype(np.float64)
        config = FlowConfig(coherence_power=2.0)

        alpha = compute_blend_weight(coherence, config)

        assert np.all(alpha >= 0.0)
        assert np.all(alpha <= 1.0)

    def test_power_function(self):
        """Test that power function is correctly applied."""
        coherence = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
        config = FlowConfig(coherence_power=2.0)

        alpha = compute_blend_weight(coherence, config)

        assert np.allclose(alpha[0, 0], 0.0)
        assert np.allclose(alpha[0, 1], 0.25)  # 0.5^2
        assert np.allclose(alpha[0, 2], 1.0)

    def test_default_config(self):
        """Test that default config works."""
        coherence = np.random.rand(20, 20).astype(np.float64)

        # Should use default config
        alpha = compute_blend_weight(coherence, None)

        assert alpha.shape == coherence.shape
        assert np.all(alpha >= 0.0)
        assert np.all(alpha <= 1.0)

    def test_clipping(self):
        """Test that values outside [0,1] are clipped."""
        # Create coherence with some out-of-range values
        coherence = np.array([[-0.1, 0.5, 1.2]], dtype=np.float64)
        config = FlowConfig(coherence_power=1.0)

        alpha = compute_blend_weight(coherence, config)

        assert alpha[0, 0] == 0.0  # Clipped from negative
        assert alpha[0, 1] == 0.5  # Unchanged
        assert alpha[0, 2] == 1.0  # Clipped from > 1


class TestBlendFlowFields:
    """Tests for blend_flow_fields function."""

    def test_high_coherence_prefers_etf(self):
        """Test that high alpha (coherence) gives more weight to ETF."""
        etf_tx = np.ones((10, 10), dtype=np.float64)
        etf_ty = np.zeros((10, 10), dtype=np.float64)

        contour_fx = np.zeros((10, 10), dtype=np.float64)
        contour_fy = np.ones((10, 10), dtype=np.float64)

        # High alpha (0.9) should prefer ETF
        alpha = np.full((10, 10), 0.9, dtype=np.float64)

        flow_x, flow_y = blend_flow_fields(
            etf_tx, etf_ty, contour_fx, contour_fy, alpha, fallback_threshold=0.1
        )

        # Result should be closer to ETF (1, 0) than contour (0, 1)
        assert np.mean(flow_x) > np.mean(flow_y)

    def test_low_coherence_prefers_contour(self):
        """Test that low alpha (coherence) gives more weight to contour flow."""
        etf_tx = np.ones((10, 10), dtype=np.float64)
        etf_ty = np.zeros((10, 10), dtype=np.float64)

        contour_fx = np.zeros((10, 10), dtype=np.float64)
        contour_fy = np.ones((10, 10), dtype=np.float64)

        # Low alpha (0.1) should prefer contour
        alpha = np.full((10, 10), 0.1, dtype=np.float64)

        flow_x, flow_y = blend_flow_fields(
            etf_tx, etf_ty, contour_fx, contour_fy, alpha, fallback_threshold=0.1
        )

        # Result should be closer to contour (0, 1) than ETF (1, 0)
        assert np.mean(flow_y) > np.mean(flow_x)

    def test_fallback_on_cancellation(self):
        """Test fallback to contour flow when vectors cancel out."""
        # Create opposing vectors that will cancel with alpha=0.5
        etf_tx = np.ones((5, 5), dtype=np.float64)
        etf_ty = np.zeros((5, 5), dtype=np.float64)

        contour_fx = np.full((5, 5), -1.0, dtype=np.float64)
        contour_fy = np.zeros((5, 5), dtype=np.float64)

        # Alpha = 0.5 will cause cancellation
        alpha = np.full((5, 5), 0.5, dtype=np.float64)

        # Set low threshold to trigger fallback
        flow_x, flow_y = blend_flow_fields(
            etf_tx, etf_ty, contour_fx, contour_fy, alpha, fallback_threshold=0.1
        )

        # Should fallback to contour flow
        assert np.allclose(flow_x, -1.0, atol=1e-6)
        assert np.allclose(flow_y, 0.0, atol=1e-6)

    def test_output_unit_length(self):
        """Test that output vectors are unit length."""
        # Random input fields
        etf_tx = np.random.randn(20, 20).astype(np.float64)
        etf_ty = np.random.randn(20, 20).astype(np.float64)
        mag = np.sqrt(etf_tx**2 + etf_ty**2)
        etf_tx = etf_tx / np.maximum(mag, 1e-10)
        etf_ty = etf_ty / np.maximum(mag, 1e-10)

        contour_fx = np.random.randn(20, 20).astype(np.float64)
        contour_fy = np.random.randn(20, 20).astype(np.float64)
        mag = np.sqrt(contour_fx**2 + contour_fy**2)
        contour_fx = contour_fx / np.maximum(mag, 1e-10)
        contour_fy = contour_fy / np.maximum(mag, 1e-10)

        alpha = np.random.rand(20, 20).astype(np.float64)

        flow_x, flow_y = blend_flow_fields(
            etf_tx, etf_ty, contour_fx, contour_fy, alpha, fallback_threshold=0.1
        )

        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        assert np.allclose(magnitude, 1.0, atol=1e-6)

    def test_linear_interpolation(self):
        """Test that blending is linear interpolation for normal cases."""
        etf_tx = np.ones((3, 3), dtype=np.float64)
        etf_ty = np.zeros((3, 3), dtype=np.float64)

        contour_fx = np.zeros((3, 3), dtype=np.float64)
        contour_fy = np.ones((3, 3), dtype=np.float64)

        # Use different alpha values
        alpha = np.array([[0.0, 0.5, 1.0],
                         [0.25, 0.5, 0.75],
                         [0.0, 0.5, 1.0]], dtype=np.float64)

        # Use high threshold to avoid fallback
        flow_x, flow_y = blend_flow_fields(
            etf_tx, etf_ty, contour_fx, contour_fy, alpha, fallback_threshold=0.01
        )

        # Check corners where blend is simple
        # At alpha=0: should be contour (0, 1) normalized
        assert flow_x[0, 0] < 0.1  # Close to 0
        assert flow_y[0, 0] > 0.9  # Close to 1

        # At alpha=1: should be ETF (1, 0) normalized
        assert flow_x[0, 2] > 0.9  # Close to 1
        assert flow_y[0, 2] < 0.1  # Close to 0
