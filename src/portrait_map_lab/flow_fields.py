"""Flow field computation combining Edge Tangent Fields with contour-based flow."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from portrait_map_lab.models import FlowConfig

__all__ = [
    "compute_contour_flow",
    "align_tangent_field",
    "compute_blend_weight",
    "blend_flow_fields",
]


def compute_contour_flow(
    signed_distance: np.ndarray,
    smooth_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute flow field from signed distance field gradient.

    Creates a flow field perpendicular to the contour by rotating the
    gradient of the signed distance field by 90 degrees counter-clockwise.
    This produces flow that follows the contour lines.

    Parameters
    ----------
    signed_distance
        Signed distance field as float64 array.
    smooth_sigma
        Standard deviation for optional Gaussian smoothing. Set to 0 to skip.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flow field components (flow_x, flow_y) as unit vectors.
    """
    # Compute gradient of signed distance field
    # np.gradient returns (grad_y, grad_x) for 2D arrays
    grad_y, grad_x = np.gradient(signed_distance)

    # Rotate 90 degrees counter-clockwise to get flow perpendicular to gradient
    # Rotation: (x, y) -> (-y, x)
    flow_x = -grad_y
    flow_y = grad_x

    # Normalize to unit vectors
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    # Avoid division by zero with small epsilon
    safe_magnitude = np.maximum(magnitude, 1e-10)
    flow_x = flow_x / safe_magnitude
    flow_y = flow_y / safe_magnitude

    # Optional smoothing with re-normalization
    if smooth_sigma > 0:
        flow_x = gaussian_filter(flow_x, sigma=smooth_sigma)
        flow_y = gaussian_filter(flow_y, sigma=smooth_sigma)

        # Re-normalize after smoothing
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        safe_magnitude = np.maximum(magnitude, 1e-10)
        flow_x = flow_x / safe_magnitude
        flow_y = flow_y / safe_magnitude

    return flow_x, flow_y


def align_tangent_field(
    tx: np.ndarray,
    ty: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align tangent field vectors to a reference flow field.

    Resolves the 180-degree ambiguity in eigenvector direction by ensuring
    tangent vectors point in the same general direction as the reference.
    Vectors with negative dot product are flipped.

    Parameters
    ----------
    tx
        Tangent field x-component.
    ty
        Tangent field y-component.
    ref_x
        Reference flow field x-component.
    ref_y
        Reference flow field y-component.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Aligned tangent field components (tx_aligned, ty_aligned).
    """
    # Compute dot product with reference field
    dot_product = tx * ref_x + ty * ref_y

    # Create copies to avoid modifying input
    tx_aligned = tx.copy()
    ty_aligned = ty.copy()

    # Flip vectors where dot product is negative
    flip_mask = dot_product < 0
    tx_aligned[flip_mask] = -tx_aligned[flip_mask]
    ty_aligned[flip_mask] = -ty_aligned[flip_mask]

    return tx_aligned, ty_aligned


def compute_blend_weight(
    coherence: np.ndarray,
    config: FlowConfig | None = None,
) -> np.ndarray:
    """Compute blend weight from ETF coherence values.

    Higher coherence (stronger edges) produces higher blend weight,
    giving more influence to the ETF. Lower coherence (flat regions)
    produces lower weight, giving more influence to contour flow.

    Parameters
    ----------
    coherence
        ETF coherence values in [0, 1].
    config
        Flow configuration containing coherence_power parameter.
        If None, uses default configuration.

    Returns
    -------
    np.ndarray
        Blend weights in [0, 1] where 1 = full ETF, 0 = full contour.
    """
    if config is None:
        config = FlowConfig()

    # Apply power function to control blend curve
    alpha = coherence ** config.coherence_power

    # Ensure values are in valid range
    alpha = np.clip(alpha, 0.0, 1.0)

    return alpha


def blend_flow_fields(
    etf_tx: np.ndarray,
    etf_ty: np.ndarray,
    contour_fx: np.ndarray,
    contour_fy: np.ndarray,
    alpha: np.ndarray,
    fallback_threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend ETF and contour flow fields using coherence-based weights.

    Performs linear blending between ETF tangent field and contour flow,
    with fallback to pure contour flow in degenerate regions where
    opposing vectors cancel out.

    Parameters
    ----------
    etf_tx
        ETF tangent field x-component (unit vectors).
    etf_ty
        ETF tangent field y-component (unit vectors).
    contour_fx
        Contour flow field x-component (unit vectors).
    contour_fy
        Contour flow field y-component (unit vectors).
    alpha
        Blend weights in [0, 1] where 1 = full ETF, 0 = full contour.
    fallback_threshold
        Magnitude threshold below which to use pure contour flow.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Blended flow field components (flow_x, flow_y) as unit vectors.
    """
    # Linear blend
    blended_x = alpha * etf_tx + (1 - alpha) * contour_fx
    blended_y = alpha * etf_ty + (1 - alpha) * contour_fy

    # Check pre-normalization magnitude for fallback
    magnitude = np.sqrt(blended_x**2 + blended_y**2)

    # Fallback to contour flow where magnitude is too low (cancellation)
    fallback_mask = magnitude < fallback_threshold
    blended_x[fallback_mask] = contour_fx[fallback_mask]
    blended_y[fallback_mask] = contour_fy[fallback_mask]

    # Normalize to unit vectors
    magnitude = np.sqrt(blended_x**2 + blended_y**2)
    safe_magnitude = np.maximum(magnitude, 1e-10)
    flow_x = blended_x / safe_magnitude
    flow_y = blended_y / safe_magnitude

    return flow_x, flow_y
