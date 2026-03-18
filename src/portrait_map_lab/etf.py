"""Edge Tangent Field computation for flow field generation."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from portrait_map_lab.models import ETFConfig, ETFResult

__all__ = [
    "compute_structure_tensor",
    "extract_tangent_field",
    "refine_tangent_field",
    "compute_etf",
]


def compute_structure_tensor(
    gray: np.ndarray,
    blur_sigma: float,
    structure_sigma: float,
    sobel_ksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the structure tensor from a grayscale image.

    The structure tensor encodes local gradient information and is fundamental
    for edge and corner detection algorithms.

    Parameters
    ----------
    gray
        Grayscale input image as float64 array with values in [0, 1].
    blur_sigma
        Standard deviation for Gaussian blur pre-processing.
    structure_sigma
        Standard deviation for smoothing structure tensor components.
    sobel_ksize
        Size of the Sobel kernel (1, 3, 5, or 7).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Structure tensor components (Jxx, Jxy, Jyy) as float64 arrays.
    """
    # Apply Gaussian blur to reduce noise
    if blur_sigma > 0:
        blurred = gaussian_filter(gray, sigma=blur_sigma)
    else:
        blurred = gray.copy()

    # Compute gradients using Sobel operator
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    # Compute structure tensor components
    Jxx = gx * gx
    Jxy = gx * gy
    Jyy = gy * gy

    # Smooth the structure tensor components
    if structure_sigma > 0:
        Jxx = gaussian_filter(Jxx, sigma=structure_sigma)
        Jxy = gaussian_filter(Jxy, sigma=structure_sigma)
        Jyy = gaussian_filter(Jyy, sigma=structure_sigma)

    return Jxx, Jxy, Jyy


def extract_tangent_field(
    Jxx: np.ndarray, Jxy: np.ndarray, Jyy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract tangent field and coherence from structure tensor.

    Computes eigenvalues and eigenvectors of the structure tensor using
    closed-form solutions, then extracts the minor eigenvector (perpendicular
    to the gradient) as the tangent direction.

    Parameters
    ----------
    Jxx
        Structure tensor xx component.
    Jxy
        Structure tensor xy component.
    Jyy
        Structure tensor yy component.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tangent field components (tx, ty) as unit vectors and coherence in [0, 1].
    """
    # Compute eigenvalues using closed-form solution
    # For 2x2 symmetric matrix: λ = trace/2 ± sqrt((trace/2)² - det)
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy

    # Discriminant
    disc = np.sqrt(np.maximum(0.0, (trace / 2.0) ** 2 - det))

    # Eigenvalues (λ₁ >= λ₂)
    lambda1 = trace / 2.0 + disc
    lambda2 = trace / 2.0 - disc

    # Minor eigenvector corresponds to λ₂
    # For symmetric matrix, eigenvector for λ₂ is perpendicular to the gradient
    # When Jxy is near zero, we need a different approach

    # Use the standard eigenvector formula, but handle degenerate cases
    # If Jxy is near zero, the eigenvector is determined by which diagonal is larger
    near_zero_mask = np.abs(Jxy) < 1e-10

    # Standard case: use (Jxy, λ₂ - Jxx) or (λ₂ - Jyy, Jxy)
    tx = np.where(
        near_zero_mask,
        np.where(Jxx > Jyy, 0.0, 1.0),  # Horizontal edge -> horizontal tangent
        Jxy,
    )
    ty = np.where(
        near_zero_mask,
        np.where(Jxx > Jyy, 1.0, 0.0),  # Vertical edge -> vertical tangent
        lambda2 - Jxx,
    )

    # Normalize to unit length
    magnitude = np.sqrt(tx * tx + ty * ty)
    magnitude = np.maximum(magnitude, 1e-10)  # Avoid division by zero
    tx = tx / magnitude
    ty = ty / magnitude

    # Compute coherence: (λ₁ - λ₂) / (λ₁ + λ₂ + ε)
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
    coherence = np.clip(coherence, 0.0, 1.0)

    return tx, ty, coherence


def refine_tangent_field(
    tx: np.ndarray, ty: np.ndarray, sigma: float, iterations: int
) -> tuple[np.ndarray, np.ndarray]:
    """Refine tangent field through iterative smoothing and renormalization.

    This process reduces noise while preserving the overall flow structure.

    Parameters
    ----------
    tx
        Tangent field x component.
    ty
        Tangent field y component.
    sigma
        Standard deviation for Gaussian smoothing.
    iterations
        Number of refinement iterations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Refined tangent field components (tx, ty) as unit vectors.
    """
    tx_refined = tx.copy()
    ty_refined = ty.copy()

    for _ in range(iterations):
        if sigma > 0:
            # Smooth the tangent field
            tx_refined = gaussian_filter(tx_refined, sigma=sigma)
            ty_refined = gaussian_filter(ty_refined, sigma=sigma)

        # Renormalize to unit length
        magnitude = np.sqrt(tx_refined * tx_refined + ty_refined * ty_refined)
        magnitude = np.maximum(magnitude, 1e-10)  # Avoid division by zero
        tx_refined = tx_refined / magnitude
        ty_refined = ty_refined / magnitude

    return tx_refined, ty_refined


def compute_etf(image: np.ndarray, config: ETFConfig | None = None) -> ETFResult:
    """Compute Edge Tangent Field from an image.

    This is the main orchestration function that combines structure tensor
    computation, tangent field extraction, and optional refinement.

    Parameters
    ----------
    image
        Input image as BGR uint8 (H, W, 3) or grayscale float64 (H, W) in [0, 1].
    config
        Configuration for ETF computation. If None, uses defaults.

    Returns
    -------
    ETFResult
        Result containing tangent field components, coherence, and gradient magnitude.
    """
    if config is None:
        config = ETFConfig()

    # Convert to grayscale if needed
    if image.ndim == 3:
        # Assume BGR input
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    elif image.ndim == 2:
        # Already grayscale
        if image.dtype == np.uint8:
            gray = image.astype(np.float64) / 255.0
        else:
            gray = image.astype(np.float64)
            # Ensure values are in [0, 1]
            if gray.max() > 1.0:
                gray = gray / gray.max()
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {image.shape}")

    # Compute structure tensor
    Jxx, Jxy, Jyy = compute_structure_tensor(
        gray,
        blur_sigma=config.blur_sigma,
        structure_sigma=config.structure_sigma,
        sobel_ksize=config.sobel_ksize,
    )

    # Extract tangent field and coherence
    tx, ty, coherence = extract_tangent_field(Jxx, Jxy, Jyy)

    # Refine tangent field if requested
    if config.refine_iterations > 0:
        tx, ty = refine_tangent_field(
            tx,
            ty,
            sigma=config.refine_sigma,
            iterations=config.refine_iterations,
        )

    # Compute gradient magnitude for reference
    # This is the magnitude before structure tensor smoothing
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=config.sobel_ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=config.sobel_ksize)
    gradient_magnitude = np.sqrt(gx * gx + gy * gy)

    return ETFResult(
        tangent_x=tx,
        tangent_y=ty,
        coherence=coherence,
        gradient_magnitude=gradient_magnitude,
    )
