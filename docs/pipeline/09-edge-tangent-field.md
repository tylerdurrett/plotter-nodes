# Stage 9: Edge Tangent Field (ETF)

## Overview

The Edge Tangent Field (ETF) stage computes a smooth vector field that follows the dominant edges and structures in an image. Using structure tensor analysis and eigenvector decomposition, it produces tangent vectors perpendicular to image gradients, creating flow lines that naturally follow visual contours and forms.

## Purpose

The ETF provides structural flow guidance by:
- Detecting dominant edge orientations at each pixel
- Computing smooth tangent flow along edges
- Measuring edge strength through coherence
- Creating natural flow patterns for artistic rendering

The tangent field ensures that:
- Flow lines follow image structures (edges, contours)
- Smooth regions have low coherence (unreliable flow)
- Strong edges have high coherence (reliable flow)
- The field is continuous and smooth

## Algorithm

### Structure Tensor

The structure tensor captures local image geometry through gradient analysis:

```python
# Gradient computation
Ix = ∂I/∂x  (horizontal gradient)
Iy = ∂I/∂y  (vertical gradient)

# Structure tensor components
Jxx = Ix * Ix  (gradient energy in x)
Jxy = Ix * Iy  (gradient correlation)
Jyy = Iy * Iy  (gradient energy in y)

# Gaussian smoothing for robustness
Jxx = Gaussian(Jxx, σ_structure)
Jxy = Gaussian(Jxy, σ_structure)
Jyy = Gaussian(Jyy, σ_structure)
```

### Eigenvector Analysis

The structure tensor's eigenvectors reveal edge orientation:

```python
# Eigenvalues (closed-form for 2x2 matrix)
trace = Jxx + Jyy
det = Jxx * Jyy - Jxy * Jxy
discriminant = sqrt((trace/2)² - det)

λ₁ = trace/2 + discriminant  (major eigenvalue)
λ₂ = trace/2 - discriminant  (minor eigenvalue)

# Minor eigenvector (tangent to edge)
if |Jxy| > ε:  # Non-degenerate case
    tx = Jxy
    ty = λ₂ - Jxx
else:  # Degenerate case (axis-aligned edge)
    if Jxx > Jyy:
        tx = 0, ty = 1  # Horizontal edge → vertical tangent
    else:
        tx = 1, ty = 0  # Vertical edge → horizontal tangent
```

### Coherence Metric

Coherence measures the reliability of the local orientation:

```python
coherence = (λ₁ - λ₂) / (λ₁ + λ₂ + ε)
```

- **High coherence** (near 1.0): Strong, unambiguous edge
- **Low coherence** (near 0.0): Uniform region or corner
- **Medium coherence**: Gradual transitions

### Iterative Refinement

The tangent field is refined through iterative smoothing:

```python
for iteration in range(n_iterations):
    # Smooth tangent components
    tx = Gaussian(tx, σ_refine)
    ty = Gaussian(ty, σ_refine)

    # Re-normalize to unit length
    magnitude = sqrt(tx² + ty²)
    tx = tx / max(magnitude, ε)
    ty = ty / max(magnitude, ε)
```

## Implementation

### Core Functions

```python
def compute_structure_tensor(gray: np.ndarray,
                            blur_sigma: float,
                            structure_sigma: float,
                            sobel_ksize: int) -> tuple[np.ndarray, ...]:
    """Compute structure tensor from grayscale image.

    Parameters:
        gray: Grayscale image [0, 1]
        blur_sigma: Pre-blur for noise reduction
        structure_sigma: Structure tensor smoothing
        sobel_ksize: Sobel kernel size (1, 3, 5, 7)

    Returns:
        Tuple of (Jxx, Jxy, Jyy) tensor components
    """

def extract_tangent_field(Jxx: np.ndarray,
                         Jxy: np.ndarray,
                         Jyy: np.ndarray) -> tuple[np.ndarray, ...]:
    """Extract tangent field from structure tensor.

    Parameters:
        Jxx, Jxy, Jyy: Structure tensor components

    Returns:
        Tuple of (tx, ty, coherence) where tx, ty are unit vectors
    """

def refine_tangent_field(tx: np.ndarray,
                        ty: np.ndarray,
                        sigma: float,
                        iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively smooth and renormalize tangent field.

    Parameters:
        tx, ty: Initial tangent components
        sigma: Gaussian smoothing sigma
        iterations: Number of refinement iterations

    Returns:
        Refined (tx, ty) unit vector field
    """

def compute_etf(image: np.ndarray,
               config: ETFConfig | None = None) -> ETFResult:
    """Complete ETF computation pipeline.

    Parameters:
        image: Input BGR or grayscale image
        config: ETF configuration parameters

    Returns:
        ETFResult with tangent field and coherence
    """
```

### Processing Pipeline

1. **Image Preparation**
   - Convert to grayscale if needed
   - Apply Gaussian blur for noise reduction
   - Normalize to [0, 1] range

2. **Gradient Computation**
   - Sobel operators for Ix, Iy
   - Use CV_64F for precision
   - Handle image boundaries

3. **Structure Tensor Assembly**
   - Compute outer product components
   - Apply Gaussian smoothing
   - Ensure symmetry (Jxy = Jyx)

4. **Eigenvector Extraction**
   - Closed-form eigenvalue computation
   - Minor eigenvector as tangent
   - Handle degenerate cases

5. **Field Refinement**
   - Iterative Gaussian smoothing
   - Maintain unit vector constraint
   - Preserve field continuity

## Configuration

### ETFConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `blur_sigma` | float | 1.5 | Initial blur for noise reduction |
| `structure_sigma` | float | 5.0 | Structure tensor smoothing |
| `refine_sigma` | float | 3.0 | Refinement smoothing sigma |
| `refine_iterations` | int | 2 | Number of refinement passes |
| `sobel_ksize` | int | 3 | Sobel operator size (1, 3, 5, 7) |

### Parameter Guidelines

**Blur Sigma** (0.5 - 3.0):
- Lower: Preserve fine details, more noise
- Higher: Smoother field, less detail

**Structure Sigma** (2.0 - 10.0):
- Lower: Local structure, detailed flow
- Higher: Global structure, smooth flow

**Refine Sigma** (1.0 - 5.0):
- Lower: Preserve discontinuities
- Higher: Smoother, more continuous field

**Refine Iterations** (0 - 5):
- 0: No refinement, raw field
- 1-2: Moderate smoothing (default)
- 3-5: Heavy smoothing, very smooth field

**Sobel Kernel Size**:
- 1: Simple gradient (fast)
- 3: Standard Sobel (default)
- 5-7: Smoother gradients (slower)

## Output Format

### ETFResult Structure

```python
@dataclass(frozen=True)
class ETFResult:
    tangent_x: np.ndarray       # X-component of unit tangent
    tangent_y: np.ndarray       # Y-component of unit tangent
    coherence: np.ndarray       # Edge coherence [0, 1]
    gradient_magnitude: np.ndarray  # Edge strength
```

### Field Properties

- **Unit vectors**: `sqrt(tx² + ty²) ≈ 1.0` everywhere
- **Continuous**: Smooth transitions between regions
- **Orientation**: Tangent to edges (perpendicular to gradients)
- **Ambiguity**: 180° ambiguity (±direction equally valid)

## Visualization

### Visualization Methods

1. **Coherence Map**
   - Heatmap showing edge strength
   - Viridis colormap recommended
   - High values along strong edges

2. **Quiver Plot**
   - Arrows showing tangent directions
   - Subsample for clarity (every 16 pixels)
   - Color by coherence or uniform

3. **Line Integral Convolution**
   - Texture showing flow patterns
   - Best visualization of field structure
   - See [Stage 11: LIC Visualization](11-lic-visualization.md)

### Expected Patterns

**Portrait Features**:
- Face contour: Strong coherent tangents following oval
- Eyes: Circular tangent patterns
- Nose: Vertical tangents along ridge
- Mouth: Horizontal tangents along lips
- Hair: Complex flowing patterns

**Coherence Distribution**:
- Edges: 0.7-1.0 (strong signal)
- Textured regions: 0.3-0.7 (moderate)
- Smooth skin: 0.0-0.3 (weak/unreliable)

## Quality Considerations

### Common Issues

1. **Noise amplification**: Too low blur_sigma
2. **Over-smoothing**: Too high structure_sigma or too many iterations
3. **Lost details**: Parameters too aggressive for image resolution
4. **Degenerate regions**: Corners and uniform areas have undefined orientation

### Validation Tests

```python
# Check unit vector property
magnitude = np.sqrt(etf.tangent_x**2 + etf.tangent_y**2)
assert np.allclose(magnitude, 1.0, atol=1e-6)

# Check coherence range
assert np.all((etf.coherence >= 0) & (etf.coherence <= 1))

# Check perpendicularity to gradient (should be ~90°)
gradient_x, gradient_y = np.gradient(gray_image)
dot_product = etf.tangent_x * gradient_x + etf.tangent_y * gradient_y
assert np.mean(np.abs(dot_product)) < 0.1  # Near zero
```

### Best Practices

1. Start with default parameters
2. Adjust blur_sigma based on image noise
3. Increase structure_sigma for smoother flow
4. Use 1-2 refinement iterations for most cases
5. Validate with coherence visualization

## Mathematical Background

### Structure Tensor Theory

The structure tensor is the outer product of the gradient:
```
J = ∇I ⊗ ∇I = [Ix] [Ix Iy]
                [Iy]
```

Its eigendecomposition reveals:
- **λ₁**: Gradient strength perpendicular to edge
- **λ₂**: Gradient strength along edge (ideally zero)
- **v₁**: Normal to edge (gradient direction)
- **v₂**: Tangent to edge (flow direction)

### Coherence Interpretation

Coherence is related to the condition number:
- **Isotropic** (λ₁ ≈ λ₂): No dominant direction
- **Anisotropic** (λ₁ >> λ₂): Clear edge direction

## Next Stage

The ETF flows into [Stage 10: Flow Fields](10-flow-fields.md), where it combines with contour-based flow to create the final blended flow field for particle guidance.