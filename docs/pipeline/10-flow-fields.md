# Stage 10: Flow Fields

## Overview

The flow fields stage combines Edge Tangent Field (ETF) information with contour-based flow to create a unified vector field for particle guidance. Using coherence-based blending and alignment techniques, it produces smooth, structure-aware flow patterns that follow both image edges and face contours.

## Purpose

Flow field blending serves critical roles in particle guidance:
- **Combines multiple flow sources**: ETF for edges, contour flow for face shape
- **Resolves directional ambiguity**: Aligns fields to prevent cancellation
- **Adaptive blending**: Uses coherence to weight reliability
- **Fallback handling**: Ensures valid flow even in degenerate regions

The blended flow field ensures particles:
- Follow strong edges when present (high coherence)
- Follow face contours in smooth regions (low coherence)
- Maintain smooth, continuous paths
- Never get stuck in degenerate zones

## Algorithm

### Contour Flow Computation

Contour flow derives from the signed distance field gradient:

```python
# Compute gradient of signed distance field
grad_y, grad_x = np.gradient(signed_distance)

# Rotate 90° counter-clockwise for tangent flow
flow_x = -grad_y  # Perpendicular to gradient
flow_y = grad_x

# Normalize to unit vectors
magnitude = sqrt(flow_x² + flow_y²)
flow_x = flow_x / max(magnitude, ε)
flow_y = flow_y / max(magnitude, ε)

# Optional smoothing
if smooth_sigma > 0:
    flow_x = Gaussian(flow_x, smooth_sigma)
    flow_y = Gaussian(flow_y, smooth_sigma)
    # Re-normalize after smoothing
```

This creates circular flow patterns around the face contour.

### Tangent Field Alignment

ETF has 180° ambiguity (both ±v are valid). Alignment resolves this:

```python
def align_tangent_field(tx, ty, ref_x, ref_y):
    # Compute dot product with reference
    dot = tx * ref_x + ty * ref_y

    # Flip where pointing opposite direction
    tx = np.where(dot < 0, -tx, tx)
    ty = np.where(dot < 0, -ty, ty)

    return tx, ty
```

This ensures ETF and contour flow point in compatible directions.

### Coherence-Based Blending

Blend weight derives from ETF coherence:

```python
# Transform coherence to blend weight
alpha = coherence ** coherence_power

# Higher coherence → prefer ETF
# Lower coherence → prefer contour flow
```

The power function allows tuning the transition sharpness.

### Linear Blending with Fallback

Final blending with degenerate case handling:

```python
# Linear interpolation
blend_x = alpha * etf_x + (1 - alpha) * contour_x
blend_y = alpha * etf_y + (1 - alpha) * contour_y

# Check for cancellation (opposing vectors)
magnitude = sqrt(blend_x² + blend_y²)
degenerate = magnitude < fallback_threshold

# Use contour flow in degenerate regions
flow_x = np.where(degenerate, contour_x, blend_x)
flow_y = np.where(degenerate, contour_y, blend_y)

# Final normalization
magnitude = sqrt(flow_x² + flow_y²)
flow_x = flow_x / max(magnitude, ε)
flow_y = flow_y / max(magnitude, ε)
```

## Implementation

### Core Functions

```python
def compute_contour_flow(signed_distance: np.ndarray,
                        smooth_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute tangent flow from signed distance field.

    Parameters:
        signed_distance: Signed distance from contour
        smooth_sigma: Optional smoothing (0 = no smoothing)

    Returns:
        Tuple of (flow_x, flow_y) unit vectors
    """

def align_tangent_field(tx: np.ndarray, ty: np.ndarray,
                       ref_x: np.ndarray, ref_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align tangent field to reference direction.

    Parameters:
        tx, ty: Tangent field to align
        ref_x, ref_y: Reference field

    Returns:
        Aligned (tx, ty) with consistent orientation
    """

def compute_blend_weight(coherence: np.ndarray,
                        config: FlowConfig | None = None) -> np.ndarray:
    """Convert coherence to blend weight.

    Parameters:
        coherence: ETF coherence [0, 1]
        config: Flow configuration

    Returns:
        Blend weight alpha [0, 1]
    """

def blend_flow_fields(etf_tx: np.ndarray, etf_ty: np.ndarray,
                     contour_fx: np.ndarray, contour_fy: np.ndarray,
                     alpha: np.ndarray,
                     fallback_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Blend ETF and contour flow with fallback.

    Parameters:
        etf_tx, etf_ty: Aligned ETF tangents
        contour_fx, contour_fy: Contour flow
        alpha: Blend weight (0=contour, 1=ETF)
        fallback_threshold: Magnitude threshold for fallback

    Returns:
        Blended (flow_x, flow_y) unit vectors
    """
```

### Processing Pipeline

1. **Contour Flow Generation**
   - Compute gradient of signed distance
   - Rotate 90° for tangent direction
   - Normalize and optionally smooth

2. **ETF Computation**
   - Full ETF pipeline (see Stage 9)
   - Produces tangent field and coherence

3. **Field Alignment**
   - Use contour flow as reference
   - Flip ETF where opposite
   - Ensure compatible directions

4. **Weight Computation**
   - Apply power function to coherence
   - Create smooth transition map

5. **Field Blending**
   - Linear interpolation by weight
   - Detect degenerate regions
   - Apply fallback to contour flow
   - Final normalization

## Configuration

### FlowConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `etf` | ETFConfig | Default ETF | ETF computation parameters |
| `contour_smooth_sigma` | float | 1.0 | Contour flow smoothing |
| `blend_mode` | str | "coherence" | Blending strategy |
| `coherence_power` | float | 2.0 | Coherence transform exponent |
| `fallback_threshold` | float | 0.1 | Degenerate magnitude threshold |

### Parameter Guidelines

**Contour Smooth Sigma** (0.0 - 3.0):
- 0.0: No smoothing, sharp transitions
- 1.0: Moderate smoothing (default)
- 3.0: Heavy smoothing, very smooth flow

**Coherence Power** (1.0 - 4.0):
- 1.0: Linear transition
- 2.0: Quadratic (default, smooth)
- 4.0: Sharp transition

**Fallback Threshold** (0.01 - 0.3):
- 0.01: Rarely trigger fallback
- 0.1: Default, handle clear cancellations
- 0.3: Aggressive fallback

## Output Format

### FlowResult Structure

```python
@dataclass(frozen=True)
class FlowResult:
    etf: ETFResult              # Complete ETF data
    contour_flow_x: np.ndarray  # Contour flow X
    contour_flow_y: np.ndarray  # Contour flow Y
    blend_weight: np.ndarray    # Alpha blending map
    flow_x: np.ndarray          # Final flow X
    flow_y: np.ndarray          # Final flow Y
```

### Flow Properties

- **Unit vectors**: All flow fields normalized
- **Continuous**: Smooth transitions between regions
- **No singularities**: Fallback prevents degenerate points
- **Structure-aware**: Follows edges and contours

## Visualization

### Output Files

The flow pipeline generates:
- `flow/etf_coherence.png`: Edge coherence map
- `flow/etf_quiver.png`: ETF direction arrows
- `flow/contour_flow_quiver.png`: Contour flow arrows
- `flow/blend_weight.png`: Blending alpha map
- `flow/flow_lic.png`: LIC visualization
- `flow/flow_lic_overlay.png`: LIC over original
- `flow/flow_quiver.png`: Final flow arrows
- `flow/flow_x_raw.npy`: Flow X component
- `flow/flow_y_raw.npy`: Flow Y component
- `flow/contact_sheet.png`: All visualizations

### Visual Characteristics

**High Coherence Regions** (edges):
- Flow follows edge tangents
- Sharp, well-defined directions
- Minimal contour influence

**Low Coherence Regions** (smooth):
- Flow follows face contour
- Circular/oval patterns
- Minimal edge influence

**Transition Zones**:
- Smooth blend between sources
- No abrupt direction changes
- Natural flow continuation

## Quality Considerations

### Common Issues

1. **Alignment artifacts**: Incorrect flipping at boundaries
2. **Cancellation zones**: Opposing flows create null regions
3. **Discontinuities**: Abrupt transitions with high coherence_power
4. **Over-smoothing**: Lost detail with high smooth_sigma

### Validation Checks

```python
# Unit vector property
magnitude = np.sqrt(flow.flow_x**2 + flow.flow_y**2)
assert np.allclose(magnitude, 1.0, atol=1e-6)

# No NaN or inf values
assert np.all(np.isfinite(flow.flow_x))
assert np.all(np.isfinite(flow.flow_y))

# Blend weight range
assert np.all((flow.blend_weight >= 0) & (flow.blend_weight <= 1))

# Fallback triggered where expected
blend_mag = np.sqrt(
    (flow.blend_weight * etf.tangent_x +
     (1 - flow.blend_weight) * contour_x)**2 +
    (flow.blend_weight * etf.tangent_y +
     (1 - flow.blend_weight) * contour_y)**2
)
fallback_mask = blend_mag < config.fallback_threshold
# Verify contour flow used in fallback regions
```

### Best Practices

1. **Default parameters work well** for most portraits
2. **Increase coherence_power** for sharper edge following
3. **Decrease coherence_power** for smoother blending
4. **Adjust ETF parameters** before flow parameters
5. **Visualize with LIC** to see actual flow patterns

## Integration

### Pipeline Context

Flow fields integrate two information sources:

```python
# From Stage 9: Edge Tangent Field
etf_result = compute_etf(image, config.etf)

# From Stage 6: Contour signed distance
contour_result = run_contour_pipeline(image)

# Compute and blend flows
flow_result = run_flow_pipeline(image, contour_result, config)
```

### Particle System Usage

The flow field guides particle movement:

```python
# At each particle position (x, y)
fx = flow_x[int(y), int(x)]
fy = flow_y[int(y), int(x)]

# Update particle position
new_x = x + step_size * fx
new_y = y + step_size * fy
```

Particles naturally:
- Follow edges when present
- Circle around face in smooth areas
- Never get stuck (no singularities)

## Advanced Topics

### Alternative Blend Modes

Future extensions could include:

**Max Coherence**: Always use most coherent source
```python
alpha = np.where(etf_coherence > contour_coherence, 1.0, 0.0)
```

**Weighted by Gradient**: Use gradient magnitude
```python
alpha = gradient_magnitude / (gradient_magnitude + ε)
```

**Bilateral Weights**: Consider spatial proximity
```python
alpha = coherence * spatial_weight(x, y)
```

### Multi-Scale Flow

Hierarchical flow computation:
1. Compute flow at multiple scales
2. Blend based on local frequency content
3. Fine details use fine-scale flow
4. Broad strokes use coarse-scale flow

## Next Stage

The blended flow field enables [Stage 11: LIC Visualization](11-lic-visualization.md), which creates texture visualizations showing the actual flow patterns for validation and artistic preview.