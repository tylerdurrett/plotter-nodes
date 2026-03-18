# Stage 4: Influence Maps (Distance-to-Influence Remapping)

## Overview

The influence maps stage transforms raw distance fields into normalized influence values through configurable remapping curves. This critical transformation controls how feature influence decreases with distance, defining the spatial extent and falloff characteristics of each facial feature's importance.

## Purpose

Influence remapping provides:
- Normalized values (0.0-1.0) suitable for combination
- Configurable falloff behaviors (sharp vs. smooth)
- Perceptually meaningful gradients
- Mathematical control over influence spread
- Artistic flexibility in importance distribution

## Remapping Curves

The pipeline supports three fundamental curve types, each with distinct characteristics:

### 1. Linear Falloff

```python
influence = max(0, 1 - distance/radius)
```

**Formula**: I(d) = max(0, 1 - d/r)

**Characteristics**:
- Constant rate of decrease
- Sharp cutoff at radius
- Predictable, uniform behavior
- Good for hard boundaries

**Parameters**:
- `radius`: Maximum influence distance (pixels)

### 2. Gaussian Falloff

```python
influence = exp(-(distance²)/(2σ²))
```

**Formula**: I(d) = e^(-d²/2σ²)

**Characteristics**:
- Natural, smooth falloff
- Never reaches exactly zero
- Bell curve profile
- Mimics natural phenomena

**Parameters**:
- `sigma` (σ): Standard deviation controlling spread
- At d=σ: influence ≈ 0.606
- At d=2σ: influence ≈ 0.135
- At d=3σ: influence ≈ 0.011

### 3. Exponential Falloff

```python
influence = exp(-distance/tau)
```

**Formula**: I(d) = e^(-d/τ)

**Characteristics**:
- Rapid initial decrease
- Long tail (slow final decay)
- Never reaches exactly zero
- Good for concentrated influence

**Parameters**:
- `tau` (τ): Decay rate constant
- At d=τ: influence ≈ 0.368
- At d=2τ: influence ≈ 0.135
- At d=3τ: influence ≈ 0.050

## Comparison of Curves

```
Distance  | Linear | Gaussian | Exponential
----------|--------|----------|-------------
0 pixels  |  1.00  |   1.00   |    1.00
25 pixels |  0.83  |   0.95   |    0.68
50 pixels |  0.67  |   0.82   |    0.46
75 pixels |  0.50  |   0.64   |    0.31
100 pixels|  0.33  |   0.44   |    0.21
150 pixels|  0.00  |   0.18   |    0.10
200 pixels|  0.00  |   0.06   |    0.04
```

*Example values with radius=150, σ=80, τ=60*

## Implementation

### Core Function

```python
def remap_influence(
    distance_field: np.ndarray,
    config: RemapConfig
) -> np.ndarray:
    """Remap distance field to influence map.

    Parameters:
        distance_field: Float array of pixel distances
        config: Remapping configuration with curve type and parameters

    Returns:
        Influence map (float64) with values in [0.0, 1.0]
    """
    # Clamp distances to maximum
    d = np.minimum(distance_field, config.clamp_distance)

    # Apply selected curve
    if config.curve == "linear":
        influence = np.maximum(0.0, 1.0 - d / config.radius)
    elif config.curve == "gaussian":
        influence = np.exp(-(d**2) / (2 * config.sigma**2))
    elif config.curve == "exponential":
        influence = np.exp(-d / config.tau)

    return np.clip(influence, 0.0, 1.0)
```

### Configuration

```python
@dataclass
class RemapConfig:
    curve: str = "gaussian"      # Curve type
    radius: float = 150.0        # Linear radius
    sigma: float = 80.0          # Gaussian spread
    tau: float = 60.0            # Exponential decay
    clamp_distance: float = 300.0 # Max distance before clamping
```

## Visual Characteristics

### Linear Influence
- Creates circular zones of influence
- Hard edge at radius boundary
- Uniform gradient within radius
- No influence beyond radius

### Gaussian Influence
- Smooth, natural-looking falloff
- No hard boundaries
- Peak sharpness controlled by σ
- Influence extends indefinitely (but diminishes)

### Exponential Influence
- Strong near-field influence
- Gradual far-field decay
- Asymmetric profile (fast→slow)
- Good for focal points

## Parameter Selection Guide

### Choosing Curve Type

| Use Case | Recommended Curve | Reasoning |
|----------|-------------------|-----------|
| Natural portraits | Gaussian | Smooth, organic appearance |
| Graphic/stylized | Linear | Clear boundaries, geometric |
| High detail focus | Exponential | Strong central emphasis |
| Uniform coverage | Linear | Predictable spread |

### Parameter Tuning

#### Gaussian σ (Sigma)
- **Small (30-50)**: Tight focus on features
- **Medium (60-100)**: Balanced influence
- **Large (100-150)**: Broad, soft influence

#### Linear Radius
- **Small (50-100)**: Localized importance
- **Medium (100-200)**: Standard coverage
- **Large (200-300)**: Extended influence

#### Exponential τ (Tau)
- **Small (20-40)**: Rapid falloff
- **Medium (40-80)**: Balanced decay
- **Large (80-120)**: Extended influence

## Advanced Techniques

### Composite Curves

Combine multiple curves for complex behaviors:
```python
# Hybrid: Sharp center + soft edges
influence = 0.7 * gaussian(d, σ=50) + 0.3 * gaussian(d, σ=150)
```

### Adaptive Parameters

Vary parameters based on image properties:
```python
# Scale with face size
face_width = bbox.width
sigma = face_width * 0.15  # 15% of face width
```

### Anisotropic Influence

Different falloff rates in different directions:
```python
# Elliptical influence for eyes
dx = x - eye_center_x
dy = y - eye_center_y
d_ellipse = sqrt((dx/a)² + (dy/b)²)
influence = exp(-d_ellipse² / (2σ²))
```

## Normalization

### Value Range
- All influence values clamped to [0.0, 1.0]
- Ensures compatibility for weighted combination
- Prevents overflow in subsequent operations

### Distance Clamping
- `clamp_distance` parameter limits maximum distance
- Improves performance (skip far pixels)
- Prevents numerical issues with large distances

## Quality Considerations

### Common Artifacts

1. **Hard edges** (Linear)
   - Visible circular boundaries
   - Solution: Use Gaussian or blend curves

2. **Over-concentration** (Exponential)
   - Too much focus on centers
   - Solution: Increase τ or use Gaussian

3. **Over-diffusion** (Gaussian with large σ)
   - Features blend together
   - Solution: Reduce σ or use exponential

### Validation

```python
# Check influence map properties
def validate_influence_map(influence):
    assert influence.min() >= 0.0
    assert influence.max() <= 1.0
    assert not np.any(np.isnan(influence))
    assert not np.any(np.isinf(influence))

    # Check reasonable coverage
    coverage = np.mean(influence > 0.1)
    if coverage < 0.05:
        warnings.warn("Very low influence coverage")
    if coverage > 0.8:
        warnings.warn("Very high influence coverage")
```

## Performance

### Computational Cost

- **Linear**: Single division + max operation
- **Gaussian**: Exponential of squared distance
- **Exponential**: Single exponential operation

All curves are O(n) for n pixels, vectorized via NumPy.

### Optimization

```python
# Pre-compute expensive operations
gaussian_factor = -1.0 / (2 * sigma**2)
influence = np.exp(distance_field**2 * gaussian_factor)

# Early termination for distant pixels
mask = distance_field < clamp_distance
influence[~mask] = 0.0
influence[mask] = compute_influence(distance_field[mask])
```

## Next Stage

The influence maps flow into [Stage 5: Combination](05-combination.md), where they are merged using weighted summation to create the final importance map.