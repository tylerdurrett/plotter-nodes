# Stage 11: Line Integral Convolution (LIC) Visualization

## Overview

Line Integral Convolution (LIC) is a texture synthesis technique that visualizes vector fields by creating streaking patterns along flow lines. The algorithm convolves noise along streamlines to produce a texture where the visual patterns directly reveal the underlying flow structure, making it ideal for validating and previewing flow fields.

## Purpose

LIC visualization serves multiple roles:
- **Flow validation**: Visually verify flow field correctness
- **Artistic preview**: See how particles would move
- **Quality assessment**: Identify discontinuities or artifacts
- **Parameter tuning**: Evaluate effects of flow parameters

The LIC texture shows:
- Flow direction through streak orientation
- Flow coherence through pattern clarity
- Field continuity through smooth transitions
- Singularities or problem areas

## Algorithm

### Core Concept

LIC works by averaging noise values along streamlines:

```python
For each pixel (x, y):
    1. Generate a streamline through (x, y)
    2. Sample noise at points along the streamline
    3. Average the sampled values
    4. Assign average to output pixel
```

This creates correlation along flow lines while maintaining independence across flow lines.

### Streamline Integration

Bidirectional streamline tracing:

```python
def trace_streamline(x0, y0, flow_x, flow_y, length, step):
    points = [(x0, y0)]

    # Forward integration
    x, y = x0, y0
    for i in range(length):
        # Sample flow at current position
        fx = flow_x[int(y), int(x)]
        fy = flow_y[int(y), int(x)]

        # Advance along flow
        x = x + step * fx
        y = y + step * fy

        # Clip to image bounds
        x = clip(x, 0, width-1)
        y = clip(y, 0, height-1)

        points.append((x, y))

    # Backward integration (similar, with -step)
    # ...

    return points
```

### Noise Generation

White noise provides the texture basis:

```python
# Generate reproducible white noise
np.random.seed(seed)
noise = np.random.random((height, width))
```

The noise must be:
- High frequency (pixel-level variation)
- Uniform distribution
- Reproducible (fixed seed)

### Convolution Process

Averaging along streamlines:

```python
def compute_lic_value(points, noise):
    # Sample noise at streamline points
    if use_bilinear:
        # Bilinear interpolation for smooth sampling
        values = [bilinear_sample(noise, x, y) for x, y in points]
    else:
        # Nearest neighbor for speed
        values = [noise[int(y), int(x)] for x, y in points]

    # Average sampled values
    return np.mean(values)
```

### Vectorized Implementation

For efficiency, process all pixels simultaneously:

```python
def compute_lic(flow_x, flow_y, config):
    height, width = flow_x.shape
    noise = generate_noise(height, width, config.seed)

    # Initialize coordinate grids
    Y, X = np.mgrid[0:height, 0:width]
    coords_x = X.astype(np.float64)
    coords_y = Y.astype(np.float64)

    # Accumulator for convolution
    accumulator = np.zeros_like(noise)
    count = 0

    # Forward integration
    for step in range(config.length):
        # Sample noise at current coordinates
        if config.use_bilinear:
            values = map_coordinates(noise, [coords_y, coords_x], order=1)
        else:
            values = noise[coords_y.astype(int), coords_x.astype(int)]

        accumulator += values
        count += 1

        # Advance coordinates along flow
        fx = map_coordinates(flow_x, [coords_y, coords_x], order=1)
        fy = map_coordinates(flow_y, [coords_y, coords_x], order=1)
        coords_x += config.step * fx
        coords_y += config.step * fy

        # Clip to bounds
        coords_x = np.clip(coords_x, 0, width-1)
        coords_y = np.clip(coords_y, 0, height-1)

    # Backward integration (similar)
    # ...

    # Average and normalize
    result = accumulator / count
    return (result - result.min()) / (result.max() - result.min())
```

## Implementation

### Core Function

```python
def compute_lic(flow_x: np.ndarray,
               flow_y: np.ndarray,
               config: LICConfig | None = None) -> np.ndarray:
    """Compute Line Integral Convolution of flow field.

    Parameters:
        flow_x: X-component of flow field (unit vectors)
        flow_y: Y-component of flow field (unit vectors)
        config: LIC configuration parameters

    Returns:
        LIC texture as float64 [0, 1]

    Notes:
        - Deterministic with fixed seed
        - Vectorized for performance
        - Handles boundary conditions
    """
```

### Configuration

```python
@dataclass
class LICConfig:
    length: int = 30          # Streamline length (steps)
    step: float = 1.0        # Step size (pixels)
    seed: int = 42           # Random seed for noise
    use_bilinear: bool = True  # Interpolation method
```

### Processing Steps

1. **Noise Generation**
   - Create white noise texture
   - Use fixed seed for reproducibility

2. **Coordinate Initialization**
   - Create coordinate grids for all pixels
   - Use float64 for sub-pixel accuracy

3. **Forward Integration**
   - Advance coordinates along flow
   - Sample and accumulate noise values
   - Clip coordinates to image bounds

4. **Backward Integration**
   - Similar to forward, negative steps
   - Ensures symmetric convolution

5. **Normalization**
   - Average accumulated values
   - Normalize to [0, 1] range

## Configuration Parameters

### LICConfig Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `length` | int | 30 | Number of integration steps |
| `step` | float | 1.0 | Step size in pixels |
| `seed` | int | 42 | Random seed for noise |
| `use_bilinear` | bool | True | Use bilinear interpolation |

### Parameter Effects

**Length** (10 - 100):
- **Short** (10-20): Local flow, sharp details
- **Medium** (30-40): Balanced (default)
- **Long** (50-100): Global flow, smooth streaks

**Step Size** (0.5 - 2.0):
- **Small** (0.5): Fine sampling, slower
- **Standard** (1.0): Pixel-level steps
- **Large** (2.0): Coarse sampling, faster

**Interpolation**:
- **Bilinear**: Smooth, accurate, slower
- **Nearest**: Fast, may show artifacts

## Visualization

### Direct LIC Display

The raw LIC texture shows flow patterns:
- **Streaks**: Follow flow direction
- **Intensity**: From noise averaging
- **Coherence**: Clear in strong flow, fuzzy in weak

### Overlay Visualization

Blend LIC with original image:

```python
def overlay_lic(lic_image, original, alpha=0.5):
    # Convert LIC to BGR
    lic_bgr = cv2.cvtColor(
        (lic_image * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    # Alpha blend
    result = alpha * lic_bgr + (1 - alpha) * original

    return result.astype(np.uint8)
```

Alpha values:
- 0.3: Subtle texture overlay
- 0.5: Balanced blend (default)
- 0.7: Strong texture, faint image

### Enhanced Visualizations

**Colored LIC**: Map flow properties to color
```python
# Color by flow angle
angle = np.arctan2(flow_y, flow_x)
hue = (angle + np.pi) / (2 * np.pi)
colored_lic = hsv_to_rgb(hue, 1.0, lic_intensity)
```

**Multi-scale LIC**: Different lengths for overview
```python
lic_fine = compute_lic(flow, length=20)
lic_coarse = compute_lic(flow, length=60)
multi_scale = 0.5 * lic_fine + 0.5 * lic_coarse
```

## Quality Characteristics

### Expected Patterns

**Good Flow Field**:
- Smooth, continuous streaks
- Clear directional patterns
- Natural transitions
- No sudden breaks

**Problem Indicators**:
- **Spotty texture**: Singularities or zeros in flow
- **Crossing patterns**: Inconsistent flow directions
- **Uniform gray**: Degenerate or constant flow
- **Sharp discontinuities**: Flow field errors

### Portrait-Specific Patterns

**Face Contour**: Circular/oval streaks around face boundary
**Eyes**: Concentric circular patterns
**Nose**: Vertical streaks along ridge
**Mouth**: Horizontal streaks along lips
**Hair**: Complex flowing patterns
**Skin**: Smooth transitions following surface

## Performance Considerations

### Computational Complexity

- **Time**: O(width × height × length)
- **Space**: O(width × height)
- **Bottleneck**: Flow field sampling

### Optimization Strategies

1. **Vectorization**: Process all pixels in parallel
2. **Caching**: Reuse flow samples when possible
3. **Early termination**: Stop at image boundaries
4. **Lower resolution**: Compute at reduced scale
5. **GPU acceleration**: Future enhancement

### Typical Performance

| Image Size | Length | Time |
|------------|--------|------|
| 640×480 | 30 | ~0.5s |
| 1920×1080 | 30 | ~3s |
| 4K | 30 | ~10s |

## Integration

### Flow Pipeline Integration

LIC is computed after flow field blending:

```python
# Compute flow field
flow_result = run_flow_pipeline(image, contour_result, config)

# Generate LIC visualization
lic_image = compute_lic(
    flow_result.flow_x,
    flow_result.flow_y,
    lic_config
)

# Create overlay
overlay = overlay_lic(lic_image, image, alpha=0.5)
```

### Validation Usage

Use LIC to verify flow field quality:

```python
def validate_flow_field(flow_x, flow_y):
    # Quick LIC with short streamlines
    lic = compute_lic(flow_x, flow_y, LICConfig(length=20))

    # Check for problem indicators
    variance = np.var(lic)
    if variance < 0.01:
        warning("Low variance: possible degenerate flow")

    # Check for discontinuities
    gradient = np.gradient(lic)
    if np.max(np.abs(gradient)) > 0.5:
        warning("Sharp discontinuities detected")

    return lic
```

## Advanced Topics

### Fast LIC

Optimization techniques:
- **Hierarchical**: Compute at multiple scales
- **Sparse**: Sample subset of pixels
- **Cached**: Reuse streamline computations
- **Parallel**: GPU or multi-threaded

### Enhanced LIC

Extensions for richer visualization:
- **OLIC**: Oriented LIC (shows direction)
- **UFLIC**: Unsteady flow LIC (time-varying)
- **PLIC**: Parallel LIC (GPU accelerated)
- **Colored LIC**: Encode additional properties

### Artistic Applications

LIC as artistic effect:
- **Painterly rendering**: Oil painting texture
- **Hair rendering**: Natural hair flow
- **Water effects**: Fluid flow visualization
- **Abstract patterns**: Decorative textures

## Conclusion

LIC provides essential visualization for flow field validation and serves as a preview of how particles would behave in the flow field. The texture patterns directly reveal the flow structure, making it invaluable for parameter tuning and quality assessment in the portrait processing pipeline.