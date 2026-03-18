# Stage 13: Complexity Map

## Overview

The Complexity Map stage measures local image complexity through gradient and texture analysis, producing scalar fields that quantify structural detail at each pixel. These maps serve dual purposes: as standalone exports for downstream consumption and as modulators for particle flow speed, where particles slow down in detail-rich areas to capture intricate features.

## Purpose

The complexity map provides structural detail measurement by:
- Detecting regions of high gradient energy (edges, textures)
- Quantifying local image variation through multiple metrics
- Normalizing complexity to a consistent [0, 1] range
- Deriving flow speed modulation from complexity values

The complexity measurement ensures that:
- Detailed regions (eyes, hair texture) have high complexity values
- Smooth regions (skin, background) have low complexity values
- Flow particles slow down in complex areas for better detail capture
- The metric can be tuned for different types of detail detection

## Algorithm

### Gradient Energy Metric

The gradient energy metric uses Sobel operators to measure edge strength:

```python
# Sobel gradient computation
Gx = Sobel(image, dx=1, dy=0)  # Horizontal gradient
Gy = Sobel(image, dx=0, dy=1)  # Vertical gradient

# Gradient magnitude
magnitude = sqrt(Gx² + Gy²)

# Gaussian smoothing for spatial coherence
energy = Gaussian(magnitude, σ)
```

### Laplacian Energy Metric

The Laplacian metric detects fine textures and rapid intensity changes:

```python
# Laplacian operator (second derivative)
L = Laplacian(image)

# Absolute value for energy
energy = |L|

# Gaussian smoothing
energy = Gaussian(energy, σ)
```

### Multiscale Gradient Energy

Combines gradient energies across multiple scales to capture both fine and coarse details:

```python
# Compute gradient energy at each scale
energies = []
for σ in scales:
    energy = compute_gradient_energy(image, σ)
    energies.append(energy)

# Weighted combination
combined = Σ(weight[i] * energy[i])
```

### Normalization

Percentile-based normalization ensures robust scaling despite outliers:

```python
# Compute normalization ceiling
ceiling = percentile(raw_complexity, p)  # Default p=99.0

# Normalize and clip
if ceiling > 0:
    normalized = raw_complexity / ceiling
    normalized = clip(normalized, 0, 1)
else:
    normalized = zeros_like(raw_complexity)
```

### Flow Speed Derivation

Linear inverse mapping converts complexity to particle speed:

```python
# Inverse linear relationship
speed = speed_max - complexity * (speed_max - speed_min)

# Ensures:
# complexity=0 (smooth) → speed=speed_max (fast)
# complexity=1 (detailed) → speed=speed_min (slow)
```

## Implementation

### Core Functions

```python
def compute_gradient_energy(gray: np.ndarray,
                           sigma: float = 3.0) -> np.ndarray:
    """Compute gradient energy using Sobel gradients.

    Parameters:
        gray: Grayscale image [0, 1]
        sigma: Gaussian smoothing sigma

    Returns:
        Raw gradient energy (not normalized)
    """

def compute_laplacian_energy(gray: np.ndarray,
                            sigma: float = 3.0) -> np.ndarray:
    """Compute Laplacian energy for fine detail detection.

    Parameters:
        gray: Grayscale image [0, 1]
        sigma: Gaussian smoothing sigma

    Returns:
        Raw Laplacian energy (not normalized)
    """

def compute_multiscale_gradient_energy(gray: np.ndarray,
                                       scales: list[float],
                                       weights: list[float]) -> np.ndarray:
    """Compute weighted sum of gradient energies at multiple scales.

    Parameters:
        gray: Grayscale image [0, 1]
        scales: List of sigma values for different scales
        weights: Weights for combining scales (must sum to 1.0)

    Returns:
        Combined multiscale energy (not normalized)
    """

def normalize_map(raw: np.ndarray,
                 percentile: float = 99.0) -> np.ndarray:
    """Normalize complexity map using percentile-based scaling.

    Parameters:
        raw: Raw complexity values
        percentile: Percentile for normalization ceiling (100.0 = max)

    Returns:
        Normalized map [0, 1]
    """

def compute_flow_speed(complexity: np.ndarray,
                      config: FlowSpeedConfig | None = None) -> np.ndarray:
    """Derive flow speed from complexity map.

    Parameters:
        complexity: Normalized complexity [0, 1]
        config: Speed configuration (speed_min, speed_max)

    Returns:
        Flow speed map [speed_min, speed_max]
    """

def compute_complexity_map(image: np.ndarray,
                          config: ComplexityConfig | None = None,
                          mask: np.ndarray | None = None) -> ComplexityResult:
    """Complete complexity computation pipeline.

    Parameters:
        image: Input BGR or grayscale image
        config: Complexity configuration
        mask: Optional binary mask to scope computation

    Returns:
        ComplexityResult with raw and normalized complexity
    """
```

### Processing Pipeline

1. **Image Preparation**
   - Convert BGR to grayscale if needed
   - Normalize to [0, 1] float64 range
   - Apply mask if provided (face region)

2. **Metric Computation**
   - Dispatch to appropriate metric function
   - Gradient: Sobel operators + magnitude
   - Laplacian: Second derivative + absolute value
   - Multiscale: Multiple gradient scales

3. **Post-Processing**
   - Apply Gaussian smoothing
   - Percentile-based normalization
   - Mask application (zero outside region)

4. **Flow Speed Derivation**
   - Linear inverse mapping
   - Configurable speed range
   - Clip to valid bounds

5. **Output Generation**
   - ComplexityResult with raw and normalized maps
   - Flow speed array for particle modulation
   - Visualization outputs (heatmaps)

## Configuration

### ComplexityConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | str | "gradient" | Complexity metric: "gradient", "laplacian", "multiscale_gradient" |
| `sigma` | float | 3.0 | Gaussian smoothing sigma for single-scale metrics |
| `scales` | list[float] | [1.0, 3.0, 8.0] | Sigma values for multiscale metric |
| `scale_weights` | list[float] | [0.5, 0.3, 0.2] | Weights for combining scales |
| `normalize_percentile` | float | 99.0 | Percentile for normalization (100.0 = max) |
| `output_dir` | str | "output" | Output directory for results |

### FlowSpeedConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speed_min` | float | 0.3 | Minimum speed in complex areas |
| `speed_max` | float | 1.0 | Maximum speed in smooth areas |

### Parameter Guidelines

**Metric Selection**:
- `"gradient"`: Best for edge-based complexity, general purpose (default)
- `"laplacian"`: Emphasizes fine textures and rapid changes
- `"multiscale_gradient"`: Captures detail at multiple scales

**Sigma** (1.0 - 10.0):
- Lower: More sensitive to fine details, potentially noisy
- Higher: Smoother complexity map, less detail sensitivity

**Normalize Percentile** (90.0 - 100.0):
- 99.0: Robust against outliers (default)
- 100.0: Use maximum value for normalization
- 95.0: More aggressive clipping for high contrast

**Speed Range**:
- `speed_min`: 0.1-0.5 (how much to slow in complex areas)
- `speed_max`: 0.8-1.0 (baseline speed in simple areas)
- Wider range: More dramatic speed variation

## Output Format

### ComplexityResult Structure

```python
@dataclass(frozen=True)
class ComplexityResult:
    raw_complexity: np.ndarray    # Unnormalized metric output
    complexity: np.ndarray         # Normalized [0, 1] complexity
    metric: str                    # Which metric was used
```

### Flow Speed Array

- **Type**: float64 ndarray
- **Shape**: Same as input image (H, W)
- **Range**: [speed_min, speed_max]
- **Semantics**: Scalar multiplier for particle velocity

### Export Format

When exported via the binary bundle:
- **File**: `complexity.bin` (float32 binary)
- **Range**: [0.0, 1.0] normalized
- **File**: `flow_speed.bin` (float32 binary)
- **Range**: [0.0, 1.0] normalized to speed range

## Visualization

### Visualization Methods

1. **Complexity Heatmap**
   - Raw energy: "viridis" colormap
   - Normalized: "inferno" colormap
   - Higher values = brighter colors

2. **Flow Speed Map**
   - "plasma" colormap recommended
   - Bright = fast (smooth regions)
   - Dark = slow (complex regions)

3. **Contact Sheet**
   - Original image for reference
   - Raw complexity energy
   - Normalized complexity
   - Flow speed (if computed)

### Expected Patterns

**Portrait Features**:
- Eyes: High complexity (iris patterns, lashes)
- Hair: Very high complexity (texture, strands)
- Nose: Moderate complexity (nostrils, shadows)
- Smooth skin: Low complexity
- Background: Variable (depends on content)

**Complexity Distribution**:
- Textured regions: 0.6-1.0 (high)
- Edge regions: 0.3-0.7 (moderate)
- Smooth regions: 0.0-0.3 (low)

**Speed Patterns** (inverse of complexity):
- Complex areas: 0.3-0.5 (slow)
- Moderate detail: 0.5-0.7 (medium)
- Smooth areas: 0.7-1.0 (fast)

## Quality Considerations

### Common Issues

1. **Over-sensitivity**: Sigma too low captures noise as complexity
2. **Under-sensitivity**: Sigma too high misses important details
3. **Normalization artifacts**: Percentile too low clips useful range
4. **Speed range too narrow**: Insufficient variation in particle behavior

### Validation Tests

```python
# Check normalization range
assert np.all((result.complexity >= 0) & (result.complexity <= 1))

# Check speed range
if flow_speed is not None:
    assert np.all((flow_speed >= config.speed_min) &
                  (flow_speed <= config.speed_max))

# Verify metric consistency
assert result.metric in ["gradient", "laplacian", "multiscale_gradient"]

# Check mask application
if mask is not None:
    assert np.allclose(result.complexity[mask == 0], 0.0)
```

### Best Practices

1. **Start with gradient metric** (most predictable)
2. **Use percentile=99.0** for robust normalization
3. **Test with visualization** before export
4. **Adjust sigma based on image resolution**
5. **Use mask to focus on face region**
6. **Preview flow speed before finalizing**

## Mathematical Background

### Gradient Energy Theory

The gradient captures first-order intensity changes:
```
∇I = [∂I/∂x, ∂I/∂y]
|∇I| = √((∂I/∂x)² + (∂I/∂y)²)
```

Gradient magnitude indicates:
- Edge strength (high at boundaries)
- Texture presence (high in detailed areas)
- Smoothness (low in uniform regions)

### Laplacian Theory

The Laplacian captures second-order changes:
```
∇²I = ∂²I/∂x² + ∂²I/∂y²
```

Laplacian characteristics:
- Zero-crossing at edges
- High response to fine details
- Sensitive to noise (requires smoothing)

### Scale Space Theory

Multiscale analysis captures details at different frequencies:
```
L(x, y, σ) = G(σ) * I(x, y)
```

Where G(σ) is a Gaussian with standard deviation σ:
- Small σ: Fine details, high frequency
- Large σ: Coarse structure, low frequency

### Speed Modulation Rationale

Linear inverse mapping provides intuitive control:
```
speed = speed_max - complexity * (speed_max - speed_min)
```

This ensures:
- Smooth interpolation between extremes
- Predictable behavior
- Easy parameter tuning

Alternative mappings (future extensions):
- Exponential: `speed = speed_min + (speed_max - speed_min) * exp(-k * complexity)`
- Sigmoid: `speed = speed_min + (speed_max - speed_min) / (1 + exp(k * (complexity - 0.5)))`

## Integration with Flow Pipeline

The complexity map integrates seamlessly with the flow field generation:

1. **Complexity computed** after contour detection
2. **Masked** to face region using contour fill mask
3. **Flow speed derived** from normalized complexity
4. **Speed array passed** to flow field computation
5. **Particles modulated** during trajectory integration

This creates natural artistic behavior where particles:
- Move quickly through smooth areas (cheeks, forehead)
- Slow down in detailed regions (eyes, hair)
- Create appropriate line density based on local complexity

## CLI Usage

### Standalone Complexity

```bash
# Basic gradient complexity
uv run python scripts/run_pipeline.py complexity image.jpg

# Laplacian for texture emphasis
uv run python scripts/run_pipeline.py complexity --metric laplacian image.jpg

# Multiscale with custom weights
uv run python scripts/run_pipeline.py complexity \
    --metric multiscale_gradient \
    --scales 1.0 5.0 10.0 \
    --scale-weights 0.6 0.3 0.1 \
    image.jpg
```

### Flow with Complexity

```bash
# Flow with gradient-based speed modulation
uv run python scripts/run_pipeline.py flow \
    --metric gradient \
    --speed-min 0.2 \
    --speed-max 1.0 \
    image.jpg
```

### Full Pipeline

```bash
# Complete pipeline with complexity
uv run python scripts/run_pipeline.py all \
    --metric gradient \
    --complexity-sigma 3.0 \
    --normalize-percentile 99.0 \
    --speed-min 0.3 \
    --speed-max 1.0 \
    --export \
    image.jpg
```

## Next Stage

The complexity map feeds into the flow field system where the derived speed values modulate particle velocities. The export bundle includes both the complexity map and flow speed for consumption by the TypeScript plotter, enabling dynamic line behavior based on local image structure.