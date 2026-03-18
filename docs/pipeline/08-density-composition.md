# Stage 8: Density Composition

## Overview

The density composition stage combines tonal targets with importance maps through configurable blend modes and gamma correction to produce final density targets for particle drawing systems. This stage implements a two-stage composition strategy: first blending the maps, then applying gamma correction for fine-tuning the density distribution.

## Purpose

Density composition creates sophisticated density targets by combining multiple information sources:
- **Tonal information**: Where the image is naturally dark/light
- **Structural importance**: Where facial features are located
- **Artistic control**: Blend modes for different aesthetic effects
- **Fine-tuning**: Gamma correction for density distribution adjustment

The composed density map ensures particles concentrate in areas that are both:
- Tonally appropriate (dark regions need more lines)
- Structurally significant (eyes, mouth need definition)

## Algorithm

### Blend Modes

The system implements four blend modes, each with distinct characteristics:

#### Multiply Mode (Default)
```python
result = map_a * map_b
```
- **Effect**: Density only where BOTH maps have high values
- **Use case**: Conservative approach, ensures importance AND darkness
- **Characteristics**: Produces lower overall density, high selectivity

#### Screen Mode
```python
result = 1.0 - (1.0 - map_a) * (1.0 - map_b)
```
- **Effect**: Density where EITHER map has high values
- **Use case**: Inclusive approach, preserves both tonal and structural information
- **Characteristics**: Produces higher overall density, low selectivity

#### Max Mode
```python
result = np.maximum(map_a, map_b)
```
- **Effect**: Takes the higher value at each pixel
- **Use case**: Preserves peaks from both maps
- **Characteristics**: No value reduction, maintains local maxima

#### Weighted Mode
```python
# Normalized weighted sum
weight_sum = weight_a + weight_b
result = (map_a * weight_a + map_b * weight_b) / weight_sum
```
- **Effect**: Linear interpolation between maps
- **Use case**: Balanced contribution from both sources
- **Characteristics**: Smooth blending, predictable results

### Gamma Correction

After blending, gamma correction adjusts the density distribution:

```python
final = blended ** gamma
```

- **Gamma < 1.0**: Brightens (increases density values)
- **Gamma = 1.0**: No change (linear)
- **Gamma > 1.0**: Darkens (decreases density values)

## Implementation

### Core Functions

```python
def compose_maps(map_a: np.ndarray,
                map_b: np.ndarray,
                mode: str = "multiply") -> np.ndarray:
    """Blend two maps using specified mode.

    Parameters:
        map_a: First map, float64 [0, 1]
        map_b: Second map, float64 [0, 1]
        mode: Blend mode (multiply, screen, max, weighted)

    Returns:
        Blended map, float64 [0, 1]

    Raises:
        ValueError: Invalid mode or shape mismatch
    """

def build_density_target(tonal_target: np.ndarray,
                       importance: np.ndarray,
                       mode: str = "multiply",
                       gamma: float = 1.0) -> np.ndarray:
    """Build final density target with gamma correction.

    Parameters:
        tonal_target: Tonal density from luminance
        importance: Combined feature importance
        mode: Blend mode for composition
        gamma: Gamma correction exponent

    Returns:
        Final density target, float64 [0, 1]
    """
```

### Processing Pipeline

1. **Input Validation**
   - Verify matching shapes
   - Check value ranges [0, 1]
   - Validate blend mode string

2. **Map Composition**
   - Apply selected blend mode
   - Handle edge cases (zeros, ones)
   - Maintain numerical stability

3. **Gamma Correction**
   - Apply power function
   - Handle zero values safely
   - Clip to [0, 1] range

4. **Output Generation**
   - Return composed density map
   - Preserve float64 precision
   - Include in DensityResult

## Configuration

### ComposeConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `luminance` | LuminanceConfig | Default config | CLAHE parameters |
| `feature_weight` | float | 0.6 | Weight for feature importance |
| `contour_weight` | float | 0.4 | Weight for contour importance |
| `tonal_blend_mode` | str | "multiply" | How to combine tonal and importance |
| `tonal_weight` | float | 1.0 | Tonal map weight (weighted mode) |
| `importance_weight` | float | 1.0 | Importance weight (weighted mode) |
| `gamma` | float | 1.0 | Gamma correction exponent |

### Mode Selection Guidelines

| Mode | When to Use | Visual Result |
|------|-------------|---------------|
| **multiply** | Default choice, natural looking | Sharp feature definition, clean backgrounds |
| **screen** | Preserve all information | Softer, more inclusive density |
| **max** | Emphasize peaks | Strong local features |
| **weighted** | Fine control needed | Balanced, tunable blend |

### Gamma Adjustment Guidelines

| Gamma | Effect | Use Case |
|-------|--------|----------|
| 0.5 | Strong brightening | Increase overall density |
| 0.7 | Moderate brightening | Subtle density boost |
| 1.0 | No change (default) | Use raw blend result |
| 1.5 | Moderate darkening | Reduce density, increase contrast |
| 2.0 | Strong darkening | Sparse, high-contrast result |

## Output Format

### DensityResult Structure

```python
@dataclass(frozen=True)
class DensityResult:
    luminance: np.ndarray        # Original grayscale
    clahe_luminance: np.ndarray  # Enhanced contrast
    tonal_target: np.ndarray     # Inverted CLAHE
    importance: np.ndarray       # Combined feature importance
    density_target: np.ndarray   # Final composed result
```

### Density Value Interpretation

- **0.0**: No particle activity (blank areas)
- **0.2**: Light coverage (subtle shading)
- **0.5**: Medium density (standard shading)
- **0.8**: Heavy coverage (dark shadows)
- **1.0**: Maximum density (deepest blacks)

## Visualization

### Output Files

The density pipeline saves:
- `density/luminance.png`: Original luminance extraction
- `density/clahe_luminance.png`: Contrast-enhanced version
- `density/tonal_target.png`: Inverted tonal map (hot colormap)
- `density/importance.png`: Combined importance (inferno colormap)
- `density/density_target.png`: Final composed result (hot colormap)
- `density/density_target_raw.npy`: Raw float64 array
- `density/contact_sheet.png`: Grid of all visualizations

### Visual Characteristics

Expected patterns by blend mode:

**Multiply Mode**:
- Clean, well-defined features
- Low density in unimportant bright areas
- High contrast between features and background

**Screen Mode**:
- Softer, more distributed density
- Preserves both tonal and structural information
- Lower contrast, smoother gradients

**Max Mode**:
- Sharp preservation of peaks
- No reduction in local maxima
- Can appear somewhat disconnected

**Weighted Mode**:
- Smooth interpolation
- Predictable, linear blending
- Good for fine-tuning balance

## Quality Considerations

### Common Issues

1. **Over-darkening**: Gamma too high, reduces detail
2. **Loss of structure**: Multiply mode with weak importance
3. **Over-saturation**: Screen mode with strong inputs
4. **Imbalance**: Poor weight selection in weighted mode

### Optimization Strategy

1. Start with defaults (multiply mode, gamma=1.0)
2. Preview intermediate results
3. Adjust blend mode based on desired aesthetic
4. Fine-tune gamma for density distribution
5. Use weighted mode for precise control

### Best Practices

- **Multiply** for portraits with clear features
- **Screen** for low-contrast or difficult images
- **Max** for preserving specific features
- **Weighted** with iterative weight tuning
- Gamma adjustment as final refinement step

## Integration

### Pipeline Context

Density composition integrates three information sources:

```python
# From Stage 1: Feature importance
feature_result = run_feature_distance_pipeline(image)

# From Stage 1: Contour importance
contour_result = run_contour_pipeline(image)

# From Stage 7: Tonal target
tonal_result = compute_tonal_target(image, config.luminance)

# Combine feature and contour importance
importance = combine_maps(
    [feature_result.combined, contour_result.influence_map],
    [config.feature_weight, config.contour_weight]
)

# Build final density target
density_target = build_density_target(
    tonal_target=tonal_result.tonal_target,
    importance=importance,
    mode=config.tonal_blend_mode,
    gamma=config.gamma
)
```

### Downstream Usage

The density target drives particle systems:
- High values → More particle visits
- Low values → Fewer particle visits
- Zero values → No particle activity

## Next Stage

The density target provides one of the key inputs for the drawing system, while [Stage 9: Edge Tangent Field](09-edge-tangent-field.md) begins computing the flow fields that will guide particle movement direction.