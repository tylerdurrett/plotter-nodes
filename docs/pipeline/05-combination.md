# Stage 5: Weighted Combination

## Overview

The combination stage merges individual influence maps (eyes, mouth) into a single importance map using weighted summation. This final stage produces the unified importance field that represents the overall visual significance across the portrait.

## Purpose

The combination stage:
- Unifies multiple feature influences into one map
- Allows artistic control over feature emphasis
- Normalizes combined values to standard range
- Handles overlapping influence zones
- Produces the final output for downstream use

## Mathematical Foundation

### Weighted Sum

Given influence maps I₁, I₂, ..., Iₙ with weights w₁, w₂, ..., wₙ:

```
Combined(x,y) = Σᵢ wᵢ × Iᵢ(x,y) / Σᵢ |wᵢ|
```

Where:
- Iᵢ(x,y) ∈ [0,1] is the influence at pixel (x,y) for feature i
- wᵢ is the weight for feature i
- Division by Σ|wᵢ| ensures output remains in [0,1]

### Properties

- **Linear combination**: Preserves smoothness of input maps
- **Normalized output**: Always in [0.0, 1.0] range
- **Commutative**: Order of maps doesn't matter
- **Interpolation**: Result interpolates between inputs

## Implementation

### Core Function

```python
def combine_maps(
    maps: dict[str, np.ndarray],
    weights: dict[str, float]
) -> np.ndarray:
    """Combine influence maps using weighted sum.

    Parameters:
        maps: Dictionary of influence maps (all same shape)
        weights: Dictionary of weights (keys match maps)

    Returns:
        Combined map (float64) normalized to [0.0, 1.0]
    """
    # Validate matching keys
    if set(weights.keys()) != set(maps.keys()):
        raise ValueError("Weight keys must match map keys")

    # Initialize result
    result = np.zeros(reference_shape, dtype=np.float64)

    # Weighted summation
    total_weight = 0.0
    for name, influence_map in maps.items():
        weight = weights[name]
        if weight != 0:
            result += weight * influence_map
            total_weight += abs(weight)

    # Normalize by total weight
    if total_weight > 0:
        result /= total_weight

    return np.clip(result, 0.0, 1.0)
```

## Weight Configuration

### Standard Weights

Default configuration emphasizes eyes over mouth:
```python
weights = {
    "eyes": 0.6,   # 60% influence from eyes
    "mouth": 0.4   # 40% influence from mouth
}
```

### Weight Interpretation

| Eyes | Mouth | Effect |
|------|-------|--------|
| 0.8 | 0.2 | Strong eye focus, minimal mouth |
| 0.6 | 0.4 | Balanced with eye preference |
| 0.5 | 0.5 | Equal importance |
| 0.3 | 0.7 | Mouth-dominant |
| 1.0 | 0.0 | Eyes only |

### Normalization

Weights don't need to sum to 1.0:
- `{eyes: 3, mouth: 2}` equivalent to `{eyes: 0.6, mouth: 0.4}`
- Automatic normalization by sum of absolute values
- Negative weights supported (subtract influence)

## Combination Behaviors

### Non-overlapping Regions

When influence zones don't overlap:
- Simple addition of weighted values
- Each region maintains its original structure
- No interaction effects

### Overlapping Regions

When influence zones overlap:
- Values sum constructively
- Creates natural blending
- Peak values where both influences strong

### Edge Cases

1. **Single feature**: `weights = {eyes: 1.0, mouth: 0.0}`
   - Output equals eye influence map

2. **Equal weights**: `weights = {eyes: 0.5, mouth: 0.5}`
   - Simple average of both maps

3. **Negative weight**: `weights = {eyes: 1.0, mouth: -0.2}`
   - Subtracts mouth influence from eyes

## Visual Characteristics

### Balanced Combination (0.6/0.4)
- Smooth transitions between features
- Natural-looking importance distribution
- Eyes slightly emphasized
- Good for realistic rendering

### Eye-Dominant (0.8/0.2)
- Strong peaks around eyes
- Mouth barely visible
- Good for eye-focused portraits
- May lose facial balance

### Mouth-Dominant (0.3/0.7)
- Emphasized lower face
- Good for expression focus
- May underrepresent eye detail

## Advanced Techniques

### Dynamic Weighting

Adjust weights based on image analysis:
```python
# Increase mouth weight for smiling portraits
if detect_smile(image):
    weights["mouth"] = 0.6
    weights["eyes"] = 0.4
```

### Multi-Feature Extension

Adding more features:
```python
weights = {
    "eyes": 0.4,
    "mouth": 0.3,
    "nose": 0.2,
    "eyebrows": 0.1
}
```

### Non-Linear Combination

Alternative combination strategies:
```python
# Maximum (instead of sum)
combined = np.maximum(eyes * 0.6, mouth * 0.4)

# Multiplicative
combined = (eyes ** 0.6) * (mouth ** 0.4)

# Harmonic mean
combined = 2 / (1/eyes + 1/mouth)
```

## Output Characteristics

### Value Distribution

Typical combined importance map:
- **Range**: [0.0, 1.0]
- **Mean**: ~0.15-0.25
- **Peak values**: 0.8-1.0 at feature centers
- **Background**: 0.0-0.1 at image edges

### Spatial Properties

- **Smoothness**: Inherited from influence maps
- **Continuity**: C∞ continuous (infinitely differentiable)
- **Gradient**: Smooth transitions, no discontinuities
- **Coverage**: 20-40% of pixels > 0.1

## Quality Validation

### Checks

```python
def validate_combined_map(combined):
    # Value range
    assert combined.min() >= 0.0
    assert combined.max() <= 1.0

    # No invalid values
    assert not np.any(np.isnan(combined))
    assert not np.any(np.isinf(combined))

    # Reasonable coverage
    high_importance = np.sum(combined > 0.5)
    total_pixels = combined.size
    coverage = high_importance / total_pixels

    if coverage < 0.01:
        warnings.warn("Very little high-importance area")
    if coverage > 0.5:
        warnings.warn("Excessive high-importance area")
```

### Visual Inspection

Check for:
- Natural falloff from features
- No hard edges or artifacts
- Balanced feature representation
- Smooth blending in overlap zones

## Performance Considerations

### Computational Cost

- **Operation**: Element-wise multiplication and addition
- **Complexity**: O(n) for n pixels
- **Memory**: Single pass, no intermediate storage
- **Vectorized**: Fully NumPy-optimized

### Optimization

```python
# Pre-allocate result array
result = np.empty_like(maps[0])

# Fused multiply-add
np.multiply(weight, influence, out=temp)
np.add(result, temp, out=result)
```

## Visualization

### Heatmap Display

```python
# Apply "hot" colormap for importance visualization
importance_rgb = cv2.applyColorMap(
    (combined * 255).astype(np.uint8),
    cv2.COLORMAP_HOT
)
```

### Contour Visualization

```python
# Show importance level contours
levels = [0.1, 0.3, 0.5, 0.7, 0.9]
contours = []
for level in levels:
    mask = (combined >= level).astype(np.uint8)
    c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.extend(c)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
```

## Output Usage

The combined importance map serves as input for:
- **Density mapping**: Line/dot density proportional to importance
- **Detail allocation**: More detail where importance high
- **Sampling strategies**: Biased sampling toward important areas
- **Computational focus**: Allocate resources by importance
- **Artistic effects**: Vary style/technique by importance level

## Pipeline Complete

The combined importance map represents the final output of the portrait importance pipeline, ready for use in downstream drawing and rendering algorithms.