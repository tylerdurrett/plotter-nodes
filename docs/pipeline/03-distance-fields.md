# Stage 3: Distance Fields Computation

## Overview

The distance fields stage computes Euclidean distance transforms from binary masks, creating continuous fields that encode the distance from each pixel to the nearest feature boundary. These fields provide smooth, continuous gradients essential for natural influence falloff.

## Purpose

Distance fields enable:
- Smooth gradients from feature boundaries
- Continuous influence falloff (no hard edges)
- Mathematical basis for remapping functions
- Natural blending between regions
- Resolution-independent representation

## Mathematical Foundation

### Euclidean Distance Transform

For a binary mask M where M(x,y) ∈ {0, 1}, the Euclidean distance transform D is:

```
D(x,y) = min{||p - (x,y)||₂ : M(p) = 1}
```

Where:
- D(x,y) is the distance at pixel (x,y)
- p ranges over all pixels where M(p) = 1 (inside mask)
- ||·||₂ denotes Euclidean (L2) distance

### Properties

- **Continuity**: D is continuous across the entire image
- **Minimum at boundaries**: D = 0 where mask transitions 0→1
- **Monotonic increase**: D increases with distance from features
- **Isotropic**: Equal distance in all directions (circular contours)

## Implementation

### Core Function

```python
def compute_distance_field(mask: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance field from binary mask.

    Parameters:
        mask: Binary mask (uint8, values 0 or 255)

    Returns:
        Distance field (float64) with pixel-unit distances
    """
    # Invert mask: compute distance FROM feature
    inverted = ~(mask > 0)

    # Apply Euclidean distance transform
    distance_field = distance_transform_edt(inverted)

    return distance_field.astype(np.float64)
```

### Algorithm Details

The implementation uses SciPy's `distance_transform_edt`:
- **Method**: Fast marching or chamfer-based algorithm
- **Complexity**: O(n) for n pixels (linear scan)
- **Accuracy**: Exact Euclidean distances
- **Memory**: In-place transformation available

## Process Flow

```
Binary Mask (0/255)
    ↓
[Invert Mask]
    ↓
Distance Transform
    ↓
Float64 Distance Field
    ↓
[Apply to Each Region]
    ↓
Distance Field Set
```

## Output Characteristics

### Field Structure

Each distance field is a 2D array:
- **Shape**: `(height, width)` matching input image
- **Dtype**: `float64` for precision
- **Values**: 0.0 to max_distance (pixels)
- **Units**: Pixels (image coordinate space)

### Value Distribution

```python
# Typical statistics for eye distance field
min: 0.0          # At eye boundary
max: ~500.0       # Far corners of image
mean: ~150.0      # Average distance
median: ~120.0    # 50th percentile
```

### Gradient Properties

- **Gradient magnitude**: 1.0 everywhere (unit speed)
- **Gradient direction**: Points toward nearest feature
- **Level sets**: Form concentric contours around features

## Visual Interpretation

### Distance Ranges

| Distance (pixels) | Interpretation | Visual Impact |
|------------------|----------------|---------------|
| 0-30 | Immediate vicinity | Core feature area |
| 30-80 | Near field | Strong influence zone |
| 80-150 | Mid field | Moderate influence |
| 150-300 | Far field | Weak influence |
| >300 | Background | Minimal/no influence |

### Heatmap Visualization

```python
# Normalize and visualize distance field
def visualize_distance_field(field, max_dist=300):
    # Clamp to reasonable range
    normalized = np.clip(field, 0, max_dist) / max_dist

    # Apply colormap (viridis: purple→blue→green→yellow)
    heatmap = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS
    )
    return heatmap
```

## Multiple Region Handling

### Separate Fields

The pipeline computes independent distance fields:
```python
distance_fields = {
    "eyes": compute_distance_field(masks["combined_eyes"]),
    "mouth": compute_distance_field(masks["mouth"])
}
```

### Field Interaction

When regions are close:
- Fields remain independent (no interaction)
- Influence combination happens later
- Overlapping influence zones blend naturally

## Performance Optimization

### Computational Complexity

- **Time**: O(n) for n pixels
- **Space**: O(n) for distance array
- **Cache efficiency**: Sequential memory access
- **Parallelizable**: Each field independent

### Optimization Strategies

1. **Downsampling**
   ```python
   # Compute at lower resolution
   small_mask = cv2.resize(mask, (width//2, height//2))
   small_field = compute_distance_field(small_mask)
   field = cv2.resize(small_field, (width, height)) * 2
   ```

2. **Approximation**
   ```python
   # Use chamfer distance (faster, less accurate)
   field = cv2.distanceTransform(
       inverted, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
   )
   ```

3. **Caching**
   ```python
   # Cache computed fields for reuse
   @lru_cache(maxsize=10)
   def cached_distance_field(mask_hash):
       return compute_distance_field(mask)
   ```

## Quality Considerations

### Numerical Precision

- **Float64**: Maintains accuracy for large distances
- **Rounding errors**: Negligible for typical image sizes
- **Boundary accuracy**: Sub-pixel precision at mask edges

### Edge Cases

1. **Empty mask**: Returns all infinite distances
2. **Full mask**: Returns all zeros
3. **Disconnected regions**: Each component treated independently
4. **Image boundaries**: Correct distances to image edge

## Physical Interpretation

Distance fields can be interpreted as:
- **Time**: Propagation time at unit speed
- **Energy**: Potential energy landscape
- **Heat**: Temperature distribution from heat sources
- **Light**: Illumination falloff from light sources

## Applications Beyond Pipeline

Distance fields enable:
- **Morphological operations**: Precise dilation/erosion
- **Skeleton extraction**: Medial axis transform
- **Voronoi diagrams**: Nearest neighbor regions
- **Path planning**: Gradient descent navigation
- **Artistic effects**: Glow, blur, stylization

## Next Stage

The distance fields flow into [Stage 4: Influence Maps](04-influence-maps.md), where they are remapped through configurable falloff curves to create normalized influence values.