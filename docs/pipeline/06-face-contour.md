# Stage 6: Face Contour Distance Map

## Overview

The face contour distance map stage generates signed distance fields from the face boundary, creating continuous gradients that encode the distance from each pixel to the face oval contour. Unlike the feature-based pipeline that focuses on eyes and mouth, this pipeline emphasizes the overall face boundary, enabling effects that radiate from or toward the face edge.

## Purpose

Face contour distance maps provide:
- Signed distance fields with interior/exterior distinction
- Continuous gradients from the face boundary
- Directional control (inward/outward/both/band)
- Natural falloff for face-edge effects
- Foundation for boundary-based artistic rendering
- Complementary maps to feature-based importance

## Technology

### MediaPipe Face Mesh Contour

The pipeline leverages MediaPipe's 478 facial landmarks to extract the face boundary:
- **Convex Hull Approach**: Computes the convex hull of all 478 landmarks
- **Full Coverage**: Captures the entire visible face including cheeks
- **Robust Boundary**: Handles 3D→2D projection artifacts
- **No Tuning Required**: Parameter-free boundary extraction

### Signed Distance Fields

The core mathematical representation:
- **Euclidean Distance Transform**: Exact distances to contour
- **Sign Convention**: Negative inside, positive outside, zero on contour
- **Continuous Gradients**: Smooth transitions across boundaries
- **Direction Modes**: Configurable influence direction

## Implementation

### Core Functions

```python
def get_face_oval_polygon(landmarks: LandmarkResult) -> np.ndarray:
    """Extract face boundary using convex hull of all landmarks.

    Parameters:
        landmarks: Detected face landmarks (478 points)

    Returns:
        Polygon vertices as (N, 2) float64 array
    """

def compute_signed_distance(
    contour_mask: np.ndarray,
    filled_mask: np.ndarray
) -> np.ndarray:
    """Compute signed distance field from contour.

    Parameters:
        contour_mask: Binary mask of contour line (0/255)
        filled_mask: Binary mask of filled region (0/255)

    Returns:
        Signed distances (negative inside, positive outside)
    """

def prepare_directional_distance(
    signed_distance: np.ndarray,
    mode: str = "inward",
    clamp_value: float = 9999.0,
    band_width: float | None = None
) -> np.ndarray:
    """Convert signed distance to directional distance.

    Parameters:
        signed_distance: Signed distance field
        mode: Direction mode (inward/outward/both/band)
        clamp_value: Value for clamped regions
        band_width: Width for band mode (pixels)

    Returns:
        Unsigned directional distance field
    """
```

## Process Flow

```
Input Image
    ↓
[Detect Landmarks]
    → 478 facial points
    ↓
[Extract Face Boundary]
    → Convex hull polygon
    ↓
[Rasterize Masks]
    ├→ Contour mask (polyline)
    └→ Filled mask (polygon)
    ↓
[Compute Signed Distance]
    → Signed distance field
    ↓
[Apply Direction Mode]
    → Directional distance
    ↓
[Remap to Influence]
    → Normalized influence map (0.0-1.0)
```

## Mathematical Foundation

### Signed Distance Field

For a contour C and filled region F, the signed distance field D is:

```
D(x,y) = {
    -min{||p - (x,y)||₂ : p ∈ C}  if (x,y) ∈ F (inside)
     min{||p - (x,y)||₂ : p ∈ C}  if (x,y) ∉ F (outside)
     0                              if (x,y) ∈ C (on contour)
}
```

Where:
- D(x,y) is the signed distance at pixel (x,y)
- C is the set of contour pixels
- F is the set of filled region pixels
- ||·||₂ denotes Euclidean distance

### Properties

- **Continuity**: D is continuous everywhere except at the contour
- **Zero Level Set**: The contour is exactly where D = 0
- **Gradient Unity**: |∇D| = 1 almost everywhere
- **Sign Preservation**: Interior/exterior distinction maintained

## Contour Extraction

### Convex Hull Method

The landmark-based contour extraction uses the convex hull of all 478 landmarks:

```python
# Compute convex hull of all 478 landmarks
points_2d = landmarks.landmarks[:, :2]
hull = ConvexHull(points_2d)
polygon = points_2d[hull.vertices]
```

**Advantages:**
- Captures full visible face boundary
- Includes lateral cheek regions
- No parameter tuning needed
- Robust to 3D projection

## Direction Modes

### Inward Mode
```python
config = ContourConfig(direction="inward")
```

**Behavior:**
- Keeps interior distances (face center)
- Clamps exterior to large value
- Influence strongest at boundary, falls toward center
- **Use Case:** Face-centric effects, interior emphasis

### Outward Mode
```python
config = ContourConfig(direction="outward")
```

**Behavior:**
- Keeps exterior distances (background)
- Clamps interior to large value
- Influence strongest at boundary, falls toward edges
- **Use Case:** Halo effects, background emphasis

### Both Mode
```python
config = ContourConfig(direction="both")
```

**Behavior:**
- Uses absolute value everywhere
- Symmetric falloff from contour
- No clamping applied
- **Use Case:** Balanced boundary effects

### Band Mode
```python
config = ContourConfig(direction="band", band_width=50.0)
```

**Behavior:**
- Uses absolute value within band_width
- Clamps beyond specified width
- Creates fixed-width influence zone
- **Use Case:** Precise boundary control

## Output Characteristics

### File Structure

```
output/<image_name>/contour/
├── contour_overlay.png           # Contour drawn on original
├── contour_mask.png              # Binary contour line
├── filled_mask.png               # Binary filled region
├── signed_distance_raw.npy       # Float64 signed distances
├── signed_distance_heatmap.png   # Diverging colormap (RdBu)
├── directional_distance_raw.npy  # Float64 directional distances
├── directional_distance_heatmap.png # Sequential colormap (viridis)
├── contour_influence.png         # Final influence map (inferno)
└── contact_sheet.png            # All visualizations combined
```

### Data Types

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| contour_polygon | float64 (N,2) | Image bounds | Boundary vertices |
| contour_mask | uint8 (H,W) | {0, 255} | Contour pixels |
| filled_mask | uint8 (H,W) | {0, 255} | Interior pixels |
| signed_distance | float64 (H,W) | [-max, +max] | Signed distances |
| directional_distance | float64 (H,W) | [0, clamp] | Unsigned distances |
| influence_map | float64 (H,W) | [0, 1] | Normalized influence |

## Configuration

### ContourConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remap` | RemapConfig | RemapConfig() | Influence remapping settings |
| `direction` | str | "inward" | Direction mode (inward/outward/both/band) |
| `band_width` | float \| None | None | Band width for band mode (pixels) |
| `contour_thickness` | int | 1 | Contour line thickness (pixels) |
| `output_dir` | str | "output" | Base output directory |

### RemapConfig Integration

The contour pipeline uses the same remapping configuration as the feature pipeline:

```python
config = ContourConfig(
    remap=RemapConfig(
        curve="gaussian",      # Remapping curve type
        sigma=80.0,           # Gaussian spread
        clamp_distance=300.0  # Maximum distance
    ),
    direction="inward"
)
```

## Visual Interpretation

### Signed Distance Heatmap

The signed distance visualization uses a diverging colormap (RdBu):

| Color | Value | Location |
|-------|-------|----------|
| Blue | Negative | Inside face (center) |
| White | ~Zero | At contour boundary |
| Red | Positive | Outside face (background) |

### Influence Map Patterns

| Direction | Center | Boundary | Background |
|-----------|--------|----------|------------|
| Inward | Low influence | High influence | No influence |
| Outward | No influence | High influence | Low influence |
| Both | Low influence | High influence | Low influence |
| Band | No influence | High influence | No influence |

## Quality Considerations

### Numerical Precision

- **Float64 Arrays**: Maintains precision for large distances
- **Contour Priority**: Pixels on contour always get distance = 0
- **Symmetric Normalization**: Diverging colormaps centered at zero

### Edge Cases

1. **No Face Detected**: Raises ValueError, pipeline skips
2. **Partial Face**: Convex hull still produces valid boundary
3. **Multiple Faces**: Uses first detected face
4. **Image Boundaries**: Correct distances to image edges

### Robustness Features

- **Convex Hull**: Always produces simple, closed polygon
- **Binary Masks**: Threshold-independent (0/255 only)
- **Signed Distance**: Preserves all spatial information

## Performance Optimization

### Computational Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Convex Hull | O(n log n) | ~1ms for 478 points |
| Mask Rasterization | O(w × h) | ~5ms for 1024×1024 |
| Distance Transform | O(w × h) | ~10ms for 1024×1024 |
| Influence Remap | O(w × h) | ~5ms for 1024×1024 |

### Optimization Strategies

1. **Precomputed Hull**: Cache hull for multiple renders
2. **Lower Resolution**: Compute at half size, upsample
3. **Band Mode**: Reduces computation outside band
4. **Parallel Processing**: Independent distance transforms

## Applications

### Artistic Rendering

- **Contour Emphasis**: Dense strokes along face boundary
- **Face Halos**: Glowing effects around face
- **Boundary Particles**: Spawn points at face edge
- **Gradient Flows**: Vector fields from signed distance

### Computer Vision

- **Face Segmentation**: Interior/exterior classification
- **Boundary Refinement**: Sub-pixel contour extraction
- **Shape Analysis**: Face shape metrics from contour
- **Tracking**: Temporal contour correspondence

### Hybrid Approaches

Combine with feature pipeline for rich effects:
```python
# Run both pipelines
feature_result = run_feature_distance_pipeline(image)
contour_result = run_contour_pipeline(image)

# Blend influence maps
combined = 0.7 * feature_result.combined + 0.3 * contour_result.influence_map
```

## Integration with Feature Pipeline

### Complementary Information

| Aspect | Feature Pipeline | Contour Pipeline |
|--------|-----------------|------------------|
| Focus | Eyes, mouth | Face boundary |
| Gradients | Multiple centers | Single boundary |
| Direction | Outward from features | In/out from edge |
| Use Case | Feature emphasis | Boundary effects |

### CLI Subcommands

```bash
# Feature pipeline only
uv run python scripts/run_pipeline.py features image.jpg

# Contour pipeline only
uv run python scripts/run_pipeline.py contour image.jpg

# Both pipelines
uv run python scripts/run_pipeline.py all image.jpg
```

### Shared Infrastructure

Both pipelines share:
- Landmark detection (MediaPipe)
- Distance field computation
- Influence remapping (RemapConfig)
- Visualization utilities
- Output management

## Next Stage

The face contour distance maps can be:
1. Used standalone for boundary-based effects
2. Combined with feature maps for hybrid importance
3. Extended to other facial regions (nose, eyebrows)
4. Integrated into generative drawing algorithms

For technical details on combining multiple influence maps, see [Stage 5: Combination](05-combination.md).