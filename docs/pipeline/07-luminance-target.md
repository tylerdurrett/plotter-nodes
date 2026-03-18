# Stage 7: Luminance & Tonal Target

## Overview

The luminance and tonal target stage extracts grayscale luminance from portrait images and processes it through Contrast Limited Adaptive Histogram Equalization (CLAHE) to create a tonal density target. This stage produces an inverted luminance map where darker regions of the original image become high-density targets for particle drawing systems.

## Purpose

Luminance-based density targeting serves as a fundamental component of the density composition pipeline, providing:
- Tonal structure extracted from the original portrait
- Enhanced local contrast through adaptive equalization
- Natural density distribution following photographic tones
- Foundation for multi-layer density composition

The inverted CLAHE result creates a density map where:
- Dark areas (shadows, hair) → High density values (more drawing activity)
- Bright areas (highlights, skin) → Low density values (less drawing activity)

## Technology

### Luminance Extraction

The pipeline uses OpenCV's color space conversion to extract grayscale luminance:
- **BGR to Gray**: Standard luminance weights (0.299R + 0.587G + 0.114B)
- **Normalization**: Float64 values in [0, 1] range
- **Preservation**: Maintains full tonal information

### CLAHE (Contrast Limited Adaptive Histogram Equalization)

CLAHE enhances local contrast while preventing over-amplification:
- **Adaptive**: Processes image in tiles, each equalized independently
- **Limited**: Clips histogram to prevent noise amplification
- **Smooth**: Bilinear interpolation between tiles
- **Configurable**: Adjustable clip limit and tile size

## Implementation

### Core Functions

```python
def extract_luminance(image: np.ndarray) -> np.ndarray:
    """Extract grayscale luminance from BGR or grayscale image.

    Parameters:
        image: BGR uint8 (H, W, 3) or grayscale uint8 (H, W)

    Returns:
        Grayscale luminance as float64 [0, 1]
    """

def apply_clahe(luminance: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE to enhance local contrast.

    Parameters:
        luminance: Float64 grayscale [0, 1]
        clip_limit: Contrast limiting threshold
        tile_size: Grid size for adaptive tiles

    Returns:
        Enhanced luminance as float64 [0, 1]
    """

def compute_tonal_target(image: np.ndarray,
                        config: LuminanceConfig | None = None) -> tuple[np.ndarray, ...]:
    """Compute complete tonal target from image.

    Parameters:
        image: Input BGR image
        config: Optional luminance configuration

    Returns:
        Tuple of (luminance, clahe_luminance, tonal_target)
    """
```

### Processing Steps

1. **Luminance Extraction**
   - Convert BGR to grayscale using standard weights
   - Normalize to float64 [0, 1] range
   - Handle both color and grayscale inputs

2. **CLAHE Enhancement**
   - Convert to uint8 for OpenCV processing
   - Create CLAHE object with specified parameters
   - Apply tile-based adaptive equalization
   - Convert back to float64 [0, 1]

3. **Tonal Target Inversion**
   - Invert CLAHE result: `tonal_target = 1.0 - clahe_luminance`
   - Dark regions become high values (dense drawing)
   - Bright regions become low values (sparse drawing)

4. **Result Packaging**
   - Return all intermediates for visualization
   - Maintain float64 precision throughout

## Configuration

### LuminanceConfig Parameters

| Parameter | Type | Default | Description | Effect |
|-----------|------|---------|-------------|---------|
| `clip_limit` | float | 2.0 | CLAHE contrast limit | Higher = more local contrast |
| `tile_size` | int | 8 | Adaptive tile grid size | Smaller = more local adaptation |

### Parameter Guidelines

**Clip Limit** (1.0 - 10.0):
- `1.0`: Minimal enhancement, preserves original contrast
- `2.0`: Moderate enhancement (default, good for most portraits)
- `4.0`: Strong enhancement, may amplify noise
- `10.0`: Extreme enhancement, artistic effect

**Tile Size** (4 - 16):
- `4`: Very local adaptation, fine details
- `8`: Balanced local/global (default)
- `16`: More global adaptation, smoother transitions

## Output Format

### Tonal Target Arrays

The stage produces three arrays in `DensityResult`:

```python
luminance: np.ndarray         # Original grayscale [0, 1]
clahe_luminance: np.ndarray   # Enhanced contrast [0, 1]
tonal_target: np.ndarray      # Inverted density target [0, 1]
```

### Value Interpretation

- **0.0**: No drawing activity (originally white)
- **0.5**: Medium density (originally mid-gray)
- **1.0**: Maximum density (originally black)

## Visualization

The tonal target visualization includes:

### Individual Components
- **Luminance**: Original grayscale extraction
- **CLAHE Luminance**: Contrast-enhanced version
- **Tonal Target**: Final inverted density map

### Colormap Recommendations
- **Luminance/CLAHE**: Grayscale or viridis (shows tones)
- **Tonal Target**: Hot or plasma (shows density distribution)

### Visual Characteristics

Expected patterns in tonal target:
- Hair regions: High values (0.8-1.0)
- Shadows: Moderate-high values (0.6-0.8)
- Skin tones: Low-moderate values (0.2-0.5)
- Highlights: Very low values (0.0-0.2)

## Quality Considerations

### Input Requirements

- Well-exposed portraits work best
- Avoid extreme over/underexposure
- Natural lighting produces better tonal gradients
- Higher resolution preserves fine tonal details

### Common Artifacts

1. **Tile boundaries**: Visible if tile_size too large for image
2. **Noise amplification**: High clip_limit on grainy images
3. **Halo effects**: Around high-contrast edges with aggressive settings
4. **Loss of detail**: Over-equalization in uniform regions

### Best Practices

- Start with default parameters (clip_limit=2.0, tile_size=8)
- Increase clip_limit gradually for more contrast
- Adjust tile_size based on image resolution
- Preview intermediates to diagnose issues

## Integration

### Density Pipeline Role

The tonal target serves as one input to density composition:

```python
# In compose.py
density_target = build_density_target(
    tonal_target=tonal_result.tonal_target,
    importance=combined_importance,
    mode="multiply",  # Multiplicative blend
    gamma=1.0
)
```

### Combination with Importance

- **Multiply mode**: Density only where both dark AND important
- **Screen mode**: Density where either dark OR important
- **Max mode**: Take higher of tonal or importance
- **Weighted mode**: Linear blend of the two

## Next Stage

The tonal target flows into [Stage 8: Density Composition](08-density-composition.md), where it combines with feature importance maps to create the final density target for particle drawing systems.