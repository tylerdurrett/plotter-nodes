# Stage 2: Region Masks Generation

## Overview

The region masks stage transforms facial landmarks into binary masks for semantic facial regions. These masks define the exact pixel boundaries of eyes and mouth, serving as the foundation for distance field computation and subsequent influence mapping.

## Purpose

Binary masks provide:
- Pixel-accurate boundaries of facial features
- Discrete regions for distance field computation
- Combined regions for unified processing (both eyes together)
- Visual debugging and validation tools
- Foundation for area-based calculations

## Process Flow

```
Landmarks (478 points)
    ↓
[Select Region Indices]
    ↓
Extract Polygon Vertices
    ↓
[Rasterize Polygon]
    ↓
Binary Mask (0 or 255)
    ↓
[Combine Masks (Optional)]
    ↓
Final Mask Set
```

## Region Definitions

### Standard Regions

The pipeline defines three primary regions by default:

#### Left Eye
```python
RegionDefinition(
    name="left_eye",
    landmark_indices=[263, 249, 390, 373, 374, 380, 381, 382,
                     362, 398, 384, 385, 386, 387, 388, 466]
)
```

#### Right Eye
```python
RegionDefinition(
    name="right_eye",
    landmark_indices=[33, 7, 163, 144, 145, 153, 154, 155,
                     133, 173, 157, 158, 159, 160, 161, 246]
)
```

#### Mouth
```python
RegionDefinition(
    name="mouth",
    landmark_indices=[61, 146, 91, 181, 84, 17, 314, 405, 321,
                     375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
)
```

### Combined Regions

The pipeline automatically generates:
- **combined_eyes**: Union of left_eye and right_eye masks
- Used for unified eye influence in final importance map

## Implementation

### Core Functions

#### Polygon Extraction
```python
def get_region_polygons(
    landmarks: LandmarkResult,
    regions: list[RegionDefinition]
) -> dict[str, np.ndarray]:
    """Extract polygon vertices for each region.

    Returns dict mapping region name to Nx2 vertex array.
    """
```

#### Mask Rasterization
```python
def rasterize_mask(
    polygon: np.ndarray,
    image_shape: tuple[int, int]
) -> np.ndarray:
    """Convert polygon to binary mask using OpenCV fillPoly.

    Returns uint8 array with values 0 (background) or 255 (feature).
    """
```

#### Batch Processing
```python
def build_region_masks(
    landmarks: LandmarkResult,
    regions: list[RegionDefinition]
) -> dict[str, np.ndarray]:
    """Generate all region masks plus combined variants.

    Returns dict with keys: left_eye, right_eye, mouth, combined_eyes
    """
```

## Rasterization Process

### Algorithm

1. **Polygon Formation**
   - Extract landmark coordinates by indices
   - Order vertices to form closed polygon
   - Convert to integer pixel coordinates

2. **Scan-line Filling**
   - OpenCV's `fillPoly` uses scan-line algorithm
   - Handles concave polygons correctly
   - Anti-aliasing disabled for binary output

3. **Mask Combination**
   - Bitwise OR operation for unions
   - Preserves individual feature boundaries
   - No overlap double-counting

### Technical Details

```python
# Rasterization implementation
mask = np.zeros(image_shape, dtype=np.uint8)
points = polygon.astype(np.int32).reshape((-1, 1, 2))
cv2.fillPoly(mask, [points], 255)
```

## Output Format

### Individual Masks

Each mask is a 2D numpy array:
- **Shape**: `(height, width)` matching input image
- **Dtype**: `uint8`
- **Values**: 0 (background) or 255 (feature region)
- **Channels**: Single channel (grayscale)

### Mask Dictionary

```python
masks = {
    "left_eye": np.ndarray,     # Left eye region
    "right_eye": np.ndarray,    # Right eye region
    "mouth": np.ndarray,        # Mouth region
    "combined_eyes": np.ndarray # Union of both eyes
}
```

## Visual Characteristics

### Eye Masks
- Elliptical or almond-shaped regions
- Include eyelid boundaries
- Exclude eyebrows and surrounding skin
- Typical area: 2-5% of image

### Mouth Mask
- Includes lips and immediate surrounding area
- Roughly elliptical or rectangular
- Excludes nose and chin
- Typical area: 3-7% of image

### Combined Eyes Mask
- Two disconnected regions (unless eyes very close)
- Preserves individual eye shapes
- Simplifies downstream processing
- Used for unified eye influence

## Quality Considerations

### Common Issues

1. **Incomplete Masks**
   - Landmarks at image edge
   - Partial face in frame
   - Extreme head angles

2. **Mask Overlap**
   - Very close features
   - Unusual expressions
   - Generally not problematic

3. **Mask Gaps**
   - Low landmark confidence
   - Occlusions affecting landmarks
   - Interpolation may be needed

### Validation

```python
# Check mask validity
def validate_mask(mask):
    area = np.sum(mask > 0)
    if area == 0:
        raise ValueError("Empty mask generated")
    if area < 100:  # pixels
        warnings.warn("Very small mask area")
    return True
```

## Customization

### Custom Regions

Define new regions by providing landmark indices:

```python
# Example: Nose region
nose_region = RegionDefinition(
    name="nose",
    landmark_indices=[1, 2, 5, 6, 19, 20, 94, 125, 235, 236]
)

# Add to pipeline config
config = PipelineConfig(
    regions=[left_eye, right_eye, mouth, nose_region]
)
```

### Region Modifications

- **Expand/Contract**: Morphological operations
- **Smooth**: Gaussian blur + threshold
- **Combine**: Bitwise operations
- **Exclude**: Bitwise NOT + AND

## Performance

- **Speed**: <5ms for all masks (1920x1080)
- **Memory**: ~2MB per mask at 1920x1080
- **Parallelizable**: Each mask independent
- **Cache-friendly**: Sequential array access

## Visualization

### Display Options

```python
# Overlay mask on image
overlay = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

# Show mask boundaries
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Create composite view
composite = np.hstack([
    cv2.cvtColor(mask_left_eye, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(mask_right_eye, cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(mask_mouth, cv2.COLOR_GRAY2BGR)
])
```

## Next Stage

The binary masks flow into [Stage 3: Distance Fields](03-distance-fields.md), where Euclidean distance transforms compute the distance from each pixel to the nearest mask boundary.