# Stage 1: Landmark Detection

## Overview

The landmark detection stage identifies and extracts facial features from a portrait image using MediaPipe's Face Mesh model. This stage produces 478 3D facial landmarks that form a detailed mesh representation of the face, providing the geometric foundation for all subsequent processing.

## Purpose

Landmark detection serves as the entry point to the pipeline, transforming raw pixel data into structured geometric information. The detected landmarks define:
- Precise boundaries of facial features (eyes, mouth, nose, face contour)
- Spatial relationships between features
- Face pose and orientation
- Confidence metrics for quality assessment

## Technology

### MediaPipe Face Mesh

The pipeline uses Google's MediaPipe Face Mesh, a real-time face geometry solution that:
- Detects 478 3D facial landmarks
- Works on single RGB images
- Provides sub-pixel precision
- Includes confidence scores per landmark
- Runs efficiently on CPU

### Model Details

- **Model**: `face_landmarker.task` (float16 variant)
- **Input**: RGB image of any size
- **Output**: 478 landmarks with (x, y, z) coordinates + presence confidence
- **Coordinate space**: Normalized to image dimensions (0.0-1.0)
- **Auto-download**: Model downloaded on first use (~3.8 MB)

## Implementation

### Core Function

```python
def detect_landmarks(image: np.ndarray) -> LandmarkResult:
    """Detect face landmarks in a BGR image.

    Parameters:
        image: BGR uint8 image array (OpenCV format)

    Returns:
        LandmarkResult with:
        - landmarks: 478x2 array of pixel coordinates
        - image_shape: (height, width) tuple
        - confidence: mean presence confidence (0.0-1.0)

    Raises:
        ValueError: If no face detected
    """
```

### Processing Steps

1. **Image Preparation**
   - Convert BGR to RGB color space
   - Wrap in MediaPipe Image format
   - No resizing needed (handled internally)

2. **Face Detection**
   - Initialize FaceLandmarker with configuration
   - Run detection on prepared image
   - Handle multi-face scenarios (use first face)

3. **Coordinate Conversion**
   - Transform normalized coordinates (0.0-1.0) to pixel space
   - Extract x, y coordinates (z-depth discarded for 2D pipeline)
   - Maintain float64 precision for accuracy

4. **Confidence Assessment**
   - Average per-landmark presence scores
   - Warn if confidence < 0.5
   - Include in result for downstream quality checks

## Landmark Structure

The 478 landmarks form a comprehensive face mesh:

### Key Landmark Groups

| Feature | Landmark Count | Indices | Usage |
|---------|---------------|---------|-------|
| Face Contour | 36 | 10, 21, 54... | Face boundary, jaw line |
| Left Eye | 16 | 263, 249, 390... | Eye region mask |
| Right Eye | 16 | 33, 7, 163... | Eye region mask |
| Mouth | 20 | 61, 146, 91... | Mouth region mask |
| Nose | 9 | 1, 2, 5... | Future: nose importance |
| Left Eyebrow | 8 | 46, 53, 52... | Future: expression analysis |
| Right Eyebrow | 8 | 276, 283, 282... | Future: expression analysis |

### Semantic Regions

Landmarks are semantically grouped for feature extraction:
```python
# Example: Left eye polygon
left_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382,
                   362, 398, 384, 385, 386, 387, 388, 466]
```

## Configuration

### Detection Parameters

```python
FaceLandmarkerOptions(
    num_faces=1,                      # Single face mode
    min_face_detection_confidence=0.5, # Detection threshold
    min_face_presence_confidence=0.5,  # Presence threshold
)
```

### Environment Variables

- `FACE_LANDMARKER_MODEL_PATH`: Override default model location
- Default location: `<project_root>/models/face_landmarker.task`

## Output Format

### LandmarkResult Structure

```python
@dataclass
class LandmarkResult:
    landmarks: np.ndarray      # Shape: (478, 2), dtype: float64
    image_shape: tuple[int, int]  # (height, width)
    confidence: float          # Mean presence confidence (0.0-1.0)
```

### Coordinate System

- **Origin**: Top-left corner of image
- **X-axis**: Horizontal, increasing rightward (0 to width)
- **Y-axis**: Vertical, increasing downward (0 to height)
- **Units**: Pixels (floating point for sub-pixel accuracy)

## Quality Considerations

### Failure Modes

1. **No face detected**
   - Image too dark/bright
   - Face partially out of frame
   - Extreme face angles
   - Non-human subjects

2. **Low confidence**
   - Motion blur
   - Occlusions (hair, hands, objects)
   - Unusual facial expressions
   - Poor image quality

### Best Practices

- Input images should be well-lit with clear facial features
- Face should occupy 20-80% of image area
- Frontal or near-frontal poses work best
- Avoid extreme expressions or occlusions
- Higher resolution improves landmark precision

## Visualization

The detected landmarks can be visualized as:
- Points overlay on original image
- Connected mesh showing face structure
- Individual feature groups with different colors
- Confidence heatmap (per-landmark confidence)

Example visualization code:
```python
def draw_landmarks(image, landmarks):
    for point in landmarks.landmarks:
        cv2.circle(image, tuple(point.astype(int)), 2, (0, 255, 0), -1)
    return image
```

## Performance

- **Speed**: ~30-50ms on modern CPU (for 1920x1080 image)
- **Memory**: ~50MB for model + image buffers
- **Accuracy**: Sub-pixel precision with good input images
- **Robustness**: Handles various face sizes, minor occlusions

## Extensions

Future enhancements could include:
- Multi-face support with face selection logic
- Temporal smoothing for video sequences
- 3D landmark usage for pose-aware processing
- Custom landmark subsets for specific features
- Confidence-weighted processing downstream
- Alternative face detection models (dlib, OpenCV)

## Next Stage

The detected landmarks flow into [Stage 2: Region Masks](02-region-masks.md), where they are used to generate binary masks for semantic facial regions.