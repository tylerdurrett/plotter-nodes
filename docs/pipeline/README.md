# Portrait Importance Map Pipeline

## Overview

The Portrait Importance Map Pipeline is a multi-stage computer vision system that generates importance maps from portrait images, with particular emphasis on facial features (eyes and mouth). These maps serve as the foundation for downstream generative drawing algorithms, providing density targets and structural guidance for artistic rendering.

The pipeline transforms a portrait photograph into a normalized importance map where values indicate the relative visual significance of each pixel region, with higher values near critical facial features.

## Pipeline Stages

The complete pipeline consists of five sequential processing stages:

1. **[Landmark Detection](01-landmark-detection.md)** - Detect face and extract 478 3D facial landmarks using MediaPipe Face Mesh
2. **[Region Masks](02-region-masks.md)** - Generate binary masks for semantic facial regions (eyes, mouth)
3. **[Distance Fields](03-distance-fields.md)** - Compute Euclidean distance transforms from region boundaries
4. **[Influence Maps](04-influence-maps.md)** - Remap distances to influence values using configurable falloff curves
5. **[Combination](05-combination.md)** - Weighted combination of individual influence maps into final importance map

## Quick Start

### Basic Usage

```python
from portrait_map_lab import load_image, run_feature_distance_pipeline
from pathlib import Path

# Load portrait image
image = load_image(Path("portrait.jpg"))

# Run pipeline with default configuration
result = run_feature_distance_pipeline(image)

# Access the final importance map (normalized 0.0-1.0)
importance_map = result.combined
```

### Custom Configuration

```python
from portrait_map_lab import PipelineConfig, RemapConfig

config = PipelineConfig(
    # Influence remapping parameters
    remap=RemapConfig(
        curve="gaussian",     # Options: linear, gaussian, exponential
        sigma=80.0,          # Gaussian spread
        clamp_distance=300.0 # Maximum distance in pixels
    ),
    # Feature weights for combination
    weights={
        "eyes": 0.6,   # 60% influence from eyes
        "mouth": 0.4   # 40% influence from mouth
    }
)

result = run_feature_distance_pipeline(image, config)
```

## Data Flow

```
Input Image (BGR)
    ↓
[Landmark Detection]
    → LandmarkResult (478 points + confidence)
    ↓
[Region Masks]
    → Binary masks (left_eye, right_eye, mouth, combined_eyes)
    ↓
[Distance Fields]
    → Euclidean distance arrays (eyes, mouth)
    ↓
[Influence Maps]
    → Normalized influence arrays (0.0-1.0)
    ↓
[Weighted Combination]
    → Final importance map (0.0-1.0)
```

## Output Structure

The pipeline returns a `PipelineResult` containing all intermediate and final outputs:

```python
result.landmarks       # Face mesh landmarks (478 points)
result.masks          # Binary masks for each region
result.distance_fields # Raw distance transforms
result.influence_maps  # Remapped influence values
result.combined       # Final importance map
```

## Key Concepts

### Importance Map
A normalized grayscale image where pixel values (0.0-1.0) represent the relative visual importance or attention weight of that region. Higher values indicate areas that should receive more detail, density, or emphasis in downstream rendering.

### Distance Field
A continuous field where each pixel contains its Euclidean distance to the nearest feature boundary. Used as the basis for smooth influence falloff.

### Influence Map
A normalized field derived from a distance field through remapping functions. Represents how strongly a feature (eye, mouth) influences each pixel in the image.

### Falloff Curves
Mathematical functions that control how influence decreases with distance:
- **Linear**: Steady linear decrease
- **Gaussian**: Bell curve falloff (smooth, natural)
- **Exponential**: Rapid initial falloff

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `curve` | str | "gaussian" | Remapping curve type: linear, gaussian, exponential |
| `sigma` | float | 80.0 | Gaussian curve spread parameter |
| `radius` | float | 150.0 | Linear curve maximum radius |
| `tau` | float | 60.0 | Exponential decay rate |
| `clamp_distance` | float | 300.0 | Maximum distance before clamping |
| `weights.eyes` | float | 0.6 | Weight for eye influence (0.0-1.0) |
| `weights.mouth` | float | 0.4 | Weight for mouth influence (0.0-1.0) |

## Visualization

The pipeline provides comprehensive visualization outputs:

- **Landmark overlay** - Input image with detected face mesh
- **Binary masks** - Individual region masks
- **Distance heatmaps** - Colorized distance fields
- **Influence maps** - Individual feature influences
- **Combined importance** - Final weighted result
- **Contact sheet** - Grid of all visualizations

## Architecture

See [architecture.md](architecture.md) for detailed technical architecture, including:
- Module responsibilities
- Data structures
- Extension points
- ComfyUI integration readiness
- Performance considerations

## Extensions

The pipeline architecture supports future extensions:

- Additional facial features (nose, eyebrows, face contour)
- Alternative importance metrics (saliency, edge density, curvature)
- Multi-face support
- Temporal consistency for video
- Custom region definitions
- Neural importance prediction

## Related Documentation

- [Technical Architecture](architecture.md) - System design and implementation details
- [Vision Document](../vision.md) - Broader project goals and roadmap
- [API Reference](https://github.com/username/portrait-map-lab/docs/api) - Complete API documentation