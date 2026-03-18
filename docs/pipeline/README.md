# Portrait Importance Map Pipeline

## Overview

The Portrait Importance Map Pipeline is a multi-stage computer vision system that generates importance maps from portrait images, with particular emphasis on facial features (eyes and mouth). These maps serve as the foundation for downstream generative drawing algorithms, providing density targets and structural guidance for artistic rendering.

The pipeline transforms a portrait photograph into a normalized importance map where values indicate the relative visual significance of each pixel region, with higher values near critical facial features.

## Pipeline Stages

The system includes two complementary pipelines that share core infrastructure:

### Feature Distance Pipeline
Generates importance maps based on facial features (eyes and mouth):

1. **[Landmark Detection](01-landmark-detection.md)** - Detect face and extract 478 3D facial landmarks using MediaPipe Face Mesh
2. **[Region Masks](02-region-masks.md)** - Generate binary masks for semantic facial regions (eyes, mouth)
3. **[Distance Fields](03-distance-fields.md)** - Compute Euclidean distance transforms from region boundaries
4. **[Influence Maps](04-influence-maps.md)** - Remap distances to influence values using configurable falloff curves
5. **[Combination](05-combination.md)** - Weighted combination of individual influence maps into final importance map

### Face Contour Pipeline
Generates distance maps based on the face boundary:

6. **[Face Contour Distance Map](06-face-contour.md)** - Compute signed distance fields from face oval contour with directional control

## Quick Start

### Feature Pipeline

```python
from portrait_map_lab import load_image, run_feature_distance_pipeline
from pathlib import Path

# Load portrait image
image = load_image(Path("portrait.jpg"))

# Run feature pipeline with default configuration
result = run_feature_distance_pipeline(image)

# Access the final importance map (normalized 0.0-1.0)
importance_map = result.combined
```

### Contour Pipeline

```python
from portrait_map_lab import load_image, run_contour_pipeline, ContourConfig

# Load portrait image
image = load_image(Path("portrait.jpg"))

# Run contour pipeline with custom configuration
config = ContourConfig(
    direction="inward",  # Focus on face interior
    band_width=None      # No band limiting
)
result = run_contour_pipeline(image, config)

# Access the influence map (normalized 0.0-1.0)
contour_map = result.influence_map
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

### Parallel Pipeline Architecture

```
                    Input Image (BGR)
                           ↓
                  [Landmark Detection]
                    478 face points
                     ↙          ↘
        Feature Pipeline      Contour Pipeline
               ↓                     ↓
        [Region Masks]        [Face Boundary]
         eyes, mouth           convex hull
               ↓                     ↓
      [Distance Fields]      [Signed Distance]
        from features         from contour
               ↓                     ↓
       [Influence Maps]    [Directional Distance]
         remapped           inward/outward/both
               ↓                     ↓
        [Combination]        [Influence Map]
        weighted sum          normalized
               ↓                     ↓
        Feature Map           Contour Map
         (0.0-1.0)             (0.0-1.0)
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

### Shared Parameters (RemapConfig)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `curve` | str | "gaussian" | Remapping curve type: linear, gaussian, exponential |
| `sigma` | float | 80.0 | Gaussian curve spread parameter |
| `radius` | float | 150.0 | Linear curve maximum radius |
| `tau` | float | 60.0 | Exponential decay rate |
| `clamp_distance` | float | 300.0 | Maximum distance before clamping |

### Feature Pipeline (PipelineConfig)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights.eyes` | float | 0.6 | Weight for eye influence (0.0-1.0) |
| `weights.mouth` | float | 0.4 | Weight for mouth influence (0.0-1.0) |

### Contour Pipeline (ContourConfig)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `direction` | str | "inward" | Direction mode: inward, outward, both, band |
| `band_width` | float \| None | None | Band width for band mode (pixels) |
| `contour_thickness` | int | 1 | Contour line thickness (pixels) |

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

## Command Line Interface

The CLI uses subcommands to run different pipelines:

```bash
# Run feature pipeline only
uv run python scripts/run_pipeline.py features portrait.jpg

# Run contour pipeline only
uv run python scripts/run_pipeline.py contour portrait.jpg --direction inward

# Run both pipelines
uv run python scripts/run_pipeline.py all portrait.jpg

# Custom configuration
uv run python scripts/run_pipeline.py contour portrait.jpg \
    --direction band \
    --band-width 50 \
    --curve gaussian \
    --sigma 100
```

Outputs are organized by pipeline:
```
output/
  <image_name>/
    features/     # Feature pipeline outputs
    contour/      # Contour pipeline outputs
```

## Extensions

The pipeline architecture supports future extensions:

- Additional facial features (nose, eyebrows)
- Alternative importance metrics (saliency, edge density, curvature)
- Multi-face support
- Temporal consistency for video
- Custom region definitions
- Neural importance prediction
- Hybrid pipeline combinations

## Related Documentation

- [Technical Architecture](architecture.md) - System design and implementation details
- [Vision Document](../vision.md) - Broader project goals and roadmap
- [API Reference](https://github.com/username/portrait-map-lab/docs/api) - Complete API documentation