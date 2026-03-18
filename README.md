# Portrait Map Lab

Portrait feature distance mapping pipeline for pen plotter artwork.

## Overview

Portrait Map Lab generates **portrait analysis maps** — distance fields and influence maps derived from facial features — that support algorithmic continuous-line drawing systems for SVG and pen-plotter output.

The long-term vision is a system where accumulated line density creates tone while remaining structurally tied to the form of the subject. This repository is the **Stage 1 map-generation lab** for that larger system, focused on building reusable pipelines for deriving density and importance maps from portraits.

The first map family centers on **distance and influence relative to the eyes and mouth**.

## What it produces

For a given portrait image, the pipeline outputs:

- **Landmark overlay** — input with 478 MediaPipe Face Mesh landmarks drawn
- **Region masks** — binary masks for left eye, right eye, mouth, and combined eyes
- **Distance fields** — Euclidean distance transforms from eye/mouth regions (raw `.npy` + heatmap)
- **Influence maps** — remapped distance fields using configurable falloff curves (linear, gaussian, exponential)
- **Combined importance map** — weighted combination of eye and mouth influence
- **Contact sheet** — labeled grid of all visualizations

## Installation

Requires Python 3.10+.

```bash
# Recommended: using uv
uv sync

# Alternative: using pip
pip install -e .

# With dev dependencies (pytest, ruff)
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
python scripts/run_pipeline.py path/to/portrait.jpg
```

Options:

```
--output-dir DIR       Base output directory (default: output)
--eye-weight FLOAT     Weight for eye influence (default: 0.6)
--mouth-weight FLOAT   Weight for mouth influence (default: 0.4)
--curve TYPE           Remapping curve: linear, gaussian, exponential (default: gaussian)
--sigma FLOAT          Gaussian curve spread (default: 80.0)
--radius FLOAT         Linear curve radius (default: 150.0)
--tau FLOAT            Exponential decay rate (default: 60.0)
--clamp-distance FLOAT Max distance in pixels (default: 300.0)
```

### Python API

```python
from portrait_map_lab import (
    load_image,
    run_feature_distance_pipeline,
    save_pipeline_outputs,
    PipelineConfig,
    RemapConfig,
)
from pathlib import Path

image = load_image(Path("portrait.jpg"))

config = PipelineConfig(
    remap=RemapConfig(curve="gaussian", sigma=80.0),
    weights={"eyes": 0.6, "mouth": 0.4},
)

result = run_feature_distance_pipeline(image, config)
save_pipeline_outputs(result, image, Path("output/portrait"))

# Access intermediate results directly
result.landmarks       # LandmarkResult with 478 face mesh points
result.masks           # dict of binary masks (left_eye, right_eye, mouth, combined_eyes)
result.distance_fields # dict of raw distance arrays (eyes, mouth)
result.influence_maps  # dict of remapped influence arrays (eyes, mouth)
result.combined        # final weighted importance map
```

## Project structure

```
src/portrait_map_lab/
    __init__.py          # Public API (21 exports)
    models.py            # Core dataclasses (LandmarkResult, PipelineConfig, etc.)
    landmarks.py         # MediaPipe Face Mesh landmark detection
    face_regions.py      # Semantic facial region definitions
    masks.py             # Binary mask rasterization from landmarks
    distance_fields.py   # Euclidean distance field computation (scipy)
    remap.py             # Distance-to-influence remapping (3 curve types)
    combine.py           # Weighted map combination
    pipelines.py         # End-to-end pipeline composition
    viz.py               # Visualization (landmark overlay, colormaps, contact sheets)
    storage.py           # Image and array I/O
scripts/
    run_pipeline.py      # CLI entry point
tests/                   # 71 tests across 8 test modules
```

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

## Architecture

The package is designed around a few key principles:

- **Modular** — each module has a single responsibility, and functions are independently usable outside the pipeline
- **ComfyUI-ready** — core logic lives in the package; the CLI is a thin wrapper. Future ComfyUI custom nodes can wrap the same functions with no duplication
- **Extensible** — the architecture supports adding new map types (silhouette, depth, saliency, curvature) without structural changes
- **Research-friendly** — all intermediate outputs (raw masks, distance fields, influence maps) are preserved for inspection and experimentation
