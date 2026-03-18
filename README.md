# Portrait Map Lab

A Python-based map and analysis lab for generative plotter drawing.

## Overview

Portrait Map Lab is a **research and prototyping environment** for building the upstream visual intelligence needed by creative-code drawing algorithms in a separate TypeScript plotter repo. It generates portrait analysis maps — importance maps, density targets, structural guidance maps, and other derived representations — that will eventually drive line, shading, and mark-making systems for pen plotting.

This repo is **not** the final plotter runtime or drawing engine. It is the place to explore and develop the maps, signals, and analysis layers that more advanced algorithms will depend on. Over time, outputs may be wrapped as ComfyUI nodes and/or exported for use in the TypeScript plotter system.

The current focus is **Stage 1**: building foundational portrait analysis maps, starting with distance fields and influence maps derived from facial features (eyes, mouth, face contour).

## Vision & Roadmap

The larger goal is a family of creative-code drawing algorithms for pen plotting — continuous-line portraits, density-based shading, form-following strokes, and other experimental approaches. This repo supports that work by providing the upstream analysis and prototyping layer, especially where Python tools (MediaPipe, scientific Python, image processing) make exploration easier.

See [docs/vision.md](docs/vision.md) for the full vision, the 7-stage development roadmap, and how this repo relates to the broader plotter system.

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
