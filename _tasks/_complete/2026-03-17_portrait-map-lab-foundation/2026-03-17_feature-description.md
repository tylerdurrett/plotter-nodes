# Feature: Portrait Map Lab — Foundation & Eye/Mouth Feature Distance Pipeline

**Date:** 2026-03-17
**Status:** Scoped

---

## Overview

Stand up the Portrait Map Lab as a modern Python project and implement the first end-to-end pipeline: eye and mouth feature-distance/influence maps from portrait images. This establishes the repo's architecture, tooling, and module structure so that all future map types, ComfyUI wrapping, and potential API exposure build on a clean, tested foundation.

---

## End-User Capabilities

1. **Run a single command** against a portrait image and receive a full set of outputs: landmark overlay, per-feature masks, raw distance fields, remapped influence maps, and a combined feature-importance image.
2. **Call any pipeline step independently** — landmark detection, mask generation, distance transform, influence remapping — as a standalone function with typed inputs and outputs.
3. **Tune pipeline behavior** via a config object with sensible defaults (eye/mouth weights, falloff radii, distance clamp, normalization mode).
4. **Inspect all intermediate outputs** saved as PNG (visual) and `.npy` (raw precision) files in an organized output directory.
5. **Import the package** from scripts, notebooks, or future wrapper layers (ComfyUI nodes, API endpoints) with no UI or framework dependencies.

---

## Architecture / Scope

### Project Tooling

| Decision | Choice | Rationale |
|---|---|---|
| Package manager | **uv** | Fast, modern, handles venvs and lockfiles, works with pyproject.toml |
| Build config | **pyproject.toml** (no setup.py) | Modern Python standard; single source of truth for metadata, deps, and tool config |
| Testing | **pytest** | De facto standard; simple, extensible |
| Image array type | **numpy ndarray** | Native to OpenCV and scipy; canonical for numerical image processing |
| Structured data | **dataclasses** | Stdlib, zero-dependency, lightweight; avoids pydantic coupling while keeping typed interfaces. Pydantic can layer on top later if an API surface is added |
| Type hints | **Throughout** | Modern Python best practice; improves IDE support and self-documentation |

### Package Layout (src layout)

```
plotter-nodes/
  pyproject.toml
  README.md
  src/
    portrait_map_lab/
      __init__.py
      landmarks.py        # MediaPipe face landmark detection
      face_regions.py      # Semantic region definitions (landmark index groups)
      masks.py             # Polygon rasterization to binary masks
      distance_fields.py   # Euclidean distance transforms
      remap.py             # Distance-to-influence remapping curves
      combine.py           # Weighted combination of influence maps
      pipelines.py         # Convenience compositions of the above steps
      viz.py               # Overlays, colormaps, debug contact sheets
      storage.py           # Image/array save/load utilities
      models.py            # Dataclasses for config, landmarks, pipeline results
  scripts/
    run_pipeline.py        # CLI entry point for single-image pipeline
  tests/
    test_landmarks.py
    test_masks.py
    test_distance_fields.py
    test_remap.py
    test_pipelines.py
  notebooks/               # Optional exploratory notebooks (not required for milestone)
```

**Changes from init.md's proposed layout:**
- `types.py` renamed to `models.py` — avoids shadowing Python's stdlib `types` module.
- `io.py` renamed to `storage.py` — avoids shadowing Python's stdlib `io` module.
- `pipelines.py` added as a *composition layer* — each underlying module remains independently callable. Pipelines are convenience, not the only entry point.
- `combine.py` broken out from `remap.py` — combination logic (weighting, blending multiple maps) is a distinct concern from single-map remapping.

### Composability Principle

Every processing step is a standalone function:

```
detect_landmarks(image) → LandmarkResult
build_region_masks(landmarks, image_shape, regions) → dict[str, ndarray]
compute_distance_field(mask) → ndarray
remap_influence(distance_field, config) → ndarray
combine_maps(maps, weights) → ndarray
```

`pipelines.py` composes these into named workflows but is never required. This directly supports:
- **ComfyUI wrapping**: each function maps to one node.
- **API exposure**: each function maps to one endpoint.
- **Notebook use**: call exactly the steps you need.

### Future API Readiness

The architecture supports later API exposure without restructuring:
- Dataclass configs serialize naturally to/from JSON.
- Functions have typed, explicit signatures — easy to wrap with FastAPI.
- If an API layer is added, pydantic models can be introduced as a translation layer at the API boundary without changing core code.

---

## Technical Details

### Dependencies

| Package | Purpose |
|---|---|
| mediapipe | Face landmark detection |
| opencv-python | Image I/O, polygon drawing, general image ops |
| numpy | Array operations, distance transforms |
| scipy | `scipy.ndimage.distance_transform_edt` for Euclidean distance fields |
| matplotlib | Visualization, colormaps, contact sheets |

Dev dependencies: `pytest`, `ruff` (linter/formatter).

### Landmark Detection

- Use MediaPipe Face Mesh (468 landmarks) via the Python API directly.
- Normalize landmarks to pixel coordinates relative to image dimensions.
- Return a `LandmarkResult` dataclass containing the coordinate array and metadata.

### Semantic Face Regions

- Centralized landmark-index-to-region mapping in `face_regions.py`.
- Initial regions: `left_eye`, `right_eye`, `mouth`.
- Each region defined as an ordered list of landmark indices forming a polygon.
- Designed for easy extension (nose, eyebrows, jawline, etc.).

### Mask Generation

- Rasterize region polygons onto image-sized arrays using OpenCV `fillPoly`.
- Output: binary `uint8` masks (0 or 255).
- Support individual and combined masks (e.g., `combined_eyes` = left | right).

### Distance Fields

- Compute via `scipy.ndimage.distance_transform_edt` on inverted masks.
- Output: `float64` ndarray of pixel-unit Euclidean distances.
- Preserve raw output — no normalization at this stage.

### Influence Remapping

- Transform raw distance into 0.0–1.0 influence values.
- Supported curve types (at minimum):
  - **Linear clamped**: `max(0, 1 - d/radius)`
  - **Gaussian**: `exp(-(d^2) / (2 * sigma^2))`
  - **Exponential decay**: `exp(-d / tau)`
- Config via `RemapConfig` dataclass with fields for curve type, radius/sigma/tau, and clamp distance.

### Combination

- Weighted sum of multiple influence maps.
- Normalize result to 0.0–1.0.
- Config via weight dict, e.g., `{"eyes": 0.6, "mouth": 0.4}`.

### Output Strategy

| Output | Format | Purpose |
|---|---|---|
| Landmark overlay | PNG | Visual verification of detection |
| Per-feature masks | PNG | Visual verification of regions |
| Raw distance fields | `.npy` + heatmap PNG | Precision data + visual inspection |
| Remapped influence maps | PNG (grayscale or colormap) | Visual inspection |
| Combined feature map | PNG (colormap) | Final result visualization |
| Debug contact sheet | PNG | All-in-one overview |

Output directory structure per run:
```
output/
  <image_name>/
    landmarks.png
    mask_left_eye.png
    mask_right_eye.png
    mask_mouth.png
    distance_eyes_raw.npy
    distance_eyes_heatmap.png
    distance_mouth_raw.npy
    distance_mouth_heatmap.png
    influence_eyes.png
    influence_mouth.png
    combined_importance.png
    contact_sheet.png
```

---

## Risks and Considerations

- **MediaPipe model download**: MediaPipe downloads model files on first run. The project should document this and handle the case where the model isn't available gracefully.
- **Face detection failures**: Not all images will have detectable faces. The pipeline must handle zero-face and multi-face cases explicitly (start with first-face-only, log warnings).
- **Landmark stability**: MediaPipe's landmark indices for specific face regions are not formally documented as stable API. The centralized region definitions in `face_regions.py` isolate this risk — if indices change, there's one place to update.
- **Large output volume**: Saving both PNG and `.npy` per map per feature can add up. Acceptable for a research lab; worth noting for future batch processing.

---

## Non-Goals / Future Iterations

These are explicitly **out of scope** for this feature:

- SVG or line generation of any kind
- ComfyUI node wrappers (architecture supports it; implementation comes later)
- API endpoints (architecture supports it; implementation comes later)
- Depth maps, normal maps, or 3D geometry
- Geodesic or mesh-surface distance
- Batch processing (single-image pipeline only)
- Notebook creation (the package should be notebook-importable, but shipping notebooks is not a deliverable)
- GUI of any kind

---

## Success Criteria

1. `uv run python scripts/run_pipeline.py <image_path>` produces all outputs listed in the Output Strategy section.
2. Each processing module is independently importable and callable with typed inputs/outputs.
3. Config defaults produce reasonable results on a standard portrait photo without manual tuning.
4. All modules have at least basic pytest coverage for the happy path.
5. `ruff check` and `ruff format --check` pass clean.
6. The package installs cleanly via `uv pip install -e .` with no manual dependency steps beyond `uv sync`.

---

## Open Questions

1. **Contact sheet layout**: Should the debug contact sheet follow a fixed grid, or adapt to the number of outputs? (Can decide during implementation — not blocking.)
2. **Multi-face handling**: For the first milestone, first-detected-face-only is sufficient. Worth defining an explicit strategy before batch processing is added.
