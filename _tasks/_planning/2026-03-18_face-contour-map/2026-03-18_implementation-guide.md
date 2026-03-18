# Implementation Guide: Face Contour Distance Map

**Date:** 2026-03-18
**Feature:** Face Contour Distance Map
**Source:** [2026-03-18_feature-description.md](2026-03-18_feature-description.md)

## Overview

This implementation adds a face contour distance pipeline alongside the existing eye/mouth feature pipeline. The core new logic lives in a single new module (`face_contour.py`) that computes signed distance fields from the MediaPipe face oval contour. Everything else — models, pipeline orchestration, viz, CLI, docs — layers on top of that core module and reuses existing infrastructure.

Phases are sequenced bottom-up: data models first, then core contour logic, then pipeline/viz, then CLI, then documentation. Each phase is independently testable upon completion. The CLI refactoring (Phase 4) is sequenced after the pipeline works so we can verify the new pipeline in isolation before wiring up CLI integration.

## File Structure

```
src/portrait_map_lab/
  models.py            # MODIFIED: add ContourConfig, ContourResult
  face_contour.py      # NEW: face oval indices, contour masks, signed distance
  viz.py               # MODIFIED: add draw_contour
  pipelines.py         # MODIFIED: add run_contour_pipeline, save_contour_outputs
  __init__.py          # MODIFIED: add new exports

scripts/
  run_pipeline.py      # MODIFIED: refactor to subcommand architecture

tests/
  test_models.py       # MODIFIED: add ContourConfig/ContourResult tests
  test_face_contour.py # NEW: unit tests for face_contour module
  test_contour_pipeline.py # NEW: integration tests for contour pipeline

docs/
  pipeline/
    README.md          # MODIFIED: add contour pipeline section
    06-face-contour.md # NEW: face contour pipeline stage documentation
    architecture.md    # MODIFIED: add contour module to tables
```

---

## Phase 1: Data Models

**Purpose:** Define the configuration and result dataclasses needed by all subsequent phases.

**Rationale:** Models have no dependencies on new code and are imported by every other phase, so they must come first.

### 1.1 Add ContourConfig and ContourResult to models.py

- [ ] Add `ContourConfig` dataclass (mutable, `slots=True`) with fields: `remap: RemapConfig`, `direction: str = "inward"`, `band_width: float | None = None`, `contour_thickness: int = 1`, `output_dir: str = "output"`
- [ ] Add `ContourResult` dataclass (frozen, `slots=True`) with fields: `landmarks: LandmarkResult`, `contour_polygon: np.ndarray`, `contour_mask: np.ndarray`, `filled_mask: np.ndarray`, `signed_distance: np.ndarray`, `directional_distance: np.ndarray`, `influence_map: np.ndarray`
- [ ] Add both classes to `__all__` in `models.py`
- [ ] Write tests in `test_models.py`: `TestContourConfig` (defaults, mutable, independent defaults) and `TestContourResult` (creation, frozen)

**Acceptance Criteria:**
- `ContourConfig()` creates instance with expected defaults (`direction="inward"`, `band_width=None`, `contour_thickness=1`)
- `ContourConfig` is mutable; `ContourResult` is frozen
- Separate `ContourConfig` instances do not share mutable state
- `pytest tests/test_models.py` passes

---

## Phase 2: Core Contour Module

**Purpose:** Implement all contour-specific logic as standalone composable functions.

**Rationale:** This is the core new functionality. Building it before the pipeline lets us unit test each function in isolation with synthetic data, with no MediaPipe dependency in tests.

### 2.1 Face oval indices and polygon extraction

- [ ] Create `src/portrait_map_lab/face_contour.py`
- [ ] Define `FACE_OVAL_INDICES: list[int]` — 36 ordered landmark indices from MediaPipe FACEMESH_FACE_OVAL, starting at forehead (10), walking clockwise
- [ ] Implement `get_face_oval_polygon(landmarks: LandmarkResult) -> np.ndarray` — extracts Nx2 pixel coordinates for the face oval from the 478-point landmark array
- [ ] Write tests in `tests/test_face_contour.py`: `TestFaceOvalIndices` (count is 36, all in 0-477 range, no duplicates) and `TestGetFaceOvalPolygon` (correct shape, dtype, coordinates within image bounds)

**Acceptance Criteria:**
- `FACE_OVAL_INDICES` has exactly 36 unique indices, all in [0, 477]
- `get_face_oval_polygon` returns `(36, 2)` float64 array with coordinates within image bounds
- `pytest tests/test_face_contour.py::TestFaceOvalIndices tests/test_face_contour.py::TestGetFaceOvalPolygon` passes

### 2.2 Contour and filled mask rasterization

- [ ] Implement `rasterize_contour_mask(polygon: np.ndarray, image_shape: tuple[int, int], thickness: int = 1) -> np.ndarray` — draws closed polyline via `cv2.polylines`, returns uint8 mask (0/255)
- [ ] Implement `rasterize_filled_mask(polygon: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray` — draws filled polygon via `cv2.fillPoly`, returns uint8 mask (0/255)
- [ ] Write tests: `TestRasterizeContourMask` (dtype/shape, binary values, nonzero pixels, thin vs thick) and `TestRasterizeFilledMask` (dtype/shape, filled area larger than contour)

**Acceptance Criteria:**
- Both functions return uint8 arrays of correct shape with only 0/255 values
- Contour mask with `thickness=1` has significantly fewer nonzero pixels than filled mask for the same polygon
- `thickness=3` produces more nonzero pixels than `thickness=1`
- `pytest tests/test_face_contour.py::TestRasterizeContourMask tests/test_face_contour.py::TestRasterizeFilledMask` passes

### 2.3 Signed distance computation

- [ ] Implement `compute_signed_distance(contour_mask: np.ndarray, filled_mask: np.ndarray) -> np.ndarray` — computes EDT from contour mask via existing `compute_distance_field`, then applies sign: negative inside (where `filled_mask > 0`), positive outside, zero on contour. Ensure pixels where `contour_mask > 0` always get distance 0.0 regardless of filled mask classification.
- [ ] Write tests: `TestComputeSignedDistance` (shape/dtype, zero on contour, negative inside, positive outside, magnitude increases from contour)

**Acceptance Criteria:**
- Output is float64, same shape as inputs
- Pixels on the contour mask have distance ~0.0
- Interior pixels (inside filled mask, not on contour) have negative values
- Exterior pixels (outside filled mask, not on contour) have positive values
- `abs(signed_distance)` increases moving away from contour in either direction
- `pytest tests/test_face_contour.py::TestComputeSignedDistance` passes

### 2.4 Directional distance preparation

- [ ] Implement `prepare_directional_distance(signed_distance: np.ndarray, mode: str = "inward", clamp_value: float = 9999.0, band_width: float | None = None) -> np.ndarray` — converts signed distance to unsigned distance based on direction mode: `"inward"` keeps interior distances and clamps exterior; `"outward"` keeps exterior and clamps interior; `"both"` uses `abs()` everywhere; `"band"` uses `abs()` clamped at `band_width`
- [ ] Write tests: `TestPrepareDirectionalDistance` (inward clamps exterior, outward clamps interior, both is all positive, band clamps beyond width, raises on unknown mode)

**Acceptance Criteria:**
- `"inward"` mode: exterior pixels get `clamp_value`, interior pixels get `abs(signed_distance)`
- `"outward"` mode: interior pixels get `clamp_value`, exterior pixels get `abs(signed_distance)`
- `"both"` mode: all values are `abs(signed_distance)`
- `"band"` mode: values beyond `band_width` are clamped to `clamp_value`
- Unknown mode raises `ValueError`
- `pytest tests/test_face_contour.py::TestPrepareDirectionalDistance` passes

### 2.5 Module exports

- [ ] Add `__all__` to `face_contour.py` listing all public names: `FACE_OVAL_INDICES`, `get_face_oval_polygon`, `rasterize_contour_mask`, `rasterize_filled_mask`, `compute_signed_distance`, `prepare_directional_distance`

**Acceptance Criteria:**
- All public functions are importable via `from portrait_map_lab.face_contour import ...`
- `ruff check src/portrait_map_lab/face_contour.py` passes

---

## Phase 3: Pipeline and Visualization

**Purpose:** Wire the core contour functions into a pipeline with visualization and file output.

**Rationale:** Depends on Phase 1 (models) and Phase 2 (face_contour). Building viz first lets the save function use it immediately.

### 3.1 Add draw_contour to viz.py

- [ ] Implement `draw_contour(image: np.ndarray, polygon: np.ndarray, color: tuple[int, int, int] = (0, 255, 255), thickness: int = 2) -> np.ndarray` — draws the face oval as a closed polyline on a copy of the image, returns annotated image without mutation

**Acceptance Criteria:**
- Returns a new image (input not mutated)
- Output has same shape and dtype as input
- Drawn pixels differ from original at contour locations

### 3.2 Implement run_contour_pipeline

- [ ] Add `run_contour_pipeline(image: np.ndarray, config: ContourConfig | None = None) -> ContourResult` to `pipelines.py`. Steps: detect_landmarks → get_face_oval_polygon → rasterize_contour_mask → rasterize_filled_mask → compute_signed_distance → prepare_directional_distance → remap_influence. Return ContourResult with all intermediates.
- [ ] Add logging at each step, consistent with existing pipeline

**Acceptance Criteria:**
- `run_contour_pipeline(test_image)` returns a `ContourResult` with all fields populated
- All array fields have shape `(h, w)` matching input image
- `influence_map` values are in `[0.0, 1.0]`
- `signed_distance` has both negative and positive values
- Works with default config (no args beyond image)

### 3.3 Implement save_contour_outputs

- [ ] Add `save_contour_outputs(result: ContourResult, image: np.ndarray, output_dir: Path) -> None` to `pipelines.py`. Saves: `contour_overlay.png`, `contour_mask.png`, `filled_mask.png`, `signed_distance_raw.npy` + heatmap (diverging colormap, symmetric range centered at zero), `directional_distance_raw.npy` + heatmap, `contour_influence.png`, `contact_sheet.png`
- [ ] Use diverging colormap (`"RdBu"`) for signed distance heatmap with symmetric normalization: map `[-max_abs, +max_abs]` to `[0, 1]` so zero maps to 0.5

**Acceptance Criteria:**
- All expected files are created in the output directory
- `.npy` files are loadable and have correct shapes
- Signed distance heatmap uses diverging colormap (visually distinct from other heatmaps)
- Contact sheet includes all visualizations

### 3.4 Write integration tests

- [ ] Create `tests/test_contour_pipeline.py` with tests using the real test image fixture: complete result populated, shapes match, influence normalized, default config works, custom config works, all direction modes produce valid output, save creates expected files, signed distance has both signs

**Acceptance Criteria:**
- `pytest tests/test_contour_pipeline.py` passes
- Tests cover default config, custom config, all 4 direction modes, and file output

### 3.5 Update __init__.py exports

- [ ] Add imports and `__all__` entries for: `ContourConfig`, `ContourResult`, `run_contour_pipeline`, `save_contour_outputs`, `FACE_OVAL_INDICES`, `get_face_oval_polygon`, `rasterize_contour_mask`, `rasterize_filled_mask`, `compute_signed_distance`, `prepare_directional_distance`, `draw_contour`

**Acceptance Criteria:**
- All new public API names are importable from `portrait_map_lab`
- `ruff check src/portrait_map_lab/__init__.py` passes

---

## Phase 4: CLI Refactoring

**Purpose:** Restructure the CLI from flat argparse to subcommands to support multiple map generators.

**Rationale:** Sequenced after the pipeline works so we can verify correctness in isolation first. The CLI is a thin wrapper; getting the pipeline right matters more.

### 4.1 Refactor run_pipeline.py to subcommand architecture

- [ ] Restructure `scripts/run_pipeline.py` to use argparse subcommands: `features` (existing eye/mouth pipeline), `contour` (new face contour pipeline), `all` (runs both)
- [ ] Extract shared remap args (`--curve`, `--sigma`, `--tau`, `--radius`, `--clamp-distance`) and shared args (`image_path`, `--output-dir`) onto the parent parser
- [ ] Add `features` subcommand with `--eye-weight`, `--mouth-weight`
- [ ] Add `contour` subcommand with `--direction` (choices: inward, outward, both, band), `--band-width`, `--contour-thickness`
- [ ] Add `all` subcommand that runs both pipelines with their respective defaults
- [ ] Each subcommand shares the landmark detection step where possible (for `all`, detect once and pass landmarks to both pipelines — or accept the duplicate detection as acceptable for now)

**Acceptance Criteria:**
- `uv run python scripts/run_pipeline.py features test_images/20230427-171404.JPG` produces the same output as the old `run_pipeline.py` invocation
- `uv run python scripts/run_pipeline.py contour test_images/20230427-171404.JPG` produces contour outputs
- `uv run python scripts/run_pipeline.py all test_images/20230427-171404.JPG` produces both sets of outputs
- `uv run python scripts/run_pipeline.py --help` shows subcommands
- `uv run python scripts/run_pipeline.py contour --help` shows contour-specific args

---

## Phase 5: Documentation

**Purpose:** Document the new contour pipeline and update existing docs to reflect the expanded system.

**Rationale:** Sequenced after all code is working and tested so documentation reflects the actual implementation.

### 5.1 Create face contour pipeline documentation

- [ ] Create `docs/pipeline/06-face-contour.md` following the existing stage document conventions (Overview, Purpose, Implementation, Process Flow with ASCII diagram, Output Format, Configuration, Quality Considerations, Visualization sections)
- [ ] Cover: face oval landmarks, contour vs filled masks, signed distance field, direction modes, influence remapping
- [ ] Include code examples for standalone function usage and pipeline usage
- [ ] Include ASCII data flow diagram specific to the contour pipeline

**Acceptance Criteria:**
- Document follows the same section structure as existing stage docs (01-05)
- Includes process flow diagram, code examples, configuration table, and output format description
- Direction modes are clearly documented with use cases

### 5.2 Update pipeline README and architecture

- [ ] Update `docs/pipeline/README.md`: add face contour to the pipeline stages list, add a contour pipeline quick start example, update the data flow diagram to show the contour branch, add contour config parameters to the configuration table
- [ ] Update `docs/pipeline/architecture.md`: add `face_contour.py` to the module responsibility table, add `ContourConfig`/`ContourResult` to data structures section, update extension points to reference the contour pipeline as a concrete example

**Acceptance Criteria:**
- Pipeline README lists the contour pipeline alongside the feature pipeline
- Architecture doc includes the new module and data structures
- Quick start example for contour pipeline is present and accurate

---

## Phase 6: Final Verification

**Purpose:** End-to-end verification that everything works together.

**Rationale:** Final pass to catch integration issues after all code and docs are in place.

### 6.1 Full test suite and linting

- [ ] Run `uv run pytest` — all tests pass (existing + new)
- [ ] Run `uv run ruff check` — clean
- [ ] Run `uv run ruff format --check` — clean

**Acceptance Criteria:**
- All tests pass with no failures or warnings
- Linting and formatting checks pass clean

### 6.2 Visual verification

- [ ] Run `uv run python scripts/run_pipeline.py contour test_images/20230427-171404.JPG` and inspect outputs
- [ ] Verify contour overlay shows the face oval correctly positioned
- [ ] Verify signed distance heatmap shows blue (negative) inside and red (positive) outside with clear zero-crossing at the contour
- [ ] Verify default `inward` influence map is strongest at the face boundary, falling off toward center
- [ ] Run `uv run python scripts/run_pipeline.py features test_images/20230427-171404.JPG` and verify existing pipeline is unaffected

**Acceptance Criteria:**
- Contour overlay traces the jaw and forehead boundary
- Signed distance heatmap is visually correct (diverging at contour)
- Influence map emphasizes face boundary
- Existing `features` pipeline output is unchanged

---

## Dependency Graph

```
Phase 1 (Models)
  1.1 (ContourConfig, ContourResult + tests)
    │
    v
Phase 2 (Core Contour Module)
  2.1 (indices + polygon) → 2.2 (masks) → 2.3 (signed distance) → 2.4 (directional) → 2.5 (exports)
    │
    v
Phase 3 (Pipeline + Viz)
  3.1 (draw_contour) ──┐
  3.2 (run_contour_pipeline) ←─┘
  3.2 → 3.3 (save_contour_outputs)
  3.3 → 3.4 (integration tests)
  3.4 → 3.5 (__init__.py)
    │
    v
Phase 4 (CLI)
  4.1 (subcommand refactoring)
    │
    v
Phase 5 (Documentation)
  5.1 (06-face-contour.md) ──┐
  5.2 (README + architecture) ←─┘
    │
    v
Phase 6 (Verification)
  6.1 (tests + linting) → 6.2 (visual verification)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Signed distance field as core intermediate | Preserves full interior/exterior information regardless of remap direction, enabling future experimentation with particle spawning and flow fields |
| Separate `face_contour.py` module (not extending `face_regions.py`) | Face regions uses filled polygons; face contour uses polylines. Different geometric semantics warrant separate modules |
| Direction mode in `prepare_directional_distance` (not in `remap_influence`) | Keeps the existing remap module unchanged; direction is a contour-specific concern handled upstream |
| Subcommand CLI architecture | Each map type gets its own parameter namespace. Scales cleanly — new map types are new subcommands, not more flags on a flat parser |
| Hardcoded face oval indices | MediaPipe Python does not reliably expose `face_mesh_connections` at runtime. Hardcoding from the canonical source is the pragmatic choice |
| Contour mask pixels always get distance=0 | Prevents polyline/fillPoly rasterization boundary mismatch from producing incorrect signed distances at the contour |
