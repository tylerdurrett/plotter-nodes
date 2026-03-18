# Implementation Guide: Portrait Map Lab — Foundation & Eye/Mouth Feature Distance Pipeline

**Date:** 2026-03-17
**Feature:** Portrait Map Lab Foundation
**Source:** [2026-03-17_feature-description.md](2026-03-17_feature-description.md)

---

## Overview

This guide builds the Portrait Map Lab from an empty repo to a working single-image pipeline in six phases. The sequencing prioritizes **incremental testability** — every phase produces something you can run and verify before moving on.

Phase 1 establishes the Python project skeleton with all tooling configured. Phases 2–4 build the processing modules bottom-up (landmarks → masks → distance → remap → combine), each with tests. Phase 5 adds output and visualization. Phase 6 wires everything into the pipeline composition and CLI script.

**Modern Python conventions used throughout:**
- Python 3.10+ type syntax: `X | None` (not `Optional[X]`), `list[str]` (not `List[str]`)
- `from __future__ import annotations` at the top of every module for consistent forward-reference behavior
- `@dataclass` with slots and frozen where appropriate
- `py.typed` marker for PEP 561 (typed package support)
- ruff handles both linting and formatting (replaces black + isort + flake8)

---

## File Structure

```
plotter-nodes/
  pyproject.toml
  README.md
  src/
    portrait_map_lab/
      __init__.py
      py.typed               # PEP 561 typed package marker
      models.py
      landmarks.py
      face_regions.py
      masks.py
      distance_fields.py
      remap.py
      combine.py
      pipelines.py
      viz.py
      storage.py
  scripts/
    run_pipeline.py
  tests/
    __init__.py
    conftest.py              # Shared fixtures (sample images, landmark data)
    test_models.py
    test_landmarks.py
    test_face_regions.py
    test_masks.py
    test_distance_fields.py
    test_remap.py
    test_combine.py
    test_pipelines.py
  test_images/               # Small test portrait(s) committed to repo
```

---

## Phase 1: Project Bootstrap

**Purpose:** Establish a fully configured, installable Python package with all tooling working before writing any processing code.

**Rationale:** Getting the project skeleton right first means every subsequent phase starts from a known-good state — imports work, tests run, linting passes. Fixing these things retroactively is annoying.

### 1.1 pyproject.toml & Package Skeleton

- [x] Create `pyproject.toml` with:
  - Project metadata (name: `portrait-map-lab`, version: `0.1.0`, requires-python: `>=3.10`)
  - Dependencies: `mediapipe`, `opencv-python`, `numpy`, `scipy`, `matplotlib`
  - Dev dependencies group: `pytest`, `ruff`
  - `[tool.ruff]` config: target Python 3.10, line-length 99, enable isort rules
  - `[tool.pytest.ini_options]` config: `testpaths = ["tests"]`
  - `[build-system]` using `hatchling` (modern, minimal build backend)
- [x] Create `src/portrait_map_lab/__init__.py` with package version and top-level docstring
- [x] Create `src/portrait_map_lab/py.typed` (empty marker file for PEP 561)
- [x] Create `tests/__init__.py` (empty)
- [x] Create `tests/conftest.py` with a placeholder fixture comment

> **Note:** Build backend is `hatchling.build` (not `hatchling.backends` as sometimes seen in examples). All 5 acceptance criteria verified passing.

**Acceptance Criteria:**
- `uv sync` installs all dependencies into a venv without errors
- `uv run python -c "import portrait_map_lab"` succeeds
- `uv run pytest` runs (0 tests collected, no errors)
- `uv run ruff check src/ tests/` passes clean
- `uv run ruff format --check src/ tests/` passes clean

### 1.2 Models (Core Dataclasses)

- [x] Create `models.py` with `from __future__ import annotations`
- [x] Define `LandmarkResult` dataclass:
  - `landmarks`: `np.ndarray` (Nx2 or Nx3 array of pixel coordinates)
  - `image_shape`: `tuple[int, int]` (height, width)
  - `confidence`: `float`
- [x] Define `RegionDefinition` dataclass:
  - `name`: `str`
  - `landmark_indices`: `list[int]`
- [x] Define `RemapConfig` dataclass with defaults:
  - `curve`: `str` (default `"gaussian"`)
  - `radius`: `float` (default `150.0`)
  - `sigma`: `float` (default `80.0`)
  - `tau`: `float` (default `60.0`)
  - `clamp_distance`: `float` (default `300.0`)
- [x] Define `PipelineConfig` dataclass with defaults:
  - `regions`: `list[RegionDefinition]` (defaults to eye + mouth regions)
  - `remap`: `RemapConfig` (default instance)
  - `weights`: `dict[str, float]` (default `{"eyes": 0.6, "mouth": 0.4}`)
  - `output_dir`: `str` (default `"output"`)
- [x] Define `PipelineResult` dataclass:
  - `landmarks`: `LandmarkResult`
  - `masks`: `dict[str, np.ndarray]`
  - `distance_fields`: `dict[str, np.ndarray]`
  - `influence_maps`: `dict[str, np.ndarray]`
  - `combined`: `np.ndarray`
- [x] Write `tests/test_models.py` — verify dataclasses instantiate with defaults, field types are correct

> **Note:** All dataclasses use `slots=True`. Immutable types (`LandmarkResult`, `RegionDefinition`, `PipelineResult`) also use `frozen=True`. Config types (`RemapConfig`, `PipelineConfig`) are mutable for user tweaking. Default regions include real MediaPipe Face Mesh landmark indices via `_default_regions()` helper. 12 tests pass covering creation, defaults, frozen/mutable behavior, and independent default instances.

**Acceptance Criteria:**
- All dataclasses instantiate with default values where defined
- `PipelineConfig()` produces a usable default config with no arguments
- Tests pass: `uv run pytest tests/test_models.py`

---

## Phase 2: Landmark Detection

**Purpose:** Get MediaPipe working and producing structured landmark data from portrait images.

**Rationale:** Landmarks are the input to everything else. Verifying this works first (including with a real test image) catches environment/dependency issues early.

### 2.1 Test Image Setup

- [x] Add some portrait test images to `test_images/`
- [x] Add `test_images/` to version control (small enough to commit)
- [x] Add a `conftest.py` fixture that loads the test image path and array

**Acceptance Criteria:**
- `conftest.py` fixture returns a valid numpy array when loaded
- Test image is under 500KB

### 2.2 Landmark Detection Module

- [ ] Create `landmarks.py` with `from __future__ import annotations`
- [ ] Implement `detect_landmarks(image: np.ndarray) -> LandmarkResult`:
  - Initialize MediaPipe Face Mesh with `static_image_mode=True`
  - Convert normalized landmarks to pixel coordinates
  - Return first face found; raise `ValueError` if no face detected
  - Use `logging` module for warnings (multi-face, low confidence)
- [ ] Write `tests/test_landmarks.py`:
  - Test: returns `LandmarkResult` with correct shape for test image
  - Test: landmark coordinates are within image bounds
  - Test: raises `ValueError` on a blank/non-face image

**Acceptance Criteria:**
- `detect_landmarks(test_image)` returns a `LandmarkResult` with 478 landmarks (MediaPipe Face Mesh v2)
- All landmark coordinates are non-negative and within image dimensions
- Blank image input raises `ValueError`
- Tests pass: `uv run pytest tests/test_landmarks.py`

---

## Phase 3: Face Regions & Masks

**Purpose:** Turn raw landmarks into semantic regions and rasterized binary masks.

**Rationale:** These two modules are tightly coupled (regions define what masks get built) and together form the next testable unit — you can visually verify masks make sense.

### 3.1 Face Region Definitions

- [ ] Create `face_regions.py` with `from __future__ import annotations`
- [ ] Define `DEFAULT_REGIONS: list[RegionDefinition]` containing:
  - `left_eye` — landmark indices for the left eye contour
  - `right_eye` — landmark indices for the right eye contour
  - `mouth` — landmark indices for the outer mouth contour
- [ ] Implement `get_region_polygons(landmarks: LandmarkResult, regions: list[RegionDefinition]) -> dict[str, np.ndarray]`:
  - Extract pixel coordinates for each region's landmark indices
  - Return dict mapping region name to Nx2 polygon array
- [ ] Write `tests/test_face_regions.py`:
  - Test: default regions contain expected names
  - Test: polygon extraction returns correct shapes with valid landmark input
  - Test: polygon coordinates are within image bounds

**Acceptance Criteria:**
- `DEFAULT_REGIONS` defines three regions with correct names
- `get_region_polygons()` returns dict with polygon arrays of shape (N, 2)
- Tests pass: `uv run pytest tests/test_face_regions.py`

### 3.2 Mask Generation

- [ ] Create `masks.py` with `from __future__ import annotations`
- [ ] Implement `rasterize_mask(polygon: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray`:
  - Use `cv2.fillPoly` on a zeros array
  - Return `uint8` array (0 or 255)
- [ ] Implement `build_region_masks(landmarks: LandmarkResult, regions: list[RegionDefinition]) -> dict[str, np.ndarray]`:
  - Call `get_region_polygons` then `rasterize_mask` for each
  - Add `combined_eyes` key (bitwise OR of left + right eye masks)
  - Return dict of all masks
- [ ] Write `tests/test_masks.py`:
  - Test: `rasterize_mask` output is correct dtype and shape
  - Test: mask contains only 0 and 255
  - Test: mask for a known polygon has nonzero area
  - Test: `build_region_masks` returns expected keys including `combined_eyes`

**Acceptance Criteria:**
- Masks are `uint8` with values only 0 or 255
- `combined_eyes` mask is the union of left and right eye masks
- Tests pass: `uv run pytest tests/test_masks.py`

---

## Phase 4: Distance Fields, Remapping & Combination

**Purpose:** Build the core image-processing modules that transform masks into influence maps and combine them.

**Rationale:** These three modules form a logical chain (distance → remap → combine) and are testable with synthetic mask inputs — no MediaPipe dependency needed for unit tests.

### 4.1 Distance Field Computation

- [ ] Create `distance_fields.py` with `from __future__ import annotations`
- [ ] Implement `compute_distance_field(mask: np.ndarray) -> np.ndarray`:
  - Invert mask (distance from non-masked regions)
  - Apply `scipy.ndimage.distance_transform_edt`
  - Return `float64` ndarray of pixel-unit distances
- [ ] Write `tests/test_distance_fields.py`:
  - Test: output shape matches input shape
  - Test: output dtype is `float64`
  - Test: pixels inside mask region have distance 0.0
  - Test: distance increases moving away from mask boundary
  - Use synthetic masks (e.g., a centered rectangle) for deterministic tests

**Acceptance Criteria:**
- Distance is 0.0 at mask pixels, positive elsewhere
- Output is `float64` and same shape as input
- Tests pass: `uv run pytest tests/test_distance_fields.py`

### 4.2 Influence Remapping

- [ ] Create `remap.py` with `from __future__ import annotations`
- [ ] Implement `remap_influence(distance_field: np.ndarray, config: RemapConfig) -> np.ndarray`:
  - Clamp distance at `config.clamp_distance`
  - Apply curve based on `config.curve`:
    - `"linear"`: `max(0, 1 - d / radius)`
    - `"gaussian"`: `exp(-(d² / (2σ²)))`
    - `"exponential"`: `exp(-d / τ)`
  - Return `float64` array in range [0.0, 1.0]
- [ ] Write `tests/test_remap.py`:
  - Test: output values are in [0.0, 1.0]
  - Test: influence is 1.0 at distance 0.0 for all curve types
  - Test: influence decreases as distance increases
  - Test: linear curve reaches 0.0 at d >= radius
  - Test: invalid curve name raises `ValueError`

**Acceptance Criteria:**
- All three curve types produce values in [0.0, 1.0]
- Influence is maximal (1.0) at mask boundary and decays outward
- Tests pass: `uv run pytest tests/test_remap.py`

### 4.3 Map Combination

- [ ] Create `combine.py` with `from __future__ import annotations`
- [ ] Implement `combine_maps(maps: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray`:
  - Weighted sum of specified maps
  - Normalize result to [0.0, 1.0]
  - Raise `ValueError` if weight keys don't match map keys
- [ ] Write `tests/test_combine.py`:
  - Test: output is normalized to [0.0, 1.0]
  - Test: equal weights on identical maps returns the same map
  - Test: zero weight for a map excludes it from result
  - Test: mismatched keys raise `ValueError`

**Acceptance Criteria:**
- Weighted combination produces array in [0.0, 1.0]
- Weight keys must match map keys
- Tests pass: `uv run pytest tests/test_combine.py`

---

## Phase 5: Storage & Visualization

**Purpose:** Build output saving and visual debugging utilities.

**Rationale:** Deferred until after processing modules so there's real data to save/visualize. These are needed before the pipeline can produce its full output set.

### 5.1 Storage Utilities

- [ ] Create `storage.py` with `from __future__ import annotations`
- [ ] Implement `save_image(image: np.ndarray, path: str | Path) -> None`:
  - Use `cv2.imwrite`, create parent dirs if needed
- [ ] Implement `save_array(array: np.ndarray, path: str | Path) -> None`:
  - Use `np.save`
- [ ] Implement `load_image(path: str | Path) -> np.ndarray`:
  - Use `cv2.imread`, raise `FileNotFoundError` if missing
  - Return BGR ndarray (OpenCV default)
- [ ] Implement `ensure_output_dir(base: str | Path, image_name: str) -> Path`:
  - Create `<base>/<image_name>/` directory structure
  - Return the created path

**Acceptance Criteria:**
- Images round-trip through save/load
- `.npy` files preserve exact float64 values
- Output directories are created automatically
- No tests required for thin I/O wrappers — tested implicitly by pipeline integration tests

### 5.2 Visualization

- [ ] Create `viz.py` with `from __future__ import annotations`
- [ ] Implement `draw_landmarks(image: np.ndarray, landmarks: LandmarkResult) -> np.ndarray`:
  - Draw landmark points on a copy of the image
  - Return annotated image (does not mutate input)
- [ ] Implement `colorize_map(array: np.ndarray, colormap: str = "inferno") -> np.ndarray`:
  - Apply matplotlib colormap to a [0, 1] float array
  - Return BGR uint8 image for saving with OpenCV
- [ ] Implement `make_contact_sheet(images: dict[str, np.ndarray], columns: int = 4) -> np.ndarray`:
  - Arrange labeled images in a grid
  - Resize to uniform cell size
  - Return single composite image

**Acceptance Criteria:**
- `draw_landmarks` returns a new array (no mutation)
- `colorize_map` returns uint8 BGR array
- `make_contact_sheet` handles varying input sizes
- No separate test file — covered by pipeline integration tests

---

## Phase 6: Pipeline Composition & CLI

**Purpose:** Wire all modules into the end-to-end pipeline and provide a command-line script.

**Rationale:** Final phase because it depends on every prior module. This is where integration testing validates the full flow.

### 6.1 Pipeline Module

- [ ] Create `pipelines.py` with `from __future__ import annotations`
- [ ] Implement `run_feature_distance_pipeline(image: np.ndarray, config: PipelineConfig | None = None) -> PipelineResult`:
  - Default config if None provided
  - Call each module in sequence: landmarks → regions → masks → distance → remap → combine
  - Return `PipelineResult` with all intermediate data
- [ ] Implement `save_pipeline_outputs(result: PipelineResult, image: np.ndarray, output_dir: Path) -> None`:
  - Save all outputs per the output directory structure in the feature description
  - Use `viz.py` for colormaps and contact sheet
  - Use `storage.py` for file I/O
- [ ] Write `tests/test_pipelines.py`:
  - Test: `run_feature_distance_pipeline` returns `PipelineResult` with all expected fields populated
  - Test: all masks, distance fields, and influence maps have correct shapes
  - Test: combined map is in [0.0, 1.0]
  - Test: pipeline with default config runs without errors on test image
  - Test: `save_pipeline_outputs` creates expected files in a temp directory

**Acceptance Criteria:**
- Pipeline runs end-to-end on test image with default config
- `PipelineResult` contains all expected intermediate data
- All output files are created in correct directory structure
- Tests pass: `uv run pytest tests/test_pipelines.py`

### 6.2 CLI Script

- [ ] Create `scripts/run_pipeline.py`:
  - Accept image path as positional argument via `argparse`
  - Optional `--output-dir` flag (default: `"output"`)
  - Optional `--eye-weight`, `--mouth-weight`, `--curve`, `--radius` flags
  - Load image, build config from args, run pipeline, save outputs
  - Print summary of what was saved
- [ ] Verify manually: `uv run python scripts/run_pipeline.py test_images/<image>.jpg` produces all expected output files

**Acceptance Criteria:**
- `uv run python scripts/run_pipeline.py <image>` runs successfully and produces output directory with all files listed in feature description
- `--help` shows available options
- Custom flags override default config values

### 6.3 Package Exports & Final Cleanup

- [ ] Update `src/portrait_map_lab/__init__.py`:
  - Import and re-export primary public functions and classes
  - Define `__all__`
- [ ] Verify `ruff check src/ tests/` and `ruff format --check src/ tests/` pass clean
- [ ] Run full test suite: `uv run pytest` — all tests pass
- [ ] Verify `uv pip install -e .` installs cleanly in a fresh venv

**Acceptance Criteria:**
- `from portrait_map_lab import run_feature_distance_pipeline, PipelineConfig` works
- All linting and formatting checks pass
- Full test suite passes
- Package installs cleanly

---

## Dependency Graph

```
Phase 1 (Bootstrap)
  1.1 pyproject.toml → 1.2 models.py
                          |
Phase 2 (Landmarks)       |
  2.1 test image → 2.2 landmarks.py
                          |
Phase 3 (Regions/Masks)   |
  3.1 face_regions.py → 3.2 masks.py
                               |
Phase 4 (Processing)           |
  4.1 distance_fields.py → 4.2 remap.py → 4.3 combine.py
                                                  |
Phase 5 (Output)                                  |
  5.1 storage.py ─┐                               |
  5.2 viz.py ─────┤                               |
                  |                               |
Phase 6 (Integration)                             |
  6.1 pipelines.py ←─────────────────────────────┘
    → 6.2 CLI script
    → 6.3 exports & cleanup
```

Phases 2–4 are strictly sequential (each depends on the prior).
Phase 5 can technically start after Phase 1 but is placed here because it's more useful once there's real data.
Phase 6 depends on everything.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `from __future__ import annotations` in every module | Enables modern type syntax consistently, avoids forward-reference issues, zero runtime cost |
| `py.typed` marker file | PEP 561 compliance — typed package support for consumers' type checkers |
| Synthetic masks for unit tests (Phases 4.x) | Decouples processing tests from MediaPipe — faster, deterministic, no model download needed |
| Real image integration test only in Phase 6 | Keeps unit tests fast; validates full flow once at the end |
| `hatchling` build backend | Minimal, modern, zero-config for src layout — preferred over setuptools for new projects |
| No `conftest.py` fixtures for Phases 4.x | Synthetic test data is trivial (numpy arrays) and clearer when inline in each test file |
| `ruff` for both lint and format | Single tool replaces black + isort + flake8 + pyflakes — modern Python standard |
| `storage.py` and `viz.py` not unit-tested separately | Thin wrappers around OpenCV/matplotlib — tested via pipeline integration tests. Avoids brittle image-comparison tests |
| `argparse` for CLI (not click/typer) | Stdlib, zero dependencies, sufficient for a simple script. Can upgrade later if CLI grows |
