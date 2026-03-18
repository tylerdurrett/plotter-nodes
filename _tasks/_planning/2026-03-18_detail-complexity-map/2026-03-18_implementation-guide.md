# Implementation Guide: Complexity Map

**Date:** 2026-03-18
**Feature:** Complexity Map
**Source:** [2026-03-18_feature-description.md](2026-03-18_feature-description.md)

## Overview

This implementation adds a complexity map that measures local image complexity and uses it in two ways: (1) as a standalone exported map for downstream consumption, and (2) as a flow speed modulator where particles slow down in detail-rich areas. The core computation lives in `complexity_map.py`, flow speed derivation in `flow_speed.py`, and both integrate into the existing pipeline, export, and CLI systems.

Phases are sequenced bottom-up: data models first, then core computation (testable with synthetic data), then flow speed derivation, then pipeline wiring, then export bundle, then CLI, then docs. The flow speed module (Phase 3) is separate from the core complexity module (Phase 2) because they're distinct concerns — one measures image structure, the other translates it into particle behavior.

**Open question resolutions:**

- **Percentile vs. max normalization**: Percentile-based with the percentile exposed as a config parameter (default 99.0). Set `normalize_percentile=100.0` for max normalization. Robust defaults, full flexibility.
- **Multiscale vs. single-scale default**: Single-scale gradient energy. Simplest to debug. Multiscale available as `metric="multiscale_gradient"`.
- **Density integration**: Removed. Complexity does not feed into density composition. It drives flow speed instead — a more natural fit for "how lines behave" vs. "where lines go."

## File Structure

```
src/portrait_map_lab/
  models.py            # MODIFIED: add ComplexityConfig, ComplexityResult, FlowSpeedConfig; add flow_speed to FlowResult
  complexity_map.py    # NEW: gradient energy, laplacian energy, multiscale gradient, normalization, masking
  flow_speed.py        # NEW: complexity-to-speed derivation
  pipelines.py         # MODIFIED: add run_complexity_pipeline, save_complexity_outputs; update run_flow_pipeline, run_all_pipelines
  export.py            # MODIFIED: add complexity and flow_speed to _MAP_DEFINITIONS
  __init__.py          # MODIFIED: add new exports

scripts/
  run_pipeline.py      # MODIFIED: add complexity subcommand, update all and flow subcommands

tests/
  test_models.py       # MODIFIED: add ComplexityConfig/ComplexityResult/FlowSpeedConfig tests
  test_complexity_map.py   # NEW: unit tests for all metrics, normalization, masking
  test_flow_speed.py       # NEW: unit tests for speed derivation
  test_complexity_pipeline.py # NEW: integration tests for complexity pipeline and flow integration

docs/
  pipeline/
    README.md                  # MODIFIED: add complexity section
    13-complexity-map.md       # NEW: complexity map pipeline documentation
    architecture.md            # MODIFIED: add new modules to tables
```

---

## Phase 1: Data Models

**Purpose:** Define the configuration and result dataclasses needed by all subsequent phases.

**Rationale:** Models have no dependencies on new code and are imported by every other phase. Adding `flow_speed` to `FlowResult` here means later phases can populate it without modifying models again.

### 1.1 Add ComplexityConfig and ComplexityResult to models.py

- [x] Add `ComplexityConfig` dataclass (mutable, `slots=True`) with fields:
  - `metric: str = "gradient"` — one of `"gradient"`, `"laplacian"`, `"multiscale_gradient"`
  - `sigma: float = 3.0` — Gaussian smoothing sigma for single-scale metrics
  - `scales: list[float] = field(default_factory=lambda: [1.0, 3.0, 8.0])` — sigma values for multiscale metric
  - `scale_weights: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])` — weights per scale
  - `normalize_percentile: float = 99.0` — percentile for normalization (100.0 = max)
  - `output_dir: str = "output"`
- [x] Add `ComplexityResult` dataclass (frozen, `slots=True`) with fields:
  - `raw_complexity: np.ndarray` — unnormalized metric output
  - `complexity: np.ndarray` — normalized [0, 1] complexity map
  - `metric: str` — which metric was used
- [x] Add both classes to `__all__` in `models.py`

### 1.2 Add FlowSpeedConfig to models.py

- [x] Add `FlowSpeedConfig` dataclass (mutable, `slots=True`) with fields:
  - `speed_min: float = 0.3` — speed in most complex areas
  - `speed_max: float = 1.0` — speed in smooth areas
- [x] Add to `__all__`

### 1.3 Add flow_speed field to FlowResult

- [x] Add `flow_speed: np.ndarray | None = None` field to `FlowResult`
- [x] Note: `FlowResult` is frozen, so this field must be set at construction time. When no complexity is available, pass `None`.

### 1.4 Write model tests

- [x] Write tests in `test_models.py`: `TestComplexityConfig` (defaults, mutable, independent defaults for list fields), `TestComplexityResult` (creation, frozen), `TestFlowSpeedConfig` (defaults, mutable)
- [x] Write test for `FlowResult` construction with `flow_speed=None` (backward compatible) and with an array

**Acceptance Criteria:**
- `ComplexityConfig()` creates instance with expected defaults (`metric="gradient"`, `sigma=3.0`, `normalize_percentile=99.0`) ✅
- `ComplexityConfig` is mutable; `ComplexityResult` is frozen ✅
- Separate `ComplexityConfig` instances do not share mutable list state ✅
- `FlowSpeedConfig()` has `speed_min=0.3`, `speed_max=1.0` ✅
- `FlowResult` accepts `flow_speed=None` without error ✅
- `pytest tests/test_models.py` passes ✅

**Implementation Notes:**
- Phase 1 completed successfully (2026-03-18)
- All data models added to models.py with proper type annotations
- Comprehensive tests added including backward compatibility test for FlowResult
- Code quality verified with ruff (all checks passing)
- All 47 existing model tests continue to pass

---

## Phase 2: Core Complexity Map Module

**Purpose:** Implement all complexity metric functions as standalone composable functions, testable with synthetic data.

**Rationale:** This is the core new functionality. Building it before pipeline integration lets us unit test each metric in isolation using synthetic images (gradients, edges, flat regions) with no real image dependency.

### 2.1 Gradient energy metric

- [x] Create `src/portrait_map_lab/complexity_map.py`
- [x] Implement `compute_gradient_energy(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray`:
  1. Compute Sobel gradients Gx, Gy (CV_64F, ksize=3)
  2. Compute magnitude: `sqrt(Gx² + Gy²)`
  3. Smooth with Gaussian (sigma)
  4. Return raw float64 (not yet normalized)
- [x] Write tests in `tests/test_complexity_map.py`: `TestComputeGradientEnergy`
  - Flat image produces near-zero energy everywhere
  - Image with a sharp edge produces high energy at the edge
  - Output shape matches input shape, dtype is float64
  - All values are non-negative
  - Larger sigma produces smoother output (lower peak, wider spread)

**Acceptance Criteria:**
- Flat input → all values < 1e-6 ✅
- Sharp vertical edge → peak energy at edge location ✅
- Output is float64, same shape as input, all values >= 0 ✅
- `pytest tests/test_complexity_map.py::TestComputeGradientEnergy` passes ✅

### 2.2 Laplacian energy metric

- [x] Implement `compute_laplacian_energy(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray`:
  1. Compute Laplacian (cv2.Laplacian, CV_64F)
  2. Take absolute value
  3. Smooth with Gaussian (sigma)
  4. Return raw float64
- [x] Write tests: `TestComputeLaplacianEnergy`
  - Flat image produces near-zero energy
  - Fine checkerboard pattern produces high energy
  - Sharp edge produces energy at the edge
  - Output is float64, non-negative

**Acceptance Criteria:**
- Flat input → all values < 1e-6 ✅
- Checkerboard input → uniformly high energy ✅
- Output is float64, same shape, all values >= 0 ✅
- `pytest tests/test_complexity_map.py::TestComputeLaplacianEnergy` passes ✅

### 2.3 Multiscale gradient energy metric

- [x] Implement `compute_multiscale_gradient_energy(gray: np.ndarray, scales: list[float], weights: list[float]) -> np.ndarray`:
  1. For each scale sigma, compute `compute_gradient_energy(gray, sigma)`
  2. Weighted sum: `Σ(wᵢ × energy_i)`
  3. Return raw float64
- [x] Validate that `len(scales) == len(weights)`, raise `ValueError` if not
- [x] Write tests: `TestComputeMultiscaleGradientEnergy`
  - Produces valid output with default scales/weights
  - Mismatched scales/weights lengths raises ValueError
  - Result has contributions from multiple scales
  - Output is float64, non-negative

**Acceptance Criteria:**
- Default scales `[1.0, 3.0, 8.0]` with weights `[0.5, 0.3, 0.2]` produces valid output ✅
- `scales=[1.0]`, `weights=[1.0, 2.0]` raises `ValueError` ✅
- Output is float64, same shape, all values >= 0 ✅
- `pytest tests/test_complexity_map.py::TestComputeMultiscaleGradientEnergy` passes ✅

### 2.4 Normalization and masking

- [x] Implement `normalize_map(raw: np.ndarray, percentile: float = 99.0) -> np.ndarray`:
  1. Compute the normalization ceiling as `np.percentile(raw, percentile)`
  2. If ceiling <= 0, return zeros
  3. Divide by ceiling, clip to [0, 1]
  4. Return float64 in [0, 1]
- [x] Implement `apply_mask(map_array: np.ndarray, mask: np.ndarray) -> np.ndarray`:
  1. Convert mask to float64 binary (0 or 1) — handle both uint8 0/255 and float 0/1 inputs
  2. Multiply element-wise by mask
  3. Return float64 in [0, 1]
- [x] Write tests: `TestNormalizeMap` (percentile=100 is max, percentile=99 clips top 1%, zero input returns zeros, output always in [0,1])
- [x] Write tests: `TestApplyMask` (zeroes out masked regions, preserves unmasked, handles uint8 0/255 and float 0/1)

**Acceptance Criteria:**
- `normalize_map(arr, percentile=100.0)` divides by max ✅
- `normalize_map(arr, percentile=99.0)` clips the top 1% of values to 1.0 ✅
- `normalize_map(np.zeros(...))` returns zeros (no divide-by-zero) ✅
- `apply_mask` zeroes out regions where mask is 0, preserves where mask is nonzero ✅
- `apply_mask` handles both uint8 (0/255) and float (0.0/1.0) masks ✅
- `pytest tests/test_complexity_map.py::TestNormalizeMap tests/test_complexity_map.py::TestApplyMask` passes ✅

### 2.5 Main entry point and module exports

- [x] Implement `compute_complexity_map(image: np.ndarray, config: ComplexityConfig | None = None, mask: np.ndarray | None = None) -> ComplexityResult`:
  1. Convert to grayscale float64 [0, 1] (handle BGR uint8 and grayscale input)
  2. Dispatch to metric function based on `config.metric`
  3. Normalize via `normalize_map`
  4. Apply mask if provided
  5. Return `ComplexityResult` with raw_complexity, complexity, metric name
- [x] Raise `ValueError` for unknown metric names
- [x] Add `__all__` listing all public names
- [x] Write tests: `TestComputeComplexityMap`
  - Default config produces valid result
  - Each metric name dispatches correctly
  - Unknown metric raises ValueError
  - Optional mask is applied when provided
  - Result fields have correct types and shapes

**Acceptance Criteria:**
- `compute_complexity_map(image)` produces `ComplexityResult` with `complexity` in [0, 1] ✅
- Each metric dispatches correctly; `"unknown"` raises `ValueError` ✅
- With mask, complexity is zero outside masked region ✅
- `ruff check src/portrait_map_lab/complexity_map.py` passes ✅
- `pytest tests/test_complexity_map.py` passes (all tests) ✅

**Implementation Notes:**
- Phase 2 completed successfully (2026-03-18)
- All complexity metric functions implemented as standalone composable functions
- Comprehensive test coverage with 30 tests, all passing
- Added robust handling for different image input formats (BGR/grayscale, uint8/float64)
- Code quality verified with ruff (all checks passing)
- Tested with real images to verify visual output is correct

---

## Phase 3: Flow Speed Module

**Purpose:** Implement the complexity-to-speed derivation as a standalone module.

**Rationale:** Speed derivation is a distinct concern from complexity measurement. Separating it lets us test the transform in isolation and makes it easy to swap in different speed curves later (gamma, sigmoid, etc.) without touching the complexity computation.

### 3.1 Implement compute_flow_speed

- [x] Create `src/portrait_map_lab/flow_speed.py`
- [x] Implement `compute_flow_speed(complexity: np.ndarray, config: FlowSpeedConfig | None = None) -> np.ndarray`:
  1. If config is None, use defaults
  2. Compute: `speed = speed_max - complexity * (speed_max - speed_min)`
  3. Clip to [speed_min, speed_max]
  4. Return float64
- [x] Add `__all__` with `compute_flow_speed`
- [x] Write tests in `tests/test_flow_speed.py`: `TestComputeFlowSpeed`
  - Zero complexity → speed_max everywhere
  - Full complexity (1.0) → speed_min everywhere
  - Mid complexity (0.5) → midpoint speed
  - Output shape matches input, dtype is float64
  - All values in [speed_min, speed_max]
  - Custom speed_min/speed_max work correctly

**Acceptance Criteria:**
- `complexity=0.0` everywhere → all values equal `speed_max` (1.0) ✅
- `complexity=1.0` everywhere → all values equal `speed_min` (0.3) ✅
- `complexity=0.5` → values at `0.65` (midpoint of 0.3–1.0) ✅
- Output is float64, same shape, all values in [speed_min, speed_max] ✅
- `pytest tests/test_flow_speed.py` passes ✅

**Implementation Notes:**
- Phase 3 completed successfully (2026-03-18)
- Created flow_speed.py module with compute_flow_speed function
- Implemented linear inverse mapping for speed derivation
- Added comprehensive test suite with 11 tests covering all edge cases
- Fixed floating-point precision issues in tests using np.isclose
- All tests pass, code quality verified with ruff

---

## Phase 4: Pipeline Integration

**Purpose:** Wire complexity and flow speed into the pipeline system with file output and visualization.

**Rationale:** Depends on Phases 1-3. This phase makes everything runnable end-to-end before adding CLI and export.

### 4.1 Implement run_complexity_pipeline

- [x] Add `run_complexity_pipeline(image: np.ndarray, config: ComplexityConfig | None = None, mask: np.ndarray | None = None) -> ComplexityResult` to `pipelines.py`
  - Thin wrapper: calls `compute_complexity_map`, adds logging
  - Consistent logging style with existing pipelines
- [x] Add imports of `compute_complexity_map`, `ComplexityConfig`, `ComplexityResult`

**Acceptance Criteria:**
- `run_complexity_pipeline(test_image)` returns `ComplexityResult` with all fields populated ✅
- `complexity` values in [0.0, 1.0] ✅
- Works with default config ✅

### 4.2 Implement save_complexity_outputs

- [x] Add `save_complexity_outputs(result: ComplexityResult, output_dir: Path, image: np.ndarray | None = None) -> None` to `pipelines.py`
  - Creates `complexity/` subdirectory under output_dir
  - Saves raw complexity heatmap (colorized `"viridis"`) as `<metric>_energy.png`
  - Saves raw as `<metric>_energy_raw.npy`
  - Saves normalized complexity as `complexity.png` (colorized `"inferno"`)
  - Saves as `complexity_raw.npy`
  - Creates contact sheet

**Acceptance Criteria:**
- All expected files created in `output_dir/complexity/` ✅
- `.npy` files loadable with correct shapes ✅
- Contact sheet includes all visualizations ✅

### 4.3 Update run_flow_pipeline to compute speed

- [x] Add optional `complexity_result: ComplexityResult | None = None` and `speed_config: FlowSpeedConfig | None = None` parameters to `run_flow_pipeline`
- [x] When `complexity_result` is provided:
  - Call `compute_flow_speed(complexity_result.complexity, speed_config)`
  - Pass result as `flow_speed` when constructing `FlowResult`
- [x] When `complexity_result` is None, pass `flow_speed=None` to `FlowResult`
- [x] Add logging for speed computation

**Acceptance Criteria:**
- `run_flow_pipeline(image, contour_result)` (no complexity) produces `FlowResult` with `flow_speed=None` — identical to current behavior ✅
- `run_flow_pipeline(image, contour_result, complexity_result=cr)` produces `FlowResult` with `flow_speed` array of correct shape, values in [speed_min, speed_max] ✅

### 4.4 Update save_flow_outputs to save speed

- [x] When `result.flow_speed` is not None:
  - Save `flow_speed.png` (colorized heatmap) and `flow_speed_raw.npy` to `flow/` directory
  - Add flow speed visualization to the flow contact sheet

**Acceptance Criteria:**
- `flow_speed.png` and `flow_speed_raw.npy` appear in flow output when speed is computed ✅
- Existing flow outputs unchanged when `flow_speed` is None ✅

### 4.5 Update run_all_pipelines

- [x] Add optional `complexity_config: ComplexityConfig | None = None` and `speed_config: FlowSpeedConfig | None = None` parameters
- [x] Run complexity pipeline after contour, before flow (since flow needs complexity for speed)
- [x] Use the filled mask from contour result as the complexity mask (if available), so complexity is automatically scoped to the face/head region
- [x] Pass complexity result to `run_flow_pipeline`
- [x] Update `ComposedResult` to include `complexity_result: ComplexityResult | None`
- [x] Update `save_all_outputs` to call `save_complexity_outputs` when complexity result is present

**Acceptance Criteria:**
- `run_all_pipelines(image)` with no complexity config produces identical output to current ✅
- `run_all_pipelines(image, complexity_config=ComplexityConfig())` runs complexity pipeline, computes flow speed, saves outputs ✅
- Existing feature, contour, density, flow direction outputs unaffected ✅

### 4.6 Write integration tests

- [x] Create `tests/test_complexity_pipeline.py`:
  - `TestRunComplexityPipeline`: shapes, normalization, all metrics, masking
  - `TestSaveComplexityOutputs`: expected files
  - `TestFlowWithSpeed`: flow result has speed when complexity provided, None when not
  - `TestAllPipelinesWithComplexity`: end-to-end with complexity enabled vs. disabled

**Acceptance Criteria:**
- `pytest tests/test_complexity_pipeline.py` passes ✅
- Tests cover standalone complexity, flow speed integration, and full pipeline ✅

### 4.7 Update __init__.py exports

- [x] Add imports and `__all__` entries for: `ComplexityConfig`, `ComplexityResult`, `FlowSpeedConfig`, `run_complexity_pipeline`, `save_complexity_outputs`, `compute_complexity_map`, `compute_gradient_energy`, `compute_laplacian_energy`, `compute_multiscale_gradient_energy`, `normalize_map`, `apply_mask`, `compute_flow_speed`

**Acceptance Criteria:**
- All new public API names importable from `portrait_map_lab` ✅
- `ruff check src/portrait_map_lab/__init__.py` passes ✅

**Implementation Notes:**
- Phase 4 completed successfully (2026-03-18)
- All pipeline integration complete with full backward compatibility
- Flow speed visualization uses "plasma" colormap for better differentiation
- Comprehensive tests written with 9 passing, 3 skipped (face detection not available in test env)
- Code quality verified with ruff (all checks passing)
- Imports automatically sorted by ruff for consistency

---

## Phase 5: Export Bundle

**Purpose:** Add complexity and flow_speed to the binary export bundle.

**Rationale:** Sequenced after pipeline integration so the maps are computed and available. The export system is the bridge to the TypeScript plotter.

### 5.1 Add complexity and flow_speed to export definitions

- [ ] Add two entries to `_MAP_DEFINITIONS` in `export.py`:
  - `("complexity", "complexity_result.complexity", (0.0, 1.0), "Local image complexity for speed modulation")`
  - `("flow_speed", "flow_result.flow_speed", (0.0, 1.0), "Particle speed scalar derived from complexity")`
- [ ] Handle the case where `complexity_result` is None (skip the map, don't fail)
- [ ] Handle the case where `flow_speed` is None (skip the map)

### 5.2 Write export tests

- [ ] Add tests to verify complexity and flow_speed appear in export when present
- [ ] Verify export works correctly when complexity is not computed (maps omitted, no error)

**Acceptance Criteria:**
- Export with complexity produces `complexity.bin` and `flow_speed.bin` in the bundle
- Export without complexity omits those files and succeeds
- Manifest includes correct metadata for new maps
- `pytest tests/test_export.py` passes (if export tests exist, otherwise add to test_complexity_pipeline.py)

---

## Phase 6: CLI Extension

**Purpose:** Add `complexity` subcommand and integrate into `all` and `flow` subcommands.

**Rationale:** Sequenced after pipeline and export work so we can verify correctness in isolation first.

### 6.1 Add complexity subcommand to run_pipeline.py

- [ ] Add `complexity` subparser with arguments:
  - `--metric` (choices: `gradient`, `laplacian`, `multiscale_gradient`, default: `gradient`)
  - `--complexity-sigma` (float, default: 3.0)
  - `--scales` (float, nargs="+", default: [1.0, 3.0, 8.0])
  - `--scale-weights` (float, nargs="+", default: [0.5, 0.3, 0.2])
  - `--normalize-percentile` (float, default: 99.0)
  - `--mask-image` (Path, optional)
- [ ] Implement `handle_complexity` function following existing handler pattern
- [ ] Add `build_complexity_config(args) -> ComplexityConfig` helper

**Acceptance Criteria:**
- `uv run python scripts/run_pipeline.py complexity test_images/20230427-171404.JPG` writes to `output/20230427-171404/complexity/`
- `uv run python scripts/run_pipeline.py complexity --metric laplacian test_images/20230427-171404.JPG` uses Laplacian
- `uv run python scripts/run_pipeline.py complexity --help` shows args

### 6.2 Integrate complexity into flow and all subcommands

- [ ] Add to `flow` subparser: `--metric`, `--complexity-sigma`, `--speed-min`, `--speed-max` — when metric is provided, compute complexity and derive speed
- [ ] Add to `all` subparser: same complexity args plus `--speed-min`, `--speed-max`
- [ ] Update `handle_flow` to optionally run complexity pipeline and pass to flow
- [ ] Update `handle_all` to optionally run complexity pipeline
- [ ] When `--metric` is not provided in `flow` or `all`, complexity is skipped (current behavior preserved)

**Acceptance Criteria:**
- `uv run python scripts/run_pipeline.py flow test_images/20230427-171404.JPG` (no complexity args) — current behavior, no speed output
- `uv run python scripts/run_pipeline.py flow --metric gradient test_images/20230427-171404.JPG` — computes complexity, produces flow_speed
- `uv run python scripts/run_pipeline.py all --metric gradient test_images/20230427-171404.JPG` — includes complexity and flow speed
- `uv run python scripts/run_pipeline.py all test_images/20230427-171404.JPG` — identical to current, no complexity

---

## Phase 7: Documentation

**Purpose:** Document the new complexity pipeline and update existing docs.

**Rationale:** Sequenced after all code is working so documentation reflects actual implementation.

### 7.1 Create complexity map pipeline documentation

- [ ] Create `docs/pipeline/13-complexity-map.md` following existing conventions
- [ ] Cover: metric options, normalization, masking, flow speed derivation, export integration
- [ ] Include code examples for standalone, pipeline, and flow integration usage
- [ ] Include comparison guidance: when to use each metric

**Acceptance Criteria:**
- Document follows existing stage doc structure
- Includes code examples, configuration table, output format
- Flow speed derivation clearly explained with formula

### 7.2 Update pipeline README and architecture

- [ ] Update `docs/pipeline/README.md`: add complexity to stages, add quick start, update data flow diagram to show complexity → flow speed path
- [ ] Update `docs/pipeline/architecture.md`: add new modules, data structures, updated flow diagram

**Acceptance Criteria:**
- Pipeline README shows complexity as a pipeline stage
- Architecture doc includes new modules and the complexity → speed data flow

---

## Phase 8: Final Verification

**Purpose:** End-to-end verification.

**Rationale:** Final pass to catch integration issues.

### 8.1 Full test suite and linting

- [ ] Run `uv run pytest` — all tests pass
- [ ] Run `uv run ruff check` — clean
- [ ] Run `uv run ruff format --check` — clean

**Acceptance Criteria:**
- All tests pass with no failures
- Linting and formatting clean

### 8.2 Visual verification

- [ ] Run `uv run python scripts/run_pipeline.py complexity test_images/20230427-171404.JPG` — inspect complexity heatmap
- [ ] Verify gradient energy shows high values at edges/detail, low in smooth regions
- [ ] Run with `--metric laplacian` — verify fine texture emphasis
- [ ] Run with `--metric multiscale_gradient` — verify multi-scale detail
- [ ] Run `uv run python scripts/run_pipeline.py flow --metric gradient test_images/20230427-171404.JPG` — inspect flow_speed heatmap
- [ ] Verify speed is low (dark) in complex areas, high (bright) in smooth areas
- [ ] Run `uv run python scripts/run_pipeline.py all test_images/20230427-171404.JPG` — verify no complexity/speed outputs (backward compat)
- [ ] Run `uv run python scripts/run_pipeline.py all --metric gradient --export test_images/20230427-171404.JPG` — verify complexity.bin and flow_speed.bin in export

**Acceptance Criteria:**
- Complexity map visually emphasizes structurally complex regions
- Flow speed is inversely correlated with complexity
- Each metric produces visibly different but reasonable results
- Existing pipeline output without complexity is unchanged
- Export bundle includes new maps when complexity is enabled

---

## Dependency Graph

```
Phase 1 (Models)
  1.1 (ComplexityConfig/Result) → 1.2 (FlowSpeedConfig) → 1.3 (FlowResult.flow_speed) → 1.4 (tests)
    │
    v
Phase 2 (Core Complexity Module)
  2.1 (gradient) → 2.2 (laplacian) → 2.3 (multiscale)
  2.1 ──────────────────────────────→ 2.4 (normalize + mask)
  2.3 + 2.4 → 2.5 (entry point)
    │
    v
Phase 3 (Flow Speed Module) ←── depends on Phase 1.2
  3.1 (compute_flow_speed + tests)
    │
    v
Phase 4 (Pipeline Integration) ←── depends on Phases 2 + 3
  4.1 (run_complexity) → 4.2 (save_complexity) → 4.3 (flow speed in run_flow)
  4.3 → 4.4 (save_flow speed) → 4.5 (run_all_pipelines) → 4.6 (tests) → 4.7 (__init__.py)
    │
    v
Phase 5 (Export Bundle)
  5.1 (add maps to definitions) → 5.2 (tests)
    │
    v
Phase 6 (CLI)
  6.1 (complexity subcommand) → 6.2 (flow + all integration)
    │
    v
Phase 7 (Documentation)
  7.1 (13-complexity-map.md) ──┐
  7.2 (README + architecture) ←┘
    │
    v
Phase 8 (Verification)
  8.1 (tests + linting) → 8.2 (visual verification)
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Complexity drives flow speed, not density | Density answers "where should lines exist" (semantic/tonal). Complexity answers "how should lines behave" (slow down in detail). Speed modulation is a more natural fit for local image complexity |
| Separate `flow_speed.py` module | Speed derivation is a distinct concern from complexity measurement. Separating them makes it easy to swap speed curves later without touching complexity computation |
| Linear speed mapping as default | `speed = max - complexity * (max - min)` is simple and predictable. Non-linear curves (gamma, sigmoid) can be added as options later without changing the interface |
| `flow_speed` is `None` when not computed (not uniform 1.0) | Explicit None lets downstream consumers distinguish "speed not computed" from "uniform speed." The plotter can default to its own behavior when speed is absent |
| Complexity not added to density composition | Per user direction. Density is already well-served by tonal + semantic maps. Complexity was a poor fit — "more detail" doesn't mean "should be darker/denser" |
| Percentile normalization (99th) as default | Robust against outlier spikes. `percentile=100.0` recovers max normalization. Parameter exposed for full flexibility |
| Single-scale gradient energy as default metric | Simplest to reason about and debug. Multiscale available as opt-in |
| Separate `complexity_map.py` module (not extending `etf.py`) | ETF computes directional edge flow; complexity computes scalar magnitude. Different purposes, different preprocessing |
| Filled contour mask as default complexity mask in `run_all_pipelines` | Scopes complexity to face/head region automatically, preventing background clutter from dominating |
| Complexity and flow_speed optional in export | When complexity isn't computed, export succeeds without those maps. No breaking change to existing export consumers |
