# Implementation Guide: Density Target Composition & Flow Fields

**Date:** 2026-03-18
**Feature:** Density Target Composition & Flow Fields
**Source:** [2026-03-18_feature-description.md](2026-03-18_feature-description.md)

## Overview

This guide implements Stage 2 (composed density targets) and Stage 3 (flow fields) of the portrait map lab in seven phases. The sequencing prioritizes incremental testability: each phase produces a working, testable artifact before the next begins.

The implementation starts with data models (everything depends on them), then builds the density target pipeline (Stage 2) as a self-contained capability, followed by the flow field pipeline (Stage 3), then LIC visualization, full integration, and finally documentation.

**Open question resolved:** The `all` CLI subcommand will be extended to include density and flow pipelines — `all` means "run everything" and should grow as the project grows. No separate `composed` subcommand is needed. Individual `density` and `flow` subcommands are added for running pieces independently.

## File Structure

```
src/portrait_map_lab/
  luminance.py          (new)  Grayscale extraction, CLAHE, tonal target
  compose.py            (new)  Multi-mode map composition
  etf.py                (new)  Edge Tangent Field computation
  flow_fields.py        (new)  Contour gradient flow, alignment, blending
  lic.py                (new)  Line Integral Convolution visualization
  models.py             (mod)  New config/result dataclasses
  pipelines.py          (mod)  New pipeline functions + save helpers
  viz.py                (mod)  Flow quiver overlay, LIC overlay
  __init__.py           (mod)  Export new public API

scripts/
  run_pipeline.py       (mod)  density, flow subcommands; extend all

tests/
  test_luminance.py     (new)
  test_compose.py       (new)
  test_etf.py           (new)
  test_flow_fields.py   (new)
  test_lic.py           (new)
  test_pipelines.py     (mod)  New integration tests

docs/pipeline/
  07-luminance-target.md    (new)
  08-density-composition.md (new)
  09-edge-tangent-field.md  (new)
  10-flow-fields.md         (new)
  11-lic-visualization.md   (new)
  architecture.md           (mod)
  README.md                 (mod)
```

## Output Directory Structure

```
output/<image>/
  features/           (existing, unchanged)
  contour/            (existing, unchanged)
  density/
    luminance.png
    clahe_luminance.png
    tonal_target.png
    importance.png
    density_target.png
    density_target_raw.npy
    contact_sheet.png
  flow/
    etf_coherence.png
    etf_quiver.png
    contour_flow_quiver.png
    blend_weight.png
    flow_lic.png
    flow_lic_overlay.png
    flow_quiver.png
    flow_x_raw.npy
    flow_y_raw.npy
    contact_sheet.png
```

---

## Phase 1: Data Models ✅ COMPLETE

**Purpose:** Define all new configuration and result dataclasses so downstream phases can import them immediately.

**Rationale:** Every new module depends on these types. Defining them first avoids circular dependencies and lets each phase focus on logic rather than data structure design.

**Implementation Notes:**
- Successfully added all 5 configuration dataclasses (LuminanceConfig, ComposeConfig, ETFConfig, FlowConfig, LICConfig)
- Successfully added all 4 result dataclasses (DensityResult, ETFResult, FlowResult, ComposedResult)
- All dataclasses follow existing patterns: mutable configs with slots=True, frozen results with slots=True
- Used field(default_factory=...) for nested mutable configs to ensure independent instances
- Added comprehensive tests for all new dataclasses (19 new test methods)
- All 37 tests pass, no lint errors

### 1.1 Add configuration dataclasses to `models.py` ✅

- [x] Add `LuminanceConfig` dataclass (mutable, `slots=True`): `clip_limit: float = 2.0`, `tile_size: int = 8`
- [x] Add `ComposeConfig` dataclass (mutable, `slots=True`): `luminance: LuminanceConfig`, `feature_weight: float = 0.6`, `contour_weight: float = 0.4`, `tonal_blend_mode: str = "multiply"`, `tonal_weight: float = 1.0`, `importance_weight: float = 1.0`, `gamma: float = 1.0`
- [x] Add `ETFConfig` dataclass (mutable, `slots=True`): `blur_sigma: float = 1.5`, `structure_sigma: float = 5.0`, `refine_sigma: float = 3.0`, `refine_iterations: int = 2`, `sobel_ksize: int = 3`
- [x] Add `FlowConfig` dataclass (mutable, `slots=True`): `etf: ETFConfig`, `contour_smooth_sigma: float = 1.0`, `blend_mode: str = "coherence"`, `coherence_power: float = 2.0`, `fallback_threshold: float = 0.1`
- [x] Add `LICConfig` dataclass (mutable, `slots=True`): `length: int = 30`, `step: float = 1.0`, `seed: int = 42`, `use_bilinear: bool = True`
- [x] Update `__all__` in models.py

**Acceptance Criteria:**
- All config dataclasses instantiate with defaults
- Configs are mutable (not frozen)
- `field(default_factory=...)` used for mutable defaults (LuminanceConfig inside ComposeConfig, ETFConfig inside FlowConfig)

### 1.2 Add result dataclasses to `models.py` ✅

- [x] Add `DensityResult` dataclass (frozen, `slots=True`): `luminance`, `clahe_luminance`, `tonal_target`, `importance`, `density_target` — all `np.ndarray`
- [x] Add `ETFResult` dataclass (frozen, `slots=True`): `tangent_x`, `tangent_y`, `coherence`, `gradient_magnitude` — all `np.ndarray`
- [x] Add `FlowResult` dataclass (frozen, `slots=True`): `etf: ETFResult`, `contour_flow_x`, `contour_flow_y`, `blend_weight`, `flow_x`, `flow_y` — all arrays except etf
- [x] Add `ComposedResult` dataclass (frozen, `slots=True`): `feature_result: PipelineResult`, `contour_result: ContourResult`, `density_result: DensityResult`, `flow_result: FlowResult`, `lic_image: np.ndarray`
- [x] Update `__all__` in models.py
- [x] Write tests in `tests/test_models.py` (extend existing file): creation with defaults, frozen behavior, independent default instances for mutable configs

**Acceptance Criteria:**
- Result dataclasses are frozen (raises `FrozenInstanceError` on attribute assignment)
- Config dataclasses with nested mutable defaults produce independent instances
- All new types appear in `__all__`

---

## Phase 2: Luminance & Density Composition

**Purpose:** Implement the complete Stage 2 density target pipeline — from luminance extraction through composed density output.

**Rationale:** Stage 2 is self-contained and produces a directly useful artifact (the density target map). Building luminance and composition together in one phase means the density pipeline is end-to-end testable at the end of this phase.

### 2.1 Implement `luminance.py` ✅ COMPLETE

**Implementation Notes:**
- Successfully created `src/portrait_map_lab/luminance.py` with all 3 required functions
- Implemented BGR-to-grayscale conversion with proper float64 normalization
- CLAHE implementation uses OpenCV with configurable clip_limit and tile_size
- Tonal target correctly inverts CLAHE result (dark areas → high density values)
- Created comprehensive test suite with 18 test cases, all passing
- Fixed minor linting issues (missing newlines at end of files)
- Note: CLAHE redistributes histogram even for uniform images, so test adjusted to check for uniformity of output rather than exact value preservation

- [x] Create `src/portrait_map_lab/luminance.py` with `__all__`
- [x] Implement `extract_luminance(image: np.ndarray) -> np.ndarray` — BGR to grayscale float64 [0, 1] via `cv2.cvtColor`
- [x] Implement `apply_clahe(luminance: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray` — convert to uint8, apply `cv2.createCLAHE`, return float64 [0, 1]
- [x] Implement `compute_tonal_target(image: np.ndarray, config: LuminanceConfig | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — returns `(luminance, clahe_luminance, tonal_target)` where tonal_target is inverted CLAHE. Accept `None` config and create defaults (matching existing pipeline pattern).
- [x] Write `tests/test_luminance.py`:
  - Output shapes match input (H, W)
  - Output dtype is float64
  - Values in [0, 1] range
  - Tonal target is inverted: bright input → low density
  - Uniform input → uniform output (CLAHE maintains uniformity but may remap values)
  - Black image → all 1.0 tonal target; white image → all 0.0

**Acceptance Criteria:**
- `compute_tonal_target` produces three float64 arrays in [0, 1] ✅
- Dark input regions produce high tonal target values ✅
- All tests pass ✅ (18 tests, all passing)

### 2.2 Implement `compose.py` ✅ COMPLETE

**Implementation Notes:**
- Successfully created `src/portrait_map_lab/compose.py` with both required functions
- Implemented all 4 blend modes: multiply, screen, max, and weighted
- Weighted mode uses simple averaging (equal weights) to match normalization approach from combine.py
- Added comprehensive input validation with helpful error messages
- Gamma correction properly handles edge cases including zeros
- Created comprehensive test suite with 24 test cases covering all modes and edge cases
- All tests pass, no lint errors

- [x] Create `src/portrait_map_lab/compose.py` with `__all__`
- [x] Implement `compose_maps(map_a: np.ndarray, map_b: np.ndarray, mode: str = "multiply") -> np.ndarray` with modes:
  - `"multiply"`: `map_a * map_b` — element-wise product
  - `"screen"`: `1.0 - (1.0 - map_a) * (1.0 - map_b)` — inverse of multiply
  - `"max"`: `np.maximum(map_a, map_b)` — element-wise maximum
  - `"weighted"`: weighted sum (reuse normalization logic from `combine.py`)
  - Validate mode string, raise `ValueError` for unknown modes
  - Validate matching shapes
  - Output clipped to [0, 1]
- [x] Implement `build_density_target(tonal_target: np.ndarray, importance: np.ndarray, mode: str = "multiply", gamma: float = 1.0) -> np.ndarray` — compose then apply gamma: `result ** gamma`, clipped to [0, 1]
- [x] Write `tests/test_compose.py`:
  - Each blend mode produces expected results on known inputs
  - Multiply: (0.5 * 0.5 = 0.25), (1.0 * x = x), (0.0 * x = 0.0)
  - Screen: screen(0, 0) = 0, screen(1, x) = 1
  - Max: max picks the higher value
  - Gamma < 1.0 brightens, gamma > 1.0 darkens, gamma = 1.0 identity
  - Output always in [0, 1]
  - Invalid mode raises ValueError
  - Shape mismatch raises ValueError

**Acceptance Criteria:**
- All four blend modes produce mathematically correct results ✅
- Gamma correction works as expected ✅
- Output always float64 in [0, 1] ✅
- All tests pass ✅ (24 tests, all passing)

### 2.3 Density pipeline integration ✅ COMPLETE

**Implementation Notes:**
- Successfully added both `run_density_pipeline` and `save_density_outputs` functions to pipelines.py
- Functions follow existing pipeline patterns with comprehensive logging and error handling
- Added 7 comprehensive integration tests covering all aspects of the pipeline
- All tests pass (7/7) with no linting errors
- Tested with real images, produces expected outputs in `density/` subdirectory
- Code review confirms production-ready quality with excellent adherence to patterns

- [x] Add `run_density_pipeline(image: np.ndarray, feature_result: PipelineResult, contour_result: ContourResult, config: ComposeConfig | None = None) -> DensityResult` to `pipelines.py`
  - Compute tonal target via `compute_tonal_target`
  - Combine feature + contour importance via existing `combine_maps` with configured weights
  - Compose importance + tonal via `build_density_target`
  - Return `DensityResult` with all intermediates
- [x] Add `save_density_outputs(result: DensityResult, output_dir: Path, image: np.ndarray | None = None) -> None` to `pipelines.py`
  - Save each intermediate as colorized heatmap PNG (inferno for importance, hot for density/tonal)
  - Save `density_target_raw.npy`
  - Build and save contact sheet
- [x] Write integration tests in `tests/test_pipelines.py` (extend existing):
  - `run_density_pipeline` returns correct result type with all fields populated
  - All arrays have matching shapes
  - density_target values in [0, 1]
  - Save function creates expected files in temp directory

**Acceptance Criteria:** ✅ All met
- `run_density_pipeline` accepts pre-computed Stage 1 results (no re-running landmark detection) ✅
- All intermediate arrays preserved in result ✅
- Output files written to `density/` subdirectory ✅
- Contact sheet includes all intermediate visualizations ✅

---

## Phase 3: Edge Tangent Field ✅ COMPLETE

**Purpose:** Implement the ETF computation — the algorithmically deepest new module.

**Rationale:** The ETF is the foundation of Stage 3 and the most complex piece. Isolating it in its own phase allows focused testing of the numerical correctness (unit vectors, coherence range, behavior on known inputs) before integrating with flow blending.

**Implementation Notes:**
- Successfully created `src/portrait_map_lab/etf.py` with all 4 required functions
- Implemented structure tensor computation using Sobel gradients and Gaussian smoothing
- Implemented eigenvector extraction with special handling for degenerate cases (when Jxy ≈ 0)
- Fixed eigenvector calculation to properly handle axis-aligned edges
- Added comprehensive refinement with iterative smoothing and renormalization
- Created test suite with 19 test cases, all passing
- Fixed linting issues (line length, missing newlines)
- Note: Special logic added to handle degenerate cases where Jxy is near zero (common for axis-aligned edges)

### 3.1 Implement `etf.py` ✅

- [x] Create `src/portrait_map_lab/etf.py` with `__all__`
- [x] Implement `compute_structure_tensor(gray: np.ndarray, blur_sigma: float, structure_sigma: float, sobel_ksize: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - Gaussian blur input image
  - Sobel gradients (gx, gy) via `cv2.Sobel` with `cv2.CV_64F`
  - Structure tensor: `Jxx = gx*gx`, `Jxy = gx*gy`, `Jyy = gy*gy`
  - Smooth each component with `scipy.ndimage.gaussian_filter(sigma=structure_sigma)`
  - Return `(Jxx, Jxy, Jyy)`
- [x] Implement `extract_tangent_field(Jxx, Jxy, Jyy) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - Closed-form eigenvalues: `λ = trace/2 ± sqrt((trace/2)² - det)`
  - Minor eigenvector: `tx = Jxy`, `ty = λ₂ - Jxx` (with special handling for Jxy ≈ 0)
  - Normalize to unit length (guard against division by zero with `max(mag, 1e-10)`)
  - Coherence: `(λ₁ - λ₂) / (λ₁ + λ₂ + 1e-10)`
  - Return `(tx, ty, coherence)`
- [x] Implement `refine_tangent_field(tx, ty, sigma, iterations) -> tuple[np.ndarray, np.ndarray]`
  - For each iteration: Gaussian smooth tx and ty, re-normalize to unit length
  - Return `(tx, ty)`
- [x] Implement `compute_etf(image: np.ndarray, config: ETFConfig | None = None) -> ETFResult`
  - Convert to grayscale if BGR
  - Orchestrate: structure tensor → tangent field → refinement
  - Return `ETFResult` with tangent_x, tangent_y, coherence, gradient_magnitude

**Acceptance Criteria:** ✅ All met
- Tangent vectors are unit length everywhere: `sqrt(tx² + ty²) ≈ 1.0` (within 1e-6) ✅
- Coherence values in [0, 1] ✅
- Uniform gray image → coherence near 0 everywhere (no edges) ✅
- Image with a sharp vertical line → tangent vectors near vertical along the line, high coherence ✅

### 3.2 ETF tests ✅

- [x] Write `tests/test_etf.py`:
  - `test_unit_length`: output tangent magnitude ≈ 1.0 on real image
  - `test_coherence_range`: coherence in [0, 1]
  - `test_uniform_image_low_coherence`: synthetic uniform image → coherence near 0
  - `test_vertical_edge_tangent_direction`: synthetic image with vertical edge → tangent vectors approximately vertical (|ty| > |tx|) at the edge
  - `test_horizontal_edge_tangent_direction`: synthetic image with horizontal edge → tangent vectors approximately horizontal
  - `test_refinement_preserves_unit_length`: after N refinement iterations, vectors still unit length
  - `test_structure_tensor_shapes`: outputs match input shape
  - `test_default_config`: ETFConfig defaults produce reasonable results
  - `test_bgr_and_gray_input`: function handles both BGR and grayscale inputs

**Acceptance Criteria:** ✅ All met
- All tests pass ✅ (19 tests, all passing)
- Synthetic edge tests confirm direction correctness within reasonable tolerance ✅

---

## Phase 4: Flow Fields ✅ COMPLETE

**Purpose:** Implement contour gradient flow, ETF alignment, coherence-based blending, and the flow pipeline.

**Rationale:** This phase depends on the ETF from Phase 3 and the contour signed distance field from Stage 1. It combines them into the final blended flow field, completing Stage 3's core processing.

**Implementation Notes:**
- Successfully created `src/portrait_map_lab/flow_fields.py` with all 4 required functions
- Implemented contour flow computation using numpy gradient with 90° CCW rotation
- Added tangent field alignment using dot product to resolve 180° ambiguity
- Created coherence-based blend weight computation with power function
- Implemented linear blending with fallback mechanism for degenerate cases
- Added comprehensive pipeline integration in pipelines.py with logging
- Created 16 unit tests for flow_fields.py, all passing
- Added 6 integration tests to test_pipelines.py, all passing
- Fixed minor linting issues (newlines, unused import)
- Tested end-to-end with real portrait image, produces expected outputs

### 4.1 Implement `flow_fields.py` ✅

- [x] Create `src/portrait_map_lab/flow_fields.py` with `__all__`
- [x] Implement `compute_contour_flow(signed_distance: np.ndarray, smooth_sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]`
  - `grad_y, grad_x = np.gradient(signed_distance)`
  - Rotate 90° CCW: `flow_x = -grad_y`, `flow_y = grad_x`
  - Normalize to unit vectors
  - Optional Gaussian smoothing + re-normalize
  - Return `(flow_x, flow_y)`
- [x] Implement `align_tangent_field(tx, ty, ref_x, ref_y) -> tuple[np.ndarray, np.ndarray]`
  - Compute dot product: `dot = tx * ref_x + ty * ref_y`
  - Flip where `dot < 0`: `tx = where(dot < 0, -tx, tx)`, same for ty
  - Return aligned `(tx, ty)`
- [x] Implement `compute_blend_weight(coherence: np.ndarray, config: FlowConfig | None = None) -> np.ndarray`
  - `alpha = coherence ** config.coherence_power`
  - Clip to [0, 1]
  - Return alpha
- [x] Implement `blend_flow_fields(etf_tx, etf_ty, contour_fx, contour_fy, alpha, fallback_threshold) -> tuple[np.ndarray, np.ndarray]`
  - Linear blend: `bx = alpha * etf_tx + (1 - alpha) * contour_fx`
  - Pre-normalization magnitude check: where `sqrt(bx² + by²) < fallback_threshold`, use contour flow
  - Normalize to unit vectors
  - Return `(flow_x, flow_y)`

**Acceptance Criteria:** ✅ All met
- Contour flow vectors are unit length and perpendicular to the distance gradient ✅
- Alignment flips exactly where dot product is negative ✅
- Blend fallback triggers in degenerate regions ✅
- All output vectors are unit length ✅

### 4.2 Flow pipeline integration ✅

- [x] Add `run_flow_pipeline(image: np.ndarray, contour_result: ContourResult, config: FlowConfig | None = None) -> FlowResult` to `pipelines.py`
  - Compute ETF via `compute_etf`
  - Compute contour flow from `contour_result.signed_distance`
  - Align ETF to contour flow
  - Compute blend weight from coherence
  - Blend flow fields
  - Return `FlowResult` with all components
- [x] Add `save_flow_outputs(result: FlowResult, output_dir: Path, image: np.ndarray | None = None) -> None` to `pipelines.py`
  - Save coherence heatmap (viridis)
  - Save blend weight heatmap (viridis)
  - Save raw .npy arrays for flow_x, flow_y
  - Contact sheet (defer quiver and LIC overlays to Phase 5)

**Acceptance Criteria:** ✅ All met
- `run_flow_pipeline` accepts pre-computed contour result ✅
- FlowResult contains all expected fields ✅
- Raw .npy flow arrays saved for downstream use ✅

### 4.3 Flow fields tests ✅

- [x] Write `tests/test_flow_fields.py`:
  - `test_contour_flow_unit_length`: output is unit vectors on synthetic distance field
  - `test_contour_flow_perpendicular`: flow is perpendicular to distance gradient (dot product ≈ 0)
  - `test_alignment_flips_opposing_vectors`: synthetic test where half the field is opposing → those get flipped
  - `test_alignment_preserves_aligned_vectors`: already-aligned vectors unchanged
  - `test_blend_weight_range`: output in [0, 1]
  - `test_blend_high_coherence_prefers_etf`: alpha near 1.0 → result close to ETF
  - `test_blend_low_coherence_prefers_contour`: alpha near 0.0 → result close to contour flow
  - `test_blend_fallback_on_cancellation`: opposing vectors with alpha=0.5 → falls back to contour flow
  - `test_blended_unit_length`: final blended vectors are unit length
- [x] Extend `tests/test_pipelines.py` with flow pipeline integration test:
  - Returns correct type
  - All array shapes match image dimensions
  - Flow vectors unit length

**Acceptance Criteria:** ✅ All met
- All tests pass ✅ (22 tests total: 16 unit + 6 integration)
- Perpendicularity test passes within tolerance (|dot| < 0.05) ✅

---

## Phase 5: LIC Visualization & Viz Extensions ✅ COMPLETE

**Purpose:** Implement LIC rendering and add flow-field visualization helpers to viz.py.

**Rationale:** LIC depends on having a flow field (Phase 4). The quiver overlay also needs the flow field. Bundling these together means all visualization for the flow pipeline is complete in one phase, and the contact sheets from Phase 4 can be enhanced.

**Implementation Notes:**
- Successfully created `src/portrait_map_lab/lic.py` with Line Integral Convolution algorithm
- Implemented bidirectional streamline tracing (forward and backward)
- Added both bilinear and nearest-neighbor interpolation support
- Created comprehensive test suite with 15 test cases covering all scenarios
- Added visualize_flow_field and overlay_lic functions to viz.py
- Updated save_flow_outputs to generate quiver plots and LIC visualizations
- All 27 tests pass, no lint errors
- Code quality review rated implementation as "EXCELLENT" and production-ready

### 5.1 Implement `lic.py` ✅

- [x] Create `src/portrait_map_lab/lic.py` with `__all__`
- [x] Implement `compute_lic(flow_x: np.ndarray, flow_y: np.ndarray, config: LICConfig | None = None) -> np.ndarray`
  - Generate white noise with fixed seed
  - Initialize coordinate grids via `np.mgrid`
  - Forward + backward tracing loop (`length` steps each direction)
  - At each step: sample flow direction, advance coordinates, accumulate noise values
  - Use `scipy.ndimage.map_coordinates(order=1)` for bilinear sampling when `use_bilinear=True`, else nearest-neighbor via integer indexing
  - Clip coordinates to image bounds
  - Average accumulated values
  - Return float64 in [0, 1] (normalize)

**Acceptance Criteria:** ✅ All met
- Output shape matches input flow field shape ✅
- Output dtype is float64 ✅
- Values in [0, 1] ✅
- Deterministic: same inputs + same seed → same output ✅
- Uniform flow (all pointing right) → horizontal streaks visible in output ✅

### 5.2 Add flow visualization to `viz.py` ✅

- [x] Implement `visualize_flow_field(flow_x, flow_y, image=None, step=16, scale=10.0, color=(0,255,0)) -> np.ndarray`
  - Draw arrows on a copy of `image` (or blank canvas if None) at grid positions every `step` pixels
  - Use `cv2.arrowedLine` for each vector
  - Return BGR uint8
- [x] Implement `overlay_lic(lic_image: np.ndarray, image: np.ndarray, alpha: float = 0.5) -> np.ndarray`
  - Blend LIC texture (converted to BGR) with source image
  - `result = alpha * lic_bgr + (1 - alpha) * image`
  - Return BGR uint8
- [x] Update `save_flow_outputs` in `pipelines.py` to include quiver plot, LIC, and LIC overlay in the contact sheet

**Acceptance Criteria:** ✅ All met
- Quiver plot has arrows at expected grid spacing ✅
- LIC overlay blends cleanly without clipping artifacts ✅
- Flow contact sheet includes: ETF coherence, contour flow quiver, blend weight, blended flow quiver, LIC, LIC overlay ✅

### 5.3 LIC and visualization tests ✅

- [x] Write `tests/test_lic.py`:
  - `test_output_shape`: matches input shape
  - `test_output_range`: values in [0, 1]
  - `test_deterministic`: same inputs → same output
  - `test_uniform_flow_horizontal_streaks`: uniform rightward flow → horizontal variance low, vertical variance high (streaks are horizontal)
  - `test_bilinear_vs_nearest`: both modes produce valid output with similar statistics
  - Plus 10 additional comprehensive tests for edge cases
- [x] Test viz functions: quiver returns valid BGR image, overlay returns valid BGR image
  - Created `tests/test_viz_flow.py` with 12 tests for visualization functions

**Acceptance Criteria:** ✅ All met
- All tests pass ✅ (27 tests total: 15 LIC + 12 viz)
- Horizontal streak test passes (variance ratio confirms directional coherence) ✅

---

## Phase 6: Full Pipeline Integration & CLI ✅ COMPLETE

**Purpose:** Wire everything together into the composed pipeline and extend the CLI.

**Rationale:** All individual components are built and tested. This phase connects them into the end-to-end workflow the user interacts with.

**Implementation Notes:**
- Successfully created helper functions to extract post-landmark logic from existing pipelines
- Implemented `run_all_pipelines()` with efficient landmark sharing (detected once, used across all stages)
- Added comprehensive CLI subcommands for density and flow pipelines
- Extended the `all` subcommand to use the new unified pipeline
- All tests pass, no lint errors
- Tested successfully with real portrait images

### 6.1 Implement composed pipeline ✅

- [x] Add `run_all_pipelines(image: np.ndarray, feature_config: PipelineConfig | None = None, contour_config: ContourConfig | None = None, compose_config: ComposeConfig | None = None, flow_config: FlowConfig | None = None, lic_config: LICConfig | None = None) -> ComposedResult` to `pipelines.py`
  - Run feature pipeline → `PipelineResult`
  - Run contour pipeline (share landmarks) → `ContourResult`
  - Run density pipeline (pass feature + contour results) → `DensityResult`
  - Run flow pipeline (pass contour result) → `FlowResult`
  - Compute LIC from flow result → `lic_image`
  - Return `ComposedResult`
- [x] Add `save_all_outputs(result: ComposedResult, output_dir: Path, image: np.ndarray) -> None` to `pipelines.py`
  - Delegate to existing `save_pipeline_outputs`, `save_contour_outputs`, `save_density_outputs`, `save_flow_outputs`
  - Each writes to its own subdirectory

**Acceptance Criteria:** ✅ All met
- Landmark detection runs once, shared across all sub-pipelines ✅
- All four result types populated in ComposedResult ✅
- LIC image present and valid ✅
- Save creates `features/`, `contour/`, `density/`, `flow/` subdirectories ✅

### 6.2 Extend CLI ✅

- [x] Add `density` subcommand to `scripts/run_pipeline.py`:
  - Runs features + contour + density internally (sharing landmarks)
  - Arguments: shared args + `--clip-limit`, `--tile-size`, `--tonal-blend` (choices: multiply, screen, max, weighted), `--gamma`, `--feature-weight`, `--contour-weight`
  - Handler: build configs, run pipelines, save density outputs
- [x] Add `flow` subcommand:
  - Runs contour + flow + LIC internally
  - Arguments: shared args + `--structure-sigma`, `--refine-iterations`, `--refine-sigma`, `--coherence-power`, `--lic-length`
  - Handler: build configs, run pipelines, save flow outputs
- [x] Extend `all` subcommand handler:
  - After existing features + contour handling, also run density + flow + LIC
  - Add relevant arguments to the `all` subparser (density and flow args)
  - Use `run_all_pipelines` internally
  - Print expanded summary with density/flow verification items
- [x] Verify backward compatibility: `features`, `contour` subcommands unchanged

**Acceptance Criteria:** ✅ All met
- `python scripts/run_pipeline.py density <image>` produces density outputs ✅
- `python scripts/run_pipeline.py flow <image>` produces flow outputs ✅
- `python scripts/run_pipeline.py all <image>` produces all outputs (features + contour + density + flow) ✅
- Existing subcommands produce identical output to before this feature ✅

### 6.3 Update `__init__.py` exports ✅

- [x] Add all new public functions and types to `__init__.py` imports and `__all__`:
  - From `luminance`: `extract_luminance`, `apply_clahe`, `compute_tonal_target`
  - From `compose`: `compose_maps`, `build_density_target`
  - From `etf`: `compute_structure_tensor`, `extract_tangent_field`, `refine_tangent_field`, `compute_etf`
  - From `flow_fields`: `compute_contour_flow`, `align_tangent_field`, `compute_blend_weight`, `blend_flow_fields`
  - From `lic`: `compute_lic`
  - From `models`: all new config/result types
  - From `pipelines`: `run_density_pipeline`, `run_flow_pipeline`, `run_all_pipelines`
  - From `viz`: `visualize_flow_field`, `overlay_lic`

**Acceptance Criteria:** ✅ All met
- All new public symbols importable from `portrait_map_lab` directly ✅
- `__all__` is complete and sorted ✅

### 6.4 End-to-end integration tests ✅

- [x] Write integration test in `tests/test_pipelines.py`:
  - `test_run_all_pipelines`: full pipeline on test image → ComposedResult with all fields populated, correct shapes, valid ranges
  - `test_all_pipeline_shared_landmarks`: verify landmarks computed once (check that feature_result.landmarks and contour_result.landmarks reference equivalent data)
  - `test_save_all_outputs`: save to temp dir, verify all expected files exist in all four subdirectories
  - `test_all_pipeline_with_custom_configs`: test with custom configurations
- [x] Run full test suite and lint check (`ruff`)

**Acceptance Criteria:** ✅ All met
- All tests pass (existing + new) ✅
- No lint errors ✅
- Full pipeline completes on test image without errors ✅

---

## Phase 7: Documentation

**Purpose:** Document the new pipeline stages and update architecture docs.

**Rationale:** Documentation comes last because it references the final API and behavior. Writing it before the code stabilizes leads to stale docs.

### 7.1 Pipeline stage docs ✅ COMPLETE

**Implementation Notes:**
- Successfully created all 5 pipeline stage documentation files
- Each document follows the established pattern from existing stage docs
- All function signatures accurately match the implementation
- Added comprehensive parameter guidelines and visual characteristics
- Included mathematical background, algorithm details, and best practices

- [x] Create `docs/pipeline/07-luminance-target.md`: overview, CLAHE rationale, function signatures, parameters, output format, visualization
- [x] Create `docs/pipeline/08-density-composition.md`: blend modes with formulas, two-stage composition strategy, gamma correction, configuration
- [x] Create `docs/pipeline/09-edge-tangent-field.md`: structure tensor algorithm, closed-form eigenvectors, iterative refinement, coherence metric, parameter tuning guidance
- [x] Create `docs/pipeline/10-flow-fields.md`: contour gradient flow derivation, alignment step, coherence-based blending, fallback behavior, combined flow output
- [x] Create `docs/pipeline/11-lic-visualization.md`: LIC algorithm, vectorized implementation, parameters, interpretation guide

**Acceptance Criteria:** ✅ All met
- Each doc follows the existing stage doc pattern (Overview, Purpose, Implementation, Output Format, Configuration, Visualization, Next Stage) ✅
- Function signatures match actual implementation ✅

### 7.2 Update architecture and README ✅ COMPLETE

**Implementation Notes:**
- Successfully updated all three documentation files with comprehensive coverage of the new pipelines
- Added detailed data flow diagram showing all 4 pipelines (features, contour, density, flow)
- Updated Module Architecture table with all 5 new modules and their responsibilities
- Added all new data structures (configs and results) to the Data Structures section
- Created extension examples for both flow field and density composition systems
- Updated README with complete quick-start examples for all pipelines
- Added CLI documentation for the new `density` and `flow` subcommands
- Updated vision.md to mark Stages 1, 2, and 3 as complete with implementation details
- All documentation is internally consistent and accurately reflects the implementation

- [x] Update `docs/pipeline/architecture.md`:
  - Add density and flow modules to the Module Architecture table
  - Extend the data flow diagram to include Stage 2 and Stage 3
  - Add new result/config types to the Data Structures section
  - Update Extension Points with flow field extension examples
- [x] Update `docs/pipeline/README.md`:
  - Add density and flow to the quick-start examples
  - Update the pipeline stage list
  - Add `all` subcommand expanded description
- [x] Update `docs/vision.md`: mark Stage 2 and Stage 3 initial flow fields as complete; note contour-following flow + ETF blending as the implemented approach

**Acceptance Criteria:** ✅ All met
- Architecture doc data flow diagram matches the actual implementation ✅
- README quick-start examples work when copy-pasted ✅
- Vision doc accurately reflects current project state ✅

---

## Dependency Graph

```
Phase 1 (Models)
  1.1 → 1.2
         |
    ┌────┴────┐
    v         v
Phase 2    Phase 3
(Density)  (ETF)
 2.1─2.2─2.3  3.1─3.2
              |
              v
           Phase 4
           (Flow)
            4.1─4.2─4.3
              |
              v
           Phase 5
           (LIC + Viz)
            5.1─5.2─5.3
                  |
    ┌─────────────┘
    v
Phase 6 (Integration)
 6.1─6.2─6.3─6.4
         |
         v
Phase 7 (Docs)
 7.1─7.2
```

Note: Phases 2 and 3 can be developed in parallel after Phase 1 is complete. Phase 4 depends on Phase 3 (ETF) and uses contour data from Stage 1 (already built). Phase 5 depends on Phase 4. Phase 6 depends on all prior phases. Phase 7 depends on Phase 6.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Extend `all` instead of adding `composed` subcommand | `all` should always mean "run everything." Adding a separate `composed` creates confusion about which is the "real" full pipeline. As the project grows, `all` naturally encompasses new capabilities. |
| Multiplicative blend as default | Density = importance x tone ensures the particle only draws heavily where the subject is both dark AND structurally important. Other modes available for experimentation. |
| Separate `compose.py` from existing `combine.py` | `combine.py` does weighted linear sum for same-type maps (e.g., multiple influence maps). `compose.py` handles cross-domain composition with richer semantics (multiply, screen, max). Different concerns. |
| Gaussian ETF refinement over Kang-Lee bilateral | Simpler, faster, adequate for initial implementation. Bilateral refinement can be added as an optional mode later if visual quality is insufficient. |
| Coherence-based flow blending | Naturally identifies where ETF is reliable (strong edges) vs. unreliable (flat skin). No manual region definitions needed. |
| ETF alignment to contour flow | Resolves the 180° eigenvector ambiguity that would otherwise cause cancellation artifacts during blending. |
| Vectorized LIC over per-pixel | Performance critical for usable iteration speed. All pixels advance simultaneously in numpy. |
| Unit vector flow fields (no magnitude) | Particle speed should be controlled by density target, not flow field magnitude. Clean separation of concerns. |
| Pre-computed Stage 1 results passed to new pipelines | Avoids duplicate landmark detection (expensive network inference). `run_all_pipelines` shares results; individual subcommands re-run as needed. |
