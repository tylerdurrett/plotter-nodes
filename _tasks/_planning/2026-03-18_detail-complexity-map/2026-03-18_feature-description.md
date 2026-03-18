# Feature: Complexity Map

**Date:** 2026-03-18
**Status:** Scoped

---

## Overview

Add a signal-based complexity map that measures local image complexity — where the image contains visually rich structure — and exposes it as a standalone exported map and as a flow speed modulator. Rather than driving density (which answers "where should lines exist?"), complexity drives particle *behavior*: particles slow down in detail-rich areas, spending more time there and producing finer strokes that naturally capture more structure.

The existing importance maps answer "where are the semantically important features?" The complexity map answers a different question: "where does the image itself contain local complexity that should affect how lines behave?"

---

## End-User Capabilities

1. **Run a complexity pipeline** against an image and receive: per-pixel complexity map, normalized output, and heatmap visualizations — saved as PNG and raw `.npy`.
2. **Choose a complexity metric** from multiple options:
   - **Gradient energy** (default) — measures edge and transition density via Sobel gradient magnitude, smoothed over a configurable window. Best general-purpose metric for line-drawing relevance.
   - **Laplacian energy** — measures fine-grained high-frequency content via absolute Laplacian response. Good for texture detection.
   - **Multiscale gradient energy** — computes gradient energy at multiple spatial scales and combines them, capturing both fine texture and coarse structural detail in a single map.
3. **Control spatial scale** via configurable blur/window parameters, allowing the map to target fine texture (small sigma), medium structure (moderate sigma), or broad tonal complexity (large sigma).
4. **Apply an optional spatial mask** to restrict the complexity map to a region of interest (e.g., face/head segmentation mask), preventing background clutter from dominating.
5. **Get a flow speed scalar field** derived from the complexity map — high complexity regions produce low speed values, causing particles to slow down and linger in detailed areas. This is saved alongside existing `flow_x` and `flow_y` outputs and included in the export bundle.
6. **Export the complexity map and flow speed** as part of the binary export bundle for consumption by the companion TypeScript plotter.
7. **Run the complexity pipeline standalone** via CLI subcommand, or as part of the full `all` pipeline.

---

## Architecture / Scope

### New module: `complexity_map.py`

All complexity computation logic lives in a new module. Pure functions, same conventions as existing modules. Functions:

- `compute_gradient_energy(gray, sigma)` → float64 array
  Sobel gradient magnitude, Gaussian-smoothed, raw (unnormalized).
- `compute_laplacian_energy(gray, sigma)` → float64 array
  Absolute Laplacian response, Gaussian-smoothed, raw.
- `compute_multiscale_gradient_energy(gray, scales, weights)` → float64 array
  Weighted combination of gradient energy at multiple sigma values, raw.
- `normalize_map(raw, percentile)` → float64 array [0, 1]
  Percentile-based normalization.
- `apply_mask(map, mask)` → float64 array [0, 1]
  Element-wise masking.
- `compute_complexity_map(image, config, mask)` → `ComplexityResult`
  Main entry point. Dispatches to the configured metric, normalizes, applies optional mask.

### New module: `flow_speed.py`

Derives a speed scalar field from the complexity map:

- `compute_flow_speed(complexity, config)` → float64 array [0, 1]
  Converts normalized complexity into a speed field. High complexity → low speed. Configurable via `speed_min` (speed in most complex areas) and `speed_max` (speed in least complex areas).

This is intentionally a separate module from `complexity_map.py` because speed derivation is a distinct concern — complexity measures image structure, speed translates that into particle behavior.

### New models in `models.py`

- `ComplexityConfig` — metric name, sigma/scale params, normalize_percentile, output_dir
- `ComplexityResult` — raw_complexity, complexity (normalized [0, 1]), metric used
- `FlowSpeedConfig` — speed_min, speed_max
- Add `flow_speed` field to `FlowResult`

### Integration points

**Flow pipeline**: `run_flow_pipeline` gains an optional complexity result input. When provided, it computes a `flow_speed` field and includes it in `FlowResult`. When not provided, `flow_speed` defaults to a uniform 1.0 array (no speed modulation).

**Export bundle**: Two new maps added to `_MAP_DEFINITIONS`:
- `complexity` — normalized [0, 1] complexity map
- `flow_speed` — speed scalar [speed_min, 1.0]

**No changes to density pipeline**: Complexity does not feed into density composition. Density is driven by tonal values and semantic importance as before.

### New pipeline functions in `pipelines.py`

- `run_complexity_pipeline(image, config, mask)` → `ComplexityResult`
- `save_complexity_outputs(result, output_dir)` — saves intermediates following existing conventions

### CLI extension

New `complexity` subcommand in `scripts/run_pipeline.py` with metric-specific args. Also integrated into the `all` subcommand.

### Reuse (no changes needed to core logic)

- `etf.py` — related approach but the complexity module uses its own lighter-weight Sobel computation
- `storage.py`, `viz.py` — colorization, saving, contact sheets

### What changes

- `models.py` — add `ComplexityConfig`, `ComplexityResult`, `FlowSpeedConfig`; add `flow_speed` to `FlowResult`
- `pipelines.py` — add complexity pipeline functions; update `run_flow_pipeline` to accept complexity and compute speed; update `run_all_pipelines`
- `export.py` — add `complexity` and `flow_speed` to `_MAP_DEFINITIONS`
- `scripts/run_pipeline.py` — new `complexity` subcommand, `all` includes complexity
- `__init__.py` — export new public functions

---

## Technical Details

### Gradient energy (default metric)

1. Convert to grayscale float64 [0, 1]
2. Compute Sobel gradients Gx, Gy
3. Compute magnitude: `sqrt(Gx² + Gy²)`
4. Smooth with Gaussian (sigma controls spatial scale of the "complexity window")
5. Normalize to [0, 1] using percentile-based normalization (99th percentile → 1.0)

This is intentionally simpler than the ETF structure tensor. The ETF exists to compute *directional* edge flow; the complexity map only needs *scalar* complexity magnitude.

### Laplacian energy

1. Compute Laplacian (sum of second derivatives)
2. Take absolute value
3. Smooth with Gaussian
4. Normalize to [0, 1]

Captures fine texture and high-frequency content more aggressively than gradient energy.

### Multiscale gradient energy

1. Compute gradient energy at N scales (e.g., sigma = [1.0, 3.0, 8.0])
2. Weight and sum: `Σ(wᵢ × energy_at_scale_i)`
3. Normalize to [0, 1]

Captures both hair-strand-level texture and broader structural transitions in one map.

### Flow speed derivation

The speed field is a simple transform of the normalized complexity map:

```
speed = speed_max - complexity * (speed_max - speed_min)
```

Where:
- `complexity = 0.0` (smooth region) → `speed = speed_max` (full speed)
- `complexity = 1.0` (most complex) → `speed = speed_min` (slowest)

Default `speed_min = 0.3`, `speed_max = 1.0`. This means particles slow to 30% speed in the most complex areas. The plotter consumes this as a per-pixel multiplier on particle step size.

### Masking behavior

The `compute_complexity_map` function accepts an optional binary mask. When provided:
- The complexity map is computed on the full image (to avoid edge artifacts at mask boundaries)
- The final normalized output is multiplied element-wise by the mask
- Values outside the mask are zero (which translates to `speed_max` in flow speed — full speed outside the mask)

### Default parameters

| Parameter | Default | Rationale |
|---|---|---|
| `metric` | `"gradient"` | Best general-purpose for line-drawing relevance |
| `sigma` | `3.0` | Moderate smoothing — captures medium-scale structure |
| `scales` | `[1.0, 3.0, 8.0]` | Fine, medium, coarse (for multiscale only) |
| `scale_weights` | `[0.5, 0.3, 0.2]` | Favor fine detail (for multiscale only) |
| `normalize_percentile` | `99.0` | Robust normalization avoiding outlier spikes |
| `speed_min` | `0.3` | Particles slow to 30% in most complex areas |
| `speed_max` | `1.0` | Full speed in smooth areas |

### Outputs per run

```
output/<image_name>/complexity/
    gradient_energy.png              # Raw complexity heatmap
    gradient_energy_raw.npy          # Raw float64 values
    complexity.png                   # Final normalized [0,1] map
    complexity_raw.npy
    contact_sheet.png

output/<image_name>/flow/
    flow_speed.png                   # Speed scalar heatmap (NEW)
    flow_speed_raw.npy               # Raw float64 speed values (NEW)
    ... (existing flow outputs unchanged)
```

### Export bundle additions

```
export/
    complexity.bin                   # float32 [0, 1] — normalized complexity
    flow_speed.bin                   # float32 [speed_min, 1.0] — speed scalar
    ... (existing exports unchanged)
```

---

## Risks and Considerations

- **Complexity ≠ importance**: Highly complex regions (noisy backgrounds, textured clothing) are not inherently important. Mitigated by: (a) supporting spatial masking, (b) the speed field only affects particle behavior, not placement, and (c) masking to face/head in the `all` pipeline.
- **Normalization sensitivity**: Percentile-based normalization is image-dependent. The 99th percentile default handles outliers but may need tuning per use case. Exposed as a config parameter.
- **Speed range tuning**: The default `speed_min=0.3` is a guess. May need visual experimentation to find the right balance between "particles linger in detail" and "particles get stuck." The parameter is fully configurable.
- **Relationship to ETF gradient magnitude**: The ETF module already computes `gradient_magnitude`. The complexity module intentionally computes its own to keep modules decoupled and allow independent preprocessing control.

---

## Non-Goals / Future Iterations

- **Density integration** — complexity does not feed into density composition. Density is driven by tonal values and semantic importance. If this changes in the future, the complexity map is available as a composable array.
- **Quadtree subdivision visualization** — the original inspiration. Could consume the complexity map to produce an interpretable block rendering. Separate future feature.
- **Wavelet decomposition** — a strong theoretical approach to multiscale analysis. Could replace or augment the multiscale gradient metric later.
- **Structure tensor coherence as a metric** — already available via ETF. Could be exposed as an additional metric option.
- **Automatic mask generation** — this feature accepts masks but does not generate them.
- **Frequency-domain analysis (FFT/DCT)** — deferred to keep the first implementation simple and interpretable.
- **Adaptive speed curves** — the current speed derivation is linear. Non-linear curves (gamma, sigmoid) could be explored later.

---

## Success Criteria

1. `uv run python scripts/run_pipeline.py complexity <image>` produces all listed outputs for each metric.
2. All three metrics (gradient, laplacian, multiscale) produce valid [0, 1] complexity maps.
3. Optional mask correctly zeros out regions outside the mask without introducing edge artifacts.
4. Flow speed field is computed when complexity is available, with configurable speed range.
5. Export bundle includes `complexity.bin` and `flow_speed.bin` with correct metadata.
6. All module functions are independently callable with typed inputs/outputs.
7. Tests pass (`pytest`), linting passes (`ruff check`, `ruff format --check`).
8. Existing feature, contour, density, and flow pipelines remain unaffected when complexity is not configured.
