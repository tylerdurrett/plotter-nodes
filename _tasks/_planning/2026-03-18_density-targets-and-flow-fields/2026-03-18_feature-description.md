# Feature: Density Target Composition & Flow Fields

**Date:** 2026-03-18
**Status:** Scoped

## Overview

Compose existing Stage 1 importance maps (feature distance, face contour) with a new luminance-derived tonal target into a unified density target map, then generate direction flow fields by blending an Edge Tangent Field with contour-following flow. These two outputs — a scalar density target and a vector flow field — are the primary inputs a continuous-line drawing particle will eventually consume: "how much to draw here" and "which direction to prefer."

## End-User Capabilities

1. Run a single CLI command (`composed`) on a portrait image and receive a **density target map** that encodes where line activity should concentrate, informed by facial feature importance, face contour structure, and image tonality.
2. Run the same command and receive a **blended flow field** (as .npy arrays) that encodes preferred line travel direction at every pixel, combining image edge structure with face-contour-following behavior.
3. Inspect a **LIC (Line Integral Convolution) visualization** of the flow field to verify directional coherence — streaky texture that reveals how a particle would travel.
4. Configure the density composition blend mode (multiplicative, screen, max, weighted), CLAHE parameters, ETF smoothing, and flow blend behavior via CLI flags or Python API.
5. Inspect all intermediate outputs (luminance, CLAHE result, tonal target, importance blend, ETF coherence, contour flow, blend weights) as individual heatmaps and in a contact sheet.

## Architecture / Scope

### New modules

| Module | Responsibility |
|---|---|
| `luminance.py` | Grayscale extraction, CLAHE equalization, tonal target (inverted CLAHE) |
| `compose.py` | Multi-mode map composition: multiply, screen, max, weighted sum |
| `etf.py` | Edge Tangent Field via structure tensor + iterative refinement |
| `flow_fields.py` | Contour gradient flow, ETF alignment, coherence-based flow blending |
| `lic.py` | Line Integral Convolution visualization |

### Modified modules

| Module | Changes |
|---|---|
| `models.py` | New config/result dataclasses for luminance, density, ETF, flow, LIC, composed |
| `pipelines.py` | `run_density_pipeline`, `run_flow_pipeline`, `run_composed_pipeline` + save helpers |
| `viz.py` | Flow field quiver overlay, LIC-on-image overlay |
| `scripts/run_pipeline.py` | `density`, `flow`, `composed` subcommands |
| `__init__.py` | Export new public API |

### Data flow

```
Source Image
  ├── [Feature Pipeline]  → feature importance (0-1)      ┐
  ├── [Contour Pipeline]  → contour importance (0-1)       ├─ Stage 1 (exists)
  │                        → signed distance field          ┘
  │
  ├── [Luminance]  → grayscale → CLAHE → invert            ─ tonal target (0-1)
  │
  ├── [Density Composition]                                 ─ Stage 2
  │     importance = weighted_sum(feature, contour)
  │     density_target = importance × tonal_target           (default: multiplicative)
  │
  ├── [ETF]  → structure tensor → eigenvectors → refine    ─ Stage 3
  │            → tangent field (H×W×2 unit vectors) + coherence
  │
  ├── [Contour Flow]  → ∇(signed_distance) → rotate 90°   ─ Stage 3
  │                     → contour flow (H×W×2 unit vectors)
  │
  ├── [Flow Blending]                                       ─ Stage 3
  │     align ETF to contour flow (resolve 180° ambiguity)
  │     alpha = coherence^power  (ETF reliable where edges are strong)
  │     flow = alpha·ETF + (1-alpha)·contour_flow → normalize
  │
  └── [LIC Visualization]  → convolve noise along flow     ─ Validation
```

### Pipeline design

`run_density_pipeline` and `run_flow_pipeline` accept pre-computed Stage 1 results to avoid duplicate landmark detection. `run_composed_pipeline` is the convenience all-in-one that shares intermediate results internally.

## Technical Details

### Luminance target

- Convert to grayscale via `cv2.cvtColor`
- Apply CLAHE (`cv2.createCLAHE`) with configurable `clip_limit` (default 2.0) and `tile_size` (default 8). CLAHE over global histogram equalization because portraits often have bimodal tonal distributions (face vs. background) where global equalization washes out detail.
- Invert: `tonal_target = 1.0 - clahe / 255.0`. Dark regions → high density. Output float64 in [0, 1].

### Density composition

Two-stage composition:
1. **Importance**: `weighted_sum(feature_importance, contour_importance)` with configurable weights (default 0.6 / 0.4)
2. **Density target**: `compose(importance, tonal_target, mode)` with configurable blend mode

Default blend mode is **multiplicative** because `density = importance × tone` means: "draw densely only where the subject is both dark AND structurally important." A bright forehead near the eyes gets moderate density rather than maximum. This gives the particle system the right signal for controlled mark-making.

Additional blend modes (screen, max, weighted sum) for experimentation. Post-composition gamma correction for overall density tuning.

### Edge Tangent Field (ETF)

1. Mild Gaussian blur (σ ~1.5) to suppress noise
2. Sobel gradients → structure tensor components (Jxx, Jxy, Jyy)
3. Smooth structure tensor (σ ~5.0) for spatial coherence
4. Closed-form 2×2 eigenvector extraction (vectorized, no per-pixel loops)
5. Minor eigenvector = edge tangent direction
6. Iterative Gaussian smoothing + re-normalization (default 2 passes, σ ~3.0)
7. Coherence metric: `(λ₁ - λ₂) / (λ₁ + λ₂)` — high near edges, low in flat regions

Output: unit vector field (tx, ty) + coherence map. The coherence map drives the flow blending.

### Contour gradient flow

- `np.gradient(signed_distance)` → gradient vectors perpendicular to contour
- Rotate 90° CCW: `(flow_x, flow_y) = (-grad_y, grad_x)` → contour-following direction
- Normalize to unit length, optional light smoothing

This produces concentric flow lines wrapping the face shape — the "line flowing around form" behavior from the vision.

### Flow blending

- **Alignment**: Flip ETF vectors where `dot(ETF, contour_flow) < 0` to resolve the 180° eigenvector ambiguity. Essential to prevent cancellation during blending.
- **Weight**: `alpha = coherence^power` (default power=2.0). ETF dominates near strong edges (eyes, lips, jawline); contour flow dominates in flat skin regions (cheeks, forehead).
- **Blend**: Linear interpolation of vector components → re-normalize. Fallback to pure contour flow where blended magnitude is below threshold.

### LIC visualization

Vectorized batch tracing: all pixels advance simultaneously through the flow field, sampling a white noise texture. Forward + backward tracing for `length` steps (default 30). Uses `scipy.ndimage.map_coordinates` for bilinear interpolation. Fixed random seed for reproducibility.

Output: grayscale float64 texture where coherent streaks reveal flow structure.

### Output structure

```
output/<image>/
  density/
    luminance.png, clahe_luminance.png, tonal_target.png
    importance.png, density_target.png, density_target_raw.npy
    contact_sheet.png
  flow/
    etf_coherence.png, etf_quiver.png
    contour_flow_quiver.png, blend_weight.png
    flow_lic.png, flow_lic_overlay.png, flow_quiver.png
    flow_x_raw.npy, flow_y_raw.npy
    contact_sheet.png
```

## Risks and Considerations

- **ETF in flat regions**: Structure tensor eigenvectors are noisy where gradient magnitude is near zero (uniform skin). The coherence-based fallback to contour flow mitigates this.
- **LIC performance**: Vectorized LIC on large images (e.g. 4K) could be slow due to memory for the full-resolution coordinate arrays. May need a downscale option for interactive use.
- **Composition tuning**: The "right" blend mode and weights are subjective and image-dependent. Providing sensible defaults with easy overrides is the mitigation.
- **ETF sign ambiguity**: The eigenvector alignment step is critical. Without it, blending ETF and contour flow produces artifacts at transition boundaries.

## Non-Goals / Future Iterations

- **Particle simulation or line generation** — that is Stage 5, not this feature
- **Depth/normal/3D-derived flow** — Stage 6; this feature works in 2D image space only
- **Export to TypeScript plotter** — Stage 7; outputs are .npy arrays and PNGs for research
- **Kang-Lee bilateral ETF refinement** — the simpler Gaussian smoothing refinement is sufficient initially; the full bilateral approach can be added later if needed
- **Non-portrait inputs** — this feature assumes a single detected face
- **Streamline visualization** — LIC is the primary visualization; streamlines can be added later
- **Real-time or interactive parameter tuning** — batch CLI workflow only

## Success Criteria

1. `python scripts/run_pipeline.py composed <image>` produces density target and flow field outputs in the expected directory structure
2. Density target visually concentrates high values in dark, structurally important regions (dark eye sockets, nostrils, lips) and low values in bright, unimportant regions (background, highlights)
3. LIC visualization shows coherent streaks that follow image edges near features and follow the face contour in flat regions
4. Flow field vectors are unit length everywhere (within float precision)
5. All intermediate outputs are inspectable as heatmaps and in contact sheets
6. All new modules have unit tests covering edge cases (zero gradients, uniform images, value ranges)
7. Full pipeline runs without re-computing landmarks (shared across sub-pipelines)
8. Existing `features`, `contour`, and `all` subcommands continue to work unchanged

## Open Questions

1. Should the `all` subcommand be updated to include the new density/flow pipelines, or should `composed` be the only entry point that runs everything?
