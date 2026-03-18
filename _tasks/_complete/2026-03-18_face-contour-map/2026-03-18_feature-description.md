# Feature: Face Contour Distance Map

**Date:** 2026-03-18
**Status:** Scoped

---

## Overview

Add a face contour distance pipeline that extracts the MediaPipe face oval contour (~36 landmarks around the jaw and forehead), computes a signed distance field from the contour line, and remaps it into a configurable influence map. This produces a **structural support map** that emphasizes the face boundary, complementing the existing eye/mouth feature maps that capture local identity-critical regions.

---

## End-User Capabilities

1. **Run a contour pipeline** against a portrait image and receive: contour overlay, contour and filled masks, signed distance field, directional distance, and remapped influence map — all saved as PNG visualizations and raw `.npy` arrays.
2. **Control falloff direction** — inward-only (default), outward-only, symmetric, or band mode — via a config field, without changing core logic.
3. **Inspect the signed distance field** which preserves full interior/exterior distance information regardless of the chosen remap direction, enabling future experimentation with particle spawning and flow fields.
4. **Call any processing step independently** — oval polygon extraction, contour mask, filled mask, signed distance, directional prep — as standalone typed functions.
5. **Use a scalable CLI** with subcommands (`features`, `contour`, `all`) that cleanly separates parameters per map type and extends naturally as more map generators are added.

---

## Architecture / Scope

### New module: `face_contour.py`

All contour-specific logic lives in a new module, keeping the filled-polygon region system in `face_regions.py` clean. Functions:

- `FACE_OVAL_INDICES` — 36 ordered landmark indices from MediaPipe FACEMESH_FACE_OVAL
- `get_face_oval_polygon(landmarks)` → Nx2 pixel coordinates
- `rasterize_contour_mask(polygon, shape, thickness)` → thin polyline mask via `cv2.polylines`
- `rasterize_filled_mask(polygon, shape)` → filled polygon mask for inside/outside classification
- `compute_signed_distance(contour_mask, filled_mask)` → signed distance field (negative inside, positive outside, zero on contour)
- `prepare_directional_distance(signed, mode, ...)` → unsigned distance field ready for existing `remap_influence`

The **signed distance field** is the core reusable intermediate — it preserves maximum information. The `prepare_directional_distance` function translates between signed distance and the unsigned distance that existing `remap_influence` expects, requiring no changes to `remap.py`.

### New models in `models.py`

- `ContourConfig` — direction mode, band_width, contour_thickness, remap config, output_dir
- `ContourResult` — landmarks, contour_polygon, contour_mask, filled_mask, signed_distance, directional_distance, influence_map

### New pipeline functions in `pipelines.py`

- `run_contour_pipeline(image, config)` → `ContourResult`
- `save_contour_outputs(result, image, output_dir)` — saves all intermediates following existing output conventions

### New viz function in `viz.py`

- `draw_contour(image, polygon)` — draws face oval polyline overlay on image copy

### CLI refactoring: `scripts/run_pipeline.py`

Refactored from flat argparse to subcommand architecture:

| Subcommand | Description |
|---|---|
| `features` | Existing eye/mouth pipeline (current behavior) |
| `contour` | New face contour pipeline |
| `all` | Runs both pipelines with defaults |

Shared remap args (`--curve`, `--sigma`, etc.) live on the parent parser. Map-specific args (e.g., `--direction`, `--eye-weight`) live on subcommand parsers. This scales cleanly — each new map type is a new subcommand with its own parameter namespace.

### Reuse (no changes needed)

- `landmarks.py` — `detect_landmarks()` shared entry point
- `distance_fields.py` — `compute_distance_field()` works with polyline masks as-is
- `remap.py` — `remap_influence()` operates on unsigned distance; directional logic handled upstream
- `combine.py` — available for future composition of contour + feature maps
- `storage.py`, `viz.py` (colorize_map, make_contact_sheet, draw_landmarks)

---

## Technical Details

### Face oval landmark indices

36 ordered points from MediaPipe FACEMESH_FACE_OVAL, starting at forehead (index 10), walking clockwise:

```
[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
```

These are hardcoded because MediaPipe Python does not reliably expose `face_mesh_connections` at runtime.

### Distance field algorithm

1. Draw face oval as closed polyline (thickness=1) → contour mask
2. Draw face oval as filled polygon → filled mask (inside/outside classification)
3. Compute EDT from contour mask via existing `compute_distance_field` → unsigned distance from contour line
4. Apply sign using filled mask: negative inside, positive outside, zero on contour
5. `prepare_directional_distance` selects which side(s) to preserve based on direction mode
6. Existing `remap_influence` converts to [0,1] influence using configured curve

### Direction modes

| Mode | Interior | Exterior | Use case |
|---|---|---|---|
| `inward` | distance from contour | clamped high (→ ~0 influence) | Default: structural face boundary emphasis |
| `outward` | clamped high (→ ~0 influence) | distance from contour | Exterior particle spawning |
| `both` | distance from contour | distance from contour | Symmetric falloff |
| `band` | clamped at band_width | clamped at band_width | Narrow contour region only |

### Outputs per run

```
output/<image_name>/
  contour_overlay.png              # Face oval polyline on original
  contour_mask.png                 # Thin polyline mask
  filled_mask.png                  # Filled face oval
  signed_distance_raw.npy          # Full signed distance field
  signed_distance_heatmap.png      # Diverging colormap (blue=inside, red=outside)
  directional_distance_raw.npy     # After direction mode applied
  directional_distance_heatmap.png
  contour_influence.png            # Final remapped influence map
  contact_sheet.png                # Grid of all above
```

---

## Risks and Considerations

- **Face oval index ordering**: Indices must trace the contour sequentially for `cv2.polylines` to draw correctly. The ordered list is derived from walking the FACEMESH_FACE_OVAL connection graph and should be verified visually on a real image.
- **Polyline/fillPoly boundary mismatch**: Pixels on the contour may be classified differently by `cv2.polylines` vs `cv2.fillPoly` due to rasterization rules. The signed distance function must ensure contour-mask pixels always get distance=0 regardless of filled-mask classification.
- **Signed distance heatmap normalization**: Needs a diverging colormap centered at zero. Use symmetric range `[-max_abs, +max_abs]` mapped to [0, 1] with 0.5 at the zero-crossing.

---

## Non-Goals / Future Iterations

- Head silhouette or hair boundary (different map family)
- Subject segmentation
- Mesh-aware or geodesic contour distance
- SVG or path generation
- Composition with eye/mouth maps (combine.py is ready but composition weights/strategy are future work)
- Batch processing

---

## Success Criteria

1. `uv run python scripts/run_pipeline.py contour <image>` produces all listed outputs.
2. The signed distance field has negative values inside the face oval and positive outside.
3. Default `inward` mode produces influence that is strongest at the face boundary and falls off toward the center.
4. All four direction modes produce valid [0,1] influence maps.
5. All contour module functions are independently callable with typed inputs/outputs.
6. Tests pass (`pytest`), linting passes (`ruff check`, `ruff format --check`).
7. Existing `features` pipeline and tests are unaffected.

---

## Open Questions

1. **Default remap parameters for contour**: The face oval is larger than eye/mouth regions, so a wider falloff (larger sigma/radius) might be more appropriate. Can tune during implementation.
2. **Contact sheet contents**: Should include original image and landmarks overlay alongside contour-specific outputs for context.
