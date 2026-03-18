---
name: project-portrait-map-lab
description: Core project context — Portrait Map Lab for portrait analysis maps feeding a pen-plotter line drawing system
type: project
---

Portrait Map Lab is a Python package for generating portrait analysis maps (distance fields, influence maps) that will later feed an algorithmic continuous-line drawing system for SVG/pen-plotter output.

**Key architectural decisions (2026-03-17):**
- Standalone Python repo first, ComfyUI wrapping later
- MediaPipe for face landmarks (direct Python API, not via ComfyUI nodes)
- 2D image-space Euclidean distance fields first (not geodesic)
- Region masks, not just center points
- Preserve raw + remapped outputs
- Each pipeline step independently callable (composable, not monolithic)
- uv as package manager, pyproject.toml, pytest, ruff
- dataclasses for structured data (not pydantic — may add pydantic at API boundary later)
- numpy ndarray as canonical image type
- Config dataclasses with sensible defaults for tunable parameters
- `models.py` not `types.py`, `storage.py` not `io.py` (avoid stdlib shadowing)

**Why:** Building a research lab for upstream maps, not a one-off script. Architecture must support future ComfyUI nodes, potential API, and additional map types.

**How to apply:** All code should be modular, typed, UI-agnostic. Each function = potential ComfyUI node = potential API endpoint.
