"""Preview PNG generation for the API server.

Bridges the per-pipeline save functions into the API session cache,
generating preview images from pipeline results and building metadata
for the API response.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from portrait_map_lab.models import ComposedResult
from portrait_map_lab.pipelines import (
    save_complexity_outputs,
    save_contour_outputs,
    save_density_outputs,
    save_flow_outputs,
    save_pipeline_outputs,
)

from .schemas import PreviewInfo

logger = logging.getLogger(__name__)


def generate_previews_full(
    result: ComposedResult,
    image: np.ndarray,
    previews_dir: Path,
) -> list[PreviewInfo]:
    """Generate all preview PNGs from a full ``ComposedResult``.

    Used by the full pipeline path (no ``maps`` filter).
    """
    previews_dir.mkdir(parents=True, exist_ok=True)

    # features and contour write directly into the given dir
    save_pipeline_outputs(result.feature_result, image, previews_dir / "features")
    save_contour_outputs(result.contour_result, image, previews_dir / "contour")

    # density, complexity, flow create their own subdirectory under the given dir
    save_density_outputs(result.density_result, previews_dir, image)

    if result.complexity_result is not None:
        save_complexity_outputs(result.complexity_result, previews_dir, image)

    save_flow_outputs(result.flow_result, previews_dir, image)

    return discover_previews(previews_dir)


def generate_previews_resolved(
    resolved_results: dict[str, Any],
    image: np.ndarray,
    previews_dir: Path,
) -> list[PreviewInfo]:
    """Generate preview PNGs only for pipelines that actually ran.

    Used by the granular/resolved pipeline path.
    """
    previews_dir.mkdir(parents=True, exist_ok=True)

    if "features" in resolved_results:
        save_pipeline_outputs(resolved_results["features"], image, previews_dir / "features")

    if "contour" in resolved_results:
        save_contour_outputs(resolved_results["contour"], image, previews_dir / "contour")

    if "density" in resolved_results:
        save_density_outputs(resolved_results["density"], previews_dir, image)

    if "complexity" in resolved_results:
        save_complexity_outputs(resolved_results["complexity"], previews_dir, image)

    if "flow" in resolved_results:
        save_flow_outputs(resolved_results["flow"], previews_dir, image)

    return discover_previews(previews_dir)


def discover_previews(previews_dir: Path) -> list[PreviewInfo]:
    """Scan a previews directory and build ``PreviewInfo`` metadata.

    Walks the directory looking for ``.png`` files, excluding contact
    sheet composites which are large and not useful as individual
    previews.  Returns URLs relative to the session directory (i.e.
    prefixed with ``previews/``).
    """
    result: list[PreviewInfo] = []

    if not previews_dir.is_dir():
        return result

    for png_path in sorted(previews_dir.rglob("*.png")):
        if "contact_sheet" in png_path.name:
            continue

        relative = png_path.relative_to(previews_dir)
        category = relative.parts[0] if len(relative.parts) > 1 else "other"
        name = png_path.stem
        url = f"previews/{relative}"

        result.append(PreviewInfo(category=category, name=name, url=url))

    return result
