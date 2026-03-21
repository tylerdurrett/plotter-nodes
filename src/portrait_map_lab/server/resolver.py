"""Pipeline dependency resolver for granular map selection.

Maps requested map keys to the minimal set of pipelines required to produce
them, then orchestrates pipeline execution in dependency order with shared
landmarks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from portrait_map_lab.models import (
    ComplexityConfig,
    ComposeConfig,
    ContourConfig,
    FlowConfig,
    FlowSpeedConfig,
    LandmarkResult,
    PipelineConfig,
)
from portrait_map_lab.pipelines import (
    run_complexity_pipeline,
    run_contour_pipeline_with_landmarks,
    run_density_pipeline,
    run_feature_pipeline_with_landmarks,
    run_flow_pipeline,
)
from portrait_map_lab.server.schemas import VALID_MAP_KEYS

__all__ = ["resolve_pipelines", "run_resolved_pipelines"]

logger = logging.getLogger(__name__)

# Inter-pipeline dependencies: pipeline → set of pipelines it requires.
_PIPELINE_DEPS: dict[str, frozenset[str]] = {
    "features": frozenset(),
    "contour": frozenset(),
    "density": frozenset({"features", "contour"}),
    "complexity": frozenset(),
    "flow": frozenset({"contour"}),
}

# All pipeline names recognised by the resolver.
ALL_PIPELINES: frozenset[str] = frozenset(
    {"features", "contour", "density", "complexity", "flow"}
)

# Map key → set of pipelines required to produce it.
_MAP_DEPENDENCIES: dict[str, frozenset[str]] = {
    "density_target": frozenset({"features", "contour", "density"}),
    "importance": frozenset({"features", "contour", "density"}),
    "flow_x": frozenset({"contour", "flow"}),
    "flow_y": frozenset({"contour", "flow"}),
    "coherence": frozenset({"contour", "flow"}),
    "complexity": frozenset({"complexity"}),
    "flow_speed": frozenset({"complexity", "contour", "flow"}),
}


assert set(_MAP_DEPENDENCIES.keys()) == VALID_MAP_KEYS, (
    f"_MAP_DEPENDENCIES keys {sorted(_MAP_DEPENDENCIES.keys())} "
    f"do not match VALID_MAP_KEYS {sorted(VALID_MAP_KEYS)}"
)


def resolve_pipelines(requested_maps: list[str]) -> set[str]:
    """Determine which pipelines are needed for *requested_maps*.

    Parameters
    ----------
    requested_maps
        Map keys the caller wants produced.  An empty list means "all maps",
        which resolves to all pipelines.

    Returns
    -------
    set[str]
        Pipeline names that must be executed.

    Raises
    ------
    ValueError
        If any key in *requested_maps* is not a recognised map key.
    """
    if not requested_maps:
        return set(ALL_PIPELINES)

    invalid = set(requested_maps) - VALID_MAP_KEYS
    if invalid:
        raise ValueError(
            f"Invalid map key(s): {sorted(invalid)}. "
            f"Valid keys: {sorted(VALID_MAP_KEYS)}"
        )

    pipelines: set[str] = set()
    for key in requested_maps:
        pipelines |= _MAP_DEPENDENCIES[key]
    return pipelines


def run_resolved_pipelines(
    image: np.ndarray,
    landmarks: LandmarkResult,
    pipelines: set[str],
    *,
    feature_config: PipelineConfig | None = None,
    contour_config: ContourConfig | None = None,
    compose_config: ComposeConfig | None = None,
    flow_config: FlowConfig | None = None,
    complexity_config: ComplexityConfig | None = None,
    speed_config: FlowSpeedConfig | None = None,
) -> dict[str, Any]:
    """Run only the pipelines in *pipelines*, collecting result objects.

    Landmarks are expected to be pre-computed and shared across all stages.
    Pipeline functions are called in dependency order so that downstream
    stages can consume earlier results.

    Parameters
    ----------
    image
        BGR uint8 image array.
    landmarks
        Pre-computed facial landmarks shared across pipelines.
    pipelines
        Set of pipeline names to execute (output of :func:`resolve_pipelines`).
    feature_config, contour_config, compose_config, flow_config,
    complexity_config, speed_config
        Optional per-pipeline configuration overrides.

    Returns
    -------
    dict[str, Any]
        Pipeline name → result object for each executed pipeline.  Callers
        can extract individual map arrays from these result objects.
    """
    # Validate dependency closure — catch misuse early with a clear message.
    for name in pipelines:
        missing = _PIPELINE_DEPS.get(name, frozenset()) - pipelines
        if missing:
            raise ValueError(
                f"Pipeline '{name}' requires {sorted(missing)}, "
                f"but they are not in the requested set {sorted(pipelines)}. "
                f"Use resolve_pipelines() to build a valid set."
            )

    results: dict[str, Any] = {}

    # --- execution order follows the dependency chain ---

    if "features" in pipelines:
        logger.info("Resolver: running feature pipeline")
        results["features"] = run_feature_pipeline_with_landmarks(
            landmarks, feature_config
        )

    if "contour" in pipelines:
        logger.info("Resolver: running contour pipeline")
        results["contour"] = run_contour_pipeline_with_landmarks(
            landmarks, contour_config, image=image
        )

    if "density" in pipelines:
        logger.info("Resolver: running density pipeline")
        results["density"] = run_density_pipeline(
            image,
            results["features"],
            results["contour"],
            compose_config,
        )

    if "complexity" in pipelines:
        logger.info("Resolver: running complexity pipeline")
        results["complexity"] = run_complexity_pipeline(
            image, complexity_config
        )

    if "flow" in pipelines:
        logger.info("Resolver: running flow pipeline")
        results["flow"] = run_flow_pipeline(
            image,
            results["contour"],
            flow_config,
            results.get("complexity"),
            speed_config,
        )

    return results
