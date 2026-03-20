"""Tests for the pipeline dependency resolver (Phase 3.2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from portrait_map_lab.models import LandmarkResult
from portrait_map_lab.server.resolver import (
    _MAP_DEPENDENCIES,
    ALL_PIPELINES,
    resolve_pipelines,
    run_resolved_pipelines,
)

# ---------------------------------------------------------------------------
# resolve_pipelines tests
# ---------------------------------------------------------------------------


class TestResolvePipelines:
    """Verify that map keys resolve to the correct pipeline sets."""

    # -- single map requests ------------------------------------------------

    def test_density_target(self) -> None:
        assert resolve_pipelines(["density_target"]) == {"features", "contour", "density"}

    def test_importance(self) -> None:
        assert resolve_pipelines(["importance"]) == {"features", "contour", "density"}

    def test_flow_x(self) -> None:
        assert resolve_pipelines(["flow_x"]) == {"contour", "flow"}

    def test_flow_y(self) -> None:
        assert resolve_pipelines(["flow_y"]) == {"contour", "flow"}

    def test_coherence(self) -> None:
        assert resolve_pipelines(["coherence"]) == {"contour", "flow"}

    def test_complexity(self) -> None:
        assert resolve_pipelines(["complexity"]) == {"complexity"}

    def test_flow_speed(self) -> None:
        assert resolve_pipelines(["flow_speed"]) == {"complexity", "contour", "flow"}

    # -- combined requests --------------------------------------------------

    def test_density_and_flow_x_union(self) -> None:
        """Requesting density_target + flow_x should union both pipeline sets."""
        result = resolve_pipelines(["density_target", "flow_x"])
        assert result == {"features", "contour", "density", "flow"}

    def test_density_and_flow_speed_union(self) -> None:
        """Requesting density_target + flow_speed should include all needed pipelines."""
        result = resolve_pipelines(["density_target", "flow_speed"])
        assert result == {"features", "contour", "density", "complexity", "flow"}

    def test_all_maps_covers_all_pipelines(self) -> None:
        """Requesting every map key should resolve to all pipelines."""
        all_keys = list(_MAP_DEPENDENCIES.keys())
        result = resolve_pipelines(all_keys)
        assert result == set(ALL_PIPELINES)

    def test_duplicate_keys_handled(self) -> None:
        """Duplicate map keys should not cause errors."""
        result = resolve_pipelines(["flow_x", "flow_x", "flow_y"])
        assert result == {"contour", "flow"}

    # -- empty list ---------------------------------------------------------

    def test_empty_list_resolves_to_all(self) -> None:
        """An empty list should resolve to all pipelines."""
        assert resolve_pipelines([]) == set(ALL_PIPELINES)

    # -- invalid keys -------------------------------------------------------

    def test_invalid_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="tonal_target"):
            resolve_pipelines(["tonal_target"])

    def test_mixed_valid_and_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="bogus"):
            resolve_pipelines(["density_target", "bogus"])

    # -- dependency table consistency ---------------------------------------

    def test_all_map_keys_have_dependencies(self) -> None:
        """Every valid map key must appear in the dependency table."""
        from portrait_map_lab.server.schemas import VALID_MAP_KEYS

        assert set(_MAP_DEPENDENCIES.keys()) == VALID_MAP_KEYS

    def test_all_dependency_values_are_valid_pipelines(self) -> None:
        """Every pipeline name in the dependency table must be a known pipeline."""
        for key, deps in _MAP_DEPENDENCIES.items():
            unknown = deps - ALL_PIPELINES
            assert deps <= ALL_PIPELINES, f"{key} references unknown pipeline(s): {unknown}"


# ---------------------------------------------------------------------------
# run_resolved_pipelines tests
# ---------------------------------------------------------------------------


def _fake_landmarks() -> LandmarkResult:
    """Build a minimal ``LandmarkResult`` for test mocks."""
    return LandmarkResult(
        landmarks=np.zeros((478, 2), dtype=np.float32),
        image_shape=(100, 100),
        confidence=0.99,
    )


class TestDependencyClosureValidation:
    """Verify that run_resolved_pipelines rejects invalid pipeline sets."""

    def test_density_without_features_raises(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="density.*features"):
            run_resolved_pipelines(image, _fake_landmarks(), {"density"})

    def test_flow_without_contour_raises(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="flow.*contour"):
            run_resolved_pipelines(image, _fake_landmarks(), {"flow"})

    def test_valid_set_from_resolver_accepted(self) -> None:
        """Sets produced by resolve_pipelines should always pass validation."""
        for key in _MAP_DEPENDENCIES:
            pipelines = resolve_pipelines([key])
            # Should not raise — just verify no ValueError
            # (actual pipeline calls will fail without real data, but
            # the dependency check runs first)
            try:
                run_resolved_pipelines(
                    np.zeros((10, 10, 3), dtype=np.uint8),
                    _fake_landmarks(),
                    pipelines,
                )
            except ValueError:
                pytest.fail(
                    f"resolve_pipelines(['{key}']) produced an "
                    f"invalid set: {pipelines}"
                )
            except (KeyError, Exception):
                # Pipeline execution errors are expected (mock-free),
                # but dependency validation should have passed.
                pass


_PATCH_PREFIX = "portrait_map_lab.server.resolver."
_PATCH_FEATURE = _PATCH_PREFIX + "run_feature_pipeline_with_landmarks"
_PATCH_CONTOUR = _PATCH_PREFIX + "run_contour_pipeline_with_landmarks"
_PATCH_DENSITY = _PATCH_PREFIX + "run_density_pipeline"
_PATCH_COMPLEXITY = _PATCH_PREFIX + "run_complexity_pipeline"
_PATCH_FLOW = _PATCH_PREFIX + "run_flow_pipeline"


class TestRunResolvedPipelines:
    """Verify that ``run_resolved_pipelines`` calls the correct pipeline functions."""

    def _make_mocks(self):
        """Return a dict of patches for all pipeline functions."""
        return {
            "features": patch(_PATCH_FEATURE, return_value=MagicMock(name="PipelineResult")),
            "contour": patch(_PATCH_CONTOUR, return_value=MagicMock(name="ContourResult")),
            "density": patch(_PATCH_DENSITY, return_value=MagicMock(name="DensityResult")),
            "complexity": patch(
                _PATCH_COMPLEXITY, return_value=MagicMock(name="ComplexityResult"),
            ),
            "flow": patch(_PATCH_FLOW, return_value=MagicMock(name="FlowResult")),
        }

    def test_only_complexity_skips_others(self) -> None:
        """Resolving only complexity should call run_complexity_pipeline and nothing else."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = _fake_landmarks()
        mocks = self._make_mocks()

        with mocks["features"] as m_feat, mocks["contour"] as m_cont, \
             mocks["density"] as m_dens, mocks["complexity"] as m_comp, \
             mocks["flow"] as m_flow:
            results = run_resolved_pipelines(
                image, landmarks, {"complexity"}
            )

        m_comp.assert_called_once()
        m_feat.assert_not_called()
        m_cont.assert_not_called()
        m_dens.assert_not_called()
        m_flow.assert_not_called()
        assert "complexity" in results
        assert "features" not in results

    def test_flow_x_runs_contour_and_flow(self) -> None:
        """Resolving flow_x pipelines should run contour + flow only."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = _fake_landmarks()
        mocks = self._make_mocks()

        with mocks["features"] as m_feat, mocks["contour"] as m_cont, \
             mocks["density"] as m_dens, mocks["complexity"] as m_comp, \
             mocks["flow"] as m_flow:
            results = run_resolved_pipelines(
                image, landmarks, {"contour", "flow"}
            )

        m_cont.assert_called_once()
        m_flow.assert_called_once()
        m_feat.assert_not_called()
        m_dens.assert_not_called()
        m_comp.assert_not_called()
        assert "contour" in results
        assert "flow" in results

    def test_density_runs_features_contour_density(self) -> None:
        """Resolving density_target pipelines should run features + contour + density."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = _fake_landmarks()
        mocks = self._make_mocks()

        with mocks["features"] as m_feat, mocks["contour"] as m_cont, \
             mocks["density"] as m_dens, mocks["complexity"] as m_comp, \
             mocks["flow"] as m_flow:
            results = run_resolved_pipelines(
                image, landmarks, {"features", "contour", "density"}
            )

        m_feat.assert_called_once()
        m_cont.assert_called_once()
        m_dens.assert_called_once()
        m_comp.assert_not_called()
        m_flow.assert_not_called()
        assert set(results.keys()) == {"features", "contour", "density"}

    def test_all_pipelines_runs_everything(self) -> None:
        """All pipelines should all be called."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = _fake_landmarks()
        mocks = self._make_mocks()

        with mocks["features"] as m_feat, mocks["contour"] as m_cont, \
             mocks["density"] as m_dens, mocks["complexity"] as m_comp, \
             mocks["flow"] as m_flow:
            results = run_resolved_pipelines(
                image, landmarks, set(ALL_PIPELINES)
            )

        m_feat.assert_called_once()
        m_cont.assert_called_once()
        m_dens.assert_called_once()
        m_comp.assert_called_once()
        m_flow.assert_called_once()
        assert set(results.keys()) == set(ALL_PIPELINES)

    def test_flow_receives_complexity_when_present(self) -> None:
        """When complexity is in the pipeline set, its result should be passed to flow."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = _fake_landmarks()
        mocks = self._make_mocks()

        with mocks["features"] as _m_feat, mocks["contour"] as _m_cont, \
             mocks["density"] as _m_dens, mocks["complexity"] as m_comp, \
             mocks["flow"] as m_flow:
            run_resolved_pipelines(
                image, landmarks, {"contour", "complexity", "flow"}
            )

        # flow_pipeline should receive the complexity result
        flow_args = m_flow.call_args
        assert flow_args[0][3] is m_comp.return_value  # complexity_result positional arg

    def test_flow_receives_none_complexity_when_absent(self) -> None:
        """When complexity is not in the pipeline set, flow should receive None."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = _fake_landmarks()
        mocks = self._make_mocks()

        with mocks["features"] as _m_feat, mocks["contour"] as _m_cont, \
             mocks["density"] as _m_dens, mocks["complexity"] as _m_comp, \
             mocks["flow"] as m_flow:
            run_resolved_pipelines(
                image, landmarks, {"contour", "flow"}
            )

        flow_args = m_flow.call_args
        assert flow_args[0][3] is None  # complexity_result is None

    def test_configs_forwarded_correctly(self) -> None:
        """Pipeline-specific configs should be forwarded to the correct functions."""
        from portrait_map_lab.models import ComplexityConfig, PipelineConfig

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = _fake_landmarks()
        feat_cfg = PipelineConfig()
        comp_cfg = ComplexityConfig()
        mocks = self._make_mocks()

        with mocks["features"] as m_feat, mocks["contour"] as _m_cont, \
             mocks["density"] as _m_dens, mocks["complexity"] as m_comp, \
             mocks["flow"] as _m_flow:
            run_resolved_pipelines(
                image, landmarks, {"features", "complexity"},
                feature_config=feat_cfg,
                complexity_config=comp_cfg,
            )

        m_feat.assert_called_once_with(landmarks, feat_cfg)
        m_comp.assert_called_once_with(image, comp_cfg)
