"""Tests for the Map Generation API server and CLI serve subcommand."""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from portrait_map_lab.export import ExportBundle, _manifest_to_dict
from portrait_map_lab.models import ExportManifest, ExportMapEntry, LandmarkResult
from portrait_map_lab.server.config import ServerConfig

# ---------------------------------------------------------------------------
# CLI subparser tests (Phase 1.3)
# ---------------------------------------------------------------------------

# Import parse_args from the CLI script
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from run_pipeline import parse_args  # noqa: E402


class TestServeSubparser:
    """Verify the ``serve`` subparser is registered and parses arguments."""

    def test_serve_subparser_exists(self) -> None:
        """The ``serve`` subparser should be recognised by argparse."""
        with patch("sys.argv", ["run_pipeline.py", "serve"]):
            args = parse_args()
        assert args.pipeline == "serve"

    def test_serve_default_host_and_port(self) -> None:
        """Default host and port should match ServerConfig defaults."""
        defaults = ServerConfig()
        with patch("sys.argv", ["run_pipeline.py", "serve"]):
            args = parse_args()
        assert args.host == defaults.host
        assert args.port == defaults.port

    def test_serve_custom_host_and_port(self) -> None:
        """Custom ``--host`` and ``--port`` should override defaults."""
        argv = ["run_pipeline.py", "serve", "--host", "0.0.0.0", "--port", "9000"]
        with patch("sys.argv", argv):
            args = parse_args()
        assert args.host == "0.0.0.0"
        assert args.port == 9000

    def test_existing_subcommands_unaffected(self) -> None:
        """Adding ``serve`` should not break the existing pipeline subcommands."""
        for cmd in ("features", "contour", "density", "flow", "complexity", "all"):
            with patch("sys.argv", ["run_pipeline.py", cmd, "dummy.jpg"]):
                args = parse_args()
            assert args.pipeline == cmd


# ---------------------------------------------------------------------------
# Server endpoint tests (Phase 1.4)
# ---------------------------------------------------------------------------

from portrait_map_lab.server.app import create_app  # noqa: E402


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an httpx async test client backed by the FastAPI app (no real server)."""
    app = create_app()
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


class TestHealthEndpoint:
    """Verify the ``/api/health`` endpoint."""

    @pytest.mark.anyio
    async def test_health_returns_200(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/health")
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_health_returns_ok(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/health")
        assert response.json() == {"status": "ok"}


class TestCORS:
    """Verify CORS headers are present."""

    @pytest.mark.anyio
    async def test_cors_headers_on_preflight(self, client: httpx.AsyncClient) -> None:
        response = await client.options(
            "/api/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in response.headers

    @pytest.mark.anyio
    async def test_cors_allow_origin_wildcard(self, client: httpx.AsyncClient) -> None:
        response = await client.get(
            "/api/health",
            headers={"Origin": "http://localhost:3000"},
        )
        assert response.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# Schema validation tests (Phase 2.1)
# ---------------------------------------------------------------------------

from pydantic import ValidationError  # noqa: E402

from portrait_map_lab.server.schemas import (  # noqa: E402
    VALID_MAP_KEYS,
    GenerateRequest,
    GenerateResponse,
    MapKeyInfo,
)


class TestSchemas:
    """Validate Pydantic request/response schemas."""

    # -- GenerateRequest -------------------------------------------------

    def test_empty_request_accepted(self) -> None:
        """A completely empty body should be valid (all fields optional)."""
        req = GenerateRequest()
        assert req.image_path is None
        assert req.maps is None
        assert req.persist is None
        assert req.config is None

    def test_valid_map_keys_accepted(self) -> None:
        """All canonical map keys should pass validation."""
        req = GenerateRequest(maps=sorted(VALID_MAP_KEYS))
        assert set(req.maps) == VALID_MAP_KEYS

    def test_invalid_map_key_rejected(self) -> None:
        """An unrecognised map key should raise a validation error."""
        with pytest.raises(ValidationError, match="tonal_target"):
            GenerateRequest(maps=["tonal_target"])

    def test_mixed_valid_and_invalid_keys_rejected(self) -> None:
        """A mix of valid and invalid keys should still be rejected."""
        with pytest.raises(ValidationError, match="bogus"):
            GenerateRequest(maps=["density_target", "bogus"])

    # -- Partial config overrides ----------------------------------------

    def test_partial_density_override(self) -> None:
        """Only specifying ``density.gamma`` should leave other fields as None."""
        req = GenerateRequest(config={"density": {"gamma": 2.0}})
        assert req.config.density.gamma == 2.0
        assert req.config.density.feature_weight is None
        assert req.config.features is None

    def test_nested_etf_override(self) -> None:
        """Nested override ``flow.etf.blur_sigma`` should parse correctly."""
        req = GenerateRequest(
            config={"flow": {"etf": {"blur_sigma": 5.0}}}
        )
        assert req.config.flow.etf.blur_sigma == 5.0
        assert req.config.flow.etf.structure_sigma is None
        assert req.config.flow.contour_smooth_sigma is None

    def test_features_weights_override(self) -> None:
        """Custom feature weights should be accepted."""
        req = GenerateRequest(
            config={"features": {"weights": {"eyes": 1.0, "mouth": 0.0}}}
        )
        assert req.config.features.weights == {"eyes": 1.0, "mouth": 0.0}

    def test_contour_override_with_nested_remap(self) -> None:
        """Contour override with nested remap should parse correctly."""
        req = GenerateRequest(
            config={"contour": {"band_width": 10.0, "remap": {"radius": 200.0}}}
        )
        assert req.config.contour.band_width == 10.0
        assert req.config.contour.remap.radius == 200.0
        assert req.config.contour.remap.sigma is None

    def test_complexity_override(self) -> None:
        """Complexity config override should be accepted."""
        req = GenerateRequest(
            config={"complexity": {"metric": "laplacian", "sigma": 5.0}}
        )
        assert req.config.complexity.metric == "laplacian"
        assert req.config.complexity.sigma == 5.0

    def test_flow_speed_override(self) -> None:
        """Flow speed config override should be accepted."""
        req = GenerateRequest(
            config={"flow_speed": {"speed_min": 0.1, "speed_max": 0.8}}
        )
        assert req.config.flow_speed.speed_min == 0.1
        assert req.config.flow_speed.speed_max == 0.8

    def test_all_config_fields_default_none(self) -> None:
        """An empty config should have all pipeline sections as None."""
        req = GenerateRequest(config={})
        cfg = req.config
        assert cfg.features is None
        assert cfg.contour is None
        assert cfg.density is None
        assert cfg.complexity is None
        assert cfg.flow is None
        assert cfg.flow_speed is None

    # -- Response schemas ------------------------------------------------

    def test_generate_response_roundtrip(self) -> None:
        """GenerateResponse should serialize and deserialize correctly."""
        resp = GenerateResponse(
            session_id="abc-123",
            manifest={"version": 1, "maps": []},
            base_url="/api/maps/abc-123",
        )
        data = resp.model_dump()
        assert data["session_id"] == "abc-123"
        assert data["base_url"] == "/api/maps/abc-123"

    def test_map_key_info_structure(self) -> None:
        """MapKeyInfo should hold key, value_range, and description."""
        info = MapKeyInfo(key="flow_x", value_range=[-1.0, 1.0], description="X component")
        assert info.key == "flow_x"
        assert info.value_range == [-1.0, 1.0]
        assert info.description == "X component"


# ---------------------------------------------------------------------------
# Map keys endpoint tests (Phase 2.2)
# ---------------------------------------------------------------------------

class TestMapKeysEndpoint:
    """Verify the ``GET /api/maps/keys`` endpoint."""

    @pytest.mark.anyio
    async def test_returns_200(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/maps/keys")
        assert response.status_code == 200

    @pytest.mark.anyio
    async def test_returns_all_map_keys(self, client: httpx.AsyncClient) -> None:
        """Response should contain all keys from _MAP_DEFINITIONS."""
        response = await client.get("/api/maps/keys")
        data = response.json()
        assert len(data) == len(VALID_MAP_KEYS)
        returned_keys = {entry["key"] for entry in data}
        assert returned_keys == VALID_MAP_KEYS

    @pytest.mark.anyio
    async def test_entry_structure(self, client: httpx.AsyncClient) -> None:
        """Each entry should have key (str), value_range (2-element list), description (str)."""
        response = await client.get("/api/maps/keys")
        for entry in response.json():
            assert isinstance(entry["key"], str)
            assert isinstance(entry["value_range"], list)
            assert len(entry["value_range"]) == 2
            assert all(isinstance(v, (int, float)) for v in entry["value_range"])
            assert isinstance(entry["description"], str)
            assert len(entry["description"]) > 0

    @pytest.mark.anyio
    async def test_known_keys_present(self, client: httpx.AsyncClient) -> None:
        """Spot-check that well-known keys are included with correct ranges."""
        response = await client.get("/api/maps/keys")
        by_key = {entry["key"]: entry for entry in response.json()}
        assert "density_target" in by_key
        assert by_key["density_target"]["value_range"] == [0.0, 1.0]
        assert "flow_x" in by_key
        assert by_key["flow_x"]["value_range"] == [-1.0, 1.0]


# ---------------------------------------------------------------------------
# Generate endpoint tests (Phase 2.3)
# ---------------------------------------------------------------------------


def _fake_landmarks() -> LandmarkResult:
    """Build a minimal ``LandmarkResult`` for test mocks."""
    return LandmarkResult(
        landmarks=np.zeros((478, 2), dtype=np.float32),
        image_shape=(100, 100),
        confidence=0.99,
    )


def _fake_export_bundle() -> ExportBundle:
    """Build a minimal ``ExportBundle`` for test mocks."""
    entry = ExportMapEntry(
        filename="density_target.bin",
        key="density_target",
        dtype="float32",
        shape=(100, 100),
        value_range=(0.0, 1.0),
        description="test density",
    )
    manifest = ExportManifest(
        version=1,
        source_image="test.jpg",
        width=100,
        height=100,
        created_at="2026-03-20T00:00:00+00:00",
        maps=(entry,),
    )
    return ExportBundle(
        manifest=manifest,
        binary_maps={"density_target": np.zeros((100, 100), dtype=np.float32).tobytes()},
        png_files={},
    )


def _fake_manifest_dict() -> dict:
    """Return a manifest dict derived from ``_fake_export_bundle``."""
    return _manifest_to_dict(_fake_export_bundle().manifest)


# Patch targets — these are the module paths where routes.py imports them.
_PATCH_LOAD = "portrait_map_lab.server.routes.load_image"
_PATCH_DETECT = "portrait_map_lab.server.routes.detect_landmarks"
_PATCH_RUN = "portrait_map_lab.server.routes.run_all_pipelines"
_PATCH_BUNDLE = "portrait_map_lab.server.routes.build_export_bundle"
_PATCH_MANIFEST = "portrait_map_lab.server.routes._manifest_to_dict"


@contextlib.contextmanager
def _mock_pipeline_stack(cache_dir: Path | None = None):
    """Context manager that mocks the entire pipeline for generate tests."""
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    sentinel_result = MagicMock(name="ComposedResult")
    bundle = _fake_export_bundle()
    manifest_dict = _fake_manifest_dict()

    patches = [
        patch(_PATCH_LOAD, return_value=fake_image),
        patch(_PATCH_DETECT, return_value=_fake_landmarks()),
        patch(_PATCH_RUN, return_value=sentinel_result),
        patch(_PATCH_BUNDLE, return_value=bundle),
        patch(_PATCH_MANIFEST, return_value=manifest_dict),
    ]
    if cache_dir is not None:
        patches.append(
            patch(
                "portrait_map_lab.server.routes.ServerConfig",
                return_value=ServerConfig(cache_dir=cache_dir),
            )
        )

    with contextlib.ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in patches]
        yield mocks


class TestGenerateEndpoint:
    """Verify the ``POST /api/generate`` endpoint."""

    @pytest.mark.anyio
    async def test_generate_with_valid_image_path(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """A valid image_path should return 200 with session_id, manifest, base_url."""
        with _mock_pipeline_stack(cache_dir=tmp_path / "cache"):
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "manifest" in data
        assert "base_url" in data
        assert data["base_url"].startswith("/api/maps/")

    @pytest.mark.anyio
    async def test_generate_invalid_image_path(
        self, client: httpx.AsyncClient
    ) -> None:
        """A non-existent image path should return 422."""
        with patch(_PATCH_LOAD, side_effect=FileNotFoundError("not found")):
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/nonexistent/image.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 422
        detail = response.json()["detail"].lower()
        assert "not found" in detail or "image" in detail

    @pytest.mark.anyio
    async def test_generate_no_face_detected(
        self, client: httpx.AsyncClient
    ) -> None:
        """An image with no detectable face should return 422."""
        fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
        with (
            patch(_PATCH_LOAD, return_value=fake_image),
            patch(_PATCH_DETECT, side_effect=ValueError("No face detected")),
        ):
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 422
        assert "no face detected" in response.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_generate_no_image_source(
        self, client: httpx.AsyncClient
    ) -> None:
        """An empty request with no image should return 422."""
        response = await client.post(
            "/api/generate",
            content=json.dumps({}),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "no image" in response.json()["detail"].lower()

    @pytest.mark.anyio
    async def test_generate_creates_cache_files(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Generate should write .bin files and manifest.json to the cache dir."""
        cache_dir = tmp_path / "cache"
        with _mock_pipeline_stack(cache_dir=cache_dir):
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        session_dir = cache_dir / session_id
        assert session_dir.is_dir()
        assert (session_dir / "manifest.json").is_file()
        assert (session_dir / "density_target.bin").is_file()

        # Verify manifest JSON is valid
        manifest = json.loads((session_dir / "manifest.json").read_text())
        assert manifest["version"] == 1
        assert manifest["width"] == 100
        assert manifest["height"] == 100

    @pytest.mark.anyio
    async def test_generate_manifest_structure(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """The manifest in the response should have all required plotter fields."""
        with _mock_pipeline_stack(cache_dir=tmp_path / "cache"):
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        manifest = response.json()["manifest"]
        # Top-level fields required by the plotter's parseManifest()
        assert "version" in manifest
        assert "source_image" in manifest
        assert "width" in manifest
        assert "height" in manifest
        assert "created_at" in manifest
        assert "maps" in manifest
        # Map entry structure
        map_entry = manifest["maps"][0]
        assert "filename" in map_entry
        assert "key" in map_entry
        assert "dtype" in map_entry
        assert "shape" in map_entry
        assert "value_range" in map_entry
        assert "description" in map_entry

    @pytest.mark.anyio
    async def test_generate_with_file_upload(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Multipart file upload should also produce a valid response."""
        # Create a small dummy JPEG-like file
        fake_bytes = b"\xff\xd8\xff" + b"\x00" * 100
        with _mock_pipeline_stack(cache_dir=tmp_path / "cache"):
            response = await client.post(
                "/api/generate",
                files={"image": ("test.jpg", fake_bytes, "image/jpeg")},
            )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["base_url"].startswith("/api/maps/")

    @pytest.mark.anyio
    async def test_generate_with_config_overrides(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Config overrides should be propagated to run_all_pipelines."""
        with _mock_pipeline_stack(cache_dir=tmp_path / "cache"):
            response = await client.post(
                "/api/generate",
                content=json.dumps({
                    "image_path": "/some/test.jpg",
                    "config": {"density": {"gamma": 2.0}},
                }),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Config merge helper tests (Phase 2.3)
# ---------------------------------------------------------------------------

from portrait_map_lab.server.schemas import (  # noqa: E402
    build_complexity_config,
    build_compose_config,
    build_contour_config,
    build_flow_config,
    build_flow_speed_config,
    build_pipeline_config,
)


class TestConfigMergeHelpers:
    """Verify that schema-to-dataclass merge helpers work correctly."""

    def test_none_schema_returns_none(self) -> None:
        """All builders should return None when schema is None."""
        assert build_pipeline_config(None) is None
        assert build_contour_config(None) is None
        assert build_compose_config(None) is None
        assert build_complexity_config(None) is None
        assert build_flow_config(None) is None
        assert build_flow_speed_config(None) is None

    def test_empty_schema_returns_defaults(self) -> None:
        """An empty schema should produce a dataclass with default values."""
        from portrait_map_lab.models import PipelineConfig
        from portrait_map_lab.server.schemas import FeaturesConfigSchema

        cfg = build_pipeline_config(FeaturesConfigSchema())
        default = PipelineConfig()
        assert cfg.weights == default.weights
        assert cfg.remap.radius == default.remap.radius

    def test_scalar_override(self) -> None:
        """A non-None scalar should override the default."""
        from portrait_map_lab.server.schemas import DensityConfigSchema

        cfg = build_compose_config(DensityConfigSchema(gamma=2.5))
        assert cfg.gamma == 2.5
        # Other fields keep defaults
        assert cfg.feature_weight == 0.6

    def test_nested_remap_override(self) -> None:
        """Nested remap overrides should merge onto the default RemapConfig."""
        from portrait_map_lab.server.schemas import (
            ContourConfigSchema,
            RemapConfigSchema,
        )

        cfg = build_contour_config(
            ContourConfigSchema(remap=RemapConfigSchema(radius=300.0))
        )
        assert cfg.remap.radius == 300.0
        # Non-overridden remap fields keep defaults
        assert cfg.remap.sigma == 80.0

    def test_nested_etf_override(self) -> None:
        """Nested ETF overrides should merge onto the default ETFConfig."""
        from portrait_map_lab.server.schemas import (
            ETFConfigSchema,
            FlowConfigSchema,
        )

        cfg = build_flow_config(
            FlowConfigSchema(etf=ETFConfigSchema(blur_sigma=5.0))
        )
        assert cfg.etf.blur_sigma == 5.0
        assert cfg.etf.structure_sigma == 5.0  # default

    def test_nested_luminance_override(self) -> None:
        """Nested luminance overrides should merge onto the default LuminanceConfig."""
        from portrait_map_lab.server.schemas import (
            DensityConfigSchema,
            LuminanceConfigSchema,
        )

        cfg = build_compose_config(
            DensityConfigSchema(luminance=LuminanceConfigSchema(clip_limit=4.0))
        )
        assert cfg.luminance.clip_limit == 4.0
        assert cfg.luminance.tile_size == 8  # default

    def test_flow_speed_override(self) -> None:
        """Flow speed overrides should apply correctly."""
        from portrait_map_lab.server.schemas import FlowSpeedConfigSchema

        cfg = build_flow_speed_config(
            FlowSpeedConfigSchema(speed_min=0.1, speed_max=0.9)
        )
        assert cfg.speed_min == 0.1
        assert cfg.speed_max == 0.9

    def test_complexity_override(self) -> None:
        """Complexity overrides should apply correctly."""
        from portrait_map_lab.server.schemas import ComplexityConfigSchema

        cfg = build_complexity_config(
            ComplexityConfigSchema(metric="laplacian", sigma=5.0)
        )
        assert cfg.metric == "laplacian"
        assert cfg.sigma == 5.0
        assert cfg.normalize_percentile == 99.0  # default
