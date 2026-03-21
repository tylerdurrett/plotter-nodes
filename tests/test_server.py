"""Tests for the Map Generation API server and CLI serve subcommand."""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from portrait_map_lab.export import ExportBundle, manifest_to_dict
from portrait_map_lab.models import ExportManifest, ExportMapEntry, LandmarkResult
from portrait_map_lab.server.cache import SessionCache
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
async def client(tmp_path: Path) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an httpx async test client backed by the FastAPI app.

    Each test gets its own app with a fresh ``SessionCache`` backed by a
    temporary directory.  The cache is initialised directly (bypassing the
    lifespan's background cleanup task) for test isolation and speed.
    """
    config = ServerConfig(cache_dir=tmp_path / "cache")
    app = create_app(config)
    # Initialise cache directly — httpx ASGITransport does not send
    # lifespan events, so we set up the cache that the lifespan would create.
    app.state.cache = SessionCache(config)
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
    return manifest_to_dict(_fake_export_bundle().manifest)


# Patch targets — these are the module paths where routes.py imports them.
_PATCH_LOAD = "portrait_map_lab.server.routes.load_image"
_PATCH_DETECT = "portrait_map_lab.server.routes.detect_landmarks"
_PATCH_RUN = "portrait_map_lab.server.routes.run_all_pipelines"
_PATCH_BUNDLE = "portrait_map_lab.server.routes.build_export_bundle"
_PATCH_MANIFEST = "portrait_map_lab.server.routes.manifest_to_dict"


@contextlib.contextmanager
def _mock_pipeline_stack():
    """Context manager that mocks the entire pipeline for generate tests.

    The cache directory is now managed by the app's ``SessionCache`` (set up
    by the ``client`` fixture), so no ``ServerConfig`` patching is needed.
    """
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
        with _mock_pipeline_stack():
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
        with _mock_pipeline_stack():
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
        with _mock_pipeline_stack():
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
        with _mock_pipeline_stack():
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
        with _mock_pipeline_stack():
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


# ---------------------------------------------------------------------------
# Map file serving tests (Phase 2.4)
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _generate_session(
    client: httpx.AsyncClient,
):
    """Generate a mocked session and yield its response data.

    The cache directory is managed by the app's ``SessionCache``, so no
    ``ServerConfig`` patching is needed.
    """
    with _mock_pipeline_stack():
        gen = await client.post(
            "/api/generate",
            content=json.dumps({"image_path": "/some/test.jpg"}),
            headers={"Content-Type": "application/json"},
        )
        assert gen.status_code == 200
        yield gen.json()


class TestMapFileServing:
    """Verify the map file serving endpoints."""

    @pytest.mark.anyio
    async def test_serve_manifest_json(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Fetching manifest.json should return the same manifest as the generate response."""
        async with _generate_session(client) as data:
            response = await client.get(f"{data['base_url']}/manifest.json")
        assert response.status_code == 200
        assert response.json() == data["manifest"]

    @pytest.mark.anyio
    async def test_serve_bin_file(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Fetching a .bin file should return valid float32 data with correct byte length."""
        async with _generate_session(client) as data:
            manifest = data["manifest"]
            width, height = manifest["width"], manifest["height"]
            map_entry = manifest["maps"][0]
            response = await client.get(f"{data['base_url']}/{map_entry['filename']}")
        assert response.status_code == 200

        raw = response.content
        assert len(raw) == height * width * 4

        array = np.frombuffer(raw, dtype=np.float32)
        assert array.shape == (height * width,)

    @pytest.mark.anyio
    async def test_fetch_all_bin_urls(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """All .bin files listed in the manifest should be fetchable."""
        async with _generate_session(client) as data:
            manifest = data["manifest"]
            expected_bytes = manifest["height"] * manifest["width"] * 4
            for map_entry in manifest["maps"]:
                response = await client.get(f"{data['base_url']}/{map_entry['filename']}")
                assert response.status_code == 200, f"Failed for {map_entry['filename']}"
                assert len(response.content) == expected_bytes

    @pytest.mark.anyio
    async def test_nonexistent_session_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        """A non-existent session ID should return 404."""
        response = await client.get("/api/maps/nonexistent-id/manifest.json")
        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_nonexistent_file_returns_404(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """A non-existent filename within a valid session should return 404."""
        async with _generate_session(client) as data:
            response = await client.get(f"{data['base_url']}/nonexistent.bin")
        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_path_traversal_session_id_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        """Path traversal in session_id should return 404."""
        response = await client.get("/api/maps/../../etc/manifest.json")
        assert response.status_code in (404, 422)

    @pytest.mark.anyio
    async def test_path_traversal_filename_returns_404(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Path traversal in filename should return 404."""
        async with _generate_session(client) as data:
            response = await client.get(f"{data['base_url']}/../../../etc/passwd")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Config override integration tests (Phase 2.5)
# ---------------------------------------------------------------------------

from portrait_map_lab.models import ComplexityConfig as _ComplexityConfig  # noqa: E402


class TestConfigOverrideIntegration:
    """Verify that config overrides sent via the API are propagated to pipelines."""

    @pytest.mark.anyio
    async def test_density_gamma_override(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """``config.density.gamma = 2.0`` should reach run_all_pipelines as compose_config."""
        with _mock_pipeline_stack() as mocks:
            response = await client.post(
                "/api/generate",
                content=json.dumps({
                    "image_path": "/some/test.jpg",
                    "config": {"density": {"gamma": 2.0}},
                }),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        run_mock = mocks[2]  # run_all_pipelines
        kwargs = run_mock.call_args.kwargs
        assert kwargs["compose_config"].gamma == 2.0
        # Non-overridden fields keep defaults
        assert kwargs["compose_config"].feature_weight == 0.6

    @pytest.mark.anyio
    async def test_features_weights_override(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """``config.features.weights`` should reach run_all_pipelines as feature_config."""
        weights = {"eyes": 1.0, "mouth": 0.0}
        with _mock_pipeline_stack() as mocks:
            response = await client.post(
                "/api/generate",
                content=json.dumps({
                    "image_path": "/some/test.jpg",
                    "config": {"features": {"weights": weights}},
                }),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        run_mock = mocks[2]  # run_all_pipelines
        kwargs = run_mock.call_args.kwargs
        assert kwargs["feature_config"].weights == weights

    @pytest.mark.anyio
    async def test_nested_etf_override(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """``config.flow.etf.blur_sigma = 5.0`` should merge onto FlowConfig.etf."""
        with _mock_pipeline_stack() as mocks:
            response = await client.post(
                "/api/generate",
                content=json.dumps({
                    "image_path": "/some/test.jpg",
                    "config": {"flow": {"etf": {"blur_sigma": 5.0}}},
                }),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        run_mock = mocks[2]  # run_all_pipelines
        kwargs = run_mock.call_args.kwargs
        assert kwargs["flow_config"].etf.blur_sigma == 5.0
        # Non-overridden ETF fields keep defaults
        assert kwargs["flow_config"].etf.structure_sigma == 5.0

    @pytest.mark.anyio
    async def test_no_config_uses_defaults(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """No config overrides: None for optional configs, default ComplexityConfig."""
        with _mock_pipeline_stack() as mocks:
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        run_mock = mocks[2]  # run_all_pipelines
        kwargs = run_mock.call_args.kwargs
        assert kwargs["feature_config"] is None
        assert kwargs["contour_config"] is None
        assert kwargs["compose_config"] is None
        assert kwargs["flow_config"] is None
        assert kwargs["speed_config"] is None
        # Complexity always gets a default config (never None)
        cx = kwargs["complexity_config"]
        assert cx is not None
        default_cx = _ComplexityConfig()
        assert cx.metric == default_cx.metric
        assert cx.sigma == default_cx.sigma


# ---------------------------------------------------------------------------
# Resolver integration tests (Phase 3.3)
# ---------------------------------------------------------------------------

# Additional patch targets for the resolver path
_PATCH_RESOLVE = "portrait_map_lab.server.routes.resolve_pipelines"
_PATCH_RUN_RESOLVED = "portrait_map_lab.server.routes.run_resolved_pipelines"
_PATCH_BUNDLE_FOR_MAPS = "portrait_map_lab.server.routes.build_export_bundle_for_maps"


def _fake_export_bundle_for_keys(keys: list[str]) -> ExportBundle:
    """Build a minimal ``ExportBundle`` containing entries for the given keys."""
    entries = []
    binary_maps = {}
    for key in keys:
        entries.append(
            ExportMapEntry(
                filename=f"{key}.bin",
                key=key,
                dtype="float32",
                shape=(100, 100),
                value_range=(0.0, 1.0),
                description=f"test {key}",
            )
        )
        binary_maps[key] = np.zeros((100, 100), dtype=np.float32).tobytes()

    manifest = ExportManifest(
        version=1,
        source_image="test.jpg",
        width=100,
        height=100,
        created_at="2026-03-20T00:00:00+00:00",
        maps=tuple(entries),
    )
    return ExportBundle(manifest=manifest, binary_maps=binary_maps, png_files={})


@contextlib.contextmanager
def _mock_resolver_stack(
    requested_maps: list[str],
    resolved_pipelines: set[str],
):
    """Context manager that mocks the resolver path for generate tests.

    Mocks ``resolve_pipelines``, ``run_resolved_pipelines``, and
    ``build_export_bundle_for_maps`` so the endpoint exercises the
    granular map selection path without real pipeline execution.
    """
    fake_image = np.zeros((100, 100, 3), dtype=np.uint8)
    bundle = _fake_export_bundle_for_keys(requested_maps)
    manifest_dict = manifest_to_dict(bundle.manifest)

    patches = [
        patch(_PATCH_LOAD, return_value=fake_image),
        patch(_PATCH_DETECT, return_value=_fake_landmarks()),
        patch(_PATCH_RESOLVE, return_value=resolved_pipelines),
        patch(
            _PATCH_RUN_RESOLVED,
            return_value={p: MagicMock(name=f"{p}_result") for p in resolved_pipelines},
        ),
        patch(_PATCH_BUNDLE_FOR_MAPS, return_value=bundle),
        patch(_PATCH_MANIFEST, return_value=manifest_dict),
    ]

    with contextlib.ExitStack() as stack:
        mocks = [stack.enter_context(p) for p in patches]
        yield mocks


class TestResolverIntegration:
    """Verify that ``POST /api/generate`` uses the resolver when ``maps`` is specified."""

    @pytest.mark.anyio
    async def test_complexity_only_skips_other_pipelines(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Requesting only ``["complexity"]`` should resolve to the complexity pipeline only."""
        cache_dir = tmp_path / "cache"
        maps = ["complexity"]
        resolved = {"complexity"}

        with _mock_resolver_stack(maps, resolved) as mocks:
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg", "maps": maps}),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        # Verify resolver was called with the requested maps
        resolve_mock = mocks[2]  # resolve_pipelines
        resolve_mock.assert_called_once_with(maps)

        # Verify run_resolved_pipelines was called (not run_all_pipelines)
        run_resolved_mock = mocks[3]  # run_resolved_pipelines
        run_resolved_mock.assert_called_once()

        # Verify only complexity.bin exists in the cache
        data = response.json()
        session_dir = cache_dir / data["session_id"]
        assert (session_dir / "complexity.bin").is_file()
        assert not (session_dir / "density_target.bin").exists()
        assert not (session_dir / "flow_x.bin").exists()

        # Manifest should only contain the requested map
        manifest = data["manifest"]
        manifest_keys = {m["key"] for m in manifest["maps"]}
        assert manifest_keys == {"complexity"}

    @pytest.mark.anyio
    async def test_flow_xy_skips_density(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Requesting ``["flow_x", "flow_y"]`` should not run the density pipeline."""
        cache_dir = tmp_path / "cache"
        maps = ["flow_x", "flow_y"]
        resolved = {"contour", "flow"}

        with _mock_resolver_stack(maps, resolved) as mocks:
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg", "maps": maps}),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        run_resolved_mock = mocks[3]  # run_resolved_pipelines
        call_kwargs = run_resolved_mock.call_args
        # The resolved pipeline set passed to run_resolved_pipelines
        assert call_kwargs[0][2] == {"contour", "flow"}

        # Verify cache contains only the requested maps
        data = response.json()
        session_dir = cache_dir / data["session_id"]
        assert (session_dir / "flow_x.bin").is_file()
        assert (session_dir / "flow_y.bin").is_file()
        assert not (session_dir / "density_target.bin").exists()

    @pytest.mark.anyio
    async def test_density_and_flow_speed_runs_needed_pipelines(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Requesting ``["density_target", "flow_speed"]`` should run all needed pipelines."""
        maps = ["density_target", "flow_speed"]
        resolved = {"features", "contour", "density", "complexity", "flow"}

        with _mock_resolver_stack(maps, resolved) as mocks:
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg", "maps": maps}),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        run_resolved_mock = mocks[3]
        call_kwargs = run_resolved_mock.call_args
        assert call_kwargs[0][2] == resolved

        # Only the two requested maps should be in the manifest
        data = response.json()
        manifest_keys = {m["key"] for m in data["manifest"]["maps"]}
        assert manifest_keys == {"density_target", "flow_speed"}

    @pytest.mark.anyio
    async def test_empty_maps_uses_all_pipelines(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Omitting ``maps`` should use the full ``run_all_pipelines`` path."""
        with _mock_pipeline_stack() as mocks:
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        # run_all_pipelines should have been called (not run_resolved_pipelines)
        run_all_mock = mocks[2]  # run_all_pipelines
        run_all_mock.assert_called_once()

    @pytest.mark.anyio
    async def test_maps_with_config_overrides(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Config overrides should be forwarded through the resolver path."""
        maps = ["complexity"]
        resolved = {"complexity"}

        with _mock_resolver_stack(maps, resolved) as mocks:
            response = await client.post(
                "/api/generate",
                content=json.dumps({
                    "image_path": "/some/test.jpg",
                    "maps": maps,
                    "config": {"complexity": {"metric": "laplacian"}},
                }),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        run_resolved_mock = mocks[3]
        call_kwargs = run_resolved_mock.call_args.kwargs
        assert call_kwargs["complexity_config"].metric == "laplacian"

    @pytest.mark.anyio
    async def test_resolver_path_returns_base_url(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """The resolver path should return a valid ``base_url`` for file serving."""
        maps = ["complexity"]
        resolved = {"complexity"}

        with _mock_resolver_stack(maps, resolved):
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg", "maps": maps}),
                headers={"Content-Type": "application/json"},
            )
        data = response.json()
        assert data["base_url"].startswith("/api/maps/")


# ---------------------------------------------------------------------------
# Session list and delete endpoint tests (Phase 3.4)
# ---------------------------------------------------------------------------


class TestSessionEndpoints:
    """Verify the ``GET /api/sessions`` and ``DELETE /api/maps/{session_id}`` endpoints."""

    @pytest.mark.anyio
    async def test_sessions_empty_initially(
        self, client: httpx.AsyncClient
    ) -> None:
        """``GET /api/sessions`` should return an empty list when no sessions exist."""
        response = await client.get("/api/sessions")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.anyio
    async def test_sessions_lists_after_generate(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """After generating, the session should appear in the sessions list."""
        with _mock_pipeline_stack():
            gen = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        assert gen.status_code == 200
        gen_data = gen.json()

        response = await client.get("/api/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 1
        session = sessions[0]
        assert session["session_id"] == gen_data["session_id"]
        assert session["source_image"] == "test.jpg"
        assert session["created_at"] == gen_data["manifest"]["created_at"]
        assert "density_target" in session["map_keys"]

    @pytest.mark.anyio
    async def test_sessions_lists_multiple(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Multiple generates should produce multiple session entries."""
        for i in range(3):
            with _mock_pipeline_stack():
                resp = await client.post(
                    "/api/generate",
                    content=json.dumps({"image_path": "/some/test.jpg"}),
                    headers={"Content-Type": "application/json"},
                )
            assert resp.status_code == 200

        response = await client.get("/api/sessions")
        assert response.status_code == 200
        assert len(response.json()) == 3

    @pytest.mark.anyio
    async def test_delete_session_returns_204(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """``DELETE /api/maps/{session_id}`` should return 204 and remove from list."""
        with _mock_pipeline_stack():
            gen = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        session_id = gen.json()["session_id"]

        response = await client.delete(f"/api/maps/{session_id}")
        assert response.status_code == 204

        # Session should be gone from the list
        sessions_resp = await client.get("/api/sessions")
        assert len(sessions_resp.json()) == 0

    @pytest.mark.anyio
    async def test_delete_session_removes_files(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Deleting a session should remove its cache directory from disk."""
        cache_dir = tmp_path / "cache"
        with _mock_pipeline_stack():
            gen = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        session_id = gen.json()["session_id"]
        session_dir = cache_dir / session_id
        assert session_dir.is_dir()

        response = await client.delete(f"/api/maps/{session_id}")
        assert response.status_code == 204
        assert not session_dir.exists()

    @pytest.mark.anyio
    async def test_delete_nonexistent_session_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        """Deleting a non-existent session should return 404."""
        response = await client.delete("/api/maps/nonexistent-session-id")
        assert response.status_code == 404

    @pytest.mark.anyio
    async def test_delete_path_traversal_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        """Path traversal in session_id on DELETE should return 404."""
        response = await client.delete("/api/maps/../../etc")
        assert response.status_code in (404, 422)


# ---------------------------------------------------------------------------
# Cache lifecycle integration tests (Phase 4.2)
# ---------------------------------------------------------------------------


class TestCacheLifecycle:
    """Verify that the ``SessionCache`` is wired into the server lifecycle."""

    @pytest.mark.anyio
    async def test_generate_registers_session_in_cache(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Generating maps should register the session in the app's SessionCache."""
        with _mock_pipeline_stack():
            gen = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        assert gen.status_code == 200
        session_id = gen.json()["session_id"]

        # Access the cache directly from app state
        cache: SessionCache = client._transport.app.state.cache  # type: ignore[attr-defined]
        sessions = cache.list_sessions()
        assert any(s.session_id == session_id for s in sessions)

    @pytest.mark.anyio
    async def test_cache_uses_configured_directory(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """The cache should use the directory from ``ServerConfig``."""
        cache: SessionCache = client._transport.app.state.cache  # type: ignore[attr-defined]
        assert cache.cache_dir == tmp_path / "cache"

    @pytest.mark.anyio
    async def test_delete_uses_cache(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Deleting a session should remove it from the ``SessionCache``."""
        with _mock_pipeline_stack():
            gen = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        session_id = gen.json()["session_id"]

        response = await client.delete(f"/api/maps/{session_id}")
        assert response.status_code == 204

        cache: SessionCache = client._transport.app.state.cache  # type: ignore[attr-defined]
        assert not any(s.session_id == session_id for s in cache.list_sessions())

    def test_expired_sessions_cleaned_up_by_cache(
        self, tmp_path: Path
    ) -> None:
        """Sessions past TTL should be removed when ``cleanup_expired`` is called.

        This tests the integration between the cache and the generate
        endpoint: a session registered by ``generate`` should be subject
        to TTL cleanup.
        """
        # Create a cache with a very short TTL
        config = ServerConfig(cache_dir=tmp_path / "cache", session_ttl_seconds=1)
        cache = SessionCache(config)

        # Simulate a session that was created 5 seconds ago
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        from portrait_map_lab.server.schemas import SessionInfo

        info = SessionInfo(
            session_id="old-session",
            source_image="test.jpg",
            created_at=old_time,
            map_keys=["density_target"],
        )
        cache.register(info)

        # Create the session directory on disk
        session_dir = cache.get_path("old-session")
        session_dir.mkdir(parents=True)
        (session_dir / "density_target.bin").write_bytes(b"\x00" * 16)

        # Run cleanup — session should be expired
        removed = cache.cleanup_expired()
        assert removed == 1
        assert not session_dir.exists()
        assert cache.list_sessions() == []

    @pytest.mark.anyio
    async def test_startup_scan_recovers_sessions(
        self, tmp_path: Path
    ) -> None:
        """Sessions from a previous run should be discoverable after re-creating the cache.

        Simulates a server restart by writing a manifest to disk, then
        creating a fresh ``SessionCache`` that scans the directory.
        """
        cache_dir = tmp_path / "cache"
        session_dir = cache_dir / "recovered-session"
        session_dir.mkdir(parents=True)
        manifest = {
            "version": 1,
            "source_image": "portrait.jpg",
            "width": 100,
            "height": 100,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "maps": [{"filename": "density_target.bin", "key": "density_target",
                       "dtype": "float32", "shape": [100, 100],
                       "value_range": [0.0, 1.0], "description": "test"}],
        }
        (session_dir / "manifest.json").write_text(json.dumps(manifest))

        # Create a new app+cache — the startup scan should find the session
        config = ServerConfig(cache_dir=cache_dir)
        app = create_app(config)
        app.state.cache = SessionCache(config)

        transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
            response = await c.get("/api/sessions")
        assert response.status_code == 200
        sessions = response.json()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "recovered-session"
        assert sessions[0]["source_image"] == "portrait.jpg"


# ---------------------------------------------------------------------------
# Persist parameter tests (Phase 5.1)
# ---------------------------------------------------------------------------


class TestPersistParameter:
    """Verify the ``persist`` parameter on ``POST /api/generate``."""

    # -- Schema validation ---------------------------------------------------

    def test_valid_persist_values_accepted(self) -> None:
        """Alphanumeric, hyphens, and underscores should be accepted."""
        for value in ("my-portrait", "test_123", "AbCdEf", "a"):
            req = GenerateRequest(persist=value)
            assert req.persist == value

    def test_persist_none_accepted(self) -> None:
        """Omitting persist should default to None."""
        req = GenerateRequest()
        assert req.persist is None

    def test_persist_path_traversal_rejected(self) -> None:
        """Path traversal attempts should be rejected by the validator."""
        for value in ("../../etc", "../secret", "a/b", "a\\b"):
            with pytest.raises(ValidationError, match="persist"):
                GenerateRequest(persist=value)

    def test_persist_special_chars_rejected(self) -> None:
        """Special characters should be rejected."""
        for value in ("hello world", "test!", "a@b", "a.b", ""):
            with pytest.raises(ValidationError, match="persist"):
                GenerateRequest(persist=value)

    # -- Integration: files created at output path ---------------------------

    @pytest.mark.anyio
    async def test_persist_creates_output_files(
        self, client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Persist should copy cache files to ``output/{persist}/export/``."""
        output_root = tmp_path / "output"
        # Monkeypatch Path("output") resolution — the route uses a relative
        # path, so we change the working directory to tmp_path so that
        # "output/{persist}/export" resolves inside tmp_path.
        monkeypatch.chdir(tmp_path)

        with _mock_pipeline_stack():
            response = await client.post(
                "/api/generate",
                content=json.dumps({
                    "image_path": "/some/test.jpg",
                    "persist": "my-portrait",
                }),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        persist_dir = output_root / "my-portrait" / "export"
        assert persist_dir.is_dir()
        assert (persist_dir / "manifest.json").is_file()
        assert (persist_dir / "density_target.bin").is_file()

    @pytest.mark.anyio
    async def test_persist_contents_match_cache(
        self, client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Persisted files should be byte-identical to the cache directory."""
        monkeypatch.chdir(tmp_path)
        cache_dir = tmp_path / "cache"

        with _mock_pipeline_stack():
            response = await client.post(
                "/api/generate",
                content=json.dumps({
                    "image_path": "/some/test.jpg",
                    "persist": "check-copy",
                }),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        session_id = response.json()["session_id"]
        session_dir = cache_dir / session_id
        persist_dir = tmp_path / "output" / "check-copy" / "export"

        for cache_file in session_dir.iterdir():
            persist_file = persist_dir / cache_file.name
            assert persist_file.exists(), f"Missing: {cache_file.name}"
            assert cache_file.read_bytes() == persist_file.read_bytes()

    @pytest.mark.anyio
    async def test_persist_manifest_is_plotter_loadable(
        self, client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The persisted manifest should have all fields required by the plotter."""
        monkeypatch.chdir(tmp_path)

        with _mock_pipeline_stack():
            response = await client.post(
                "/api/generate",
                content=json.dumps({
                    "image_path": "/some/test.jpg",
                    "persist": "plotter-test",
                }),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        persist_dir = tmp_path / "output" / "plotter-test" / "export"
        manifest = json.loads((persist_dir / "manifest.json").read_text())

        # Top-level fields required by parseManifest()
        assert "version" in manifest
        assert "source_image" in manifest
        assert "width" in manifest
        assert "height" in manifest
        assert "created_at" in manifest
        assert "maps" in manifest
        # Map entry structure
        for entry in manifest["maps"]:
            assert "filename" in entry
            assert "key" in entry
            assert "dtype" in entry
            assert "shape" in entry
            assert "value_range" in entry
            assert "description" in entry

    # -- Persistent sessions exempt from TTL ---------------------------------

    def test_persistent_session_exempt_from_ttl(self, tmp_path: Path) -> None:
        """Persistent sessions should NOT be cleaned up by ``cleanup_expired``."""
        config = ServerConfig(cache_dir=tmp_path / "cache", session_ttl_seconds=1)
        cache = SessionCache(config)

        # Create an old but persistent session
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        from portrait_map_lab.server.schemas import SessionInfo as _SI

        info = _SI(
            session_id="persistent-session",
            source_image="test.jpg",
            created_at=old_time,
            map_keys=["density_target"],
            persistent=True,
        )
        cache.register(info)
        session_dir = cache.get_path("persistent-session")
        session_dir.mkdir(parents=True)
        (session_dir / "density_target.bin").write_bytes(b"\x00" * 16)

        removed = cache.cleanup_expired()
        assert removed == 0
        assert session_dir.exists()
        assert len(cache.list_sessions()) == 1

    def test_non_persistent_session_still_cleaned_up(self, tmp_path: Path) -> None:
        """Non-persistent expired sessions should still be cleaned up."""
        config = ServerConfig(cache_dir=tmp_path / "cache", session_ttl_seconds=1)
        cache = SessionCache(config)

        old_time = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        from portrait_map_lab.server.schemas import SessionInfo as _SI

        info = _SI(
            session_id="ephemeral-session",
            source_image="test.jpg",
            created_at=old_time,
            map_keys=["density_target"],
            persistent=False,
        )
        cache.register(info)
        session_dir = cache.get_path("ephemeral-session")
        session_dir.mkdir(parents=True)
        (session_dir / "density_target.bin").write_bytes(b"\x00" * 16)

        removed = cache.cleanup_expired()
        assert removed == 1
        assert not session_dir.exists()

    @pytest.mark.anyio
    async def test_persist_marks_session_persistent(
        self, client: httpx.AsyncClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sessions created with ``persist`` should be marked persistent in the cache."""
        monkeypatch.chdir(tmp_path)

        with _mock_pipeline_stack():
            response = await client.post(
                "/api/generate",
                content=json.dumps({
                    "image_path": "/some/test.jpg",
                    "persist": "mark-test",
                }),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        cache: SessionCache = client._transport.app.state.cache  # type: ignore[attr-defined]
        sessions = cache.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].persistent is True

    @pytest.mark.anyio
    async def test_no_persist_session_not_persistent(
        self, client: httpx.AsyncClient, tmp_path: Path
    ) -> None:
        """Sessions created without ``persist`` should NOT be marked persistent."""
        with _mock_pipeline_stack():
            response = await client.post(
                "/api/generate",
                content=json.dumps({"image_path": "/some/test.jpg"}),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 200

        cache: SessionCache = client._transport.app.state.cache  # type: ignore[attr-defined]
        sessions = cache.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].persistent is False
