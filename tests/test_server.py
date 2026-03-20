"""Tests for the Map Generation API server and CLI serve subcommand."""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

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
