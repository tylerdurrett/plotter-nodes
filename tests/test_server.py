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
