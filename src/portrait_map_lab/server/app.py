"""FastAPI application factory for the Map Generation API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .cache import SessionCache
from .config import ServerConfig
from .routes import router

__all__ = ["create_app"]

logger = logging.getLogger(__name__)

_CLEANUP_INTERVAL_SECONDS = 60


async def _cleanup_loop(cache: SessionCache) -> None:
    """Periodically run TTL-based session cleanup."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
        try:
            removed = await asyncio.to_thread(cache.cleanup_expired)
            if removed:
                logger.info("Background cleanup removed %d session(s)", removed)
        except Exception:
            logger.exception("Error during background session cleanup")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    On startup, initialises the :class:`SessionCache` (which scans the cache
    directory for existing sessions) and starts a background cleanup task.
    On shutdown, cancels the cleanup task cleanly.
    """
    config: ServerConfig = getattr(app.state, "_server_config", None) or ServerConfig()
    cache = SessionCache(config)
    app.state.cache = cache

    cleanup_task = asyncio.create_task(_cleanup_loop(cache))
    logger.info(
        "Server started — cache dir: %s, TTL: %ds",
        cache.cache_dir,
        cache.ttl_seconds,
    )
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Server shutdown — cleanup task cancelled")


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    config : ServerConfig | None
        Server configuration.  When *None*, :class:`ServerConfig` defaults
        are used.  The config is stored on ``app.state._server_config`` so
        the lifespan handler can pick it up.

    Returns
    -------
    FastAPI
        Configured application instance with CORS, routes, and lifespan.
    """
    app = FastAPI(
        title="Portrait Map Lab API",
        description="Map Generation API for the portrait map lab pipeline.",
        version=version("portrait-map-lab"),
        lifespan=_lifespan,
    )

    if config is not None:
        app.state._server_config = config

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    app.include_router(router)

    return app
