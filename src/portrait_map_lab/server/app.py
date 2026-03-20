"""FastAPI application factory for the Map Generation API."""

from __future__ import annotations

from importlib.metadata import version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

__all__ = ["create_app"]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns
    -------
    FastAPI
        Configured application instance with CORS and routes.
    """
    app = FastAPI(
        title="Portrait Map Lab API",
        description="Map Generation API for the portrait map lab pipeline.",
        version=version("portrait-map-lab"),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    app.include_router(router)

    return app
