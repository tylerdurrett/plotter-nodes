"""API route handlers for the Map Generation API."""

from __future__ import annotations

from fastapi import APIRouter

from .schemas import MAP_KEY_INFOS, MapKeyInfo

__all__ = ["router"]

router = APIRouter(prefix="/api")


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return server health status."""
    return {"status": "ok"}


@router.get("/maps/keys")
def list_map_keys() -> list[MapKeyInfo]:
    """Return metadata for all available map keys."""
    return MAP_KEY_INFOS
