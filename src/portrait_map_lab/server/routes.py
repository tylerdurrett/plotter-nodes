"""API route handlers for the Map Generation API."""

from __future__ import annotations

from fastapi import APIRouter

__all__ = ["router"]

router = APIRouter(prefix="/api")


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return server health status."""
    return {"status": "ok"}
