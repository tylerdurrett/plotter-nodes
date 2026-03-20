"""Server configuration for the Map Generation API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ServerConfig"]


@dataclass(slots=True)
class ServerConfig:
    """Configuration for the API server.

    Parameters
    ----------
    host : str
        Bind address for the server.
    port : int
        Port number for the server.
    cache_dir : Path
        Directory for disk-backed session cache.
    session_ttl_seconds : int
        Time-to-live for cached sessions in seconds.
    """

    host: str = "127.0.0.1"
    port: int = 8100
    cache_dir: Path = Path(".cache/api")
    session_ttl_seconds: int = 1800
