"""Disk-backed session cache with TTL-based expiration.

Provides a ``SessionCache`` class that tracks generated map sessions in
memory, persists them to disk, recovers sessions on startup by scanning the
cache directory, and expires stale sessions after a configurable TTL.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import ServerConfig
from .schemas import SessionInfo

__all__ = ["SessionCache"]

logger = logging.getLogger(__name__)


class SessionCache:
    """Thread-safe session registry backed by a disk cache directory.

    On construction the cache directory is scanned for existing sessions
    (subdirectories containing ``manifest.json``), allowing the server to
    recover session metadata across restarts.

    Parameters
    ----------
    config : ServerConfig | None
        Server configuration providing ``cache_dir`` and
        ``session_ttl_seconds``.  When *None*, ``ServerConfig()`` defaults
        are used.
    """

    def __init__(self, config: ServerConfig | None = None) -> None:
        cfg = config or ServerConfig()
        self._cache_dir = cfg.cache_dir
        self._ttl_seconds = cfg.session_ttl_seconds
        self._registry: dict[str, SessionInfo] = {}
        self._lock = threading.Lock()
        self._scan_existing()

    # -- public properties ---------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        """Root directory for cached session data (read-only)."""
        return self._cache_dir

    @property
    def ttl_seconds(self) -> int:
        """Time-to-live for sessions in seconds (read-only)."""
        return self._ttl_seconds

    # -- public methods ------------------------------------------------------

    def register(self, info: SessionInfo) -> None:
        """Add or update a session in the registry.

        Parameters
        ----------
        info : SessionInfo
            Session metadata to store.  The ``session_id`` field is used
            as the registry key.
        """
        with self._lock:
            self._registry[info.session_id] = info
        logger.info("Registered session %s", info.session_id)

    def get_path(self, session_id: str) -> Path:
        """Return the cache directory path for *session_id*.

        This is a pure path computation — the directory may not exist.
        """
        return self._cache_dir / session_id

    def exists(self, session_id: str) -> bool:
        """Check whether a session directory exists on disk."""
        return (self._cache_dir / session_id).is_dir()

    def list_sessions(self) -> list[SessionInfo]:
        """Return a snapshot of all registered sessions."""
        with self._lock:
            return list(self._registry.values())

    def delete(self, session_id: str) -> bool:
        """Remove a session from the registry and delete its directory.

        Returns ``True`` if the session was found (in registry or on disk),
        ``False`` otherwise.
        """
        with self._lock:
            in_registry = self._registry.pop(session_id, None) is not None

        session_dir = self._cache_dir / session_id
        on_disk = self._rmtree_safe(session_dir)

        return in_registry or on_disk

    def cleanup_expired(self) -> int:
        """Remove sessions that have exceeded the TTL.

        Returns the number of sessions removed.
        """
        now = datetime.now(timezone.utc)

        # Snapshot the registry so age computation (which may stat() the
        # filesystem as a fallback) happens outside the lock.
        with self._lock:
            snapshot = list(self._registry.items())

        expired_ids: list[str] = []
        for sid, info in snapshot:
            if info.persistent:
                continue
            age = self._session_age_seconds(sid, info, now)
            if age is not None and age > self._ttl_seconds:
                expired_ids.append(sid)

        if not expired_ids:
            return 0

        with self._lock:
            for sid in expired_ids:
                self._registry.pop(sid, None)

        # Filesystem cleanup outside the lock.
        for sid in expired_ids:
            self._rmtree_safe(self._cache_dir / sid)
            logger.info("Expired session %s", sid)

        logger.info("Cleaned up %d expired session(s)", len(expired_ids))
        return len(expired_ids)

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _dir_mtime_utc(path: Path) -> datetime | None:
        """Return the modification time of *path* as a UTC datetime.

        Returns ``None`` if the path does not exist or is inaccessible.
        """
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except OSError:
            return None

    @staticmethod
    def _rmtree_safe(path: Path) -> bool:
        """Remove a directory tree, returning whether it existed.

        Tolerates concurrent deletion (``FileNotFoundError``) and logs
        a warning for other OS errors instead of propagating them.
        """
        try:
            shutil.rmtree(path)
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("Failed to remove %s: %s", path, exc)
            return True  # Directory existed even if removal failed.

    def _session_age_seconds(
        self,
        session_id: str,
        info: SessionInfo,
        now: datetime,
    ) -> float | None:
        """Compute the age of a session in seconds.

        Parses ``info.created_at`` as ISO 8601.  Falls back to the session
        directory's modification time if parsing fails.  Returns ``None`` if
        the age cannot be determined (session should not be expired).
        """
        # Try parsing the stored ISO timestamp.
        try:
            created = datetime.fromisoformat(info.created_at)
            # Ensure timezone-aware comparison.
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return (now - created).total_seconds()
        except (ValueError, TypeError):
            pass

        # Fallback: directory modification time.
        mtime = self._dir_mtime_utc(self._cache_dir / session_id)
        if mtime is not None:
            return (now - mtime).total_seconds()

        return None

    def _scan_existing(self) -> None:
        """Scan the cache directory and recover sessions from disk.

        Each subdirectory containing a ``manifest.json`` is re-registered
        so that sessions survive server restarts.  Called only from
        ``__init__`` before the instance is shared across threads.
        """
        if not self._cache_dir.is_dir():
            return

        recovered = 0
        for entry in self._cache_dir.iterdir():
            if not entry.is_dir():
                continue
            manifest_path = entry / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text())
            except (json.JSONDecodeError, OSError):
                # Missing or corrupt manifest — skip this directory.
                if manifest_path.exists():
                    logger.warning(
                        "Skipping session %s: could not read manifest",
                        entry.name,
                    )
                continue

            try:
                info = self._info_from_manifest(entry.name, manifest, entry)
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "Skipping session %s: malformed manifest: %s",
                    entry.name,
                    exc,
                )
                continue

            self._registry[info.session_id] = info
            recovered += 1

        if recovered:
            logger.info(
                "Recovered %d session(s) from %s", recovered, self._cache_dir
            )

    @staticmethod
    def _info_from_manifest(
        session_id: str, manifest: dict, session_dir: Path
    ) -> SessionInfo:
        """Build a ``SessionInfo`` from a parsed manifest dict.

        Falls back to the directory modification time when ``created_at``
        is absent from the manifest.

        Raises ``KeyError`` or ``TypeError`` if the manifest structure is
        invalid (e.g. a map entry lacks a ``"key"`` field).
        """
        created_at = manifest.get("created_at", "")
        if not created_at:
            mtime = SessionCache._dir_mtime_utc(session_dir)
            created_at = mtime.isoformat() if mtime is not None else ""

        return SessionInfo(
            session_id=session_id,
            source_image=manifest.get("source_image", ""),
            created_at=created_at,
            map_keys=[m["key"] for m in manifest.get("maps", [])],
        )
