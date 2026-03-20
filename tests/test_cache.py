"""Tests for the disk-backed session cache (Phase 4.1)."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from portrait_map_lab.server.cache import SessionCache
from portrait_map_lab.server.config import ServerConfig
from portrait_map_lab.server.schemas import SessionInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_info(
    session_id: str = "test-session-1",
    source_image: str = "photo.jpg",
    created_at: str | None = None,
    map_keys: list[str] | None = None,
) -> SessionInfo:
    """Build a ``SessionInfo`` with sensible defaults."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    if map_keys is None:
        map_keys = ["density_target", "importance"]
    return SessionInfo(
        session_id=session_id,
        source_image=source_image,
        created_at=created_at,
        map_keys=map_keys,
    )


def _write_manifest(
    session_dir: Path,
    *,
    source_image: str = "test.jpg",
    created_at: str | None = None,
    map_keys: list[str] | None = None,
) -> dict:
    """Write a ``manifest.json`` to *session_dir* and return the dict."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    if map_keys is None:
        map_keys = ["density_target"]

    manifest: dict = {
        "version": 1,
        "source_image": source_image,
        "width": 100,
        "height": 100,
        "created_at": created_at,
        "maps": [
            {
                "filename": f"{key}.bin",
                "key": key,
                "dtype": "float32",
                "shape": [100, 100],
                "value_range": [0.0, 1.0],
                "description": f"test {key}",
            }
            for key in map_keys
        ],
    }
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> SessionCache:
    """Create a ``SessionCache`` backed by a temp directory."""
    config = ServerConfig(cache_dir=tmp_path / "cache")
    return SessionCache(config)


# ---------------------------------------------------------------------------
# Register & list
# ---------------------------------------------------------------------------


class TestRegisterAndList:
    """Verify register/list lifecycle."""

    def test_list_sessions_empty_initially(self, cache: SessionCache) -> None:
        """A fresh cache should have no sessions."""
        assert cache.list_sessions() == []

    def test_register_adds_to_list(self, cache: SessionCache) -> None:
        """Registering a session makes it visible in ``list_sessions``."""
        info = _make_session_info()
        cache.register(info)
        sessions = cache.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == "test-session-1"

    def test_register_multiple_sessions(self, cache: SessionCache) -> None:
        """Multiple registrations should all appear."""
        for i in range(3):
            cache.register(_make_session_info(session_id=f"s-{i}"))
        assert len(cache.list_sessions()) == 3

    def test_register_overwrites_existing(self, cache: SessionCache) -> None:
        """Re-registering with the same ID replaces the entry."""
        cache.register(_make_session_info(source_image="old.jpg"))
        cache.register(_make_session_info(source_image="new.jpg"))
        sessions = cache.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].source_image == "new.jpg"


# ---------------------------------------------------------------------------
# get_path
# ---------------------------------------------------------------------------


class TestGetPath:
    """Verify ``get_path`` behaviour."""

    def test_returns_expected_directory(self, cache: SessionCache) -> None:
        """The returned path should be ``cache_dir / session_id``."""
        path = cache.get_path("abc-123")
        assert path == cache.cache_dir / "abc-123"

    def test_does_not_create_directory(self, cache: SessionCache) -> None:
        """``get_path`` should not create the directory on disk."""
        path = cache.get_path("abc-123")
        assert not path.exists()


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestExists:
    """Verify ``exists`` behaviour."""

    def test_false_when_no_directory(self, cache: SessionCache) -> None:
        """Session that is only in the registry (no directory) is not 'existing'."""
        cache.register(_make_session_info(session_id="s1"))
        assert not cache.exists("s1")

    def test_true_when_directory_exists(self, cache: SessionCache) -> None:
        """A session whose directory is on disk should be reported as existing."""
        cache.get_path("s1").mkdir(parents=True)
        assert cache.exists("s1")

    def test_false_after_delete(self, cache: SessionCache) -> None:
        """After deletion the directory should be gone."""
        session_dir = cache.get_path("s1")
        session_dir.mkdir(parents=True)
        (session_dir / "data.bin").write_bytes(b"\x00" * 16)
        cache.register(_make_session_info(session_id="s1"))
        cache.delete("s1")
        assert not cache.exists("s1")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    """Verify ``delete`` behaviour."""

    def test_removes_from_registry(self, cache: SessionCache) -> None:
        """Deleted session should disappear from ``list_sessions``."""
        cache.register(_make_session_info(session_id="s1"))
        cache.delete("s1")
        assert cache.list_sessions() == []

    def test_removes_directory_from_disk(self, cache: SessionCache) -> None:
        """Deleting should ``shutil.rmtree`` the session directory."""
        session_dir = cache.get_path("s1")
        session_dir.mkdir(parents=True)
        (session_dir / "file.bin").write_bytes(b"\x00")
        cache.register(_make_session_info(session_id="s1"))
        cache.delete("s1")
        assert not session_dir.exists()

    def test_returns_true_when_in_registry(self, cache: SessionCache) -> None:
        """``delete`` returns ``True`` when the session was registered."""
        cache.register(_make_session_info(session_id="s1"))
        assert cache.delete("s1") is True

    def test_returns_true_when_only_on_disk(self, cache: SessionCache) -> None:
        """``delete`` returns ``True`` for an unregistered session with a directory."""
        session_dir = cache.get_path("disk-only")
        session_dir.mkdir(parents=True)
        (session_dir / "manifest.json").write_text("{}")
        assert cache.delete("disk-only") is True
        assert not session_dir.exists()

    def test_returns_false_when_not_found(self, cache: SessionCache) -> None:
        """``delete`` returns ``False`` for a completely unknown session."""
        assert cache.delete("nonexistent") is False

    def test_handles_registry_only(self, cache: SessionCache) -> None:
        """Deleting a registry-only session (no directory) should succeed."""
        cache.register(_make_session_info(session_id="reg-only"))
        assert cache.delete("reg-only") is True
        assert cache.list_sessions() == []


# ---------------------------------------------------------------------------
# cleanup_expired
# ---------------------------------------------------------------------------


class TestCleanupExpired:
    """Verify TTL-based expiration."""

    def test_removes_old_sessions(self, tmp_path: Path) -> None:
        """Sessions older than TTL should be removed."""
        config = ServerConfig(cache_dir=tmp_path / "cache", session_ttl_seconds=60)
        cache = SessionCache(config)
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        cache.register(_make_session_info(session_id="old", created_at=old_time))
        removed = cache.cleanup_expired()
        assert removed == 1
        assert cache.list_sessions() == []

    def test_keeps_recent_sessions(self, tmp_path: Path) -> None:
        """Sessions within TTL should survive cleanup."""
        config = ServerConfig(cache_dir=tmp_path / "cache", session_ttl_seconds=1800)
        cache = SessionCache(config)
        recent_time = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        cache.register(_make_session_info(session_id="recent", created_at=recent_time))
        removed = cache.cleanup_expired()
        assert removed == 0
        assert len(cache.list_sessions()) == 1

    def test_deletes_directory(self, tmp_path: Path) -> None:
        """Expired session directories should be removed from disk."""
        config = ServerConfig(cache_dir=tmp_path / "cache", session_ttl_seconds=60)
        cache = SessionCache(config)
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        cache.register(_make_session_info(session_id="old", created_at=old_time))
        session_dir = cache.get_path("old")
        session_dir.mkdir(parents=True)
        (session_dir / "data.bin").write_bytes(b"\x00")

        cache.cleanup_expired()
        assert not session_dir.exists()

    def test_returns_count(self, tmp_path: Path) -> None:
        """``cleanup_expired`` should return the number of removed sessions."""
        config = ServerConfig(cache_dir=tmp_path / "cache", session_ttl_seconds=60)
        cache = SessionCache(config)
        old_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        recent_time = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

        cache.register(_make_session_info(session_id="old1", created_at=old_time))
        cache.register(_make_session_info(session_id="old2", created_at=old_time))
        cache.register(_make_session_info(session_id="fresh", created_at=recent_time))

        assert cache.cleanup_expired() == 2
        assert len(cache.list_sessions()) == 1
        assert cache.list_sessions()[0].session_id == "fresh"

    def test_handles_unparseable_created_at(self, tmp_path: Path) -> None:
        """Fall back to directory mtime when ``created_at`` is not valid ISO."""
        config = ServerConfig(cache_dir=tmp_path / "cache", session_ttl_seconds=60)
        cache = SessionCache(config)

        cache.register(
            _make_session_info(session_id="bad-ts", created_at="not-a-date")
        )
        # Create the directory and set its mtime to 2 minutes ago.
        session_dir = cache.get_path("bad-ts")
        session_dir.mkdir(parents=True)
        old_mtime = time.time() - 120
        os.utime(session_dir, (old_mtime, old_mtime))

        assert cache.cleanup_expired() == 1
        assert cache.list_sessions() == []


# ---------------------------------------------------------------------------
# Startup scan
# ---------------------------------------------------------------------------


class TestStartupScan:
    """Verify that ``SessionCache`` recovers sessions from disk on init."""

    def test_recovers_sessions_from_disk(self, tmp_path: Path) -> None:
        """Directories with manifest.json should be recovered on startup."""
        cache_dir = tmp_path / "cache"
        _write_manifest(cache_dir / "sess-a", source_image="a.jpg")
        _write_manifest(cache_dir / "sess-b", source_image="b.jpg")

        cache = SessionCache(ServerConfig(cache_dir=cache_dir))
        ids = {s.session_id for s in cache.list_sessions()}
        assert ids == {"sess-a", "sess-b"}

    def test_ignores_directories_without_manifest(self, tmp_path: Path) -> None:
        """Directories without ``manifest.json`` should be skipped."""
        cache_dir = tmp_path / "cache"
        (cache_dir / "no-manifest").mkdir(parents=True)
        _write_manifest(cache_dir / "with-manifest")

        cache = SessionCache(ServerConfig(cache_dir=cache_dir))
        assert len(cache.list_sessions()) == 1
        assert cache.list_sessions()[0].session_id == "with-manifest"

    def test_reconstructs_session_info_correctly(self, tmp_path: Path) -> None:
        """Recovered ``SessionInfo`` should match the manifest contents."""
        cache_dir = tmp_path / "cache"
        ts = "2026-03-20T12:00:00+00:00"
        keys = ["density_target", "flow_x"]
        _write_manifest(
            cache_dir / "precise",
            source_image="portrait.jpg",
            created_at=ts,
            map_keys=keys,
        )

        cache = SessionCache(ServerConfig(cache_dir=cache_dir))
        info = cache.list_sessions()[0]
        assert info.session_id == "precise"
        assert info.source_image == "portrait.jpg"
        assert info.created_at == ts
        assert info.map_keys == keys

    def test_uses_mtime_when_created_at_missing(self, tmp_path: Path) -> None:
        """When ``created_at`` is absent, the directory mtime should be used."""
        cache_dir = tmp_path / "cache"
        session_dir = cache_dir / "no-ts"
        _write_manifest(session_dir, created_at="")

        # The helper writes "" for created_at — _info_from_manifest should
        # fall back to mtime.  Verify a timestamp was populated.
        cache = SessionCache(ServerConfig(cache_dir=cache_dir))
        info = cache.list_sessions()[0]
        assert info.created_at != ""
        # It should be a parseable ISO timestamp.
        datetime.fromisoformat(info.created_at)

    def test_handles_empty_cache_dir(self, tmp_path: Path) -> None:
        """An empty cache directory should produce an empty session list."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache = SessionCache(ServerConfig(cache_dir=cache_dir))
        assert cache.list_sessions() == []

    def test_handles_nonexistent_cache_dir(self, tmp_path: Path) -> None:
        """A nonexistent cache directory should not raise."""
        cache_dir = tmp_path / "does-not-exist"
        cache = SessionCache(ServerConfig(cache_dir=cache_dir))
        assert cache.list_sessions() == []

    def test_skips_corrupt_manifest(self, tmp_path: Path) -> None:
        """Directories with unparseable JSON should be skipped with a warning."""
        cache_dir = tmp_path / "cache"

        # Good session.
        _write_manifest(cache_dir / "good")

        # Corrupt session.
        corrupt_dir = cache_dir / "corrupt"
        corrupt_dir.mkdir(parents=True)
        (corrupt_dir / "manifest.json").write_text("{invalid json!!")

        cache = SessionCache(ServerConfig(cache_dir=cache_dir))
        assert len(cache.list_sessions()) == 1
        assert cache.list_sessions()[0].session_id == "good"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Basic concurrency sanity check."""

    def test_concurrent_register_and_list(self, cache: SessionCache) -> None:
        """Concurrent registrations from multiple threads should not lose data."""
        n_sessions = 100

        def _register(i: int) -> None:
            cache.register(_make_session_info(session_id=f"t-{i}"))

        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(_register, range(n_sessions)))

        assert len(cache.list_sessions()) == n_sessions
