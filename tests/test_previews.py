"""Tests for the preview generation and discovery module."""

from __future__ import annotations

from pathlib import Path

from portrait_map_lab.server.previews import discover_previews
from portrait_map_lab.server.schemas import PreviewInfo


class TestDiscoverPreviews:
    """Verify ``discover_previews`` scans directories correctly."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """An empty previews directory should return no previews."""
        previews_dir = tmp_path / "previews"
        previews_dir.mkdir()
        assert discover_previews(previews_dir) == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """A nonexistent directory should return no previews."""
        assert discover_previews(tmp_path / "does_not_exist") == []

    def test_finds_png_files(self, tmp_path: Path) -> None:
        """Should discover PNG files in category subdirectories."""
        previews_dir = tmp_path / "previews"
        density_dir = previews_dir / "density"
        density_dir.mkdir(parents=True)
        (density_dir / "density_target.png").write_bytes(b"\x89PNG")
        (density_dir / "luminance.png").write_bytes(b"\x89PNG")

        result = discover_previews(previews_dir)
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == {"density_target", "luminance"}
        # All should be in the density category
        assert all(p.category == "density" for p in result)

    def test_excludes_contact_sheets(self, tmp_path: Path) -> None:
        """PNG files with 'contact_sheet' in the name should be excluded."""
        previews_dir = tmp_path / "previews"
        flow_dir = previews_dir / "flow"
        flow_dir.mkdir(parents=True)
        (flow_dir / "etf_coherence.png").write_bytes(b"\x89PNG")
        (flow_dir / "contact_sheet.png").write_bytes(b"\x89PNG")

        result = discover_previews(previews_dir)
        assert len(result) == 1
        assert result[0].name == "etf_coherence"

    def test_url_format(self, tmp_path: Path) -> None:
        """URLs should be relative to the session dir, prefixed with 'previews/'."""
        previews_dir = tmp_path / "previews"
        density_dir = previews_dir / "density"
        density_dir.mkdir(parents=True)
        (density_dir / "density_target.png").write_bytes(b"\x89PNG")

        result = discover_previews(previews_dir)
        assert result[0].url == "previews/density/density_target.png"

    def test_multiple_categories(self, tmp_path: Path) -> None:
        """Should discover PNGs across multiple category subdirectories."""
        previews_dir = tmp_path / "previews"
        for category in ("density", "flow", "features"):
            cat_dir = previews_dir / category
            cat_dir.mkdir(parents=True)
            (cat_dir / f"{category}_preview.png").write_bytes(b"\x89PNG")

        result = discover_previews(previews_dir)
        assert len(result) == 3
        categories = {p.category for p in result}
        assert categories == {"density", "flow", "features"}

    def test_returns_preview_info_type(self, tmp_path: Path) -> None:
        """Results should be PreviewInfo instances."""
        previews_dir = tmp_path / "previews"
        (previews_dir / "density").mkdir(parents=True)
        (previews_dir / "density" / "test.png").write_bytes(b"\x89PNG")

        result = discover_previews(previews_dir)
        assert len(result) == 1
        assert isinstance(result[0], PreviewInfo)

    def test_ignores_non_png_files(self, tmp_path: Path) -> None:
        """Non-PNG files (e.g. .npy) should be ignored."""
        previews_dir = tmp_path / "previews"
        (previews_dir / "features").mkdir(parents=True)
        (previews_dir / "features" / "landmarks.png").write_bytes(b"\x89PNG")
        (previews_dir / "features" / "raw_data.npy").write_bytes(b"\x00" * 10)

        result = discover_previews(previews_dir)
        assert len(result) == 1
        assert result[0].name == "landmarks"
