"""Tests for the export module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from portrait_map_lab.export import (
    ExportBundle,
    build_export_bundle,
    save_export_bundle,
)
from portrait_map_lab.models import (
    ComplexityResult,
    ComposedResult,
    ContourResult,
    DensityResult,
    ETFResult,
    FlowResult,
    LandmarkResult,
    PipelineResult,
)


def _make_synthetic_composed_result(
    height: int = 48,
    width: int = 64,
    include_complexity: bool = False,
    include_flow_speed: bool = False,
) -> ComposedResult:
    """Build a minimal ComposedResult with synthetic arrays for testing."""
    shape = (height, width)
    landmarks = LandmarkResult(
        landmarks=np.zeros((468, 2), dtype=np.float64),
        image_shape=(height, width),
        confidence=0.95,
    )
    feature_result = PipelineResult(
        landmarks=landmarks,
        masks={"combined_eyes": np.zeros(shape, dtype=np.uint8)},
        distance_fields={"eyes": np.zeros(shape, dtype=np.float64)},
        influence_maps={"eyes": np.zeros(shape, dtype=np.float64)},
        combined=np.random.default_rng(42).random(shape),
    )
    contour_result = ContourResult(
        landmarks=landmarks,
        contour_polygon=np.zeros((10, 2), dtype=np.int32),
        contour_mask=np.zeros(shape, dtype=np.uint8),
        filled_mask=np.zeros(shape, dtype=np.uint8),
        signed_distance=np.zeros(shape, dtype=np.float64),
        directional_distance=np.zeros(shape, dtype=np.float64),
        influence_map=np.zeros(shape, dtype=np.float64),
    )
    rng = np.random.default_rng(42)
    density_result = DensityResult(
        luminance=rng.random(shape),
        clahe_luminance=rng.random(shape),
        tonal_target=rng.random(shape),
        importance=rng.random(shape),
        density_target=rng.random(shape),
    )
    etf_result = ETFResult(
        tangent_x=rng.random(shape),
        tangent_y=rng.random(shape),
        coherence=rng.random(shape),
        gradient_magnitude=rng.random(shape),
    )
    # Create unit-vector flow fields
    angle = rng.random(shape) * 2 * np.pi
    flow_speed = None
    if include_flow_speed:
        flow_speed = rng.random(shape) * 0.7 + 0.3  # Range [0.3, 1.0]

    flow_result = FlowResult(
        etf=etf_result,
        contour_flow_x=np.cos(angle),
        contour_flow_y=np.sin(angle),
        blend_weight=rng.random(shape),
        flow_x=np.cos(angle),
        flow_y=np.sin(angle),
        flow_speed=flow_speed,
    )

    # Add complexity result if requested
    complexity_result = None
    if include_complexity:
        complexity_result = ComplexityResult(
            raw_complexity=rng.random(shape) * 10.0,  # Some raw values
            complexity=rng.random(shape),  # Normalized [0, 1]
            metric="gradient",
        )

    return ComposedResult(
        feature_result=feature_result,
        contour_result=contour_result,
        density_result=density_result,
        flow_result=flow_result,
        lic_image=rng.random(shape),
        complexity_result=complexity_result,
    )


class TestBuildExportBundle:
    """Tests for build_export_bundle pure function."""

    def test_returns_export_bundle(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        assert isinstance(bundle, ExportBundle)

    def test_manifest_has_five_maps(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        assert len(bundle.manifest.maps) == 5

    def test_binary_maps_has_five_keys(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        expected_keys = {"density_target", "flow_x", "flow_y", "importance", "coherence"}
        assert set(bundle.binary_maps.keys()) == expected_keys

    def test_manifest_dimensions_match(self):
        result = _make_synthetic_composed_result(height=48, width=64)
        bundle = build_export_bundle(result, "test.jpg")
        assert bundle.manifest.height == 48
        assert bundle.manifest.width == 64

    def test_manifest_source_image(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "portrait.jpg")
        assert bundle.manifest.source_image == "portrait.jpg"

    def test_manifest_version(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        assert bundle.manifest.version == 1

    def test_manifest_created_at_is_iso8601(self):
        from datetime import datetime

        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        # Should parse without error
        datetime.fromisoformat(bundle.manifest.created_at)

    def test_binary_map_size_matches_dimensions(self):
        """Each binary map should be H * W * 4 bytes (float32)."""
        height, width = 48, 64
        result = _make_synthetic_composed_result(height=height, width=width)
        bundle = build_export_bundle(result, "test.jpg")
        expected_size = height * width * 4
        for key, data in bundle.binary_maps.items():
            assert len(data) == expected_size, f"{key}: expected {expected_size}, got {len(data)}"

    def test_binary_maps_decodable_as_float32(self):
        """Binary data should be decodable as float32 arrays."""
        height, width = 48, 64
        result = _make_synthetic_composed_result(height=height, width=width)
        bundle = build_export_bundle(result, "test.jpg")
        for key, data in bundle.binary_maps.items():
            array = np.frombuffer(data, dtype=np.float32).reshape(height, width)
            assert array.shape == (height, width), f"{key}: wrong shape"
            assert np.all(np.isfinite(array)), f"{key}: contains NaN or Inf"

    def test_binary_map_values_in_declared_range(self):
        """Decoded values should fall within the declared range."""
        height, width = 48, 64
        result = _make_synthetic_composed_result(height=height, width=width)
        bundle = build_export_bundle(result, "test.jpg")
        for entry in bundle.manifest.maps:
            data = bundle.binary_maps[entry.key]
            array = np.frombuffer(data, dtype=np.float32).reshape(height, width)
            lo, hi = entry.value_range
            assert np.all(array >= lo - 1e-6), f"{entry.key}: values below {lo}"
            assert np.all(array <= hi + 1e-6), f"{entry.key}: values above {hi}"

    def test_float32_precision_preserved(self):
        """Decoded bytes should match the original array cast to float32."""
        height, width = 48, 64
        result = _make_synthetic_composed_result(height=height, width=width)
        bundle = build_export_bundle(result, "test.jpg")

        original = result.density_result.density_target.astype(np.float32)
        decoded = np.frombuffer(
            bundle.binary_maps["density_target"], dtype=np.float32
        ).reshape(height, width)
        np.testing.assert_array_equal(original, decoded)

    def test_little_endian_byte_order(self):
        """Bytes should decode correctly as little-endian float32 (TypeScript compat)."""
        height, width = 48, 64
        result = _make_synthetic_composed_result(height=height, width=width)
        bundle = build_export_bundle(result, "test.jpg")

        data = bundle.binary_maps["density_target"]
        # Explicitly decode as little-endian
        array = np.frombuffer(data, dtype="<f4").reshape(height, width)
        original = result.density_result.density_target.astype(np.float32)
        np.testing.assert_array_equal(original, array)

    def test_no_png_files_without_source_dir(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        assert bundle.png_files == {}

    def test_includes_complexity_and_flow_speed_when_present(self):
        """Test that complexity and flow_speed maps are included when available."""
        result = _make_synthetic_composed_result(
            include_complexity=True,
            include_flow_speed=True,
        )
        bundle = build_export_bundle(result, "test.jpg")
        assert len(bundle.manifest.maps) == 7  # 5 base + 2 new
        assert "complexity" in bundle.binary_maps
        assert "flow_speed" in bundle.binary_maps

    def test_excludes_optional_maps_when_absent(self):
        """Test backward compatibility - no complexity means only 5 maps."""
        result = _make_synthetic_composed_result()  # No complexity
        bundle = build_export_bundle(result, "test.jpg")
        assert len(bundle.manifest.maps) == 5  # Original 5 maps only
        assert "complexity" not in bundle.binary_maps
        assert "flow_speed" not in bundle.binary_maps

    def test_partial_optional_maps(self):
        """Test with complexity but no flow_speed."""
        result = _make_synthetic_composed_result(include_complexity=True)
        bundle = build_export_bundle(result, "test.jpg")
        assert len(bundle.manifest.maps) == 6  # 5 base + complexity
        assert "complexity" in bundle.binary_maps
        assert "flow_speed" not in bundle.binary_maps

    def test_collects_png_files_from_source_dir(self):
        result = _make_synthetic_composed_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some dummy PNGs
            features_dir = Path(tmpdir) / "features"
            features_dir.mkdir()
            (features_dir / "contact_sheet.png").write_bytes(b"fake png")
            (features_dir / "landmarks.png").write_bytes(b"fake png")
            density_dir = Path(tmpdir) / "density"
            density_dir.mkdir()
            (density_dir / "density_target.png").write_bytes(b"fake png")

            bundle = build_export_bundle(result, "test.jpg", png_source_dir=Path(tmpdir))
            assert len(bundle.png_files) == 3
            assert "features/contact_sheet.png" in bundle.png_files
            assert "density/density_target.png" in bundle.png_files


class TestManifestJsonRoundtrip:
    """Tests for manifest JSON serialization."""

    def test_manifest_roundtrips_through_json(self):
        from portrait_map_lab.export import _manifest_to_dict

        result = _make_synthetic_composed_result(height=48, width=64)
        bundle = build_export_bundle(result, "test.jpg")

        # Serialize to JSON string and back
        json_str = json.dumps(_manifest_to_dict(bundle.manifest))
        loaded = json.loads(json_str)

        assert loaded["version"] == 1
        assert loaded["source_image"] == "test.jpg"
        assert loaded["width"] == 64
        assert loaded["height"] == 48
        assert len(loaded["maps"]) == 5

    def test_manifest_map_entries_have_all_fields(self):
        from portrait_map_lab.export import _manifest_to_dict

        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        data = _manifest_to_dict(bundle.manifest)

        required_fields = {"filename", "key", "dtype", "shape", "value_range", "description"}
        for entry in data["maps"]:
            assert set(entry.keys()) == required_fields


class TestSaveExportBundle:
    """Tests for save_export_bundle disk writer."""

    def test_creates_export_directory(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = save_export_bundle(bundle, tmpdir)
            assert export_dir.is_dir()
            assert export_dir.name == "export"

    def test_creates_manifest_json(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = save_export_bundle(bundle, tmpdir)
            manifest_path = export_dir / "manifest.json"
            assert manifest_path.exists()
            # Should be valid JSON
            data = json.loads(manifest_path.read_text())
            assert data["version"] == 1

    def test_creates_all_bin_files(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = save_export_bundle(bundle, tmpdir)
            expected_files = [
                "density_target.bin",
                "flow_x.bin",
                "flow_y.bin",
                "importance.bin",
                "coherence.bin",
            ]
            for filename in expected_files:
                assert (export_dir / filename).exists(), f"Missing {filename}"

    def test_bin_files_are_loadable(self):
        height, width = 48, 64
        result = _make_synthetic_composed_result(height=height, width=width)
        bundle = build_export_bundle(result, "test.jpg")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = save_export_bundle(bundle, tmpdir)
            data = (export_dir / "density_target.bin").read_bytes()
            array = np.frombuffer(data, dtype=np.float32).reshape(height, width)
            assert array.shape == (height, width)
            assert np.all(np.isfinite(array))

    def test_copies_png_previews(self):
        result = _make_synthetic_composed_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy PNGs in a source dir
            src_dir = Path(tmpdir) / "src"
            features_dir = src_dir / "features"
            features_dir.mkdir(parents=True)
            (features_dir / "contact_sheet.png").write_bytes(b"fake png data")

            bundle = build_export_bundle(result, "test.jpg", png_source_dir=src_dir)

            out_dir = Path(tmpdir) / "out"
            export_dir = save_export_bundle(bundle, out_dir)
            preview_path = export_dir / "previews" / "features" / "contact_sheet.png"
            assert preview_path.exists()
            assert preview_path.read_bytes() == b"fake png data"

    def test_no_previews_dir_without_pngs(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = save_export_bundle(bundle, tmpdir)
            previews_dir = export_dir / "previews"
            assert not previews_dir.exists()

    def test_returns_export_dir_path(self):
        result = _make_synthetic_composed_result()
        bundle = build_export_bundle(result, "test.jpg")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = save_export_bundle(bundle, tmpdir)
            assert isinstance(export_dir, Path)
            assert export_dir == Path(tmpdir) / "export"

    def test_creates_all_bin_files_with_complexity(self):
        """Test that all 7 bin files are created when complexity is included."""
        result = _make_synthetic_composed_result(
            include_complexity=True,
            include_flow_speed=True,
        )
        bundle = build_export_bundle(result, "test.jpg")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_dir = save_export_bundle(bundle, tmpdir)
            expected_files = [
                "density_target.bin",
                "flow_x.bin",
                "flow_y.bin",
                "importance.bin",
                "coherence.bin",
                "complexity.bin",
                "flow_speed.bin",
            ]
            for filename in expected_files:
                assert (export_dir / filename).exists(), f"Missing {filename}"
