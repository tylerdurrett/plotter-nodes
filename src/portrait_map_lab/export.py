"""Export bundle creation for cross-language consumption.

Builds self-contained export bundles with raw float32 binary maps
and a JSON manifest, designed for easy consumption by TypeScript/JavaScript
via ``new Float32Array(buffer)``.

The core function :func:`build_export_bundle` is a pure function that
returns in-memory data without performing I/O, making it reusable by
both the CLI disk writer and a future HTTP API layer.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from portrait_map_lab.models import ComposedResult, ExportManifest, ExportMapEntry

__all__ = [
    "ExportBundle",
    "build_export_bundle",
    "export_composed_result",
    "save_export_bundle",
]

logger = logging.getLogger(__name__)

# Map definitions: (key, source_path_on_result, value_range, description)
_MAP_DEFINITIONS: list[tuple[str, str, tuple[float, float], str]] = [
    (
        "density_target",
        "density_result.density_target",
        (0.0, 1.0),
        "How dark each region should be — density target for particle placement",
    ),
    (
        "flow_x",
        "flow_result.flow_x",
        (-1.0, 1.0),
        "Flow field X component (unit vectors)",
    ),
    (
        "flow_y",
        "flow_result.flow_y",
        (-1.0, 1.0),
        "Flow field Y component (unit vectors)",
    ),
    (
        "importance",
        "density_result.importance",
        (0.0, 1.0),
        "Feature and contour combined importance for particle attraction bias",
    ),
    (
        "coherence",
        "flow_result.etf.coherence",
        (0.0, 1.0),
        "Flow field confidence and reliability",
    ),
    (
        "complexity",
        "complexity_result.complexity",
        (0.0, 1.0),
        "Local image complexity for speed modulation",
    ),
    (
        "flow_speed",
        "flow_result.flow_speed",
        (0.0, 1.0),
        "Particle speed scalar derived from complexity",
    ),
]


def _resolve_attr(obj: object, dotted_path: str) -> object:
    """Resolve a dotted attribute path like 'flow_result.etf.coherence'."""
    current = obj
    for part in dotted_path.split("."):
        current = getattr(current, part)
    return current


@dataclass(frozen=True, slots=True)
class ExportBundle:
    """In-memory export bundle ready for serialization.

    Attributes
    ----------
    manifest
        Manifest describing the bundle contents and image metadata.
    binary_maps
        Map key to raw float32 bytes, ready for disk write or HTTP streaming.
    png_files
        Relative path (e.g. ``features/contact_sheet.png``) to absolute
        source path on disk. Empty dict when no PNG source directory was given.
    """

    manifest: ExportManifest
    binary_maps: dict[str, bytes]
    png_files: dict[str, Path]


def build_export_bundle(
    result: ComposedResult,
    source_image_name: str,
    png_source_dir: Path | None = None,
) -> ExportBundle:
    """Build an in-memory export bundle from a composed pipeline result.

    This is a pure function (no I/O) so it can be reused by both the CLI
    disk writer and a future HTTP API layer.

    Parameters
    ----------
    result
        Complete composed pipeline result containing all map data.
    source_image_name
        Original image filename (stored in manifest for reference).
    png_source_dir
        If provided, scans this directory recursively for ``.png`` files
        and includes their paths in the bundle for copying during save.

    Returns
    -------
    ExportBundle
        Bundle containing manifest, binary map data, and PNG file references.
    """
    # Extract maps and convert to float32 bytes
    binary_maps: dict[str, bytes] = {}
    map_entries: list[ExportMapEntry] = []
    height: int | None = None
    width: int | None = None

    for key, attr_path, value_range, description in _MAP_DEFINITIONS:
        try:
            array = _resolve_attr(result, attr_path)
            if array is None:
                continue  # Skip None values
            if not isinstance(array, np.ndarray):
                msg = f"Expected ndarray for {attr_path}, got {type(array)}"
                raise TypeError(msg)

            if height is None:
                height, width = array.shape[:2]
            assert height is not None
            assert width is not None

            float32_array = array.astype(np.float32)
            binary_maps[key] = float32_array.tobytes()

            map_entries.append(
                ExportMapEntry(
                    filename=f"{key}.bin",
                    key=key,
                    dtype="float32",
                    shape=(height, width),
                    value_range=value_range,
                    description=description,
                )
            )
        except AttributeError:
            # Skip missing optional fields (e.g., complexity_result is None)
            continue

    assert height is not None
    assert width is not None

    # Build manifest
    manifest = ExportManifest(
        version=1,
        source_image=source_image_name,
        width=width,
        height=height,
        created_at=datetime.now(timezone.utc).isoformat(),
        maps=tuple(map_entries),
    )

    # Collect PNG files if source directory provided
    png_files: dict[str, Path] = {}
    if png_source_dir is not None:
        png_source_dir = Path(png_source_dir)
        if png_source_dir.is_dir():
            for png_path in sorted(png_source_dir.rglob("*.png")):
                relative = str(png_path.relative_to(png_source_dir))
                png_files[relative] = png_path

    logger.info(
        "Built export bundle: %d maps (%dx%d), %d preview PNGs",
        len(map_entries),
        width,
        height,
        len(png_files),
    )

    return ExportBundle(
        manifest=manifest,
        binary_maps=binary_maps,
        png_files=png_files,
    )


def _manifest_to_dict(manifest: ExportManifest) -> dict:
    """Convert an ExportManifest to a JSON-serializable dict.

    Uses ``dataclasses.asdict`` but converts tuples to lists for JSON
    compatibility.
    """
    data = asdict(manifest)
    # Convert shape tuples and value_range tuples to lists for JSON
    for map_entry in data["maps"]:
        map_entry["shape"] = list(map_entry["shape"])
        map_entry["value_range"] = list(map_entry["value_range"])
    return data


def save_export_bundle(
    bundle: ExportBundle,
    output_dir: str | Path,
) -> Path:
    """Write an export bundle to disk.

    Creates the following structure under ``output_dir/export/``::

        export/
            manifest.json
            density_target.bin
            flow_x.bin
            flow_y.bin
            importance.bin
            coherence.bin
            previews/
                features/*.png
                contour/*.png
                density/*.png
                flow/*.png

    Parameters
    ----------
    bundle
        In-memory export bundle to write.
    output_dir
        Base output directory. An ``export/`` subdirectory will be created.

    Returns
    -------
    Path
        Path to the created export directory.
    """
    export_dir = Path(output_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Write binary maps
    for key, data in bundle.binary_maps.items():
        bin_path = export_dir / f"{key}.bin"
        bin_path.write_bytes(data)
        logger.info("Saved binary map: %s (%d bytes)", bin_path.name, len(data))

    # Write manifest
    manifest_path = export_dir / "manifest.json"
    manifest_dict = _manifest_to_dict(bundle.manifest)
    manifest_path.write_text(json.dumps(manifest_dict, indent=2) + "\n")
    logger.info("Saved manifest: %s", manifest_path)

    # Copy PNG previews
    if bundle.png_files:
        previews_dir = export_dir / "previews"
        for relative_path, source_path in bundle.png_files.items():
            dest_path = previews_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
        logger.info("Copied %d preview PNGs to %s", len(bundle.png_files), previews_dir)

    logger.info("Export bundle saved to %s", export_dir)
    return export_dir


def export_composed_result(
    result: ComposedResult,
    source_image_name: str,
    output_dir: str | Path,
    image: np.ndarray | None = None,
) -> Path:
    """High-level convenience: save all outputs then create export bundle.

    Calls :func:`~portrait_map_lab.pipelines.save_all_outputs` to write the
    standard pipeline outputs (PNGs, .npy files), then builds and saves the
    export bundle with previews.

    Parameters
    ----------
    result
        Complete composed pipeline result.
    source_image_name
        Original image filename for the manifest.
    output_dir
        Base output directory.
    image
        Original input image for visualizations. If None, save_all_outputs
        will skip image-dependent visualizations.

    Returns
    -------
    Path
        Path to the created export directory.
    """
    # Lazy import to avoid circular dependency
    from portrait_map_lab.pipelines import save_all_outputs

    output_dir = Path(output_dir)

    # Save standard pipeline outputs (PNGs, .npy files)
    if image is not None:
        save_all_outputs(result, output_dir, image)

    # Build and save export bundle
    bundle = build_export_bundle(result, source_image_name, png_source_dir=output_dir)
    return save_export_bundle(bundle, output_dir)
