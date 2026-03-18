# Stage 12: Export Bundle

## Overview

The export bundle stage packages the essential algorithm inputs into a cross-language format designed for consumption by the companion TypeScript/React/Vite plotter repo. It produces raw float32 binary files with a JSON manifest, alongside PNG previews of all pipeline visualizations.

## Purpose

The Python repo generates maps and fields that drive the drawing algorithm in the TypeScript repo. The export bundle bridges this gap by:

- **Converting to a TypeScript-friendly format**: Raw float32 binary files readable via `new Float32Array(buffer)` — no parsing libraries needed
- **Selecting algorithm-relevant maps**: Only the 5 maps the drawing algorithm actually needs, not all intermediates
- **Including visual reference**: PNG previews for human inspection alongside the binary data
- **Documenting the payload**: A JSON manifest with dimensions, value ranges, and descriptions

## Architecture

The export system separates bundle *building* from bundle *writing*:

```
ComposedResult
       │
       v
build_export_bundle()    ← Pure function, no I/O
       │
       v
  ExportBundle           ← In-memory: manifest + binary maps + PNG refs
       │
       ├──→ save_export_bundle()     ← Disk writer (CLI)
       │
       └──→ Future HTTP API          ← Stream bytes directly
```

This separation means the same `build_export_bundle()` function can be reused by both the CLI (writes to disk) and a future API layer (streams over HTTP).

## Exported Maps

| Map | Source | Value Range | Description |
|-----|--------|-------------|-------------|
| `density_target` | Density pipeline | [0, 1] | How dark each region should be |
| `flow_x` | Flow pipeline | [-1, 1] | Flow field X component (unit vectors) |
| `flow_y` | Flow pipeline | [-1, 1] | Flow field Y component (unit vectors) |
| `importance` | Density pipeline | [0, 1] | Feature + contour combined importance |
| `coherence` | ETF pipeline | [0, 1] | Flow field confidence/reliability |

## Binary Format Specification

Each `.bin` file contains:
- **Encoding**: Raw IEEE 754 float32 values
- **Byte order**: Little-endian (native to x86/ARM and JavaScript)
- **Layout**: Row-major (C order), no header or padding
- **Size**: `height * width * 4` bytes per file

To decode in TypeScript/JavaScript:
```typescript
const buf = fs.readFileSync("density_target.bin");
const data = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
// Access pixel at (x, y): data[y * width + x]
```

To decode in Python (for verification):
```python
array = np.fromfile("density_target.bin", dtype=np.float32).reshape(height, width)
```

## Manifest Schema

The `manifest.json` file describes the bundle contents:

```json
{
  "version": 1,
  "source_image": "portrait.jpg",
  "width": 640,
  "height": 480,
  "created_at": "2026-03-18T14:30:00+00:00",
  "maps": [
    {
      "filename": "density_target.bin",
      "key": "density_target",
      "dtype": "float32",
      "shape": [480, 640],
      "value_range": [0.0, 1.0],
      "description": "How dark each region should be — density target for particle placement"
    }
  ]
}
```

### Manifest Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | int | Schema version (currently 1) |
| `source_image` | string | Original image filename |
| `width` | int | Image width in pixels |
| `height` | int | Image height in pixels |
| `created_at` | string | ISO 8601 UTC timestamp |
| `maps` | array | Array of map entry objects |

### Map Entry Fields

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Binary file name (e.g., `density_target.bin`) |
| `key` | string | Programmatic key for lookup |
| `dtype` | string | Data type (`float32`) |
| `shape` | [int, int] | Array dimensions `[height, width]` |
| `value_range` | [float, float] | Expected value range |
| `description` | string | Human-readable description |

## Output Structure

```
output/<image_name>/
  features/         (existing pipeline outputs)
  contour/          (existing pipeline outputs)
  density/          (existing pipeline outputs)
  flow/             (existing pipeline outputs)
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
```

## Usage

### Python API

```python
from portrait_map_lab import (
    load_image, run_all_pipelines,
    build_export_bundle, save_export_bundle,
)

image = load_image("portrait.jpg")
result = run_all_pipelines(image)

# Build bundle (pure, no I/O)
bundle = build_export_bundle(result, "portrait.jpg")

# Write to disk
export_dir = save_export_bundle(bundle, "output/portrait")
```

### Convenience Function

```python
from portrait_map_lab import load_image, run_all_pipelines, export_composed_result

image = load_image("portrait.jpg")
result = run_all_pipelines(image)

# Saves all outputs + creates export bundle in one call
export_dir = export_composed_result(result, "portrait.jpg", "output/portrait", image)
```

### CLI

```bash
# Run all pipelines and generate export bundle
uv run python scripts/run_pipeline.py all portrait.jpg --export

# With custom configuration
uv run python scripts/run_pipeline.py all portrait.jpg --export --gamma 0.8 --clip-limit 3.0
```

## Configuration

The export stage has no configuration parameters — it packages whatever the upstream pipelines produce. To change the exported maps, configure the density, flow, and feature pipelines.

## Data Models

```python
@dataclass(frozen=True, slots=True)
class ExportMapEntry:
    filename: str              # "density_target.bin"
    key: str                   # "density_target"
    dtype: str                 # "float32"
    shape: tuple[int, int]     # (height, width)
    value_range: tuple[float, float]  # (0.0, 1.0)
    description: str

@dataclass(frozen=True, slots=True)
class ExportManifest:
    version: int
    source_image: str
    width: int
    height: int
    created_at: str            # ISO 8601
    maps: tuple[ExportMapEntry, ...]

@dataclass(frozen=True, slots=True)
class ExportBundle:
    manifest: ExportManifest
    binary_maps: dict[str, bytes]    # key → raw float32 bytes
    png_files: dict[str, Path]       # relative path → absolute source
```

## Next Stage

The export bundle is consumed by the TypeScript plotter repo to drive the continuous-line drawing algorithm. The drawing system uses:

- **density_target** to know how much line density each region needs
- **flow_x/flow_y** to guide particle movement direction
- **importance** to bias particle attraction toward key features
- **coherence** to modulate randomness vs. structure-following behavior
