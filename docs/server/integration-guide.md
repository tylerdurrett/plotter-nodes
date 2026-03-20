# Map Generation API — Integration Guide

## Overview

The Map Generation API exposes the portrait map lab pipeline as an HTTP service, allowing client applications (such as the companion TypeScript plotter) to generate and fetch importance maps without running Python locally. The API accepts portrait images, runs the configured pipelines, and serves the resulting float32 binary maps via a JSON manifest — the same export bundle format used by the CLI.

## Starting the Server

```bash
# Start with default settings (127.0.0.1:8100)
uv run python scripts/run_pipeline.py serve

# Custom host and port
uv run python scripts/run_pipeline.py serve --host 0.0.0.0 --port 9000
```

The server requires the optional `api` dependencies:

```bash
uv pip install -e ".[api]"
```

Without these, the `serve` command prints an error and exits with code 1.

### Default Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `host` | `127.0.0.1` | Bind address |
| `port` | `8100` | Listen port |
| `cache_dir` | `.cache/api` | Session cache directory |
| `session_ttl_seconds` | `1800` | Session lifetime (30 minutes) |

## Health Check

```bash
curl http://127.0.0.1:8100/api/health
```

```json
{"status": "ok"}
```

Use this endpoint to verify the server is running before sending generation requests.

## Discovering Available Maps

```bash
curl http://127.0.0.1:8100/api/maps/keys
```

Returns metadata for every map the pipeline can produce:

```json
[
  {
    "key": "density_target",
    "value_range": [0.0, 1.0],
    "description": "How dark each region should be — density target for particle placement"
  },
  {
    "key": "flow_x",
    "value_range": [-1.0, 1.0],
    "description": "Flow field X component (unit vectors)"
  },
  {
    "key": "flow_y",
    "value_range": [-1.0, 1.0],
    "description": "Flow field Y component (unit vectors)"
  },
  {
    "key": "importance",
    "value_range": [0.0, 1.0],
    "description": "Feature and contour combined importance for particle attraction bias"
  },
  {
    "key": "coherence",
    "value_range": [0.0, 1.0],
    "description": "Flow field confidence and reliability"
  },
  {
    "key": "complexity",
    "value_range": [0.0, 1.0],
    "description": "Local image complexity for speed modulation"
  },
  {
    "key": "flow_speed",
    "value_range": [0.0, 1.0],
    "description": "Particle speed scalar derived from complexity"
  }
]
```

## Generating Maps

### With an Image Path (JSON)

Send a JSON body with `image_path` pointing to a file accessible to the server:

```bash
curl -X POST http://127.0.0.1:8100/api/generate \
  -H "Content-Type: application/json" \
  -d '{"image_path": "path/to/portrait.jpg"}'
```

### With a File Upload (Multipart)

Upload an image directly via multipart form data:

```bash
curl -X POST http://127.0.0.1:8100/api/generate \
  -F "image=@portrait.jpg"
```

To include request options with a file upload, pass them as a `request_body` form field:

```bash
curl -X POST http://127.0.0.1:8100/api/generate \
  -F "image=@portrait.jpg" \
  -F 'request_body={"maps": ["density_target", "flow_x", "flow_y"]}'
```

### Response Format

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "manifest": {
    "version": 1,
    "source_image": "portrait.jpg",
    "width": 640,
    "height": 480,
    "created_at": "2026-03-20T14:30:00+00:00",
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
  },
  "base_url": "/api/maps/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

The `manifest` is embedded directly in the response so clients can begin processing immediately. The `base_url` is the prefix for fetching individual map files.

### Requesting Specific Maps

By default, all 7 maps are generated. To request a subset, pass the `maps` array. The server automatically resolves which pipelines to run — requesting fewer maps can be significantly faster.

```bash
# Only density and flow fields (skips complexity pipeline)
curl -X POST http://127.0.0.1:8100/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "portrait.jpg",
    "maps": ["density_target", "flow_x", "flow_y"]
  }'
```

#### Pipeline Dependency Table

The resolver determines which pipelines to run based on requested maps:

| Map Key | Required Pipelines |
|---------|--------------------|
| `density_target` | features, contour, density |
| `importance` | features, contour, density |
| `flow_x` | contour, flow |
| `flow_y` | contour, flow |
| `coherence` | contour, flow |
| `complexity` | complexity |
| `flow_speed` | complexity, contour, flow |

Requesting `["complexity"]` runs only the complexity pipeline. Requesting `["density_target", "flow_speed"]` runs features, contour, density, complexity, and flow.

### Error Responses

| Status | Condition | Example Detail |
|--------|-----------|----------------|
| 422 | No image provided | `"No image provided. Supply 'image_path' in JSON or upload a file."` |
| 422 | Image not found | `"Image not found: path/to/missing.jpg"` |
| 422 | No face detected | `"No face detected in image"` |
| 422 | Invalid map key | `"Invalid map key(s): ['invalid']. Valid keys: [...]"` |
| 422 | Invalid persist value | Pydantic validation error |

## Fetching Map Files

After generation, fetch individual files using the `base_url` from the response:

```bash
# Fetch the manifest
curl http://127.0.0.1:8100/api/maps/{session_id}/manifest.json

# Fetch a binary map
curl -o density_target.bin \
  http://127.0.0.1:8100/api/maps/{session_id}/density_target.bin
```

### Binary Map Format

Each `.bin` file contains raw IEEE 754 float32 values in little-endian byte order, row-major (C order), with no header or padding. File size is `height * width * 4` bytes.

To decode in TypeScript/JavaScript:

```typescript
const response = await fetch(`${baseUrl}/density_target.bin`);
const buffer = await response.arrayBuffer();
const data = new Float32Array(buffer);
// Access pixel at (x, y): data[y * width + x]
```

To decode in Python:

```python
import numpy as np
array = np.fromfile("density_target.bin", dtype=np.float32).reshape(height, width)
```

## Plotter Integration Pattern

The response format is designed to map directly to the plotter's `MapBundle.load()` pattern:

1. **`POST /api/generate`** returns `base_url` and `manifest`
2. **`manifest`** contains `maps[]` with `filename`, `key`, `dtype`, `shape`, `value_range`
3. **Binary files** are at `{base_url}/{filename}`

```typescript
// 1. Generate maps
const res = await fetch("http://127.0.0.1:8100/api/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ image_path: "portrait.jpg" }),
});
const { base_url, manifest } = await res.json();

// 2. base_url → pass to MapBundle.load() as baseUrl
// 3. manifest.json at {base_url}/manifest.json → fetched automatically
// 4. .bin files at {base_url}/{filename} → fetched on demand

// Example: load a specific map
const fullBaseUrl = `http://127.0.0.1:8100${base_url}`;
for (const entry of manifest.maps) {
  const binRes = await fetch(`${fullBaseUrl}/${entry.filename}`);
  const buffer = await binRes.arrayBuffer();
  const data = new Float32Array(buffer);
  // data.length === entry.shape[0] * entry.shape[1]
}
```

The `base_url` in the response is a relative path (e.g., `/api/maps/{session_id}`). Prepend the server origin to form the full URL.

## Config Overrides

All pipeline parameters can be overridden via the `config` object in the request body. Only include the fields you want to change — omitted fields keep their pipeline defaults.

### Features Pipeline

Controls facial feature influence mapping (eyes and mouth).

```json
{
  "image_path": "portrait.jpg",
  "config": {
    "features": {
      "weights": {"eyes": 0.8, "mouth": 0.2},
      "remap": {
        "curve": "gaussian",
        "sigma": 100.0,
        "clamp_distance": 400.0
      }
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `{string: float}` | Feature weights (e.g., `{"eyes": 0.6, "mouth": 0.4}`) |
| `remap.curve` | string | Remapping curve: `"linear"`, `"gaussian"`, `"exponential"` |
| `remap.sigma` | float | Gaussian spread parameter |
| `remap.radius` | float | Linear curve maximum radius |
| `remap.tau` | float | Exponential decay rate |
| `remap.clamp_distance` | float | Maximum distance before clamping |

### Contour Pipeline

Controls face boundary distance mapping.

```json
{
  "image_path": "portrait.jpg",
  "config": {
    "contour": {
      "direction": "inward",
      "contour_thickness": 2,
      "remap": {
        "curve": "exponential",
        "tau": 80.0
      }
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `contour_method` | string | Contour detection method |
| `direction` | string | `"inward"`, `"outward"`, `"both"`, `"band"` |
| `band_width` | float | Band width for band mode (pixels) |
| `contour_thickness` | int | Contour line thickness |
| `epsilon_factor` | float | Contour approximation factor |
| `smooth_contour` | bool | Enable contour smoothing |
| `remap.*` | | Same remap fields as features |

### Density Pipeline

Controls density composition from features, contour, and tonal information.

```json
{
  "image_path": "portrait.jpg",
  "config": {
    "density": {
      "gamma": 0.8,
      "feature_weight": 0.5,
      "contour_weight": 0.3,
      "luminance": {
        "clip_limit": 3.0,
        "tile_size": 8
      }
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `feature_weight` | float | Weight for feature importance |
| `contour_weight` | float | Weight for contour importance |
| `tonal_blend_mode` | string | Tonal blending mode |
| `tonal_weight` | float | Weight for tonal component |
| `importance_weight` | float | Weight for importance component |
| `gamma` | float | Gamma correction |
| `luminance.clip_limit` | float | CLAHE clip limit |
| `luminance.tile_size` | int | CLAHE tile size |

### Complexity Pipeline

Controls local complexity measurement for speed modulation.

```json
{
  "image_path": "portrait.jpg",
  "config": {
    "complexity": {
      "metric": "gradient",
      "sigma": 3.0,
      "normalize_percentile": 99.0
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `metric` | string | `"gradient"`, `"laplacian"`, `"multiscale_gradient"` |
| `sigma` | float | Gaussian smoothing sigma |
| `scales` | float[] | Scales for multiscale metric |
| `scale_weights` | float[] | Weights per scale |
| `normalize_percentile` | float | Robust normalization percentile |

### Flow Pipeline

Controls edge tangent flow field computation.

```json
{
  "image_path": "portrait.jpg",
  "config": {
    "flow": {
      "blend_mode": "slerp",
      "coherence_power": 2.0,
      "etf": {
        "blur_sigma": 5.0,
        "refine_iterations": 4
      }
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `contour_smooth_sigma` | float | Contour flow smoothing |
| `blend_mode` | string | Flow blending mode |
| `coherence_power` | float | Coherence exponent |
| `fallback_threshold` | float | Fallback threshold |
| `etf.blur_sigma` | float | ETF blur sigma |
| `etf.structure_sigma` | float | Structure tensor sigma |
| `etf.refine_sigma` | float | Refinement sigma |
| `etf.refine_iterations` | int | Number of refinement iterations |
| `etf.sobel_ksize` | int | Sobel kernel size |

### Flow Speed

Controls how complexity maps to particle speed.

```json
{
  "image_path": "portrait.jpg",
  "config": {
    "flow_speed": {
      "speed_min": 0.2,
      "speed_max": 1.0
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `speed_min` | float | Minimum particle speed |
| `speed_max` | float | Maximum particle speed |

## Persisting Bundles

To save the generated maps to a permanent output directory (outside the cache), include the `persist` parameter:

```bash
curl -X POST http://127.0.0.1:8100/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "portrait.jpg",
    "persist": "my-portrait"
  }'
```

This copies the session files to `output/my-portrait/export/`, creating a durable bundle identical to what the CLI produces with `--export`. The persist value must match `[a-zA-Z0-9_-]+`.

Persisted sessions are also exempt from automatic TTL cleanup.

## Session Lifecycle and TTL

### Session Creation

Each `POST /api/generate` request creates a new session with a UUID. Session data (manifest and `.bin` files) is stored in `.cache/api/{session_id}/`.

### Listing Sessions

```bash
curl http://127.0.0.1:8100/api/sessions
```

```json
[
  {
    "session_id": "a1b2c3d4-...",
    "source_image": "portrait.jpg",
    "created_at": "2026-03-20T14:30:00+00:00",
    "map_keys": ["density_target", "flow_x", "flow_y", "importance", "coherence", "complexity", "flow_speed"],
    "persistent": false
  }
]
```

### Deleting a Session

```bash
curl -X DELETE http://127.0.0.1:8100/api/maps/{session_id}
# Returns 204 No Content on success, 404 if not found
```

### Automatic TTL Cleanup

A background task runs every 60 seconds and removes sessions older than the configured TTL (default: 30 minutes). Sessions marked as `persistent` (created with the `persist` parameter) are exempt from TTL cleanup.

### Server Restart Recovery

On startup, the server scans the `.cache/api/` directory and re-registers any sessions that contain a valid `manifest.json`. This means sessions survive server restarts. Note: the `persistent` flag is not recovered — only the `output/` copy is the durable artifact.

## Interactive API Documentation

FastAPI automatically generates interactive API documentation available at:

- **Swagger UI**: `http://127.0.0.1:8100/docs`
- **ReDoc**: `http://127.0.0.1:8100/redoc`

These pages show all endpoints, request/response schemas, and allow testing requests directly from the browser.

## Concurrency

Pipeline execution is serialized with a threading lock — only one image can be processed at a time. File-serving requests (`GET /api/maps/...`) are not blocked and can be served concurrently while a generation is in progress.

## CORS

The server allows all origins (`*`) by default, so browser-based clients can call the API directly without proxy configuration.
