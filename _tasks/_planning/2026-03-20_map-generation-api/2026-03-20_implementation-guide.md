# Implementation Guide: Map Generation API

**Date:** 2026-03-20
**Feature:** Map Generation API
**Source:** [2026-03-20_feature-description.md](2026-03-20_feature-description.md)

## Overview

This guide implements a FastAPI server that exposes the existing pipeline functions as an HTTP API. The approach prioritizes getting a working end-to-end request/response loop first (Phase 1-2), then layering in the smart features: granular map selection with dependency resolution (Phase 3), disk-backed caching with TTL (Phase 4), and documentation (Phase 5).

Key sequencing decisions:
- **Phase 1 builds the skeleton**: dependencies, module structure, FastAPI app, and the `serve` subcommand — testable with a health check before any pipeline code is wired in.
- **Phase 2 wires up full pipeline generation**: the simplest useful version — accept an image, run all pipelines, return maps. This validates the core integration before adding complexity.
- **Phase 3 adds the pipeline resolver**: granular map selection and the dependency table that determines which pipelines to run. Separated from Phase 2 because the resolver is new logic that needs its own testing.
- **Phase 4 adds the disk-backed cache**: session management, TTL eviction, file serving. Deferred because Phases 1-3 work fine by generating and returning maps synchronously — caching is an optimization, not a prerequisite.
- **Phase 5 is documentation and the persist feature**: integration guide, gitignore, and the `persist` parameter. Last because the API contract must be stable before documenting it.

The private helpers `_run_feature_pipeline_with_landmarks` and `_run_contour_pipeline_with_landmarks` need to be promoted to public functions so the API's pipeline resolver can call individual pipelines with shared landmarks. This is a small, safe refactor in Phase 3.

## File Structure

```
src/portrait_map_lab/
├── server/                     # New package
│   ├── __init__.py
│   ├── app.py                  # FastAPI application, lifespan, CORS
│   ├── routes.py               # Endpoint handlers
│   ├── schemas.py              # Pydantic request/response models
│   ├── resolver.py             # Map-to-pipeline dependency resolver
│   ├── cache.py                # Disk-backed session cache + TTL
│   └── config.py               # Server configuration (port, host, TTL, etc.)
├── pipelines.py                # (modified) Promote _with_landmarks helpers to public
├── export.py                   # (unchanged) Reused for serialization
├── ...
scripts/
├── run_pipeline.py             # (modified) Add `serve` subcommand
tests/
├── test_server.py              # API endpoint tests
├── test_resolver.py            # Pipeline dependency resolver tests
├── test_cache.py               # Session cache tests
docs/
├── server/
│   └── integration-guide.md    # Client integration documentation
.gitignore                      # (modified) Add .cache/
pyproject.toml                  # (modified) Add [api] optional dependency
```

---

## Phase 1: Bootstrap — Dependencies, Module Skeleton, Serve Command

**Purpose:** Get a running FastAPI server reachable via the existing CLI, with zero pipeline logic.

**Rationale:** Validates the packaging, optional dependency handling, and CLI integration before introducing any complexity. Every subsequent phase builds on this working foundation.

### 1.1 Add Optional Dependencies

- [x] Add `[project.optional-dependencies] api` section to `pyproject.toml` with `fastapi>=0.115`, `uvicorn>=0.34`, `python-multipart>=0.0.9`
- [x] Add `httpx>=0.28` to the `dev` optional dependencies (for `TestClient`)
- [x] Run `uv pip install -e ".[api,dev]"` and verify installation succeeds

**Acceptance Criteria:**
- `python -c "import fastapi; import uvicorn"` succeeds after install
- `uv pip install -e ".[dev]"` (without `api`) still works — fastapi is not imported at module level

### 1.2 Create Server Module Skeleton

- [x] Create `src/portrait_map_lab/server/__init__.py` — re-exports `create_app` and `ServerConfig` (divergence: not empty, added re-exports per code review)
- [x] Create `src/portrait_map_lab/server/config.py` with a `ServerConfig` dataclass: `host` (default `"127.0.0.1"`), `port` (default `8100`), `cache_dir` (default `Path(".cache/api")` — uses `Path` not `str` per code review), `session_ttl_seconds` (default `1800`)
- [x] Create `src/portrait_map_lab/server/app.py` with:
  - `create_app()` factory function returning a FastAPI instance
  - `CORSMiddleware` with `allow_origins=["*"]`, `allow_methods=["*"]`, `allow_headers=["*"]`
  - A `GET /api/health` endpoint (in `routes.py`) returning `{"status": "ok"}`
  - Note: health endpoint placed in `routes.py` with `APIRouter(prefix="/api")`, included via `app.include_router(router)` — follows planned file structure
  - Note: version sourced from `importlib.metadata.version()` instead of hardcoded
- [x] Create `src/portrait_map_lab/server/schemas.py` — empty for now, will be populated in Phase 2

**Acceptance Criteria:**
- `from portrait_map_lab.server.app import create_app` works
- The app instance has CORS middleware configured
- `/api/health` returns `{"status": "ok"}`

### 1.3 Add `serve` Subcommand to CLI

- [ ] Add a `serve` subparser to `scripts/run_pipeline.py` with `--host` and `--port` arguments
- [ ] The `handle_serve()` function lazily imports `fastapi` and `uvicorn` inside the function body
- [ ] If `fastapi` or `uvicorn` are not installed, print a clear error: `"The API server requires additional dependencies. Install with: uv pip install -e '.[api]'"`
- [ ] When dependencies are available, call `uvicorn.run(app, host=host, port=port)`
- [ ] Write tests: verify the `serve` subparser is registered and parses `--host`/`--port` arguments

**Acceptance Criteria:**
- `uv run python scripts/run_pipeline.py serve` starts a server on `127.0.0.1:8100`
- `uv run python scripts/run_pipeline.py serve --port 9000` starts on port 9000
- `curl http://127.0.0.1:8100/api/health` returns `{"status":"ok"}`
- Existing CLI subcommands (`features`, `contour`, etc.) are unaffected
- Without `.[api]` installed, `serve` prints a helpful error and exits with code 1

### 1.4 Add Endpoint Tests with TestClient

- [ ] Create `tests/test_server.py`
- [ ] Add a `client` pytest fixture that creates a `httpx.ASGITransport` + `httpx.Client` from `create_app()` (no real server needed)
- [ ] Test `GET /api/health` returns 200 with `{"status": "ok"}`
- [ ] Test CORS headers are present in response
- [ ] Run tests: `uv run pytest tests/test_server.py -v`

**Acceptance Criteria:**
- All server tests pass
- Tests run without starting a real uvicorn process

---

## Phase 2: Full Pipeline Generation Endpoint

**Purpose:** Accept an image via the API and return generated maps in the export bundle format — the simplest useful version.

**Rationale:** Gets end-to-end generation working before adding the resolver complexity of Phase 3. This phase always runs all pipelines (like the CLI's `all` command), which is well-tested existing behavior. The response format is designed from the start to match the plotter's `parseManifest()`.

### 2.1 Define Pydantic Request/Response Schemas

- [ ] Create request schema in `schemas.py`:
  - `GenerateRequest`: `image_path` (optional str), `maps` (optional list of str), `persist` (optional str)
  - `config` sub-schemas mirroring the dataclass hierarchy: `FeaturesConfig`, `ContourConfigSchema`, `DensityConfigSchema`, `ComplexityConfigSchema`, `FlowConfigSchema`, `FlowSpeedConfigSchema` — each with all fields optional (overrides only)
  - Nested `RemapConfigSchema` and `LuminanceConfigSchema` for sub-objects
- [ ] Create response schema:
  - `GenerateResponse`: `session_id` (str), `manifest` (dict), `base_url` (str)
  - `MapKeyInfo`: `key`, `value_range`, `description` — for the `/api/maps/keys` endpoint
- [ ] Write tests: validate that schemas accept partial overrides and reject invalid map keys

**Acceptance Criteria:**
- All config fields have correct types matching the dataclass counterparts
- Omitted fields default to `None` (meaning "use pipeline default")
- Invalid map keys (e.g., `"tonal_target"`) are rejected by validation

### 2.2 Implement `/api/maps/keys` Endpoint

- [ ] Add `GET /api/maps/keys` route in `routes.py`
- [ ] Source data from `_MAP_DEFINITIONS` in `export.py` — import and iterate to build response
- [ ] Return list of `{key, value_range, description}` for each defined map
- [ ] Write tests: verify response contains all 7 map keys with correct structure

**Acceptance Criteria:**
- Response lists exactly the 7 keys from `_MAP_DEFINITIONS`
- Each entry has `key` (string), `value_range` (2-element list), `description` (string)

### 2.3 Implement `POST /api/generate` — Full Pipeline Path

- [ ] Add `POST /api/generate` route in `routes.py`
- [ ] Handle dual image input: accept `image_path` (JSON body) or multipart file upload
  - For `image_path`: validate path exists and is loadable via `load_image()`
  - For file upload: write to temp file, load via `load_image()`, clean up after processing
- [ ] Validate image has a detectable face (call `detect_landmarks()`); return 422 with `"No face detected in image"` if not
- [ ] Build config objects from request schema, merging overrides onto defaults
- [ ] Call `run_all_pipelines()` with the constructed configs
- [ ] Call `build_export_bundle()` on the result to get `ExportBundle`
- [ ] Generate a UUID4 session ID
- [ ] Write `.bin` files and `manifest.json` to `.cache/api/{session_id}/`
- [ ] Return `GenerateResponse` with full manifest (via `_manifest_to_dict()`) and `base_url`
- [ ] Define the endpoint as a regular function (not `async def`) so FastAPI runs it in a threadpool
- [ ] Add a `threading.Lock` around pipeline execution to serialize concurrent requests

**Acceptance Criteria:**
- `POST /api/generate` with a valid `image_path` returns 200 with a manifest containing map entries
- The `manifest` field in the response passes the plotter's `parseManifest()` validation (has `version`, `source_image`, `width`, `height`, `created_at`, `maps` array with `filename`, `key`, `dtype`, `shape`, `value_range`, `description`)
- The `base_url` field points to `/api/maps/{session_id}`
- Invalid image path returns 422
- Missing face returns 422 with descriptive message
- Concurrent requests are serialized (second request waits for first to complete)

### 2.4 Implement Map File Serving

- [ ] Add `GET /api/maps/{session_id}/manifest.json` route — serve manifest from cache dir
- [ ] Add `GET /api/maps/{session_id}/{filename}` route — serve `.bin` files via `FileResponse`
- [ ] Return 404 if session or file doesn't exist
- [ ] Write integration test: generate maps, then fetch each `.bin` URL from the response and verify it returns valid float32 data with correct byte length (`width * height * 4`)

**Acceptance Criteria:**
- Fetching `{base_url}/manifest.json` returns the same manifest as the generate response
- Fetching `{base_url}/density_target.bin` returns bytes decodable as float32 array
- Byte length equals `height * width * 4` (float32)
- Non-existent session returns 404
- Non-existent filename within a valid session returns 404

### 2.5 Config Override Integration Tests

- [ ] Write test: generate with `config.density.gamma = 2.0` and verify the result differs from default
- [ ] Write test: generate with `config.features.weights = {"eyes": 1.0, "mouth": 0.0}` and verify it's accepted
- [ ] Write test: generate with nested override `config.flow.etf.blur_sigma = 5.0` and verify it's applied
- [ ] Write test: generate with no config overrides and verify defaults produce valid output

**Acceptance Criteria:**
- Config overrides are correctly propagated to pipeline config dataclasses
- Partial overrides work (only override what's specified, keep defaults for the rest)
- Nested config objects (remap, luminance, etf) merge correctly

---

## Phase 3: Granular Map Selection and Pipeline Resolver

**Purpose:** Allow requesting specific maps and only run the pipelines needed to produce them.

**Rationale:** Separated from Phase 2 because the resolver is new logic not present in the existing codebase. It needs its own unit tests and has edge cases (e.g., requesting `flow_speed` implies complexity + contour + flow). Phase 2 validates that end-to-end generation works; Phase 3 optimizes which pipelines run.

### 3.1 Promote Private Pipeline Helpers to Public

- [ ] Rename `_run_feature_pipeline_with_landmarks` to `run_feature_pipeline_with_landmarks` in `pipelines.py`
- [ ] Rename `_run_contour_pipeline_with_landmarks` to `run_contour_pipeline_with_landmarks` in `pipelines.py`
- [ ] Update `run_all_pipelines` to call the renamed functions
- [ ] Add both to `__init__.py` exports
- [ ] Run existing test suite to verify nothing breaks

**Acceptance Criteria:**
- Both functions are importable from `portrait_map_lab`
- All existing tests pass unchanged
- `run_all_pipelines` still works identically

### 3.2 Implement Pipeline Dependency Resolver

- [ ] Create `src/portrait_map_lab/server/resolver.py`
- [ ] Define the dependency table as a constant mapping map keys to required pipeline sets:
  ```
  density_target → {features, contour, density}
  importance     → {features, contour, density}
  flow_x         → {contour, flow}
  flow_y         → {contour, flow}
  coherence      → {contour, flow}
  complexity     → {complexity}
  flow_speed     → {complexity, contour, flow}
  ```
- [ ] Implement `resolve_pipelines(requested_maps: list[str]) -> set[str]` that unions pipeline sets for all requested maps
- [ ] Implement `run_resolved_pipelines(image, pipelines: set[str], configs) -> dict[str, np.ndarray]` that:
  - Detects landmarks once (if any pipeline needs them)
  - Calls individual pipeline functions in dependency order with shared landmarks
  - Collects the result arrays keyed by map name
  - Only computes LIC if not needed (the API doesn't export it)
- [ ] Write thorough unit tests for the resolver:
  - Single map requests resolve to correct pipeline set
  - Combined requests union correctly (e.g., `["density_target", "flow_x"]` → `{features, contour, density, flow}`)
  - Invalid map keys raise ValueError
  - Empty list resolves to all pipelines

**Acceptance Criteria:**
- `resolve_pipelines(["flow_x"])` returns `{"contour", "flow"}`
- `resolve_pipelines(["density_target", "flow_speed"])` returns `{"features", "contour", "density", "complexity", "flow"}`
- `resolve_pipelines([])` returns all pipeline names
- Invalid keys raise a clear error

### 3.3 Integrate Resolver into Generate Endpoint

- [ ] Modify `POST /api/generate` to use the resolver when `maps` is specified
- [ ] When `maps` is omitted or empty, run all pipelines (existing behavior)
- [ ] When `maps` is specified, call `run_resolved_pipelines()` instead of `run_all_pipelines()`
- [ ] Build a partial `ExportBundle` containing only the requested maps (filter `_MAP_DEFINITIONS` to requested keys)
- [ ] Write integration tests:
  - Request only `["complexity"]` — verify features/contour/density/flow pipelines are NOT run (test by checking that only `complexity.bin` exists in cache dir)
  - Request `["flow_x", "flow_y"]` — verify density pipeline is not run
  - Request `["density_target", "flow_speed"]` — verify all needed pipelines run

**Acceptance Criteria:**
- Requesting `["complexity"]` completes significantly faster than requesting all maps
- The manifest in the response only contains entries for requested maps
- The `.bin` files in the cache directory match the requested maps

### 3.4 Implement Sessions List and Delete Endpoints

- [ ] Add `GET /api/sessions` route — list session IDs with metadata (source image name, created_at timestamp, map keys)
- [ ] Add `DELETE /api/maps/{session_id}` route — delete cache directory, return 204; return 404 if not found
- [ ] Track sessions in a lightweight in-memory registry (dict of session_id → metadata)
- [ ] Write tests for both endpoints

**Acceptance Criteria:**
- After generating, `GET /api/sessions` lists the session
- After deleting, session is gone from list and `.bin` files are removed from disk
- Deleting a non-existent session returns 404

---

## Phase 4: Session Cache Management

**Purpose:** Add TTL-based automatic cleanup and server-restart resilience.

**Rationale:** Deferred to Phase 4 because Phases 1-3 produce a fully functional API. Without TTL cleanup, the cache directory grows indefinitely — acceptable for development but not for sustained use. This phase adds the lifecycle management.

### 4.1 Implement Disk-Backed Cache Manager

- [ ] Create `src/portrait_map_lab/server/cache.py` with a `SessionCache` class:
  - `register(session_id, metadata)` — add to registry with timestamp
  - `get_path(session_id) -> Path` — return cache directory path
  - `exists(session_id) -> bool` — check if session exists on disk
  - `list_sessions() -> list[dict]` — return all sessions with metadata
  - `delete(session_id)` — remove from registry and delete directory
  - `cleanup_expired()` — remove sessions older than TTL
- [ ] On initialization, scan existing `.cache/api/` subdirectories:
  - Re-register any directories that contain a `manifest.json`
  - Use file modification time as the creation timestamp
  - This handles server restarts without orphaning sessions
- [ ] Write unit tests with a temp directory:
  - Register, get, list, delete lifecycle
  - Expired sessions are cleaned up
  - Startup scan re-registers existing sessions

**Acceptance Criteria:**
- Sessions created before server restart are discoverable after restart
- Expired sessions are cleaned up by `cleanup_expired()`
- Deleting a session removes its directory from disk
- The cache directory path is configurable via `ServerConfig`

### 4.2 Integrate Cache into Server Lifecycle

- [ ] Use FastAPI's lifespan context manager to:
  - Initialize the `SessionCache` on startup (triggers directory scan)
  - Start a background task that runs `cleanup_expired()` every 60 seconds
  - Cancel the background task on shutdown
- [ ] Wire the `SessionCache` into route handlers via FastAPI dependency injection (or app state)
- [ ] Update `POST /api/generate` to use `SessionCache.register()` instead of ad-hoc directory creation
- [ ] Update `GET /api/sessions` and `DELETE` to use `SessionCache`
- [ ] Update file-serving endpoints to verify session exists via cache before serving
- [ ] Write integration test: generate a session, wait for TTL to pass (use a short TTL in test config), verify session is cleaned up

**Acceptance Criteria:**
- Background cleanup task runs without blocking request handling
- Expired sessions are automatically removed
- Server shutdown is clean (no orphaned background tasks)
- All existing tests still pass

---

## Phase 5: Persist, Documentation, and Polish

**Purpose:** Add the disk persistence feature, write the integration guide, and handle remaining polish items.

**Rationale:** Last phase because the API contract must be stable before documenting it, and the `persist` feature is a convenience that layers on top of the working cache.

### 5.1 Implement Persist Parameter

- [ ] When `persist` is specified in the generate request:
  - Sanitize the value: allow only `[a-zA-Z0-9_-]` characters, reject anything else with 422
  - Copy the session's cache directory contents to `output/{persist}/export/`
  - Exempt the session from TTL cleanup (or mark it as persistent in the registry)
- [ ] Write tests:
  - Persist creates files at the expected output path
  - Persisted bundle has identical contents to the cache directory
  - Path traversal attempts (e.g., `"../../etc"`) are rejected with 422
  - The persisted bundle is loadable by the plotter (manifest format is correct)

**Acceptance Criteria:**
- `persist: "my-portrait"` writes to `output/my-portrait/export/`
- The output directory contains `manifest.json` and all requested `.bin` files
- Unsafe persist values are rejected

### 5.2 Add `.cache/` to `.gitignore`

- [ ] Add `.cache/` to the project's `.gitignore`
- [ ] Verify with `git status` that `.cache/api/` directories are ignored

**Acceptance Criteria:**
- Generated cache files do not appear in `git status`

### 5.3 Write Integration Guide

- [ ] Create `docs/server/integration-guide.md` covering:
  - Starting the server (`uv run python scripts/run_pipeline.py serve`)
  - Checking health (`GET /api/health`)
  - Discovering available maps (`GET /api/maps/keys`)
  - Generating maps with `POST /api/generate` — full examples with curl
  - Fetching individual maps by URL
  - Using config overrides (with examples for each pipeline)
  - Persisting bundles
  - How the response maps to the plotter's `MapBundle.load()` pattern:
    - `base_url` from response → pass to `MapBundle.load()` as `baseUrl`
    - `manifest.json` at `{base_url}/manifest.json` → fetched automatically
    - `.bin` files at `{base_url}/{filename}` → fetched on demand
  - Session lifecycle and TTL behavior
- [ ] Update `docs/pipeline/README.md` with a link to the server docs
- [ ] Verify the Swagger docs at `/docs` are complete and accurate

**Acceptance Criteria:**
- Integration guide covers the full client workflow from health check to map consumption
- The plotter integration pattern is explicitly documented with examples
- Swagger UI at `/docs` shows all endpoints with schemas

### 5.4 Final Verification

- [ ] Run full test suite: `uv run pytest tests/ -v`
- [ ] Run linter: `uv run ruff check .`
- [ ] Manual smoke test: start server, generate maps via curl, fetch a `.bin` file, verify it's valid float32 data
- [ ] Verify that existing CLI commands still work unchanged

**Acceptance Criteria:**
- All tests pass
- No linting errors
- CLI `features`, `contour`, `density`, `complexity`, `flow`, `all` subcommands work as before
- Server starts, generates, and serves maps correctly

---

## Dependency Graph

```
Phase 1 (Bootstrap)
  1.1 (deps) → 1.2 (skeleton) → 1.3 (CLI serve) → 1.4 (tests)
                                      │
                                Phase 2 (Full Pipeline)
                                  2.1 (schemas) → 2.2 (keys endpoint)
                                       │
                                       ├→ 2.3 (generate endpoint) → 2.4 (file serving) → 2.5 (config tests)
                                       │                                                       │
                                Phase 3 (Resolver)                                             │
                                  3.1 (promote helpers) → 3.2 (resolver) → 3.3 (integrate) → 3.4 (sessions)
                                                                                                │
                                                                               Phase 4 (Cache Management)
                                                                                 4.1 (cache manager) → 4.2 (lifecycle)
                                                                                                            │
                                                                                           Phase 5 (Polish)
                                                                                  5.1 (persist) ─┐
                                                                                  5.2 (gitignore) ├→ 5.4 (verification)
                                                                                  5.3 (docs) ─────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `serve` as CLI subcommand, not separate script | Follows existing pattern — one entry point (`run_pipeline.py`). Users already know how to invoke it. |
| Lazy import of fastapi/uvicorn in handler | The `serve` handler is the only code that needs these deps. Importing at module level would break the CLI for users without `.[api]` installed. |
| Regular function (not `async def`) for generate endpoint | Pipeline code is CPU-bound (numpy/OpenCV/MediaPipe). A sync function lets FastAPI auto-run it in a threadpool, keeping the event loop free for concurrent file-serving requests. |
| Threading lock around pipeline execution | MediaPipe may not be thread-safe; bounding to one concurrent pipeline run also caps peak memory at ~200-400 MB. |
| Promote `_with_landmarks` helpers to public | The resolver needs to call individual pipelines with shared landmarks. The alternative (duplicating landmark detection) wastes 1-2 seconds per request. |
| Disk-backed cache, not in-memory | A full map set is ~55 MB as float32. Holding multiple sessions in memory would consume hundreds of MB with no benefit — the data is only read once when the client fetches it. |
| Startup scan of cache directory | Avoids orphaning sessions on server restart. Simpler than requiring clean shutdown. |
| Full `ExportManifest` in response (not simplified) | The plotter's `parseManifest()` validates `created_at`, `maps[].filename`, `maps[].shape`, etc. A simplified format would require plotter-side changes, defeating the "no modification" goal. |
| `base_url` in response instead of per-map `urls` | Matches the plotter's `MapBundle.load(baseUrl)` pattern. The client fetches `{baseUrl}/manifest.json` and `{baseUrl}/{filename}` — providing a `base_url` makes this natural. |
| Config schema mirrors dataclass hierarchy | 1:1 correspondence makes implementation straightforward (merge Pydantic model → dataclass) and avoids naming ambiguity (e.g., both features and contour have `remap`). |
| Separate `resolver.py` module | The dependency table and pipeline orchestration logic will grow when presets and recipes are added. Isolating it makes future extension clean. |
