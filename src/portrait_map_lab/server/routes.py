"""API route handlers for the Map Generation API."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from portrait_map_lab.export import (
    build_export_bundle,
    build_export_bundle_for_maps,
    manifest_to_dict,
)
from portrait_map_lab.landmarks import detect_landmarks
from portrait_map_lab.models import ComplexityConfig
from portrait_map_lab.pipelines import run_all_pipelines
from portrait_map_lab.storage import load_image

from .cache import SessionCache
from .previews import generate_previews_full, generate_previews_resolved
from .resolver import resolve_pipelines, run_resolved_pipelines
from .schemas import (
    MAP_KEY_INFOS,
    GenerateRequest,
    GenerateResponse,
    MapKeyInfo,
    SessionInfo,
    build_complexity_config,
    build_compose_config,
    build_contour_config,
    build_flow_config,
    build_flow_speed_config,
    build_pipeline_config,
)

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Serialize pipeline execution — MediaPipe may not be thread-safe and
# bounding to one concurrent run caps peak memory.
_pipeline_lock = threading.Lock()


def _get_cache(request: Request) -> SessionCache:
    """Retrieve the :class:`SessionCache` from application state."""
    return request.app.state.cache


async def _parse_request(
    request: Request,
) -> tuple[GenerateRequest, UploadFile | None]:
    """Parse the incoming request, handling both JSON and multipart bodies."""
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        image_file = form.get("image")
        body_str = form.get("request_body", "{}")
        if isinstance(body_str, str):
            body = GenerateRequest.model_validate_json(body_str)
        else:
            body = GenerateRequest()
        upload = image_file if isinstance(image_file, (UploadFile, StarletteUploadFile)) else None
        return body, upload

    # JSON or empty body
    body_bytes = await request.body()
    if body_bytes:
        body = GenerateRequest.model_validate_json(body_bytes)
    else:
        body = GenerateRequest()
    return body, None


@router.get("/health")
def health_check() -> dict[str, str]:
    """Return server health status."""
    return {"status": "ok"}


@router.get("/maps/keys")
def list_map_keys() -> list[MapKeyInfo]:
    """Return metadata for all available map keys."""
    return MAP_KEY_INFOS


@router.post("/generate")
def generate_maps(
    request: Request,
    parsed: tuple[GenerateRequest, UploadFile | None] = Depends(_parse_request),
) -> GenerateResponse:
    """Generate maps from a portrait image.

    Accepts either a JSON body with ``image_path`` or a multipart file upload
    with field name ``image``.  The endpoint runs all pipelines synchronously
    and writes the resulting binary maps to the session cache directory.
    """
    cache = _get_cache(request)
    body, upload_file = parsed
    temp_path: Path | None = None

    try:
        # --- resolve image source ---
        if upload_file is not None:
            suffix = Path(upload_file.filename or "upload.jpg").suffix
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False
            ) as tmp:
                shutil.copyfileobj(upload_file.file, tmp)
                temp_path = Path(tmp.name)
            source_name = upload_file.filename or "upload"
            try:
                image = load_image(temp_path)
            except FileNotFoundError:
                raise HTTPException(
                    status_code=422, detail="Uploaded file could not be loaded as an image"
                )
        elif body.image_path is not None:
            source_name = Path(body.image_path).name
            try:
                image = load_image(body.image_path)
            except FileNotFoundError:
                raise HTTPException(
                    status_code=422,
                    detail=f"Image not found: {body.image_path}",
                )
        else:
            raise HTTPException(
                status_code=422,
                detail="No image provided. Supply 'image_path' in JSON or upload a file.",
            )

        # --- validate face is detectable (fast-fail before pipeline lock) ---
        try:
            landmarks = detect_landmarks(image)
        except ValueError:
            raise HTTPException(
                status_code=422, detail="No face detected in image"
            )

        # --- build config objects from overrides ---
        cfg = body.config
        feature_config = build_pipeline_config(cfg.features if cfg else None)
        contour_config = build_contour_config(cfg.contour if cfg else None)
        compose_config = build_compose_config(cfg.density if cfg else None)
        flow_config = build_flow_config(cfg.flow if cfg else None)
        speed_config = build_flow_speed_config(cfg.flow_speed if cfg else None)

        # --- run pipelines and build bundle ---
        requested_maps = body.maps or []

        if requested_maps:
            # Granular path: resolve only the pipelines needed for the
            # requested maps, then build a partial export bundle.
            pipelines = resolve_pipelines(requested_maps)

            # Only enable complexity if it's in the resolved set.
            complexity_config: ComplexityConfig | None = None
            if "complexity" in pipelines:
                complexity_config = build_complexity_config(
                    cfg.complexity if cfg else None
                ) or ComplexityConfig()

            with _pipeline_lock:
                resolved_results = run_resolved_pipelines(
                    image,
                    landmarks,
                    pipelines,
                    feature_config=feature_config,
                    contour_config=contour_config,
                    compose_config=compose_config,
                    flow_config=flow_config,
                    complexity_config=complexity_config,
                    speed_config=speed_config,
                )

            bundle = build_export_bundle_for_maps(
                resolved_results, requested_maps, source_name
            )
        else:
            # Full path: run all pipelines (existing behavior).
            # Always enable complexity so all 7 maps are produced.
            complexity_config = build_complexity_config(
                cfg.complexity if cfg else None
            ) or ComplexityConfig()

            with _pipeline_lock:
                result = run_all_pipelines(
                    image,
                    feature_config=feature_config,
                    contour_config=contour_config,
                    compose_config=compose_config,
                    flow_config=flow_config,
                    complexity_config=complexity_config,
                    speed_config=speed_config,
                )

            bundle = build_export_bundle(result, source_name)

        # --- write to cache ---
        session_id = str(uuid.uuid4())
        session_dir = cache.get_path(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        for key, data in bundle.binary_maps.items():
            (session_dir / f"{key}.bin").write_bytes(data)

        manifest_dict = manifest_to_dict(bundle.manifest)
        (session_dir / "manifest.json").write_text(
            json.dumps(manifest_dict, indent=2) + "\n"
        )

        # --- generate preview PNGs ---
        previews_dir = session_dir / "previews"
        try:
            if requested_maps:
                preview_list = generate_previews_resolved(
                    resolved_results, image, previews_dir
                )
            else:
                preview_list = generate_previews_full(result, image, previews_dir)
        except Exception:
            # Preview generation is supplementary — don't block the
            # primary .bin output if it fails.
            logger.exception(
                "Failed to generate preview PNGs for session %s", session_id
            )
            preview_list = []

        logger.info(
            "Session %s: wrote %d maps and %d previews to %s",
            session_id, len(bundle.binary_maps), len(preview_list), session_dir,
        )

        # --- persist to output directory if requested ---
        if body.persist:
            output_dir = Path("output") / body.persist / "export"
            shutil.copytree(session_dir, output_dir, dirs_exist_ok=True)
            logger.info(
                "Session %s: persisted to %s", session_id, output_dir,
            )

        # Register session metadata in the cache.
        info = SessionInfo(
            session_id=session_id,
            source_image=bundle.manifest.source_image,
            created_at=bundle.manifest.created_at,
            map_keys=[m.key for m in bundle.manifest.maps],
            persistent=bool(body.persist),
            previews=preview_list,
        )
        cache.register(info)

        return GenerateResponse(
            session_id=session_id,
            manifest=manifest_dict,
            base_url=f"/api/maps/{session_id}",
            previews=preview_list,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _resolve_session_dir(session_id: str, cache_dir: Path) -> Path:
    """Resolve and validate a session cache directory path.

    Raises :class:`~fastapi.HTTPException` with 404 if the *session_id*
    contains a path-traversal attempt (e.g. ``../../etc``).

    Returns the resolved :class:`~pathlib.Path` to the session directory
    (which may or may not exist on disk yet).
    """
    cache_root = cache_dir.resolve()
    session_dir = (cache_root / session_id).resolve()

    if not session_dir.is_relative_to(cache_root):
        raise HTTPException(status_code=404, detail="Session not found")

    return session_dir


def _resolve_session_file(
    session_id: str, filename: str, cache_dir: Path
) -> Path:
    """Resolve and validate a file path within a session cache directory.

    Raises :class:`~fastapi.HTTPException` with 404 if the session directory
    or the requested file does not exist.  Path-traversal attempts (e.g.
    ``../`` in *session_id* or *filename*) are rejected.
    """
    session_dir = _resolve_session_dir(session_id, cache_dir)
    file_path = (session_dir / filename).resolve()

    # Guard against path traversal in filename (e.g. "../secret.txt")
    if not file_path.is_relative_to(session_dir):
        raise HTTPException(status_code=404, detail="File not found")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return file_path


@router.get("/maps/{session_id}/manifest.json")
def get_session_manifest(session_id: str, request: Request) -> Response:
    """Serve the manifest JSON for a cached session."""
    cache = _get_cache(request)
    file_path = _resolve_session_file(session_id, "manifest.json", cache.cache_dir)
    return Response(content=file_path.read_bytes(), media_type="application/json")


@router.get("/maps/{session_id}/{filename:path}")
def get_session_file(session_id: str, filename: str, request: Request) -> FileResponse:
    """Serve a binary map or preview PNG file from a cached session."""
    cache = _get_cache(request)
    file_path = _resolve_session_file(session_id, filename, cache.cache_dir)
    # Serve PNGs with correct content type for browser rendering
    media_type = "image/png" if file_path.suffix == ".png" else "application/octet-stream"
    return FileResponse(file_path, media_type=media_type)


@router.get("/sessions")
def list_sessions(request: Request) -> list[SessionInfo]:
    """Return metadata for all tracked sessions."""
    cache = _get_cache(request)
    return cache.list_sessions()


@router.delete("/maps/{session_id}", status_code=204)
def delete_session(session_id: str, request: Request) -> Response:
    """Delete a cached session directory and remove it from the registry.

    Returns 204 No Content on success, 404 if the session does not exist.
    """
    cache = _get_cache(request)
    _resolve_session_dir(session_id, cache.cache_dir)  # validates path traversal

    if not cache.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return Response(status_code=204)
