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

from portrait_map_lab.export import _manifest_to_dict, build_export_bundle
from portrait_map_lab.landmarks import detect_landmarks
from portrait_map_lab.models import ComplexityConfig
from portrait_map_lab.pipelines import run_all_pipelines
from portrait_map_lab.storage import load_image

from .config import ServerConfig
from .schemas import (
    MAP_KEY_INFOS,
    GenerateRequest,
    GenerateResponse,
    MapKeyInfo,
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
    parsed: tuple[GenerateRequest, UploadFile | None] = Depends(_parse_request),
) -> GenerateResponse:
    """Generate maps from a portrait image.

    Accepts either a JSON body with ``image_path`` or a multipart file upload
    with field name ``image``.  The endpoint runs all pipelines synchronously
    and writes the resulting binary maps to the session cache directory.
    """
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
            detect_landmarks(image)
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

        # Always enable complexity so all 7 maps are produced.
        complexity_config = build_complexity_config(
            cfg.complexity if cfg else None
        ) or ComplexityConfig()

        # --- run pipelines under lock ---
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

        # --- build export bundle ---
        bundle = build_export_bundle(result, source_name)

        # --- write to cache ---
        session_id = str(uuid.uuid4())
        config = ServerConfig()
        session_dir = config.cache_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        for key, data in bundle.binary_maps.items():
            (session_dir / f"{key}.bin").write_bytes(data)

        manifest_dict = _manifest_to_dict(bundle.manifest)
        (session_dir / "manifest.json").write_text(
            json.dumps(manifest_dict, indent=2) + "\n"
        )

        logger.info(
            "Session %s: wrote %d maps to %s",
            session_id, len(bundle.binary_maps), session_dir,
        )

        return GenerateResponse(
            session_id=session_id,
            manifest=manifest_dict,
            base_url=f"/api/maps/{session_id}",
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _resolve_session_file(session_id: str, filename: str) -> Path:
    """Resolve and validate a file path within a session cache directory.

    Raises :class:`~fastapi.HTTPException` with 404 if the session directory
    or the requested file does not exist.  Path-traversal attempts (e.g.
    ``../`` in *session_id* or *filename*) are rejected.
    """
    config = ServerConfig()
    cache_root = config.cache_dir.resolve()
    session_dir = (cache_root / session_id).resolve()

    # Guard against path traversal in session_id (e.g. "../../etc")
    if not session_dir.is_relative_to(cache_root):
        raise HTTPException(status_code=404, detail="Session not found")

    file_path = (session_dir / filename).resolve()

    # Guard against path traversal in filename (e.g. "../secret.txt")
    if not file_path.is_relative_to(session_dir):
        raise HTTPException(status_code=404, detail="File not found")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return file_path


@router.get("/maps/{session_id}/manifest.json")
def get_session_manifest(session_id: str) -> Response:
    """Serve the manifest JSON for a cached session."""
    file_path = _resolve_session_file(session_id, "manifest.json")
    return Response(content=file_path.read_bytes(), media_type="application/json")


@router.get("/maps/{session_id}/{filename}")
def get_session_file(session_id: str, filename: str) -> FileResponse:
    """Serve a binary map file from a cached session."""
    file_path = _resolve_session_file(session_id, filename)
    return FileResponse(file_path, media_type="application/octet-stream")
