from typing import Any, Dict, Literal, List, Optional
from pathlib import Path

import uuid
from pathlib import Path

import redis
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from celery.result import AsyncResult

from app.core.config import get_settings
from app.worker.celery_app import celery_app


router = APIRouter(prefix="/api/v1")


class ProcessRequest(BaseModel):
    duration_s: int = 5


@router.get("/", summary="Hello World")
def hello_world() -> Dict[str, str]:
    return {"message": "Hello World"}


@router.get("/healthz", summary="Liveness probe")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
def readyz() -> Dict[str, str]:
    settings = get_settings()
    # If no Redis configured yet, treat as not ready to surface misconfig early.
    if not settings.effective_broker_url:
        raise HTTPException(status_code=503, detail="redis not configured")
    try:
        client = redis.Redis.from_url(settings.effective_broker_url, socket_connect_timeout=0.5)
        pong: Any = client.ping()
        if pong is True:
            return {"status": "ready"}
        raise RuntimeError("unexpected redis ping result")
    except Exception as exc:  # noqa: BLE001 - surface readiness failure as 503
        raise HTTPException(status_code=503, detail=f"redis not reachable: {exc}")


@router.post("/process", summary="Upload an image and start processing", status_code=status.HTTP_202_ACCEPTED)
async def enqueue_process(
    file: UploadFile = File(...),
    depth: Literal["cleaned", "full"] = Form(...),
    debug: bool = Form(False),
    force: bool = Form(False),
) -> Dict[str, str]:
    """Accept an image upload and enqueue the AI pipeline task.

    Stores the upload under artifacts/uploads and triggers background processing.
    """
    # Resolve repo root -> artifacts/uploads
    repo_root = Path(__file__).resolve().parents[4]
    uploads_dir = repo_root / "artifacts" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # Derive a safe filename
    original_name = file.filename or "upload"
    ext = Path(original_name).suffix
    if not ext:
        # best-effort based on content-type
        ct = (file.content_type or "").lower()
        if "jpeg" in ct:
            ext = ".jpg"
        elif "png" in ct:
            ext = ".png"
        else:
            ext = ".img"
    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = uploads_dir / saved_name

    # Persist upload to disk
    contents = await file.read()
    with open(saved_path, "wb") as f:
        f.write(contents)

    # Enqueue processing task
    async_result = celery_app.send_task(
        "app.worker.tasks.process_page_task",
        args=[str(saved_path)],
        kwargs={
            "depth": "full" if depth == "full" else "cleaned",
            "debug": bool(debug),
            "force": bool(force),
        },
    )
    return {"task_id": async_result.id}


@router.get("/process/{task_id}", summary="Get task status")
def get_process_status(task_id: str) -> Dict[str, Any]:
    """Poll the status/result of a Celery task by id."""
    result = AsyncResult(task_id, app=celery_app)
    payload: Dict[str, Any] = {"task_id": task_id, "state": result.state}
    # Include progress meta when available
    info = result.info  # may carry {stage, progress}
    if isinstance(info, dict) and ("stage" in info or "progress" in info):
        payload["meta"] = {k: info.get(k) for k in ("stage", "progress")}
    if result.ready():
        if result.failed():
            payload["error"] = str(result.result)
        else:
            payload["result"] = result.result
    return payload


class EditRecord(BaseModel):
    id: int
    en_text: Optional[str] = None
    font_size: Optional[int] = None


class ApplyEditsRequest(BaseModel):
    edits: List[EditRecord]


@router.post("/jobs/{task_id}/edits", summary="Persist edits and enqueue re-typeset", status_code=status.HTTP_202_ACCEPTED)
def apply_edits(task_id: str, body: ApplyEditsRequest) -> Dict[str, str]:
    """Save edits and enqueue a background re-typeset job.

    Returns the id of the background task (reuse the original task id for artifact paths).
    """
    async_result = celery_app.send_task(
        "app.worker.tasks.apply_edits_task",
        args=[task_id, [e.model_dump() for e in body.edits]],
        kwargs={},
    )
    return {"task_id": async_result.id, "job_id": task_id}


@router.get("/jobs/{task_id}/exports", summary="Get export artifact URLs")
def get_exports(task_id: str) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    job_dir = repo_root / "artifacts" / "jobs" / task_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    final = job_dir / "final.png"
    text_layer = job_dir / "text_layer.png"
    payload: Dict[str, Any] = {"task_id": task_id}
    if final.exists():
        payload["final_url"] = f"/artifacts/jobs/{task_id}/final.png"
    if text_layer.exists():
        payload["text_layer_url"] = f"/artifacts/jobs/{task_id}/text_layer.png"
    return payload


@router.get("/jobs/{task_id}/download", summary="Download packaged artifacts (zip)")
def download_package(task_id: str):
    """Stream a zip containing available artifacts for a job.

    Contents:
      - cleaned.png (required)
      - final.png (when present)
      - text_layer.png (when present)
      - text.json (when present)
      - editor_payload.json (when present)
    """
    import io
    import zipfile

    repo_root = Path(__file__).resolve().parents[4]
    job_dir = repo_root / "artifacts" / "jobs" / task_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")

    cleaned = job_dir / "cleaned.png"
    if not cleaned.exists():
        raise HTTPException(status_code=404, detail="cleaned artifact not ready")

    final = job_dir / "final.png"
    text_layer = job_dir / "text_layer.png"
    text_json = job_dir / "text.json"
    editor_payload = job_dir / "editor_payload.json"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(cleaned, arcname="cleaned.png")
        if final.exists():
            zf.write(final, arcname="final.png")
        if text_layer.exists():
            zf.write(text_layer, arcname="text_layer.png")
        if text_json.exists():
            zf.write(text_json, arcname="text.json")
        if editor_payload.exists():
            zf.write(editor_payload, arcname="editor_payload.json")
    buf.seek(0)

    filename = f"mangafuse_{task_id}.zip"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


