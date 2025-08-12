from typing import Any, Dict, Literal

import uuid
from pathlib import Path

import redis
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
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


