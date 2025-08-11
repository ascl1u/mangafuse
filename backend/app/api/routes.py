from typing import Any, Dict

import redis
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from celery.result import AsyncResult

from app.core.config import get_settings
from app.worker.celery_app import celery_app
from app.worker.tasks import demo_task


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


@router.post("/process", summary="Enqueue demo task", status_code=status.HTTP_202_ACCEPTED)
def enqueue_process(req: ProcessRequest) -> Dict[str, str]:
    """Enqueue a demo Celery task and return its task id."""
    async_result = demo_task.delay(req.duration_s)
    return {"task_id": async_result.id}


@router.get("/process/{task_id}", summary="Get task status")
def get_process_status(task_id: str) -> Dict[str, Any]:
    """Poll the status/result of a Celery task by id."""
    result = AsyncResult(task_id, app=celery_app)
    payload: Dict[str, Any] = {"task_id": task_id, "state": result.state}
    if result.ready():
        if result.failed():
            # result.result contains the exception info when failed
            payload["error"] = str(result.result)
        else:
            payload["result"] = result.result
    return payload


