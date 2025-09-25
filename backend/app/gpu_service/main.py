from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from app.core.paths import get_assets_root
from app.pipeline.model_registry import ModelRegistry
from app.gpu_service.runner import run_and_callback


class JobInput(BaseModel):
    storage_key: Optional[str] = None
    path: Optional[str] = None
    download_url: Optional[str] = None


class SubmitJobBody(BaseModel):
    job_id: str
    mode: str  # "cleaned" | "full"
    input: JobInput
    callback_url: Optional[str] = None
    outputs: Optional[Dict[str, Dict[str, str]]] = None  # name -> {storage_key, put_url}


app = FastAPI(title="MangaFuse GPU Service", version="0.1.0")


def _run_pipeline_and_callback(body: SubmitJobBody) -> None:
    run_and_callback(
        job_id=body.job_id,
        mode=body.mode,
        job_input=body.input.model_dump(),
        outputs=body.outputs,
        callback_url=body.callback_url,
        callback_secret=None,
        models=getattr(app.state, "models", None),
    )


@app.post("/jobs", status_code=202)
def submit_job(body: SubmitJobBody, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    background_tasks.add_task(_run_pipeline_and_callback, body)
    return {"job_id": body.job_id, "status": "QUEUED"}


@app.on_event("startup")
def _startup_load_models() -> None:
    """Preload heavy models once at service startup."""
    try:
        assets_root = get_assets_root()
        seg_model_path = assets_root / "models" / "model.pt"
        registry = ModelRegistry.load(seg_model_path=seg_model_path, preload_ocr=True)
        app.state.models = registry
    except Exception:
        # If preload fails, leave models unset; pipeline will fallback to on-demand
        app.state.models = None
