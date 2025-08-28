from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import json
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.core.paths import get_artifacts_root, get_job_dir, get_assets_root
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.model_registry import ModelRegistry


class JobInput(BaseModel):
    storage_key: Optional[str] = None
    path: Optional[str] = None


class SubmitJobBody(BaseModel):
    job_id: str
    mode: str = "full"  # "cleaned" | "full"
    input: JobInput
    callback_url: Optional[str] = None
    outputs: Optional[Dict[str, Dict[str, str]]] = None  # name -> {storage_key, put_url}


app = FastAPI(title="MangaFuse GPU Service", version="0.1.0")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _read_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _run_pipeline_and_callback(body: SubmitJobBody) -> None:
    job_id = body.job_id
    mode = body.mode if body.mode in ("cleaned", "full") else "full"
    job_dir = get_job_dir(job_id)
    artifacts_root = get_artifacts_root()
    status = "COMPLETED"
    error_detail = None

    try:
        # Resolve input image path
        if body.input.path:
            src_path = Path(body.input.path)
        elif body.input.storage_key:
            src_path = artifacts_root / body.input.storage_key
        else:
            raise ValueError("missing input path or storage_key")
        if not src_path.exists():
            raise ValueError("input not found")

        # Copy input into job_dir/source_image
        dst_path = job_dir / "source_image"
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes(dst_path, _read_bytes(src_path))

        # Run pipeline without translate or typeset; produce cleaned + text.json (+ optional overlay)
        # Pull preloaded models from app state (may be None in dev)
        models = getattr(app.state, "models", None)

        result = run_pipeline(
            job_id=job_id,
            image_path=str(dst_path),
            depth=mode,
            include_typeset=False,
            include_translate=False,
            models=models,
        )

        # If outputs with presigned PUTs were provided, upload via HTTP PUT and then remove local files
        if body.outputs:
            import httpx
            artifacts = result.get("artifacts", {})
            name_to_path = {
                "CLEANED_PAGE": artifacts.get("CLEANED_PAGE"),
                "TEXT_JSON": result.get("paths", {}).get("json"),
            }
            with httpx.Client(timeout=30.0) as client:
                for name, spec in body.outputs.items():
                    p = name_to_path.get(name)
                    if not p or not Path(p).exists():
                        continue
                    with open(p, "rb") as f:
                        url = spec.get("put_url", "")
                        if url:
                            client.put(url, content=f.read())
    except Exception as e:
        status = "FAILED"
        error_detail = str(e)
        logging.getLogger(__name__).exception("pipeline_failed", extra={"job_id": job_id})

    # In local dev, we do not push outputs anywhere; backend will read from artifacts volume.
    # A production variant would PUT to presigned URLs and then POST the callback.
    if body.callback_url:
        try:
            import httpx

            payload = {"job_id": job_id, "status": status}
            if status == "COMPLETED" and body.outputs:
                payload["artifacts"] = {name: spec.get("storage_key") for name, spec in body.outputs.items() if spec.get("storage_key")}
            if error_detail:
                payload["error"] = error_detail
            with httpx.Client(timeout=10.0) as client:
                client.post(body.callback_url, json=payload)
        except Exception:
            logging.getLogger(__name__).exception("callback_failed", extra={"job_id": job_id})


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
