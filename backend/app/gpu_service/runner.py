from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import logging


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def ensure_segmentation_model_available() -> Path:
    """Ensure YOLO segmentation weights exist at assets/models/model.pt.

    - If the file is missing and R2 (S3-compatible) credentials are configured,
      download once to the assets directory. Subsequent runs reuse the file
      (ideally on a mounted volume in serverless).
    - If configuration is incomplete, do nothing.
    """
    try:
        from app.core.paths import get_assets_root
        from app.core.config import get_settings
    except Exception:
        # If imports fail, return a best-effort path without side effects
        return Path("assets/models/model.pt")

    dst = get_assets_root() / "models" / "model.pt"
    if dst.exists():
        return dst

    settings = get_settings()
    endpoint = settings.r2_endpoint_url
    bucket = settings.r2_bucket_name
    access_key = settings.r2_access_key_id
    secret_key = settings.r2_secret_access_key
    key = "models/model.pt"

    if not (endpoint and bucket and access_key and secret_key):
        # Configuration not present; skip download to avoid breaking local
        logging.getLogger(__name__).info("seg_model_skip_download_missing_config", extra={"path": str(dst)})
        return dst

    try:
        import boto3  # type: ignore
        from botocore.config import Config  # type: ignore

        dst.parent.mkdir(parents=True, exist_ok=True)
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
        s3.download_file(bucket, key, str(dst))
        logging.getLogger(__name__).info(
            "seg_model_downloaded", extra={"bucket": bucket, "key": key, "path": str(dst)}
        )
    except Exception:
        logging.getLogger(__name__).exception(
            "seg_model_download_failed", extra={"bucket": bucket, "key": key, "path": str(dst)}
        )
    return dst


def run_and_callback(
    *,
    job_id: str,
    mode: str,
    job_input: Dict[str, Any],
    outputs: Optional[Dict[str, Dict[str, str]]],
    callback_url: Optional[str],
    callback_secret: Optional[str],
    models: Any | None = None,
) -> None:
    """
    Execute the pipeline for a single job and POST a signed callback.

    - job_input supports keys: download_url | path | storage_key
    - outputs maps name -> { storage_key, put_url }
    - callback_url is optional; when provided the function will POST results
    """
    from app.core.paths import get_artifacts_root, get_job_dir
    from app.pipeline.orchestrator import run_pipeline

    status = "COMPLETED"
    error_detail: Optional[str] = None

    job_dir = get_job_dir(job_id)
    artifacts_root = get_artifacts_root()

    # Track which output artifacts we successfully uploaded when presigned PUT URLs are provided.
    uploaded_names: set[str] = set()

    try:
        # Ensure required model assets exist before running the pipeline (no-op if already present)
        ensure_segmentation_model_available()

        # Resolve input image
        dst_path = job_dir / "source_image"
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        download_url = job_input.get("download_url")
        if download_url:
            import httpx
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(str(download_url))
                resp.raise_for_status()
                _write_bytes(dst_path, resp.content)
        else:
            if job_input.get("path"):
                src_path = Path(str(job_input["path"]))
            elif job_input.get("storage_key"):
                src_path = artifacts_root / str(job_input["storage_key"])
            else:
                raise ValueError("missing input path or storage_key")

            if not src_path.exists():
                raise ValueError("input not found")
            _write_bytes(dst_path, src_path.read_bytes())

        # Run pipeline (no translate/typeset here)
        mode_val = mode if mode in ("cleaned", "full") else "full"
        result = run_pipeline(
            job_id=job_id,
            image_path=str(dst_path),
            depth=mode_val,
            include_typeset=False,
            include_translate=False,
            models=models,
        )

        # Upload outputs when presigned PUTs are provided
        if outputs:
            import httpx
            artifacts = result.get("artifacts", {})
            name_to_path = {
                # Source cleaned image directly from pipeline paths
                "CLEANED_PAGE": result.get("paths", {}).get("cleaned"),
                "TEXT_JSON": result.get("paths", {}).get("json"),
            }
            with httpx.Client(timeout=30.0) as client:
                for name, spec in outputs.items():
                    p = name_to_path.get(name)
                    if not p:
                        continue
                    fp = Path(p)
                    if not fp.exists():
                        continue
                    put_url = spec.get("put_url", "")
                    if not put_url:
                        continue
                    client.put(put_url, content=fp.read_bytes())
                    uploaded_names.add(name)
    except Exception as e:
        status = "FAILED"
        error_detail = str(e)
        logging.getLogger(__name__).exception("pipeline_failed", extra={"job_id": job_id})

    # Prepare and send callback
    if callback_url:
        import httpx
        import hmac
        import hashlib
        import base64

        payload: Dict[str, Any] = {"job_id": job_id, "status": status}
        if error_detail:
            payload["error"] = error_detail

        if status == "COMPLETED":
            artifacts_to_report: Dict[str, str] = {}
            if outputs:
                # Report only the artifacts that were actually uploaded
                artifacts_to_report = {
                    name: spec.get("storage_key")
                    for name, spec in outputs.items()
                    if name in uploaded_names and spec.get("storage_key")
                }
            else:
                # Local/dev relative paths
                json_path = result.get("paths", {}).get("json") if 'result' in locals() else None
                if json_path and Path(json_path).exists():
                    artifacts_to_report["TEXT_JSON"] = str(Path(json_path).relative_to(artifacts_root)).replace("\\", "/")
                cleaned_path = result.get("paths", {}).get("cleaned") if 'result' in locals() else None
                if cleaned_path and Path(cleaned_path).exists():
                    artifacts_to_report["CLEANED_PAGE"] = str(Path(cleaned_path).relative_to(artifacts_root)).replace("\\", "/")

            if artifacts_to_report:
                payload["artifacts"] = artifacts_to_report

        headers = {"content-type": "application/json"}
        try:
            raw = json_dumps(payload)
            if callback_secret:
                mac = hmac.new(callback_secret.encode("utf-8"), raw, hashlib.sha256).digest()
                headers["x-gpu-signature"] = base64.b64encode(mac).decode("ascii")
            with httpx.Client(timeout=10.0) as client:
                client.post(str(callback_url), content=raw, headers=headers)
        except Exception:
            logging.getLogger(__name__).exception("callback_failed", extra={"job_id": job_id})


def json_dumps(obj: Any) -> bytes:
    import json
    return json.dumps(obj).encode("utf-8")