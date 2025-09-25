from __future__ import annotations

from typing import Any, Dict, Optional
# Runpod serverless uses a special handler contract. We avoid importing their SDK
# to keep dependencies minimal; the container can simply expose a function named 'handler'.

from app.gpu_service.runner import run_and_callback
from app.core.config import get_settings


def handler(event: Dict[str, Any], _context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runpod serverless entrypoint.

    Expected event schema:
    {
      "input": {
        "job_id": str,
        "mode": "cleaned"|"full",
        "input": {"download_url"|"path"|"storage_key": str},
        "outputs": { name: {"storage_key": str, "put_url": str} },
        "callback_url": str
      },
      "userProvidedId": str | None
    }
    """
    settings = get_settings()
    inp = event.get("input") or {}

    job_id = str(inp.get("job_id") or "")
    mode = str(inp.get("mode") or "full")
    body_input = inp.get("input") or {}
    outputs = inp.get("outputs")
    callback_url = inp.get("callback_url")

    if not job_id:
        return {"status": "FAILED", "error": "missing job_id"}

    run_and_callback(
        job_id=job_id,
        mode=mode,
        job_input=body_input,
        outputs=outputs,
        callback_url=callback_url,
        callback_secret=settings.gpu_callback_secret,
    )

    return {"status": "QUEUED", "job_id": job_id}