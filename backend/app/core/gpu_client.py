from __future__ import annotations

from typing import Optional, Protocol, Dict, Tuple
import hmac
import hashlib
import base64
import json
import logging
import os

import httpx

from app.core.config import get_settings
import random
import time

logger = logging.getLogger(__name__)


class GpuClient(Protocol):
    """Interface for submitting jobs to a GPU service."""

    def submit_job(
        self,
        *,
        job_id: str,
        storage_key: str,
        mode: str = "full",
        outputs: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> None:
        """Submit a job to the GPU service."""
        ...


class LocalGpuClient:
    """Client for submitting jobs to a local GPU service, intended for development."""

    def __init__(self, base_url: str, public_backend_url: str, callback_secret: Optional[str] = None) -> None:
        self._base_url = base_url
        self._public_backend = public_backend_url
        self._callback_secret = callback_secret

    def _sign(self, payload: bytes) -> str | None:
        if not self._callback_secret:
            return None
        mac = hmac.new(self._callback_secret.encode("utf-8"), payload, hashlib.sha256).digest()
        return base64.b64encode(mac).decode("ascii")

    def submit_job(
        self,
        *,
        job_id: str,
        storage_key: str,
        mode: str = "full",
        outputs: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> None:
        callback_url = f"{self._public_backend}/api/v1/gpu/callback"
        body = {
            "job_id": job_id,
            "mode": mode,
            "input": {"storage_key": storage_key},
            "callback_url": callback_url,
        }
        if outputs:
            body["outputs"] = {
                name: {"storage_key": sk, "put_url": url} for name, (sk, url) in outputs.items()
            }
        headers = {"content-type": "application/json"}
        payload = json.dumps(body).encode("utf-8")
        sig = self._sign(payload)
        if sig:
            headers["x-gpu-signature"] = sig

        settings = get_settings()
        max_attempts = int(os.getenv("GPU_SUBMIT_MAX_RETRIES", "3"))
        base_backoff_ms = int(os.getenv("GPU_SUBMIT_INITIAL_BACKOFF_MS", "500"))
        jitter_ms = int(os.getenv("GPU_SUBMIT_JITTER_MS", "200"))

        attempt = 0
        while True:
            attempt += 1
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(f"{self._base_url}/jobs", content=payload, headers=headers)
                    resp.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if 400 <= status_code < 500:
                    logger.error("gpu_submit_client_error", extra={"status": status_code, "job_id": job_id})
                    raise RuntimeError(f"gpu submit failed: {exc}") from exc
                # else treat as retryable
            except (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as exc:  # type: ignore[attr-defined]
                # retryable
                pass

            if attempt >= max_attempts:
                logger.error("gpu_submit_failed", extra={"attempts": attempt, "job_id": job_id})
                raise RuntimeError("gpu submit failed after retries")

            sleep_ms = base_backoff_ms * (2 ** (attempt - 1)) + random.randint(0, jitter_ms)
            time.sleep(sleep_ms / 1000.0)


class CloudGpuClient:
    """Client for submitting jobs to a cloud-based GPU service (placeholder)."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        # ... other cloud-specific config

    def submit_job(self, *, job_id: str, storage_key: str, mode: str = "full") -> None:
        raise NotImplementedError("Cloud GPU client is not yet implemented.")


def get_gpu_client() -> GpuClient:
    """FastAPI dependency provider for the GPU client."""
    settings = get_settings()
    if settings.gpu_service_provider == "local":
        if not settings.gpu_service_base_url or not settings.public_backend_base_url:
            raise RuntimeError("Local GPU client requires GPU_SERVICE_BASE_URL and PUBLIC_BACKEND_BASE_URL")
        return LocalGpuClient(
            base_url=settings.gpu_service_base_url,
            public_backend_url=settings.public_backend_base_url,
            callback_secret=settings.gpu_callback_secret,
        )
    if settings.gpu_service_provider == "cloud":
        raise NotImplementedError("Cloud GPU client is not yet implemented.")
    raise ValueError(f"Unknown GPU service provider: {settings.gpu_service_provider}")
