from __future__ import annotations

import logging
import os


def main() -> None:
    mode = (os.getenv("RUN_MODE", "uvicorn").strip().lower())
    if mode == "serverless":
        # Start Runpod serverless worker loop
        try:
            import runpod  # type: ignore
            from app.gpu_service.runpod_handler import handler
            try:
                version = getattr(runpod, "__version__", "unknown")
            except Exception:
                version = "unknown"
            logging.getLogger(__name__).info(
                "runpod_serverless_start", extra={"runpod_version": version, "handler_signature": "handler(job)"}
            )
            runpod.serverless.start({"handler": handler})
        except Exception:
            logging.getLogger(__name__).exception("runpod_serverless_start_failed")
            raise
    else:
        # Default to local GPU FastAPI server
        import uvicorn  # type: ignore
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "5001"))
        uvicorn.run("app.gpu_service.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()


