from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.v1.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401 - FastAPI lifespan signature
    settings = get_settings()
    configure_logging(settings.log_level)
    logging.getLogger(__name__).info(
        "app_start",
        extra={
            "env": settings.app_env,
            "redis_configured": bool(settings.effective_broker_url),
            "redis_url": settings.effective_broker_url,
        },
    )
    yield
    logging.getLogger(__name__).info("app_stop")


def create_app() -> FastAPI:
    app = FastAPI(title="MangaFuse API", version="0.1.0", lifespan=lifespan)

    # CORS for local development and future frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Phase 1: permissive; will restrict later
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    # Serve artifacts/ for local verification (Step 3.1)
    repo_root = Path(__file__).resolve().parents[2]  # backend/ -> repo root
    artifacts_dir = repo_root / "artifacts"
    app.mount("/artifacts", StaticFiles(directory=str(artifacts_dir), html=False), name="artifacts")
    # Serve assets/ (fonts, models) for local development and editor previews
    assets_dir = repo_root / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir), html=False), name="assets")
    return app


app = create_app()


