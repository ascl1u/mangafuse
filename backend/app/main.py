from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.v1.routes import router as api_router
from app.api.v1.billing import router as billing_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.paths import get_artifacts_root, get_assets_root
from app.db.session import create_engine_and_sessionmaker


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401 - FastAPI lifespan signature
    settings = get_settings()
    configure_logging(settings.log_level)

    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    engine, sessionmaker = create_engine_and_sessionmaker()
    app.state.db_engine = engine
    app.state.db_sessionmaker = sessionmaker

    logging.getLogger(__name__).info(
        "app_start",
        extra={
            "env": settings.app_env,
        },
    )
    yield
    app.state.db_engine.dispose()
    logging.getLogger(__name__).info("app_stop")


def create_app() -> FastAPI:
    app = FastAPI(title="MangaFuse API", version="0.1.0", lifespan=lifespan)

    # Log CORS configuration on startup
    logging.getLogger(__name__).info(
        "CORS configured",
        extra={
            "allow_origins": [
                "http://localhost:5173",
                "https://mangafuse.com",
                "https://www.mangafuse.com",
            ]
        }
    )

    # CORS for local development and production frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",          # Local development
            "https://mangafuse.com",          # Production domain
            "https://www.mangafuse.com",      # WWW subdomain
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.include_router(billing_router)

    # CORS debug endpoint
    @app.get("/cors-test")
    def cors_test():
        return {"message": "CORS is working!", "origin": "allowed"}

    settings = get_settings()
    # Serve artifacts/ only in development (local filesystem storage)
    artifacts_dir = get_artifacts_root()
    if settings.app_env == "development":
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        static_files_app = StaticFiles(directory=str(artifacts_dir), html=False)
        # Mount the StaticFiles app, wrapped in a new CORSMiddleware instance
        app.mount("/artifacts", CORSMiddleware(
            static_files_app,
            allow_origins=["http://localhost:5173"], # Be specific for dev
            allow_credentials=True,
            allow_methods=["GET"], # Only GET is needed for static files
            allow_headers=["*"],
        ), name="artifacts")
    assets_dir = get_assets_root()
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir), html=False), name="assets")
    return app


app = create_app()
