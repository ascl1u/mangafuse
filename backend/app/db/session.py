from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Tuple

from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, create_engine

from app.core.config import get_settings


def _normalize_database_url(url: str) -> str:
    lower = url.lower()
    if lower.startswith("postgres://"):
        return "postgresql+psycopg://" + url.split("://", 1)[1]
    if lower.startswith("postgresql://") and "+" not in lower.split("://", 1)[0]:
        return "postgresql+psycopg://" + url.split("://", 1)[1]
    if lower.startswith("postgresql+psycopg2://"):
        return "postgresql+psycopg://" + url.split("://", 1)[1]
    return url


def create_engine_and_sessionmaker() -> Tuple[Engine, sessionmaker]:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    normalized = _normalize_database_url(settings.database_url)
    engine = create_engine(
        normalized,
        echo=(settings.app_env == "development"),
        pool_pre_ping=True,
        future=True,
    )
    SessionLocal = sessionmaker(bind=engine, class_=Session, autoflush=False, autocommit=False)
    return engine, SessionLocal


def check_database_connection(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except OperationalError:
        return False


# The following functions are intended for use by Celery workers or scripts,
# where the FastAPI app state is not available.

_worker_engine: Engine | None = None
_worker_sessionmaker: sessionmaker | None = None

def get_worker_sessionmaker() -> sessionmaker:
    """Get a sessionmaker for a Celery worker. Caches the engine and sessionmaker."""
    global _worker_engine, _worker_sessionmaker
    if _worker_engine is None:
        _worker_engine, _worker_sessionmaker = create_engine_and_sessionmaker()
    assert _worker_sessionmaker is not None
    return _worker_sessionmaker


@contextmanager
def worker_session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations for a worker."""
    SessionLocal = get_worker_sessionmaker()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
