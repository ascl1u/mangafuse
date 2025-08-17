from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, Session, create_engine

from app.core.config import get_settings


_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def _normalize_database_url(url: str) -> str:
    lower = url.lower()
    if lower.startswith("postgres://"):
        return "postgresql+psycopg://" + url.split("://", 1)[1]
    if lower.startswith("postgresql://") and "+" not in lower.split("://", 1)[0]:
        return "postgresql+psycopg://" + url.split("://", 1)[1]
    if lower.startswith("postgresql+psycopg2://"):
        return "postgresql+psycopg://" + url.split("://", 1)[1]
    return url


def get_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        if not settings.database_url:
            raise RuntimeError("DATABASE_URL is not configured")
        normalized = _normalize_database_url(settings.database_url)
        _engine = create_engine(
            normalized,
            echo=(settings.app_env == "development"),
            pool_pre_ping=True,
            future=True,
        )
        _SessionLocal = sessionmaker(bind=_engine, class_=Session, autoflush=False, autocommit=False)
    assert _engine is not None
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    SessionLocal = get_sessionmaker()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_connection() -> bool:
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except OperationalError:
        return False


def init_database_schema(create_all: bool = False) -> None:
    """Optionally create tables for simple local dev when migrations are not used.

    In production and normal development flows we will use Alembic migrations.
    """
    if create_all:
        engine = get_engine()
        SQLModel.metadata.create_all(engine)


