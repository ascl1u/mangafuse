from __future__ import annotations

from collections.abc import Generator

from sqlmodel import Session

from app.db.session import get_sessionmaker


def get_db_session() -> Generator[Session, None, None]:
    SessionLocal = get_sessionmaker()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


