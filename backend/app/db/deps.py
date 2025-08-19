from __future__ import annotations

from collections.abc import Generator

from sqlmodel import Session, select

from fastapi import Depends, HTTPException, status, Request
from typing import Optional
from functools import lru_cache
from app.core.config import get_settings
from app.api.v1.schemas import AuthenticatedUser
from app.db.models import User
from clerk_backend_api import Clerk
from clerk_backend_api import AuthenticateRequestOptions, RequestState


def get_db_session(request: Request) -> Generator[Session, None, None]:
    sessionmaker = request.app.state.db_sessionmaker
    session = sessionmaker()
    try:
        yield session
    finally:
        session.close()


@lru_cache(maxsize=1)
def _get_clerk_client() -> Clerk:
    settings = get_settings()
    if not settings.clerk_secret_key:
        raise RuntimeError("CLERK_SECRET_KEY not configured")
    return Clerk(bearer_auth=settings.clerk_secret_key)


def get_current_user(request: Request, session: Session = Depends(get_db_session)) -> AuthenticatedUser:
    """Validate a Clerk-issued JWT from the Authorization header and return the user."""
    settings = get_settings()
    clerk = _get_clerk_client()
    try:
        request_state: RequestState = clerk.authenticate_request(
            request,
            AuthenticateRequestOptions(authorized_parties=settings.authorized_parties),
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")

    if not request_state.is_signed_in:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="not signed in")

    claims = request_state.payload
    clerk_user_id: Optional[str] = claims.get("sub")
    if not clerk_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="token missing user id")

    user = session.exec(select(User).where(User.clerk_user_id == clerk_user_id)).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="user not provisioned")

    return AuthenticatedUser(clerk_user_id=clerk_user_id, email=user.email)
