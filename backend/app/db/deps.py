from __future__ import annotations

from collections.abc import Generator

from sqlmodel import Session, select

from fastapi import HTTPException, status, Request, Depends
from typing import Optional
from functools import lru_cache
from app.core.config import get_settings
from app.api.v1.schemas import AuthenticatedUser
from clerk_backend_api import Clerk
from clerk_backend_api import AuthenticateRequestOptions, RequestState
from app.db.models import Subscription


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


def get_current_user(request: Request) -> AuthenticatedUser:
    """Validate a Clerk-issued JWT from the Authorization header and return the user.

    Uses Clerk Direct ID approach - no database lookup required.
    """
    settings = get_settings()

    # Local development: allow mock authentication
    if settings.app_env == "development" and request.headers.get("X-Mock-Auth"):
        mock_user_id = request.headers.get("X-Mock-User-ID", "user_test_123")
        mock_email = request.headers.get("X-Mock-Email", "test@example.com")
        return AuthenticatedUser(clerk_user_id=mock_user_id, email=mock_email)

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

    # Extract email from JWT claims instead of database lookup
    email: Optional[str] = claims.get("email")

    return AuthenticatedUser(clerk_user_id=clerk_user_id, email=email)


def get_current_subscription(
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session)
) -> Subscription | None:
    """
    Retrieves the current user's subscription record directly using their clerk_user_id.
    Returns None if no subscription exists (free tier).
    """
    subscription = session.exec(
        select(Subscription).where(Subscription.user_id == user.clerk_user_id)
    ).first()
    return subscription
