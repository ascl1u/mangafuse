from typing import Any, Dict
import uuid

import redis
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Depends, Request
from fastapi.responses import StreamingResponse
from celery.result import AsyncResult

from app.core.config import get_settings
from app.worker.celery_app import celery_app
from app.api.v1.schemas import ApplyEditsRequest, AuthenticatedUser, ClerkWebhookEvent
from app.db.session import check_database_connection
from app.db.deps import get_current_user, get_db_session
from sqlmodel import Session, select
from app.db.models import User, Project, ProjectArtifact, ArtifactType
from svix.webhooks import Webhook, WebhookVerificationError
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from app.core.storage import get_storage_service, StorageService


router = APIRouter(prefix="/api/v1")

@router.get("/me", summary="Get current user")
def who_am_i(user: AuthenticatedUser = Depends(get_current_user)) -> Dict[str, Any]:
    return {"user": user.model_dump()}


@router.get("/healthz", summary="Liveness probe")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
def readyz() -> Dict[str, str]:
    settings = get_settings()
    # If no Redis configured yet, treat as not ready to surface misconfig early.
    if not settings.effective_broker_url:
        raise HTTPException(status_code=503, detail="redis not configured")
    try:
        client = redis.Redis.from_url(settings.effective_broker_url, socket_connect_timeout=0.5)
        pong: Any = client.ping()
        if pong is True:
            return {"status": "ready"}
        raise RuntimeError("unexpected redis ping result")
    except Exception as exc:  # noqa: BLE001 - surface readiness failure as 503
        raise HTTPException(status_code=503, detail=f"redis not reachable: {exc}")


@router.get("/dbz", summary="Database readiness probe")
def dbz(request: Request) -> Dict[str, str]:
    if not check_database_connection(request.app.state.db_engine):
        raise HTTPException(status_code=503, detail="database not reachable")
    return {"status": "ready"}


@router.post("/projects/upload-url", summary="Get a URL to upload a source file")
def get_upload_url(
    filename: str,
    user: AuthenticatedUser = Depends(get_current_user),
    storage: StorageService = Depends(get_storage_service),
) -> Dict[str, str]:
    """Get a URL to upload a source file."""
    # In the future, this will generate a presigned URL for direct-to-cloud upload.
    # For local dev, it just confirms the user is authenticated and returns a path.
    project_id = str(uuid.uuid4())
    url = storage.get_upload_url(project_id, filename)
    return {"project_id": project_id, "url": url}


@router.post("/projects", summary="Create a new project", status_code=status.HTTP_202_ACCEPTED)
def create_project(
    project_id: str,
    filename: str,
    storage_key: str,
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> Dict[str, str]:
    """Create a project record and enqueue the processing task."""
    db_user = session.exec(select(User).where(User.clerk_user_id == user.clerk_user_id)).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    project = Project(id=project_id, user_id=db_user.id, title=filename)
    artifact = ProjectArtifact(
        project_id=project.id,
        artifact_type=ArtifactType.SOURCE_RAW,
        storage_key=storage_key,
    )
    session.add(project)
    session.add(artifact)
    session.commit()
    session.refresh(project)

    async_result = celery_app.send_task(
        "app.worker.tasks.process_page_task",
        args=[str(project.id)],
        kwargs={},
    )
    project.celery_task_id = async_result.id
    session.add(project)
    session.commit()

    return {"project_id": str(project.id), "task_id": async_result.id}


@router.post("/projects/{project_id}/upload", summary="Upload a source file (local dev)")
async def upload_source_file(
    project_id: str,
    file: UploadFile = File(...),
    user: AuthenticatedUser = Depends(get_current_user),
    storage: StorageService = Depends(get_storage_service),
) -> Dict[str, str]:
    """Endpoint for local development to handle file uploads."""
    if get_settings().app_env != "development":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")

    contents = await file.read()
    storage_key = storage.save_artifact(project_id, file.filename, contents)
    return {"storage_key": storage_key}


@router.get("/projects/{project_id}", summary="Get project status and results")
def get_project(
    project_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
    storage: StorageService = Depends(get_storage_service),
) -> Dict[str, Any]:
    """Get project status, and if complete, the artifact URLs."""
    db_user = session.exec(select(User).where(User.clerk_user_id == user.clerk_user_id)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    project = session.exec(
        select(Project).where(Project.id == project_id, Project.user_id == db_user.id)
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    payload: Dict[str, Any] = {"project_id": str(project.id), "status": project.status}
    if project.celery_task_id:
        result = AsyncResult(project.celery_task_id, app=celery_app)
        payload["task_state"] = result.state
        if isinstance(result.info, dict):
            payload["meta"] = {k: result.info.get(k) for k in ("stage", "progress")}
        if result.failed():
            payload["error"] = str(result.result)

    if project.status == "COMPLETED":
        artifacts = session.exec(select(ProjectArtifact).where(ProjectArtifact.project_id == project.id)).all()
        payload["artifacts"] = {
            art.artifact_type.value: storage.get_download_url(art.storage_key) for art in artifacts
        }
        payload["editor_data"] = project.editor_data

    return payload


@router.put("/projects/{project_id}", summary="Update project and re-typeset", status_code=status.HTTP_202_ACCEPTED)
def update_project(
    project_id: str,
    body: ApplyEditsRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> Dict[str, str]:
    """Update project editor data and enqueue a re-typeset task."""
    db_user = session.exec(select(User).where(User.clerk_user_id == user.clerk_user_id)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    project = session.exec(
        select(Project).where(Project.id == project_id, Project.user_id == db_user.id)
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Update editor_data (simplified for now)
    if not project.editor_data:
        project.editor_data = {}
    project.editor_data["edits"] = [e.model_dump() for e in body.edits]
    session.add(project)
    session.commit()

    async_result = celery_app.send_task(
        "app.worker.tasks.apply_edits_task",
        args=[str(project.id)],
        kwargs={},
    )
    return {"task_id": async_result.id}


@router.get("/projects/{project_id}/download", summary="Download packaged artifacts (zip)")
def download_package(
    project_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
    storage: StorageService = Depends(get_storage_service),
):
    """Stream a zip containing available artifacts for a project."""
    import io
    import zipfile

    db_user = session.exec(select(User).where(User.clerk_user_id == user.clerk_user_id)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    project = session.exec(
        select(Project).where(Project.id == project_id, Project.user_id == db_user.id)
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    artifacts = session.exec(select(ProjectArtifact).where(ProjectArtifact.project_id == project.id)).all()
    if not artifacts:
        raise HTTPException(status_code=404, detail="No artifacts found for this project")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for art in artifacts:
            try:
                data = storage.get_artifact(art.storage_key)
                zf.writestr(f"{art.artifact_type.value.lower()}.png", data.read())
            except FileNotFoundError:
                # In a real app, you might log this or handle it differently
                continue
    buf.seek(0)

    filename = f"mangafuse_{project_id}.zip"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "Cache-Control": "no-cache",
    }
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


# TODO: This is a placeholder for the Clerk webhook. We need to implement the actual webhook from dashboard  

def _verify_clerk_webhook(request: Request, body: bytes) -> None:
    """Verify Clerk webhook using Svix headers when secret is configured.

    If no secret configured, accept (useful for local dev).
    """
    settings = get_settings()
    secret = settings.clerk_webhook_secret
    if not secret:
        return
    svix_id = request.headers.get("svix-id")
    svix_timestamp = request.headers.get("svix-timestamp")
    svix_signature = request.headers.get("svix-signature")
    if not (svix_id and svix_timestamp and svix_signature):
        raise HTTPException(status_code=401, detail="missing svix headers")
    headers = {"svix-id": svix_id, "svix-timestamp": svix_timestamp, "svix-signature": svix_signature}
    try:
        Webhook(secret).verify(body, headers)
    except WebhookVerificationError:
        raise HTTPException(status_code=401, detail="invalid signature")

# TODO: This is a placeholder for the Clerk webhook. We need to implement the actual webhook from dashboard  
@router.post("/clerk/webhook", summary="Clerk user sync webhook", status_code=status.HTTP_200_OK)
async def clerk_webhook(request: Request, session: Session = Depends(get_db_session)) -> Dict[str, str]:
    raw = await request.body()
    _verify_clerk_webhook(request, raw)
    try:
        payload = ClerkWebhookEvent.model_validate_json(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid payload")

    event_type = payload.type
    data = payload.data
    primary_email: str | None = None
    if data.primary_email_address_id and data.email_addresses:
        for ent in data.email_addresses:
            if ent.get("id") == data.primary_email_address_id:
                primary_email = ent.get("email_address")
                break
    if not primary_email and data.email_addresses:
        # Fallback to first email
        primary_email = data.email_addresses[0].get("email_address")

    if event_type in {"user.created", "user.updated"}:
        try:
            with session.begin():
                existing = session.exec(select(User).where(User.clerk_user_id == data.id)).first()
                if existing:
                    if data.deleted:
                        existing.deactivated_at = existing.deactivated_at or datetime.now(timezone.utc)
                    else:
                        if primary_email and existing.email != primary_email:
                            existing.email = primary_email
                    session.add(existing)
                else:
                    if not data.deleted:
                        if not primary_email:
                            raise HTTPException(status_code=400, detail="missing email")
                        session.add(User(clerk_user_id=data.id, email=primary_email))
        except IntegrityError:
            # Likely a unique constraint conflict on email; allow Clerk to retry or treat as idempotent
            raise HTTPException(status_code=409, detail="user conflict")
        except SQLAlchemyError:
            raise HTTPException(status_code=503, detail="database unavailable")
    elif event_type == "user.deleted":
        try:
            with session.begin():
                existing = session.exec(select(User).where(User.clerk_user_id == data.id)).first()
                if existing and not existing.deactivated_at:
                    existing.deactivated_at = datetime.now(timezone.utc)
                    session.add(existing)
        except SQLAlchemyError:
            raise HTTPException(status_code=503, detail="database unavailable")

    return {"status": "ok"}
