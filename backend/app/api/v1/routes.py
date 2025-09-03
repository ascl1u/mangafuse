from typing import Any, Dict
import uuid
import json
import hmac
import hashlib
import base64
from pathlib import Path
import logging

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import attributes

from app.core.config import get_settings
from app.api.v1.schemas import ApplyEditsRequest, AuthenticatedUser, ClerkWebhookEvent
from app.db.session import check_database_connection
from app.db.deps import get_current_user, get_db_session
from sqlmodel import Session, select
from app.db.models import User, Project, ProjectArtifact, ArtifactType, ProjectStatus
from svix.webhooks import Webhook, WebhookVerificationError
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from app.core.storage import get_storage_service, StorageService
from app.worker.queue import get_default_queue, get_high_priority_queue
from app.worker.tasks import process_translation, retypeset_after_edits
from app.core.gpu_client import GpuClient, get_gpu_client


router = APIRouter(prefix="/api/v1")

@router.get("/me", summary="Get current user")
def who_am_i(user: AuthenticatedUser = Depends(get_current_user)) -> Dict[str, Any]:
    return {"user": user.model_dump()}


@router.get("/healthz", summary="Liveness probe")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness probe")
def readyz(request: Request) -> Dict[str, str]:
    # Check database connectivity and GPU service configuration only
    if not check_database_connection(request.app.state.db_engine):
        raise HTTPException(status_code=503, detail="database not reachable")
    settings = get_settings()
    if not settings.gpu_service_base_url:
        raise HTTPException(status_code=503, detail="gpu service not configured")
    return {"status": "ready"}


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
    project_id = str(uuid.uuid4())
    url = storage.get_upload_url(project_id, filename)
    planned_key = storage.get_planned_upload_storage_key(project_id, filename)
    return {"project_id": project_id, "url": url, "storage_key": planned_key}


@router.post("/projects", summary="Create a new project", status_code=status.HTTP_202_ACCEPTED)
async def create_project(
    project_id: str,
    filename: str,
    storage_key: str,
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
    gpu_client: GpuClient = Depends(get_gpu_client),
    storage: StorageService = Depends(get_storage_service),
    file: UploadFile | None = File(None),
) -> Dict[str, str]:
    """Create a project record and submit GPU job. Returns immediately with 202."""
    # In local dev, a file may be uploaded directly to this endpoint.
    # If so, we save it to the shared artifacts volume.
    if get_settings().app_env == "development" and file:
        contents = await file.read()
        storage_key = storage.save_artifact(project_id, filename, contents)

    db_user = session.exec(select(User).where(User.clerk_user_id == user.clerk_user_id)).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Idempotency handling using Redis (via RQ queue connection)
    idemp_key = request.headers.get("X-Idempotency-Key")
    q = get_default_queue()
    r = q.connection
    idemp_redis_key = None
    if idemp_key:
        # Namespace by user to avoid cross-user collisions
        idemp_redis_key = f"idemp:create_project:{db_user.id}:{idemp_key}"
        cached = r.get(idemp_redis_key)
        if cached:
            try:
                raw = cached.decode("utf-8") if isinstance(cached, (bytes, bytearray)) else cached
                resp = json.loads(raw)
                # Return immediately with cached response
                return {"project_id": resp.get("project_id", project_id)}
            except Exception:
                # Fall through on parse errors
                pass

        # If the project already exists, treat as idempotent success
        existing = session.exec(
            select(Project).where(Project.id == project_id, Project.user_id == db_user.id)
        ).first()
        if existing:
            return {"project_id": str(existing.id)}

        # Acquire a short lock to prevent duplicate GPU submissions on concurrent retries
        lock_key = f"{idemp_redis_key}:lock"
        acquired = r.set(lock_key, "1", nx=True, ex=60)
        if not acquired:
            # Another identical request is in-flight; return a stable response without re-submitting
            return {"project_id": project_id}

    project = Project(id=project_id, user_id=db_user.id, title=filename)
    artifact = ProjectArtifact(project_id=project.id, artifact_type=ArtifactType.SOURCE_RAW, storage_key=storage_key)
    # Mark as processing immediately
    project.status = ProjectStatus.PROCESSING

    # Submit GPU job first; commit only if submission succeeds to avoid orphaned rows
    try:
        # Prepare presigned outputs for decoupled artifact handling (if supported by storage)
        outputs: dict[str, tuple[str, str]] = {}
        for name in ("CLEANED_PAGE", "TEXT_JSON"):
            pair = storage.get_output_upload_url(project_id, f"{name.lower()}.tmp")
            if pair:
                outputs[name] = pair
        gpu_client.submit_job(job_id=str(project.id), storage_key=storage_key, mode="full", outputs=outputs or None)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"gpu submit failed: {exc}")

    session.add(project)
    session.add(artifact)
    session.commit()

    response_payload = {"project_id": str(project.id)}
    # Cache idempotent result for a limited time
    if idemp_key and idemp_redis_key:
        try:
            r.setex(idemp_redis_key, 60 * 60 * 24, json.dumps(response_payload))
            r.delete(f"{idemp_redis_key}:lock")
        except Exception:
            # Best-effort; ignore caching errors
            pass

    return response_payload


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

    payload: Dict[str, Any] = {"project_id": str(project.id), "status": project.status, "error": project.failure_reason}
    # Expose editor revision so the frontend can wait for a specific update
    payload["editor_data_rev"] = int(project.editor_data_rev or 0)

    # Provide coarse progress and stage derived from status for a better UX
    try:
        status_val: ProjectStatus = project.status
        stage_map = {
            ProjectStatus.PENDING: ("pending", 0.0),
            ProjectStatus.PROCESSING: ("processing", 0.3),
            ProjectStatus.TRANSLATING: ("translating", 0.7),
            ProjectStatus.TYPESETTING: ("typesetting", 0.9),
            ProjectStatus.UPDATING: ("updating", 0.6),
            ProjectStatus.COMPLETED: ("completed", 1.0),
            ProjectStatus.FAILED: ("failed", 1.0),
        }
        stage, progress = stage_map.get(status_val, ("unknown", None))
        if progress is not None:
            payload["meta"] = {"stage": stage, "progress": float(progress)}
        else:
            payload["meta"] = {"stage": stage}
    except Exception:
        # Avoid breaking the endpoint if mapping fails for any reason
        payload["meta"] = {"stage": "unknown"}

    if project.status in [ProjectStatus.COMPLETED, ProjectStatus.FAILED]:
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
    request: Request,
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> Dict[str, str]:
    """Atomically update project data and enqueue a re-typeset job."""
    db_user = session.exec(select(User).where(User.clerk_user_id == user.clerk_user_id)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    project = session.exec(
        select(Project).where(Project.id == project_id, Project.user_id == db_user.id)
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not isinstance(project.editor_data, dict) or "bubbles" not in project.editor_data:
        raise HTTPException(status_code=409, detail="Project is not in a valid state for editing.")

    # Idempotency handling for apply-edits using Redis
    idemp_key = request.headers.get("X-Idempotency-Key")
    idemp_redis_key = None
    if idemp_key:
        q = get_high_priority_queue()
        r = q.connection
        idemp_redis_key = f"idemp:update_project:{project.user_id}:{idemp_key}"
        cached = r.get(idemp_redis_key)
        if cached:
            return {"project_id": project_id}
        lock_key = f"{idemp_redis_key}:lock"
        acquired = r.set(lock_key, "1", nx=True, ex=60)
        if not acquired:
            return {"project_id": project_id}
    # Create a mutable copy of the editor data and a lookup map for bubbles.
    new_editor_data = dict(project.editor_data)
    bubbles = new_editor_data.get("bubbles", [])
    bubbles_by_id = {b.get("id"): b for b in bubbles if "id" in b}

    # Apply incoming edits directly to the bubbles list.
    edited_bubble_ids = []
    for edit in body.edits:
        bubble = bubbles_by_id.get(edit.id)
        if bubble:
            edited_bubble_ids.append(edit.id)
            if edit.en_text is not None:
                bubble["en_text"] = edit.en_text
            if edit.font_size is not None:
                bubble["font_size"] = edit.font_size
            else:  # If font size is not provided, clear any existing one to trigger recalculation.
                bubble.pop("font_size", None)

    # The 'edits' list is no longer stored, 'bubbles' is the single source of truth.
    new_editor_data.pop("edits", None)
    project.editor_data = new_editor_data
    # Mark the JSONB field as modified to ensure SQLAlchemy detects the change.
    attributes.flag_modified(project, "editor_data")

    project.editor_data_rev = int(project.editor_data_rev or 0) + 1
    project.status = ProjectStatus.UPDATING
    session.add(project)
    session.commit()

    # Enqueue job, now passing the list of edited bubble IDs for optimization.
    try:
        q = get_high_priority_queue()
        job_id = f"retypeset-{project_id}-{project.editor_data_rev}"
        q.enqueue(
            retypeset_after_edits,
            project_id,
            project.editor_data_rev,
            edited_bubble_ids=edited_bubble_ids,  # Pass the list of IDs
            job_id=job_id,
            at_front=True,
        )
    except Exception as exc:
        # If enqueuing fails, roll back the status to avoid getting stuck in UPDATING.
        project.status = ProjectStatus.FAILED
        project.failure_reason = f"enqueue_failed: {exc}"
        session.add(project)
        session.commit()
        raise HTTPException(status_code=503, detail="Queue unavailable")

    # Cache idempotent result for a limited time
    if idemp_key and idemp_redis_key:
        try:
            r = get_high_priority_queue().connection
            r.setex(idemp_redis_key, 60 * 60 * 24, json.dumps({"project_id": project_id}))
            r.delete(f"{idemp_redis_key}:lock")
        except Exception:
            pass

    return {"project_id": project_id}


def _verify_gpu_webhook_signature(raw: bytes, header_sig: str | None) -> None:
    settings = get_settings()
    secret = settings.gpu_callback_secret
    if not secret:
        return
    if not header_sig:
        raise HTTPException(status_code=401, detail="missing signature")
    mac = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("ascii")
    if not hmac.compare_digest(expected, header_sig):
        raise HTTPException(status_code=401, detail="invalid signature")


@router.post("/gpu/callback", summary="GPU job completion webhook")
async def gpu_callback(
    request: Request, session: Session = Depends(get_db_session), storage: StorageService = Depends(get_storage_service)
) -> Dict[str, str]:
    raw = await request.body()
    _verify_gpu_webhook_signature(raw, request.headers.get("x-gpu-signature"))
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    job_id = payload.get("job_id")
    status_val = payload.get("status")
    if not job_id or status_val not in ("COMPLETED", "FAILED"):
        raise HTTPException(status_code=400, detail="invalid payload")

    project = session.get(Project, job_id)
    if not project:
        raise HTTPException(status_code=404, detail="project not found")

    if status_val == "FAILED":
        project.status = ProjectStatus.FAILED
        project.failure_reason = payload.get("error")
        session.add(project)
        session.commit()
        return {"status": "ok"}

    # Persist artifact references only; enqueue worker job
    artifacts = payload.get("artifacts") or {}
    project.status = ProjectStatus.TRANSLATING
    session.add(project)
    session.commit()

    try:
        q = get_default_queue()
        job_id_enq = f"translate-{job_id}"
        # Idempotency for duplicate GPU callbacks: no-op if the job already exists
        existing = q.fetch_job(job_id_enq)
        if not existing:
            q.enqueue(
                process_translation,
                job_id,
                artifacts,
                job_id=job_id_enq,
            )
    except Exception as exc:
        # Treat enqueue errors as transient; avoid flipping to FAILED on duplicate callback races
        logging.getLogger(__name__).exception("enqueue_translate_failed", extra={"project_id": job_id})
        raise HTTPException(status_code=503, detail="Queue unavailable")

    return {"status": "ok"}


@router.get("/projects/{project_id}/download", summary="Download packaged artifacts (zip)")
def download_package(
    project_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
    storage: StorageService = Depends(get_storage_service),
):
    """
    Returns a redirect to a presigned URL for the project's downloadable zip archive.
    The zip file is generated asynchronously by a background worker.
    """
    db_user = session.exec(select(User).where(User.clerk_user_id == user.clerk_user_id)).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    project = session.exec(
        select(Project).where(Project.id == project_id, Project.user_id == db_user.id)
    ).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Find the pre-generated zip artifact in the database.
    zip_artifact = session.exec(
        select(ProjectArtifact).where(
            ProjectArtifact.project_id == project.id,
            ProjectArtifact.artifact_type == ArtifactType.DOWNLOADABLE_ZIP,
        )
    ).first()

    if not zip_artifact:
        raise HTTPException(
            status_code=404,
            detail="Download package not found. It may still be generating.",
        )

    # Generate a presigned URL for direct download from cloud storage.
    download_url = storage.get_download_url(zip_artifact.storage_key)

    # Redirect the client to the presigned URL.
    return RedirectResponse(url=download_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


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