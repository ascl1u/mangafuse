from typing import Any, Dict
import uuid
import json
import hmac
import hashlib
import base64
from pathlib import Path
import logging

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Depends, Request
from fastapi.responses import StreamingResponse

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
from app.core.paths import get_job_dir
from app.pipeline.orchestrator import apply_edits as orchestrator_apply_edits
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
    return {"project_id": str(project.id)}


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
    """Update project editor data and synchronously re-typeset on CPU."""
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

    # Run CPU re-typeset synchronously (with local cache of cleaned image)
    job_dir = get_job_dir(project_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = job_dir / "cleaned.png"
    if not cleaned_path.exists():
        cleaned_artifact = session.exec(
            select(ProjectArtifact).where(
                ProjectArtifact.project_id == project_id,
                ProjectArtifact.artifact_type == ArtifactType.CLEANED_PAGE,
            )
        ).first()
        if not cleaned_artifact:
            cleaned_artifact = session.exec(
                select(ProjectArtifact).where(
                    ProjectArtifact.project_id == project_id,
                    ProjectArtifact.artifact_type == ArtifactType.FINAL_PNG,
                )
            ).first()
        if not cleaned_artifact:
            raise HTTPException(status_code=400, detail="No cleaned or final artifact found for re-typesetting")
        storage_dep: StorageService = get_storage_service()
        with open(cleaned_path, "wb") as wf:
            with storage_dep.get_artifact(cleaned_artifact.storage_key) as rf:
                wf.write(rf.read())

    # 2) Reconstruct text.json from editor_data.bubbles
    bubbles = project.editor_data.get("bubbles") if isinstance(project.editor_data, dict) else None
    if not bubbles or not isinstance(bubbles, list):
        raise HTTPException(status_code=400, detail="Editor data missing bubbles for re-typesetting")
    json_path = job_dir / "text.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"bubbles": bubbles}, f, ensure_ascii=False, indent=2)

    # 3) Apply edits from request body
    edits = [e.model_dump() for e in body.edits]
    result = orchestrator_apply_edits(job_dir, edits)

    # Upload updated artifacts
    storage = get_storage_service()
    # Upload artifacts (final, text_layer) sequentially (can be parallelized later)
    for artifact_type, artifact_path in result.get("artifacts", {}).items():
        with open(artifact_path, "rb") as f:
            payload_bytes = f.read()
        storage_key = storage.save_artifact(project_id, Path(artifact_path).name, payload_bytes)
        existing_artifact = session.exec(
            select(ProjectArtifact).where(
                ProjectArtifact.project_id == project_id,
                ProjectArtifact.artifact_type == ArtifactType(artifact_type),
            )
        ).first()
        if existing_artifact:
            existing_artifact.storage_key = storage_key
            session.add(existing_artifact)
        else:
            new_artifact = ProjectArtifact(
                project_id=project_id,
                artifact_type=ArtifactType(artifact_type),
                storage_key=storage_key,
            )
            session.add(new_artifact)
    session.commit()
    return {"status": "ok"}


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
async def gpu_callback(request: Request, session: Session = Depends(get_db_session), storage: StorageService = Depends(get_storage_service)) -> Dict[str, str]:
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

    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = job_dir / "cleaned.png"
    json_path = job_dir / "text.json"

    # Prefer artifacts from callback (R2 storage_keys); fallback to shared FS for local dev
    try:
        artifacts = payload.get("artifacts") or {}
        if artifacts.get("CLEANED_PAGE"):
            with open(cleaned_path, "wb") as wf:
                with storage.get_artifact(artifacts["CLEANED_PAGE"]) as rf:
                    wf.write(rf.read())
        if artifacts.get("TEXT_JSON"):
            with open(json_path, "wb") as wf:
                with storage.get_artifact(artifacts["TEXT_JSON"]) as rf:
                    wf.write(rf.read())
    except Exception:
        pass

    if not cleaned_path.exists() or not json_path.exists():
        raise HTTPException(status_code=400, detail="missing cleaned or text.json from GPU result")

    # Translate on CPU (if ja_text exists and en_text missing), then initial CPU typeset
    try:
        from app.pipeline.utils.textio import read_text_json, save_text_records
        from app.pipeline.translate.gemini import GeminiTranslator
        import os
        data = read_text_json(json_path)
        bubbles = data.get("bubbles", [])
        texts: list[str] = []
        indices: list[int] = []
        for i, rec in enumerate(bubbles):
            ja = (rec.get("ja_text") or "").strip()
            has_en = isinstance(rec.get("en_text"), str) and rec["en_text"].strip() != ""
            if ja and not has_en:
                indices.append(i)
                texts.append(ja)
        if texts:
            translator = GeminiTranslator(api_key=os.getenv("GOOGLE_API_KEY", ""))
            en_list = translator.translate_batch(texts)
            for idx, en in zip(indices, en_list):
                bubbles[idx]["en_text"] = en
            save_text_records(json_path, bubbles)
    except Exception as exc:
        logging.warning("translation_failed", extra={"job_id": job_id, "error": str(exc)})
        # Proceed without translation on failure; typesetter will fallback to ja_text

    # Initial CPU typeset (no edits yet)
    result = orchestrator_apply_edits(job_dir, [])

    # Upload artifacts and update DB
    storage = get_storage_service()
    for artifact_type, artifact_path in result.get("artifacts", {}).items():
        with open(artifact_path, "rb") as f:
            payload_bytes = f.read()
        storage_key = storage.save_artifact(job_id, Path(artifact_path).name, payload_bytes)
        existing = session.exec(
            select(ProjectArtifact).where(
                ProjectArtifact.project_id == project.id,
                ProjectArtifact.artifact_type == ArtifactType(artifact_type),
            )
        ).first()
        if existing:
            existing.storage_key = storage_key
            session.add(existing)
        else:
            session.add(ProjectArtifact(project_id=project.id, artifact_type=ArtifactType(artifact_type), storage_key=storage_key))

    # Attach editor payload to project
    editor_payload_path = job_dir / "editor_payload.json"
    if editor_payload_path.exists():
        try:
            with open(editor_payload_path, "r", encoding="utf-8") as f:
                project.editor_data = json.load(f)
        except Exception:
            project.editor_data = None

    project.status = ProjectStatus.COMPLETED
    session.add(project)
    session.commit()
    return {"status": "ok"}


@router.get("/projects/{project_id}/download", summary="Download packaged artifacts (zip)")
def download_package(
    project_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
    storage: StorageService = Depends(get_storage_service),
):
    """Stream a zip containing available artifacts for a project without loading into memory."""
    import os
    import zipfile
    import tempfile
    from typing import Iterator

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

    # Build zip on disk to limit memory usage, then stream in chunks
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_path = tmp.name
    tmp.close()
    try:
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for art in artifacts:
                try:
                    with storage.get_artifact(art.storage_key) as rf:
                        # Write each artifact under a stable name
                        arcname = f"{art.artifact_type.value.lower()}.png"
                        # ZipFile.writestr accepts bytes; read in chunks to avoid loading entire file
                        # Accumulate into a temporary file inside the zip API using writestr once
                        data = rf.read()
                        zf.writestr(arcname, data)
                except FileNotFoundError:
                    logging.warning("artifact_missing", extra={"project_id": project_id, "key": art.storage_key})
                    continue

        def file_iterator(path: str, chunk_size: int = 1024 * 1024) -> Iterator[bytes]:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            try:
                os.remove(path)
            except Exception:
                pass

        filename = f"mangafuse_{project_id}.zip"
        headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "Cache-Control": "no-cache",
        }
        return StreamingResponse(file_iterator(tmp_path), media_type="application/zip", headers=headers)
    except Exception:
        # Ensure temp file is removed on failure
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


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
