from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from sqlmodel import select

from app.core.paths import get_job_dir
from app.core.storage import get_storage_service
from app.db.models import Project, ProjectArtifact, ArtifactType, ProjectStatus
from app.db.session import worker_session_scope
from app.pipeline.orchestrator import apply_edits as orchestrator_apply_edits
from app.worker.queue import get_default_queue


logger = logging.getLogger(__name__)


def _ensure_cleaned_and_text(job_id: str, artifacts: Dict[str, str] | None = None) -> tuple[Path, Path]:
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = job_dir / "cleaned.png"
    json_path = job_dir / "text.json"

    storage = get_storage_service()
    if artifacts:
        if artifacts.get("CLEANED_PAGE") and not cleaned_path.exists():
            with open(cleaned_path, "wb") as wf:
                with storage.get_artifact(artifacts["CLEANED_PAGE"]) as rf:
                    wf.write(rf.read())
        if artifacts.get("TEXT_JSON") and not json_path.exists():
            with open(json_path, "wb") as wf:
                with storage.get_artifact(artifacts["TEXT_JSON"]) as rf:
                    wf.write(rf.read())

    if not cleaned_path.exists() or not json_path.exists():
        raise RuntimeError("missing cleaned or text.json from GPU result")

    return cleaned_path, json_path


def process_translation(project_id: str, artifacts: Dict[str, str] | None = None) -> None:
    with worker_session_scope() as session:
        project = session.get(Project, project_id)
        if not project:
            logger.warning("project_missing", extra={"project_id": project_id})
            return

    # Ensure inputs present
    _, json_path = _ensure_cleaned_and_text(project_id, artifacts)

    # Translate missing en_text values, but do not fail hard if translation errors
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
        logger.warning("translation_failed", extra={"project_id": project_id, "error": str(exc)})

    # Chain to initial typeset on the default queue
    try:
        q = get_default_queue()
        job_id_enq = f"initial-typeset-{project_id}"
        q.enqueue(process_initial_typeset, project_id, job_id=job_id_enq)
    except Exception as exc:
        # If enqueue fails, mark project failed so the user sees a clear state
        with worker_session_scope() as session:
            project = session.get(Project, project_id)
            if project:
                project.status = ProjectStatus.FAILED
                project.failure_reason = f"enqueue_failed: {exc}"
                session.add(project)
        logger.exception("enqueue_initial_typeset_failed", extra={"project_id": project_id})


def process_initial_typeset(project_id: str) -> None:
    job_dir = get_job_dir(project_id)
    try:
        result = orchestrator_apply_edits(job_dir)
    except Exception as exc:
        # Mark project failed with a clear reason and stop
        with worker_session_scope() as session:
            project = session.get(Project, project_id)
            if project:
                project.status = ProjectStatus.FAILED
                project.failure_reason = f"typeset_failed: {exc}"
                session.add(project)
        logger.exception("initial_typeset_failed", extra={"project_id": project_id})
        return

    storage = get_storage_service()
    with worker_session_scope() as session:
        project = session.get(Project, project_id)
        if not project:
            return
        for artifact_type, artifact_path in result.get("artifacts", {}).items():
            with open(artifact_path, "rb") as f:
                payload_bytes = f.read()
            storage_key = storage.save_artifact(project_id, Path(artifact_path).name, payload_bytes)
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

        # Attach editor payload to project if present
        editor_payload_path = job_dir / "editor_payload.json"
        if editor_payload_path.exists():
            try:
                with open(editor_payload_path, "r", encoding="utf-8") as f:
                    project.editor_data = json.load(f)
            except Exception:
                project.editor_data = None

        project.status = ProjectStatus.COMPLETED
        session.add(project)


def retypeset_after_edits(project_id: str, revision: int, edited_bubble_ids: Optional[List[int]] = None) -> None:
    job_dir = get_job_dir(project_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    with worker_session_scope() as session:
        project = session.get(Project, project_id)
        if not project:
            logger.warning("project_missing", extra={"project_id": project_id})
            return
        # Skip stale job
        if project.editor_data_rev > revision:
            logger.info("stale_edit_job", extra={"project_id": project_id, "job_rev": revision, "current_rev": project.editor_data_rev})
            return

        # Ensure cleaned.png is present; fetch from storage if missing
        cleaned_path = job_dir / "cleaned.png"
        if not cleaned_path.exists():
            cleaned_artifact = session.exec(
                select(ProjectArtifact).where(
                    ProjectArtifact.project_id == project_id,
                    ProjectArtifact.artifact_type == ArtifactType.CLEANED_PAGE,
                )
            ).first()
            if not cleaned_artifact:
                # Fallback to FINAL_PNG if CLEANED_PAGE is missing
                cleaned_artifact = session.exec(
                    select(ProjectArtifact).where(
                        ProjectArtifact.project_id == project_id,
                        ProjectArtifact.artifact_type == ArtifactType.FINAL_PNG,
                    )
                ).first()
            if not cleaned_artifact:
                raise RuntimeError("No cleaned or final artifact found for re-typesetting")
            storage = get_storage_service()
            with open(cleaned_path, "wb") as wf:
                with storage.get_artifact(cleaned_artifact.storage_key) as rf:
                    wf.write(rf.read())

        # The API has already updated the bubbles list. This worker's only job
        # is to read that canonical state and re-run the typesetter.
        bubbles = project.editor_data.get("bubbles") if isinstance(project.editor_data, dict) else None
        if not bubbles or not isinstance(bubbles, list):
            raise RuntimeError("Editor data missing bubbles for re-typesetting")

        # Rebuild text.json from the canonical 'bubbles' list in the database.
        json_path = job_dir / "text.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"bubbles": bubbles}, f, ensure_ascii=False, indent=2)

    try:
        # Pass the edited_bubble_ids to the orchestrator for partial re-rendering.
        result = orchestrator_apply_edits(job_dir, edited_bubble_ids=edited_bubble_ids)
    except Exception as exc:
        with worker_session_scope() as session:
            project = session.get(Project, project_id)
            if project:
                project.status = ProjectStatus.FAILED
                project.failure_reason = f"typeset_failed: {exc}"
                # On failure, persist the editor payload which now contains all error fields.
                # This makes the complete set of errors available to the frontend.
                editor_payload_path = job_dir / "editor_payload.json"
                if editor_payload_path.exists():
                    try:
                        with open(editor_payload_path, "r", encoding="utf-8") as f:
                            project.editor_data = json.load(f)
                    except Exception:
                        logger.exception("failed_to_persist_editor_data_on_error", extra={"project_id": project_id})

                session.add(project)
        logger.exception("retypeset_failed", extra={"project_id": project_id, "revision": revision})
        return

    storage = get_storage_service()
    with worker_session_scope() as session:
        project = session.get(Project, project_id)
        if not project:
            return

        for artifact_type, artifact_path in result.get("artifacts", {}).items():
            with open(artifact_path, "rb") as f:
                payload_bytes = f.read()
            storage_key = storage.save_artifact(project_id, Path(artifact_path).name, payload_bytes)
            existing_artifact = session.exec(
                select(ProjectArtifact).where(
                    ProjectArtifact.project_id == project.id,
                    ProjectArtifact.artifact_type == ArtifactType(artifact_type),
                )
            ).first()
            if existing_artifact:
                existing_artifact.storage_key = storage_key
                session.add(existing_artifact)
            else:
                new_artifact = ProjectArtifact(
                    project_id=project.id,
                    artifact_type=ArtifactType(artifact_type),
                    storage_key=storage_key,
                )
                session.add(new_artifact)

        # Refresh editor payload with post-edit values if present
        editor_payload_path = job_dir / "editor_payload.json"
        if editor_payload_path.exists():
            try:
                with open(editor_payload_path, "r", encoding="utf-8") as f:
                    project.editor_data = json.load(f)
            except Exception:
                pass

        project.status = ProjectStatus.COMPLETED
        session.add(project)