from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict

from app.worker.celery_app import celery_app
from app.pipeline.orchestrator import run_pipeline as orchestrator_run_pipeline, apply_edits as orchestrator_apply_edits
from app.core.storage import get_storage_service
from app.core.paths import get_job_dir
from app.db.session import worker_session_scope
from app.db.models import Project, ProjectArtifact, ArtifactType, ProjectStatus
from sqlmodel import select

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.worker.tasks.process_page_task")
def process_page_task(self, project_id: str) -> Dict[str, Any]:
    """Run the MangaFuse pipeline for a project."""
    storage = get_storage_service()
    with worker_session_scope() as session:
        project = session.get(Project, project_id)
        if not project:
            logger.error("Project not found", extra={"project_id": project_id})
            return {"error": "Project not found"}

        source_artifact = session.exec(
            select(ProjectArtifact).where(
                ProjectArtifact.project_id == project_id,
                ProjectArtifact.artifact_type == ArtifactType.SOURCE_RAW,
            )
        ).first()
        if not source_artifact:
            project.status = ProjectStatus.FAILED
            project.failure_reason = "Source artifact not found"
            session.add(project)
            return {"error": "Source artifact not found"}

        # Capture storage key before leaving the session to avoid detached instance issues
        source_storage_key = source_artifact.storage_key

        project.status = ProjectStatus.PROCESSING
        session.add(project)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "source_image"
        with open(tmp_path, "wb") as f:
            with storage.get_artifact(source_storage_key) as rf:
                f.write(rf.read())

        def _update(stage: str, progress: float) -> None:
            self.update_state(state="PROGRESS", meta={"stage": stage, "progress": progress})

        try:
            result = orchestrator_run_pipeline(
                job_id=project_id,
                image_path=str(tmp_path),
                depth="full",  # Simplified for now
                progress_callback=_update,
            )

            with worker_session_scope() as session:
                project = session.get(Project, project_id)
                # Upload artifacts
                for artifact_type, artifact_path in result["artifacts"].items():
                    # Read file fully before writing to avoid truncation when src == dest
                    with open(artifact_path, "rb") as f:
                        payload = f.read()
                    storage_key = storage.save_artifact(project_id, Path(artifact_path).name, payload)
                    # Upsert artifact record for idempotency
                    existing = session.exec(
                        select(ProjectArtifact).where(
                            ProjectArtifact.project_id == project_id,
                            ProjectArtifact.artifact_type == ArtifactType(artifact_type),
                        )
                    ).first()
                    if existing:
                        existing.storage_key = storage_key
                        session.add(existing)
                    else:
                        artifact = ProjectArtifact(
                            project_id=project_id,
                            artifact_type=ArtifactType(artifact_type),
                            storage_key=storage_key,
                        )
                        session.add(artifact)

                project.status = ProjectStatus.COMPLETED
                project.editor_data = result["editor_payload"]
                session.add(project)
            return {"project_id": project_id, "status": "completed"}

        except Exception as e:
            logger.exception("Pipeline failed", extra={"project_id": project_id})
            with worker_session_scope() as session:
                project = session.get(Project, project_id)
                project.status = ProjectStatus.FAILED
                project.failure_reason = str(e)
                session.add(project)
            raise


@celery_app.task(bind=True, name="app.worker.tasks.apply_edits_task")
def apply_edits_task(self, project_id: str) -> Dict[str, Any]:
    """Apply user edits and re-typeset."""
    storage = get_storage_service()
    with worker_session_scope() as session:
        project = session.get(Project, project_id)
        if not project or not project.editor_data:
            return {"error": "Project or editor data not found"}

        # Ensure local job directory exists and required inputs are present
        job_dir = get_job_dir(project_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        # 1) Download cleaned background image from storage
        cleaned_artifact = session.exec(
            select(ProjectArtifact).where(
                ProjectArtifact.project_id == project_id,
                ProjectArtifact.artifact_type == ArtifactType.CLEANED_PAGE,
            )
        ).first()
        # Fallback: if CLEANED_PAGE is missing, try FINAL_PNG as background
        if not cleaned_artifact:
            cleaned_artifact = session.exec(
                select(ProjectArtifact).where(
                    ProjectArtifact.project_id == project_id,
                    ProjectArtifact.artifact_type == ArtifactType.FINAL_PNG,
                )
            ).first()
        if not cleaned_artifact:
            return {"error": "No cleaned or final artifact found for re-typesetting"}

        cleaned_path = job_dir / "cleaned.png"
        with open(cleaned_path, "wb") as wf:
            with storage.get_artifact(cleaned_artifact.storage_key) as rf:
                wf.write(rf.read())

        # 2) Reconstruct text.json from editor_data.bubbles
        bubbles = project.editor_data.get("bubbles") if isinstance(project.editor_data, dict) else None
        if not bubbles or not isinstance(bubbles, list):
            return {"error": "Editor data missing bubbles for re-typesetting"}

        # Minimal writer to match pipeline format: {"bubbles": [...]}
        import json
        json_path = job_dir / "text.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"bubbles": bubbles}, f, ensure_ascii=False, indent=2)

        # 3) Invoke orchestrator to apply edits
        edits = project.editor_data.get("edits", [])
        result = orchestrator_apply_edits(job_dir, edits)

        # Upload new artifacts
        for artifact_type, artifact_path in result["artifacts"].items():
            with open(artifact_path, "rb") as f:
                payload = f.read()
            storage_key = storage.save_artifact(project_id, Path(artifact_path).name, payload)
            
            # Update existing artifact or create new one
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
        
        return {"project_id": project_id, "status": "edits applied"}
