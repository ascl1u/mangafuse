from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import attributes  # 👈 Add this import
from sqlmodel import select

from app.core.paths import get_job_dir
from app.core.storage import get_storage_service
from app.db.models import Project, ProjectArtifact, ArtifactType, ProjectStatus
from app.db.session import worker_session_scope
from app.pipeline.orchestrator import apply_edits as orchestrator_apply_edits
from app.worker.queue import get_default_queue


logger = logging.getLogger(__name__)


def _upload_crops_for_project(project_id: str, job_dir: Path, storage, session) -> None:
    """Generate and upload bubble crops to storage using deterministic keys.

    This avoids persisting environment-specific URLs and ensures the API can
    presign crop URLs on read. Uses polygon fallback cropping; masks are not
    required at this stage.
    """
    try:
        # Read the latest editor_data from the database for authoritative bubble polygons
        project = session.get(Project, project_id)
        if not project or not isinstance(project.editor_data, dict):
            return
        bubbles = project.editor_data.get("bubbles") or []
        if not bubbles:
            return

        # Prefer RAW source image for crops; fallback to cleaned if unavailable
        import cv2  # type: ignore
        raw_path = job_dir / "source_image"
        img_path = None
        crop_source = "raw"

        if not raw_path.exists():
            # Try fetching SOURCE_RAW artifact from storage to local path
            try:
                raw_art = session.exec(
                    select(ProjectArtifact).where(
                        ProjectArtifact.project_id == project_id,
                        ProjectArtifact.artifact_type == ArtifactType.SOURCE_RAW,
                    )
                ).first()
                if raw_art:
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(raw_path, "wb") as wf:
                        with storage.get_artifact(raw_art.storage_key) as rf:
                            wf.write(rf.read())
            except Exception:
                pass

        if raw_path.exists():
            img_path = raw_path
            crop_source = "raw"
        else:
            cleaned_path = job_dir / "cleaned.png"
            if cleaned_path.exists():
                img_path = cleaned_path
                crop_source = "cleaned"

        if img_path is None or not img_path.exists():
            return

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return

        # Ensure local crops directory exists for temporary files
        crops_dir = job_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Crop and upload per bubble
        from app.pipeline.ocr.crops import tight_crop_from_mask  # lazy import
        uploaded = 0
        for rec in bubbles:
            try:
                bid = int(rec.get("id"))
            except Exception:
                continue
            polygon = rec.get("polygon") or []
            if not polygon:
                continue
            crop_bgr, _ = tight_crop_from_mask(img, None, polygon)
            # Skip degenerate crops
            if crop_bgr is None:
                continue
            h, w = crop_bgr.shape[:2]
            if h <= 0 or w <= 0:
                continue

            # Save to disk first, then upload via storage abstraction
            out_path = crops_dir / f"{bid}.png"
            try:
                cv2.imwrite(str(out_path), crop_bgr)
            except Exception:
                continue

            try:
                with open(out_path, "rb") as f:
                    storage.save_artifact(project_id, f"crops/{bid}.png", f)
                uploaded += 1
            except Exception:
                # Best-effort; continue with other crops
                continue

        logger.info("crops_uploaded", extra={"project_id": project_id, "count": uploaded, "source": crop_source})
    except Exception:
        # Swallow errors to avoid failing the whole job on crop upload issues
        logger.exception("crops_upload_failed", extra={"project_id": project_id})

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


def _package_and_upload_zip(project_id: str, storage, session) -> None:
    """Create a flat zip with final.png, cleaned.png, text_layer.png and upload.

    - Best-effort: failures are logged but do not alter project status.
    - Idempotent: updates existing DOWNLOADABLE_ZIP artifact if present.
    """
    try:
        # Ensure any prior DB writes are visible to subsequent selects
        try:
            session.flush()
        except Exception:
            pass

        job_dir = get_job_dir(project_id)
        zip_name = f"mangafuse_{project_id}.zip"
        zip_tmp_path = job_dir / f".tmp-{zip_name}"

        desired: List[tuple[ArtifactType, str]] = [
            (ArtifactType.FINAL_PNG, "final.png"),
            (ArtifactType.CLEANED_PAGE, "cleaned.png"),
            (ArtifactType.TEXT_LAYER_PNG, "text_layer.png"),
        ]

        # Build zip on disk to avoid large in-memory buffers. Use a tmp name to avoid self-overwrite in local storage.
        with zipfile.ZipFile(zip_tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            num_written = 0
            for art_type, arcname in desired:
                # Prefer local job_dir files first for freshest artifacts
                local_path = job_dir / arcname
                if local_path.exists():
                    try:
                        zf.write(local_path, arcname=arcname)
                        num_written += 1
                        continue
                    except Exception:
                        logger.exception(
                            "zip_add_file_failed",
                            extra={"project_id": project_id, "file": str(local_path)},
                        )
                # Fallback to storage if local file missing
                artifact_row = session.exec(
                    select(ProjectArtifact).where(
                        ProjectArtifact.project_id == project_id,
                        ProjectArtifact.artifact_type == art_type,
                    )
                ).first()
                if artifact_row and artifact_row.storage_key:
                    try:
                        with storage.get_artifact(artifact_row.storage_key) as rf:
                            zf.writestr(arcname, rf.read())
                            num_written += 1
                    except Exception:
                        logger.exception(
                            "zip_read_storage_failed",
                            extra={"project_id": project_id, "artifact_type": art_type.value},
                        )
            if num_written == 0:
                logger.info("zip_skipped_no_files", extra={"project_id": project_id})
                try:
                    if zip_tmp_path.exists():
                        zip_tmp_path.unlink()
                except Exception:
                    logger.warning("zip_tmp_cleanup_failed", extra={"project_id": project_id, "path": str(zip_tmp_path)})
                return

        size_bytes = zip_tmp_path.stat().st_size if zip_tmp_path.exists() else 0
        # Stream upload using file handle to avoid loading entire zip into memory
        with open(zip_tmp_path, "rb") as f:
            storage_key = storage.save_artifact(project_id, zip_name, f)

        # Upsert the DOWNLOADABLE_ZIP artifact
        existing_zip = session.exec(
            select(ProjectArtifact).where(
                ProjectArtifact.project_id == project_id,
                ProjectArtifact.artifact_type == ArtifactType.DOWNLOADABLE_ZIP,
            )
        ).first()
        if existing_zip:
            existing_zip.storage_key = storage_key
            session.add(existing_zip)
        else:
            session.add(
                ProjectArtifact(
                    project_id=project_id,
                    artifact_type=ArtifactType.DOWNLOADABLE_ZIP,
                    storage_key=storage_key,
                )
            )

        # Cleanup the temporary zip file regardless of storage backend.
        try:
            if zip_tmp_path.exists():
                zip_tmp_path.unlink()
        except Exception:
            # Non-fatal cleanup failure
            logger.warning("zip_tmp_cleanup_failed", extra={"project_id": project_id, "path": str(zip_tmp_path)})

        logger.info(
            "zip_packaged",
            extra={
                "project_id": project_id,
                "size_bytes": size_bytes,
                "num_files": num_written,
            },
        )
    except Exception:
        logger.exception("zip_package_failed", extra={"project_id": project_id})


def process_translation(project_id: str) -> None:
    # This function is now responsible for translating AND persisting the result to the DB.
    from app.pipeline.utils.textio import read_text_json
    from app.pipeline.translate.gemini import GeminiTranslator
    import os

    # No initial session scope needed, as we'll open one to save the final state.

    # 1. Download necessary artifacts from storage (now queries database instead of using parameter)
    _, json_path = _ensure_cleaned_and_text(project_id, None)

    # 2. Perform translation in memory
    data = read_text_json(json_path)
    bubbles = data.get("bubbles", [])

    try:
        texts_to_translate: list[str] = []
        indices_to_translate: list[int] = []
        for i, rec in enumerate(bubbles):
            ja = (rec.get("ja_text") or "").strip()
            has_en = isinstance(rec.get("en_text"), str) and rec["en_text"].strip() != ""
            if ja and not has_en:
                indices_to_translate.append(i)
                texts_to_translate.append(ja)

        if texts_to_translate:
            translator = GeminiTranslator(api_key=os.getenv("GOOGLE_API_KEY", ""))
            en_list = translator.translate_batch(texts_to_translate)
            for idx, en in zip(indices_to_translate, en_list):
                bubbles[idx]["en_text"] = en

        # 3. ✅ PERSIST THE TRANSLATED STATE TO THE DATABASE
        with worker_session_scope() as session:
            project = session.get(Project, project_id)
            if not project:
                logger.warning("project_missing_after_translation", extra={"project_id": project_id})
                return

            # The full data payload now includes the translations
            project.editor_data = data # data object now contains the modified bubbles list
            attributes.flag_modified(project, "editor_data")
            session.add(project)

    except Exception as exc:
        logger.warning("translation_failed", extra={"project_id": project_id, "error": str(exc)})
        with worker_session_scope() as session:
            project = session.get(Project, project_id)
            if project:
                project.completion_warnings = "Automated translation service failed. Please add text manually."
                session.add(project)

    # 4. Enqueue the next step
    try:
        q = get_default_queue()
        job_id_enq = f"initial-typeset-{project_id}"
        q.enqueue(process_initial_typeset, project_id, job_id=job_id_enq)
    except Exception as exc:
        with worker_session_scope() as session:
            project = session.get(Project, project_id)
            if project:
                project.status = ProjectStatus.FAILED
                project.failure_reason = f"enqueue_failed: {exc}"
                session.add(project)
        logger.exception("enqueue_initial_typeset_failed", extra={"project_id": project_id})


def process_initial_typeset(project_id: str) -> None:
    job_dir = get_job_dir(project_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    # 1. ✅ FETCH STATE FROM DATABASE
    # This is the crucial first step. It gets the translated data from the previous worker.
    with worker_session_scope() as session:
        project = session.get(Project, project_id)
        if not project:
            logger.warning("project_missing_for_typeset", extra={"project_id": project_id})
            return
        # If editor_data is missing, this task is now responsible for initializing it.
        # This occurs in the "cleaned" flow where the translation step is bypassed.
        if not project.editor_data:
            logger.info("initializing_editor_data_from_artifact", extra={"project_id": project_id})
            storage = get_storage_service()
            text_json_artifact = session.exec(
                select(ProjectArtifact).where(
                    ProjectArtifact.project_id == project_id, ProjectArtifact.artifact_type == ArtifactType.TEXT_JSON
                )
            ).first()

            if not text_json_artifact:
                project.status = ProjectStatus.FAILED
                project.failure_reason = "Critical artifact TEXT_JSON was not found after GPU processing."
                session.add(project)
                logger.error("missing_text_json_artifact", extra={"project_id": project_id})
                return

            try:
                with storage.get_artifact(text_json_artifact.storage_key) as f:
                    editor_data_payload = json.load(f)
                project.editor_data = editor_data_payload
                attributes.flag_modified(project, "editor_data")
                session.add(project)
                logger.info("editor_data_initialized_successfully", extra={"project_id": project_id})
            except Exception as e:
                project.status = ProjectStatus.FAILED
                project.failure_reason = f"Failed to load or parse TEXT_JSON artifact: {e}"
                session.add(project)
                logger.exception("failed_to_initialize_editor_data", extra={"project_id": project_id})
                return

        if not project.editor_data or not project.editor_data.get("bubbles"):
             # Handle case where segmentation found no bubbles. Mark as complete but with a warning.
            project.status = ProjectStatus.COMPLETED
            project.completion_warnings = "No speech bubbles were detected. The project is complete but no text was processed."
            session.add(project)
            return

        # 2. ✅ WRITE THE STATE TO A LOCAL text.json for the orchestrator
        json_path = job_dir / "text.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(project.editor_data, f, ensure_ascii=False, indent=2)

        # 3. ✅ Download the cleaned image, as it's not guaranteed to be local
        cleaned_path = job_dir / "cleaned.png"
        if not cleaned_path.exists():
            cleaned_artifact = session.exec(select(ProjectArtifact).where(
                ProjectArtifact.project_id == project_id,
                ProjectArtifact.artifact_type == ArtifactType.CLEANED_PAGE
            )).first()
            if not cleaned_artifact:
                raise RuntimeError(f"CLEANED_PAGE artifact not found for project {project_id}")

            storage = get_storage_service()
            with open(cleaned_path, "wb") as wf:
                with storage.get_artifact(cleaned_artifact.storage_key) as rf:
                    wf.write(rf.read())

    # 4. Run the typesetter and finalize the project (logic is now the same as retypeset)
    # Update project status to TYPESETTING
    with worker_session_scope() as session:
        project = session.get(Project, project_id)
        if project:
            project.status = ProjectStatus.TYPESETTING
            session.add(project)

    # The rest of this function now mirrors the logic from retypeset_after_edits' success path
    try:
        result = orchestrator_apply_edits(job_dir)
        # Check for typesetting errors and set completion warnings if needed
        if result.get("had_typesetting_errors", False):
            error_count = result.get("typesetting_error_count", 0)
            with worker_session_scope() as session:
                project = session.get(Project, project_id)
                if project:
                    if error_count == 1:
                        project.completion_warnings = "Text in 1 bubble could not fit properly. You can edit the text manually in the editor."
                    else:
                        project.completion_warnings = f"Text in {error_count} bubbles could not fit properly. You can edit the text manually in the editor."
                    session.add(project)
    except Exception as exc:
        # Mark project failed with a clear reason and stop
        with worker_session_scope() as session:
            project = session.get(Project, project_id)
            if project:
                project.status = ProjectStatus.FAILED
                project.failure_reason = f"Typesetting failed due to system error: {exc}"
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

        # Update editor_data with the final, typeset-aware data
        final_json_path_str = result.get("artifacts", {}).get("TEXT_JSON")
        if final_json_path_str:
            final_json_path = Path(final_json_path_str)
            if final_json_path.exists():
                with open(final_json_path, "r", encoding="utf-8") as f:
                    project.editor_data = json.load(f)

        project.status = ProjectStatus.COMPLETED
        session.add(project)

        # Upload bubble crops (best-effort) then package zip
        _upload_crops_for_project(project_id, job_dir, storage, session)
        _package_and_upload_zip(project_id, storage, session)


def retypeset_after_edits(project_id: str, revision: int, edited_bubble_ids: Optional[List[int]] = None) -> None:
    # This function's logic is now almost identical to the final part of process_initial_typeset
    # The key difference is it starts with data already in the DB from user edits.
    job_dir = get_job_dir(project_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    with worker_session_scope() as session:
        project = session.get(Project, project_id)
        if not project:
            logger.warning("project_missing", extra={"project_id": project_id})
            return
        if project.editor_data_rev > revision:
            logger.info("stale_edit_job", extra={"project_id": project_id, "job_rev": revision, "current_rev": project.editor_data_rev})
            return

        # Fetch cleaned image artifact
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

        # The API already updated editor_data. We just need to write it locally for the orchestrator.
        if not project.editor_data:
            raise RuntimeError("Editor data is missing for re-typesetting")

        json_path = job_dir / "text.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(project.editor_data, f, ensure_ascii=False, indent=2)

    # The rest of the finalization process is identical to process_initial_typeset
    try:
        result = orchestrator_apply_edits(job_dir, edited_bubble_ids=edited_bubble_ids)
        # Check for typesetting errors and set completion warnings if needed
        if result.get("had_typesetting_errors", False):
            error_count = result.get("typesetting_error_count", 0)
            with worker_session_scope() as session:
                project = session.get(Project, project_id)
                if project:
                    if error_count == 1:
                        project.completion_warnings = "Text in 1 bubble could not fit properly. You can edit the text manually in the editor."
                    else:
                        project.completion_warnings = f"Text in {error_count} bubbles could not fit properly. You can edit the text manually in the editor."
                    session.add(project)
    except Exception as exc:
        with worker_session_scope() as session:
            project = session.get(Project, project_id)
            if project:
                project.status = ProjectStatus.FAILED
                project.failure_reason = f"Typesetting failed due to system error: {exc}"
                # The orchestrator updated text.json with error details right before it failed.
                # We must read from it to preserve those details.
                json_path = job_dir / "text.json"
                if json_path.exists():
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            project.editor_data = json.load(f)
                    except Exception:
                        logger.exception("failed_to_persist_editor_data_on_error", extra={"project_id": project_id})

                session.add(project)

                # Package a zip so users can download partial results on failure.
                try:
                    storage = get_storage_service()
                    _package_and_upload_zip(project_id, storage, session)
                except Exception:
                    logger.exception("zip_package_failed_on_error", extra={"project_id": project_id})
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

        # Update editor_data with the final, re-typeset data
        final_json_path_str = result.get("artifacts", {}).get("TEXT_JSON")
        if final_json_path_str:
            final_json_path = Path(final_json_path_str)
            if final_json_path.exists():
                with open(final_json_path, "r", encoding="utf-8") as f:
                    project.editor_data = json.load(f)

        project.status = ProjectStatus.COMPLETED
        session.add(project)

        # Best-effort: package and upload downloadable zip after re-typeset
        _package_and_upload_zip(project_id, storage, session)