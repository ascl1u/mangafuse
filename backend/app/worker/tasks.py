from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from app.worker.celery_app import celery_app
from app.pipeline.io import ensure_dir
from app.core.paths import get_job_dir
from app.pipeline.orchestrator import run_pipeline as orchestrator_run_pipeline, apply_edits as orchestrator_apply_edits


logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.worker.tasks.process_page_task")
def process_page_task(
    self,
    image_path: str,
    *,
    depth: Literal["cleaned", "full"] = "cleaned",
    debug: bool = False,
    force: bool = False,
    seg_model_path: Optional[str] = None,
    font_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the MangaFuse pipeline for a single page according to depth.

    Delegates to the pipeline orchestrator. Writes artifacts under artifacts/jobs/{task_id}/.
    """
    task_id: str = str(getattr(self.request, "id", "unknown"))
    job_dir = get_job_dir(task_id)
    ensure_dir(job_dir)
    ensure_dir(job_dir / "masks")

    # Progress helper passed to orchestrator
    def _update(stage: str, progress: float) -> None:
        self.update_state(state="PROGRESS", meta={"stage": stage, "progress": progress})

    result = orchestrator_run_pipeline(
        job_id=task_id,
        image_path=image_path,
        depth=depth,
        debug=debug,
        force=force,
        seg_model_path=seg_model_path,
        font_path=font_path,
        progress_callback=_update,
    )
    # augment payload with urls including task_id
    result["task_id"] = task_id
    return result


@celery_app.task(bind=True, name="app.worker.tasks.apply_edits_task")
def apply_edits_task(original_task_id: str, edits: list[dict]) -> Dict[str, Any]:
    """Apply user edits to an existing job and re-typeset the final image."""
    result = orchestrator_apply_edits(original_task_id, edits)
    logger.info("task_completed", extra={"task": "apply_edits_task", "result": result})
    return result