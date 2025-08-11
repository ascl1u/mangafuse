from __future__ import annotations

import logging
import time
from typing import Any, Dict

from app.worker.celery_app import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.worker.tasks.demo_task")
def demo_task(self, duration_s: int = 5) -> Dict[str, Any]:
    """A simple demo task that sleeps then returns a payload.

    Parameters
    ----------
    duration_s: int
        Number of seconds to sleep before completing.
    """
    duration = max(0, int(duration_s or 0))
    logger.info("task_started", extra={"task": "demo_task", "duration_s": duration})
    time.sleep(duration)
    result: Dict[str, Any] = {"status": "completed", "slept_seconds": duration}
    logger.info("task_completed", extra={"task": "demo_task", "result": result})
    return result


__all__ = ["demo_task"]


