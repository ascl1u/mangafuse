from __future__ import annotations

import logging

from celery import Celery

from app.core.config import get_settings
from app.core.logging import configure_logging


# Initialize settings and logging so Celery worker logs are structured JSON
settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


# Warn clearly if broker/backend are not configured to avoid Celery's AMQP default
if not settings.effective_broker_url or not settings.effective_result_backend:
    logger.warning(
        "celery_config_missing",
        extra={
            "broker": settings.effective_broker_url,
            "backend": settings.effective_result_backend,
        },
    )


celery_app = Celery(
    "mangafuse",
    broker=settings.effective_broker_url,
    backend=settings.effective_result_backend,
    include=["app.worker.tasks"],
)

# Conservative, JSON-only configuration suitable for development
celery_app.conf.update(
    task_time_limit=settings.celery_task_time_limit,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

__all__ = ["celery_app"]


