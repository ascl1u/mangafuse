from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import redis
from rq import Queue

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _get_redis_connection() -> redis.Redis:
    settings = get_settings()
    url: Optional[str] = settings.redis_url or os.getenv("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL is not configured")
    return redis.from_url(url)


def get_queue(name: str) -> Queue:
    conn = _get_redis_connection()
    settings = get_settings()
    timeout = settings.rq_job_timeout_seconds
    return Queue(name=name, connection=conn, default_timeout=timeout)


def get_high_priority_queue() -> Queue:
    settings = get_settings()
    return get_queue(settings.rq_queue_high)


def get_default_queue() -> Queue:
    settings = get_settings()
    return get_queue(settings.rq_queue_default)


