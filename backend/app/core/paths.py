from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_repo_root() -> Path:
    """Best-effort repository root detection.

    Walk up from this file and return the first directory that looks like the
    project root. Falls back to a static relative parent for local dev.
    """
    here = Path(__file__).resolve()
    # Try a bounded walk upward
    for p in [here] + list(here.parents)[:8]:
        if (p / "backend").exists() and ((p / "frontend").exists()):
            return p
    # Fallback: backend/app/core/paths.py -> repo root is parents[3]
    try:
        return here.parents[3]
    except Exception:
        return here.parent


@lru_cache(maxsize=1)
def get_artifacts_root() -> Path:
    path = os.getenv("ARTIFACTS_ROOT")
    return Path(path) if path else (get_repo_root() / "artifacts")


@lru_cache(maxsize=1)
def get_assets_root() -> Path:
    path = os.getenv("ASSETS_ROOT")
    return Path(path) if path else (get_repo_root() / "assets")


def get_job_dir(task_id: str) -> Path:
    return get_artifacts_root() / "jobs" / task_id


@lru_cache(maxsize=1)
def get_uploads_dir() -> Path:
    return get_artifacts_root() / "uploads"



