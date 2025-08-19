from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import IO, Literal

from app.core.config import get_settings
from app.core.paths import get_job_dir, get_artifacts_root


class StorageService(ABC):
    """Abstract interface for a storage service."""

    @abstractmethod
    def get_upload_url(self, project_id: str, filename: str) -> str:
        """Generate a URL/path for the client to upload a source file."""
        pass

    @abstractmethod
    def get_download_url(self, storage_key: str) -> str:
        """Generate a URL for the client to download a file."""
        pass

    @abstractmethod
    def save_artifact(self, project_id: str, artifact_name: str, data: IO[bytes]) -> str:
        """Save a pipeline artifact and return its storage key."""
        pass

    @abstractmethod
    def get_artifact(self, storage_key: str) -> IO[bytes]:
        """Retrieve an artifact as a file-like object."""
        pass


class LocalStorageService(StorageService):
    """Storage service for local development, using the filesystem."""

    def get_upload_url(self, project_id: str, filename: str) -> str:
        # For local dev, the "upload URL" is just a server endpoint.
        # The actual file saving will be handled by the endpoint logic.
        return f"/api/v1/projects/{project_id}/upload"

    def get_download_url(self, storage_key: str) -> str:
        # Expose under FastAPI StaticFiles mount at /artifacts
        return f"/artifacts/{storage_key}"

    def save_artifact(self, project_id: str, artifact_name: str, data: IO[bytes] | bytes) -> str:
        job_dir = get_job_dir(project_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = job_dir / artifact_name
        with open(artifact_path, "wb") as f:
            if hasattr(data, "read"):
                f.write(data.read())  # type: ignore[arg-type]
            else:
                # bytes-like passed directly
                f.write(data)  # type: ignore[arg-type]
        # The storage key for local is the path relative to the artifacts root.
        artifacts_root = get_artifacts_root()
        return str(artifact_path.relative_to(artifacts_root)).replace("\\", "/")

    def get_artifact(self, storage_key: str) -> IO[bytes]:
        # In local storage, the storage_key is the path relative to artifacts dir.
        path = get_artifacts_root() / storage_key
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {path}")
        return open(path, "rb")


class CloudStorageService(StorageService):
    """Storage service for production, using a cloud provider (e.g., R2)."""

    def __init__(self):
        raise NotImplementedError("CloudStorageService is not yet implemented.")

    def get_upload_url(self, project_id: str, filename: str) -> str:
        raise NotImplementedError

    def get_download_url(self, storage_key: str) -> str:
        raise NotImplementedError

    def save_artifact(self, project_id: str, artifact_name: str, data: IO[bytes]) -> str:
        raise NotImplementedError

    def get_artifact(self, storage_key: str) -> IO[bytes]:
        raise NotImplementedError


@lru_cache(maxsize=1)
def get_storage_service() -> StorageService:
    """
    Factory function to get the appropriate storage service based on the environment.
    This is used as a FastAPI dependency.
    """
    settings = get_settings()
    if settings.app_env == "production":
        return CloudStorageService()
    return LocalStorageService()
