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
    def get_planned_upload_storage_key(self, project_id: str, filename: str) -> str:
        """Return the storage key that will be used for the client direct upload."""
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

    # Optional: presigned upload for pipeline outputs (cloud-only)
    def get_output_upload_url(self, project_id: str, artifact_name: str) -> tuple[str, str] | None:  # pragma: no cover - default not implemented
        """Return (storage_key, presigned_put_url) for uploading a pipeline output artifact.

        Local storage may return None to indicate unsupported; callers should handle this by uploading via the backend.
        """
        return None


class LocalStorageService(StorageService):
    """Storage service for local development, using the filesystem."""

    def get_upload_url(self, project_id: str, filename: str) -> str:
        # For local dev, the "upload URL" is just a server endpoint.
        # The actual file saving will be handled by the endpoint logic.
        return f"/api/v1/projects/{project_id}/upload"

    def get_planned_upload_storage_key(self, project_id: str, filename: str) -> str:
        # For local dev, mirror the path produced by save_artifact for predictability
        return f"jobs/{project_id}/{filename}"

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
        # Lazy import so local dev doesn't require boto3
        import boto3
        from botocore.config import Config

        settings = get_settings()
        if not (settings.r2_endpoint_url and settings.r2_bucket_name and settings.r2_access_key_id and settings.r2_secret_access_key):
            raise RuntimeError("R2 configuration is incomplete. Ensure R2_ACCOUNT_ID (or R2_S3_ENDPOINT), R2_BUCKET_NAME, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY are set.")

        self._bucket = settings.r2_bucket_name
        self._presign_expiration = settings.r2_presign_expiration_seconds
        # Use path-style addressing for R2 default endpoint
        self._s3 = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint_url,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def get_upload_url(self, project_id: str, filename: str) -> str:
        key = self.get_planned_upload_storage_key(project_id, filename)
        return self._s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=self._presign_expiration,
        )

    def get_planned_upload_storage_key(self, project_id: str, filename: str) -> str:
        # Separate upload namespace from generated artifacts
        return f"uploads/{project_id}/{filename}"

    def get_download_url(self, storage_key: str) -> str:
        return self._s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self._bucket, "Key": storage_key},
            ExpiresIn=self._presign_expiration,
        )

    def save_artifact(self, project_id: str, artifact_name: str, data: IO[bytes]) -> str:
        import mimetypes
        key = f"jobs/{project_id}/{artifact_name}"
        # Determine content type; fall back to octet-stream
        content_type, _ = mimetypes.guess_type(artifact_name)
        body = data.read() if hasattr(data, "read") else data
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,  # type: ignore[arg-type]
            ContentType=content_type or "application/octet-stream",
        )
        return key

    def get_artifact(self, storage_key: str) -> IO[bytes]:
        import io
        obj = self._s3.get_object(Bucket=self._bucket, Key=storage_key)
        # Read fully to return a simple in-memory file-like object
        payload: bytes = obj["Body"].read()
        return io.BytesIO(payload)

    def get_output_upload_url(self, project_id: str, artifact_name: str) -> tuple[str, str]:
        key = f"jobs/{project_id}/{artifact_name}"
        url = self._s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=self._presign_expiration,
        )
        return key, url


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
