from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from dotenv import load_dotenv


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Uses a local .env file in development for convenience.
    """

    app_env: Literal["development", "production"] = "development"
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"

    # Redis and future Celery configuration
    redis_url: Optional[str] = None
    celery_broker_url: Optional[str] = None
    celery_result_backend: Optional[str] = None
    celery_task_time_limit: int = 120
    google_api_key: Optional[str] = None

    # Database configuration
    database_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL"),
        description="SQLAlchemy URL to the primary database (e.g., postgresql+psycopg://user:pass@host:5432/db)",
    )

    # Clerk (Auth) configuration
    clerk_issuer: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CLERK_ISSUER"),
        description="Expected Clerk JWT issuer, e.g. https://your-app.clerk.accounts.dev",
    )
    clerk_jwks_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CLERK_JWKS_URL"),
        description="Override JWKS URL. If not provided, derived as {issuer}/.well-known/jwks.json",
    )
    clerk_webhook_secret: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CLERK_WEBHOOK_SECRET"),
        description="Secret used to verify Clerk webhooks (svix)",
    )
    clerk_secret_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CLERK_SECRET_KEY"),
        description="Clerk secret key for backend API calls",
    )
    authorized_parties: list[str] = Field(
        default=[],
        description="A list of authorized parties for JWT audience verification.",
    )

    # Cloudflare R2 (S3-compatible) configuration
    r2_account_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("R2_ACCOUNT_ID"),
        description="Cloudflare account ID for R2 endpoint construction",
    )
    r2_bucket_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("R2_BUCKET_NAME"),
        description="Cloudflare R2 bucket name",
    )
    r2_access_key_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("R2_ACCESS_KEY_ID"),
        description="Access key ID for R2 S3 API",
    )
    r2_secret_access_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("R2_SECRET_ACCESS_KEY"),
        description="Secret access key for R2 S3 API",
    )
    r2_access_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("R2_ACCESS_TOKEN"),
        description="Optional Cloudflare API token (not required for S3 presign)",
    )
    r2_presign_expiration_seconds: int = Field(
        default=3600,
        description="Expiration (seconds) for presigned URLs",
    )
    r2_endpoint_override: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("R2_S3_ENDPOINT", "R2_ENDPOINT_URL"),
        description="Override the default constructed R2 S3 endpoint URL",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def effective_broker_url(self) -> Optional[str]:
        return self.celery_broker_url or self.redis_url

    @property
    def effective_result_backend(self) -> Optional[str]:
        return self.celery_result_backend or self.redis_url

    @property
    def r2_endpoint_url(self) -> Optional[str]:
        if self.r2_endpoint_override:
            return self.r2_endpoint_override
        if not self.r2_account_id:
            return None
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Lazily load and cache settings to avoid re-parsing .env on each import."""
    # Ensure environment variables are loaded in both repo root and backend contexts
    # 1) Load backend/.env if present (works when running from repo root)
    backend_dir = Path(__file__).resolve().parents[2]
    backend_env_path = backend_dir / ".env"
    if backend_env_path.exists():
        load_dotenv(backend_env_path, override=False)

    # 2) Load nearest .env discovered from CWD upward without overriding existing vars
    load_dotenv(override=False)

    return Settings()
