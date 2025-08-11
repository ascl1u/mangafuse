from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
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


