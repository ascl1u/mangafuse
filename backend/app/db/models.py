from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlmodel import Field, SQLModel


class ProjectStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ArtifactType(str, Enum):
    SOURCE_RAW = "SOURCE_RAW"
    CLEANED_PAGE = "CLEANED_PAGE"
    FINAL_PNG = "FINAL_PNG"
    TEXT_LAYER_PNG = "TEXT_LAYER_PNG"


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=sa.Column(PGUUID(as_uuid=True), primary_key=True, nullable=False),
    )
    clerk_user_id: str = Field(sa_column=sa.Column(sa.String(length=191), unique=True, index=True, nullable=False))
    email: str = Field(sa_column=sa.Column(sa.String(length=320), unique=True, index=True, nullable=False))
    deactivated_at: Optional[datetime] = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True),
    )
    created_at: datetime = Field(
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        )
    )
    updated_at: datetime = Field(
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        )
    )


class Project(SQLModel, table=True):
    __tablename__ = "projects"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=sa.Column(PGUUID(as_uuid=True), primary_key=True, nullable=False),
    )
    user_id: uuid.UUID = Field(sa_column=sa.Column(PGUUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False))
    title: str = Field(sa_column=sa.Column(sa.String(length=255), nullable=False))
    status: ProjectStatus = Field(
        sa_column=sa.Column(sa.Enum(ProjectStatus, name="project_status_enum"), nullable=False, index=True),
        default=ProjectStatus.PENDING,
    )
    editor_data: Optional[dict] = Field(default=None, sa_column=sa.Column(JSONB, nullable=True))
    celery_task_id: Optional[str] = Field(default=None, sa_column=sa.Column(sa.String(length=191), nullable=True))
    failure_reason: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text, nullable=True))
    created_at: datetime = Field(
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        )
    )
    updated_at: datetime = Field(
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        )
    )


class ProjectArtifact(SQLModel, table=True):
    __tablename__ = "project_artifacts"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=sa.Column(PGUUID(as_uuid=True), primary_key=True, nullable=False),
    )
    project_id: uuid.UUID = Field(sa_column=sa.Column(PGUUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True))
    artifact_type: ArtifactType = Field(sa_column=sa.Column(sa.Enum(ArtifactType, name="artifact_type_enum"), nullable=False, index=True))
    storage_key: str = Field(sa_column=sa.Column(sa.String(length=512), nullable=False))
    created_at: datetime = Field(
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        )
    )


__all__ = [
    "User",
    "Project",
    "ProjectArtifact",
    "ProjectStatus",
    "ArtifactType",
]
