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
    TRANSLATING = "TRANSLATING"
    TYPESETTING = "TYPESETTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProcessingMode(str, Enum):
    FULL = "FULL"
    CLEANED = "CLEANED"


class ArtifactType(str, Enum):
    SOURCE_RAW = "SOURCE_RAW"
    CLEANED_PAGE = "CLEANED_PAGE"
    FINAL_PNG = "FINAL_PNG"
    TEXT_LAYER_PNG = "TEXT_LAYER_PNG"
    DOWNLOADABLE_ZIP = "DOWNLOADABLE_ZIP"


# User model removed - using Clerk Direct ID approach
# clerk_user_id is used directly as user identifier throughout the application


class Project(SQLModel, table=True):
    __tablename__ = "projects"

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=sa.Column(PGUUID(as_uuid=True), primary_key=True, nullable=False),
    )
    user_id: str = Field(sa_column=sa.Column(sa.String(length=191), index=True, nullable=False))  # Clerk user ID
    title: str = Field(sa_column=sa.Column(sa.String(length=255), nullable=False))
    status: ProjectStatus = Field(
        sa_column=sa.Column(sa.Enum(ProjectStatus, name="project_status_enum"), nullable=False, index=True),
        default=ProjectStatus.PENDING,
    )
    processing_mode: ProcessingMode = Field(
        sa_column=sa.Column(sa.Enum(ProcessingMode, name="processing_mode_enum"), nullable=False),
        default=ProcessingMode.FULL,
    )
    editor_data: Optional[dict] = Field(default=None, sa_column=sa.Column(JSONB, nullable=True))
    editor_data_rev: int = Field(default=0, sa_column=sa.Column(sa.Integer, nullable=False, server_default="0"))
    failure_reason: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text, nullable=True))
    completion_warnings: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text, nullable=True))
    created_at: datetime = Field(
        sa_column=sa.Column(
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            index=True,
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


class Customer(SQLModel, table=True):
    __tablename__ = "customers"

    # One-to-one with Clerk users; user_id (clerk_user_id) is the primary key to enforce uniqueness
    user_id: str = Field(sa_column=sa.Column(sa.String(length=191), primary_key=True))  # Clerk user ID
    stripe_customer_id: str = Field(sa_column=sa.Column(sa.String, unique=True, index=True, nullable=False))


class Subscription(SQLModel, table=True):
    __tablename__ = "subscriptions"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, sa_column=sa.Column(PGUUID(as_uuid=True), primary_key=True, nullable=False))
    user_id: str = Field(sa_column=sa.Column(sa.String(length=191), unique=True, index=True, nullable=False))  # Clerk user ID
    stripe_subscription_id: str = Field(sa_column=sa.Column(sa.String, unique=True, index=True, nullable=False))
    plan_id: str = Field(sa_column=sa.Column(sa.String, nullable=False))  # Stripe Price ID for paid plans; "free" for free tier
    status: str = Field(sa_column=sa.Column(sa.String, nullable=False))  # Stripe subscription status
    current_period_end: datetime = Field(sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False))
    cancel_at_period_end: bool = Field(default=False, sa_column=sa.Column(sa.Boolean, nullable=False, server_default=sa.text("false")))


__all__ = [
    "Project",
    "ProjectArtifact",
    "ProjectStatus",
    "ProcessingMode",
    "ArtifactType",
    "Customer",
    "Subscription",
]
