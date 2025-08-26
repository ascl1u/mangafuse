"""add statuses and editor_data_rev

Revision ID: a1b2c3d4e5f6
Revises: ff6586f9d6f0
Create Date: 2025-09-01 00:01:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'ff6586f9d6f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new enum values TRANSLATING and UPDATING to project_status_enum (idempotent)
    op.execute("ALTER TYPE project_status_enum ADD VALUE IF NOT EXISTS 'TRANSLATING'")
    op.execute("ALTER TYPE project_status_enum ADD VALUE IF NOT EXISTS 'UPDATING'")

    # Add editor_data_rev column with default 0 (idempotent-friendly for fresh db too)
    with op.batch_alter_table('projects') as batch_op:
        batch_op.add_column(sa.Column('editor_data_rev', sa.Integer(), nullable=False, server_default='0'))


def downgrade() -> None:
    # Removing enum values is non-trivial; we leave them in place.
    with op.batch_alter_table('projects') as batch_op:
        try:
            batch_op.drop_column('editor_data_rev')
        except Exception:
            # Column might not exist in some environments
            pass


