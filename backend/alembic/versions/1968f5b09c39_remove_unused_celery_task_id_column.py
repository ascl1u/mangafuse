"""remove unused celery_task_id column

Revision ID: 1968f5b09c39
Revises: a1b2c3d4e5f6
Create Date: 2025-08-26 15:05:37.729057

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1968f5b09c39'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_column("celery_task_id")


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('projects') as batch_op:
        batch_op.add_column(sa.Column('celery_task_id', sa.String(length=191), nullable=True))
