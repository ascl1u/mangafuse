"""add typesetting enum for project status

Revision ID: 763bbff280a1
Revises: d7b8d350a223
Create Date: 2025-09-01 15:55:39.884774

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '763bbff280a1'
down_revision: Union[str, Sequence[str], None] = 'd7b8d350a223'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.execute("ALTER TYPE project_status_enum ADD VALUE IF NOT EXISTS 'TYPESETTING'")

def downgrade() -> None:
    pass # No downgrade needed
