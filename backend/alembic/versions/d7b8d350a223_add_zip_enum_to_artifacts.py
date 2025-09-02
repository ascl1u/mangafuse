"""add zip enum to artifacts

Revision ID: d7b8d350a223
Revises: 1968f5b09c39
Create Date: 2025-09-01 14:40:11.725363

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd7b8d350a223'
down_revision: Union[str, Sequence[str], None] = '1968f5b09c39'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TYPE artifact_type_enum ADD VALUE IF NOT EXISTS 'DOWNLOADABLE_ZIP'")

def downgrade() -> None:
    """Downgrade schema."""
    # Downgrading by removing an enum value is complex and often unnecessary.
    # It's safe to leave the value in the type.
    pass
