"""remove UPDATING enum from ProjectStatus, add processing_mode to projects

Revision ID: 551deb4ca9b8
Revises: dd3d99ca1131
Create Date: 2025-09-08 10:15:21.462721

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '551deb4ca9b8'
down_revision: Union[str, Sequence[str], None] = 'dd3d99ca1131'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### Manually adjusted Alembic commands ###

    # 1. Define the new ENUM type and create it in the database.
    #    This must be done before creating a column that uses it.
    processing_mode_enum = sa.Enum("FULL", "CLEANED", name="processing_mode_enum")
    processing_mode_enum.create(op.get_bind())

    # 2. Add the new column, referencing the created ENUM type.
    #    A server_default is necessary because the column is not nullable,
    #    ensuring existing rows are populated correctly.
    op.add_column(
        "projects",
        sa.Column(
            "processing_mode",
            processing_mode_enum,
            nullable=False,
            server_default="FULL",
        ),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### Manually adjusted Alembic commands ###

    # 1. Drop the column first.
    op.drop_column("projects", "processing_mode")

    # 2. After the column is gone, drop the ENUM type from the database.
    processing_mode_enum = sa.Enum("FULL", "CLEANED", name="processing_mode_enum")
    processing_mode_enum.drop(op.get_bind())
    # ### end Alembic commands ###