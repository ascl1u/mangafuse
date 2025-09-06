"""init db

Revision ID: ff6586f9d6f0
Revises: 
Create Date: 2025-08-17 11:52:44.731533

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'ff6586f9d6f0'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### Clerk Direct ID approach - no users table, clerk_user_id used directly ###

    # Define the ENUM types without creating them. SQLAlchemy will handle this.
    project_status_enum = sa.Enum(
        'PENDING', 'PROCESSING', 'TRANSLATING', 'TYPESETTING', 'UPDATING', 'COMPLETED', 'FAILED',
        name='project_status_enum'
    )
    artifact_type_enum = sa.Enum(
        'SOURCE_RAW', 'CLEANED_PAGE', 'FINAL_PNG', 'TEXT_LAYER_PNG', 'DOWNLOADABLE_ZIP',
        name='artifact_type_enum'
    )

    # Projects table - user_id is now clerk_user_id (string)
    op.create_table('projects',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.String(length=191), nullable=False),  # Clerk user ID
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('status', project_status_enum, nullable=False),
        sa.Column('editor_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('editor_data_rev', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_projects_status'), 'projects', ['status'], unique=False)
    op.create_index(op.f('ix_projects_user_id'), 'projects', ['user_id'], unique=False)
    op.create_index(op.f('ix_projects_created_at'), 'projects', ['created_at'], unique=False)

    # Project artifacts table
    op.create_table('project_artifacts',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('project_id', sa.UUID(), nullable=False),
        sa.Column('artifact_type', artifact_type_enum, nullable=False),
        sa.Column('storage_key', sa.String(length=512), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_project_artifacts_artifact_type'), 'project_artifacts', ['artifact_type'], unique=False)
    op.create_index(op.f('ix_project_artifacts_project_id'), 'project_artifacts', ['project_id'], unique=False)

    # Customers table - user_id is now clerk_user_id (string, primary key for one-to-one)
    op.create_table('customers',
        sa.Column('user_id', sa.String(length=191), nullable=False),  # Clerk user ID as primary key
        sa.Column('stripe_customer_id', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('user_id')
    )
    op.create_index('ix_customers_stripe_customer_id', 'customers', ['stripe_customer_id'], unique=True)

    # Subscriptions table - user_id is now clerk_user_id (string)
    op.create_table('subscriptions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.String(length=191), nullable=False),  # Clerk user ID
        sa.Column('stripe_subscription_id', sa.String(), nullable=False),
        sa.Column('plan_id', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('cancel_at_period_end', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_subscriptions_user_id', 'subscriptions', ['user_id'], unique=True)
    op.create_index('ix_subscriptions_stripe_subscription_id', 'subscriptions', ['stripe_subscription_id'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_subscriptions_stripe_subscription_id', table_name='subscriptions')
    op.drop_index('ix_subscriptions_user_id', table_name='subscriptions')
    op.drop_table('subscriptions')
    op.drop_index('ix_customers_stripe_customer_id', table_name='customers')
    op.drop_table('customers')
    op.drop_index(op.f('ix_project_artifacts_project_id'), table_name='project_artifacts')
    op.drop_index(op.f('ix_project_artifacts_artifact_type'), table_name='project_artifacts')
    op.drop_table('project_artifacts')
    op.drop_index(op.f('ix_projects_created_at'), table_name='projects')
    op.drop_index(op.f('ix_projects_user_id'), table_name='projects')
    op.drop_index(op.f('ix_projects_status'), table_name='projects')
    op.drop_table('projects')

    # Drop enums - this is the correct place for explicit drop commands
    op.execute("DROP TYPE IF EXISTS project_status_enum")
    op.execute("DROP TYPE IF EXISTS artifact_type_enum")
