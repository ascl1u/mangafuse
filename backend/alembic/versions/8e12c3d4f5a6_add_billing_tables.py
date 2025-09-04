"""add billing tables

Revision ID: 8e12c3d4f5a6
Revises: 763bbff280a1
Create Date: 2025-09-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8e12c3d4f5a6'
down_revision: Union[str, Sequence[str], None] = '763bbff280a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # customers table: one-to-one mapping from users to Stripe customer id
    op.create_table(
        'customers',
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('stripe_customer_id', sa.String(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('user_id'),
    )
    op.create_index('ix_customers_stripe_customer_id', 'customers', ['stripe_customer_id'], unique=True)

    # subscriptions table: local cache of subscription state
    op.create_table(
        'subscriptions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('stripe_subscription_id', sa.String(), nullable=False),
        sa.Column('plan_id', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('current_period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('cancel_at_period_end', sa.Boolean(), server_default=sa.text('false'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_subscriptions_user_id', 'subscriptions', ['user_id'], unique=True)
    op.create_index('ix_subscriptions_stripe_subscription_id', 'subscriptions', ['stripe_subscription_id'], unique=True)

    # support monthly usage query
    try:
        op.create_index('ix_projects_created_at', 'projects', ['created_at'], unique=False)
    except Exception:
        # Index might already exist
        pass


def downgrade() -> None:
    try:
        op.drop_index('ix_projects_created_at', table_name='projects')
    except Exception:
        pass
    op.drop_index('ix_subscriptions_stripe_subscription_id', table_name='subscriptions')
    op.drop_index('ix_subscriptions_user_id', table_name='subscriptions')
    op.drop_table('subscriptions')
    op.drop_index('ix_customers_stripe_customer_id', table_name='customers')
    op.drop_table('customers')


