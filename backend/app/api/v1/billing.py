import logging
from datetime import datetime, timezone
from typing import Dict

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.deps import get_db_session
from app.db.models import Customer, Subscription
from app.api.v1.schemas import AuthenticatedUser
from app.db.deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/billing")

def get_or_create_stripe_customer(session: Session, user: AuthenticatedUser) -> str:
    """
    Get existing Stripe customer or create one on-demand for Clerk Direct ID approach.
    Returns the Stripe customer ID.
    """
    settings = get_settings()
    stripe.api_key = settings.stripe_secret_key

    # Check if customer already exists in our database
    existing_customer = session.exec(
        select(Customer).where(Customer.user_id == user.clerk_user_id)
    ).first()

    if existing_customer:
        return existing_customer.stripe_customer_id

    # Create Stripe customer on-demand
    try:
        customer_obj = stripe.Customer.create(
            email=user.email,
            name=user.clerk_user_id,
            metadata={"clerk_user_id": user.clerk_user_id}
        )

        # Store the mapping in our database
        local_customer = Customer(
            user_id=user.clerk_user_id,
            stripe_customer_id=customer_obj.id
        )
        session.add(local_customer)
        session.commit()

        logger.info(
            "stripe_customer_created",
            extra={"clerk_user_id": user.clerk_user_id, "stripe_customer_id": customer_obj.id}
        )

        return customer_obj.id
    except Exception as e:
        logger.error(
            "stripe_customer_creation_failed",
            extra={"clerk_user_id": user.clerk_user_id, "error": str(e)}
        )
        raise HTTPException(status_code=503, detail="Failed to create billing customer")

def sync_stripe_data(session: Session, stripe_customer_id: str):
    """
    Fetches all subscription data for a customer from Stripe and upserts it into the local database.
    This is our single source of truth for syncing.
    """
    settings = get_settings()
    stripe.api_key = settings.stripe_secret_key
    
    try:
        subscriptions = stripe.Subscription.list(customer=stripe_customer_id, status='all', limit=1)
    except Exception as e:
        logger.error("stripe_api_error_on_sync", extra={"customer_id": stripe_customer_id, "error": str(e)})
        raise HTTPException(status_code=503, detail="Could not sync with Stripe at this time.")

    customer = session.exec(select(Customer).where(Customer.stripe_customer_id == stripe_customer_id)).first()
    if not customer:
        logger.warning("sync_customer_not_found", extra={"stripe_customer_id": stripe_customer_id})
        return

    # If no subscriptions, ensure no active sub is in the DB
    if not subscriptions.data:
        existing_sub = session.exec(select(Subscription).where(Subscription.user_id == customer.user_id)).first()
        if existing_sub:
            session.delete(existing_sub)
        return

    # Assume one subscription per customer
    stripe_sub = subscriptions.data[0]
    
    plan_id = stripe_sub.items.data[0].price.id if stripe_sub.items and stripe_sub.items.data else ""

    existing_sub = session.exec(select(Subscription).where(Subscription.user_id == customer.user_id)).first()
    if not existing_sub:
        sub = Subscription(user_id=customer.user_id)
        session.add(sub)
    else:
        sub = existing_sub
    
    sub.stripe_subscription_id = stripe_sub.id
    sub.status = stripe_sub.status
    sub.plan_id = plan_id
    sub.current_period_end = datetime.fromtimestamp(stripe_sub.current_period_end, tz=timezone.utc)
    sub.cancel_at_period_end = stripe_sub.cancel_at_period_end
    
    session.commit()

@router.post("/stripe-webhook", include_in_schema=False)
async def stripe_webhook(request: Request, session: Session = Depends(get_db_session)):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    settings = get_settings()
    if not settings.stripe_webhook_secret or not sig_header:
        raise HTTPException(status_code=400, detail="Webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Extract customer ID and call our central sync function
    if hasattr(event.data, 'object') and 'customer' in event.data.object:
        customer_id = event.data.object['customer']
        if isinstance(customer_id, str):
            sync_stripe_data(session, customer_id)
            
    return {"status": "ok"}


@router.get("/customer", summary="Get or create Stripe customer for current user")
def get_customer(
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> Dict[str, str]:
    """
    Get existing Stripe customer or create one on-demand for Clerk Direct ID approach.
    This demonstrates on-demand customer creation when user interacts with billing features.
    """
    stripe_customer_id = get_or_create_stripe_customer(session, user)
    return {"stripe_customer_id": stripe_customer_id}