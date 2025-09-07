import logging
from datetime import datetime, timezone
from typing import Dict, Any

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select, func

from app.core.config import get_settings
from app.db.deps import get_db_session
from app.db.models import Customer, Subscription, Project
from app.api.v1.schemas import AuthenticatedUser
from app.db.deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/billing")

def get_project_count_current_month(session: Session, user_id: str) -> int:
    """Counts projects created by a user in the current calendar month (UTC)."""
    # Get the first day of the current month in UTC
    start_of_month = func.date_trunc('month', func.now())

    statement = (
        select(func.count(Project.id))
        .where(Project.user_id == user_id)
        .where(Project.created_at >= start_of_month)
    )
    count = session.exec(statement).one()
    return int(count)


def get_stripe_client():
    """A FastAPI dependency that returns a configured Stripe client."""
    settings = get_settings()
    stripe.api_key = settings.stripe_secret_key
    stripe.api_version = settings.stripe_api_version
    return stripe

def get_or_create_stripe_customer(session: Session, user: AuthenticatedUser, stripe_client) -> str:
    """
    Get existing Stripe customer or create one on-demand for Clerk Direct ID approach.
    Returns the Stripe customer ID.
    """
    # Check if customer already exists in our database
    existing_customer = session.exec(
        select(Customer).where(Customer.user_id == user.clerk_user_id)
    ).first()

    if existing_customer:
        return existing_customer.stripe_customer_id

    # Create Stripe customer on-demand
    try:
        customer_obj = stripe_client.Customer.create(
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

def sync_stripe_data(session: Session, stripe_customer_id: str, stripe_client):
    """
    Fetches all subscription data for a customer from Stripe and upserts it into the local database.
    This is our single source of truth for syncing.
    """
    try:
        subscriptions = stripe_client.Subscription.list(customer=stripe_customer_id, status='all', limit=1)

        customer = session.exec(select(Customer).where(Customer.stripe_customer_id == stripe_customer_id)).first()
        if not customer:
            logger.warning("sync_customer_not_found", extra={"stripe_customer_id": stripe_customer_id})
            return

        if not subscriptions.data:
            existing_sub = session.exec(select(Subscription).where(Subscription.user_id == customer.user_id)).first()
            if existing_sub:
                session.delete(existing_sub)
            session.commit()
            return

        stripe_sub = subscriptions.data[0]

        if stripe_sub.get('status') not in ["active", "trialing", "past_due", "canceled"]:
            logger.info("skipping_sync_for_non_active_sub",
                        extra={"stripe_sub_id": stripe_sub.get('id'), "status": stripe_sub.get('status')})
            return

        items_data = stripe_sub.get('items', {}).get('data', [])
        plan_id = items_data[0].get('price', {}).get('id', '') if items_data else ""

        existing_sub = session.exec(select(Subscription).where(Subscription.user_id == customer.user_id)).first()

        sub = existing_sub
        if not sub:
            sub = Subscription(user_id=customer.user_id)
            session.add(sub)

        sub.stripe_subscription_id = stripe_sub.get('id')
        sub.status = stripe_sub.get('status')
        sub.plan_id = plan_id

        # Safe access to current_period_end
        period_end_ts = stripe_sub.get('current_period_end')
        if period_end_ts is None:
            logger.error("FATAL: current_period_end is None despite passing status checks.",
                        extra={"stripe_sub_id": stripe_sub.get('id'), "status": stripe_sub.get('status')})
            raise TypeError("'NoneType' object cannot be interpreted as an integer")

        sub.current_period_end = datetime.fromtimestamp(period_end_ts, tz=timezone.utc)
        sub.cancel_at_period_end = stripe_sub.get('cancel_at_period_end', False)

        session.commit()

    except Exception as e:
        logger.exception("subscription_sync_failed", extra={"stripe_customer_id": stripe_customer_id})
        raise HTTPException(status_code=503, detail=f"Failed to sync subscription data: {e}")

@router.post("/stripe-webhook", include_in_schema=False)
async def stripe_webhook(
    request: Request,
    session: Session = Depends(get_db_session),
    stripe_client = Depends(get_stripe_client)
):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    settings = get_settings()
    if not settings.stripe_webhook_secret or not sig_header:
        raise HTTPException(status_code=400, detail="Webhook secret not configured")

    try:
        event = stripe_client.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Define which events should trigger a subscription sync.
    # We only care about events that modify the subscription's state or items.
    subscription_events = {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
        "checkout.session.completed" # Sync after a successful checkout
    }

    if event.type in subscription_events:
        event_data = event.data.object
        customer_id = None

        # The customer ID is in different places depending on the event type
        if event.type == "checkout.session.completed":
            customer_id = event_data.get('customer')
        elif 'customer' in event_data:
            customer_id = event_data['customer']

        if customer_id and isinstance(customer_id, str):
            logger.info(f"Triggering subscription sync for customer {customer_id} from event {event.type}")
            sync_stripe_data(session, customer_id, stripe_client)
        else:
            logger.warning(f"Could not extract customer ID from {event.type} event.", extra={"event_id": event.id})
    else:
        # It's safe to ignore other events for subscription syncing purposes.
        logger.info(f"Ignoring event type '{event.type}' for subscription sync.")

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
    stripe_customer_id = get_or_create_stripe_customer(session, user, stripe_client)
    return {"stripe_customer_id": stripe_customer_id}


@router.post("/create-checkout-session")
def create_checkout_session(
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
    stripe_client = Depends(get_stripe_client)
):
    """Create a Stripe checkout session for subscription."""
    settings = get_settings()
    stripe_customer_id = get_or_create_stripe_customer(session, user, stripe_client)
    frontend_base_url = settings.public_frontend_base_url or 'http://localhost:5173'

    try:
        checkout_session = stripe_client.checkout.Session.create(
            customer=stripe_customer_id,
            payment_method_types=["card"],
            line_items=[{"price": settings.stripe_price_id_pro, "quantity": 1}],
            mode="subscription",
            success_url=f"{frontend_base_url}/checkout-success",
            cancel_url=f"{frontend_base_url}/pricing",
        )
        logger.info(
            "checkout_session_created",
            extra={
                "user_id": user.clerk_user_id,
                "session_id": checkout_session.id,
                "customer_id": stripe_customer_id
            }
        )
        return {"sessionId": checkout_session.id, "url": checkout_session.url}
    except Exception as e:
        logger.error(
            "checkout_session_creation_failed",
            extra={
                "user_id": user.clerk_user_id,
                "customer_id": stripe_customer_id,
                "error": str(e)
            }
        )
        raise HTTPException(status_code=500, detail="Failed to create checkout session")


@router.post("/create-portal-session")
def create_portal_session(
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
    stripe_client = Depends(get_stripe_client)
):
    """Create a Stripe customer portal session for subscription management."""
    settings = get_settings()
    stripe_customer_id = get_or_create_stripe_customer(session, user, stripe_client)
    frontend_base_url = settings.public_frontend_base_url or 'http://localhost:5173'

    try:
        portal_session = stripe_client.billing_portal.Session.create(
            customer=stripe_customer_id,
            return_url=f"{frontend_base_url}/projects",
        )
        logger.info(
            "portal_session_created",
            extra={
                "user_id": user.clerk_user_id,
                "customer_id": stripe_customer_id
            }
        )
        return {"url": portal_session.url}
    except Exception as e:
        logger.error(
            "portal_session_creation_failed",
            extra={
                "user_id": user.clerk_user_id,
                "customer_id": stripe_customer_id,
                "error": str(e)
            }
        )
        raise HTTPException(status_code=500, detail="Failed to create portal session")


@router.post("/sync-subscription")
def sync_user_subscription(
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
):
    """Endpoint for the frontend to trigger an eager sync after checkout."""
    customer = session.exec(select(Customer).where(Customer.user_id == user.clerk_user_id)).first()
    if not customer:
        logger.warning(
            "sync_customer_not_found",
            extra={"user_id": user.clerk_user_id}
        )
        raise HTTPException(status_code=404, detail="Billing customer not found")

    try:
        sync_stripe_data(session, customer.stripe_customer_id)
        logger.info(
            "subscription_sync_completed",
            extra={"user_id": user.clerk_user_id, "customer_id": customer.stripe_customer_id}
        )
        return {"status": "ok"}
    except Exception as e:
        logger.error(
            "subscription_sync_failed",
            extra={
                "user_id": user.clerk_user_id,
                "customer_id": customer.stripe_customer_id,
                "error": str(e)
            }
        )
        raise HTTPException(status_code=500, detail="Failed to sync subscription")


@router.get("/status", summary="Get current subscription and usage status")
def get_subscription_status(
    user: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """
    Fetches the user's subscription status and current monthly project usage.
    """
    settings = get_settings()
    subscription = session.exec(
        select(Subscription).where(Subscription.user_id == user.clerk_user_id)
    ).first()

    project_count = get_project_count_current_month(session, user.clerk_user_id)
    current_plan_id = subscription.plan_id if subscription and subscription.plan_id in settings.plan_limits else "free"

    plan_limit = settings.plan_limits.get(current_plan_id, 0)

    return {
        "plan_id": current_plan_id,
        "status": subscription.status if subscription else "active",
        "period_end": subscription.current_period_end if subscription else None,
        "project_count": project_count,
        "project_limit": plan_limit,
    }