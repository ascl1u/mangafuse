from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class EditRecord(BaseModel):
    id: int
    en_text: Optional[str] = None
    font_size: Optional[int] = None


class ApplyEditsRequest(BaseModel):
    edits: List[EditRecord]


class AuthenticatedUser(BaseModel):
    clerk_user_id: str
    email: Optional[str] = None


class ClerkWebhookUser(BaseModel):
    id: str
    email_addresses: List[dict]
    primary_email_address_id: Optional[str] = None
    deleted: Optional[bool] = None


class ClerkWebhookEvent(BaseModel):
    type: str
    data: ClerkWebhookUser



