from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class EditRecord(BaseModel):
    id: int
    en_text: Optional[str] = None
    font_size: Optional[int] = None


class ApplyEditsRequest(BaseModel):
    edits: List[EditRecord]



