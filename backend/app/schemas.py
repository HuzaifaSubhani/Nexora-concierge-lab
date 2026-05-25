from pydantic import BaseModel, Field
from typing import List, Optional, Literal


AllowedIntent = Literal[
    "towel_request",
    "food_order",
    "maintenance_request",
    "room_service",
    "housekeeping",
    "general_request",
]

AllowedDepartment = Literal[
    "housekeeping",
    "kitchen",
    "maintenance",
    "front_desk",
    "room_service",
]


class IntentExtractionRequest(BaseModel):
    text: str = Field(min_length=1)


class IntentExtractionResult(BaseModel):
    intent: AllowedIntent
    department: AllowedDepartment
    items: List[str] = Field(default_factory=list)
    quantity: Optional[int] = None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_confirmation: bool = True
    raw_text: str
    source: str


class TaskRouteRequest(BaseModel):
    text: str = Field(min_length=1)


class TaskRouteResult(BaseModel):
    intent: AllowedIntent
    department: AllowedDepartment
    queue: str
    priority: Literal["low", "normal", "high"] = "normal"
    confidence: float = Field(ge=0.0, le=1.0)
    needs_confirmation: bool = True
    should_create_task: bool = True
    raw_text: str
    source: str
    route_reason: str


class TaskCreateRequest(BaseModel):
    text: str = Field(min_length=1)


class TaskRecord(BaseModel):
    id: int
    text: str
    intent: AllowedIntent
    department: AllowedDepartment
    queue: str
    priority: Literal["low", "normal", "high"]
    status: Literal["queued", "in_progress", "done"] = "queued"
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str
    source: str
    created_at: str


class TranslationRequest(BaseModel):
    text: str = Field(min_length=1)
    src_lang: Optional[str] = None


class TranslationResult(BaseModel):
    translation: str
    translation_language: str = "en"
