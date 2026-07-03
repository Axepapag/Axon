"""Typed JSON message protocol for the Axon collaboration bus."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Board item models
# ---------------------------------------------------------------------------
class Objective(BaseModel):
    id: str
    text: str
    priority: Literal["low", "normal", "high"] = "normal"
    owner: str = ""
    created_at: str


class Task(BaseModel):
    id: str
    text: str
    status: Literal["open", "in_progress", "blocked", "done"] = "open"
    owner: str = ""
    created_at: str


class TalkingPoint(BaseModel):
    id: str
    text: str
    source: str = ""
    created_at: str


class Question(BaseModel):
    id: str
    text: str
    status: Literal["open", "answered", "stale"] = "open"
    asked_by: str = ""
    created_at: str


class DirectMessage(BaseModel):
    msg_id: str
    from_agent: str
    to_agent: str
    content: str
    in_reply_to: str | None = None
    timestamp: str
    delivered: bool = False


BOARD_FIELDS = {"objectives", "tasks", "talking_points", "questions", "direct_messages"}


class BoardState(BaseModel):
    version: int = 1
    updated_at: str = Field(default_factory=utc_now)
    objectives: list[Objective] = []
    tasks: list[Task] = []
    talking_points: list[TalkingPoint] = []
    questions: list[Question] = []
    direct_messages: list[DirectMessage] = []
    participants: list[dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Message envelope
# ---------------------------------------------------------------------------
MessageType = Literal[
    "hello",
    "hello_ack",
    "board_state",
    "status",
    "board_update",
    "broadcast",
    "direct_message",
    "dm_ack",
    "ping",
    "pong",
    "error",
]


class MessageEnvelope(BaseModel):
    msg_id: str = Field(default_factory=lambda: uuid4().hex)
    type: MessageType
    from_agent: str = ""
    to_agent: str | None = None
    topic: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Typed payloads
# ---------------------------------------------------------------------------
class HelloPayload(BaseModel):
    agent_id: str
    capabilities: list[str] = Field(default_factory=list)
    last_event_id: str | None = None
    last_dm_id: str | None = None


class StatusPayload(BaseModel):
    status: str = "idle"
    current_task: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class BoardUpdatePayload(BaseModel):
    field: Literal["objectives", "tasks", "talking_points", "questions"]
    value: dict[str, Any]


class BroadcastPayload(BaseModel):
    topic: str
    payload: dict[str, Any] = Field(default_factory=dict)


class DirectMessagePayload(BaseModel):
    to_agent: str
    content: str
    in_reply_to: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BOARD_ITEM_MODELS = {
    "objectives": Objective,
    "tasks": Task,
    "talking_points": TalkingPoint,
    "questions": Question,
}


def topic_matches(topic: str, subscription: str) -> bool:
    """Salvaged topic matcher from Dream Team event_bus.py."""
    if subscription == "*":
        return True
    if subscription.endswith(".*"):
        return topic.startswith(subscription[:-1])
    return topic == subscription
