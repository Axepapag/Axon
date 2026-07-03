"""Schema validation tests for the Axon collaboration bus."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bus.schema import (
    BoardState,
    BoardUpdatePayload,
    DirectMessagePayload,
    HelloPayload,
    MessageEnvelope,
    Objective,
    Task,
    topic_matches,
)


def test_objective_defaults() -> None:
    obj = Objective(id="o1", text="build bus", created_at="2026-06-30T00:00:00Z")
    assert obj.priority == "normal"
    assert obj.owner == ""


def test_objective_invalid_priority() -> None:
    with pytest.raises(ValidationError):
        Objective(id="o1", text="build bus", priority="urgent", created_at="2026-06-30T00:00:00Z")


def test_task_invalid_status() -> None:
    with pytest.raises(ValidationError):
        Task(id="t1", text="wire ws", status="pending", created_at="2026-06-30T00:00:00Z")


def test_hello_payload() -> None:
    payload = HelloPayload(agent_id="claude", capabilities=["read", "dm"])
    assert payload.agent_id == "claude"
    assert "read" in payload.capabilities


def test_board_update_payload_valid_field() -> None:
    p = BoardUpdatePayload(field="tasks", value={"id": "t1", "text": "x", "created_at": "now"})
    assert p.field == "tasks"


def test_board_update_payload_invalid_field() -> None:
    with pytest.raises(ValidationError):
        BoardUpdatePayload(field="direct_messages", value={"id": "m1"})


def test_direct_message_payload() -> None:
    p = DirectMessagePayload(to_agent="kimi", content="hello")
    assert p.to_agent == "kimi"
    assert p.content == "hello"


def test_message_envelope_factory_id() -> None:
    e1 = MessageEnvelope(type="ping")
    e2 = MessageEnvelope(type="ping")
    assert e1.msg_id != e2.msg_id
    assert len(e1.msg_id) == 32


def test_board_state_with_items() -> None:
    state = BoardState(
        updated_at="2026-06-30T00:00:00Z",
        tasks=[Task(id="t1", text="wire ws", created_at="2026-06-30T00:00:00Z")],
    )
    assert len(state.tasks) == 1


def test_topic_matches() -> None:
    assert topic_matches("status", "*")
    assert topic_matches("task.foo", "task.*")
    assert topic_matches("task.foo", "task.foo")
    assert not topic_matches("task.foo", "status")
    assert topic_matches("task.foo.bar", "task.*")
