"""State persistence tests for the Axon collaboration bus."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bus.log import read_jsonl
from bus.schema import BoardState, DirectMessage, Objective, Task
from bus.state import BoardStateStore


@pytest.fixture
def tmp_store(tmp_path: Path) -> BoardStateStore:
    return BoardStateStore(tmp_path)


def test_load_empty(tmp_store: BoardStateStore) -> None:
    state = tmp_store.get()
    assert state.version == 1
    assert state.objectives == []


def test_patch_adds_item(tmp_store: BoardStateStore) -> None:
    state = tmp_store.patch("tasks", {"id": "t1", "text": "wire ws", "created_at": "now"})
    assert len(state.tasks) == 1
    assert state.tasks[0].text == "wire ws"


def test_patch_replaces_by_id(tmp_store: BoardStateStore) -> None:
    tmp_store.patch("tasks", {"id": "t1", "text": "wire ws", "created_at": "now"})
    tmp_store.patch("tasks", {"id": "t1", "text": "done ws", "status": "done", "created_at": "now"})
    state = tmp_store.get()
    assert len(state.tasks) == 1
    assert state.tasks[0].text == "done ws"
    assert state.tasks[0].status == "done"


def test_invalid_field_raises(tmp_store: BoardStateStore) -> None:
    with pytest.raises(ValueError):
        tmp_store.patch("direct_messages", {"id": "m1"})


def test_overwrite_board(tmp_store: BoardStateStore) -> None:
    tmp_store.patch("tasks", {"id": "t1", "text": "wire ws", "created_at": "now"})
    state = tmp_store.overwrite(
        {
            "updated_at": "now",
            "objectives": [{"id": "o1", "text": "ship it", "created_at": "now"}],
        }
    )
    assert len(state.objectives) == 1
    assert state.tasks == []


def test_persistence_roundtrip(tmp_path: Path) -> None:
    store = BoardStateStore(tmp_path)
    store.patch("objectives", {"id": "o1", "text": "ship it", "created_at": "now"})
    del store
    store2 = BoardStateStore(tmp_path)
    state = store2.get()
    assert len(state.objectives) == 1
    assert state.objectives[0].text == "ship it"


def test_add_direct_message(tmp_store: BoardStateStore) -> None:
    dm = DirectMessage(
        msg_id="m1",
        from_agent="claude",
        to_agent="kimi",
        content="hi",
        timestamp="now",
    )
    tmp_store.add_direct_message(dm)
    assert len(tmp_store.get().direct_messages) == 1


def test_direct_message_dedup(tmp_store: BoardStateStore) -> None:
    dm = DirectMessage(
        msg_id="m1",
        from_agent="claude",
        to_agent="kimi",
        content="hi",
        timestamp="now",
    )
    tmp_store.add_direct_message(dm)
    dm2 = DirectMessage(
        msg_id="m1",
        from_agent="claude",
        to_agent="kimi",
        content="hi again",
        timestamp="now2",
    )
    tmp_store.add_direct_message(dm2)
    assert len(tmp_store.get().direct_messages) == 1
    assert tmp_store.get().direct_messages[0].content == "hi again"


def test_load_messages_dedups_append_only_corrections(tmp_path: Path) -> None:
    messages_path = tmp_path / "messages.jsonl"
    messages_path.write_text(
        "\n".join(
            [
                '{"msg_id":"m1","from_agent":"claude","to_agent":"kimi","content":"hi","timestamp":"now","delivered":false}',
                '{"msg_id":"m1","from_agent":"claude","to_agent":"kimi","content":"hi","timestamp":"now","delivered":true}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    store = BoardStateStore(tmp_path)
    messages = store.get().direct_messages

    assert len(messages) == 1
    assert messages[0].msg_id == "m1"
    assert messages[0].delivered is True
