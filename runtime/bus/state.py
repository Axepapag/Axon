"""Board state persistence and patch logic."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .log import read_jsonl, write_json
from .schema import (
    BOARD_FIELDS,
    BOARD_ITEM_MODELS,
    BoardState,
    DirectMessage,
)


class BoardStateStore:
    """Persistent in-memory board state backed by board.json + messages.jsonl."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.board_path = self.state_dir / "board.json"
        self.messages_path = self.state_dir / "messages.jsonl"
        self._state = self._load()

    def _load(self) -> BoardState:
        if self.board_path.exists():
            try:
                data = BoardState.model_validate_json(self.board_path.read_text(encoding="utf-8"))
                # Always reload messages from the log to avoid stale DM list.
                data.direct_messages = self._load_messages()
                return data
            except (ValidationError, json.JSONDecodeError):
                pass
        return BoardState(updated_at=_now(), direct_messages=self._load_messages())

    def _load_messages(self) -> list[DirectMessage]:
        messages_by_id: dict[str, DirectMessage] = {}
        for record in read_jsonl(self.messages_path):
            try:
                message = DirectMessage.model_validate(record)
            except ValidationError:
                continue
            messages_by_id[message.msg_id] = message
        return list(messages_by_id.values())

    def _save(self) -> None:
        self._state.updated_at = _now()
        write_json(self.board_path, self._state.model_dump())

    def get(self) -> BoardState:
        return self._state

    def set_participants(self, participants: list[dict[str, Any]]) -> None:
        self._state.participants = participants
        self._save()

    def patch(self, field: str, value: dict[str, Any]) -> BoardState:
        if field not in BOARD_FIELDS - {"direct_messages"}:
            raise ValueError(f"invalid board field: {field}")
        model = BOARD_ITEM_MODELS[field]
        item = model.model_validate(value)
        # Replace by id if present, else append.
        current = getattr(self._state, field)
        ids = {it.id: idx for idx, it in enumerate(current)}
        if item.id in ids:
            current[ids[item.id]] = item
        else:
            current.append(item)
        self._save()
        return self._state

    def overwrite(self, data: dict[str, Any]) -> BoardState:
        state = BoardState.model_validate(data)
        # Preserve messages from the log, not the incoming snapshot.
        state.direct_messages = self._load_messages()
        self._state = state
        self._save()
        return self._state

    def add_direct_message(self, dm: DirectMessage) -> BoardState:
        # De-duplicate by msg_id.
        existing = {m.msg_id: idx for idx, m in enumerate(self._state.direct_messages)}
        if dm.msg_id in existing:
            self._state.direct_messages[existing[dm.msg_id]] = dm
        else:
            self._state.direct_messages.append(dm)
        self._save()
        return self._state


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
