"""Core routing: broadcast, direct delivery, and event replay."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .log import append_jsonl, read_jsonl, read_jsonl_after_id, read_jsonl_reverse
from .registry import Participant, ParticipantRegistry
from .schema import (
    BOARD_ITEM_MODELS,
    BOARD_FIELDS,
    BoardState,
    BroadcastPayload,
    DirectMessage,
    DirectMessagePayload,
    HelloPayload,
    MessageEnvelope,
    StatusPayload,
    topic_matches,
)
from .state import BoardStateStore


class Bus:
    """Single-node Axon collaboration bus."""

    def __init__(self, state_dir: Path, registry: ParticipantRegistry | None = None) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = BoardStateStore(self.state_dir)
        self.registry = registry or ParticipantRegistry()
        self.events_path = self.state_dir / "events.jsonl"
        self.messages_path = self.state_dir / "messages.jsonl"
        self.participants_path = self.state_dir / "participants.jsonl"
        self._seq = self._init_seq()

    def _init_seq(self) -> int:
        seq = 0
        for path in (self.events_path, self.messages_path, self.participants_path):
            for record in read_jsonl(path):
                mid = record.get("msg_id", "")
                try:
                    seq = max(seq, int(mid))
                except (ValueError, TypeError):
                    continue
        return seq

    def _next_id(self) -> str:
        self._seq += 1
        return f"{self._seq:020d}"

    # -----------------------------------------------------------------------
    # Events / logging
    # -----------------------------------------------------------------------
    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _log_event(self, envelope: MessageEnvelope) -> None:
        envelope.msg_id = self._next_id()
        record = envelope.model_dump()
        append_jsonl(self.events_path, record)

    def _log_participant(self, agent_id: str, event: str, meta: dict[str, Any] | None = None) -> None:
        record = {
            "msg_id": self._next_id(),
            "agent_id": agent_id,
            "event": event,
            "timestamp": self._now(),
            "meta": meta or {},
        }
        append_jsonl(self.participants_path, record)

    def recent_events(self, after_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        return read_jsonl_after_id(self.events_path, after_id, id_field="msg_id", limit=limit)

    # -----------------------------------------------------------------------
    # Board operations
    # -----------------------------------------------------------------------
    def patch_board(self, field: str, value: dict[str, Any], from_agent: str = "") -> BoardState:
        if field not in BOARD_FIELDS - {"direct_messages"}:
            raise ValueError(f"invalid board field: {field}")
        # Normalize id if missing.
        if "id" not in value:
            value["id"] = uuid4().hex[:12]
        if "created_at" not in value:
            value["created_at"] = self._now()
        state = self.store.patch(field, value)
        env = MessageEnvelope(
            type="board_update",
            from_agent=from_agent,
            topic="board",
            payload={"field": field, "value": value},
        )
        self._log_event(env)
        return state

    def overwrite_board(self, data: dict[str, Any], from_agent: str = "") -> BoardState:
        state = self.store.overwrite(data)
        env = MessageEnvelope(
            type="board_update",
            from_agent=from_agent,
            topic="board",
            payload={"full_board": True, "version": state.version},
        )
        self._log_event(env)
        return state

    # -----------------------------------------------------------------------
    # Broadcast
    # -----------------------------------------------------------------------
    def broadcast(
        self, topic: str, payload: dict[str, Any], from_agent: str = ""
    ) -> MessageEnvelope:
        env = MessageEnvelope(
            type="broadcast",
            from_agent=from_agent,
            topic=topic,
            payload=payload,
        )
        self._log_event(env)
        return env

    def matching_recipients(self, topic: str) -> list[Participant]:
        return [p for p in self.registry.all() if topic_matches(topic, p.subscription)]

    # -----------------------------------------------------------------------
    # Direct messages
    # -----------------------------------------------------------------------
    def send_direct_message(
        self, payload: DirectMessagePayload, from_agent: str
    ) -> tuple[DirectMessage, bool]:
        dm = DirectMessage(
            msg_id=self._next_id(),
            from_agent=from_agent,
            to_agent=payload.to_agent,
            content=payload.content,
            in_reply_to=payload.in_reply_to,
            timestamp=self._now(),
            delivered=False,
        )
        append_jsonl(self.messages_path, dm.model_dump())
        delivered = False
        recipient = self.registry.get(payload.to_agent)
        if recipient is not None:
            env = MessageEnvelope(
                type="direct_message",
                from_agent=from_agent,
                to_agent=payload.to_agent,
                payload=dm.model_dump(),
            )
            try:
                recipient.queue.put_nowait(env.model_dump())
                delivered = True
            except asyncio.QueueFull:
                delivered = False
        dm.delivered = delivered
        # Update board state DM list.
        self.store.add_direct_message(dm)
        # Rewrite the messages log entry with delivered status (append corrected entry).
        append_jsonl(self.messages_path, dm.model_dump())
        return dm, delivered

    def direct_messages_for(
        self, agent_id: str, with_agent: str | None = None, after_id: str | None = None, limit: int = 50
    ) -> list[DirectMessage]:
        dms = self.store.get().direct_messages
        if with_agent:
            dms = [m for m in dms if m.to_agent == agent_id and m.from_agent == with_agent]
        else:
            dms = [m for m in dms if m.to_agent == agent_id]
        if after_id is not None:
            dms = [m for m in dms if m.msg_id > after_id]
        if limit and limit < len(dms):
            dms = dms[-limit:]
        return dms

    # -----------------------------------------------------------------------
    # Participant lifecycle
    # -----------------------------------------------------------------------
    def connect(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        subscription: str = "*",
        last_event_id: str | None = None,
        last_dm_id: str | None = None,
    ) -> tuple[Participant, dict[str, Any]]:
        participant = self.registry.register(
            agent_id=agent_id,
            capabilities=capabilities or [],
            subscription=subscription,
            connected_at=self._now(),
        )
        self._log_participant(agent_id, "connect", {"capabilities": capabilities})
        # Sync board participant list.
        self.store.set_participants(self.registry.snapshot())
        # Replay.
        missed_events = self.recent_events(after_id=last_event_id, limit=200)
        missed_dms = [
            dm.model_dump()
            for dm in self.direct_messages_for(agent_id, after_id=last_dm_id, limit=200)
        ]
        info = {
            "missed_events": missed_events,
            "missed_dms": missed_dms,
        }
        return participant, info

    def disconnect(self, agent_id: str) -> None:
        self.registry.unregister(agent_id)
        self._log_participant(agent_id, "disconnect")
        self.store.set_participants(self.registry.snapshot())

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "agents": len(self.registry.all()),
            "board_version": self.store.get().version,
        }
