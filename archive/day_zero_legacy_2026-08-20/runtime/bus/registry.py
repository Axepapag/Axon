"""In-memory participant registry for the collaboration bus."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Participant:
    agent_id: str
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    capabilities: list[str] = field(default_factory=list)
    subscription: str = "*"
    connected_at: str | None = None
    last_seen: str | None = None
    last_pong_at: float = field(default_factory=time.monotonic)


class ParticipantRegistry:
    """Holds currently connected bus participants."""

    def __init__(self) -> None:
        self._participants: dict[str, Participant] = {}

    def register(
        self,
        agent_id: str,
        capabilities: list[str] | None = None,
        subscription: str = "*",
        connected_at: str | None = None,
    ) -> Participant:
        if agent_id in self._participants:
            old = self._participants[agent_id]
            participant = Participant(
                agent_id=agent_id,
                queue=old.queue,
                capabilities=capabilities or old.capabilities,
                subscription=subscription,
                connected_at=connected_at,
                last_seen=connected_at,
                last_pong_at=time.monotonic(),
            )
        else:
            participant = Participant(
                agent_id=agent_id,
                capabilities=capabilities or [],
                subscription=subscription,
                connected_at=connected_at,
                last_seen=connected_at,
                last_pong_at=time.monotonic(),
            )
        self._participants[agent_id] = participant
        return participant

    def unregister(self, agent_id: str) -> Participant | None:
        return self._participants.pop(agent_id, None)

    def get(self, agent_id: str) -> Participant | None:
        return self._participants.get(agent_id)

    def is_online(self, agent_id: str) -> bool:
        return agent_id in self._participants

    def all(self) -> list[Participant]:
        return list(self._participants.values())

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "agent_id": p.agent_id,
                "capabilities": p.capabilities,
                "subscription": p.subscription,
                "connected_at": p.connected_at,
                "last_seen": p.last_seen,
            }
            for p in self._participants.values()
        ]
