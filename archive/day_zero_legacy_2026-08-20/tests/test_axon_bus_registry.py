"""Participant registry tests for the Axon collaboration bus."""
from __future__ import annotations

import asyncio

from bus.registry import ParticipantRegistry


def test_register_and_get() -> None:
    reg = ParticipantRegistry()
    p = reg.register("kimi", capabilities=["read"])
    assert reg.get("kimi") is p
    assert p.capabilities == ["read"]


def test_register_reuses_queue() -> None:
    reg = ParticipantRegistry()
    p1 = reg.register("kimi")
    p1.queue.put_nowait({"type": "x"})
    p2 = reg.register("kimi", capabilities=["write"])
    assert p1.queue is p2.queue
    assert p2.queue.qsize() == 1


def test_unregister() -> None:
    reg = ParticipantRegistry()
    reg.register("kimi")
    removed = reg.unregister("kimi")
    assert removed is not None
    assert removed.agent_id == "kimi"
    assert reg.get("kimi") is None


def test_snapshot() -> None:
    reg = ParticipantRegistry()
    reg.register("kimi", capabilities=["read"], connected_at="now")
    reg.register("codex", capabilities=["write"], connected_at="now")
    snap = reg.snapshot()
    assert len(snap) == 2
    ids = {s["agent_id"] for s in snap}
    assert ids == {"kimi", "codex"}


def test_is_online() -> None:
    reg = ParticipantRegistry()
    reg.register("kimi")
    assert reg.is_online("kimi")
    assert not reg.is_online("claude")
