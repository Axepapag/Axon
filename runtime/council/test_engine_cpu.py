#!/usr/bin/env python3
"""Verification gate 2 (CONTRACT.md): CPU end-to-end council tick test.

Engine start -> 3 cores -> submit "How are you?" -> ticks run, consolidator
rotates round-robin (crown visits cores 0, 1, 2), the draft commits
"I am doing well.", stable halt, turn lands in conversation_history.

Run from D:/Axon with the global python:
    python runtime/council/test_engine_cpu.py

Kimmy engine subagent / kimi-k2 / 2026-08-17
"""
from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.council.engine import CouncilEngine  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent / "council_config.test.json"
SOULS_DIR = Path(__file__).resolve().parent / "souls"
TIMEOUT_S = 300.0


async def main() -> int:
    # Clean boot: default config + fresh souls so the run is deterministic.
    if SOULS_DIR.exists():
        shutil.rmtree(SOULS_DIR)
    events: list[dict] = []
    engine = CouncilEngine(CONFIG_PATH, events.append)
    engine.apply_config({"tick_delay_ms": 10, "cores": 3, "device": "cpu",
                         "stable_ticks": 2, "temperature_spread": 0.0})

    await engine.start()
    assert engine.status()["running"], "engine not running after start()"
    await engine.submit_user_message("How are you?")

    t0 = time.time()
    axon_reply: str | None = None
    while time.time() - t0 < TIMEOUT_S:
        for e in events:
            if e["type"] == "chat" and e.get("role") == "axon":
                axon_reply = e["text"]
        if axon_reply is not None:
            break
        await asyncio.sleep(0.5)
    await engine.stop()

    consolidators = [e["consolidator"] for e in events if e["type"] == "tick_start"]
    soul_events = [e for e in events if e["type"] == "soul"]
    core_deltas = [e for e in events if e["type"] == "core_delta"]
    status = engine.status()
    history = status["canonical_state"]["conversation_history"]

    print(f"ticks run: {status['tick']}, consolidators seen: {consolidators[:6]}")
    print(f"events: {len(events)} total, {len(core_deltas)} core_delta, {len(soul_events)} soul")
    print(f"axon reply: {axon_reply!r}")
    print(f"conversation_history tail: {history[-80:]!r}")
    print(f"souls saved: {sorted(p.name for p in SOULS_DIR.glob('core_*.pt'))}")

    failures = []
    if axon_reply is None:
        failures.append("no axon chat reply within timeout")
    else:
        if axon_reply != "I am doing well.":
            failures.append(f"draft mismatch: {axon_reply!r} != 'I am doing well.'")
        if f"Assistant: {axon_reply}" not in history:
            failures.append("turn did not land in conversation_history")
    if consolidators[:3] != [0, 1, 2]:
        failures.append(f"consolidator did not rotate 0,1,2: {consolidators[:3]}")
    if not soul_events:
        failures.append("no soul inhale/exhale events")
    phases = {e["phase"] for e in core_deltas}
    if phases != {"A", "B"}:
        failures.append(f"missing phase A/B core_delta events: {phases}")
    if len(status["cores"]) != 3:
        failures.append("status does not report 3 cores")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("GATE 2 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
