"""Gate 3 harness: live smoke with the real bus and a real kimi seat.

This file implements the harness but does NOT execute gate 3 automatically.
Gate 3 requires an officer present and a live bus; run it manually via
`runtime/table/run_gate3.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime.table import buslink, waker


BRIEF = Path(__file__).parent / "fixtures" / "table" / "test_brief.md"


@pytest.mark.skip(reason="Gate 3 requires officer presence and live bus; run runtime/table/run_gate3.py manually.")
def test_live_smoke_with_kimi():
    """Harness entry point for gate 3.

    When un-skipped and run with BUS_TOKEN set, this test performs the live
    smoke round.  It is kept skipped to prevent accidental automated execution.
    """
    assert "BUS_TOKEN" in os.environ, "BUS_TOKEN must be exported"

    round_id = "gate3-live-harness"
    table = waker.RoundTable()
    table.open_round(
        round_id=round_id,
        brief_path=str(BRIEF),
        seats=["echo", "kimi"],
        synthesizer="kimi",
        max_cycles=1,
        per_turn_timeout_s=600,
        budgets={"max_invocations": 6, "board_view_events": 40, "board_view_chars": 24000},
        offline=False,
        officers=["codex", "claude"],
    )

    result = table.run(round_id)
    assert result["halted"] is True

    link = buslink.BusLink(offline=False)
    import asyncio

    events = asyncio.run(link.replay_events(patterns=["agent.kimi.*"], timeout=5.0))
    assert any(e.get("topic", "").startswith("agent.kimi.") for e in events)

    resolution_path = result.get("resolution_path")
    assert resolution_path
    assert Path(resolution_path).exists()


def test_gate3_runner_script_exists():
    """Verify the one-command runner script is present and documented."""
    runner = Path(__file__).parent.parent / "runtime" / "table" / "run_gate3.py"
    assert runner.exists()
    text = runner.read_text(encoding="utf-8")
    assert "BUS_TOKEN" in text
    assert "officer" in text.lower()
    assert "agent.kimi.turn.completed" in text
