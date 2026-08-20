"""Gate 5: staleness drill — SOT change mid-round halts with stale_doctrine."""

from __future__ import annotations

from pathlib import Path

from runtime.table import waker


FIXTURES = Path(__file__).parent / "fixtures" / "table"
BRIEF = FIXTURES / "test_brief.md"


def test_staleness_drill(tmp_path):
    state = tmp_path / "State" / "table"
    # Scratch copy of the SOT.
    sot = tmp_path / "SOURCE_OF_TRUTH.md"
    sot.write_text("# SOT\n\nOriginal doctrine.\n", encoding="utf-8")

    table = waker.RoundTable(state_root=state)
    round_id = "stale-gate5-2026-07-03"

    table.open_round(
        round_id=round_id,
        brief_path=str(BRIEF),
        seats=["echo_a", "echo_b"],
        synthesizer="synth",
        max_cycles=2,
        per_turn_timeout_s=30,
        budgets={"max_invocations": 10, "board_view_events": 40, "board_view_chars": 24000},
        offline=True,
        officers=[],
        sot_path=str(sot),
    )

    # First turn completes.
    result1 = table.step(round_id)
    assert result1["halted"] is False

    # Modify the scratch SOT mid-round.
    sot.write_text("# SOT\n\nModified doctrine.\n", encoding="utf-8")

    # Next turn must halt with stale_doctrine.
    result2 = table.step(round_id)
    assert result2["halted"] is True
    assert result2["reason"] == "stale_doctrine"

    cfg = table._load_config(round_id)
    assert cfg.status == "stale_doctrine"
