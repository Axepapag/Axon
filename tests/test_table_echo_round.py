"""Gate 2: two echo seats, 2 cycles, offline mode."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from runtime.table import transcript, waker


FIXTURES = Path(__file__).parent / "fixtures" / "table"
BRIEF = FIXTURES / "test_brief.md"


@pytest.fixture
def brief():
    return str(BRIEF)


def test_two_echo_seats_two_cycles_offline(brief, tmp_path):
    state = tmp_path / "State" / "table"
    table = waker.RoundTable(state_root=state)
    round_id = "echo-gate2-2026-07-03"

    cfg = table.open_round(
        round_id=round_id,
        brief_path=brief,
        seats=["echo_a", "echo_b"],
        synthesizer="synth",
        max_cycles=2,
        per_turn_timeout_s=30,
        budgets={"max_invocations": 10, "board_view_events": 40, "board_view_chars": 24000},
        offline=True,
        officers=[],
    )
    assert cfg.status == "open"

    result = table.run(round_id)
    assert result["halted"] is True
    assert result["reason"] == "end_conditions_met"
    assert result["resolution_path"] is not None
    assert Path(result["resolution_path"]).exists()

    round_dir = table._round_dir(round_id)
    entries = transcript.read_jsonl(transcript.get_transcript_path(round_dir))

    # Exactly 2 cycles * 2 seats = 4 completed turns + 1 synthesizer entry.
    turns = [e for e in entries if e.get("type") == "turn"]
    completed = [e for e in turns if e.get("status") == "completed"]
    assert len(completed) == 4

    # Cycles are whole.
    cycles = [e.get("cycle") for e in completed]
    assert cycles == [1, 1, 2, 2]

    # No failures.
    assert not any(e.get("status") == "failed" for e in turns)

    # The transcript.md is rendered.
    md_path = transcript.get_rendered_path(round_dir)
    assert md_path.exists()
    md_text = md_path.read_text()
    assert "echo_a" in md_text
    assert "echo_b" in md_text

    # Save fixtures.
    fixture_dir = FIXTURES / "echo_round"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(transcript.get_transcript_path(round_dir), fixture_dir / "transcript.jsonl")
    shutil.copy(md_path, fixture_dir / "transcript.md")
    shutil.copy(round_dir / "round.json", fixture_dir / "round.json")


def test_fixture_files_committed():
    """Ensure gate 2 fixtures exist after a successful run."""
    fixture_dir = FIXTURES / "echo_round"
    assert (fixture_dir / "transcript.jsonl").exists()
    assert (fixture_dir / "transcript.md").exists()
    assert (fixture_dir / "round.json").exists()
    data = json.loads((fixture_dir / "transcript.jsonl").read_text().splitlines()[0])
    assert data.get("type") == "turn"
