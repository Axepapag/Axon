"""Gate 4: timeout failure drill — seat disabled after second failure."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from runtime.table import transcript, waker
from runtime.table.manifests import ManifestDir


FIXTURES = Path(__file__).parent / "fixtures" / "table"
BRIEF = FIXTURES / "test_brief.md"


def _make_manifests(tmp_path: Path) -> Path:
    d = tmp_path / "manifests"
    d.mkdir()
    for name in ("echo_a.json", "echo_b.json", "synth.json"):
        shutil.copy(ManifestDir / name, d / name)
    slow = {
        "client_id": "slow",
        "display_name": "Slow",
        "wake": {
            "argv": ["python", "-c", "import time; time.sleep(999)"],
            "prompt_via": "argv",
            "timeout_seconds": 2,
            "workdir": str(tmp_path),
        },
        "capabilities": ["test"],
        "identity_stamp": "Slow / slow-seat / set-at-wake",
        "enabled": True,
        "notes": "Always times out.",
    }
    (d / "slow.json").write_text(json.dumps(slow), encoding="utf-8")
    return d


def test_timeout_failure_drill(tmp_path):
    manifests_dir = _make_manifests(tmp_path)
    state = tmp_path / "State" / "table"
    table = waker.RoundTable(state_root=state, manifests_dir=manifests_dir)
    round_id = "failure-gate4-2026-07-03"

    table.open_round(
        round_id=round_id,
        brief_path=str(BRIEF),
        seats=["echo_a", "slow"],
        synthesizer="synth",
        max_cycles=2,
        per_turn_timeout_s=2,
        budgets={"max_invocations": 10, "board_view_events": 40, "board_view_chars": 24000},
        offline=True,
        officers=[],
    )

    result = table.run(round_id)
    assert result["halted"] is True

    cfg = table._load_config(round_id)
    assert "slow" in cfg.disabled_seats

    round_dir = table._round_dir(round_id)
    entries = transcript.read_jsonl(transcript.get_transcript_path(round_dir))
    failures = [e for e in entries if e.get("type") == "turn" and e.get("status") == "failed" and e.get("seat") == "slow"]
    assert len(failures) >= 2
    assert all(e["error"]["reason"] == "timeout" for e in failures)

    # The round continued despite failures (echo_a completed turns exist).
    echo_completed = [e for e in entries if e.get("seat") == "echo_a" and e.get("status") == "completed"]
    assert len(echo_completed) >= 2

    # .turn.failed events were published to offline outbox.
    outbox_events = table.bus._read_outbox()
    failed_topics = [e.topic for e in outbox_events if e.topic == "agent.slow.turn.failed"]
    assert len(failed_topics) >= 2
