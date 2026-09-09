from __future__ import annotations

import json
from pathlib import Path

from scripts.append_engineers_ledger_event import append_event


def _event(event_id: str) -> dict[str, object]:
    return {
        "schema": "axon-engineers-ledger-event-v1",
        "event_id": event_id,
        "timestamp": "2026-09-09T00:00:00+00:00",
        "agent": {"name": "test", "model": "test", "session": "test"},
        "turn": {"request": "test", "status": "completed", "summary": "test"},
        "actions": [],
        "files": {"created": [], "modified": [], "moved": [], "deleted": []},
        "verification": [],
        "decisions": [],
        "flags": [],
        "next_steps": [],
        "identity_stamp": "test / test / 2026-09-09",
    }


def _write_event(path: Path, event: dict[str, object]) -> None:
    path.write_text(
        json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_append_preserves_historical_blank_lines(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    first = _event("evt-first")
    second = _event("evt-second")
    ledger.write_text(
        json.dumps(first, ensure_ascii=False, separators=(",", ":")) + "\n\n",
        encoding="utf-8",
    )
    event_path = tmp_path / "event.json"
    _write_event(event_path, second)

    assert append_event(ledger, event_path) == "evt-second"
    assert ledger.read_text(encoding="utf-8").splitlines() == [
        json.dumps(first, ensure_ascii=False, separators=(",", ":")),
        "",
        json.dumps(second, ensure_ascii=False, separators=(",", ":")),
    ]
    assert append_event(ledger, event_path) == "evt-second"
    assert ledger.read_text(encoding="utf-8").splitlines().count("") == 1
