"""Durable training progress survives its launching process."""

from __future__ import annotations

import json

from runtime.trainer import TrainingProgressJournal


def test_progress_appends_jsonl_updates_current_and_resumes_sequence(tmp_path, capsys) -> None:
    root = tmp_path / "progress"
    first = TrainingProgressJournal(root, job_id="job-1")
    event_one = first.emit("starting", candidate="c")
    event_two = first.emit("training", global_step=1, loss=2.5)

    current = json.loads((root / "current.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [item["sequence"] for item in events] == [1, 2]
    assert current["event_id"] == event_two.event_id
    assert event_one.event_id != event_two.event_id
    assert "AXON_PROGRESS" in capsys.readouterr().out

    resumed = TrainingProgressJournal(root, job_id="job-1")
    event_three = resumed.emit("paused", global_step=1)
    assert event_three.sequence == 3


def test_progress_directory_cannot_be_reused_by_another_job(tmp_path) -> None:
    root = tmp_path / "progress"
    TrainingProgressJournal(root, job_id="job-1").emit("starting")
    try:
        TrainingProgressJournal(root, job_id="job-2")
    except RuntimeError as exc:
        assert "another job" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("cross-job progress reuse was accepted")
