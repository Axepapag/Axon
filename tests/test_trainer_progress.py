"""Durable training progress survives its launching process."""

from __future__ import annotations

import io
import json
import sys

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


def test_evaluation_metrics_and_unicode_samples_are_durably_supported(tmp_path, monkeypatch) -> None:
    output = io.BytesIO()
    legacy_console = io.TextIOWrapper(output, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", legacy_console)
    journal = TrainingProgressJournal(tmp_path, job_id="cloud-resume")
    event = journal.emit(
        "evaluated", phase="initial", global_step=360, heldout_mean_loss=2.738,
        qa_transcripts=[{"prompt": "Copy: 水🙂", "predicted_payload": "水🙂", "expected_payload": "水🙂"}],
    )
    current = json.loads(journal.current_path.read_text(encoding="utf-8"))
    assert current["event_id"] == event.event_id
    assert current["status"] == "evaluated"
    assert current["details"]["qa_transcripts"][0]["predicted_payload"] == "水🙂"
    printed = output.getvalue().decode("cp1252").removeprefix("AXON_PROGRESS ")
    assert json.loads(printed) == current
