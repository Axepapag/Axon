"""Durable training progress survives its launching process."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

from runtime.heart import HeartHost
from runtime.trainer import TrainingProgressJournal
from training.first_form_curriculum import load_first_form_curriculum, publish_first_form_curriculum
from training.foundation_motor_curriculum import (
    compile_foundation_motor_v2,
    compile_foundation_motor_v2_unicode_walk,
)

from ._short_tmp import short_state_root

IDENTITY = "Axon is Axon. Every brother attends the complete exact field."
ROOT = Path(__file__).resolve().parents[1]


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


def test_local_events_attribute_each_step_and_transcript_to_its_curriculum(tmp_path) -> None:
    """Two curricula sharing one family and stage must stay distinguishable.

    The lane name is built from the family and stage alone, so two manifests of
    family F0 both produce the same lane. Before the emitted details carried the
    source manifest, a dashboard could not tell which curriculum a step came
    from. The lane name is deliberately left alone; attribution is a field.

    The trainer rejects duplicate episodes and refuses to mix motor generations,
    so both curricula are motor v2 compiled by different lesson generators. A
    fresh motor-v2 campaign opens at the first stage, which both manifests share.
    """
    state_root = short_state_root("progress-attribution") / "State"
    with HeartHost(state_root=state_root) as host:
        host.amend_identity(
            IDENTITY,
            amendment_id="progress-attribution-identity-v1",
            evidence_ids=("progress-attribution-evidence",),
            provenance="tests/test_trainer_progress.py",
        )
    curricula = (
        compile_foundation_motor_v2(identity_text=IDENTITY, requested_counts=(("F0", (36, 36, 36)),)),
        compile_foundation_motor_v2_unicode_walk(
            identity_text=IDENTITY, requested_counts=(("F0", (36, 36, 36)),)
        ),
    )
    manifests = [
        publish_first_form_curriculum(curriculum, state_root=state_root)
        for curriculum in curricula
    ]
    expected = {
        load_first_form_curriculum(path).teaching_living_curriculum.train_manifest_id
        for path in manifests
    }
    assert len(expected) == 2
    progress_dir = tmp_path / "progress"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_living_reasoning_smoke.py"),
            "--state-root", str(state_root),
            "--curriculum-manifest", str(manifests[0]),
            "--curriculum-manifest", str(manifests[1]),
            "--candidate-label", "progress-attribution",
            "--device", "cpu",
            "--tranche-steps", "3",
            "--checkpoint-interval", "1",
            "--evaluation-case-limit", "2",
            "--ffn-dim", "128",
            "--layers", "1",
            "--seed", "20260906",
            "--progress-dir", str(progress_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, f"{completed.stdout[-2000:]}\n{completed.stderr[-4000:]}"
    events = [
        json.loads(line)
        for line in (progress_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    steps = [event for event in events if event["status"] == "training"]
    assert len(steps) == 3
    # Both manifests contribute the identical lane name, and a staged motor-v2
    # campaign omits the mechanism lane.
    assert {step["details"]["curriculum_lane"] for step in steps} == {"ffcs-F0-copy_alignment"}
    # The emitted manifest identity, not the lane name, separates the two.
    assert {step["details"]["source_manifest_id"] for step in steps} == expected
    for step in steps:
        assert step["details"]["material_kind"] == "first_form_case"
        assert step["details"]["material_id"]
        assert step["details"]["material_label"]

    # The evaluation surface must be attributable too: `episode_id` alone is a
    # hash, so a failing transcript could not say which case or curriculum it
    # came from, and an `isolated_manifest_evaluations` table can silently
    # collapse to one manifest when the evaluated surface is limited.
    transcripts = [
        row
        for event in events
        if event["status"] == "evaluated"
        for row in event["details"].get("qa_transcripts") or []
    ]
    assert transcripts, "the tranche must report evaluation transcripts"
    for row in transcripts:
        assert row["episode_label"]
        assert len(str(row["case_id"])) == 64
        assert row["family"] == "F0"
        # The evaluation must reuse the identifier the training lane tags with.
        assert row["source_manifest_id"] in expected
        assert row["manifest_id"]
