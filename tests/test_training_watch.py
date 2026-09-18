"""Operator telemetry must distinguish new work from replay and global progress."""

import re

from scripts import axon_training_watch
from scripts.axon_training_watch import Watcher


def _watcher():
    return Watcher(window=60, show_qa=True, transcript_lines=12)


def _event(event_id, status, **details):
    return {
        "schema": "axon-training-progress-event-v1",
        "event_id": event_id,
        "status": status,
        "details": details,
    }


def test_live_replay_never_counts_a_step_or_bundle_twice():
    watcher = _watcher()
    event = _event(
        "step-390", "training", global_step=390, loss=2.5, curriculum_lane="ffcs-L0",
        checkpoint_id="checkpoint", accepted_step_bundle_id="bundle", wall_seconds=6.0,
    )
    watcher.consume(event)
    watcher.consume(event)
    assert watcher.event_count == 1
    assert watcher.checkpoints == watcher.bundles == 1
    assert list(watcher.losses) == [2.5]


def test_resumed_tranche_progress_is_not_mislabeled_complete():
    watcher = _watcher()
    watcher.consume(_event("start", "starting", tranche_steps=360))
    watcher.consume(_event("eval", "evaluating", phase="initial", global_step=360))
    assert "tranche: 0/360  global step: 360" in watcher.render()
    watcher.consume(_event("step", "training", global_step=361, segment_end_step=720))
    assert watcher.segment_start == 360
    assert "tranche: 1/360  global step: 361" in watcher.render()
    assert "not autonomous conversation" in watcher.render()


def test_motor_v2_copy_metrics_and_recent_steps_are_visible():
    watcher = _watcher()
    watcher.job_id = "a" * 64
    watcher.job_title = "Axon D64 mixer 4L FFN256 multi-cell copy teach"
    watcher.consume(
        _event(
            "eval-initial",
            "evaluated",
            phase="initial",
            heldout_mean_loss=1.2,
            foundation_motor_v2_heldout_probe={
                "case_count": 8,
                "alignment_copy_gate_accuracy": 1.0,
                "alignment_position_accuracy": 1.0,
                "pair_copy_gate": 1.0,
                "pair_position": 1.0,
            },
            foundation_motor_v2_regression_probe={
                "case_count": 8,
                "alignment_copy_gate_accuracy": 1.0,
                "alignment_position_accuracy": 0.667,
                "pair_copy_gate": 1.0,
                "pair_position": 0.667,
            },
        )
    )
    watcher.consume(
        _event(
            "step-121",
            "training",
            global_step=121,
            loss=0.42,
            curriculum_lane="copy_alignment",
            material_kind="authored",
            wall_seconds=3.5,
        )
    )
    rendered = watcher.render()
    assert "Axon D64 mixer 4L FFN256 multi-cell copy teach" in rendered
    assert "motor v2 initial/heldout" in rendered
    assert "copy-gate 1.000" in rendered
    assert "position 0.667" in rendered
    assert "121" in rendered and "0.420" in rendered
    assert "copy_alignment" in rendered
    assert "sync:" not in rendered


def test_sync_dashboard_reports_verified_windows_and_disabled_recipe():
    watcher = _watcher()
    watcher.note_sync(
        enabled=True,
        verified_through_step=45,
        verified_ranges=[[16, 30], [31, 45]],
        released_ranges=[[1, 15]],
        pulled=1,
    )
    rendered = watcher.render()
    assert "verified through step 45" in rendered
    assert "kept 2/3 windows" in rendered
    assert "released 1 older payloads" in rendered
    watcher.note_sync(enabled=False, waiting="recipe did not enable mid-run checkpoint uploads")
    assert "did not enable mid-run checkpoint uploads" in watcher.render()


def test_axon_sync_kernel_receipt_is_visible_on_the_dashboard():
    watcher = _watcher()
    watcher.consume(
        {
            "schema": "axon-mid-run-sync-receipt-v1",
            "status": "uploaded",
            "details": {"step_range": [16, 30]},
        }
    )
    assert "uploaded steps 16-30" in watcher.render()


def test_runner_failure_is_visible_even_before_first_training_event():
    watcher = _watcher()
    watcher.consume({
        "schema": "axon-kaggle-runner-event-v1",
        "status": "failed",
        "details": {"error": "CUDA probe failed"},
    })
    assert watcher.status == "failed"
    assert "CUDA probe failed" in watcher.render()


def test_events_flag_watches_a_local_journal_without_a_job_id(tmp_path, monkeypatch):
    journal = tmp_path / "events.jsonl"
    journal.write_text(
        '{"schema":"axon-training-progress-event-v1","job_id":"local-9",'
        '"status":"training","sequence":1,"details":{"global_step":1,"loss":9.0}}\n',
        encoding="utf-8",
    )
    seen = {}

    def fake_follow(path, watcher, poll):
        seen["path"] = path
        seen["job_id"] = watcher.job_id
        return 0

    monkeypatch.setattr(axon_training_watch, "_follow_local", fake_follow)
    assert axon_training_watch.follow_job(None, events_path=journal) == 0
    assert seen["path"] == journal
    assert seen["job_id"] == "local-9"


def test_events_flag_replays_a_local_journal_from_the_start(tmp_path, monkeypatch):
    journal = tmp_path / "events.jsonl"
    journal.write_text(
        '{"schema":"axon-training-progress-event-v1","job_id":"local-9","status":"training",'
        '"sequence":1,"details":{"global_step":1,"loss":9.0}}\n',
        encoding="utf-8",
    )
    seen = {}

    def fake_replay(path, watcher, *, follow, poll_seconds):
        seen["path"] = path
        return 0

    monkeypatch.setattr(axon_training_watch, "_replay_local", fake_replay)
    assert axon_training_watch.follow_job(None, events_path=journal, replay=True) == 0
    assert seen["path"] == journal


def test_events_flag_fails_closed_when_the_journal_never_appears(tmp_path, monkeypatch):
    monkeypatch.setattr(axon_training_watch, "_wait_for_events", lambda path, timeout=60.0: False)
    try:
        axon_training_watch.follow_job(None, events_path=tmp_path / "absent.jsonl")
    except SystemExit as exc:
        assert "no local events at" in str(exc)
    else:  # pragma: no cover - the assertion path is the contract
        raise AssertionError("a local journal that never appears must fail closed")


def test_local_flag_without_a_job_id_still_fails_closed():
    try:
        axon_training_watch.follow_job(None, local=True)
    except SystemExit as exc:
        assert "job id is required" in str(exc)
    else:  # pragma: no cover - the assertion path is the contract
        raise AssertionError("--local without a job id must fail closed")


def test_two_curricula_sharing_a_lane_name_stay_attributable():
    watcher = _watcher()
    for index, manifest in enumerate(("1" * 64, "2" * 64), start=1):
        watcher.consume(
            _event(
                f"step-{index}",
                "training",
                global_step=index,
                loss=1.0,
                curriculum_lane="ffcs-F0-copy_alignment",
                material_kind="first_form_case",
                material_id=f"case-{index}",
                material_label=f"unicode-walk-train-insert-00{index}-0",
                source_manifest_id=manifest,
                wall_seconds=2.0,
            )
        )
    rendered = watcher.render()
    assert "ffcs-F0-copy_alignment@1111111…:1" in rendered
    assert "ffcs-F0-copy_alignment@2222222…:1" in rendered
    assert "unicode-walk-train-insert-001-0" in rendered
    assert "unicode-walk-train-insert-002-0" in rendered


def test_training_events_without_manifest_attribution_still_render():
    """Cloud and pre-existing journals carry no manifest or case fields."""
    watcher = _watcher()
    watcher.consume(
        _event(
            "step-1",
            "training",
            global_step=1,
            loss=1.0,
            curriculum_lane="ffcs-L0",
            material_kind="first_form_case",
            wall_seconds=2.0,
        )
    )
    rendered = watcher.render()
    lane_line = next(line for line in rendered.splitlines() if line.startswith(" lanes:"))
    assert lane_line == " lanes: ffcs-L0:1"
    step_line = next(line for line in rendered.splitlines() if "first_form_case" in line)
    assert "@" not in step_line


def test_evaluation_transcripts_name_their_case_and_curriculum():
    """A failing transcript must say which case it is and who taught it.

    An evaluated row carries `episode_id` — a 64-character hash — and previously
    nothing else identifying the case. A dashboard could therefore show twelve
    failures without revealing that every one of them came from a single
    curriculum, and no one reading the run could tell.
    """
    watcher = _watcher()
    watcher.consume(
        _event(
            "eval-final",
            "evaluated",
            phase="final",
            global_step=60,
            heldout_mean_loss=3.4,
            qa_transcripts=[
                {
                    "episode_id": "1" * 64,
                    "episode_label": "unicode-walk-holdout-insert-000-0",
                    "source_manifest_id": "a872278fd0e8ef926370e1712d01dcf0a277672c4a0088af44aef483d8417740",
                    "case_id": "9a7f64f9" + "0" * 56,
                    "family": "F0",
                    "prompt": "Insert the current SOURCE_SYMBOL between the brackets.",
                    "predicted_payload": "",
                    "expected_payload": "😂",
                    "exact_match": False,
                },
                {
                    "episode_id": "2" * 64,
                    "episode_label": "plain-holdout-insert-000-1",
                    "source_manifest_id": "12df454776145707502363d801eda3a3ef7a3f81ab400b6578461f00ea12566b",
                    "prompt": "Insert the current SOURCE_SYMBOL between the brackets.",
                    "predicted_payload": "い",
                    "expected_payload": "い",
                    "exact_match": True,
                },
            ],
        )
    )
    rendered = watcher.render()
    assert "[unicode-walk-holdout-insert-000-0@a872278…]" in rendered
    assert "[plain-holdout-insert-000-1@12df454…]" in rendered
    assert "expected '😂'" in rendered


def test_evaluation_transcripts_without_attribution_still_render():
    """A cloud or pre-existing transcript has no case or manifest fields."""
    watcher = _watcher()
    watcher.consume(
        _event(
            "eval-legacy",
            "evaluated",
            phase="initial",
            global_step=1,
            heldout_mean_loss=2.0,
            qa_transcripts=[
                {"prompt": "Copy: 水🙂", "predicted_payload": "水🙂", "expected_payload": "水🙂"},
            ],
        )
    )
    qa_line = next(line for line in watcher.render().splitlines() if "Q:" in line)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", qa_line)
    assert "[" not in plain
    assert "@" not in plain


def test_transcript_failure_names_the_binding_condition_and_tallies_it():
    """A row can reproduce the expected payload and still fail on the stop token.

    ``exact_match`` is ``terminated and payload == target.payload``
    (living_reasoning_curriculum.py:789), so an empty expected payload answered
    with nothing renders as ``A: '' (expected '') x`` and reads as a broken
    display.  The failure must name why, and the verdict line must tally the
    reasons so a run's binding constraint is visible without expanding rows.
    """
    watcher = _watcher()
    watcher.consume(
        _event(
            "eval-final",
            "evaluated",
            phase="final",
            global_step=600,
            heldout_mean_loss=1.056,
            qa_transcripts=[
                {
                    "prompt": "Delete exactly response position 1; emit no replacement.",
                    "predicted_payload": "",
                    "expected_payload": "",
                    "exact_match": False,
                    "terminated": False,
                    "decision": "DELTA",
                    "operation": "REPLACE",
                    "region": "TOOL_RESULTS",
                },
                {
                    "prompt": "Insert the current SOURCE_SYMBOL between the brackets.",
                    "predicted_payload": "",
                    "expected_payload": "Α",
                    "exact_match": False,
                    "terminated": False,
                    "decision": "DELTA",
                    "operation": "REPLACE",
                    "region": "TOOL_RESULTS",
                },
                {
                    "prompt": "Insert the current SOURCE_SYMBOL between the brackets.",
                    "predicted_payload": "Α",
                    "expected_payload": "Α",
                    "exact_match": False,
                    "terminated": False,
                    "decision": "DELTA",
                },
            ],
        )
    )
    rendered = watcher.render()
    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered)
    # The empty-expected row that matches character for character is the stop
    # token and nothing else.
    assert "A: '' ✗ stop" in plain
    assert "A: '' (expected 'Α') ✗ payload+stop" in plain
    # A row that reproduces its expected payload carries no redundant echo of it.
    assert "A: 'Α' ✗ stop" in plain
    # The predicted typed fields are what expose a constant answer.
    assert "DELTA/REPLACE/TOOL_RESULTS" in plain
    # One line tallies the binding constraint for the whole sample.
    qa_line = next(line for line in plain.splitlines() if line.startswith(" qa:"))
    assert "0/3 exact" in qa_line
    assert "payload+stop 1" in qa_line
    assert "stop 2" in qa_line


def test_transcript_without_a_verdict_names_no_reason():
    """A pre-existing transcript carries no ``exact_match``; inventing a reason
    for it would claim a failure the monitor never observed."""
    watcher = _watcher()
    watcher.consume(
        _event(
            "eval-old",
            "evaluated",
            phase="initial",
            global_step=1,
            qa_transcripts=[{"prompt": "Copy: x", "predicted_payload": "x"}],
        )
    )
    plain = re.sub(r"\x1b\[[0-9;]*m", "", watcher.render())
    qa_line = next(line for line in plain.splitlines() if "Q:" in line)
    assert "typed" not in qa_line
    verdict = next(line for line in plain.splitlines() if line.startswith(" qa:"))
    assert "0/1 exact" in verdict


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_a_rate_at_its_constant_floor_is_not_rendered_as_progress():
    """A bare percentage reads as progress even when nothing was learned.

    The emit-nothing policy scores every no-op and delete case for free, so its
    typed exactness sits exactly at the constant-answer floor.  The dashboard
    must say so rather than print 33.3% and let the operator infer progress.
    """
    watcher = _watcher()
    watcher.consume(
        _event(
            "eval-initial",
            "evaluated",
            phase="initial",
            global_step=0,
            heldout_mean_loss=5.206,
            typed_emission_exact_rate=1.0 / 3.0,
            payload_transport_exact_rate=0.0,
            payload_teacher_forced_token_accuracy=0.0,
            constant_typed_emission_exact_floor=1.0 / 3.0,
            constant_payload_transport_exact_floor=1.0 / 3.0,
            constant_payload_token_accuracy_floor=0.436,
        )
    )
    line = next(line for line in watcher.render().splitlines() if "eval[" in line)
    plain = _plain(line)
    assert "typed_exact 33.3% floor 33.3% AT-FLOOR" in plain
    assert "payload_exact 0.0% floor 33.3% BELOW" in plain


def test_a_rate_above_its_floor_is_rendered_as_beaten():
    watcher = _watcher()
    watcher.consume(
        _event(
            "eval-final",
            "evaluated",
            phase="final",
            global_step=600,
            heldout_mean_loss=1.1,
            typed_emission_exact_rate=0.9,
            payload_transport_exact_rate=0.8,
            payload_teacher_forced_token_accuracy=0.95,
            constant_typed_emission_exact_floor=1.0 / 3.0,
            constant_payload_transport_exact_floor=1.0 / 3.0,
            constant_payload_token_accuracy_floor=0.436,
        )
    )
    plain = _plain(next(line for line in watcher.render().splitlines() if "eval[" in line))
    assert "typed_exact 90.0% floor 33.3% BEATEN" in plain
    assert "payload_exact 80.0% floor 33.3% BEATEN" in plain
    assert "token_acc 95.0% floor 43.6% BEATEN" in plain
    assert plain.count("BEATEN") == 3


def test_a_stale_evaluation_names_the_step_it_was_measured_at():
    """The smoke trainer only evaluates at a tranche's start and end."""
    watcher = _watcher()
    watcher.consume(_event("eval", "evaluated", phase="initial", global_step=0, heldout_mean_loss=5.2))
    for step in (300, 554):
        watcher.consume(
            _event(f"step-{step}", "training", global_step=step, loss=1.0, wall_seconds=7.2)
        )
    plain = _plain(next(line for line in watcher.render().splitlines() if "eval[" in line))
    assert "eval[initial @step 0]" in plain


def test_a_disabled_sync_receipt_states_why_it_is_disabled():
    """`kernel disabled` alone cannot be acted on; the receipt carries a reason."""
    watcher = _watcher()
    watcher.consume(
        {
            "schema": "axon-mid-run-sync-receipt-v1",
            "status": "disabled",
            "details": {"reason": "sync credentials unavailable: SyncCredentialsMissing"},
        }
    )
    rendered = watcher.render()
    assert "disabled (sync credentials unavailable: SyncCredentialsMissing)" in rendered


def test_the_teacher_forced_panel_is_opt_in_with_a_one_line_verdict():
    """The transcript panel used to consume most of the screen by default."""
    watcher = Watcher(window=60, show_qa=False, transcript_lines=12)
    watcher.consume(
        _event(
            "eval-initial",
            "evaluated",
            phase="initial",
            global_step=0,
            heldout_mean_loss=5.206,
            qa_transcripts=[
                {
                    "episode_id": "1" * 64,
                    "prompt": "Insert the current SOURCE_SYMBOL between the brackets.",
                    "predicted_payload": "",
                    "expected_payload": "Α",
                    "exact_match": False,
                },
                {
                    "episode_id": "1" * 64,
                    "prompt": "Insert the current SOURCE_SYMBOL between the brackets.",
                    "predicted_payload": "",
                    "expected_payload": "Α",
                    "exact_match": False,
                },
            ],
        )
    )
    rendered = watcher.render()
    assert "not autonomous conversation" not in rendered
    assert "Q:" not in rendered
    verdict = next(line for line in rendered.splitlines() if "qa:" in line)
    plain = _plain(verdict)
    assert "sample of 1 teacher-forced cases" in plain
    assert "0/1 exact" in plain
    assert "--qa for rows" in plain


def test_expanded_transcript_panel_collapses_repeated_cases():
    watcher = _watcher()
    watcher.consume(
        _event(
            "eval-initial",
            "evaluated",
            phase="initial",
            global_step=0,
            heldout_mean_loss=5.206,
            qa_transcripts=[
                {
                    "episode_id": "9" * 64,
                    "prompt": "Delete exactly response position 1; emit no replacement.",
                    "predicted_payload": "",
                    "expected_payload": "",
                    "exact_match": False,
                }
            ],
        )
    )
    # The trainer reports each evaluation twice; content keying must dedupe it.
    watcher.consume(
        _event(
            "eval-mirror",
            "evaluated",
            phase="initial",
            global_step=0,
            heldout_mean_loss=5.206,
            qa_transcripts=[
                {
                    "episode_id": "9" * 64,
                    "prompt": "Delete exactly response position 1; emit no replacement.",
                    "predicted_payload": "",
                    "expected_payload": "",
                    "exact_match": False,
                }
            ],
        )
    )
    rendered = watcher.render()
    assert rendered.count("Delete exactly response position 1") == 1
    assert watcher.qa_shown == 1
