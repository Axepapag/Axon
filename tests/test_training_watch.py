"""Operator telemetry must distinguish new work from replay and global progress."""

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


def test_runner_failure_is_visible_even_before_first_training_event():
    watcher = _watcher()
    watcher.consume({
        "schema": "axon-kaggle-runner-event-v1",
        "status": "failed",
        "details": {"error": "CUDA probe failed"},
    })
    assert watcher.status == "failed"
    assert "CUDA probe failed" in watcher.render()
