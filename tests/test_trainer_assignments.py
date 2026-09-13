"""Tests for the durable Heart-owned assignment lifecycle store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.field import canonical_sha256
from runtime.trainer.assignments import (
    ASSIGNMENT_SCHEMA,
    EVENT_SCHEMA,
    AssignmentAttempt,
    AssignmentKind,
    AssignmentLease,
    AssignmentStatus,
    AssignmentStore,
    AssignmentStoreError,
    AttemptCritique,
    AttemptOutcome,
    TrainingAssignment,
)


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


COHORT = {"core_mode": "OFFLINE_TRAINING", "core_ids": ["core-alpha", "core-beta"]}
BINDING = {"field_id": "field-1", "view_id": "view-1"}
EMISSION = {"rail": "perspective", "emission_type": "typed_delta", "digest": "e" * 64}
SOUL_LINEAGE = {"before_soul_id": "a" * 64, "after_soul_id": "b" * 64}
EVIDENCE = {"loss": 1.25, "payload_sha256": "c" * 64}


def make_store(tmp_path: Path, clock: FakeClock | None = None) -> AssignmentStore:
    return AssignmentStore(tmp_path / "assignments", clock=clock or FakeClock())


def claim(
    store: AssignmentStore,
    *,
    key: str = "claim-1",
    holder: str = "core-alpha",
    created_at: float | None = None,
    threshold: int = 3,
) -> TrainingAssignment:
    return store.create_assignment(
        kind=AssignmentKind.LEARNING,
        curriculum_ref="curriculum://abc-sequence",
        cohort_eligibility=COHORT,
        field_binding=BINDING,
        holder_core_id=holder,
        origin="supervisor:test",
        lease_seconds=60.0,
        idempotency_key=key,
        failure_threshold=threshold,
        created_at=created_at,
    )


def activate(store: AssignmentStore, record: TrainingAssignment, *, key: str = "activate-1") -> TrainingAssignment:
    return store.activate(
        record.assignment_id,
        holder_core_id=record.lease.holder_core_id,
        expected_revision=record.revision,
        idempotency_key=key,
    )


def record_attempt(
    store: AssignmentStore,
    record: TrainingAssignment,
    *,
    key: str,
    emission: dict | None = None,
    core_id: str | None = None,
) -> AssignmentAttempt:
    holder = core_id if core_id is not None else record.lease.holder_core_id  # type: ignore[union-attr]
    return store.record_attempt(
        record.assignment_id,
        core_id=holder,
        emission_ref=EMISSION if emission is None else emission,
        soul_lineage=SOUL_LINEAGE,
        parameter_generation="param-gen-0",
        optimizer_generation="opt-gen-0",
        evidence=EVIDENCE,
        expected_revision=record.revision,
        idempotency_key=key,
    )


def fail_attempt(store: AssignmentStore, record: TrainingAssignment, attempt: AssignmentAttempt, *, key: str):
    return store.record_attempt_outcome(
        attempt.attempt_id,
        outcome=AttemptOutcome.FAIL,
        expected_revision=record.revision,
        idempotency_key=key,
    )


def test_happy_path_full_lifecycle(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = claim(store)
    assert record.status is AssignmentStatus.CLAIMED
    assert record.lease is not None and record.lease.holder_core_id == "core-alpha"
    assert record.revision == 0
    assert len(record.assignment_id) == 64

    record = activate(store, record)
    assert record.status is AssignmentStatus.ACTIVE
    assert record.revision == 1

    attempt = record_attempt(store, record, key="attempt-1")
    assert attempt.attempt_index == 0
    assert attempt.outcome is AttemptOutcome.PENDING
    record = store.get_assignment(record.assignment_id)
    assert record.attempt_count == 1

    attempt, record = store.record_attempt_outcome(
        attempt.attempt_id,
        outcome=AttemptOutcome.PASS,
        expected_revision=record.revision,
        idempotency_key="outcome-1",
    )
    assert attempt.outcome is AttemptOutcome.PASS
    assert record.consecutive_failures == 0

    record = store.complete(
        record.assignment_id,
        holder_core_id="core-alpha",
        expected_revision=record.revision,
        idempotency_key="complete-1",
    )
    assert record.status is AssignmentStatus.COMPLETED
    assert record.lease is None

    events = store.events_for(record.assignment_id)
    kinds = [event["event"] for event in events]
    assert kinds == [
        "assignment_claimed",
        "assignment_activate",
        "attempt_recorded",
        "attempt_outcome",
        "assignment_complete",
    ]
    revisions = [event["to_revision"] for event in events]
    assert revisions == [0, 1, 2, 3, 4]


def test_records_are_content_addressed_and_verified(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store))
    path = store.assignments_dir / f"{record.assignment_id}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["attempt_count"] = 99
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(AssignmentStoreError, match="identity/hash mismatch"):
        store.get_assignment(record.assignment_id)


def test_tampered_journal_event_detected(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = claim(store)
    journal = store.events_dir / f"{record.assignment_id}.jsonl"
    lines = journal.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["reason"] = "forged"
    lines[0] = json.dumps(event, sort_keys=True)
    journal.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AssignmentStoreError, match="event identity mismatch"):
        store.events_for(record.assignment_id)


def test_double_claim_rejected_and_holder_exclusive(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store))
    with pytest.raises(AssignmentStoreError, match="does not hold the assignment lease"):
        store.pause(
            record.assignment_id,
            holder_core_id="core-beta",
            expected_revision=record.revision,
            idempotency_key="pause-evil",
        )
    with pytest.raises(AssignmentStoreError, match="does not hold the assignment lease"):
        store.record_attempt(
            record.assignment_id,
            core_id="core-beta",
            emission_ref=EMISSION,
            soul_lineage=SOUL_LINEAGE,
            parameter_generation="p",
            optimizer_generation="o",
            evidence=EVIDENCE,
            expected_revision=record.revision,
            idempotency_key="attempt-evil",
        )
    # Cohort membership is also enforced.
    store2 = make_store(tmp_path / "other")
    with pytest.raises(AssignmentStoreError, match="not eligible"):
        claim(store2, holder="core-outsider")


def test_stale_revision_cas_failure(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = claim(store)
    stale = record.revision  # 0
    record = activate(store, record)
    with pytest.raises(AssignmentStoreError, match="stale revision"):
        store.pause(
            record.assignment_id,
            holder_core_id="core-alpha",
            expected_revision=stale,
            idempotency_key="pause-stale",
        )
    # The failed command must not have journaled anything.
    assert [e["command"] for e in store.events_for(record.assignment_id)] == [
        "create_assignment",
        "activate",
    ]


def test_idempotent_replay_applies_once(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = claim(store)
    again = activate(store, record, key="activate-1")
    replayed = activate(store, record, key="activate-1")
    assert replayed == again
    assert replayed.revision == 1
    assert [e["command"] for e in store.events_for(record.assignment_id)].count("activate") == 1

    # Same key with a different command payload fails closed.
    with pytest.raises(AssignmentStoreError, match="different command"):
        store.pause(
            record.assignment_id,
            holder_core_id="core-alpha",
            expected_revision=replayed.revision,
            idempotency_key="activate-1",
        )


def test_lease_expiry_reclaim_preserves_attempts(tmp_path: Path) -> None:
    clock = FakeClock()
    store = make_store(tmp_path, clock)
    record = activate(store, claim(store))
    attempt = record_attempt(store, record, key="attempt-1")

    # Reclaim before expiry fails closed.
    record = store.get_assignment(record.assignment_id)
    with pytest.raises(AssignmentStoreError, match="has not expired"):
        store.reclaim_expired(
            record.assignment_id,
            expected_revision=record.revision,
            idempotency_key="reclaim-early",
        )

    clock.advance(120.0)  # lease was 60s
    record = store.get_assignment(record.assignment_id)
    record = store.reclaim_expired(
        record.assignment_id,
        expected_revision=record.revision,
        idempotency_key="reclaim-1",
    )
    assert record.status is AssignmentStatus.PAUSED
    assert record.lease is None

    # Attempts survive reclaim untouched; timeout recorded no outcome.
    attempts = store.attempts_for(record.assignment_id)
    assert len(attempts) == 1
    assert attempts[0].attempt_id == attempt.attempt_id
    assert attempts[0].outcome is AttemptOutcome.PENDING

    # The reclaimed assignment no longer accepts attempts without a lease.
    with pytest.raises(AssignmentStoreError, match="no active lease"):
        record_attempt(store, record, key="attempt-2", core_id="core-alpha")

    # A fresh claim resumes continuity on the same assignment.
    record = store.resume(
        record.assignment_id,
        core_id="core-beta",
        expected_revision=record.revision,
        idempotency_key="resume-1",
        lease_seconds=60.0,
    )
    assert record.status is AssignmentStatus.ACTIVE
    assert record.lease is not None and record.lease.holder_core_id == "core-beta"
    next_attempt = record_attempt(store, record, key="attempt-2")
    assert next_attempt.attempt_index == 1


def test_pause_resume_continuity(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store))
    attempt = record_attempt(store, record, key="attempt-1")
    record = store.get_assignment(record.assignment_id)
    record = store.pause(
        record.assignment_id,
        holder_core_id="core-alpha",
        expected_revision=record.revision,
        idempotency_key="pause-1",
        reason="heartbeat_lost",
    )
    assert record.status is AssignmentStatus.PAUSED
    assert record.lease is None

    record = store.resume(
        record.assignment_id,
        core_id="core-alpha",
        expected_revision=record.revision,
        idempotency_key="resume-1",
        lease_seconds=30.0,
    )
    assert record.status is AssignmentStatus.ACTIVE
    assert record.attempt_count == 1
    assert store.attempts_for(record.assignment_id)[0].attempt_id == attempt.attempt_id


def test_escalation_after_failed_series_never_fabricates_success(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store, threshold=2))

    attempt1 = record_attempt(store, record, key="attempt-1")
    record = store.get_assignment(record.assignment_id)
    _, record = fail_attempt(store, record, attempt1, key="fail-1")
    assert record.status is AssignmentStatus.ACTIVE
    assert record.consecutive_failures == 1

    attempt2 = record_attempt(store, record, key="attempt-2")
    record = store.get_assignment(record.assignment_id)
    _, record = fail_attempt(store, record, attempt2, key="fail-2")
    assert record.status is AssignmentStatus.ESCALATED
    assert record.consecutive_failures == 2
    assert record.lease is None

    # Every attempt preserved with its real outcome; nothing was auto-passed.
    attempts = store.attempts_for(record.assignment_id)
    assert [a.outcome for a in attempts] == [AttemptOutcome.FAIL, AttemptOutcome.FAIL]

    # No new attempts on an escalated assignment.
    with pytest.raises(AssignmentStoreError):
        record_attempt(store, record, key="attempt-3", core_id="core-alpha")

    # Supervisor inspection path: critique + curriculum adjustment + rollback.
    attempt, record = store.record_attempt_outcome(
        attempt2.attempt_id,
        outcome=AttemptOutcome.FAIL,
        expected_revision=record.revision,
        idempotency_key="fail-2",
    )
    critique = store.add_critique(
        attempt2.attempt_id,
        author="supervisor:jeff",
        guidance="masking window too narrow; revisit region spans",
        idempotency_key="critique-1",
    )
    assert isinstance(critique, AttemptCritique)
    assert store.critiques_for(attempt2.attempt_id)[0].critique_id == critique.critique_id

    stable_assignment_id = record.assignment_id
    record = store.adjust_curriculum(
        record.assignment_id,
        author="supervisor:jeff",
        expected_revision=record.revision,
        idempotency_key="adjust-1",
        curriculum_ref="curriculum://abc-sequence-v2",
        reason="inspection after failed series",
    )
    assert record.curriculum_ref == "curriculum://abc-sequence-v2"
    assert record.assignment_id == stable_assignment_id
    assert store.get_assignment(stable_assignment_id) == record
    assert record.status is AssignmentStatus.ESCALATED

    record = store.rollback(
        record.assignment_id,
        author="supervisor:jeff",
        expected_revision=record.revision,
        idempotency_key="rollback-1",
        parameter_generation="param-gen-0",
        optimizer_generation="opt-gen-0",
        reason="return to pre-series generations",
    )
    assert record.rollback_target == {
        "parameter_generation": "param-gen-0",
        "optimizer_generation": "opt-gen-0",
        "reason": "return to pre-series generations",
    }

    # Terminal escalation rejects further transitions.
    with pytest.raises(AssignmentStoreError, match="terminal"):
        store.escalate(
            record.assignment_id,
            author="supervisor:jeff",
            expected_revision=record.revision,
            idempotency_key="escalate-again",
            reason="reconfirm",
        )


def test_terminal_assignment_rejects_transitions(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store))
    record = store.complete(
        record.assignment_id,
        holder_core_id="core-alpha",
        expected_revision=record.revision,
        idempotency_key="complete-1",
    )
    assert record.status is AssignmentStatus.COMPLETED
    with pytest.raises(AssignmentStoreError, match="terminal"):
        store.pause(
            record.assignment_id,
            holder_core_id="core-alpha",
            expected_revision=record.revision,
            idempotency_key="pause-done",
        )
    with pytest.raises(AssignmentStoreError, match="terminal"):
        store.adjust_curriculum(
            record.assignment_id,
            author="supervisor:jeff",
            expected_revision=record.revision,
            idempotency_key="adjust-done",
            curriculum_ref="curriculum://other",
            reason="too late",
        )


def test_counterfactual_only_does_not_reset_failure_series(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store, threshold=3))
    a1 = record_attempt(store, record, key="a1")
    record = store.get_assignment(record.assignment_id)
    _, record = fail_attempt(store, record, a1, key="f1")
    a2 = record_attempt(store, record, key="a2")
    record = store.get_assignment(record.assignment_id)
    attempt, record = store.record_attempt_outcome(
        a2.attempt_id,
        outcome=AttemptOutcome.COUNTERFACTUAL_ONLY,
        expected_revision=record.revision,
        idempotency_key="cf1",
    )
    # Counterfactual-only is evidence, not a pass: consecutive failures stay.
    assert record.consecutive_failures == 1
    a3 = record_attempt(store, record, key="a3")
    record = store.get_assignment(record.assignment_id)
    _, record = fail_attempt(store, record, a3, key="f2")
    assert record.consecutive_failures == 2
    assert record.status is AssignmentStatus.ACTIVE
    a4 = record_attempt(store, record, key="a4")
    record = store.get_assignment(record.assignment_id)
    _, record = fail_attempt(store, record, a4, key="f3")
    assert record.status is AssignmentStatus.ESCALATED


def test_outcome_idempotency_and_fail_closed_double_outcome(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store))
    attempt = record_attempt(store, record, key="attempt-1")
    record = store.get_assignment(record.assignment_id)

    again, record_again = store.record_attempt_outcome(
        attempt.attempt_id,
        outcome=AttemptOutcome.FAIL,
        expected_revision=record.revision,
        idempotency_key="fail-1",
    )
    replay, record_replay = store.record_attempt_outcome(
        attempt.attempt_id,
        outcome=AttemptOutcome.FAIL,
        expected_revision=record.revision,
        idempotency_key="fail-1",
    )
    assert replay == again
    assert record_replay.revision == record_again.revision

    with pytest.raises(AssignmentStoreError, match="already has outcome"):
        store.record_attempt_outcome(
            attempt.attempt_id,
            outcome=AttemptOutcome.PASS,
            expected_revision=record_replay.revision,
            idempotency_key="pass-late",
        )


def test_preemption_by_supervisor(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store))
    record = store.preempt(
        record.assignment_id,
        author="supervisor:jeff",
        expected_revision=record.revision,
        idempotency_key="preempt-1",
        reason="kaggle preemption drill",
    )
    assert record.status is AssignmentStatus.PAUSED
    assert record.lease is None


def test_renew_lease_extends_expiry(tmp_path: Path) -> None:
    clock = FakeClock()
    store = make_store(tmp_path, clock)
    record = activate(store, claim(store))
    first_expiry = record.lease.expires_at
    record = store.renew_lease(
        record.assignment_id,
        holder_core_id="core-alpha",
        expected_revision=record.revision,
        idempotency_key="renew-1",
        lease_seconds=300.0,
    )
    assert record.lease.expires_at > first_expiry
    assert record.lease.lease_epoch == 2
    clock.advance(500.0)
    record = store.reclaim_expired(
        record.assignment_id,
        expected_revision=record.revision,
        idempotency_key="reclaim-1",
    )
    assert record.status is AssignmentStatus.PAUSED


def test_crash_between_journal_and_record_write_finalized_on_retry(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store))

    original = store._write_assignment_atomic

    def crash_once(assignm: TrainingAssignment) -> None:
        if assignm.revision == 2:
            raise RuntimeError("simulated crash before record write")
        original(assignm)

    store._write_assignment_atomic = crash_once  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated crash"):
        store.pause(
            record.assignment_id,
            holder_core_id="core-alpha",
            expected_revision=record.revision,
            idempotency_key="pause-1",
        )
    store._write_assignment_atomic = original  # type: ignore[method-assign]

    # The journal holds the durable intent; re-issuing finalizes it.
    finalized = store.pause(
        record.assignment_id,
        holder_core_id="core-alpha",
        expected_revision=record.revision,
        idempotency_key="pause-1",
    )
    assert finalized.status is AssignmentStatus.PAUSED
    assert finalized.revision == 2
    assert store.get_assignment(record.assignment_id) == finalized
    pause_events = [e for e in store.events_for(record.assignment_id) if e["command"] == "pause"]
    assert len(pause_events) == 1


def test_recover_finalizes_orphaned_tmp_and_intents(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store))
    attempt = record_attempt(store, record, key="attempt-1")

    # 1. A valid temp file whose os.replace never ran is promoted.
    record = store.get_assignment(record.assignment_id)
    candidate = store.pause(
        record.assignment_id,
        holder_core_id="core-alpha",
        expected_revision=record.revision,
        idempotency_key="pause-direct",
    )
    target = store.assignments_dir / f"{candidate.assignment_id}.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    target.unlink()
    tmp = target.with_name(target.name + ".9999.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # 2. A truncated temp file is removed, never promoted.
    junk = store.attempts_dir / f"{attempt.attempt_id}.json.9999.tmp"
    junk.write_text('{"schema": "axon-trainer-assignment-attemp', encoding="utf-8")

    report = store.recover()
    assert target.is_file()
    assert report.recovered_tmp_files == (tmp.name,)
    assert report.removed_tmp_files == (junk.name,)
    assert report.assignments == 1
    assert report.attempts == 1
    assert report.events == 4

    # Recovery is idempotent: a second pass changes nothing.
    again = store.recover()
    assert again.recovered_tmp_files == ()
    assert again.finalized_assignments == ()
    assert store.get_assignment(record.assignment_id).status is AssignmentStatus.PAUSED


def test_recover_finalizes_journaled_intent_when_record_missing(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store))
    record = store.pause(
        record.assignment_id,
        holder_core_id="core-alpha",
        expected_revision=record.revision,
        idempotency_key="pause-1",
    )
    path = store.assignments_dir / f"{record.assignment_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()
    report = store.recover()
    assert report.finalized_assignments == (record.assignment_id,)
    assert AssignmentStore(tmp_path / "assignments").get_assignment(record.assignment_id).to_canonical_dict() == payload


def test_deterministic_ids_under_fixed_clock(tmp_path: Path) -> None:
    clock_a, clock_b = FakeClock(500.0), FakeClock(500.0)
    store_a = AssignmentStore(tmp_path / "a", clock=clock_a)
    store_b = AssignmentStore(tmp_path / "b", clock=clock_b)
    rec_a = claim(store_a, created_at=500.0)
    rec_b = claim(store_b, created_at=500.0)
    assert rec_a.assignment_id == rec_b.assignment_id
    assert rec_a.record_hash == rec_b.record_hash
    body_a = (store_a.assignments_dir / f"{rec_a.assignment_id}.json").read_bytes()
    body_b = (store_b.assignments_dir / f"{rec_b.assignment_id}.json").read_bytes()
    assert body_a == body_b


def test_assignment_identity_binds_immutable_fields(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = claim(store, created_at=100.0, key="c1")
    second = claim(store, created_at=200.0, key="c2")
    assert first.assignment_id != second.assignment_id
    identity = first._identity_dict()
    assert canonical_sha256(identity) == first.assignment_id
    loaded = store.get_assignment(first.assignment_id)
    assert loaded == first
    assert loaded.to_canonical_dict()["schema"] == ASSIGNMENT_SCHEMA


def test_supervisor_actions_require_keys_and_revision(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = activate(store, claim(store))
    with pytest.raises(ValueError):
        store.adjust_curriculum(
            record.assignment_id,
            author="supervisor:jeff",
            expected_revision=record.revision,
            idempotency_key="adjust-empty",
            reason="no-op",
        )
    with pytest.raises(AssignmentStoreError, match="stale revision"):
        store.rollback(
            record.assignment_id,
            author="supervisor:jeff",
            expected_revision=99,
            idempotency_key="rollback-stale",
            parameter_generation="p",
            optimizer_generation="o",
            reason="stale",
        )


def test_lease_validation_and_serialization() -> None:
    lease = AssignmentLease(holder_core_id="core-alpha", issued_at=10.0, expires_at=70.0, lease_epoch=1)
    assert AssignmentLease.from_dict(lease.to_dict()) == lease
    assert not lease.expired(69.9)
    assert lease.expired(70.0)
    with pytest.raises(ValueError):
        AssignmentLease(holder_core_id="core-alpha", issued_at=10.0, expires_at=10.0, lease_epoch=1)
    with pytest.raises(ValueError):
        AssignmentLease(holder_core_id="core-alpha", issued_at=10.0, expires_at=70.0, lease_epoch=0)


def test_event_schema_and_command_hash_binding(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = claim(store)
    (event,) = store.events_for(record.assignment_id)
    assert event["schema"] == EVENT_SCHEMA
    assert event["command_sha256"] == canonical_sha256(
        {
            "command": "create_assignment",
            "params": {
                "kind": "learning",
                "curriculum_ref": "curriculum://abc-sequence",
                "cohort_eligibility": COHORT,
                "field_binding": BINDING,
                "holder_core_id": "core-alpha",
                "origin": "supervisor:test",
                "lease_seconds": 60.0,
                "failure_threshold": 3,
                "created_at": 1000.0,
            },
        }
    )


def test_active_classmethod_uses_injected_state_root(tmp_path: Path) -> None:
    store = AssignmentStore.active(tmp_path, clock=FakeClock())
    assert store.root == (tmp_path / "training" / "assignments").resolve()
    record = claim(store)
    assert store.get_assignment(record.assignment_id) == record
