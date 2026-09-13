"""Scope B3 tests: end-to-end training session + mode transition (handoff 9-10).

Every fixture runs the REAL CompleteField64D model through the REAL
assignment store, training-attention view, rail-auth envelopes, supervisory
gates, SoulStore, and step-bundle coordinator.  Nothing is simulated:
worker metrics are evidence, gate recomputation is authority, and every
outcome is committed through idempotently keyed store commands.

These tests prove the handoff's minimum acceptance gates before Kaggle
learning, including the stop/restart/reclaim resume and the
training->reasoning->training mode round trip with a byte-exact resume
probe.

Note on paths: content-addressed accepted-step artifacts carry 128+ hex
filename characters; on Windows the system temp directory keeps the state
root short enough to stay under the classic 260-character path limit (the
production ``D:\\Axon\\State`` root is far shorter still).
"""
from __future__ import annotations

import hashlib
import io
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from runtime.field import (
    LogicalRegion,
    RegionMaskPolicy,
    SharedFieldSnapshot,
    TRAINING_REGIONS,
    canonical_region_order,
    canonical_sha256,
)
from runtime.field.compiler_d64 import D64FieldCompiler
from runtime.field.training_view import TrainingHistoryWindow
from runtime.heart import (
    AuthorityGrant,
    CoreBinding,
    CoreDescriptor,
    CoreRegistry,
    CoreStatus,
    InvalidModeTransitionError,
    OperatorCoreGrant,
    UnknownCoreError,
    UngrantedCoreRegistrationError,
)
from runtime.heart.registry import InvalidCoreBindingError
from runtime.soul import SoulStore, SoulTemperature
from runtime.trainer.assignments import (
    AssignmentKind,
    AssignmentStatus,
    AssignmentStore,
    AssignmentStoreError,
    AttemptOutcome,
)
from runtime.trainer.local_worker import (
    EOS_TOKEN_MARKER,
    LOCAL_TRAINING_ARCHITECTURE,
    LocalTrainingWorkerError,
)
from runtime.trainer.soul_candidates import CandidateSoulWorkspace
from runtime.trainer.step_bundle import CandidateStepBundleCoordinator
from runtime.trainer.supervisory_gates import (
    PROBE_RAIL,
    PROBE_RAIL_CONTENT,
    PROBE_SOUL,
    GateDecision,
    SubmissionKind,
    WorkerEvidenceBundle,
    assert_resume_continuity,
)
from runtime.trainer.training_session import (
    TrainingSession,
    TrainingSessionConfig,
    TrainingSessionError,
)
from runtime.trainer.store import TrainerStateStore
from substrate import get_letter_bank
from training.complete_field_64d import (
    CompleteField64D,
    ReaderConfig,
    sequence_cross_entropy,
    teacher_char_accuracy,
)

_CORE = "tc"
_SECRET = "k" * 48
_ENVIRON = {"AXON_RAIL_SECRET": _SECRET}
_T0 = 1_000.0
_ASSIGNMENT_TEXT = "ABC? Recite the alphabet forward from 'a'."
_IMPOSSIBLE_TEXT = "a" * 28


@pytest.fixture
def state_root():
    tmp = tempfile.mkdtemp(prefix="b3")
    yield Path(tmp) / "S"
    shutil.rmtree(tmp, ignore_errors=True)


def _snapshot(
    text: str = _ASSIGNMENT_TEXT, tick_id: int = 7
) -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.TRAINER_INSTRUCTIONS: text,
            LogicalRegion.TRAINING_RESPONSES: "",
        },
        tick_id=tick_id,
        source="test-trainer",
        provenance="heart:committed-assignment",
    )


def _registry() -> tuple[CoreRegistry, OperatorCoreGrant]:
    registry = CoreRegistry(
        (
            CoreDescriptor(
                core_id=_CORE,
                d_model=64,
                architecture_id=LOCAL_TRAINING_ARCHITECTURE,
                parameter_generation="untrained",
            ),
        )
    )
    grant = OperatorCoreGrant(core_id=_CORE, d_model=64)
    registry.issue_operator_grant(grant)
    return registry, grant


def _session(
    state_root: Path,
    registry: CoreRegistry,
    grant: OperatorCoreGrant,
    *,
    snapshot: SharedFieldSnapshot | None = None,
    kind: AssignmentKind = AssignmentKind.LEARNING,
    threshold: int = 3,
    max_attempts: int = 8,
    clock=None,
    target_text: str | None = None,
) -> TrainingSession:
    config = TrainingSessionConfig(
        core_id=_CORE,
        curriculum_ref="abc-sequence:train:0000",
        cohort_core_ids=(_CORE,),
        assignment_kind=kind,
        lease_seconds=600.0,
        failure_threshold=threshold,
        max_attempts=max_attempts,
        target_text=target_text,
    )
    return TrainingSession(
        state_root=state_root,
        registry=registry,
        snapshot=snapshot if snapshot is not None else _snapshot(),
        config=config,
        operator_grant=grant,
        clock=clock if clock is not None else (lambda: _T0),
        rail_environ=_ENVIRON,
    )


def _serialize(model: torch.nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


# --- the assignment reaches the core as exact D64 rows -----------------------


def test_assignment_reaches_core_as_exact_d64_rows(state_root) -> None:
    snapshot = _snapshot()
    registry, grant = _registry()
    session = _session(state_root, registry, grant)

    view = session.view
    # The session's view is byte-identical to an independent masked compile
    # of the unchanged canonical snapshot: same rail, same rows.
    policies = {
        region: RegionMaskPolicy("none") for region in canonical_region_order()
    }
    policies[LogicalRegion.TRAINER_INSTRUCTIONS] = RegionMaskPolicy("all")
    independent = D64FieldCompiler().compile(snapshot, region_masks=policies)

    assert view.compiled.rail_id == independent.rail_id
    assert np.array_equal(view.compiled.rows, independent.rows)
    assert view.compiled.row_count > 0
    assert view.source_field_id == snapshot.field_id
    assert view.source_tick_id == snapshot.tick_id
    assert session.objective_text == _ASSIGNMENT_TEXT

    # Every emitted lane dereferences to the canonical training regions.
    view.verify(snapshot)
    for address in view.valid_addresses:
        assert address.region in TRAINING_REGIONS
        assert snapshot.region(address.region).text[address.region_position] == address.character


# --- the first emission is a typed model output -------------------------------


def test_first_emission_is_typed_model_output_not_fixture_text(state_root) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    result = session.run(max_attempts=1)
    step = result.steps[0]
    emission = step.worker_bundle.emission

    bank_chars = set(get_letter_bank().chars[:-1])
    assert emission.token_ids
    for token, character in zip(emission.token_ids, emission.characters, strict=True):
        if token == len(bank_chars):
            assert character == EOS_TOKEN_MARKER
        else:
            assert character in bank_chars
    # The emission is NOT the fixture text: it is the model's own argmax.
    assert "".join(emission.characters) != _ASSIGNMENT_TEXT
    for key in step.worker_bundle.to_canonical_dict():
        assert key not in {"text", "prose", "answer", "narrative"}

    # The emission equals the free-running attempt-time decode.  The worker
    # re-seeds torch and runs in train mode just before that forward, so the
    # exact autoregressive result is replayable from a byte-identical fresh
    # session: provably model-derived without target-prefix leakage.
    # An eval-mode pass on the POST-step model cannot match (dropout + the
    # accepted optimizer step both move the graph).
    tmp = tempfile.mkdtemp(prefix="b3r")
    try:
        replay = _session(Path(tmp) / "S", *_registry())
        replay.worker._ensure_model(replay.view.source_schema_version)
        model = replay.worker.model
        torch.manual_seed(11)  # TrainingSessionConfig.seed default
        model.train()
        reader_state, memory, _ = model.read_compiled_with_memory(replay.view.compiled)
        attempted_text, terminated = model.decode_greedy(
            reader_state,
            head=0,
            work_units=replay.worker._emission_work_units,
            memory=memory,
        )
        token_ids = tuple(model.char_to_index[character] for character in attempted_text)
        if terminated:
            token_ids = (*token_ids, model.eos_index)
        assert token_ids == emission.token_ids
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- real loss, gradients, optimizer step, frozen controls --------------------


def test_real_loss_nonzero_gradients_and_optimizer_changes_intended_bytes(
    state_root,
) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    session.worker._ensure_model(session.view.source_schema_version)
    model = session.worker.model
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    buffers_before = {
        name: tensor.detach().to(device="cpu").contiguous().numpy().tobytes(order="C")
        for name, tensor in model.named_buffers()
    }

    result = session.run(max_attempts=1)
    step = result.steps[0]
    recomputed = step.verdict.recomputed

    # Discrete categorical loss: finite, positive, and the same graph that
    # produced it yields nonzero, finite gradients (gate-recomputed).
    assert 0.0 < recomputed.loss < 20.0
    assert recomputed.grad_norm is not None
    assert 0.0 < recomputed.grad_norm < float("inf")
    # The optimizer changed intended parameter bytes...
    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), parameters_before[name])
    ]
    assert changed
    assert step.worker_bundle.parameter_generation_after != (
        step.worker_bundle.parameter_generation_before
    )
    # ...while every frozen control buffer stays byte-identical.
    for name, tensor in model.named_buffers():
        observed = tensor.detach().to(device="cpu").contiguous().numpy().tobytes(order="C")
        assert observed == buffers_before[name], f"control buffer {name!r} changed"
    assert {"bank16", "char_lift"} <= {name for name, _ in model.named_buffers()}


# --- Soul survives a full restart and binds to the accepted checkpoint -------


def test_soul_payload_survives_full_restart_and_binds_to_checkpoint(
    state_root,
) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    result = session.run(max_attempts=2)
    last = result.steps[-1]

    # FULL process-restart simulation: every store object is rebuilt from the
    # same roots, and the assignment store runs recover().
    fresh_souls = SoulStore.active(state_root)
    head = fresh_souls.branch(_CORE).load_head()
    assert head.soul_id == last.worker_bundle.soul_id_after
    assert head.soul_id != result.steps[0].worker_bundle.soul_id_before
    assert head.generation == 2
    hot = head.layer(SoulTemperature.HOT)
    assert hot.payload_sha256 == hashlib.sha256(hot.payload).hexdigest()

    fresh_store = AssignmentStore.active(state_root, clock=lambda: _T0)
    report = fresh_store.recover()
    assert report.attempts == 2
    fresh_coordinator = CandidateStepBundleCoordinator(state_root)
    accepted = session.latest_accepted_bundle()
    assert accepted is not None
    checkpoint = fresh_coordinator.checkpoint_for_bundle(accepted)
    payload = TrainerStateStore.active(state_root=state_root).load_verified_candidate_checkpoint(
        checkpoint
    )
    # The accepted checkpoint binds the exact live parameter bytes.
    assert _serialize(session.worker.model) == _serialize_state(payload["module_state_dict"])
    assert payload["optimizer_state_dict"] is not None
    # The accepted Soul boundary is the candidate branch head.
    branch = CandidateSoulWorkspace(state_root).branch(
        accepted.candidate_generation_id, _CORE
    )
    assert branch.load_head().soul_id == accepted.after_soul_id


def _serialize_state(state_dict) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()


# --- Heart commits exactly once; tampering fails closed -----------------------


def _honest_submission(
    session: TrainingSession,
    record,
    **overrides,
) -> WorkerEvidenceBundle:
    view = session.view
    model = session.worker.model
    model.eval()
    with torch.no_grad():
        reader_state, memory, _ = model.read_compiled_with_memory(view.compiled)
        logits, targets = model.decode_teacher(
            reader_state, session.objective_text, head=0, memory=memory
        )
    loss = float(sequence_cross_entropy(logits, targets).item())
    accuracy = float(teacher_char_accuracy(logits, targets))
    params = {
        "core_id": _CORE,
        "assignment_id": record.assignment_id,
        "view_id": view.view_id,
        "field_id": view.source_field_id,
        "tick_id": view.source_tick_id,
        "submission_kind": SubmissionKind.LEARNING_RESULT,
        "parameter_bytes": _serialize(model),
        "claimed_loss": loss,
        "claimed_grad_norm": None,
        "claimed_accuracy": accuracy,
        "prior_parameter_sha256": "p" * 64,
        "optimizer_generation": "opt",
    }
    params.update(overrides)
    bundle = WorkerEvidenceBundle(**params)
    if params.get("envelope", "seal") == "seal":
        from runtime.heart.rail_auth import seal_envelope

        import dataclasses

        bundle = dataclasses.replace(
            bundle,
            envelope=seal_envelope(
                core_identity=params["core_id"],
                assignment_id=params["assignment_id"],
                payload=bundle.sealed_payload_bytes,
                ttl_seconds=300,
                now=int(_T0),
                environ=_ENVIRON,
            ),
        )
    return bundle


def test_heart_commits_accepted_response_exactly_once(state_root) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    result = session.run(max_attempts=1)
    step = result.steps[0]
    assert step.verdict.decision is GateDecision.ACCEPTED
    attempt_id = step.attempt.attempt_id
    store = session.assignment_store
    record = store.get_assignment(result.assignment.assignment_id)

    # Replaying the identical keyed command returns the original outcome and
    # never applies twice.
    replayed, _ = store.record_attempt_outcome(
        attempt_id,
        outcome=step.attempt.outcome,
        evidence=dict(step.attempt.evidence),
        expected_revision=record.revision,
        idempotency_key=f"training-session-outcome:{attempt_id}",
    )
    assert replayed == store.get_attempt(attempt_id)
    assert store.get_assignment(record.assignment_id).attempt_count == 1
    assert len(store.attempts_for(record.assignment_id)) == 1
    # A second, differently keyed outcome for the same attempt fails closed.
    with pytest.raises(AssignmentStoreError, match="already has outcome"):
        store.record_attempt_outcome(
            attempt_id,
            outcome=AttemptOutcome.FAIL,
            evidence=dict(step.attempt.evidence),
            expected_revision=store.get_assignment(record.assignment_id).revision,
            idempotency_key=f"double-outcome:{attempt_id}",
        )
    assert store.get_attempt(attempt_id).outcome is AttemptOutcome.PASS


def test_replayed_stale_wrong_core_wrong_tick_wrong_view_and_unauthenticated_fail_closed(
    state_root,
) -> None:
    from runtime.heart.rail_auth import seal_envelope

    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    result = session.run(max_attempts=1)
    record = session.assignment_store.get_assignment(result.assignment.assignment_id)
    view = session.view

    def evaluate(bundle, **overrides):
        params = {
            "target_text": session.objective_text,
            "store": None,
            "snapshot": _snapshot(),
            "nonce_cache": session.nonce_cache,
        }
        params.update(overrides)
        return session.gate.evaluate(bundle, record, view, **params)

    # Replayed envelope: identical sealed envelope evaluated twice.
    bundle = _honest_submission(session, record)
    first = evaluate(bundle)
    assert first.decision in (
        GateDecision.ACCEPTED,
        GateDecision.PAUSED,
    )  # a valid sealed submission verifies once
    replayed = evaluate(bundle)
    assert replayed.decision is GateDecision.REJECTED
    assert any("replay" in reason for reason in replayed.reasons)

    # Unauthenticated: no envelope at all.
    unauthenticated = _honest_submission(session, record, envelope=None)
    verdict = evaluate(unauthenticated)
    assert verdict.decision is GateDecision.REJECTED
    assert any("unauthenticated" in reason for reason in verdict.reasons)

    # Wrong core.
    wrong_core = _honest_submission(session, record, core_id="core-impostor")
    wrong_core = _reseal(wrong_core, core_id="core-impostor")
    verdict = evaluate(wrong_core)
    assert verdict.decision is GateDecision.REJECTED
    assert any("wrong-core" in reason for reason in verdict.reasons)

    # Wrong tick.
    wrong_tick = _honest_submission(session, record, tick_id=view.source_tick_id + 1)
    wrong_tick = _reseal(wrong_tick, tick_id=view.source_tick_id + 1)
    verdict = evaluate(wrong_tick)
    assert verdict.decision is GateDecision.REJECTED
    assert any("wrong-tick" in reason for reason in verdict.reasons)

    # Wrong view.
    wrong_view = _honest_submission(session, record, view_id="v" * 64)
    wrong_view = _reseal(wrong_view, view_id="v" * 64)
    verdict = evaluate(wrong_view)
    assert verdict.decision is GateDecision.REJECTED
    assert any("wrong-view" in reason for reason in verdict.reasons)

    # Stale view: the issued view does not verify against a newer snapshot.
    stale = _honest_submission(session, record)
    stale = _reseal(stale)
    newer = _snapshot(tick_id=view.source_tick_id + 5)
    verdict = evaluate(stale, snapshot=newer)
    assert verdict.decision is GateDecision.REJECTED
    assert any("stale-view" in reason for reason in verdict.reasons)

    # Expired envelope.
    expired = _honest_submission(session, record)
    expired = _reseal(expired, now=int(_T0))
    verdict = evaluate(expired, now=_T0 + 10_000)
    assert verdict.decision is GateDecision.REJECTED
    assert any("expired" in reason for reason in verdict.reasons)


def _reseal(bundle: WorkerEvidenceBundle, **overrides) -> WorkerEvidenceBundle:
    from dataclasses import replace as dc_replace

    from runtime.heart.rail_auth import seal_envelope

    core_id = overrides.get("core_id", bundle.core_id)
    assignment_id = overrides.get("assignment_id", bundle.assignment_id)
    now = overrides.get("now", int(_T0))
    return dc_replace(
        bundle,
        envelope=seal_envelope(
            core_identity=core_id,
            assignment_id=assignment_id,
            payload=bundle.sealed_payload_bytes,
            ttl_seconds=300,
            now=now,
            environ=_ENVIRON,
        ),
    )


# --- learning curve -----------------------------------------------------------


def test_loss_falls_and_accuracy_exceeds_constant_output_floor(state_root) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    result = session.run(max_attempts=5)
    assert len(result.steps) == 5
    assert all(step.verdict.decision is GateDecision.ACCEPTED for step in result.steps)
    losses = [step.verdict.recomputed.loss for step in result.steps]
    accuracies = [step.verdict.recomputed.accuracy for step in result.steps]
    floors = [step.verdict.accuracy_floor for step in result.steps]
    assert losses[-1] < losses[0], f"loss did not fall: {losses}"
    assert all(losses[index] <= losses[index - 1] for index in range(1, len(losses)))
    assert accuracies[-1] > floors[-1], (
        f"accuracy {accuracies[-1]} did not exceed the constant-output floor {floors[-1]}"
    )
    assert accuracies[-1] > accuracies[0]


# --- counterfactual probes -----------------------------------------------------


def test_counterfactual_probes_prove_each_claimed_mechanism(state_root) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    session.run(max_attempts=1)
    unrelated = _snapshot(
        "A completely different assignment about zebras, quarks, and violins.",
        tick_id=11,
    )
    reports = session.run_counterfactual_probes(
        (PROBE_RAIL, PROBE_RAIL_CONTENT, PROBE_SOUL, "unsupported_probe"),
        probe_snapshot=unrelated,
    )
    by_name = {report.name: report for report in reports}
    # The core genuinely attends the rail, THIS rail's content, and its Soul.
    assert by_name[PROBE_RAIL].used is True
    assert by_name[PROBE_RAIL].max_abs_delta > 0.0
    assert by_name[PROBE_RAIL_CONTENT].used is True
    assert by_name[PROBE_SOUL].used is True
    assert by_name["unsupported_probe"].used is False


# --- failed series escalates, preserving every attempt -------------------------


def test_impossible_target_series_escalates_preserving_every_attempt(
    state_root,
) -> None:
    # An EVALUATION assignment never steps the optimizer, so an unreachable
    # constant-output floor (single repeated character) honestly fails every
    # attempt: no competence can be fabricated, no attempt is deleted.
    registry, grant = _registry()
    session = _session(
        state_root,
        registry,
        grant,
        snapshot=_snapshot(_IMPOSSIBLE_TEXT),
        kind=AssignmentKind.EVALUATION,
        threshold=3,
        max_attempts=6,
    )
    result = session.run()
    assert result.stopped_reason == "escalated"
    assert len(result.steps) == 3
    attempts = session.assignment_store.attempts_for(result.assignment.assignment_id)
    assert len(attempts) == 3
    assert [attempt.attempt_index for attempt in attempts] == [0, 1, 2]
    for attempt in attempts:
        assert attempt.outcome is AttemptOutcome.FAIL
        assert attempt.evidence["decision"] == GateDecision.REJECTED.value
        assert "below-floor" in " ".join(attempt.evidence["reasons"])
        assert attempt.evidence["metrics"]["loss"] > 0.0
    record = result.assignment
    assert record.status is AssignmentStatus.ESCALATED
    assert record.consecutive_failures == 3
    assert record.lease is None
    # Supervisor inspection is durable and bound to the exact final attempt.
    critiques = session.assignment_store.critiques_for(attempts[-1].attempt_id)
    assert len(critiques) == 1
    assert "supervisor inspection" in critiques[0].guidance
    # No accepted bundle and no fabricated landmark exists.
    assert session.latest_accepted_bundle() is None
    # The frozen model never learned (evaluation kind): parameters unchanged.
    generations = {attempt.parameter_generation for attempt in attempts}
    assert len(generations) == 1


# --- stop / restart / reclaim resumes exact lineage ----------------------------


def test_preemption_reclaim_resumes_exact_lineage_without_duplicate_attempts(
    state_root,
) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant, max_attempts=8)
    first = session.run(max_attempts=2)
    assignment_id = first.assignment.assignment_id
    del session  # kill the session mid-flight

    # The lease expires before anything reconnects (Kaggle preemption).
    late_clock = lambda: _T0 + 10_000.0
    revived = _session(state_root, registry, grant, max_attempts=8, clock=late_clock)
    second = revived.run(assignment_id=assignment_id, max_attempts=1)
    assert second.stopped_reason == "max_attempts"

    store = revived.assignment_store
    record = store.get_assignment(assignment_id)
    assert record.status is AssignmentStatus.ACTIVE
    attempts = store.attempts_for(assignment_id)
    # No duplicated attempts; indices resume exactly where they stopped.
    assert [attempt.attempt_index for attempt in attempts] == [0, 1, 2]
    assert_resume_continuity(attempts)
    # Exact parameter lineage across the preemption boundary.
    for predecessor, successor in zip(attempts, attempts[1:]):
        assert successor.evidence["prior_parameter_sha256"] == (
            predecessor.evidence["parameter_sha256"]
        )
        assert successor.soul_lineage["soul_id_before"] == (
            predecessor.soul_lineage["soul_id_after"]
        )
    # The resumed worker holds exactly the last ACCEPTED parameter/optimizer
    # bytes: rehydration from the verified checkpoint is byte-exact.
    accepted = revived.latest_accepted_bundle()
    checkpoint = CandidateStepBundleCoordinator(state_root).checkpoint_for_bundle(
        accepted
    )
    payload = TrainerStateStore.active(state_root=state_root).load_verified_candidate_checkpoint(
        checkpoint
    )
    assert _serialize(revived.worker.model) == _serialize_state(
        payload["module_state_dict"]
    )
    # Loss keeps falling after the resume.
    all_losses = [step.verdict.recomputed.loss for step in (*first.steps, *second.steps)]
    assert all_losses[-1] < all_losses[0]


def test_rejected_learning_attempt_resumes_exact_wip_lineage(state_root) -> None:
    registry, grant = _registry()
    session = _session(
        state_root,
        registry,
        grant,
        threshold=5,
        target_text="a",
    )
    first = session.run(max_attempts=1)
    assignment_id = first.assignment.assignment_id
    assert first.steps[0].verdict.decision is GateDecision.REJECTED
    assert session.latest_accepted_bundle() is None
    first_attempt = first.steps[0].attempt
    del session

    def late_clock() -> float:
        return _T0 + 10_000.0

    revived = _session(
        state_root,
        registry,
        grant,
        threshold=5,
        clock=late_clock,
        target_text="a",
    )
    second = revived.run(assignment_id=assignment_id, max_attempts=1)
    assert second.steps[0].verdict.decision is GateDecision.REJECTED

    attempts = revived.assignment_store.attempts_for(assignment_id)
    assert [attempt.attempt_index for attempt in attempts] == [0, 1]
    assert_resume_continuity(attempts)
    assert attempts[1].evidence["prior_parameter_sha256"] == (
        first_attempt.evidence["parameter_sha256"]
    )
    assert attempts[1].soul_lineage["soul_id_before"] == (
        first_attempt.soul_lineage["soul_id_after"]
    )


# --- training -> reasoning -> training preserves identity ----------------------


def test_mode_round_trip_preserves_identity_and_passes_byte_exact_resume_probe(
    state_root,
) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    result = session.run(max_attempts=3)
    assignment_id = result.assignment.assignment_id

    probe_before = session.probe_output_sha256()
    accepted = session.latest_accepted_bundle()
    binding = session.exit_to_reasoning()

    # Reasoning mode restored with the exact accepted generations.
    assert binding.mode is CoreStatus.ACTIVE
    assert registry.get(_CORE).status is CoreStatus.ACTIVE
    assert len(registry.active(64)) == 1
    assert binding.parameter_generation == accepted.candidate_generation_id
    assert binding.soul_id == accepted.after_soul_id
    assert binding.architecture_id == LOCAL_TRAINING_ARCHITECTURE

    # Landmarks record the meaningful points.  Each accepted step's landmark
    # lives under its own candidate-generation dir ("acceptance-gates-met" for
    # the first accepted step, "accepted-step:N" afterwards), so collect
    # across every generation.
    candidate_root = state_root / "training" / "trainer" / "candidates" / _CORE
    labels = {
        __import__("json").loads(path.read_text(encoding="utf-8"))["label"]
        for path in candidate_root.glob("*/accepted_steps/landmarks/*.json")
    }
    assert "acceptance-gates-met" in labels
    assert "mode:reasoning-restored" in labels

    # Back into training on the same assignment with a FULLY fresh session
    # (fresh store objects, recover(), rehydration from the checkpoint).
    fresh = _session(state_root, registry, grant)
    fresh.resume_lineage(assignment_id)
    probe_after = fresh.probe_output_sha256()
    assert probe_after == probe_before
    # The durable Soul binds the accepted boundary across the round trip.
    head = SoulStore.active(state_root).branch(_CORE).load_head()
    last = result.steps[-1]
    assert head.soul_id == last.worker_bundle.soul_id_after
    # Training continues exactly: the next attempt is index 3.
    continued = fresh.run(assignment_id=assignment_id, max_attempts=1)
    attempts = fresh.assignment_store.attempts_for(assignment_id)
    assert [attempt.attempt_index for attempt in attempts] == [0, 1, 2, 3]
    assert continued.steps[0].attempt.attempt_index == 3


# --- the registry exit path fails closed ---------------------------------------


def test_exit_offline_training_fails_closed_on_every_invalid_combination(
    state_root,
) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    session.run(max_attempts=1)
    accepted = session.latest_accepted_bundle()
    good = SimpleNamespace(
        core_id=_CORE,
        bundle_id=accepted.bundle_id,
        after_soul_id=accepted.after_soul_id,
    )

    # A valid exit works and is mode-symmetric.
    binding = registry.exit_offline_training(
        _CORE, grant=AuthorityGrant.trainer(), accepted_bundle=good
    )
    assert binding.mode is CoreStatus.ACTIVE
    # Exiting again (not in training) fails closed.
    with pytest.raises(InvalidModeTransitionError, match="not bound in offline training"):
        registry.exit_offline_training(
            _CORE, grant=AuthorityGrant.trainer(), accepted_bundle=good
        )

    # Rebuild a registry stuck in training for the remaining cases.
    registry2, grant2 = _registry()
    session2 = _session(state_root / "other", registry2, grant2)
    session2.run(max_attempts=1)
    bound = registry2.binding(_CORE)
    assert bound.mode is CoreStatus.OFFLINE_TRAINING

    with pytest.raises(UnknownCoreError):
        registry2.exit_offline_training(
            "core-phantom",
            grant=AuthorityGrant.trainer(),
            accepted_bundle=good,
        )
    for grant_obj in (
        AuthorityGrant.core(),
        AuthorityGrant.consolidator(),
        AuthorityGrant.dormant_valve(),
        AuthorityGrant.identity_steward(),
        AuthorityGrant.ingress("user"),
    ):
        with pytest.raises(InvalidModeTransitionError, match="trainer authority"):
            registry2.exit_offline_training(
                _CORE, grant=grant_obj, accepted_bundle=good
            )
    with pytest.raises(TypeError):
        registry2.exit_offline_training(
            _CORE, grant="trainer", accepted_bundle=good  # type: ignore[arg-type]
        )
    # Missing bundle.
    with pytest.raises(InvalidModeTransitionError, match="requires an accepted bundle"):
        registry2.exit_offline_training(
            _CORE, grant=AuthorityGrant.trainer(), accepted_bundle=None
        )
    # Malformed bundle records fail closed.
    for malformed in ("bundle", SimpleNamespace(core_id=_CORE), object()):
        with pytest.raises(InvalidModeTransitionError):
            registry2.exit_offline_training(
                _CORE, grant=AuthorityGrant.trainer(), accepted_bundle=malformed
            )
    # Foreign-core bundle.
    foreign = SimpleNamespace(
        core_id="core-impostor",
        bundle_id="a" * 64,
        after_soul_id=bound.soul_id,
    )
    with pytest.raises(InvalidModeTransitionError, match="different core"):
        registry2.exit_offline_training(
            _CORE, grant=AuthorityGrant.trainer(), accepted_bundle=foreign
        )
    # Stale bundle: terminal Soul predates the binding's current generation.
    stale = SimpleNamespace(
        core_id=_CORE,
        bundle_id="b" * 64,
        after_soul_id="c" * 64,
    )
    with pytest.raises(InvalidModeTransitionError, match="stale or mismatched"):
        registry2.exit_offline_training(
            _CORE, grant=AuthorityGrant.trainer(), accepted_bundle=stale
        )


def test_session_refuses_exit_without_an_accepted_boundary(state_root) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    with pytest.raises(TrainingSessionError, match="accepted bundle"):
        session.exit_to_reasoning()


# --- the narrow registry rebind path --------------------------------------------


def test_rebind_core_advances_generations_and_fails_closed() -> None:
    registry, grant = _registry()
    binding = registry.bind_core(
        CoreBinding(
            core_id=_CORE,
            architecture_id=LOCAL_TRAINING_ARCHITECTURE,
            parameter_generation="untrained",
            optimizer_generation="optimizer-init",
            soul_id="a" * 64,
        ),
        grant=grant,
    )
    advanced = CoreBinding(
        core_id=_CORE,
        architecture_id=LOCAL_TRAINING_ARCHITECTURE,
        parameter_generation="b" * 64,
        optimizer_generation="c" * 64,
        soul_id="d" * 64,
    )
    rebound = registry.rebind_core(advanced, grant=grant)
    assert rebound is advanced
    assert registry.binding(_CORE) is advanced
    # Descriptor and binding moved together.
    assert registry.get(_CORE).parameter_generation == "b" * 64

    # Architecture is immutable across a rebind.
    with pytest.raises(InvalidCoreBindingError, match="architecture"):
        registry.rebind_core(
            CoreBinding(
                core_id=_CORE,
                architecture_id="foreign-arch",
                parameter_generation="e" * 64,
                optimizer_generation="c" * 64,
                soul_id="d" * 64,
            ),
            grant=grant,
        )
    # A rebind cannot change mode; mode belongs to the transitions.
    registry.enter_offline_training(
        _CORE, grant=AuthorityGrant.trainer(), claim="claim-1"
    )
    with pytest.raises(InvalidModeTransitionError, match="cannot change mode"):
        registry.rebind_core(
            CoreBinding(
                core_id=_CORE,
                architecture_id=LOCAL_TRAINING_ARCHITECTURE,
                parameter_generation="f" * 64,
                optimizer_generation="c" * 64,
                soul_id="d" * 64,
                mode=CoreStatus.ACTIVE,
            ),
            grant=grant,
        )
    # A phantom core, an unknown grant, or a non-grant fails closed.
    with pytest.raises(UnknownCoreError):
        registry.rebind_core(
            CoreBinding(
                core_id="core-phantom",
                architecture_id=LOCAL_TRAINING_ARCHITECTURE,
                parameter_generation="b" * 64,
                optimizer_generation="c" * 64,
                soul_id="d" * 64,
            ),
            grant=grant,
        )
    stranger = OperatorCoreGrant(core_id=_CORE, d_model=64, issued_by="impostor")
    with pytest.raises(UngrantedCoreRegistrationError):
        registry.rebind_core(advanced, grant=stranger)
    with pytest.raises(TypeError):
        registry.rebind_core(advanced, grant="grant")  # type: ignore[arg-type]
