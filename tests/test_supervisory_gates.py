"""Tests for the trainer-side supervisory gate engine (SCOPE B2).

Every fixture runs the REAL model (training.complete_field_64d.CompleteField64D)
over a REAL compiled training-attention view.  The "worker" fixtures execute
the actual forward/backward/optimizer; nothing is simulated.
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from runtime.field import (
    FieldSpan,
    LogicalRegion,
    RegionState,
    SharedFieldSnapshot,
)
from runtime.field.training_view import TrainingAttentionView, TrainingAttentionViewCompiler
from runtime.heart.rail_auth import RailAuthError, seal_envelope, verify_envelope
from runtime.trainer.assignments import (
    AssignmentKind,
    AssignmentStatus,
    AssignmentStore,
    AttemptOutcome,
)
from runtime.trainer.local_worker import encode_hot_payload
from runtime.trainer.supervisory_gates import (
    LOSS_RECOMPUTATION_TOLERANCE,
    PROBE_RAIL,
    PROBE_RAIL_CONTENT,
    PROBE_SOUL,
    GateDecision,
    GateEngine,
    GateEvaluationError,
    MemoryNonceCache,
    ResumeContinuityError,
    SubmissionKind,
    WorkerEvidenceBundle,
    assert_exact_generation_lineage,
    assert_no_duplicate_attempt_indices,
    assert_resume_continuity,
    constant_output_floor,
)
from training.complete_field_64d import (
    CompleteField64D,
    ReaderConfig,
    sequence_cross_entropy,
    teacher_char_accuracy,
)

_SECRET = "k" * 48
_ENVIRON = {"AXON_RAIL_SECRET": _SECRET}
_CORE = "core-alpha"
_ENVELOPE_NOW = 1_000_000
_EVAL_NOW = 1_000_030
_ASSIGNMENT_TEXT = "Predict the next character after 'ABC?'."
_TARGET_TEXT = "abcabcabcabca"


class FakeClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now


# --------------------------------------------------------------------- fixtures


def _snapshot(
    attempts: tuple[str, ...] = (),
    *,
    assignment: str = _ASSIGNMENT_TEXT,
    core_id: str = _CORE,
    tick_id: int = 7,
) -> SharedFieldSnapshot:
    responses = RegionState(
        name=LogicalRegion.TRAINING_RESPONSES,
        spans=tuple(
            FieldSpan(
                span_id=f"attempt:{core_id}:{index}",
                text=text,
                kind="training_attempt",
                source=core_id,
                provenance="heart:committed-attempt",
            )
            for index, text in enumerate(attempts)
        ),
    )
    return SharedFieldSnapshot(
        tick_id=tick_id,
        regions=(
            RegionState.from_text(
                LogicalRegion.TRAINER_INSTRUCTIONS,
                assignment,
                source="trainer",
                provenance="heart:committed-assignment",
            ),
            responses,
        ),
    )


def _view(snapshot: SharedFieldSnapshot) -> TrainingAttentionView:
    return TrainingAttentionViewCompiler().compile(
        snapshot, core_id=_CORE, cohort_core_ids=(_CORE,)
    )


def _serialize(model: CompleteField64D) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _objective(model: CompleteField64D, view: TrainingAttentionView, target_text: str, *, with_grad: bool):
    """The typed objective contract exactly as the gate engine recomputes it."""
    model.zero_grad(set_to_none=True)
    reader_state, memory, _manifest = model.read_compiled_with_memory(view.compiled)
    logits, targets = model.decode_teacher(reader_state, target_text, head=0, memory=memory)
    grad_norm: float | None = None
    if with_grad:
        loss = sequence_cross_entropy(logits, targets)
        loss.backward()
        total = torch.zeros((), dtype=torch.float64)
        for param in model.parameters():
            if param.grad is not None:
                total = total + param.grad.detach().to(torch.float64).pow(2).sum()
        grad_norm = math.sqrt(float(total.item()))
    return logits.detach(), targets.detach(), grad_norm


def _train_worker(view: TrainingAttentionView, target_text: str, *, steps: int = 150, seed: int = 1234):
    """Run the real model and really optimize it; return honest evidence."""
    torch.manual_seed(seed)
    model = CompleteField64D(ReaderConfig()).eval()
    prior = _serialize(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        reader_state, memory, _ = model.read_compiled_with_memory(view.compiled)
        logits, targets = model.decode_teacher(reader_state, target_text, head=0, memory=memory)
        loss = sequence_cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
    final_logits, targets, grad_norm = _objective(model, view, target_text, with_grad=True)
    loss = float(sequence_cross_entropy(final_logits, targets).item())
    accuracy = float(teacher_char_accuracy(final_logits, targets))
    return _serialize(model), prior, loss, grad_norm, accuracy


def _seal(bundle: WorkerEvidenceBundle, **overrides) -> dict:
    params = {
        "core_identity": bundle.core_id,
        "assignment_id": bundle.assignment_id,
        "payload": bundle.sealed_payload_bytes,
        "ttl_seconds": 300,
        "now": _ENVELOPE_NOW,
        "environ": _ENVIRON,
    }
    params.update(overrides)
    return seal_envelope(**params)


def _make_bundle(
    parameter_bytes: bytes,
    *,
    prior_sha256: str | None,
    loss: float,
    grad_norm: float | None,
    accuracy: float,
    kind: SubmissionKind = SubmissionKind.LEARNING_RESULT,
    claims: tuple[str, ...] = (),
    view: TrainingAttentionView,
    field_id: str | None = None,
    tick_id: int | None = None,
) -> WorkerEvidenceBundle:
    return WorkerEvidenceBundle(
        core_id=_CORE,
        assignment_id="placeholder",
        view_id=view.view_id,
        field_id=field_id if field_id is not None else view.source_field_id,
        tick_id=tick_id if tick_id is not None else view.source_tick_id,
        submission_kind=kind,
        parameter_bytes=parameter_bytes,
        claimed_loss=loss,
        claimed_grad_norm=grad_norm,
        claimed_accuracy=accuracy,
        prior_parameter_sha256=prior_sha256,
        optimizer_generation="opt-gen-1",
        counterfactual_claims=claims,
    )


def _attach_envelope(bundle: WorkerEvidenceBundle, **seal_overrides) -> WorkerEvidenceBundle:
    return replace(bundle, envelope=_seal(bundle, **seal_overrides))


def _store(tmp_path: Path, clock: FakeClock | None = None) -> AssignmentStore:
    return AssignmentStore(tmp_path / "assignments", clock=clock or FakeClock())


def _activate(store: AssignmentStore, view: TrainingAttentionView, *, threshold: int = 3, key: str = "claim-1"):
    record = store.create_assignment(
        kind=AssignmentKind.LEARNING,
        curriculum_ref="curriculum://abc-sequence",
        cohort_eligibility={"core_mode": "OFFLINE_TRAINING", "core_ids": [_CORE]},
        field_binding={"field_id": view.source_field_id, "view_id": view.view_id},
        holder_core_id=_CORE,
        origin="supervisor:test",
        lease_seconds=600.0,
        idempotency_key=key,
        failure_threshold=threshold,
    )
    return store.activate(
        record.assignment_id,
        holder_core_id=_CORE,
        expected_revision=record.revision,
        idempotency_key=f"{key}:activate",
    )


def _evaluate(engine, bundle, assignment, view, store, nonce_cache, **kwargs):
    return engine.evaluate(
        bundle,
        assignment,
        view,
        target_text=_TARGET_TEXT,
        store=store,
        snapshot=None,
        nonce_cache=nonce_cache,
        now=_EVAL_NOW,
        **kwargs,
    )


def _honest_trained_submission(view: TrainingAttentionView, assignment_id: str):
    params, prior, loss, grad_norm, accuracy = _train_worker(view, _TARGET_TEXT)
    bundle = _make_bundle(
        params,
        prior_sha256=hashlib.sha256(prior).hexdigest(),
        loss=loss,
        grad_norm=grad_norm,
        accuracy=accuracy,
        view=view,
    )
    return replace(bundle, assignment_id=assignment_id)


# ------------------------------------------------------------------ unit pieces


def test_constant_output_floor_is_most_frequent_token_share() -> None:
    assert constant_output_floor([1, 1, 1, 2]) == pytest.approx(0.75)
    assert constant_output_floor([5, 5, 5, 5]) == 1.0
    with pytest.raises(ValueError):
        constant_output_floor([])


def test_memory_nonce_cache_rejects_repeats_and_persists(tmp_path: Path) -> None:
    cache = MemoryNonceCache(tmp_path)
    assert cache("ab" * 16) is True
    assert cache("ab" * 16) is False
    assert cache("cd" * 16) is True
    reloaded = MemoryNonceCache(tmp_path)
    assert reloaded("ab" * 16) is False
    assert reloaded("cd" * 16) is False
    assert reloaded("ef" * 16) is True


def test_sealed_payload_binds_every_claim() -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    params, prior, loss, grad_norm, accuracy = _train_worker(view, _TARGET_TEXT, steps=5)
    bundle = _make_bundle(
        params,
        prior_sha256=hashlib.sha256(prior).hexdigest(),
        loss=loss,
        grad_norm=grad_norm,
        accuracy=accuracy,
        view=view,
    )
    sealed = bundle.sealed_payload_bytes
    verified = verify_envelope(
        _seal(bundle),
        payload=sealed,
        expected_core_identity=_CORE,
        expected_assignment_id=bundle.assignment_id,
        nonce_cache=MemoryNonceCache(),
        now=_EVAL_NOW,
        environ=_ENVIRON,
    )
    assert verified["payload_sha256"] == bundle.bundle_sha256
    # Any claim change breaks the sealed identity.
    tampered = replace(bundle, claimed_loss=loss + 1.0)
    with pytest.raises(RailAuthError):
        verify_envelope(
            _seal(bundle),
            payload=tampered.sealed_payload_bytes,
            expected_core_identity=_CORE,
            expected_assignment_id=bundle.assignment_id,
            nonce_cache=None,
            now=_EVAL_NOW,
            environ=_ENVIRON,
        )


# ------------------------------------------------------------- acceptance paths


def test_honest_worker_is_accepted_and_committed_exactly_once(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    bundle = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))
    nonce_cache = MemoryNonceCache(tmp_path / "nonces")

    verdict = _evaluate(engine, bundle, assignment, view, store, nonce_cache)

    assert verdict.decision is GateDecision.ACCEPTED
    assert verdict.outcome is AttemptOutcome.PASS
    assert verdict.attempt_id is not None
    assert verdict.recomputed is not None
    assert verdict.recomputed.accuracy > verdict.accuracy_floor
    assert verdict.parameters_changed is True
    attempt = store.get_attempt(verdict.attempt_id)
    assert attempt.outcome is AttemptOutcome.PASS
    assert attempt.parameter_generation == bundle.parameter_sha256
    assert attempt.evidence["parameter_sha256"] == bundle.parameter_sha256
    assert attempt.evidence["prior_parameter_sha256"] == bundle.prior_parameter_sha256
    assert store.get_assignment(assignment.assignment_id).attempt_count == 1
    assert_resume_continuity(
        store.attempts_for(assignment.assignment_id),
        expected_initial_parameter_sha256=bundle.prior_parameter_sha256,
    )


def test_evaluation_without_store_returns_verdict_without_commit(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    bundle = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))

    verdict = engine.evaluate(
        bundle,
        assignment,
        view,
        target_text=_TARGET_TEXT,
        store=None,
        nonce_cache=MemoryNonceCache(),
        now=_EVAL_NOW,
    )

    assert verdict.decision is GateDecision.ACCEPTED
    assert verdict.attempt_id is None


# ------------------------------------------------------------------ cheating paths


def test_cheating_worker_claiming_low_loss_is_rejected(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    honest = _honest_trained_submission(view, assignment.assignment_id)
    cheating = replace(honest, claimed_loss=0.000001)
    bundle = _attach_envelope(cheating)

    verdict = _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache())

    assert verdict.decision is GateDecision.REJECTED
    assert verdict.outcome is AttemptOutcome.FAIL
    assert any("disagrees with recomputed loss" in reason for reason in verdict.reasons)
    assert verdict.recomputed is not None
    assert abs(verdict.recomputed.loss - honest.claimed_loss) <= LOSS_RECOMPUTATION_TOLERANCE
    # The lie is preserved as evidence, exactly once.
    attempts = store.attempts_for(assignment.assignment_id)
    assert len(attempts) == 1
    assert attempts[0].outcome is AttemptOutcome.FAIL


def test_cheating_worker_claiming_wrong_accuracy_is_rejected(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    honest = _honest_trained_submission(view, assignment.assignment_id)
    bundle = _attach_envelope(replace(honest, claimed_accuracy=0.0))

    verdict = _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache())

    assert verdict.decision is GateDecision.REJECTED
    assert any("disagrees with recomputed accuracy" in reason for reason in verdict.reasons)


def test_architecture_mismatch_params_are_rejected(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    params, prior, loss, grad_norm, accuracy = _train_worker(view, _TARGET_TEXT, steps=5)
    state_dict = torch.load(io.BytesIO(params), map_location="cpu")
    del state_dict["decoder_output.weight"]
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    bundle = _attach_envelope(
        replace(
            _make_bundle(
                buffer.getvalue(),
                prior_sha256=hashlib.sha256(prior).hexdigest(),
                loss=loss,
                grad_norm=grad_norm,
                accuracy=accuracy,
                view=view,
            ),
            assignment_id=assignment.assignment_id,
        )
    )

    verdict = _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache())

    assert verdict.decision is GateDecision.REJECTED
    assert any("architecture" in reason for reason in verdict.reasons)


# ------------------------------------------------------------------- auth paths


def test_replayed_envelope_fails_closed(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    bundle = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))
    nonce_cache = MemoryNonceCache(tmp_path / "nonces")

    first = _evaluate(engine, bundle, assignment, view, store, nonce_cache)
    assert first.decision is GateDecision.ACCEPTED

    replayed = _evaluate(engine, bundle, assignment, view, store, nonce_cache)
    assert replayed.decision is GateDecision.REJECTED
    assert any("replay" in reason for reason in replayed.reasons)
    assert len(store.attempts_for(assignment.assignment_id)) == 1


def test_tampered_signature_fails_closed(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    bundle = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))
    envelope = dict(bundle.envelope)
    flipped = "0" if envelope["signature"][0] != "0" else "1"
    envelope["signature"] = flipped + envelope["signature"][1:]
    bundle = replace(bundle, envelope=envelope)

    verdict = _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache())

    assert verdict.decision is GateDecision.REJECTED
    assert any("unauthenticated" in reason for reason in verdict.reasons)


def test_expired_envelope_fails_closed(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    bundle = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))

    verdict = engine.evaluate(
        bundle,
        assignment,
        view,
        target_text=_TARGET_TEXT,
        store=store,
        nonce_cache=MemoryNonceCache(),
        now=_ENVELOPE_NOW + 400,
    )

    assert verdict.decision is GateDecision.REJECTED
    assert any("expired" in reason for reason in verdict.reasons)


def test_missing_secret_fails_closed(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    bundle = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))
    engine = GateEngine(environ={})

    verdict = _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache())

    assert verdict.decision is GateDecision.REJECTED
    assert any("unauthenticated" in reason for reason in verdict.reasons)


def test_wrong_core_submission_fails_before_authentication(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    honest = _honest_trained_submission(view, assignment.assignment_id)
    impostor = replace(honest, core_id="core-impostor")
    bundle = _attach_envelope(impostor)  # valid envelope, wrong core

    verdict = _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache())

    assert verdict.decision is GateDecision.REJECTED
    assert any("wrong-core" in reason for reason in verdict.reasons)


def test_wrong_tick_and_wrong_view_fail_closed(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    honest = _honest_trained_submission(view, assignment.assignment_id)

    wrong_tick = _attach_envelope(replace(honest, tick_id=view.source_tick_id + 1))
    verdict = _evaluate(engine, wrong_tick, assignment, view, store, MemoryNonceCache())
    assert verdict.decision is GateDecision.REJECTED
    assert any("wrong-tick" in reason for reason in verdict.reasons)

    wrong_view = _attach_envelope(replace(honest, view_id="f" * 64))
    verdict = _evaluate(engine, wrong_view, assignment, view, store, MemoryNonceCache())
    assert verdict.decision is GateDecision.REJECTED
    assert any("wrong-view" in reason for reason in verdict.reasons)


def test_stale_view_fails_against_canonical_snapshot(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    bundle = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))
    advanced = _snapshot(tick_id=snapshot.tick_id + 1)

    verdict = engine.evaluate(
        bundle,
        assignment,
        view,
        target_text=_TARGET_TEXT,
        store=store,
        snapshot=advanced,
        nonce_cache=MemoryNonceCache(),
        now=_EVAL_NOW,
    )

    assert verdict.decision is GateDecision.REJECTED
    assert any("stale-view" in reason for reason in verdict.reasons)


def test_envelope_for_a_different_assignment_fails_closed(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    bundle = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))

    cross = replace(bundle, envelope=_seal(bundle, assignment_id="assignment-other"))
    verdict = _evaluate(engine, cross, assignment, view, store, MemoryNonceCache())

    assert verdict.decision is GateDecision.REJECTED
    assert any("unauthenticated" in reason for reason in verdict.reasons)


# ------------------------------------------------------------- evaluation gates


def test_below_constant_output_floor_is_rejected_not_accepted(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    # A model honestly trained on one target but evaluated against a target
    # it never learned: real gradients, real parameter change, real low score.
    trained_view = view
    params, prior, _loss, grad_norm, _accuracy = _train_worker(trained_view, _TARGET_TEXT)
    honest_model = CompleteField64D(ReaderConfig()).eval()
    honest_model.load_state_dict(torch.load(io.BytesIO(params), map_location="cpu"))
    logits, targets, grad_norm = _objective(honest_model, view, "zzzzzzzzzz", with_grad=True)
    loss = float(sequence_cross_entropy(logits, targets).item())
    accuracy = float(teacher_char_accuracy(logits, targets))
    assert accuracy <= constant_output_floor([int(t) for t in targets.flatten().tolist()])
    bundle = _attach_envelope(
        replace(
            _make_bundle(
                params,
                prior_sha256=hashlib.sha256(prior).hexdigest(),
                loss=loss,
                grad_norm=grad_norm,
                accuracy=accuracy,
                view=view,
            ),
            assignment_id=assignment.assignment_id,
        )
    )

    verdict = engine.evaluate(
        bundle,
        assignment,
        view,
        target_text="zzzzzzzzzz",
        store=store,
        nonce_cache=MemoryNonceCache(),
        now=_EVAL_NOW,
    )

    assert verdict.decision is GateDecision.REJECTED
    assert any("below-floor" in reason for reason in verdict.reasons)
    assert verdict.recomputed is not None and verdict.recomputed.grad_norm > 0.0


def test_unchanged_parameters_on_learning_assignment_pauses(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    params, _prior, loss, grad_norm, accuracy = _train_worker(view, _TARGET_TEXT)
    # The worker claims a learning step but returned the untouched parameters.
    bundle = _attach_envelope(
        replace(
            _make_bundle(
                params,
                prior_sha256=hashlib.sha256(params).hexdigest(),
                loss=loss,
                grad_norm=grad_norm,
                accuracy=accuracy,
                view=view,
            ),
            assignment_id=assignment.assignment_id,
        )
    )

    verdict = _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache())

    assert verdict.decision is GateDecision.PAUSED
    assert any("did not change" in reason for reason in verdict.reasons)
    record = store.get_assignment(assignment.assignment_id)
    assert record.status is AssignmentStatus.PAUSED
    attempts = store.attempts_for(assignment.assignment_id)
    assert len(attempts) == 1 and attempts[0].outcome is AttemptOutcome.FAIL


def test_nonfinite_recomputed_metrics_pause_instead_of_accept(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    params, prior, _loss, _grad, _acc = _train_worker(view, _TARGET_TEXT)
    state_dict = {key: value.clone() for key, value in torch.load(io.BytesIO(params), map_location="cpu").items()}
    state_dict["decoder_output.bias"][0] = float("nan")
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    bundle = _attach_envelope(
        replace(
            _make_bundle(
                buffer.getvalue(),
                prior_sha256=hashlib.sha256(prior).hexdigest(),
                loss=1.0,  # whatever the worker claims: recomputation is authority
                grad_norm=1.0,
                accuracy=0.5,
                view=view,
            ),
            assignment_id=assignment.assignment_id,
        )
    )

    verdict = _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache())

    assert verdict.decision is GateDecision.PAUSED
    assert any("non-finite" in reason for reason in verdict.reasons)
    assert store.get_assignment(assignment.assignment_id).status is AssignmentStatus.PAUSED


def test_zero_gradient_norm_pauses(tmp_path: Path) -> None:
    # One-character field, repeated-character target: the crafted parameters
    # make the typed output provably independent of every parameter (single
    # copy source, saturated copy gate), so the recomputed gradient is
    # exactly zero.  This is a real forward/backward through the real model.
    snapshot = _snapshot(assignment="a")
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    target = "aaaa"
    torch.manual_seed(99)
    model = CompleteField64D(ReaderConfig()).eval()
    prior = _serialize(model)
    with torch.no_grad():
        model.position_query.weight.zero_()
        model.position_key.weight.zero_()
        model.copy_gate.bias.fill_(-100.0)  # sigmoid underflows to exactly 0.0f
    params = _serialize(model)
    logits, targets, grad_norm = _objective(model, view, target, with_grad=True)
    assert grad_norm == 0.0
    loss = float(sequence_cross_entropy(logits, targets).item())
    accuracy = float(teacher_char_accuracy(logits, targets))
    bundle = _attach_envelope(
        replace(
            _make_bundle(
                params,
                prior_sha256=hashlib.sha256(prior).hexdigest(),
                loss=loss,
                grad_norm=grad_norm,
                accuracy=accuracy,
                view=view,
            ),
            assignment_id=assignment.assignment_id,
        )
    )

    verdict = engine.evaluate(
        bundle,
        assignment,
        view,
        target_text=target,
        store=store,
        nonce_cache=MemoryNonceCache(),
        now=_EVAL_NOW,
    )

    assert verdict.decision is GateDecision.PAUSED
    assert any("gradient norm is zero" in reason for reason in verdict.reasons)
    assert store.get_assignment(assignment.assignment_id).status is AssignmentStatus.PAUSED


# ------------------------------------------------------------------ probe paths


def test_counterfactual_probe_submission_is_never_a_pass(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    params, _prior, loss, grad_norm, accuracy = _train_worker(view, _TARGET_TEXT)
    probe_snapshot = _snapshot(assignment="unrelated content entirely")
    bundle = _attach_envelope(
        replace(
            _make_bundle(
                params,
                prior_sha256=None,
                loss=loss,
                grad_norm=grad_norm,
                accuracy=accuracy,
                kind=SubmissionKind.COUNTERFACTUAL_PROBE,
                claims=(PROBE_RAIL, PROBE_SOUL, PROBE_RAIL_CONTENT, "bogus_mechanism"),
                view=view,
            ),
            assignment_id=assignment.assignment_id,
            soul_payload=encode_hot_payload(
                reader_state=torch.full((1, 4, 64), 0.25),
                metrics={"source": "test"},
            ),
            soul_media_type="application/x-axon-local-worker-soul-hot",
            soul_tensor_layout="reader-state-float32",
        )
    )

    verdict = _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache(), probe_snapshot=probe_snapshot)

    assert verdict.outcome is AttemptOutcome.COUNTERFACTUAL_ONLY
    assert verdict.decision is GateDecision.REJECTED
    reports = {report.name: report for report in verdict.mechanism_reports}
    assert set(reports) == {PROBE_RAIL, PROBE_SOUL, PROBE_RAIL_CONTENT, "bogus_mechanism"}
    # The real core genuinely uses every claimed, supported mechanism.
    assert reports[PROBE_RAIL].used is True
    assert reports[PROBE_SOUL].used is True
    assert reports[PROBE_RAIL_CONTENT].used is True
    assert reports["bogus_mechanism"].used is False
    assert "unsupported" in reports["bogus_mechanism"].note
    # Probe outcomes are transparent to the failure series.
    record = store.get_assignment(assignment.assignment_id)
    assert record.consecutive_failures == 0
    assert len(store.attempts_for(assignment.assignment_id)) == 1


def test_core_without_mechanism_use_is_reported(tmp_path: Path) -> None:
    # A core whose output path cannot see the rail reports rail unused.
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    torch.manual_seed(5)
    model = CompleteField64D(ReaderConfig()).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    params = _serialize(model)
    bundle = _attach_envelope(
        replace(
            _make_bundle(
                params,
                prior_sha256=None,
                loss=0.0,
                grad_norm=0.0,
                accuracy=0.0,
                kind=SubmissionKind.COUNTERFACTUAL_PROBE,
                claims=(PROBE_RAIL,),
                view=view,
            ),
            assignment_id=assignment.assignment_id,
        )
    )

    verdict = _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache())

    reports = {report.name: report for report in verdict.mechanism_reports}
    assert reports[PROBE_RAIL].used is False
    assert reports[PROBE_RAIL].max_abs_delta == 0.0


# -------------------------------------------------------------- failed series


def _failing_worker_submission(view: TrainingAttentionView, assignment_id: str, seed: int):
    """An honest worker that learned the wrong target: real everything, low score."""
    params, prior, _l, _g, _a = _train_worker(view, _TARGET_TEXT, seed=seed)
    model = CompleteField64D(ReaderConfig()).eval()
    model.load_state_dict(torch.load(io.BytesIO(params), map_location="cpu"))
    logits, targets, grad_norm = _objective(model, view, "zzzzzzzzzz", with_grad=True)
    loss = float(sequence_cross_entropy(logits, targets).item())
    accuracy = float(teacher_char_accuracy(logits, targets))
    return replace(
        _make_bundle(
            params,
            prior_sha256=hashlib.sha256(prior).hexdigest(),
            loss=loss,
            grad_norm=grad_norm,
            accuracy=accuracy,
            view=view,
        ),
        assignment_id=assignment_id,
    )


def test_failed_series_escalates_to_supervisor_and_preserves_every_attempt(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view, threshold=2)
    engine = GateEngine(environ=_ENVIRON)
    nonce_cache = MemoryNonceCache(tmp_path / "nonces")

    first = _evaluate(
        engine,
        _attach_envelope(_failing_worker_submission(view, assignment.assignment_id, seed=11)),
        assignment,
        view,
        store,
        nonce_cache,
    )
    assert first.decision is GateDecision.REJECTED
    current = store.get_assignment(assignment.assignment_id)
    assert current.status is AssignmentStatus.ACTIVE
    assert current.consecutive_failures == 1

    second = _evaluate(
        engine,
        _attach_envelope(_failing_worker_submission(view, assignment.assignment_id, seed=22)),
        assignment,
        view,
        store,
        nonce_cache,
    )
    assert second.decision is GateDecision.ESCALATED
    record = store.get_assignment(assignment.assignment_id)
    assert record.status is AssignmentStatus.ESCALATED
    assert record.consecutive_failures == 2
    attempts = store.attempts_for(assignment.assignment_id)
    assert len(attempts) == 2
    assert all(attempt.outcome is AttemptOutcome.FAIL for attempt in attempts)
    assert_no_duplicate_attempt_indices(attempts)


# ------------------------------------------------------------- pause and resume


def test_pause_then_resume_preserves_exact_continuity(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    params, _prior, loss, grad_norm, accuracy = _train_worker(view, _TARGET_TEXT)
    # The first submission reports an honest evaluation but returned the
    # untouched parameters: verified no-learning -> pause.
    stale = _attach_envelope(
        replace(
            _make_bundle(
                params,
                prior_sha256=hashlib.sha256(params).hexdigest(),
                loss=loss,
                grad_norm=grad_norm,
                accuracy=accuracy,
                view=view,
            ),
            assignment_id=assignment.assignment_id,
        )
    )
    paused = _evaluate(engine, stale, assignment, view, store, MemoryNonceCache(tmp_path / "n1"))
    assert paused.decision is GateDecision.PAUSED

    # --- restart: brand-new store objects over the same durable root ---
    clock = FakeClock(1_050.0)
    store2 = _store(tmp_path, clock)
    report = store2.recover()
    assert report.attempts == 1
    record = store2.get_assignment(assignment.assignment_id)
    assert record.status is AssignmentStatus.PAUSED
    resumed = store2.resume(
        record.assignment_id,
        core_id=_CORE,
        expected_revision=record.revision,
        idempotency_key="resume-1",
        lease_seconds=600.0,
        reason="worker reconnected after preemption",
    )
    assert resumed.status is AssignmentStatus.ACTIVE

    # A second honest learning step from the returned parameters.
    torch.manual_seed(777)
    model = CompleteField64D(ReaderConfig()).eval()
    model.load_state_dict(torch.load(io.BytesIO(params), map_location="cpu"))
    prior2 = params
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(20):
        model.zero_grad(set_to_none=True)
        reader_state, memory, _ = model.read_compiled_with_memory(view.compiled)
        logits, targets = model.decode_teacher(reader_state, _TARGET_TEXT, head=0, memory=memory)
        loss2 = sequence_cross_entropy(logits, targets)
        loss2.backward()
        optimizer.step()
    params2 = _serialize(model)
    logits, targets, grad_norm2 = _objective(model, view, _TARGET_TEXT, with_grad=True)
    bundle2 = _attach_envelope(
        replace(
            _make_bundle(
                params2,
                prior_sha256=hashlib.sha256(prior2).hexdigest(),
                loss=float(sequence_cross_entropy(logits, targets).item()),
                grad_norm=grad_norm2,
                accuracy=float(teacher_char_accuracy(logits, targets)),
                view=view,
            ),
            assignment_id=assignment.assignment_id,
        )
    )
    verdict2 = _evaluate(engine, bundle2, resumed, view, store2, MemoryNonceCache(tmp_path / "n2"))
    assert verdict2.decision is GateDecision.ACCEPTED

    attempts = store2.attempts_for(assignment.assignment_id)
    assert [attempt.attempt_index for attempt in attempts] == [0, 1]
    assert_resume_continuity(attempts)
    assert_exact_generation_lineage(attempts)


def test_resume_continuity_rejects_duplicate_indices(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    bundle = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))
    _evaluate(engine, bundle, assignment, view, store, MemoryNonceCache())
    attempts = list(store.attempts_for(assignment.assignment_id))
    assert_resume_continuity(attempts)

    duplicated = [*attempts, replace(attempts[0])]
    with pytest.raises(ResumeContinuityError):
        assert_no_duplicate_attempt_indices(duplicated)


def test_resume_continuity_rejects_broken_generation_lineage(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    # Two accepted attempts that do not chain: the second begins from the
    # wrong prior parameters.
    first = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))
    verdict1 = _evaluate(engine, first, assignment, view, store, MemoryNonceCache(tmp_path / "n1"))
    assert verdict1.decision is GateDecision.ACCEPTED

    params, _prior, loss, grad_norm, accuracy = _train_worker(view, _TARGET_TEXT, seed=4242)
    orphan = _attach_envelope(
        replace(
            _make_bundle(
                params,
                prior_sha256="0" * 64,  # does not match attempt 0's returned parameters
                loss=loss,
                grad_norm=grad_norm,
                accuracy=accuracy,
                view=view,
            ),
            assignment_id=assignment.assignment_id,
        )
    )
    current = store.get_assignment(assignment.assignment_id)
    verdict2 = _evaluate(engine, orphan, current, view, store, MemoryNonceCache(tmp_path / "n1"))
    assert verdict2.decision is GateDecision.ACCEPTED

    attempts = store.attempts_for(assignment.assignment_id)
    assert len(attempts) == 2
    with pytest.raises(ResumeContinuityError):
        assert_exact_generation_lineage(attempts)
    with pytest.raises(ResumeContinuityError):
        assert_resume_continuity(attempts)


def test_acceptance_that_cannot_commit_fails_closed(tmp_path: Path) -> None:
    snapshot = _snapshot()
    view = _view(snapshot)
    store = _store(tmp_path)
    assignment = _activate(store, view)
    engine = GateEngine(environ=_ENVIRON)
    bundle = _attach_envelope(_honest_trained_submission(view, assignment.assignment_id))
    # Exhaust the lease so the durable commit must refuse; an acceptance that
    # cannot be committed exactly once must not stand.
    clock_store = _store(tmp_path, FakeClock(5_000.0))
    far = clock_store.get_assignment(assignment.assignment_id)

    with pytest.raises((GateEvaluationError, Exception)):
        _evaluate(engine, bundle, far, view, clock_store, MemoryNonceCache())
