"""Regression tests for the remediation of the nine Codex blocking findings.

Each test pins one previously-open finding against the current mechanisms:

1. the inhaled Soul payload conditions the worker's recurrent forward;
2. a hashed bound parameter generation without the exact accepted
   checkpoint bytes fails closed (no untrusted fresh-seed generation);
6. accepted outcomes are materialized into the canonical
   ``training_responses`` region through the Heart transaction boundary
   under ``AuthorityGrant.trainer()``, and the next view compiles from the
   successor field/tick;
7. an explicit assignment target binds the loss, the emission, and the gate
   recomputation (task semantics, not instruction reconstruction);
8. a missing nonce cache fails closed at the ``GateEngine.evaluate``
   boundary before any other check.

Everything runs the real model on injected state roots with fixed seeds and
a fixed clock; no network, no threads.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from runtime.field import (
    CanonicalStateBranch,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_sha256,
)
from runtime.field.training_view import (
    TrainingAttentionView,
    TrainingAttentionViewCompiler,
    TrainingHistoryWindow,
)
from runtime.heart.authority import AuthorityClass, AuthorityGrant
from runtime.heart.registry import (
    CoreBinding,
    CoreDescriptor,
    CoreRegistry,
    CoreStatus,
    OperatorCoreGrant,
)
from runtime.heart.transaction import HeartTransactionBoundary
from runtime.soul import SoulStore, SoulTemperature
from runtime.trainer.assignments import AssignmentKind, AssignmentStore
from runtime.trainer.local_worker import (
    LOCAL_TRAINING_ARCHITECTURE,
    LocalTrainingWorker,
    LocalTrainingWorkerError,
    decode_hot_payload,
)
from runtime.trainer.supervisory_gates import (
    GateDecision,
    MemoryNonceCache,
    SubmissionKind,
    WorkerEvidenceBundle,
)
from runtime.trainer.training_session import (
    TrainingSession,
    TrainingSessionConfig,
)
from training.complete_field_64d import (
    sequence_cross_entropy,
    teacher_char_accuracy,
)

_CORE = "tr"
_SECRET = "k" * 48
_ENVIRON = {"AXON_RAIL_SECRET": _SECRET}
_T0 = 1_000.0
_ASSIGNMENT_TEXT = "ABC? Recite the alphabet forward from 'a'."
_EXPLICIT_TARGET = "recite the alphabet forward from 'a'."


@pytest.fixture
def state_root():
    tmp = tempfile.mkdtemp(prefix="r")
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
    target_text: str | None = None,
    heart: HeartTransactionBoundary | None = None,
    canonical_branch: CanonicalStateBranch | None = None,
) -> TrainingSession:
    config = TrainingSessionConfig(
        core_id=_CORE,
        curriculum_ref="abc-sequence:train:0000",
        cohort_core_ids=(_CORE,),
        assignment_kind=AssignmentKind.LEARNING,
        lease_seconds=600.0,
        failure_threshold=3,
        max_attempts=8,
        target_text=target_text,
    )
    return TrainingSession(
        state_root=state_root,
        registry=registry,
        snapshot=snapshot if snapshot is not None else _snapshot(),
        config=config,
        operator_grant=grant,
        clock=lambda: _T0,
        rail_environ=_ENVIRON,
        heart=heart,
        canonical_branch=canonical_branch,
    )


def _serialize(model: torch.nn.Module) -> bytes:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def _evaled_metrics(
    session: TrainingSession, supervised_text: str
) -> tuple[float, float]:
    model = session.worker.model
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        reader_state, memory, _ = model.read_compiled_with_memory(
            session.view.compiled
        )
        logits, targets = model.decode_teacher(
            reader_state, supervised_text, head=0, memory=memory
        )
    return (
        float(sequence_cross_entropy(logits, targets).item()),
        float(teacher_char_accuracy(logits, targets)),
    )


# --- finding 1: the inhaled Soul conditions the worker forward ---------------


def test_inhaled_soul_payload_conditions_the_worker_forward(state_root) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    result = session.run(max_attempts=2)
    assert all(step.verdict.decision is GateDecision.ACCEPTED for step in result.steps)

    # The durable Soul head carries the exact hot payload exhaled at the
    # final attempt; decoding it yields the recurrent reader state.
    head = SoulStore.active(state_root).branch(_CORE).load_head()
    hot = head.layer(SoulTemperature.HOT)
    reader_array, metrics = decode_hot_payload(hot.payload)
    model = session.worker.model
    # The exhale persists the batched reader state; the model also accepts
    # the unbatched bound shape.  Anything else failed closed at the worker.
    assert reader_array.shape == (1, *tuple(model.initial_state.shape))
    assert metrics["attempt_index"] == 1

    initial_state = torch.from_numpy(reader_array)

    def forward_logits(initial: torch.Tensor | None) -> torch.Tensor:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            reader_state, memory, _ = model.read_compiled_with_memory(
                session.view.compiled, initial_state=initial
            )
            logits, _targets = model.decode_teacher(
                reader_state, session.objective_text, head=0, memory=memory
            )
        return logits

    conditioned = forward_logits(initial_state)
    # The decode is deterministic: the same exact Soul state reproduces.
    assert torch.equal(conditioned, forward_logits(initial_state))
    # The persisted Soul genuinely drives the forward: zeroing it (the gate's
    # soul counterfactual) moves the typed output.
    zeroed = forward_logits(torch.zeros_like(model.initial_state))
    assert not torch.equal(conditioned, zeroed)
    assert (conditioned - zeroed).abs().max().item() > 0.0
    # The hot payload is the exact reader state exhale, not a digest of it.
    assert hot.payload_sha256 == hashlib.sha256(hot.payload).hexdigest()


# --- finding 2: no untrusted fresh-seed generation ---------------------------


def test_hashed_generation_without_checkpoint_bytes_fails_closed(
    state_root,
) -> None:
    snapshot = _snapshot()
    view = TrainingAttentionViewCompiler().compile(
        snapshot,
        core_id=_CORE,
        cohort_core_ids=(_CORE,),
        history_window=TrainingHistoryWindow.none(),
    )
    store = AssignmentStore.active(state_root, clock=lambda: _T0)
    assignment = store.create_assignment(
        kind=AssignmentKind.LEARNING,
        curriculum_ref="abc-sequence:train:0000",
        cohort_eligibility={"core_ids": [_CORE]},
        field_binding={
            "field_id": snapshot.field_id,
            "view_id": view.view_id,
            "tick_id": view.source_tick_id,
        },
        holder_core_id=_CORE,
        origin="test-trainer",
        lease_seconds=600.0,
        idempotency_key="claim-hashed",
    )
    assignment = store.activate(
        assignment.assignment_id,
        holder_core_id=_CORE,
        expected_revision=assignment.revision,
        idempotency_key="activate-hashed",
    )
    souls = SoulStore.active(state_root)
    head = souls.ensure_core(
        core_id=_CORE,
        architecture_id=LOCAL_TRAINING_ARCHITECTURE,
        parameter_generation="untrained",
    )
    # A binding that names hashed accepted lineage the worker never loaded:
    # the worker must refuse to fabricate a fresh seeded model for it.
    foreign = CoreBinding(
        core_id=_CORE,
        architecture_id=LOCAL_TRAINING_ARCHITECTURE,
        parameter_generation="a" * 64,
        optimizer_generation="b" * 64,
        soul_id=head.soul_id,
        mode=CoreStatus.OFFLINE_TRAINING,
    )
    worker = LocalTrainingWorker(
        assignment_store=store, clock=lambda: _T0, rail_environ=_ENVIRON
    )
    with pytest.raises(LocalTrainingWorkerError, match="does not hold"):
        worker.execute_attempt(foreign, assignment, view, souls.branch(_CORE))
    assert worker._model is None  # nothing was fabricated


# --- finding 8: replay protection is mandatory at the gate boundary ----------


def test_missing_nonce_cache_fails_closed_before_any_other_check(
    state_root,
) -> None:
    from runtime.heart.rail_auth import seal_envelope

    registry, grant = _registry()
    session = _session(state_root, registry, grant)
    result = session.run(max_attempts=1)
    record = session.assignment_store.get_assignment(
        result.assignment.assignment_id
    )
    view = session.view

    loss, accuracy = _evaled_metrics(session, session.objective_text)
    bundle = WorkerEvidenceBundle(
        core_id=_CORE,
        assignment_id=record.assignment_id,
        view_id=view.view_id,
        field_id=view.source_field_id,
        tick_id=view.source_tick_id,
        submission_kind=SubmissionKind.LEARNING_RESULT,
        parameter_bytes=_serialize(session.worker.model),
        claimed_loss=loss,
        claimed_grad_norm=None,
        claimed_accuracy=accuracy,
        prior_parameter_sha256="p" * 64,
        optimizer_generation="opt",
    )
    bundle = dataclasses.replace(
        bundle,
        envelope=seal_envelope(
            core_identity=_CORE,
            assignment_id=record.assignment_id,
            payload=bundle.sealed_payload_bytes,
            ttl_seconds=300,
            now=int(_T0),
            environ=_ENVIRON,
        ),
    )

    # Without a nonce cache the submission fails closed for that alone...
    denied = session.gate.evaluate(
        bundle,
        record,
        view,
        target_text=session.objective_text,
        nonce_cache=None,
    )
    assert denied.decision is GateDecision.REJECTED
    assert any("nonce cache is mandatory" in reason for reason in denied.reasons)
    # ...and the identical submission verifies once a cache is supplied.
    allowed = session.gate.evaluate(
        bundle,
        record,
        view,
        target_text=session.objective_text,
        nonce_cache=MemoryNonceCache(),
    )
    assert allowed.decision in (
        GateDecision.ACCEPTED,
        GateDecision.PAUSED,
    )


# --- finding 7: explicit target binding --------------------------------------


def test_explicit_target_binds_loss_emission_and_gate_recomputation(
    state_root,
) -> None:
    registry, grant = _registry()
    session = _session(state_root, registry, grant, target_text=_EXPLICIT_TARGET)
    result = session.run(max_attempts=3)
    assert all(step.verdict.decision is GateDecision.ACCEPTED for step in result.steps)

    # The durable assignment record itself carries the explicit target.
    record = session.assignment_store.get_assignment(
        result.assignment.assignment_id
    )
    assert record.target_text == _EXPLICIT_TARGET
    target_sha = canonical_sha256(_EXPLICIT_TARGET)

    for step in result.steps:
        # The worker supervised EXACTLY the explicit target, not the
        # instructions.  Its free-running emission is a separate artifact.
        assert step.worker_bundle.metrics.supervised_characters == len(
            _EXPLICIT_TARGET
        )
        assert (
            step.attempt.evidence["explicit_target_sha256"] == target_sha
        )
        assert step.attempt.evidence["explicit_target_sha256"] != (
            canonical_sha256(_ASSIGNMENT_TEXT)
        )
        assert step.verdict.recomputed is not None

    # Independently: the final accepted parameters score the explicit target
    # differently from the instructions text (different contracts)...
    target_loss, _target_accuracy = _evaled_metrics(session, _EXPLICIT_TARGET)
    instruction_loss, _ = _evaled_metrics(session, _ASSIGNMENT_TEXT)
    assert abs(target_loss - instruction_loss) > 1e-3
    # The gate's authoritative attempt-time recomputation also differs from
    # instruction reconstruction.  It used the precise pre-attempt Soul,
    # whereas the independent final score above uses the post-attempt Soul,
    # so their numeric values are not expected to be identical.
    assert abs(
        result.steps[-1].verdict.recomputed.loss - instruction_loss
    ) > 1e-3


# --- finding 6: canonical Heart commit + successor view -----------------------


def test_heart_commits_typed_response_and_successor_view_drives_next_attendance(
    state_root,
) -> None:
    registry, grant = _registry()
    heart = HeartTransactionBoundary()
    session = _session(state_root, registry, grant, heart=heart)
    base_tick = session.view.source_tick_id
    base_field = session.view.source_field_id
    result = session.run(max_attempts=2)
    assert all(step.verdict.decision is GateDecision.ACCEPTED for step in result.steps)

    # Every attempt produced one Heart commit under trainer authority.
    commits = [step.heart_commit for step in result.steps]
    assert all(commit is not None for commit in commits)
    for commit in commits:
        assert commit.grant.authority_class is AuthorityClass.TRAINER
        assert commit.grant is not AuthorityGrant.core()
    # The successor chain advances field and tick exactly along the commits.
    assert commits[0].base_field_id == base_field
    assert commits[0].successor.parent_field_id == base_field
    assert commits[0].successor.tick_id == base_tick + 1
    assert commits[1].base_field_id == commits[0].successor.field_id
    assert commits[1].successor.tick_id == base_tick + 2

    # The canonical head now holds the typed responses: one attempt span per
    # attempt, authored by the core, bound to the attempt and the verdict.
    head = session.canonical_head
    assert head.tick_id == base_tick + 2
    responses = head.region(LogicalRegion.TRAINING_RESPONSES)
    assert len(responses.spans) == 2
    for span, step in zip(responses.spans, result.steps, strict=True):
        expected_text = "".join(
            character
            for character in step.worker_bundle.emission.characters
            if character != "<eos>"
        )
        assert span.text == expected_text
        assert span.source == _CORE
        assert step.attempt.attempt_id in span.container_refs
        assert step.verdict.verdict_id in span.container_refs
        assert step.worker_bundle.emission.emission_id in span.container_refs
    assert responses.text == "".join(span.text for span in responses.spans)

    # Each committed response triggered the next training tick.  The live
    # session view and the durable assignment binding now name the canonical
    # successor instead of replaying the original frozen view.
    assert result.steps[1].worker_bundle.field_id == commits[0].successor.field_id
    assert result.steps[1].worker_bundle.tick_id == base_tick + 1
    assert session.view.source_field_id == head.field_id
    assert session.view.source_tick_id == head.tick_id
    session.view.verify(head)
    assert result.assignment.assignment_id == result.steps[0].attempt.assignment_id
    assert result.assignment.field_binding == {
        "field_id": session.view.source_field_id,
        "view_id": session.view.view_id,
        "tick_id": session.view.source_tick_id,
    }
    next_view = session.successor_view(history_window=TrainingHistoryWindow.latest_n(1))
    assert isinstance(next_view, TrainingAttentionView)
    assert next_view.source_field_id == head.field_id
    assert next_view.source_tick_id == head.tick_id
    next_view.verify(head)
    # The successor view attends the assignment plus exactly the newest
    # committed response span, compiled from the canonical successor only.
    assert next_view.compiled.region_text(LogicalRegion.TRAINER_INSTRUCTIONS) == (
        _ASSIGNMENT_TEXT
    )
    assert next_view.compiled.region_text(LogicalRegion.TRAINING_RESPONSES) == (
        responses.spans[-1].text
    )


def test_training_response_advances_durable_canonical_branch(state_root) -> None:
    registry, grant = _registry()
    branch = CanonicalStateBranch.training("runtime-training", state_root=state_root)
    session = _session(
        state_root,
        registry,
        grant,
        heart=HeartTransactionBoundary(),
        canonical_branch=branch,
    )

    result = session.run(max_attempts=1)

    assert result.steps[0].heart_commit is not None
    assert branch.load_head() == session.canonical_head
    assert branch.load_head_record().generation == 1
    assert branch.load_head().region(LogicalRegion.TRAINING_RESPONSES).spans
