"""Scope B1 tests: the local OFFLINE_TRAINING worker (handoff step 7).

End-to-end on an ABC fixture assignment: an `ABC?` assignment reaches the
worker as exact D64 rows through a TrainingAttentionView; the worker inhales
its exact Soul, runs the real CompleteField64D motor, computes a discrete
categorical next-character loss on the same autograd graph, performs a real
backward pass and optimizer step for learning assignments, exhales the real
Soul state through SoulStore, records metrics as evidence only, and seals the
AttemptEvidenceBundle behind the same rail_auth transport boundary intended
for Kaggle — including fail-closed verification on wrong core, wrong
assignment, tampered payload, and replay.
"""
from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from runtime.field import LogicalRegion, SharedFieldSnapshot, canonical_json_bytes
from runtime.field.training_view import (
    TrainingAttentionView,
    TrainingAttentionViewCompiler,
    TrainingHistoryWindow,
)
from runtime.heart.rail_auth import RailAuthError, verify_envelope
from runtime.heart.registry import CoreBinding, CoreStatus
from runtime.soul import SoulStore
from runtime.trainer.assignments import (
    AssignmentKind,
    AssignmentStatus,
    AssignmentStore,
    AttemptOutcome,
)
from runtime.trainer.local_worker import (
    EOS_TOKEN_MARKER,
    LOCAL_TRAINING_ARCHITECTURE,
    AttemptEvidenceBundle,
    LocalTrainingWorker,
    LocalTrainingWorkerError,
    decode_hot_payload,
)

_CORE = "core-training-alpha"
_SECRET = "k" * 48
_ENVIRON = {"AXON_RAIL_SECRET": _SECRET}
_T0 = 1_000.0
_ASSIGNMENT_TEXT = "ABC? Recite the alphabet forward from 'a'."


def _snapshot(assignment_text: str = _ASSIGNMENT_TEXT, tick_id: int = 7) -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.TRAINER_INSTRUCTIONS: assignment_text,
            LogicalRegion.TRAINING_RESPONSES: "",
        },
        tick_id=tick_id,
        source="test-trainer",
        provenance="heart:committed-assignment",
    )


def _compile_view(snapshot: SharedFieldSnapshot) -> TrainingAttentionView:
    return TrainingAttentionViewCompiler().compile(
        snapshot,
        core_id=_CORE,
        cohort_core_ids=(_CORE,),
        history_window=TrainingHistoryWindow.none(),
    )


def _environment(tmp_path, kind: AssignmentKind = AssignmentKind.LEARNING, *, seed: int = 11):
    """One full local training environment: field, view, assignment, soul, worker."""

    snapshot = _snapshot()
    view = _compile_view(snapshot)
    store = AssignmentStore(tmp_path / "assignments", clock=lambda: _T0)
    assignment = store.create_assignment(
        kind=kind,
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
        idempotency_key="claim-1",
        created_at=_T0,
    )
    assignment = store.activate(
        assignment.assignment_id,
        holder_core_id=_CORE,
        expected_revision=assignment.revision,
        idempotency_key="activate-1",
    )
    soul_store = SoulStore(tmp_path / "souls")
    soul_snapshot = soul_store.ensure_core(
        core_id=_CORE,
        architecture_id=LOCAL_TRAINING_ARCHITECTURE,
        parameter_generation="untrained",
    )
    binding = CoreBinding(
        core_id=_CORE,
        architecture_id=LOCAL_TRAINING_ARCHITECTURE,
        parameter_generation="untrained",
        optimizer_generation="optimizer-init",
        soul_id=soul_snapshot.soul_id,
        mode=CoreStatus.OFFLINE_TRAINING,
    )
    worker = LocalTrainingWorker(
        assignment_store=store,
        seed=seed,
        clock=lambda: _T0,
        rail_environ=_ENVIRON,
    )
    return {
        "snapshot": snapshot,
        "view": view,
        "store": store,
        "assignment": assignment,
        "soul_store": soul_store,
        "soul_branch": soul_store.branch(_CORE),
        "binding": binding,
        "worker": worker,
    }


def _verify(bundle: AttemptEvidenceBundle, envelope: dict, nonce_cache, **overrides):
    params = {
        "payload": canonical_json_bytes(bundle.to_canonical_dict()),
        "expected_core_identity": _CORE,
        "expected_assignment_id": bundle.assignment_id,
        "nonce_cache": nonce_cache,
        "now": 2_000_010,
        "environ": _ENVIRON,
    }
    params.update(overrides)
    return verify_envelope(envelope, **params)


def _fresh_nonce_cache():
    seen: set[str] = set()

    def cache(nonce: str) -> bool:
        if nonce in seen:
            return False
        seen.add(nonce)
        return True

    return cache


# --- end-to-end learning attempt ---------------------------------------------


def test_learning_attempt_runs_real_model_and_records_evidence(tmp_path) -> None:
    env = _environment(tmp_path)
    bundle = env["worker"].execute_attempt(
        env["binding"], env["assignment"], env["view"], env["soul_branch"]
    )

    # Typed emission: causal model decode -> token ids, never host prose or
    # the teacher-forced target sequence.
    emission = bundle.emission
    assert isinstance(emission.token_ids, tuple)
    assert 1 <= len(emission.token_ids) <= 64
    assert len(emission.characters) == len(emission.token_ids)
    assert emission.terminated is (emission.characters[-1] == EOS_TOKEN_MARKER)
    from substrate import get_letter_bank

    bank_chars = set(get_letter_bank().chars[:-1])
    for token, character in zip(emission.token_ids, emission.characters, strict=True):
        if token == len(bank_chars):
            assert character == EOS_TOKEN_MARKER
        else:
            assert character in bank_chars
    # No free-form prose anywhere in the canonical bundle.
    for key in bundle.to_canonical_dict():
        assert key not in {"text", "prose", "answer", "narrative"}

    # Real metrics: finite, nonzero loss and gradient.
    assert bundle.metrics.loss == pytest.approx(bundle.metrics.loss)  # finite (NaN != itself)
    assert 0.0 < bundle.metrics.loss < 20.0
    assert bundle.metrics.grad_norm > 0.0
    assert 0.0 <= bundle.metrics.exact_next_char_accuracy <= 1.0
    assert bundle.metrics.supervised_characters == len(_ASSIGNMENT_TEXT)
    assert bundle.metrics.supervised_positions == len(_ASSIGNMENT_TEXT) + 1

    # Lineage: generations move for a learning attempt.
    assert bundle.architecture_generation_before == LOCAL_TRAINING_ARCHITECTURE
    assert bundle.architecture_generation_after == LOCAL_TRAINING_ARCHITECTURE
    assert bundle.parameter_generation_after != bundle.parameter_generation_before
    assert bundle.optimizer_generation_after != bundle.optimizer_generation_before
    assert bundle.soul_id_after != bundle.soul_id_before

    # Identity: bundle_id is the canonical sha256 of the exact bundle.
    from runtime.field import canonical_sha256

    assert bundle.bundle_id == canonical_sha256(bundle.to_canonical_dict())

    # Evidence landed in the assignment store; the outcome stays Heart's.
    attempts = env["store"].attempts_for(bundle.assignment_id)
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.attempt_index == bundle.attempt_index == 0
    assert attempt.attempt_id == bundle.attempt_id
    assert attempt.core_id == _CORE
    assert attempt.outcome is AttemptOutcome.PENDING
    assert attempt.evidence["metrics"]["loss"] == bundle.metrics.loss
    assert attempt.evidence["learning_step_applied"] is True
    assert attempt.parameter_generation == bundle.parameter_generation_after
    assert attempt.optimizer_generation == bundle.optimizer_generation_after
    record = env["store"].get_assignment(bundle.assignment_id)
    assert record.attempt_count == 1
    assert record.status is AssignmentStatus.ACTIVE


def test_free_running_emission_does_not_read_expected_target_prefix(tmp_path) -> None:
    first = _environment(tmp_path / "first")
    second = _environment(tmp_path / "second")

    bundle_a = first["worker"].execute_attempt(
        first["binding"],
        first["assignment"],
        first["view"],
        first["soul_branch"],
        target_text="D",
    )
    bundle_b = second["worker"].execute_attempt(
        second["binding"],
        second["assignment"],
        second["view"],
        second["soul_branch"],
        target_text="XYZ",
    )

    assert bundle_a.emission.token_ids == bundle_b.emission.token_ids
    assert bundle_a.emission.characters == bundle_b.emission.characters
    assert bundle_a.emission.terminated is bundle_b.emission.terminated
    assert bundle_a.metrics.supervised_positions != bundle_b.metrics.supervised_positions


def test_optimizer_changes_parameters_and_controls_stay_byte_identical(tmp_path) -> None:
    env = _environment(tmp_path)
    worker = env["worker"]
    worker._ensure_model(env["view"].source_schema_version)
    model = worker.model
    parameter_bytes_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    buffer_bytes_before = {
        name: tensor.detach().to(device="cpu").contiguous().numpy().tobytes(order="C")
        for name, tensor in model.named_buffers()
    }

    worker.execute_attempt(
        env["binding"], env["assignment"], env["view"], env["soul_branch"]
    )

    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), parameter_bytes_before[name])
    ]
    assert changed, "optimizer step must change at least one intended parameter"
    for name, tensor in model.named_buffers():
        observed = tensor.detach().to(device="cpu").contiguous().numpy().tobytes(order="C")
        assert observed == buffer_bytes_before[name], f"control buffer {name!r} changed"
    # The frozen substrate transport bank and lift are among the controls.
    assert {"bank16", "char_lift"} <= {name for name, _ in model.named_buffers()}


def test_soul_exhale_round_trips_through_soulstore(tmp_path) -> None:
    env = _environment(tmp_path)
    worker = env["worker"]
    before = env["soul_branch"].load_head()
    bundle = worker.execute_attempt(
        env["binding"], env["assignment"], env["view"], env["soul_branch"]
    )

    # Reload through a fresh SoulStore handle over the same root: the exact
    # persisted payload and lineage survive, proving the exhale mechanism.
    reloaded_store = SoulStore(env["soul_store"].root)
    head = reloaded_store.branch(_CORE).load_head()
    assert head.soul_id == bundle.soul_id_after
    assert head.soul_id != before.soul_id
    assert head.generation == before.generation + 1
    assert head.parent_soul_id == before.soul_id
    assert head.architecture_id == LOCAL_TRAINING_ARCHITECTURE

    hot = head.layer("hot")
    array, metrics = decode_hot_payload(hot.payload)
    assert array.shape == (1, 4, 64)
    assert array.dtype.name == "float32"
    assert bool(torch.isfinite(torch.from_numpy(array)).all())
    assert metrics["loss"] == bundle.metrics.loss
    assert metrics["attempt_index"] == 0

    # The Soul transition committed exactly once under its content identity.
    receipts = reloaded_store.branch(_CORE).receipts_after(before.soul_id)
    assert len(receipts) == 1
    assert receipts[0].after_soul_id == bundle.soul_id_after
    assert receipts[0].phase == "training"


def test_sealed_payload_verifies_under_rail_auth_and_replay_fails(tmp_path) -> None:
    env = _environment(tmp_path)
    bundle = env["worker"].execute_attempt(
        env["binding"], env["assignment"], env["view"], env["soul_branch"]
    )
    wire = bundle.sealed_payload(now=2_000_000)
    envelope = json.loads(wire.decode("utf-8"))
    cache = _fresh_nonce_cache()
    verified = _verify(bundle, envelope, cache)
    assert verified["payload_sha256"] == envelope["payload_sha256"]
    # Replay of the same nonce through the same retention hook fails closed.
    with pytest.raises(RailAuthError, match="replay"):
        _verify(bundle, envelope, cache)


def test_sealed_payload_fails_closed_wrong_core_wrong_assignment_tamper_and_secret(
    tmp_path,
) -> None:
    env = _environment(tmp_path)
    bundle = env["worker"].execute_attempt(
        env["binding"], env["assignment"], env["view"], env["soul_branch"]
    )
    envelope = json.loads(bundle.sealed_payload(now=2_000_000).decode("utf-8"))
    cache = _fresh_nonce_cache()
    with pytest.raises(RailAuthError, match="core identity mismatch"):
        _verify(bundle, envelope, cache, expected_core_identity="core-impostor")
    with pytest.raises(RailAuthError, match="assignment identity mismatch"):
        _verify(bundle, envelope, cache, expected_assignment_id="a" * 64)
    with pytest.raises(RailAuthError, match="payload digest mismatch"):
        _verify(bundle, envelope, cache, payload=b"substituted bundle bytes")
    with pytest.raises(RailAuthError, match="signature mismatch"):
        _verify(bundle, envelope, cache, environ={"AXON_RAIL_SECRET": "z" * 48})


# --- evaluation attempts ------------------------------------------------------


def test_evaluation_attempt_computes_real_loss_but_never_steps(tmp_path) -> None:
    env = _environment(tmp_path, kind=AssignmentKind.EVALUATION)
    worker = env["worker"]
    worker._ensure_model(env["view"].source_schema_version)
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in worker.model.named_parameters()
    }
    bundle = worker.execute_attempt(
        env["binding"], env["assignment"], env["view"], env["soul_branch"]
    )
    # Real forward/backward evidence...
    assert 0.0 < bundle.metrics.loss < 20.0
    assert bundle.metrics.grad_norm > 0.0
    # ...but no parameter, optimizer, or Soul mutation.
    assert bundle.parameter_generation_after == bundle.parameter_generation_before
    assert bundle.optimizer_generation_after == bundle.optimizer_generation_before
    assert bundle.soul_id_after == bundle.soul_id_before
    for name, parameter in worker.model.named_parameters():
        assert torch.equal(parameter.detach(), parameters_before[name])
    attempt = env["store"].attempts_for(bundle.assignment_id)[0]
    assert attempt.evidence["learning_step_applied"] is False
    assert attempt.outcome is AttemptOutcome.PENDING


# --- determinism ---------------------------------------------------------------


def test_identical_runs_produce_identical_bundle_identity(tmp_path) -> None:
    first = _environment(tmp_path / "run-a")
    second = _environment(tmp_path / "run-b")
    bundle_a = first["worker"].execute_attempt(
        first["binding"], first["assignment"], first["view"], first["soul_branch"]
    )
    bundle_b = second["worker"].execute_attempt(
        second["binding"], second["assignment"], second["view"], second["soul_branch"]
    )
    assert bundle_a.bundle_id == bundle_b.bundle_id
    assert bundle_a.metrics == bundle_b.metrics
    assert bundle_a.emission == bundle_b.emission
    assert bundle_a.parameter_generation_after == bundle_b.parameter_generation_after
    assert bundle_a.soul_id_after == bundle_b.soul_id_after
    # Identity paths carry no RNG: only the transport nonce differs.
    assert bundle_a.sealed_payload(now=2_000_000) != bundle_b.sealed_payload(now=2_000_000)


# --- learning curve ------------------------------------------------------------


def test_loss_falls_over_repeated_learning_attempts(tmp_path) -> None:
    env = _environment(tmp_path)
    worker = env["worker"]
    binding = env["binding"]
    losses: list[float] = []
    for _ in range(5):
        bundle = worker.execute_attempt(
            binding, env["assignment"], env["view"], env["soul_branch"]
        )
        losses.append(bundle.metrics.loss)
        binding = dataclasses.replace(
            binding,
            parameter_generation=bundle.parameter_generation_after,
            optimizer_generation=bundle.optimizer_generation_after,
            soul_id=bundle.soul_id_after,
        )
    assert losses[-1] < losses[0], f"loss did not fall: {losses}"
    record = env["store"].get_assignment(env["assignment"].assignment_id)
    assert record.attempt_count == 5
    assert len(env["store"].attempts_for(env["assignment"].assignment_id)) == 5
    # Every attempt is preserved; none carries a fabricated outcome.
    assert all(
        attempt.outcome is AttemptOutcome.PENDING
        for attempt in env["store"].attempts_for(env["assignment"].assignment_id)
    )
    head = env["soul_branch"].load_head()
    assert head.generation == 5


# --- fail-closed validation ----------------------------------------------------


def test_fail_closed_on_wrong_binding_mode_and_architecture(tmp_path) -> None:
    env = _environment(tmp_path)
    reasoning = dataclasses.replace(env["binding"], mode=CoreStatus.ACTIVE)
    with pytest.raises(LocalTrainingWorkerError, match="OFFLINE_TRAINING"):
        env["worker"].execute_attempt(
            reasoning, env["assignment"], env["view"], env["soul_branch"]
        )
    foreign = dataclasses.replace(env["binding"], architecture_id="foreign-arch-v9")
    with pytest.raises(LocalTrainingWorkerError, match="architecture"):
        env["worker"].execute_attempt(
            foreign, env["assignment"], env["view"], env["soul_branch"]
        )


def test_fail_closed_on_view_and_assignment_mismatch(tmp_path) -> None:
    env = _environment(tmp_path)
    other_snapshot = _snapshot("Recite the digits forward from '0'.", tick_id=8)
    foreign_view = TrainingAttentionViewCompiler().compile(
        other_snapshot,
        core_id="core-other",
        cohort_core_ids=("core-other",),
        history_window=TrainingHistoryWindow.none(),
    )
    with pytest.raises(LocalTrainingWorkerError, match="different core"):
        env["worker"].execute_attempt(
            env["binding"], env["assignment"], foreign_view, env["soul_branch"]
        )
    mismatched_view = TrainingAttentionViewCompiler().compile(
        other_snapshot,
        core_id=_CORE,
        cohort_core_ids=(_CORE,),
        history_window=TrainingHistoryWindow.none(),
    )
    with pytest.raises(LocalTrainingWorkerError, match="field binding"):
        env["worker"].execute_attempt(
            env["binding"], env["assignment"], mismatched_view, env["soul_branch"]
        )


def test_fail_closed_on_paused_assignment_and_expired_lease(tmp_path) -> None:
    env = _environment(tmp_path)
    record = env["store"].pause(
        env["assignment"].assignment_id,
        holder_core_id=_CORE,
        expected_revision=env["assignment"].revision,
        idempotency_key="pause-1",
    )
    with pytest.raises(LocalTrainingWorkerError, match="not live"):
        env["worker"].execute_attempt(
            env["binding"], env["assignment"], env["view"], env["soul_branch"]
        )
    env["store"].resume(
        record.assignment_id,
        core_id=_CORE,
        expected_revision=record.revision,
        idempotency_key="resume-1",
        lease_seconds=10.0,
    )
    late_worker = LocalTrainingWorker(
        assignment_store=env["store"],
        clock=lambda: _T0 + 3600.0,
        rail_environ=_ENVIRON,
    )
    with pytest.raises(LocalTrainingWorkerError, match="expired"):
        late_worker.execute_attempt(
            env["binding"], env["assignment"], env["view"], env["soul_branch"]
        )


def test_fail_closed_on_foreign_soul_and_stale_parameter_lineage(tmp_path) -> None:
    env = _environment(tmp_path)
    foreign_binding = dataclasses.replace(env["binding"], soul_id="f" * 64)
    with pytest.raises(LocalTrainingWorkerError, match="soul"):
        env["worker"].execute_attempt(
            foreign_binding, env["assignment"], env["view"], env["soul_branch"]
        )

    bundle = env["worker"].execute_attempt(
        env["binding"], env["assignment"], env["view"], env["soul_branch"]
    )
    stale = dataclasses.replace(env["binding"], parameter_generation="untrained")
    with pytest.raises(LocalTrainingWorkerError, match="stale"):
        env["worker"].execute_attempt(
            stale, env["assignment"], env["view"], env["soul_branch"]
        )
    # The tracked lineage check passes once the binding carries the measured
    # generations from the previous bundle.
    fresh = dataclasses.replace(
        env["binding"],
        parameter_generation=bundle.parameter_generation_after,
        optimizer_generation=bundle.optimizer_generation_after,
        soul_id=bundle.soul_id_after,
    )
    second = env["worker"].execute_attempt(
        fresh, env["assignment"], env["view"], env["soul_branch"]
    )
    assert second.attempt_index == 1
    assert second.soul_id_before == bundle.soul_id_after
