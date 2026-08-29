from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.soul import (
    SoulIntegrityError,
    SoulLayer,
    SoulPromotion,
    SoulStore,
    SoulTemperature,
    SoulTransition,
)
from runtime.trainer import CandidateSoulWorkspace


def _transition(
    before,
    *,
    tick_uid: str = "tick-1",
    request_id: str = "request-1",
    phase: str = "first",
    payload: bytes = b"hot successor",
) -> SoulTransition:
    return SoulTransition(
        core_id=before.core_id,
        architecture_id=before.architecture_id,
        parameter_generation=before.parameter_generation,
        before_soul_id=before.soul_id,
        before_generation=before.generation,
        tick_uid=tick_uid,
        request_id=request_id,
        phase=phase,
        updates=(SoulLayer(SoulTemperature.HOT, payload, tensor_layout="test-private-v1"),),
    )


def test_opaque_layers_roundtrip_exact_bytes_without_configured_size_ceiling(tmp_path: Path) -> None:
    store = SoulStore.active(tmp_path)
    initial = store.ensure_core(
        core_id="core64",
        architecture_id="d64-candidate-a",
        parameter_generation="g0",
    )
    payload = bytes(range(256)) * 4097
    receipt = store.branch("core64").commit_transition(
        _transition(initial, payload=payload)
    )
    reloaded = store.branch("core64").load_snapshot(receipt.after_soul_id)
    assert reloaded.layer(SoulTemperature.HOT).payload == payload
    assert reloaded.generation == 1

    path = store.branch("core64").snapshots_dir / f"{reloaded.soul_id}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["layers"][0]["payload_base64"] = "AA=="
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SoulIntegrityError, match="payload identity"):
        store.branch("core64").load_head()


def test_runtime_transition_requires_hot_and_colder_layers_require_vetted_promotion() -> None:
    store = SoulStore()
    before = store.branch("never-created")
    assert not before.initialized
    with pytest.raises(ValueError, match="hot-layer"):
        SoulTransition(
            core_id="core64",
            architecture_id="d64",
            parameter_generation="g0",
            before_soul_id="0" * 64,
            before_generation=0,
            tick_uid="tick",
            request_id="request",
            phase="first",
            updates=(SoulLayer(SoulTemperature.WARM, b"memory"),),
        )

    with pytest.raises(ValueError, match="successful outcome evidence"):
        SoulPromotion(
            source=SoulTemperature.WARM,
            target=SoulTemperature.COLD,
            source_payload_sha256="0" * 64,
            outcome_quality="observed",
            repeated_observations=2,
        )

    promotion = SoulPromotion(
        source=SoulTemperature.COLD,
        target=SoulTemperature.DEEP_COLD,
        source_payload_sha256="0" * 64,
        outcome_quality="endorsed",
        evidence_ids=("human-endorsement",),
        repeated_observations=3,
        validation_ids=("heldout-replay",),
    )
    assert promotion.target is SoulTemperature.DEEP_COLD


def test_external_commit_transition_recovers_only_with_canonical_binding(tmp_path: Path) -> None:
    store = SoulStore.active(tmp_path)
    before = store.ensure_core(
        core_id="core64",
        architecture_id="d64",
        parameter_generation="g0",
    )
    branch = store.branch("core64")
    transition = _transition(before, phase="consolidated")
    prepared = branch.prepare_transition(transition, requires_external_commit=True)
    assert branch.load_head().soul_id == before.soul_id
    assert branch.recover() == ()
    recovered = branch.recover(
        external_commit_bindings={transition.transition_id: "canonical-field:" + "f" * 64}
    )
    assert len(recovered) == 1
    assert branch.load_head().soul_id == prepared.soul_id
    assert branch.recover() == ()


def test_candidate_soul_is_exact_fork_and_live_advancement_requires_replay(tmp_path: Path) -> None:
    live = SoulStore.active(tmp_path)
    initial = live.ensure_core(
        core_id="core64",
        architecture_id="d64-candidate-a",
        parameter_generation="g0",
    )
    first = live.branch("core64").commit_transition(_transition(initial))
    workspace = CandidateSoulWorkspace(tmp_path)
    manifest = workspace.prepare(
        candidate_id="candidate-a",
        core_id="core64",
        runtime_episode_session_id="session-1",
        whole_episode_split="train",
        soul_trajectory_ids=("a" * 64,),
        candidate_parameter_generation="g1-candidate",
    )
    candidate = workspace.branch("candidate-a", "core64")
    assert candidate.load_head().soul_id == manifest.candidate_initial_soul_id
    assert candidate.load_head().soul_id != first.after_soul_id
    assert candidate.load_head().layers == live.branch("core64").load_head().layers
    assert candidate.load_head().parameter_generation == "g1-candidate"
    ready = workspace.promotion_plan(manifest)
    assert ready.status == "exact_base_ready_for_gates"
    assert ready.replay_receipt_ids == ()

    live_head = live.branch("core64").load_head()
    second = live.branch("core64").commit_transition(
        _transition(
            live_head,
            tick_uid="tick-2",
            request_id="request-2",
            phase="refined",
            payload=b"new lived experience",
        )
    )
    blocked = workspace.promotion_plan(manifest)
    assert blocked.status == "live_replay_required"
    assert blocked.replay_receipt_ids == (second.receipt_id,)
    assert candidate.load_head().soul_id == manifest.candidate_initial_soul_id
