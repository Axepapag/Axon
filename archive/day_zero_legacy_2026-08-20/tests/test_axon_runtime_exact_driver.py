from __future__ import annotations

from pathlib import Path

import torch

from cores.core import AxonCore, CoreConfig
from runtime.axon_runtime.checkpoint import (
    CheckpointSourceEvidence,
    ExactV4Checkpoint,
    HOT_ONLY,
    inference_core_state_sha256,
)
from runtime.axon_runtime.contracts import (
    CoreIdentity,
    ProjectionManifest,
    RuntimeHead,
)
from runtime.axon_runtime.core_backend import CoreBackend
from runtime.axon_runtime.engine import ObservedField
from runtime.axon_runtime.exact_driver import (
    ExactV4RuntimeDriver,
    initialize_logical_core_state,
)
from runtime.axon_runtime.soul_store import (
    PrivateRuntimeStateStore,
    SoulBlobStore,
)
from runtime.field import (
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_sha256,
)


def _tiny_checkpoint(tmp_path: Path) -> ExactV4Checkpoint:
    cfg = CoreConfig(
        d_model=16,
        n_heads=1,
        n_layers=1,
        ffn_dim=32,
        dropout=0.0,
        n_ticks=2,
        grad_ticks=1,
        soul_rows=2,
        soul_mode="act_reflect_v2",
        soul_gate_init=0.05,
        soul_hot_rows=2,
        soul_write_mode="compartments",
        n_soul_compartments=2,
        char_slot_mode=True,
        char_slot_max_slots=384,
        char_n_regions=3,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(707)
        core = AxonCore(cfg)
        soul = torch.randn(cfg.total_soul_rows(), cfg.d_model)
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    core.eval()
    return ExactV4Checkpoint(
        path=tmp_path / "synthetic-exact-v4.pt",
        checkpoint_sha256=canonical_sha256("synthetic-checkpoint"),
        core_state_sha256=inference_core_state_sha256(core.state_dict()),
        byte_length=0,
        step=1,
        cfg=cfg,
        core=core,
        initial_soul=soul,
        initial_soul_mask=torch.ones(
            cfg.total_soul_rows(),
            dtype=torch.bool,
        ),
        soul_compressor=None,
        soul_tier_status=HOT_ONLY,
        mmap_used=False,
        source_evidence=CheckpointSourceEvidence(
            optimizer_state_present=True,
            rng_state_keys=(
                "numpy_random_state",
                "python_random_state",
                "torch_rng_state",
            ),
        ),
    )


def _identity(checkpoint: ExactV4Checkpoint) -> CoreIdentity:
    return CoreIdentity(
        core_id="core-a",
        display_name="Core A",
        base_checkpoint_path=str(checkpoint.path),
        base_checkpoint_sha256=checkpoint.checkpoint_sha256,
        model_id="tiny-v4",
        core_state_sha256=checkpoint.core_state_sha256,
        lineage=("exact-v4", "runtime-clone"),
        soul_id="soul-a",
        adapter_namespace="adapter-a",
    )


def _foundation(tmp_path: Path):
    checkpoint = _tiny_checkpoint(tmp_path)
    identity = _identity(checkpoint)
    genesis = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.CONVERSATION_HISTORY: "prior context",
            LogicalRegion.USER_INPUT: "please respond",
            LogicalRegion.RESPONSE_DRAFT: "",
        }
    )
    backend = CoreBackend()
    backend.register_loaded_model("tiny-v4", checkpoint)
    logical = backend.create_logical_core(
        model_id="tiny-v4",
        core_id=identity.core_id,
        soul_id=identity.soul_id,
        committed_field_id=genesis.field_id,
        adapter_manifest={
            "schema": "axon-runtime-adapter-manifest-v1",
            "namespace": identity.adapter_namespace,
            "adapters": [],
        },
        rng_manifest={
            "schema": "axon-runtime-rng-manifest-v1",
            "stream_id": identity.core_id,
            "generation": 0,
        },
    )
    souls = SoulBlobStore(tmp_path / "souls")
    private = PrivateRuntimeStateStore(tmp_path / "private")
    initial = initialize_logical_core_state(
        logical=logical,
        identity=identity,
        genesis=genesis,
        soul_store=souls,
        private_state_store=private,
    )
    driver = ExactV4RuntimeDriver(
        backend=backend,
        soul_store=souls,
        private_state_store=private,
        identities=(identity,),
    )
    return checkpoint, identity, genesis, backend, logical, initial, driver


def _head(
    *,
    snapshot: SharedFieldSnapshot,
    manifest,
    generation: int,
    tick_seq: int,
) -> RuntimeHead:
    projection = ProjectionManifest(
        generation=generation,
        tick_seq=tick_seq,
        field_id=snapshot.field_id,
        field_hash=snapshot.canonical_hash,
        core_state_manifest_ids=((manifest.core_id, manifest.manifest_id),),
        role_index=0,
        role_assignment_hash=canonical_sha256("assignment"),
        parent_projection_id=None,
    )
    return RuntimeHead(
        generation=generation,
        tick_seq=tick_seq,
        snapshot=snapshot,
        role_index=0,
        role_assignment_hash=projection.role_assignment_hash,
        projection_manifest_id=projection.projection_id,
        core_state_manifests=(manifest,),
    )


def test_exact_driver_persists_before_install_and_installs_after_commit(
    tmp_path: Path,
) -> None:
    (
        checkpoint,
        identity,
        genesis,
        backend,
        logical,
        initial,
        driver,
    ) = _foundation(tmp_path)
    working = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.CONVERSATION_HISTORY: "prior context",
            LogicalRegion.USER_INPUT: "please respond",
            LogicalRegion.SITUATION_AWARENESS: "same-tick ingress sealed",
            LogicalRegion.RESPONSE_DRAFT: "",
        },
        tick_id=genesis.tick_id,
        parent_field_id=genesis.field_id,
    )
    turn = driver.stage_turn(
        core_id=identity.core_id,
        observation=ObservedField(
            snapshot=working,
            ancestor_field_ids=(genesis.field_id,),
        ),
        canonical_base_field_id=genesis.field_id,
        tick_seq=0,
        target_region=LogicalRegion.RESPONSE_DRAFT,
    )
    output_field_id = canonical_sha256(
        ["output", turn.proposal.text, genesis.field_id]
    )
    prepared = driver.prepare_commit(
        turn=turn,
        accepted_text=turn.proposal.text,
        output_field_id=output_field_id,
        runtime_generation=1,
        next_tick_seq=1,
        prior_manifest=initial,
    )

    assert logical.core_generation == 0
    assert logical.soul_generation == 0
    assert logical.committed_field_id == genesis.field_id
    assert prepared.manifest.core_state_sha256 == checkpoint.core_state_sha256
    assert (
        driver.private_state_store.load(
            prepared.manifest.cursor_state_sha256
        ).committed_field_id
        == output_field_id
    )

    driver.finalize_commit(prepared)

    assert logical.core_generation == 1
    assert logical.soul_generation == 1
    assert logical.committed_field_id == output_field_id


def test_reconcile_head_replaces_stale_or_missing_live_private_state(
    tmp_path: Path,
) -> None:
    (
        checkpoint,
        identity,
        genesis,
        backend,
        logical,
        initial,
        driver,
    ) = _foundation(tmp_path)
    working = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.USER_INPUT: "hello",
            LogicalRegion.RESPONSE_DRAFT: "",
        },
        tick_id=0,
        parent_field_id=genesis.field_id,
    )
    turn = driver.stage_turn(
        core_id=identity.core_id,
        observation=ObservedField(
            snapshot=working,
            ancestor_field_ids=(genesis.field_id,),
        ),
        canonical_base_field_id=genesis.field_id,
        tick_seq=0,
        target_region=LogicalRegion.RESPONSE_DRAFT,
    )
    output = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.USER_INPUT: "hello",
            LogicalRegion.RESPONSE_DRAFT: turn.proposal.text,
        },
        tick_id=1,
        parent_field_id=genesis.field_id,
    )
    prepared = driver.prepare_commit(
        turn=turn,
        accepted_text=turn.proposal.text,
        output_field_id=output.field_id,
        runtime_generation=1,
        next_tick_seq=1,
        prior_manifest=initial,
    )

    # Simulate process loss after SQLite committed but before RAM installation:
    # the backend still holds generation zero.
    assert backend.logical_core(identity.core_id).core_generation == 0
    driver.reconcile_head(
        _head(
            snapshot=output,
            manifest=prepared.manifest,
            generation=1,
            tick_seq=1,
        )
    )

    restored = backend.logical_core(identity.core_id)
    assert restored is not logical
    assert restored.shared_model.checkpoint is checkpoint
    assert restored.core_generation == 1
    assert restored.soul_generation == 1
    assert restored.committed_field_id == output.field_id
    assert (
        restored.last_candidate_manifest_id
        == prepared.commit_token.prepared_state.receipt.action.manifest.manifest_id
    )

