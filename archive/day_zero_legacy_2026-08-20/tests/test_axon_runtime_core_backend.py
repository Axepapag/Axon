from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from cores.core import AxonCore, CoreConfig
from runtime.axon_runtime.checkpoint import file_sha256
from runtime.axon_runtime.core_backend import (
    CandidateValidationError,
    CoreBackend,
    install_validated_candidate,
    persist_candidate_soul,
    persist_committed_runtime_state,
    prepare_persisted_runtime_state,
    validate_candidate_install,
)
from runtime.axon_runtime.soul_store import (
    PRIVATE_RUNTIME_STATE_MAGIC,
    PrivateRuntimeStateStore,
    SoulBlobCorruptionError,
    SoulBlobStore,
    decode_private_runtime_state,
)
from runtime.field import (
    FieldViewCursor,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
)


def write_tiny_checkpoint(tmp_path: Path) -> tuple[Path, str]:
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
        torch.manual_seed(109)
        core = AxonCore(cfg)
        soul = torch.randn(1, cfg.total_soul_rows(), cfg.d_model)
    path = tmp_path / "runtime-tiny.pt"
    torch.save(
        {
            "checkpoint_schema": "axon_exact_v4_checkpoint_v1",
            "step": 33,
            "cfg": cfg.to_dict(),
            "core_state": deepcopy(core.state_dict()),
            "optimizer_state": {"state": {}, "param_groups": []},
            "rng_state": {
                "python_random_state": ("source",),
                "numpy_random_state": ("source",),
                "torch_rng_state": torch.arange(8, dtype=torch.uint8),
            },
            "soul_state": {
                "soul": soul,
                "soul_mask": torch.ones(
                    1,
                    cfg.total_soul_rows(),
                    dtype=torch.bool,
                ),
            },
        },
        path,
    )
    return path, file_sha256(path)


def make_backend_and_clones(tmp_path: Path):
    checkpoint, digest = write_tiny_checkpoint(tmp_path)
    canonical = SharedFieldSnapshot.from_texts(
        {
            "conversation_history": "prior context",
            "user_input": "please answer",
            "structured_knowledge": "known fact",
            "response_draft": "draft",
        },
        tick_id=5,
    )
    observed = SharedFieldSnapshot.from_texts(
        {
            "conversation_history": "prior context",
            "user_input": "please answer",
            "structured_knowledge": "known fact",
            "situation_awareness": "sealed ingress",
            "response_draft": "draft",
        },
        tick_id=5,
        parent_field_id=canonical.field_id,
    )
    backend = CoreBackend()
    first_load = backend.load_model(
        model_id="tiny-exact-v4",
        checkpoint_path=checkpoint,
        checkpoint_sha256=digest,
    )
    second_load = backend.load_model(
        model_id="tiny-exact-v4",
        checkpoint_path=checkpoint,
        checkpoint_sha256=digest,
    )
    clone_a = backend.create_logical_core(
        model_id="tiny-exact-v4",
        core_id="core-a",
        soul_id="soul-a",
        committed_field_id=canonical.field_id,
        adapter_manifest={
            "schema": "axon-runtime-adapter-manifest-v1",
            "adapters": [],
        },
        rng_manifest={
            "schema": "axon-runtime-rng-manifest-v1",
            "stream_id": "a",
            "generation": 0,
        },
    )
    clone_b = backend.create_logical_core(
        model_id="tiny-exact-v4",
        core_id="core-b",
        soul_id="soul-b",
        committed_field_id=canonical.field_id,
        adapter_manifest={
            "schema": "axon-runtime-adapter-manifest-v1",
            "adapters": [],
        },
        rng_manifest={
            "schema": "axon-runtime-rng-manifest-v1",
            "stream_id": "b",
            "generation": 0,
        },
    )
    return (
        backend,
        canonical,
        observed,
        first_load,
        second_load,
        clone_a,
        clone_b,
    )


def prepare_kwargs(
    canonical: SharedFieldSnapshot,
    observed: SharedFieldSnapshot,
) -> dict[str, object]:
    return {
        "canonical_base_field_id": canonical.field_id,
        "expected_observed_field_id": observed.field_id,
        "expected_observed_tick_id": observed.tick_id,
    }


def numpy_rng_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_model_is_shared_once_while_logical_state_is_private(tmp_path: Path) -> None:
    _, _, _, first, second, clone_a, clone_b = make_backend_and_clones(tmp_path)
    assert first is second
    assert clone_a.core is clone_b.core
    param_a = next(clone_a.core.parameters())
    param_b = next(clone_b.core.parameters())
    assert param_a.data_ptr() == param_b.data_ptr()
    assert clone_a.soul.data_ptr() != clone_b.soul.data_ptr()
    assert clone_a.soul_mask.data_ptr() != clone_b.soul_mask.data_ptr()
    assert clone_a.cursor is not clone_b.cursor
    assert clone_a.adapter_manifest is not clone_b.adapter_manifest
    assert clone_a.rng_manifest is not clone_b.rng_manifest
    clone_a.soul[0, 0] += 1
    clone_a.adapter_manifest["adapters"].append("private-a")
    clone_a.rng_manifest["generation"] = 1
    assert not torch.equal(clone_a.soul, clone_b.soul)
    assert clone_b.adapter_manifest["adapters"] == []
    assert clone_b.rng_manifest["generation"] == 0


def test_prepare_action_is_pure_repeatable_and_restores_rng_and_flags(
    tmp_path: Path,
) -> None:
    _, canonical, observed, _, _, logical, _ = make_backend_and_clones(
        tmp_path
    )
    core = logical.core
    core.train()
    modules = tuple(core.modules())
    modules[-1].eval()
    flags_before = tuple(module.training for module in modules)
    parameters_before = {
        name: tensor.detach().clone()
        for name, tensor in core.state_dict().items()
    }
    soul_before = logical.soul.clone()
    mask_before = logical.soul_mask.clone()
    cursor_before = logical.cursor
    adapters_before = deepcopy(logical.adapter_manifest)
    rng_manifest_before = deepcopy(logical.rng_manifest)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    first = logical.prepare_action(
        observed,
        tick_index=9,
        **prepare_kwargs(canonical, observed),
    )
    second = logical.prepare_action(
        observed,
        tick_index=9,
        **prepare_kwargs(canonical, observed),
    )

    assert first.proposal == second.proposal
    assert first.manifest.manifest_id == second.manifest.manifest_id
    assert first.candidate_blob.data == second.candidate_blob.data
    assert torch.equal(first.candidate_soul, second.candidate_soul)
    assert torch.equal(logical.soul, soul_before)
    assert torch.equal(logical.soul_mask, mask_before)
    assert logical.cursor == cursor_before == FieldViewCursor()
    assert logical.adapter_manifest == adapters_before
    assert logical.rng_manifest == rng_manifest_before
    assert first.candidate_soul.data_ptr() != logical.soul.data_ptr()
    assert tuple(module.training for module in modules) == flags_before
    assert random.getstate() == python_before
    assert numpy_rng_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)
    for name, tensor in core.state_dict().items():
        assert torch.equal(tensor, parameters_before[name])


def test_acceptance_validation_is_exact_and_persistence_is_separate(
    tmp_path: Path,
) -> None:
    _, canonical, observed, shared, _, logical, _ = make_backend_and_clones(
        tmp_path
    )
    action = logical.prepare_action(
        observed,
        **prepare_kwargs(canonical, observed),
    )
    final_text = action.proposal.text

    with pytest.raises(CandidateValidationError, match="core generation"):
        validate_candidate_install(
            logical,
            action,
            expected_core_generation=1,
            expected_soul_generation=0,
            canonical_base_field_id=canonical.field_id,
            expected_observed_field_id=observed.field_id,
            expected_observed_tick_id=observed.tick_id,
            final_accepted_text=final_text,
        )
    with pytest.raises(CandidateValidationError, match="base field"):
        validate_candidate_install(
            logical,
            action,
            expected_core_generation=0,
            expected_soul_generation=0,
            canonical_base_field_id="another-field",
            expected_observed_field_id=observed.field_id,
            expected_observed_tick_id=observed.tick_id,
            final_accepted_text=final_text,
        )
    with pytest.raises(CandidateValidationError, match="accepted text"):
        validate_candidate_install(
            logical,
            action,
            expected_core_generation=0,
            expected_soul_generation=0,
            canonical_base_field_id=canonical.field_id,
            expected_observed_field_id=observed.field_id,
            expected_observed_tick_id=observed.tick_id,
            final_accepted_text=final_text + "x",
        )

    receipt = validate_candidate_install(
        logical,
        action,
        expected_core_generation=0,
        expected_soul_generation=0,
        canonical_base_field_id=canonical.field_id,
        expected_observed_field_id=observed.field_id,
        expected_observed_tick_id=observed.tick_id,
        final_accepted_text=final_text,
    )
    store = SoulBlobStore(tmp_path / "runtime-souls")
    state_store = PrivateRuntimeStateStore(tmp_path / "runtime-states")
    live_soul = logical.soul.clone()
    live_mask = logical.soul_mask.clone()
    live_cursor = logical.cursor
    live_anchors = logical.cursor_anchors
    live_field = logical.committed_field_id
    assert logical.core_generation == 0
    prepared_state = prepare_persisted_runtime_state(
        soul_store=store,
        state_store=state_store,
        logical=logical,
        receipt=receipt,
        intended_committed_field_id="accepted-field-id",
    )
    stored = prepared_state.stored_soul
    stored_state = prepared_state.stored_private_state
    assert prepared_state.soul_state_sha256 == action.candidate_blob.sha256
    assert prepared_state.cursor_state_sha256 == stored_state.sha256
    assert logical.core_generation == 0
    assert logical.soul_generation == 0
    assert logical.committed_field_id == live_field
    assert logical.cursor == live_cursor
    assert logical.cursor_anchors == live_anchors
    assert torch.equal(logical.soul, live_soul)
    assert torch.equal(logical.soul_mask, live_mask)
    with pytest.raises(RuntimeError, match="simulated DB failure"):
        raise RuntimeError("simulated DB failure")
    assert logical.core_generation == 0
    assert logical.committed_field_id == live_field

    install_validated_candidate(
        logical,
        receipt,
        committed_field_id="accepted-field-id",
    )
    assert logical.core_generation == 1
    assert logical.soul_generation == 1
    assert logical.committed_field_id == "accepted-field-id"
    assert logical.soul.data_ptr() != action.candidate_soul.data_ptr()
    verified_state = persist_committed_runtime_state(
        state_store,
        logical,
        receipt,
        stored,
    )
    assert verified_state.sha256 == stored_state.sha256
    assert stored_state.cursor_state_sha256 == stored_state.sha256
    decoded_state = state_store.load(stored_state.sha256)
    assert decoded_state.soul_blob_sha256 == stored.sha256
    assert decoded_state.candidate_manifest_id == action.manifest.manifest_id
    assert decoded_state.cursor == logical.cursor
    assert (
        decoded_state.cursor_anchor_sha256
        == logical.cursor_anchors.anchor_sha256
    )
    assert decoded_state.adapter_manifest == logical.adapter_manifest
    assert decoded_state.rng_manifest == logical.rng_manifest

    restarted = CoreBackend()
    restarted.load_model(
        model_id=logical.model_id,
        checkpoint_path=shared.checkpoint.path,
        checkpoint_sha256=shared.checkpoint.checkpoint_sha256,
    )
    restored = restarted.restore_logical_core(
        cursor_state_sha256=stored_state.sha256,
        state_store=state_store,
        soul_store=store,
    )
    assert restored.core_generation == logical.core_generation
    assert restored.soul_generation == logical.soul_generation
    assert restored.committed_field_id == logical.committed_field_id
    assert restored.cursor == logical.cursor
    assert restored.cursor_anchors == logical.cursor_anchors
    assert restored.adapter_manifest == logical.adapter_manifest
    assert restored.rng_manifest == logical.rng_manifest
    assert restored.last_candidate_manifest_id == action.manifest.manifest_id
    assert torch.equal(restored.soul, logical.soul)
    assert restored.soul.data_ptr() != logical.soul.data_ptr()

    payload = json.loads(
        stored_state.path.read_bytes()[len(PRIVATE_RUNTIME_STATE_MAGIC) :]
    )
    payload["cursor_anchor_manifest"]["anchors"][0][
        "character_length"
    ] += 1
    tampered_anchor = (
        PRIVATE_RUNTIME_STATE_MAGIC + canonical_json_bytes(payload)
    )
    tampered_hash = hashlib.sha256(tampered_anchor).hexdigest()
    with pytest.raises(SoulBlobCorruptionError, match="anchor"):
        decode_private_runtime_state(tampered_anchor, tampered_hash)

    corrupt_state = bytearray(stored_state.path.read_bytes())
    corrupt_state[-1] ^= 1
    stored_state.path.write_bytes(bytes(corrupt_state))
    with pytest.raises(SoulBlobCorruptionError):
        state_store.load(stored_state.sha256)

    with pytest.raises(CandidateValidationError, match="generation"):
        validate_candidate_install(
            logical,
            action,
            expected_core_generation=0,
            expected_soul_generation=0,
            canonical_base_field_id=canonical.field_id,
            expected_observed_field_id=observed.field_id,
            expected_observed_tick_id=observed.tick_id,
            final_accepted_text=final_text,
        )


def test_modified_or_nonfinite_candidate_is_rejected(tmp_path: Path) -> None:
    _, canonical, observed, _, _, logical, _ = make_backend_and_clones(tmp_path)
    action = logical.prepare_action(
        observed,
        **prepare_kwargs(canonical, observed),
    )
    action.candidate_soul[0, 0] += 1
    with pytest.raises(CandidateValidationError, match="modified"):
        logical.validate_candidate(
            action,
            expected_core_generation=0,
            expected_soul_generation=0,
            canonical_base_field_id=canonical.field_id,
            expected_observed_field_id=observed.field_id,
            expected_observed_tick_id=observed.tick_id,
            final_accepted_text=action.proposal.text,
        )

    action = logical.prepare_action(
        observed,
        **prepare_kwargs(canonical, observed),
    )
    action.candidate_soul[0, 0] = float("nan")
    with pytest.raises(CandidateValidationError, match="candidate"):
        logical.validate_candidate(
            action,
            expected_core_generation=0,
            expected_soul_generation=0,
            canonical_base_field_id=canonical.field_id,
            expected_observed_field_id=observed.field_id,
            expected_observed_tick_id=observed.tick_id,
            final_accepted_text=action.proposal.text,
        )


def test_authenticated_overlay_and_lagging_private_field_are_distinct(
    tmp_path: Path,
) -> None:
    backend, canonical, observed, _, _, _, _ = make_backend_and_clones(
        tmp_path
    )
    lagging = backend.create_logical_core(
        model_id="tiny-exact-v4",
        core_id="core-lagging",
        soul_id="soul-lagging",
        committed_field_id="private-head-h0",
    )

    with pytest.raises(CandidateValidationError, match="descendant"):
        lagging.prepare_action(
            observed,
            canonical_base_field_id="stale-canonical-head",
            expected_observed_field_id=observed.field_id,
            expected_observed_tick_id=observed.tick_id,
        )
    with pytest.raises(CandidateValidationError, match="field/tick"):
        lagging.prepare_action(
            observed,
            canonical_base_field_id=canonical.field_id,
            expected_observed_field_id="modified-observed-id",
            expected_observed_tick_id=observed.tick_id,
        )

    action = lagging.prepare_action(
        observed,
        **prepare_kwargs(canonical, observed),
    )
    assert action.manifest.prior_private_field_id == "private-head-h0"
    assert action.manifest.canonical_base_field_id == canonical.field_id
    assert action.manifest.observed_field_id == observed.field_id
    assert action.manifest.observed_tick_id == observed.tick_id
    assert action.manifest.observed_ancestor_field_ids == (
        canonical.field_id,
    )

    foreign = SharedFieldSnapshot.from_texts(
        {
            "user_input": "please answer",
            "situation_awareness": "foreign overlay",
        },
        tick_id=observed.tick_id,
        parent_field_id=canonical.field_id,
    )
    with pytest.raises(CandidateValidationError, match="observed"):
        lagging.validate_candidate(
            action,
            expected_core_generation=0,
            expected_soul_generation=0,
            canonical_base_field_id=canonical.field_id,
            expected_observed_field_id=foreign.field_id,
            expected_observed_tick_id=foreign.tick_id,
            final_accepted_text=action.proposal.text,
        )

    receipt = lagging.validate_candidate(
        action,
        expected_core_generation=0,
        expected_soul_generation=0,
        canonical_base_field_id=canonical.field_id,
        expected_observed_field_id=observed.field_id,
        expected_observed_tick_id=observed.tick_id,
        final_accepted_text=action.proposal.text,
    )
    install_validated_candidate(
        lagging,
        receipt,
        committed_field_id="final-head-h4",
    )
    assert lagging.committed_field_id == "final-head-h4"
    with pytest.raises(CandidateValidationError, match="generation"):
        lagging.validate_candidate(
            action,
            expected_core_generation=0,
            expected_soul_generation=0,
            canonical_base_field_id=canonical.field_id,
            expected_observed_field_id=observed.field_id,
            expected_observed_tick_id=observed.tick_id,
            final_accepted_text=action.proposal.text,
        )


def test_cursor_anchors_retain_append_prefix_and_reset_replaced_source(
    tmp_path: Path,
) -> None:
    backend, canonical, _, _, _, logical, _ = make_backend_and_clones(tmp_path)
    observed_long = SharedFieldSnapshot.from_texts(
        {
            "conversation_history": "a" * 600,
            "user_input": "u" * 100,
        },
        tick_id=canonical.tick_id,
        parent_field_id=canonical.field_id,
    )
    first = logical.prepare_action(
        observed_long,
        **prepare_kwargs(canonical, observed_long),
    )
    receipt = logical.validate_candidate(
        first,
        expected_core_generation=0,
        expected_soul_generation=0,
        canonical_base_field_id=canonical.field_id,
        expected_observed_field_id=observed_long.field_id,
        expected_observed_tick_id=observed_long.tick_id,
        final_accepted_text=first.proposal.text,
    )
    install_validated_candidate(
        logical,
        receipt,
        committed_field_id="private-after-first-page",
    )
    assert logical.cursor.page_index > 0
    assert logical.cursor_anchors.anchors
    live_cursor_hash = canonical_sha256(logical.cursor.to_canonical_dict())

    next_canonical = SharedFieldSnapshot.from_texts(
        {"task_state": "current head"},
        tick_id=canonical.tick_id + 1,
        parent_field_id=canonical.field_id,
    )
    appended = SharedFieldSnapshot.from_texts(
        {
            "conversation_history": "a" * 600 + "b" * 20,
            "user_input": "u" * 100 + "v" * 10,
        },
        tick_id=next_canonical.tick_id,
        parent_field_id=next_canonical.field_id,
    )
    append_action = logical.prepare_action(
        appended,
        **prepare_kwargs(next_canonical, appended),
    )
    assert append_action.manifest.cursor_before_sha256 == live_cursor_hash
    assert append_action.manifest.rebased_cursor_sha256 == live_cursor_hash

    replaced = SharedFieldSnapshot.from_texts(
        {
            "conversation_history": "z" * 600,
            "user_input": "changed",
        },
        tick_id=next_canonical.tick_id,
        parent_field_id=next_canonical.field_id,
    )
    replace_action = logical.prepare_action(
        replaced,
        **prepare_kwargs(next_canonical, replaced),
    )
    empty_hash = canonical_sha256(FieldViewCursor().to_canonical_dict())
    assert replace_action.manifest.cursor_before_sha256 == live_cursor_hash
    assert replace_action.manifest.rebased_cursor_sha256 == empty_hash
    assert (
        replace_action.manifest.cursor_anchor_after_sha256
        != append_action.manifest.cursor_anchor_after_sha256
    )
