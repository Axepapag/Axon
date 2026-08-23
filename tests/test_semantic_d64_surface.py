from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from runtime.axon_runtime import CanonicalD64RuntimeAdapter
from runtime.field import (
    CompiledD64DualSurface,
    D64FieldCompiler,
    D64SemanticBindingError,
    D64SemanticSurface,
    D64SemanticSurfaceCompiler,
    FieldSpan,
    LogicalRegion,
    RegionMaskPolicy,
    RegionState,
    SharedFieldSnapshot,
    StaleD64SemanticSurfaceError,
)
from runtime.heart import FrozenTickImage, HeartbeatClock, StaleRailBindingError


def _snapshot() -> SharedFieldSnapshot:
    return SharedFieldSnapshot(
        tick_id=7,
        regions=(
            RegionState(
                name=LogicalRegion.USER_INPUT,
                spans=(
                    FieldSpan(
                        span_id="user:a",
                        text="Alpha beta. ",
                        kind="user_text",
                        source="test-user",
                        provenance="unit",
                    ),
                    FieldSpan(
                        span_id="user:b",
                        text="Gamma delta.",
                        kind="user_text",
                        source="test-user",
                        provenance="unit",
                    ),
                ),
            ),
            RegionState(
                name=LogicalRegion.CORTEX,
                spans=(
                    FieldSpan(
                        span_id="knowledge:fastapi",
                        text="FastAPI exposes endpoint.",
                        kind="dormant_fact",
                        source="dormant",
                        provenance="exact-jsonl",
                        container_refs=("container:fastapi",),
                        edge_refs=("edge:fastapi:endpoint",),
                    ),
                ),
            ),
        ),
    )


def test_semantic_surface_is_grounded_and_preserves_exact_refs() -> None:
    snapshot = _snapshot()
    exact = D64FieldCompiler().compile(snapshot)
    surface = D64SemanticSurfaceCompiler().compile(snapshot, exact)

    exact.verify_roundtrip(snapshot)
    surface.verify_grounding(snapshot, exact)
    assert surface.source_rail_id == exact.rail_id
    assert surface.slot_count > 0

    word_texts = {
        surface.slot_text(slot, exact)
        for slot in surface.slots
        if slot.slot_kind == "word"
    }
    assert {"Alpha", "beta", "Gamma", "delta", "FastAPI", "exposes", "endpoint"} <= word_texts

    grounded = next(
        slot
        for slot in surface.slots
        if slot.slot_kind == "field_span" and slot.source_kind == "dormant_fact"
    )
    assert surface.slot_text(grounded, exact) == "FastAPI exposes endpoint."
    assert grounded.container_refs == ("container:fastapi",)
    assert grounded.edge_refs == ("edge:fastapi:endpoint",)
    assert grounded.features64.shape == (64,)
    assert np.isfinite(grounded.features64).all()
    assert not grounded.features64.flags.writeable


def test_semantic_surface_is_deterministic_for_same_exact_rail() -> None:
    snapshot = _snapshot()
    exact = D64FieldCompiler().compile(snapshot)
    compiler = D64SemanticSurfaceCompiler()

    first = compiler.compile(snapshot, exact)
    second = compiler.compile(snapshot, exact)

    assert first.surface_id == second.surface_id
    assert [slot.slot_id for slot in first.slots] == [slot.slot_id for slot in second.slots]
    for left, right in zip(first.slots, second.slots):
        np.testing.assert_array_equal(left.features64, right.features64)


def test_masked_view_skips_partially_unattended_structural_slots() -> None:
    snapshot = _snapshot()
    exact_all = D64FieldCompiler().compile(snapshot)
    masked = D64FieldCompiler().compile(
        snapshot,
        region_masks={LogicalRegion.USER_INPUT: RegionMaskPolicy("last_n_spans", 1)},
    )
    compiler = D64SemanticSurfaceCompiler()
    all_surface = compiler.compile(snapshot, exact_all)
    masked_surface = compiler.compile(snapshot, masked)

    assert exact_all.source_field_id == masked.source_field_id == snapshot.field_id
    assert exact_all.rail_id != masked.rail_id
    assert all_surface.surface_id != masked_surface.surface_id

    user_slots = [slot for slot in masked_surface.slots if slot.region is LogicalRegion.USER_INPUT]
    second_span_start = len("Alpha beta. ")
    assert user_slots
    assert all(slot.region_start >= second_span_start for slot in user_slots)
    assert "Alpha" not in {masked_surface.slot_text(slot, masked) for slot in user_slots}
    assert "Gamma" in {masked_surface.slot_text(slot, masked) for slot in user_slots}


def test_semantic_surface_fails_closed_for_stale_exact_rail() -> None:
    snapshot = _snapshot()
    exact = D64FieldCompiler().compile(snapshot)
    surface = D64SemanticSurfaceCompiler().compile(snapshot, exact)

    other = SharedFieldSnapshot.from_texts(
        {LogicalRegion.USER_INPUT: "different field"}, tick_id=8
    )
    other_exact = D64FieldCompiler().compile(other)

    with pytest.raises(StaleD64SemanticSurfaceError):
        surface.assert_fresh(other, other_exact)
    with pytest.raises(D64SemanticBindingError):
        CompiledD64DualSurface(exact=other_exact, semantic=surface)


def test_grounding_rejects_lane_reference_tampering() -> None:
    snapshot = _snapshot()
    exact = D64FieldCompiler().compile(snapshot)
    surface = D64SemanticSurfaceCompiler().compile(snapshot, exact)
    slot = next(item for item in surface.slots if item.slot_kind == "word" and surface.slot_text(item, exact) == "beta")

    tampered_refs = tuple(ref + 1 for ref in slot.exact_lane_refs)
    tampered_slot = replace(slot, exact_lane_refs=tampered_refs)
    tampered_slots = tuple(tampered_slot if item.slot_id == slot.slot_id else item for item in surface.slots)
    tampered_slots = tuple(
        sorted(
            tampered_slots,
            key=lambda item: (
                list(LogicalRegion).index(item.region),
                item.region_start,
                item.region_end,
                item.slot_kind,
                item.source_kind,
                item.slot_id,
            ),
        )
    )
    tampered_surface = D64SemanticSurface(
        source_field_id=surface.source_field_id,
        source_tick_id=surface.source_tick_id,
        source_rail_id=surface.source_rail_id,
        feature_generation=surface.feature_generation,
        slots=tampered_slots,
    )
    with pytest.raises(D64SemanticBindingError):
        tampered_surface.verify_grounding(snapshot, exact)


def test_frozen_tick_binds_semantic_surface_generation_and_identity() -> None:
    snapshot = _snapshot()
    exact = D64FieldCompiler().compile(snapshot)
    surface = D64SemanticSurfaceCompiler().compile(snapshot, exact)
    identity = HeartbeatClock().open_tick(snapshot)

    image = FrozenTickImage.from_compiled(
        identity,
        {64: exact},
        semantic_surfaces={64: surface},
    )
    binding = image.require_rail(64)
    assert binding.semantic_surface_id == surface.surface_id
    assert binding.semantic_generation == surface.feature_generation
    assert binding.semantic_slot_count == surface.slot_count

    exact_only = FrozenTickImage.from_compiled(identity, {64: exact})
    assert exact_only.require_rail(64).semantic_surface_id is None
    assert exact_only.image_id != image.image_id


def test_frozen_tick_rejects_semantic_surface_from_another_exact_view() -> None:
    snapshot = _snapshot()
    exact = D64FieldCompiler().compile(snapshot)
    masked = D64FieldCompiler().compile(
        snapshot,
        region_masks={LogicalRegion.USER_INPUT: RegionMaskPolicy("last_n_spans", 1)},
    )
    masked_surface = D64SemanticSurfaceCompiler().compile(snapshot, masked)
    identity = HeartbeatClock().open_tick(snapshot)

    with pytest.raises(StaleRailBindingError):
        FrozenTickImage.from_compiled(
            identity,
            {64: exact},
            semantic_surfaces={64: masked_surface},
        )


def test_runtime_adapter_exposes_dual_surface_without_commit_authority() -> None:
    snapshot = _snapshot()
    adapter = CanonicalD64RuntimeAdapter()
    dual = adapter.compile_dual(snapshot)

    dual.assert_fresh(snapshot)
    assert dual.exact.source_field_id == snapshot.field_id
    assert dual.semantic.source_field_id == snapshot.field_id
    assert dual.semantic.source_rail_id == dual.exact.rail_id


def test_grounding_recomputes_features_and_exact_evidence_refs() -> None:
    snapshot = _snapshot()
    exact = D64FieldCompiler().compile(snapshot)
    surface = D64SemanticSurfaceCompiler().compile(snapshot, exact)
    slot = next(
        item
        for item in surface.slots
        if item.slot_kind == "field_span" and item.source_kind == "dormant_fact"
    )

    changed_features = np.array(slot.features64, copy=True)
    changed_features[0] += np.float32(0.125)
    tampered = replace(
        slot,
        features64=changed_features,
        container_refs=("container:forged",),
    )
    slots = tuple(tampered if item.slot_id == slot.slot_id else item for item in surface.slots)
    slots = tuple(
        sorted(
            slots,
            key=lambda item: (
                list(LogicalRegion).index(item.region),
                item.region_start,
                item.region_end,
                item.slot_kind,
                item.source_kind,
                item.slot_id,
            ),
        )
    )
    tampered_surface = D64SemanticSurface(
        source_field_id=surface.source_field_id,
        source_tick_id=surface.source_tick_id,
        source_rail_id=surface.source_rail_id,
        feature_generation=surface.feature_generation,
        slots=slots,
    )
    with pytest.raises(D64SemanticBindingError):
        tampered_surface.verify_grounding(snapshot, exact)
