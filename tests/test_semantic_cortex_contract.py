from __future__ import annotations

import pytest

from Cortext import (
    ExactEvidenceRef,
    SemanticObservation,
    SemanticProjectionRef,
    SemanticQuery,
)
from runtime.field import D64FieldCompiler, D64SemanticSurfaceCompiler, LogicalRegion, RegionState, SharedFieldSnapshot
from runtime.heart import ValveState, primitive_valve_registry


def _evidence() -> ExactEvidenceRef:
    return ExactEvidenceRef(
        source_kind="dormant_container",
        source_id="c-1",
        provenance="fixture:container:c-1",
        sha256="a" * 64,
    )


def _projection() -> tuple[SemanticProjectionRef, object]:
    snapshot = SharedFieldSnapshot(
        tick_id=2,
        regions=(
            RegionState.from_text(
                LogicalRegion.USER_INPUT,
                "what kind of system is Axon?",
                span_id="q-1",
                kind="semantic_eval_text",
                source="fixture",
                provenance="fixture:q-1",
            ),
        ),
    )
    exact = D64FieldCompiler().compile(snapshot)
    surface = D64SemanticSurfaceCompiler().compile(snapshot, exact)
    slot = next(
        item
        for item in surface.slots
        if item.slot_kind == "field_span" and item.source_span_ids == ("q-1",)
    )
    return SemanticProjectionRef.from_surface(surface, slot_ids=(slot.slot_id,)), surface


def test_semantic_query_and_observation_have_stable_grounded_identity() -> None:
    ref = _evidence()
    projection, surface = _projection()
    projection.assert_matches(surface)

    query = SemanticQuery(
        lane="taxonomy",
        text="what kind of system is Axon?",
        projection=projection,
        evidence_refs=(ref,),
    )
    same_query = SemanticQuery(
        lane="taxonomy",
        text="what kind of system is Axon?",
        projection=projection,
        evidence_refs=(ref,),
    )
    assert query.query_id == same_query.query_id
    assert query.source_field_id == projection.source_field_id
    assert query.source_tick_id == projection.source_tick_id

    observation = SemanticObservation(
        lane="taxonomy",
        query_id=query.query_id,
        projection=projection,
        microtick_sequence=7,
        model_generation="taxonomy-g0",
        subject="Axon",
        relation="is_a",
        object="system",
        confidence=0.6,
        relevance=0.9,
        novelty=0.3,
        evidence_refs=(ref,),
        retrieval_candidate_ids=("c-1",),
    )
    assert len(observation.observation_id) == 64
    assert observation.source_field_id == query.source_field_id
    assert observation.source_tick_id == query.source_tick_id
    assert observation.evidence_refs == (ref,)
    assert not hasattr(observation, "grant")
    assert not hasattr(observation, "authority_grant")
    assert not hasattr(observation, "target_region")


def test_semantic_projection_fails_closed_when_surface_identity_is_tampered() -> None:
    projection, surface = _projection()
    tampered = SemanticProjectionRef(
        source_field_id=projection.source_field_id,
        source_tick_id=projection.source_tick_id,
        rail_id="b" * 64,
        semantic_surface_id=projection.semantic_surface_id,
        feature_generation=projection.feature_generation,
        slots=projection.slots,
    )
    with pytest.raises(ValueError, match=r"stale|another D64 surface"):
        tampered.assert_matches(surface)


def test_semantic_observation_fails_closed_without_exact_evidence() -> None:
    projection, _ = _projection()
    query = SemanticQuery(
        lane="taxonomy",
        text="what kind of system is Axon?",
        projection=projection,
    )
    with pytest.raises(ValueError, match="exact evidence"):
        SemanticObservation(
            lane="taxonomy",
            query_id=query.query_id,
            projection=projection,
            microtick_sequence=0,
            model_generation="taxonomy-g0",
            subject="Axon",
            relation="is_a",
            object="system",
            confidence=0.5,
            relevance=0.5,
            novelty=0.5,
            evidence_refs=(),
        )


def test_semantic_cortex_valve_remains_closed() -> None:
    registry = primitive_valve_registry()
    valve = registry.get("semantic_cortex")
    assert valve.state is ValveState.CLOSED
    assert valve.budget.items_per_beat == 0
    assert valve.budget.target_chars_per_beat == 0
