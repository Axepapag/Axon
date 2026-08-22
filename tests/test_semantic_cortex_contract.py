from __future__ import annotations

import pytest

from Cortext import ExactEvidenceRef, SemanticObservation, SemanticQuery
from runtime.heart import ValveState, primitive_valve_registry


def _evidence() -> ExactEvidenceRef:
    return ExactEvidenceRef(
        source_kind="dormant_container",
        source_id="c-1",
        provenance="fixture:container:c-1",
        sha256="a" * 64,
    )


def test_semantic_query_and_observation_have_stable_grounded_identity() -> None:
    ref = _evidence()
    query = SemanticQuery(
        lane="taxonomy",
        source_field_id="f" * 64,
        source_tick_id=2,
        text="what kind of system is Axon?",
        evidence_refs=(ref,),
    )
    same_query = SemanticQuery(
        lane="taxonomy",
        source_field_id="f" * 64,
        source_tick_id=2,
        text="what kind of system is Axon?",
        evidence_refs=(ref,),
    )
    assert query.query_id == same_query.query_id

    observation = SemanticObservation(
        lane="taxonomy",
        source_field_id=query.source_field_id,
        source_tick_id=query.source_tick_id,
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
    assert observation.evidence_refs == (ref,)
    assert not hasattr(observation, "grant")
    assert not hasattr(observation, "authority_grant")
    assert not hasattr(observation, "target_region")


def test_semantic_observation_fails_closed_without_exact_evidence() -> None:
    with pytest.raises(ValueError, match="exact evidence"):
        SemanticObservation(
            lane="taxonomy",
            source_field_id="f" * 64,
            source_tick_id=0,
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
    assert valve.budget.pending_cap == 0
