from __future__ import annotations

import pytest

from runtime.field import LogicalRegion
from runtime.heart import (
    AuthorityClass,
    HeartValveDefinition,
    HeartValveRegistry,
    UnknownValveError,
    ValveBudget,
    ValveDecision,
    ValveEnvelope,
    ValveReceipt,
    ValveState,
    primitive_valve_registry,
)
from runtime.heart.valve import _PRIMITIVE_REAL_VALVE_IDS


def _envelope(
    valve_id: str = "user_ingress",
    source_id: str = "external_user",
    payload: str = "hello",
    provenance: str = "test",
    envelope_type: str = "text/plain",
) -> ValveEnvelope:
    return ValveEnvelope(
        valve_id=valve_id,
        source_id=source_id,
        payload=payload,
        provenance=provenance,
        envelope_type=envelope_type,
    )


def test_closed_valve_rejects() -> None:
    registry = primitive_valve_registry()
    envelope = _envelope(valve_id="semantic_cortex")
    decision = registry.decide(envelope)
    assert isinstance(decision, ValveDecision)
    assert decision.admitted is False
    assert decision.quarantine is False
    assert decision.receipt is None
    assert decision.reason == "valve is closed"


def test_capped_primitive_valves_admit_valid_envelopes() -> None:
    registry = primitive_valve_registry()
    for valve_id, source_id, region in (
        ("user_ingress", "external_user", LogicalRegion.USER_INPUT),
        ("tool_ingress", "external_tool", LogicalRegion.TOOL_RESULTS),
        ("advisor_ingress", "external_advisor", LogicalRegion.ADVISOR_INPUT),
        ("dormant_recall", "dormant_valve", LogicalRegion.CORTEX),
    ):
        envelope = _envelope(valve_id=valve_id, source_id=source_id)
        decision = registry.decide(envelope)
        assert decision.admitted is True, valve_id
        assert decision.quarantine is False, valve_id
        assert decision.reason == "admitted", valve_id
        receipt = decision.receipt
        assert isinstance(receipt, ValveReceipt), valve_id
        assert receipt.valve_id == valve_id
        assert receipt.valve_version == 2
        assert receipt.source_id == source_id
        assert receipt.authority_class is registry.get(valve_id).authority_class
        assert receipt.governed_regions == frozenset({region})
        assert receipt.item_id
        assert receipt.enqueued_at is not None


def test_unknown_valve_id_fails() -> None:
    registry = primitive_valve_registry()
    envelope = _envelope(valve_id="no_such_valve")
    decision = registry.decide(envelope)
    assert decision.admitted is False
    assert decision.quarantine is False
    assert decision.receipt is None
    assert "unknown valve_id" in decision.reason


def test_get_unknown_valve_raises() -> None:
    registry = primitive_valve_registry()
    with pytest.raises(UnknownValveError, match="unknown valve_id"):
        registry.get("no_such_valve")


def test_mismatched_source_class_fails() -> None:
    registry = primitive_valve_registry()
    envelope = _envelope(valve_id="user_ingress", source_id="external_tool")
    decision = registry.decide(envelope)
    assert decision.admitted is False
    assert decision.quarantine is False
    assert "source class" in decision.reason


def test_mismatched_envelope_type_fails() -> None:
    registry = primitive_valve_registry()
    envelope = _envelope(valve_id="user_ingress", envelope_type="application/json")
    decision = registry.decide(envelope)
    assert decision.admitted is False
    assert decision.quarantine is False
    assert "envelope type" in decision.reason


def test_payload_beyond_old_item_ceiling_is_admitted_whole() -> None:
    registry = primitive_valve_registry()
    oversized = "x" * 20_000
    envelope = _envelope(valve_id="user_ingress", payload=oversized)
    decision = registry.decide(envelope)
    assert decision.admitted is True
    assert decision.quarantine is False
    assert registry.tracker.chars_this_beat("user_ingress") == len(oversized)


def test_rejection_policy_does_not_create_a_payload_ceiling() -> None:
    registry = HeartValveRegistry(
        [
            HeartValveDefinition(
                valve_id="reject_only",
                version=1,
                state=ValveState.CAPPED,
                source_class="test_source",
                authority_class=AuthorityClass.EXTERNAL_INGRESS,
                governed_regions=frozenset({LogicalRegion.USER_INPUT}),
                envelope_type="text/plain",
                budget=ValveBudget(
                    items_per_beat=10,
                    target_chars_per_beat=10,
                ),
                rejection_policy="reject",
            ),
        ]
    )
    envelope = _envelope(
        valve_id="reject_only", source_id="test_source", payload="x" * 11
    )
    decision = registry.decide(envelope)
    assert decision.admitted is True
    assert decision.quarantine is False


def test_budget_exhaustion_rejects_further_items() -> None:
    budget = ValveBudget(
        items_per_beat=2,
        target_chars_per_beat=1000,
    )
    registry = HeartValveRegistry(
        [
            HeartValveDefinition(
                valve_id="throttled",
                version=1,
                state=ValveState.CAPPED,
                source_class="test_source",
                authority_class=AuthorityClass.EXTERNAL_INGRESS,
                governed_regions=frozenset({LogicalRegion.USER_INPUT}),
                envelope_type="text/plain",
                budget=budget,
                rejection_policy="quarantine",
            ),
        ]
    )
    envelope = _envelope(valve_id="throttled", source_id="test_source", payload="a")
    first = registry.decide(envelope)
    assert first.admitted is True
    second = registry.decide(envelope)
    assert second.admitted is True
    third = registry.decide(envelope)
    assert third.admitted is False
    assert "items-per-beat budget exceeded" in third.reason


def test_target_chars_per_beat_defers_only_later_items() -> None:
    budget = ValveBudget(
        items_per_beat=100,
        target_chars_per_beat=2,
    )
    registry = HeartValveRegistry(
        [
            HeartValveDefinition(
                valve_id="char_throttled",
                version=1,
                state=ValveState.CAPPED,
                source_class="test_source",
                authority_class=AuthorityClass.EXTERNAL_INGRESS,
                governed_regions=frozenset({LogicalRegion.USER_INPUT}),
                envelope_type="text/plain",
                budget=budget,
                rejection_policy="quarantine",
            ),
        ]
    )
    envelope = _envelope(
        valve_id="char_throttled", source_id="test_source", payload="a"
    )
    assert registry.decide(envelope).admitted is True
    assert registry.decide(envelope).admitted is True
    decision = registry.decide(envelope)
    assert decision.admitted is False
    assert "target-chars-per-beat work allocation exhausted" in decision.reason


def test_envelope_cannot_smuggle_authority_grant() -> None:
    envelope = _envelope()
    assert not hasattr(envelope, "grant")
    assert not hasattr(envelope, "authority_grant")


def test_twenty_bootstrap_slots_are_present_and_registry_has_no_slot_ceiling() -> None:
    registry = primitive_valve_registry()
    expected_ids = [
        "user_ingress",
        "tool_ingress",
        "advisor_ingress",
        "dormant_recall",
        "semantic_cortex",
        "core_initial_proposal",
        "core_refinement",
        "consolidator",
        "vision",
        "hearing",
        "speech_feedback",
        "episodic_memory",
        "engineer_memory_service",
        "self_model",
        "planner",
        "actuation_feedback",
        "training_promotion",
        "external_sensor",
        "future_organ_a",
        "future_organ_b",
    ]
    assert len(expected_ids) == 20
    for valve_id in expected_ids:
        definition = registry.get(valve_id)
        assert definition.valve_id == valve_id

    capped = {
        vid
        for vid in expected_ids
        if registry.get(vid).state is ValveState.CAPPED
    }
    assert capped == set(_PRIMITIVE_REAL_VALVE_IDS)

    for valve_id in expected_ids:
        assert registry.is_open_for_mutation(valve_id) is (
            valve_id in _PRIMITIVE_REAL_VALVE_IDS
        )

    definitions = list(registry)
    definitions.append(
        HeartValveDefinition(
            valve_id="future_organ_c",
            version=1,
            state=ValveState.CLOSED,
            source_class="reserved",
            authority_class=AuthorityClass.CORE,
            governed_regions=frozenset({LogicalRegion.SCRATCH}),
            envelope_type="none",
            budget=ValveBudget(),
            rejection_policy="reject",
        )
    )
    expanded = HeartValveRegistry(definitions)
    assert expanded.get("future_organ_c").valve_id == "future_organ_c"


def test_valve_definition_validation() -> None:
    base = dict(
        valve_id="test",
        version=1,
        state=ValveState.CAPPED,
        source_class="source",
        authority_class=AuthorityClass.EXTERNAL_INGRESS,
        governed_regions=frozenset({LogicalRegion.USER_INPUT}),
        envelope_type="text/plain",
        budget=ValveBudget(),
        rejection_policy="quarantine",
    )

    # version must be >= 1
    with pytest.raises(ValueError, match="version must be >= 1"):
        HeartValveDefinition(**{**base, "version": 0})

    # governed_regions must be non-empty
    with pytest.raises(ValueError, match="governed_regions must be non-empty"):
        HeartValveDefinition(**{**base, "governed_regions": frozenset()})

    # rejection_policy must be reject or quarantine
    with pytest.raises(ValueError, match="rejection_policy"):
        HeartValveDefinition(**{**base, "rejection_policy": "ignore"})


def test_valve_budget_validation() -> None:
    with pytest.raises(ValueError, match="target_chars_per_beat"):
        ValveBudget(target_chars_per_beat=-1)
    with pytest.raises(TypeError, match="items_per_beat"):
        ValveBudget(items_per_beat="a")


def test_global_budget_enforced() -> None:
    budget = ValveBudget(
        items_per_beat=100,
        target_chars_per_beat=1000,
    )
    registry = HeartValveRegistry(
        [
            HeartValveDefinition(
                valve_id="v1",
                version=1,
                state=ValveState.CAPPED,
                source_class="s1",
                authority_class=AuthorityClass.EXTERNAL_INGRESS,
                governed_regions=frozenset({LogicalRegion.USER_INPUT}),
                envelope_type="text/plain",
                budget=budget,
                rejection_policy="quarantine",
            ),
            HeartValveDefinition(
                valve_id="v2",
                version=1,
                state=ValveState.CAPPED,
                source_class="s2",
                authority_class=AuthorityClass.EXTERNAL_INGRESS,
                governed_regions=frozenset({LogicalRegion.TOOL_RESULTS}),
                envelope_type="text/plain",
                budget=budget,
                rejection_policy="quarantine",
            ),
        ]
    )
    global_budget = ValveBudget(
        items_per_beat=1,
        target_chars_per_beat=1000,
    )
    first = registry.decide(
        ValveEnvelope("v1", "s1", "a", "test", "text/plain"),
        global_budget=global_budget,
    )
    assert first.admitted is True
    second = registry.decide(
        ValveEnvelope("v2", "s2", "b", "test", "text/plain"),
        global_budget=global_budget,
    )
    assert second.admitted is False
    assert "global items-per-beat budget exceeded" in second.reason


def test_beats_remaining_budget_override() -> None:
    budget = ValveBudget(
        items_per_beat=100,
        target_chars_per_beat=1000,
    )
    registry = HeartValveRegistry(
        [
            HeartValveDefinition(
                valve_id="override",
                version=1,
                state=ValveState.CAPPED,
                source_class="test_source",
                authority_class=AuthorityClass.EXTERNAL_INGRESS,
                governed_regions=frozenset({LogicalRegion.USER_INPUT}),
                envelope_type="text/plain",
                budget=budget,
                rejection_policy="quarantine",
            ),
        ]
    )
    envelope = _envelope(valve_id="override", source_id="test_source", payload="a")
    decision = registry.decide(envelope, beats_remaining_budget={"override": {"items": 0}})
    assert decision.admitted is False
    assert "caller-supplied beat item budget exhausted" in decision.reason
