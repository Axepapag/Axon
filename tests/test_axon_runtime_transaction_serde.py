from __future__ import annotations

import copy

import pytest

from runtime.axon_runtime.contracts import (
    ConsolidationDecision,
    CoreStateManifest,
    ProposalRecord,
    RuntimeContractError,
    SerializationContractError,
    TickCommitRecord,
    ToolRequestRecord,
)
from runtime.axon_runtime.field_transaction import (
    AppendSpans,
    ClearRegion,
    ReplaceSpans,
    SystemFieldUpdate,
    apply_system_update,
    compose_tick_transaction,
)
from runtime.axon_runtime.serde import (
    deserialize_consolidation_decision,
    deserialize_core_state_manifest,
    deserialize_proposal_record,
    deserialize_system_field_operation,
    deserialize_system_field_update,
    deserialize_tick_commit,
    deserialize_tool_request,
    serialize_consolidation_decision,
    serialize_proposal_record,
    serialize_system_field_operation,
    serialize_system_field_update,
    serialize_tick_commit,
    serialize_tool_request,
)
from runtime.field import (
    FieldDelta,
    FieldSpan,
    InsertText,
    SharedFieldSnapshot,
    canonical_sha256,
)


def _span(span_id: str, text: str) -> FieldSpan:
    return FieldSpan(
        span_id=span_id,
        text=text,
        kind="queued_evidence",
        source="runtime_queue",
        provenance=f"exact:{span_id}",
        confidence=0.75,
        container_refs=("container-β",),
        edge_refs=("edge-🙂",),
    )


def _records():
    head = SharedFieldSnapshot.from_texts(
        {"response_draft": "Draft", "task_state": "old"},
        tick_id=8,
        source="genesis",
        provenance="exact-head",
    )
    update = SystemFieldUpdate(
        base_field_id=head.field_id,
        base_tick_id=head.tick_id,
        operations=(
            AppendSpans("user_input", (_span("user-8", "café 🙂"),)),
            ReplaceSpans("task_state", (_span("task-8", "execute"),)),
            ClearRegion("advisor_input"),
        ),
        evidence=("queue:8",),
    )
    working = apply_system_update(head, update).working_snapshot
    proposer_delta = FieldDelta(
        base_field_id=working.field_id,
        base_tick_id=working.tick_id,
        author_core_id="core-proposer",
        pass_id="proposal-8",
        operations=(
            InsertText(
                region="scratch",
                offset=0,
                text="candidate",
                provenance="proposal",
            ),
        ),
    )
    proposal = ProposalRecord(
        tick_seq=head.tick_id,
        assignment_hash=canonical_sha256({"assignment": 8}),
        proposer_core_id="core-proposer",
        system_update_id=update.update_id,
        base_field_id=working.field_id,
        base_tick_id=working.tick_id,
        delta=proposer_delta,
        evidence=("user-8",),
    )
    committed_delta = FieldDelta(
        base_field_id=working.field_id,
        base_tick_id=working.tick_id,
        author_core_id="core-consolidator",
        pass_id="consolidate-8",
        operations=(
            InsertText(
                region="response_draft",
                offset=len(working.region("response_draft").text),
                text=" ready",
                provenance="consolidated",
            ),
        ),
    )
    audit = compose_tick_transaction(
        head,
        update,
        committed_delta,
        consolidator_author_core_id="core-consolidator",
    )
    decision = ConsolidationDecision(
        tick_seq=head.tick_id,
        assignment_hash=canonical_sha256({"assignment": 8}),
        consolidator_core_id="core-consolidator",
        proposal_id=proposal.proposal_id,
        base_field_id=working.field_id,
        accepted=True,
        output_field_id=audit.final_field_id,
        committed_delta=committed_delta,
        reason="accepted exact working-overlay delta",
    )
    request = ToolRequestRecord(
        tick_seq=head.tick_id,
        requester_core_id="core-consolidator",
        base_field_id=working.field_id,
        tool_name="lookup",
        arguments={"query": "café", "limit": 3},
        parent_proposal_id=proposal.proposal_id,
    )
    commit = TickCommitRecord(
        generation=9,
        tick_seq=head.tick_id,
        assignment_hash=canonical_sha256({"assignment": 8}),
        proposal_id=proposal.proposal_id,
        decision_id=decision.decision_id,
        input_field_id=head.field_id,
        system_update_id=update.update_id,
        working_field_id=working.field_id,
        field_transaction_audit_id=audit.audit_id,
        output_field_id=audit.final_field_id,
        updated_core_state_manifest_ids=("core-state-8",),
        projection_id="projection-9",
        tool_request_ids=(request.request_id,),
    )
    return update, proposal, decision, request, commit, working, audit


def test_system_operation_and_update_round_trip_exactly() -> None:
    update, *_ = _records()
    for operation in update.operations:
        serialized = serialize_system_field_operation(operation)
        assert deserialize_system_field_operation(serialized) == operation

    serialized_update = serialize_system_field_update(update)
    restored = deserialize_system_field_update(serialized_update)
    assert restored == update
    assert restored.update_id == update.update_id
    appended = next(
        operation
        for operation in restored.operations
        if isinstance(operation, AppendSpans)
    )
    assert appended.spans[0].text.encode("utf-8") == "café 🙂".encode("utf-8")
    assert appended.spans[0].provenance == "exact:user-8"
    assert appended.spans[0].container_refs == ("container-β",)
    assert appended.spans[0].edge_refs == ("edge-🙂",)


def test_v2_runtime_records_round_trip_with_nested_deltas() -> None:
    update, proposal, decision, request, commit, working, audit = _records()
    assert deserialize_proposal_record(
        serialize_proposal_record(proposal)
    ) == proposal
    assert deserialize_proposal_record(proposal.to_dict()) == proposal
    assert deserialize_consolidation_decision(
        serialize_consolidation_decision(decision)
    ) == decision
    assert deserialize_consolidation_decision(decision.to_dict()) == decision
    assert deserialize_tool_request(
        serialize_tool_request(request)
    ) == request
    assert deserialize_tick_commit(serialize_tick_commit(commit)) == commit

    assert proposal.to_canonical_dict()["schema"] == "axon-proposal-record-v2"
    assert proposal.system_update_id == update.update_id
    assert proposal.base_field_id == working.field_id
    assert decision.to_canonical_dict()["schema"] == (
        "axon-consolidation-decision-v2"
    )
    assert decision.base_field_id == working.field_id
    assert request.to_canonical_dict()["schema"] == "axon-tool-request-v2"
    assert request.base_field_id == working.field_id
    assert commit.to_canonical_dict()["schema"] == "axon-tick-commit-v2"
    assert commit.system_update_id == update.update_id
    assert commit.working_field_id == working.field_id
    assert commit.field_transaction_audit_id == audit.audit_id


@pytest.mark.parametrize(
    ("serializer", "deserializer", "missing_key"),
    (
        (
            serialize_system_field_update,
            deserialize_system_field_update,
            "update_id",
        ),
        (
            serialize_proposal_record,
            deserialize_proposal_record,
            "system_update_id",
        ),
        (
            serialize_consolidation_decision,
            deserialize_consolidation_decision,
            "decision_id",
        ),
        (
            serialize_tool_request,
            deserialize_tool_request,
            "request_id",
        ),
        (
            serialize_tick_commit,
            deserialize_tick_commit,
            "working_field_id",
        ),
    ),
)
def test_full_serializers_reject_missing_and_extra_fields(
    serializer,
    deserializer,
    missing_key: str,
) -> None:
    records = _records()[:5]
    by_serializer = {
        serialize_system_field_update: records[0],
        serialize_proposal_record: records[1],
        serialize_consolidation_decision: records[2],
        serialize_tool_request: records[3],
        serialize_tick_commit: records[4],
    }
    serialized = serializer(by_serializer[serializer])

    missing = copy.deepcopy(serialized)
    missing.pop(missing_key)
    with pytest.raises(SerializationContractError):
        deserializer(missing)

    extra = copy.deepcopy(serialized)
    extra["unexpected"] = True
    with pytest.raises(SerializationContractError):
        deserializer(extra)


def test_nested_delta_and_all_derived_ids_are_verified() -> None:
    update, proposal, decision, request, commit, *_ = _records()

    malformed_update = serialize_system_field_update(update)
    malformed_update["canonical_hash"] = "0" * 64
    with pytest.raises(SerializationContractError, match="hash mismatch"):
        deserialize_system_field_update(malformed_update)

    malformed_proposal = serialize_proposal_record(proposal)
    assert malformed_proposal["delta"] is not None
    malformed_proposal["delta"]["canonical_hash"] = "0" * 64
    with pytest.raises(SerializationContractError, match="hash mismatch"):
        deserialize_proposal_record(malformed_proposal)

    malformed_proposal = serialize_proposal_record(proposal)
    assert malformed_proposal["delta"] is not None
    malformed_proposal["delta"]["unexpected"] = "not canonical"
    with pytest.raises(SerializationContractError, match="fields mismatch"):
        deserialize_proposal_record(malformed_proposal)

    malformed_decision = serialize_consolidation_decision(decision)
    malformed_decision["committed_delta_id"] = "0" * 64
    with pytest.raises(SerializationContractError, match="delta ID mismatch"):
        deserialize_consolidation_decision(malformed_decision)

    malformed_request = serialize_tool_request(request)
    malformed_request["request_id"] = "tool-request-forged"
    with pytest.raises(SerializationContractError, match="request ID mismatch"):
        deserialize_tool_request(malformed_request)

    malformed_commit = serialize_tick_commit(commit)
    malformed_commit["commit_id"] = "tick-commit-forged"
    with pytest.raises(SerializationContractError, match="commit ID mismatch"):
        deserialize_tick_commit(malformed_commit)


def test_core_state_v2_requires_distinct_persisted_cursor_hash() -> None:
    cursor_hash = "A" * 64
    rng_hash = "b" * 64
    state = CoreStateManifest(
        core_id="core-0",
        soul_id="soul-0",
        model_id="model",
        core_state_sha256="c" * 64,
        generation=3,
        tick_seq=3,
        committed_field_id="field-3",
        soul_state_sha256="d" * 64,
        cursor_state_sha256=cursor_hash,
        rng_state_sha256=rng_hash,
    )
    assert state.to_canonical_dict()["schema"] == (
        "axon-core-state-manifest-v2"
    )
    assert state.cursor_state_sha256 == cursor_hash.lower()
    assert state.rng_state_sha256 == rng_hash
    assert deserialize_core_state_manifest(state.to_dict()) == state

    missing = state.to_dict()
    missing.pop("cursor_state_sha256")
    with pytest.raises(SerializationContractError):
        deserialize_core_state_manifest(missing)

    with pytest.raises(RuntimeContractError, match="cursor_state_sha256"):
        CoreStateManifest(
            core_id="core-0",
            soul_id="soul-0",
            model_id="model",
            core_state_sha256="c" * 64,
            generation=3,
            tick_seq=3,
            committed_field_id="field-3",
            soul_state_sha256="d" * 64,
            cursor_state_sha256="not-a-hash",
        )
