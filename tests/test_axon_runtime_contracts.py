from __future__ import annotations

import copy

import pytest

from runtime.axon_runtime import (
    ConsolidationDecision,
    CoreIdentity,
    CoreStateManifest,
    IdentityContractError,
    ProjectionManifest,
    ProposalRecord,
    RoleContractError,
    RoleScheduler,
    SerializationContractError,
    SourceEvent,
    SpanLifecycleEvent,
    TickCommitRecord,
    ToolRequestRecord,
    ToolResultRecord,
    deserialize_core_identity,
    deserialize_core_state_manifest,
    deserialize_field_delta,
    deserialize_field_span,
    deserialize_field_view_cursor,
    deserialize_projection_manifest,
    deserialize_region_state,
    deserialize_role_assignment,
    deserialize_shared_field,
    serialize_field_delta,
    serialize_field_span,
    serialize_field_view_cursor,
    serialize_region_state,
    serialize_shared_field,
    validate_identity_population,
)
from runtime.field import (
    DeleteText,
    FieldDelta,
    FieldSpan,
    FieldViewCursor,
    InsertText,
    LogicalRegion,
    RegionState,
    ReplaceText,
    SharedFieldSnapshot,
    apply_delta,
    canonical_sha256,
)


def _identity(index: int, *, enabled: bool = True) -> CoreIdentity:
    return CoreIdentity(
        core_id=f"core-{index}",
        display_name=f"Core {index}",
        base_checkpoint_path=r"D:\Axon\checkpoints\shared.pt",
        base_checkpoint_sha256="a" * 64,
        model_id="shared-64d-model",
        core_state_sha256="b" * 64,
        lineage=("exact-v4", "phase-a"),
        parent_core_id=None if index == 0 else "core-0",
        soul_id=f"soul-{index}",
        adapter_namespace=f"personal/core-{index}/root",
        enabled=enabled,
    )


def test_role_ring_rejects_small_population_and_is_fair() -> None:
    with pytest.raises(RoleContractError):
        RoleScheduler([_identity(0), _identity(1)])

    for size in (3, 4):
        identities = [_identity(index) for index in range(size)]
        scheduler = RoleScheduler(identities)
        assignments = [scheduler.assignment(tick) for tick in range(size)]
        for assignment in assignments:
            assert len(
                {
                    assignment.proposer_core_id,
                    assignment.consolidator_core_id,
                    assignment.sleeper_core_id,
                }
            ) == 3
            assert len(assignment.standby_core_ids) == size - 3
            restored = deserialize_role_assignment(assignment.to_dict())
            assert restored == assignment
        for role in (
            "proposer_core_id",
            "consolidator_core_id",
            "sleeper_core_id",
        ):
            assert {getattr(value, role) for value in assignments} == set(
                scheduler.core_ids
            )
        restarted = RoleScheduler(identities)
        assert [
            restarted.assignment(tick) for tick in range(size * 2)
        ] == [
            scheduler.assignment(tick) for tick in range(size * 2)
        ]

    disabled = [_identity(0), _identity(1), _identity(2, enabled=False)]
    with pytest.raises(RoleContractError):
        RoleScheduler(disabled)


def test_core_clone_identity_isolation_and_runtime_record_ids() -> None:
    identities = [_identity(index) for index in range(3)]
    validate_identity_population(identities)
    assert len({value.identity_hash for value in identities}) == 3
    assert len({value.soul_id for value in identities}) == 3
    assert len({value.adapter_namespace for value in identities}) == 3
    assert len({value.model_id for value in identities}) == 1
    assert len({value.base_checkpoint_sha256 for value in identities}) == 1
    for identity in identities:
        assert deserialize_core_identity(identity.to_dict()) == identity

    duplicate_soul = copy.deepcopy(identities[1].to_canonical_dict())
    duplicate_soul.pop("schema")
    duplicate_soul["soul_id"] = identities[0].soul_id
    duplicate_soul["lineage"] = tuple(duplicate_soul["lineage"])
    duplicate_soul["role_capabilities"] = tuple(
        duplicate_soul["role_capabilities"]
    )
    with pytest.raises(IdentityContractError):
        validate_identity_population(
            [identities[0], CoreIdentity(**duplicate_soul), identities[2]]
        )

    source = SourceEvent(
        idempotency_key="source-1",
        source_kind="user",
        source_ref="input:1",
        payload={"text": "hello"},
    )
    assert SourceEvent(
        idempotency_key="source-1",
        source_kind="user",
        source_ref="input:1",
        payload={"text": "hello"},
    ).source_event_id == source.source_event_id
    lifecycle = SpanLifecycleEvent(
        action="masked",
        span_id="span-1",
        region="conversation_history",
        field_id="f" * 64,
        payload={"reason": "policy"},
        parent_ids=(source.source_event_id,),
    )
    request = ToolRequestRecord(
        tick_seq=0,
        requester_core_id="core-0",
        base_field_id="f" * 64,
        tool_name="search",
        arguments={"q": "axon"},
        parent_proposal_id="proposal-parent",
    )
    result = ToolResultRecord(
        idempotency_key="tool-result-1",
        request_id=request.request_id,
        status="complete",
        payload={"rows": 2},
        parent_ids=(request.request_id,),
    )
    assert lifecycle.lifecycle_event_id.startswith("span-lifecycle-")
    assert request.request_id.startswith("tool-request-")
    assert result.result_id.startswith("tool-result-")


def test_strict_field_and_cursor_serialization_round_trips() -> None:
    span = FieldSpan(
        span_id="fact-1",
        text="Axon remembers.",
        kind="fact",
        source="D:/00/memory.db",
        provenance="row:1",
        confidence=0.75,
        container_refs=("container-2", "container-1"),
        edge_refs=("edge-1",),
    )
    region = RegionState(
        name=LogicalRegion.STRUCTURED_KNOWLEDGE,
        spans=(span,),
    )
    snapshot = SharedFieldSnapshot(
        tick_id=0,
        regions=(
            region,
            RegionState.from_text(
                LogicalRegion.RESPONSE_DRAFT,
                "draft",
                span_id="draft-1",
            ),
        ),
        source_manifest_ids=("manifest-1",),
    )
    delta = FieldDelta(
        base_field_id=snapshot.field_id,
        base_tick_id=snapshot.tick_id,
        author_core_id="core-0",
        pass_id="tick-0",
        operations=(
            InsertText(
                LogicalRegion.RESPONSE_DRAFT,
                0,
                "A",
                provenance="proposal",
            ),
            ReplaceText(
                LogicalRegion.RESPONSE_DRAFT,
                1,
                3,
                "BC",
                container_refs=("container-1",),
            ),
            DeleteText(
                LogicalRegion.RESPONSE_DRAFT,
                3,
                5,
                edge_refs=("edge-1",),
            ),
        ),
        evidence=("row:1",),
    )
    cursor = FieldViewCursor(
        page_index=3,
        context_offsets=(("conversation_history", 12),),
        user_offset=4,
    )

    assert deserialize_field_span(serialize_field_span(span)) == span
    assert deserialize_region_state(serialize_region_state(region)) == region
    assert deserialize_shared_field(serialize_shared_field(snapshot)) == snapshot
    assert deserialize_field_delta(serialize_field_delta(delta)) == delta
    assert (
        deserialize_field_view_cursor(serialize_field_view_cursor(cursor))
        == cursor
    )
    assert (
        deserialize_shared_field(serialize_shared_field(snapshot)).canonical_hash
        == snapshot.canonical_hash
    )

    malformed = serialize_shared_field(snapshot)
    malformed["extra"] = True
    with pytest.raises(SerializationContractError):
        deserialize_shared_field(malformed)

    malformed = serialize_region_state(region)
    malformed["visibility"] = "unknown"
    with pytest.raises(SerializationContractError):
        deserialize_region_state(malformed)

    malformed = serialize_field_delta(delta)
    malformed["canonical_hash"] = "0" * 64
    with pytest.raises(SerializationContractError):
        deserialize_field_delta(malformed)

    malformed = serialize_field_span(span)
    malformed["schema"] = "future-span"
    with pytest.raises(SerializationContractError):
        deserialize_field_span(malformed)

    malformed = serialize_field_view_cursor(cursor)
    malformed["context_offsets"]["not_a_region"] = 2
    with pytest.raises((SerializationContractError, ValueError)):
        deserialize_field_view_cursor(malformed)


def test_manifest_proposal_decision_projection_and_commit_hashes() -> None:
    identity = _identity(0)
    snapshot = SharedFieldSnapshot.from_texts({"response_draft": ""})
    state = CoreStateManifest(
        core_id=identity.core_id,
        soul_id=identity.soul_id,
        model_id=identity.model_id,
        core_state_sha256=identity.core_state_sha256,
        generation=0,
        tick_seq=0,
        committed_field_id=snapshot.field_id,
        soul_state_sha256=canonical_sha256({"soul": 0}),
        cursor_state_sha256=canonical_sha256({"cursor": 0}),
    )
    assert deserialize_core_state_manifest(state.to_dict()) == state
    assignment = RoleScheduler([_identity(i) for i in range(3)]).assignment(0)
    system_update_id = canonical_sha256({"system_update": 0})
    proposer_delta = FieldDelta(
        base_field_id=snapshot.field_id,
        base_tick_id=0,
        author_core_id=assignment.proposer_core_id,
        pass_id="proposal",
        operations=(InsertText(LogicalRegion.RESPONSE_DRAFT, 0, "A"),),
    )
    proposal = ProposalRecord(
        tick_seq=0,
        assignment_hash=assignment.assignment_hash,
        proposer_core_id=assignment.proposer_core_id,
        system_update_id=system_update_id,
        base_field_id=snapshot.field_id,
        base_tick_id=0,
        delta=proposer_delta,
    )
    committed_delta = FieldDelta(
        base_field_id=snapshot.field_id,
        base_tick_id=0,
        author_core_id=assignment.consolidator_core_id,
        pass_id="consolidation",
        operations=(InsertText(LogicalRegion.RESPONSE_DRAFT, 0, "B"),),
    )
    successor = apply_delta(snapshot, committed_delta)
    decision = ConsolidationDecision(
        tick_seq=0,
        assignment_hash=assignment.assignment_hash,
        consolidator_core_id=assignment.consolidator_core_id,
        proposal_id=proposal.proposal_id,
        base_field_id=snapshot.field_id,
        accepted=True,
        output_field_id=successor.field_id,
        committed_delta=committed_delta,
    )
    assert decision.committed_delta_id == committed_delta.delta_id
    assert (
        decision.to_canonical_dict()["committed_delta"]
        == committed_delta.to_canonical_dict()
    )
    projection = ProjectionManifest(
        generation=1,
        tick_seq=1,
        field_id=successor.field_id,
        field_hash=successor.canonical_hash,
        core_state_manifest_ids=((state.core_id, state.manifest_id),),
        role_index=1,
        role_assignment_hash=RoleScheduler(
            [_identity(i) for i in range(3)]
        ).assignment(1).assignment_hash,
        parent_projection_id="projection-parent",
    )
    assert deserialize_projection_manifest(projection.to_dict()) == projection
    commit = TickCommitRecord(
        generation=1,
        tick_seq=0,
        assignment_hash=assignment.assignment_hash,
        proposal_id=proposal.proposal_id,
        decision_id=decision.decision_id,
        input_field_id=snapshot.field_id,
        system_update_id=system_update_id,
        working_field_id=canonical_sha256({"working": 0}),
        field_transaction_audit_id=canonical_sha256({"audit": 0}),
        output_field_id=successor.field_id,
        updated_core_state_manifest_ids=(),
        projection_id=projection.projection_id,
    )
    assert proposal.proposal_id.startswith("proposal-")
    assert decision.decision_id.startswith("decision-")
    assert projection.projection_id.startswith("projection-")
    assert commit.commit_id.startswith("tick-commit-")
