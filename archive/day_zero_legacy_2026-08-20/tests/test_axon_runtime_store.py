from __future__ import annotations

import sqlite3

import pytest

from runtime.axon_runtime import (
    ConsolidationDecision,
    CoreIdentity,
    CoreStateManifest,
    CoreStateUpdateError,
    IdempotencyConflictError,
    JournalIntegrityError,
    ProjectionManifest,
    ProposalRecord,
    RoleOwnershipError,
    RoleScheduler,
    RuntimeStore,
    SourceEvent,
    SpanLifecycleEvent,
    StaleHeadError,
    StoreContractError,
    ToolRequestRecord,
    ToolResultRecord,
    canonical_json_text,
    deserialize_shared_field,
    parse_json_object,
)
from runtime.axon_runtime.field_transaction import (
    AppendSpans,
    SystemFieldUpdate,
    apply_system_update,
    compose_tick_transaction,
)
from runtime.axon_runtime.ingress import (
    ConsumptionReceipt,
    IngressConsumptionError,
    IngressEvent,
    IngressQueue,
    build_system_update,
)
from runtime.field import (
    FieldDelta,
    FieldSpan,
    InsertText,
    LogicalRegion,
    SharedFieldSnapshot,
    apply_delta,
    canonical_sha256,
)


def _identities(count: int = 4) -> tuple[CoreIdentity, ...]:
    return tuple(
        CoreIdentity(
            core_id=f"core-{index}",
            display_name=f"Core {index}",
            base_checkpoint_path=r"D:\Axon\checkpoints\shared.pt",
            base_checkpoint_sha256="a" * 64,
            model_id="shared-model",
            core_state_sha256="b" * 64,
            lineage=("exact-v4",),
            parent_core_id=None if index == 0 else "core-0",
            soul_id=f"soul-{index}",
            adapter_namespace=f"personal/core-{index}/root",
        )
        for index in range(count)
    )


def _initial_states(
    identities: tuple[CoreIdentity, ...],
    snapshot: SharedFieldSnapshot,
) -> tuple[CoreStateManifest, ...]:
    return tuple(
        CoreStateManifest(
            core_id=identity.core_id,
            soul_id=identity.soul_id,
            model_id=identity.model_id,
            core_state_sha256=identity.core_state_sha256,
            generation=0,
            tick_seq=0,
            committed_field_id=snapshot.field_id,
            soul_state_sha256=canonical_sha256(
                {"core_id": identity.core_id, "generation": 0}
            ),
            cursor_state_sha256=canonical_sha256(
                {"core_id": identity.core_id, "cursor_generation": 0}
            ),
        )
        for identity in identities
    )


def _tick_inputs(
    store: RuntimeStore,
    identities: tuple[CoreIdentity, ...],
    *,
    no_op: bool = False,
    update_role: str = "proposer",
    queue_tool: bool = False,
    proposal_text: str | None = None,
    committed_text: str | None = None,
    accepted: bool = True,
    system_operations=(),
    system_update_override: SystemFieldUpdate | None = None,
):
    head = store.current_head()
    scheduler = RoleScheduler(identities)
    assignment = scheduler.assignment(head.tick_seq)
    system_update = (
        system_update_override
        if system_update_override is not None
        else SystemFieldUpdate(
            base_field_id=head.snapshot.field_id,
            base_tick_id=head.snapshot.tick_id,
            operations=tuple(system_operations),
            evidence=(f"tick:{head.tick_seq}",),
        )
    )
    working = apply_system_update(
        head.snapshot,
        system_update,
    ).working_snapshot
    if no_op:
        proposal_delta = None
        committed_delta = None
    else:
        response = working.region(LogicalRegion.RESPONSE_DRAFT).text
        default_text = chr(ord("a") + head.tick_seq)
        proposal_operation = InsertText(
            region=LogicalRegion.RESPONSE_DRAFT,
            offset=len(response),
            text=proposal_text if proposal_text is not None else default_text,
            provenance=f"tick:{head.tick_seq}",
        )
        committed_operation = InsertText(
            region=LogicalRegion.RESPONSE_DRAFT,
            offset=len(response),
            text=(
                committed_text
                if committed_text is not None
                else default_text
            ),
            provenance=f"consolidated-tick:{head.tick_seq}",
        )
        proposal_delta = FieldDelta(
            base_field_id=working.field_id,
            base_tick_id=working.tick_id,
            author_core_id=assignment.proposer_core_id,
            pass_id=f"tick-{head.tick_seq}",
            operations=(proposal_operation,),
        )
        committed_delta = (
            FieldDelta(
                base_field_id=working.field_id,
                base_tick_id=working.tick_id,
                author_core_id=assignment.consolidator_core_id,
                pass_id=f"consolidate-{head.tick_seq}",
                operations=(committed_operation,),
            )
            if accepted
            else None
        )
    audit = compose_tick_transaction(
        head.snapshot,
        system_update,
        committed_delta,
        consolidator_author_core_id=(
            None
            if committed_delta is None
            else assignment.consolidator_core_id
        ),
    )
    output = audit.final_snapshot
    proposal = ProposalRecord(
        tick_seq=head.tick_seq,
        assignment_hash=assignment.assignment_hash,
        proposer_core_id=assignment.proposer_core_id,
        system_update_id=system_update.update_id,
        base_field_id=working.field_id,
        base_tick_id=working.tick_id,
        delta=proposal_delta,
    )
    decision = ConsolidationDecision(
        tick_seq=head.tick_seq,
        assignment_hash=assignment.assignment_hash,
        consolidator_core_id=assignment.consolidator_core_id,
        proposal_id=proposal.proposal_id,
        base_field_id=working.field_id,
        accepted=accepted,
        output_field_id=output.field_id,
        committed_delta=committed_delta,
        reason="accepted",
    )
    role_to_core = {
        "proposer": assignment.proposer_core_id,
        "consolidator": assignment.consolidator_core_id,
        "sleeper": assignment.sleeper_core_id,
    }
    update_core_id = role_to_core[update_role]
    identity_by_id = {identity.core_id: identity for identity in identities}
    current_by_id = {
        state.core_id: state for state in head.core_state_manifests
    }
    identity = identity_by_id[update_core_id]
    previous = current_by_id[update_core_id]
    updated = CoreStateManifest(
        core_id=identity.core_id,
        soul_id=identity.soul_id,
        model_id=identity.model_id,
        core_state_sha256=identity.core_state_sha256,
        generation=head.generation + 1,
        tick_seq=head.tick_seq + 1,
        committed_field_id=output.field_id,
        soul_state_sha256=canonical_sha256(
            {
                "core_id": identity.core_id,
                "generation": head.generation + 1,
                "field_id": output.field_id,
            }
        ),
        cursor_state_sha256=canonical_sha256(
            {
                "core_id": identity.core_id,
                "cursor_generation": head.generation + 1,
                "field_id": output.field_id,
            }
        ),
        parent_manifest_id=previous.manifest_id,
    )
    updates = (updated,) if accepted else ()
    projected = dict(current_by_id)
    if accepted:
        projected[updated.core_id] = updated
    next_assignment = scheduler.assignment(head.tick_seq + 1)
    projection = ProjectionManifest(
        generation=head.generation + 1,
        tick_seq=head.tick_seq + 1,
        field_id=output.field_id,
        field_hash=output.canonical_hash,
        core_state_manifest_ids=tuple(
            (core_id, state.manifest_id)
            for core_id, state in projected.items()
        ),
        role_index=scheduler.role_index(head.tick_seq + 1),
        role_assignment_hash=next_assignment.assignment_hash,
        parent_projection_id=head.projection_manifest_id,
    )
    requests = ()
    if queue_tool and accepted:
        requests = (
            ToolRequestRecord(
                tick_seq=head.tick_seq,
                requester_core_id=assignment.consolidator_core_id,
                base_field_id=working.field_id,
                tool_name="lookup",
                arguments={"tick": head.tick_seq},
                parent_proposal_id=proposal.proposal_id,
            ),
        )
    return (
        head,
        assignment,
        proposal,
        decision,
        output,
        updates,
        projection,
        requests,
        system_update,
    )


def _commit(
    store: RuntimeStore,
    values,
    *,
    ingress_queue: IngressQueue | None = None,
    consumed_ingress_event_ids=(),
) -> object:
    (
        head,
        assignment,
        proposal,
        decision,
        output,
        updates,
        projection,
        requests,
        system_update,
    ) = values
    return store.commit_tick(
        head.generation,
        head.snapshot.field_id,
        assignment,
        system_update,
        proposal,
        decision,
        output,
        updates,
        projection,
        requests,
        ingress_queue=ingress_queue,
        consumed_ingress_event_ids=consumed_ingress_event_ids,
    )


def test_initialize_reopen_recover_and_six_deterministic_ticks(tmp_path) -> None:
    database = tmp_path / "runtime.sqlite3"
    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.CONVERSATION_HISTORY: "start",
            LogicalRegion.RESPONSE_DRAFT: "",
        },
        source_manifest_ids=("source-manifest",),
    )
    with RuntimeStore(database) as store:
        head = store.initialize(
            genesis,
            identities,
            _initial_states(identities, genesis),
        )
        assert head.generation == 0
        assert head.tick_seq == 0
        assert head.snapshot == genesis
        assert store.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert (
            store.connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        )
        assert (
            store.connection.execute("PRAGMA journal_mode").fetchone()[0]
            == "wal"
        )
        genesis_record = store.connection.execute(
            """
            SELECT record_id, canonical_json
            FROM journal
            WHERE event_type='genesis'
            """
        ).fetchone()
        assert genesis_record["record_id"] == genesis.field_id
        assert deserialize_shared_field(
            parse_json_object(genesis_record["canonical_json"])
        ) == genesis

    with RuntimeStore(database) as store:
        assert store.recover().snapshot == genesis
        source = SourceEvent(
            idempotency_key="source-input-1",
            source_kind="user_input",
            source_ref="phone-call:1",
            payload={"text": "continue"},
        )
        assert store.append_source_event(source) == store.append_source_event(
            source
        )
        with pytest.raises(IdempotencyConflictError):
            store.append_source_event(
                SourceEvent(
                    idempotency_key="source-input-1",
                    source_kind="user_input",
                    source_ref="phone-call:1",
                    payload={"text": "different"},
                )
            )
        store.append_span_lifecycle(
            SpanLifecycleEvent(
                action="attended",
                span_id="conversation_history:0",
                region="conversation_history",
                field_id=genesis.field_id,
                payload={"mask": "visible"},
                parent_ids=(source.source_event_id,),
            )
        )

        snapshots = [genesis]
        assignments = []
        request = None
        for tick in range(6):
            values = _tick_inputs(
                store,
                identities,
                no_op=tick == 2,
                accepted=tick != 2,
                queue_tool=tick == 0,
                system_operations=(
                    (
                        AppendSpans(
                            region=LogicalRegion.USER_INPUT,
                            spans=(
                                FieldSpan(
                                    span_id="user-input-1",
                                    text="continue",
                                    kind="user_input",
                                    source="phone-call:1",
                                    provenance=source.source_event_id,
                                ),
                            ),
                        ),
                    )
                    if tick == 0
                    else ()
                ),
            )
            assignments.append(values[1])
            if values[7]:
                request = values[7][0]
            _commit(store, values)
            snapshots.append(values[4])
            assert snapshots[-1].tick_id == snapshots[-2].tick_id + 1
            assert snapshots[-1].parent_field_id == snapshots[-2].field_id

        head = store.recover()
        assert head.generation == 6
        assert head.tick_seq == 6
        assert head.snapshot == snapshots[-1]
        assert head.snapshot.region("response_draft").text == "abdef"
        assert head.snapshot.region("user_input").text == "continue"
        assert store.replay_journal() == head
        scheduler = RoleScheduler(identities)
        assert assignments == [scheduler.assignment(tick) for tick in range(6)]
        assert head.role_assignment_hash == scheduler.assignment(
            6
        ).assignment_hash

        assert request is not None
        result = ToolResultRecord(
            idempotency_key="tool-result-1",
            request_id=request.request_id,
            status="complete",
            payload={"value": 42},
            parent_ids=(request.request_id,),
        )
        first_event = store.append_tool_result(result)
        assert store.append_tool_result(result) == first_event
        with pytest.raises(IdempotencyConflictError):
            store.append_tool_result(
                ToolResultRecord(
                    idempotency_key="tool-result-1",
                    request_id=request.request_id,
                    status="complete",
                    payload={"value": 99},
                    parent_ids=(request.request_id,),
                )
            )
        store.verify_journal_chain()


def test_recover_detects_projection_drift_and_rebuilds_from_journal(
    tmp_path,
) -> None:
    database = tmp_path / "projection-repair.sqlite3"
    identities = _identities(3)
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    with RuntimeStore(database) as store:
        store.initialize(genesis, identities, _initial_states(identities, genesis))
        _commit(store, _tick_inputs(store, identities))
        truth = store.replay_journal()
        journal_before = tuple(
            tuple(row)
            for row in store.connection.execute(
                """
                SELECT
                    seq,
                    event_id,
                    previous_event_id,
                    event_type,
                    record_id,
                    canonical_json,
                    idempotency_key
                FROM journal
                ORDER BY seq
                """
            ).fetchall()
        )

        store.connection.execute(
            """
            UPDATE runtime_head
            SET generation=generation + 10,
                tick_seq=tick_seq + 10,
                role_index=role_index + 1
            WHERE singleton=1
            """
        )
        with pytest.raises(
            JournalIntegrityError,
            match="drift from journal truth",
        ):
            store.recover()
        assert store.rebuild_runtime_head_from_journal() == truth
        assert store.recover() == truth

        core_id = truth.core_state_manifests[0].core_id
        store.connection.execute(
            """
            UPDATE core_states
            SET manifest_id='drifted-manifest-id'
            WHERE core_id=?
            """,
            (core_id,),
        )
        with pytest.raises(
            JournalIntegrityError,
            match="projection columns conflict",
        ):
            store.recover()
        assert store.rebuild_runtime_head_from_journal() == truth

        store.connection.execute(
            "UPDATE core_states SET manifest_json='{}' WHERE core_id=?",
            (core_id,),
        )
        with pytest.raises(
            JournalIntegrityError,
            match="mutable runtime projections are invalid",
        ):
            store.recover()
        assert store.rebuild_runtime_head_from_journal() == truth

        store.connection.execute(
            "DELETE FROM runtime_head WHERE singleton=1"
        )
        with pytest.raises(
            JournalIntegrityError,
            match="mutable runtime projections are invalid",
        ):
            store.recover()
        assert store.rebuild_runtime_head_from_journal() == truth
        assert store.recover() == truth

        journal_after = tuple(
            tuple(row)
            for row in store.connection.execute(
                """
                SELECT
                    seq,
                    event_id,
                    previous_event_id,
                    event_type,
                    record_id,
                    canonical_json,
                    idempotency_key
                FROM journal
                ORDER BY seq
                """
            ).fetchall()
        )
        assert journal_after == journal_before


def test_tool_result_retry_repairs_projection_and_record_ids_are_unique(
    tmp_path,
) -> None:
    database = tmp_path / "tool-result-repair.sqlite3"
    identities = _identities(3)
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    with RuntimeStore(database) as store:
        store.initialize(genesis, identities, _initial_states(identities, genesis))
        values = _tick_inputs(store, identities, queue_tool=True)
        _commit(store, values)
        request = values[7][0]
        result = ToolResultRecord(
            idempotency_key="result-repair-1",
            request_id=request.request_id,
            status="complete",
            payload={"nested": {"values": [1, 2, 3]}},
            parent_ids=(request.request_id,),
        )
        event_id = store.append_tool_result(result)
        journal_count = store.connection.execute(
            "SELECT COUNT(*) FROM journal"
        ).fetchone()[0]

        store.connection.execute(
            """
            UPDATE tool_requests
            SET status='queued', result_id=NULL, result_json=NULL
            WHERE request_id=?
            """,
            (request.request_id,),
        )
        assert store.append_tool_result(result) == event_id
        repaired = store.connection.execute(
            """
            SELECT status, result_id, result_json
            FROM tool_requests
            WHERE request_id=?
            """,
            (request.request_id,),
        ).fetchone()
        assert repaired["status"] == result.status
        assert repaired["result_id"] == result.result_id
        assert repaired["result_json"] == canonical_json_text(result.to_dict())
        assert (
            store.connection.execute("SELECT COUNT(*) FROM journal").fetchone()[0]
            == journal_count
        )

        store.connection.execute(
            """
            UPDATE tool_requests
            SET status='queued', result_id=NULL, result_json=NULL
            WHERE request_id=?
            """,
            (request.request_id,),
        )
        conflicting = ToolResultRecord(
            idempotency_key="result-repair-conflict",
            request_id=request.request_id,
            status="complete",
            payload={"nested": {"values": [9]}},
            parent_ids=(request.request_id,),
        )
        with pytest.raises(
            IdempotencyConflictError,
            match="journal already has a different result",
        ):
            store.append_tool_result(conflicting)
        assert (
            store.connection.execute("SELECT COUNT(*) FROM journal").fetchone()[0]
            == journal_count
        )
        assert store.append_tool_result(result) == event_id

        prior = store.connection.execute(
            "SELECT event_id FROM journal ORDER BY seq DESC LIMIT 1"
        ).fetchone()["event_id"]
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            store.connection.execute(
                """
                INSERT INTO journal(
                    event_id,
                    previous_event_id,
                    event_type,
                    record_id,
                    canonical_json
                ) VALUES (?, ?, 'tool_result', ?, ?)
                """,
                (
                    "duplicate-event-id",
                    prior,
                    result.result_id,
                    "{}",
                ),
            )


def test_consolidator_replaces_proposer_candidate_and_replay_yields_decision(
    tmp_path,
) -> None:
    database = tmp_path / "consolidator.sqlite3"
    identities = _identities(3)
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    with RuntimeStore(database) as store:
        store.initialize(genesis, identities, _initial_states(identities, genesis))
        values = _tick_inputs(
            store,
            identities,
            proposal_text="A",
            committed_text="B",
        )
        proposal = values[2]
        decision = values[3]
        assert proposal.delta is not None
        assert decision.committed_delta is not None
        assert proposal.delta.delta_id != decision.committed_delta_id
        assert proposal.delta.author_core_id == values[1].proposer_core_id
        assert (
            decision.committed_delta.author_core_id
            == values[1].consolidator_core_id
        )
        working = apply_system_update(genesis, values[8]).working_snapshot
        assert apply_delta(working, proposal.delta).region(
            "response_draft"
        ).text == "A"
        assert compose_tick_transaction(
            genesis,
            values[8],
            decision.committed_delta,
            consolidator_author_core_id=values[1].consolidator_core_id,
        ).final_snapshot == values[4]
        assert values[4].parent_field_id == genesis.field_id

        _commit(store, values)
        assert store.current_head().snapshot.region("response_draft").text == "B"
        assert "A" not in store.current_head().snapshot.region(
            "response_draft"
        ).text


def test_rejected_tick_preserves_system_overlay_without_private_effects(
    tmp_path,
) -> None:
    database = tmp_path / "rejected-overlay.sqlite3"
    identities = _identities(3)
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    with RuntimeStore(database) as store:
        store.initialize(genesis, identities, _initial_states(identities, genesis))
        rejected = _tick_inputs(
            store,
            identities,
            accepted=False,
            system_operations=(
                AppendSpans(
                    region=LogicalRegion.USER_INPUT,
                    spans=(
                        FieldSpan(
                            span_id="rejected-tick-input",
                            text="sealed evidence",
                            kind="user_input",
                            source="test",
                            provenance="source-event:test",
                        ),
                    ),
                ),
            ),
        )
        assert rejected[5] == ()
        assert rejected[7] == ()
        candidate_state = _tick_inputs(store, identities)[5]
        leaked_tool = ToolRequestRecord(
            tick_seq=rejected[0].tick_seq,
            requester_core_id=rejected[1].consolidator_core_id,
            base_field_id=rejected[2].base_field_id,
            tool_name="must-not-run",
            arguments={},
            parent_proposal_id=rejected[2].proposal_id,
        )
        journal_before = store.connection.execute(
            "SELECT COUNT(*) FROM journal"
        ).fetchone()[0]

        for updates, requests in (
            (candidate_state, ()),
            ((), (leaked_tool,)),
        ):
            with pytest.raises(
                StoreContractError,
                match="rejected consolidation",
            ):
                store.commit_tick(
                    rejected[0].generation,
                    rejected[0].snapshot.field_id,
                    rejected[1],
                    rejected[8],
                    rejected[2],
                    rejected[3],
                    rejected[4],
                    updates,
                    rejected[6],
                    requests,
                )
            assert store.current_head() == rejected[0]
            assert (
                store.connection.execute(
                    "SELECT COUNT(*) FROM journal"
                ).fetchone()[0]
                == journal_before
            )

        commit = _commit(store, rejected)
        assert commit.updated_core_state_manifest_ids == ()
        assert commit.tool_request_ids == ()
        head = store.recover()
        assert head.generation == 1
        assert head.tick_seq == 1
        assert head.snapshot.parent_field_id == genesis.field_id
        assert head.snapshot.region("user_input").text == "sealed evidence"
        assert head.snapshot.region("response_draft").text == ""
        assert store.replay_journal() == head


def test_ingress_consumption_is_atomic_and_restart_safe(tmp_path) -> None:
    database = tmp_path / "atomic-ingress.sqlite3"
    identities = _identities(3)
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    first_event = IngressEvent(
        idempotency_key="phone-0",
        source="phone",
        source_sequence=0,
        event_kind="user_input",
        exact_text="first",
        provenance="exact:phone:0",
    )
    second_event = IngressEvent(
        idempotency_key="phone-1",
        source="phone",
        source_sequence=1,
        event_kind="user_input",
        exact_text="second",
        provenance="exact:phone:1",
    )

    with RuntimeStore(database) as store:
        store.initialize(genesis, identities, _initial_states(identities, genesis))
        queue = IngressQueue(store.connection)
        queue.enqueue(first_event)
        first_update = build_system_update(
            store.current_head().snapshot,
            queue.pending(),
        )
        first_values = _tick_inputs(
            store,
            identities,
            system_update_override=first_update,
        )
        first_commit = _commit(
            store,
            first_values,
            ingress_queue=queue,
            consumed_ingress_event_ids=(first_event.event_id,),
        )
        assert queue.pending() == ()
        assert queue.receipts()[0].tick_commit_id == first_commit.commit_id
        assert store.recover() == store.current_head()

        queue.enqueue(second_event)
        head_before = store.current_head()
        journal_before = tuple(
            tuple(row)
            for row in store.connection.execute(
                """
                SELECT
                    seq,
                    event_id,
                    previous_event_id,
                    event_type,
                    record_id,
                    canonical_json,
                    idempotency_key
                FROM journal
                ORDER BY seq
                """
            ).fetchall()
        )
        core_states_before = tuple(
            tuple(row)
            for row in store.connection.execute(
                """
                SELECT core_id, manifest_id, manifest_json
                FROM core_states
                ORDER BY core_id
                """
            ).fetchall()
        )
        receipts_before = queue.receipts()
        second_compiled = build_system_update(
            head_before.snapshot,
            queue.pending(),
        )
        failing_update = SystemFieldUpdate(
            base_field_id=head_before.snapshot.field_id,
            base_tick_id=head_before.snapshot.tick_id,
            operations=second_compiled.operations,
            evidence=(second_event.event_id, first_event.event_id),
        )
        failing_values = _tick_inputs(
            store,
            identities,
            system_update_override=failing_update,
        )
        with pytest.raises(IngressConsumptionError, match="already consumed"):
            _commit(
                store,
                failing_values,
                ingress_queue=queue,
                consumed_ingress_event_ids=(
                    second_event.event_id,
                    first_event.event_id,
                ),
            )
        assert store.current_head() == head_before
        assert queue.pending() == (second_event,)
        assert queue.receipts() == receipts_before
        assert tuple(
            tuple(row)
            for row in store.connection.execute(
                """
                SELECT
                    seq,
                    event_id,
                    previous_event_id,
                    event_type,
                    record_id,
                    canonical_json,
                    idempotency_key
                FROM journal
                ORDER BY seq
                """
            ).fetchall()
        ) == journal_before
        assert tuple(
            tuple(row)
            for row in store.connection.execute(
                """
                SELECT core_id, manifest_id, manifest_json
                FROM core_states
                ORDER BY core_id
                """
            ).fetchall()
        ) == core_states_before

    with RuntimeStore(database) as store:
        queue = IngressQueue(store.connection)
        assert store.recover() == head_before
        assert queue.pending() == (second_event,)
        retry_update = build_system_update(
            store.current_head().snapshot,
            queue.pending(),
        )
        retry_values = _tick_inputs(
            store,
            identities,
            system_update_override=retry_update,
        )
        second_commit = _commit(
            store,
            retry_values,
            ingress_queue=queue,
            consumed_ingress_event_ids=(second_event.event_id,),
        )
        assert queue.pending() == ()
        assert tuple(
            receipt.tick_commit_id for receipt in queue.receipts()
        ) == (first_commit.commit_id, second_commit.commit_id)
        head = store.recover()
        assert head.generation == 2
        assert head.snapshot.region("user_input").text == "second"
        assert head.snapshot.region("conversation_history").text == "firstsecond"


def test_replay_rejects_ingress_receipt_absent_from_tick_evidence(
    tmp_path,
) -> None:
    database = tmp_path / "forged-ingress-receipt.sqlite3"
    identities = _identities(3)
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    included = IngressEvent(
        idempotency_key="included",
        source="source",
        source_sequence=0,
        event_kind="diary",
        exact_text="included",
        provenance="exact:included",
    )
    omitted = IngressEvent(
        idempotency_key="omitted",
        source="source",
        source_sequence=1,
        event_kind="diary",
        exact_text="omitted",
        provenance="exact:omitted",
    )
    with RuntimeStore(database) as store:
        store.initialize(genesis, identities, _initial_states(identities, genesis))
        queue = IngressQueue(store.connection)
        queue.enqueue(included)
        update = build_system_update(
            store.current_head().snapshot,
            queue.pending(),
        )
        values = _tick_inputs(
            store,
            identities,
            system_update_override=update,
        )
        commit = _commit(
            store,
            values,
            ingress_queue=queue,
            consumed_ingress_event_ids=(included.event_id,),
        )
        queue.enqueue(omitted)
        forged = ConsumptionReceipt(
            event_id=omitted.event_id,
            tick_commit_id=commit.commit_id,
        )
        store.connection.execute(
            """
            INSERT INTO axon_ingress_consumption_receipts(
                receipt_id,
                event_id,
                tick_commit_id,
                canonical_json,
                canonical_sha256
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                forged.receipt_id,
                forged.event_id,
                forged.tick_commit_id,
                canonical_json_text(forged.to_dict()),
                forged.receipt_id,
            ),
        )
        with pytest.raises(
            JournalIntegrityError,
            match="absent from tick evidence",
        ):
            store.replay_journal()
        with pytest.raises(
            JournalIntegrityError,
            match="absent from tick evidence",
        ):
            store.recover()


def test_stale_role_and_cross_core_rejections_roll_back_atomically(tmp_path) -> None:
    database = tmp_path / "rollback.sqlite3"
    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    with RuntimeStore(database) as store:
        store.initialize(genesis, identities, _initial_states(identities, genesis))
        valid = _tick_inputs(store, identities)
        before = store.current_head()
        journal_before = store.connection.execute(
            "SELECT COUNT(*) FROM journal"
        ).fetchone()[0]
        with pytest.raises(StaleHeadError):
            store.commit_tick(
                before.generation + 1,
                before.snapshot.field_id,
                valid[1],
                valid[8],
                valid[2],
                valid[3],
                valid[4],
                valid[5],
                valid[6],
                valid[7],
            )
        assert store.current_head() == before
        assert (
            store.connection.execute("SELECT COUNT(*) FROM journal").fetchone()[0]
            == journal_before
        )

        wrong_decision = ConsolidationDecision(
            tick_seq=valid[3].tick_seq,
            assignment_hash=valid[3].assignment_hash,
            consolidator_core_id=valid[1].proposer_core_id,
            proposal_id=valid[3].proposal_id,
            base_field_id=valid[3].base_field_id,
            accepted=valid[3].accepted,
            output_field_id=valid[3].output_field_id,
            committed_delta=valid[3].committed_delta,
            reason="wrong owner",
        )
        with pytest.raises(RoleOwnershipError):
            store.commit_tick(
                before.generation,
                before.snapshot.field_id,
                valid[1],
                valid[8],
                valid[2],
                wrong_decision,
                valid[4],
                valid[5],
                valid[6],
                valid[7],
            )
        assert store.current_head() == before
        assert (
            store.connection.execute("SELECT COUNT(*) FROM journal").fetchone()[0]
            == journal_before
        )

        assert valid[3].committed_delta is not None
        proposer_authored_commit = FieldDelta(
            base_field_id=valid[3].base_field_id,
            base_tick_id=valid[2].base_tick_id,
            author_core_id=valid[1].proposer_core_id,
            pass_id="invalid-proposer-commit",
            operations=valid[3].committed_delta.operations,
        )
        wrong_author = ConsolidationDecision(
            tick_seq=valid[3].tick_seq,
            assignment_hash=valid[3].assignment_hash,
            consolidator_core_id=valid[1].consolidator_core_id,
            proposal_id=valid[3].proposal_id,
            base_field_id=valid[3].base_field_id,
            accepted=True,
            output_field_id=valid[3].output_field_id,
            committed_delta=proposer_authored_commit,
            reason="wrong delta author",
        )
        with pytest.raises(RoleOwnershipError):
            store.commit_tick(
                before.generation,
                before.snapshot.field_id,
                valid[1],
                valid[8],
                valid[2],
                wrong_author,
                valid[4],
                valid[5],
                valid[6],
                valid[7],
            )

        wrong_base_commit = FieldDelta(
            base_field_id="0" * 64,
            base_tick_id=before.snapshot.tick_id,
            author_core_id=valid[1].consolidator_core_id,
            pass_id="invalid-base-commit",
            operations=valid[3].committed_delta.operations,
        )
        wrong_base = ConsolidationDecision(
            tick_seq=valid[3].tick_seq,
            assignment_hash=valid[3].assignment_hash,
            consolidator_core_id=valid[1].consolidator_core_id,
            proposal_id=valid[3].proposal_id,
            base_field_id=valid[3].base_field_id,
            accepted=True,
            output_field_id=valid[3].output_field_id,
            committed_delta=wrong_base_commit,
            reason="wrong delta base",
        )
        with pytest.raises(RoleOwnershipError):
            store.commit_tick(
                before.generation,
                before.snapshot.field_id,
                valid[1],
                valid[8],
                valid[2],
                wrong_base,
                valid[4],
                valid[5],
                valid[6],
                valid[7],
            )

        proposer_tool_request = ToolRequestRecord(
            tick_seq=before.tick_seq,
            requester_core_id=valid[1].proposer_core_id,
            base_field_id=valid[2].base_field_id,
            tool_name="forbidden-proposer-tool",
            arguments={},
            parent_proposal_id=valid[2].proposal_id,
        )
        with pytest.raises(RoleOwnershipError):
            store.commit_tick(
                before.generation,
                before.snapshot.field_id,
                valid[1],
                valid[8],
                valid[2],
                valid[3],
                valid[4],
                valid[5],
                valid[6],
                (proposer_tool_request,),
            )

        sleeper_update = _tick_inputs(
            store,
            identities,
            update_role="sleeper",
        )
        with pytest.raises(CoreStateUpdateError):
            _commit(store, sleeper_update)
        assert store.current_head() == before
        assert (
            store.connection.execute("SELECT COUNT(*) FROM journal").fetchone()[0]
            == journal_before
        )


def test_journal_is_append_only_and_tamper_is_detected(tmp_path) -> None:
    database = tmp_path / "journal.sqlite3"
    identities = _identities(3)
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    with RuntimeStore(database) as store:
        store.initialize(genesis, identities, _initial_states(identities, genesis))
        store.verify_journal_chain()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.connection.execute(
                "UPDATE journal SET event_type='tampered' WHERE seq=1"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.connection.execute("DELETE FROM journal WHERE seq=1")
        store.verify_journal_chain()

        store.connection.execute("DROP TRIGGER journal_reject_update")
        store.connection.execute(
            "UPDATE journal SET canonical_json='{}' WHERE seq=1"
        )
        with pytest.raises(JournalIntegrityError):
            store.verify_journal_chain()


def test_replay_rejects_hash_valid_forward_journal_reference(tmp_path) -> None:
    database = tmp_path / "forward-reference.sqlite3"
    identities = _identities(3)
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    with RuntimeStore(database) as store:
        store.initialize(genesis, identities, _initial_states(identities, genesis))
        _commit(store, _tick_inputs(store, identities))
        proposal_seq = store.connection.execute(
            "SELECT seq FROM journal WHERE event_type='proposal'"
        ).fetchone()["seq"]
        commit_seq = store.connection.execute(
            "SELECT seq FROM journal WHERE event_type='tick_commit'"
        ).fetchone()["seq"]
        store.connection.execute("DROP TRIGGER journal_reject_update")
        store.connection.execute(
            "UPDATE journal SET seq=-1 WHERE seq=?",
            (proposal_seq,),
        )
        store.connection.execute(
            "UPDATE journal SET seq=? WHERE seq=?",
            (proposal_seq, commit_seq),
        )
        store.connection.execute(
            "UPDATE journal SET seq=? WHERE seq=-1",
            (commit_seq,),
        )

        rows = store.connection.execute(
            """
            SELECT seq, event_type, record_id, canonical_json
            FROM journal
            ORDER BY seq
            """
        ).fetchall()
        for row in rows:
            store.connection.execute(
                """
                UPDATE journal
                SET event_id=?, previous_event_id=NULL
                WHERE seq=?
                """,
                (f"temporary-{row['seq']}", row["seq"]),
            )
        previous = None
        for row in rows:
            payload = parse_json_object(row["canonical_json"])
            event_id = canonical_sha256(
                {
                    "previous_event_id": previous,
                    "event_type": row["event_type"],
                    "record_id": row["record_id"],
                    "payload": payload,
                }
            )
            store.connection.execute(
                """
                UPDATE journal
                SET event_id=?, previous_event_id=?
                WHERE seq=?
                """,
                (event_id, previous, row["seq"]),
            )
            previous = event_id

        store.verify_journal_chain()
        with pytest.raises(
            JournalIntegrityError,
            match="forward journal reference",
        ):
            store.replay_journal()
