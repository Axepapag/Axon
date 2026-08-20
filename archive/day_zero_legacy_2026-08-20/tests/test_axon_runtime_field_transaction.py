from __future__ import annotations

import pytest

from runtime.axon_runtime.field_transaction import (
    AppendSpans,
    ClearRegion,
    ConsolidatorAuthorError,
    FieldTransactionAudit,
    ReplayMismatchError,
    ReplaceSpans,
    StaleSystemUpdateError,
    SystemFieldUpdate,
    SystemRegionWriteError,
    apply_system_update,
    compose_tick_successor,
    compose_tick_transaction,
    replay_field_transaction,
)
from runtime.field import (
    FieldDelta,
    FieldSpan,
    InsertText,
    LogicalRegion,
    RegionState,
    SealedRegionWriteError,
    SharedFieldSnapshot,
    StaleDeltaError,
)


def _span(
    span_id: str,
    text: str,
    *,
    source: str,
    provenance: str,
    container_refs: tuple[str, ...] = (),
    edge_refs: tuple[str, ...] = (),
) -> FieldSpan:
    return FieldSpan(
        span_id=span_id,
        text=text,
        kind="test_evidence",
        source=source,
        provenance=provenance,
        confidence=0.875,
        container_refs=container_refs,
        edge_refs=edge_refs,
    )


def _head() -> SharedFieldSnapshot:
    return SharedFieldSnapshot(
        tick_id=19,
        regions=(
            RegionState(
                name="conversation_history",
                spans=(
                    _span(
                        "history-0",
                        "Earlier turn.\n",
                        source="transcript",
                        provenance="exact:turn:0",
                    ),
                ),
            ),
            RegionState.from_text(
                "scratch",
                "plan",
                span_id="scratch-0",
                source="core-old",
                provenance="tick:18",
            ),
            RegionState.from_text(
                "response_draft",
                "Draft",
                span_id="draft-0",
                source="core-old",
                provenance="tick:18",
            ),
        ),
        source_manifest_ids=("manifest-19",),
    )


def _update(head: SharedFieldSnapshot) -> SystemFieldUpdate:
    return SystemFieldUpdate(
        base_field_id=head.field_id,
        base_tick_id=head.tick_id,
        operations=(
            AppendSpans(
                "diary",
                (
                    _span(
                        "diary-19",
                        "Learned café 🙂.",
                        source="runtime_diary",
                        provenance="diary:19:utf8",
                    ),
                ),
            ),
            AppendSpans(
                "tool_results",
                (
                    _span(
                        "tool-19",
                        '{"temperature":"21°C"}',
                        source="weather_tool",
                        provenance="tool-call:abc:result",
                        container_refs=("container-weather",),
                        edge_refs=("edge-temperature",),
                    ),
                ),
            ),
            AppendSpans(
                "user_input",
                (
                    _span(
                        "user-19",
                        "Please continue.",
                        source="user",
                        provenance="queue:message:19",
                    ),
                ),
            ),
            AppendSpans(
                "advisor_input",
                (
                    _span(
                        "advisor-19",
                        "Preserve exact evidence.",
                        source="advisor",
                        provenance="advisor-call:19",
                    ),
                ),
            ),
        ),
        evidence=("queue:19", "tool-call:abc"),
    )


def test_queued_runtime_appends_are_visible_in_working_and_final() -> None:
    head = _head()
    update = _update(head)
    application = apply_system_update(head, update)
    working = application.working_snapshot

    assert working.tick_id == head.tick_id
    assert working.parent_field_id == head.field_id
    assert application.head_field_id == head.field_id
    assert application.working_field_id == working.field_id
    assert working.region("user_input").text == "Please continue."
    assert working.region("tool_results").text == '{"temperature":"21°C"}'
    assert working.region("advisor_input").text == "Preserve exact evidence."
    assert working.region("diary").text == "Learned café 🙂."

    delta = FieldDelta(
        base_field_id=working.field_id,
        base_tick_id=working.tick_id,
        author_core_id="core-128-consolidator",
        pass_id="consolidate-19",
        operations=(
            InsertText(
                region="response_draft",
                offset=len(working.region("response_draft").text),
                text=" ready",
                provenance="consolidator:19",
            ),
        ),
    )
    final = compose_tick_successor(
        head,
        update,
        delta,
        consolidator_author_core_id="core-128-consolidator",
    )
    assert final.tick_id == head.tick_id + 1
    assert final.parent_field_id == head.field_id
    assert final.region("user_input").text == "Please continue."
    assert final.region("tool_results").text == '{"temperature":"21°C"}'
    assert final.region("advisor_input").text == "Preserve exact evidence."
    assert final.region("diary").text == "Learned café 🙂."
    assert final.region("response_draft").text == "Draft ready"


def test_source_head_is_immutable_and_append_history_never_deletes() -> None:
    head = _head()
    existing = head.region("conversation_history").spans
    appended = _span(
        "history-1",
        "New turn.\n",
        source="transcript",
        provenance="exact:turn:1",
    )
    update = SystemFieldUpdate(
        base_field_id=head.field_id,
        base_tick_id=head.tick_id,
        operations=(AppendSpans("conversation_history", (appended,)),),
    )
    working = apply_system_update(head, update).working_snapshot

    assert head.region("conversation_history").spans == existing
    assert working.region("conversation_history").spans == (*existing, appended)
    assert working.region("conversation_history").spans[: len(existing)] == existing
    assert head.tick_id == 19
    assert head.parent_field_id is None


def test_core_still_cannot_write_sealed_runtime_regions() -> None:
    head = _head()
    update = _update(head)
    working = apply_system_update(head, update).working_snapshot
    illegal = FieldDelta(
        base_field_id=working.field_id,
        base_tick_id=working.tick_id,
        author_core_id="core-64-consolidator",
        pass_id="illegal",
        operations=(
            InsertText(
                region=LogicalRegion.CONVERSATION_HISTORY,
                offset=0,
                text="forged",
            ),
        ),
    )
    with pytest.raises(SealedRegionWriteError):
        compose_tick_successor(
            head,
            update,
            illegal,
            consolidator_author_core_id="core-64-consolidator",
        )


@pytest.mark.parametrize("region", ["scratch", "response_draft"])
def test_system_update_cannot_write_core_writable_regions(region: str) -> None:
    head = _head()
    update = SystemFieldUpdate(
        base_field_id=head.field_id,
        base_tick_id=head.tick_id,
        operations=(
            AppendSpans(
                region,
                (
                    _span(
                        f"illegal-{region}",
                        "runtime must not write this",
                        source="runtime",
                        provenance="forbidden",
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(SystemRegionWriteError):
        apply_system_update(head, update)


def test_stale_system_update_and_head_based_delta_are_rejected() -> None:
    head = _head()
    stale_update = SystemFieldUpdate(
        base_field_id="stale-field",
        base_tick_id=head.tick_id,
    )
    with pytest.raises(StaleSystemUpdateError):
        apply_system_update(head, stale_update)

    update = _update(head)
    stale_delta = FieldDelta(
        # The core saw the head, not the exact runtime overlay.
        base_field_id=head.field_id,
        base_tick_id=head.tick_id,
        author_core_id="core-consolidator",
        pass_id="stale-overlay",
        operations=(
            InsertText(region="scratch", offset=0, text="x"),
        ),
    )
    with pytest.raises(StaleDeltaError):
        compose_tick_successor(
            head,
            update,
            stale_delta,
            consolidator_author_core_id="core-consolidator",
        )


def test_delta_author_must_be_supplied_and_match_independently() -> None:
    head = _head()
    update = _update(head)
    working = apply_system_update(head, update).working_snapshot
    delta = FieldDelta(
        base_field_id=working.field_id,
        base_tick_id=working.tick_id,
        author_core_id="actual-core",
        pass_id="author-check",
        operations=(InsertText(region="scratch", offset=0, text="x"),),
    )
    with pytest.raises(ConsolidatorAuthorError):
        compose_tick_successor(head, update, delta)
    with pytest.raises(ConsolidatorAuthorError):
        compose_tick_successor(
            head,
            update,
            delta,
            consolidator_author_core_id="different-core",
        )


def test_noop_successor_still_advances_exactly_one_tick() -> None:
    head = _head()
    update = SystemFieldUpdate(
        base_field_id=head.field_id,
        base_tick_id=head.tick_id,
    )
    audit = compose_tick_transaction(head, update, None)
    final = audit.final_snapshot

    assert audit.before_snapshot is head
    assert audit.working_snapshot.tick_id == head.tick_id
    assert audit.before_field_id == head.field_id
    assert audit.working_field_id == audit.working_snapshot.field_id
    assert audit.final_field_id == final.field_id
    assert final.tick_id == head.tick_id + 1
    assert final.parent_field_id == head.field_id
    assert final.regions == head.regions
    assert compose_tick_successor(head, update, None) == final


def test_exact_audit_replay_and_unicode_span_provenance_are_preserved() -> None:
    head = _head()
    update = _update(head)
    working = apply_system_update(head, update).working_snapshot
    original_tool_span = next(
        span
        for span in working.region("tool_results").spans
        if span.span_id == "tool-19"
    )
    delta = FieldDelta(
        base_field_id=working.field_id,
        base_tick_id=working.tick_id,
        author_core_id="core-exact",
        pass_id="exact-replay",
        operations=(
            InsertText(
                region="scratch",
                offset=len(working.region("scratch").text),
                text=" → résumé",
                provenance="unicode-core-output",
            ),
        ),
    )
    audit = compose_tick_transaction(
        head,
        update,
        delta,
        consolidator_author_core_id="core-exact",
    )
    replayed = replay_field_transaction(audit)

    assert replayed.to_canonical_dict() == audit.final_snapshot.to_canonical_dict()
    assert replayed.field_id == audit.final_field_id
    replayed_tool_span = next(
        span
        for span in replayed.region("tool_results").spans
        if span.span_id == "tool-19"
    )
    assert replayed_tool_span == original_tool_span
    assert replayed_tool_span.text.encode("utf-8") == (
        '{"temperature":"21°C"}'.encode("utf-8")
    )
    assert replayed_tool_span.provenance == "tool-call:abc:result"
    assert replayed_tool_span.container_refs == ("container-weather",)
    assert replayed_tool_span.edge_refs == ("edge-temperature",)
    assert audit.audit_id == audit.canonical_hash


def test_replay_detects_a_forged_exact_final_snapshot() -> None:
    head = _head()
    update = SystemFieldUpdate(
        base_field_id=head.field_id,
        base_tick_id=head.tick_id,
    )
    audit = compose_tick_transaction(head, update, None)
    forged_final = SharedFieldSnapshot(
        tick_id=audit.final_snapshot.tick_id,
        parent_field_id=head.field_id,
        regions=(
            RegionState.from_text(
                "conversation_history",
                "forged",
                span_id="forged",
            ),
        ),
        source_manifest_ids=head.source_manifest_ids,
    )
    forged_audit = FieldTransactionAudit(
        before_snapshot=audit.before_snapshot,
        system_update=audit.system_update,
        working_snapshot=audit.working_snapshot,
        consolidator_delta=None,
        consolidator_author_core_id=None,
        final_snapshot=forged_final,
    )
    with pytest.raises(ReplayMismatchError):
        replay_field_transaction(forged_audit)


def test_replace_and_clear_are_explicit_whole_region_operations() -> None:
    head = _head()
    replacement = _span(
        "task-19",
        "phase=execute",
        source="scheduler",
        provenance="task-state:19",
    )
    with_old_state = SharedFieldSnapshot(
        tick_id=head.tick_id,
        parent_field_id=head.parent_field_id,
        source_manifest_ids=head.source_manifest_ids,
        regions=(
            *(
                region
                for region in head.regions
                if region.name
                not in {
                    LogicalRegion.TASK_STATE,
                    LogicalRegion.USER_INPUT,
                }
            ),
            RegionState.from_text(
                "task_state",
                "phase=old",
                span_id="task-old",
            ),
            RegionState.from_text(
                "user_input",
                "consumed input",
                span_id="user-old",
            ),
        ),
    )
    update = SystemFieldUpdate(
        base_field_id=with_old_state.field_id,
        base_tick_id=with_old_state.tick_id,
        operations=(
            ReplaceSpans("task_state", (replacement,)),
            ClearRegion("user_input"),
        ),
    )
    working = apply_system_update(with_old_state, update).working_snapshot
    assert working.region("task_state").spans == (replacement,)
    assert working.region("user_input").spans == ()
