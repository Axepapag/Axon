from __future__ import annotations

import sqlite3

import pytest

from runtime.axon_runtime.field_transaction import (
    AppendSpans,
    ClearRegion,
    ReplaceSpans,
    apply_system_update,
    compose_tick_successor,
)
from runtime.axon_runtime.ingress import (
    IngressConflictError,
    IngressConsumptionError,
    IngressEvent,
    IngressQueue,
    IngressSequenceError,
    build_system_update,
)
from runtime.field import FieldSpan, RegionState, SharedFieldSnapshot


def _event(
    key: str,
    source: str,
    sequence: int,
    kind: str,
    text: str,
    *,
    provenance: str | None = None,
) -> IngressEvent:
    return IngressEvent(
        idempotency_key=key,
        source=source,
        source_sequence=sequence,
        event_kind=kind,
        exact_text=text,
        provenance=provenance or f"exact:{key}",
    )


def _head(*, user_text: str = "") -> SharedFieldSnapshot:
    regions = [
        RegionState(
            name="conversation_history",
            spans=(
                FieldSpan(
                    span_id="history-existing",
                    text="Earlier.\n",
                    source="transcript",
                    provenance="exact:earlier",
                ),
            ),
        )
    ]
    if user_text:
        regions.append(
            RegionState.from_text(
                "user_input",
                user_text,
                span_id="user-previous",
                source="user",
                provenance="prior-tick",
            )
        )
    return SharedFieldSnapshot(tick_id=4, regions=tuple(regions))


def test_restart_persistence_and_pending_fifo(tmp_path) -> None:
    database = tmp_path / "ingress.sqlite3"
    events = (
        _event("user-0", "phone", 0, "user_input", "first"),
        _event("tool-0", "weather", 0, "tool_result", "21°C"),
        _event("user-1", "phone", 1, "user_input", "second"),
    )
    connection = sqlite3.connect(database)
    queue = IngressQueue(connection)
    assert not connection.in_transaction
    for event in events:
        queue.enqueue(event)
    connection.commit()
    connection.close()

    reopened = sqlite3.connect(database)
    queue = IngressQueue(reopened)
    assert queue.pending() == events
    assert queue.pending(limit=2) == events[:2]
    assert queue.get_event(events[1].event_id) == events[1]
    reopened.close()


def test_idempotency_conflicts_and_source_sequence_regression_fail() -> None:
    connection = sqlite3.connect(":memory:")
    queue = IngressQueue(connection)
    original = _event("message-7", "phone", 7, "user_input", "hello")
    assert queue.enqueue(original) == original
    assert queue.enqueue(original) == original

    with pytest.raises(IngressConflictError, match="idempotency"):
        queue.enqueue(
            _event(
                "message-7",
                "phone",
                7,
                "user_input",
                "different",
            )
        )
    with pytest.raises(IngressSequenceError, match="exceed"):
        queue.enqueue(
            _event(
                "message-other-key",
                "phone",
                7,
                "user_input",
                "same position",
            )
        )
    with pytest.raises(IngressSequenceError, match="exceed"):
        queue.enqueue(
            _event(
                "message-old",
                "phone",
                6,
                "user_input",
                "older",
            )
        )
    assert queue.pending() == (original,)


def test_consumption_receipts_share_outer_transaction_and_rollback() -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    queue = IngressQueue(connection)
    events = (
        _event("event-0", "source", 0, "diary", "one"),
        _event("event-1", "source", 1, "diary", "two"),
    )
    for event in events:
        queue.enqueue(event)

    with pytest.raises(IngressConsumptionError, match="begin"):
        queue.consume_in_current_transaction(
            tuple(event.event_id for event in events),
            "tick-commit-1",
        )

    connection.execute("BEGIN IMMEDIATE")
    receipts = queue.consume_in_current_transaction(
        tuple(event.event_id for event in events),
        "tick-commit-1",
    )
    assert tuple(receipt.event_id for receipt in receipts) == tuple(
        event.event_id for event in events
    )
    assert queue.pending() == ()
    connection.rollback()
    assert queue.pending() == events
    assert queue.receipts() == ()

    connection.execute("BEGIN IMMEDIATE")
    queue.consume_in_current_transaction(
        tuple(event.event_id for event in events),
        "tick-commit-2",
    )
    connection.commit()
    assert queue.pending() == ()
    assert len(queue.receipts()) == 2
    assert queue.get_event(events[0].event_id) == events[0]


def test_batch_consumption_uses_savepoint_to_avoid_partial_receipts() -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    queue = IngressQueue(connection)
    event = _event("event-0", "source", 0, "diary", "one")
    queue.enqueue(event)
    connection.execute("BEGIN")
    with pytest.raises(IngressConsumptionError, match="unknown"):
        queue.consume_in_current_transaction(
            (event.event_id, "0" * 64),
            "tick-commit-partial",
        )
    assert queue.receipts() == ()
    assert queue.pending() == (event,)
    connection.rollback()


def test_build_update_preserves_exact_sources_and_combines_regions() -> None:
    connection = sqlite3.connect(":memory:")
    queue = IngressQueue(connection)
    events = (
        _event("user-0", "phone", 0, "user_input", "café 🙂"),
        _event("user-1", "phone", 1, "user_input", "第二条"),
        _event(
            "history-0",
            "archive",
            0,
            "conversation_history",
            "Archived exact line.\n",
        ),
        _event(
            "tool-0",
            "weather_tool",
            0,
            "tool_result",
            '{"temperature":"21°C","raw":[1,2]}',
            provenance="tool-call:raw-result",
        ),
        _event("advisor-0", "advisor", 0, "advisor_result", "Keep evidence."),
        _event("diary-0", "diary", 0, "diary", "Remember résumé."),
        _event(
            "knowledge-0",
            "knowledge_import",
            0,
            "structured_knowledge",
            "Axon uses tools",
        ),
        _event(
            "situation-0",
            "sensor",
            0,
            "situation_awareness",
            "old situation",
        ),
        _event(
            "situation-1",
            "sensor",
            1,
            "situation_awareness",
            "new situation",
        ),
        _event("task-0", "scheduler", 0, "task_state", "queued"),
        _event("task-1", "scheduler", 1, "task_state", "running"),
    )
    for event in events:
        queue.enqueue(event)
    pending = queue.pending()
    head = _head(user_text="previous input")
    update = build_system_update(head, pending)

    regions = [operation.region for operation in update.operations]
    assert len(regions) == len(set(regions))
    assert set(update.evidence) == {event.event_id for event in events}
    assert sum(
        isinstance(operation, AppendSpans)
        and operation.region.value == "conversation_history"
        for operation in update.operations
    ) == 1
    assert sum(
        isinstance(operation, ReplaceSpans)
        and operation.region.value == "situation_awareness"
        for operation in update.operations
    ) == 1

    working = apply_system_update(head, update).working_snapshot
    assert head.region("conversation_history").text == "Earlier.\n"
    assert working.region("conversation_history").text == (
        "Earlier.\ncafé 🙂第二条Archived exact line.\n"
    )
    assert working.region("user_input").text == "café 🙂第二条"
    assert working.region("tool_results").text == (
        '{"temperature":"21°C","raw":[1,2]}'
    )
    tool_span = working.region("tool_results").spans[0]
    assert tool_span.text.encode("utf-8") == (
        '{"temperature":"21°C","raw":[1,2]}'.encode("utf-8")
    )
    assert tool_span.source == "weather_tool"
    assert tool_span.provenance == "tool-call:raw-result"
    assert tool_span.container_refs == ()
    assert tool_span.edge_refs == ()
    assert working.region("advisor_input").text == "Keep evidence."
    assert working.region("diary").text == "Remember résumé."
    assert working.region("structured_knowledge").text == "Axon uses tools"
    assert working.region("situation_awareness").text == "new situation"
    assert working.region("task_state").text == "running"

    # Building an update never consumes, rewrites, or discards source events.
    assert queue.pending() == events
    assert queue.get_event(events[7].event_id).exact_text == "old situation"


def test_user_input_is_visible_for_one_committed_tick_then_cleared() -> None:
    head = _head()
    user = _event("user-once", "phone", 0, "user_input", "one tick only")
    first_update = build_system_update(head, (user,))
    first = compose_tick_successor(head, first_update, None)
    assert first.region("user_input").text == "one tick only"
    assert first.region("conversation_history").text.endswith("one tick only")

    clear_update = build_system_update(first, ())
    assert any(
        isinstance(operation, ClearRegion)
        and operation.region.value == "user_input"
        for operation in clear_update.operations
    )
    second = compose_tick_successor(first, clear_update, None)
    assert second.region("user_input").spans == ()
    assert second.region("conversation_history").text == (
        first.region("conversation_history").text
    )

    already_clear = build_system_update(second, ())
    assert all(
        operation.region.value != "user_input"
        for operation in already_clear.operations
    )


def test_event_and_receipt_tables_are_insert_only() -> None:
    connection = sqlite3.connect(":memory:", isolation_level=None)
    queue = IngressQueue(connection)
    event = _event("event-0", "source", 0, "diary", "immutable")
    queue.enqueue(event)
    connection.execute("BEGIN")
    queue.consume_in_current_transaction((event.event_id,), "tick-commit-1")
    connection.commit()

    with pytest.raises(sqlite3.IntegrityError, match="insert-only"):
        connection.execute(
            """
            UPDATE axon_ingress_events
            SET exact_utf8=?
            WHERE event_id=?
            """,
            (b"forged", event.event_id),
        )
    with pytest.raises(sqlite3.IntegrityError, match="insert-only"):
        connection.execute(
            "DELETE FROM axon_ingress_events WHERE event_id=?",
            (event.event_id,),
        )
    with pytest.raises(sqlite3.IntegrityError, match="insert-only"):
        connection.execute(
            """
            UPDATE axon_ingress_consumption_receipts
            SET tick_commit_id='forged'
            WHERE event_id=?
            """,
            (event.event_id,),
        )
