from __future__ import annotations

import sqlite3

from runtime.axon_runtime.dormant import DormantStore
from runtime.axon_runtime.idle import DormantIdleWorker
from runtime.axon_runtime.projection import ProjectionPolicy, build_active_projection
from runtime.field import (
    CANONICAL_REGION_ORDER,
    FieldSpan,
    LogicalRegion,
    RegionState,
    SharedFieldSnapshot,
)


def _policy() -> ProjectionPolicy:
    budgets = {
        region: 64 for region in CANONICAL_REGION_ORDER
    }
    budgets[LogicalRegion.CONVERSATION_HISTORY] = 16
    return ProjectionPolicy(
        global_char_budget=256,
        max_selected_spans=32,
        region_char_budgets=tuple(
            (region.value, budgets[region])
            for region in CANONICAL_REGION_ORDER
        ),
    )


def test_rotating_sleeper_quantum_preserves_source_and_builds_candidates() -> None:
    connection = sqlite3.connect(":memory:")
    store = DormantStore(connection)
    worker = DormantIdleWorker(store)
    old = FieldSpan(
        span_id="history-old",
        text="Axon uses durable memory.",
        source="conversation",
        provenance="turn:1",
    )
    recent = FieldSpan(
        span_id="history-recent",
        text="new",
        source="conversation",
        provenance="turn:2",
    )
    full = SharedFieldSnapshot(
        tick_id=4,
        regions=(
            RegionState(
                name=LogicalRegion.CONVERSATION_HISTORY,
                spans=(old, recent),
            ),
        ),
    )
    projection = build_active_projection(full, policy=_policy(), store=store)
    assert projection.active_snapshot.region(
        LogicalRegion.CONVERSATION_HISTORY
    ).text == recent.text

    result = worker.run_projection_quantum(
        projection,
        owner_core_id="core-64-b",
        now_tick=4,
        capture_limit=1,
        job_limit=1,
    )
    assert len(result.captured_source_ids) == 1
    assert len(result.completed_job_keys) == 1
    assert result.dormant_span_ids == (old.span_id,)
    loaded = store.get_source(result.captured_source_ids[0])
    assert loaded.exact_text == old.text
    assert loaded.exact_text.encode("utf-8") == old.text.encode("utf-8")
    assert full.region(LogicalRegion.CONVERSATION_HISTORY).spans == (old, recent)

    triples = connection.execute(
        "SELECT revision_id FROM triple_revisions"
    ).fetchall()
    assert len(triples) == 1
    lifecycle = store.latest_knowledge(triples[0][0])
    assert lifecycle is not None
    assert lifecycle.state == "candidate"
    assert "not semantic proof" in lifecycle.reason

    after = build_active_projection(full, policy=_policy(), store=store)
    assert after.active_snapshot.region(
        LogicalRegion.CONVERSATION_HISTORY
    ).text == recent.text
    assert next(
        item for item in after.manifest.masked if item.span_id == old.span_id
    ).reason == "lifecycle_dormant"


def test_idle_quantum_is_bounded_and_idempotent_after_completion() -> None:
    connection = sqlite3.connect(":memory:")
    store = DormantStore(connection)
    worker = DormantIdleWorker(store)
    spans = tuple(
        FieldSpan(
            span_id=f"old-{index}",
            text=f"Axon has memory {index}.",
            source="conversation",
            provenance=f"turn:{index}",
        )
        for index in range(2)
    )
    full = SharedFieldSnapshot(
        tick_id=9,
        regions=(
            RegionState(
                name=LogicalRegion.CONVERSATION_HISTORY,
                spans=spans,
            ),
        ),
    )
    projection = build_active_projection(full, policy=_policy(), store=store)
    first = worker.run_projection_quantum(
        projection,
        owner_core_id="core-128-a",
        now_tick=9,
        capture_limit=1,
        job_limit=1,
    )
    assert len(first.captured_source_ids) == 1
    assert len(first.completed_job_keys) == 1
    second_projection = build_active_projection(full, policy=_policy(), store=store)
    second = worker.run_projection_quantum(
        second_projection,
        owner_core_id="core-64-a",
        now_tick=10,
        capture_limit=1,
        job_limit=1,
    )
    assert len(second.captured_source_ids) == 1
    assert len(second.completed_job_keys) == 1
    assert set(first.dormant_span_ids + second.dormant_span_ids) == {
        span.span_id for span in spans
    }
    idle = worker.process_quantum(
        owner_core_id="core-128-b",
        now_tick=11,
        job_limit=1,
    )
    assert idle.leased_job_keys == ()
    assert idle.completed_job_keys == ()


def test_capture_retry_reuses_exact_source_identity_after_crash_boundary() -> None:
    connection = sqlite3.connect(":memory:")
    store = DormantStore(connection)
    worker = DormantIdleWorker(store)
    old = FieldSpan(
        span_id="retry-old",
        text="Axon needs restart safety.",
        source="conversation",
        provenance="turn:retry",
    )
    full = SharedFieldSnapshot(
        tick_id=12,
        regions=(
            RegionState(
                name=LogicalRegion.CONVERSATION_HISTORY,
                spans=(old,),
            ),
        ),
    )
    projection = build_active_projection(full, policy=_policy(), store=store)
    first = worker.capture_projection_omissions(
        projection,
        tick_id=12,
        limit=1,
    )
    # Simulate restart before a sleeper leased/completed the queued job.
    retry_projection = build_active_projection(full, policy=_policy(), store=store)
    second = DormantIdleWorker(store).capture_projection_omissions(
        retry_projection,
        tick_id=13,
        limit=1,
    )
    assert second == first
    assert (
        connection.execute("SELECT COUNT(*) FROM source_records").fetchone()[0]
        == 1
    )
    completed = worker.process_quantum(
        owner_core_id="core-restart",
        now_tick=13,
        job_limit=1,
    )
    assert completed.dormant_span_ids == (old.span_id,)
