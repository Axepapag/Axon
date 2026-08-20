from __future__ import annotations

from dataclasses import replace
import hashlib
import sqlite3

import pytest

from runtime.axon_runtime.dormant import (
    BOOTSTRAP_EXTRACTOR_ID,
    DormantConflictError,
    DormantStore,
    ExtractionBundle,
    SourceRecord,
    bootstrap_extract,
)
from runtime.axon_runtime.projection import (
    ProjectionPolicy,
    build_active_projection,
    retrieve_and_surface,
)
from runtime.field import (
    CANONICAL_REGION_ORDER,
    FieldSpan,
    LogicalRegion,
    RegionState,
    SharedFieldSnapshot,
)


def _store() -> DormantStore:
    return DormantStore(sqlite3.connect(":memory:"))


def _source(
    text: str,
    *,
    sequence: int = 0,
    stream: str = "conversation",
    tick: int = 0,
) -> SourceRecord:
    return SourceRecord(
        stream_id=stream,
        sequence=sequence,
        region=LogicalRegion.CONVERSATION_HISTORY,
        exact_text=text,
        source="test:conversation",
        provenance=f"test-record:{sequence}",
        tick_id=tick,
    )


def _policy(
    *,
    global_budget: int = 512,
    conversation_budget: int = 128,
) -> ProjectionPolicy:
    budgets = {
        region: 64
        for region in CANONICAL_REGION_ORDER
    }
    budgets[LogicalRegion.CONVERSATION_HISTORY] = conversation_budget
    budgets[LogicalRegion.STRUCTURED_KNOWLEDGE] = 128
    return ProjectionPolicy(
        global_char_budget=global_budget,
        max_selected_spans=64,
        region_char_budgets=tuple(
            (region.value, budgets[region])
            for region in CANONICAL_REGION_ORDER
        ),
    )


def _snapshot(*regions: RegionState) -> SharedFieldSnapshot:
    return SharedFieldSnapshot(tick_id=7, regions=tuple(regions))


def _accept_bundle(
    store: DormantStore,
    source: SourceRecord,
    *,
    tick: int,
):
    store.append_source(source)
    bundle = bootstrap_extract(source)
    store.append_extraction_bundle(bundle)
    for triple in bundle.triples:
        store.transition_knowledge(
            target_kind="triple",
            target_revision_id=triple.revision_id,
            state="candidate",
            reason="bootstrap candidate only",
            tick_id=tick,
        )
        store.transition_knowledge(
            target_kind="triple",
            target_revision_id=triple.revision_id,
            state="accepted",
            reason="test validator accepted readable assertion",
            tick_id=tick + 1,
        )
    return bundle


def test_million_character_source_is_preserved_but_projection_is_bounded() -> None:
    enormous = "x" * 1_000_000
    old_span = FieldSpan(
        span_id="history-million",
        text=enormous,
        source="test",
        provenance="exact-million",
    )
    recent_span = FieldSpan(
        span_id="history-recent",
        text="recent exact history",
        source="test",
        provenance="exact-recent",
    )
    source = _snapshot(
        RegionState(
            name=LogicalRegion.CONVERSATION_HISTORY,
            spans=(old_span, recent_span),
        )
    )
    source_id = source.field_id

    result = build_active_projection(
        source,
        policy=_policy(global_budget=256, conversation_budget=64),
    )

    assert result.full_snapshot is source
    assert source.field_id == source_id
    assert source.region("conversation_history").text == enormous + recent_span.text
    assert result.active_snapshot.region("conversation_history").text == recent_span.text
    assert result.manifest.selected_char_count <= 256
    assert result.manifest.source_char_count == len(enormous) + len(recent_span.text)
    omitted = {item.span_id: item for item in result.manifest.masked}
    assert omitted["history-million"].char_count == len(enormous)
    assert omitted["history-million"].exact_text_sha256 == hashlib.sha256(
        enormous.encode("utf-8")
    ).hexdigest()
    assert omitted["history-million"].reason == "whole_span_exceeds_region_budget"


def test_projection_pins_runtime_critical_regions_without_slicing() -> None:
    regions = []
    expected = {}
    for region in (
        LogicalRegion.USER_INPUT,
        LogicalRegion.RESPONSE_DRAFT,
        LogicalRegion.SCRATCH,
        LogicalRegion.TOOL_RESULTS,
    ):
        text = f"exact {region.value}"
        expected[region] = text
        regions.append(
            RegionState.from_text(
                region,
                text,
                span_id=f"pin-{region.value}",
                source="test",
                provenance="pinned",
            )
        )
    result = build_active_projection(_snapshot(*regions), policy=_policy())
    for region, text in expected.items():
        assert result.active_snapshot.region(region).text == text
        selected = next(
            item
            for item in result.manifest.selected
            if item.logical_region is region
        )
        assert selected.pinned is True


def test_dormant_tool_results_are_masked_even_though_region_is_pinned() -> None:
    store = _store()
    old = FieldSpan(
        span_id="tool-old",
        text="old exact tool output",
        source="tool:test",
        provenance="result:old",
    )
    current = FieldSpan(
        span_id="tool-current",
        text="current tool output",
        source="tool:test",
        provenance="result:current",
    )
    source = _source(old.text)
    store.append_source(source)
    store.transition_span(
        span_id=old.span_id,
        source_record_id=source.record_id,
        state="active",
        reason="captured",
        tick_id=0,
    )
    store.transition_span(
        span_id=old.span_id,
        source_record_id=source.record_id,
        state="masked",
        reason="working-set rotation",
        tick_id=1,
    )
    store.transition_span(
        span_id=old.span_id,
        source_record_id=source.record_id,
        state="dormant",
        reason="processed",
        tick_id=2,
    )
    snapshot = _snapshot(
        RegionState(
            name=LogicalRegion.TOOL_RESULTS,
            spans=(old, current),
        )
    )
    result = build_active_projection(snapshot, policy=_policy(), store=store)
    assert snapshot.region(LogicalRegion.TOOL_RESULTS).spans == (old, current)
    assert result.active_snapshot.region(LogicalRegion.TOOL_RESULTS).spans == (
        current,
    )
    omitted = next(
        item for item in result.manifest.masked if item.span_id == old.span_id
    )
    assert omitted.reason == "lifecycle_dormant"


def test_active_masked_dormant_resurfaced_is_append_only() -> None:
    store = _store()
    span = FieldSpan(
        span_id="history-one",
        text="Axon has memory",
        source="test:conversation",
        provenance="test-record:0",
    )
    source_record = _source(span.text)
    store.append_source(source_record)
    store.transition_span(
        span_id=span.span_id,
        source_record_id=source_record.record_id,
        state="active",
        reason="captured",
        tick_id=0,
    )
    store.transition_span(
        span_id=span.span_id,
        source_record_id=source_record.record_id,
        state="masked",
        reason="working-set rotation",
        tick_id=1,
    )
    store.transition_span(
        span_id=span.span_id,
        source_record_id=source_record.record_id,
        state="dormant",
        reason="idle extraction complete",
        tick_id=2,
    )
    snapshot = _snapshot(
        RegionState(
            name=LogicalRegion.CONVERSATION_HISTORY,
            spans=(span,),
        )
    )
    dormant = build_active_projection(snapshot, policy=_policy(), store=store)
    assert dormant.active_snapshot.region("conversation_history").text == ""
    assert dormant.manifest.masked[0].reason == "lifecycle_dormant"

    resurfaced_event = store.transition_span(
        span_id=span.span_id,
        source_record_id=source_record.record_id,
        state="resurfaced",
        reason="retrieval lease",
        tick_id=3,
    )
    resurfaced = build_active_projection(snapshot, policy=_policy(), store=store)
    assert resurfaced.active_snapshot.region("conversation_history").text == span.text
    selected = next(
        item for item in resurfaced.manifest.selected if item.span_id == span.span_id
    )
    assert selected.lifecycle_event_id == resurfaced_event.event_id
    assert (
        store.connection.execute(
            "SELECT COUNT(*) FROM span_lifecycle_events WHERE span_id=?",
            (span.span_id,),
        ).fetchone()[0]
        == 4
    )


def test_unicode_source_archive_and_bootstrap_provenance_are_exact() -> None:
    store = _store()
    exact = "Axon uses café 🙂"
    source = _source(exact)
    assert store.append_source(source) is True
    assert store.append_source(source) is False
    loaded = store.get_source(source.record_id)
    assert loaded.exact_text == exact
    assert loaded.exact_text.encode("utf-8") == exact.encode("utf-8")

    bundle = bootstrap_extract(loaded)
    assert len(bundle.triples) == 1
    triple = bundle.triples[0]
    assert triple.object_text == "café 🙂"
    assert triple.extractor_id == BOOTSTRAP_EXTRACTOR_ID
    assert "not-semantic-proof" in triple.provenance
    store.append_extraction_bundle(bundle)
    ref = triple.provenance_refs[0]
    assert loaded.exact_text[ref.char_start : ref.char_end] == exact
    assert ref.exact_text_sha256 == hashlib.sha256(
        exact.encode("utf-8")
    ).hexdigest()


def test_bootstrap_extraction_ids_are_stable_and_patterns_are_conservative() -> None:
    source = _source(
        "Axon has memory. Axon uses tools.\n"
        "This resembles a relation but contains no allowed verb."
    )
    first = bootstrap_extract(source)
    second = bootstrap_extract(source)
    assert first.output_revision_ids == second.output_revision_ids
    assert [item.readable_text for item in first.triples] == [
        "Axon has memory",
        "Axon uses tools",
    ]
    assert all(
        item.extractor_id == BOOTSTRAP_EXTRACTOR_ID
        for item in first.triples
    )


def test_extraction_bundle_rejects_unknown_entity_refs_atomically() -> None:
    store = _store()
    source = _source("Axon uses tools.")
    store.append_source(source)
    valid = bootstrap_extract(source)
    forged = replace(valid.triples[0], subject_entity_id="entity-missing")
    bundle = ExtractionBundle(
        source_record_id=valid.source_record_id,
        extractor_id=valid.extractor_id,
        containers=valid.containers,
        entities=valid.entities,
        triples=(forged,),
    )
    with pytest.raises(DormantConflictError, match="subject_entity_id"):
        store.append_extraction_bundle(bundle)
    assert (
        store.connection.execute(
            "SELECT COUNT(*) FROM container_revisions"
        ).fetchone()[0]
        == 0
    )
    assert (
        store.connection.execute(
            "SELECT COUNT(*) FROM entity_revisions"
        ).fetchone()[0]
        == 0
    )


def test_conflicting_claims_coexist_as_distinct_immutable_triples() -> None:
    store = _store()
    tea = _accept_bundle(store, _source("Axon wants tea.", sequence=0), tick=0)
    coffee = _accept_bundle(
        store,
        _source("Axon wants coffee.", sequence=1),
        tick=2,
    )
    assert tea.triples[0].triple_id != coffee.triples[0].triple_id
    assert tea.triples[0].revision_id != coffee.triples[0].revision_id
    rows = store.connection.execute(
        "SELECT revision_id FROM triple_revisions ORDER BY revision_id"
    ).fetchall()
    assert len(rows) == 2
    retrieval = store.retrieve("Axon wants", limit=10)
    assert {hit.text for hit in retrieval.hits} == {
        "Axon wants tea",
        "Axon wants coffee",
    }


def test_bounded_job_leasing_and_expired_lease_recovery() -> None:
    store = _store()
    first = _source("Axon has memory.", sequence=0)
    second = _source("Axon uses tools.", sequence=1)
    store.append_source(first)
    store.append_source(second)
    first_job = store.enqueue_extraction(first.record_id)
    store.enqueue_extraction(second.record_id)

    leased = store.lease_jobs(
        owner="worker-a",
        now_tick=10,
        lease_ticks=2,
        limit=1,
    )
    assert len(leased) == 1
    assert leased[0].job_key == first_job.job_key
    assert store.recover_expired_jobs(now_tick=11, limit=1) == ()
    recovered = store.recover_expired_jobs(now_tick=12, limit=1)
    assert len(recovered) == 1
    assert recovered[0].state == "queued"
    assert recovered[0].attempt == 1

    released = store.lease_jobs(
        owner="worker-b",
        now_tick=12,
        lease_ticks=3,
        limit=1,
    )
    assert len(released) == 1
    completed = store.complete_job(
        released[0].job_key,
        owner="worker-b",
        output_revision_ids=("revision-one",),
    )
    assert completed.state == "completed"
    assert completed.output_revision_ids == ("revision-one",)
    assert (
        store.connection.execute(
            "SELECT COUNT(*) FROM extraction_job_events WHERE job_key=?",
            (first_job.job_key,),
        ).fetchone()[0]
        == 5
    )


def test_retrieval_and_surfacing_are_deterministic_with_explicit_omissions() -> None:
    store = _store()
    _accept_bundle(store, _source("Axon uses tools.", sequence=0), tick=0)
    _accept_bundle(store, _source("Axon uses memory.", sequence=1), tick=2)
    first = store.retrieve("Axon uses", limit=1)
    second = store.retrieve("Axon uses", limit=1)
    assert first == second
    assert len(first.hits) == 1
    assert len(first.omitted_revision_ids) == 1

    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.USER_INPUT: "What does Axon use",
            LogicalRegion.RESPONSE_DRAFT: "",
        },
        tick_id=5,
    )
    surfaced_a = retrieve_and_surface(
        store, snapshot, "Axon uses", limit=1
    )
    surfaced_b = retrieve_and_surface(
        store, snapshot, "Axon uses", limit=1
    )
    assert surfaced_a.record.surfacing_id == surfaced_b.record.surfacing_id
    assert surfaced_a.surfaced_snapshot.field_id == surfaced_b.surfaced_snapshot.field_id
    assert snapshot.region("structured_knowledge").text == ""
    assert surfaced_a.surfaced_snapshot.region("structured_knowledge").text
    assert len(surfaced_a.omissions) == 1
    span = surfaced_a.spans[0]
    assert span.container_refs
    assert span.edge_refs
    assert "exact_sources" in span.provenance


def test_insert_only_triggers_and_source_position_conflicts() -> None:
    store = _store()
    source = _source("Axon is persistent.")
    store.append_source(source)
    with pytest.raises(sqlite3.IntegrityError, match="insert-only"):
        store.connection.execute(
            "UPDATE source_records SET region='scratch' WHERE record_id=?",
            (source.record_id,),
        )
    store.connection.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="insert-only"):
        store.connection.execute(
            "DELETE FROM source_records WHERE record_id=?",
            (source.record_id,),
        )
    store.connection.rollback()
    with pytest.raises(DormantConflictError, match="stream position"):
        store.append_source(_source("different bytes", sequence=0))

    triggers = {
        row[0]
        for row in store.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
    }
    for table in (
        "source_records",
        "span_lifecycle_events",
        "container_revisions",
        "entity_revisions",
        "triple_revisions",
        "knowledge_lifecycle_events",
        "extraction_job_events",
        "surfacing_records",
    ):
        assert f"{table}_no_update" in triggers
        assert f"{table}_no_delete" in triggers
