from __future__ import annotations

import numpy as np
import pytest

from runtime.field import (
    CyclingFieldViewProvider,
    FieldViewCursor,
    LogicalRegion,
    SharedFieldSnapshot,
    SlotKind,
    audit_read_cycle_coverage,
    collect_read_cycle,
    compile_field_view,
    compile_next_read_page,
)
from runtime.multi_tick_refiner import RegionProposal, run_multi_tick_refinement


def _long_snapshot() -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(
        {
            "conversation_history": "history-" * 90,
            "structured_knowledge": "knowledge-" * 70,
            "situation_awareness": "situation-" * 40,
            "tool_results": "tool-" * 35,
            "advisor_input": "advisor-" * 30,
            "task_state": "task-" * 45,
            "diary": "diary-" * 32,
            "scratch": "scratch-" * 30,
            "user_input": "question-" * 30,
            "response_draft": "draft-" * 20,
        }
    )


def test_read_cycle_covers_every_supported_character_once() -> None:
    snapshot = _long_snapshot()
    pages = collect_read_cycle(snapshot)
    report = audit_read_cycle_coverage(snapshot, pages)

    assert len(pages) > 2
    assert report.passed
    assert report.expected_characters == report.observed_characters
    assert report.duplicate_characters == 0
    assert pages[-1].read_cycle_complete is True
    assert pages[-1].next_cursor == FieldViewCursor()
    assert all(page.view.field16.shape == (384, 16) for page in pages)
    assert all(
        np.all(page.view.role_ids[:256] == 0)
        and np.all(page.view.role_ids[256:320] == 1)
        and np.all(page.view.role_ids[320:] == 2)
        for page in pages
    )


def test_cursor_pages_are_distinct_auditable_and_do_not_page_proposal() -> None:
    snapshot = _long_snapshot()
    first = compile_next_read_page(snapshot)
    second = compile_next_read_page(snapshot, cursor=first.next_cursor)

    assert first.read_cycle_complete is False
    assert second.cursor.page_index == 1
    assert first.view.view_hash != second.view.view_hash
    assert any(
        omission.reason == "context_cursor_before"
        for omission in second.view.omissions
    )
    assert any(
        omission.reason == "user_cursor_before"
        for omission in second.view.omissions
    )
    first_proposal = [
        ref.rendered_char
        for ref in first.view.slot_refs[320:]
        if ref.kind is SlotKind.SPAN
    ]
    second_proposal = [
        ref.rendered_char
        for ref in second.view.slot_refs[320:]
        if ref.kind is SlotKind.SPAN
    ]
    assert first_proposal == second_proposal


def test_one_cursor_survives_proposal_switch_and_covers_full_read_cycle() -> None:
    snapshot = _long_snapshot()
    cursor = FieldViewCursor()
    pages = []
    proposal_schedule = (
        LogicalRegion.SCRATCH,
        LogicalRegion.RESPONSE_DRAFT,
    )
    for tick_index in range(256):
        proposal = proposal_schedule[tick_index % len(proposal_schedule)]
        page = compile_next_read_page(
            snapshot,
            proposal_region=proposal,
            cursor=cursor,
        )
        pages.append(page)
        if page.read_cycle_complete:
            break
        cursor = page.next_cursor
    else:  # pragma: no cover - deterministic guard
        raise AssertionError("read cycle did not complete")

    report = audit_read_cycle_coverage(
        snapshot,
        tuple(pages),
        proposal_region=None,
    )
    assert report.passed
    assert report.expected_characters == report.observed_characters
    assert report.duplicate_characters == 0
    assert pages[0].view.proposal_region is LogicalRegion.SCRATCH
    assert pages[1].view.proposal_region is LogicalRegion.RESPONSE_DRAFT
    assert pages[1].cursor == pages[0].next_cursor
    assert pages[1].cursor.page_index == 1
    assert pages[-1].next_cursor == FieldViewCursor()
    assert len({page.view.view_hash for page in pages}) == len(pages)

    for page in pages[:2]:
        proposal_refs = [
            ref
            for ref in page.view.slot_refs[320:]
            if ref.kind is SlotKind.SPAN
        ]
        assert proposal_refs
        expected_last = len(
            snapshot.region(page.view.proposal_region).text
        ) - 1
        assert proposal_refs[-1].region_char_index == expected_last
        audit = page.to_audit_dict()
        assert audit["page_index"] == page.cursor.page_index
        assert audit["read_view_hash"] == page.view.view_hash
        assert audit["cursor_before"] == page.cursor.to_canonical_dict()
        assert audit["cursor_after"] == page.next_cursor.to_canonical_dict()
        assert audit["coverage"]["duplicate_character_count"] == 0


def test_compile_offsets_fail_closed() -> None:
    snapshot = SharedFieldSnapshot.from_texts(
        {"conversation_history": "abc", "user_input": "x"}
    )
    with pytest.raises(ValueError, match="context offset"):
        compile_field_view(
            snapshot,
            context_offsets={LogicalRegion.CONVERSATION_HISTORY: 4},
        )
    with pytest.raises(ValueError, match="user_input"):
        compile_field_view(snapshot, user_offset=2)
    with pytest.raises(ValueError, match="not a context region"):
        compile_field_view(
            snapshot,
            context_offsets={LogicalRegion.RESPONSE_DRAFT: 0},
        )


def test_multi_tick_refiner_can_use_one_read_page_per_tick() -> None:
    snapshot = _long_snapshot()
    provider = CyclingFieldViewProvider()

    def no_op_proposer(
        *,
        snapshot: SharedFieldSnapshot,
        view,
        tick_index: int,
        target_region: LogicalRegion,
    ) -> RegionProposal:
        del view, tick_index
        return RegionProposal(
            target_region=target_region,
            text=snapshot.region(target_region).text,
        )

    result = run_multi_tick_refinement(
        initial_snapshot=snapshot,
        proposer=no_op_proposer,
        target_schedule=[
            LogicalRegion.SCRATCH,
            LogicalRegion.RESPONSE_DRAFT,
            LogicalRegion.SCRATCH,
        ],
        author_core_id="schedule-test",
        max_ticks=3,
        min_ticks=3,
        stable_ticks=4,
        view_provider=provider,
    )

    assert len(result.records) == 3
    assert len(provider.records) == 3
    assert [record.cursor.page_index for record in provider.records] == [0, 1, 2]
    assert [record.view.proposal_region for record in provider.records] == [
        LogicalRegion.SCRATCH,
        LogicalRegion.RESPONSE_DRAFT,
        LogicalRegion.SCRATCH,
    ]
    assert len({record.input_view_hash for record in result.records}) == 3
