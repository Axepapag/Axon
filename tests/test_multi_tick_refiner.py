from __future__ import annotations

import pytest

from runtime.field import LogicalRegion, SharedFieldSnapshot
from runtime.multi_tick_refiner import (
    MultiTickContractError,
    RegionProposal,
    character_edit_distance,
    run_multi_tick_refinement,
)


def _snapshot(**texts: str) -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(texts, source="fixture")


def test_free_running_second_tick_reads_its_own_committed_draft() -> None:
    initial = _snapshot(
        conversation_history="The user asked for a concise answer.",
        user_input="Answer now",
        response_draft="",
    )
    seen: list[tuple[int, str, str]] = []

    def proposer(*, snapshot, view, tick_index, target_region):
        prior = snapshot.region(target_region).text
        visible = view.decode(320, 384)
        seen.append((tick_index, prior, visible))
        if tick_index == 0:
            assert prior == ""
            assert "gold answer" not in visible
            return RegionProposal(target_region, "rough draft")
        if tick_index == 1:
            assert prior == "rough draft"
            assert "rough draft" in visible
            return RegionProposal(target_region, "final answer")
        assert prior == "final answer"
        return RegionProposal(target_region, "final answer")

    outcomes = []
    result = run_multi_tick_refinement(
        initial_snapshot=initial,
        proposer=proposer,
        target_schedule=[
            LogicalRegion.RESPONSE_DRAFT,
            LogicalRegion.RESPONSE_DRAFT,
            LogicalRegion.RESPONSE_DRAFT,
        ],
        author_core_id="tiny64",
        max_ticks=3,
        min_ticks=2,
        stable_ticks=1,
        on_outcome=lambda snapshot, record: outcomes.append(
            (snapshot.field_id, record.tick_index)
        ),
    )

    assert result.free_running is True
    assert result.stopped_stable is True
    assert result.final_snapshot.region("response_draft").text == "final answer"
    assert [record.proposed_text for record in result.records] == [
        "rough draft",
        "final answer",
        "final answer",
    ]
    assert len(result.deltas) == 2
    assert result.replay().field_id == result.final_snapshot.field_id
    assert len(outcomes) == 3
    assert len(result.run_hash) == 64


def test_ticks_can_plan_in_scratch_then_answer_from_committed_plan() -> None:
    initial = _snapshot(user_input="What comes next?", scratch="", response_draft="")
    targets = [LogicalRegion.SCRATCH, LogicalRegion.RESPONSE_DRAFT]

    def proposer(*, snapshot, view, tick_index, target_region):
        if tick_index == 0:
            assert target_region is LogicalRegion.SCRATCH
            assert view.proposal_region is LogicalRegion.SCRATCH
            return RegionProposal("scratch", "step one then step two")
        assert target_region is LogicalRegion.RESPONSE_DRAFT
        assert snapshot.region("scratch").text == "step one then step two"
        assert "step one then step two" in view.decode(0, 256)
        return RegionProposal("response_draft", "done")

    result = run_multi_tick_refinement(
        initial_snapshot=initial,
        proposer=proposer,
        target_schedule=targets,
        author_core_id="tiny64",
        max_ticks=2,
    )

    assert result.final_snapshot.region("scratch").text == "step one then step two"
    assert result.final_snapshot.region("response_draft").text == "done"
    assert [record.target_region for record in result.records] == targets


def test_teacher_forcing_is_explicit_and_keeps_the_model_proposal() -> None:
    initial = _snapshot(response_draft="")

    def proposer(**kwargs):
        return RegionProposal(kwargs["target_region"], "model proposal")

    result = run_multi_tick_refinement(
        initial_snapshot=initial,
        proposer=proposer,
        target_schedule=[LogicalRegion.RESPONSE_DRAFT],
        author_core_id="tiny64",
        max_ticks=1,
        teacher_targets={0: "teacher commit"},
        reference_targets={0: "teacher commit"},
    )

    assert result.free_running is False
    assert result.records[0].teacher_forced is True
    assert result.records[0].proposed_text == "model proposal"
    assert result.records[0].committed_text == "teacher commit"
    assert result.records[0].reference_edit_distance == 0
    assert result.final_snapshot.region("response_draft").text == "teacher commit"


def test_schedule_and_proposal_fail_closed_on_sealed_or_wrong_regions() -> None:
    initial = _snapshot(response_draft="")

    with pytest.raises(MultiTickContractError, match="sealed"):
        run_multi_tick_refinement(
            initial_snapshot=initial,
            proposer=lambda **kwargs: RegionProposal("response_draft", "x"),
            target_schedule=[LogicalRegion.USER_INPUT],
            author_core_id="tiny64",
            max_ticks=1,
        )

    with pytest.raises(MultiTickContractError, match="scheduled"):
        run_multi_tick_refinement(
            initial_snapshot=initial,
            proposer=lambda **kwargs: RegionProposal("scratch", "x"),
            target_schedule=[LogicalRegion.RESPONSE_DRAFT],
            author_core_id="tiny64",
            max_ticks=1,
        )


def test_proposer_failure_is_not_mistaken_for_a_completed_rollout() -> None:
    initial = _snapshot(response_draft="")

    def broken(**kwargs):
        raise RuntimeError("boom")

    with pytest.raises(MultiTickContractError, match="proposer failed"):
        run_multi_tick_refinement(
            initial_snapshot=initial,
            proposer=broken,
            target_schedule=[LogicalRegion.RESPONSE_DRAFT],
            author_core_id="tiny64",
            max_ticks=1,
        )


def test_character_edit_distance_examples() -> None:
    assert character_edit_distance("", "") == 0
    assert character_edit_distance("draft", "draft") == 0
    assert character_edit_distance("kitten", "sitting") == 3
    assert character_edit_distance("a", "") == 1
