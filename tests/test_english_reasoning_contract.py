from __future__ import annotations

import pytest

from runtime.field import (
    DeleteText,
    FieldDelta,
    InsertText,
    LogicalRegion,
    RegionState,
    ReplaceText,
    SharedFieldSnapshot,
    apply_delta,
)
from runtime.heart import AuthorityGrant
from runtime.heart.english_reasoning import (
    EnglishProposal,
    EnglishReasoningContractError,
    TechnicalFinalVerdict,
    parse_final_verdict,
    render_final_verdict,
)


def _base() -> SharedFieldSnapshot:
    return SharedFieldSnapshot(
        tick_id=17,
        regions=(
            RegionState.from_text(LogicalRegion.RESPONSE_DRAFT, "Old answer."),
            RegionState.from_text(LogicalRegion.SCRATCH, "old scratch"),
            RegionState.from_text(LogicalRegion.TASK_STATE, "abcd"),
        ),
    )


def _apply(base: SharedFieldSnapshot, delta: FieldDelta) -> SharedFieldSnapshot:
    return apply_delta(
        base,
        delta,
        permitted_regions=AuthorityGrant.consolidator().governed_regions,
    )


def test_english_proposal_is_nonempty_exact_unicode_and_width_independent() -> None:
    proposal = EnglishProposal(
        base_field_id="a" * 64,
        base_tick_id=9,
        author_core_id="core-a",
        pass_id="first",
        rail_d_model=64,
        text="I think the response is correct, but I would preserve the caveat about 東京.",
        evidence=("z", "a", "z"),
    )

    assert proposal.evidence == ("a", "z")
    assert proposal.frame_for(64).text == proposal.text
    assert proposal.frame_for(128).text == proposal.text
    assert proposal.frame_for(256).text == proposal.text
    assert len(proposal.frame_for(64).rows) > len(proposal.frame_for(256).rows)


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_english_proposal_rejects_empty_contribution(text: str) -> None:
    with pytest.raises(EnglishReasoningContractError, match="nonempty"):
        EnglishProposal(
            base_field_id="a" * 64,
            base_tick_id=1,
            author_core_id="core-a",
            pass_id="first",
            rail_d_model=64,
            text=text,
        )


def test_english_proposal_rejects_the_old_consolidated_or_permission_surface() -> None:
    with pytest.raises(EnglishReasoningContractError, match=r"first.*refined"):
        EnglishProposal(
            base_field_id="a" * 64,
            base_tick_id=1,
            author_core_id="core-a",
            pass_id="consolidated",
            rail_d_model=64,
            text="I have a contribution.",
        )


def test_final_verdict_is_the_simple_tagged_surface_requested_by_doctrine() -> None:
    base = _base()
    text = "#responseDraft# Hello Jeff.\n#scratch# Something to write down."
    verdict = TechnicalFinalVerdict.parse(text, base=base, author_core_id="core-64-a")
    successor = _apply(base, verdict.delta)

    assert verdict.text == text
    assert successor.region(LogicalRegion.RESPONSE_DRAFT).text == "Hello Jeff."
    assert successor.region(LogicalRegion.SCRATCH).text == "Something to write down."
    assert verdict.frame_for(64).text == text
    assert verdict.frame_for(256).text == text
    assert "base field" not in text.lower()
    assert "mutation" not in text.lower()
    assert "{" not in text


def test_final_verdict_preserves_multiline_unicode_and_literal_tag_shaped_text() -> None:
    base = _base()
    desired = "Observe 東京.\n#responseDraft# this is literal text.\n\\leading slash"
    text = render_final_verdict(
        {
            LogicalRegion.SCRATCH: desired,
            LogicalRegion.RESPONSE_DRAFT: "Hello Jeff.",
        }
    )
    assert "\\#responseDraft# this is literal text." in text
    assert "\\\\leading slash" in text

    parsed = parse_final_verdict(text, base=base, author_core_id="core")
    successor = _apply(base, parsed)
    assert successor.region(LogicalRegion.SCRATCH).text == desired
    assert successor.region(LogicalRegion.RESPONSE_DRAFT).text == "Hello Jeff."


def test_journal_is_the_public_tag_for_the_historical_diary_region() -> None:
    base = _base()
    text = render_final_verdict({"journal": "Today I learned something useful."})
    assert text == "#journal# Today I learned something useful."
    parsed = parse_final_verdict(text, base=base, author_core_id="core")
    successor = _apply(base, parsed)
    assert successor.region(LogicalRegion.DIARY).text == "Today I learned something useful."


def test_unmentioned_regions_are_unchanged_and_empty_section_can_clear_a_region() -> None:
    base = _base()
    parsed = parse_final_verdict(
        "#scratch#",
        base=base,
        author_core_id="core",
    )
    successor = _apply(base, parsed)
    assert successor.region(LogicalRegion.SCRATCH).text == ""
    assert successor.region(LogicalRegion.RESPONSE_DRAFT).text == "Old answer."
    assert successor.region(LogicalRegion.TASK_STATE).text == "abcd"


def test_parser_rejects_unknown_duplicate_unauthorized_and_effective_noop_tags() -> None:
    base = _base()
    with pytest.raises(EnglishReasoningContractError, match="unknown final-verdict tag"):
        parse_final_verdict("#respnseDraft# typo", base=base, author_core_id="core")
    with pytest.raises(EnglishReasoningContractError, match="duplicate"):
        parse_final_verdict(
            "#scratch# one\n#scratch# two",
            base=base,
            author_core_id="core",
        )
    with pytest.raises(Exception, match="identity"):
        parse_final_verdict("#identity# rewrite me", base=base, author_core_id="core")
    with pytest.raises(EnglishReasoningContractError, match="change at least one"):
        parse_final_verdict("#responseDraft# Old answer.", base=base, author_core_id="core")


def test_sparse_historical_delta_can_be_rendered_as_equivalent_simple_region_text() -> None:
    base = _base()
    original = FieldDelta(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id="core-64-a",
        pass_id="consolidated",
        operations=(
            InsertText(region=LogicalRegion.SCRATCH, offset=3, text=" INSERT "),
            ReplaceText(
                region=LogicalRegion.RESPONSE_DRAFT,
                start=0,
                end=3,
                text="New",
            ),
            DeleteText(region=LogicalRegion.TASK_STATE, start=1, end=3),
        ),
        evidence=("source-b", "source-a"),
    )
    expected = _apply(base, original)
    verdict = TechnicalFinalVerdict.from_delta(base, original)
    reconstructed = _apply(base, verdict.delta)

    assert verdict.text.startswith("#taskState#") or verdict.text.startswith("#scratch#")
    assert "Mutation" not in verdict.text
    for region in (LogicalRegion.SCRATCH, LogicalRegion.RESPONSE_DRAFT, LogicalRegion.TASK_STATE):
        assert reconstructed.region(region).text == expected.region(region).text


def test_unicode_surrogate_is_rejected_before_transport() -> None:
    with pytest.raises(EnglishReasoningContractError, match="valid exact Unicode"):
        EnglishProposal(
            base_field_id="a" * 64,
            base_tick_id=1,
            author_core_id="core",
            pass_id="refined",
            rail_d_model=64,
            text="bad surrogate: \ud800",
        )
