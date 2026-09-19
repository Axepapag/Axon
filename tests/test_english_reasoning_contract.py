from __future__ import annotations

import json

import pytest

from runtime.field import DeleteText, FieldDelta, InsertText, LogicalRegion, ReplaceText
from runtime.heart.english_reasoning import (
    EnglishProposal,
    EnglishReasoningContractError,
    TechnicalFinalVerdict,
    parse_final_verdict,
    render_final_verdict,
)


def _delta() -> FieldDelta:
    return FieldDelta(
        base_field_id="f" * 64,
        base_tick_id=17,
        author_core_id="core-64-a",
        pass_id="consolidated",
        operations=(
            InsertText(
                region=LogicalRegion.SCRATCH,
                offset=3,
                text="Observe 東京 carefully.\nSecond line.",
                provenance="proposal:1",
                container_refs=("c2", "c1"),
                edge_refs=("e1",),
            ),
            ReplaceText(
                region=LogicalRegion.RESPONSE_DRAFT,
                start=0,
                end=5,
                text='Paris — and "France".',
                provenance="consolidator",
            ),
            DeleteText(
                region=LogicalRegion.TASK_STATE,
                start=2,
                end=4,
                provenance="cleanup",
            ),
        ),
        evidence=("source-b", "source-a", "source-a"),
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


def test_final_verdict_roundtrips_all_field_operations_exactly() -> None:
    original = _delta()
    text = render_final_verdict(original)
    parsed = parse_final_verdict(text)

    assert parsed.to_canonical_dict() == original.to_canonical_dict()
    verdict = TechnicalFinalVerdict.parse(text)
    assert verdict.delta.delta_id == original.delta_id
    assert verdict.frame_for(64).text == text
    assert verdict.frame_for(256).text == text


def test_final_verdict_is_technical_english_with_canonical_json_payloads() -> None:
    text = render_final_verdict(_delta())
    lines = text.split("\n")

    assert lines[0] == "AXON FINAL VERDICT V1"
    assert lines[1].startswith("The base field is ")
    assert lines[5] == "There are 3 canonical mutations."
    assert lines[6].startswith("Mutation 1 is {")
    assert lines[-1] == "END AXON FINAL VERDICT V1"
    payload = lines[6][len("Mutation 1 is ") : -1]
    decoded = json.loads(payload)
    assert decoded["op"] == "insert"
    assert decoded["text"] == "Observe 東京 carefully.\nSecond line."


def test_final_verdict_parser_rejects_noncanonical_or_ambiguous_text() -> None:
    text = render_final_verdict(_delta())
    mutation_line = text.split("\n")[6]
    payload = json.loads(mutation_line[len("Mutation 1 is ") : -1])
    pretty = json.dumps(payload, ensure_ascii=False, sort_keys=True)  # spaces => noncanonical
    tampered = text.replace(mutation_line, f"Mutation 1 is {pretty}.")

    with pytest.raises(EnglishReasoningContractError, match="canonical JSON"):
        parse_final_verdict(tampered)


def test_final_verdict_parser_rejects_count_or_numbering_drift() -> None:
    text = render_final_verdict(_delta())
    with pytest.raises(EnglishReasoningContractError, match="count"):
        parse_final_verdict(text.replace("There are 3 canonical mutations.", "There are 2 canonical mutations."))
    with pytest.raises(EnglishReasoningContractError, match="consecutively"):
        parse_final_verdict(text.replace("Mutation 2 is ", "Mutation 7 is ", 1))


def test_final_verdict_requires_real_canonical_mutation() -> None:
    text = "\n".join(
        [
            "AXON FINAL VERDICT V1",
            'The base field is "field".',
            "The base tick is 1.",
            'The author core is "core".',
            "The evidence references are [].",
            "There are 0 canonical mutations.",
            "END AXON FINAL VERDICT V1",
        ]
    )
    with pytest.raises(EnglishReasoningContractError, match="at least one"):
        parse_final_verdict(text)


def test_verdict_rejects_a_non_consolidated_delta() -> None:
    delta = FieldDelta(
        base_field_id="f" * 64,
        base_tick_id=1,
        author_core_id="core",
        pass_id="first",
        operations=(InsertText(region=LogicalRegion.SCRATCH, offset=0, text="x"),),
    )
    with pytest.raises(EnglishReasoningContractError, match="consolidated"):
        render_final_verdict(delta)


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
