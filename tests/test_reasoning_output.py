from __future__ import annotations

import pytest

from runtime.field import (
    D64FieldCompiler,
    LogicalRegion,
    RegionMaskPolicy,
    SealedRegionWriteError,
    SharedFieldSnapshot,
)
from runtime.heart import (
    EMPTY_CATEGORY_ID,
    EOS_CATEGORY_ID,
    REASONING_CATEGORY_COUNT,
    AuthorityGrant,
    CategoricalTextFrame,
    ReasoningDecision,
    ReasoningEmission,
    ReasoningOperationEmission,
    ReasoningOperationKind,
    ReasoningOutputError,
)
from substrate import TRANSPORT_VOCAB_SIZE, byte_token_id


def _base() -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.USER_INPUT: "hello",
            LogicalRegion.SCRATCH: "seed",
        },
        tick_id=3,
        source="test",
        provenance="test",
    )


def test_reasoning_categories_preserve_arbitrary_unicode_without_a_context_ceiling() -> None:
    text = ("Python: print('λ🧠\\n')\nC++: std::cout << \"✓\";\n" * 500) + "\x00終"
    frame = CategoricalTextFrame.from_text(text, d_model=64)
    assert TRANSPORT_VOCAB_SIZE == 351
    assert EMPTY_CATEGORY_ID == 351
    assert EOS_CATEGORY_ID == 352
    assert REASONING_CATEGORY_COUNT == 353
    assert frame.text == text
    assert len(frame.rows) > 192
    assert all(len(row) == 4 for row in frame.rows)


def test_reasoning_text_frame_rejects_malformed_eos_and_unicode_transport() -> None:
    with pytest.raises(ReasoningOutputError, match="exactly one EOS"):
        CategoricalTextFrame(d_model=64, rows=((EMPTY_CATEGORY_ID,) * 4,))
    with pytest.raises(ReasoningOutputError, match="only EMPTY"):
        CategoricalTextFrame(
            d_model=64,
            rows=((EOS_CATEGORY_ID, 0, EMPTY_CATEGORY_ID, EMPTY_CATEGORY_ID),),
        )
    with pytest.raises(ReasoningOutputError, match="EMPTY categories may appear only after EOS"):
        CategoricalTextFrame(
            d_model=64,
            rows=((0, EMPTY_CATEGORY_ID, EOS_CATEGORY_ID, EMPTY_CATEGORY_ID),),
        )
    with pytest.raises(ReasoningOutputError, match="canonical Unicode transport"):
        CategoricalTextFrame(
            d_model=64,
            rows=((byte_token_id(0xF0), EOS_CATEGORY_ID, EMPTY_CATEGORY_ID, EMPTY_CATEGORY_ID),),
        )


def test_reasoning_emission_decodes_exact_addresses_and_enforces_authority() -> None:
    base = _base()
    payload = CategoricalTextFrame.from_text(" + exact", d_model=64)
    emission = ReasoningEmission(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id="core-a",
        pass_id="first",
        rail_d_model=64,
        decision=ReasoningDecision.DELTA,
        operations=(
            ReasoningOperationEmission(
                kind=ReasoningOperationKind.INSERT,
                region=LogicalRegion.SCRATCH,
                start=4,
                end=4,
                payload=payload,
            ),
        ),
    )
    delta = emission.decode_delta(base, AuthorityGrant.core())
    assert delta is not None
    assert delta.operations[0].offset == 4
    assert delta.operations[0].text == " + exact"

    sealed = ReasoningEmission(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id="core-a",
        pass_id="first",
        rail_d_model=64,
        decision=ReasoningDecision.DELTA,
        operations=(
            ReasoningOperationEmission(
                kind=ReasoningOperationKind.REPLACE,
                region=LogicalRegion.USER_INPUT,
                start=0,
                end=5,
                payload=CategoricalTextFrame.from_text("poison", d_model=64),
            ),
        ),
    )
    with pytest.raises(SealedRegionWriteError):
        sealed.decode_delta(base, AuthorityGrant.core())


def test_no_op_and_abstain_are_explicit_non_delta_decisions() -> None:
    base = _base()
    no_op = ReasoningEmission.no_op(
        base=base,
        author_core_id="core-a",
        pass_id="first",
        rail_d_model=64,
        detail="nothing grounded to change",
    )
    abstain = ReasoningEmission.abstain(
        base=base,
        author_core_id="core-b",
        pass_id="refined",
        rail_d_model=64,
        detail="insufficient evidence",
    )
    assert no_op.decode_delta(base, AuthorityGrant.core()) is None
    assert abstain.decode_delta(base, AuthorityGrant.core()) is None
    assert no_op.detail is not None and no_op.detail.text == "nothing grounded to change"
    assert abstain.detail is not None and abstain.detail.text == "insufficient evidence"


def test_reasoning_delta_cannot_address_masked_dormant_cells() -> None:
    base = _base()
    surface = D64FieldCompiler().compile(
        base,
        region_masks={LogicalRegion.SCRATCH: RegionMaskPolicy("tail_percent", 50)},
    )

    def replace_scratch(start: int, end: int) -> ReasoningEmission:
        return ReasoningEmission(
            base_field_id=base.field_id,
            base_tick_id=base.tick_id,
            author_core_id="core-a",
            pass_id="first",
            rail_d_model=64,
            decision=ReasoningDecision.DELTA,
            operations=(
                ReasoningOperationEmission(
                    kind=ReasoningOperationKind.REPLACE,
                    region=LogicalRegion.SCRATCH,
                    start=start,
                    end=end,
                    payload=CategoricalTextFrame.from_text("ok", d_model=64),
                ),
            ),
        )

    with pytest.raises(ReasoningOutputError, match="masked canonical cells"):
        replace_scratch(0, 2).decode_delta(
            base,
            AuthorityGrant.core(),
            attended_surface=surface,
        )
    visible = replace_scratch(2, 4).decode_delta(
        base,
        AuthorityGrant.core(),
        attended_surface=surface,
    )
    assert visible is not None
