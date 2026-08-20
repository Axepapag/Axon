from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from runtime.field import (
    CANONICAL_REGION_ORDER,
    CONTEXT_END,
    DeleteText,
    DeltaValidationError,
    FieldDelta,
    FieldSpan,
    InsertText,
    LOGICAL_REGION_IDS,
    LogicalRegion,
    OverlappingDeltaError,
    PROPOSAL_START,
    PhysicalRole,
    RegionState,
    ReplaceText,
    SealedRegionWriteError,
    SharedFieldSnapshot,
    SlotKind,
    StaleDeltaError,
    USER_END,
    USER_START,
    apply_delta,
    compile_field_view,
)
from substrate import default_alphabet, get_letter_bank


def _snapshot(**texts: str) -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(texts, tick_id=7)


def test_snapshot_has_all_ten_regions_and_rejects_dormant() -> None:
    snapshot = _snapshot(user_input="hello")
    assert tuple(region.name for region in snapshot.regions) == CANONICAL_REGION_ORDER
    assert len(snapshot.regions) == 10
    assert snapshot.region("user_input").text == "hello"
    assert "dormant" not in {region.value for region in CANONICAL_REGION_ORDER}
    with pytest.raises(ValueError, match="dormant state is surfaced"):
        RegionState.from_text("dormant", "hidden")


def test_snapshot_is_immutable_unbounded_and_hash_deterministic() -> None:
    text = "a" * 10_000
    left = _snapshot(conversation_history=text)
    right = _snapshot(conversation_history=text)
    assert left.region(LogicalRegion.CONVERSATION_HISTORY).text == text
    assert left.field_id == right.field_id == left.canonical_hash
    with pytest.raises(FrozenInstanceError):
        left.tick_id = 8  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        left.regions[0].spans[0].text = "changed"  # type: ignore[misc]


def test_canonical_unicode_is_lossless_and_compiler_audits_unsupported_ranges() -> None:
    text = "a\u00e9\U0001f642b"
    span = FieldSpan(
        span_id="unicode-source",
        text=text,
        source=r"D:\00\axon_memory.db",
        provenance="messages:42",
    )
    snapshot = SharedFieldSnapshot(
        tick_id=0,
        regions=(
            RegionState(
                name=LogicalRegion.CONVERSATION_HISTORY,
                spans=(span,),
            ),
        ),
    )
    assert snapshot.region("conversation_history").text == text

    view = compile_field_view(snapshot)
    rendered_refs = [
        ref
        for ref in view.slot_refs
        if ref.kind is SlotKind.SPAN and ref.span_id == "unicode-source"
    ]
    assert [ref.rendered_char for ref in rendered_refs] == ["a", "b"]
    assert [ref.span_char_index for ref in rendered_refs] == [0, 3]
    assert [ref.region_char_index for ref in rendered_refs] == [0, 3]

    unsupported = [
        omission
        for omission in view.omissions
        if omission.reason == "unsupported_substrate"
    ]
    assert len(unsupported) == 1
    omission = unsupported[0]
    assert omission.logical_region is LogicalRegion.CONVERSATION_HISTORY
    assert omission.span_id == "unicode-source"
    assert (omission.span_char_start, omission.span_char_end) == (1, 3)
    assert (omission.region_char_start, omission.region_char_end) == (1, 3)
    assert omission.omitted_text_sha256 == hashlib.sha256(
        "\u00e9\U0001f642".encode("utf-8")
    ).hexdigest()
    assert omission.source == r"D:\00\axon_memory.db"
    assert omission.provenance == "messages:42"


def test_compile_preserves_checkpoint_geometry_and_active_blank_proposal() -> None:
    view = compile_field_view(_snapshot(user_input="question"))
    assert view.field16.shape == (384, 16)
    assert np.all(view.role_ids[:256] == int(PhysicalRole.CONTEXT))
    assert np.all(view.role_ids[256:320] == int(PhysicalRole.USER))
    assert np.all(view.role_ids[320:] == int(PhysicalRole.PROPOSAL))
    assert not np.any(view.write_mask[:PROPOSAL_START])
    assert np.all(view.write_mask[PROPOSAL_START:])
    assert np.all(view.attention_mask[PROPOSAL_START:])
    blank_refs = [
        ref
        for ref in view.slot_refs[PROPOSAL_START:]
        if ref.kind is SlotKind.BLANK
    ]
    assert blank_refs
    assert all(
        ref.logical_region is LogicalRegion.RESPONSE_DRAFT for ref in blank_refs
    )
    assert all(ref.provenance == "active_blank_proposal" for ref in blank_refs)


def test_compile_can_alternate_proposal_between_response_and_scratch() -> None:
    snapshot = _snapshot(
        scratch="think" * 30,
        response_draft="answer",
    )
    response_view = compile_field_view(snapshot)
    scratch_view = compile_field_view(snapshot, proposal_region="scratch")

    assert response_view.proposal_region is LogicalRegion.RESPONSE_DRAFT
    assert scratch_view.proposal_region is LogicalRegion.SCRATCH
    assert "[response_draft]" in response_view.decode(PROPOSAL_START, 384)
    assert "[scratch]" in scratch_view.decode(PROPOSAL_START, 384)
    scratch_context = scratch_view.decode(0, CONTEXT_END)
    assert "[response_draft]\nanswer" in scratch_context
    assert "[scratch]" not in scratch_context
    assert "think" not in scratch_context
    assert np.all(
        scratch_view.logical_region_ids[PROPOSAL_START:]
        == LOGICAL_REGION_IDS[LogicalRegion.SCRATCH]
    )
    assert all(
        ref.logical_region is LogicalRegion.SCRATCH
        for ref in scratch_view.slot_refs[PROPOSAL_START:]
    )
    assert any(
        omission.logical_region is LogicalRegion.SCRATCH
        and omission.reason == "proposal_capacity"
        for omission in scratch_view.omissions
    )
    assert response_view.view_hash != scratch_view.view_hash

    with pytest.raises(ValueError, match="sealed"):
        compile_field_view(
            snapshot,
            proposal_region=LogicalRegion.CONVERSATION_HISTORY,
        )
    sealed_scratch = SharedFieldSnapshot(
        tick_id=0,
        regions=(
            RegionState.from_text(
                LogicalRegion.SCRATCH,
                "private",
                write_policy="sealed",
            ),
        ),
    )
    with pytest.raises(ValueError, match="sealed"):
        compile_field_view(sealed_scratch, proposal_region="scratch")


def test_compile_renders_every_logical_region_tag() -> None:
    view = compile_field_view(SharedFieldSnapshot.empty())
    context = view.decode(0, CONTEXT_END)
    user = view.decode(USER_START, USER_END)
    proposal = view.decode(PROPOSAL_START, 384)
    for region in (
        LogicalRegion.CONVERSATION_HISTORY,
        LogicalRegion.STRUCTURED_KNOWLEDGE,
        LogicalRegion.SITUATION_AWARENESS,
        LogicalRegion.TOOL_RESULTS,
        LogicalRegion.ADVISOR_INPUT,
        LogicalRegion.TASK_STATE,
        LogicalRegion.SCRATCH,
        LogicalRegion.DIARY,
    ):
        assert f"[{region.value}]" in context
    assert "[user_input]" in user
    assert "[response_draft]" in proposal


def test_95_character_substrate_roundtrip_with_exact_slot_refs() -> None:
    alphabet = "".join(default_alphabet())
    span = FieldSpan(
        span_id="alphabet",
        text=alphabet,
        source="substrate-test",
        provenance="frozen-95-gate",
    )
    snapshot = SharedFieldSnapshot(
        tick_id=0,
        regions=(
            RegionState(
                name=LogicalRegion.CONVERSATION_HISTORY,
                spans=(span,),
            ),
        ),
    )
    view = compile_field_view(snapshot)
    refs = [
        ref
        for ref in view.slot_refs
        if ref.kind is SlotKind.SPAN and ref.span_id == "alphabet"
    ]
    assert [ref.span_char_index for ref in refs] == list(range(95))
    assert [ref.region_char_index for ref in refs] == list(range(95))
    assert all(ref.source == "substrate-test" for ref in refs)
    assert all(ref.provenance == "frozen-95-gate" for ref in refs)
    rows = view.field16[[ref.slot_index for ref in refs]]
    assert get_letter_bank().decode_sequence(rows, blanks_as="") == alphabet


def test_view_omissions_are_explicit_without_truncating_canonical_text() -> None:
    long_text = "history" * 500
    snapshot = _snapshot(conversation_history=long_text, scratch="working")
    view = compile_field_view(snapshot)
    assert snapshot.region("conversation_history").text == long_text
    omissions = [
        omission
        for omission in view.omissions
        if omission.logical_region is LogicalRegion.CONVERSATION_HISTORY
    ]
    assert omissions
    assert omissions[0].reason == "context_capacity"
    assert omissions[0].region_char_end == len(long_text)
    assert len(omissions[0].omitted_text_sha256) == 64


def test_compiled_view_and_arrays_are_deterministic_and_read_only() -> None:
    snapshot = _snapshot(
        conversation_history="history",
        user_input="question",
        response_draft="draft",
    )
    first = compile_field_view(snapshot)
    second = compile_field_view(snapshot)
    assert first.view_hash == second.view_hash
    assert np.array_equal(first.field16, second.field16)
    with pytest.raises(ValueError):
        first.field16[0, 0] = 0.0
    with pytest.raises(ValueError):
        first.role_ids.setflags(write=True)


def test_delta_applies_only_scratch_and_response_and_replays_identically() -> None:
    base = _snapshot(scratch="plan", response_draft="hello world")
    delta = FieldDelta(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id="coreB128",
        pass_id=2,
        operations=(
            InsertText(
                region=LogicalRegion.SCRATCH,
                offset=4,
                text=" next",
                provenance="pass2 scratch extension",
            ),
            ReplaceText(
                region=LogicalRegion.RESPONSE_DRAFT,
                start=6,
                end=11,
                text="Jeff",
                provenance="supported response refinement",
            ),
        ),
        evidence=("container-2", "container-1"),
    )
    first = apply_delta(base, delta)
    replay = apply_delta(base, delta)
    assert first.region("scratch").text == "plan next"
    assert first.region("response_draft").text == "hello Jeff"
    assert first.tick_id == base.tick_id + 1
    assert first.parent_field_id == base.field_id
    assert first.field_id == replay.field_id
    assert first.to_canonical_dict() == replay.to_canonical_dict()
    response_delta_span = next(
        span
        for span in first.region("response_draft").spans
        if span.source == "coreB128"
    )
    assert response_delta_span.provenance == "supported response refinement"
    assert delta.delta_id == delta.canonical_hash


def test_delta_delete_and_adjacent_insert_have_deterministic_boundary_order() -> None:
    base = _snapshot(scratch="abcdef")
    delta = FieldDelta(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id="coreA64",
        pass_id="refine",
        operations=(
            DeleteText(region="scratch", start=1, end=3),
            InsertText(region="scratch", offset=3, text="XY"),
        ),
    )
    assert apply_delta(base, delta).region("scratch").text == "aXYdef"


def test_delta_preserves_unicode_in_canonical_writable_regions() -> None:
    base = _snapshot(scratch="plan")
    delta = FieldDelta(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id="coreA64",
        pass_id="unicode",
        operations=(
            InsertText(region="scratch", offset=4, text=" caf\u00e9 \U0001f642"),
        ),
    )
    updated = apply_delta(base, delta)
    assert updated.region("scratch").text == "plan caf\u00e9 \U0001f642"
    view = compile_field_view(updated, proposal_region="scratch")
    assert any(
        omission.reason == "unsupported_substrate"
        and omission.logical_region is LogicalRegion.SCRATCH
        for omission in view.omissions
    )


def test_delta_rejects_stale_sealed_bounds_and_overlap() -> None:
    base = _snapshot(
        conversation_history="sealed",
        scratch="abcdef",
        response_draft="draft",
    )
    stale = FieldDelta(
        base_field_id="0" * 64,
        base_tick_id=base.tick_id,
        author_core_id="core",
        pass_id=1,
        operations=(InsertText(region="scratch", offset=0, text="x"),),
    )
    with pytest.raises(StaleDeltaError):
        apply_delta(base, stale)

    sealed = FieldDelta(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id="core",
        pass_id=1,
        operations=(
            ReplaceText(
                region="conversation_history",
                start=0,
                end=1,
                text="S",
            ),
        ),
    )
    with pytest.raises(SealedRegionWriteError):
        apply_delta(base, sealed)

    sealed_scratch_base = SharedFieldSnapshot(
        tick_id=base.tick_id,
        regions=(
            RegionState.from_text(
                LogicalRegion.SCRATCH,
                "private",
                write_policy="sealed",
            ),
        ),
    )
    sealed_scratch_delta = FieldDelta(
        base_field_id=sealed_scratch_base.field_id,
        base_tick_id=sealed_scratch_base.tick_id,
        author_core_id="core",
        pass_id=1,
        operations=(InsertText(region="scratch", offset=0, text="x"),),
    )
    with pytest.raises(SealedRegionWriteError):
        apply_delta(sealed_scratch_base, sealed_scratch_delta)

    out_of_bounds = FieldDelta(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id="core",
        pass_id=1,
        operations=(InsertText(region="scratch", offset=99, text="x"),),
    )
    with pytest.raises(DeltaValidationError, match="exceed"):
        apply_delta(base, out_of_bounds)

    overlap = FieldDelta(
        base_field_id=base.field_id,
        base_tick_id=base.tick_id,
        author_core_id="core",
        pass_id=1,
        operations=(
            ReplaceText(region="scratch", start=1, end=4, text="x"),
            DeleteText(region="scratch", start=3, end=5),
        ),
    )
    with pytest.raises(OverlappingDeltaError):
        apply_delta(base, overlap)


def test_region_ids_are_exact_for_canonical_characters() -> None:
    snapshot = _snapshot(user_input="abc", response_draft="xyz")
    view = compile_field_view(snapshot)
    user_refs = [
        ref
        for ref in view.slot_refs
        if ref.kind is SlotKind.SPAN
        and ref.logical_region is LogicalRegion.USER_INPUT
    ]
    response_refs = [
        ref
        for ref in view.slot_refs
        if ref.kind is SlotKind.SPAN
        and ref.logical_region is LogicalRegion.RESPONSE_DRAFT
    ]
    assert user_refs and response_refs
    assert all(
        view.logical_region_ids[ref.slot_index]
        == LOGICAL_REGION_IDS[LogicalRegion.USER_INPUT]
        for ref in user_refs
    )
    assert all(
        view.logical_region_ids[ref.slot_index]
        == LOGICAL_REGION_IDS[LogicalRegion.RESPONSE_DRAFT]
        for ref in response_refs
    )
