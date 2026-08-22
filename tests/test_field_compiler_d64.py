from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from runtime.axon_runtime.d64_adapter import CanonicalD64RuntimeAdapter
from runtime.field import (
    AttendedInterval,
    CANONICAL_REGION_ORDER,
    D64_LANES_PER_ROW,
    D64FieldCompiler,
    FieldSpan,
    LogicalRegion,
    RegionMaskPolicy,
    RegionState,
    SharedFieldSnapshot,
    StaleCompiledFieldError,
    UnsupportedActiveCharacterError,
    apply_compiled_delta,
    replacement_delta,
)
from substrate import char_to_slot
from training.canonical_d64 import canonicalize_r0_record
from training.complete_field_64d import CompleteField64D, ReaderConfig


def test_d64_packs_four_exact_16d_cells_and_roundtrips() -> None:
    snapshot = SharedFieldSnapshot.from_texts({"user_input": "dog!"})
    compiled = D64FieldCompiler().compile(snapshot)
    addresses = compiled.region_addresses("user_input")
    assert len(addresses) == 4
    assert len({address.row_index for address in addresses}) == 1
    assert [address.lane_index for address in addresses] == [0, 1, 2, 3]
    row = compiled.rows[addresses[0].row_index]
    for lane, char in enumerate("dog!"):
        start = lane * 16
        assert np.array_equal(row[start : start + 16], char_to_slot(char))
    assert compiled.region_text("user_input") == "dog!"
    assert compiled.coverage.complete is True


def test_rows_never_cross_region_boundaries_and_padding_is_explicit() -> None:
    snapshot = SharedFieldSnapshot.from_texts(
        {"conversation_history": "abc", "user_input": "wxyz1"}
    )
    compiled = D64FieldCompiler().compile(snapshot)
    for row in range(compiled.row_count):
        regions = {
            address.region
            for lane in range(D64_LANES_PER_ROW)
            if (address := compiled.address(row, lane)) is not None
        }
        assert len(regions) <= 1
    assert compiled.coverage.padding_lanes == 4  # 1 pad + 3 pads
    assert compiled.region_text("conversation_history") == "abc"
    assert compiled.region_text("user_input") == "wxyz1"


def test_compiler_visits_all_regions_and_preserves_masking() -> None:
    masked = RegionState.from_text(
        LogicalRegion.DIARY,
        "secretÃ°Å¸â„¢â€š",
        visibility="masked",
    )
    snapshot = SharedFieldSnapshot(
        tick_id=2,
        regions=(RegionState.from_text("user_input", "hello"), masked),
    )
    compiled = D64FieldCompiler().compile(snapshot)
    assert compiled.coverage.visited_regions == tuple(
        region.value for region in CANONICAL_REGION_ORDER
    )
    assert compiled.region_text("user_input") == "hello"
    assert compiled.region_text("diary") == ""
    assert snapshot.region("diary").text == "secretÃ°Å¸â„¢â€š"


def test_unsupported_attended_character_fails_closed() -> None:
    snapshot = SharedFieldSnapshot.from_texts({"user_input": "helloÃ°Å¸â„¢â€š"})
    with pytest.raises(UnsupportedActiveCharacterError, match="user_input"):
        D64FieldCompiler().compile(snapshot)


def test_lane_addresses_preserve_span_source_and_exact_positions() -> None:
    span = FieldSpan(
        span_id="evidence-1",
        text="park",
        source="container-42",
        provenance="dormant:42",
    )
    snapshot = SharedFieldSnapshot(
        tick_id=3,
        regions=(RegionState(name="structured_knowledge", spans=(span,)),),
    )
    compiled = D64FieldCompiler().compile(snapshot)
    addresses = compiled.region_addresses("structured_knowledge")
    assert [address.region_position for address in addresses] == [0, 1, 2, 3]
    assert [address.span_position for address in addresses] == [0, 1, 2, 3]
    assert all(address.span_id == "evidence-1" for address in addresses)
    assert all(address.source == "container-42" for address in addresses)
    assert all(address.provenance == "dormant:42" for address in addresses)


def test_stale_rail_is_rejected_after_real_typed_delta() -> None:
    snapshot = SharedFieldSnapshot.from_texts({"scratch": "old"}, tick_id=5)
    compiled = D64FieldCompiler().compile(snapshot)
    delta = replacement_delta(
        snapshot,
        compiled,
        region="scratch",
        text="new",
        author_core_id="core64",
        pass_id="scratch",
    )
    updated = apply_compiled_delta(snapshot, compiled, delta)
    assert updated.region("scratch").text == "new"
    assert updated.parent_field_id == snapshot.field_id
    with pytest.raises(StaleCompiledFieldError):
        compiled.assert_fresh(updated)


def test_cartography_marks_words_sentences_and_paragraphs_exactly() -> None:
    text = "The dog is home.\n\nSecond paragraph!"
    snapshot = SharedFieldSnapshot.from_texts({"user_input": text})
    compiled = D64FieldCompiler().compile(snapshot)
    spans = [span for span in compiled.cartography if span.region is LogicalRegion.USER_INPUT]
    kinds = {span.kind for span in spans}
    assert {"region_text", "word", "sentence", "paragraph"} <= kinds
    dog = next(span for span in spans if span.kind == "word" and span.start == 4)
    assert text[dog.start : dog.end] == "dog"
    assert dog.text_sha256 == hashlib.sha256(b"dog").hexdigest()


def test_compiled_arrays_are_read_only_and_deterministic() -> None:
    snapshot = SharedFieldSnapshot.from_texts({"user_input": "deterministic"})
    first = D64FieldCompiler().compile(snapshot)
    second = D64FieldCompiler().compile(snapshot)
    assert first.rail_id == second.rail_id
    assert np.array_equal(first.rows, second.rows)
    with pytest.raises(ValueError):
        first.rows[0, 0] = 0.0
    with pytest.raises(ValueError):
        first.lane_valid[0, 0] = False


def test_runtime_and_training_materialize_the_same_rail() -> None:
    record = {
        "example_id": "shared-adapter",
        "field": {"user_input": "The dog is at the park."},
        "targets": {"scratch": "locate dog", "response_draft": "The dog is at the park."},
    }
    example = canonicalize_r0_record(record)
    runtime_compiled = CanonicalD64RuntimeAdapter().compile(example.snapshot)
    assert runtime_compiled.rail_id == example.compiled.rail_id
    assert np.array_equal(runtime_compiled.rows, example.compiled.rows)


def test_d64_reader_consumes_only_the_canonical_compiled_rail() -> None:
    field = {
        "conversation_history": "history",
        "user_input": "question",
        "structured_knowledge": "evidence",
        "scratch": "plan",
    }
    snapshot = SharedFieldSnapshot.from_texts(field)
    model = CompleteField64D(
        ReaderConfig(page_size=4, max_output_chars=128, dropout=0.0)
    )
    model.eval()
    with torch.no_grad():
        state, memory, coverage, compiled = model.read_snapshot_with_memory(snapshot)
    assert compiled.coverage.complete
    assert compiled.source_field_id == snapshot.field_id
    assert coverage.complete
    assert state.shape[-1] == 64
    assert memory.states.shape[-1] == 64
    valid_positions = memory.region_positions[memory.region_positions >= 0]
    assert int(valid_positions.numel()) == compiled.coverage.compiled_active_characters
    assert compiled.active_texts() == {
        region.value: snapshot.region(region).text
        for region in CANONICAL_REGION_ORDER
    }


def test_compiler_attends_last_n_spans_exactly() -> None:
    spans = tuple(
        FieldSpan(span_id=f"turn-{i}", text=f"{i}\n") for i in range(5)
    )
    state = RegionState(
        name=LogicalRegion.CONVERSATION_HISTORY,
        spans=spans,
        mask_policy=RegionMaskPolicy("last_n_spans", 2),
    )
    snapshot = SharedFieldSnapshot(tick_id=7, regions=(state,))
    compiled = D64FieldCompiler().compile(snapshot)
    assert compiled.coverage.complete
    assert compiled.coverage.expected_active_characters == 4
    assert compiled.coverage.compiled_active_characters == 4
    assert compiled.region_text("conversation_history") == "3\n4\n"
    assert snapshot.region("conversation_history").text == "0\n1\n2\n3\n4\n"


def test_compiler_attends_disjoint_explicit_intervals() -> None:
    spans = (FieldSpan(span_id="s0", text="abcdefghij"),)
    state = RegionState(
        name=LogicalRegion.USER_INPUT,
        spans=spans,
        attended_intervals=(
            AttendedInterval(0, 2),
            AttendedInterval(5, 7),
        ),
    )
    snapshot = SharedFieldSnapshot(tick_id=3, regions=(state,))
    compiled = D64FieldCompiler().compile(snapshot)
    assert compiled.coverage.complete
    assert compiled.region_text("user_input") == "abfg"
    addresses = compiled.region_addresses("user_input")
    assert [a.region_position for a in addresses] == [0, 1, 5, 6]
    assert [a.span_position for a in addresses] == [0, 1, 5, 6]


def test_compiler_all_policy_attends_full_region_text() -> None:
    state = RegionState.from_text(
        LogicalRegion.USER_INPUT,
        "hello",
        mask_policy=RegionMaskPolicy("all"),
    )
    snapshot = SharedFieldSnapshot(tick_id=0, regions=(state,))
    compiled = D64FieldCompiler().compile(snapshot)
    assert compiled.coverage.complete
    assert compiled.region_text("user_input") == "hello"


def test_compiler_none_policy_yields_empty_rail_for_region() -> None:
    state = RegionState.from_text(
        LogicalRegion.USER_INPUT,
        "secret",
        mask_policy=RegionMaskPolicy("none"),
    )
    snapshot = SharedFieldSnapshot(tick_id=0, regions=(state,))
    compiled = D64FieldCompiler().compile(snapshot)
    assert compiled.coverage.complete
    assert compiled.region_text("user_input") == ""
    assert compiled.region_addresses("user_input") == ()
