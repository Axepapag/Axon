"""The 95-character law, enforced at every boundary, and the 16D -> 1024D chain."""
from __future__ import annotations

import numpy as np
import pytest

from runtime.field import (
    D16ViewDelta,
    FieldSpan,
    InsertText,
    LogicalRegion,
    RegionState,
    SharedFieldSnapshot,
    apply_delta,
    materialize_d16_view,
    replacement_delta,
)
from substrate import UnsupportedCharacterError, char_to_slot, native
from substrate import substrate_1024 as s1024

OUTSIDERS = ["\U0001F600", "\t", "\r", "\u00e9", "`", "\u2014", "\u201c", "\x00", "\u200b"]


def test_native_alphabet_is_the_95_and_matches_the_1024_substrate():
    assert native.NATIVE_COUNT == 95 and len(set(native.ALPHABET)) == 95
    assert native.ALPHABET == s1024.ALPHABET
    assert np.array_equal(native.native_bank(), s1024.lane_bank()[:95])


@pytest.mark.parametrize("bad", OUTSIDERS)
def test_every_outsider_fails_closed_at_the_substrate(bad):
    with pytest.raises(UnsupportedCharacterError):
        char_to_slot(bad)
    with pytest.raises(UnsupportedCharacterError):
        native.encode_ids("ok" + bad)
    with pytest.raises(UnsupportedCharacterError):
        native.native_id(bad)


@pytest.mark.parametrize("bad", OUTSIDERS)
def test_every_outsider_fails_closed_at_the_shared_field(bad):
    with pytest.raises(UnsupportedCharacterError):
        FieldSpan(span_id="s", text="hello" + bad)
    with pytest.raises(UnsupportedCharacterError):
        RegionState.from_text(LogicalRegion.SCRATCH, "x" + bad)
    with pytest.raises(UnsupportedCharacterError):
        SharedFieldSnapshot.from_texts({"scratch": "x" + bad})


@pytest.mark.parametrize("bad", ["\U0001F600", "\t", "\u00e9"])
def test_outsiders_cannot_enter_through_deltas(bad):
    base = SharedFieldSnapshot.from_texts({"scratch": "old"})
    with pytest.raises(UnsupportedCharacterError):
        apply_delta(base, replacement_delta(base, region="scratch", text="new" + bad, author_core_id="c", pass_id=1))
    # A well-formed delta against the same base still works and stays exact.
    ok = apply_delta(base, replacement_delta(base, region="scratch", text="new", author_core_id="c", pass_id=1))
    assert ok.region("scratch").text == "new"


def test_all_95_characters_round_trip_heart_to_16d_to_1024d_and_back():
    text = native.ALPHABET * 2  # 190 characters, spans three 64-lane vectors
    field = SharedFieldSnapshot.from_texts({"scratch": text})
    view = materialize_d16_view(field)
    region = view.region(LogicalRegion.SCRATCH)
    assert region.text == text and region.cells16.shape == (190, 16)
    assert all(a.native_id == native.native_id(a.character) for a in region.addresses)

    vectors = s1024.cells16_to_vectors(region.cells16)
    assert vectors.shape == (3, 1024) and np.array_equal(vectors, s1024.encode_text(text))
    lowered = s1024.vectors_to_cells16(vectors)
    assert np.array_equal(lowered, region.cells16)
    assert s1024.decode_vectors(vectors) == text


def test_heart_view_delta_changes_only_the_edited_region():
    base = SharedFieldSnapshot.from_texts({"user_input": "hello", "scratch": "old"})
    successor = apply_delta(
        base, replacement_delta(base, region="scratch", text="new", author_core_id="c", pass_id=1)
    )
    before, after = materialize_d16_view(base), materialize_d16_view(successor)
    delta = D16ViewDelta.between(before, after)
    assert [patch.region for patch in delta.patches] == [LogicalRegion.SCRATCH]
    assert delta.apply(before).identity == after.identity
    assert np.array_equal(delta.apply(before).cells16, after.cells16)


def test_insert_operation_text_is_also_checked():
    base = SharedFieldSnapshot.from_texts({"scratch": "abc"})
    from runtime.field import FieldDelta

    with pytest.raises(UnsupportedCharacterError):
        apply_delta(
            base,
            FieldDelta(
                base_field_id=base.field_id,
                base_tick_id=base.tick_id,
                author_core_id="c",
                pass_id=1,
                operations=(InsertText(region=LogicalRegion.SCRATCH, offset=1, text="\U0001F600"),),
            ),
        )
