from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from runtime.field import (
    AttendedInterval,
    D64FieldCompiler,
    FieldSpan,
    LogicalRegion,
    RegionMaskPolicy,
    RegionState,
    SharedFieldSnapshot,
    apply_compiled_delta,
    replacement_delta,
)
from runtime.heart import D64HeartCodec, D64HeartCodecError
from runtime.heart.translation_core import HeartTranslationCore, HeartTranslationCoreConfig
from substrate import (
    NATIVE_TOKEN_COUNT,
    TRANSPORT_VOCAB_SIZE,
    InvalidUnicodeScalarError,
    MalformedUnicodeTransportError,
    NonCanonicalUnicodeTransportError,
    byte_token_id,
    char_to_slot,
    decode_unicode_tokens,
    default_alphabet,
    encode_unicode_character,
    encode_unicode_text,
    native_token_id,
    transport_token_cell16,
    unicode_transport_bank,
    unicode_transport_geometry,
)
from training.complete_field_64d import CompleteField64D, ReaderConfig


def _legacy_heart_model() -> HeartTranslationCore:
    return HeartTranslationCore(
        HeartTranslationCoreConfig(
            d_model=64,
            n_heads=4,
            n_layers=1,
            ffn_dim=128,
            dropout=0.0,
            source_page_chars=8,
        )
    )


def test_native_token_ids_and_cells_are_frozen_unchanged() -> None:
    alphabet = tuple(default_alphabet())
    bank = unicode_transport_bank()
    assert len(alphabet) == NATIVE_TOKEN_COUNT == 95
    assert bank.shape == (TRANSPORT_VOCAB_SIZE, 16)
    for token_id, character in enumerate(alphabet):
        assert native_token_id(character) == token_id
        assert encode_unicode_character(character).token_ids == (token_id,)
        assert np.array_equal(bank[token_id], transport_token_cell16(token_id))
        assert np.array_equal(bank[token_id], char_to_slot(character))


def test_byte_codebook_has_deterministic_error_correcting_geometry() -> None:
    geometry = unicode_transport_geometry()
    assert geometry["vocab_size"] == 351
    assert geometry["byte_worst_nn_cos"] == pytest.approx(0.5, abs=1e-7)
    assert geometry["byte_native_worst_cos"] < 0.70
    assert geometry["bank_sha256"] == "fb7ee523f9fb61e282c31525df7795c3ee0458c6f6c02dfe3f348c4115bfb148"
    rows = unicode_transport_bank()
    assert len({row.tobytes() for row in rows}) == TRANSPORT_VOCAB_SIZE


def test_every_unicode_scalar_roundtrips_through_canonical_utf8_transport() -> None:
    count = 0
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        character = chr(codepoint)
        encoded = encode_unicode_character(character)
        assert decode_unicode_tokens(encoded.token_ids) == character
        assert 1 <= len(encoded.token_ids) <= 4
        count += 1
    assert count == 1_112_064


@pytest.mark.parametrize("character", ("\ud800", "\udfff"))
def test_unpaired_surrogates_fail_closed(character: str) -> None:
    with pytest.raises(InvalidUnicodeScalarError, match="surrogate"):
        encode_unicode_text(character)


@pytest.mark.parametrize(
    "token_ids",
    (
        (byte_token_id(0xC0),),
        (byte_token_id(0xC2),),
        (byte_token_id(0xE2), native_token_id("a"), byte_token_id(0x80)),
        (byte_token_id(0xED), byte_token_id(0xA0), byte_token_id(0x80)),
        (byte_token_id(0xF4), byte_token_id(0x90), byte_token_id(0x80), byte_token_id(0x80)),
        (byte_token_id(0x80),),
    ),
)
def test_malformed_utf8_transport_fails_closed(token_ids: tuple[int, ...]) -> None:
    with pytest.raises(MalformedUnicodeTransportError):
        decode_unicode_tokens(token_ids)


def test_native_character_cannot_be_spelled_through_byte_alias() -> None:
    with pytest.raises(NonCanonicalUnicodeTransportError, match="native character"):
        decode_unicode_tokens((byte_token_id(ord("a")),))


def test_mixed_unicode_d64_roundtrip_and_receipts_are_exact() -> None:
    text = "`\t\r café e\u0301 — 🙂 中文 العربية"
    snapshot = SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: text}, tick_id=11)
    compiled = D64FieldCompiler().compile(snapshot)
    compiled.verify_roundtrip(snapshot)

    assert compiled.region_text(LogicalRegion.USER_INPUT) == text
    assert compiled.coverage.expected_active_characters == len(text)
    assert compiled.coverage.compiled_active_characters == len(text)
    assert compiled.coverage.compiled_transport_units == len(encode_unicode_text(text))
    assert compiled.coverage.valid_lanes == compiled.coverage.compiled_transport_units
    assert compiled.coverage.native_transport_units + compiled.coverage.utf8_byte_transport_units == (
        compiled.coverage.compiled_transport_units
    )
    assert len(compiled.region_character_addresses(LogicalRegion.USER_INPUT)) == len(text)

    emoji_group = tuple(
        address for address in compiled.region_addresses(LogicalRegion.USER_INPUT) if address.character == "🙂"
    )
    assert len(emoji_group) == 4
    assert tuple(item.transport_unit_index for item in emoji_group) == (0, 1, 2, 3)
    assert all(item.transport_unit_count == 4 for item in emoji_group)
    assert len({item.region_position for item in emoji_group}) == 1
    assert len({item.global_position for item in emoji_group}) == 1


@pytest.mark.parametrize(
    "text",
    (
        'def greet(name: str) -> str:\n\treturn f"Hello, {name} 🙂\\n"\n',
        'class Main { public static void main(String[] a) { System.out.println("café\\n世界"); } }',
        '#include <iostream>\nint main() { std::cout << "π 🙂\\n"; return 0; }\n',
        '{"escaped":"line\\n\\tquote:\\"","raw":"العربية 中文 🙂"}',
    ),
)
def test_programming_language_and_escape_surfaces_roundtrip_exactly(text: str) -> None:
    snapshot = SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: text})
    compiled = D64FieldCompiler().compile(snapshot)
    assert compiled.region_text(LogicalRegion.USER_INPUT) == text
    compiled.verify_roundtrip(snapshot)


def test_character_pages_never_split_one_unicode_scalar() -> None:
    text = "A🙂BéC界"
    compiled = D64FieldCompiler().compile(SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: text}))
    pages = tuple(page for page in compiled.iter_character_pages(2) if page.region is LogicalRegion.USER_INPUT)
    assert "".join(page.text for page in pages) == text
    assert all(len(page.text) <= 2 for page in pages)
    for page in pages:
        for address in page.addresses:
            peers = tuple(item for item in page.addresses if item.region_position == address.region_position)
            assert len(peers) == address.transport_unit_count


def test_masks_select_canonical_characters_before_utf8_expansion() -> None:
    text = "A🙂Bé"
    snapshot = SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: text})
    compiled = D64FieldCompiler().compile(
        snapshot,
        region_masks={LogicalRegion.USER_INPUT: RegionMaskPolicy("tail_percent", 50)},
    )
    assert compiled.region_text(LogicalRegion.USER_INPUT) == "Bé"
    assert compiled.coverage.expected_active_characters == 2
    assert compiled.coverage.compiled_transport_units == 3
    assert snapshot.region(LogicalRegion.USER_INPUT).text == text


def test_unicode_transport_rows_preserve_interval_span_and_provenance_boundaries() -> None:
    state = RegionState(
        name=LogicalRegion.USER_INPUT,
        spans=(
            FieldSpan("left", "🙂a", source="user", provenance="event:1"),
            FieldSpan("right", "界b", source="tool", provenance="event:2"),
        ),
        attended_intervals=(AttendedInterval(0, 2), AttendedInterval(2, 4)),
    )
    compiled = D64FieldCompiler().compile(SharedFieldSnapshot(regions=(state,), tick_id=0))
    for row_index in range(compiled.row_count):
        row = tuple(address for lane in range(4) if (address := compiled.address(row_index, lane)) is not None)
        assert len({item.attended_interval_index for item in row}) <= 1
        assert len({item.span_id for item in row}) <= 1
        assert len({item.provenance for item in row}) <= 1


def test_unicode_replacement_delta_remains_canonical_text() -> None:
    snapshot = SharedFieldSnapshot.from_texts({LogicalRegion.RESPONSE_DRAFT: "old"})
    compiled = D64FieldCompiler().compile(snapshot)
    replacement = "Привет, 世界 🙂"
    delta = replacement_delta(
        snapshot,
        compiled,
        region=LogicalRegion.RESPONSE_DRAFT,
        text=replacement,
        author_core_id="unicode-test",
        pass_id="proposal",
    )
    successor = apply_compiled_delta(snapshot, compiled, delta)
    assert successor.region(LogicalRegion.RESPONSE_DRAFT).text == replacement
    recompiled = D64FieldCompiler().compile(successor)
    assert recompiled.region_text(LogicalRegion.RESPONSE_DRAFT) == replacement


def test_heart_frame_carries_unicode_transport_but_legacy_model_fails_explicitly() -> None:
    text = "Axon🙂"
    frame = D64HeartCodec().compile(SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: text}))
    assert frame.exact_text == text
    assert len(frame.transport_token_ids) == len("Axon") + 4
    assert len(frame.cells16) == len(frame.transport_token_ids)
    assert any(right == left for left, right in itertools.pairwise(frame.canonical_positions))
    with pytest.raises(D64HeartCodecError, match="legacy 95-character"):
        D64HeartCodec.batch_for_model(_legacy_heart_model(), (frame,))


def test_permanent_d64_reader_visits_every_unicode_transport_unit_on_cpu() -> None:
    text = "Python:\n```\nprint('café 🙂')\n```\n"
    compiled = D64FieldCompiler().compile(SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: text}))
    model = CompleteField64D(ReaderConfig(page_size=3, dropout=0.0))
    started = time.perf_counter()
    _, memory, manifest = model.read_compiled_with_memory(compiled)
    elapsed = time.perf_counter() - started
    assert manifest.complete
    assert manifest.observed_characters == len(text)
    assert memory.states.shape[1] >= compiled.coverage.compiled_transport_units
    assert elapsed < 30.0
