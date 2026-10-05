"""Conformance tests for the frozen 1024D substrate (64 lanes x 16 floats)."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from substrate import substrate_1024 as s

ROOT = Path(__file__).resolve().parents[1]


def _text(length: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    return "".join(s.ALPHABET[int(i)] for i in rng.integers(0, s.ALPHABET_SIZE, size=length))


def test_geometry_and_seal():
    assert (s.WIDTH, s.LANE_DIM, s.LANES) == (1024, 16, 64)
    assert s.ALPHABET_SIZE == 95 and len(set(s.ALPHABET)) == 95
    bank = s.lane_bank()
    assert bank.shape == (96, 16) and bank.dtype == np.float32 and not bank.flags.writeable
    assert hashlib.sha256(bank.astype("<f4").tobytes()).hexdigest() == s.LANE_BANK_SHA256


def test_lane_codes_are_the_frozen_16d_substrate():
    from substrate import char_to_slot, default_alphabet

    assert "".join(default_alphabet()) == s.ALPHABET
    rows = np.stack([char_to_slot(c) for c in s.ALPHABET] + [char_to_slot("<empty>")])
    assert np.allclose(rows, s.lane_bank(), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("length", [0, 1, 15, 63, 64, 65, 127, 128, 129, 1000])
def test_text_roundtrip_is_bit_exact(length):
    text = _text(length, seed=length)
    vectors = s.encode_text(text)
    assert vectors.shape == (-(-length // 64), 1024) and vectors.dtype == np.float32
    assert s.decode_vectors(vectors) == text
    assert np.array_equal(s.encode_text(s.decode_vectors(vectors)), vectors)


def test_layout_is_literal_lanes():
    vectors = s.encode_text("cat")
    bank = s.lane_bank()
    for position, ch in enumerate("cat"):
        assert np.array_equal(vectors[0, position * 16:(position + 1) * 16], bank[s.ALPHABET.index(ch)])
    assert np.array_equal(vectors[0, 3 * 16:].reshape(61, 16), np.tile(bank[s.EMPTY_ID], (61, 1)))
    assert s.decode_vectors(s.encode_text("cat")) != s.decode_vectors(s.encode_text("owl"))


def test_cells16_lift_and_lower_match_text_path():
    from substrate import text_to_field

    text = _text(150, seed=7)
    cells = text_to_field(text)
    vectors = s.cells16_to_vectors(cells)
    assert np.array_equal(vectors, s.encode_text(text))
    assert np.array_equal(s.vectors_to_cells16(vectors), cells)


@pytest.mark.parametrize("bad", ["\U0001F600", "\t", "\r", "\u00e9", "a\u200bb", "\ud800", "\x00"])
def test_characters_outside_95_fail_closed(bad):
    with pytest.raises(s.UnsupportedCharacterError):
        s.encode_text("ok" + bad)


def test_old_byte_transport_cells_are_rejected():
    # The retired byte-transport cells were unit-length +/-0.25 codewords.
    byte_style_cell = np.where(np.arange(16) % 2 == 0, 0.25, -0.25).astype(np.float32)[None, :]
    with pytest.raises(s.MalformedVectorError):
        s.cells16_to_vectors(byte_style_cell)


def test_noncanonical_vectors_fail_closed():
    bank = s.lane_bank()
    good = s.encode_text("x" * 70)
    shifted = good.copy(); shifted[0, 0] += 0.25
    gap = good.copy(); gap[1, 16:32] = bank[s.EMPTY_ID]
    empty_last = np.concatenate([s.encode_text("y" * 64), np.tile(bank[s.EMPTY_ID], 64)[None, :]])
    short_first = s.encode_text("a" * 10)
    short_first = np.concatenate([short_first, s.encode_text("b" * 64)])
    nan = good.copy(); nan[0, 5] = np.nan
    for case in (shifted, gap, empty_last, short_first, nan, good[:, :1000], good.astype(np.int32), good[0]):
        with pytest.raises(s.MalformedVectorError):
            s.vectors_to_ids(case)


def test_empty_cannot_be_lifted_from_the_heart():
    with pytest.raises(s.MalformedVectorError):
        s.cells16_to_vectors(s.lane_bank()[[s.EMPTY_ID]])


def test_bounded_tolerance_is_safe_and_never_ambiguous():
    vectors = s.encode_text("hello world")
    noisy = vectors + np.float32(0.05) * np.sign(np.sin(np.arange(1024, dtype=np.float32)))[None, :] / 4
    with pytest.raises(s.MalformedVectorError):
        s.decode_vectors(noisy)
    assert s.decode_vectors(noisy, tolerance=0.2) == "hello world"
    for bad_tolerance in (-0.1, s.MAX_SAFE_TOLERANCE + 0.01, 0.5):
        with pytest.raises(s.Substrate1024Error):
            s.decode_vectors(vectors, tolerance=bad_tolerance)
    bank = s.lane_bank()
    separation = np.linalg.norm(bank[:, None] - bank[None], axis=2)
    np.fill_diagonal(separation, np.inf)
    assert 2 * s.MAX_SAFE_TOLERANCE < separation.min()


def test_self_test_exits_zero():
    result = subprocess.run(
        [sys.executable, str(ROOT / "substrate" / "substrate_1024.py"), "--quiet"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
