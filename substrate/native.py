"""The native 95-character API: the only characters Axon knows.

There is no byte transport, no escape hatch and no normalization here. A
character is one of the 95 frozen native characters (id 0..94) and has exactly
one frozen 16D cell, or it is rejected with UnsupportedCharacterError.
Converting outside text into the 95 is an explicit, logged tool that runs
before this boundary; it never happens silently inside it.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

from .substrate import (
    SLOT_DIM,
    UnsupportedCharacterError,
    char_to_slot,
    default_alphabet,
)

NATIVE_SCHEMA = "axon-substrate-native95-16d-v1"
ALPHABET: str = "".join(default_alphabet())
NATIVE_COUNT = len(ALPHABET)  # 95
_CHAR_TO_ID = {ch: index for index, ch in enumerate(ALPHABET)}

if NATIVE_COUNT != 95 or len(_CHAR_TO_ID) != 95:
    raise RuntimeError("the native alphabet must be exactly 95 distinct characters")


@lru_cache(maxsize=1)
def native_bank() -> np.ndarray:
    """The frozen (95, 16) bank: row i is the cell of character id i."""
    bank = np.stack([char_to_slot(ch) for ch in ALPHABET]).astype(np.float32, copy=True)
    bank.setflags(write=False)
    return bank


def native_bank_sha256() -> str:
    return hashlib.sha256(native_bank().astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def native_id(character: str) -> int:
    if not isinstance(character, str) or len(character) != 1:
        raise TypeError("native_id requires exactly one character")
    try:
        return _CHAR_TO_ID[character]
    except KeyError:
        raise UnsupportedCharacterError(
            f"character U+{ord(character):04X} is outside the frozen 95-character substrate"
        ) from None


def native_char(character_id: int) -> str:
    if isinstance(character_id, bool) or not isinstance(character_id, (int, np.integer)):
        raise TypeError("character id must be an integer")
    if not 0 <= int(character_id) < NATIVE_COUNT:
        raise UnsupportedCharacterError(f"character id {character_id} is outside 0..94")
    return ALPHABET[int(character_id)]


def native_cell16(character_id: int) -> np.ndarray:
    native_char(character_id)
    return native_bank()[int(character_id)]


def assert_native_text(text: str) -> str:
    """Return text unchanged, or reject it (fail closed) naming what is outside."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    bad = sorted({ch for ch in text if ch not in _CHAR_TO_ID})
    if bad:
        shown = ", ".join(f"U+{ord(ch):04X}" for ch in bad[:8])
        raise UnsupportedCharacterError(
            f"text has characters outside the 95-character substrate and fails closed: {shown}"
        )
    return text


def encode_ids(text: str) -> tuple[int, ...]:
    return tuple(_CHAR_TO_ID[ch] for ch in assert_native_text(text))


def decode_ids(ids: Iterable[int] | Sequence[int]) -> str:
    return "".join(native_char(i) for i in ids)


def text_to_cells16(text: str) -> np.ndarray:
    """(N, 16) exact cells for text, one per character."""
    ids = encode_ids(text)
    if not ids:
        return np.zeros((0, SLOT_DIM), dtype=np.float32)
    return native_bank()[np.asarray(ids, dtype=np.int64)].copy()


__all__ = [
    "ALPHABET",
    "NATIVE_COUNT",
    "NATIVE_SCHEMA",
    "UnsupportedCharacterError",
    "assert_native_text",
    "decode_ids",
    "encode_ids",
    "native_bank",
    "native_bank_sha256",
    "native_char",
    "native_cell16",
    "native_id",
    "text_to_cells16",
]
