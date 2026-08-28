"""Exact additive Unicode transport over Axon's frozen 16D substrate width.

Canonical state remains ordinary Python/Unicode text.  This module defines a
derived categorical transport only:

* the 95 frozen native characters retain their existing token ids and cells;
* any non-native Unicode scalar is encoded as its strict UTF-8 bytes;
* each byte is one deterministic 16D extended-Hamming code cell;
* decoding rejects malformed UTF-8 and non-canonical byte spellings.

The byte cells are not additions to the native character bank and do not
change any pre-existing substrate vector.  Token kind is categorical and is
carried in compiler receipts; nearest-vector guessing has no authority.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal, Sequence

import numpy as np

from .substrate import ALPHABET_SET, SLOT_DIM, char_to_slot, default_alphabet

UNICODE_TRANSPORT_SCHEMA = "axon-unicode-transport-utf8-16d-v1"
NATIVE_TOKEN_COUNT = len(default_alphabet())
BYTE_TOKEN_COUNT = 256
TRANSPORT_VOCAB_SIZE = NATIVE_TOKEN_COUNT + BYTE_TOKEN_COUNT

_NATIVE_CHARACTERS = tuple(default_alphabet())
_NATIVE_TO_ID = {character: index for index, character in enumerate(_NATIVE_CHARACTERS)}
_HAMMING_DATA_POSITIONS = (3, 5, 6, 7, 9, 10, 11, 12)
_HAMMING_PARITY_POSITIONS = (1, 2, 4, 8)


class UnicodeTransportError(ValueError):
    """Base class for exact Unicode transport failures."""


class InvalidUnicodeScalarError(UnicodeTransportError):
    """Canonical text contains an unpaired UTF-16 surrogate."""


class MalformedUnicodeTransportError(UnicodeTransportError):
    """A categorical transport stream is not strict UTF-8."""


class NonCanonicalUnicodeTransportError(UnicodeTransportError):
    """A valid stream uses byte tokens for a registered native character."""


@dataclass(frozen=True, slots=True)
class EncodedUnicodeCharacter:
    """One canonical Unicode scalar and its one-to-four transport token ids."""

    character: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.character, str) or len(self.character) != 1:
            raise TypeError("EncodedUnicodeCharacter.character must be one character")
        if not self.token_ids:
            raise ValueError("encoded Unicode character cannot contain zero transport tokens")

    @property
    def expanded(self) -> bool:
        return len(self.token_ids) > 1 or self.token_ids[0] >= NATIVE_TOKEN_COUNT


def _extended_hamming_codeword(value: int) -> np.ndarray:
    """Encode one byte as an extended-Hamming [16,11,4] signed codeword.

    Eight data bits occupy a fixed subset of the eleven data positions.  The
    remaining data positions stay zero.  Any two byte codewords therefore
    differ in at least four dimensions.  Mapping bits to +/-0.25 produces
    finite unit-norm 16D cells with byte-to-byte cosine at most 0.5.
    """

    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError("byte value must be an integer in [0, 255]")
    bits = np.zeros((16,), dtype=np.uint8)
    for source_bit, position in enumerate(_HAMMING_DATA_POSITIONS):
        bits[position - 1] = (value >> source_bit) & 1
    for parity_position in _HAMMING_PARITY_POSITIONS:
        parity = 0
        for position in range(1, 16):
            if position != parity_position and position & parity_position:
                parity ^= int(bits[position - 1])
        bits[parity_position - 1] = parity
    bits[15] = int(bits[:15].sum()) & 1
    cell = np.where(bits == 1, np.float32(0.25), np.float32(-0.25)).astype(np.float32)
    cell.setflags(write=False)
    return cell


@lru_cache(maxsize=1)
def byte_transport_bank() -> np.ndarray:
    bank = np.stack([_extended_hamming_codeword(value) for value in range(BYTE_TOKEN_COUNT)])
    bank.setflags(write=False)
    return bank


@lru_cache(maxsize=1)
def unicode_transport_bank() -> np.ndarray:
    native = np.stack([char_to_slot(character) for character in _NATIVE_CHARACTERS]).astype(
        np.float32,
        copy=False,
    )
    bank = np.concatenate((native, byte_transport_bank()), axis=0)
    if bank.shape != (TRANSPORT_VOCAB_SIZE, SLOT_DIM):
        raise RuntimeError("Unicode transport bank has invalid shape")
    bank.setflags(write=False)
    return bank


def unicode_transport_bank_sha256() -> str:
    return hashlib.sha256(unicode_transport_bank().astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def native_token_id(character: str) -> int:
    try:
        return _NATIVE_TO_ID[character]
    except KeyError as exc:
        raise UnicodeTransportError(f"character is not in the native transport bank: {character!r}") from exc


def byte_token_id(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
        raise ValueError("byte value must be an integer in [0, 255]")
    return NATIVE_TOKEN_COUNT + value


def transport_token_kind(token_id: int) -> Literal["native", "utf8_byte"]:
    _validate_token_id(token_id)
    return "native" if token_id < NATIVE_TOKEN_COUNT else "utf8_byte"


def transport_token_value(token_id: int) -> str | int:
    _validate_token_id(token_id)
    if token_id < NATIVE_TOKEN_COUNT:
        return _NATIVE_CHARACTERS[token_id]
    return token_id - NATIVE_TOKEN_COUNT


def transport_token_cell16(token_id: int) -> np.ndarray:
    _validate_token_id(token_id)
    cell = unicode_transport_bank()[token_id]
    cell.setflags(write=False)
    return cell


def encode_unicode_character(character: str) -> EncodedUnicodeCharacter:
    if not isinstance(character, str) or len(character) != 1:
        raise TypeError("encode_unicode_character requires one character")
    if character in ALPHABET_SET:
        return EncodedUnicodeCharacter(character, (native_token_id(character),))
    try:
        payload = character.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise InvalidUnicodeScalarError(
            f"canonical text contains an unpaired surrogate U+{ord(character):04X}"
        ) from exc
    return EncodedUnicodeCharacter(
        character,
        tuple(byte_token_id(value) for value in payload),
    )


def encode_unicode_text(text: str) -> tuple[int, ...]:
    if not isinstance(text, str):
        raise TypeError("encode_unicode_text requires text")
    return tuple(token_id for character in text for token_id in encode_unicode_character(character).token_ids)


def decode_unicode_tokens(token_ids: Sequence[int] | Iterable[int]) -> str:
    """Decode one canonical categorical stream, rejecting alternate spellings."""

    tokens = tuple(token_ids)
    output: list[str] = []
    index = 0
    while index < len(tokens):
        token_id = tokens[index]
        _validate_token_id(token_id)
        if token_id < NATIVE_TOKEN_COUNT:
            output.append(_NATIVE_CHARACTERS[token_id])
            index += 1
            continue

        lead = token_id - NATIVE_TOKEN_COUNT
        width = _utf8_sequence_width(lead)
        values = [lead]
        for offset in range(1, width):
            next_index = index + offset
            if next_index >= len(tokens):
                raise MalformedUnicodeTransportError("truncated UTF-8 transport sequence")
            continuation_id = tokens[next_index]
            _validate_token_id(continuation_id)
            if continuation_id < NATIVE_TOKEN_COUNT:
                raise MalformedUnicodeTransportError("native token interrupts a UTF-8 byte sequence")
            continuation = continuation_id - NATIVE_TOKEN_COUNT
            if not 0x80 <= continuation <= 0xBF:
                raise MalformedUnicodeTransportError(f"invalid UTF-8 continuation byte 0x{continuation:02X}")
            values.append(continuation)
        try:
            character = bytes(values).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise MalformedUnicodeTransportError("invalid UTF-8 transport sequence") from exc
        if len(character) != 1:
            raise MalformedUnicodeTransportError("UTF-8 transport unit did not decode to one scalar")
        if character in ALPHABET_SET:
            raise NonCanonicalUnicodeTransportError(
                f"registered native character must use its native token: {character!r}"
            )
        output.append(character)
        index += width
    return "".join(output)


def unicode_transport_geometry() -> dict[str, float | int | str]:
    native = unicode_transport_bank()[:NATIVE_TOKEN_COUNT]
    native = native / np.linalg.norm(native, axis=1, keepdims=True).clip(min=1e-12)
    byte = byte_transport_bank()
    byte_cos = byte @ byte.T
    np.fill_diagonal(byte_cos, -2.0)
    cross = byte @ native.T
    byte_pair = np.unravel_index(int(np.argmax(byte_cos)), byte_cos.shape)
    cross_pair = np.unravel_index(int(np.argmax(cross)), cross.shape)
    return {
        "schema": UNICODE_TRANSPORT_SCHEMA,
        "native_tokens": NATIVE_TOKEN_COUNT,
        "byte_tokens": BYTE_TOKEN_COUNT,
        "vocab_size": TRANSPORT_VOCAB_SIZE,
        "byte_worst_nn_cos": float(byte_cos[byte_pair]),
        "byte_worst_pair": f"0x{byte_pair[0]:02X}-0x{byte_pair[1]:02X}",
        "byte_native_worst_cos": float(cross[cross_pair]),
        "byte_native_worst_pair": (f"0x{cross_pair[0]:02X}-{_NATIVE_CHARACTERS[cross_pair[1]]!r}"),
        "bank_sha256": unicode_transport_bank_sha256(),
    }


def _utf8_sequence_width(lead: int) -> int:
    if 0x00 <= lead <= 0x7F:
        return 1
    if 0xC2 <= lead <= 0xDF:
        return 2
    if 0xE0 <= lead <= 0xEF:
        return 3
    if 0xF0 <= lead <= 0xF4:
        return 4
    raise MalformedUnicodeTransportError(f"invalid UTF-8 lead byte 0x{lead:02X}")


def _validate_token_id(token_id: int) -> None:
    if isinstance(token_id, bool) or not isinstance(token_id, int):
        raise TypeError("transport token ids must be integers")
    if not 0 <= token_id < TRANSPORT_VOCAB_SIZE:
        raise UnicodeTransportError(f"transport token id out of range: {token_id}")


__all__ = [
    "BYTE_TOKEN_COUNT",
    "NATIVE_TOKEN_COUNT",
    "TRANSPORT_VOCAB_SIZE",
    "UNICODE_TRANSPORT_SCHEMA",
    "EncodedUnicodeCharacter",
    "InvalidUnicodeScalarError",
    "MalformedUnicodeTransportError",
    "NonCanonicalUnicodeTransportError",
    "UnicodeTransportError",
    "byte_token_id",
    "byte_transport_bank",
    "decode_unicode_tokens",
    "encode_unicode_character",
    "encode_unicode_text",
    "native_token_id",
    "transport_token_cell16",
    "transport_token_kind",
    "transport_token_value",
    "unicode_transport_bank",
    "unicode_transport_bank_sha256",
    "unicode_transport_geometry",
]
