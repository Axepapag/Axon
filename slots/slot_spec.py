"""Axon slot spec: deterministic 8192D slot pack/unpack on the frozen 16D substrate.

The shared field is made of 8192D slots. Each slot has three payload sections
plus a reserved zero tail:

  dims 0-4095    text payload   (256 chars x 16D = 4096D)
  dims 4096-6143 edge payload  (128 chars x 16D = 2048D)
  dims 6144-6655 control block  (32 chars x 16D = 512D)
  dims 6656-8191 reserved       (1536D zeros)

Pack = concatenation of frozen 16D substrate codes. Unpack = nearest-code
decode via the LetterBank. pack -> unpack -> identical text is a hard gate.

Long text (>256 chars) chains across slots. Dense edges (>128 chars in the
edge payload) chain an edge-continuation slot. The control block records kind,
length, chain index/total, status, and edge-form flags — all as substrate
characters.

Self-test: `python slots/slot_spec.py` round-trips a mixed paragraph with
dense edges, chained text, and every control field — prints PASS/FAIL.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

# When run as a script (python slots/slot_spec.py), ensure the repo root and
# substrate package are importable.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from substrate import (  # noqa: E402
    SLOT_DIM,
    char_to_slot,
    default_alphabet,
    get_letter_bank,
)

# --------------------------------------------------------------------------- #
# Constants — the locked 8192D slot layout
# --------------------------------------------------------------------------- #

SLOT_WIDTH = 8192

TEXT_DIMS = 4096          # dims 0-4095
EDGE_DIMS = 2048          # dims 4096-6143
CONTROL_DIMS = 512         # dims 6144-6655
RESERVED_DIMS = 1536       # dims 6656-8191

TEXT_OFFSET = 0
EDGE_OFFSET = TEXT_DIMS            # 4096
CONTROL_OFFSET = EDGE_OFFSET + EDGE_DIMS  # 6144
RESERVED_OFFSET = CONTROL_OFFSET + CONTROL_DIMS  # 6656

MAX_TEXT_CHARS = TEXT_DIMS // SLOT_DIM       # 256
MAX_EDGE_CHARS = EDGE_DIMS // SLOT_DIM      # 128
MAX_CONTROL_CHARS = CONTROL_DIMS // SLOT_DIM  # 32

ALPHABET_SET = set(default_alphabet())

# Control-block field positions (character indices within the control section)
CTRL_KIND_END = 1           # 1 char for kind
CTRL_LEN_END = 4            # 3 chars for length (zero-padded decimal)
CTRL_CHAIN_IDX_END = 8      # 4 chars for chain index
CTRL_CHAIN_TOTAL_END = 12   # 4 chars for chain total
CTRL_STATUS_END = 13        # 1 char for status
CTRL_EDGE_FORM_END = 14     # 1 char for edge form flag
# chars 14-31 reserved for future control fields

# Kind codes (single substrate character)
KIND_TEXT = "T"
KIND_EDGE_CONT = "E"       # edge-continuation slot
KIND_RESPONSE_DRAFT = "R"
KIND_CONTROL = "C"
KIND_EMPTY = " "

# Status codes
STATUS_ACTIVE = "A"
STATUS_DORMANT = "D"
STATUS_DRAFT = "d"
STATUS_COMMITTED = "K"
STATUS_EMPTY = " "

# Edge form flags
EDGE_FORM_FULL = "F"       # full English word-edges like "is a.animal"
EDGE_FORM_NONE = "N"        # no edges in this slot


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _validate_text(text: str) -> None:
    """Reject characters outside the substrate alphabet."""
    bad = sorted(set(c for c in text if c not in ALPHABET_SET))
    if bad:
        raise ValueError(
            f"text contains characters outside the substrate alphabet: {bad!r}"
        )


def _validate_edge_text(edge_str: str) -> None:
    """Edge payload must be substrate-valid."""
    bad = sorted(set(c for c in edge_str if c not in ALPHABET_SET))
    if bad:
        raise ValueError(
            f"edge payload contains characters outside the substrate alphabet: {bad!r}"
        )


# --------------------------------------------------------------------------- #
# Control block encoding
# --------------------------------------------------------------------------- #

@dataclass
class ControlBlock:
    """Decoded control block from a slot's control section."""
    kind: str = KIND_EMPTY
    length: int = 0
    chain_index: int = 0
    chain_total: int = 1
    status: str = STATUS_EMPTY
    edge_form: str = EDGE_FORM_NONE

    def to_control_chars(self) -> str:
        """Encode control fields as a fixed-length substrate-safe string."""
        s = (
            self.kind.ljust(CTRL_KIND_END)[:CTRL_KIND_END]
            + str(self.length).zfill(3)[:3]
            + str(self.chain_index).zfill(4)[:4]
            + str(self.chain_total).zfill(4)[:4]
            + self.status.ljust(1)[:1]
            + self.edge_form.ljust(1)[:1]
        )
        # Pad to full control section
        s = s.ljust(MAX_CONTROL_CHARS)
        # Truncate to exact
        s = s[:MAX_CONTROL_CHARS]
        return s

    @classmethod
    def from_control_chars(cls, chars: str) -> "ControlBlock":
        """Decode a control string (must be MAX_CONTROL_CHARS long)."""
        if len(chars) < CTRL_EDGE_FORM_END:
            return cls()
        kind = chars[0] if chars[0] != " " else KIND_EMPTY
        length_str = chars[1:CTRL_LEN_END].strip()
        chain_idx_str = chars[CTRL_LEN_END:CTRL_CHAIN_IDX_END].strip()
        chain_total_str = chars[CTRL_CHAIN_IDX_END:CTRL_CHAIN_TOTAL_END].strip()
        status = chars[CTRL_CHAIN_TOTAL_END] if chars[CTRL_CHAIN_TOTAL_END] != " " else STATUS_EMPTY
        edge_form = chars[CTRL_CHAIN_TOTAL_END + 1] if len(chars) > CTRL_CHAIN_TOTAL_END + 1 and chars[CTRL_CHAIN_TOTAL_END + 1] != " " else EDGE_FORM_NONE

        try:
            length = int(length_str) if length_str else 0
        except ValueError:
            length = 0
        try:
            chain_index = int(chain_idx_str) if chain_idx_str else 0
        except ValueError:
            chain_index = 0
        try:
            chain_total = int(chain_total_str) if chain_total_str else 1
        except ValueError:
            chain_total = 1

        return cls(
            kind=kind, length=length, chain_index=chain_index,
            chain_total=chain_total, status=status, edge_form=edge_form,
        )


# --------------------------------------------------------------------------- #
# Slot dataclass
# --------------------------------------------------------------------------- #

@dataclass
class Slot:
    """One 8192D slot vector + decoded metadata."""
    vector: np.ndarray
    text: str
    edges: str
    control: ControlBlock

    def __post_init__(self) -> None:
        if self.vector.shape != (SLOT_WIDTH,):
            raise ValueError(f"slot vector must be ({SLOT_WIDTH},), got {self.vector.shape}")
        if self.vector.dtype != np.float32:
            self.vector = self.vector.astype(np.float32)


# --------------------------------------------------------------------------- #
# Pack / unpack primitives
# --------------------------------------------------------------------------- #

def _pack_chars(text: str, out: np.ndarray, offset: int, max_chars: int) -> None:
    """Write frozen 16D char slots into `out` starting at `offset`."""
    for i, char in enumerate(text[:max_chars]):
        slot = char_to_slot(char)
        out[offset + i * SLOT_DIM : offset + (i + 1) * SLOT_DIM] = slot


def _unpack_chars(vec: np.ndarray, offset: int, n_chars: int) -> str:
    """Decode n_chars from the slot vector starting at `offset`."""
    if n_chars == 0:
        return ""
    bank = get_letter_bank()
    chars: list[str] = []
    for i in range(n_chars):
        start = offset + i * SLOT_DIM
        char_vec = vec[start : start + SLOT_DIM]
        chars.append(bank.decode_letter(char_vec))
    return "".join(chars)


def _pack_control(control: ControlBlock, out: np.ndarray) -> None:
    """Pack the control block into dims 6144-6655."""
    ctrl_str = control.to_control_chars()
    _pack_chars(ctrl_str, out, CONTROL_OFFSET, MAX_CONTROL_CHARS)


def _unpack_control(vec: np.ndarray) -> ControlBlock:
    """Unpack the control block from dims 6144-6655."""
    ctrl_str = _unpack_chars(vec, CONTROL_OFFSET, MAX_CONTROL_CHARS)
    return ControlBlock.from_control_chars(ctrl_str)


def pack_slot(
    text: str = "",
    edges: str = "",
    kind: str = KIND_TEXT,
    status: str = STATUS_ACTIVE,
    chain_index: int = 0,
    chain_total: int = 1,
    edge_form: str = EDGE_FORM_NONE,
) -> Slot:
    """Pack text, edges, and control into one 8192D slot.

    Text is truncated at MAX_TEXT_CHARS (256). Edges at MAX_EDGE_CHARS (128).
    The caller is responsible for chaining when text or edges overflow —
    use pack_text_chain / pack_edge_chain for that.
    """
    _validate_text(text)
    _validate_edge_text(edges)

    vec = np.zeros(SLOT_WIDTH, dtype=np.float32)
    text_len = min(len(text), MAX_TEXT_CHARS)
    edge_len = min(len(edges), MAX_EDGE_CHARS)

    _pack_chars(text, vec, TEXT_OFFSET, MAX_TEXT_CHARS)
    _pack_chars(edges, vec, EDGE_OFFSET, MAX_EDGE_CHARS)

    ctrl = ControlBlock(
        kind=kind,
        length=text_len,
        chain_index=chain_index,
        chain_total=chain_total,
        status=status,
        edge_form=edge_form if edge_len > 0 else EDGE_FORM_NONE,
    )
    _pack_control(ctrl, vec)

    return Slot(
        vector=vec,
        text=text[:text_len],
        edges=edges[:edge_len],
        control=ctrl,
    )


def unpack_slot(vec: np.ndarray) -> Slot:
    """Unpack an 8192D slot vector into text, edges, and control."""
    if vec.shape != (SLOT_WIDTH,):
        raise ValueError(f"slot vector must be ({SLOT_WIDTH},), got {vec.shape}")

    ctrl = _unpack_control(vec)
    text = _unpack_chars(vec, TEXT_OFFSET, ctrl.length)
    # Edge length is not stored separately in v1 — decode until we hit
    # a zero-norm (empty) slot or the max edge chars.
    bank = get_letter_bank()
    edge_chars: list[str] = []
    for i in range(MAX_EDGE_CHARS):
        start = EDGE_OFFSET + i * SLOT_DIM
        char_vec = vec[start : start + SLOT_DIM]
        if bank.is_empty(char_vec):
            break
        edge_chars.append(bank.decode_letter(char_vec))
    edges = "".join(edge_chars)

    return Slot(vector=vec.astype(np.float32), text=text, edges=edges, control=ctrl)


# --------------------------------------------------------------------------- #
# Chaining for long text and dense edges
# --------------------------------------------------------------------------- #

def pack_text_chain(
    text: str,
    edges: str = "",
    kind: str = KIND_TEXT,
    status: str = STATUS_ACTIVE,
    edge_form: str = EDGE_FORM_NONE,
) -> list[Slot]:
    """Pack text that may exceed 256 chars into a chain of slots.

    Text is split into 256-char chunks. Each chunk becomes one slot.
    All slots in the chain carry the same edges (if they fit) — only the
    first slot carries edges; continuation slots have empty edge payloads
    unless edges overflow too (then use pack_full_chain).
    """
    _validate_text(text)
    _validate_edge_text(edges)

    if not text:
        return [pack_slot(text="", edges=edges, kind=kind, status=status,
                          edge_form=edge_form)]

    # Split text into MAX_TEXT_CHARS chunks
    chunks = [text[i : i + MAX_TEXT_CHARS] for i in range(0, len(text), MAX_TEXT_CHARS)]
    total = len(chunks)

    # Edge handling: if edges fit in 128 chars, put them in the first slot
    edge_str = edges[:MAX_EDGE_CHARS] if edges else ""
    overflow_edges = edges[MAX_EDGE_CHARS:] if len(edges) > MAX_EDGE_CHARS else ""

    slots: list[Slot] = []
    for i, chunk in enumerate(chunks):
        slot_edges = edge_str if i == 0 else ""
        slot_edge_form = edge_form if (i == 0 and edge_str) else EDGE_FORM_NONE
        slots.append(pack_slot(
            text=chunk,
            edges=slot_edges,
            kind=kind,
            status=status,
            chain_index=i,
            chain_total=total,
            edge_form=slot_edge_form,
        ))

    # If edges overflow, chain edge-continuation slots
    if overflow_edges:
        edge_chunks = [overflow_edges[j : j + MAX_EDGE_CHARS]
                       for j in range(0, len(overflow_edges), MAX_EDGE_CHARS)]
        for j, echunk in enumerate(edge_chunks):
            slots.append(pack_slot(
                text="",
                edges=echunk,
                kind=KIND_EDGE_CONT,
                status=status,
                chain_index=j,
                chain_total=len(edge_chunks),
                edge_form=edge_form,
            ))

    return slots


def unpack_chain(slots: Sequence[Slot]) -> tuple[str, str]:
    """Unpack a chain of slots into (full_text, full_edges)."""
    text_parts: list[str] = []
    edge_parts: list[str] = []

    for slot in slots:
        if slot.control.kind == KIND_EDGE_CONT:
            edge_parts.append(slot.edges)
        else:
            text_parts.append(slot.text)
            # Edges from the first non-continuation slot
            if slot.edges and not edge_parts:
                edge_parts.append(slot.edges)

    return "".join(text_parts), "".join(edge_parts)


# --------------------------------------------------------------------------- #
# Region packing / unpacking
# --------------------------------------------------------------------------- #

def pack_region(slots: Sequence[Slot]) -> np.ndarray:
    """Pack a list of Slot objects into a (N, 8192) array."""
    if not slots:
        return np.zeros((0, SLOT_WIDTH), dtype=np.float32)
    return np.stack([s.vector for s in slots], axis=0)


def unpack_region(arr: np.ndarray) -> list[Slot]:
    """Unpack a (N, 8192) array into a list of Slot objects."""
    if arr.ndim != 2 or arr.shape[1] != SLOT_WIDTH:
        raise ValueError(f"region array must be (N, {SLOT_WIDTH}), got {arr.shape}")
    return [unpack_slot(arr[i]) for i in range(arr.shape[0])]


# --------------------------------------------------------------------------- #
# Edge payload formatting helpers
# --------------------------------------------------------------------------- #

# Edge payload encoding uses only substrate characters:
#   - period '.' separates edge_type from target:  isa.animal
#   - space ' ' separates multiple edges:           isa.animal hasp.furry
# Underscores in edge types are dropped (is_a -> isa) to stay alphanumeric.

def format_edge_full(edge_type: str, target: str, sense: str = "") -> str:
    """Format a full word-edge using substrate-safe characters.

    Example: format_edge_full("is_a", "animal") -> "isa.animal"
             format_edge_full("is_a", "bank", "river") -> "isa.bank.river"
    """
    et = edge_type.replace("_", "")
    s = f"{et}.{target}"
    if sense:
        s += f".{sense}"
    _validate_edge_text(s)
    return s


def format_edges(edges: list[str], max_chars: int = MAX_EDGE_CHARS) -> tuple[str, str, bool]:
    """Format a list of edge strings into the edge payload.

    Returns (packed_str, edge_form, overflowed).
    Joins edges with spaces. If the result exceeds max_chars, returns what
    fits and sets overflowed=True.
    """
    if not edges:
        return "", EDGE_FORM_NONE, False

    packed = " ".join(edges)
    if len(packed) <= max_chars:
        return packed, EDGE_FORM_FULL, False

    # Does not fit: return what fits, mark overflow. Continuation slots carry
    # the rest as full English edge text.
    return packed[:max_chars], EDGE_FORM_FULL, True


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def selftest() -> bool:
    """Round-trip a mixed paragraph with dense edges, chained text, and control fields."""
    bank = get_letter_bank()
    all_ok = True
    checks: list[tuple[str, bool]] = []

    # 1. Simple slot round-trip
    text = "The dog is a good boy."
    slot = pack_slot(text=text, kind=KIND_TEXT, status=STATUS_ACTIVE)
    decoded = unpack_slot(slot.vector)
    ok = decoded.text == text
    checks.append(("simple_slot_roundtrip", ok))
    all_ok &= ok

    # 2. Slot with edges
    edges = format_edge_full("is_a", "animal") + " " + format_edge_full("has_property", "furry")
    slot = pack_slot(text="dog", edges=edges, edge_form=EDGE_FORM_FULL)
    decoded = unpack_slot(slot.vector)
    ok = decoded.text == "dog" and decoded.edges == edges
    checks.append(("slot_with_edges", ok))
    all_ok &= ok

    # 3. Chained text (long paragraph > 256 chars)
    long_text = "This is a long paragraph. " * 15  # ~360 chars
    slots = pack_text_chain(long_text, kind=KIND_TEXT, status=STATUS_ACTIVE)
    full_text, _ = unpack_chain(slots)
    ok = full_text == long_text
    checks.append(("chained_text_roundtrip", ok))
    all_ok &= ok

    # 4. Chain metadata correctness
    ok = (len(slots) >= 2 and
          slots[0].control.chain_index == 0 and
          slots[-1].control.chain_index == len(slots) - 1 and
          all(s.control.chain_total == len(slots) for s in slots))
    checks.append(("chain_metadata", ok))
    all_ok &= ok

    # 5. Dense edges with overflow (edge continuation)
    dense_edges = " ".join(format_edge_full("is_a", f"cat{i}") for i in range(30))
    slots = pack_text_chain("dog", edges=dense_edges, edge_form=EDGE_FORM_FULL)
    _, full_edges = unpack_chain(slots)
    ok = full_edges == dense_edges
    checks.append(("dense_edge_overflow_chain", ok))
    all_ok &= ok

    # 6. Control block round-trip
    ctrl = ControlBlock(kind=KIND_RESPONSE_DRAFT, length=42, chain_index=1,
                        chain_total=3, status=STATUS_DRAFT, edge_form=EDGE_FORM_FULL)
    ctrl_str = ctrl.to_control_chars()
    ctrl2 = ControlBlock.from_control_chars(ctrl_str)
    ok = (ctrl2.kind == ctrl.kind and
          ctrl2.length == ctrl.length and
          ctrl2.chain_index == ctrl.chain_index and
          ctrl2.chain_total == ctrl.chain_total and
          ctrl2.status == ctrl.status and
          ctrl2.edge_form == ctrl.edge_form)
    checks.append(("control_block_roundtrip", ok))
    all_ok &= ok

    # 7. Region pack/unpack
    slots_in = [
        pack_slot(text="hello", kind=KIND_TEXT),
        pack_slot(text="world", kind=KIND_TEXT),
        pack_slot(text="", edges=format_edge_full("is_a", "animal"), kind=KIND_EDGE_CONT,
                   edge_form=EDGE_FORM_FULL),
    ]
    arr = pack_region(slots_in)
    slots_out = unpack_region(arr)
    ok = (len(slots_out) == 3 and
          slots_out[0].text == "hello" and
          slots_out[1].text == "world" and
          slots_out[2].control.kind == KIND_EDGE_CONT)
    checks.append(("region_pack_unpack", ok))
    all_ok &= ok

    # 8. Empty slot
    slot = pack_slot(text="", kind=KIND_EMPTY, status=STATUS_EMPTY)
    decoded = unpack_slot(slot.vector)
    ok = decoded.text == "" and decoded.control.kind == KIND_EMPTY
    checks.append(("empty_slot", ok))
    all_ok &= ok

    # 9. Invalid character rejection
    try:
        pack_slot(text="hello\tworld")
        checks.append(("invalid_char_rejection", False))
        all_ok = False
    except ValueError:
        checks.append(("invalid_char_rejection", True))

    # 10. Overlength text is handled by chaining, not silent truncation
    big_text = "A" * 300  # > 256 chars
    slots = pack_text_chain(big_text)
    full, _ = unpack_chain(slots)
    ok = full == big_text and len(slots) == 2  # 256 + 44
    checks.append(("overlength_text_chain", ok))
    all_ok &= ok

    # Print results
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}")

    print()
    if all_ok:
        print(f"slot_spec self-test: PASS ({len(checks)} checks)")
    else:
        failed = [n for n, ok in checks if not ok]
        print(f"slot_spec self-test: FAIL ({len(failed)} failed: {failed})")
    return all_ok


if __name__ == "__main__":
    ok = selftest()
    sys.exit(0 if ok else 1)
