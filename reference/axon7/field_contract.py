"""Axon v7 field contract — LAW 0: the ONE field-builder.

This module is the single source of truth for the shape of the field.
The trainer imports it. The runtime imports it. Nothing else may
materialize a field. See FIELD_CONTRACT.md Sections 4-6.

The contract is grammar, not geometry: fixed region ORDER, one anchor
row before each region's contents, sizes fully variable. Region anchors
are one-hot 16D vectors; the cores learn region identity from anchors,
never from absolute positions.

The per-core private state (the SOUL) is NOT built here: it is latent —
S rows at d_model, carried tick to tick by the caller, never encoded
from or decoded to letters. This module only shapes the shared,
letter-space field.

Self-test: `python field_contract.py --selftest` exits 1 on any failure
(train.bat uses this as a pre-flight gate).
"""
from __future__ import annotations

import sys

import numpy as np

from substrate import SLOT_DIM, char_to_slot, get_letter_bank

# ---------------------------------------------------------------------------
# Region tables (FIELD_CONTRACT.md 4.1)
# ---------------------------------------------------------------------------

# (name, ++code, anchor_dim, filled_by)
SHARED_REGIONS: list[tuple[str, str, int, str]] = [
    ("input_window",        "I", 0, "harness"),
    ("response_draft",      "R", 1, "axon"),
    ("working_memory",      "W", 2, "axon"),
    ("rolling_summary",     "S", 3, "axon"),
    ("situation_awareness", "A", 4, "axon"),
    ("structured_knowledge","K", 5, "harness"),
    ("diary",               "D", 6, "axon"),
    ("advisor",             "V", 7, "harness"),
    ("tool_results",        "T", 8, "harness"),
    ("task_state",          "P", 9, "axon"),
]
# Anchor dims 10-15 are reserved for future regions.

SHARED_ORDER: tuple[str, ...] = tuple(name for name, _, _, _ in SHARED_REGIONS)
PLUS_CODES: dict[str, str] = {code: name for name, code, _, _ in SHARED_REGIONS}

# Standing buffer (FIELD_CONTRACT.md 4.3). The response draft is the only
# region with reserved empty slots; every other region grows via ++ calls
# executed by the harness between ticks.
DRAFT_BUFFER = 500
BUFFERED_REGIONS: dict[str, int] = {
    "response_draft": DRAFT_BUFFER,
}

_ANCHOR_CACHE: dict[str, np.ndarray] = {}
_ANCHOR_DIMS: dict[str, int] = {name: d for name, _, d, _ in SHARED_REGIONS}


def get_anchor(region: str) -> np.ndarray:
    """One-hot 16D anchor for a region. Frozen, deterministic."""
    v = _ANCHOR_CACHE.get(region)
    if v is None:
        if region not in _ANCHOR_DIMS:
            raise KeyError(f"unknown region: {region}")
        v = np.zeros(SLOT_DIM, dtype=np.float32)
        v[_ANCHOR_DIMS[region]] = 1.0
        v.setflags(write=False)
        _ANCHOR_CACHE[region] = v
    return v


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------

EMPTY16 = char_to_slot("<empty>")

# Row kinds
KIND_ANCHOR = "region"
KIND_LETTER = "letter"


def materialize(
    shared: dict[str, str],
    buffers: dict[str, int] | None = None,
) -> tuple[np.ndarray, list[str], list[tuple[str, int]]]:
    """Build the shared 16D field for one core, per the contract.

    Layout: for each region in canonical order: one anchor row, then one
    row per character of the region's text, then `buffers[region]` empty
    rows (default: the standing draft buffer).

    Soul rows are NOT built here — they live at d_model and are appended
    after ingress by the caller (trainer / runtime).

    Returns (slots, kinds, row_map):
      slots   (N, 16) float32
      kinds   list[str], KIND_ANCHOR or KIND_LETTER per row
      row_map list of (region_name, char_index); char_index -1 marks the
              region's anchor row; char_index >= len(text) marks a
              buffer (empty) row.
    """
    if buffers is None:
        buffers = BUFFERED_REGIONS
    rows: list[np.ndarray] = []
    kinds: list[str] = []
    row_map: list[tuple[str, int]] = []

    for name in SHARED_ORDER:
        text = shared.get(name, "")
        rows.append(get_anchor(name))
        kinds.append(KIND_ANCHOR)
        row_map.append((name, -1))
        for i, c in enumerate(text):
            rows.append(char_to_slot(c))
            kinds.append(KIND_LETTER)
            row_map.append((name, i))
        for j in range(buffers.get(name, 0)):
            rows.append(EMPTY16)
            kinds.append(KIND_LETTER)
            row_map.append((name, len(text) + j))

    slots = np.stack(rows, axis=0).astype(np.float32)
    return slots, kinds, row_map


def target_field(
    shared_after: dict[str, str],
    row_map: list[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Lay the after-state onto the BEFORE-field's rows (N in = N out).

    For each letter row (region, i): target is after_text[i] when i is
    within the after-text, else <empty> (shrinkage = writing toward
    empty). Raises ValueError if any region's after-text is longer than
    the rows available to it — growth without a buffer is a contract
    violation (it needs a ++ call between ticks).

    Returns (targets (N,16) float32, letter_mask (N,) bool). Anchor rows
    get the anchor vector itself and letter_mask False (never letter-
    supervised).
    """
    avail: dict[str, int] = {}
    for region, i in row_map:
        if i >= 0:
            avail[region] = max(avail.get(region, 0), i + 1)
    for region, text in shared_after.items():
        if len(text) > avail.get(region, 0):
            raise ValueError(
                f"region '{region}' grows {avail.get(region, 0)} -> "
                f"{len(text)} within a tick; growth needs a ++ call "
                f"between ticks (contract 4.2)"
            )
    targets = np.zeros((len(row_map), SLOT_DIM), dtype=np.float32)
    letter_mask = np.zeros(len(row_map), dtype=bool)
    for r, (region, i) in enumerate(row_map):
        if i == -1:
            targets[r] = get_anchor(region)
        else:
            text = shared_after.get(region, "")
            targets[r] = char_to_slot(text[i]) if i < len(text) else EMPTY16
            letter_mask[r] = True
    return targets, letter_mask


def decode_region(
    letters16: np.ndarray, row_map: list[tuple[str, int]], region: str
) -> str:
    """Render one region's letter rows from a (N,16) prediction."""
    bank = get_letter_bank()
    chars = []
    for r, (name, i) in enumerate(row_map):
        if name == region and i >= 0:
            chars.append(bank.decode_letter(letters16[r]))
    return "".join(chars)


# ---------------------------------------------------------------------------
# Self-test (train.bat pre-flight; LAW 0 gate)
# ---------------------------------------------------------------------------

def selftest(verbose: bool = True) -> bool:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and cond
        if verbose:
            print(f"  {'PASS' if cond else 'FAIL':4s}  {name}")

    # 1. Anchors orthonormal
    A = np.stack([get_anchor(r) for r in SHARED_ORDER], axis=0)
    C = A @ A.T
    check("anchors_orthonormal", bool(np.allclose(C, np.eye(len(SHARED_ORDER)))))

    # 2. Determinism: two builds bit-identical
    shared = {"input_window": "hello axon", "response_draft": "Hi.",
              "working_memory": "note one", "rolling_summary": "we said hello"}
    s1, k1, m1 = materialize(shared)
    s2, k2, m2 = materialize(shared)
    check("deterministic", bool(np.array_equal(s1, s2)) and k1 == k2 and m1 == m2)

    # 3. Layout: first row of each region is its anchor, in canonical order
    anchor_rows = [(r, i) for r, (name, i) in enumerate(m1) if i == -1]
    names_in_order = [m1[r][0] for r, _ in anchor_rows]
    check("region_order", list(names_in_order) == list(SHARED_ORDER))
    check("anchor_vectors", all(
        np.array_equal(s1[r], get_anchor(m1[r][0])) for r, _ in anchor_rows))

    # 4. Letters decode back exactly
    check("roundtrip_draft", decode_region(s1, m1, "response_draft")
          .startswith("Hi."))
    got = decode_region(s1, m1, "input_window")
    check("roundtrip_input", got[:len("hello axon")] == "hello axon")

    # 5. Buffer present on the draft and nowhere else
    n_draft = sum(1 for name, i in m1 if name == "response_draft" and i >= 0)
    check("draft_buffer", n_draft == len("Hi.") + DRAFT_BUFFER)
    n_wm = sum(1 for name, i in m1 if name == "working_memory" and i >= 0)
    check("no_buffer_elsewhere", n_wm == len("note one"))

    # 6. Target mapping: shrink pads with empty; growth raises
    after = dict(shared)
    after["working_memory"] = "note"                      # shrink: ok
    after["response_draft"] = "Hi. More words now."       # fits buffer: ok
    t, lm = target_field(after, m1)
    check("target_shape", t.shape == s1.shape and lm.shape[0] == s1.shape[0])
    bad = dict(shared)
    bad["working_memory"] = "note one plus growth"        # grow: must raise
    try:
        target_field(bad, m1)
        check("growth_raises", False)
    except ValueError:
        check("growth_raises", True)

    # 7. N in == N out is representable: kinds/lengths agree
    check("row_bookkeeping", len(k1) == s1.shape[0] == len(m1))

    return ok


if __name__ == "__main__":
    ok = selftest(verbose=True)
    print("\nfield contract:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
