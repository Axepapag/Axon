"""Axon 1024D substrate: 64 exact characters per frozen 1024-float vector.

THE LAW (set by Jeff, 2026-10-05):

* One 1024D vector is 64 lanes of 16 floats. Lane k holds the character at
  position k. A lane is exactly one of 96 frozen codes: the 95 native
  characters of the 16D substrate, or EMPTY (unused trailing lane).
* The codes are stored in the committed artifact ``d1024_lane_bank.npy`` and
  pinned by the SHA-256 below. They are never recomputed, tuned, or re-derived.
  Changing any value orphans every core trained on this substrate.
* NO LEARNED PARAMETERS. NO RNG. NO HASHING OF TEXT. NO PROJECTION. The
  mapping between text and vectors is a fixed, mechanical, reversible layout.
* The substrate is an ALPHABET of exactly 95 characters. Anything else
  (emoji, tabs, accented letters, raw bytes) fails closed. There is no byte
  transport and no escape hatch.
* Decoding is verification against the frozen table, never nearest-vector
  guessing. A lane must match one code exactly (or within an explicit,
  bounded tolerance that provably cannot be ambiguous), or decoding fails.
* Canonical form: text of N characters is ceil(N/64) vectors; only the last
  vector may contain EMPTY lanes, and only as a trailing run, and it must
  hold at least one character. Text -> vectors -> text and vectors -> text ->
  vectors are both exact.

Self-test: ``python substrate_1024.py`` prints the conformance report and
exits 1 on any gate failure (``--quiet`` prints nothing).
"""
from __future__ import annotations

import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

SUBSTRATE_1024_SCHEMA = "axon-substrate-1024-lanes-v1"

WIDTH = 1024
LANE_DIM = 16
LANES = WIDTH // LANE_DIM  # 64 characters per vector

# ORDER IS FROZEN: identical to the 16D substrate's native alphabet.
ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .\n!?"
    "(=[,{+')*];}-\"</#:%@&>\\|_^$~"
)
ALPHABET_SIZE = len(ALPHABET)  # 95
EMPTY_ID = ALPHABET_SIZE  # lane id 95; not a character
BANK_ROWS = ALPHABET_SIZE + 1

LANE_BANK_FILE = "d1024_lane_bank.npy"
LANE_BANK_SHA256 = "1b974ecbead74b114bd85e6576976fd59335a8d2db363dff246c867c713f469f"

# Closest pair of lane codes is 0.4946 apart in L2 ('.' vs '!'). A tolerance
# below half of that can never match two codes at once.
MIN_CODE_SEPARATION_L2 = 0.4946
MAX_SAFE_TOLERANCE = 0.24

_CHAR_TO_ID = {ch: i for i, ch in enumerate(ALPHABET)}
_ALPHABET_ARRAY = np.array(list(ALPHABET), dtype=object)


class Substrate1024Error(ValueError):
    """Base class for every fail-closed 1024D substrate rejection."""


class UnsupportedCharacterError(Substrate1024Error):
    """Text contains a character outside the 95-character alphabet."""


class MalformedVectorError(Substrate1024Error):
    """A vector stream is not canonical frozen-lane substrate."""


@lru_cache(maxsize=1)
def lane_bank() -> np.ndarray:
    """The (96, 16) frozen lane codes: rows 0..94 characters, row 95 EMPTY."""
    path = Path(__file__).resolve().parent / LANE_BANK_FILE
    bank = np.load(path, allow_pickle=False)
    if bank.shape != (BANK_ROWS, LANE_DIM) or bank.dtype != np.dtype("<f4"):
        raise Substrate1024Error("lane bank artifact has the wrong shape or dtype")
    digest = hashlib.sha256(bank.astype("<f4", copy=False).tobytes(order="C")).hexdigest()
    if digest != LANE_BANK_SHA256:
        raise Substrate1024Error("lane bank artifact does not match its sealed SHA-256")
    bank = np.array(bank, dtype=np.float32, copy=True)
    bank.setflags(write=False)
    return bank


def text_to_ids(text: str) -> np.ndarray:
    """Exact character ids. Rejects every character outside the alphabet."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    bad = sorted({ch for ch in text if ch not in _CHAR_TO_ID})
    if bad:
        shown = ", ".join(f"U+{ord(ch):04X}" for ch in bad[:8])
        raise UnsupportedCharacterError(
            f"{sum(ch not in _CHAR_TO_ID for ch in text)} character(s) are outside the "
            f"95-character substrate and fail closed: {shown}"
        )
    return np.fromiter((_CHAR_TO_ID[ch] for ch in text), dtype=np.int64, count=len(text))


def ids_to_text(ids: np.ndarray) -> str:
    ids = np.asarray(ids)
    if ids.size and (ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() >= ALPHABET_SIZE):
        raise Substrate1024Error("character ids must be integers in [0, 94]")
    return "".join(_ALPHABET_ARRAY[ids.reshape(-1)].tolist())


def ids_to_vectors(ids: np.ndarray) -> np.ndarray:
    """(N,) character ids -> (ceil(N/64), 1024) float32; trailing lanes EMPTY."""
    ids = np.asarray(ids)
    if ids.ndim != 1:
        raise Substrate1024Error("ids must be one-dimensional")
    if ids.size and (ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() >= ALPHABET_SIZE):
        raise Substrate1024Error("character ids must be integers in [0, 94]")
    count = int(ids.size)
    vectors = -(-count // LANES)
    padded = np.full(vectors * LANES, EMPTY_ID, dtype=np.int64)
    padded[:count] = ids
    return np.ascontiguousarray(lane_bank()[padded].reshape(vectors, WIDTH))


def encode_text(text: str) -> np.ndarray:
    return ids_to_vectors(text_to_ids(text))


def _match_lanes(lanes: np.ndarray, tolerance: float) -> np.ndarray:
    """Lane vectors (M,16) -> bank row ids (M,). Exact (or bounded) or fail."""
    bank = lane_bank()
    bank64 = bank.astype(np.float64)
    bank_norm = np.sum(bank64 * bank64, axis=1)[None, :]
    ids = np.empty(lanes.shape[0], dtype=np.int64)
    misses = 0
    worst = 0.0
    # Chunked so corpus-sized input never builds a giant (lanes x 96) array.
    # The matmul only picks a candidate; the exact distance below is the authority.
    for start in range(0, lanes.shape[0], 1 << 17):
        chunk = lanes[start:start + (1 << 17)].astype(np.float64, copy=False)
        pick = (bank_norm - 2.0 * (chunk @ bank64.T)).argmin(axis=1)
        best = np.linalg.norm(chunk - bank64[pick], axis=1)
        misses += int((best > tolerance).sum())
        worst = max(worst, float(best.max(initial=0.0)))
        ids[start:start + chunk.shape[0]] = pick
    if misses:
        raise MalformedVectorError(
            f"{misses} lane(s) do not match any frozen substrate code "
            f"(closest miss {worst:.6g} > tolerance {tolerance:g})"
        )
    return ids


def _check_tolerance(tolerance: float) -> float:
    tolerance = float(tolerance)
    if not (0.0 <= tolerance <= MAX_SAFE_TOLERANCE):
        raise Substrate1024Error(
            f"tolerance must be in [0, {MAX_SAFE_TOLERANCE}] so a lane can never match two codes"
        )
    return tolerance


def vectors_to_ids(vectors: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
    """(N,1024) vectors -> (characters,) ids. Fails closed on any non-canonical input."""
    tolerance = _check_tolerance(tolerance)
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape[1] != WIDTH:
        raise MalformedVectorError(f"expected shape (N, {WIDTH}); got {vectors.shape}")
    if vectors.dtype.kind != "f":
        raise MalformedVectorError("vectors must be floating point")
    if not bool(np.isfinite(vectors).all()):
        raise MalformedVectorError("vectors contain NaN or infinity")
    if vectors.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    lanes = vectors.astype(np.float32, copy=False).reshape(-1, LANE_DIM)
    ids = _match_lanes(lanes, tolerance)
    empty = ids == EMPTY_ID
    if empty.any():
        first_empty = int(np.argmax(empty))
        if not bool(empty[first_empty:].all()):
            raise MalformedVectorError("EMPTY lane followed by a character")
        if first_empty < (vectors.shape[0] - 1) * LANES + 1:
            raise MalformedVectorError("EMPTY lanes may only pad the last vector, which must hold a character")
    return ids[~empty]


def decode_vectors(vectors: np.ndarray, tolerance: float = 0.0) -> str:
    return ids_to_text(vectors_to_ids(vectors, tolerance))


def cells16_to_vectors(cells: np.ndarray) -> np.ndarray:
    """Mechanical lift: exact 16D cells from the Heart -> 1024D vectors."""
    cells = np.asarray(cells)
    if cells.ndim != 2 or cells.shape[1] != LANE_DIM or cells.dtype.kind != "f":
        raise MalformedVectorError(f"expected float cells of shape (N, {LANE_DIM}); got {cells.shape}")
    if not bool(np.isfinite(cells).all()):
        raise MalformedVectorError("cells contain NaN or infinity")
    if cells.shape[0] == 0:
        return np.zeros((0, WIDTH), dtype=np.float32)
    ids = _match_lanes(cells.astype(np.float32, copy=False), 0.0)
    if bool((ids == EMPTY_ID).any()):
        raise MalformedVectorError("EMPTY is padding, not a character; it cannot be lifted from the Heart")
    return ids_to_vectors(ids)


def vectors_to_cells16(vectors: np.ndarray, tolerance: float = 0.0) -> np.ndarray:
    """Mechanical lower: 1024D vectors -> exact 16D cells to hand back to the Heart."""
    ids = vectors_to_ids(vectors, tolerance)
    return np.ascontiguousarray(lane_bank()[ids])


def _load_16d():
    try:
        from .substrate import char_to_slot, default_alphabet
    except ImportError:
        try:
            from substrate import char_to_slot, default_alphabet
        except ImportError:
            return None
    return char_to_slot, default_alphabet


def verify_substrate_1024(verbose: bool = True) -> bool:
    """Conformance gate. HARD RULES: never loosen.

    0. The artifact matches its sealed SHA-256 (checked on load).
    1. Geometry: 1024 = 64 lanes x 16; 95 unique characters; EMPTY is row 95.
    2. Every lane code is finite, nonzero, and distinct; closest pair >= 0.4946.
    3. The maximum safe tolerance is below half the closest pair.
    4. Text round-trips bit-exactly at every length class (0, 1, 63, 64, 65, 200).
    5. Every character outside the alphabet fails closed.
    6. Non-canonical vectors fail closed (bad lane, EMPTY gap, empty last vector, NaN).
    7. The lane codes equal the frozen 16D substrate (atol 1e-5) when it is importable.
    """
    rules: dict[str, bool] = {}
    details: dict[str, object] = {}
    try:
        bank = lane_bank()
        rules["artifact_sealed"] = True
    except Substrate1024Error as exc:
        details["artifact_error"] = str(exc)
        rules = {"artifact_sealed": False}
        bank = None

    if bank is not None:
        rules["geometry_64x16"] = (
            WIDTH == 1024 and LANES == 64 and LANE_DIM == 16 and ALPHABET_SIZE == 95
            and len(set(ALPHABET)) == 95 and EMPTY_ID == 95 and bank.shape == (96, 16)
        )
        distance = np.linalg.norm(bank[:, None, :] - bank[None, :, :], axis=2)
        np.fill_diagonal(distance, np.inf)
        details["closest_pair_l2"] = float(distance.min())
        rules["codes_finite_nonzero_distinct"] = bool(
            np.isfinite(bank).all() and (np.abs(bank).sum(axis=1) > 0).all()
            and distance.min() >= MIN_CODE_SEPARATION_L2
        )
        rules["tolerance_cannot_be_ambiguous"] = bool(2 * MAX_SAFE_TOLERANCE < distance.min())

        roundtrip = True
        for length in (0, 1, 63, 64, 65, 200):
            text = "".join(ALPHABET[(i * 7 + 3) % ALPHABET_SIZE] for i in range(length))
            vectors = encode_text(text)
            again = decode_vectors(vectors)
            roundtrip &= again == text and vectors.shape == (-(-length // LANES), WIDTH)
            roundtrip &= bool(np.array_equal(encode_text(again), vectors))
        full = ALPHABET * 3
        roundtrip &= decode_vectors(encode_text(full)) == full
        rules["text_roundtrip_exact"] = bool(roundtrip)

        closed = True
        for bad in ("\U0001F600", "\t", "\r", "\u00e9", "a\u200bb", "\ud800"):
            try:
                encode_text(bad)
                closed = False
            except UnsupportedCharacterError:
                pass
        rules["out_of_alphabet_fails_closed"] = closed

        good = encode_text("x" * 70)
        cases = []
        shifted = good.copy(); shifted[0, 0] += 0.25; cases.append(shifted)
        gap = good.copy(); gap[1, 16:32] = bank[EMPTY_ID]; cases.append(gap)
        empty_last = np.concatenate([encode_text("y" * 64), np.tile(bank[EMPTY_ID], LANES)[None, :]])
        cases.append(empty_last)
        nan = good.copy(); nan[0, 5] = np.nan; cases.append(nan)
        cases.append(good[:, :1000])
        rejected = True
        for case in cases:
            try:
                vectors_to_ids(case)
                rejected = False
            except MalformedVectorError:
                pass
        rules["noncanonical_vectors_fail_closed"] = rejected

        loaded = _load_16d()
        if loaded is None:
            details["frozen_16d_cross_check"] = "skipped: 16D substrate not importable"
        else:
            char_to_slot, default_alphabet = loaded
            same_alphabet = "".join(default_alphabet()) == ALPHABET
            rows = np.stack([char_to_slot(c) for c in ALPHABET] + [char_to_slot("<empty>")])
            rules["lane_codes_equal_frozen_16d"] = bool(same_alphabet and np.allclose(rows, bank, atol=1e-5, rtol=0.0))

    if verbose:
        for key, value in details.items():
            print(f"  {key:34s} {value}")
        for key, ok in rules.items():
            print(f"  {'PASS' if ok else 'FAIL':4s}  {key}")
    return bool(rules) and all(rules.values())


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    ok = verify_substrate_1024(verbose=not quiet)
    if not quiet:
        print("\n1024D substrate conformance:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
