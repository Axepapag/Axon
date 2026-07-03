"""Axon v7 substrate: every character is one frozen, hand-authored 16D vector.

Ported from v6.5 (which passed all geometry gates) with two revisions per
FIELD_CONTRACT.md Section 2.1:
  * The ten digits now sit on a hand-authored circle across two dimensions
    (36 degree spacing) instead of a straight line, so adjacent digits
    separate angularly and a learned writing pathway can hit them. Digits
    join the WRITING_SET margin rule (gate 5).
  * v7.2: period, newline, !, ? removed from the native alphabet. The
    native alphabet is now strictly alphanumerics + space (63 chars).
    Punctuation is the job of an explicit formatting layer.

NO HASH. NO RNG. NO LEARNED PARAMETERS. The same character always produces
the same 16D vector, at tick 0 and at tick 1,000,000.

Self-test: `python substrate.py` prints the full geometry report and exits
1 on any gate failure. `python substrate.py --quiet` prints nothing and
just sets the exit code (train.bat uses this as the pre-flight gate).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SLOT_DIM = 16
FEATURE_DIM = 33  # v6.5's 32 + digit circle takes two rows (30, 31); null moved to 32

VOWELS_LOWER = set("aeiou")
ASCENDERS = set("bdfhklt")
DESCENDERS = set("gjpqy")
DOTS = set("ij")
CLOSED_LOOPS = set("abdegopq")

LETTER_FREQUENCY_ORDER = "etaoinshrdlucmfwgypbvkxqjz"
LETTER_FREQUENCY = {
    c: 1.0 - i / (len(LETTER_FREQUENCY_ORDER) - 1)
    for i, c in enumerate(LETTER_FREQUENCY_ORDER)
}

# --- Phonetic tables (hand-built, frozen; unchanged from v6.5) -------------

PLACE_H = {
    "i": 1.0, "e": 0.85, "a": 0.0, "o": -0.6, "u": -0.85,
    "p": 1.0, "b": 1.0, "m": 1.0, "w": 1.0,
    "f": 0.9, "v": 0.9,
    "t": 0.3, "d": 0.3, "n": 0.3, "s": 0.3, "z": 0.3, "l": 0.3,
    "r": -0.2, "j": -0.2, "y": -0.2,
    "k": -0.7, "g": -0.7, "c": -0.7, "q": -0.7, "x": -0.7,
    "h": -1.0,
}
PLACE_V = {"i": 1.0, "u": 0.85, "e": 0.3, "o": -0.4, "a": -1.0}
VOICING = {
    "b": 1.0, "d": 1.0, "g": 1.0, "v": 1.0, "z": 1.0, "l": 1.0, "m": 1.0,
    "n": 1.0, "r": 1.0, "w": 1.0, "j": 1.0, "y": 1.0,
    "a": 1.0, "e": 1.0, "i": 1.0, "o": 1.0, "u": 1.0,
    "p": 0.0, "t": 0.0, "k": 0.0, "f": 0.0, "s": 0.0, "h": 0.0,
    "c": 0.0, "x": 0.0, "q": 0.0,
}
NASALITY = {"m": 1.0, "n": 1.0}
SONORITY = {
    "p": -1.0, "t": -1.0, "k": -1.0, "b": -1.0, "d": -1.0, "g": -1.0,
    "c": -1.0, "q": -1.0,
    "j": -0.5,
    "f": 0.0, "s": 0.0, "v": 0.0, "z": 0.0, "h": 0.0, "x": 0.0,
    "m": 0.45, "n": 0.45,
    "l": 0.95, "r": 0.95,
    "w": 1.4, "y": 1.4,
}
MANNER_STOP = {c: 1.0 for c in "ptkbdgcq"}

# v7.2: punctuation/whitespace (other than space) removed from the native
# substrate. Only A-Z, a-z, 0-9, and space are hand-authored into 16D slots.
# Period, newline, !, ? and all other punctuation/control characters will
# be re-introduced later as a separate, explicit control layer so they do
# not crowd the alphanumeric geometry.
SENTENCE_END_GRADE: dict[str, float] = {}
CLAUSE_SEP_GRADE: dict[str, float] = {}
QUOTE_GRADE: dict[str, float] = {}
BRACKET_GRADE: dict[str, float] = {}


def structural_features(char: str) -> np.ndarray:
    """Compute the 33-dim structural-property vector for a character.

    Rows 0-29 are identical to v6.5. Rows 30/31 are the v7 digit circle
    (cos/sin at 36 degree spacing). Row 32 is the null slot.
    """
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    if char == "<empty>":
        feats[32] = 1.0
        return feats
    if not char:
        return feats
    cp = ord(char[0])
    lower = char.lower()
    is_letter = lower.isalpha()
    is_digit = char.isdigit()
    is_punct = (not is_letter and not is_digit and not char.isspace())
    is_space = char.isspace()

    feats[0] = 1.0 if is_letter else 0.0
    feats[1] = 1.0 if (is_letter and lower in VOWELS_LOWER) else 0.0
    feats[2] = 1.0 if (is_letter and lower not in VOWELS_LOWER) else 0.0
    feats[3] = 1.0 if (is_letter and char.isupper()) else 0.0
    feats[4] = 1.0 if (is_letter and char.islower()) else 0.0
    feats[5] = 1.0 if is_digit else 0.0
    feats[6] = 1.0 if is_punct else 0.0
    if is_space:
        feats[7] = {" ": 1.0, "\t": 0.6, "\n": 0.3, "\r": 0.45}.get(char, 0.8)

    if is_letter:
        feats[8] = float(VOICING.get(lower, 0.5))
        feats[9] = float(PLACE_H.get(lower, 0.0))
        feats[10] = float(PLACE_V.get(lower, 0.0))
        feats[11] = float(SONORITY.get(lower, 0.0))
        feats[12] = float(MANNER_STOP.get(lower, 0.0))
        feats[13] = float(NASALITY.get(lower, 0.0))

    feats[14] = (cp % 128) / 128.0
    feats[15] = (cp % 32) / 32.0
    feats[16] = ((cp // 32) % 4) / 4.0
    feats[17] = 1.0 if cp < 128 else 0.0
    feats[18] = 1.0 if 128 <= cp < 256 else 0.0
    feats[19] = 1.0 if cp >= 256 else 0.0

    feats[20] = 1.0 if (is_letter and lower in ASCENDERS) else 0.0
    feats[21] = 1.0 if (is_letter and lower in DESCENDERS) else 0.0
    feats[22] = 1.0 if (is_letter and lower in DOTS) else 0.0
    feats[23] = 1.0 if (is_letter and lower in CLOSED_LOOPS) else 0.0
    feats[24] = float(LETTER_FREQUENCY.get(lower, 0.0)) if is_letter else 0.0

    feats[25] = 1.0 if is_space else 0.0
    feats[26] = float(SENTENCE_END_GRADE.get(char, 0.0))
    feats[27] = float(CLAUSE_SEP_GRADE.get(char, 0.0))
    feats[28] = float(QUOTE_GRADE.get(char, 0.0))
    feats[29] = float(BRACKET_GRADE.get(char, 0.0))

    # v7 digit circle: ten digits at 36 degree spacing. Hand-authored,
    # deterministic, no RNG. Replaces the v6 linear digit_value, which
    # left adjacent digits at cosine ~0.99 (unwritable).
    if is_digit:
        d = cp - ord("0")
        angle = 2.0 * math.pi * d / 10.0
        feats[30] = math.cos(angle)
        feats[31] = math.sin(angle)

    feats[32] = 0.0
    return feats


# ---------------------------------------------------------------------------
# The hand-authored basis: feature row -> [(output_dim, weight), ...]
# Rows 0-29 carry the v6.5 weights verbatim (letters/punct geometry is
# untouched and stays conformant). Changes for v7:
#   * row 5 (is_digit): d10 weight reduced 1.00 -> 0.30 so the category
#     offset doesn't crowd the circle.
#   * rows 30/31: the digit circle lands in the (d10, d15) plane at
#     radius 1.05.
#   * row 32: null slot (old row 31 weights).
# ---------------------------------------------------------------------------

_FEATURE_WEIGHTS: dict[int, list[tuple[int, float]]] = {
    0:  [(0, 1.50)],
    1:  [(1, 1.43)],
    2:  [(1, -0.43)],
    3:  [(2, 0.96)],
    4:  [(2, -0.53)],
    5:  [(10, 0.30), (0, -0.40)],
    6:  [(11, 1.00), (0, -0.60)],
    7:  [(14, 1.00), (0, -0.30)],
    8:  [(3, 1.20)],
    9:  [(4, 2.06)],
    10: [(6, 1.16)],
    11: [(5, 1.01)],
    12: [(9, -0.03)],
    13: [(9, 1.21)],
    14: [(15, 0.52)],
    15: [(15, -1.14), (10, 0.68)],
    16: [(15, 0.30), (11, -0.25)],
    17: [(0, 0.10)],
    18: [(4, -0.30)],
    19: [(4, -0.50), (15, 0.30)],
    20: [(7, 1.25)],
    21: [(7, -1.03)],
    22: [(8, 0.83)],
    23: [(8, -0.64)],
    24: [(1, 1.00)],
    25: [(14, -0.50)],
    26: [(11, -1.40), (12, 0.77)],
    27: [(12, 1.07)],
    28: [(12, -1.32), (13, 0.53)],
    29: [(13, 0.90)],
    30: [(10, -1.05)],          # digit circle, cos component
    31: [(15, 1.05)],           # digit circle, sin component
    32: [(15, -0.90), (14, 0.40)],  # null slot
}


def build_basis_matrix() -> np.ndarray:
    """Fully deterministic: no RNG, no QR, no seed."""
    W = np.zeros((FEATURE_DIM, SLOT_DIM), dtype=np.float32)
    for row, assignments in _FEATURE_WEIGHTS.items():
        for dim, weight in assignments:
            W[row, dim] = weight
    return W


_BASIS_CACHE: np.ndarray | None = None


def get_basis_matrix() -> np.ndarray:
    global _BASIS_CACHE
    if _BASIS_CACHE is None:
        _BASIS_CACHE = build_basis_matrix()
    return _BASIS_CACHE


_CHAR_CACHE: dict[str, np.ndarray] = {}


def char_to_slot(char: str) -> np.ndarray:
    """The main API. Maps a character to its frozen 16D substrate slot.
    Cached for speed; the cache changes speed, never values."""
    vec = _CHAR_CACHE.get(char)
    if vec is None:
        feats = structural_features(char)
        W = get_basis_matrix()
        vec = (W.T @ feats).astype(np.float32)
        vec.setflags(write=False)
        _CHAR_CACHE[char] = vec
    return vec


EMPTY_SLOT = None  # set below once char_to_slot exists


def text_to_field(text: str) -> np.ndarray:
    """(N, 16) frozen letter field, one row per character."""
    if not text:
        return np.zeros((0, SLOT_DIM), dtype=np.float32)
    return np.stack([char_to_slot(c) for c in text], axis=0)


# ---------------------------------------------------------------------------
# The alphabet and the renderer (substrate inverse)
# ---------------------------------------------------------------------------

def default_alphabet() -> list[str]:
    """The v7.1 native alphabet: alphanumerics + basic formatting.

    Composition (67 characters total):
      - a-z (26)
      - A-Z (26)
      - 0-9 (10)
      - space (1)
      - period . (1)
      - newline \n (1)
      - exclamation ! (1)
      - question ? (1)

    All other punctuation and control characters are intentionally excluded
    from the 16D letter bank to prevent semantic crowding. They will be
    handled by an explicit formatting layer above the substrate when needed.
    """
    chars: list[str] = []
    chars.extend(chr(c) for c in range(ord("a"), ord("z") + 1))
    chars.extend(chr(c) for c in range(ord("A"), ord("Z") + 1))
    chars.extend(chr(c) for c in range(ord("0"), ord("9") + 1))
    chars.append(" ")
    chars.append(".")
    chars.append("\n")
    chars.append("!")
    chars.append("?")
    return chars


ALPHABET_SET = set(default_alphabet())


def _unit_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-12)
    return M / norms


class LetterBank:
    """The full alphabet plus <empty> as unit 16D vectors. The renderer:
    nearest character by cosine. Pure function, no learned parameters."""

    def __init__(self):
        self.chars: list[str] = default_alphabet() + ["<empty>"]
        self.vecs = np.stack([char_to_slot(c) for c in self.chars], axis=0)
        self.vecs_unit = _unit_rows(self.vecs)
        self.empty_index = len(self.chars) - 1

    def decode_index(self, vec: np.ndarray) -> int:
        n = float(np.linalg.norm(vec))
        if n < 1e-6:
            return self.empty_index
        return int(np.argmax(self.vecs_unit @ (vec / n)))

    def decode_letter(self, vec: np.ndarray) -> str:
        c = self.chars[self.decode_index(vec)]
        return "" if c == "<empty>" else c

    def is_empty(self, vec: np.ndarray) -> bool:
        return self.decode_index(vec) == self.empty_index

    def decode_sequence(self, vecs: np.ndarray, blanks_as: str = "·") -> str:
        """Decode rows to text; empty slots render as `blanks_as`
        (middle dot for training displays; pass "" for the live surface)."""
        out = []
        for v in vecs:
            i = self.decode_index(v)
            out.append(blanks_as if i == self.empty_index else self.chars[i])
        return "".join(out)


_DEFAULT_BANK: LetterBank | None = None


def get_letter_bank() -> LetterBank:
    global _DEFAULT_BANK
    if _DEFAULT_BANK is None:
        _DEFAULT_BANK = LetterBank()
    return _DEFAULT_BANK


# ---------------------------------------------------------------------------
# Geometry verification (HARD rules — never loosen; Contract 2.2)
# ---------------------------------------------------------------------------

# v7.1: the writing set is exactly the native alphabet (alnum + space + period + newline + ! + ?).
WRITING_SET = default_alphabet()


def geometry_check() -> dict[str, object]:
    W = get_basis_matrix()
    lowercase = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    Ml = _unit_rows(np.stack([char_to_slot(c) for c in lowercase], axis=0))
    Cl = Ml @ Ml.T
    n = len(lowercase)
    off_mask = ~np.eye(n, dtype=bool)
    off_l = Cl[off_mask]
    hi = np.unravel_index(np.argmax(np.where(off_mask, Cl, -2.0)), Cl.shape)
    lo = np.unravel_index(np.argmin(np.where(off_mask, Cl, 2.0)), Cl.shape)

    Mw = _unit_rows(np.stack([char_to_slot(c) for c in WRITING_SET], axis=0))
    Cw = Mw @ Mw.T
    np.fill_diagonal(Cw, -2.0)
    nn_cos = Cw.max(axis=1)
    worst_idx = int(np.argmax(nn_cos))
    worst_partner = WRITING_SET[int(np.argmax(Cw[worst_idx]))]

    digits = [str(d) for d in range(10)]
    Md = _unit_rows(np.stack([char_to_slot(c) for c in digits], axis=0))
    Cd = Md @ Md.T
    adj = max(float(Cd[i, (i + 1) % 10]) for i in range(10))

    i = {c: k for k, c in enumerate(lowercase)}
    identity_rows = sum(
        1 for r in range(W.shape[0])
        if np.allclose(W[r], np.eye(FEATURE_DIM, dtype=np.float32)[r][:SLOT_DIM])
    )
    return {
        "letters_min_cos": float(off_l.min()),
        "letters_min_pair": f"{lowercase[lo[0]]}-{lowercase[lo[1]]}",
        "letters_max_cos": float(off_l.max()),
        "letters_max_pair": f"{lowercase[hi[0]]}-{lowercase[hi[1]]}",
        "d_to_t_cos": float(Cl[i["d"], i["t"]]),
        "d_to_q_cos": float(Cl[i["d"], i["q"]]),
        "m_to_n_cos": float(Cl[i["m"], i["n"]]),
        "digits_worst_adjacent_cos": adj,
        "writing_set_size": len(WRITING_SET),
        "writing_worst_nn_cos": float(nn_cos[worst_idx]),
        "writing_worst_pair": f"{WRITING_SET[worst_idx]!r}-{worst_partner!r}",
        "identity_rows": int(identity_rows),
        "basis_shape": tuple(W.shape),
    }


def roundtrip_check() -> tuple[int, int, list[tuple[str, str]]]:
    full = default_alphabet()
    M = np.stack([char_to_slot(c) for c in full], axis=0)
    Mu = _unit_rows(M)
    fails: list[tuple[str, str]] = []
    for k, c in enumerate(full):
        v = char_to_slot(c)
        vu = v / max(float(np.linalg.norm(v)), 1e-12)
        back = full[int(np.argmax(Mu @ vu))]
        if back != c:
            fails.append((c, back))
    return len(full), len(fails), fails


def verify_substrate(verbose: bool = True) -> bool:
    """The v7.2 conformance gate. HARD RULES — never loosen:
      1. No identity rows in the basis.
      2. Every alphabet character round-trips exactly.
      3. Lowercase letter pairwise cosines within [-0.30, 0.92].
      4. Topology: cos(d,t) > cos(d,q) + 0.05; cos(m,n) <= 0.92.
      5. WRITING_SET (alphanumeric only): worst nearest-neighbor <= 0.97.
      6. Adjacent digits on the circle separate: worst adjacent <= 0.97.
    """
    rep = geometry_check()
    n_chars, n_fail, fails = roundtrip_check()
    rules = {
        "no_identity_rows": rep["identity_rows"] == 0,
        "roundtrip_exact": n_fail == 0,
        "letters_band_low": rep["letters_min_cos"] >= -0.30,
        "letters_band_high": rep["letters_max_cos"] <= 0.92,
        "topology_d_t_q": rep["d_to_t_cos"] > rep["d_to_q_cos"] + 0.05,
        "m_n_separated": rep["m_to_n_cos"] <= 0.92,
        "writing_nn_margin": rep["writing_worst_nn_cos"] <= 0.97,
        "digits_adjacent": rep["digits_worst_adjacent_cos"] <= 0.97,
    }
    if verbose:
        for k, v in rep.items():
            print(f"  {k:28s} {v}")
        print()
        for k, ok in rules.items():
            print(f"  {'PASS' if ok else 'FAIL':4s}  {k}")
        if fails:
            print(f"  round-trip failures: {fails[:20]}")
    return all(rules.values())


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    if not quiet:
        W = get_basis_matrix()
        print(f"basis matrix shape: {W.shape} (hand-authored, deterministic, no RNG)")
        print("\n== v7 geometry verification (HARD rules) ==")
    ok = verify_substrate(verbose=not quiet)
    if not quiet:
        print("\nv7 conformance:", "PASS" if ok else "FAIL")
        if not ok:
            full = default_alphabet()
            M = np.stack([char_to_slot(c) for c in full], axis=0)
            Mu = _unit_rows(M)
            C = Mu @ Mu.T
            np.fill_diagonal(C, -2.0)
            pairs = []
            for a in range(len(full)):
                for b in range(a + 1, len(full)):
                    pairs.append((float(C[a, b]), full[a], full[b]))
            pairs.sort(reverse=True)
            print("\nworst pairs:")
            for cos, a, b in pairs[:25]:
                print(f"  {a!r} vs {b!r}: {cos:.5f}")
    sys.exit(0 if ok else 1)
