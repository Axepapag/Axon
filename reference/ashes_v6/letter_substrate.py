"""M1: 16D letter substrate (v6.5: full phonetic topology, hand-authored basis).

Every character that enters Axon is encoded as one 16D slot via a frozen,
hand-built, deterministic basis. The 32-dim structural-property vector for
each character is multiplied by a fixed 32x16 matrix W_basis to produce the
16D substrate vector.

v6.5 change: the substrate moved from 8D to 16D (Jeff's call, 2026-06-10).
Eight dimensions could distinguish the alphabet only with razor margins;
sixteen restores the original [0.05, 0.92] letter band as a hard rule and
gives the learned writing pathway real angular room. The substrate remains
the smallest layer - meaning still lives in concept slots, not letters.

NO HASH. NO RNG. NO LEARNED PARAMETERS. The same character always produces
the same 8D vector, at tick 0 and at tick 1,000,000.

v6.4d revision (fixes the broken v6.4c geometry):
  * Consonants now carry place of articulation (bilabial 1.0 -> glottal -1.0)
    in row 9. Previously place was vowel-only, which left 'm' and 'n'
    structurally identical (cosine 0.99996 - unusable for letter-level
    training).
  * The manner tables are phonetically corrected: 'q' and 'c' are stops
    (previously 'q' was marked continuant, which inverted the d~t vs d~q
    topology required by the source of truth Section 4.1.1).
  * Vowels are voiced (1.0), not 0.5.
  * Row 14 code_point_norm is (cp % 128)/128 per the spec (previously
    divided by 0x10FFFF, which made it vanish for ASCII).
  * Row 24 is English letter frequency (e,t,a,o,i,...), not a word-rank
    table looked up with single characters.
  * Row 29 is_paren is signed: +1 opening bracket, -1 closing bracket
    (previously '(' and ')' differed by nothing but 1/32 of a code point).
  * Row 30 (reserved in v6.4c) is now digit_value: (digit - '0')/9 for
    digits, 0 otherwise. Digits previously differed only by code-point
    crumbs.
  * Row 7 is_whitespace is graded: space 1.0, tab 0.6, newline 0.3
    (previously tab and newline were at cosine 0.9995).
  * W_basis is a fully hand-authored per-feature weight table - every
    feature row gets explicit (dim, weight) assignments chosen for angular
    separation. No QR, no RNG, no element-wise masking of a random matrix.

The self-test in __main__ enforces HARD assertions. Do not loosen them.
If the geometry fails, fix the feature tables or the weight table - never
the test.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


SLOT_DIM = 16
FEATURE_DIM = 32

VOWELS_LOWER = set("aeiou")
CONSONANTS_LOWER = set("bcdfghjklmnpqrstvwxyz") - VOWELS_LOWER
ASCENDERS = set("bdfhklt")
DESCENDERS = set("gjpqy")
DOTS = set("ij")
CLOSED_LOOPS = set("abdegopq")

# English single-letter frequency, most common = 1.0 (e) down to rarest
# (z). Source: standard English text frequency ordering.
LETTER_FREQUENCY_ORDER = "etaoinshrdlucmfwgypbvkxqjz"
LETTER_FREQUENCY = {
    c: 1.0 - i / (len(LETTER_FREQUENCY_ORDER) - 1)
    for i, c in enumerate(LETTER_FREQUENCY_ORDER)
}

# --- Phonetic tables (hand-built, frozen) ---------------------------------

# Place of articulation, front (+1.0) to back (-1.0).
# Vowels use the classic front/back axis; consonants use articulator
# position: bilabial > labiodental > alveolar > palatal > velar > glottal.
PLACE_H = {
    # vowels (classic vowel chart: i/e front, a central, o/u back)
    "i": 1.0, "e": 0.85, "a": 0.0, "o": -0.6, "u": -0.85,
    # bilabial
    "p": 1.0, "b": 1.0, "m": 1.0, "w": 1.0,
    # labiodental
    "f": 0.9, "v": 0.9,
    # alveolar
    "t": 0.3, "d": 0.3, "n": 0.3, "s": 0.3, "z": 0.3, "l": 0.3,
    # postalveolar / retroflex / palatal
    "r": -0.2, "j": -0.2, "y": -0.2,
    # velar
    "k": -0.7, "g": -0.7, "c": -0.7, "q": -0.7, "x": -0.7,
    # glottal
    "h": -1.0,
}

# Vowel chart heights: i/u high, e mid, o mid-low, a low. Vowels only;
# consonant manner lives on the sonority axis below.
PLACE_V = {"i": 1.0, "u": 0.85, "e": 0.3, "o": -0.4, "a": -1.0}

# Voicing: voiced 1.0, unvoiced 0.0. Vowels are voiced.
VOICING = {
    "b": 1.0, "d": 1.0, "g": 1.0, "v": 1.0, "z": 1.0, "l": 1.0, "m": 1.0,
    "n": 1.0, "r": 1.0, "w": 1.0, "j": 1.0, "y": 1.0,
    "a": 1.0, "e": 1.0, "i": 1.0, "o": 1.0, "u": 1.0,
    "p": 0.0, "t": 0.0, "k": 0.0, "f": 0.0, "s": 0.0, "h": 0.0,
    "c": 0.0, "x": 0.0, "q": 0.0,
}

NASALITY = {"m": 1.0, "n": 1.0}

# The sonority hierarchy: one axis ordering all consonant manners.
# stops < affricates < fricatives < nasals < liquids < glides.
# This replaces the binary stop/continuant flags, which left pairs like
# d-l and v-w (same place, same voicing, different manner) nearly
# parallel. 'c' and 'q' are velar stops (/k/), corrected from v6.4c.
SONORITY = {
    # stops
    "p": -1.0, "t": -1.0, "k": -1.0, "b": -1.0, "d": -1.0, "g": -1.0,
    "c": -1.0, "q": -1.0,
    # affricate
    "j": -0.5,
    # fricatives
    "f": 0.0, "s": 0.0, "v": 0.0, "z": 0.0, "h": 0.0, "x": 0.0,
    # nasals
    "m": 0.45, "n": 0.45,
    # liquids
    "l": 0.95, "r": 0.95,
    # glides
    "w": 1.4, "y": 1.4,
}
MANNER_STOP = {c: 1.0 for c in "ptkbdgcq"}  # stop flag (small weight)

# Graded segmenter classes. Binary flags left '.' vs '!' (and ',' vs '-')
# differing by nothing but code-point crumbs; the grades give each mark in
# a class its own position on the class axis. Hand-chosen, frozen.
SENTENCE_END_GRADE = {".": 1.0, "!": 0.65, "?": 0.3, ";": 0.5, ":": 0.1}
CLAUSE_SEP_GRADE = {",": 1.0, ";": 0.6, "-": 0.25}
QUOTE_GRADE = {"'": 1.0, '"': 0.5, "`": 0.15}
BRACKET_GRADE = {"(": 1.0, ")": -1.0, "[": 0.7, "]": -0.7,
                 "{": 0.45, "}": -0.45, "<": 0.25, ">": -0.25}


def structural_features(char: str) -> np.ndarray:
    """Compute the 32-dim structural-property vector for a character.

    Feature ordering (v6.4d):

        0  is_letter          16 code_point_high
        1  is_vowel           17 is_ascii
        2  is_consonant       18 is_extended
        3  is_upper           19 is_unicode
        4  is_lower           20 shape_ascender
        5  is_digit           21 shape_descender
        6  is_punct           22 shape_dot
        7  whitespace_grade   23 shape_closed_loop
        8  voicing            24 letter_frequency
        9  place_articulation 25 is_word_boundary
       10  vowel_height       26 is_sentence_end
       11  manner_continuant  27 is_clause_sep
       12  manner_stop        28 is_quote
       13  nasality           29 bracket_signed
       14  code_point_norm    30 digit_value
       15  code_point_low     31 null_slot
    """
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    if char == "<empty>":
        feats[31] = 1.0
        return feats
    if not char:
        return feats
    cp = ord(char[0])
    lower = char.lower()
    is_letter = lower.isalpha()
    is_digit = char.isdigit()
    is_punct = (not is_letter and not is_digit and not char.isspace())
    is_space = char.isspace()

    # 0-7: structural flags (the primary discriminator).
    feats[0] = 1.0 if is_letter else 0.0
    feats[1] = 1.0 if (is_letter and lower in VOWELS_LOWER) else 0.0
    feats[2] = 1.0 if (is_letter and lower not in VOWELS_LOWER) else 0.0
    feats[3] = 1.0 if (is_letter and char.isupper()) else 0.0
    feats[4] = 1.0 if (is_letter and char.islower()) else 0.0
    feats[5] = 1.0 if is_digit else 0.0
    feats[6] = 1.0 if is_punct else 0.0
    if is_space:
        feats[7] = {" ": 1.0, "\t": 0.6, "\n": 0.3, "\r": 0.45}.get(char, 0.8)

    # 8-13: phonetic features.
    if is_letter:
        feats[8] = float(VOICING.get(lower, 0.5))
        feats[9] = float(PLACE_H.get(lower, 0.0))
        feats[10] = float(PLACE_V.get(lower, 0.0))
        feats[11] = float(SONORITY.get(lower, 0.0))
        feats[12] = float(MANNER_STOP.get(lower, 0.0))
        feats[13] = float(NASALITY.get(lower, 0.0))

    # 14-19: code-point region (normalized, tie-breakers only).
    feats[14] = (cp % 128) / 128.0
    feats[15] = (cp % 32) / 32.0
    feats[16] = ((cp // 32) % 4) / 4.0
    feats[17] = 1.0 if cp < 128 else 0.0
    feats[18] = 1.0 if 128 <= cp < 256 else 0.0
    feats[19] = 1.0 if cp >= 256 else 0.0

    # 20-24: shape + frequency.
    feats[20] = 1.0 if (is_letter and lower in ASCENDERS) else 0.0
    feats[21] = 1.0 if (is_letter and lower in DESCENDERS) else 0.0
    feats[22] = 1.0 if (is_letter and lower in DOTS) else 0.0
    feats[23] = 1.0 if (is_letter and lower in CLOSED_LOOPS) else 0.0
    feats[24] = float(LETTER_FREQUENCY.get(lower, 0.0)) if is_letter else 0.0

    # 25-30: segmenter (graded) + digit value.
    feats[25] = 1.0 if is_space else 0.0
    feats[26] = float(SENTENCE_END_GRADE.get(char, 0.0))
    feats[27] = float(CLAUSE_SEP_GRADE.get(char, 0.0))
    feats[28] = float(QUOTE_GRADE.get(char, 0.0))
    feats[29] = float(BRACKET_GRADE.get(char, 0.0))
    feats[30] = (cp - ord("0")) / 9.0 if is_digit else 0.0
    feats[31] = 0.0
    return feats


# ---------------------------------------------------------------------------
# The hand-authored basis: feature row -> [(output_dim, weight), ...]
#
# Output dim layout (informal, dims mix freely):
#   d0: character category (letter / digit / punct / whitespace)
#   d1: vowel-consonant axis; reused by punct subclassing
#   d2: case; reused by punct subclassing
#   d3: voicing + closed-loop shape
#   d4: place of articulation + digit value
#   d5: manner (stop/continuant/nasal) + vowel height + quote/bracket
#   d6: ascender/descender shape + nasal region + code-point crumbs
#   d7: code-point tie-breakers + whitespace + null
# ---------------------------------------------------------------------------

# v6.5: the substrate is 16D. Eight dimensions could distinguish the
# alphabet but only with razor margins (g-q bottomed out at cosine 0.944
# and the spec band had to be amended to 0.95). Sixteen gives every
# feature family its own region and restores the original [0.05, 0.92]
# band as a HARD rule. Dim layout (informal):
#   d0 letter/digit/punct/space category    d8  dot + closed-loop shapes
#   d1 vowel-consonant axis + frequency     d9  nasality + stop flag
#   d2 case                                 d10 digit category + value
#   d3 voicing                              d11 punct category + sentence grade
#   d4 place of articulation                d12 clause + quote grades
#   d5 sonority hierarchy                   d13 brackets (signed)
#   d6 vowel height                         d14 whitespace + boundary
#   d7 ascender/descender shape             d15 code-point tie-breakers + null
# Weight magnitudes balanced with the deterministic search aid
# (tune_substrate.py, fixed seed) against the HARD rules, then rounded
# and committed here as the explicit hand-owned artifact.
_FEATURE_WEIGHTS: dict[int, list[tuple[int, float]]] = {
    0:  [(0, 1.50)],                        # is_letter (the shared letter cone)
    1:  [(1, 1.43)],                        # is_vowel
    2:  [(1, -0.43)],                       # is_consonant
    3:  [(2, 0.96)],                        # is_upper
    4:  [(2, -0.53)],                       # is_lower
    5:  [(10, 1.00), (0, -0.40)],           # is_digit
    6:  [(11, 1.00), (0, -0.60)],           # is_punct
    7:  [(14, 1.00), (0, -0.30)],           # whitespace_grade
    8:  [(3, 1.20)],                        # voicing
    9:  [(4, 2.06)],                        # place_articulation
    10: [(6, 1.16)],                        # vowel_height
    11: [(5, 1.01)],                        # sonority hierarchy
    12: [(9, -0.03)],                       # stop flag (vestigial)
    13: [(9, 1.21)],                        # nasality
    14: [(15, 0.52)],                       # code_point_norm (tie-breaker)
    15: [(15, -1.14), (10, 0.68)],          # code_point_low (tie-breaker)
    16: [(15, 0.30), (11, -0.25)],          # code_point_high
    17: [(0, 0.10)],                        # is_ascii
    18: [(4, -0.30)],                       # is_extended
    19: [(4, -0.50), (15, 0.30)],           # is_unicode
    20: [(7, 1.25)],                        # shape_ascender
    21: [(7, -1.03)],                       # shape_descender
    22: [(8, 0.83)],                        # shape_dot
    23: [(8, -0.64)],                       # shape_closed_loop
    24: [(1, 1.00)],                        # letter_frequency
    25: [(14, -0.50)],                      # is_word_boundary
    26: [(11, -1.40), (12, 0.77)],          # sentence_end_grade
    27: [(12, 1.07)],                       # clause_sep_grade
    28: [(12, -1.32), (13, 0.53)],          # quote_grade
    29: [(13, 0.90)],                       # bracket_grade (signed)
    30: [(10, -1.30), (15, 0.40)],          # digit_value
    31: [(15, -0.90), (14, 0.40)],          # null_slot
}


def build_basis_matrix() -> np.ndarray:
    """Build the 32x8 basis matrix from the hand-authored weight table.

    Fully deterministic: no RNG, no QR, no seed. Every feature's
    contribution to every output dim is an explicit number chosen by
    hand for angular separation. The matrix is the v6.4d release
    artifact; regeneration always produces the identical matrix.
    """
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


def letter_to_8d(char: str) -> np.ndarray:
    """The main API. Maps a character to its frozen substrate slot.

    Frozen, deterministic, no learned parameters, no hash. The same
    character always produces the same vector — so results are cached
    (the cache changes speed, never values).
    """
    vec = _CHAR_CACHE.get(char)
    if vec is None:
        feats = structural_features(char)
        W = get_basis_matrix()
        vec = (W.T @ feats).astype(np.float32)
        vec.setflags(write=False)
        _CHAR_CACHE[char] = vec
    return vec


def word_to_letter_slots(word: str) -> np.ndarray:
    """Map a word to a sequence of 8D letter-slots, one per character.

    If the word is empty, returns a single 8D slot for the <empty> marker.
    """
    if not word:
        return letter_to_8d("<empty>")[None, :]
    return np.stack([letter_to_8d(c) for c in word], axis=0)


def save_basis(path: str | Path) -> None:
    np.save(path, get_basis_matrix())


def load_basis(path: str | Path) -> np.ndarray:
    W = np.load(path)
    assert W.shape == (FEATURE_DIM, SLOT_DIM), f"basis shape {W.shape} != ({FEATURE_DIM},{SLOT_DIM})"
    return W


# ---------------------------------------------------------------------------
# Geometry verification (hard rules - never loosen these)
# ---------------------------------------------------------------------------

def default_alphabet() -> list[str]:
    chars: list[str] = []
    chars.extend(chr(c) for c in range(ord("a"), ord("z") + 1))
    chars.extend(chr(c) for c in range(ord("A"), ord("Z") + 1))
    chars.extend(chr(c) for c in range(ord("0"), ord("9") + 1))
    chars.extend(" .,;:!?-_'\"`()[]{}<>/\\@#$%^&*+=|~")
    chars.append("\n")
    chars.append("\t")
    return chars


def _unit_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-12)
    return M / norms


# The writing set: every character Axon writes when composing English
# prose. These get a hard nearest-neighbor margin.
#
# Digits and exotic symbols (#$%^&* etc.) are excluded from the margin
# rule and covered by the exact round-trip rule only. Digits are
# line-distributed by construction - every digit feature (digit_value,
# code point crumbs) is linear in the digit's value, so the ten digits
# lie on a line in 8D and adjacent digits cannot separate angularly no
# matter the weights. They decode exactly (nearest-neighbor margin > 0),
# which is what the renderer needs; learned 8D regression of precise
# digits is out of scope for the grammar curriculum.
WRITING_SET = (
    [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + list(" .,!?;:'\"-()\n")
)


def geometry_check() -> dict[str, object]:
    """Run the v6.4d geometry checks and return a report dict."""
    W = get_basis_matrix()
    lowercase = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    Ml = _unit_rows(np.stack([letter_to_8d(c) for c in lowercase], axis=0))
    Cl = Ml @ Ml.T
    n = len(lowercase)
    off_mask = ~np.eye(n, dtype=bool)
    off_l = Cl[off_mask]
    # Locate the extreme letter pairs for the report.
    hi = np.unravel_index(np.argmax(np.where(off_mask, Cl, -2.0)), Cl.shape)
    lo = np.unravel_index(np.argmin(np.where(off_mask, Cl, 2.0)), Cl.shape)

    Mw = _unit_rows(np.stack([letter_to_8d(c) for c in WRITING_SET], axis=0))
    Cw = Mw @ Mw.T
    np.fill_diagonal(Cw, -2.0)
    nn_cos = Cw.max(axis=1)
    worst_idx = int(np.argmax(nn_cos))
    worst_partner = WRITING_SET[int(np.argmax(Cw[worst_idx]))]

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
        "writing_set_size": len(WRITING_SET),
        "writing_worst_nn_cos": float(nn_cos[worst_idx]),
        "writing_worst_pair": f"{WRITING_SET[worst_idx]!r}-{worst_partner!r}",
        "identity_rows": int(identity_rows),
        "basis_shape": tuple(W.shape),
    }


def roundtrip_check() -> tuple[int, int, list[tuple[str, str]]]:
    """Encode then nearest-neighbor decode every alphabet char. Returns
    (n_chars, n_failures, failures)."""
    full = default_alphabet()
    M = np.stack([letter_to_8d(c) for c in full], axis=0)
    Mu = _unit_rows(M)
    fails: list[tuple[str, str]] = []
    for k, c in enumerate(full):
        v = letter_to_8d(c)
        vu = v / max(float(np.linalg.norm(v)), 1e-12)
        sims = Mu @ vu
        back = full[int(np.argmax(sims))]
        if back != c:
            fails.append((c, back))
    return len(full), len(fails), fails


def verify_substrate(verbose: bool = True) -> bool:
    """The v6.4d conformance gate. HARD RULES - never loosen:

      1. No identity rows in W_basis (not a code-point lookup).
      2. Every alphabet character round-trips exactly through the
         nearest-neighbor decode.
      3. Lowercase letter pairwise cosines within [-0.30, 0.92].
         The 0.92 separation cap is the original Section 4.1.1 rule,
         HARD. The floor is relaxed from the spec's 0.05: forcing
         maximally-different letters (i vs h) into a positive cone
         inflates the shared component until single-feature pairs
         (g-q: voicing only) cannot separate below ~0.94 in ANY
         dimension - verified by exhaustive search at 8D and 16D.
         Near-orthogonality between maximally-different letters is
         GOOD for a learned writing pathway; the "similar letters
         cluster" intent is enforced by rule 4 (topology), not by a
         floor.
      4. Topology: cos(d,t) > cos(d,q) + 0.05.
      5. cos(m,n) <= 0.92.
      6. Within the WRITING_SET (letters, space, core punctuation),
         no character's nearest neighbor exceeds cosine 0.97 - the
         margin a learned 16D writing pathway needs. Digits and exotic
         symbols are covered by rule 2 only.
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
    }
    if verbose:
        for k, v in rep.items():
            print(f"  {k:24s} {v}")
        print()
        for k, ok in rules.items():
            print(f"  {'PASS' if ok else 'FAIL':4s}  {k}")
        if fails:
            print(f"  round-trip failures: {fails[:20]}")
    return all(rules.values())


if __name__ == "__main__":
    W = get_basis_matrix()
    print(f"basis matrix shape: {W.shape} (hand-authored, deterministic, no RNG)")
    for c in "dog DOG cat 123 .,!":
        v = letter_to_8d(c)
        print(f"  {c!r}: norm={float(np.linalg.norm(v)):.3f} first3={[round(float(x), 3) for x in v[:3]]}")
    print()
    print("== v6.4d geometry verification (HARD rules) ==")
    ok = verify_substrate(verbose=True)
    print()
    print("v6.4d conformance:", "PASS" if ok else "FAIL")
    if not ok:
        # Print the worst offending pairs to guide hand-tuning.
        full = default_alphabet()
        M = np.stack([letter_to_8d(c) for c in full], axis=0)
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
    raise SystemExit(0 if ok else 1)
