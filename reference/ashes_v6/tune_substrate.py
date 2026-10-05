"""Throwaway weight-search aid for the v6.4d basis (not part of the runtime).

Searches the per-feature weight magnitudes for a combination that satisfies
the HARD geometry rules in letter_substrate.verify_substrate. The winning
weights get rounded to 2 decimals and committed by hand into
letter_substrate._FEATURE_WEIGHTS. Deterministic (fixed seed).
"""
from __future__ import annotations

import numpy as np

import letter_substrate as ls


LOWER = [chr(c) for c in range(ord("a"), ord("z") + 1)]
WRITING_NO_DIGITS = (
    LOWER + [c.upper() for c in LOWER] + list(" .,!?;:'\"-()\n")
)
FULL = ls.default_alphabet()

# Feature matrix is fixed; only weights move.
F_LOWER = np.stack([ls.structural_features(c) for c in LOWER])
F_WRITE = np.stack([ls.structural_features(c) for c in WRITING_NO_DIGITS])
F_FULL = np.stack([ls.structural_features(c) for c in FULL])

I = {c: k for k, c in enumerate(LOWER)}


def build_W(p: dict[str, float]) -> np.ndarray:
    W = np.zeros((32, 16), dtype=np.float64)
    W[0, 0] = p["letter"]
    W[1, 1] = p["vowel"]
    W[2, 1] = -p["cons"]
    W[3, 2] = p["upper"]
    W[4, 2] = -p["low"]
    W[5, 10], W[5, 0] = 1.00, -0.40       # is_digit
    W[6, 11], W[6, 0] = 1.00, -0.60       # is_punct
    W[7, 14], W[7, 0] = 1.00, -0.30       # whitespace_grade
    W[8, 3] = p["voice"]
    W[9, 4] = p["place"]
    W[10, 6] = p["height"]
    W[11, 5] = p["sonority"]
    W[12, 9] = -p["stopflag"]
    W[13, 9] = p["nasal"]
    W[14, 15] = p["cpn"]
    W[15, 15], W[15, 10] = -p["cpl"], p["cpl"] * 0.6
    W[16, 15], W[16, 11] = 0.30, -0.25
    W[17, 0] = 0.10
    W[18, 4] = -0.30
    W[19, 4], W[19, 15] = -0.50, 0.30
    W[20, 7] = p["shape"]
    W[21, 7] = -p["desc"]
    W[22, 8] = p["dot"]
    W[23, 8] = -p["loop"]
    W[24, 1] = p["freq"]
    W[25, 14] = -0.50
    W[26, 11], W[26, 12] = -p["sent"], p["sent"] * 0.55   # sentence_end
    W[27, 12] = p["clause"]                                # clause_sep
    W[28, 12], W[28, 13] = -p["quote"], p["quote"] * 0.4   # quote
    W[29, 13] = 0.90                       # bracket (signed feature)
    W[30, 10], W[30, 15] = -1.30, 0.40    # digit_value
    W[31, 15], W[31, 14] = -0.90, 0.40    # null_slot
    return W


def cos_matrix(F: np.ndarray, W: np.ndarray) -> np.ndarray:
    M = F @ W
    n = np.linalg.norm(M, axis=1, keepdims=True).clip(min=1e-12)
    Mu = M / n
    return Mu @ Mu.T


def evaluate(p: dict[str, float]) -> tuple[float, dict[str, float]]:
    W = build_W(p)
    Cl = cos_matrix(F_LOWER, W)
    off = ~np.eye(len(LOWER), dtype=bool)
    lmin = Cl[off].min()
    lmax = Cl[off].max()
    dt, dq, mn = Cl[I["d"], I["t"]], Cl[I["d"], I["q"]], Cl[I["m"], I["n"]]

    Cw = cos_matrix(F_WRITE, W)
    np.fill_diagonal(Cw, -2.0)
    wmax = Cw.max()

    # Round-trip over the full alphabet.
    Cf = cos_matrix(F_FULL, W)
    np.fill_diagonal(Cf, -2.0)
    rt_ok = True
    for k in range(len(FULL)):
        # exact NN means no other char ties or exceeds self-sim 1.0
        if Cf[k].max() >= 1.0 - 1e-9:
            rt_ok = False
            break

    viol = (
        max(0.0, -0.28 - lmin)
        + max(0.0, lmax - 0.90)
        + max(0.0, mn - 0.90)
        + max(0.0, dq + 0.07 - dt)
        + max(0.0, wmax - 0.96)
        + (1.0 if not rt_ok else 0.0)
    )
    # Secondary: prefer wide margins
    score = viol * 100 + max(0.0, lmax - 0.82) + max(0.0, wmax - 0.93)
    return score, {"lmin": lmin, "lmax": lmax, "dt": dt, "dq": dq,
                   "mn": mn, "wmax": wmax, "rt": float(rt_ok)}


BOUNDS = {
    "letter": (1.4, 3.6), "vowel": (0.5, 1.6), "cons": (0.25, 1.2),
    "upper": (0.5, 1.3), "low": (0.2, 0.8), "voice": (0.8, 2.4),
    "place": (1.2, 3.4), "height": (0.6, 2.4), "sonority": (0.7, 2.2),
    "stopflag": (0.0, 0.5), "nasal": (0.4, 1.9), "shape": (0.5, 1.9),
    "desc": (0.4, 1.6),
    "dot": (0.3, 1.3), "loop": (0.2, 1.0), "freq": (0.05, 1.0),
    "cpn": (0.3, 1.0), "cpl": (0.3, 1.5),
    "sent": (0.5, 1.4), "clause": (0.5, 1.4), "quote": (0.5, 1.4),
}


def main() -> None:
    rng = np.random.default_rng(206)
    keys = list(BOUNDS)
    best_p, best_s, best_m = None, float("inf"), None
    for it in range(60000):
        p = {k: float(rng.uniform(*BOUNDS[k])) for k in keys}
        s, m = evaluate(p)
        if s < best_s:
            best_s, best_p, best_m = s, p, m
    # Local refine
    step = 0.05
    for _ in range(8):
        improved = False
        for k in keys:
            for d in (-step, step):
                q = dict(best_p)
                q[k] = float(np.clip(q[k] + d, *BOUNDS[k]))
                s, m = evaluate(q)
                if s < best_s:
                    best_s, best_p, best_m = s, q, m
                    improved = True
        if not improved:
            step /= 2
    # Round to 2 decimals and re-verify
    rounded = {k: round(v, 2) for k, v in best_p.items()}
    s, m = evaluate(rounded)
    print("best score (raw):", best_s)
    print("rounded score:", s)
    print("weights:", rounded)
    print("metrics:", {k: round(v, 4) for k, v in m.items()})


if __name__ == "__main__":
    main()
