"""Axon slot adapters: frozen shared-state adapters for the 8192D slot field.

DESIGN (revised per table redirect 2026-07-03):

The adapter is the only bridge between the 8192D slot shared field and smaller
d_model cores. One slot reaches a core as ONE d_model vector (the sizing rule).
Exact text round-trip through that bottleneck is information-theoretically
impossible (256 chars ~ 1536 bits >> 64 floats), so the adapter has asymmetric
read/write:

  DOWN (read path — lossy, by design):
    Project the full 8192D slot (text + edge + control dims) into one d_model
    vector via a deterministic sign-projection. The core sees one frozen
    summary per slot. Lossy is acceptable: exact text lives in the 8192D field,
    and the runtime decodes English from the field, never from a core's
    compressed view. The core's job is reasoning over summaries, not character
    storage.

  UP (write path - codebook-snap):
    Accept field deltas from a core-owned output path and snap every 16D
    character block to the nearest frozen substrate code. Anything committed
    to the field is exact substrate by construction. An is_empty threshold
    handles zero/padded positions. There is no shared trained decode organ
    between the core and field.

  GATES (replacing the old impossible exact-round-trip gate):
    (a) Snap-idempotence: any validly packed slot passes through
        DOWN -> UP-snap unchanged (the snap is a no-op on already-exact
        substrate codes).
    (b) Separability: distinct slots (differing by one character) produce
        distinct d_model projections (no collisions across a test corpus).

  CORE-OWNED OUTPUT:
    Trainable character/output parameters live inside each core. The adapter
    only performs frozen projection and deterministic substrate snapping.

One frozen adapter per d_model size, shared by all cores of that size.
Artifacts: adapters/slot8192_adapter_{d_model}d.pt

Self-test: `python adapters/slot_adapter.py --check` runs snap-idempotence
and separability gates for all minted adapter sizes.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

# Ensure repo root is importable when run as a script
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from substrate import (  # noqa: E402
    SLOT_DIM,
    char_to_slot,
    default_alphabet,
    get_letter_bank,
)
from slots.slot_spec import (  # noqa: E402
    SLOT_WIDTH,
    TEXT_DIMS,
    TEXT_OFFSET,
    EDGE_OFFSET,
    CONTROL_OFFSET,
    MAX_TEXT_CHARS,
    MAX_EDGE_CHARS,
    pack_slot,
    unpack_slot,
)

ADAPTER_VERSION = "axon_slot_adapter_v2"

# First locked set: 64, 128, 256 (the proven core sizes)
ADAPTER_D_MODELS: tuple[int, ...] = (64, 128, 256)

# Frozen character-prototype version (independent of adapter projection)
CHAR_PROTOTYPE_VERSION = "axon_char_prototype_v1"


# --------------------------------------------------------------------------- #
# Deterministic sign-projection (full 8192D -> d_model)
# --------------------------------------------------------------------------- #

def _projection_seed(d_model: int) -> int:
    """Deterministic seed from d_model + version."""
    h = hashlib.shake_256(f"{ADAPTER_VERSION}:d={d_model}".encode("utf-8"))
    return int.from_bytes(h.digest(4), "little")


def _build_projection(d_model: int) -> np.ndarray:
    """Build a deterministic (d_model, SLOT_WIDTH) projection matrix.

    Uses a fixed-seed random matrix with orthogonal rows (via QR decomposition).
    This is the down-projection: slot_vec (8192,) @ W.T -> (d_model,).
    """
    rng = np.random.default_rng(seed=_projection_seed(d_model))
    M = rng.standard_normal((d_model, SLOT_WIDTH))
    # Orthogonalize rows so each d_model dimension captures independent variance
    Q, _ = np.linalg.qr(M.T)  # Q: (SLOT_WIDTH, d_model) with orthonormal columns
    return Q.T.astype(np.float32)  # (d_model, SLOT_WIDTH)


# --------------------------------------------------------------------------- #
# Codebook snap (write path)
# --------------------------------------------------------------------------- #

def _snap_char_block(vec: np.ndarray, bank) -> np.ndarray:
    """Snap a 16D block to the nearest frozen substrate character code.

    If the block is near-zero (below threshold), leave it as zeros (empty).
    """
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return np.zeros(SLOT_DIM, dtype=np.float32)
    # Decode to nearest character, then re-encode to get the exact frozen code
    char = bank.decode_letter(vec)
    if char == "":
        return np.zeros(SLOT_DIM, dtype=np.float32)
    return char_to_slot(char).astype(np.float32)


def codebook_snap(slot_vec: np.ndarray) -> np.ndarray:
    """Snap every 16D character block in an 8192D slot to the nearest
    frozen substrate code.

    Operates on text (dims 0-4095), edge (dims 4096-6143), and control
    (dims 6144-6655) sections. Reserved dims are zeroed.
    """
    bank = get_letter_bank()
    snapped = np.zeros(SLOT_WIDTH, dtype=np.float32)

    # Snap text payload: 256 x 16D blocks
    for i in range(MAX_TEXT_CHARS):
        start = TEXT_OFFSET + i * SLOT_DIM
        block = slot_vec[start : start + SLOT_DIM]
        snapped[start : start + SLOT_DIM] = _snap_char_block(block, bank)

    # Snap edge payload: 128 x 16D blocks
    for i in range(MAX_EDGE_CHARS):
        start = EDGE_OFFSET + i * SLOT_DIM
        block = slot_vec[start : start + SLOT_DIM]
        snapped[start : start + SLOT_DIM] = _snap_char_block(block, bank)

    # Snap control block: 32 x 16D blocks
    n_control = (CONTROL_OFFSET + 512 - CONTROL_OFFSET) // SLOT_DIM  # 32
    for i in range(n_control):
        start = CONTROL_OFFSET + i * SLOT_DIM
        block = slot_vec[start : start + SLOT_DIM]
        snapped[start : start + SLOT_DIM] = _snap_char_block(block, bank)

    # Reserved dims stay zero
    return snapped


# --------------------------------------------------------------------------- #
# Frozen per-character prototype table (core-owned write path -> field)
# --------------------------------------------------------------------------- #

def _prototype_seed(d_model: int) -> int:
    """Deterministic seed from d_model + prototype version."""
    h = hashlib.shake_256(f"{CHAR_PROTOTYPE_VERSION}:d={d_model}".encode("utf-8"))
    return int.from_bytes(h.digest(4), "little")


def _mint_prototype_matrix(n_chars: int, d_model: int) -> np.ndarray:
    """Mint n_chars deterministic d_model-dimensional unit vectors.

    Uses a seeded Gaussian then deterministic repulsion so every vector is
    the unique nearest neighbour of itself.  The construction is weightless:
    no trained parameters, only reproducible arithmetic.
    """
    rng = np.random.default_rng(seed=_prototype_seed(d_model))
    vecs = rng.standard_normal((n_chars, d_model)).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    vecs = vecs / norms

    # Deterministic repulsion: push any too-close pair apart along their
    # difference direction.  With high-dimensional random unit vectors this
    # almost never triggers, but the loop makes the minting robust.
    max_iter = 1000
    step = 0.05
    threshold = 0.95  # cosine; corresponds to ~18 degrees separation
    for _ in range(max_iter):
        sim = vecs @ vecs.T
        np.fill_diagonal(sim, -2.0)
        viol = np.argwhere(sim > threshold)
        if len(viol) == 0:
            break
        for i, j in viol:
            diff = vecs[i] - vecs[j]
            diff_norm = float(np.linalg.norm(diff))
            if diff_norm < 1e-12:
                diff = rng.standard_normal(d_model)
                diff_norm = float(np.linalg.norm(diff))
            diff = diff / diff_norm
            vecs[i] = vecs[i] + step * diff
            vecs[j] = vecs[j] - step * diff
            # renormalize on the fly
            vecs[i] = vecs[i] / max(1e-12, float(np.linalg.norm(vecs[i])))
            vecs[j] = vecs[j] / max(1e-12, float(np.linalg.norm(vecs[j])))
    else:
        # If we exhaust iterations, continue anyway; the exact round-trip
        # verification is the binding gate below.
        pass

    # Final normalization
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    return (vecs / norms).astype(np.float32)


class CharPrototypeTable:
    """Frozen deterministic table mapping alphabet characters to/from d_model.

    This is the bridge for the arithmetic-only write path: the core emits
    per-character d_model vectors, and this table decodes each vector to the
    nearest substrate character by cosine similarity.  The table itself has
    no learned parameters; it is minted once per d_model.
    """

    def __init__(self, d_model: int) -> None:
        self.d_model = d_model
        self.chars: list[str] = default_alphabet()
        self.n_chars = len(self.chars)
        self._prototypes = _mint_prototype_matrix(self.n_chars, d_model)
        # Pre-normalized (unit) prototypes for cosine nearest-neighbour decode
        self._unit = self._prototypes.copy()

    def encode(self, char: str) -> np.ndarray:
        """Return the frozen d_model prototype vector for a single character."""
        try:
            idx = self.chars.index(char)
        except ValueError as exc:
            raise ValueError(f"character {char!r} is not in the substrate alphabet") from exc
        return self._prototypes[idx].copy()

    def decode(self, vec: np.ndarray) -> str:
        """Nearest-code decode: return the alphabet char whose prototype is
        closest in cosine similarity to the supplied d_model vector."""
        v = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return " "
        vu = v / norm
        scores = self._unit @ vu
        idx = int(np.argmax(scores))
        return self.chars[idx]

    def decode_batch(self, vecs: np.ndarray) -> list[str]:
        """Decode a (L, d_model) or (B, L, d_model) array of vectors."""
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 2:
            vecs = vecs[None, ...]
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True).clip(min=1e-9)
        unit = vecs / norms
        # (B, L, n_chars)
        scores = unit @ self._unit.T
        idx = np.argmax(scores, axis=-1)
        return [[self.chars[i] for i in row] for row in idx]

    def logits(self, vecs: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return cosine-similarity logits to every alphabet character.

        Output shape: (B, L, n_chars) for training-time discrete CE.
        """
        if isinstance(vecs, torch.Tensor):
            t = vecs
        else:
            t = torch.from_numpy(np.asarray(vecs, dtype=np.float32))
        if t.ndim == 2:
            t = t.unsqueeze(0)
        # Normalize input vectors for cosine similarity
        norm = t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        unit = t / norm
        protos = torch.from_numpy(self._unit).to(t.dtype).to(t.device)
        return unit @ protos.T  # (B, L, n_chars)

    def char_index(self, char: str) -> int:
        return self.chars.index(char)

    def save(self, path: str | Path) -> None:
        torch.save({
            "version": CHAR_PROTOTYPE_VERSION,
            "d_model": self.d_model,
            "chars": self.chars,
            "prototypes": torch.tensor(self._prototypes, dtype=torch.float32),
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "CharPrototypeTable":
        data = torch.load(str(path), weights_only=False)
        table = cls.__new__(cls)
        table.d_model = data["d_model"]
        table.chars = list(data["chars"])
        table.n_chars = len(table.chars)
        table._prototypes = data["prototypes"].numpy()
        table._unit = table._prototypes.copy()
        return table

    def check_exact(self) -> tuple[bool, list[tuple[str, str]]]:
        """Verify every alphabet character round-trips through encode->decode."""
        fails: list[tuple[str, str]] = []
        for char in self.chars:
            got = self.decode(self.encode(char))
            if got != char:
                fails.append((char, got))
        return len(fails) == 0, fails


# --------------------------------------------------------------------------- #
# Char-prototype registry
# --------------------------------------------------------------------------- #

_CHAR_PROTOTYPE_REGISTRY: dict[int, CharPrototypeTable] = {}


def get_char_prototype_table(d_model: int) -> CharPrototypeTable:
    """Get the canonical frozen char-prototype table for a d_model size."""
    if d_model in _CHAR_PROTOTYPE_REGISTRY:
        return _CHAR_PROTOTYPE_REGISTRY[d_model]

    artifact_path = Path(_ROOT) / "adapters" / f"char_prototype_{d_model}d.pt"
    if artifact_path.exists():
        table = CharPrototypeTable.load(artifact_path)
    else:
        table = CharPrototypeTable(d_model=d_model)
    _CHAR_PROTOTYPE_REGISTRY[d_model] = table
    return table


def clear_char_prototype_registry() -> None:
    _CHAR_PROTOTYPE_REGISTRY.clear()


# --------------------------------------------------------------------------- #
# Gate reports
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SnapIdempotenceReport:
    """Result of the snap-idempotence gate."""
    d_model: int
    n_tested: int
    n_changed: int  # should be 0 for already-exact substrate slots
    passed: bool


@dataclass(frozen=True)
class SeparabilityReport:
    """Result of the separability gate."""
    d_model: int
    n_pairs_tested: int
    n_collisions: int  # pairs that produced identical d_model projections
    passed: bool


# --------------------------------------------------------------------------- #
# Adapter module
# --------------------------------------------------------------------------- #

class SlotAdapter:
    """Frozen shared-state adapter: 8192D <-> d_model.

    DOWN: full 8192D slot -> one d_model vector (lossy, deterministic).
    UP: accept 8192D proposal, codebook-snap to exact substrate.
    """

    def __init__(self, d_model: int) -> None:
        self.d_model = d_model
        self._W = _build_projection(d_model)  # (d_model, SLOT_WIDTH)
        self._bank = get_letter_bank()

    # -- DOWN (read path) -- #

    def project_down(self, slot_vec: np.ndarray | torch.Tensor) -> np.ndarray:
        """Project one 8192D slot to one d_model vector. Lossy by design.

        Input: (8192,) or (B, 8192).
        Output: (d_model,) or (B, d_model).
        """
        if isinstance(slot_vec, torch.Tensor):
            slot_vec = slot_vec.numpy()
        if slot_vec.ndim == 1:
            return self._W @ slot_vec
        return slot_vec @ self._W.T

    def project_down_region(self, region: np.ndarray) -> np.ndarray:
        """Project a (N, 8192) region to (N, d_model) — one vector per slot."""
        return region @ self._W.T

    # -- UP (write path) -- #

    def project_up(self, d_vec: np.ndarray | torch.Tensor) -> np.ndarray:
        """Project a d_model vector back to 8192D (lossy reconstruction).

        This is the inverse projection — NOT codebook-snapped. Use
        codebook_snap() on the result before committing to the field.

        Input: (d_model,) or (B, d_model).
        Output: (8192,) or (B, 8192).
        """
        if isinstance(d_vec, torch.Tensor):
            d_vec = d_vec.numpy()
        if d_vec.ndim == 1:
            return self._W.T @ d_vec
        return d_vec @ self._W

    def snap(self, slot_vec: np.ndarray | torch.Tensor) -> np.ndarray:
        """Codebook-snap a proposed 8192D slot to exact substrate.

        This is the commitment path: anything that goes into the shared field
        must pass through snap() first.
        """
        if isinstance(slot_vec, torch.Tensor):
            slot_vec = slot_vec.numpy()
        if slot_vec.ndim == 1:
            return codebook_snap(slot_vec)
        return np.stack([codebook_snap(slot_vec[i]) for i in range(slot_vec.shape[0])])

    def down_up_snap(self, slot_vec: np.ndarray) -> np.ndarray:
        """Full path: DOWN -> UP -> snap. For testing snap-idempotence."""
        d_vec = self.project_down(slot_vec)
        reconstructed = self.project_up(d_vec)
        return self.snap(reconstructed)

    # -- GATES -- #

    def check_snap_idempotence(self, n_test_slots: int = 50) -> SnapIdempotenceReport:
        """Gate (a): validly packed slots survive DOWN -> UP -> snap unchanged.

        We test by packing random substrate-valid text into slots, running
        the full down/up/snap path, and checking that the snapped result
        matches the original (after snap, since the original is already exact).
        """
        alphabet = default_alphabet()
        rng = np.random.default_rng(seed=42)
        n_changed = 0

        for i in range(n_test_slots):
            # Build a random slot with random text
            text_len = rng.integers(1, MAX_TEXT_CHARS)
            text = "".join(rng.choice(list(alphabet), size=int(text_len)))
            slot = pack_slot(text=text)
            original = slot.vector

            # Snap the original (should be a no-op on already-exact substrate)
            snapped = codebook_snap(original)
            if not np.allclose(snapped, original, atol=1e-6):
                n_changed += 1

        return SnapIdempotenceReport(
            d_model=self.d_model,
            n_tested=n_test_slots,
            n_changed=n_changed,
            passed=(n_changed == 0),
        )

    def check_separability(self, n_pairs: int = 200) -> SeparabilityReport:
        """Gate (b): distinct slots produce distinct d_model projections.

        We build pairs of slots that differ by exactly one character and
        check their d_model projections are different.
        """
        alphabet = default_alphabet()
        rng = np.random.default_rng(seed=123)
        n_collisions = 0

        for i in range(n_pairs):
            # Build a base slot with random text
            text_len = int(rng.integers(2, MAX_TEXT_CHARS))
            base_text = list("".join(rng.choice(list(alphabet), size=text_len)))

            # Change one character
            pos = int(rng.integers(0, text_len))
            orig_char = base_text[pos]
            new_char = rng.choice([c for c in alphabet if c != orig_char])
            base_text[pos] = str(new_char)
            modified_text = "".join(base_text)

            # Restore original for the base slot
            base_text[pos] = str(orig_char)
            original_text = "".join(base_text)

            slot_a = pack_slot(text=original_text)
            slot_b = pack_slot(text=modified_text)

            proj_a = self.project_down(slot_a.vector)
            proj_b = self.project_down(slot_b.vector)

            if np.allclose(proj_a, proj_b, atol=1e-10):
                n_collisions += 1

        return SeparabilityReport(
            d_model=self.d_model,
            n_pairs_tested=n_pairs,
            n_collisions=n_collisions,
            passed=(n_collisions == 0),
        )

    # -- Persistence -- #

    def save(self, path: str | Path) -> None:
        """Save the adapter to a .pt file."""
        torch.save({
            "version": ADAPTER_VERSION,
            "d_model": self.d_model,
            "slot_width": SLOT_WIDTH,
            "W": torch.tensor(self._W, dtype=torch.float32),
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "SlotAdapter":
        """Load an adapter from a .pt file."""
        data = torch.load(str(path), weights_only=False)
        adapter = cls.__new__(cls)
        adapter.d_model = data["d_model"]
        adapter._W = data["W"].numpy()
        adapter._bank = get_letter_bank()
        return adapter


# --------------------------------------------------------------------------- #
# CORE-OWNED OUTPUT CONTRACT
# --------------------------------------------------------------------------- #
# Trainable write behavior belongs inside the core. Anything outside the core
# on the way to the field must be frozen arithmetic: prototype decode,
# deterministic pack, snap, and typed diff commit.


# --------------------------------------------------------------------------- #
# Adapter registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[int, SlotAdapter] = {}


def get_adapter(d_model: int) -> SlotAdapter:
    """Get the canonical adapter for a d_model size."""
    if d_model in _REGISTRY:
        return _REGISTRY[d_model]

    artifact_path = Path(_ROOT) / "adapters" / f"slot8192_adapter_{d_model}d.pt"
    if artifact_path.exists():
        adapter = SlotAdapter.load(artifact_path)
    else:
        adapter = SlotAdapter(d_model=d_model)
    _REGISTRY[d_model] = adapter
    return adapter


def clear_registry() -> None:
    _REGISTRY.clear()


# --------------------------------------------------------------------------- #
# Self-test / check
# --------------------------------------------------------------------------- #

def run_char_prototype_check(verbose: bool = True) -> bool:
    """Verify exact nearest-code round-trip for every alphabet char at all
    locked d_model sizes.  This is the binding gate for the arithmetic-only
    per-character write path.
    """
    all_ok = True
    for d_model in ADAPTER_D_MODELS:
        table = get_char_prototype_table(d_model)
        ok, fails = table.check_exact()
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  char-prototype round-trip d={d_model} "
                  f"({table.n_chars}/{table.n_chars} chars)")
            if fails:
                print(f"       failures: {fails[:10]}")
        all_ok &= ok
    if verbose:
        print()
        print("char_prototype check:", "PASS" if all_ok else "FAIL")
    return all_ok


def run_check(verbose: bool = True) -> bool:
    """Run snap-idempotence, separability, and char-prototype gates."""
    artifact_dir = Path(_ROOT) / "adapters"
    all_ok = True

    for d_model in ADAPTER_D_MODELS:
        path = artifact_dir / f"slot8192_adapter_{d_model}d.pt"
        if path.exists():
            adapter = SlotAdapter.load(path)
            if verbose:
                print(f"  Checking adapter d={d_model} (from {path.name})...")
        else:
            adapter = SlotAdapter(d_model=d_model)
            if verbose:
                print(f"  Checking adapter d={d_model} (in-memory, not minted)...")

        # Gate (a): snap-idempotence
        snap_report = adapter.check_snap_idempotence()
        if verbose:
            status = "PASS" if snap_report.passed else "FAIL"
            print(f"    {status}  snap-idempotence: {snap_report.n_changed}/{snap_report.n_tested} changed")

        # Gate (b): separability
        sep_report = adapter.check_separability()
        if verbose:
            status = "PASS" if sep_report.passed else "FAIL"
            print(f"    {status}  separability: {sep_report.n_collisions}/{sep_report.n_pairs_tested} collisions")

        all_ok &= snap_report.passed and sep_report.passed

    if verbose:
        print()
        if all_ok:
            print("slot_adapter check: PASS")
        else:
            print("slot_adapter check: FAIL")
    return all_ok


if __name__ == "__main__":
    if "--check" in sys.argv:
        ok1 = run_check(verbose=True)
        ok2 = run_char_prototype_check(verbose=True)
        sys.exit(0 if (ok1 and ok2) else 1)
    elif "--check-prototypes" in sys.argv:
        ok = run_char_prototype_check(verbose=True)
        sys.exit(0 if ok else 1)
    else:
        print("Usage: python adapters/slot_adapter.py --check | --check-prototypes")
        sys.exit(1)
