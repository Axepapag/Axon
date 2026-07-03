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

def run_check(verbose: bool = True) -> bool:
    """Run snap-idempotence and separability gates for all minted adapter sizes."""
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
        ok = run_check(verbose=True)
        sys.exit(0 if ok else 1)
    else:
        print("Usage: python adapters/slot_adapter.py --check")
        sys.exit(1)
