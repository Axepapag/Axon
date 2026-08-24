"""Grounded deterministic codec between canonical fields and D64 Heart tissue."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import torch

from runtime.field import (
    CompiledD64DualSurface,
    D64FieldCompiler,
    D64SemanticSurfaceCompiler,
    FieldDelta,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_sha256,
    replacement_delta,
)

if TYPE_CHECKING:
    from .translation_core import HeartTranslationCore


D64_HEART_FRAME_SCHEMA = "axon-heart-d64-frame-v1"


class D64HeartCodecError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class D64HeartFrame:
    source_field_id: str
    source_tick_id: int
    rail_id: str
    semantic_surface_id: str
    characters: tuple[str, ...]
    canonical_positions: tuple[int, ...]
    cells16: np.ndarray
    address_sha256: str
    coverage_sha256: str
    frame_id: str = field(init=False)

    def __post_init__(self) -> None:
        cells = np.asarray(self.cells16, dtype=np.float32)
        if cells.shape != (len(self.characters), 16):
            raise ValueError("D64 Heart frame cells must have shape [characters, 16]")
        if not self.characters:
            raise ValueError("D64 Heart frame cannot be empty")
        if any(not isinstance(char, str) or len(char) != 1 for char in self.characters):
            raise ValueError("D64 Heart frame characters must be literal single characters")
        positions = tuple(int(item) for item in self.canonical_positions)
        if len(positions) != len(self.characters) or any(item < 0 for item in positions):
            raise ValueError("D64 Heart frame canonical positions are invalid")
        if any(right <= left for left, right in zip(positions, positions[1:])):
            raise ValueError("D64 Heart frame canonical positions must be strictly increasing")
        object.__setattr__(self, "canonical_positions", positions)
        if not np.isfinite(cells).all():
            raise ValueError("D64 Heart frame cells must be finite")
        cells.setflags(write=False)
        object.__setattr__(self, "cells16", cells)
        object.__setattr__(self, "frame_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def exact_text(self) -> str:
        return "".join(self.characters)

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": D64_HEART_FRAME_SCHEMA,
            "source_field_id": self.source_field_id,
            "source_tick_id": self.source_tick_id,
            "rail_id": self.rail_id,
            "semantic_surface_id": self.semantic_surface_id,
            "character_count": len(self.characters),
            "exact_text_sha256": hashlib.sha256(self.exact_text.encode("utf-8")).hexdigest(),
            "cells16_sha256": hashlib.sha256(self.cells16.tobytes(order="C")).hexdigest(),
            "canonical_positions_sha256": hashlib.sha256(
                np.asarray(self.canonical_positions, dtype=np.int64).tobytes(order="C")
            ).hexdigest(),
            "address_sha256": self.address_sha256,
            "coverage_sha256": self.coverage_sha256,
        }
        if include_id:
            value["frame_id"] = self.frame_id
        return value


@dataclass(slots=True)
class D64HeartModelBatch:
    source_indices: torch.Tensor
    source_cells16: torch.Tensor
    source_positions: torch.Tensor
    source_mask: torch.Tensor
    frame_ids: tuple[str, ...]
    field_ids: tuple[str, ...]
    rail_ids: tuple[str, ...]

    def to(self, device: torch.device | str) -> "D64HeartModelBatch":
        return D64HeartModelBatch(
            source_indices=self.source_indices.to(device),
            source_cells16=self.source_cells16.to(device),
            source_positions=self.source_positions.to(device),
            source_mask=self.source_mask.to(device),
            frame_ids=self.frame_ids,
            field_ids=self.field_ids,
            rail_ids=self.rail_ids,
        )


class D64HeartCodec:
    """Verify a real compiled dual surface and expose its literal lane cells."""

    def __init__(
        self,
        compiler: D64SemanticSurfaceCompiler | None = None,
        exact_compiler: D64FieldCompiler | None = None,
    ) -> None:
        self.compiler = compiler or D64SemanticSurfaceCompiler()
        self.exact_compiler = exact_compiler or D64FieldCompiler()

    def compile(
        self,
        snapshot: SharedFieldSnapshot,
        dual: CompiledD64DualSurface | None = None,
    ) -> D64HeartFrame:
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("snapshot must be SharedFieldSnapshot")
        if dual is None:
            exact = self.exact_compiler.compile(snapshot)
            dual = self.compiler.compile_dual(snapshot, exact)
        dual.assert_fresh(snapshot)
        dual.exact.verify_roundtrip(snapshot)
        dual.semantic.verify_grounding(snapshot, dual.exact)
        addresses = tuple(address for address in dual.exact.addresses if address is not None)
        if len(addresses) != dual.exact.coverage.expected_active_characters:
            raise D64HeartCodecError("D64 frame address count does not match complete coverage")
        positions = tuple(address.global_position for address in addresses)
        if any(right <= left for left, right in zip(positions, positions[1:])):
            raise D64HeartCodecError("D64 frame addresses are not in canonical ordering")
        cells = np.stack(
            [dual.exact.lane_cell16(address.row_index, address.lane_index) for address in addresses],
            axis=0,
        ).astype(np.float32, copy=False)
        return D64HeartFrame(
            source_field_id=snapshot.field_id,
            source_tick_id=snapshot.tick_id,
            rail_id=dual.exact.rail_id,
            semantic_surface_id=dual.semantic.surface_id,
            characters=tuple(address.character for address in addresses),
            canonical_positions=positions,
            cells16=cells,
            address_sha256=dual.exact.coverage.address_sha256,
            coverage_sha256=canonical_sha256(dual.exact.coverage.to_canonical_dict()),
        )

    @staticmethod
    def batch_for_model(
        model: "HeartTranslationCore", frames: Iterable[D64HeartFrame]
    ) -> D64HeartModelBatch:
        rows = tuple(frames)
        if not rows:
            raise ValueError("cannot batch zero D64 Heart frames")
        max_chars = max(len(item.characters) for item in rows)
        indices = torch.full((len(rows), max_chars), model.source_pad_index, dtype=torch.long)
        cells = torch.zeros((len(rows), max_chars, 16), dtype=torch.float32)
        positions = torch.zeros((len(rows), max_chars), dtype=torch.long)
        mask = torch.zeros((len(rows), max_chars), dtype=torch.bool)
        bank = model.bank16.detach().cpu().numpy()
        for row_index, frame in enumerate(rows):
            try:
                source = [model.char_to_index[char] for char in frame.characters]
            except KeyError as exc:
                raise D64HeartCodecError(f"compiled D64 frame contains unsupported model character: {exc}") from exc
            expected = bank[np.asarray(source, dtype=np.int64)]
            if not np.allclose(frame.cells16, expected, rtol=0.0, atol=1e-6):
                raise D64HeartCodecError("compiled D64 lane cell does not match frozen Heart substrate")
            length = len(source)
            indices[row_index, :length] = torch.tensor(source, dtype=torch.long)
            cells[row_index, :length] = torch.from_numpy(frame.cells16.copy())
            positions[row_index, :length] = torch.tensor(frame.canonical_positions, dtype=torch.long)
            mask[row_index, :length] = True
        return D64HeartModelBatch(
            source_indices=indices,
            source_cells16=cells,
            source_positions=positions,
            source_mask=mask,
            frame_ids=tuple(item.frame_id for item in rows),
            field_ids=tuple(item.source_field_id for item in rows),
            rail_ids=tuple(item.rail_id for item in rows),
        )

    @staticmethod
    def proposed_replacement(
        snapshot: SharedFieldSnapshot,
        dual: CompiledD64DualSurface,
        *,
        region: LogicalRegion | str,
        text: str,
        author_core_id: str,
        pass_id: str | int,
        frame: D64HeartFrame,
        evidence: Iterable[str] = (),
    ) -> FieldDelta:
        dual.assert_fresh(snapshot)
        if frame.source_field_id != snapshot.field_id or frame.rail_id != dual.exact.rail_id:
            raise D64HeartCodecError("Heart proposal frame is stale or rail-substituted")
        return replacement_delta(
            snapshot,
            dual.exact,
            region=region,
            text=text,
            author_core_id=author_core_id,
            pass_id=pass_id,
            evidence=(frame.frame_id, *tuple(evidence)),
            provenance="heart_d64_grounded_translation_proposal_v1",
        )


__all__ = [
    "D64_HEART_FRAME_SCHEMA",
    "D64HeartCodecError",
    "D64HeartFrame",
    "D64HeartModelBatch",
    "D64HeartCodec",
]
