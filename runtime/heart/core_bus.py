"""Typed width-independent D16 Core Bus contracts.

The bus is a deterministic event protocol between Heart and resident Cores.
Exact textual payloads travel through Axon's registered 16D Unicode transport;
Core-private hidden width is not part of the wire contract.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from runtime.field import canonical_json_bytes, canonical_sha256
from runtime.field.d16_view import D16ResyncRequired, D16View, D16ViewDelta, D16ViewIdentity
from substrate import (
    SLOT_DIM,
    UNICODE_TRANSPORT_SCHEMA,
    decode_unicode_tokens,
    encode_unicode_text,
    transport_token_cell16,
)

CORE_BUS_SCHEMA = "axon-d16-core-bus-v1"
CORE_BUS_TEXT_FRAME_SCHEMA = "axon-d16-core-bus-text-frame-v1"
MIRROR_ACK_SCHEMA = "axon-d16-mirror-ack-v1"


class CoreBusEventKind(str, Enum):
    FIELD_SNAPSHOT = "FIELD_SNAPSHOT"
    FIELD_DELTA = "FIELD_DELTA"
    MIRROR_ACK = "MIRROR_ACK"
    RESYNC_REQUIRED = "RESYNC_REQUIRED"
    FIRST_REQUEST = "FIRST_REQUEST"
    FIRST_PROPOSAL = "FIRST_PROPOSAL"
    PROPOSAL_SET = "PROPOSAL_SET"
    REFINED_REQUEST = "REFINED_REQUEST"
    REFINED_PROPOSAL = "REFINED_PROPOSAL"
    REFINED_SET = "REFINED_SET"
    FINAL_REQUEST = "FINAL_REQUEST"
    FINAL_VERDICT = "FINAL_VERDICT"
    CANONICAL_SYNC = "CANONICAL_SYNC"


@dataclass(frozen=True, slots=True)
class D16TextFrame:
    """Exact Unicode text carried as categorical D16 transport cells."""

    text: str
    token_ids: tuple[int, ...] = field(init=False)
    cells16: np.ndarray = field(init=False, repr=False, compare=False)
    cells_sha256: str = field(init=False)
    frame_id: str = field(init=False)

    def __post_init__(self) -> None:
        if SLOT_DIM != 16:
            raise RuntimeError(f"frozen substrate width changed: expected 16, got {SLOT_DIM}")
        token_ids = encode_unicode_text(self.text)
        cells = (
            np.stack([transport_token_cell16(token_id) for token_id in token_ids], axis=0).astype(np.float32, copy=False)
            if token_ids
            else np.zeros((0, 16), dtype=np.float32)
        )
        cells.setflags(write=False)
        if decode_unicode_tokens(token_ids) != self.text:
            raise ValueError("D16 text frame failed exact Unicode roundtrip")
        cells_sha = hashlib.sha256(cells.astype("<f4", copy=False).tobytes(order="C")).hexdigest()
        object.__setattr__(self, "token_ids", tuple(token_ids))
        object.__setattr__(self, "cells16", cells)
        object.__setattr__(self, "cells_sha256", cells_sha)
        object.__setattr__(
            self,
            "frame_id",
            canonical_sha256(
                {
                    "schema": CORE_BUS_TEXT_FRAME_SCHEMA,
                    "transport_schema": UNICODE_TRANSPORT_SCHEMA,
                    "text": self.text,
                    "token_ids": list(token_ids),
                    "cells_sha256": cells_sha,
                }
            ),
        )

    def decode(self) -> str:
        for index, token_id in enumerate(self.token_ids):
            if not np.array_equal(self.cells16[index], np.asarray(transport_token_cell16(token_id), dtype=np.float32)):
                raise ValueError("D16 text frame cell no longer matches registered transport cell")
        return decode_unicode_tokens(self.token_ids)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": CORE_BUS_TEXT_FRAME_SCHEMA,
            "transport_schema": UNICODE_TRANSPORT_SCHEMA,
            "text": self.text,
            "token_ids": list(self.token_ids),
            "cells_sha256": self.cells_sha256,
            "frame_id": self.frame_id,
        }


@dataclass(frozen=True, slots=True)
class FieldSnapshotEvent:
    sequence: int
    view: D16View
    event_id: str = field(init=False)

    kind: CoreBusEventKind = field(default=CoreBusEventKind.FIELD_SNAPSHOT, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("FIELD_SNAPSHOT sequence must be a non-negative integer")
        object.__setattr__(self, "event_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def target_identity(self) -> D16ViewIdentity:
        return self.view.identity

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": CORE_BUS_SCHEMA,
            "kind": self.kind.value,
            "sequence": self.sequence,
            "target": self.view.identity.to_canonical_dict(),
        }
        if include_id:
            value["event_id"] = self.event_id
        return value


@dataclass(frozen=True, slots=True)
class FieldDeltaEvent:
    sequence: int
    delta: D16ViewDelta
    event_id: str = field(init=False)

    kind: CoreBusEventKind = field(default=CoreBusEventKind.FIELD_DELTA, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("FIELD_DELTA sequence must be a non-negative integer")
        object.__setattr__(self, "event_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def target_identity(self) -> D16ViewIdentity:
        return self.delta.target

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": CORE_BUS_SCHEMA,
            "kind": self.kind.value,
            "sequence": self.sequence,
            "delta": self.delta.to_canonical_dict(),
        }
        if include_id:
            value["event_id"] = self.event_id
        return value


@dataclass(frozen=True, slots=True)
class MirrorAck:
    core_id: str
    core_generation: int
    event_sequence: int
    event_id: str
    identity: D16ViewIdentity
    ack_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.core_id:
            raise ValueError("MIRROR_ACK core_id must be non-empty")
        if isinstance(self.core_generation, bool) or not isinstance(self.core_generation, int) or self.core_generation < 0:
            raise ValueError("MIRROR_ACK core_generation must be a non-negative integer")
        if isinstance(self.event_sequence, bool) or not isinstance(self.event_sequence, int) or self.event_sequence < 0:
            raise ValueError("MIRROR_ACK event_sequence must be a non-negative integer")
        if not self.event_id:
            raise ValueError("MIRROR_ACK event_id must be non-empty")
        object.__setattr__(self, "ack_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": MIRROR_ACK_SCHEMA,
            "kind": CoreBusEventKind.MIRROR_ACK.value,
            "core_id": self.core_id,
            "core_generation": self.core_generation,
            "event_sequence": self.event_sequence,
            "event_id": self.event_id,
            "identity": self.identity.to_canonical_dict(),
        }
        if include_id:
            value["ack_id"] = self.ack_id
        return value


class D16CoreMirror:
    """Deterministic non-authoritative mirror used by B1/B2 proofs and adapters."""

    def __init__(self, core_id: str, core_generation: int = 0) -> None:
        if not core_id:
            raise ValueError("core_id must be non-empty")
        self.core_id = core_id
        self.core_generation = core_generation
        self._view: D16View | None = None
        self._last_sequence: int | None = None
        self._last_event_id: str | None = None

    @property
    def view(self) -> D16View | None:
        return self._view

    @property
    def identity(self) -> D16ViewIdentity | None:
        return None if self._view is None else self._view.identity

    @property
    def last_sequence(self) -> int | None:
        return self._last_sequence

    def apply_snapshot(self, event: FieldSnapshotEvent) -> MirrorAck:
        self._view = event.view
        self._last_sequence = event.sequence
        self._last_event_id = event.event_id
        return self._ack()

    def apply_delta(self, event: FieldDeltaEvent) -> MirrorAck:
        if self._view is None or self._last_sequence is None:
            raise D16ResyncRequired("Core has no D16 snapshot; full resync required")
        expected = self._last_sequence + 1
        if event.sequence != expected:
            raise D16ResyncRequired(
                f"D16 Core Bus sequence gap: expected {expected}, received {event.sequence}"
            )
        self._view = event.delta.apply(self._view)
        self._last_sequence = event.sequence
        self._last_event_id = event.event_id
        return self._ack()

    def _ack(self) -> MirrorAck:
        if self._view is None or self._last_sequence is None or self._last_event_id is None:
            raise RuntimeError("cannot ACK an uninitialized D16 mirror")
        return MirrorAck(
            core_id=self.core_id,
            core_generation=self.core_generation,
            event_sequence=self._last_sequence,
            event_id=self._last_event_id,
            identity=self._view.identity,
        )


__all__ = [
    "CORE_BUS_SCHEMA",
    "CORE_BUS_TEXT_FRAME_SCHEMA",
    "CoreBusEventKind",
    "D16CoreMirror",
    "D16TextFrame",
    "FieldDeltaEvent",
    "FieldSnapshotEvent",
    "MIRROR_ACK_SCHEMA",
    "MirrorAck",
]
