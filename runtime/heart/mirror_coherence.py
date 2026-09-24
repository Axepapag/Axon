"""Heart-side mirror-coherence registry for resident D16 Core Bus participants."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from runtime.field.d16_view import D16ViewIdentity

from .core_bus import FieldDeltaEvent, FieldSnapshotEvent, MirrorAck


class MirrorCoherenceError(RuntimeError):
    """Heart cannot prove that a Core mirror is synchronized."""


class MirrorSyncState(str, Enum):
    SYNCED = "SYNCED"
    APPLYING = "APPLYING"
    STALE = "STALE"
    RESYNC_REQUIRED = "RESYNC_REQUIRED"
    OFFLINE = "OFFLINE"


@dataclass(slots=True)
class MirrorCoherenceRecord:
    core_id: str
    core_generation: int
    state: MirrorSyncState = MirrorSyncState.RESYNC_REQUIRED
    acknowledged_identity: D16ViewIdentity | None = None
    last_ack_id: str | None = None
    last_event_sequence: int | None = None
    last_event_id: str | None = None
    expected_identity: D16ViewIdentity | None = None
    expected_event_sequence: int | None = None
    expected_event_id: str | None = None
    detail: str = "initial synchronization required"

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "core_id": self.core_id,
            "core_generation": self.core_generation,
            "state": self.state.value,
            "acknowledged_identity": None
            if self.acknowledged_identity is None
            else self.acknowledged_identity.to_canonical_dict(),
            "last_ack_id": self.last_ack_id,
            "last_event_sequence": self.last_event_sequence,
            "last_event_id": self.last_event_id,
            "expected_identity": None if self.expected_identity is None else self.expected_identity.to_canonical_dict(),
            "expected_event_sequence": self.expected_event_sequence,
            "expected_event_id": self.expected_event_id,
            "detail": self.detail,
        }


class MirrorCoherenceRegistry:
    """Heart-owned proof that each active Core's exact D16 mirror is current."""

    def __init__(self) -> None:
        self._records: dict[str, MirrorCoherenceRecord] = {}

    def register(self, core_id: str, core_generation: int = 0) -> MirrorCoherenceRecord:
        if not core_id:
            raise ValueError("core_id must be non-empty")
        existing = self._records.get(core_id)
        if existing is not None:
            if existing.core_generation != core_generation:
                record = MirrorCoherenceRecord(core_id=core_id, core_generation=core_generation)
                self._records[core_id] = record
                return record
            return existing
        record = MirrorCoherenceRecord(core_id=core_id, core_generation=core_generation)
        self._records[core_id] = record
        return record

    def record(self, core_id: str) -> MirrorCoherenceRecord:
        try:
            return self._records[core_id]
        except KeyError as exc:
            raise KeyError(f"unregistered Core {core_id!r}") from exc

    def expect(self, core_id: str, event: FieldSnapshotEvent | FieldDeltaEvent) -> MirrorCoherenceRecord:
        record = self.record(core_id)
        if record.state is MirrorSyncState.OFFLINE:
            raise MirrorCoherenceError(f"Core {core_id!r} is offline")
        if isinstance(event, FieldDeltaEvent):
            if record.acknowledged_identity is None:
                self._force_resync(record, "delta offered before initial snapshot acknowledgement")
                raise MirrorCoherenceError(record.detail)
            if record.acknowledged_identity != event.delta.base:
                self._force_resync(record, "delta base does not match Heart's last acknowledged Core mirror")
                raise MirrorCoherenceError(record.detail)
            if record.last_event_sequence is not None and event.sequence != record.last_event_sequence + 1:
                self._force_resync(record, "delta sequence is not contiguous with Heart's last acknowledged event")
                raise MirrorCoherenceError(record.detail)
        record.state = MirrorSyncState.APPLYING
        record.expected_identity = event.target_identity
        record.expected_event_sequence = event.sequence
        record.expected_event_id = event.event_id
        record.detail = f"awaiting MIRROR_ACK for {event.event_id}"
        return record

    def accept_ack(self, ack: MirrorAck) -> MirrorCoherenceRecord:
        record = self.record(ack.core_id)
        if ack.core_generation != record.core_generation:
            self._force_resync(record, "MIRROR_ACK Core generation mismatch")
            raise MirrorCoherenceError(record.detail)
        if record.state is not MirrorSyncState.APPLYING:
            self._force_resync(record, "unexpected MIRROR_ACK while no update is applying")
            raise MirrorCoherenceError(record.detail)
        if (
            ack.event_sequence != record.expected_event_sequence
            or ack.event_id != record.expected_event_id
            or ack.identity != record.expected_identity
        ):
            self._force_resync(record, "MIRROR_ACK does not match Heart's expected event/identity")
            raise MirrorCoherenceError(record.detail)
        record.state = MirrorSyncState.SYNCED
        record.acknowledged_identity = ack.identity
        record.last_ack_id = ack.ack_id
        record.last_event_sequence = ack.event_sequence
        record.last_event_id = ack.event_id
        record.expected_identity = None
        record.expected_event_sequence = None
        record.expected_event_id = None
        record.detail = "mirror coherent"
        return record

    def mark_stale(self, core_id: str, detail: str = "mirror stale") -> MirrorCoherenceRecord:
        record = self.record(core_id)
        record.state = MirrorSyncState.STALE
        record.expected_identity = None
        record.expected_event_sequence = None
        record.expected_event_id = None
        record.detail = detail
        return record

    def require_resync(self, core_id: str, detail: str = "full resynchronization required") -> MirrorCoherenceRecord:
        record = self.record(core_id)
        self._force_resync(record, detail)
        return record

    def mark_offline(self, core_id: str, detail: str = "Core offline") -> MirrorCoherenceRecord:
        record = self.record(core_id)
        record.state = MirrorSyncState.OFFLINE
        record.expected_identity = None
        record.expected_event_sequence = None
        record.expected_event_id = None
        record.detail = detail
        return record

    def barrier_eligible(self, core_id: str, required_identity: D16ViewIdentity) -> bool:
        record = self.record(core_id)
        return record.state is MirrorSyncState.SYNCED and record.acknowledged_identity == required_identity

    def require_barrier_eligible(self, core_id: str, required_identity: D16ViewIdentity) -> MirrorCoherenceRecord:
        record = self.record(core_id)
        if not self.barrier_eligible(core_id, required_identity):
            raise MirrorCoherenceError(
                f"Core {core_id!r} is not mirror-coherent for the required field/view identity: state={record.state.value}"
            )
        return record

    def synchronized_core_ids(self, required_identity: D16ViewIdentity) -> tuple[str, ...]:
        return tuple(
            sorted(
                core_id
                for core_id, record in self._records.items()
                if record.state is MirrorSyncState.SYNCED and record.acknowledged_identity == required_identity
            )
        )

    @staticmethod
    def _force_resync(record: MirrorCoherenceRecord, detail: str) -> None:
        record.state = MirrorSyncState.RESYNC_REQUIRED
        record.expected_identity = None
        record.expected_event_sequence = None
        record.expected_event_id = None
        record.detail = detail


__all__ = [
    "MirrorCoherenceError",
    "MirrorCoherenceRecord",
    "MirrorCoherenceRegistry",
    "MirrorSyncState",
]
