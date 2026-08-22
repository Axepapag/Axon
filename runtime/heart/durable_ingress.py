"""Durable, recoverable ingress spool for the permanent Axon Heart.

The spool is storage, never authority.  Valve-local admission happens before a
record is appended; the Heart final gate revalidates every recovered record
before constructing authority or touching canonical state.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import PoisonEventError, ReplayEventError, ValveAdmissionError
from .valve import ValveDecision, ValveEnvelope

_INGRESS_SCHEMA = "axon-heart-ingress-event-v2"
_QUARANTINE_SCHEMA = "axon-heart-ingress-quarantine-v1"
_REJECTION_SCHEMA = "axon-heart-ingress-rejection-v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class IngressRecord:
    event_id: str
    valve_id: str
    valve_version: int
    source_id: str
    envelope_type: str
    payload: str
    provenance: str
    enqueued_at: str

    def to_envelope(self) -> ValveEnvelope:
        return ValveEnvelope(
            valve_id=self.valve_id,
            source_id=self.source_id,
            payload=self.payload,
            provenance=self.provenance,
            envelope_type=self.envelope_type,
        )


class DurableIngressSpool:
    """Append-only global FIFO with ack cursor, quarantine, and rejection logs."""

    def __init__(self, spool_dir: Path | str) -> None:
        self.spool_dir = Path(spool_dir)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.spool_dir / "ingress.jsonl"
        self.ack_cursor_path = self.spool_dir / "ack_cursor.json"
        self.quarantine_path = self.spool_dir / "quarantine.jsonl"
        self.rejection_path = self.spool_dir / "rejections.jsonl"

    def submit(
        self,
        envelope: ValveEnvelope,
        *,
        decision: ValveDecision,
        valve_version: int,
    ) -> IngressRecord:
        """Persist one locally admitted envelope or record its explicit rejection."""

        if not isinstance(envelope, ValveEnvelope):
            raise TypeError("DurableIngressSpool.submit requires a ValveEnvelope")
        if isinstance(valve_version, bool) or not isinstance(valve_version, int) or valve_version < 1:
            raise ValueError("valve_version must be a positive integer")
        record = IngressRecord(
            event_id=(
                decision.receipt.item_id
                if decision.receipt is not None
                else uuid.uuid4().hex
            ),
            valve_id=envelope.valve_id,
            valve_version=valve_version,
            source_id=envelope.source_id,
            envelope_type=envelope.envelope_type,
            payload=envelope.payload,
            provenance=envelope.provenance,
            enqueued_at=_now_iso(),
        )
        if not decision.admitted:
            self.record_rejection(record, decision.reason)
            if decision.quarantine:
                self._append_quarantine(record, decision.reason)
                return record
            raise ValveAdmissionError(decision.reason)
        if not envelope.payload:
            self._append_quarantine(record, "empty_payload")
            raise PoisonEventError("empty ingress payload was quarantined")
        self._append_jsonl(self.journal_path, self._event_dict(record))
        return record

    def reject_envelope(
        self,
        envelope: ValveEnvelope,
        *,
        reason: str,
        valve_version: int = 0,
        quarantine: bool = False,
    ) -> IngressRecord:
        """Durably record an envelope rejected before journal admission."""

        record = IngressRecord(
            event_id=uuid.uuid4().hex,
            valve_id=envelope.valve_id,
            valve_version=valve_version,
            source_id=envelope.source_id,
            envelope_type=envelope.envelope_type,
            payload=envelope.payload,
            provenance=envelope.provenance,
            enqueued_at=_now_iso(),
        )
        self.record_rejection(record, reason)
        if quarantine:
            self._append_quarantine(record, reason)
        return record

    def pending(self, count: int | None = None) -> tuple[IngressRecord, ...]:
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("count must be a non-negative integer or None")
            if count == 0:
                return ()
        acked = self._acked_count()
        records: list[IngressRecord] = []
        for index, data in enumerate(self._journal_records()):
            if index < acked:
                continue
            records.append(_record_from_dict(data))
            if count is not None and len(records) >= count:
                break
        return tuple(records)

    def recover(self) -> tuple[IngressRecord, ...]:
        records = self.pending(None)
        seen: set[str] = set()
        for record in records:
            if record.event_id in seen:
                raise ReplayEventError(
                    f"duplicate event_id in unacknowledged ingress tail: {record.event_id}"
                )
            seen.add(record.event_id)
        return records

    @property
    def pending_count(self) -> int:
        return max(0, len(self._journal_records()) - self._acked_count())

    def pending_count_for_valve(self, valve_id: str) -> int:
        return sum(1 for record in self.pending(None) if record.valve_id == valve_id)

    @property
    def quarantine_count(self) -> int:
        return self._line_count(self.quarantine_path)

    @property
    def rejection_count(self) -> int:
        return self._line_count(self.rejection_path)

    def acknowledge(self, event_id: str) -> None:
        """Ack only the FIFO head; already-acked event ids are idempotent."""

        records = [_record_from_dict(item) for item in self._journal_records()]
        acked = self._acked_count()
        for index, record in enumerate(records):
            if record.event_id != event_id:
                continue
            if index < acked:
                return
            if index != acked:
                raise ReplayEventError(
                    f"cannot acknowledge {event_id} before its FIFO predecessors"
                )
            self._write_ack_cursor(event_id, index + 1)
            return
        raise ReplayEventError(f"cannot acknowledge unknown event_id: {event_id}")

    def quarantine_front(self, record: IngressRecord, reason: str) -> None:
        pending = self.pending(1)
        if not pending or pending[0].event_id != record.event_id:
            raise ReplayEventError("only the durable FIFO head may be quarantined")
        self._append_quarantine(record, reason)
        self.record_rejection(record, reason)
        self.acknowledge(record.event_id)

    def record_rejection(self, record: IngressRecord, reason: str) -> None:
        self._append_jsonl(
            self.rejection_path,
            {
                "schema": _REJECTION_SCHEMA,
                "event_id": record.event_id,
                "valve_id": record.valve_id,
                "valve_version": record.valve_version,
                "source_id": record.source_id,
                "reason": reason,
                "recorded_at": _now_iso(),
            },
        )

    def _append_quarantine(self, record: IngressRecord, reason: str) -> None:
        self._append_jsonl(
            self.quarantine_path,
            {
                "schema": _QUARANTINE_SCHEMA,
                **self._record_payload(record),
                "reason": reason,
                "quarantined_at": _now_iso(),
            },
        )

    def _event_dict(self, record: IngressRecord) -> dict[str, Any]:
        return {"schema": _INGRESS_SCHEMA, **self._record_payload(record)}

    @staticmethod
    def _record_payload(record: IngressRecord) -> dict[str, Any]:
        return {
            "event_id": record.event_id,
            "valve_id": record.valve_id,
            "valve_version": record.valve_version,
            "source_id": record.source_id,
            "envelope_type": record.envelope_type,
            "payload": record.payload,
            "provenance": record.provenance,
            "enqueued_at": record.enqueued_at,
        }

    def _journal_records(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            lines = self.journal_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ReplayEventError(f"cannot read ingress journal: {self.journal_path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayEventError(
                    f"corrupt ingress journal JSON at line {line_number}"
                ) from exc
            if data.get("schema") != _INGRESS_SCHEMA:
                raise ReplayEventError(
                    f"unsupported ingress journal schema at line {line_number}"
                )
            records.append(data)
        return records

    def _acked_count(self) -> int:
        if not self.ack_cursor_path.exists():
            return 0
        try:
            data = json.loads(self.ack_cursor_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReplayEventError("corrupt durable ingress ack cursor") from exc
        count = data.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ReplayEventError("durable ingress ack count is invalid")
        if count > len(self._journal_records()):
            raise ReplayEventError("durable ingress ack cursor exceeds journal length")
        return count

    def _write_ack_cursor(self, event_id: str, count: int) -> None:
        value = {"last_event_id": event_id, "count": count}
        temporary = self.ack_cursor_path.with_name(
            self.ack_cursor_path.name + f".{os.getpid()}.tmp"
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.ack_cursor_path)

    @staticmethod
    def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _line_count(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _record_from_dict(data: dict[str, Any]) -> IngressRecord:
    try:
        valve_version = data["valve_version"]
        if isinstance(valve_version, bool) or not isinstance(valve_version, int) or valve_version < 1:
            raise ValueError("invalid valve_version")
        return IngressRecord(
            event_id=str(data["event_id"]),
            valve_id=str(data["valve_id"]),
            valve_version=valve_version,
            source_id=str(data["source_id"]),
            envelope_type=str(data["envelope_type"]),
            payload=str(data["payload"]),
            provenance=str(data.get("provenance", "")),
            enqueued_at=str(data["enqueued_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ReplayEventError("malformed ingress journal record") from exc


__all__ = ["IngressRecord", "DurableIngressSpool"]
