"""Durable append-only ingress and consumption receipts.

The queue owns no transaction boundary.  It accepts a caller-supplied SQLite
connection so receipt insertion can participate in the same ``BEGIN
IMMEDIATE`` transaction that commits a runtime tick and its tool outbox.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
import sqlite3
from typing import Any, Iterable, Sequence

from runtime.field import (
    FieldSpan,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
)

from .field_transaction import (
    AppendSpans,
    ClearRegion,
    ReplaceSpans,
    SystemFieldUpdate,
)


INGRESS_SCHEMA_VERSION = "axon-ingress-v1"

INGRESS_EVENT_KINDS = frozenset(
    {
        "user_input",
        "conversation_history",
        "tool_result",
        "advisor_result",
        "diary",
        "situation_awareness",
        "task_state",
        "structured_knowledge",
    }
)

_APPEND_REGION_BY_KIND = {
    "conversation_history": LogicalRegion.CONVERSATION_HISTORY,
    "tool_result": LogicalRegion.TOOL_RESULTS,
    "advisor_result": LogicalRegion.ADVISOR_INPUT,
    "diary": LogicalRegion.DIARY,
    "structured_knowledge": LogicalRegion.STRUCTURED_KNOWLEDGE,
}

_REPLACE_REGION_BY_KIND = {
    "situation_awareness": LogicalRegion.SITUATION_AWARENESS,
    "task_state": LogicalRegion.TASK_STATE,
}


class IngressError(ValueError):
    """Base class for ingress contract violations."""


class IngressConflictError(IngressError):
    """An idempotency key was reused for different immutable content."""


class IngressSequenceError(IngressError):
    """A source attempted to move its sequence backward or reuse a position."""


class IngressConsumptionError(IngressError):
    """A receipt could not be appended to the current transaction."""


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise IngressError(f"{label} must be a non-empty string")
    return value


def _sequence(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IngressError("source_sequence must be a non-negative integer")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


@dataclass(frozen=True, slots=True)
class IngressEvent:
    """One exact source event with a monotonic per-source sequence."""

    idempotency_key: str
    source: str
    source_sequence: int
    event_kind: str
    exact_text: str
    provenance: str
    exact_utf8_sha256: str = field(init=False)
    event_id: str = field(init=False)
    canonical_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.idempotency_key, "idempotency_key")
        _nonempty(self.source, "source")
        _sequence(self.source_sequence)
        if self.event_kind not in INGRESS_EVENT_KINDS:
            raise IngressError(f"unknown ingress event kind {self.event_kind!r}")
        if not isinstance(self.exact_text, str) or not self.exact_text:
            raise IngressError("exact_text must be a non-empty string")
        _nonempty(self.provenance, "provenance")
        text_hash = _sha256_text(self.exact_text)
        object.__setattr__(self, "exact_utf8_sha256", text_hash)
        digest = canonical_sha256(self.to_canonical_dict())
        object.__setattr__(self, "event_id", digest)
        object.__setattr__(self, "canonical_hash", digest)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-ingress-event-v1",
            "idempotency_key": self.idempotency_key,
            "source": self.source,
            "source_sequence": self.source_sequence,
            "event_kind": self.event_kind,
            "exact_text": self.exact_text,
            "provenance": self.provenance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_canonical_dict(),
            "exact_utf8_sha256": self.exact_utf8_sha256,
            "event_id": self.event_id,
            "canonical_hash": self.canonical_hash,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "IngressEvent":
        if not isinstance(value, dict):
            raise IngressError("serialized ingress event must be an object")
        expected = {
            "schema",
            "idempotency_key",
            "source",
            "source_sequence",
            "event_kind",
            "exact_text",
            "provenance",
            "exact_utf8_sha256",
            "event_id",
            "canonical_hash",
        }
        if set(value) != expected:
            raise IngressError("serialized ingress event fields mismatch")
        if value["schema"] != "axon-ingress-event-v1":
            raise IngressError("unknown ingress event schema")
        event = cls(
            idempotency_key=value["idempotency_key"],
            source=value["source"],
            source_sequence=value["source_sequence"],
            event_kind=value["event_kind"],
            exact_text=value["exact_text"],
            provenance=value["provenance"],
        )
        if value["exact_utf8_sha256"] != event.exact_utf8_sha256:
            raise IngressConflictError("ingress exact-text hash mismatch")
        if value["event_id"] != event.event_id:
            raise IngressConflictError("ingress event ID mismatch")
        if value["canonical_hash"] != event.canonical_hash:
            raise IngressConflictError("ingress canonical hash mismatch")
        return event


@dataclass(frozen=True, slots=True)
class ConsumptionReceipt:
    """Immutable proof that one event entered one committed tick."""

    event_id: str
    tick_commit_id: str
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.event_id):
            raise IngressConsumptionError(
                "receipt event_id must be lowercase SHA-256"
            )
        _nonempty(self.tick_commit_id, "tick_commit_id")
        object.__setattr__(
            self,
            "receipt_id",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-ingress-consumption-receipt-v1",
            "event_id": self.event_id,
            "tick_commit_id": self.tick_commit_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_canonical_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, value: Any) -> "ConsumptionReceipt":
        if not isinstance(value, dict):
            raise IngressConsumptionError(
                "serialized consumption receipt must be an object"
            )
        expected = {
            "schema",
            "event_id",
            "tick_commit_id",
            "receipt_id",
        }
        if set(value) != expected:
            raise IngressConsumptionError(
                "serialized consumption receipt fields mismatch"
            )
        if value["schema"] != "axon-ingress-consumption-receipt-v1":
            raise IngressConsumptionError("unknown consumption receipt schema")
        receipt = cls(
            event_id=value["event_id"],
            tick_commit_id=value["tick_commit_id"],
        )
        if value["receipt_id"] != receipt.receipt_id:
            raise IngressConflictError("consumption receipt ID mismatch")
        return receipt


_INSERT_ONLY_TABLES = (
    "axon_ingress_events",
    "axon_ingress_consumption_receipts",
)


def ensure_ingress_schema(connection: sqlite3.Connection) -> None:
    """Create ingress tables without committing the caller's transaction."""

    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be sqlite3.Connection")
    statements = (
        """
        CREATE TABLE IF NOT EXISTS axon_ingress_events (
            arrival_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            idempotency_key TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            source_sequence INTEGER NOT NULL CHECK(source_sequence >= 0),
            event_kind TEXT NOT NULL,
            exact_utf8 BLOB NOT NULL,
            exact_utf8_sha256 TEXT NOT NULL,
            canonical_json TEXT NOT NULL,
            canonical_sha256 TEXT NOT NULL,
            UNIQUE(source, source_sequence)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS axon_ingress_fifo
        ON axon_ingress_events(arrival_seq)
        """,
        """
        CREATE TABLE IF NOT EXISTS axon_ingress_consumption_receipts (
            receipt_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id TEXT NOT NULL UNIQUE,
            event_id TEXT NOT NULL UNIQUE
                REFERENCES axon_ingress_events(event_id),
            tick_commit_id TEXT NOT NULL,
            canonical_json TEXT NOT NULL,
            canonical_sha256 TEXT NOT NULL
        )
        """,
    )
    for statement in statements:
        connection.execute(statement)
    for table in _INSERT_ONLY_TABLES:
        connection.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table}_reject_update
            BEFORE UPDATE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'ingress table is insert-only');
            END
            """
        )
        connection.execute(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table}_reject_delete
            BEFORE DELETE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'ingress table is insert-only');
            END
            """
        )


class IngressQueue:
    """Append-only ingress facade over a caller-owned SQLite connection."""

    def __init__(self, connection: sqlite3.Connection):
        if not isinstance(connection, sqlite3.Connection):
            raise TypeError("connection must be sqlite3.Connection")
        self.connection = connection
        ensure_ingress_schema(connection)

    @staticmethod
    def _event_json(event: IngressEvent) -> str:
        return _canonical_text(event.to_dict())

    @staticmethod
    def _receipt_json(receipt: ConsumptionReceipt) -> str:
        return _canonical_text(receipt.to_dict())

    def _event_from_row(self, row: Any) -> IngressEvent:
        value = json.loads(row[0])
        event = IngressEvent.from_dict(value)
        exact_bytes = bytes(row[1])
        if exact_bytes != event.exact_text.encode("utf-8"):
            raise IngressConflictError(
                "stored ingress bytes differ from canonical event"
            )
        if row[2] != event.exact_utf8_sha256:
            raise IngressConflictError(
                "stored ingress byte hash differs from canonical event"
            )
        if row[3] != event.canonical_hash:
            raise IngressConflictError(
                "stored ingress canonical hash differs from event"
            )
        return event

    def get_event(self, event_id: str) -> IngressEvent:
        row = self.connection.execute(
            """
            SELECT
                canonical_json,
                exact_utf8,
                exact_utf8_sha256,
                canonical_sha256
            FROM axon_ingress_events
            WHERE event_id=?
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError(event_id)
        return self._event_from_row(row)

    def enqueue(self, event: IngressEvent) -> IngressEvent:
        """Insert once; an exact idempotent retry returns the stored event."""

        if not isinstance(event, IngressEvent):
            raise TypeError("event must be IngressEvent")
        existing = self.connection.execute(
            """
            SELECT
                canonical_json,
                exact_utf8,
                exact_utf8_sha256,
                canonical_sha256
            FROM axon_ingress_events
            WHERE idempotency_key=?
            """,
            (event.idempotency_key,),
        ).fetchone()
        if existing is not None:
            stored = self._event_from_row(existing)
            if stored != event:
                raise IngressConflictError(
                    "idempotency key has conflicting ingress content"
                )
            return stored

        latest = self.connection.execute(
            """
            SELECT MAX(source_sequence)
            FROM axon_ingress_events
            WHERE source=?
            """,
            (event.source,),
        ).fetchone()[0]
        if latest is not None and event.source_sequence <= int(latest):
            raise IngressSequenceError(
                f"source {event.source!r} sequence must exceed {latest}"
            )

        payload = self._event_json(event)
        try:
            self.connection.execute(
                """
                INSERT INTO axon_ingress_events(
                    event_id,
                    idempotency_key,
                    source,
                    source_sequence,
                    event_kind,
                    exact_utf8,
                    exact_utf8_sha256,
                    canonical_json,
                    canonical_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.idempotency_key,
                    event.source,
                    event.source_sequence,
                    event.event_kind,
                    event.exact_text.encode("utf-8"),
                    event.exact_utf8_sha256,
                    payload,
                    event.canonical_hash,
                ),
            )
        except sqlite3.IntegrityError as exc:
            retry = self.connection.execute(
                """
                SELECT
                    canonical_json,
                    exact_utf8,
                    exact_utf8_sha256,
                    canonical_sha256
                FROM axon_ingress_events
                WHERE idempotency_key=?
                """,
                (event.idempotency_key,),
            ).fetchone()
            if retry is not None and self._event_from_row(retry) == event:
                return event
            raise IngressConflictError(
                "ingress event conflicts with immutable queue state"
            ) from exc
        return event

    def pending(self, *, limit: int | None = None) -> tuple[IngressEvent, ...]:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise IngressError("pending limit must be a non-negative integer")
        sql = """
            SELECT
                event.canonical_json,
                event.exact_utf8,
                event.exact_utf8_sha256,
                event.canonical_sha256
            FROM axon_ingress_events AS event
            LEFT JOIN axon_ingress_consumption_receipts AS receipt
                ON receipt.event_id=event.event_id
            WHERE receipt.event_id IS NULL
            ORDER BY event.arrival_seq
        """
        parameters: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            parameters = (limit,)
        rows = self.connection.execute(sql, parameters).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def receipts(self) -> tuple[ConsumptionReceipt, ...]:
        rows = self.connection.execute(
            """
            SELECT canonical_json, canonical_sha256
            FROM axon_ingress_consumption_receipts
            ORDER BY receipt_seq
            """
        ).fetchall()
        result: list[ConsumptionReceipt] = []
        for row in rows:
            receipt = ConsumptionReceipt.from_dict(json.loads(row[0]))
            if row[1] != receipt.receipt_id:
                raise IngressConflictError(
                    "stored consumption hash differs from receipt"
                )
            result.append(receipt)
        return tuple(result)

    def consume_in_current_transaction(
        self,
        event_ids: Sequence[str],
        tick_commit_id: str,
    ) -> tuple[ConsumptionReceipt, ...]:
        """Append receipts without opening or committing a transaction."""

        if not self.connection.in_transaction:
            raise IngressConsumptionError(
                "caller must begin the tick transaction before consumption"
            )
        _nonempty(tick_commit_id, "tick_commit_id")
        identifiers = tuple(str(event_id) for event_id in event_ids)
        if any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in identifiers):
            raise IngressConsumptionError(
                "event_ids must contain lowercase SHA-256 values"
            )
        if len(identifiers) != len(set(identifiers)):
            raise IngressConsumptionError("event_ids contain duplicates")

        receipts: list[ConsumptionReceipt] = []
        self.connection.execute("SAVEPOINT axon_ingress_consume")
        try:
            for event_id in identifiers:
                if self.connection.execute(
                    "SELECT 1 FROM axon_ingress_events WHERE event_id=?",
                    (event_id,),
                ).fetchone() is None:
                    raise IngressConsumptionError(
                        f"cannot consume unknown ingress event {event_id}"
                    )
                if self.connection.execute(
                    """
                    SELECT 1
                    FROM axon_ingress_consumption_receipts
                    WHERE event_id=?
                    """,
                    (event_id,),
                ).fetchone() is not None:
                    raise IngressConsumptionError(
                        f"ingress event {event_id} is already consumed"
                    )
                receipt = ConsumptionReceipt(
                    event_id=event_id,
                    tick_commit_id=tick_commit_id,
                )
                self.connection.execute(
                    """
                    INSERT INTO axon_ingress_consumption_receipts(
                        receipt_id,
                        event_id,
                        tick_commit_id,
                        canonical_json,
                        canonical_sha256
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        receipt.receipt_id,
                        receipt.event_id,
                        receipt.tick_commit_id,
                        self._receipt_json(receipt),
                        receipt.receipt_id,
                    ),
                )
                receipts.append(receipt)
        except Exception:
            self.connection.execute("ROLLBACK TO axon_ingress_consume")
            self.connection.execute("RELEASE axon_ingress_consume")
            raise
        else:
            self.connection.execute("RELEASE axon_ingress_consume")
        return tuple(receipts)


def _span_for_event(
    event: IngressEvent,
    region: LogicalRegion,
) -> FieldSpan:
    span_id = "ingress-" + canonical_sha256(
        {
            "event_id": event.event_id,
            "logical_region": region.value,
        }
    )
    return FieldSpan(
        span_id=span_id,
        text=event.exact_text,
        kind=f"ingress_{event.event_kind}",
        source=event.source,
        provenance=event.provenance,
    )


def build_system_update(
    head: SharedFieldSnapshot,
    pending_events: Iterable[IngressEvent],
) -> SystemFieldUpdate:
    """Compile FIFO ingress into one whole-span operation per region."""

    if not isinstance(head, SharedFieldSnapshot):
        raise TypeError("head must be SharedFieldSnapshot")
    events = tuple(pending_events)
    if not all(isinstance(event, IngressEvent) for event in events):
        raise TypeError("pending_events must contain only IngressEvent values")
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise IngressError("pending_events contain duplicate event IDs")

    append_spans: dict[LogicalRegion, list[FieldSpan]] = {}
    latest_replace: dict[LogicalRegion, FieldSpan] = {}
    user_input_spans: list[FieldSpan] = []

    for event in events:
        if event.event_kind == "user_input":
            append_spans.setdefault(
                LogicalRegion.CONVERSATION_HISTORY,
                [],
            ).append(
                _span_for_event(event, LogicalRegion.CONVERSATION_HISTORY)
            )
            user_input_spans.append(
                _span_for_event(event, LogicalRegion.USER_INPUT)
            )
            continue
        append_region = _APPEND_REGION_BY_KIND.get(event.event_kind)
        if append_region is not None:
            append_spans.setdefault(append_region, []).append(
                _span_for_event(event, append_region)
            )
            continue
        replace_region = _REPLACE_REGION_BY_KIND.get(event.event_kind)
        if replace_region is not None:
            # FIFO order makes the final event the newest live value.  Earlier
            # events remain exact immutable queue records and audit evidence.
            latest_replace[replace_region] = _span_for_event(
                event,
                replace_region,
            )
            continue
        raise IngressError(f"unhandled ingress kind {event.event_kind!r}")

    operations: list[AppendSpans | ReplaceSpans | ClearRegion] = []
    for region, spans in append_spans.items():
        operations.append(AppendSpans(region=region, spans=tuple(spans)))
    for region, span in latest_replace.items():
        operations.append(ReplaceSpans(region=region, spans=(span,)))
    if user_input_spans:
        operations.append(
            ReplaceSpans(
                region=LogicalRegion.USER_INPUT,
                spans=tuple(user_input_spans),
            )
        )
    elif head.region(LogicalRegion.USER_INPUT).spans:
        operations.append(ClearRegion(region=LogicalRegion.USER_INPUT))

    return SystemFieldUpdate(
        base_field_id=head.field_id,
        base_tick_id=head.tick_id,
        operations=tuple(operations),
        evidence=tuple(event_ids),
    )


__all__ = [
    "INGRESS_SCHEMA_VERSION",
    "INGRESS_EVENT_KINDS",
    "IngressError",
    "IngressConflictError",
    "IngressSequenceError",
    "IngressConsumptionError",
    "IngressEvent",
    "ConsumptionReceipt",
    "ensure_ingress_schema",
    "IngressQueue",
    "build_system_update",
]
