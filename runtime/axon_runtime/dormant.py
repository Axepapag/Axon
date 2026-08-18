"""Append-only dormant-memory records and a small durable SQLite store.

This module is deliberately conservative.  Source text is authoritative and is
stored byte-for-byte.  Containers, entities, triples, lifecycle changes,
extraction jobs, and surfacing decisions are immutable interpretations of that
source.  They may be superseded by later events, but never updated or deleted.

The built-in extractor recognizes only explicit, readable relation clauses
(``is``, ``has``, ``uses``, ``needs``, and ``wants``).  Its output is labelled
as a bootstrap candidate and is *not* semantic proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from runtime.field import FieldSpan, LogicalRegion, canonical_json_bytes


DORMANT_SCHEMA_VERSION = "axon-dormant-v1"
BOOTSTRAP_EXTRACTOR_ID = (
    "bootstrap-explicit-readable-relations-v1:not-semantic-proof"
)
BOOTSTRAP_RULESET_SHA256 = hashlib.sha256(
    b"anchored clauses: subject (is|has|uses|needs|wants) object"
).hexdigest()

SPAN_STATES = frozenset({"active", "masked", "dormant", "resurfaced"})
KNOWLEDGE_STATES = frozenset(
    {"candidate", "accepted", "superseded", "contradicted", "retracted"}
)
JOB_STATES = frozenset({"queued", "leased", "completed", "failed"})

_SPAN_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "active": frozenset({"masked"}),
    "masked": frozenset({"active", "dormant", "resurfaced"}),
    "dormant": frozenset({"resurfaced"}),
    "resurfaced": frozenset({"active", "masked", "dormant"}),
}
_KNOWLEDGE_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "candidate": frozenset(
        {"accepted", "superseded", "contradicted", "retracted"}
    ),
    "accepted": frozenset({"superseded", "contradicted", "retracted"}),
    "contradicted": frozenset({"accepted", "superseded", "retracted"}),
    "superseded": frozenset(),
    "retracted": frozenset(),
}


class DormantError(ValueError):
    """Base error for a dormant-store contract violation."""


class DormantConflictError(DormantError):
    """An immutable identifier already exists with different content."""


class DormantTransitionError(DormantError):
    """An append-only lifecycle transition is not legal."""


def _canonical_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _content_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_json_bytes(payload)).hexdigest()}"


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DormantError(f"{label} must be a non-empty string")
    return value


def _tick(value: int, label: str = "tick_id") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DormantError(f"{label} must be a non-negative integer")
    return value


def _confidence(value: float) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise DormantError("confidence must be in [0, 1]")
    return result


def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({_nonempty(str(value), "reference") for value in values}))


def _ordered_unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _nonempty(str(value), "reference")
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def _normal_key(text: str) -> str:
    return " ".join(text.casefold().split())


@dataclass(frozen=True, slots=True)
class ExactProvenance:
    """An exact character slice of one immutable :class:`SourceRecord`."""

    source_record_id: str
    span_id: str
    char_start: int
    char_end: int
    exact_text_sha256: str
    source: str
    provenance: str

    def __post_init__(self) -> None:
        _nonempty(self.source_record_id, "source_record_id")
        _nonempty(self.span_id, "span_id")
        if (
            isinstance(self.char_start, bool)
            or isinstance(self.char_end, bool)
            or not isinstance(self.char_start, int)
            or not isinstance(self.char_end, int)
            or self.char_start < 0
            or self.char_end <= self.char_start
        ):
            raise DormantError("provenance bounds must satisfy 0 <= start < end")
        if not re.fullmatch(r"[0-9a-f]{64}", self.exact_text_sha256):
            raise DormantError("exact_text_sha256 must be lowercase SHA-256")
        _nonempty(self.source, "source")
        _nonempty(self.provenance, "provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_record_id": self.source_record_id,
            "span_id": self.span_id,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "exact_text_sha256": self.exact_text_sha256,
            "source": self.source,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExactProvenance":
        return cls(
            source_record_id=str(value["source_record_id"]),
            span_id=str(value["span_id"]),
            char_start=int(value["char_start"]),
            char_end=int(value["char_end"]),
            exact_text_sha256=str(value["exact_text_sha256"]),
            source=str(value["source"]),
            provenance=str(value["provenance"]),
        )


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """Exact append-only source text captured from the living field."""

    stream_id: str
    sequence: int
    region: LogicalRegion | str
    exact_text: str
    source: str
    provenance: str
    tick_id: int
    parent_source_id: str | None = None
    record_id: str = field(init=False)
    exact_utf8_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.stream_id, "stream_id")
        _tick(self.sequence, "sequence")
        _tick(self.tick_id)
        try:
            region = (
                self.region
                if isinstance(self.region, LogicalRegion)
                else LogicalRegion(self.region)
            )
        except (TypeError, ValueError) as exc:
            raise DormantError(f"unknown logical region {self.region!r}") from exc
        if not isinstance(self.exact_text, str):
            raise DormantError("exact_text must be a string")
        _nonempty(self.source, "source")
        _nonempty(self.provenance, "provenance")
        if self.parent_source_id is not None:
            _nonempty(self.parent_source_id, "parent_source_id")
        object.__setattr__(self, "region", region)
        digest = _sha256_text(self.exact_text)
        object.__setattr__(self, "exact_utf8_sha256", digest)
        object.__setattr__(self, "record_id", _content_id("src", self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "source-record-v1",
            "stream_id": self.stream_id,
            "sequence": self.sequence,
            "region": self.region.value,
            "exact_text": self.exact_text,
            "source": self.source,
            "provenance": self.provenance,
            "tick_id": self.tick_id,
            "parent_source_id": self.parent_source_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.payload(),
            "record_id": self.record_id,
            "exact_utf8_sha256": self.exact_utf8_sha256,
        }

    def exact_ref(
        self,
        span_id: str,
        char_start: int = 0,
        char_end: int | None = None,
    ) -> ExactProvenance:
        end = len(self.exact_text) if char_end is None else char_end
        if char_start < 0 or end <= char_start or end > len(self.exact_text):
            raise DormantError("source-reference bounds exceed exact source text")
        return ExactProvenance(
            source_record_id=self.record_id,
            span_id=span_id,
            char_start=char_start,
            char_end=end,
            exact_text_sha256=_sha256_text(self.exact_text[char_start:end]),
            source=self.source,
            provenance=self.provenance,
        )

    @classmethod
    def from_field_span(
        cls,
        span: FieldSpan,
        *,
        stream_id: str,
        sequence: int,
        region: LogicalRegion | str,
        tick_id: int,
        parent_source_id: str | None = None,
    ) -> "SourceRecord":
        return cls(
            stream_id=stream_id,
            sequence=sequence,
            region=region,
            exact_text=span.text,
            source=span.source or "shared_field",
            provenance=span.provenance or f"field_span:{span.span_id}",
            tick_id=tick_id,
            parent_source_id=parent_source_id,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRecord":
        result = cls(
            stream_id=str(value["stream_id"]),
            sequence=int(value["sequence"]),
            region=str(value["region"]),
            exact_text=str(value["exact_text"]),
            source=str(value["source"]),
            provenance=str(value["provenance"]),
            tick_id=int(value["tick_id"]),
            parent_source_id=(
                None
                if value.get("parent_source_id") is None
                else str(value["parent_source_id"])
            ),
        )
        if value.get("record_id", result.record_id) != result.record_id:
            raise DormantConflictError("source record ID does not match content")
        if (
            value.get("exact_utf8_sha256", result.exact_utf8_sha256)
            != result.exact_utf8_sha256
        ):
            raise DormantConflictError("source text hash does not match content")
        return result


@dataclass(frozen=True, slots=True)
class SpanLifecycle:
    span_id: str
    source_record_id: str
    state: str
    reason: str
    tick_id: int
    previous_event_id: str | None = None
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.span_id, "span_id")
        _nonempty(self.source_record_id, "source_record_id")
        if self.state not in SPAN_STATES:
            raise DormantError(f"unknown span lifecycle state {self.state!r}")
        _nonempty(self.reason, "reason")
        _tick(self.tick_id)
        if self.previous_event_id is not None:
            _nonempty(self.previous_event_id, "previous_event_id")
        object.__setattr__(self, "event_id", _content_id("spanlife", self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "span-lifecycle-v1",
            "span_id": self.span_id,
            "source_record_id": self.source_record_id,
            "state": self.state,
            "reason": self.reason,
            "tick_id": self.tick_id,
            "previous_event_id": self.previous_event_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "event_id": self.event_id}


def _revision_payload(
    *,
    schema: str,
    exact_text: str,
    provenance_refs: Sequence[ExactProvenance],
    provenance: str,
    confidence: float,
    parent_revision_id: str | None,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": schema,
        "exact_text": exact_text,
        "provenance_refs": [ref.to_dict() for ref in provenance_refs],
        "provenance": provenance,
        "confidence": confidence,
        "parent_revision_id": parent_revision_id,
        **dict(extra),
    }


@dataclass(frozen=True, slots=True)
class ContainerRevision:
    exact_text: str
    kind: str
    provenance_refs: tuple[ExactProvenance, ...]
    provenance: str
    confidence: float = 1.0
    parent_revision_id: str | None = None
    container_id: str = field(init=False)
    revision_id: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.exact_text, "exact_text")
        _nonempty(self.kind, "kind")
        refs = tuple(self.provenance_refs)
        if not refs or not all(isinstance(ref, ExactProvenance) for ref in refs):
            raise DormantError("container requires exact provenance refs")
        _nonempty(self.provenance, "provenance")
        object.__setattr__(self, "provenance_refs", refs)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "container_id",
            _content_id(
                "container",
                {"kind": self.kind, "text_key": _normal_key(self.exact_text)},
            ),
        )
        object.__setattr__(self, "revision_id", _content_id("containerrev", self.payload()))

    def payload(self) -> dict[str, Any]:
        return _revision_payload(
            schema="container-revision-v1",
            exact_text=self.exact_text,
            provenance_refs=self.provenance_refs,
            provenance=self.provenance,
            confidence=self.confidence,
            parent_revision_id=self.parent_revision_id,
            extra={"kind": self.kind, "container_id": self.container_id},
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "revision_id": self.revision_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContainerRevision":
        result = cls(
            exact_text=str(value["exact_text"]),
            kind=str(value["kind"]),
            provenance_refs=tuple(
                ExactProvenance.from_dict(item)
                for item in value["provenance_refs"]
            ),
            provenance=str(value["provenance"]),
            confidence=float(value["confidence"]),
            parent_revision_id=(
                None
                if value.get("parent_revision_id") is None
                else str(value["parent_revision_id"])
            ),
        )
        if value.get("container_id", result.container_id) != result.container_id:
            raise DormantConflictError("container ID does not match content")
        if value.get("revision_id", result.revision_id) != result.revision_id:
            raise DormantConflictError("container revision ID does not match content")
        return result


@dataclass(frozen=True, slots=True)
class EntityRevision:
    exact_label: str
    entity_type: str
    aliases: tuple[str, ...]
    provenance_refs: tuple[ExactProvenance, ...]
    provenance: str
    confidence: float = 1.0
    parent_revision_id: str | None = None
    entity_id: str = field(init=False)
    revision_id: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.exact_label, "exact_label")
        _nonempty(self.entity_type, "entity_type")
        refs = tuple(self.provenance_refs)
        if not refs:
            raise DormantError("entity requires exact provenance refs")
        aliases = _unique_strings((*self.aliases, self.exact_label))
        _nonempty(self.provenance, "provenance")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "provenance_refs", refs)
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "entity_id",
            _content_id(
                "entity",
                {
                    "label_key": _normal_key(self.exact_label),
                    "entity_type": self.entity_type,
                },
            ),
        )
        object.__setattr__(self, "revision_id", _content_id("entityrev", self.payload()))

    def payload(self) -> dict[str, Any]:
        return _revision_payload(
            schema="entity-revision-v1",
            exact_text=self.exact_label,
            provenance_refs=self.provenance_refs,
            provenance=self.provenance,
            confidence=self.confidence,
            parent_revision_id=self.parent_revision_id,
            extra={
                "entity_id": self.entity_id,
                "entity_type": self.entity_type,
                "aliases": list(self.aliases),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "revision_id": self.revision_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EntityRevision":
        result = cls(
            exact_label=str(value["exact_text"]),
            entity_type=str(value["entity_type"]),
            aliases=tuple(str(item) for item in value.get("aliases", ())),
            provenance_refs=tuple(
                ExactProvenance.from_dict(item)
                for item in value["provenance_refs"]
            ),
            provenance=str(value["provenance"]),
            confidence=float(value["confidence"]),
            parent_revision_id=(
                None
                if value.get("parent_revision_id") is None
                else str(value["parent_revision_id"])
            ),
        )
        if value.get("entity_id", result.entity_id) != result.entity_id:
            raise DormantConflictError("entity ID does not match content")
        if value.get("revision_id", result.revision_id) != result.revision_id:
            raise DormantConflictError("entity revision ID does not match content")
        return result


@dataclass(frozen=True, slots=True)
class TripleRevision:
    subject_entity_id: str
    subject_text: str
    predicate: str
    object_text: str
    provenance_refs: tuple[ExactProvenance, ...]
    provenance: str
    container_revision_ids: tuple[str, ...] = ()
    object_entity_id: str | None = None
    extractor_id: str = BOOTSTRAP_EXTRACTOR_ID
    confidence: float = 0.5
    parent_revision_id: str | None = None
    triple_id: str = field(init=False)
    revision_id: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.subject_entity_id, "subject_entity_id")
        _nonempty(self.subject_text, "subject_text")
        predicate = _normal_key(_nonempty(self.predicate, "predicate"))
        if predicate not in {"is", "has", "uses", "needs", "wants"}:
            raise DormantError("bootstrap triples require an explicit allowed predicate")
        _nonempty(self.object_text, "object_text")
        refs = tuple(self.provenance_refs)
        if not refs:
            raise DormantError("triple requires exact provenance refs")
        _nonempty(self.provenance, "provenance")
        _nonempty(self.extractor_id, "extractor_id")
        object.__setattr__(self, "predicate", predicate)
        object.__setattr__(self, "provenance_refs", refs)
        object.__setattr__(
            self,
            "container_revision_ids",
            _unique_strings(self.container_revision_ids),
        )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "triple_id",
            _content_id(
                "triple",
                {
                    "subject": _normal_key(self.subject_text),
                    "predicate": predicate,
                    "object": _normal_key(self.object_text),
                },
            ),
        )
        object.__setattr__(self, "revision_id", _content_id("triplerev", self.payload()))

    @property
    def readable_text(self) -> str:
        return f"{self.subject_text} {self.predicate} {self.object_text}"

    def payload(self) -> dict[str, Any]:
        return _revision_payload(
            schema="triple-revision-v1",
            exact_text=self.readable_text,
            provenance_refs=self.provenance_refs,
            provenance=self.provenance,
            confidence=self.confidence,
            parent_revision_id=self.parent_revision_id,
            extra={
                "triple_id": self.triple_id,
                "subject_entity_id": self.subject_entity_id,
                "subject_text": self.subject_text,
                "predicate": self.predicate,
                "object_text": self.object_text,
                "object_entity_id": self.object_entity_id,
                "container_revision_ids": list(self.container_revision_ids),
                "extractor_id": self.extractor_id,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "revision_id": self.revision_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TripleRevision":
        result = cls(
            subject_entity_id=str(value["subject_entity_id"]),
            subject_text=str(value["subject_text"]),
            predicate=str(value["predicate"]),
            object_text=str(value["object_text"]),
            object_entity_id=(
                None
                if value.get("object_entity_id") is None
                else str(value["object_entity_id"])
            ),
            container_revision_ids=tuple(
                str(item) for item in value.get("container_revision_ids", ())
            ),
            extractor_id=str(value["extractor_id"]),
            provenance_refs=tuple(
                ExactProvenance.from_dict(item)
                for item in value["provenance_refs"]
            ),
            provenance=str(value["provenance"]),
            confidence=float(value["confidence"]),
            parent_revision_id=(
                None
                if value.get("parent_revision_id") is None
                else str(value["parent_revision_id"])
            ),
        )
        if value.get("triple_id", result.triple_id) != result.triple_id:
            raise DormantConflictError("triple ID does not match content")
        if value.get("revision_id", result.revision_id) != result.revision_id:
            raise DormantConflictError("triple revision ID does not match content")
        return result


@dataclass(frozen=True, slots=True)
class KnowledgeLifecycle:
    target_kind: str
    target_revision_id: str
    state: str
    reason: str
    tick_id: int
    previous_event_id: str | None = None
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.target_kind not in {"container", "entity", "triple"}:
            raise DormantError("target_kind must be container, entity, or triple")
        _nonempty(self.target_revision_id, "target_revision_id")
        if self.state not in KNOWLEDGE_STATES:
            raise DormantError(f"unknown knowledge state {self.state!r}")
        _nonempty(self.reason, "reason")
        _tick(self.tick_id)
        object.__setattr__(self, "event_id", _content_id("knowlife", self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "knowledge-lifecycle-v1",
            "target_kind": self.target_kind,
            "target_revision_id": self.target_revision_id,
            "state": self.state,
            "reason": self.reason,
            "tick_id": self.tick_id,
            "previous_event_id": self.previous_event_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "event_id": self.event_id}


@dataclass(frozen=True, slots=True)
class ExtractionJob:
    source_record_id: str
    extractor_id: str
    ruleset_sha256: str
    state: str
    attempt: int
    previous_event_id: str | None = None
    lease_owner: str = ""
    lease_until_tick: int | None = None
    output_revision_ids: tuple[str, ...] = ()
    detail: str = ""
    job_key: str = field(init=False)
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        _nonempty(self.source_record_id, "source_record_id")
        _nonempty(self.extractor_id, "extractor_id")
        if not re.fullmatch(r"[0-9a-f]{64}", self.ruleset_sha256):
            raise DormantError("ruleset_sha256 must be lowercase SHA-256")
        if self.state not in JOB_STATES:
            raise DormantError(f"unknown extraction job state {self.state!r}")
        _tick(self.attempt, "attempt")
        if self.state == "leased":
            _nonempty(self.lease_owner, "lease_owner")
            if self.lease_until_tick is None:
                raise DormantError("leased job requires lease_until_tick")
            _tick(self.lease_until_tick, "lease_until_tick")
        elif self.lease_until_tick is not None:
            _tick(self.lease_until_tick, "lease_until_tick")
        object.__setattr__(
            self, "output_revision_ids", _unique_strings(self.output_revision_ids)
        )
        key_payload = {
            "source_record_id": self.source_record_id,
            "extractor_id": self.extractor_id,
            "ruleset_sha256": self.ruleset_sha256,
        }
        object.__setattr__(self, "job_key", _content_id("job", key_payload))
        object.__setattr__(self, "event_id", _content_id("jobevent", self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "extraction-job-event-v1",
            "job_key": self.job_key,
            "source_record_id": self.source_record_id,
            "extractor_id": self.extractor_id,
            "ruleset_sha256": self.ruleset_sha256,
            "state": self.state,
            "attempt": self.attempt,
            "previous_event_id": self.previous_event_id,
            "lease_owner": self.lease_owner,
            "lease_until_tick": self.lease_until_tick,
            "output_revision_ids": list(self.output_revision_ids),
            "detail": self.detail,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "event_id": self.event_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExtractionJob":
        result = cls(
            source_record_id=str(value["source_record_id"]),
            extractor_id=str(value["extractor_id"]),
            ruleset_sha256=str(value["ruleset_sha256"]),
            state=str(value["state"]),
            attempt=int(value["attempt"]),
            previous_event_id=(
                None
                if value.get("previous_event_id") is None
                else str(value["previous_event_id"])
            ),
            lease_owner=str(value.get("lease_owner", "")),
            lease_until_tick=(
                None
                if value.get("lease_until_tick") is None
                else int(value["lease_until_tick"])
            ),
            output_revision_ids=tuple(
                str(item) for item in value.get("output_revision_ids", ())
            ),
            detail=str(value.get("detail", "")),
        )
        if value.get("job_key", result.job_key) != result.job_key:
            raise DormantConflictError("job key does not match content")
        if value.get("event_id", result.event_id) != result.event_id:
            raise DormantConflictError("job event ID does not match content")
        return result


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    triple_revision_id: str
    text: str
    score: int
    container_refs: tuple[str, ...]
    edge_refs: tuple[str, ...]
    provenance_refs: tuple[ExactProvenance, ...]


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    hits: tuple[RetrievalHit, ...]
    omitted_revision_ids: tuple[str, ...]
    omitted_sha256: str


@dataclass(frozen=True, slots=True)
class Surfacing:
    query: str
    source_field_id: str
    retrieved_revision_ids: tuple[str, ...]
    omitted_revision_ids: tuple[str, ...]
    omitted_sha256: str
    output_span_ids: tuple[str, ...]
    record_limit: int
    provenance: str
    surfacing_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise DormantError("query must be a string")
        _nonempty(self.source_field_id, "source_field_id")
        _tick(self.record_limit, "record_limit")
        _nonempty(self.provenance, "provenance")
        object.__setattr__(
            self,
            "retrieved_revision_ids",
            _ordered_unique_strings(self.retrieved_revision_ids),
        )
        object.__setattr__(
            self, "omitted_revision_ids", _unique_strings(self.omitted_revision_ids)
        )
        object.__setattr__(self, "output_span_ids", _unique_strings(self.output_span_ids))
        object.__setattr__(self, "surfacing_id", _content_id("surface", self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "surfacing-record-v1",
            "query": self.query,
            "source_field_id": self.source_field_id,
            "retrieved_revision_ids": list(self.retrieved_revision_ids),
            "omitted_revision_ids": list(self.omitted_revision_ids),
            "omitted_sha256": self.omitted_sha256,
            "output_span_ids": list(self.output_span_ids),
            "record_limit": self.record_limit,
            "provenance": self.provenance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "surfacing_id": self.surfacing_id}


@dataclass(frozen=True, slots=True)
class ExtractionBundle:
    source_record_id: str
    extractor_id: str
    containers: tuple[ContainerRevision, ...]
    entities: tuple[EntityRevision, ...]
    triples: tuple[TripleRevision, ...]

    @property
    def output_revision_ids(self) -> tuple[str, ...]:
        return _unique_strings(
            [
                *(item.revision_id for item in self.containers),
                *(item.revision_id for item in self.entities),
                *(item.revision_id for item in self.triples),
            ]
        )


_AUTHORITATIVE_TABLES = (
    "source_records",
    "span_lifecycle_events",
    "container_revisions",
    "entity_revisions",
    "triple_revisions",
    "knowledge_lifecycle_events",
    "extraction_job_events",
    "surfacing_records",
)


def ensure_dormant_schema(connection: sqlite3.Connection) -> None:
    """Create insert-only authoritative tables and protective triggers."""

    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS dormant_schema_meta (
            version TEXT PRIMARY KEY
        );
        CREATE TABLE IF NOT EXISTS source_records (
            record_id TEXT PRIMARY KEY,
            stream_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            region TEXT NOT NULL,
            exact_bytes BLOB NOT NULL,
            exact_sha256 TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            UNIQUE(stream_id, sequence)
        );
        CREATE TABLE IF NOT EXISTS span_lifecycle_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            span_id TEXT NOT NULL,
            source_record_id TEXT NOT NULL,
            state TEXT NOT NULL,
            tick_id INTEGER NOT NULL,
            previous_event_id TEXT,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS span_lifecycle_latest
            ON span_lifecycle_events(span_id, seq DESC);
        CREATE TABLE IF NOT EXISTS container_revisions (
            revision_id TEXT PRIMARY KEY,
            container_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS entity_revisions (
            revision_id TEXT PRIMARY KEY,
            entity_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS triple_revisions (
            revision_id TEXT PRIMARY KEY,
            triple_id TEXT NOT NULL,
            subject_entity_id TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_text TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS knowledge_lifecycle_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            target_kind TEXT NOT NULL,
            target_revision_id TEXT NOT NULL,
            state TEXT NOT NULL,
            tick_id INTEGER NOT NULL,
            previous_event_id TEXT,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS knowledge_lifecycle_latest
            ON knowledge_lifecycle_events(target_revision_id, seq DESC);
        CREATE TABLE IF NOT EXISTS extraction_job_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,
            job_key TEXT NOT NULL,
            source_record_id TEXT NOT NULL,
            state TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            lease_owner TEXT NOT NULL,
            lease_until_tick INTEGER,
            previous_event_id TEXT,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS extraction_jobs_latest
            ON extraction_job_events(job_key, seq DESC);
        CREATE TABLE IF NOT EXISTS surfacing_records (
            surfacing_id TEXT PRIMARY KEY,
            source_field_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT OR IGNORE INTO dormant_schema_meta(version) VALUES (?)",
        (DORMANT_SCHEMA_VERSION,),
    )
    for table in _AUTHORITATIVE_TABLES:
        connection.executescript(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table}_no_update
            BEFORE UPDATE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'authoritative table is insert-only');
            END;
            CREATE TRIGGER IF NOT EXISTS {table}_no_delete
            BEFORE DELETE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'authoritative table is insert-only');
            END;
            """
        )
    connection.commit()


def _payload_sha(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


class DormantStore:
    """Small insert-only SQLite facade used by the runtime vertical slice."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        ensure_dormant_schema(connection)

    def _append(
        self,
        table: str,
        id_column: str,
        identifier: str,
        payload: Mapping[str, Any],
        columns: Mapping[str, Any],
    ) -> bool:
        payload_json = _canonical_text(payload)
        payload_sha256 = _payload_sha(payload_json)
        existing = self.connection.execute(
            f"SELECT payload_json, payload_sha256 FROM {table} "
            f"WHERE {id_column}=?",
            (identifier,),
        ).fetchone()
        if existing is not None:
            if existing[0] == payload_json and existing[1] == payload_sha256:
                return False
            raise DormantConflictError(
                f"{table} immutable ID {identifier!r} has conflicting content"
            )
        names = [id_column, *columns.keys(), "payload_json", "payload_sha256"]
        values = [identifier, *columns.values(), payload_json, payload_sha256]
        placeholders = ",".join("?" for _ in names)
        self.connection.execute(
            f"INSERT INTO {table}({','.join(names)}) VALUES ({placeholders})",
            values,
        )
        return True

    def append_source(self, record: SourceRecord) -> bool:
        with self.connection:
            by_position = self.connection.execute(
                "SELECT record_id FROM source_records "
                "WHERE stream_id=? AND sequence=?",
                (record.stream_id, record.sequence),
            ).fetchone()
            if by_position is not None and by_position[0] != record.record_id:
                raise DormantConflictError(
                    "source stream position already contains different content"
                )
            return self._append(
                "source_records",
                "record_id",
                record.record_id,
                record.to_dict(),
                {
                    "stream_id": record.stream_id,
                    "sequence": record.sequence,
                    "region": record.region.value,
                    "exact_bytes": record.exact_text.encode("utf-8"),
                    "exact_sha256": record.exact_utf8_sha256,
                },
            )

    def get_source(self, record_id: str) -> SourceRecord:
        row = self.connection.execute(
            "SELECT payload_json, exact_bytes, exact_sha256 "
            "FROM source_records WHERE record_id=?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(record_id)
        value = json.loads(row[0])
        record = SourceRecord.from_dict(value)
        exact_bytes = bytes(row[1])
        if exact_bytes != record.exact_text.encode("utf-8"):
            raise DormantConflictError("source BLOB differs from canonical payload")
        if row[2] != record.exact_utf8_sha256:
            raise DormantConflictError("source BLOB hash differs from canonical payload")
        return record

    def source_at(
        self,
        stream_id: str,
        sequence: int,
    ) -> SourceRecord | None:
        """Return one exact source position, preserving retry identity."""

        row = self.connection.execute(
            """
            SELECT record_id
            FROM source_records
            WHERE stream_id=? AND sequence=?
            """,
            (stream_id, sequence),
        ).fetchone()
        return None if row is None else self.get_source(str(row[0]))

    def _validate_refs(self, refs: Sequence[ExactProvenance]) -> None:
        for ref in refs:
            source = self.get_source(ref.source_record_id)
            if ref.char_end > len(source.exact_text):
                raise DormantConflictError("provenance range exceeds source")
            exact = source.exact_text[ref.char_start : ref.char_end]
            if _sha256_text(exact) != ref.exact_text_sha256:
                raise DormantConflictError("provenance slice hash mismatch")
            if ref.source != source.source or ref.provenance != source.provenance:
                raise DormantConflictError("provenance source metadata mismatch")

    def latest_span_lifecycle(self, span_id: str) -> SpanLifecycle | None:
        row = self.connection.execute(
            "SELECT payload_json FROM span_lifecycle_events "
            "WHERE span_id=? ORDER BY seq DESC LIMIT 1",
            (span_id,),
        ).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        return SpanLifecycle(
            span_id=value["span_id"],
            source_record_id=value["source_record_id"],
            state=value["state"],
            reason=value["reason"],
            tick_id=int(value["tick_id"]),
            previous_event_id=value.get("previous_event_id"),
        )

    def append_span_lifecycle(self, event: SpanLifecycle) -> bool:
        with self.connection:
            existing = self.connection.execute(
                "SELECT payload_json FROM span_lifecycle_events WHERE event_id=?",
                (event.event_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] == _canonical_text(event.to_dict()):
                    return False
                raise DormantConflictError("span lifecycle event conflicts")
            self.get_source(event.source_record_id)
            latest = self.latest_span_lifecycle(event.span_id)
            if latest is not None:
                if event.previous_event_id != latest.event_id:
                    raise DormantTransitionError(
                        "span lifecycle event does not extend the latest event"
                    )
                if event.state not in _SPAN_TRANSITIONS[latest.state]:
                    raise DormantTransitionError(
                        f"illegal span transition {latest.state}->{event.state}"
                    )
            elif event.previous_event_id is not None:
                raise DormantTransitionError(
                    "first span lifecycle event cannot name a predecessor"
                )
            return self._append(
                "span_lifecycle_events",
                "event_id",
                event.event_id,
                event.to_dict(),
                {
                    "span_id": event.span_id,
                    "source_record_id": event.source_record_id,
                    "state": event.state,
                    "tick_id": event.tick_id,
                    "previous_event_id": event.previous_event_id,
                },
            )

    def transition_span(
        self,
        *,
        span_id: str,
        source_record_id: str,
        state: str,
        reason: str,
        tick_id: int,
    ) -> SpanLifecycle:
        latest = self.latest_span_lifecycle(span_id)
        event = SpanLifecycle(
            span_id=span_id,
            source_record_id=source_record_id,
            state=state,
            reason=reason,
            tick_id=tick_id,
            previous_event_id=None if latest is None else latest.event_id,
        )
        self.append_span_lifecycle(event)
        return event

    def _append_container_raw(self, item: ContainerRevision) -> bool:
        self._validate_refs(item.provenance_refs)
        return self._append(
            "container_revisions",
            "revision_id",
            item.revision_id,
            item.to_dict(),
            {"container_id": item.container_id},
        )

    def append_container(self, item: ContainerRevision) -> bool:
        with self.connection:
            return self._append_container_raw(item)

    def _append_entity_raw(self, item: EntityRevision) -> bool:
        self._validate_refs(item.provenance_refs)
        return self._append(
            "entity_revisions",
            "revision_id",
            item.revision_id,
            item.to_dict(),
            {"entity_id": item.entity_id},
        )

    def append_entity(self, item: EntityRevision) -> bool:
        with self.connection:
            return self._append_entity_raw(item)

    def _append_triple_raw(self, item: TripleRevision) -> bool:
        self._validate_refs(item.provenance_refs)
        subject = self.connection.execute(
            "SELECT 1 FROM entity_revisions WHERE entity_id=? LIMIT 1",
            (item.subject_entity_id,),
        ).fetchone()
        if subject is None:
            raise DormantConflictError(
                "triple subject_entity_id has no entity revision"
            )
        if item.object_entity_id is not None:
            object_entity = self.connection.execute(
                "SELECT 1 FROM entity_revisions WHERE entity_id=? LIMIT 1",
                (item.object_entity_id,),
            ).fetchone()
            if object_entity is None:
                raise DormantConflictError(
                    "triple object_entity_id has no entity revision"
                )
        for revision_id in item.container_revision_ids:
            container = self.connection.execute(
                "SELECT 1 FROM container_revisions WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            if container is None:
                raise DormantConflictError(
                    "triple container_revision_ids contains an unknown revision"
                )
        return self._append(
            "triple_revisions",
            "revision_id",
            item.revision_id,
            item.to_dict(),
            {
                "triple_id": item.triple_id,
                "subject_entity_id": item.subject_entity_id,
                "predicate": item.predicate,
                "object_text": item.object_text,
            },
        )

    def append_triple(self, item: TripleRevision) -> bool:
        with self.connection:
            return self._append_triple_raw(item)

    def append_extraction_bundle(self, bundle: ExtractionBundle) -> None:
        with self.connection:
            for item in bundle.containers:
                self._append_container_raw(item)
            for item in bundle.entities:
                self._append_entity_raw(item)
            for item in bundle.triples:
                self._append_triple_raw(item)

    def _latest_knowledge(
        self, target_revision_id: str
    ) -> KnowledgeLifecycle | None:
        row = self.connection.execute(
            "SELECT payload_json FROM knowledge_lifecycle_events "
            "WHERE target_revision_id=? ORDER BY seq DESC LIMIT 1",
            (target_revision_id,),
        ).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        return KnowledgeLifecycle(
            target_kind=value["target_kind"],
            target_revision_id=value["target_revision_id"],
            state=value["state"],
            reason=value["reason"],
            tick_id=int(value["tick_id"]),
            previous_event_id=value.get("previous_event_id"),
        )

    def latest_knowledge(
        self, target_revision_id: str
    ) -> KnowledgeLifecycle | None:
        """Return the newest immutable lifecycle event for one revision."""

        return self._latest_knowledge(target_revision_id)

    def transition_knowledge(
        self,
        *,
        target_kind: str,
        target_revision_id: str,
        state: str,
        reason: str,
        tick_id: int,
    ) -> KnowledgeLifecycle:
        latest = self._latest_knowledge(target_revision_id)
        event = KnowledgeLifecycle(
            target_kind=target_kind,
            target_revision_id=target_revision_id,
            state=state,
            reason=reason,
            tick_id=tick_id,
            previous_event_id=None if latest is None else latest.event_id,
        )
        with self.connection:
            if latest is not None:
                if event.state not in _KNOWLEDGE_TRANSITIONS[latest.state]:
                    raise DormantTransitionError(
                        f"illegal knowledge transition {latest.state}->{event.state}"
                    )
            elif state != "candidate":
                raise DormantTransitionError(
                    "knowledge lifecycle must begin as candidate"
                )
            self._append(
                "knowledge_lifecycle_events",
                "event_id",
                event.event_id,
                event.to_dict(),
                {
                    "target_kind": event.target_kind,
                    "target_revision_id": event.target_revision_id,
                    "state": event.state,
                    "tick_id": event.tick_id,
                    "previous_event_id": event.previous_event_id,
                },
            )
        return event

    def enqueue_extraction(
        self,
        source_record_id: str,
        *,
        extractor_id: str = BOOTSTRAP_EXTRACTOR_ID,
        ruleset_sha256: str = BOOTSTRAP_RULESET_SHA256,
    ) -> ExtractionJob:
        self.get_source(source_record_id)
        seed = ExtractionJob(
            source_record_id=source_record_id,
            extractor_id=extractor_id,
            ruleset_sha256=ruleset_sha256,
            state="queued",
            attempt=0,
            detail="idle extraction queued",
        )
        latest = self.latest_job(seed.job_key)
        if latest is not None:
            return latest
        with self.connection:
            self._append_job_raw(seed)
        return seed

    def _append_job_raw(self, event: ExtractionJob) -> bool:
        return self._append(
            "extraction_job_events",
            "event_id",
            event.event_id,
            event.to_dict(),
            {
                "job_key": event.job_key,
                "source_record_id": event.source_record_id,
                "state": event.state,
                "attempt": event.attempt,
                "lease_owner": event.lease_owner,
                "lease_until_tick": event.lease_until_tick,
                "previous_event_id": event.previous_event_id,
            },
        )

    def latest_job(self, job_key: str) -> ExtractionJob | None:
        row = self.connection.execute(
            "SELECT payload_json FROM extraction_job_events "
            "WHERE job_key=? ORDER BY seq DESC LIMIT 1",
            (job_key,),
        ).fetchone()
        return None if row is None else ExtractionJob.from_dict(json.loads(row[0]))

    def _latest_jobs(self) -> list[ExtractionJob]:
        rows = self.connection.execute(
            """
            SELECT e.payload_json
            FROM extraction_job_events e
            JOIN (
                SELECT job_key, MAX(seq) AS max_seq, MIN(seq) AS first_seq
                FROM extraction_job_events GROUP BY job_key
            ) latest ON latest.max_seq=e.seq
            ORDER BY latest.first_seq, e.job_key
            """
        ).fetchall()
        return [ExtractionJob.from_dict(json.loads(row[0])) for row in rows]

    def recover_expired_jobs(
        self, *, now_tick: int, limit: int
    ) -> tuple[ExtractionJob, ...]:
        _tick(now_tick, "now_tick")
        if limit <= 0:
            raise DormantError("recovery limit must be positive")
        expired = [
            job
            for job in self._latest_jobs()
            if job.state == "leased"
            and job.lease_until_tick is not None
            and job.lease_until_tick <= now_tick
        ][:limit]
        recovered: list[ExtractionJob] = []
        with self.connection:
            for job in expired:
                event = ExtractionJob(
                    source_record_id=job.source_record_id,
                    extractor_id=job.extractor_id,
                    ruleset_sha256=job.ruleset_sha256,
                    state="queued",
                    attempt=job.attempt + 1,
                    previous_event_id=job.event_id,
                    detail="expired lease recovered",
                )
                self._append_job_raw(event)
                recovered.append(event)
        return tuple(recovered)

    def lease_jobs(
        self,
        *,
        owner: str,
        now_tick: int,
        lease_ticks: int,
        limit: int,
    ) -> tuple[ExtractionJob, ...]:
        _nonempty(owner, "owner")
        _tick(now_tick, "now_tick")
        if lease_ticks <= 0 or limit <= 0:
            raise DormantError("lease_ticks and limit must be positive")
        self.recover_expired_jobs(now_tick=now_tick, limit=limit)
        queued = [job for job in self._latest_jobs() if job.state == "queued"][:limit]
        leased: list[ExtractionJob] = []
        with self.connection:
            for job in queued:
                event = ExtractionJob(
                    source_record_id=job.source_record_id,
                    extractor_id=job.extractor_id,
                    ruleset_sha256=job.ruleset_sha256,
                    state="leased",
                    attempt=job.attempt,
                    previous_event_id=job.event_id,
                    lease_owner=owner,
                    lease_until_tick=now_tick + lease_ticks,
                    detail="bounded idle lease",
                )
                self._append_job_raw(event)
                leased.append(event)
        return tuple(leased)

    def complete_job(
        self,
        job_key: str,
        *,
        owner: str,
        output_revision_ids: Sequence[str],
    ) -> ExtractionJob:
        latest = self.latest_job(job_key)
        if latest is None:
            raise KeyError(job_key)
        if latest.state != "leased" or latest.lease_owner != owner:
            raise DormantTransitionError("only the current lease owner may complete")
        event = ExtractionJob(
            source_record_id=latest.source_record_id,
            extractor_id=latest.extractor_id,
            ruleset_sha256=latest.ruleset_sha256,
            state="completed",
            attempt=latest.attempt,
            previous_event_id=latest.event_id,
            output_revision_ids=tuple(output_revision_ids),
            detail="extraction committed",
        )
        with self.connection:
            self._append_job_raw(event)
        return event

    def append_surfacing(self, record: Surfacing) -> bool:
        with self.connection:
            return self._append(
                "surfacing_records",
                "surfacing_id",
                record.surfacing_id,
                record.to_dict(),
                {"source_field_id": record.source_field_id},
            )

    def retrieve(
        self,
        query: str,
        *,
        limit: int,
        accepted_only: bool = True,
    ) -> RetrievalResult:
        if limit < 0:
            raise DormantError("retrieval limit must be non-negative")
        terms = set(re.findall(r"\w+", query.casefold(), flags=re.UNICODE))
        candidates: list[RetrievalHit] = []
        rows = self.connection.execute(
            "SELECT payload_json FROM triple_revisions ORDER BY revision_id"
        ).fetchall()
        for row in rows:
            triple = TripleRevision.from_dict(json.loads(row[0]))
            latest = self._latest_knowledge(triple.revision_id)
            if accepted_only and (latest is None or latest.state != "accepted"):
                continue
            text = triple.readable_text
            score = len(
                terms
                & set(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))
            )
            if score <= 0:
                continue
            candidates.append(
                RetrievalHit(
                    triple_revision_id=triple.revision_id,
                    text=text,
                    score=score,
                    container_refs=triple.container_revision_ids,
                    edge_refs=(triple.revision_id,),
                    provenance_refs=triple.provenance_refs,
                )
            )
        candidates.sort(
            key=lambda item: (-item.score, _normal_key(item.text), item.triple_revision_id)
        )
        admitted = tuple(candidates[:limit])
        omitted = tuple(
            sorted(item.triple_revision_id for item in candidates[limit:])
        )
        return RetrievalResult(
            query=query,
            hits=admitted,
            omitted_revision_ids=omitted,
            omitted_sha256=hashlib.sha256(
                canonical_json_bytes(list(omitted))
            ).hexdigest(),
        )


_CLAUSE_RE = re.compile(r"[^.\r\n!?]+")
_RELATION_RE = re.compile(
    r"^\s*(?P<subject>.+?)\s+"
    r"(?P<predicate>is|has|uses|needs|wants)\s+"
    r"(?P<object>.+?)\s*$",
    flags=re.IGNORECASE,
)


def _trimmed_group_bounds(
    match: re.Match[str], group: str, clause_start: int
) -> tuple[str, int, int]:
    raw = match.group(group)
    left = len(raw) - len(raw.lstrip())
    right = len(raw.rstrip())
    text = raw[left:right]
    start = clause_start + match.start(group) + left
    return text, start, start + len(text)


def bootstrap_extract(source: SourceRecord) -> ExtractionBundle:
    """Derive only explicit readable relation candidates from exact source.

    Every output points back to an exact character slice.  No source byte is
    normalized, replaced, clipped, or deleted.  The candidates remain
    ``not-semantic-proof`` until an explicit knowledge lifecycle accepts them.
    """

    containers: dict[str, ContainerRevision] = {}
    entities: dict[str, EntityRevision] = {}
    triples: dict[str, TripleRevision] = {}
    for clause_index, clause_match in enumerate(_CLAUSE_RE.finditer(source.exact_text)):
        raw_clause = clause_match.group(0)
        left = len(raw_clause) - len(raw_clause.lstrip())
        right = len(raw_clause.rstrip())
        if right <= left:
            continue
        clause_start = clause_match.start() + left
        clause_end = clause_match.start() + right
        clause = source.exact_text[clause_start:clause_end]
        relation = _RELATION_RE.fullmatch(clause)
        if relation is None:
            continue
        subject, subject_start, subject_end = _trimmed_group_bounds(
            relation, "subject", clause_start
        )
        object_text, object_start, object_end = _trimmed_group_bounds(
            relation, "object", clause_start
        )
        if not subject or not object_text or len(subject) > 256 or len(object_text) > 256:
            continue
        predicate = relation.group("predicate").casefold()
        clause_ref = source.exact_ref(
            f"{source.record_id}:clause:{clause_index}",
            clause_start,
            clause_end,
        )
        subject_ref = source.exact_ref(
            f"{source.record_id}:subject:{clause_index}",
            subject_start,
            subject_end,
        )
        object_ref = source.exact_ref(
            f"{source.record_id}:object:{clause_index}",
            object_start,
            object_end,
        )
        label = BOOTSTRAP_EXTRACTOR_ID
        container = ContainerRevision(
            exact_text=clause,
            kind="bootstrap_explicit_relation",
            provenance_refs=(clause_ref,),
            provenance=label,
            confidence=0.5,
        )
        subject_entity = EntityRevision(
            exact_label=subject,
            entity_type="bootstrap_mention",
            aliases=(subject,),
            provenance_refs=(subject_ref,),
            provenance=label,
            confidence=0.5,
        )
        object_entity = EntityRevision(
            exact_label=object_text,
            entity_type="bootstrap_mention",
            aliases=(object_text,),
            provenance_refs=(object_ref,),
            provenance=label,
            confidence=0.5,
        )
        triple = TripleRevision(
            subject_entity_id=subject_entity.entity_id,
            subject_text=subject,
            predicate=predicate,
            object_text=object_text,
            object_entity_id=object_entity.entity_id,
            container_revision_ids=(container.revision_id,),
            provenance_refs=(clause_ref,),
            provenance=label,
            extractor_id=label,
            confidence=0.5,
        )
        containers[container.revision_id] = container
        entities[subject_entity.revision_id] = subject_entity
        entities[object_entity.revision_id] = object_entity
        triples[triple.revision_id] = triple
    return ExtractionBundle(
        source_record_id=source.record_id,
        extractor_id=BOOTSTRAP_EXTRACTOR_ID,
        containers=tuple(containers.values()),
        entities=tuple(entities.values()),
        triples=tuple(triples.values()),
    )


__all__ = [
    "DORMANT_SCHEMA_VERSION",
    "BOOTSTRAP_EXTRACTOR_ID",
    "BOOTSTRAP_RULESET_SHA256",
    "DormantError",
    "DormantConflictError",
    "DormantTransitionError",
    "ExactProvenance",
    "SourceRecord",
    "SpanLifecycle",
    "ContainerRevision",
    "EntityRevision",
    "TripleRevision",
    "KnowledgeLifecycle",
    "ExtractionJob",
    "RetrievalHit",
    "RetrievalResult",
    "Surfacing",
    "ExtractionBundle",
    "ensure_dormant_schema",
    "DormantStore",
    "bootstrap_extract",
]
