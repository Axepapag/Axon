"""Immutable canonical shared-field schema.

The canonical field is a logical, auditable document.  It is deliberately not
limited to the current 384-position model window; that limit belongs to the
compiled :class:`~runtime.field.view.FieldView`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Mapping

SCHEMA_VERSION = "shared-field-v1"


class LogicalRegion(str, Enum):
    """Canonical logical regions directly represented in the active field."""

    CONVERSATION_HISTORY = "conversation_history"
    USER_INPUT = "user_input"
    STRUCTURED_KNOWLEDGE = "structured_knowledge"
    SITUATION_AWARENESS = "situation_awareness"
    TOOL_RESULTS = "tool_results"
    ADVISOR_INPUT = "advisor_input"
    TASK_STATE = "task_state"
    SCRATCH = "scratch"
    RESPONSE_DRAFT = "response_draft"
    DIARY = "diary"


CANONICAL_REGION_ORDER: tuple[LogicalRegion, ...] = (
    LogicalRegion.CONVERSATION_HISTORY,
    LogicalRegion.USER_INPUT,
    LogicalRegion.STRUCTURED_KNOWLEDGE,
    LogicalRegion.SITUATION_AWARENESS,
    LogicalRegion.TOOL_RESULTS,
    LogicalRegion.ADVISOR_INPUT,
    LogicalRegion.TASK_STATE,
    LogicalRegion.SCRATCH,
    LogicalRegion.RESPONSE_DRAFT,
    LogicalRegion.DIARY,
)

LOGICAL_REGION_IDS: Mapping[LogicalRegion, int] = MappingProxyType(
    {region: index for index, region in enumerate(CANONICAL_REGION_ORDER)}
)


class RegionVisibility(str, Enum):
    ATTENDED = "attended"
    MASKED = "masked"


class WritePolicy(str, Enum):
    SEALED = "sealed"
    CORE_WRITABLE = "core_writable"


class PhysicalRole(IntEnum):
    """Checkpoint-compatible learned ``char_type_emb`` row numbers."""

    CONTEXT = 0
    USER = 1
    PROPOSAL = 2


CORE_WRITABLE_REGIONS: frozenset[LogicalRegion] = frozenset(
    {
        LogicalRegion.SCRATCH,
        LogicalRegion.RESPONSE_DRAFT,
    }
)


def _as_logical_region(value: LogicalRegion | str) -> LogicalRegion:
    if isinstance(value, LogicalRegion):
        return value
    try:
        return LogicalRegion(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"unknown logical region {value!r}; dormant state is surfaced into "
            "an active logical region before it can be attended"
        ) from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-safe value with one stable byte representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class FieldSpan:
    """One immutable, provenance-bearing run of canonical characters."""

    span_id: str
    text: str
    kind: str = "text"
    source: str = ""
    provenance: str = ""
    confidence: float = 1.0
    container_refs: tuple[str, ...] = ()
    edge_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.span_id, str) or not self.span_id:
            raise ValueError("FieldSpan.span_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("FieldSpan.text must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("FieldSpan.kind must be a non-empty string")
        if not isinstance(self.source, str):
            raise TypeError("FieldSpan.source must be a string")
        if not isinstance(self.provenance, str):
            raise TypeError("FieldSpan.provenance must be a string")
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("FieldSpan.confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(
            self,
            "container_refs",
            tuple(sorted(set(str(ref) for ref in self.container_refs))),
        )
        object.__setattr__(
            self,
            "edge_refs",
            tuple(sorted(set(str(ref) for ref in self.edge_refs))),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "text": self.text,
            "kind": self.kind,
            "source": self.source,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "container_refs": list(self.container_refs),
            "edge_refs": list(self.edge_refs),
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class RegionState:
    """Immutable state of one canonical logical region."""

    name: LogicalRegion | str
    spans: tuple[FieldSpan, ...] = ()
    visibility: RegionVisibility | str = RegionVisibility.ATTENDED
    write_policy: WritePolicy | str | None = None

    def __post_init__(self) -> None:
        name = _as_logical_region(self.name)
        object.__setattr__(self, "name", name)

        spans = tuple(self.spans)
        if not all(isinstance(span, FieldSpan) for span in spans):
            raise TypeError("RegionState.spans must contain only FieldSpan values")
        span_ids = [span.span_id for span in spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError(f"duplicate span_id in region {name.value!r}")
        object.__setattr__(self, "spans", spans)

        visibility = (
            self.visibility
            if isinstance(self.visibility, RegionVisibility)
            else RegionVisibility(self.visibility)
        )
        object.__setattr__(self, "visibility", visibility)

        policy = self.write_policy
        if policy is None:
            policy = (
                WritePolicy.CORE_WRITABLE
                if name in CORE_WRITABLE_REGIONS
                else WritePolicy.SEALED
            )
        elif not isinstance(policy, WritePolicy):
            policy = WritePolicy(policy)
        if policy is WritePolicy.CORE_WRITABLE and name not in CORE_WRITABLE_REGIONS:
            raise ValueError(f"logical region {name.value!r} is always sealed")
        object.__setattr__(self, "write_policy", policy)

    @property
    def text(self) -> str:
        """The complete canonical string; no view/window truncation applies."""

        return "".join(span.text for span in self.spans)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "visibility": self.visibility.value,
            "write_policy": self.write_policy.value,
            "spans": [span.to_canonical_dict() for span in self.spans],
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    @classmethod
    def from_text(
        cls,
        name: LogicalRegion | str,
        text: str,
        *,
        span_id: str | None = None,
        kind: str = "text",
        source: str = "",
        provenance: str = "",
        confidence: float = 1.0,
        container_refs: tuple[str, ...] = (),
        edge_refs: tuple[str, ...] = (),
        visibility: RegionVisibility | str = RegionVisibility.ATTENDED,
        write_policy: WritePolicy | str | None = None,
    ) -> "RegionState":
        logical_name = _as_logical_region(name)
        spans: tuple[FieldSpan, ...]
        if text:
            spans = (
                FieldSpan(
                    span_id=span_id or f"{logical_name.value}:0",
                    text=text,
                    kind=kind,
                    source=source,
                    provenance=provenance,
                    confidence=confidence,
                    container_refs=container_refs,
                    edge_refs=edge_refs,
                ),
            )
        else:
            spans = ()
        return cls(
            name=logical_name,
            spans=spans,
            visibility=visibility,
            write_policy=write_policy,
        )


@dataclass(frozen=True, slots=True)
class SharedFieldSnapshot:
    """A complete immutable field state linked to its parent by hash."""

    tick_id: int
    regions: tuple[RegionState, ...] = ()
    parent_field_id: str | None = None
    source_manifest_ids: tuple[str, ...] = ()
    field_id: str = field(init=False)
    canonical_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.tick_id, bool) or not isinstance(self.tick_id, int):
            raise TypeError("SharedFieldSnapshot.tick_id must be an integer")
        if self.tick_id < 0:
            raise ValueError("SharedFieldSnapshot.tick_id must be non-negative")
        if self.parent_field_id is not None and (
            not isinstance(self.parent_field_id, str) or not self.parent_field_id
        ):
            raise ValueError("parent_field_id must be None or a non-empty string")

        supplied: dict[LogicalRegion, RegionState] = {}
        for region_state in tuple(self.regions):
            if not isinstance(region_state, RegionState):
                raise TypeError(
                    "SharedFieldSnapshot.regions must contain RegionState values"
                )
            if region_state.name in supplied:
                raise ValueError(
                    f"duplicate logical region {region_state.name.value!r}"
                )
            supplied[region_state.name] = region_state
        normalized = tuple(
            supplied.get(region, RegionState(name=region))
            for region in CANONICAL_REGION_ORDER
        )
        object.__setattr__(self, "regions", normalized)

        manifests = tuple(
            sorted(set(str(item) for item in self.source_manifest_ids))
        )
        if any(not item for item in manifests):
            raise ValueError("source_manifest_ids cannot contain empty values")
        object.__setattr__(self, "source_manifest_ids", manifests)

        digest = canonical_sha256(self.to_canonical_dict())
        object.__setattr__(self, "field_id", digest)
        object.__setattr__(self, "canonical_hash", digest)

    def region(self, name: LogicalRegion | str) -> RegionState:
        logical_name = _as_logical_region(name)
        return self.regions[LOGICAL_REGION_IDS[logical_name]]

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "tick_id": self.tick_id,
            "parent_field_id": self.parent_field_id,
            "source_manifest_ids": list(self.source_manifest_ids),
            "regions": [region.to_canonical_dict() for region in self.regions],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["field_id"] = self.field_id
        value["canonical_hash"] = self.canonical_hash
        return value

    @classmethod
    def empty(
        cls,
        *,
        tick_id: int = 0,
        parent_field_id: str | None = None,
        source_manifest_ids: tuple[str, ...] = (),
    ) -> "SharedFieldSnapshot":
        return cls(
            tick_id=tick_id,
            parent_field_id=parent_field_id,
            source_manifest_ids=source_manifest_ids,
        )

    @classmethod
    def from_texts(
        cls,
        texts: Mapping[LogicalRegion | str, str],
        *,
        tick_id: int = 0,
        parent_field_id: str | None = None,
        source_manifest_ids: tuple[str, ...] = (),
        source: str = "",
        provenance: str = "",
    ) -> "SharedFieldSnapshot":
        normalized: dict[LogicalRegion, str] = {}
        for raw_name, text in texts.items():
            name = _as_logical_region(raw_name)
            if name in normalized:
                raise ValueError(f"duplicate logical region {name.value!r}")
            normalized[name] = text
        regions = tuple(
            RegionState.from_text(
                name,
                text,
                source=source,
                provenance=provenance,
            )
            for name, text in normalized.items()
        )
        return cls(
            tick_id=tick_id,
            regions=regions,
            parent_field_id=parent_field_id,
            source_manifest_ids=source_manifest_ids,
        )


__all__ = [
    "SCHEMA_VERSION",
    "LogicalRegion",
    "CANONICAL_REGION_ORDER",
    "LOGICAL_REGION_IDS",
    "RegionVisibility",
    "WritePolicy",
    "PhysicalRole",
    "CORE_WRITABLE_REGIONS",
    "FieldSpan",
    "RegionState",
    "SharedFieldSnapshot",
    "canonical_json_bytes",
    "canonical_sha256",
]
