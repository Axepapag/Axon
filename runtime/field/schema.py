"""Immutable canonical shared-field schema.

The canonical regional body is a logical, auditable document with no character
ceiling.  Physical pages and packed rows are processing units only; neither the
Shared Field view nor any compiled rail may silently omit an unmasked cell.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Iterator, Mapping

LEGACY_SCHEMA_VERSION = "shared-field-v1"
CORTEX_SCHEMA_VERSION = "shared-field-v2"
SCHEMA_VERSION = "shared-field-v3"
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset(
    {LEGACY_SCHEMA_VERSION, CORTEX_SCHEMA_VERSION, SCHEMA_VERSION}
)
LEGACY_CORTEX_REGION_NAME = "structured_knowledge"


class LogicalRegion(str, Enum):
    """Canonical logical regions directly represented in the active field."""

    CONVERSATION_HISTORY = "conversation_history"
    USER_INPUT = "user_input"
    CORTEX = "cortex"
    SITUATION_AWARENESS = "situation_awareness"
    TOOL_RESULTS = "tool_results"
    ADVISOR_INPUT = "advisor_input"
    TASK_STATE = "task_state"
    SCRATCH = "scratch"
    RESPONSE_DRAFT = "response_draft"
    DIARY = "diary"
    IDENTITY = "identity"


PRE_IDENTITY_REGION_ORDER: tuple[LogicalRegion, ...] = (
    LogicalRegion.CONVERSATION_HISTORY,
    LogicalRegion.USER_INPUT,
    LogicalRegion.CORTEX,
    LogicalRegion.SITUATION_AWARENESS,
    LogicalRegion.TOOL_RESULTS,
    LogicalRegion.ADVISOR_INPUT,
    LogicalRegion.TASK_STATE,
    LogicalRegion.SCRATCH,
    LogicalRegion.RESPONSE_DRAFT,
    LogicalRegion.DIARY,
)

CANONICAL_REGION_ORDER: tuple[LogicalRegion, ...] = (
    *PRE_IDENTITY_REGION_ORDER,
    LogicalRegion.IDENTITY,
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


@dataclass(frozen=True, slots=True)
class AttendedInterval:
    """One contiguous attended character interval inside a region's full text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("AttendedInterval.start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int):
            raise TypeError("AttendedInterval.end must be an integer")
        if self.start < 0:
            raise ValueError("AttendedInterval.start must be non-negative")
        if self.end < self.start:
            raise ValueError("AttendedInterval.end must be >= start")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True, slots=True)
class RegionMaskPolicy:
    """A reusable policy that resolves one region to exact attended intervals.

    ``tail_percent`` is the operational 0-100 slider policy.  It exposes an
    exact newest-character suffix using integer ceiling arithmetic, so every
    non-zero setting exposes at least one cell when the region is non-empty.
    The percentage is only control input; compilers receive exact intervals.
    """

    kind: str
    limit: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("RegionMaskPolicy.kind must be a non-empty string")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("RegionMaskPolicy.limit must be an integer")
        if self.limit < 0:
            raise ValueError("RegionMaskPolicy.limit must be non-negative")
        if self.kind not in {"all", "none", "last_n_spans", "tail_percent"}:
            raise ValueError(
                f"unsupported RegionMaskPolicy.kind {self.kind!r}; "
                "expected 'all', 'none', 'last_n_spans', or 'tail_percent'"
            )
        if self.kind == "tail_percent" and self.limit > 100:
            raise ValueError("tail_percent limit must be in [0, 100]")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "limit": self.limit}


def resolve_mask_policy(
    spans: tuple[FieldSpan, ...],
    policy: RegionMaskPolicy,
) -> tuple[AttendedInterval, ...]:
    """Resolve a mask policy to explicit attended intervals over span text."""

    if not isinstance(policy, RegionMaskPolicy):
        raise TypeError("resolve_mask_policy requires a RegionMaskPolicy")
    if policy.kind == "none":
        return ()
    text_length = sum(len(span.text) for span in spans)
    if text_length == 0:
        return ()
    if policy.kind == "all":
        return (AttendedInterval(0, text_length),)
    if policy.kind == "tail_percent":
        if policy.limit == 0:
            return ()
        attended_count = (text_length * policy.limit + 99) // 100
        return (AttendedInterval(text_length - attended_count, text_length),)
    if policy.kind == "last_n_spans":
        limit = policy.limit
        total = len(spans)
        if limit <= 0 or total == 0:
            return ()
        start_span_index = max(0, total - limit)
        intervals: list[AttendedInterval] = []
        offset = 0
        for index, span in enumerate(spans):
            span_len = len(span.text)
            if index >= start_span_index:
                intervals.append(AttendedInterval(offset, offset + span_len))
            offset += span_len
        return tuple(intervals)
    raise ValueError(f"unsupported RegionMaskPolicy.kind {policy.kind!r}")


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


def _serialized_region_name(region: LogicalRegion, schema_version: str) -> str:
    if schema_version == LEGACY_SCHEMA_VERSION and region is LogicalRegion.CORTEX:
        return LEGACY_CORTEX_REGION_NAME
    return region.value


def canonical_region_order(schema_version: str = SCHEMA_VERSION) -> tuple[LogicalRegion, ...]:
    """Return the immutable region order belonging to one field schema.

    V1 and v2 hashes were defined over the original ten regions.  V3 appends
    ``identity`` without reinterpreting any earlier numeric region identity.
    """

    if schema_version in {LEGACY_SCHEMA_VERSION, CORTEX_SCHEMA_VERSION}:
        return PRE_IDENTITY_REGION_ORDER
    if schema_version == SCHEMA_VERSION:
        return CANONICAL_REGION_ORDER
    raise ValueError(f"unsupported shared-field schema {schema_version!r}")


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
    """Immutable state of one canonical logical region.

    The region preserves its complete canonical text in ``spans``.  Attention
    masks select which characters are attended by a rail; masked text remains
    canonical in this same position-stable region but is not compiled.  The
    serialized ``visibility`` value remains for legacy snapshot compatibility;
    the live Heart path uses its independent mask controller instead.
    """

    name: LogicalRegion | str
    spans: tuple[FieldSpan, ...] = ()
    visibility: RegionVisibility | str = RegionVisibility.ATTENDED
    write_policy: WritePolicy | str | None = None
    attended_intervals: tuple[AttendedInterval, ...] | None = None
    mask_policy: RegionMaskPolicy | None = None

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
            self.visibility if isinstance(self.visibility, RegionVisibility) else RegionVisibility(self.visibility)
        )
        object.__setattr__(self, "visibility", visibility)

        policy = self.write_policy
        if policy is None:
            policy = WritePolicy.CORE_WRITABLE if name in CORE_WRITABLE_REGIONS else WritePolicy.SEALED
        elif not isinstance(policy, WritePolicy):
            policy = WritePolicy(policy)
        if policy is WritePolicy.CORE_WRITABLE and name not in CORE_WRITABLE_REGIONS:
            raise ValueError(f"logical region {name.value!r} is always sealed")
        object.__setattr__(self, "write_policy", policy)

        mask_policy = self.mask_policy
        if mask_policy is not None and not isinstance(mask_policy, RegionMaskPolicy):
            mask_policy = RegionMaskPolicy(mask_policy)
        object.__setattr__(self, "mask_policy", mask_policy)

        text_length = sum(len(span.text) for span in spans)
        intervals = self.attended_intervals
        if intervals is None:
            if mask_policy is not None:
                intervals = resolve_mask_policy(spans, mask_policy)
            elif visibility is RegionVisibility.ATTENDED:
                intervals = (AttendedInterval(0, text_length),) if text_length > 0 else ()
            else:
                intervals = ()
        else:
            intervals = tuple(intervals)
            self._validate_intervals(intervals, text_length, name)
        object.__setattr__(self, "attended_intervals", intervals)

    @staticmethod
    def _validate_intervals(
        intervals: tuple[AttendedInterval, ...],
        text_length: int,
        name: LogicalRegion,
    ) -> None:
        previous_end = 0
        for interval in intervals:
            if not isinstance(interval, AttendedInterval):
                raise TypeError("RegionState.attended_intervals must contain AttendedInterval values")
            if interval.start < 0 or interval.end > text_length:
                raise ValueError(
                    f"attended interval [{interval.start}, {interval.end}) exceeds "
                    f"region {name.value!r} length {text_length}"
                )
            if interval.start < previous_end:
                raise ValueError(f"attended intervals in region {name.value!r} must be sorted and non-overlapping")
            previous_end = interval.end

    @property
    def text(self) -> str:
        """The complete canonical string; no view/window truncation applies."""

        return "".join(span.text for span in self.spans)

    @property
    def attended_text(self) -> str:
        """The attended substring selected by the current intervals."""

        text = self.text
        return "".join(text[interval.start : interval.end] for interval in self.attended_intervals)

    def attended_offsets(self) -> Iterator[tuple[int, int, FieldSpan, int]]:
        """Yield (global_offset_in_region, span, span_offset) for attended chars."""

        text_offset = 0
        interval_index = 0
        intervals = tuple(self.attended_intervals)
        for span in self.spans:
            span_len = len(span.text)
            span_start = text_offset
            span_end = text_offset + span_len
            while interval_index < len(intervals) and intervals[interval_index].end <= span_start:
                interval_index += 1
            local_index = interval_index
            while local_index < len(intervals) and intervals[local_index].start < span_end:
                interval = intervals[local_index]
                attend_start = max(interval.start, span_start)
                attend_end = min(interval.end, span_end)
                if attend_start < attend_end:
                    yield (attend_start, attend_end, span, attend_start - span_start)
                local_index += 1
            text_offset += span_len

    def with_intervals(
        self,
        intervals: tuple[AttendedInterval, ...],
    ) -> "RegionState":
        """Return an identical region with new explicit attended intervals."""

        return RegionState(
            name=self.name,
            spans=self.spans,
            visibility=self.visibility,
            write_policy=self.write_policy,
            attended_intervals=intervals,
            mask_policy=None,
        )

    def with_policy(
        self,
        policy: RegionMaskPolicy,
    ) -> "RegionState":
        """Return an identical region with a new mask policy (intervals resolved)."""

        return RegionState(
            name=self.name,
            spans=self.spans,
            visibility=self.visibility,
            write_policy=self.write_policy,
            attended_intervals=None,
            mask_policy=policy,
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        # Attention masks are derived views of the canonical text, not part of
        # the canonical identity.  Changing a mask must not create a new
        # canonical field; the canonical body is the ordered spans and region
        # metadata (visibility/write_policy), not which characters are attended.
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
        attended_intervals: tuple[AttendedInterval, ...] | None = None,
        mask_policy: RegionMaskPolicy | None = None,
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
            attended_intervals=attended_intervals,
            mask_policy=mask_policy,
        )


@dataclass(frozen=True, slots=True)
class SharedFieldSnapshot:
    """A complete immutable field state linked to its parent by hash."""

    tick_id: int
    regions: tuple[RegionState, ...] = ()
    parent_field_id: str | None = None
    source_manifest_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    field_id: str = field(init=False)
    canonical_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported shared-field schema {self.schema_version!r}; "
                f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)!r}"
            )
        if isinstance(self.tick_id, bool) or not isinstance(self.tick_id, int):
            raise TypeError("SharedFieldSnapshot.tick_id must be an integer")
        if self.tick_id < 0:
            raise ValueError("SharedFieldSnapshot.tick_id must be non-negative")
        if self.parent_field_id is not None and (not isinstance(self.parent_field_id, str) or not self.parent_field_id):
            raise ValueError("parent_field_id must be None or a non-empty string")

        supplied: dict[LogicalRegion, RegionState] = {}
        for region_state in tuple(self.regions):
            if not isinstance(region_state, RegionState):
                raise TypeError("SharedFieldSnapshot.regions must contain RegionState values")
            if region_state.name in supplied:
                raise ValueError(f"duplicate logical region {region_state.name.value!r}")
            supplied[region_state.name] = region_state
        order = canonical_region_order(self.schema_version)
        disallowed = set(supplied) - set(order)
        if disallowed:
            names = ", ".join(sorted(region.value for region in disallowed))
            raise ValueError(
                f"shared-field schema {self.schema_version!r} does not contain regions: {names}"
            )
        normalized = tuple(supplied.get(region, RegionState(name=region)) for region in order)
        object.__setattr__(self, "regions", normalized)

        manifests = tuple(sorted(set(str(item) for item in self.source_manifest_ids)))
        if any(not item for item in manifests):
            raise ValueError("source_manifest_ids cannot contain empty values")
        object.__setattr__(self, "source_manifest_ids", manifests)

        digest = canonical_sha256(self.to_canonical_dict())
        object.__setattr__(self, "field_id", digest)
        object.__setattr__(self, "canonical_hash", digest)

    def region(self, name: LogicalRegion | str) -> RegionState:
        logical_name = _as_logical_region(name)
        order = canonical_region_order(self.schema_version)
        try:
            return self.regions[order.index(logical_name)]
        except ValueError as exc:
            raise KeyError(
                f"region {logical_name.value!r} is absent from schema {self.schema_version!r}"
            ) from exc

    def to_canonical_dict(self) -> dict[str, Any]:
        regions = []
        for region in self.regions:
            payload = region.to_canonical_dict()
            payload["name"] = _serialized_region_name(region.name, self.schema_version)
            regions.append(payload)
        return {
            "schema": self.schema_version,
            "tick_id": self.tick_id,
            "parent_field_id": self.parent_field_id,
            "source_manifest_ids": list(self.source_manifest_ids),
            "regions": regions,
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
    "CANONICAL_REGION_ORDER",
    "CORE_WRITABLE_REGIONS",
    "CORTEX_SCHEMA_VERSION",
    "LEGACY_CORTEX_REGION_NAME",
    "LEGACY_SCHEMA_VERSION",
    "LOGICAL_REGION_IDS",
    "PRE_IDENTITY_REGION_ORDER",
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "AttendedInterval",
    "FieldSpan",
    "LogicalRegion",
    "PhysicalRole",
    "RegionMaskPolicy",
    "RegionState",
    "RegionVisibility",
    "SharedFieldSnapshot",
    "WritePolicy",
    "canonical_json_bytes",
    "canonical_region_order",
    "canonical_sha256",
    "resolve_mask_policy",
]
