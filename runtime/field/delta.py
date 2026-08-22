"""Typed, validated and deterministically replayable shared-field deltas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .schema import (
    CORE_WRITABLE_REGIONS,
    FieldSpan,
    LogicalRegion,
    RegionState,
    SharedFieldSnapshot,
    WritePolicy,
    canonical_sha256,
)


class DeltaValidationError(ValueError):
    """Base class for a rejected field delta."""


class StaleDeltaError(DeltaValidationError):
    """The delta was authored against a different field snapshot."""


class SealedRegionWriteError(DeltaValidationError):
    """A delta attempted to mutate runtime-owned canonical evidence."""


class OverlappingDeltaError(DeltaValidationError):
    """Two operations address overlapping canonical character ranges."""


def _coerce_region(value: LogicalRegion | str) -> LogicalRegion:
    try:
        return value if isinstance(value, LogicalRegion) else LogicalRegion(value)
    except (TypeError, ValueError) as exc:
        raise DeltaValidationError(f"unknown delta region {value!r}") from exc


def _refs(value: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(str(item) for item in value)))


@dataclass(frozen=True, slots=True)
class InsertText:
    region: LogicalRegion | str
    offset: int
    text: str
    provenance: str = ""
    container_refs: tuple[str, ...] = ()
    edge_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "region", _coerce_region(self.region))
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise TypeError("InsertText.offset must be an integer")
        if self.offset < 0:
            raise DeltaValidationError("InsertText.offset must be non-negative")
        if not self.text:
            raise DeltaValidationError("InsertText.text must be non-empty")
        object.__setattr__(self, "container_refs", _refs(self.container_refs))
        object.__setattr__(self, "edge_refs", _refs(self.edge_refs))

    @property
    def start(self) -> int:
        return self.offset

    @property
    def end(self) -> int:
        return self.offset

    @property
    def replacement_text(self) -> str:
        return self.text

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "op": "insert",
            "region": self.region.value,
            "offset": self.offset,
            "text": self.text,
            "provenance": self.provenance,
            "container_refs": list(self.container_refs),
            "edge_refs": list(self.edge_refs),
        }


@dataclass(frozen=True, slots=True)
class DeleteText:
    region: LogicalRegion | str
    start: int
    end: int
    provenance: str = ""
    container_refs: tuple[str, ...] = ()
    edge_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "region", _coerce_region(self.region))
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
        ):
            raise TypeError("DeleteText bounds must be integers")
        if self.start < 0 or self.end <= self.start:
            raise DeltaValidationError(
                "DeleteText requires 0 <= start < end"
            )
        object.__setattr__(self, "container_refs", _refs(self.container_refs))
        object.__setattr__(self, "edge_refs", _refs(self.edge_refs))

    @property
    def replacement_text(self) -> str:
        return ""

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "op": "delete",
            "region": self.region.value,
            "start": self.start,
            "end": self.end,
            "provenance": self.provenance,
            "container_refs": list(self.container_refs),
            "edge_refs": list(self.edge_refs),
        }


@dataclass(frozen=True, slots=True)
class ReplaceText:
    region: LogicalRegion | str
    start: int
    end: int
    text: str
    provenance: str = ""
    container_refs: tuple[str, ...] = ()
    edge_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "region", _coerce_region(self.region))
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
        ):
            raise TypeError("ReplaceText bounds must be integers")
        if self.start < 0 or self.end < self.start:
            raise DeltaValidationError(
                "ReplaceText requires 0 <= start <= end"
            )
        if self.start == self.end and not self.text:
            raise DeltaValidationError("ReplaceText cannot be an empty no-op")
        object.__setattr__(self, "container_refs", _refs(self.container_refs))
        object.__setattr__(self, "edge_refs", _refs(self.edge_refs))

    @property
    def replacement_text(self) -> str:
        return self.text

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "op": "replace",
            "region": self.region.value,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "provenance": self.provenance,
            "container_refs": list(self.container_refs),
            "edge_refs": list(self.edge_refs),
        }


FieldOperation: TypeAlias = InsertText | DeleteText | ReplaceText


@dataclass(frozen=True, slots=True)
class FieldDelta:
    """A proposal authored against exactly one immutable base snapshot."""

    base_field_id: str
    base_tick_id: int
    author_core_id: str
    pass_id: str | int
    operations: tuple[FieldOperation, ...]
    evidence: tuple[str, ...] = ()
    delta_id: str = field(init=False)
    canonical_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base_field_id, str) or not self.base_field_id:
            raise ValueError("FieldDelta.base_field_id must be non-empty")
        if (
            isinstance(self.base_tick_id, bool)
            or not isinstance(self.base_tick_id, int)
            or self.base_tick_id < 0
        ):
            raise ValueError("FieldDelta.base_tick_id must be non-negative")
        if not isinstance(self.author_core_id, str) or not self.author_core_id:
            raise ValueError("FieldDelta.author_core_id must be non-empty")
        pass_id = str(self.pass_id)
        if not pass_id:
            raise ValueError("FieldDelta.pass_id must be non-empty")
        object.__setattr__(self, "pass_id", pass_id)
        operations = tuple(self.operations)
        if not operations:
            raise ValueError("FieldDelta.operations must be non-empty")
        if not all(
            isinstance(operation, (InsertText, DeleteText, ReplaceText))
            for operation in operations
        ):
            raise TypeError("FieldDelta contains an unsupported operation type")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self,
            "evidence",
            tuple(sorted(set(str(item) for item in self.evidence))),
        )
        digest = canonical_sha256(self.to_canonical_dict())
        object.__setattr__(self, "delta_id", digest)
        object.__setattr__(self, "canonical_hash", digest)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "shared-field-delta-v1",
            "base_field_id": self.base_field_id,
            "base_tick_id": self.base_tick_id,
            "author_core_id": self.author_core_id,
            "pass_id": str(self.pass_id),
            "operations": [
                operation.to_canonical_dict() for operation in self.operations
            ],
            "evidence": list(self.evidence),
        }


def _bounds(operation: FieldOperation) -> tuple[int, int]:
    return operation.start, operation.end


def _is_insert_like(operation: FieldOperation) -> bool:
    start, end = _bounds(operation)
    return start == end


def _operations_overlap(
    left: FieldOperation,
    right: FieldOperation,
) -> bool:
    left_start, left_end = _bounds(left)
    right_start, right_end = _bounds(right)
    left_insert = left_start == left_end
    right_insert = right_start == right_end
    if left_insert and right_insert:
        return left_start == right_start
    if left_insert:
        return right_start < left_start < right_end
    if right_insert:
        return left_start < right_start < left_end
    return max(left_start, right_start) < min(left_end, right_end)


def validate_delta(
    snapshot: SharedFieldSnapshot,
    delta: FieldDelta,
    *,
    permitted_regions: frozenset[LogicalRegion] | None = None,
) -> None:
    if delta.base_field_id != snapshot.field_id or delta.base_tick_id != snapshot.tick_id:
        raise StaleDeltaError(
            "delta base does not match the current field id and tick"
        )

    allowed = (
        CORE_WRITABLE_REGIONS
        if permitted_regions is None
        else (CORE_WRITABLE_REGIONS | permitted_regions)
    )

    by_region: dict[LogicalRegion, list[FieldOperation]] = {}
    for operation in delta.operations:
        region_name = operation.region
        if region_name not in allowed:
            raise SealedRegionWriteError(
                f"logical region {region_name.value!r} is sealed"
            )
        region = snapshot.region(region_name)
        start, end = _bounds(operation)
        if start < 0 or end < start or end > len(region.text):
            raise DeltaValidationError(
                f"operation bounds [{start}, {end}) exceed "
                f"{region_name.value!r} length {len(region.text)}"
            )
        by_region.setdefault(region_name, []).append(operation)

    for region_name, operations in by_region.items():
        for left_index, left in enumerate(operations):
            for right in operations[left_index + 1 :]:
                if _operations_overlap(left, right):
                    raise OverlappingDeltaError(
                        f"overlapping operations in {region_name.value!r}"
                    )


def _slice_spans(region: RegionState, start: int, end: int) -> list[FieldSpan]:
    if start == end:
        return []
    result: list[FieldSpan] = []
    region_offset = 0
    for span in region.spans:
        span_start = region_offset
        span_end = span_start + len(span.text)
        intersection_start = max(start, span_start)
        intersection_end = min(end, span_end)
        if intersection_start < intersection_end:
            local_start = intersection_start - span_start
            local_end = intersection_end - span_start
            if local_start == 0 and local_end == len(span.text):
                result.append(span)
            else:
                result.append(
                    FieldSpan(
                        span_id=f"{span.span_id}@{local_start}:{local_end}",
                        text=span.text[local_start:local_end],
                        kind=span.kind,
                        source=span.source,
                        provenance=span.provenance,
                        confidence=span.confidence,
                        container_refs=span.container_refs,
                        edge_refs=span.edge_refs,
                    )
                )
        region_offset = span_end
        if region_offset >= end:
            break
    return result


def _apply_region_operations(
    region: RegionState,
    indexed_operations: list[tuple[int, FieldOperation]],
    delta: FieldDelta,
) -> RegionState:
    ordered = sorted(
        indexed_operations,
        key=lambda item: (
            item[1].start,
            0 if _is_insert_like(item[1]) else 1,
            item[0],
        ),
    )
    spans: list[FieldSpan] = []
    cursor = 0
    for operation_index, operation in ordered:
        spans.extend(_slice_spans(region, cursor, operation.start))
        replacement = operation.replacement_text
        if replacement:
            operation_name = operation.to_canonical_dict()["op"]
            spans.append(
                FieldSpan(
                    span_id=(
                        f"delta-{delta.delta_id[:20]}-"
                        f"{operation_index:04d}"
                    ),
                    text=replacement,
                    kind=f"delta_{operation_name}",
                    source=delta.author_core_id,
                    provenance=(
                        operation.provenance
                        or f"field_delta:{delta.delta_id}:op:{operation_index}"
                    ),
                    container_refs=operation.container_refs,
                    edge_refs=operation.edge_refs,
                )
            )
        # Insertions consume no source characters, but the untouched prefix
        # before their boundary has already been emitted.
        cursor = operation.end
    spans.extend(_slice_spans(region, cursor, len(region.text)))
    # If the region carries a mask policy, re-resolve it against the new spans.
    # Otherwise default to full attention.  Incremental interval tracking is a
    # future refinement; Build B is policy-driven.
    return RegionState(
        name=region.name,
        spans=tuple(spans),
        visibility=region.visibility,
        write_policy=region.write_policy,
        attended_intervals=None,
        mask_policy=region.mask_policy,
    )


def apply_delta(
    snapshot: SharedFieldSnapshot,
    delta: FieldDelta,
    *,
    permitted_regions: frozenset[LogicalRegion] | None = None,
) -> SharedFieldSnapshot:
    """Validate and apply ``delta`` as one deterministic transaction."""

    if not isinstance(snapshot, SharedFieldSnapshot):
        raise TypeError("apply_delta requires SharedFieldSnapshot")
    if not isinstance(delta, FieldDelta):
        raise TypeError("apply_delta requires FieldDelta")
    validate_delta(snapshot, delta, permitted_regions=permitted_regions)

    indexed_by_region: dict[
        LogicalRegion,
        list[tuple[int, FieldOperation]],
    ] = {}
    for index, operation in enumerate(delta.operations):
        indexed_by_region.setdefault(operation.region, []).append(
            (index, operation)
        )

    next_regions = tuple(
        _apply_region_operations(
            region,
            indexed_by_region[region.name],
            delta,
        )
        if region.name in indexed_by_region
        else region
        for region in snapshot.regions
    )
    return SharedFieldSnapshot(
        tick_id=snapshot.tick_id + 1,
        regions=next_regions,
        parent_field_id=snapshot.field_id,
        source_manifest_ids=snapshot.source_manifest_ids,
    )


def replay_deltas(
    base: SharedFieldSnapshot,
    deltas: tuple[FieldDelta, ...],
) -> SharedFieldSnapshot:
    snapshot = base
    for delta in deltas:
        snapshot = apply_delta(snapshot, delta)
    return snapshot


# Explicit aliases make call sites readable without weakening the operation type.
InsertOperation = InsertText
DeleteOperation = DeleteText
ReplaceOperation = ReplaceText


__all__ = [
    "DeltaValidationError",
    "StaleDeltaError",
    "SealedRegionWriteError",
    "OverlappingDeltaError",
    "InsertText",
    "DeleteText",
    "ReplaceText",
    "InsertOperation",
    "DeleteOperation",
    "ReplaceOperation",
    "FieldOperation",
    "FieldDelta",
    "validate_delta",
    "apply_delta",
    "replay_deltas",
]
