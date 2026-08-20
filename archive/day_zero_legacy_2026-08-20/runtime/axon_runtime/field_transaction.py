"""Pure, deterministic transactions over the canonical shared field.

Runtime-owned evidence arrives between canonical ticks: queued user input,
tool results, advisor material, diary entries, and similar sealed context.  It
must be visible to a core before that core authors a delta, without creating a
second canonical tick or weakening the core-write boundary.

This module therefore has two explicit stages:

* :func:`apply_system_update` creates an immutable same-tick working overlay.
* :func:`compose_tick_successor` validates a core delta against that exact
  overlay and emits one parent-linked canonical successor.

System operations are whole-span operations.  Existing ``FieldSpan`` objects
are reused exactly, so their text, provenance, confidence, container
references, and edge references are not reconstructed or normalized here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

from runtime.field import (
    CANONICAL_REGION_ORDER,
    FieldDelta,
    FieldSpan,
    LogicalRegion,
    RegionState,
    SharedFieldSnapshot,
    WritePolicy,
    apply_delta,
    canonical_sha256,
    validate_delta,
)


RUNTIME_SYSTEM_AUTHOR = "runtime"

RUNTIME_OWNED_REGIONS: frozenset[LogicalRegion] = frozenset(
    {
        LogicalRegion.CONVERSATION_HISTORY,
        LogicalRegion.USER_INPUT,
        LogicalRegion.STRUCTURED_KNOWLEDGE,
        LogicalRegion.SITUATION_AWARENESS,
        LogicalRegion.TOOL_RESULTS,
        LogicalRegion.ADVISOR_INPUT,
        LogicalRegion.TASK_STATE,
        LogicalRegion.DIARY,
    }
)

_REGION_ORDER = {
    region: index for index, region in enumerate(CANONICAL_REGION_ORDER)
}


class FieldTransactionError(ValueError):
    """Base class for a rejected runtime-owned field transaction."""


class StaleSystemUpdateError(FieldTransactionError):
    """A system update was authored against a different canonical head."""


class SystemRegionWriteError(FieldTransactionError):
    """A system operation addressed a non-runtime-owned or writable region."""


class DuplicateSystemOperationError(FieldTransactionError):
    """More than one system operation addressed the same logical region."""


class ConsolidatorAuthorError(FieldTransactionError):
    """The caller did not independently authenticate the delta author."""


class ReplayMismatchError(FieldTransactionError):
    """Recomputation did not reproduce an audit record byte-for-byte."""


def _coerce_region(value: LogicalRegion | str) -> LogicalRegion:
    try:
        return value if isinstance(value, LogicalRegion) else LogicalRegion(value)
    except (TypeError, ValueError) as exc:
        raise SystemRegionWriteError(
            f"unknown system-update region {value!r}"
        ) from exc


def _whole_spans(values: tuple[FieldSpan, ...], label: str) -> tuple[FieldSpan, ...]:
    spans = tuple(values)
    if not spans:
        raise FieldTransactionError(f"{label} requires at least one whole span")
    if not all(isinstance(span, FieldSpan) for span in spans):
        raise TypeError(f"{label} accepts only FieldSpan values")
    span_ids = [span.span_id for span in spans]
    if len(span_ids) != len(set(span_ids)):
        raise FieldTransactionError(f"{label} contains duplicate span IDs")
    return spans


@dataclass(frozen=True, slots=True)
class AppendSpans:
    """Append complete spans, preserving every span already in the region."""

    region: LogicalRegion | str
    spans: tuple[FieldSpan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "region", _coerce_region(self.region))
        object.__setattr__(
            self,
            "spans",
            _whole_spans(self.spans, "AppendSpans"),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "op": "append_spans",
            "region": self.region.value,
            "spans": [span.to_canonical_dict() for span in self.spans],
        }


@dataclass(frozen=True, slots=True)
class ReplaceSpans:
    """Replace one runtime-owned region with the supplied complete spans."""

    region: LogicalRegion | str
    spans: tuple[FieldSpan, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "region", _coerce_region(self.region))
        object.__setattr__(
            self,
            "spans",
            _whole_spans(self.spans, "ReplaceSpans"),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "op": "replace_spans",
            "region": self.region.value,
            "spans": [span.to_canonical_dict() for span in self.spans],
        }


@dataclass(frozen=True, slots=True)
class ClearRegion:
    """Deliberately clear every span from one runtime-owned region."""

    region: LogicalRegion | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "region", _coerce_region(self.region))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "op": "clear_region",
            "region": self.region.value,
        }


SystemFieldOperation: TypeAlias = AppendSpans | ReplaceSpans | ClearRegion


@dataclass(frozen=True, slots=True)
class SystemFieldUpdate:
    """One immutable runtime-authored update against an exact field head."""

    base_field_id: str
    base_tick_id: int
    operations: tuple[SystemFieldOperation, ...] = ()
    evidence: tuple[str, ...] = ()
    author: str = RUNTIME_SYSTEM_AUTHOR
    update_id: str = field(init=False)
    canonical_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.base_field_id, str) or not self.base_field_id:
            raise FieldTransactionError("base_field_id must be non-empty")
        if (
            isinstance(self.base_tick_id, bool)
            or not isinstance(self.base_tick_id, int)
            or self.base_tick_id < 0
        ):
            raise FieldTransactionError("base_tick_id must be non-negative")
        if self.author != RUNTIME_SYSTEM_AUTHOR:
            raise FieldTransactionError(
                f"SystemFieldUpdate author must be {RUNTIME_SYSTEM_AUTHOR!r}"
            )
        operations = tuple(self.operations)
        if not all(
            isinstance(operation, (AppendSpans, ReplaceSpans, ClearRegion))
            for operation in operations
        ):
            raise TypeError("SystemFieldUpdate contains an unsupported operation")
        region_names = [operation.region for operation in operations]
        if len(region_names) != len(set(region_names)):
            raise DuplicateSystemOperationError(
                "one SystemFieldUpdate may address each region only once"
            )
        operations = tuple(
            sorted(operations, key=lambda item: _REGION_ORDER[item.region])
        )
        evidence = tuple(sorted(set(str(item) for item in self.evidence)))
        if any(not item for item in evidence):
            raise FieldTransactionError("evidence cannot contain empty values")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "evidence", evidence)
        digest = canonical_sha256(self.to_canonical_dict())
        object.__setattr__(self, "update_id", digest)
        object.__setattr__(self, "canonical_hash", digest)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-system-field-update-v1",
            "base_field_id": self.base_field_id,
            "base_tick_id": self.base_tick_id,
            "author": self.author,
            "operations": [
                operation.to_canonical_dict() for operation in self.operations
            ],
            "evidence": list(self.evidence),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_canonical_dict(),
            "update_id": self.update_id,
            "canonical_hash": self.canonical_hash,
        }


@dataclass(frozen=True, slots=True)
class SystemUpdateApplication:
    """Exact receipt for a same-tick runtime working overlay."""

    head_snapshot: SharedFieldSnapshot
    update: SystemFieldUpdate
    working_snapshot: SharedFieldSnapshot
    application_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.head_snapshot, SharedFieldSnapshot):
            raise TypeError("head_snapshot must be SharedFieldSnapshot")
        if not isinstance(self.update, SystemFieldUpdate):
            raise TypeError("update must be SystemFieldUpdate")
        if not isinstance(self.working_snapshot, SharedFieldSnapshot):
            raise TypeError("working_snapshot must be SharedFieldSnapshot")
        if self.update.base_field_id != self.head_snapshot.field_id:
            raise StaleSystemUpdateError("receipt update does not match its head")
        if self.update.base_tick_id != self.head_snapshot.tick_id:
            raise StaleSystemUpdateError("receipt update tick does not match its head")
        if self.working_snapshot.tick_id != self.head_snapshot.tick_id:
            raise FieldTransactionError("working overlay advanced the canonical tick")
        if self.working_snapshot.parent_field_id != self.head_snapshot.field_id:
            raise FieldTransactionError("working overlay is not linked to its head")
        object.__setattr__(
            self,
            "application_id",
            canonical_sha256(self.to_canonical_dict()),
        )

    @property
    def head_field_id(self) -> str:
        return self.head_snapshot.field_id

    @property
    def working_field_id(self) -> str:
        return self.working_snapshot.field_id

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-system-update-application-v1",
            "head_field_id": self.head_snapshot.field_id,
            "head_tick_id": self.head_snapshot.tick_id,
            "update_id": self.update.update_id,
            "working_field_id": self.working_snapshot.field_id,
            "working_tick_id": self.working_snapshot.tick_id,
        }


@dataclass(frozen=True, slots=True)
class FieldTransactionAudit:
    """Self-contained exact before/working/final transaction evidence."""

    before_snapshot: SharedFieldSnapshot
    system_update: SystemFieldUpdate
    working_snapshot: SharedFieldSnapshot
    consolidator_delta: FieldDelta | None
    consolidator_author_core_id: str | None
    final_snapshot: SharedFieldSnapshot
    audit_id: str = field(init=False)
    canonical_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("before_snapshot", "working_snapshot", "final_snapshot"):
            if not isinstance(getattr(self, name), SharedFieldSnapshot):
                raise TypeError(f"{name} must be SharedFieldSnapshot")
        if not isinstance(self.system_update, SystemFieldUpdate):
            raise TypeError("system_update must be SystemFieldUpdate")
        if self.consolidator_delta is not None and not isinstance(
            self.consolidator_delta,
            FieldDelta,
        ):
            raise TypeError("consolidator_delta must be FieldDelta or None")
        if self.consolidator_delta is None:
            if self.consolidator_author_core_id is not None:
                raise ConsolidatorAuthorError(
                    "a no-delta transaction cannot claim a delta author"
                )
        elif (
            not isinstance(self.consolidator_author_core_id, str)
            or not self.consolidator_author_core_id
        ):
            raise ConsolidatorAuthorError(
                "a consolidator delta requires a separately supplied author"
            )
        if self.working_snapshot.tick_id != self.before_snapshot.tick_id:
            raise FieldTransactionError("audit working snapshot advanced the tick")
        if self.working_snapshot.parent_field_id != self.before_snapshot.field_id:
            raise FieldTransactionError("audit working snapshot has the wrong head")
        if self.final_snapshot.tick_id != self.before_snapshot.tick_id + 1:
            raise FieldTransactionError("audit final snapshot is not the next tick")
        if self.final_snapshot.parent_field_id != self.before_snapshot.field_id:
            raise FieldTransactionError("audit final snapshot has the wrong parent")
        digest = canonical_sha256(self.to_canonical_dict())
        object.__setattr__(self, "audit_id", digest)
        object.__setattr__(self, "canonical_hash", digest)

    @property
    def before_field_id(self) -> str:
        return self.before_snapshot.field_id

    @property
    def working_field_id(self) -> str:
        return self.working_snapshot.field_id

    @property
    def final_field_id(self) -> str:
        return self.final_snapshot.field_id

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-field-transaction-audit-v1",
            "before_snapshot": self.before_snapshot.to_canonical_dict(),
            "before_field_id": self.before_snapshot.field_id,
            "system_update": self.system_update.to_canonical_dict(),
            "system_update_id": self.system_update.update_id,
            "working_snapshot": self.working_snapshot.to_canonical_dict(),
            "working_field_id": self.working_snapshot.field_id,
            "consolidator_delta": (
                None
                if self.consolidator_delta is None
                else self.consolidator_delta.to_canonical_dict()
            ),
            "consolidator_delta_id": (
                None
                if self.consolidator_delta is None
                else self.consolidator_delta.delta_id
            ),
            "consolidator_author_core_id": self.consolidator_author_core_id,
            "final_snapshot": self.final_snapshot.to_canonical_dict(),
            "final_field_id": self.final_snapshot.field_id,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_canonical_dict(),
            "audit_id": self.audit_id,
            "canonical_hash": self.canonical_hash,
        }


def validate_system_update(
    head: SharedFieldSnapshot,
    update: SystemFieldUpdate,
) -> None:
    """Validate runtime authority and the exact head binding."""

    if not isinstance(head, SharedFieldSnapshot):
        raise TypeError("head must be SharedFieldSnapshot")
    if not isinstance(update, SystemFieldUpdate):
        raise TypeError("update must be SystemFieldUpdate")
    if (
        update.base_field_id != head.field_id
        or update.base_tick_id != head.tick_id
    ):
        raise StaleSystemUpdateError(
            "system update base does not match the current field id and tick"
        )
    for operation in update.operations:
        region_name = operation.region
        if region_name not in RUNTIME_OWNED_REGIONS:
            raise SystemRegionWriteError(
                f"runtime cannot system-write {region_name.value!r}; "
                "scratch and response_draft remain core-writable"
            )
        region = head.region(region_name)
        if region.write_policy is not WritePolicy.SEALED:
            raise SystemRegionWriteError(
                f"runtime-owned region {region_name.value!r} is not sealed"
            )
        if isinstance(operation, AppendSpans):
            existing = {span.span_id for span in region.spans}
            duplicate_ids = existing & {
                span.span_id for span in operation.spans
            }
            if duplicate_ids:
                raise FieldTransactionError(
                    f"append would duplicate span IDs in {region_name.value!r}: "
                    + ", ".join(sorted(duplicate_ids))
                )


def _apply_system_operation(
    region: RegionState,
    operation: SystemFieldOperation,
) -> RegionState:
    if isinstance(operation, AppendSpans):
        next_spans = (*region.spans, *operation.spans)
        if tuple(next_spans[: len(region.spans)]) != region.spans:
            raise FieldTransactionError("append discarded an existing span")
    elif isinstance(operation, ReplaceSpans):
        next_spans = operation.spans
    elif isinstance(operation, ClearRegion):
        next_spans = ()
    else:  # pragma: no cover - guarded by SystemFieldUpdate
        raise TypeError("unsupported system operation")
    return RegionState(
        name=region.name,
        spans=tuple(next_spans),
        visibility=region.visibility,
        write_policy=region.write_policy,
    )


def apply_system_update(
    head: SharedFieldSnapshot,
    update: SystemFieldUpdate,
) -> SystemUpdateApplication:
    """Create a deterministic same-tick overlay without mutating ``head``."""

    validate_system_update(head, update)
    by_region = {operation.region: operation for operation in update.operations}
    working_regions = tuple(
        _apply_system_operation(region, by_region[region.name])
        if region.name in by_region
        else region
        for region in head.regions
    )
    working = SharedFieldSnapshot(
        tick_id=head.tick_id,
        regions=working_regions,
        parent_field_id=head.field_id,
        source_manifest_ids=head.source_manifest_ids,
    )
    return SystemUpdateApplication(
        head_snapshot=head,
        update=update,
        working_snapshot=working,
    )


def _validated_consolidator_author(
    delta: FieldDelta | None,
    consolidator_author_core_id: str | None,
) -> None:
    if delta is None:
        if consolidator_author_core_id is not None:
            raise ConsolidatorAuthorError(
                "consolidator author supplied without a delta"
            )
        return
    if (
        not isinstance(consolidator_author_core_id, str)
        or not consolidator_author_core_id
    ):
        raise ConsolidatorAuthorError(
            "caller must separately supply the consolidator delta author"
        )
    if delta.author_core_id != consolidator_author_core_id:
        raise ConsolidatorAuthorError(
            "delta author does not match the separately supplied consolidator"
        )


def compose_tick_transaction(
    head: SharedFieldSnapshot,
    update: SystemFieldUpdate,
    consolidator_delta_or_none: FieldDelta | None,
    *,
    consolidator_author_core_id: str | None = None,
) -> FieldTransactionAudit:
    """Compose and audit exactly one canonical tick transition."""

    application = apply_system_update(head, update)
    working = application.working_snapshot
    delta = consolidator_delta_or_none
    if delta is not None:
        if not isinstance(delta, FieldDelta):
            raise TypeError("consolidator delta must be FieldDelta or None")
        # Staleness and core-region authority are checked against the exact
        # same-tick overlay, never against the pre-overlay canonical head.
        validate_delta(working, delta)
    _validated_consolidator_author(delta, consolidator_author_core_id)

    if delta is None:
        final_regions = working.regions
    else:
        applied = apply_delta(working, delta)
        final_regions = applied.regions
    final = SharedFieldSnapshot(
        tick_id=head.tick_id + 1,
        regions=final_regions,
        parent_field_id=head.field_id,
        source_manifest_ids=head.source_manifest_ids,
    )
    return FieldTransactionAudit(
        before_snapshot=head,
        system_update=update,
        working_snapshot=working,
        consolidator_delta=delta,
        consolidator_author_core_id=consolidator_author_core_id,
        final_snapshot=final,
    )


def compose_tick_successor(
    head: SharedFieldSnapshot,
    update: SystemFieldUpdate,
    consolidator_delta_or_none: FieldDelta | None,
    *,
    consolidator_author_core_id: str | None = None,
) -> SharedFieldSnapshot:
    """Return the one canonical successor produced by the transaction."""

    return compose_tick_transaction(
        head,
        update,
        consolidator_delta_or_none,
        consolidator_author_core_id=consolidator_author_core_id,
    ).final_snapshot


def replay_field_transaction(
    audit: FieldTransactionAudit,
) -> SharedFieldSnapshot:
    """Recompute an audit and require exact before/working/final identity."""

    if not isinstance(audit, FieldTransactionAudit):
        raise TypeError("audit must be FieldTransactionAudit")
    replay = compose_tick_transaction(
        audit.before_snapshot,
        audit.system_update,
        audit.consolidator_delta,
        consolidator_author_core_id=audit.consolidator_author_core_id,
    )
    if replay.to_canonical_dict() != audit.to_canonical_dict():
        raise ReplayMismatchError("transaction replay differs from exact audit")
    if replay.audit_id != audit.audit_id:
        raise ReplayMismatchError("transaction replay audit hash differs")
    return replay.final_snapshot


# Readable aliases for call sites that prefer the region-qualified spelling.
AppendRegionSpans = AppendSpans
ReplaceRegionSpans = ReplaceSpans
ClearRegionSpans = ClearRegion
replay_tick_transaction = replay_field_transaction


__all__ = [
    "RUNTIME_SYSTEM_AUTHOR",
    "RUNTIME_OWNED_REGIONS",
    "FieldTransactionError",
    "StaleSystemUpdateError",
    "SystemRegionWriteError",
    "DuplicateSystemOperationError",
    "ConsolidatorAuthorError",
    "ReplayMismatchError",
    "AppendSpans",
    "ReplaceSpans",
    "ClearRegion",
    "AppendRegionSpans",
    "ReplaceRegionSpans",
    "ClearRegionSpans",
    "SystemFieldOperation",
    "SystemFieldUpdate",
    "SystemUpdateApplication",
    "FieldTransactionAudit",
    "validate_system_update",
    "apply_system_update",
    "compose_tick_transaction",
    "compose_tick_successor",
    "replay_field_transaction",
    "replay_tick_transaction",
]
