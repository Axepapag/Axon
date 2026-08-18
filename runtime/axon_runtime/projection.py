"""Derived bounded active projections over immutable canonical field history.

The source :class:`~runtime.field.SharedFieldSnapshot` is never mutated,
shortened, or treated as disposable.  Projection selects whole spans only and
emits an exact omission record for every span it masks.  Retrieval surfacing is
also derived: it appends provenance-bearing structured-knowledge spans to a new
snapshot while preserving the input snapshot unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

from runtime.field import (
    CANONICAL_REGION_ORDER,
    FieldSpan,
    LogicalRegion,
    RegionState,
    RegionVisibility,
    SharedFieldSnapshot,
    canonical_json_bytes,
)

from .dormant import (
    DormantStore,
    RetrievalResult,
    Surfacing,
)


PINNED_REGIONS: frozenset[LogicalRegion] = frozenset(
    {
        LogicalRegion.USER_INPUT,
        LogicalRegion.RESPONSE_DRAFT,
        LogicalRegion.SCRATCH,
        LogicalRegion.TOOL_RESULTS,
    }
)

DEFAULT_REGION_BUDGETS: tuple[tuple[str, int], ...] = (
    (LogicalRegion.CONVERSATION_HISTORY.value, 1024),
    (LogicalRegion.USER_INPUT.value, 512),
    (LogicalRegion.STRUCTURED_KNOWLEDGE.value, 1024),
    (LogicalRegion.SITUATION_AWARENESS.value, 384),
    (LogicalRegion.TOOL_RESULTS.value, 512),
    (LogicalRegion.ADVISOR_INPUT.value, 384),
    (LogicalRegion.TASK_STATE.value, 384),
    (LogicalRegion.SCRATCH.value, 512),
    (LogicalRegion.RESPONSE_DRAFT.value, 512),
    (LogicalRegion.DIARY.value, 256),
)


class ProjectionError(ValueError):
    """Base error for an invalid active projection."""


class ProjectionBudgetError(ProjectionError):
    """A pinned whole span cannot fit without forbidden slicing."""


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _content_id(prefix: str, value: Any) -> str:
    return f"{prefix}-{_canonical_sha(value)}"


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectionPolicy:
    """Deterministic whole-span active-working-set policy."""

    global_char_budget: int = 4096
    max_selected_spans: int = 256
    region_char_budgets: tuple[tuple[str, int], ...] = DEFAULT_REGION_BUDGETS
    pinned_regions: tuple[str, ...] = tuple(
        sorted(region.value for region in PINNED_REGIONS)
    )
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.global_char_budget, bool)
            or not isinstance(self.global_char_budget, int)
            or self.global_char_budget <= 0
        ):
            raise ProjectionError("global_char_budget must be positive")
        if (
            isinstance(self.max_selected_spans, bool)
            or not isinstance(self.max_selected_spans, int)
            or self.max_selected_spans <= 0
        ):
            raise ProjectionError("max_selected_spans must be positive")
        normalized_budgets: dict[LogicalRegion, int] = {}
        for raw_region, raw_budget in self.region_char_budgets:
            try:
                region = LogicalRegion(raw_region)
            except (TypeError, ValueError) as exc:
                raise ProjectionError(
                    f"unknown region budget {raw_region!r}"
                ) from exc
            if (
                isinstance(raw_budget, bool)
                or not isinstance(raw_budget, int)
                or raw_budget < 0
            ):
                raise ProjectionError("region budgets must be non-negative")
            if region in normalized_budgets:
                raise ProjectionError(f"duplicate budget for {region.value}")
            normalized_budgets[region] = raw_budget
        missing = set(CANONICAL_REGION_ORDER) - set(normalized_budgets)
        if missing:
            raise ProjectionError(
                "missing region budgets: "
                + ", ".join(sorted(region.value for region in missing))
            )
        try:
            pinned = frozenset(LogicalRegion(value) for value in self.pinned_regions)
        except (TypeError, ValueError) as exc:
            raise ProjectionError("pinned_regions contains an unknown region") from exc
        if not PINNED_REGIONS.issubset(pinned):
            missing_pins = PINNED_REGIONS - pinned
            raise ProjectionError(
                "required pinned regions missing: "
                + ", ".join(sorted(region.value for region in missing_pins))
            )
        normalized = tuple(
            (region.value, normalized_budgets[region])
            for region in CANONICAL_REGION_ORDER
        )
        object.__setattr__(self, "region_char_budgets", normalized)
        object.__setattr__(
            self, "pinned_regions", tuple(sorted(region.value for region in pinned))
        )
        object.__setattr__(self, "policy_id", _content_id("projection-policy", self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "active-projection-policy-v1",
            "global_char_budget": self.global_char_budget,
            "max_selected_spans": self.max_selected_spans,
            "region_char_budgets": {
                region: budget for region, budget in self.region_char_budgets
            },
            "pinned_regions": list(self.pinned_regions),
            "selection_order": "pinned-first; newest-whole-span-per-region",
        }

    def budget_for(self, region: LogicalRegion) -> int:
        return dict(self.region_char_budgets)[region.value]


@dataclass(frozen=True, slots=True)
class SelectedSpan:
    logical_region: LogicalRegion
    span_id: str
    canonical_hash: str
    exact_text_sha256: str
    char_count: int
    pinned: bool
    lifecycle_state: str
    lifecycle_event_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_region": self.logical_region.value,
            "span_id": self.span_id,
            "canonical_hash": self.canonical_hash,
            "exact_text_sha256": self.exact_text_sha256,
            "char_count": self.char_count,
            "pinned": self.pinned,
            "lifecycle_state": self.lifecycle_state,
            "lifecycle_event_id": self.lifecycle_event_id,
        }


@dataclass(frozen=True, slots=True)
class MaskedSpan:
    logical_region: LogicalRegion
    span_id: str
    canonical_hash: str
    exact_text_sha256: str
    char_count: int
    reason: str
    source: str
    provenance: str
    container_refs: tuple[str, ...]
    edge_refs: tuple[str, ...]
    lifecycle_state: str
    lifecycle_event_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_region": self.logical_region.value,
            "span_id": self.span_id,
            "canonical_hash": self.canonical_hash,
            "exact_text_sha256": self.exact_text_sha256,
            "char_count": self.char_count,
            "reason": self.reason,
            "source": self.source,
            "provenance": self.provenance,
            "container_refs": list(self.container_refs),
            "edge_refs": list(self.edge_refs),
            "lifecycle_state": self.lifecycle_state,
            "lifecycle_event_id": self.lifecycle_event_id,
        }


@dataclass(frozen=True, slots=True)
class ProjectionManifest:
    source_field_id: str
    active_field_id: str
    policy_id: str
    selected: tuple[SelectedSpan, ...]
    masked: tuple[MaskedSpan, ...]
    selected_char_count: int
    source_char_count: int
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.selected_char_count > self.source_char_count:
            raise ProjectionError("selected characters exceed source characters")
        object.__setattr__(self, "manifest_id", _content_id("projection", self.payload()))

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "active-projection-manifest-v1",
            "source_field_id": self.source_field_id,
            "active_field_id": self.active_field_id,
            "policy_id": self.policy_id,
            "selected": [item.to_dict() for item in self.selected],
            "masked": [item.to_dict() for item in self.masked],
            "selected_char_count": self.selected_char_count,
            "source_char_count": self.source_char_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "manifest_id": self.manifest_id}


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """The original full snapshot plus a separate bounded derivative."""

    full_snapshot: SharedFieldSnapshot
    active_snapshot: SharedFieldSnapshot
    manifest: ProjectionManifest


def _lifecycle(
    store: DormantStore | None, span_id: str
) -> tuple[str, str | None]:
    if store is None:
        return "active", None
    latest = store.latest_span_lifecycle(span_id)
    if latest is None:
        return "active", None
    return latest.state, latest.event_id


def _masked(
    region: RegionState,
    span: FieldSpan,
    *,
    reason: str,
    lifecycle_state: str,
    lifecycle_event_id: str | None,
) -> MaskedSpan:
    return MaskedSpan(
        logical_region=region.name,
        span_id=span.span_id,
        canonical_hash=span.canonical_hash,
        exact_text_sha256=_text_sha(span.text),
        char_count=len(span.text),
        reason=reason,
        source=span.source,
        provenance=span.provenance,
        container_refs=span.container_refs,
        edge_refs=span.edge_refs,
        lifecycle_state=lifecycle_state,
        lifecycle_event_id=lifecycle_event_id,
    )


def build_active_projection(
    snapshot: SharedFieldSnapshot,
    *,
    policy: ProjectionPolicy | None = None,
    store: DormantStore | None = None,
) -> ProjectionResult:
    """Select a bounded, whole-span active snapshot without changing source."""

    if not isinstance(snapshot, SharedFieldSnapshot):
        raise TypeError("snapshot must be SharedFieldSnapshot")
    active_policy = ProjectionPolicy() if policy is None else policy
    if not isinstance(active_policy, ProjectionPolicy):
        raise TypeError("policy must be ProjectionPolicy")
    source_hash_before = snapshot.field_id
    pinned = frozenset(LogicalRegion(value) for value in active_policy.pinned_regions)

    selected_by_region: dict[LogicalRegion, list[FieldSpan]] = {
        region: [] for region in CANONICAL_REGION_ORDER
    }
    selected_audit: list[SelectedSpan] = []
    masked_audit: list[MaskedSpan] = []
    selected_ids: set[tuple[LogicalRegion, str]] = set()
    selected_chars = 0
    selected_spans = 0

    # Pinned evidence is admitted before any discretionary history.  A pinned
    # oversize span fails closed because slicing it would silently alter text.
    for region_name in CANONICAL_REGION_ORDER:
        if region_name not in pinned:
            continue
        region = snapshot.region(region_name)
        if region.visibility is RegionVisibility.MASKED and region.spans:
            raise ProjectionBudgetError(
                f"pinned region {region_name.value} is explicitly masked"
            )
        eligible: list[tuple[FieldSpan, str, str | None]] = []
        for span in region.spans:
            state, event_id = _lifecycle(store, span.span_id)
            if state in {"masked", "dormant"}:
                # "Pinned" means currently active critical evidence is
                # admitted before discretionary history.  It must not mean
                # that an append-only tool-result region grows in RAM forever.
                masked_audit.append(
                    _masked(
                        region,
                        span,
                        reason=f"lifecycle_{state}",
                        lifecycle_state=state,
                        lifecycle_event_id=event_id,
                    )
                )
                continue
            eligible.append((span, state, event_id))

        region_total = sum(len(span.text) for span, _, _ in eligible)
        if region_total > active_policy.budget_for(region_name):
            raise ProjectionBudgetError(
                f"pinned region {region_name.value} requires {region_total} "
                "characters and cannot fit without slicing"
            )
        if selected_chars + region_total > active_policy.global_char_budget:
            raise ProjectionBudgetError("pinned spans exceed global character budget")
        if selected_spans + len(eligible) > active_policy.max_selected_spans:
            raise ProjectionBudgetError("pinned spans exceed selected-span budget")
        for span, state, event_id in eligible:
            selected_by_region[region_name].append(span)
            selected_ids.add((region_name, span.span_id))
            selected_chars += len(span.text)
            selected_spans += 1
            selected_audit.append(
                SelectedSpan(
                    logical_region=region_name,
                    span_id=span.span_id,
                    canonical_hash=span.canonical_hash,
                    exact_text_sha256=_text_sha(span.text),
                    char_count=len(span.text),
                    pinned=True,
                    lifecycle_state=state,
                    lifecycle_event_id=event_id,
                )
            )

    # For non-pinned regions prefer newer whole spans.  Restore canonical span
    # order in the derived region after selection.
    for region_name in CANONICAL_REGION_ORDER:
        if region_name in pinned:
            continue
        region = snapshot.region(region_name)
        region_remaining = active_policy.budget_for(region_name)
        admitted_reverse: list[FieldSpan] = []
        for span in reversed(region.spans):
            state, event_id = _lifecycle(store, span.span_id)
            if region.visibility is RegionVisibility.MASKED:
                masked_audit.append(
                    _masked(
                        region,
                        span,
                        reason="region_masked",
                        lifecycle_state=state,
                        lifecycle_event_id=event_id,
                    )
                )
                continue
            if state in {"masked", "dormant"}:
                masked_audit.append(
                    _masked(
                        region,
                        span,
                        reason=f"lifecycle_{state}",
                        lifecycle_state=state,
                        lifecycle_event_id=event_id,
                    )
                )
                continue
            length = len(span.text)
            reason: str | None = None
            if length > region_remaining:
                reason = "whole_span_exceeds_region_budget"
            elif selected_chars + length > active_policy.global_char_budget:
                reason = "global_character_budget"
            elif selected_spans >= active_policy.max_selected_spans:
                reason = "global_span_budget"
            if reason is not None:
                masked_audit.append(
                    _masked(
                        region,
                        span,
                        reason=reason,
                        lifecycle_state=state,
                        lifecycle_event_id=event_id,
                    )
                )
                continue
            admitted_reverse.append(span)
            selected_ids.add((region_name, span.span_id))
            selected_chars += length
            selected_spans += 1
            region_remaining -= length
            selected_audit.append(
                SelectedSpan(
                    logical_region=region_name,
                    span_id=span.span_id,
                    canonical_hash=span.canonical_hash,
                    exact_text_sha256=_text_sha(span.text),
                    char_count=length,
                    pinned=False,
                    lifecycle_state=state,
                    lifecycle_event_id=event_id,
                )
            )
        selected_by_region[region_name] = list(reversed(admitted_reverse))

    # Defensive completeness gate: every source span is either selected or
    # represented once in the omission manifest.
    omitted_keys = {
        (item.logical_region, item.span_id) for item in masked_audit
    }
    source_keys = {
        (region.name, span.span_id)
        for region in snapshot.regions
        for span in region.spans
    }
    if source_keys != selected_ids | omitted_keys:
        raise ProjectionError("projection did not account for every source span")
    if selected_ids & omitted_keys:
        raise ProjectionError("projection both selected and masked one span")

    active_regions = tuple(
        RegionState(
            name=region.name,
            spans=tuple(selected_by_region[region.name]),
            visibility=region.visibility,
            write_policy=region.write_policy,
        )
        for region in snapshot.regions
    )
    active_snapshot = SharedFieldSnapshot(
        tick_id=snapshot.tick_id,
        regions=active_regions,
        parent_field_id=snapshot.field_id,
        source_manifest_ids=snapshot.source_manifest_ids,
    )
    selected_audit.sort(
        key=lambda item: (
            CANONICAL_REGION_ORDER.index(item.logical_region),
            next(
                index
                for index, span in enumerate(
                    snapshot.region(item.logical_region).spans
                )
                if span.span_id == item.span_id
            ),
        )
    )
    masked_audit.sort(
        key=lambda item: (
            CANONICAL_REGION_ORDER.index(item.logical_region),
            next(
                index
                for index, span in enumerate(
                    snapshot.region(item.logical_region).spans
                )
                if span.span_id == item.span_id
            ),
        )
    )
    source_chars = sum(
        len(span.text) for region in snapshot.regions for span in region.spans
    )
    manifest = ProjectionManifest(
        source_field_id=snapshot.field_id,
        active_field_id=active_snapshot.field_id,
        policy_id=active_policy.policy_id,
        selected=tuple(selected_audit),
        masked=tuple(masked_audit),
        selected_char_count=selected_chars,
        source_char_count=source_chars,
    )
    if snapshot.field_id != source_hash_before:
        raise ProjectionError("projection mutated the canonical source snapshot")
    if selected_chars > active_policy.global_char_budget:
        raise ProjectionError("projection exceeds global character budget")
    return ProjectionResult(
        full_snapshot=snapshot,
        active_snapshot=active_snapshot,
        manifest=manifest,
    )


@dataclass(frozen=True, slots=True)
class SurfacingOmission:
    triple_revision_id: str
    reason: str
    omission_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "triple_revision_id": self.triple_revision_id,
            "reason": self.reason,
            "omission_sha256": self.omission_sha256,
        }


@dataclass(frozen=True, slots=True)
class SurfacingResult:
    source_snapshot: SharedFieldSnapshot
    surfaced_snapshot: SharedFieldSnapshot
    spans: tuple[FieldSpan, ...]
    omissions: tuple[SurfacingOmission, ...]
    record: Surfacing


def surface_retrieval(
    snapshot: SharedFieldSnapshot,
    retrieval: RetrievalResult,
    *,
    store: DormantStore | None = None,
) -> SurfacingResult:
    """Append retrieved readable triples to a derived knowledge region."""

    if not isinstance(snapshot, SharedFieldSnapshot):
        raise TypeError("snapshot must be SharedFieldSnapshot")
    spans: list[FieldSpan] = []
    for rank, hit in enumerate(retrieval.hits):
        provenance_payload = {
            "schema": "surfaced-triple-provenance-v1",
            "query": retrieval.query,
            "rank": rank,
            "triple_revision_id": hit.triple_revision_id,
            "exact_sources": [ref.to_dict() for ref in hit.provenance_refs],
            "retrieval_score": hit.score,
        }
        span_payload = {
            "source_field_id": snapshot.field_id,
            "text": hit.text,
            "provenance": provenance_payload,
            "container_refs": list(hit.container_refs),
            "edge_refs": list(hit.edge_refs),
        }
        spans.append(
            FieldSpan(
                span_id=_content_id("surfaced-span", span_payload),
                text=hit.text,
                kind="surfaced_bootstrap_knowledge",
                source="dormant_store",
                provenance=json.dumps(
                    provenance_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                confidence=0.5,
                container_refs=hit.container_refs,
                edge_refs=hit.edge_refs,
            )
        )
    omissions = tuple(
        SurfacingOmission(
            triple_revision_id=revision_id,
            reason="explicit_retrieval_limit",
            omission_sha256=_canonical_sha(
                {
                    "revision_id": revision_id,
                    "retrieval_omitted_sha256": retrieval.omitted_sha256,
                }
            ),
        )
        for revision_id in retrieval.omitted_revision_ids
    )
    knowledge = snapshot.region(LogicalRegion.STRUCTURED_KNOWLEDGE)
    regions = tuple(
        RegionState(
            name=region.name,
            spans=(
                (*region.spans, *spans)
                if region.name is LogicalRegion.STRUCTURED_KNOWLEDGE
                else region.spans
            ),
            visibility=region.visibility,
            write_policy=region.write_policy,
        )
        for region in snapshot.regions
    )
    del knowledge  # the region lookup above also validates canonical presence
    surfaced_snapshot = SharedFieldSnapshot(
        tick_id=snapshot.tick_id,
        regions=regions,
        parent_field_id=snapshot.field_id,
        source_manifest_ids=snapshot.source_manifest_ids,
    )
    record = Surfacing(
        query=retrieval.query,
        source_field_id=snapshot.field_id,
        retrieved_revision_ids=tuple(
            hit.triple_revision_id for hit in retrieval.hits
        ),
        omitted_revision_ids=retrieval.omitted_revision_ids,
        omitted_sha256=retrieval.omitted_sha256,
        output_span_ids=tuple(span.span_id for span in spans),
        record_limit=len(retrieval.hits),
        provenance="deterministic_text_overlap_retrieval_v1",
    )
    if store is not None:
        store.append_surfacing(record)
    return SurfacingResult(
        source_snapshot=snapshot,
        surfaced_snapshot=surfaced_snapshot,
        spans=tuple(spans),
        omissions=omissions,
        record=record,
    )


def retrieve_and_surface(
    store: DormantStore,
    snapshot: SharedFieldSnapshot,
    query: str,
    *,
    limit: int,
) -> SurfacingResult:
    retrieval = store.retrieve(query, limit=limit, accepted_only=True)
    return surface_retrieval(snapshot, retrieval, store=store)


__all__ = [
    "PINNED_REGIONS",
    "DEFAULT_REGION_BUDGETS",
    "ProjectionError",
    "ProjectionBudgetError",
    "ProjectionPolicy",
    "SelectedSpan",
    "MaskedSpan",
    "ProjectionManifest",
    "ProjectionResult",
    "build_active_projection",
    "SurfacingOmission",
    "SurfacingResult",
    "surface_retrieval",
    "retrieve_and_surface",
]
