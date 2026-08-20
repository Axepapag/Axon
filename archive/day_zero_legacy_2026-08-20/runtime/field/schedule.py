"""Deterministic multi-tick paging over checkpoint-compatible field views.

The canonical field is not bounded by the inherited 384-position interface.
This module advances one immutable, episode-level read cursor so successive
ticks cover all supported, attended characters in the read windows.  Proposal
region changes do not reset that cursor.  The writable proposal window remains
independent and always renders the tag-aware tail of the current region.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from substrate import ALPHABET_SET

from .schema import (
    CORE_WRITABLE_REGIONS,
    LogicalRegion,
    RegionVisibility,
    SharedFieldSnapshot,
    canonical_sha256,
)
from .view import (
    CONTEXT_CANDIDATE_REGIONS,
    CONTEXT_END,
    CONTEXT_START,
    USER_END,
    USER_START,
    FieldView,
    SlotKind,
    compile_field_view,
    proposal_tail_offset,
)

# Scratch and response-draft are writable transaction surfaces.  They remain
# visible as either the proposal tail or opportunistic context, but a read
# cycle is complete when every sealed context region plus user_input has been
# covered.  This makes cycle completion independent of which writable region
# happens to be the proposal on a given tick.
PAGED_CONTEXT_REGIONS: tuple[LogicalRegion, ...] = tuple(
    region
    for region in CONTEXT_CANDIDATE_REGIONS
    if region not in CORE_WRITABLE_REGIONS
)
READ_PAGING_PROTOCOL = "canonical_read_cursor_v1"


@dataclass(frozen=True, slots=True)
class FieldViewCursor:
    """Immutable cursor for one read-region coverage cycle."""

    page_index: int = 0
    context_offsets: tuple[tuple[str, int], ...] = ()
    user_offset: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.page_index, bool)
            or not isinstance(self.page_index, int)
            or self.page_index < 0
        ):
            raise ValueError("page_index must be a nonnegative integer")
        if (
            isinstance(self.user_offset, bool)
            or not isinstance(self.user_offset, int)
            or self.user_offset < 0
        ):
            raise ValueError("user_offset must be a nonnegative integer")
        normalized: list[tuple[str, int]] = []
        seen: set[LogicalRegion] = set()
        allowed = set(CONTEXT_CANDIDATE_REGIONS)
        for raw_region, raw_offset in self.context_offsets:
            try:
                region = LogicalRegion(raw_region)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"unknown cursor region {raw_region!r}"
                ) from exc
            if region not in allowed:
                raise ValueError(
                    f"{region.value!r} is not a pageable context region"
                )
            if region in seen:
                raise ValueError(
                    f"duplicate cursor region {region.value!r}"
                )
            if (
                isinstance(raw_offset, bool)
                or not isinstance(raw_offset, int)
                or raw_offset < 0
            ):
                raise ValueError(
                    f"cursor offset for {region.value!r} must be nonnegative"
                )
            seen.add(region)
            if raw_offset:
                normalized.append((region.value, raw_offset))
        object.__setattr__(
            self,
            "context_offsets",
            tuple(sorted(normalized)),
        )

    def context_mapping(self) -> dict[LogicalRegion, int]:
        return {
            LogicalRegion(region): offset
            for region, offset in self.context_offsets
        }

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "page_index": self.page_index,
            "context_offsets": {
                region: offset for region, offset in self.context_offsets
            },
            "user_offset": self.user_offset,
        }


@dataclass(frozen=True, slots=True)
class ReadPageCoverage:
    """Exact character-reference coverage for one bounded read page."""

    character_count: int
    unique_character_count: int
    duplicate_character_count: int
    region_character_counts: tuple[tuple[str, int], ...]
    refs_sha256: str

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "character_count": self.character_count,
            "unique_character_count": self.unique_character_count,
            "duplicate_character_count": self.duplicate_character_count,
            "region_character_counts": {
                region: count
                for region, count in self.region_character_counts
            },
            "refs_sha256": self.refs_sha256,
        }


@dataclass(frozen=True, slots=True)
class PagedFieldView:
    """One view plus the cursor required for the following tick."""

    view: FieldView
    cursor: FieldViewCursor
    next_cursor: FieldViewCursor
    proposal_offset: int
    read_cycle_complete: bool
    coverage: ReadPageCoverage

    def to_audit_dict(self) -> dict[str, object]:
        """Return the exact JSON-safe paging contract for a built tick."""

        return {
            "protocol": READ_PAGING_PROTOCOL,
            "page_index": self.cursor.page_index,
            "cursor_before": self.cursor.to_canonical_dict(),
            "cursor_after": self.next_cursor.to_canonical_dict(),
            "read_cycle_complete": self.read_cycle_complete,
            "read_view_hash": self.view.view_hash,
            "proposal_region": self.view.proposal_region.value,
            "proposal_view_offset": self.proposal_offset,
            "coverage": self.coverage.to_canonical_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReadCycleCoverage:
    expected_characters: int
    observed_characters: int
    duplicate_characters: int
    missing_refs: tuple[tuple[str, str, int], ...]

    @property
    def passed(self) -> bool:
        return not self.missing_refs and self.duplicate_characters == 0


def _actual_context_regions(
    proposal_region: LogicalRegion,
) -> tuple[LogicalRegion, ...]:
    return tuple(
        region
        for region in CONTEXT_CANDIDATE_REGIONS
        if region is not proposal_region
    )


def _selected_counts(
    view: FieldView,
) -> tuple[dict[LogicalRegion, int], int]:
    context: dict[LogicalRegion, int] = {}
    for ref in view.slot_refs[CONTEXT_START:CONTEXT_END]:
        if ref.kind is SlotKind.SPAN and ref.logical_region is not None:
            context[ref.logical_region] = (
                context.get(ref.logical_region, 0) + 1
            )
    user = sum(
        ref.kind is SlotKind.SPAN
        and ref.logical_region is LogicalRegion.USER_INPUT
        for ref in view.slot_refs[USER_START:USER_END]
    )
    return context, int(user)


def _supported_character_count(
    snapshot: SharedFieldSnapshot,
    region: LogicalRegion,
) -> int:
    state = snapshot.region(region)
    if state.visibility is not RegionVisibility.ATTENDED:
        return 0
    return sum(char in ALPHABET_SET for span in state.spans for char in span.text)


def _read_cycle_has_more(
    snapshot: SharedFieldSnapshot,
    *,
    context_offsets: Mapping[LogicalRegion, int],
    user_offset: int,
) -> bool:
    return any(
        context_offsets.get(region, 0)
        < _supported_character_count(snapshot, region)
        for region in PAGED_CONTEXT_REGIONS
    ) or (
        user_offset
        < _supported_character_count(snapshot, LogicalRegion.USER_INPUT)
    )


def read_page_coverage(view: FieldView) -> ReadPageCoverage:
    """Hash and count the exact canonical span references read on one page."""

    refs = [
        (
            ref.logical_region.value,
            ref.span_id,
            ref.span_char_index,
        )
        for ref in view.slot_refs[:USER_END]
        if (
            ref.kind is SlotKind.SPAN
            and ref.logical_region is not None
            and ref.span_id is not None
            and ref.span_char_index is not None
        )
    ]
    unique_refs = set(refs)
    region_counts: dict[str, int] = {}
    for region, _, _ in refs:
        region_counts[region] = region_counts.get(region, 0) + 1
    return ReadPageCoverage(
        character_count=len(refs),
        unique_character_count=len(unique_refs),
        duplicate_character_count=len(refs) - len(unique_refs),
        region_character_counts=tuple(sorted(region_counts.items())),
        refs_sha256=canonical_sha256(refs),
    )


def compile_next_read_page(
    snapshot: SharedFieldSnapshot,
    *,
    proposal_region: LogicalRegion | str = LogicalRegion.RESPONSE_DRAFT,
    cursor: FieldViewCursor | None = None,
    proposal_offset: int | None = None,
) -> PagedFieldView:
    """Compile one read page and deterministically advance its cursor."""

    if not isinstance(snapshot, SharedFieldSnapshot):
        raise TypeError("snapshot must be SharedFieldSnapshot")
    try:
        proposal = (
            proposal_region
            if isinstance(proposal_region, LogicalRegion)
            else LogicalRegion(proposal_region)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown proposal region {proposal_region!r}") from exc
    current = FieldViewCursor() if cursor is None else cursor
    if not isinstance(current, FieldViewCursor):
        raise TypeError("cursor must be FieldViewCursor")
    if proposal_offset is None:
        proposal_offset = proposal_tail_offset(
            proposal,
            snapshot.region(proposal).text,
        )
    if (
        isinstance(proposal_offset, bool)
        or not isinstance(proposal_offset, int)
        or proposal_offset < 0
    ):
        raise ValueError("proposal_offset must be a nonnegative integer")

    all_offsets = current.context_mapping()
    actual_regions = set(_actual_context_regions(proposal))
    active_offsets = {
        region: offset
        for region, offset in all_offsets.items()
        if region in actual_regions
    }
    view = compile_field_view(
        snapshot,
        proposal_region=proposal,
        context_offsets=active_offsets,
        user_offset=current.user_offset,
        proposal_offset=proposal_offset,
    )
    context_counts, user_count = _selected_counts(view)
    next_offsets = dict(all_offsets)
    for region, count in context_counts.items():
        next_offsets[region] = next_offsets.get(region, 0) + count
    next_user_offset = current.user_offset + user_count
    has_more = _read_cycle_has_more(
        snapshot,
        context_offsets=next_offsets,
        user_offset=next_user_offset,
    )
    progress = user_count + sum(
        context_counts.get(region, 0)
        for region in PAGED_CONTEXT_REGIONS
    )
    if has_more and progress == 0:
        raise RuntimeError(
            "read-page cursor cannot advance despite capacity omissions"
        )

    if has_more:
        next_cursor = FieldViewCursor(
            page_index=current.page_index + 1,
            context_offsets=tuple(
                (region.value, offset)
                for region, offset in next_offsets.items()
            ),
            user_offset=next_user_offset,
        )
    else:
        next_cursor = FieldViewCursor()
    return PagedFieldView(
        view=view,
        cursor=current,
        next_cursor=next_cursor,
        proposal_offset=proposal_offset,
        read_cycle_complete=not has_more,
        coverage=read_page_coverage(view),
    )


def collect_read_cycle(
    snapshot: SharedFieldSnapshot,
    *,
    proposal_region: LogicalRegion | str = LogicalRegion.RESPONSE_DRAFT,
) -> tuple[PagedFieldView, ...]:
    """Materialize exactly one complete, finite read-coverage cycle."""

    pages: list[PagedFieldView] = []
    cursor = FieldViewCursor()
    seen: set[FieldViewCursor] = set()
    while True:
        if cursor in seen:
            raise RuntimeError("read-page cursor repeated before cycle completion")
        seen.add(cursor)
        page = compile_next_read_page(
            snapshot,
            proposal_region=proposal_region,
            cursor=cursor,
        )
        pages.append(page)
        if page.read_cycle_complete:
            return tuple(pages)
        cursor = page.next_cursor


def audit_read_cycle_coverage(
    snapshot: SharedFieldSnapshot,
    pages: tuple[PagedFieldView, ...],
    *,
    proposal_region: LogicalRegion | str | None = LogicalRegion.RESPONSE_DRAFT,
) -> ReadCycleCoverage:
    """Prove a page cycle showed every supported attended read character once."""

    proposal = None
    if proposal_region is not None:
        proposal = (
            proposal_region
            if isinstance(proposal_region, LogicalRegion)
            else LogicalRegion(proposal_region)
        )
    expected: set[tuple[str, str, int]] = set()
    for region_name in (
        *PAGED_CONTEXT_REGIONS,
        LogicalRegion.USER_INPUT,
    ):
        region = snapshot.region(region_name)
        if region.visibility is not RegionVisibility.ATTENDED:
            continue
        for span in region.spans:
            for index, char in enumerate(span.text):
                if char in ALPHABET_SET:
                    expected.add((region_name.value, span.span_id, index))

    observed: list[tuple[str, str, int]] = []
    for page in pages:
        if page.view.source_field_id != snapshot.field_id:
            raise ValueError("page belongs to a different canonical snapshot")
        if proposal is not None and page.view.proposal_region is not proposal:
            raise ValueError("page proposal region differs from the audit")
        for ref in page.view.slot_refs[:USER_END]:
            if (
                ref.kind is SlotKind.SPAN
                and ref.logical_region is not None
                and ref.span_id is not None
                and ref.span_char_index is not None
            ):
                key = (
                    ref.logical_region.value,
                    ref.span_id,
                    ref.span_char_index,
                )
                if key in expected:
                    observed.append(key)
    observed_set = set(observed)
    return ReadCycleCoverage(
        expected_characters=len(expected),
        observed_characters=len(observed_set),
        duplicate_characters=len(observed) - len(observed_set),
        missing_refs=tuple(sorted(expected - observed_set)),
    )


class CyclingFieldViewProvider:
    """Stateful provider for callers that want one read page per tick.

    One cursor is maintained for the complete episode, including across
    scratch/response proposal switches.  If a canonical field change makes a
    stored offset invalid, compilation fails closed; the caller must explicitly
    start a new coverage cycle instead of silently skipping or repeating
    material.
    """

    def __init__(self) -> None:
        self._cursor = FieldViewCursor()
        self.records: list[PagedFieldView] = []

    def __call__(
        self,
        snapshot: SharedFieldSnapshot,
        target_region: LogicalRegion,
        tick_index: int,
    ) -> FieldView:
        del tick_index
        page = compile_next_read_page(
            snapshot,
            proposal_region=target_region,
            cursor=self._cursor,
            proposal_offset=proposal_tail_offset(
                target_region,
                snapshot.region(target_region).text,
            ),
        )
        self._cursor = page.next_cursor
        self.records.append(page)
        return page.view


__all__ = [
    "PAGED_CONTEXT_REGIONS",
    "READ_PAGING_PROTOCOL",
    "FieldViewCursor",
    "ReadPageCoverage",
    "PagedFieldView",
    "ReadCycleCoverage",
    "read_page_coverage",
    "compile_next_read_page",
    "collect_read_cycle",
    "audit_read_cycle_coverage",
    "CyclingFieldViewProvider",
]
