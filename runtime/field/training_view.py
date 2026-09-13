"""Derived, content-addressed training-attention view over the canonical field.

A TrainingAttentionView is compiled FROM one unchanged source
SharedFieldSnapshot.  It is never a clone with emptied regions: the canonical
body, exact region/span/character addresses, and stored history are untouched.
Masking changes attendance only.

The view attends exactly the two dedicated training regions:

- ``trainer_instructions``: the full committed assignment text;
- ``training_responses``: a selectable history window over the attending
  core's own prior attempts.

Every other canonical region is masked to nothing for this view.  All D64
rows are produced by the existing ``D64FieldCompiler`` masked path
(``region_masks`` parameter); this module never re-implements packing and
never mutates the snapshot.

Receipts can dereference every emitted lane back to its exact canonical
source address because the compiled rail's ``CanonicalCharAddress`` records
already carry region, region_position, global_position, span_id,
span_position, source, and provenance.  ``TrainingAttentionView.verify``
proves each of those addresses against the source snapshot.

Attempt-span convention: one committed attempt is one ``FieldSpan`` in
``training_responses`` whose ``source`` is the authoring core id.  Heart
commits attempts under that convention; the window selects suffixes of those
spans.  The existing compiler masked path expresses policies only as
all/none/last_n_spans/tail_percent over the whole region, so a core-filtered
window is accepted only when the selected attempts are exactly the trailing
spans of the region; anything else fails closed with ``TrainingViewError``
rather than silently attending the wrong cells.

The view identity (``view_id``) is a canonical digest over the source field
identity, source tick, schema version, cohort core ids, mask-policy id, and
the typed history-window spec.  Identical specs over an identical source are
byte-identical; any spec or source change changes ``view_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from substrate import transport_token_cell16

from .compiler_d64 import (
    D64_COMPILER_SCHEMA,
    CanonicalCharAddress,
    CompiledD64Field,
    D64FieldCompiler,
    StaleCompiledFieldError,
)
from .schema import (
    TRAINING_REGIONS,
    LogicalRegion,
    RegionMaskPolicy,
    RegionState,
    SharedFieldSnapshot,
    canonical_region_order,
    canonical_sha256,
)

TRAINING_VIEW_SCHEMA = "axon-field-training-view-v1"

HISTORY_WINDOW_KINDS = ("none", "all", "latest_n")


class TrainingViewError(RuntimeError):
    """Deterministic failure to compile or verify a training-attention view."""


@dataclass(frozen=True, slots=True)
class TrainingHistoryWindow:
    """Typed history window over one core's own prior attempt spans.

    ``none`` attends no prior attempts.  ``all`` attends every retained
    attempt of the core.  ``latest_n`` attends the newest ``limit`` attempts
    of the core (fewer when the retained history is shorter).  The window
    selects attendance only; it never deletes, moves, or rewrites canonical
    history.
    """

    kind: str = "none"
    limit: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in HISTORY_WINDOW_KINDS:
            raise ValueError(f"unsupported history window kind {self.kind!r}; expected one of {HISTORY_WINDOW_KINDS!r}")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise TypeError("TrainingHistoryWindow.limit must be an integer")
        if self.limit < 0:
            raise ValueError("TrainingHistoryWindow.limit must be non-negative")
        if self.kind != "latest_n" and self.limit != 0:
            raise ValueError("history window limit is only meaningful for latest_n")
        if self.kind == "latest_n" and self.limit < 1:
            raise ValueError("latest_n history window requires limit >= 1")

    @classmethod
    def none(cls) -> "TrainingHistoryWindow":
        return cls(kind="none")

    @classmethod
    def all(cls) -> "TrainingHistoryWindow":
        return cls(kind="all")

    @classmethod
    def latest_n(cls, limit: int) -> "TrainingHistoryWindow":
        return cls(kind="latest_n", limit=limit)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "limit": self.limit}


def _attempt_span_indices(spans: tuple[Any, ...], core_id: str) -> list[int]:
    return [index for index, span in enumerate(spans) if span.source == core_id]


def _selected_attempt_indices(
    state: RegionState,
    core_id: str,
    window: TrainingHistoryWindow,
) -> list[int]:
    """Return the region span indices the history window selects."""

    if window.kind == "none":
        return []
    attempts = _attempt_span_indices(state.spans, core_id)
    if window.kind == "latest_n":
        attempts = attempts[-window.limit :] if window.limit else []
    return attempts


def _window_attended_text(state: RegionState, core_id: str, window: TrainingHistoryWindow) -> str:
    """Exact text the window attends inside one unchanged region state."""

    selected = _selected_attempt_indices(state, core_id, window)
    return "".join(state.spans[index].text for index in selected)


def _resolve_responses_policy(
    state: RegionState,
    core_id: str,
    window: TrainingHistoryWindow,
) -> RegionMaskPolicy:
    """Express one core's attempt window through the compiler's masked path.

    The existing D64 compiler resolves ``RegionMaskPolicy`` values over all
    region spans.  A core-filtered window is therefore admitted only when the
    selected attempts are exactly the trailing spans of the region, which
    ``last_n_spans`` expresses exactly.  Any other shape fails closed.
    """

    if not state.spans:
        return RegionMaskPolicy("none")
    selected = _selected_attempt_indices(state, core_id, window)
    if not selected:
        return RegionMaskPolicy("none")
    tail_start = len(state.spans) - len(selected)
    if selected != list(range(tail_start, len(state.spans))):
        raise TrainingViewError(
            "the core's selected attempt history is not the trailing span set of "
            f"training_responses (selected span indices {selected!r} over "
            f"{len(state.spans)} spans); the existing compiler masked path "
            "cannot express this window without mutating canonical state"
        )
    return RegionMaskPolicy("last_n_spans", len(selected))


def _normalize_cohort(cohort_core_ids: Any, core_id: str) -> tuple[str, ...]:
    if isinstance(cohort_core_ids, (str, bytes)) or not isinstance(cohort_core_ids, tuple):
        raise TypeError("cohort_core_ids must be a tuple of core id strings")
    normalized: list[str] = []
    for raw in cohort_core_ids:
        if not isinstance(raw, str) or not raw:
            raise ValueError("cohort core ids must be non-empty strings")
        if raw not in normalized:
            normalized.append(raw)
    if core_id not in normalized:
        raise ValueError("the attending core must be a member of its training cohort")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class TrainingAttentionView:
    """One derived, content-addressed attended view of the canonical field.

    Carries the source identities, the typed history-window spec, the exact
    per-region mask policies, the compiled D64 rail, and the full lane
    address mapping so receipts dereference back to canonical addresses.
    """

    source_field_id: str
    source_tick_id: int
    source_schema_version: str
    source_canonical_hash: str
    core_id: str
    cohort_core_ids: tuple[str, ...]
    history_window: TrainingHistoryWindow
    region_policies: tuple[tuple[LogicalRegion, RegionMaskPolicy], ...]
    mask_policy_id: str
    compiled: CompiledD64Field
    view_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.compiled, CompiledD64Field):
            raise TypeError("TrainingAttentionView.compiled must be a CompiledD64Field")
        if self.compiled.source_field_id != self.source_field_id:
            raise TrainingViewError("compiled rail source field does not match the view source")
        if self.compiled.source_tick_id != self.source_tick_id:
            raise TrainingViewError("compiled rail source tick does not match the view source")
        if not isinstance(self.history_window, TrainingHistoryWindow):
            raise TypeError("history_window must be a TrainingHistoryWindow")
        seen: set[LogicalRegion] = set()
        normalized_policies: list[tuple[LogicalRegion, RegionMaskPolicy]] = []
        for raw_region, raw_policy in self.region_policies:
            region = raw_region if isinstance(raw_region, LogicalRegion) else LogicalRegion(raw_region)
            if region in seen:
                raise TrainingViewError(f"duplicate region policy for {region.value!r}")
            seen.add(region)
            if not isinstance(raw_policy, RegionMaskPolicy):
                raise TypeError("region policies must be RegionMaskPolicy values")
            normalized_policies.append((region, raw_policy))
        object.__setattr__(self, "region_policies", tuple(normalized_policies))
        object.__setattr__(self, "view_id", canonical_sha256(self._identity_dict()))

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_VIEW_SCHEMA,
            "d64_compiler_schema": D64_COMPILER_SCHEMA,
            "source_field_id": self.source_field_id,
            "source_tick_id": self.source_tick_id,
            "source_schema_version": self.source_schema_version,
            "source_canonical_hash": self.source_canonical_hash,
            "core_id": self.core_id,
            "cohort_core_ids": list(self.cohort_core_ids),
            "history_window": self.history_window.to_canonical_dict(),
            "mask_policy_id": self.mask_policy_id,
        }

    def policy_for(self, region: LogicalRegion | str) -> RegionMaskPolicy:
        logical = region if isinstance(region, LogicalRegion) else LogicalRegion(region)
        for policy_region, policy in self.region_policies:
            if policy_region is logical:
                return policy
        raise KeyError(f"region {logical.value!r} has no policy in this training view")

    @property
    def address_mapping(self) -> tuple[CanonicalCharAddress | None, ...]:
        """Full row-major lane mapping; every valid lane dereferences canonically."""

        return self.compiled.addresses

    @property
    def valid_addresses(self) -> tuple[CanonicalCharAddress, ...]:
        return tuple(address for address in self.compiled.addresses if address is not None)

    def address(self, row: int, lane: int) -> CanonicalCharAddress | None:
        return self.compiled.address(row, lane)

    def dereference(
        self,
        receipt: CanonicalCharAddress,
        snapshot: SharedFieldSnapshot,
    ) -> CanonicalCharAddress:
        """Prove one emitted lane receipt maps to its exact canonical address.

        The receipt must be the view's own address record for its
        (row_index, lane_index), and the canonical source snapshot must agree
        at the receipt's exact region/span/character position.  Returns the
        canonical source address on success; fails closed otherwise.
        """

        if not isinstance(receipt, CanonicalCharAddress):
            raise TypeError("receipt must be a CanonicalCharAddress")
        bound = self.address(receipt.row_index, receipt.lane_index)
        if bound is None or bound != receipt:
            raise TrainingViewError("receipt does not match this view's emitted lane address")
        self._assert_source_character(snapshot, receipt)
        return bound

    def verify(self, snapshot: SharedFieldSnapshot) -> None:
        """Prove the view is fresh and every emitted row dereferences exactly.

        Checks, against the unchanged source snapshot:
        - source identity freshness (field id, tick, schema, canonical hash);
        - attended text per region equals the full instructions text, the
          exact history-window text, and nothing for all other regions;
        - every valid lane address matches the canonical region/span text at
          its exact positions and its lane bytes match the exact transport
          cell for its token.
        """

        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("verify requires the source SharedFieldSnapshot")
        if snapshot.field_id != self.source_field_id or snapshot.tick_id != self.source_tick_id:
            raise StaleCompiledFieldError("training view is stale for the supplied canonical snapshot")
        if snapshot.schema_version != self.source_schema_version:
            raise TrainingViewError("source snapshot schema does not match the training view")
        if snapshot.canonical_hash != self.source_canonical_hash:
            raise TrainingViewError("source snapshot canonical hash does not match the training view")

        expected_texts = _expected_attended_texts(snapshot, self.core_id, self.history_window)
        for region, expected in expected_texts.items():
            observed = self.compiled.region_text(region)
            if observed != expected:
                raise TrainingViewError(
                    f"training view attended text mismatch in {region.value!r}: "
                    f"expected {len(expected)} chars, observed {len(observed)}"
                )
        for address in self.valid_addresses:
            if address.region not in TRAINING_REGIONS:
                raise TrainingViewError(
                    f"training view emitted a lane outside the training regions: {address.region.value!r}"
                )
            self._assert_source_character(snapshot, address)
            observed_cell = self.compiled.lane_cell16(address.row_index, address.lane_index)
            expected_cell = transport_token_cell16(address.transport_token_id)
            if not np.array_equal(observed_cell, expected_cell):
                raise TrainingViewError(
                    f"training view lane bytes do not match the exact transport cell at "
                    f"row {address.row_index}, lane {address.lane_index}"
                )

    @staticmethod
    def _assert_source_character(snapshot: SharedFieldSnapshot, address: CanonicalCharAddress) -> None:
        state = snapshot.region(address.region)
        if address.region_position >= len(state.text) or state.text[address.region_position] != address.character:
            raise TrainingViewError(
                f"address region_position {address.region_position} does not dereference to "
                f"character {address.character!r} in region {address.region.value!r}"
            )
        span = next((item for item in state.spans if item.span_id == address.span_id), None)
        if (
            span is None
            or address.span_position >= len(span.text)
            or span.text[address.span_position] != address.character
            or span.source != address.source
            or span.provenance != address.provenance
        ):
            raise TrainingViewError(
                f"address span {address.span_id!r} does not dereference to the exact canonical "
                f"span character in region {address.region.value!r}"
            )

    def to_canonical_dict(self) -> dict[str, Any]:
        """Serialization-safe identity for receipts and transport binding."""

        return {
            **self._identity_dict(),
            "view_id": self.view_id,
            "mask_policy_id": self.mask_policy_id,
            "region_policies": [
                {"region": region.value, "policy": policy.to_canonical_dict()}
                for region, policy in self.region_policies
            ],
            "rail_id": self.compiled.rail_id,
            "rows_sha256": self.compiled.coverage.rows_sha256,
            "address_sha256": self.compiled.coverage.address_sha256,
            "roundtrip_sha256": self.compiled.coverage.roundtrip_sha256,
        }


def _expected_attended_texts(
    snapshot: SharedFieldSnapshot,
    core_id: str,
    window: TrainingHistoryWindow,
) -> dict[LogicalRegion, str]:
    """The exact text a training view must attend in every canonical region."""

    expected: dict[LogicalRegion, str] = {}
    for region in canonical_region_order(snapshot.schema_version):
        state = snapshot.region(region)
        if region is LogicalRegion.TRAINER_INSTRUCTIONS:
            expected[region] = state.text
        elif region is LogicalRegion.TRAINING_RESPONSES:
            expected[region] = _window_attended_text(state, core_id, window)
        else:
            expected[region] = ""
    return expected


class TrainingAttentionViewCompiler:
    """Compile derived training-attention views from unchanged canonical state."""

    schema = TRAINING_VIEW_SCHEMA

    def compile(
        self,
        snapshot: SharedFieldSnapshot,
        *,
        core_id: str,
        cohort_core_ids: tuple[str, ...],
        history_window: TrainingHistoryWindow | None = None,
    ) -> TrainingAttentionView:
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("TrainingAttentionViewCompiler.compile requires SharedFieldSnapshot")
        region_order = canonical_region_order(snapshot.schema_version)
        if not set(TRAINING_REGIONS) <= set(region_order):
            raise TrainingViewError(
                f"shared-field schema {snapshot.schema_version!r} has no training regions; "
                "training-attention views require the schema-v4 training surface"
            )
        if not isinstance(core_id, str) or not core_id:
            raise ValueError("core_id must be a non-empty string")
        cohort = _normalize_cohort(cohort_core_ids, core_id)
        window = history_window if history_window is not None else TrainingHistoryWindow.none()
        if not isinstance(window, TrainingHistoryWindow):
            raise TypeError("history_window must be a TrainingHistoryWindow")

        responses_policy = _resolve_responses_policy(
            snapshot.region(LogicalRegion.TRAINING_RESPONSES),
            core_id,
            window,
        )
        policies: dict[LogicalRegion, RegionMaskPolicy] = {}
        for region in region_order:
            if region is LogicalRegion.TRAINER_INSTRUCTIONS:
                policies[region] = RegionMaskPolicy("all")
            elif region is LogicalRegion.TRAINING_RESPONSES:
                policies[region] = responses_policy
            else:
                policies[region] = RegionMaskPolicy("none")

        compiled = D64FieldCompiler().compile(snapshot, region_masks=dict(policies))
        mask_policy_id = canonical_sha256(
            {region.value: policies[region].to_canonical_dict() for region in region_order}
        )
        view = TrainingAttentionView(
            source_field_id=snapshot.field_id,
            source_tick_id=snapshot.tick_id,
            source_schema_version=snapshot.schema_version,
            source_canonical_hash=snapshot.canonical_hash,
            core_id=core_id,
            cohort_core_ids=cohort,
            history_window=window,
            region_policies=tuple((region, policies[region]) for region in region_order),
            mask_policy_id=mask_policy_id,
            compiled=compiled,
        )
        # Fail closed at compile time: the freshly compiled view must prove
        # itself against the exact source snapshot it was derived from.
        view.verify(snapshot)
        return view


__all__ = [
    "HISTORY_WINDOW_KINDS",
    "TRAINING_VIEW_SCHEMA",
    "TrainingAttentionView",
    "TrainingAttentionViewCompiler",
    "TrainingHistoryWindow",
    "TrainingViewError",
]
