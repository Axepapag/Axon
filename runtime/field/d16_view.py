"""Exact width-independent D16 Shared Field views and deterministic view deltas.

This is the continuous-Core transport surface.  It materializes the permitted
Shared Field directly in Axon's registered 16D Unicode transport without
packing cells into a model-width rail.  The resulting view is derived and
non-authoritative; Heart's canonical SharedFieldSnapshot remains truth.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from substrate import (
    SLOT_DIM,
    UNICODE_TRANSPORT_SCHEMA,
    InvalidUnicodeScalarError,
    decode_unicode_tokens,
    encode_unicode_character,
    encode_unicode_text,
    transport_token_cell16,
    transport_token_kind,
    transport_token_value,
)

from .schema import (
    LogicalRegion,
    RegionMaskPolicy,
    RegionVisibility,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_region_order,
    canonical_sha256,
    resolve_mask_policy,
)

D16_VIEW_SCHEMA = "axon-field-d16-view-v1"
D16_VIEW_DELTA_SCHEMA = "axon-field-d16-view-delta-v1"
D16_WIDTH = 16


class D16ViewError(RuntimeError):
    """Base class for deterministic D16 view failures."""


class D16ResyncRequired(D16ViewError):
    """The current mirror cannot safely consume the offered incremental event."""


@dataclass(frozen=True, slots=True)
class D16CellAddress:
    region: LogicalRegion
    region_position: int
    global_position: int
    span_id: str
    span_position: int
    source: str
    provenance: str
    character: str
    transport_token_id: int
    transport_kind: str
    transport_value: str | int
    transport_unit_index: int
    transport_unit_count: int
    attended_interval_index: int
    attended_interval_start: int
    attended_interval_end: int

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "region": self.region.value,
            "region_position": self.region_position,
            "global_position": self.global_position,
            "span_id": self.span_id,
            "span_position": self.span_position,
            "source": self.source,
            "provenance": self.provenance,
            "character": self.character,
            "transport_token_id": self.transport_token_id,
            "transport_kind": self.transport_kind,
            "transport_value": self.transport_value,
            "transport_unit_index": self.transport_unit_index,
            "transport_unit_count": self.transport_unit_count,
            "attended_interval_index": self.attended_interval_index,
            "attended_interval_start": self.attended_interval_start,
            "attended_interval_end": self.attended_interval_end,
        }


@dataclass(frozen=True, slots=True)
class D16RegionView:
    region: LogicalRegion
    text: str
    cells16: np.ndarray
    addresses: tuple[D16CellAddress, ...]
    cells_sha256: str = field(init=False)
    address_sha256: str = field(init=False)
    region_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cells = np.asarray(self.cells16, dtype=np.float32)
        addresses = tuple(self.addresses)
        if cells.ndim != 2 or cells.shape[1] != D16_WIDTH:
            raise ValueError("D16RegionView.cells16 must have shape [N, 16]")
        if cells.shape[0] != len(addresses):
            raise ValueError("D16 region cells and addresses must have equal length")
        token_ids = tuple(address.transport_token_id for address in addresses)
        decoded = decode_unicode_tokens(token_ids)
        if decoded != self.text:
            raise D16ViewError("D16 region token stream does not roundtrip to declared text")
        for index, address in enumerate(addresses):
            expected = np.asarray(transport_token_cell16(address.transport_token_id), dtype=np.float32)
            if not np.array_equal(cells[index], expected):
                raise D16ViewError("D16 region cell does not equal its registered transport token cell")
            if address.region is not self.region:
                raise D16ViewError("D16 region address belongs to a different logical region")
        cells = cells.astype(np.float32, copy=False)
        cells.setflags(write=False)
        object.__setattr__(self, "cells16", cells)
        object.__setattr__(self, "addresses", addresses)
        cells_sha = hashlib.sha256(cells.astype("<f4", copy=False).tobytes(order="C")).hexdigest()
        address_sha = hashlib.sha256(
            canonical_json_bytes([address.to_canonical_dict() for address in addresses])
        ).hexdigest()
        object.__setattr__(self, "cells_sha256", cells_sha)
        object.__setattr__(self, "address_sha256", address_sha)
        object.__setattr__(
            self,
            "region_hash",
            canonical_sha256(
                {
                    "region": self.region.value,
                    "text": self.text,
                    "cells_sha256": cells_sha,
                    "address_sha256": address_sha,
                }
            ),
        )

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(address.transport_token_id for address in self.addresses)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "region": self.region.value,
            "text": self.text,
            "transport_units": len(self.addresses),
            "cells_sha256": self.cells_sha256,
            "address_sha256": self.address_sha256,
            "region_hash": self.region_hash,
        }


@dataclass(frozen=True, slots=True)
class D16ViewIdentity:
    field_id: str
    tick_id: int
    view_id: str
    mask_id: str
    transport_schema: str
    view_hash: str
    region_hashes: tuple[tuple[str, str], ...]

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "tick_id": self.tick_id,
            "view_id": self.view_id,
            "mask_id": self.mask_id,
            "transport_schema": self.transport_schema,
            "view_hash": self.view_hash,
            "region_hashes": [list(item) for item in self.region_hashes],
        }


@dataclass(frozen=True, slots=True)
class D16View:
    source_field_id: str
    source_tick_id: int
    mask_id: str
    regions: tuple[D16RegionView, ...]
    transport_schema: str = UNICODE_TRANSPORT_SCHEMA
    view_hash: str = field(init=False)
    view_id: str = field(init=False)

    def __post_init__(self) -> None:
        if SLOT_DIM != D16_WIDTH:
            raise D16ViewError(f"frozen substrate width changed: expected 16, got {SLOT_DIM}")
        if not self.source_field_id:
            raise ValueError("source_field_id must be non-empty")
        regions = tuple(self.regions)
        if len({item.region for item in regions}) != len(regions):
            raise ValueError("D16 view regions must be unique")
        object.__setattr__(self, "regions", regions)
        view_hash = canonical_sha256(
            {
                "schema": D16_VIEW_SCHEMA,
                "mask_id": self.mask_id,
                "transport_schema": self.transport_schema,
                "regions": [item.to_canonical_dict() for item in regions],
            }
        )
        view_id = canonical_sha256(
            {
                "schema": D16_VIEW_SCHEMA,
                "field_id": self.source_field_id,
                "tick_id": self.source_tick_id,
                "mask_id": self.mask_id,
                "transport_schema": self.transport_schema,
                "view_hash": view_hash,
            }
        )
        object.__setattr__(self, "view_hash", view_hash)
        object.__setattr__(self, "view_id", view_id)

    @property
    def identity(self) -> D16ViewIdentity:
        return D16ViewIdentity(
            field_id=self.source_field_id,
            tick_id=self.source_tick_id,
            view_id=self.view_id,
            mask_id=self.mask_id,
            transport_schema=self.transport_schema,
            view_hash=self.view_hash,
            region_hashes=tuple((item.region.value, item.region_hash) for item in self.regions),
        )

    @property
    def cells16(self) -> np.ndarray:
        if not self.regions or sum(item.cells16.shape[0] for item in self.regions) == 0:
            cells = np.zeros((0, D16_WIDTH), dtype=np.float32)
        else:
            cells = np.concatenate(tuple(item.cells16 for item in self.regions), axis=0).astype(np.float32, copy=False)
        cells.setflags(write=False)
        return cells

    def region(self, region: LogicalRegion | str) -> D16RegionView:
        logical = region if isinstance(region, LogicalRegion) else LogicalRegion(region)
        for item in self.regions:
            if item.region is logical:
                return item
        raise KeyError(f"D16 view has no region {logical.value!r}")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": D16_VIEW_SCHEMA,
            "source_field_id": self.source_field_id,
            "source_tick_id": self.source_tick_id,
            "mask_id": self.mask_id,
            "transport_schema": self.transport_schema,
            "view_hash": self.view_hash,
            "view_id": self.view_id,
            "regions": [item.to_canonical_dict() for item in self.regions],
        }


@dataclass(frozen=True, slots=True)
class D16RegionPatch:
    region: LogicalRegion
    target: D16RegionView

    def __post_init__(self) -> None:
        if self.target.region is not self.region:
            raise ValueError("D16 region patch target region mismatch")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"region": self.region.value, "target": self.target.to_canonical_dict()}


@dataclass(frozen=True, slots=True)
class D16ViewDelta:
    base: D16ViewIdentity
    target: D16ViewIdentity
    patches: tuple[D16RegionPatch, ...]
    delta_id: str = field(init=False)

    def __post_init__(self) -> None:
        patches = tuple(self.patches)
        if len({item.region for item in patches}) != len(patches):
            raise ValueError("D16 view delta patches must target unique regions")
        object.__setattr__(self, "patches", patches)
        object.__setattr__(
            self,
            "delta_id",
            canonical_sha256(
                {
                    "schema": D16_VIEW_DELTA_SCHEMA,
                    "base": self.base.to_canonical_dict(),
                    "target": self.target.to_canonical_dict(),
                    "patches": [item.to_canonical_dict() for item in patches],
                }
            ),
        )

    @classmethod
    def between(cls, base: D16View, target: D16View) -> "D16ViewDelta":
        if base.transport_schema != target.transport_schema:
            raise D16ResyncRequired("transport schema changed; full D16 snapshot required")
        if base.mask_id != target.mask_id:
            raise D16ResyncRequired("mask policy identity changed; full D16 snapshot required")
        if tuple(item.region for item in base.regions) != tuple(item.region for item in target.regions):
            raise D16ResyncRequired("logical region inventory changed; full D16 snapshot required")
        patches = tuple(
            D16RegionPatch(region=after.region, target=after)
            for before, after in zip(base.regions, target.regions, strict=True)
            if before.region_hash != after.region_hash
        )
        return cls(base=base.identity, target=target.identity, patches=patches)

    def apply(self, current: D16View) -> D16View:
        if current.identity != self.base:
            raise D16ResyncRequired("D16 delta base identity does not match current mirror")
        replacements = {patch.region: patch.target for patch in self.patches}
        regions = tuple(replacements.get(item.region, item) for item in current.regions)
        result = D16View(
            source_field_id=self.target.field_id,
            source_tick_id=self.target.tick_id,
            mask_id=self.target.mask_id,
            transport_schema=self.target.transport_schema,
            regions=regions,
        )
        if result.identity != self.target:
            raise D16ResyncRequired("D16 delta reconstruction does not match target identity")
        return result

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": D16_VIEW_DELTA_SCHEMA,
            "delta_id": self.delta_id,
            "base": self.base.to_canonical_dict(),
            "target": self.target.to_canonical_dict(),
            "patches": [item.to_canonical_dict() for item in self.patches],
        }


def _effective_mask_identity(snapshot: SharedFieldSnapshot, masks: Mapping[LogicalRegion, RegionMaskPolicy]) -> str:
    payload: list[dict[str, Any]] = []
    for region in canonical_region_order(snapshot.schema_version):
        state = snapshot.region(region)
        override = masks.get(region)
        if override is not None:
            spec: dict[str, Any] = {"mode": "override", "policy": override.to_canonical_dict()}
        elif state.mask_policy is not None:
            spec = {"mode": "region_policy", "policy": state.mask_policy.to_canonical_dict()}
        elif state.visibility is RegionVisibility.ATTENDED and (
            (not state.text and not state.attended_intervals)
            or (
                len(state.attended_intervals) == 1
                and state.attended_intervals[0].start == 0
                and state.attended_intervals[0].end == len(state.text)
            )
        ):
            spec = {"mode": "all"}
        elif not state.attended_intervals:
            spec = {"mode": "none"}
        else:
            spec = {
                "mode": "explicit_intervals",
                "intervals": [item.to_canonical_dict() for item in state.attended_intervals],
            }
        payload.append({"region": region.value, "spec": spec})
    return canonical_sha256({"schema": "axon-d16-mask-policy-v1", "regions": payload})


def materialize_d16_view(
    snapshot: SharedFieldSnapshot,
    *,
    region_masks: Mapping[LogicalRegion, RegionMaskPolicy] | None = None,
) -> D16View:
    """Materialize the exact permitted Shared Field directly as D16 region streams."""

    if not isinstance(snapshot, SharedFieldSnapshot):
        raise TypeError("materialize_d16_view requires SharedFieldSnapshot")
    if SLOT_DIM != D16_WIDTH:
        raise D16ViewError(f"frozen substrate width changed: expected 16, got {SLOT_DIM}")
    masks = dict(region_masks or {})
    mask_id = _effective_mask_identity(snapshot, masks)
    regions: list[D16RegionView] = []
    global_position = 0

    for region in canonical_region_order(snapshot.schema_version):
        state = snapshot.region(region)
        text = state.text
        try:
            encode_unicode_text(text)
        except InvalidUnicodeScalarError as exc:
            raise D16ViewError(f"canonical region {region.value!r} contains invalid Unicode scalar text") from exc
        override = masks.get(region)
        intervals = resolve_mask_policy(state.spans, override) if override is not None else tuple(state.attended_intervals)
        visible_text = "".join(text[item.start : item.end] for item in intervals)

        span_ranges: list[tuple[int, int, Any]] = []
        offset = 0
        for span in state.spans:
            end = offset + len(span.text)
            span_ranges.append((offset, end, span))
            offset = end

        cells: list[np.ndarray] = []
        addresses: list[D16CellAddress] = []
        for interval_index, interval in enumerate(intervals):
            for span_start, span_end, span in span_ranges:
                group_start = max(interval.start, span_start)
                group_end = min(interval.end, span_end)
                if group_start >= group_end:
                    continue
                for region_position in range(group_start, group_end):
                    span_position = region_position - span_start
                    character = span.text[span_position]
                    try:
                        encoded = encode_unicode_character(character)
                    except InvalidUnicodeScalarError as exc:
                        raise D16ViewError(
                            f"attended canonical value is not a Unicode scalar at {region.value}:{region_position}"
                        ) from exc
                    for unit_index, token_id in enumerate(encoded.token_ids):
                        cells.append(np.asarray(transport_token_cell16(token_id), dtype=np.float32))
                        addresses.append(
                            D16CellAddress(
                                region=region,
                                region_position=region_position,
                                global_position=global_position + region_position,
                                span_id=span.span_id,
                                span_position=span_position,
                                source=span.source,
                                provenance=span.provenance,
                                character=character,
                                transport_token_id=token_id,
                                transport_kind=transport_token_kind(token_id),
                                transport_value=transport_token_value(token_id),
                                transport_unit_index=unit_index,
                                transport_unit_count=len(encoded.token_ids),
                                attended_interval_index=interval_index,
                                attended_interval_start=interval.start,
                                attended_interval_end=interval.end,
                            )
                        )

        array = np.stack(cells, axis=0).astype(np.float32, copy=False) if cells else np.zeros((0, D16_WIDTH), dtype=np.float32)
        region_view = D16RegionView(region=region, text=visible_text, cells16=array, addresses=tuple(addresses))
        if region_view.text != visible_text:
            raise D16ViewError("D16 region materialization changed visible text")
        regions.append(region_view)
        global_position += len(text)

    return D16View(
        source_field_id=snapshot.field_id,
        source_tick_id=snapshot.tick_id,
        mask_id=mask_id,
        regions=tuple(regions),
    )


__all__ = [
    "D16CellAddress",
    "D16RegionPatch",
    "D16RegionView",
    "D16ResyncRequired",
    "D16View",
    "D16ViewDelta",
    "D16ViewError",
    "D16ViewIdentity",
    "D16_VIEW_DELTA_SCHEMA",
    "D16_VIEW_SCHEMA",
    "D16_WIDTH",
    "materialize_d16_view",
]
