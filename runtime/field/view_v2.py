"""Pure v2 identity-aware paging into the inherited 384 x 16 checkpoint view.

This module deliberately has no state, model, filesystem, or v1-compiler
integration. It exposes a hash-bound read-page protocol that makes the bounded
view explicit while preserving the complete canonical v2 field outside it.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np

from substrate import ALPHABET_SET, SLOT_DIM, char_to_slot, get_letter_bank

from runtime.axon_runtime.identity_v2_config import (
    IDENTITY_V2_CONTRACT_SCHEMA,
    IdentityV2Contract,
)

from .schema_v2 import (
    CANONICAL_REGION_ORDER_V2,
    CORE_WRITABLE_REGIONS_V2,
    LOGICAL_REGION_IDS_V2,
    CoreIdentityViewV2,
    IdentityCharterV2,
    LogicalRegionV2,
    PhysicalRoleV2,
    RegionVisibilityV2,
    SharedFieldSnapshotV2,
    WritePolicyV2,
    canonical_json_bytes,
    canonical_sha256,
    render_core_identity_view_v2,
)


V2_PROJECTION_PROTOCOL = "axon-v2-identity-read-page-v1"
N_SLOTS_V2 = 384
CONTEXT_START_V2 = 0
CONTEXT_END_V2 = 256
USER_START_V2 = 256
USER_END_V2 = 320
PROPOSAL_START_V2 = 320
PROPOSAL_END_V2 = 384
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _projection_plan_descriptor() -> dict[str, object]:
    """Return a fresh explicit, hash-bound description of this pure page plan."""

    return {
        "schema": V2_PROJECTION_PROTOCOL,
        "geometry": {
            "context": [CONTEXT_START_V2, CONTEXT_END_V2],
            "user": [USER_START_V2, USER_END_V2],
            "proposal": [PROPOSAL_START_V2, PROPOSAL_END_V2],
            "slot_dim": SLOT_DIM,
            "physical_roles": [
                int(PhysicalRoleV2.CONTEXT),
                int(PhysicalRoleV2.USER),
                int(PhysicalRoleV2.PROPOSAL),
            ],
        },
        "identity_policy": "derived_envelope_always_context_raw_charter_pageable",
        "context_policy": "one_explicit_region_per_page",
        "cursor_policy": "finite_canonical_region_cycle_v1",
        "unsupported_character_policy": "fail_closed",
    }


V2_PROJECTION_PLAN_HASH = canonical_sha256(_projection_plan_descriptor())


class V2ProjectionError(ValueError):
    """The v2 projection boundary rejected invalid or unauditable input."""


class V2SlotKind(str, Enum):
    TAG = "tag"
    SPAN = "span"
    DERIVED_IDENTITY = "derived_identity"
    BLANK = "blank"
    PADDING = "padding"


def _as_region(value: LogicalRegionV2 | str, label: str) -> LogicalRegionV2:
    if type(value) is LogicalRegionV2:
        return value
    if type(value) is not str:
        raise V2ProjectionError(f"{label} must be a LogicalRegionV2 or string")
    try:
        return LogicalRegionV2(value)
    except ValueError as exc:
        raise V2ProjectionError(f"unknown {label} {value!r}") from exc


def _as_role(value: PhysicalRoleV2 | int, label: str) -> PhysicalRoleV2:
    if type(value) is PhysicalRoleV2:
        return value
    if type(value) is not int:
        raise V2ProjectionError(f"{label} must be a PhysicalRoleV2 or integer")
    try:
        return PhysicalRoleV2(value)
    except ValueError as exc:
        raise V2ProjectionError(f"unknown {label} {value!r}") from exc


def _as_slot_kind(value: V2SlotKind | str) -> V2SlotKind:
    if type(value) is V2SlotKind:
        return value
    if type(value) is not str:
        raise V2ProjectionError("slot kind must be V2SlotKind or string")
    try:
        return V2SlotKind(value)
    except ValueError as exc:
        raise V2ProjectionError(f"unknown slot kind {value!r}") from exc


def _strict_nonnegative(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise V2ProjectionError(f"{label} must be a non-negative integer")
    return value


def _strict_hash(value: object, label: str) -> str:
    if type(value) is not str or _HEX64_RE.fullmatch(value) is None:
        raise V2ProjectionError(
            f"{label} must be an exact lower-case 64-character SHA-256 string"
        )
    return value


def _readonly_array(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(contiguous.shape)


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class V2ReadCursor:
    """Explicit no-state cursor for one v2 page and its finite read cycle."""

    context_region: LogicalRegionV2 | str = LogicalRegionV2.CONVERSATION_HISTORY
    context_offset: int = 0
    user_offset: int = 0
    proposal_offset: int = 0
    completed_context_regions: tuple[LogicalRegionV2 | str, ...] = ()

    def __post_init__(self) -> None:
        if type(self) is not V2ReadCursor:
            raise V2ProjectionError("V2ReadCursor does not permit subclass instances")
        region = _as_region(self.context_region, "cursor context_region")
        if region is LogicalRegionV2.USER_INPUT:
            raise V2ProjectionError("cursor context_region cannot be user_input")
        object.__setattr__(self, "context_region", region)
        for label in ("context_offset", "user_offset", "proposal_offset"):
            _strict_nonnegative(getattr(self, label), f"cursor {label}")
        if type(self.completed_context_regions) is not tuple:
            raise V2ProjectionError("cursor completed_context_regions must be a tuple")
        completed = tuple(
            _as_region(value, "cursor completed_context_regions item")
            for value in self.completed_context_regions
        )
        object.__setattr__(self, "completed_context_regions", completed)

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "context_region": self.context_region.value,
            "context_offset": self.context_offset,
            "user_offset": self.user_offset,
            "proposal_offset": self.proposal_offset,
            "completed_context_regions": [
                region.value for region in self.completed_context_regions
            ],
        }


def _strict_cursor_value(cursor: V2ReadCursor, label: str) -> V2ReadCursor:
    """Capture and rebuild an exact cursor so forged slots cannot leak outward."""

    if type(cursor) is not V2ReadCursor:
        raise V2ProjectionError(f"{label} must be exact V2ReadCursor")
    try:
        context_region = cursor.context_region
        context_offset = cursor.context_offset
        user_offset = cursor.user_offset
        proposal_offset = cursor.proposal_offset
        completed_context_regions = cursor.completed_context_regions
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise V2ProjectionError(f"{label} is missing required primitive fields") from exc
    try:
        return V2ReadCursor(
            context_region=context_region,
            context_offset=context_offset,
            user_offset=user_offset,
            proposal_offset=proposal_offset,
            completed_context_regions=completed_context_regions,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise V2ProjectionError(f"{label} reconstruction failed") from exc


@dataclass(frozen=True, slots=True)
class V2SlotRef:
    """Exact provenance of a physical v2 page row."""

    slot_index: int
    role: PhysicalRoleV2 | int
    kind: V2SlotKind | str
    logical_region: LogicalRegionV2 | str | None
    rendered_char: str | None = None
    span_id: str | None = None
    span_char_index: int | None = None
    region_char_index: int | None = None
    tag_char_index: int | None = None
    identity_envelope_char_index: int | None = None
    source: str = ""
    provenance: str = ""

    def __post_init__(self) -> None:
        if type(self) is not V2SlotRef:
            raise V2ProjectionError("V2SlotRef does not permit subclass instances")
        if type(self.slot_index) is not int or not 0 <= self.slot_index < N_SLOTS_V2:
            raise V2ProjectionError("V2SlotRef.slot_index is outside the 384-slot view")
        role = _as_role(self.role, "V2SlotRef.role")
        kind = _as_slot_kind(self.kind)
        region = self.logical_region
        if region is not None:
            region = _as_region(region, "V2SlotRef.logical_region")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "logical_region", region)

        if self.rendered_char is not None:
            if type(self.rendered_char) is not str or len(self.rendered_char) != 1:
                raise V2ProjectionError(
                    "V2SlotRef.rendered_char must be one exact character or None"
                )
            if self.rendered_char not in ALPHABET_SET:
                raise V2ProjectionError("V2SlotRef.rendered_char is unsupported")

        if kind in {
            V2SlotKind.TAG,
            V2SlotKind.SPAN,
            V2SlotKind.DERIVED_IDENTITY,
        } and self.rendered_char is None:
            raise V2ProjectionError(f"{kind.value} refs require rendered_char")

        if kind is V2SlotKind.SPAN:
            if region is None or type(self.span_id) is not str or not self.span_id:
                raise V2ProjectionError("span ref requires a logical region and span_id")
            if self.span_char_index is None or self.region_char_index is None:
                raise V2ProjectionError("span ref requires exact character offsets")
            _strict_nonnegative(self.span_char_index, "span_char_index")
            _strict_nonnegative(self.region_char_index, "region_char_index")
            if self.identity_envelope_char_index is not None:
                raise V2ProjectionError("raw span ref cannot carry envelope index")

        if kind is V2SlotKind.DERIVED_IDENTITY:
            if region is not LogicalRegionV2.IDENTITY:
                raise V2ProjectionError("derived identity must be logical identity")
            if self.identity_envelope_char_index is None:
                raise V2ProjectionError("derived identity requires envelope index")
            _strict_nonnegative(
                self.identity_envelope_char_index,
                "identity_envelope_char_index",
            )
            if (
                self.span_id is not None
                or self.span_char_index is not None
                or self.region_char_index is not None
            ):
                raise V2ProjectionError(
                    "derived identity must not impersonate raw charter span offsets"
                )

        if kind is V2SlotKind.PADDING:
            if self.logical_region is not None or self.rendered_char is not None:
                raise V2ProjectionError("padding cannot carry region or character")
        if kind is V2SlotKind.BLANK and self.rendered_char is not None:
            raise V2ProjectionError("blank proposal slot cannot carry a character")

        for label in ("source", "provenance"):
            if type(getattr(self, label)) is not str:
                raise V2ProjectionError(f"V2SlotRef.{label} must be a string")

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "slot_index": self.slot_index,
            "role": int(self.role),
            "kind": self.kind.value,
            "logical_region": (
                None if self.logical_region is None else self.logical_region.value
            ),
            "rendered_char": self.rendered_char,
            "span_id": self.span_id,
            "span_char_index": self.span_char_index,
            "region_char_index": self.region_char_index,
            "tag_char_index": self.tag_char_index,
            "identity_envelope_char_index": self.identity_envelope_char_index,
            "source": self.source,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class V2ViewOmission:
    """One exact contiguous raw canonical character range absent from this page."""

    logical_region: LogicalRegionV2 | str
    span_id: str
    span_char_start: int
    span_char_end: int
    region_char_start: int
    region_char_end: int
    reason: str
    omitted_text_sha256: str
    source: str = ""
    provenance: str = ""

    def __post_init__(self) -> None:
        if type(self) is not V2ViewOmission:
            raise V2ProjectionError("V2ViewOmission does not permit subclass instances")
        region = _as_region(self.logical_region, "V2ViewOmission.logical_region")
        object.__setattr__(self, "logical_region", region)
        if type(self.span_id) is not str or not self.span_id:
            raise V2ProjectionError("V2ViewOmission.span_id must be non-empty")
        if not (
            type(self.span_char_start) is int
            and type(self.span_char_end) is int
            and type(self.region_char_start) is int
            and type(self.region_char_end) is int
            and 0 <= self.span_char_start < self.span_char_end
            and 0 <= self.region_char_start < self.region_char_end
        ):
            raise V2ProjectionError("V2ViewOmission requires a non-empty valid range")
        if type(self.reason) is not str or not self.reason:
            raise V2ProjectionError("V2ViewOmission.reason must be non-empty")
        _strict_hash(self.omitted_text_sha256, "V2ViewOmission.omitted_text_sha256")
        if type(self.source) is not str or type(self.provenance) is not str:
            raise V2ProjectionError("V2ViewOmission source/provenance must be strings")

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "logical_region": self.logical_region.value,
            "span_id": self.span_id,
            "span_char_start": self.span_char_start,
            "span_char_end": self.span_char_end,
            "region_char_start": self.region_char_start,
            "region_char_end": self.region_char_end,
            "reason": self.reason,
            "omitted_text_sha256": self.omitted_text_sha256,
            "source": self.source,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class _RawChar:
    char: str
    region: LogicalRegionV2
    span_id: str
    span_char_index: int
    region_char_index: int
    source: str
    provenance: str

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.region.value, self.span_id, self.span_char_index)


@dataclass(frozen=True, slots=True)
class _RenderedChar:
    char: str
    region: LogicalRegionV2
    kind: V2SlotKind
    span_id: str | None = None
    span_char_index: int | None = None
    region_char_index: int | None = None
    tag_char_index: int | None = None
    identity_envelope_char_index: int | None = None
    source: str = ""
    provenance: str = ""


@dataclass(frozen=True, slots=True)
class _Authority:
    contract_id: str
    charter: IdentityCharterV2
    envelope_budget: int
    contract_source_manifest_id: str


def _raw_chars(snapshot: SharedFieldSnapshotV2, region: LogicalRegionV2) -> tuple[_RawChar, ...]:
    state = snapshot.region(region)
    chars: list[_RawChar] = []
    region_offset = 0
    for span in state.spans:
        for span_offset, char in enumerate(span.text):
            if char not in ALPHABET_SET:
                raise V2ProjectionError(
                    f"canonical {region.value!r} text contains unsupported character"
                )
            chars.append(
                _RawChar(
                    char=char,
                    region=region,
                    span_id=span.span_id,
                    span_char_index=span_offset,
                    region_char_index=region_offset + span_offset,
                    source=span.source,
                    provenance=span.provenance,
                )
            )
        region_offset += len(span.text)
    return tuple(chars)


def _render_tag(region: LogicalRegionV2, label: str | None = None) -> list[_RenderedChar]:
    text = f"[{label or region.value}]\n"
    return [
        _RenderedChar(
            char=char,
            region=region,
            kind=V2SlotKind.TAG,
            tag_char_index=index,
            source="v2_projection",
            provenance="logical_region_tag",
        )
        for index, char in enumerate(text)
    ]


def _render_envelope(view: CoreIdentityViewV2) -> list[_RenderedChar]:
    return [
        _RenderedChar(
            char=char,
            region=LogicalRegionV2.IDENTITY,
            kind=V2SlotKind.DERIVED_IDENTITY,
            identity_envelope_char_index=index,
            source="identity_v2_contract",
            provenance="derived_core_identity_envelope",
        )
        for index, char in enumerate(view.envelope)
    ]


def _strict_snapshot(snapshot: SharedFieldSnapshotV2) -> SharedFieldSnapshotV2:
    if type(snapshot) is not SharedFieldSnapshotV2:
        raise V2ProjectionError("snapshot must be exact SharedFieldSnapshotV2")
    try:
        tick_id = snapshot.tick_id
        regions = snapshot.regions
        parent_field_id = snapshot.parent_field_id
        source_manifest_ids = snapshot.source_manifest_ids
        field_id = snapshot.field_id
        canonical_hash = snapshot.canonical_hash
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise V2ProjectionError("snapshot is missing required primitive fields") from exc

    supplied_field_id = _strict_hash(field_id, "snapshot.field_id")
    supplied_canonical_hash = _strict_hash(
        canonical_hash,
        "snapshot.canonical_hash",
    )
    try:
        fresh = SharedFieldSnapshotV2(
            tick_id=tick_id,
            regions=regions,
            parent_field_id=parent_field_id,
            source_manifest_ids=source_manifest_ids,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise V2ProjectionError("snapshot reconstruction failed") from exc
    if (
        supplied_field_id != fresh.field_id
        or supplied_canonical_hash != fresh.canonical_hash
    ):
        raise V2ProjectionError("supplied snapshot hashes do not match canonical state")
    return fresh


def _authority_from_contract(contract: IdentityV2Contract) -> _Authority:
    if type(contract) is not IdentityV2Contract:
        raise V2ProjectionError("contract must be exact IdentityV2Contract")
    try:
        descriptor = contract.to_canonical_descriptor()
    except Exception as exc:
        raise V2ProjectionError("contract authority verification failed") from exc

    try:
        descriptor_bytes = canonical_json_bytes(descriptor)
        contract_id = hashlib.sha256(descriptor_bytes).hexdigest()
        identity = descriptor["identity"]
        if type(identity) is not dict:
            raise TypeError("identity descriptor is not an object")
        charter_text = identity["charter_text"]
        declared_hash = identity["charter_hash"]
        budget = identity["envelope_char_budget"]
        if descriptor["schema"] != IDENTITY_V2_CONTRACT_SCHEMA:
            raise ValueError("unexpected contract schema")
        charter = IdentityCharterV2(text=charter_text, charter_version=1)
        if type(declared_hash) is not str or declared_hash != charter.charter_hash:
            raise ValueError("contract charter hash mismatch")
        if type(budget) is not int or not 1 <= budget <= 192:
            raise ValueError("contract envelope budget is invalid")
    except (AttributeError, KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise V2ProjectionError("verified contract descriptor is malformed") from exc

    return _Authority(
        contract_id=contract_id,
        charter=charter,
        envelope_budget=budget,
        contract_source_manifest_id=(
            f"{IDENTITY_V2_CONTRACT_SCHEMA}:{contract_id}"
        ),
    )


def _verify_snapshot_identity(
    snapshot: SharedFieldSnapshotV2,
    authority: _Authority,
) -> None:
    identity = snapshot.region(LogicalRegionV2.IDENTITY)
    expected = authority.charter.build_identity_region()
    if (
        identity.visibility is not RegionVisibilityV2.ATTENDED
        or identity.write_policy is not WritePolicyV2.SEALED
        or len(identity.spans) != 1
        or identity.to_canonical_dict() != expected.to_canonical_dict()
    ):
        raise V2ProjectionError(
            "snapshot identity does not match sealed contract charter"
        )
    if authority.charter.source_manifest_id not in snapshot.source_manifest_ids:
        raise V2ProjectionError("snapshot is missing charter source manifest")
    if authority.contract_source_manifest_id not in snapshot.source_manifest_ids:
        raise V2ProjectionError("snapshot is missing contract source manifest")


def _render_identity_view(
    authority: _Authority,
    *,
    core_id: str,
    display_name: str,
    model_label: str,
    role_capabilities: tuple[str, ...],
    current_role: str,
) -> CoreIdentityViewV2:
    try:
        return render_core_identity_view_v2(
            authority.charter,
            core_id=core_id,
            display_name=display_name,
            model_label=model_label,
            role_capabilities=role_capabilities,
            current_role=current_role,
            envelope_char_budget=authority.envelope_budget,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise V2ProjectionError("core identity envelope is invalid or over budget") from exc


def _context_regions(proposal_region: LogicalRegionV2) -> tuple[LogicalRegionV2, ...]:
    return tuple(
        region
        for region in CANONICAL_REGION_ORDER_V2
        if region not in {LogicalRegionV2.USER_INPUT, proposal_region}
    )


def _validate_cursor(
    cursor: V2ReadCursor,
    *,
    snapshot: SharedFieldSnapshotV2,
    proposal_region: LogicalRegionV2,
) -> V2ReadCursor:
    cursor = _strict_cursor_value(cursor, "cursor")
    region = cursor.context_region
    order = _context_regions(proposal_region)
    completed = cursor.completed_context_regions
    if len(completed) > len(order):
        raise V2ProjectionError("cursor completed context region count is invalid")
    if region not in order:
        raise V2ProjectionError(
            "cursor context_region cannot be user_input or current proposal"
        )
    if completed != order[: len(completed)]:
        raise V2ProjectionError("cursor completed context regions are not canonical")
    if len(completed) == len(order):
        if region is not order[-1]:
            raise V2ProjectionError("terminal context cursor must remain at final region")
        terminal_length = len(_raw_chars(snapshot, region))
        if cursor.context_offset != terminal_length:
            raise V2ProjectionError("terminal context cursor has invalid offset")
    else:
        expected_region = order[len(completed)]
        if region is not expected_region:
            raise V2ProjectionError("cursor context_region must follow canonical v2 cycle")
    for label, region_name, offset in (
        ("context", region, cursor.context_offset),
        ("user", LogicalRegionV2.USER_INPUT, cursor.user_offset),
        ("proposal", proposal_region, cursor.proposal_offset),
    ):
        raw = _raw_chars(snapshot, region_name)
        if offset > len(raw):
            raise V2ProjectionError(
                f"{label} cursor offset exceeds canonical character count"
            )
    return cursor


def _write_rendered(
    *,
    start: int,
    rendered: Iterable[_RenderedChar],
    role: PhysicalRoleV2,
    field16: np.ndarray,
    logical_region_ids: np.ndarray,
    attention_mask: np.ndarray,
    refs: list[V2SlotRef],
) -> tuple[_RawChar, ...]:
    raw: list[_RawChar] = []
    for relative, item in enumerate(rendered):
        index = start + relative
        field16[index] = char_to_slot(item.char)
        logical_region_ids[index] = LOGICAL_REGION_IDS_V2[item.region]
        attention_mask[index] = True
        refs[index] = V2SlotRef(
            slot_index=index,
            role=role,
            kind=item.kind,
            logical_region=item.region,
            rendered_char=item.char,
            span_id=item.span_id,
            span_char_index=item.span_char_index,
            region_char_index=item.region_char_index,
            tag_char_index=item.tag_char_index,
            identity_envelope_char_index=item.identity_envelope_char_index,
            source=item.source,
            provenance=item.provenance,
        )
        if item.kind is V2SlotKind.SPAN:
            raw.append(
                _RawChar(
                    char=item.char,
                    region=item.region,
                    span_id=item.span_id or "",
                    span_char_index=item.span_char_index or 0,
                    region_char_index=item.region_char_index or 0,
                    source=item.source,
                    provenance=item.provenance,
                )
            )
    return tuple(raw)


def _initial_page_buffers(
    proposal_region: LogicalRegionV2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[V2SlotRef]]:
    field16 = np.zeros((N_SLOTS_V2, SLOT_DIM), dtype=np.float32)
    role_ids = np.empty(N_SLOTS_V2, dtype=np.int64)
    role_ids[CONTEXT_START_V2:CONTEXT_END_V2] = int(PhysicalRoleV2.CONTEXT)
    role_ids[USER_START_V2:USER_END_V2] = int(PhysicalRoleV2.USER)
    role_ids[PROPOSAL_START_V2:PROPOSAL_END_V2] = int(PhysicalRoleV2.PROPOSAL)
    logical_ids = np.full(N_SLOTS_V2, -1, dtype=np.int64)
    attention = np.zeros(N_SLOTS_V2, dtype=np.bool_)
    write = np.zeros(N_SLOTS_V2, dtype=np.bool_)
    write[PROPOSAL_START_V2:PROPOSAL_END_V2] = True
    attention[PROPOSAL_START_V2:PROPOSAL_END_V2] = True
    logical_ids[PROPOSAL_START_V2:PROPOSAL_END_V2] = LOGICAL_REGION_IDS_V2[
        proposal_region
    ]

    refs: list[V2SlotRef] = []
    for index in range(N_SLOTS_V2):
        if index < CONTEXT_END_V2:
            role = PhysicalRoleV2.CONTEXT
            refs.append(
                V2SlotRef(
                    slot_index=index,
                    role=role,
                    kind=V2SlotKind.PADDING,
                    logical_region=None,
                    source="v2_projection",
                    provenance="masked_padding",
                )
            )
        elif index < USER_END_V2:
            role = PhysicalRoleV2.USER
            refs.append(
                V2SlotRef(
                    slot_index=index,
                    role=role,
                    kind=V2SlotKind.PADDING,
                    logical_region=None,
                    source="v2_projection",
                    provenance="masked_padding",
                )
            )
        else:
            role = PhysicalRoleV2.PROPOSAL
            refs.append(
                V2SlotRef(
                    slot_index=index,
                    role=role,
                    kind=V2SlotKind.BLANK,
                    logical_region=proposal_region,
                    source="v2_projection",
                    provenance="active_blank_proposal",
                )
            )
    return field16, role_ids, logical_ids, attention, write, refs


def _raw_to_rendered(items: Iterable[_RawChar]) -> list[_RenderedChar]:
    return [
        _RenderedChar(
            char=item.char,
            region=item.region,
            kind=V2SlotKind.SPAN,
            span_id=item.span_id,
            span_char_index=item.span_char_index,
            region_char_index=item.region_char_index,
            source=item.source,
            provenance=item.provenance,
        )
        for item in items
    ]


def _dedicated_render(
    snapshot: SharedFieldSnapshotV2,
    region: LogicalRegionV2,
    *,
    capacity: int,
    offset: int,
) -> tuple[list[_RenderedChar], tuple[_RawChar, ...]]:
    tag = _render_tag(region)
    if len(tag) > capacity:
        raise V2ProjectionError(f"{region.value} tag exceeds physical capacity")
    raw = _raw_chars(snapshot, region)
    if offset > len(raw):
        raise V2ProjectionError(f"{region.value} offset exceeds canonical length")
    state = snapshot.region(region)
    selected = (
        raw[offset : offset + capacity - len(tag)]
        if state.visibility is RegionVisibilityV2.ATTENDED
        else ()
    )
    return tag + _raw_to_rendered(selected), tuple(selected)


def _omissions(
    snapshot: SharedFieldSnapshotV2,
    *,
    proposal_region: LogicalRegionV2,
    cursor: V2ReadCursor,
    selected_keys: set[tuple[str, str, int]],
) -> tuple[V2ViewOmission, ...]:
    omissions: list[V2ViewOmission] = []

    def reason_for(item: _RawChar) -> str | None:
        state = snapshot.region(item.region)
        if state.visibility is not RegionVisibilityV2.ATTENDED:
            return "region_masked"
        if item.key in selected_keys:
            return None
        if item.region is cursor.context_region:
            if item.region_char_index < cursor.context_offset:
                return "context_cursor_before"
            return "context_capacity"
        if item.region is LogicalRegionV2.USER_INPUT:
            if item.region_char_index < cursor.user_offset:
                return "user_cursor_before"
            return "user_capacity"
        if item.region is proposal_region:
            if item.region_char_index < cursor.proposal_offset:
                return "proposal_cursor_before"
            return "proposal_capacity"
        return "context_region_not_selected"

    for region in CANONICAL_REGION_ORDER_V2:
        run: list[_RawChar] = []
        run_reason: str | None = None

        def flush() -> None:
            nonlocal run, run_reason
            if not run:
                return
            first = run[0]
            last = run[-1]
            text = "".join(item.char for item in run)
            omissions.append(
                V2ViewOmission(
                    logical_region=first.region,
                    span_id=first.span_id,
                    span_char_start=first.span_char_index,
                    span_char_end=last.span_char_index + 1,
                    region_char_start=first.region_char_index,
                    region_char_end=last.region_char_index + 1,
                    reason=run_reason or "context_region_not_selected",
                    omitted_text_sha256=hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    source=first.source,
                    provenance=first.provenance,
                )
            )
            run = []
            run_reason = None

        for item in _raw_chars(snapshot, region):
            reason = reason_for(item)
            if reason is None:
                flush()
                continue
            contiguous = bool(run) and (
                item.span_id == run[-1].span_id
                and item.span_char_index == run[-1].span_char_index + 1
                and item.region_char_index == run[-1].region_char_index + 1
                and item.source == run[-1].source
                and item.provenance == run[-1].provenance
                and reason == run_reason
            )
            if run and not contiguous:
                flush()
            run.append(item)
            run_reason = reason
        flush()
    return tuple(omissions)


def _audit_partition(
    snapshot: SharedFieldSnapshotV2,
    refs: tuple[V2SlotRef, ...],
    omissions: tuple[V2ViewOmission, ...],
) -> None:
    expected: dict[tuple[str, str, int], _RawChar] = {}
    by_span: dict[tuple[str, str], tuple[_RawChar, ...]] = {}
    for region in CANONICAL_REGION_ORDER_V2:
        raw = _raw_chars(snapshot, region)
        for item in raw:
            expected[item.key] = item
        for span_id in {item.span_id for item in raw}:
            by_span[(region.value, span_id)] = tuple(
                item for item in raw if item.span_id == span_id
            )

    selected: list[tuple[str, str, int]] = []
    for ref in refs:
        if ref.kind is V2SlotKind.SPAN:
            if (
                ref.logical_region is None
                or ref.span_id is None
                or ref.span_char_index is None
            ):
                raise V2ProjectionError("raw span ref is incomplete")
            key = (
                ref.logical_region.value,
                ref.span_id,
                ref.span_char_index,
            )
            if key not in expected:
                raise V2ProjectionError("raw span ref does not belong to snapshot")
            selected.append(key)

    omitted: list[tuple[str, str, int]] = []
    for omission in omissions:
        items = by_span.get((omission.logical_region.value, omission.span_id))
        if items is None:
            raise V2ProjectionError("omission references unknown raw span")
        span_items = [
            item
            for item in items
            if omission.span_char_start <= item.span_char_index < omission.span_char_end
        ]
        if not span_items:
            raise V2ProjectionError("omission has no matching raw characters")
        first = span_items[0]
        last = span_items[-1]
        if (
            first.region_char_index != omission.region_char_start
            or last.region_char_index + 1 != omission.region_char_end
            or first.source != omission.source
            or first.provenance != omission.provenance
        ):
            raise V2ProjectionError("omission provenance or offsets do not match raw span")
        text = "".join(item.char for item in span_items)
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != omission.omitted_text_sha256:
            raise V2ProjectionError("omission text hash does not match raw span")
        omitted.extend(item.key for item in span_items)

    if len(selected) != len(set(selected)):
        raise V2ProjectionError("raw canonical character was selected twice")
    if len(omitted) != len(set(omitted)):
        raise V2ProjectionError("raw canonical character was omitted twice")
    if set(selected) & set(omitted):
        raise V2ProjectionError("raw canonical character was both selected and omitted")
    if set(selected) | set(omitted) != set(expected):
        raise V2ProjectionError("raw canonical character accounting is incomplete")


def _next_context(
    snapshot: SharedFieldSnapshotV2,
    *,
    proposal_region: LogicalRegionV2,
    region: LogicalRegionV2,
    offset: int,
    selected_count: int,
) -> tuple[LogicalRegionV2, int, tuple[LogicalRegionV2, ...], bool]:
    order = _context_regions(proposal_region)
    current_raw = _raw_chars(snapshot, region)
    current_visible = (
        len(current_raw)
        if snapshot.region(region).visibility is RegionVisibilityV2.ATTENDED
        else 0
    )
    current_end = offset + selected_count
    if current_end < current_visible:
        return region, current_end, (), False

    index = order.index(region)
    completed: list[LogicalRegionV2] = [region]
    for candidate in order[index + 1 :]:
        raw = _raw_chars(snapshot, candidate)
        if (
            snapshot.region(candidate).visibility is RegionVisibilityV2.ATTENDED
            and raw
        ):
            return candidate, 0, tuple(completed), False
        completed.append(candidate)
    return region, current_visible, tuple(completed), True


@dataclass(frozen=True, slots=True)
class IdentityAwareFieldViewV2:
    """One immutable, fully auditable v2 384 x 16 read page."""

    source_field_id: str
    source_canonical_hash: str
    source_parent_field_id: str | None
    source_tick_id: int
    contract_id: str
    charter_hash: str
    core_id: str
    display_name: str
    model_label: str
    role_capabilities: tuple[str, ...]
    current_role: str
    identity_view_hash: str
    identity_envelope_sha256: str
    projection_plan_hash: str
    proposal_region: LogicalRegionV2 | str
    cursor: V2ReadCursor
    next_cursor: V2ReadCursor
    read_cycle_complete: bool
    field16: np.ndarray
    role_ids: np.ndarray
    logical_region_ids: np.ndarray
    attention_mask: np.ndarray
    write_mask: np.ndarray
    slot_refs: tuple[V2SlotRef, ...]
    omissions: tuple[V2ViewOmission, ...]
    view_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not IdentityAwareFieldViewV2:
            raise V2ProjectionError(
                "IdentityAwareFieldViewV2 does not permit subclass instances"
            )
        for label in (
            "source_field_id",
            "source_canonical_hash",
            "contract_id",
            "charter_hash",
            "identity_view_hash",
            "identity_envelope_sha256",
            "projection_plan_hash",
        ):
            _strict_hash(getattr(self, label), label)
        if self.source_field_id != self.source_canonical_hash:
            raise V2ProjectionError("source field_id and canonical_hash must agree")
        if self.projection_plan_hash != V2_PROJECTION_PLAN_HASH:
            raise V2ProjectionError("projection plan hash is not the frozen v2 plan")
        if self.source_parent_field_id is not None:
            _strict_hash(self.source_parent_field_id, "source_parent_field_id")
        _strict_nonnegative(self.source_tick_id, "source_tick_id")
        for label in (
            "core_id",
            "display_name",
            "model_label",
            "current_role",
        ):
            if type(getattr(self, label)) is not str or not getattr(self, label):
                raise V2ProjectionError(f"{label} must be a non-empty exact string")
        if type(self.role_capabilities) is not tuple or not self.role_capabilities:
            raise V2ProjectionError("role_capabilities must be a non-empty tuple")
        if any(type(value) is not str or not value for value in self.role_capabilities):
            raise V2ProjectionError("role_capabilities must contain exact strings")
        if self.role_capabilities != tuple(sorted(set(self.role_capabilities))):
            raise V2ProjectionError("role_capabilities must be sorted and unique")
        if self.current_role not in self.role_capabilities:
            raise V2ProjectionError("current_role must be in role_capabilities")
        proposal = _as_region(self.proposal_region, "proposal_region")
        if proposal not in CORE_WRITABLE_REGIONS_V2:
            raise V2ProjectionError("proposal_region must be core-writable")
        cursor = _strict_cursor_value(self.cursor, "page cursor")
        next_cursor = _strict_cursor_value(self.next_cursor, "page next_cursor")
        if type(self.read_cycle_complete) is not bool:
            raise V2ProjectionError("read_cycle_complete must be boolean")

        field16 = _readonly_array(self.field16, np.dtype("<f4"))
        role_ids = _readonly_array(self.role_ids, np.dtype("<i8"))
        logical_ids = _readonly_array(self.logical_region_ids, np.dtype("<i8"))
        attention = _readonly_array(self.attention_mask, np.dtype("?"))
        write = _readonly_array(self.write_mask, np.dtype("?"))
        if field16.shape != (N_SLOTS_V2, SLOT_DIM):
            raise V2ProjectionError("field16 must be exactly (384, 16)")
        for label, array in (
            ("role_ids", role_ids),
            ("logical_region_ids", logical_ids),
            ("attention_mask", attention),
            ("write_mask", write),
        ):
            if array.shape != (N_SLOTS_V2,):
                raise V2ProjectionError(f"{label} must be exactly length 384")

        expected_roles = np.empty(N_SLOTS_V2, dtype=np.int64)
        expected_roles[:CONTEXT_END_V2] = int(PhysicalRoleV2.CONTEXT)
        expected_roles[USER_START_V2:USER_END_V2] = int(PhysicalRoleV2.USER)
        expected_roles[PROPOSAL_START_V2:] = int(PhysicalRoleV2.PROPOSAL)
        if not np.array_equal(role_ids, expected_roles):
            raise V2ProjectionError("role IDs do not preserve 0/1/2 checkpoint geometry")
        if np.any((logical_ids < -1) | (logical_ids >= len(LOGICAL_REGION_IDS_V2))):
            raise V2ProjectionError("logical region IDs are outside v2 metadata range")
        if np.any(write[:PROPOSAL_START_V2]) or not np.all(write[PROPOSAL_START_V2:]):
            raise V2ProjectionError("write mask must authorize exactly proposal window")
        if not np.all(attention[PROPOSAL_START_V2:]):
            raise V2ProjectionError("all proposal slots must stay attended")
        proposal_id = LOGICAL_REGION_IDS_V2[proposal]
        if not np.all(logical_ids[PROPOSAL_START_V2:] == proposal_id):
            raise V2ProjectionError("proposal metadata must name proposal region")

        refs = tuple(self.slot_refs)
        omissions = tuple(self.omissions)
        if len(refs) != N_SLOTS_V2:
            raise V2ProjectionError("slot_refs must contain exactly 384 entries")
        if any(type(ref) is not V2SlotRef for ref in refs):
            raise V2ProjectionError("slot_refs must contain exact V2SlotRef values")
        if any(ref.slot_index != index for index, ref in enumerate(refs)):
            raise V2ProjectionError("slot_refs must be in physical order")
        if any(
            ref.role is not PhysicalRoleV2.PROPOSAL
            for ref in refs[PROPOSAL_START_V2:]
        ):
            raise V2ProjectionError("proposal refs must use proposal physical role")
        if any(
            ref.logical_region is not proposal
            for ref in refs[PROPOSAL_START_V2:]
        ):
            raise V2ProjectionError("proposal refs must use proposal logical region")
        if any(type(item) is not V2ViewOmission for item in omissions):
            raise V2ProjectionError("omissions must contain exact V2ViewOmission values")

        envelope_chars: list[tuple[int, str]] = []
        for index, ref in enumerate(refs):
            expected_role = (
                PhysicalRoleV2.CONTEXT
                if index < CONTEXT_END_V2
                else PhysicalRoleV2.USER
                if index < USER_END_V2
                else PhysicalRoleV2.PROPOSAL
            )
            if ref.role is not expected_role:
                raise V2ProjectionError("slot ref role disagrees with physical geometry")
            if ref.kind in {
                V2SlotKind.TAG,
                V2SlotKind.SPAN,
                V2SlotKind.DERIVED_IDENTITY,
            }:
                if ref.logical_region is None or ref.rendered_char is None:
                    raise V2ProjectionError("rendered slot ref is incomplete")
                if (
                    not attention[index]
                    or bool(write[index]) != (index >= PROPOSAL_START_V2)
                ):
                    raise V2ProjectionError("rendered input slot has invalid masks")
                if logical_ids[index] != LOGICAL_REGION_IDS_V2[ref.logical_region]:
                    raise V2ProjectionError("slot ref logical region disagrees with metadata")
                if not np.array_equal(field16[index], char_to_slot(ref.rendered_char)):
                    raise V2ProjectionError("slot vector does not match frozen character")
            elif ref.kind is V2SlotKind.PADDING:
                if (
                    attention[index]
                    or write[index]
                    or logical_ids[index] != -1
                    or np.any(field16[index] != 0.0)
                ):
                    raise V2ProjectionError("padding slot does not preserve zero/mask contract")
            elif ref.kind is V2SlotKind.BLANK:
                if (
                    index < PROPOSAL_START_V2
                    or ref.logical_region is not proposal
                    or not attention[index]
                    or not write[index]
                    or logical_ids[index] != proposal_id
                    or np.any(field16[index] != 0.0)
                ):
                    raise V2ProjectionError("blank slot does not preserve proposal contract")
            if ref.kind is V2SlotKind.DERIVED_IDENTITY:
                if (
                    index >= CONTEXT_END_V2
                    or ref.logical_region is not LogicalRegionV2.IDENTITY
                    or ref.identity_envelope_char_index is None
                    or ref.rendered_char is None
                ):
                    raise V2ProjectionError("derived identity must remain context metadata")
                envelope_chars.append(
                    (ref.identity_envelope_char_index, ref.rendered_char)
                )

        if not envelope_chars:
            raise V2ProjectionError("identity-aware page is missing derived envelope")
        envelope_chars.sort()
        if [index for index, _ in envelope_chars] != list(range(len(envelope_chars))):
            raise V2ProjectionError("derived identity envelope indices are not contiguous")
        envelope = "".join(char for _, char in envelope_chars)
        if hashlib.sha256(envelope.encode("utf-8")).hexdigest() != self.identity_envelope_sha256:
            raise V2ProjectionError("identity envelope hash does not match slot refs")
        try:
            identity_view = CoreIdentityViewV2(
                charter_hash=self.charter_hash,
                core_id=self.core_id,
                display_name=self.display_name,
                model_label=self.model_label,
                role_capabilities=self.role_capabilities,
                current_role=self.current_role,
                envelope=envelope,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise V2ProjectionError("identity metadata does not reconstruct") from exc
        if identity_view.view_hash != self.identity_view_hash:
            raise V2ProjectionError("identity view hash does not match slot refs")

        object.__setattr__(self, "proposal_region", proposal)
        object.__setattr__(self, "cursor", cursor)
        object.__setattr__(self, "next_cursor", next_cursor)
        object.__setattr__(self, "field16", field16)
        object.__setattr__(self, "role_ids", role_ids)
        object.__setattr__(self, "logical_region_ids", logical_ids)
        object.__setattr__(self, "attention_mask", attention)
        object.__setattr__(self, "write_mask", write)
        object.__setattr__(self, "slot_refs", refs)
        object.__setattr__(self, "omissions", omissions)
        manifest = {
            "schema": V2_PROJECTION_PROTOCOL,
            "projection_plan_hash": self.projection_plan_hash,
            "source": {
                "field_id": self.source_field_id,
                "canonical_hash": self.source_canonical_hash,
                "parent_field_id": self.source_parent_field_id,
                "tick_id": self.source_tick_id,
            },
            "identity": {
                "contract_id": self.contract_id,
                "charter_hash": self.charter_hash,
                "core_id": self.core_id,
                "display_name": self.display_name,
                "model_label": self.model_label,
                "role_capabilities": list(self.role_capabilities),
                "current_role": self.current_role,
                "identity_view_hash": self.identity_view_hash,
                "identity_envelope_sha256": self.identity_envelope_sha256,
            },
            "page": {
                "proposal_region": proposal.value,
                "cursor": self.cursor.to_canonical_dict(),
                "next_cursor": self.next_cursor.to_canonical_dict(),
                "read_cycle_complete": self.read_cycle_complete,
            },
            "arrays": {
                "field16": {
                    "shape": list(field16.shape),
                    "dtype": field16.dtype.str,
                    "sha256": _array_sha256(field16),
                },
                "role_ids": {
                    "shape": list(role_ids.shape),
                    "dtype": role_ids.dtype.str,
                    "sha256": _array_sha256(role_ids),
                },
                "logical_region_ids": {
                    "shape": list(logical_ids.shape),
                    "dtype": logical_ids.dtype.str,
                    "sha256": _array_sha256(logical_ids),
                },
                "attention_mask": {
                    "shape": list(attention.shape),
                    "dtype": attention.dtype.str,
                    "sha256": _array_sha256(attention),
                },
                "write_mask": {
                    "shape": list(write.shape),
                    "dtype": write.dtype.str,
                    "sha256": _array_sha256(write),
                },
            },
            "slot_refs": [ref.to_canonical_dict() for ref in refs],
            "omissions": [item.to_canonical_dict() for item in omissions],
        }
        object.__setattr__(self, "view_hash", canonical_sha256(manifest))

    @property
    def region_ids(self) -> np.ndarray:
        """Audit-only alias; these values are never extra physical role IDs."""

        return self.logical_region_ids

    def decode(
        self,
        start: int = 0,
        end: int = N_SLOTS_V2,
        *,
        blanks_as: str = "",
    ) -> str:
        if not 0 <= start <= end <= N_SLOTS_V2:
            raise V2ProjectionError("decode range is outside 384 slots")
        return get_letter_bank().decode_sequence(
            self.field16[start:end],
            blanks_as=blanks_as,
        )


def _strict_page(page: IdentityAwareFieldViewV2) -> IdentityAwareFieldViewV2:
    """Rebuild one exact page and prove its supplied digest is still authentic."""

    if type(page) is not IdentityAwareFieldViewV2:
        raise V2ProjectionError("coverage pages must be exact v2 page values")
    try:
        supplied_view_hash = page.view_hash
        values = {
            "source_field_id": page.source_field_id,
            "source_canonical_hash": page.source_canonical_hash,
            "source_parent_field_id": page.source_parent_field_id,
            "source_tick_id": page.source_tick_id,
            "contract_id": page.contract_id,
            "charter_hash": page.charter_hash,
            "core_id": page.core_id,
            "display_name": page.display_name,
            "model_label": page.model_label,
            "role_capabilities": page.role_capabilities,
            "current_role": page.current_role,
            "identity_view_hash": page.identity_view_hash,
            "identity_envelope_sha256": page.identity_envelope_sha256,
            "projection_plan_hash": page.projection_plan_hash,
            "proposal_region": page.proposal_region,
            "cursor": page.cursor,
            "next_cursor": page.next_cursor,
            "read_cycle_complete": page.read_cycle_complete,
            "field16": page.field16,
            "role_ids": page.role_ids,
            "logical_region_ids": page.logical_region_ids,
            "attention_mask": page.attention_mask,
            "write_mask": page.write_mask,
            "slot_refs": page.slot_refs,
            "omissions": page.omissions,
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise V2ProjectionError("coverage page is missing required fields") from exc
    supplied = _strict_hash(supplied_view_hash, "coverage page view_hash")
    try:
        fresh = IdentityAwareFieldViewV2(**values)
    except V2ProjectionError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise V2ProjectionError("coverage page reconstruction failed") from exc
    if supplied != fresh.view_hash:
        raise V2ProjectionError("coverage page view_hash does not match content")
    return fresh


PagedIdentityFieldViewV2 = IdentityAwareFieldViewV2


@dataclass(frozen=True, slots=True)
class V2ReadCycleCoverage:
    expected_characters: int
    observed_characters: int
    duplicate_characters: int
    missing_refs: tuple[tuple[str, str, int], ...]

    @property
    def passed(self) -> bool:
        return self.duplicate_characters == 0 and not self.missing_refs


def compile_identity_aware_v2_read_page(
    contract: IdentityV2Contract,
    snapshot: SharedFieldSnapshotV2,
    *,
    core_id: str,
    display_name: str,
    model_label: str,
    role_capabilities: tuple[str, ...],
    current_role: str,
    proposal_region: LogicalRegionV2 | str = LogicalRegionV2.RESPONSE_DRAFT,
    cursor: V2ReadCursor | None = None,
) -> IdentityAwareFieldViewV2:
    """Compile one explicit in-memory v2 page without mutating inputs or state."""

    authority = _authority_from_contract(contract)
    fresh = _strict_snapshot(snapshot)
    _verify_snapshot_identity(fresh, authority)
    try:
        proposal = _as_region(proposal_region, "proposal_region")
    except V2ProjectionError:
        raise
    if proposal not in CORE_WRITABLE_REGIONS_V2:
        raise V2ProjectionError("proposal_region must be scratch or response_draft")
    current = V2ReadCursor() if cursor is None else cursor
    current = _validate_cursor(current, snapshot=fresh, proposal_region=proposal)
    identity_view = _render_identity_view(
        authority,
        core_id=core_id,
        display_name=display_name,
        model_label=model_label,
        role_capabilities=role_capabilities,
        current_role=current_role,
    )

    selected_context = current.context_region
    context_state = fresh.region(selected_context)
    context_tag = _render_tag(selected_context)
    identity_prefix = (
        _render_tag(LogicalRegionV2.IDENTITY, "identity_view")
        + _render_envelope(identity_view)
        + [
            _RenderedChar(
                char="\n",
                region=LogicalRegionV2.IDENTITY,
                kind=V2SlotKind.TAG,
                tag_char_index=len("[identity_view]\n"),
                source="v2_projection",
                provenance="identity_view_separator",
            )
        ]
    )
    if len(identity_prefix) + len(context_tag) > CONTEXT_END_V2:
        raise V2ProjectionError("identity envelope and mandatory tags exceed context")

    context_raw = _raw_chars(fresh, selected_context)
    if current.context_offset > len(context_raw):
        raise V2ProjectionError("context cursor offset exceeds canonical length")
    context_selected = (
        context_raw[
            current.context_offset : current.context_offset
            + CONTEXT_END_V2 - len(identity_prefix) - len(context_tag)
        ]
        if context_state.visibility is RegionVisibilityV2.ATTENDED
        else ()
    )
    context_rendered = (
        identity_prefix + context_tag + _raw_to_rendered(context_selected)
    )

    user_rendered, user_selected = _dedicated_render(
        fresh,
        LogicalRegionV2.USER_INPUT,
        capacity=USER_END_V2 - USER_START_V2,
        offset=current.user_offset,
    )
    proposal_rendered, proposal_selected = _dedicated_render(
        fresh,
        proposal,
        capacity=PROPOSAL_END_V2 - PROPOSAL_START_V2,
        offset=current.proposal_offset,
    )

    (
        field16,
        role_ids,
        logical_ids,
        attention,
        write,
        refs,
    ) = _initial_page_buffers(proposal)
    rendered_raw = set()
    for start, rendered, role in (
        (CONTEXT_START_V2, context_rendered, PhysicalRoleV2.CONTEXT),
        (USER_START_V2, user_rendered, PhysicalRoleV2.USER),
        (PROPOSAL_START_V2, proposal_rendered, PhysicalRoleV2.PROPOSAL),
    ):
        limit = (
            CONTEXT_END_V2
            if role is PhysicalRoleV2.CONTEXT
            else USER_END_V2
            if role is PhysicalRoleV2.USER
            else PROPOSAL_END_V2
        )
        if start + len(rendered) > limit:
            raise V2ProjectionError("rendered page exceeds physical window")
        written = _write_rendered(
            start=start,
            rendered=rendered,
            role=role,
            field16=field16,
            logical_region_ids=logical_ids,
            attention_mask=attention,
            refs=refs,
        )
        rendered_raw.update(item.key for item in written)

    expected_selected = {
        item.key for item in (*context_selected, *user_selected, *proposal_selected)
    }
    if rendered_raw != expected_selected:
        raise V2ProjectionError("renderer lost or forged selected raw character")

    omissions = _omissions(
        fresh,
        proposal_region=proposal,
        cursor=current,
        selected_keys=rendered_raw,
    )
    refs_tuple = tuple(refs)
    _audit_partition(fresh, refs_tuple, omissions)

    next_region, next_context_offset, completed_added, context_done = _next_context(
        fresh,
        proposal_region=proposal,
        region=selected_context,
        offset=current.context_offset,
        selected_count=len(context_selected),
    )
    user_total = len(_raw_chars(fresh, LogicalRegionV2.USER_INPUT))
    proposal_total = len(_raw_chars(fresh, proposal))
    next_user_offset = min(user_total, current.user_offset + len(user_selected))
    next_proposal_offset = min(
        proposal_total,
        current.proposal_offset + len(proposal_selected),
    )
    user_done = (
        fresh.region(LogicalRegionV2.USER_INPUT).visibility
        is not RegionVisibilityV2.ATTENDED
        or next_user_offset >= user_total
    )
    proposal_done = (
        fresh.region(proposal).visibility is not RegionVisibilityV2.ATTENDED
        or next_proposal_offset >= proposal_total
    )
    complete = context_done and user_done and proposal_done
    completed_context = current.completed_context_regions
    if len(completed_context) < len(_context_regions(proposal)):
        completed_context = completed_context + completed_added
    if context_done and len(completed_context) != len(_context_regions(proposal)):
        raise V2ProjectionError("context cursor completed an invalid region sequence")
    next_cursor = V2ReadCursor(
        context_region=next_region,
        context_offset=next_context_offset,
        user_offset=next_user_offset,
        proposal_offset=next_proposal_offset,
        completed_context_regions=completed_context,
    )

    return IdentityAwareFieldViewV2(
        source_field_id=fresh.field_id,
        source_canonical_hash=fresh.canonical_hash,
        source_parent_field_id=fresh.parent_field_id,
        source_tick_id=fresh.tick_id,
        contract_id=authority.contract_id,
        charter_hash=authority.charter.charter_hash,
        core_id=identity_view.core_id,
        display_name=identity_view.display_name,
        model_label=identity_view.model_label,
        role_capabilities=identity_view.role_capabilities,
        current_role=identity_view.current_role,
        identity_view_hash=identity_view.view_hash,
        identity_envelope_sha256=hashlib.sha256(
            identity_view.envelope.encode("utf-8")
        ).hexdigest(),
        projection_plan_hash=V2_PROJECTION_PLAN_HASH,
        proposal_region=proposal,
        cursor=current,
        next_cursor=next_cursor,
        read_cycle_complete=complete,
        field16=field16,
        role_ids=role_ids,
        logical_region_ids=logical_ids,
        attention_mask=attention,
        write_mask=write,
        slot_refs=refs_tuple,
        omissions=omissions,
    )


def collect_identity_aware_v2_read_cycle(
    contract: IdentityV2Contract,
    snapshot: SharedFieldSnapshotV2,
    *,
    core_id: str,
    display_name: str,
    model_label: str,
    role_capabilities: tuple[str, ...],
    current_role: str,
    proposal_region: LogicalRegionV2 | str = LogicalRegionV2.RESPONSE_DRAFT,
) -> tuple[IdentityAwareFieldViewV2, ...]:
    """Build one finite explicit coverage cycle without writing any state."""

    pages: list[IdentityAwareFieldViewV2] = []
    cursor = V2ReadCursor()
    seen: set[V2ReadCursor] = set()
    while True:
        if cursor in seen:
            raise V2ProjectionError("v2 read cursor repeated before completion")
        seen.add(cursor)
        page = compile_identity_aware_v2_read_page(
            contract,
            snapshot,
            core_id=core_id,
            display_name=display_name,
            model_label=model_label,
            role_capabilities=role_capabilities,
            current_role=current_role,
            proposal_region=proposal_region,
            cursor=cursor,
        )
        pages.append(page)
        if page.read_cycle_complete:
            return tuple(pages)
        cursor = page.next_cursor


def audit_v2_read_cycle_coverage(
    contract: IdentityV2Contract,
    snapshot: SharedFieldSnapshotV2,
    pages: tuple[IdentityAwareFieldViewV2, ...],
) -> V2ReadCycleCoverage:
    """Prove pages are the exact contract compiler output and cover raw input once.

    A coverage count alone is not authority: a malicious caller could build a
    self-consistent page with rewritten provenance, omissions, or source
    metadata.  The verified contract is therefore an explicit audit input and
    every supplied page is rebuilt through the sole compiler before it can
    contribute to coverage.
    """

    authority = _authority_from_contract(contract)
    fresh = _strict_snapshot(snapshot)
    _verify_snapshot_identity(fresh, authority)
    if not pages:
        raise V2ProjectionError("coverage requires at least one v2 page")
    expected: dict[tuple[str, str, int], _RawChar] = {}
    for region in CANONICAL_REGION_ORDER_V2:
        if fresh.region(region).visibility is RegionVisibilityV2.ATTENDED:
            expected.update(
                {item.key: item for item in _raw_chars(fresh, region)}
            )

    observed: list[tuple[str, str, int]] = []
    previous_next: V2ReadCursor | None = None
    proposal_region: LogicalRegionV2 | None = None
    cycle_identity: tuple[str, str, str, tuple[str, ...], str] | None = None
    for page_index, page in enumerate(pages):
        page = _strict_page(page)
        try:
            if page_index == 0:
                if page.cursor != V2ReadCursor():
                    raise V2ProjectionError("coverage cycle must start at canonical cursor")
                proposal_region = page.proposal_region
            elif page.cursor != previous_next:
                raise V2ProjectionError("coverage page cursor does not follow prior page")
            if page.proposal_region is not proposal_region:
                raise V2ProjectionError("coverage page proposal region changed mid-cycle")
            if page.read_cycle_complete != (page_index == len(pages) - 1):
                raise V2ProjectionError("coverage completion marker is not terminal")
            if (
                page.source_field_id != fresh.field_id
                or page.source_canonical_hash != fresh.canonical_hash
            ):
                raise V2ProjectionError("coverage page belongs to another source field")
            if page.source_parent_field_id != fresh.parent_field_id:
                raise V2ProjectionError("coverage page parent provenance differs from source")
            if page.source_tick_id != fresh.tick_id:
                raise V2ProjectionError("coverage page tick provenance differs from source")
            if page.contract_id != authority.contract_id:
                raise V2ProjectionError("coverage page contract differs from verified authority")
            if page.charter_hash != authority.charter.charter_hash:
                raise V2ProjectionError("coverage page charter hash differs from source")
            identity = (
                page.core_id,
                page.display_name,
                page.model_label,
                page.role_capabilities,
                page.current_role,
            )
            if cycle_identity is None:
                cycle_identity = identity
            elif identity != cycle_identity:
                raise V2ProjectionError("coverage page core identity changed mid-cycle")
        except V2ProjectionError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise V2ProjectionError("coverage page is missing required fields") from exc
        expected_page = compile_identity_aware_v2_read_page(
            contract,
            fresh,
            core_id=page.core_id,
            display_name=page.display_name,
            model_label=page.model_label,
            role_capabilities=page.role_capabilities,
            current_role=page.current_role,
            proposal_region=page.proposal_region,
            cursor=page.cursor,
        )
        if page.view_hash != expected_page.view_hash:
            raise V2ProjectionError(
                "coverage page does not equal exact contract compiler output"
            )
        refs = page.slot_refs
        _audit_partition(fresh, refs, page.omissions)
        for ref in refs:
            if (
                ref.kind is V2SlotKind.SPAN
                and ref.logical_region is not None
                and ref.span_id is not None
                and ref.span_char_index is not None
            ):
                key = (
                    ref.logical_region.value,
                    ref.span_id,
                    ref.span_char_index,
                )
                expected_item = expected.get(key)
                if expected_item is not None:
                    if (
                        ref.rendered_char != expected_item.char
                        or ref.region_char_index != expected_item.region_char_index
                        or ref.source != expected_item.source
                        or ref.provenance != expected_item.provenance
                    ):
                        raise V2ProjectionError(
                            "coverage page raw span ref disagrees with source"
                        )
                    observed.append(key)
        previous_next = page.next_cursor
    observed_set = set(observed)
    return V2ReadCycleCoverage(
        expected_characters=len(expected),
        observed_characters=len(observed_set),
        duplicate_characters=len(observed) - len(observed_set),
        missing_refs=tuple(sorted(set(expected) - observed_set)),
    )


__all__ = [
    "V2_PROJECTION_PROTOCOL",
    "V2_PROJECTION_PLAN_HASH",
    "N_SLOTS_V2",
    "CONTEXT_START_V2",
    "CONTEXT_END_V2",
    "USER_START_V2",
    "USER_END_V2",
    "PROPOSAL_START_V2",
    "PROPOSAL_END_V2",
    "V2ProjectionError",
    "V2SlotKind",
    "V2ReadCursor",
    "V2SlotRef",
    "V2ViewOmission",
    "IdentityAwareFieldViewV2",
    "PagedIdentityFieldViewV2",
    "V2ReadCycleCoverage",
    "compile_identity_aware_v2_read_page",
    "collect_identity_aware_v2_read_cycle",
    "audit_v2_read_cycle_coverage",
]
