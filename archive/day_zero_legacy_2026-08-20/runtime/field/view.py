"""Compile an unbounded shared-field snapshot into the legacy 384-slot view."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np

from substrate import ALPHABET_SET, SLOT_DIM, char_to_slot, get_letter_bank

from .schema import (
    CORE_WRITABLE_REGIONS,
    LOGICAL_REGION_IDS,
    LogicalRegion,
    PhysicalRole,
    RegionState,
    RegionVisibility,
    SharedFieldSnapshot,
    WritePolicy,
    canonical_json_bytes,
)


N_SLOTS = 384
CONTEXT_START = 0
CONTEXT_END = 256
USER_START = 256
USER_END = 320
PROPOSAL_START = 320
PROPOSAL_END = 384

CONTEXT_REGIONS: tuple[LogicalRegion, ...] = (
    LogicalRegion.CONVERSATION_HISTORY,
    LogicalRegion.STRUCTURED_KNOWLEDGE,
    LogicalRegion.SITUATION_AWARENESS,
    LogicalRegion.TOOL_RESULTS,
    LogicalRegion.ADVISOR_INPUT,
    LogicalRegion.TASK_STATE,
    LogicalRegion.SCRATCH,
    LogicalRegion.DIARY,
)

# The active proposal target is removed at compile time.  With the default
# response target this reduces exactly to CONTEXT_REGIONS, preserving the
# inherited view byte-for-byte; a scratch-target tick instead reads the current
# response draft without duplicating scratch in two physical roles.
CONTEXT_CANDIDATE_REGIONS: tuple[LogicalRegion, ...] = (
    LogicalRegion.CONVERSATION_HISTORY,
    LogicalRegion.STRUCTURED_KNOWLEDGE,
    LogicalRegion.SITUATION_AWARENESS,
    LogicalRegion.TOOL_RESULTS,
    LogicalRegion.ADVISOR_INPUT,
    LogicalRegion.TASK_STATE,
    LogicalRegion.SCRATCH,
    LogicalRegion.RESPONSE_DRAFT,
    LogicalRegion.DIARY,
)


def proposal_payload_capacity(
    proposal_region: LogicalRegion | str,
) -> int:
    """Return proposal text slots after the runtime-rendered region tag."""

    try:
        region = (
            proposal_region
            if isinstance(proposal_region, LogicalRegion)
            else LogicalRegion(proposal_region)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown proposal_region {proposal_region!r}") from exc
    if region not in CORE_WRITABLE_REGIONS:
        raise ValueError(f"proposal region {region.value!r} is sealed")
    capacity = (PROPOSAL_END - PROPOSAL_START) - len(
        _tag_chars(region, trailing_newline=False)
    )
    if capacity <= 0:
        raise RuntimeError("proposal-region tag consumes the proposal window")
    return capacity


def proposal_tail_offset(
    proposal_region: LogicalRegion | str,
    current_text: str,
) -> int:
    """Return the cursor that keeps the newest canonical character visible."""

    if not isinstance(current_text, str):
        raise TypeError("current_text must be a string")
    return max(
        0,
        len(current_text) - proposal_payload_capacity(proposal_region),
    )


class SlotKind(str, Enum):
    TAG = "tag"
    SPAN = "span"
    BLANK = "blank"
    PADDING = "padding"


@dataclass(frozen=True, slots=True)
class SlotRef:
    """Exact trace from one physical slot to its rendered character."""

    slot_index: int
    role: PhysicalRole | int
    kind: SlotKind | str
    logical_region: LogicalRegion | str | None
    rendered_char: str | None = None
    span_id: str | None = None
    span_char_index: int | None = None
    region_char_index: int | None = None
    tag_char_index: int | None = None
    source: str = ""
    provenance: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.slot_index < N_SLOTS:
            raise ValueError("SlotRef.slot_index is outside the 384-slot view")
        role = self.role if isinstance(self.role, PhysicalRole) else PhysicalRole(self.role)
        kind = self.kind if isinstance(self.kind, SlotKind) else SlotKind(self.kind)
        region = self.logical_region
        if region is not None and not isinstance(region, LogicalRegion):
            region = LogicalRegion(region)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "logical_region", region)
        if self.rendered_char is not None and len(self.rendered_char) != 1:
            raise ValueError("SlotRef.rendered_char must be one character or None")
        if kind in (SlotKind.TAG, SlotKind.SPAN) and self.rendered_char is None:
            raise ValueError(f"{kind.value} slots require rendered_char")
        if kind is SlotKind.SPAN:
            if self.span_id is None:
                raise ValueError("span slot requires span_id")
            if self.span_char_index is None or self.region_char_index is None:
                raise ValueError("span slot requires exact character offsets")

    def to_canonical_dict(self) -> dict[str, Any]:
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
            "source": self.source,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class ViewOmission:
    """An exact canonical span range left out of a bounded physical view."""

    logical_region: LogicalRegion | str
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
        region = (
            self.logical_region
            if isinstance(self.logical_region, LogicalRegion)
            else LogicalRegion(self.logical_region)
        )
        object.__setattr__(self, "logical_region", region)
        if not (
            0 <= self.span_char_start < self.span_char_end
            and 0 <= self.region_char_start < self.region_char_end
        ):
            raise ValueError("ViewOmission must describe a non-empty valid range")
        if not self.reason:
            raise ValueError("ViewOmission.reason must be non-empty")

    def to_canonical_dict(self) -> dict[str, Any]:
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


def _readonly_array(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    # A bytes-backed ndarray cannot be made writeable again by a caller.
    return np.frombuffer(contiguous.tobytes(), dtype=dtype).reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class FieldView:
    """Immutable checkpoint-compatible character view of a field snapshot."""

    source_field_id: str
    proposal_region: LogicalRegion | str
    field16: np.ndarray
    role_ids: np.ndarray
    logical_region_ids: np.ndarray
    attention_mask: np.ndarray
    write_mask: np.ndarray
    slot_refs: tuple[SlotRef, ...]
    omissions: tuple[ViewOmission, ...] = ()
    view_hash: str = field(init=False)

    def __post_init__(self) -> None:
        proposal_region = (
            self.proposal_region
            if isinstance(self.proposal_region, LogicalRegion)
            else LogicalRegion(self.proposal_region)
        )
        if proposal_region not in CORE_WRITABLE_REGIONS:
            raise ValueError(
                f"proposal region {proposal_region.value!r} is sealed"
            )
        field16 = _readonly_array(self.field16, np.dtype("<f4"))
        role_ids = _readonly_array(self.role_ids, np.dtype("<i8"))
        region_ids = _readonly_array(self.logical_region_ids, np.dtype("<i8"))
        attention = _readonly_array(self.attention_mask, np.dtype("?"))
        write = _readonly_array(self.write_mask, np.dtype("?"))
        if field16.shape != (N_SLOTS, SLOT_DIM):
            raise ValueError(f"field16 must have shape ({N_SLOTS}, {SLOT_DIM})")
        for name, value in (
            ("role_ids", role_ids),
            ("logical_region_ids", region_ids),
            ("attention_mask", attention),
            ("write_mask", write),
        ):
            if value.shape != (N_SLOTS,):
                raise ValueError(f"{name} must have shape ({N_SLOTS},)")

        expected_roles = np.empty(N_SLOTS, dtype=np.int64)
        expected_roles[CONTEXT_START:CONTEXT_END] = int(PhysicalRole.CONTEXT)
        expected_roles[USER_START:USER_END] = int(PhysicalRole.USER)
        expected_roles[PROPOSAL_START:PROPOSAL_END] = int(PhysicalRole.PROPOSAL)
        if not np.array_equal(role_ids, expected_roles):
            raise ValueError("role_ids do not preserve the legacy 0/1/2 slot geometry")
        if np.any(write[:PROPOSAL_START]) or not np.all(write[PROPOSAL_START:]):
            raise ValueError("write_mask must authorize exactly the proposal window")
        if not np.all(attention[PROPOSAL_START:]):
            raise ValueError("every proposal slot, including blanks, must stay active")
        proposal_region_id = LOGICAL_REGION_IDS[proposal_region]
        if not np.all(region_ids[PROPOSAL_START:] == proposal_region_id):
            raise ValueError(
                "proposal logical_region_ids must follow proposal_region"
            )
        if np.any((region_ids < -1) | (region_ids >= len(LOGICAL_REGION_IDS))):
            raise ValueError("logical_region_ids contains an unknown region id")

        refs = tuple(self.slot_refs)
        if len(refs) != N_SLOTS:
            raise ValueError(f"slot_refs must contain exactly {N_SLOTS} entries")
        if any(ref.slot_index != index for index, ref in enumerate(refs)):
            raise ValueError("slot_refs must be in exact physical slot order")
        if any(
            ref.logical_region is not proposal_region
            for ref in refs[PROPOSAL_START:]
        ):
            raise ValueError("proposal slot_refs must follow proposal_region")
        omissions = tuple(self.omissions)

        object.__setattr__(self, "proposal_region", proposal_region)
        object.__setattr__(self, "field16", field16)
        object.__setattr__(self, "role_ids", role_ids)
        object.__setattr__(self, "logical_region_ids", region_ids)
        object.__setattr__(self, "attention_mask", attention)
        object.__setattr__(self, "write_mask", write)
        object.__setattr__(self, "slot_refs", refs)
        object.__setattr__(self, "omissions", omissions)

        hasher = hashlib.sha256()
        hasher.update(self.source_field_id.encode("ascii"))
        hasher.update(proposal_region.value.encode("ascii"))
        for array in (field16, role_ids, region_ids, attention, write):
            hasher.update(array.tobytes(order="C"))
        hasher.update(
            canonical_json_bytes(
                {
                    "slot_refs": [ref.to_canonical_dict() for ref in refs],
                    "omissions": [
                        omission.to_canonical_dict() for omission in omissions
                    ],
                }
            )
        )
        object.__setattr__(self, "view_hash", hasher.hexdigest())

    @property
    def slot_role_ids(self) -> np.ndarray:
        """Alias matching the model-interface terminology."""

        return self.role_ids

    @property
    def region_ids(self) -> np.ndarray:
        return self.logical_region_ids

    @property
    def provenance(self) -> tuple[str, ...]:
        return tuple(ref.provenance for ref in self.slot_refs)

    def decode(self, start: int = 0, end: int = N_SLOTS, *, blanks_as: str = "") -> str:
        if not 0 <= start <= end <= N_SLOTS:
            raise ValueError("decode bounds are outside the 384-slot view")
        return get_letter_bank().decode_sequence(
            self.field16[start:end],
            blanks_as=blanks_as,
        )


@dataclass(frozen=True, slots=True)
class _RenderedChar:
    char: str
    region: LogicalRegion
    kind: SlotKind
    span_id: str | None
    span_char_index: int | None
    region_char_index: int | None
    tag_char_index: int | None
    source: str
    provenance: str


def _tag_chars(region: LogicalRegion, *, trailing_newline: bool) -> list[_RenderedChar]:
    tag = f"[{region.value}]\n"
    if trailing_newline:
        tag += "\n"
    return [
        _RenderedChar(
            char=char,
            region=region,
            kind=SlotKind.TAG,
            span_id=None,
            span_char_index=None,
            region_char_index=None,
            tag_char_index=index,
            source="shared_field_renderer",
            provenance="logical_region_tag",
        )
        for index, char in enumerate(tag)
    ]


def _renderable_span_chars(region: RegionState) -> list[_RenderedChar]:
    """Return only characters representable by the frozen 95-char substrate."""

    rendered: list[_RenderedChar] = []
    region_offset = 0
    for span in region.spans:
        for span_offset, char in enumerate(span.text):
            if char not in ALPHABET_SET:
                continue
            rendered.append(
                _RenderedChar(
                    char=char,
                    region=region.name,
                    kind=SlotKind.SPAN,
                    span_id=span.span_id,
                    span_char_index=span_offset,
                    region_char_index=region_offset + span_offset,
                    tag_char_index=None,
                    source=span.source,
                    provenance=span.provenance,
                )
            )
        region_offset += len(span.text)
    return rendered


def _omissions_from_chars(
    chars: list[_RenderedChar],
    reason: str,
) -> list[ViewOmission]:
    """Group omitted characters into exact contiguous source-span ranges."""

    omissions: list[ViewOmission] = []
    if not chars:
        return omissions

    run: list[_RenderedChar] = []

    def flush() -> None:
        if not run:
            return
        first = run[0]
        last = run[-1]
        text = "".join(item.char for item in run)
        omissions.append(
            ViewOmission(
                logical_region=first.region,
                span_id=first.span_id or "",
                span_char_start=first.span_char_index or 0,
                span_char_end=(last.span_char_index or 0) + 1,
                region_char_start=first.region_char_index or 0,
                region_char_end=(last.region_char_index or 0) + 1,
                reason=reason,
                omitted_text_sha256=hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                source=first.source,
                provenance=first.provenance,
            )
        )
        run.clear()

    for item in chars:
        if not run:
            run.append(item)
            continue
        previous = run[-1]
        contiguous = (
            item.region is previous.region
            and item.span_id == previous.span_id
            and item.span_char_index == (previous.span_char_index or 0) + 1
            and item.region_char_index == (previous.region_char_index or 0) + 1
            and item.source == previous.source
            and item.provenance == previous.provenance
        )
        if not contiguous:
            flush()
        run.append(item)
    flush()
    return omissions


def _unsupported_chars(region: RegionState) -> list[_RenderedChar]:
    unsupported: list[_RenderedChar] = []
    region_offset = 0
    for span in region.spans:
        for span_offset, char in enumerate(span.text):
            if char in ALPHABET_SET:
                continue
            unsupported.append(
                _RenderedChar(
                    char=char,
                    region=region.name,
                    kind=SlotKind.SPAN,
                    span_id=span.span_id,
                    span_char_index=span_offset,
                    region_char_index=region_offset + span_offset,
                    tag_char_index=None,
                    source=span.source,
                    provenance=span.provenance,
                )
            )
        region_offset += len(span.text)
    return unsupported


def _fair_prefix_allocations(
    regions: tuple[RegionState, ...],
    renderable: tuple[list[_RenderedChar], ...],
    budget: int,
) -> list[int]:
    allocations = [0] * len(regions)
    active = [
        index
        for index, region in enumerate(regions)
        if region.visibility is RegionVisibility.ATTENDED and renderable[index]
    ]
    while budget > 0 and active:
        next_active: list[int] = []
        for index in active:
            if budget == 0:
                next_active.append(index)
                continue
            if allocations[index] < len(renderable[index]):
                allocations[index] += 1
                budget -= 1
            if allocations[index] < len(renderable[index]):
                next_active.append(index)
        active = next_active
    return allocations


def _compile_context(
    snapshot: SharedFieldSnapshot,
    proposal_region: LogicalRegion,
    offsets: Mapping[LogicalRegion, int],
) -> tuple[list[_RenderedChar], list[ViewOmission]]:
    region_names = tuple(
        name
        for name in CONTEXT_CANDIDATE_REGIONS
        if name is not proposal_region
    )
    regions = tuple(snapshot.region(name) for name in region_names)
    frames = [_tag_chars(region.name, trailing_newline=True) for region in regions]
    framing_chars = sum(len(frame) for frame in frames)
    capacity = CONTEXT_END - CONTEXT_START
    if framing_chars > capacity:
        raise AssertionError("canonical logical-region tags exceed context capacity")
    renderable = tuple(_renderable_span_chars(region) for region in regions)
    remaining: list[list[_RenderedChar]] = []
    for region, available in zip(regions, renderable, strict=True):
        offset = offsets.get(region.name, 0)
        if offset < 0 or offset > len(available):
            raise ValueError(
                f"context offset for {region.name.value!r} must be in "
                f"[0, {len(available)}]"
            )
        remaining.append(available[offset:])
    allocations = _fair_prefix_allocations(
        regions,
        tuple(remaining),
        capacity - framing_chars,
    )

    rendered: list[_RenderedChar] = []
    omissions: list[ViewOmission] = []
    for region, frame, available, selected in zip(
        regions,
        frames,
        renderable,
        allocations,
        strict=True,
    ):
        offset = offsets.get(region.name, 0)
        # The final newline belongs after the selected content, not before it.
        rendered.extend(frame[:-1])
        rendered.extend(available[offset : offset + selected])
        rendered.extend(frame[-1:])
        omissions.extend(
            _omissions_from_chars(
                _unsupported_chars(region),
                "unsupported_substrate",
            )
        )
        if offset:
            omissions.extend(
                _omissions_from_chars(
                    available[:offset],
                    "context_cursor_before",
                )
            )
        if offset + selected < len(available):
            reason = (
                "region_masked"
                if region.visibility is RegionVisibility.MASKED
                else "context_capacity"
            )
            omissions.extend(
                _omissions_from_chars(
                    available[offset + selected :],
                    reason,
                )
            )
    return rendered, omissions


def _compile_dedicated(
    region: RegionState,
    capacity: int,
    omission_reason: str,
    *,
    offset: int,
    cursor_reason: str,
) -> tuple[list[_RenderedChar], list[ViewOmission]]:
    tag = _tag_chars(region.name, trailing_newline=False)
    if len(tag) > capacity:
        raise AssertionError(f"{region.name.value} tag exceeds its physical window")
    available = _renderable_span_chars(region)
    if offset < 0 or offset > len(available):
        raise ValueError(
            f"offset for {region.name.value!r} must be in [0, {len(available)}]"
        )
    selected = (
        min(len(available) - offset, capacity - len(tag))
        if region.visibility is RegionVisibility.ATTENDED
        else 0
    )
    rendered = tag + available[offset : offset + selected]
    omissions = _omissions_from_chars(
        _unsupported_chars(region),
        "unsupported_substrate",
    )
    if offset:
        omissions.extend(
            _omissions_from_chars(
                available[:offset],
                cursor_reason,
            )
        )
    if offset + selected < len(available):
        reason = (
            "region_masked"
            if region.visibility is RegionVisibility.MASKED
            else omission_reason
        )
        omissions.extend(
            _omissions_from_chars(
                available[offset + selected :],
                reason,
            )
        )
    return rendered, omissions


def _normalize_context_offsets(
    value: Mapping[LogicalRegion | str, int] | None,
    *,
    proposal_region: LogicalRegion,
) -> dict[LogicalRegion, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("context_offsets must be a mapping")
    allowed = {
        region
        for region in CONTEXT_CANDIDATE_REGIONS
        if region is not proposal_region
    }
    normalized: dict[LogicalRegion, int] = {}
    for raw_region, raw_offset in value.items():
        try:
            region = (
                raw_region
                if isinstance(raw_region, LogicalRegion)
                else LogicalRegion(raw_region)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unknown context offset region {raw_region!r}"
            ) from exc
        if region not in allowed:
            raise ValueError(
                f"{region.value!r} is not a context region for proposal "
                f"{proposal_region.value!r}"
            )
        if isinstance(raw_offset, bool) or not isinstance(raw_offset, int):
            raise TypeError(
                f"context offset for {region.value!r} must be an integer"
            )
        if raw_offset < 0:
            raise ValueError(
                f"context offset for {region.value!r} must be nonnegative"
            )
        normalized[region] = raw_offset
    return normalized


def _nonnegative_offset(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _padding_ref(
    index: int,
    role: PhysicalRole,
    proposal_region: LogicalRegion,
) -> SlotRef:
    if role is PhysicalRole.PROPOSAL:
        return SlotRef(
            slot_index=index,
            role=role,
            kind=SlotKind.BLANK,
            logical_region=proposal_region,
            source="shared_field_renderer",
            provenance="active_blank_proposal",
        )
    return SlotRef(
        slot_index=index,
        role=role,
        kind=SlotKind.PADDING,
        logical_region=None,
        source="shared_field_renderer",
        provenance="masked_padding",
    )


def _write_rendered(
    *,
    start: int,
    rendered: Iterable[_RenderedChar],
    role: PhysicalRole,
    field16: np.ndarray,
    region_ids: np.ndarray,
    attention: np.ndarray,
    refs: list[SlotRef],
) -> None:
    for relative_index, item in enumerate(rendered):
        index = start + relative_index
        field16[index] = char_to_slot(item.char)
        region_ids[index] = LOGICAL_REGION_IDS[item.region]
        attention[index] = True
        refs[index] = SlotRef(
            slot_index=index,
            role=role,
            kind=item.kind,
            logical_region=item.region,
            rendered_char=item.char,
            span_id=item.span_id,
            span_char_index=item.span_char_index,
            region_char_index=item.region_char_index,
            tag_char_index=item.tag_char_index,
            source=item.source,
            provenance=item.provenance,
        )


def compile_field_view(
    snapshot: SharedFieldSnapshot,
    *,
    proposal_region: LogicalRegion | str = LogicalRegion.RESPONSE_DRAFT,
    context_offsets: Mapping[LogicalRegion | str, int] | None = None,
    user_offset: int = 0,
    proposal_offset: int = 0,
) -> FieldView:
    """Compile ``snapshot`` without mutating or truncating its canonical text.

    Physical slot positions and learned role IDs exactly match the existing
    checkpoint contract:

    * ``0:256``   -> role 0 (CONTEXT)
    * ``256:320`` -> role 1 (USER)
    * ``320:384`` -> role 2 (PROPOSAL)
    """

    if not isinstance(snapshot, SharedFieldSnapshot):
        raise TypeError("compile_field_view requires SharedFieldSnapshot")
    try:
        proposal_region = (
            proposal_region
            if isinstance(proposal_region, LogicalRegion)
            else LogicalRegion(proposal_region)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown proposal_region {proposal_region!r}") from exc
    proposal_state = snapshot.region(proposal_region)
    if (
        proposal_region not in CORE_WRITABLE_REGIONS
        or proposal_state.write_policy is not WritePolicy.CORE_WRITABLE
    ):
        raise ValueError(f"proposal region {proposal_region.value!r} is sealed")
    normalized_context_offsets = _normalize_context_offsets(
        context_offsets,
        proposal_region=proposal_region,
    )
    user_offset = _nonnegative_offset(user_offset, label="user_offset")
    proposal_offset = _nonnegative_offset(
        proposal_offset,
        label="proposal_offset",
    )

    field16 = np.zeros((N_SLOTS, SLOT_DIM), dtype=np.float32)
    role_ids = np.empty(N_SLOTS, dtype=np.int64)
    role_ids[CONTEXT_START:CONTEXT_END] = int(PhysicalRole.CONTEXT)
    role_ids[USER_START:USER_END] = int(PhysicalRole.USER)
    role_ids[PROPOSAL_START:PROPOSAL_END] = int(PhysicalRole.PROPOSAL)
    region_ids = np.full(N_SLOTS, -1, dtype=np.int64)
    attention = np.zeros(N_SLOTS, dtype=np.bool_)
    write = np.zeros(N_SLOTS, dtype=np.bool_)
    write[PROPOSAL_START:PROPOSAL_END] = True
    attention[PROPOSAL_START:PROPOSAL_END] = True
    region_ids[PROPOSAL_START:PROPOSAL_END] = LOGICAL_REGION_IDS[proposal_region]

    refs = [
        _padding_ref(
            index,
            (
                PhysicalRole.CONTEXT
                if index < CONTEXT_END
                else PhysicalRole.USER
                if index < USER_END
                else PhysicalRole.PROPOSAL
            ),
            proposal_region,
        )
        for index in range(N_SLOTS)
    ]

    context, context_omissions = _compile_context(
        snapshot,
        proposal_region,
        normalized_context_offsets,
    )
    user, user_omissions = _compile_dedicated(
        snapshot.region(LogicalRegion.USER_INPUT),
        USER_END - USER_START,
        "user_capacity",
        offset=user_offset,
        cursor_reason="user_cursor_before",
    )
    proposal, proposal_omissions = _compile_dedicated(
        proposal_state,
        PROPOSAL_END - PROPOSAL_START,
        "proposal_capacity",
        offset=proposal_offset,
        cursor_reason="proposal_cursor_before",
    )

    _write_rendered(
        start=CONTEXT_START,
        rendered=context,
        role=PhysicalRole.CONTEXT,
        field16=field16,
        region_ids=region_ids,
        attention=attention,
        refs=refs,
    )
    _write_rendered(
        start=USER_START,
        rendered=user,
        role=PhysicalRole.USER,
        field16=field16,
        region_ids=region_ids,
        attention=attention,
        refs=refs,
    )
    _write_rendered(
        start=PROPOSAL_START,
        rendered=proposal,
        role=PhysicalRole.PROPOSAL,
        field16=field16,
        region_ids=region_ids,
        attention=attention,
        refs=refs,
    )

    return FieldView(
        source_field_id=snapshot.field_id,
        proposal_region=proposal_region,
        field16=field16,
        role_ids=role_ids,
        logical_region_ids=region_ids,
        attention_mask=attention,
        write_mask=write,
        slot_refs=tuple(refs),
        omissions=tuple(
            context_omissions + user_omissions + proposal_omissions
        ),
    )


__all__ = [
    "N_SLOTS",
    "CONTEXT_START",
    "CONTEXT_END",
    "USER_START",
    "USER_END",
    "PROPOSAL_START",
    "PROPOSAL_END",
    "proposal_payload_capacity",
    "proposal_tail_offset",
    "CONTEXT_REGIONS",
    "CONTEXT_CANDIDATE_REGIONS",
    "SlotKind",
    "SlotRef",
    "ViewOmission",
    "FieldView",
    "compile_field_view",
]
