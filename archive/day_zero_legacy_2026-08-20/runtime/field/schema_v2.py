"""Isolated shared-field-v2 schema with a sealed identity region.

This module is deliberately independent of the v1 ``shared-field`` types.
It reuses only pure primitives: ``canonical_json_bytes``, ``canonical_sha256``,
and the frozen 95-character ``ALPHABET_SET``.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, Mapping

from substrate import ALPHABET_SET

FIELD_SCHEMA_V2 = "shared-field-v2"
CHARTER_SCHEMA_V2 = "axon-identity-charter-v1"
CHARTER_VERSION_V2 = 1
CHARTER_MAX_CHARS_V2 = 512
ENVELOPE_CHAR_BUDGET_MAX_V2 = 192
VIEW_SCHEMA_V2 = "axon-core-identity-view-v1"

_CHARTER_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_PARENT_HASH_RE = _CHARTER_HASH_RE


def _assert_exact_base_type(value: object, base: type) -> None:
    """Reject subclass instances before overrides can influence canonical behavior."""

    if type(value) is not base:
        raise TypeError(
            f"{base.__name__} requires the exact base type; "
            f"got {type(value).__name__}"
        )


class LogicalRegionV2(str, Enum):
    """Canonical v2 logical regions.  v1 names/IDs 0..9 are preserved;
    ``identity`` is appended at ID 10.
    """

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
    IDENTITY = "identity"


CANONICAL_REGION_ORDER_V2: tuple[LogicalRegionV2, ...] = (
    LogicalRegionV2.CONVERSATION_HISTORY,
    LogicalRegionV2.USER_INPUT,
    LogicalRegionV2.STRUCTURED_KNOWLEDGE,
    LogicalRegionV2.SITUATION_AWARENESS,
    LogicalRegionV2.TOOL_RESULTS,
    LogicalRegionV2.ADVISOR_INPUT,
    LogicalRegionV2.TASK_STATE,
    LogicalRegionV2.SCRATCH,
    LogicalRegionV2.RESPONSE_DRAFT,
    LogicalRegionV2.DIARY,
    LogicalRegionV2.IDENTITY,
)

LOGICAL_REGION_IDS_V2: Mapping[LogicalRegionV2, int] = MappingProxyType(
    {region: index for index, region in enumerate(CANONICAL_REGION_ORDER_V2)}
)

CORE_WRITABLE_REGIONS_V2: frozenset[LogicalRegionV2] = frozenset(
    {
        LogicalRegionV2.SCRATCH,
        LogicalRegionV2.RESPONSE_DRAFT,
    }
)

SEALED_REGIONS_V2: frozenset[LogicalRegionV2] = frozenset(
    set(CANONICAL_REGION_ORDER_V2) - CORE_WRITABLE_REGIONS_V2
)


class RegionVisibilityV2(str, Enum):
    ATTENDED = "attended"
    MASKED = "masked"


class WritePolicyV2(str, Enum):
    SEALED = "sealed"
    CORE_WRITABLE = "core_writable"


class PhysicalRoleV2(IntEnum):
    """Checkpoint-compatible learned ``char_type_emb`` row numbers."""

    CONTEXT = 0
    USER = 1
    PROPOSAL = 2


def _as_logical_region_v2(value: LogicalRegionV2 | str) -> LogicalRegionV2:
    if type(value) is LogicalRegionV2:
        return value
    if type(value) is not str:
        raise TypeError(
            f"logical region must be a LogicalRegionV2 or exact built-in str; "
            f"got {type(value).__name__}"
        )
    try:
        return LogicalRegionV2(value)
    except ValueError as exc:
        raise ValueError(
            f"unknown logical region {value!r}; v2 field expects one of "
            f"{[r.value for r in CANONICAL_REGION_ORDER_V2]}"
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


def _assert_supported_text(text: str, label: str) -> None:
    """Fail closed if ``text`` contains characters outside the frozen substrate."""

    if type(text) is not str:
        raise TypeError(f"{label} must be an exact built-in str")
    unsupported = [ch for ch in text if ch not in ALPHABET_SET]
    if unsupported:
        unique = "".join(sorted(set(unsupported)))
        raise ValueError(
            f"{label} contains unsupported characters outside the frozen "
            f"95-character substrate alphabet: {unique!r}"
        )


def _strict_string_sequence(value: Any, label: str) -> tuple[str, ...]:
    """Reject non-sequence or non-string refs without coercing or reordering."""

    if type(value) not in (list, tuple):
        raise TypeError(f"{label} must be a list or tuple of strings")
    for item in value:
        if type(item) is not str or not item:
            raise ValueError(f"{label} must contain non-empty strings")
    return tuple(value)


def _reconstruct_charter(charter: IdentityCharterV2) -> IdentityCharterV2:
    """Return a fresh exact-base charter from primitive fields.

    Converts forged or mutated state into a fail-closed construction error.
    """

    if type(charter) is not IdentityCharterV2:
        raise TypeError(
            "charter must be the exact base IdentityCharterV2 type; "
            "subclasses are not permitted"
        )
    try:
        text = charter.text
        version = charter.charter_version
    except AttributeError as exc:
        raise TypeError(
            "charter is missing required primitive fields"
        ) from exc
    return IdentityCharterV2(text=text, charter_version=version)


def _validate_identity_snapshot(snapshot: "SharedFieldSnapshotV2") -> None:
    """Strict schema-level invariant: one sealed, attended, canonical charter span."""

    identity = snapshot.region(LogicalRegionV2.IDENTITY)

    if identity.visibility is not RegionVisibilityV2.ATTENDED:
        raise ValueError("identity region must be attended")
    if identity.write_policy is not WritePolicyV2.SEALED:
        raise ValueError("identity region must be sealed")
    if len(identity.spans) != 1:
        raise ValueError(
            f"identity region must contain exactly one charter span, got {len(identity.spans)}"
        )

    span = identity.spans[0]
    if span.kind != "identity_charter":
        raise ValueError("identity span kind must be 'identity_charter'")
    if type(span.text) is not str or not span.text:
        raise ValueError("identity span text must be a non-empty string")

    try:
        charter = IdentityCharterV2(text=span.text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"identity span text is not a valid charter: {exc}") from exc

    expected_region = IdentityCharterV2.build_identity_region(charter)
    expected = expected_region.spans[0]
    if span.span_id != expected.span_id:
        raise ValueError("identity span_id does not match canonical charter span_id")
    if span.text != expected.text:
        raise ValueError("identity span text does not match canonical charter text")
    if span.source != expected.source:
        raise ValueError(
            "identity span source does not match canonical charter source manifest"
        )
    if span.provenance != expected.provenance:
        raise ValueError(
            "identity span provenance does not match genesis provenance"
        )
    if canonical_sha256(
        FieldSpanV2.to_canonical_dict(span)
    ) != canonical_sha256(FieldSpanV2.to_canonical_dict(expected)):
        raise ValueError(
            "identity span does not match the canonical charter span byte-for-byte"
        )
    if span.container_refs or span.edge_refs:
        raise ValueError("identity span refs must be empty")

    if charter.source_manifest_id not in snapshot.source_manifest_ids:
        raise ValueError(
            "source_manifest_ids is missing the charter source manifest"
        )


@dataclass(frozen=True, slots=True)
class FieldSpanV2:
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
        if type(self) is not FieldSpanV2:
            raise TypeError(
                "FieldSpanV2 does not permit subclass instances; "
                "use the exact base FieldSpanV2 type"
            )

        if type(self.span_id) is not str or not self.span_id:
            raise ValueError("FieldSpanV2.span_id must be a non-empty string")
        if type(self.text) is not str or not self.text:
            raise ValueError("FieldSpanV2.text must be a non-empty string")
        _assert_supported_text(self.text, "FieldSpanV2.text")
        if type(self.kind) is not str or not self.kind:
            raise ValueError("FieldSpanV2.kind must be a non-empty string")
        if type(self.source) is not str:
            raise TypeError("FieldSpanV2.source must be a string")
        if type(self.provenance) is not str:
            raise TypeError("FieldSpanV2.provenance must be a string")

        if type(self.confidence) is not float:
            raise TypeError("FieldSpanV2.confidence must be a finite float")
        if not math.isfinite(self.confidence) or not (
            0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(
                "FieldSpanV2.confidence must be finite and in the range [0, 1]"
            )
        if self.confidence == 0.0 and math.copysign(1.0, self.confidence) < 0:
            raise ValueError(
                "FieldSpanV2.confidence must use canonical +0.0; "
                "signed negative zero is rejected"
            )

        container_refs = _strict_string_sequence(
            self.container_refs, "FieldSpanV2.container_refs"
        )
        edge_refs = _strict_string_sequence(
            self.edge_refs, "FieldSpanV2.edge_refs"
        )
        object.__setattr__(self, "container_refs", container_refs)
        object.__setattr__(self, "edge_refs", edge_refs)

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
class RegionStateV2:
    """Immutable state of one canonical v2 logical region."""

    name: LogicalRegionV2 | str
    spans: tuple[FieldSpanV2, ...] = ()
    visibility: RegionVisibilityV2 | str = RegionVisibilityV2.ATTENDED
    write_policy: WritePolicyV2 | str | None = None

    def __post_init__(self) -> None:
        if type(self) is not RegionStateV2:
            raise TypeError(
                "RegionStateV2 does not permit subclass instances; "
                "use the exact base RegionStateV2 type"
            )

        name = _as_logical_region_v2(self.name)
        object.__setattr__(self, "name", name)

        if type(self.spans) not in (list, tuple):
            raise TypeError("RegionStateV2.spans must be a list or tuple")
        supplied_spans = tuple(self.spans)
        reconstructed_spans: list[FieldSpanV2] = []
        for span in supplied_spans:
            if type(span) is not FieldSpanV2:
                raise TypeError(
                    "RegionStateV2.spans must contain only exact FieldSpanV2 values"
                )
            try:
                reconstructed_spans.append(
                    FieldSpanV2(
                        span_id=span.span_id,
                        text=span.text,
                        kind=span.kind,
                        source=span.source,
                        provenance=span.provenance,
                        confidence=span.confidence,
                        container_refs=span.container_refs,
                        edge_refs=span.edge_refs,
                    )
                )
            except AttributeError as exc:
                raise TypeError(
                    "FieldSpanV2 has missing or invalid attributes"
                ) from exc
        spans = tuple(reconstructed_spans)
        span_ids = [s.span_id for s in spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError(f"duplicate span_id in region {name.value!r}")
        object.__setattr__(self, "spans", spans)

        if type(self.visibility) is RegionVisibilityV2:
            visibility = self.visibility
        elif type(self.visibility) is str:
            visibility = RegionVisibilityV2(self.visibility)
        else:
            raise TypeError(
                "RegionStateV2.visibility must be a RegionVisibilityV2 or exact built-in str"
            )
        object.__setattr__(self, "visibility", visibility)

        policy = self.write_policy
        if policy is None:
            policy = (
                WritePolicyV2.CORE_WRITABLE
                if name in CORE_WRITABLE_REGIONS_V2
                else WritePolicyV2.SEALED
            )
        elif type(policy) is WritePolicyV2:
            pass
        elif type(policy) is str:
            policy = WritePolicyV2(policy)
        else:
            raise TypeError(
                "RegionStateV2.write_policy must be a WritePolicyV2, exact built-in str, or None"
            )
        if policy is WritePolicyV2.CORE_WRITABLE and name not in CORE_WRITABLE_REGIONS_V2:
            raise ValueError(
                f"logical region {name.value!r} is always sealed in v2"
            )
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
            "spans": [FieldSpanV2.to_canonical_dict(span) for span in self.spans],
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class SharedFieldSnapshotV2:
    """A complete immutable v2 field state linked to its parent by hash."""

    tick_id: int
    regions: tuple[RegionStateV2, ...] = ()
    parent_field_id: str | None = None
    source_manifest_ids: tuple[str, ...] = ()
    field_id: str = field(init=False)
    canonical_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not SharedFieldSnapshotV2:
            raise TypeError(
                "SharedFieldSnapshotV2 does not permit subclass instances; "
                "use the exact base SharedFieldSnapshotV2 type"
            )

        if type(self.tick_id) is not int:
            raise TypeError("SharedFieldSnapshotV2.tick_id must be an integer")
        if self.tick_id < 0:
            raise ValueError("SharedFieldSnapshotV2.tick_id must be non-negative")

        if self.tick_id == 0:
            if self.parent_field_id is not None:
                raise ValueError(
                    "genesis snapshot (tick_id == 0) must have parent_field_id=None"
                )
        elif self.parent_field_id is None:
            raise ValueError(
                "non-genesis snapshot must declare a parent_field_id hash link"
            )
        elif (
            type(self.parent_field_id) is not str
            or not _PARENT_HASH_RE.fullmatch(self.parent_field_id)
        ):
            raise ValueError(
                "parent_field_id must be a 64-character lowercase hex SHA-256"
            )

        if type(self.regions) not in (list, tuple):
            raise TypeError(
                "SharedFieldSnapshotV2.regions must be a list or tuple of RegionStateV2 values"
            )
        supplied_regions = tuple(self.regions)
        if not all(type(r) is RegionStateV2 for r in supplied_regions):
            raise TypeError(
                "SharedFieldSnapshotV2.regions must contain exact RegionStateV2 values"
            )

        reconstructed_regions: list[RegionStateV2] = []
        for region in supplied_regions:
            try:
                reconstructed_regions.append(
                    RegionStateV2(
                        name=region.name,
                        spans=region.spans,
                        visibility=region.visibility,
                        write_policy=region.write_policy,
                    )
                )
            except AttributeError as exc:
                raise TypeError(
                    "RegionStateV2 has missing or invalid attributes"
                ) from exc
        supplied_regions = tuple(reconstructed_regions)

        if len(supplied_regions) != len(CANONICAL_REGION_ORDER_V2):
            raise ValueError(
                f"v2 snapshot requires exactly {len(CANONICAL_REGION_ORDER_V2)} "
                f"regions, got {len(supplied_regions)}"
            )

        for index, expected in enumerate(CANONICAL_REGION_ORDER_V2):
            actual = supplied_regions[index].name
            if actual != expected:
                raise ValueError(
                    f"region at index {index} must be {expected.value!r} in "
                    f"canonical v2 order, got {actual.value!r}"
                )

        object.__setattr__(self, "regions", supplied_regions)

        manifests = _strict_string_sequence(
            self.source_manifest_ids,
            "SharedFieldSnapshotV2.source_manifest_ids",
        )
        if len(manifests) != len(set(manifests)):
            raise ValueError(
                "SharedFieldSnapshotV2.source_manifest_ids must not contain duplicates"
            )
        object.__setattr__(self, "source_manifest_ids", manifests)

        _validate_identity_snapshot(self)

        digest = canonical_sha256(self.to_canonical_dict())
        object.__setattr__(self, "field_id", digest)
        object.__setattr__(self, "canonical_hash", digest)

    def region(self, name: LogicalRegionV2 | str) -> RegionStateV2:
        logical_name = _as_logical_region_v2(name)
        return self.regions[LOGICAL_REGION_IDS_V2[logical_name]]

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": FIELD_SCHEMA_V2,
            "tick_id": self.tick_id,
            "parent_field_id": self.parent_field_id,
            "source_manifest_ids": list(self.source_manifest_ids),
            "regions": [
                RegionStateV2.to_canonical_dict(region) for region in self.regions
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["field_id"] = self.field_id
        value["canonical_hash"] = self.canonical_hash
        return value


@dataclass(frozen=True, slots=True)
class IdentityCharterV2:
    """A hash-pinned, sealed canonical identity charter for v2 genesis.

    R2 supports exactly ``axon-identity-charter-v1`` with charter version ``1``.
    """

    text: str
    charter_version: int = CHARTER_VERSION_V2

    def __post_init__(self) -> None:
        if type(self) is not IdentityCharterV2:
            raise TypeError(
                "IdentityCharterV2 does not permit subclass instances; "
                "use the exact base IdentityCharterV2 type"
            )
        if type(self.charter_version) is not int:
            raise TypeError(
                "IdentityCharterV2.charter_version must be an integer"
            )
        if self.charter_version != CHARTER_VERSION_V2:
            raise ValueError(
                "IdentityCharterV2.charter_version must be "
                f"{CHARTER_VERSION_V2} for this wire contract"
            )
        if type(self.text) is not str or not self.text:
            raise ValueError("IdentityCharterV2.text must be a non-empty string")
        if len(self.text) > CHARTER_MAX_CHARS_V2:
            raise ValueError(
                f"IdentityCharterV2.text exceeds {CHARTER_MAX_CHARS_V2} characters"
            )
        _assert_supported_text(self.text, "IdentityCharterV2.text")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": CHARTER_SCHEMA_V2,
            "version": self.charter_version,
            "text": self.text,
        }

    @property
    def charter_hash(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    @property
    def span_id(self) -> str:
        return f"charter:{self.charter_hash}"

    @property
    def source_manifest_id(self) -> str:
        return f"{CHARTER_SCHEMA_V2}:{self.charter_hash}"

    def build_identity_region(self) -> RegionStateV2:
        return RegionStateV2(
            name=LogicalRegionV2.IDENTITY,
            spans=(
                FieldSpanV2(
                    span_id=self.span_id,
                    text=self.text,
                    kind="identity_charter",
                    source=self.source_manifest_id,
                    provenance="genesis",
                ),
            ),
            visibility=RegionVisibilityV2.ATTENDED,
            write_policy=WritePolicyV2.SEALED,
        )

    def build_genesis_snapshot(
        self,
        *,
        source_manifest_ids: tuple[str, ...] = (),
    ) -> SharedFieldSnapshotV2:
        extras = _strict_string_sequence(
            source_manifest_ids, "IdentityCharterV2.source_manifest_ids"
        )
        if len(extras) != len(set(extras)):
            raise ValueError(
                "IdentityCharterV2.source_manifest_ids must not contain duplicates"
            )

        identity_region = self.build_identity_region()
        other_regions = tuple(
            RegionStateV2(name=region)
            for region in CANONICAL_REGION_ORDER_V2
            if region is not LogicalRegionV2.IDENTITY
        )
        all_regions = other_regions + (identity_region,)
        all_manifests = (self.source_manifest_id,) + extras
        return SharedFieldSnapshotV2(
            tick_id=0,
            regions=all_regions,
            parent_field_id=None,
            source_manifest_ids=all_manifests,
        )


def _format_identity_envelope(
    *,
    charter_hash: str,
    core_id: str,
    display_name: str,
    model_label: str,
    role_capabilities: tuple[str, ...],
    current_role: str,
) -> str:
    return (
        f"c={core_id};n={display_name};m={model_label};"
        f"r={','.join(role_capabilities)};R={current_role};h={charter_hash}"
    )


_ENVELOPE_FORBIDDEN_CHARS = set(";=,\n\r")


def _assert_safe_descriptor_value(value: str, label: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty string")
    _assert_supported_text(value, label)
    forbidden = [ch for ch in value if ch in _ENVELOPE_FORBIDDEN_CHARS]
    if forbidden:
        raise ValueError(
            f"{label} contains envelope delimiter characters: {''.join(sorted(set(forbidden)))!r}"
        )


@dataclass(frozen=True, slots=True)
class CoreIdentityViewV2:
    """A deterministic, bounded per-core identity envelope derived from a
    sealed canonical charter.
    """

    charter_hash: str
    core_id: str
    display_name: str
    model_label: str
    role_capabilities: tuple[str, ...]
    current_role: str
    envelope: str
    view_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self) is not CoreIdentityViewV2:
            raise TypeError(
                "CoreIdentityViewV2 does not permit subclass instances; "
                "use the exact base CoreIdentityViewV2 type"
            )

        if type(self.charter_hash) is not str or not _CHARTER_HASH_RE.fullmatch(
            self.charter_hash
        ):
            raise ValueError(
                "CoreIdentityViewV2.charter_hash must be a 64-character lowercase hex SHA-256"
            )

        _assert_safe_descriptor_value(self.core_id, "CoreIdentityViewV2.core_id")
        _assert_safe_descriptor_value(
            self.display_name, "CoreIdentityViewV2.display_name"
        )
        _assert_safe_descriptor_value(
            self.model_label, "CoreIdentityViewV2.model_label"
        )
        _assert_safe_descriptor_value(
            self.current_role, "CoreIdentityViewV2.current_role"
        )

        if type(self.role_capabilities) not in (list, tuple):
            raise TypeError(
                "CoreIdentityViewV2.role_capabilities must be a list or tuple of strings"
            )
        capabilities = tuple(self.role_capabilities)
        if not capabilities:
            raise ValueError(
                "CoreIdentityViewV2.role_capabilities must not be empty"
            )
        if not all(type(capability) is str for capability in capabilities):
            raise TypeError(
                "CoreIdentityViewV2.role_capabilities must be a list or tuple of strings"
            )
        for capability in capabilities:
            _assert_safe_descriptor_value(
                capability, "CoreIdentityViewV2.role_capabilities item"
            )
        if capabilities != tuple(sorted(set(capabilities))):
            raise ValueError(
                "CoreIdentityViewV2.role_capabilities must be sorted and unique"
            )
        if self.current_role not in capabilities:
            raise ValueError(
                "CoreIdentityViewV2.current_role must be present in role_capabilities"
            )
        object.__setattr__(self, "role_capabilities", capabilities)

        if type(self.envelope) is not str or not self.envelope:
            raise ValueError("CoreIdentityViewV2.envelope must be a non-empty string")
        _assert_supported_text(self.envelope, "CoreIdentityViewV2.envelope")
        if len(self.envelope) > ENVELOPE_CHAR_BUDGET_MAX_V2:
            raise ValueError(
                f"CoreIdentityViewV2.envelope exceeds {ENVELOPE_CHAR_BUDGET_MAX_V2} characters"
            )

        expected = _format_identity_envelope(
            charter_hash=self.charter_hash,
            core_id=self.core_id,
            display_name=self.display_name,
            model_label=self.model_label,
            role_capabilities=capabilities,
            current_role=self.current_role,
        )
        if self.envelope != expected:
            raise ValueError(
                "CoreIdentityViewV2.envelope does not match the validated descriptor"
            )

        object.__setattr__(
            self,
            "view_hash",
            canonical_sha256(
                CoreIdentityViewV2.to_canonical_dict(self, include_envelope=True)
            ),
        )

    def to_canonical_dict(self, *, include_envelope: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": VIEW_SCHEMA_V2,
            "charter_hash": self.charter_hash,
            "core_id": self.core_id,
            "display_name": self.display_name,
            "model_label": self.model_label,
            "role_capabilities": list(self.role_capabilities),
            "current_role": self.current_role,
        }
        if include_envelope:
            value["envelope"] = self.envelope
        return value


def render_core_identity_view_v2(
    charter: IdentityCharterV2,
    *,
    core_id: str,
    display_name: str,
    model_label: str,
    role_capabilities: tuple[str, ...],
    current_role: str,
    envelope_char_budget: int,
) -> CoreIdentityViewV2:
    """Render a compact, bounded, deterministic per-core identity envelope.

    The envelope is never silently truncated or normalized.  It fails closed if
    the required canonical content cannot fit within ``envelope_char_budget``
    characters or if any input contains unsupported substrate characters.
    """

    safe_charter = _reconstruct_charter(charter)

    if (
        type(envelope_char_budget) is not int
        or envelope_char_budget <= 0
        or envelope_char_budget > ENVELOPE_CHAR_BUDGET_MAX_V2
    ):
        raise ValueError(
            f"envelope_char_budget must be an integer in "
            f"[1, {ENVELOPE_CHAR_BUDGET_MAX_V2}]"
        )

    _assert_safe_descriptor_value(core_id, "core_id")
    _assert_safe_descriptor_value(display_name, "display_name")
    _assert_safe_descriptor_value(model_label, "model_label")
    _assert_safe_descriptor_value(current_role, "current_role")

    if type(role_capabilities) not in (list, tuple):
        raise TypeError("role_capabilities must be a list or tuple of strings")
    capabilities = tuple(role_capabilities)
    if not capabilities:
        raise ValueError("role_capabilities must not be empty")
    if not all(type(capability) is str for capability in capabilities):
        raise TypeError("role_capabilities must be a list or tuple of strings")
    for capability in capabilities:
        _assert_safe_descriptor_value(capability, "role_capability")
    if capabilities != tuple(sorted(set(capabilities))):
        raise ValueError("role_capabilities must be sorted and unique")
    if current_role not in capabilities:
        raise ValueError("current_role must be present in role_capabilities")

    envelope = _format_identity_envelope(
        charter_hash=safe_charter.charter_hash,
        core_id=core_id,
        display_name=display_name,
        model_label=model_label,
        role_capabilities=capabilities,
        current_role=current_role,
    )

    if len(envelope) > envelope_char_budget:
        raise ValueError(
            f"rendered identity envelope ({len(envelope)} chars) exceeds "
            f"budget ({envelope_char_budget} chars)"
        )

    return CoreIdentityViewV2(
        charter_hash=safe_charter.charter_hash,
        core_id=core_id,
        display_name=display_name,
        model_label=model_label,
        role_capabilities=capabilities,
        current_role=current_role,
        envelope=envelope,
    )


__all__ = [
    "FIELD_SCHEMA_V2",
    "CHARTER_SCHEMA_V2",
    "CHARTER_VERSION_V2",
    "CHARTER_MAX_CHARS_V2",
    "ENVELOPE_CHAR_BUDGET_MAX_V2",
    "LogicalRegionV2",
    "CANONICAL_REGION_ORDER_V2",
    "LOGICAL_REGION_IDS_V2",
    "RegionVisibilityV2",
    "WritePolicyV2",
    "PhysicalRoleV2",
    "CORE_WRITABLE_REGIONS_V2",
    "SEALED_REGIONS_V2",
    "FieldSpanV2",
    "RegionStateV2",
    "SharedFieldSnapshotV2",
    "IdentityCharterV2",
    "CoreIdentityViewV2",
    "render_core_identity_view_v2",
    "canonical_json_bytes",
    "canonical_sha256",
]
