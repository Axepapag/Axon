"""Neural-free v3 protocol contracts for Axon's multi-core runtime.

This module defines the first isolated slice of the v3 journal contract.  Every
record is an immutable frozen dataclass, Torch-free, canonical-JSON safe, and
content-addressed.  Validation is fail-closed at construction and at
deserialization.

Design choices documented here (smallest representation that proves the
invariants):

* Protocol version is the exact integer ``3``; the canonical protocol name is
  ``axon-runtime-protocol-v3``.
* Populations whose order is semantic (the online ring) are stored as tuples in
  the order supplied; duplicates are rejected.  Content-addressed sets and
  mappings are stored as sorted tuples so serialization is deterministic.
* Passes and boards carry phase ``INITIAL`` or ``REFINE``; ``FINAL`` is only for
  the consolidation record.  A pass references the working field ``W`` by ID and
  a read-cycle ID.
* A soul transition encodes the phase/substep and an optional board parent; the
  pair is validated for mutual consistency.
* Only ``ConsolidationRecord`` and ``TickCommitRecordV3`` may carry invocation
  request IDs.  Passes and boards have no invocation field.
* A ``ReadCycleManifest`` may be constructed with ``coverage_complete=False`` so
  an incomplete read is representable, but it cannot be referenced by a valid
  ``CorePassRecord`` (validated by ``validate_pass_against_read_cycle``).
* Structurally invalid raw material becomes an ``ArtifactRejectionRecord``; it
  is never promoted to a pass, board, transition, or commit.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from runtime.field import canonical_json_bytes, canonical_sha256


PROTOCOL_VERSION = 3
PROTOCOL_NAME = "axon-runtime-protocol-v3"
MAX_PROTOCOL_INT = (1 << 63) - 1
MAX_JSON_DEPTH = 128
MAX_JSON_NODES = 100_000
MAX_COLLECTION_ITEMS = 4_096
MAX_CONTEXT_JSON_NODES = 50_000
MAX_ONLINE_CORES = (MAX_COLLECTION_ITEMS - 1) // 2


class V3ContractError(ValueError):
    """Base class for a rejected v3 runtime contract."""


class V3PopulationError(V3ContractError):
    """A core population or role assignment is invalid."""


class V3PhaseError(V3ContractError):
    """A phase/substep/board-parent combination is invalid."""


class V3SerializationError(V3ContractError):
    """A serialized v3 object is malformed."""


class V3BoardError(V3ContractError):
    """A proposal/refinement board is malformed."""


class V3SoulError(V3ContractError):
    """A soul transition or disposition is invalid."""


_PHASES = frozenset({"INITIAL", "REFINE", "FINAL"})
_PASS_PHASES = frozenset({"INITIAL", "REFINE"})
_BOARD_PHASES = frozenset({"INITIAL", "REFINE"})

_DISPOSITIONS = frozenset(
    {
        "COMMITTED",
        "ACCEPTED_NOOP",
        "REJECTED_NOT_INSTALLED",
        "QUARANTINED",
        "SUPERSEDED",
    }
)


def _assert_string_json_keys(
    value: Any,
    label: str,
    *,
    path: tuple[str, ...] = (),
    max_depth: int = MAX_JSON_DEPTH,
    max_nodes: int = MAX_JSON_NODES,
) -> None:
    stack: list[tuple[Any, tuple[str, ...], int, bool]] = [
        (value, path, 0, False)
    ]
    active_container_ids: set[int] = set()
    node_count = 0
    while stack:
        current, current_path, depth, exiting = stack.pop()
        if exiting:
            active_container_ids.remove(id(current))
            continue
        node_count += 1
        if node_count > max_nodes:
            raise V3ContractError(
                f"{label} exceeds the JSON node limit"
            )
        current_type = type(current)
        if current_type in (dict, list, tuple):
            if depth >= max_depth:
                raise V3ContractError(
                    f"{label} exceeds the JSON depth limit"
                )
            if len(current) > MAX_COLLECTION_ITEMS:
                raise V3ContractError(
                    f"{label} exceeds the JSON collection-item limit"
                )
            current_id = id(current)
            if current_id in active_container_ids:
                raise V3ContractError(f"{label} contains a JSON cycle")
            active_container_ids.add(current_id)
            stack.append((current, current_path, depth, True))
            if current_type is dict:
                children: list[tuple[Any, tuple[str, ...]]] = []
                for key, nested in current.items():
                    location = ".".join(current_path) or "<root>"
                    if type(key) is not str:
                        raise V3ContractError(
                            f"{label} JSON key at {location} must be a string"
                        )
                    try:
                        key.encode("utf-8")
                    except UnicodeEncodeError as exc:
                        raise V3ContractError(
                            f"{label} JSON key at {location} "
                            "must be valid UTF-8"
                        ) from exc
                    children.append((nested, (*current_path, key)))
            else:
                children = [
                    (nested, (*current_path, str(index)))
                    for index, nested in enumerate(current)
                ]
            for nested, nested_path in reversed(children):
                stack.append((nested, nested_path, depth + 1, False))
            continue
        location = ".".join(current_path) or "<root>"
        if current_type is str:
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise V3ContractError(
                    f"{label} JSON value at {location} must be valid UTF-8"
                ) from exc
        elif current is None or current_type is bool:
            continue
        elif current_type is int:
            if not (-MAX_PROTOCOL_INT - 1 <= current <= MAX_PROTOCOL_INT):
                raise V3ContractError(
                    f"{label} JSON integer at {location} is out of range"
                )
        elif current_type is float:
            import math

            if not math.isfinite(current):
                raise V3ContractError(
                    f"{label} JSON float at {location} must be finite"
                )
        else:
            raise V3ContractError(
                f"{label} JSON value at {location} has an unsupported type"
            )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {
                key: _freeze_json(nested)
                for key, nested in value.items()
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(nested) for nested in value]
    return value


def _nonempty(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise V3ContractError(f"{label} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise V3ContractError(f"{label} must be valid UTF-8 text") from exc
    return value


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if (
        type(value) is not int
        or value < 0
        or value > MAX_PROTOCOL_INT
    ):
        raise V3ContractError(
            f"{label} must be a non-negative integer within signed 64-bit range"
        )
    return value


def _exact_positive_int(value: Any, label: str) -> int:
    if (
        type(value) is not int
        or value <= 0
        or value > MAX_PROTOCOL_INT
    ):
        raise V3ContractError(
            f"{label} must be a positive integer within signed 64-bit range"
        )
    return value


def _sha256(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise V3ContractError(f"{label} must be a 64-character SHA-256")
    return value.lower()


def _phase(value: Any, label: str, *, allowed: frozenset[str] | None = None) -> str:
    if type(value) is not str:
        raise V3PhaseError(f"{label} must be a phase string")
    value = value.upper()
    if value not in (allowed or _PHASES):
        raise V3PhaseError(f"{label} must be one of {sorted(allowed or _PHASES)}")
    return value


def _disposition(value: Any, label: str) -> str:
    if type(value) is not str:
        raise V3SoulError(f"{label} must be a disposition string")
    value = value.upper()
    if value not in _DISPOSITIONS:
        raise V3SoulError(f"{label} must be one of {sorted(_DISPOSITIONS)}")
    return value


def _json_safe(value: Any, label: str) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise V3ContractError(f"{label} must be a mapping")
    _assert_string_json_keys(
        value,
        label,
        max_depth=MAX_JSON_DEPTH - 1,
        max_nodes=MAX_CONTEXT_JSON_NODES,
    )
    try:
        encoded = canonical_json_bytes(dict(value))
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise V3ContractError(f"{label} must be JSON-safe") from exc
    import json

    decoded = json.loads(encoded.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise V3ContractError(f"{label} must encode a JSON object")
    return _freeze_json(decoded)


def _sequence(values: Any, label: str) -> tuple[Any, ...]:
    if type(values) not in (list, tuple):
        raise V3ContractError(f"{label} must be an ordered sequence")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise V3ContractError(
            f"{label} exceeds the collection-item limit"
        )
    return tuple(values)


def _ordered_population(values: Sequence[str], label: str) -> tuple[str, ...]:
    """Preserve author order for the semantic online ring; reject duplicates."""
    try:
        items = _sequence(values, label)
    except V3ContractError as exc:
        raise V3PopulationError(str(exc)) from exc
    normalized = tuple(_nonempty(value, f"{label} item") for value in items)
    if len(normalized) > MAX_ONLINE_CORES:
        raise V3PopulationError(
            f"{label} exceeds the online-core limit"
        )
    if len(normalized) != len(set(normalized)):
        raise V3PopulationError(f"{label} cannot contain duplicates")
    if not normalized:
        raise V3PopulationError(f"{label} must contain at least one core")
    return normalized


def _sorted_strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    normalized = tuple(
        _nonempty(value, f"{label} item")
        for value in _sequence(values, label)
    )
    if len(normalized) != len(set(normalized)):
        raise V3ContractError(f"{label} cannot contain duplicates")
    return tuple(sorted(normalized))


def _derived_id(prefix: str, value: Mapping[str, Any]) -> str:
    try:
        digest = canonical_sha256(dict(value))
    except (TypeError, ValueError) as exc:
        raise V3ContractError(
            "record canonical payload is not valid JSON/UTF-8"
        ) from exc
    return f"{prefix}-{digest}"


def _sorted_pairs(
    pairs: Sequence[tuple[str, str]], label: str
) -> tuple[tuple[str, str], ...]:
    normalized_items: list[tuple[str, str]] = []
    for index, raw_pair in enumerate(_sequence(pairs, label)):
        if type(raw_pair) not in (list, tuple):
            raise V3ContractError(f"{label} item {index} must be a key/value pair")
        pair = tuple(raw_pair)
        if len(pair) != 2:
            raise V3ContractError(f"{label} item {index} must have length 2")
        normalized_items.append(
            (
                _nonempty(pair[0], f"{label} key"),
                _nonempty(pair[1], f"{label} value"),
            )
        )
    normalized = tuple(normalized_items)
    if len({key for key, _ in normalized}) != len(normalized):
        raise V3ContractError(f"{label} cannot contain duplicate keys")
    if len({value for _, value in normalized}) != len(normalized):
        raise V3ContractError(f"{label} cannot contain duplicate values")
    return tuple(sorted(normalized))


def _sorted_sha256_pairs(
    pairs: Sequence[tuple[str, str]], label: str
) -> tuple[tuple[str, str], ...]:
    """Canonicalize a keyed SHA map while allowing equal content hashes."""

    normalized_items: list[tuple[str, str]] = []
    for index, raw_pair in enumerate(_sequence(pairs, label)):
        if type(raw_pair) not in (list, tuple):
            raise V3ContractError(f"{label} item {index} must be a key/value pair")
        pair = tuple(raw_pair)
        if len(pair) != 2:
            raise V3ContractError(f"{label} item {index} must have length 2")
        normalized_items.append(
            (
                _nonempty(pair[0], f"{label} key"),
                _sha256(pair[1], f"{label} value"),
            )
        )
    normalized = tuple(normalized_items)
    if len({key for key, _ in normalized}) != len(normalized):
        raise V3ContractError(f"{label} cannot contain duplicate keys")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class TickPhasePlan:
    """Sealed role/ring epoch for one v3 cognitive tick."""

    tick_seq: int
    protocol_version: int
    input_field_id: str
    system_update_id: str
    working_field_id: str
    no_core_delta_output_field_id: str
    projection_id: str
    online_core_ids: tuple[str, ...]
    input_core_state_leaf_ids: tuple[tuple[str, str], ...]
    input_soul_sha256_by_core: tuple[tuple[str, str], ...]
    offline_core_id: str
    consolidator_core_id: str
    model_binding_epoch_id: str
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tick_seq",
            _exact_nonnegative_int(self.tick_seq, "tick_seq"),
        )
        object.__setattr__(
            self,
            "protocol_version",
            _exact_nonnegative_int(self.protocol_version, "protocol_version"),
        )
        if self.protocol_version != PROTOCOL_VERSION:
            raise V3ContractError(
                f"protocol_version must be {PROTOCOL_VERSION} for v3"
            )
        for name in (
            "input_field_id",
            "working_field_id",
            "no_core_delta_output_field_id",
            "projection_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("system_update_id", "model_binding_epoch_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        online = _ordered_population(self.online_core_ids, "online_core_ids")
        offline = _nonempty(self.offline_core_id, "offline_core_id")
        consolidator = _nonempty(self.consolidator_core_id, "consolidator_core_id")
        if offline in online:
            raise V3PopulationError("offline core cannot be online")
        if consolidator not in online:
            raise V3PopulationError("consolidator must be online")
        input_leaves = _sorted_pairs(
            self.input_core_state_leaf_ids,
            "input_core_state_leaf_ids",
        )
        input_souls = _sorted_sha256_pairs(
            self.input_soul_sha256_by_core,
            "input_soul_sha256_by_core",
        )
        online_set = set(online)
        if {core_id for core_id, _ in input_leaves} != online_set:
            raise V3PopulationError(
                "input core-state leaf authors must match online cores"
            )
        if {core_id for core_id, _ in input_souls} != online_set:
            raise V3PopulationError(
                "input soul-hash authors must match online cores"
            )
        object.__setattr__(self, "online_core_ids", online)
        object.__setattr__(
            self,
            "input_core_state_leaf_ids",
            input_leaves,
        )
        object.__setattr__(
            self,
            "input_soul_sha256_by_core",
            input_souls,
        )
        object.__setattr__(self, "offline_core_id", offline)
        object.__setattr__(self, "consolidator_core_id", consolidator)
        object.__setattr__(
            self,
            "plan_id",
            _derived_id("tick-phase-plan", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-tick-phase-plan-v3",
            "tick_seq": self.tick_seq,
            "protocol_version": self.protocol_version,
            "protocol_name": PROTOCOL_NAME,
            "input_field_id": self.input_field_id,
            "system_update_id": self.system_update_id,
            "working_field_id": self.working_field_id,
            "no_core_delta_output_field_id": (
                self.no_core_delta_output_field_id
            ),
            "projection_id": self.projection_id,
            "online_core_ids": list(self.online_core_ids),
            "input_core_state_leaf_ids": dict(
                self.input_core_state_leaf_ids
            ),
            "input_soul_sha256_by_core": dict(
                self.input_soul_sha256_by_core
            ),
            "offline_core_id": self.offline_core_id,
            "consolidator_core_id": self.consolidator_core_id,
            "model_binding_epoch_id": self.model_binding_epoch_id,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["plan_id"] = self.plan_id
        return value


@dataclass(frozen=True, slots=True)
class ReadCycleManifest:
    """Ordered page/view hashes for one core's attendance over a projection."""

    core_id: str
    tick_seq: int
    phase: str
    working_field_id: str
    projection_id: str
    board_id: str | None
    page_view_hashes: tuple[str, ...]
    page_character_counts: tuple[int, ...]
    cursor_chain: tuple[str, ...]
    expected_character_count: int
    coverage_complete: bool
    selected_span_fingerprint: str
    cycle_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_id", _nonempty(self.core_id, "core_id"))
        object.__setattr__(
            self,
            "tick_seq",
            _exact_nonnegative_int(self.tick_seq, "tick_seq"),
        )
        object.__setattr__(
            self,
            "phase",
            _phase(self.phase, "phase"),
        )
        object.__setattr__(
            self,
            "projection_id",
            _sha256(self.projection_id, "projection_id"),
        )
        object.__setattr__(
            self,
            "working_field_id",
            _sha256(self.working_field_id, "working_field_id"),
        )
        if self.phase == "INITIAL":
            if self.board_id is not None:
                raise V3PhaseError(
                    "INITIAL read cycle cannot reference a proposal board"
                )
        else:
            object.__setattr__(
                self,
                "board_id",
                _nonempty(self.board_id, "board_id"),
            )
        object.__setattr__(
            self,
            "selected_span_fingerprint",
            _sha256(self.selected_span_fingerprint, "selected_span_fingerprint"),
        )
        if type(self.coverage_complete) is not bool:
            raise V3ContractError("coverage_complete must be boolean")
        page_views = tuple(
            _sha256(value, "page_view_hashes item")
            for value in _sequence(self.page_view_hashes, "page_view_hashes")
        )
        cursor_chain = tuple(
            _sha256(value, "cursor_chain item")
            for value in _sequence(self.cursor_chain, "cursor_chain")
        )
        page_character_counts = tuple(
            _exact_positive_int(value, "page_character_counts item")
            for value in _sequence(
                self.page_character_counts,
                "page_character_counts",
            )
        )
        expected_character_count = _exact_positive_int(
            self.expected_character_count,
            "expected_character_count",
        )
        if not (
            len(page_views)
            == len(cursor_chain)
            == len(page_character_counts)
        ):
            raise V3ContractError(
                "page views, character counts, and cursor chain must have equal lengths"
            )
        covered_character_count = sum(page_character_counts)
        if covered_character_count > expected_character_count:
            raise V3ContractError(
                "read cycle cannot cover more characters than expected"
            )
        if self.coverage_complete and (
            not page_views
            or covered_character_count != expected_character_count
        ):
            raise V3ContractError(
                "a complete read cycle must cover every expected character"
            )
        if (
            not self.coverage_complete
            and covered_character_count == expected_character_count
        ):
            raise V3ContractError(
                "an incomplete read cycle cannot cover every expected character"
            )
        object.__setattr__(self, "page_view_hashes", page_views)
        object.__setattr__(
            self,
            "page_character_counts",
            page_character_counts,
        )
        object.__setattr__(self, "cursor_chain", cursor_chain)
        object.__setattr__(
            self,
            "expected_character_count",
            expected_character_count,
        )
        object.__setattr__(
            self,
            "cycle_id",
            _derived_id("read-cycle", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-read-cycle-manifest-v3",
            "core_id": self.core_id,
            "tick_seq": self.tick_seq,
            "phase": self.phase,
            "working_field_id": self.working_field_id,
            "projection_id": self.projection_id,
            "board_id": self.board_id,
            "page_view_hashes": list(self.page_view_hashes),
            "page_character_counts": list(self.page_character_counts),
            "cursor_chain": list(self.cursor_chain),
            "expected_character_count": self.expected_character_count,
            "coverage_complete": self.coverage_complete,
            "selected_span_fingerprint": self.selected_span_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["cycle_id"] = self.cycle_id
        return value


@dataclass(frozen=True, slots=True)
class SoulTransitionRecord:
    """One immutable staged private-state transition inside a tick."""

    core_id: str
    tick_seq: int
    phase: str
    substep: int
    input_soul_sha256: str
    output_soul_sha256: str
    working_field_id: str
    projection_id: str
    read_cycle_id: str
    board_id: str | None
    delta_id: str
    candidate_private_state_id: str
    parent_transition_id: str | None
    cursor_state_sha256: str
    rng_state_sha256: str | None
    binding_manifest_id: str
    transition_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_id", _nonempty(self.core_id, "core_id"))
        object.__setattr__(
            self,
            "tick_seq",
            _exact_nonnegative_int(self.tick_seq, "tick_seq"),
        )
        phase = _phase(self.phase, "phase")
        object.__setattr__(self, "phase", phase)
        substep = _exact_nonnegative_int(self.substep, "substep")
        if phase == "INITIAL":
            if substep != 0:
                raise V3PhaseError("INITIAL transition substep must be 0")
            if self.board_id is not None:
                raise V3PhaseError("INITIAL transition cannot claim a board parent")
            if self.parent_transition_id is not None:
                raise V3PhaseError(
                    "INITIAL transition cannot claim a transition parent"
                )
        elif phase == "REFINE":
            if substep != 1:
                raise V3PhaseError("REFINE transition substep must be 1")
            if self.board_id is None:
                raise V3PhaseError(
                    "REFINE transition requires an initial-board parent"
                )
            if self.parent_transition_id is None:
                raise V3PhaseError(
                    "REFINE transition requires an INITIAL transition parent"
                )
        elif phase == "FINAL":
            if substep != 2:
                raise V3PhaseError("FINAL transition substep must be 2")
            if self.board_id is None:
                raise V3PhaseError(
                    "FINAL transition requires a refinement-board parent"
                )
            if self.parent_transition_id is None:
                raise V3PhaseError(
                    "FINAL transition requires a REFINE transition parent"
                )
        object.__setattr__(self, "substep", substep)
        for name in ("input_soul_sha256", "output_soul_sha256", "cursor_state_sha256"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("working_field_id", "projection_id"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("read_cycle_id", "delta_id", "candidate_private_state_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.board_id is not None:
            object.__setattr__(
                self,
                "board_id",
                _nonempty(self.board_id, "board_id"),
            )
        if self.parent_transition_id is not None:
            object.__setattr__(
                self,
                "parent_transition_id",
                _nonempty(self.parent_transition_id, "parent_transition_id"),
            )
        object.__setattr__(
            self,
            "rng_state_sha256",
            _sha256(self.rng_state_sha256, "rng_state_sha256", optional=True),
        )
        object.__setattr__(
            self,
            "binding_manifest_id",
            _nonempty(self.binding_manifest_id, "binding_manifest_id"),
        )
        object.__setattr__(
            self,
            "transition_id",
            _derived_id("soul-transition", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-soul-transition-v3",
            "core_id": self.core_id,
            "tick_seq": self.tick_seq,
            "phase": self.phase,
            "substep": self.substep,
            "input_soul_sha256": self.input_soul_sha256,
            "output_soul_sha256": self.output_soul_sha256,
            "working_field_id": self.working_field_id,
            "projection_id": self.projection_id,
            "read_cycle_id": self.read_cycle_id,
            "board_id": self.board_id,
            "delta_id": self.delta_id,
            "candidate_private_state_id": self.candidate_private_state_id,
            "parent_transition_id": self.parent_transition_id,
            "cursor_state_sha256": self.cursor_state_sha256,
            "rng_state_sha256": self.rng_state_sha256,
            "binding_manifest_id": self.binding_manifest_id,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["transition_id"] = self.transition_id
        return value


@dataclass(frozen=True, slots=True)
class SoulTransitionDispositionRecord:
    """Outcome applied to a staged soul transition after the tick decision."""

    transition_id: str
    disposition: str
    reason: str
    superseded_by_transition_id: str | None = None
    disposition_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transition_id",
            _nonempty(self.transition_id, "transition_id"),
        )
        object.__setattr__(
            self,
            "disposition",
            _disposition(self.disposition, "disposition"),
        )
        object.__setattr__(self, "reason", _nonempty(self.reason, "reason"))
        if self.superseded_by_transition_id is not None:
            object.__setattr__(
                self,
                "superseded_by_transition_id",
                _nonempty(
                    self.superseded_by_transition_id,
                    "superseded_by_transition_id",
                ),
            )
        if self.disposition == "SUPERSEDED":
            if self.superseded_by_transition_id is None:
                raise V3SoulError(
                    "SUPERSEDED disposition requires superseded_by_transition_id"
                )
            if self.superseded_by_transition_id == self.transition_id:
                raise V3SoulError(
                    "SUPERSEDED disposition cannot point to the same transition"
                )
        elif self.superseded_by_transition_id is not None:
            raise V3SoulError(
                "only SUPERSEDED disposition may carry "
                "superseded_by_transition_id"
            )
        object.__setattr__(
            self,
            "disposition_id",
            _derived_id("soul-disposition", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-soul-transition-disposition-v3",
            "transition_id": self.transition_id,
            "disposition": self.disposition,
            "reason": self.reason,
            "superseded_by_transition_id": self.superseded_by_transition_id,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["disposition_id"] = self.disposition_id
        return value


@dataclass(frozen=True, slots=True)
class CorePassRecord:
    """One core's pass delta authored against the same working field W."""

    core_id: str
    tick_seq: int
    phase: str
    working_field_id: str
    board_id: str | None
    read_cycle_id: str
    delta_id: str
    soul_transition_id: str
    candidate_private_state_id: str
    pass_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_id", _nonempty(self.core_id, "core_id"))
        object.__setattr__(
            self,
            "tick_seq",
            _exact_nonnegative_int(self.tick_seq, "tick_seq"),
        )
        phase = _phase(self.phase, "phase", allowed=_PASS_PHASES)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(
            self,
            "working_field_id",
            _sha256(self.working_field_id, "working_field_id"),
        )
        if phase == "INITIAL" and self.board_id is not None:
            raise V3PhaseError("INITIAL pass cannot claim a board parent")
        if phase == "REFINE" and self.board_id is None:
            raise V3PhaseError("REFINE pass requires an initial-board parent")
        if self.board_id is not None:
            object.__setattr__(
                self,
                "board_id",
                _nonempty(self.board_id, "board_id"),
            )
        for name in (
            "read_cycle_id",
            "delta_id",
            "soul_transition_id",
            "candidate_private_state_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "pass_id",
            _derived_id("core-pass", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-core-pass-v3",
            "core_id": self.core_id,
            "tick_seq": self.tick_seq,
            "phase": self.phase,
            "working_field_id": self.working_field_id,
            "board_id": self.board_id,
            "read_cycle_id": self.read_cycle_id,
            "delta_id": self.delta_id,
            "soul_transition_id": self.soul_transition_id,
            "candidate_private_state_id": self.candidate_private_state_id,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["pass_id"] = self.pass_id
        return value


@dataclass(frozen=True, slots=True)
class ProposalBoardManifest:
    """Sealed, order-independent set of passes for one phase."""

    working_field_id: str
    phase: str
    pass_ids_by_author: tuple[tuple[str, str], ...]
    required_authors: tuple[str, ...]
    explicit_conflicts: tuple[str, ...] = ()
    board_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "working_field_id",
            _sha256(self.working_field_id, "working_field_id"),
        )
        phase = _phase(self.phase, "phase", allowed=_BOARD_PHASES)
        object.__setattr__(self, "phase", phase)
        by_author = _sorted_pairs(self.pass_ids_by_author, "pass_ids_by_author")
        required = _sorted_strings(self.required_authors, "required_authors")
        if not required:
            raise V3BoardError("board must require at least one online author")
        authors = {author for author, _ in by_author}
        if authors != set(required):
            raise V3BoardError(
                "board required authors must exactly match pass-author mapping"
            )
        object.__setattr__(self, "pass_ids_by_author", by_author)
        object.__setattr__(self, "required_authors", required)
        object.__setattr__(
            self,
            "explicit_conflicts",
            _sorted_strings(self.explicit_conflicts, "explicit_conflicts"),
        )
        object.__setattr__(
            self,
            "board_id",
            _derived_id("proposal-board", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-proposal-board-v3",
            "working_field_id": self.working_field_id,
            "phase": self.phase,
            "pass_ids_by_author": dict(self.pass_ids_by_author),
            "required_authors": list(self.required_authors),
            "explicit_conflicts": list(self.explicit_conflicts),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["board_id"] = self.board_id
        return value


@dataclass(frozen=True, slots=True)
class ConsolidationRecord:
    """Final pass authored by the rotating consolidator against W + B1."""

    core_id: str
    tick_seq: int
    working_field_id: str
    refinement_board_id: str
    final_delta_id: str
    final_soul_transition_id: str
    accepted: bool
    reason: str
    invocation_request_ids: tuple[str, ...] = ()
    consolidation_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "core_id", _nonempty(self.core_id, "core_id"))
        object.__setattr__(
            self,
            "tick_seq",
            _exact_nonnegative_int(self.tick_seq, "tick_seq"),
        )
        object.__setattr__(
            self,
            "working_field_id",
            _sha256(self.working_field_id, "working_field_id"),
        )
        object.__setattr__(
            self,
            "refinement_board_id",
            _nonempty(self.refinement_board_id, "refinement_board_id"),
        )
        for name in ("final_delta_id", "final_soul_transition_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if type(self.accepted) is not bool:
            raise V3ContractError("accepted must be boolean")
        object.__setattr__(self, "reason", _nonempty(self.reason, "reason"))
        object.__setattr__(
            self,
            "invocation_request_ids",
            _sorted_strings(
                self.invocation_request_ids,
                "invocation_request_ids",
            ),
        )
        if not self.accepted and self.invocation_request_ids:
            raise V3ContractError(
                "rejected consolidation cannot carry invocation requests"
            )
        object.__setattr__(
            self,
            "consolidation_id",
            _derived_id("consolidation", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-consolidation-v3",
            "core_id": self.core_id,
            "tick_seq": self.tick_seq,
            "working_field_id": self.working_field_id,
            "refinement_board_id": self.refinement_board_id,
            "final_delta_id": self.final_delta_id,
            "final_soul_transition_id": self.final_soul_transition_id,
            "accepted": self.accepted,
            "reason": self.reason,
            "invocation_request_ids": list(self.invocation_request_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["consolidation_id"] = self.consolidation_id
        return value


@dataclass(frozen=True, slots=True)
class TickCommitRecordV3:
    """Atomic v3 tick commit referencing every required artifact."""

    tick_seq: int
    plan_id: str
    initial_board_id: str
    refine_board_id: str
    initial_pass_ids: tuple[tuple[str, str], ...]
    refine_pass_ids: tuple[tuple[str, str], ...]
    consolidation_id: str
    disposition_ids_by_transition: tuple[tuple[str, str], ...]
    final_core_state_leaf_ids: tuple[tuple[str, str], ...]
    field_transaction_audit_id: str
    output_field_id: str
    invocation_request_ids: tuple[str, ...] = ()
    commit_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tick_seq",
            _exact_nonnegative_int(self.tick_seq, "tick_seq"),
        )
        for name in (
            "plan_id",
            "initial_board_id",
            "refine_board_id",
            "consolidation_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        initial_pass_ids = _sorted_pairs(
            self.initial_pass_ids,
            "initial_pass_ids",
        )
        refine_pass_ids = _sorted_pairs(
            self.refine_pass_ids,
            "refine_pass_ids",
        )
        final_core_state_leaf_ids = _sorted_pairs(
            self.final_core_state_leaf_ids,
            "final_core_state_leaf_ids",
        )
        disposition_ids_by_transition = _sorted_pairs(
            self.disposition_ids_by_transition,
            "disposition_ids_by_transition",
        )
        if not initial_pass_ids:
            raise V3ContractError(
                "tick commit must reference at least one online core"
            )
        if not disposition_ids_by_transition:
            raise V3ContractError(
                "tick commit must reference transition dispositions"
            )
        initial_authors = {author for author, _ in initial_pass_ids}
        if {author for author, _ in refine_pass_ids} != initial_authors:
            raise V3ContractError(
                "initial and refine pass author sets must match"
            )
        if {
            author for author, _ in final_core_state_leaf_ids
        } != initial_authors:
            raise V3ContractError(
                "final state-leaf authors must match pass authors"
            )
        object.__setattr__(self, "initial_pass_ids", initial_pass_ids)
        object.__setattr__(self, "refine_pass_ids", refine_pass_ids)
        object.__setattr__(
            self,
            "final_core_state_leaf_ids",
            final_core_state_leaf_ids,
        )
        object.__setattr__(
            self,
            "disposition_ids_by_transition",
            disposition_ids_by_transition,
        )
        object.__setattr__(
            self,
            "field_transaction_audit_id",
            _sha256(self.field_transaction_audit_id, "field_transaction_audit_id"),
        )
        object.__setattr__(
            self,
            "output_field_id",
            _sha256(self.output_field_id, "output_field_id"),
        )
        object.__setattr__(
            self,
            "invocation_request_ids",
            _sorted_strings(
                self.invocation_request_ids,
                "invocation_request_ids",
            ),
        )
        object.__setattr__(
            self,
            "commit_id",
            _derived_id("tick-commit-v3", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-tick-commit-v3",
            "tick_seq": self.tick_seq,
            "plan_id": self.plan_id,
            "initial_board_id": self.initial_board_id,
            "refine_board_id": self.refine_board_id,
            "initial_pass_ids": dict(self.initial_pass_ids),
            "refine_pass_ids": dict(self.refine_pass_ids),
            "consolidation_id": self.consolidation_id,
            "disposition_ids_by_transition": dict(
                self.disposition_ids_by_transition
            ),
            "final_core_state_leaf_ids": dict(self.final_core_state_leaf_ids),
            "field_transaction_audit_id": self.field_transaction_audit_id,
            "output_field_id": self.output_field_id,
            "invocation_request_ids": list(self.invocation_request_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["commit_id"] = self.commit_id
        return value


@dataclass(frozen=True, slots=True)
class ArtifactRejectionRecord:
    """Structural invalidity audit; malformed material never becomes a pass."""

    artifact_kind: str
    reason: str
    source_context: Mapping[str, Any]
    raw_fingerprint_sha256: str
    rejection_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_kind",
            _nonempty(self.artifact_kind, "artifact_kind"),
        )
        object.__setattr__(self, "reason", _nonempty(self.reason, "reason"))
        object.__setattr__(
            self,
            "source_context",
            _json_safe(self.source_context, "source_context"),
        )
        object.__setattr__(
            self,
            "raw_fingerprint_sha256",
            _sha256(self.raw_fingerprint_sha256, "raw_fingerprint_sha256"),
        )
        object.__setattr__(
            self,
            "rejection_id",
            _derived_id("artifact-rejection", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-artifact-rejection-v3",
            "artifact_kind": self.artifact_kind,
            "reason": self.reason,
            "source_context": _thaw_json(self.source_context),
            "raw_fingerprint_sha256": self.raw_fingerprint_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["rejection_id"] = self.rejection_id
        return value


def validate_pass_against_read_cycle(
    pass_record: CorePassRecord,
    read_cycle: ReadCycleManifest,
) -> None:
    """A valid pass must be paired with a complete read cycle over the same W."""
    if type(pass_record) is not CorePassRecord:
        raise V3ContractError("pass_record must be CorePassRecord")
    if type(read_cycle) is not ReadCycleManifest:
        raise V3ContractError("read_cycle must be ReadCycleManifest")
    if pass_record.core_id != read_cycle.core_id:
        raise V3ContractError("pass and read-cycle core IDs mismatch")
    if pass_record.tick_seq != read_cycle.tick_seq:
        raise V3ContractError("pass and read-cycle tick_seq mismatch")
    if pass_record.phase != read_cycle.phase:
        raise V3ContractError("pass and read-cycle phase mismatch")
    if pass_record.working_field_id != read_cycle.working_field_id:
        raise V3ContractError("pass and read-cycle working field mismatch")
    if pass_record.board_id != read_cycle.board_id:
        raise V3ContractError("pass and read-cycle board parent mismatch")
    if pass_record.read_cycle_id != read_cycle.cycle_id:
        raise V3ContractError("pass read_cycle_id does not match manifest")
    if not read_cycle.coverage_complete:
        raise V3ContractError("pass cannot reference an incomplete read cycle")


def validate_consolidation_against_plan(
    consolidation: ConsolidationRecord,
    plan: TickPhasePlan,
    refinement_board: ProposalBoardManifest,
) -> None:
    """Bind FINAL authority to the plan and its complete REFINE board."""
    if type(consolidation) is not ConsolidationRecord:
        raise V3ContractError("consolidation must be ConsolidationRecord")
    if type(plan) is not TickPhasePlan:
        raise V3ContractError("plan must be TickPhasePlan")
    if type(refinement_board) is not ProposalBoardManifest:
        raise V3ContractError(
            "refinement_board must be ProposalBoardManifest"
        )
    if consolidation.core_id != plan.consolidator_core_id:
        raise V3ContractError("consolidator core_id does not match plan")
    if consolidation.tick_seq != plan.tick_seq:
        raise V3ContractError("consolidation tick_seq does not match plan")
    if consolidation.working_field_id != plan.working_field_id:
        raise V3ContractError("consolidation working field does not match plan")
    if refinement_board.phase != "REFINE":
        raise V3ContractError("final consolidation requires a REFINE board")
    if refinement_board.working_field_id != plan.working_field_id:
        raise V3ContractError("refinement board working field does not match plan")
    if set(refinement_board.required_authors) != set(plan.online_core_ids):
        raise V3ContractError(
            "refinement board authors do not match the online population"
        )
    if consolidation.refinement_board_id != refinement_board.board_id:
        raise V3ContractError(
            "consolidation does not reference the supplied REFINE board"
        )


def _records_by_id(
    records: Any,
    *,
    record_type: type[Any],
    id_attribute: str,
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, record in enumerate(_sequence(records, label)):
        if type(record) is not record_type:
            raise V3ContractError(
                f"{label} item {index} must be {record_type.__name__}"
            )
        record_id = getattr(record, id_attribute)
        if record_id in result:
            raise V3ContractError(f"{label} contains duplicate record IDs")
        result[record_id] = record
    return result


def validate_tick_artifact_graph(
    *,
    plan: TickPhasePlan,
    read_cycles: Sequence[ReadCycleManifest],
    soul_transitions: Sequence[SoulTransitionRecord],
    initial_passes: Sequence[CorePassRecord],
    initial_board: ProposalBoardManifest,
    refine_passes: Sequence[CorePassRecord],
    refine_board: ProposalBoardManifest,
    consolidation: ConsolidationRecord,
    dispositions: Sequence[SoulTransitionDispositionRecord],
    commit: TickCommitRecordV3,
) -> None:
    """Validate the complete neural-free v3 artifact graph for one tick.

    This proves population completeness, board sealing, same-W references,
    within-tick soul chaining, FINAL authority, disposition coverage, installed
    state leaves, and commit/invocation identity. Delta payload authorship and
    field-transaction replay are intentionally deferred to the transaction
    layer because this contracts module stores only their content IDs.
    """

    if type(plan) is not TickPhasePlan:
        raise V3ContractError("plan must be TickPhasePlan")
    if type(initial_board) is not ProposalBoardManifest:
        raise V3ContractError("initial_board must be ProposalBoardManifest")
    if type(refine_board) is not ProposalBoardManifest:
        raise V3ContractError("refine_board must be ProposalBoardManifest")
    if type(consolidation) is not ConsolidationRecord:
        raise V3ContractError("consolidation must be ConsolidationRecord")
    if type(commit) is not TickCommitRecordV3:
        raise V3ContractError("commit must be TickCommitRecordV3")

    online = tuple(plan.online_core_ids)
    online_set = set(online)
    input_leaf_by_author = dict(plan.input_core_state_leaf_ids)
    input_soul_by_author = dict(plan.input_soul_sha256_by_core)
    for board, phase, label in (
        (initial_board, "INITIAL", "initial board"),
        (refine_board, "REFINE", "refine board"),
    ):
        if board.phase != phase:
            raise V3ContractError(f"{label} has the wrong phase")
        if board.working_field_id != plan.working_field_id:
            raise V3ContractError(f"{label} working field does not match plan")
        if set(board.required_authors) != online_set:
            raise V3ContractError(
                f"{label} authors do not match the online population"
            )

    def validate_pass_phase(
        records: Sequence[CorePassRecord],
        *,
        phase: str,
        board: ProposalBoardManifest,
        expected_parent_board_id: str | None,
    ) -> dict[str, CorePassRecord]:
        items = _sequence(records, f"{phase} passes")
        if len(items) != len(online):
            raise V3ContractError(
                f"{phase} must contain exactly one pass per online core"
            )
        by_author: dict[str, CorePassRecord] = {}
        for pass_record in items:
            if type(pass_record) is not CorePassRecord:
                raise V3ContractError(
                    f"{phase} passes must contain CorePassRecord values"
                )
            if pass_record.core_id not in online_set:
                raise V3ContractError(
                    f"{phase} pass author is not an online core"
                )
            if pass_record.core_id in by_author:
                raise V3ContractError(
                    f"{phase} contains duplicate pass authors"
                )
            if pass_record.tick_seq != plan.tick_seq:
                raise V3ContractError(f"{phase} pass tick does not match plan")
            if pass_record.phase != phase:
                raise V3ContractError(f"{phase} pass has the wrong phase")
            if pass_record.working_field_id != plan.working_field_id:
                raise V3ContractError(
                    f"{phase} pass working field does not match plan"
                )
            if pass_record.board_id != expected_parent_board_id:
                raise V3ContractError(
                    f"{phase} pass has the wrong board parent"
                )
            by_author[pass_record.core_id] = pass_record
        expected_mapping = tuple(
            sorted(
                (author, pass_record.pass_id)
                for author, pass_record in by_author.items()
            )
        )
        if board.pass_ids_by_author != expected_mapping:
            raise V3ContractError(
                f"{phase} board does not exactly reference supplied passes"
            )
        return by_author

    initial_by_author = validate_pass_phase(
        initial_passes,
        phase="INITIAL",
        board=initial_board,
        expected_parent_board_id=None,
    )
    refine_by_author = validate_pass_phase(
        refine_passes,
        phase="REFINE",
        board=refine_board,
        expected_parent_board_id=initial_board.board_id,
    )

    cycle_by_id = _records_by_id(
        read_cycles,
        record_type=ReadCycleManifest,
        id_attribute="cycle_id",
        label="read_cycles",
    )
    transition_by_id = _records_by_id(
        soul_transitions,
        record_type=SoulTransitionRecord,
        id_attribute="transition_id",
        label="soul_transitions",
    )
    referenced_cycle_ids: set[str] = set()
    referenced_transition_ids: set[str] = set()

    def validate_pass_artifacts(
        pass_record: CorePassRecord,
    ) -> SoulTransitionRecord:
        try:
            read_cycle = cycle_by_id[pass_record.read_cycle_id]
        except KeyError as exc:
            raise V3ContractError(
                "pass references a missing read-cycle manifest"
            ) from exc
        validate_pass_against_read_cycle(pass_record, read_cycle)
        if read_cycle.projection_id != plan.projection_id:
            raise V3ContractError(
                "pass read cycle projection does not match plan"
            )
        referenced_cycle_ids.add(read_cycle.cycle_id)

        try:
            transition = transition_by_id[pass_record.soul_transition_id]
        except KeyError as exc:
            raise V3ContractError(
                "pass references a missing soul transition"
            ) from exc
        for actual, expected, label in (
            (transition.core_id, pass_record.core_id, "core"),
            (transition.tick_seq, pass_record.tick_seq, "tick"),
            (transition.phase, pass_record.phase, "phase"),
            (
                transition.working_field_id,
                pass_record.working_field_id,
                "working field",
            ),
            (transition.projection_id, plan.projection_id, "projection"),
            (
                transition.read_cycle_id,
                pass_record.read_cycle_id,
                "read cycle",
            ),
            (transition.board_id, pass_record.board_id, "board parent"),
            (transition.delta_id, pass_record.delta_id, "delta"),
            (
                transition.candidate_private_state_id,
                pass_record.candidate_private_state_id,
                "candidate private state",
            ),
            (
                transition.binding_manifest_id,
                plan.model_binding_epoch_id,
                "model binding",
            ),
        ):
            if actual != expected:
                raise V3ContractError(
                    f"pass and soul transition {label} mismatch"
                )
        referenced_transition_ids.add(transition.transition_id)
        return transition

    initial_transition_by_author = {
        author: validate_pass_artifacts(pass_record)
        for author, pass_record in initial_by_author.items()
    }
    refine_transition_by_author = {
        author: validate_pass_artifacts(pass_record)
        for author, pass_record in refine_by_author.items()
    }
    for author in online:
        initial_transition = initial_transition_by_author[author]
        refine_transition = refine_transition_by_author[author]
        if initial_transition.input_soul_sha256 != input_soul_by_author[author]:
            raise V3ContractError(
                "INITIAL transition input soul does not match sealed prestate"
            )
        if refine_transition.parent_transition_id != initial_transition.transition_id:
            raise V3ContractError(
                "REFINE transition does not chain from the core's INITIAL transition"
            )
        if refine_transition.input_soul_sha256 != initial_transition.output_soul_sha256:
            raise V3ContractError(
                "REFINE transition input soul does not match INITIAL output soul"
            )

    validate_consolidation_against_plan(
        consolidation,
        plan,
        refine_board,
    )
    try:
        final_transition = transition_by_id[
            consolidation.final_soul_transition_id
        ]
    except KeyError as exc:
        raise V3ContractError(
            "consolidation references a missing FINAL soul transition"
        ) from exc
    consolidator_refine = refine_transition_by_author[
        plan.consolidator_core_id
    ]
    for actual, expected, label in (
        (final_transition.core_id, plan.consolidator_core_id, "core"),
        (final_transition.tick_seq, plan.tick_seq, "tick"),
        (final_transition.phase, "FINAL", "phase"),
        (
            final_transition.working_field_id,
            plan.working_field_id,
            "working field",
        ),
        (final_transition.projection_id, plan.projection_id, "projection"),
        (final_transition.board_id, refine_board.board_id, "board parent"),
        (
            final_transition.parent_transition_id,
            consolidator_refine.transition_id,
            "transition parent",
        ),
        (
            final_transition.input_soul_sha256,
            consolidator_refine.output_soul_sha256,
            "input soul",
        ),
        (
            final_transition.delta_id,
            consolidation.final_delta_id,
            "final delta",
        ),
        (
            final_transition.binding_manifest_id,
            plan.model_binding_epoch_id,
            "model binding",
        ),
    ):
        if actual != expected:
            raise V3ContractError(
                f"FINAL soul transition {label} mismatch"
            )
    try:
        final_read_cycle = cycle_by_id[final_transition.read_cycle_id]
    except KeyError as exc:
        raise V3ContractError(
            "FINAL transition references a missing read cycle"
        ) from exc
    if (
        final_read_cycle.core_id != plan.consolidator_core_id
        or final_read_cycle.tick_seq != plan.tick_seq
        or final_read_cycle.phase != "FINAL"
        or final_read_cycle.working_field_id != plan.working_field_id
        or final_read_cycle.projection_id != plan.projection_id
        or final_read_cycle.board_id != refine_board.board_id
        or not final_read_cycle.coverage_complete
        or final_transition.read_cycle_id != final_read_cycle.cycle_id
    ):
        raise V3ContractError(
            "FINAL transition read cycle does not match the plan"
        )
    referenced_cycle_ids.add(final_read_cycle.cycle_id)
    referenced_transition_ids.add(final_transition.transition_id)

    if referenced_cycle_ids != set(cycle_by_id):
        raise V3ContractError("tick contains orphan read-cycle manifests")
    if referenced_transition_ids != set(transition_by_id):
        raise V3ContractError("tick contains orphan soul transitions")

    disposition_by_transition: dict[
        str,
        SoulTransitionDispositionRecord,
    ] = {}
    for disposition in _sequence(dispositions, "dispositions"):
        if type(disposition) is not SoulTransitionDispositionRecord:
            raise V3ContractError(
                "dispositions must contain SoulTransitionDispositionRecord values"
            )
        if disposition.transition_id in disposition_by_transition:
            raise V3ContractError(
                "tick contains duplicate transition dispositions"
            )
        disposition_by_transition[disposition.transition_id] = disposition
    if set(disposition_by_transition) != set(transition_by_id):
        raise V3ContractError(
            "tick must disposition every and only supplied soul transition"
        )
    expected_disposition_ids = tuple(
        sorted(
            (
                transition_id,
                disposition.disposition_id,
            )
            for transition_id, disposition in disposition_by_transition.items()
        )
    )
    if commit.disposition_ids_by_transition != expected_disposition_ids:
        raise V3ContractError(
            "commit disposition references do not match supplied dispositions"
        )

    if consolidation.accepted:
        for author in online:
            initial_transition = initial_transition_by_author[author]
            refine_transition = refine_transition_by_author[author]
            initial_disposition = disposition_by_transition[
                initial_transition.transition_id
            ]
            if (
                initial_disposition.disposition != "SUPERSEDED"
                or initial_disposition.superseded_by_transition_id
                != refine_transition.transition_id
            ):
                raise V3ContractError(
                    "accepted tick must supersede INITIAL with REFINE"
                )
        consolidator_refine_disposition = disposition_by_transition[
            consolidator_refine.transition_id
        ]
        if (
            consolidator_refine_disposition.disposition != "SUPERSEDED"
            or consolidator_refine_disposition.superseded_by_transition_id
            != final_transition.transition_id
        ):
            raise V3ContractError(
                "accepted tick must supersede consolidator REFINE with FINAL"
            )
        terminal_by_author = {
            author: (
                final_transition
                if author == plan.consolidator_core_id
                else refine_transition_by_author[author]
            )
            for author in online
        }
        expected_leaf_items: list[tuple[str, str]] = []
        for author, transition in terminal_by_author.items():
            terminal_disposition = disposition_by_transition[
                transition.transition_id
            ].disposition
            if terminal_disposition not in {"COMMITTED", "ACCEPTED_NOOP"}:
                raise V3ContractError(
                    "accepted tick terminal transitions must be installed"
                )
            if terminal_disposition == "COMMITTED":
                expected_leaf = transition.candidate_private_state_id
            else:
                expected_leaf = input_leaf_by_author[author]
                if (
                    transition.candidate_private_state_id != expected_leaf
                    or transition.output_soul_sha256
                    != input_soul_by_author[author]
                ):
                    raise V3ContractError(
                        "ACCEPTED_NOOP must preserve the sealed input private state"
                    )
            expected_leaf_items.append((author, expected_leaf))
        expected_leaves = tuple(sorted(expected_leaf_items))
        if commit.final_core_state_leaf_ids != expected_leaves:
            raise V3ContractError(
                "commit final state leaves do not match terminal transitions"
            )
    else:
        if any(
            disposition.disposition != "REJECTED_NOT_INSTALLED"
            for disposition in disposition_by_transition.values()
        ):
            raise V3ContractError(
                "rejected tick transitions must be REJECTED_NOT_INSTALLED"
            )
        if commit.final_core_state_leaf_ids != plan.input_core_state_leaf_ids:
            raise V3ContractError(
                "rejected tick must preserve sealed input state leaves"
            )
        if (
            commit.output_field_id
            != plan.no_core_delta_output_field_id
        ):
            raise V3ContractError(
                "rejected final delta must use the sealed no-core-delta successor"
            )

    if commit.tick_seq != plan.tick_seq:
        raise V3ContractError("commit tick does not match plan")
    if commit.plan_id != plan.plan_id:
        raise V3ContractError("commit plan_id does not match plan")
    if commit.initial_board_id != initial_board.board_id:
        raise V3ContractError("commit initial_board_id mismatch")
    if commit.refine_board_id != refine_board.board_id:
        raise V3ContractError("commit refine_board_id mismatch")
    if commit.initial_pass_ids != initial_board.pass_ids_by_author:
        raise V3ContractError("commit INITIAL pass references mismatch")
    if commit.refine_pass_ids != refine_board.pass_ids_by_author:
        raise V3ContractError("commit REFINE pass references mismatch")
    if commit.consolidation_id != consolidation.consolidation_id:
        raise V3ContractError("commit consolidation_id mismatch")
    if set(author for author, _ in commit.final_core_state_leaf_ids) != online_set:
        raise V3ContractError(
            "commit final state-leaf authors do not match online population"
        )
    if commit.invocation_request_ids != consolidation.invocation_request_ids:
        raise V3ContractError(
            "commit invocation requests do not match consolidation"
        )


__all__ = [
    "PROTOCOL_VERSION",
    "PROTOCOL_NAME",
    "MAX_PROTOCOL_INT",
    "MAX_JSON_DEPTH",
    "MAX_JSON_NODES",
    "MAX_COLLECTION_ITEMS",
    "MAX_CONTEXT_JSON_NODES",
    "MAX_ONLINE_CORES",
    "V3ContractError",
    "V3PopulationError",
    "V3PhaseError",
    "V3SerializationError",
    "V3BoardError",
    "V3SoulError",
    "TickPhasePlan",
    "ReadCycleManifest",
    "SoulTransitionRecord",
    "SoulTransitionDispositionRecord",
    "CorePassRecord",
    "ProposalBoardManifest",
    "ConsolidationRecord",
    "TickCommitRecordV3",
    "ArtifactRejectionRecord",
    "validate_pass_against_read_cycle",
    "validate_consolidation_against_plan",
    "validate_tick_artifact_graph",
]
