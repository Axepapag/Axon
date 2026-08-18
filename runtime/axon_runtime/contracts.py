"""Canonical contracts for Axon's journal-backed multi-core runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from runtime.field import FieldDelta, canonical_json_bytes, canonical_sha256


ACTIVE_ROLES = frozenset({"proposer", "consolidator", "sleeper"})
RUNTIME_SCHEMA_VERSION = "axon-runtime-v1"


class RuntimeContractError(ValueError):
    """Base class for a rejected runtime contract."""


class IdentityContractError(RuntimeContractError):
    """A core identity or clone population is invalid."""


class RoleContractError(RuntimeContractError):
    """A role ring or assignment is invalid."""


class SerializationContractError(RuntimeContractError):
    """A serialized canonical object is malformed."""


class StoreContractError(RuntimeContractError):
    """A durable-store operation violated the runtime contract."""


class AlreadyInitializedError(StoreContractError):
    """The runtime store already has a genesis head."""


class NotInitializedError(StoreContractError):
    """The runtime store has no genesis head."""


class StaleHeadError(StoreContractError):
    """A transaction was authored against a stale generation or field."""


class RoleOwnershipError(StoreContractError):
    """A proposal, decision, or update was authored by the wrong role."""


class CoreStateUpdateError(StoreContractError):
    """A core-state update crossed identity or role boundaries."""


class JournalIntegrityError(StoreContractError):
    """The insert-only journal or one of its projections is inconsistent."""


class IdempotencyConflictError(StoreContractError):
    """An idempotency key was reused for different canonical content."""


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeContractError(f"{label} must be a non-empty string")
    return value


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeContractError(f"{label} must be a non-negative integer")
    return value


def _sha256(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise RuntimeContractError(f"{label} must be a 64-character SHA-256")
    return value.lower()


class _FrozenJsonDict(dict[str, Any]):
    """A JSON-serializable mapping that cannot drift after ID derivation."""

    @staticmethod
    def _immutable(*_: Any, **__: Any) -> None:
        raise TypeError("canonical JSON mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenJsonDict(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _json_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeContractError(f"{label} must be a mapping")
    try:
        encoded = canonical_json_bytes(dict(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeContractError(f"{label} must be JSON-safe") from exc
    import json

    decoded = json.loads(encoded.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeContractError(f"{label} must encode a JSON object")
    return _freeze_json(decoded)


def _normalized_strings(
    values: Sequence[str],
    label: str,
    *,
    sort: bool = True,
) -> tuple[str, ...]:
    normalized = tuple(_nonempty(str(value), label) for value in values)
    if len(normalized) != len(set(normalized)):
        raise RuntimeContractError(f"{label} cannot contain duplicates")
    return tuple(sorted(normalized)) if sort else normalized


def _derived_id(prefix: str, value: Mapping[str, Any]) -> str:
    return f"{prefix}-{canonical_sha256(dict(value))}"


@dataclass(frozen=True, slots=True)
class CoreIdentity:
    """One logical core identity over immutable, potentially shared weights."""

    core_id: str
    display_name: str
    base_checkpoint_path: str
    base_checkpoint_sha256: str
    model_id: str
    core_state_sha256: str
    lineage: tuple[str, ...]
    soul_id: str
    adapter_namespace: str
    enabled: bool = True
    role_capabilities: tuple[str, ...] = (
        "proposer",
        "consolidator",
        "sleeper",
    )
    parent_core_id: str | None = None
    identity_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "core_id",
            "display_name",
            "base_checkpoint_path",
            "model_id",
            "soul_id",
            "adapter_namespace",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "base_checkpoint_sha256",
            _sha256(self.base_checkpoint_sha256, "base_checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "core_state_sha256",
            _sha256(self.core_state_sha256, "core_state_sha256"),
        )
        if not isinstance(self.enabled, bool):
            raise IdentityContractError("enabled must be boolean")
        lineage = _normalized_strings(self.lineage, "lineage", sort=False)
        if not lineage:
            raise IdentityContractError("lineage must contain at least one entry")
        object.__setattr__(self, "lineage", lineage)
        capabilities = _normalized_strings(
            self.role_capabilities,
            "role_capabilities",
        )
        unknown = set(capabilities) - ACTIVE_ROLES
        if unknown:
            raise IdentityContractError(
                f"unknown role capabilities: {sorted(unknown)}"
            )
        object.__setattr__(self, "role_capabilities", capabilities)
        if self.parent_core_id is not None:
            parent = _nonempty(self.parent_core_id, "parent_core_id")
            if parent == self.core_id:
                raise IdentityContractError("a core cannot parent itself")
            object.__setattr__(self, "parent_core_id", parent)
        object.__setattr__(
            self,
            "identity_hash",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-core-identity-v1",
            "core_id": self.core_id,
            "display_name": self.display_name,
            "base_checkpoint_path": self.base_checkpoint_path,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "model_id": self.model_id,
            "core_state_sha256": self.core_state_sha256,
            "lineage": list(self.lineage),
            "parent_core_id": self.parent_core_id,
            "soul_id": self.soul_id,
            "adapter_namespace": self.adapter_namespace,
            "enabled": self.enabled,
            "role_capabilities": list(self.role_capabilities),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["identity_hash"] = self.identity_hash
        return value


def validate_identity_population(identities: Sequence[CoreIdentity]) -> None:
    values = tuple(identities)
    if not values:
        raise IdentityContractError("at least one core identity is required")
    if not all(isinstance(value, CoreIdentity) for value in values):
        raise IdentityContractError("identities must contain CoreIdentity values")
    for label, attributes in (
        ("core_id", [value.core_id for value in values]),
        ("soul_id", [value.soul_id for value in values]),
        (
            "adapter_namespace",
            [value.adapter_namespace for value in values],
        ),
    ):
        if len(attributes) != len(set(attributes)):
            raise IdentityContractError(f"{label} values must be unique")
    core_ids = {value.core_id for value in values}
    for value in values:
        if value.parent_core_id is not None and value.parent_core_id not in core_ids:
            raise IdentityContractError(
                f"parent core {value.parent_core_id!r} is not present"
            )


@dataclass(frozen=True, slots=True)
class RoleAssignment:
    tick_seq: int
    proposer_core_id: str
    consolidator_core_id: str
    sleeper_core_id: str
    standby_core_ids: tuple[str, ...] = ()
    assignment_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tick_seq",
            _exact_nonnegative_int(self.tick_seq, "tick_seq"),
        )
        active = (
            _nonempty(self.proposer_core_id, "proposer_core_id"),
            _nonempty(self.consolidator_core_id, "consolidator_core_id"),
            _nonempty(self.sleeper_core_id, "sleeper_core_id"),
        )
        if len(set(active)) != 3:
            raise RoleContractError("active role core IDs must be unique")
        standby = _normalized_strings(
            self.standby_core_ids,
            "standby_core_ids",
            sort=False,
        )
        if set(active) & set(standby):
            raise RoleContractError("standby cores cannot hold an active role")
        object.__setattr__(self, "standby_core_ids", standby)
        object.__setattr__(
            self,
            "assignment_hash",
            canonical_sha256(self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-role-assignment-v1",
            "tick_seq": self.tick_seq,
            "proposer_core_id": self.proposer_core_id,
            "consolidator_core_id": self.consolidator_core_id,
            "sleeper_core_id": self.sleeper_core_id,
            "standby_core_ids": list(self.standby_core_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["assignment_hash"] = self.assignment_hash
        return value


@dataclass(frozen=True, slots=True)
class CoreStateManifest:
    """Durable per-core state, including its independent field-view cursor."""

    core_id: str
    soul_id: str
    model_id: str
    core_state_sha256: str
    generation: int
    tick_seq: int
    committed_field_id: str
    soul_state_sha256: str
    cursor_state_sha256: str
    compressor_state_sha256: str | None = None
    adapter_set_sha256: str | None = None
    rng_state_sha256: str | None = None
    parent_manifest_id: str | None = None
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("core_id", "soul_id", "model_id", "committed_field_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in ("generation", "tick_seq"):
            object.__setattr__(
                self,
                name,
                _exact_nonnegative_int(getattr(self, name), name),
            )
        for name in (
            "core_state_sha256",
            "soul_state_sha256",
            "cursor_state_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name),
            )
        for name in (
            "compressor_state_sha256",
            "adapter_set_sha256",
            "rng_state_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name, optional=True),
            )
        if self.parent_manifest_id is not None:
            object.__setattr__(
                self,
                "parent_manifest_id",
                _nonempty(self.parent_manifest_id, "parent_manifest_id"),
            )
        object.__setattr__(
            self,
            "manifest_id",
            _derived_id("core-state", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-core-state-manifest-v2",
            "core_id": self.core_id,
            "soul_id": self.soul_id,
            "model_id": self.model_id,
            "core_state_sha256": self.core_state_sha256,
            "generation": self.generation,
            "tick_seq": self.tick_seq,
            "committed_field_id": self.committed_field_id,
            "soul_state_sha256": self.soul_state_sha256,
            "cursor_state_sha256": self.cursor_state_sha256,
            "compressor_state_sha256": self.compressor_state_sha256,
            "adapter_set_sha256": self.adapter_set_sha256,
            "rng_state_sha256": self.rng_state_sha256,
            "parent_manifest_id": self.parent_manifest_id,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["manifest_id"] = self.manifest_id
        return value


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    """Proposer output authored against the exact same-tick working field W."""

    tick_seq: int
    assignment_hash: str
    proposer_core_id: str
    system_update_id: str
    base_field_id: str
    base_tick_id: int
    delta: FieldDelta | None = None
    evidence: tuple[str, ...] = ()
    proposal_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tick_seq",
            _exact_nonnegative_int(self.tick_seq, "tick_seq"),
        )
        object.__setattr__(
            self,
            "base_tick_id",
            _exact_nonnegative_int(self.base_tick_id, "base_tick_id"),
        )
        object.__setattr__(
            self,
            "assignment_hash",
            _sha256(self.assignment_hash, "assignment_hash"),
        )
        object.__setattr__(
            self,
            "system_update_id",
            _sha256(self.system_update_id, "system_update_id"),
        )
        for name in ("proposer_core_id", "base_field_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if self.delta is not None and not isinstance(self.delta, FieldDelta):
            raise RuntimeContractError("delta must be FieldDelta or None")
        object.__setattr__(
            self,
            "evidence",
            _normalized_strings(self.evidence, "evidence"),
        )
        object.__setattr__(
            self,
            "proposal_id",
            _derived_id("proposal", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-proposal-record-v2",
            "tick_seq": self.tick_seq,
            "assignment_hash": self.assignment_hash,
            "proposer_core_id": self.proposer_core_id,
            "system_update_id": self.system_update_id,
            "base_field_id": self.base_field_id,
            "base_tick_id": self.base_tick_id,
            "delta": (
                None if self.delta is None else self.delta.to_canonical_dict()
            ),
            "evidence": list(self.evidence),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["proposal_id"] = self.proposal_id
        return value


@dataclass(frozen=True, slots=True)
class ConsolidationDecision:
    """Consolidator decision whose base and committed delta both target W."""

    tick_seq: int
    assignment_hash: str
    consolidator_core_id: str
    proposal_id: str
    base_field_id: str
    accepted: bool
    output_field_id: str
    committed_delta: FieldDelta | None = None
    reason: str = ""
    committed_delta_id: str | None = field(init=False)
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tick_seq",
            _exact_nonnegative_int(self.tick_seq, "tick_seq"),
        )
        object.__setattr__(
            self,
            "assignment_hash",
            _sha256(self.assignment_hash, "assignment_hash"),
        )
        for name in (
            "consolidator_core_id",
            "proposal_id",
            "base_field_id",
            "output_field_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if not isinstance(self.accepted, bool):
            raise RuntimeContractError("accepted must be boolean")
        if self.committed_delta is not None and not isinstance(
            self.committed_delta,
            FieldDelta,
        ):
            raise RuntimeContractError(
                "committed_delta must be FieldDelta or None"
            )
        if not isinstance(self.reason, str):
            raise RuntimeContractError("reason must be a string")
        if not self.accepted and self.committed_delta is not None:
            raise RuntimeContractError(
                "a rejected decision cannot commit a delta"
            )
        object.__setattr__(
            self,
            "committed_delta_id",
            (
                None
                if self.committed_delta is None
                else self.committed_delta.delta_id
            ),
        )
        object.__setattr__(
            self,
            "decision_id",
            _derived_id("decision", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-consolidation-decision-v2",
            "tick_seq": self.tick_seq,
            "assignment_hash": self.assignment_hash,
            "consolidator_core_id": self.consolidator_core_id,
            "proposal_id": self.proposal_id,
            "base_field_id": self.base_field_id,
            "accepted": self.accepted,
            "output_field_id": self.output_field_id,
            "committed_delta": (
                None
                if self.committed_delta is None
                else self.committed_delta.to_canonical_dict()
            ),
            "committed_delta_id": self.committed_delta_id,
            "reason": self.reason,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["decision_id"] = self.decision_id
        return value


@dataclass(frozen=True, slots=True)
class ProjectionManifest:
    generation: int
    tick_seq: int
    field_id: str
    field_hash: str
    core_state_manifest_ids: tuple[tuple[str, str], ...]
    role_index: int
    role_assignment_hash: str
    parent_projection_id: str | None = None
    projection_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("generation", "tick_seq", "role_index"):
            object.__setattr__(
                self,
                name,
                _exact_nonnegative_int(getattr(self, name), name),
            )
        object.__setattr__(self, "field_id", _nonempty(self.field_id, "field_id"))
        object.__setattr__(
            self,
            "field_hash",
            _sha256(self.field_hash, "field_hash"),
        )
        object.__setattr__(
            self,
            "role_assignment_hash",
            _sha256(self.role_assignment_hash, "role_assignment_hash"),
        )
        pairs = tuple(
            (
                _nonempty(core_id, "core_state core_id"),
                _nonempty(manifest_id, "core_state manifest_id"),
            )
            for core_id, manifest_id in self.core_state_manifest_ids
        )
        if len({core_id for core_id, _ in pairs}) != len(pairs):
            raise RuntimeContractError(
                "core_state_manifest_ids contain duplicate core IDs"
            )
        object.__setattr__(
            self,
            "core_state_manifest_ids",
            tuple(sorted(pairs)),
        )
        if self.parent_projection_id is not None:
            object.__setattr__(
                self,
                "parent_projection_id",
                _nonempty(self.parent_projection_id, "parent_projection_id"),
            )
        object.__setattr__(
            self,
            "projection_id",
            _derived_id("projection", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-projection-manifest-v1",
            "generation": self.generation,
            "tick_seq": self.tick_seq,
            "field_id": self.field_id,
            "field_hash": self.field_hash,
            "core_state_manifest_ids": {
                core_id: manifest_id
                for core_id, manifest_id in self.core_state_manifest_ids
            },
            "role_index": self.role_index,
            "role_assignment_hash": self.role_assignment_hash,
            "parent_projection_id": self.parent_projection_id,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["projection_id"] = self.projection_id
        return value


@dataclass(frozen=True, slots=True)
class ToolRequestRecord:
    """Outbox request grounded in the exact working field W."""

    tick_seq: int
    requester_core_id: str
    base_field_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    parent_proposal_id: str
    request_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tick_seq",
            _exact_nonnegative_int(self.tick_seq, "tick_seq"),
        )
        for name in (
            "requester_core_id",
            "base_field_id",
            "tool_name",
            "parent_proposal_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "arguments",
            _json_mapping(self.arguments, "arguments"),
        )
        object.__setattr__(
            self,
            "request_id",
            _derived_id("tool-request", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-tool-request-v2",
            "tick_seq": self.tick_seq,
            "requester_core_id": self.requester_core_id,
            "base_field_id": self.base_field_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "parent_proposal_id": self.parent_proposal_id,
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["request_id"] = self.request_id
        return value


@dataclass(frozen=True, slots=True)
class ToolResultRecord:
    idempotency_key: str
    request_id: str
    status: str
    payload: Mapping[str, Any]
    parent_ids: tuple[str, ...] = ()
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("idempotency_key", "request_id", "status"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "payload",
            _json_mapping(self.payload, "payload"),
        )
        object.__setattr__(
            self,
            "parent_ids",
            _normalized_strings(self.parent_ids, "parent_ids"),
        )
        object.__setattr__(
            self,
            "result_id",
            _derived_id("tool-result", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-tool-result-v1",
            "idempotency_key": self.idempotency_key,
            "request_id": self.request_id,
            "status": self.status,
            "payload": dict(self.payload),
            "parent_ids": list(self.parent_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["result_id"] = self.result_id
        return value


@dataclass(frozen=True, slots=True)
class SourceEvent:
    idempotency_key: str
    source_kind: str
    source_ref: str
    payload: Mapping[str, Any]
    parent_ids: tuple[str, ...] = ()
    source_event_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("idempotency_key", "source_kind", "source_ref"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "payload",
            _json_mapping(self.payload, "payload"),
        )
        object.__setattr__(
            self,
            "parent_ids",
            _normalized_strings(self.parent_ids, "parent_ids"),
        )
        object.__setattr__(
            self,
            "source_event_id",
            _derived_id("source-event", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-source-event-v1",
            "idempotency_key": self.idempotency_key,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "payload": dict(self.payload),
            "parent_ids": list(self.parent_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["source_event_id"] = self.source_event_id
        return value


@dataclass(frozen=True, slots=True)
class SpanLifecycleEvent:
    action: str
    span_id: str
    region: str
    field_id: str
    payload: Mapping[str, Any]
    parent_ids: tuple[str, ...] = ()
    lifecycle_event_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("action", "span_id", "region", "field_id"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "payload",
            _json_mapping(self.payload, "payload"),
        )
        object.__setattr__(
            self,
            "parent_ids",
            _normalized_strings(self.parent_ids, "parent_ids"),
        )
        object.__setattr__(
            self,
            "lifecycle_event_id",
            _derived_id("span-lifecycle", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-span-lifecycle-v1",
            "action": self.action,
            "span_id": self.span_id,
            "region": self.region,
            "field_id": self.field_id,
            "payload": dict(self.payload),
            "parent_ids": list(self.parent_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["lifecycle_event_id"] = self.lifecycle_event_id
        return value


@dataclass(frozen=True, slots=True)
class TickCommitRecord:
    """Compact H -> U -> W -> delta -> F transaction receipt."""

    generation: int
    tick_seq: int
    assignment_hash: str
    proposal_id: str
    decision_id: str
    input_field_id: str
    system_update_id: str
    working_field_id: str
    field_transaction_audit_id: str
    output_field_id: str
    updated_core_state_manifest_ids: tuple[str, ...]
    projection_id: str
    tool_request_ids: tuple[str, ...] = ()
    commit_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("generation", "tick_seq"):
            object.__setattr__(
                self,
                name,
                _exact_nonnegative_int(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "assignment_hash",
            _sha256(self.assignment_hash, "assignment_hash"),
        )
        for name in (
            "system_update_id",
            "working_field_id",
            "field_transaction_audit_id",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), name),
            )
        for name in (
            "proposal_id",
            "decision_id",
            "input_field_id",
            "output_field_id",
            "projection_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        object.__setattr__(
            self,
            "updated_core_state_manifest_ids",
            _normalized_strings(
                self.updated_core_state_manifest_ids,
                "updated_core_state_manifest_ids",
            ),
        )
        object.__setattr__(
            self,
            "tool_request_ids",
            _normalized_strings(self.tool_request_ids, "tool_request_ids"),
        )
        object.__setattr__(
            self,
            "commit_id",
            _derived_id("tick-commit", self.to_canonical_dict()),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": "axon-tick-commit-v2",
            "generation": self.generation,
            "tick_seq": self.tick_seq,
            "assignment_hash": self.assignment_hash,
            "proposal_id": self.proposal_id,
            "decision_id": self.decision_id,
            "input_field_id": self.input_field_id,
            "system_update_id": self.system_update_id,
            "working_field_id": self.working_field_id,
            "field_transaction_audit_id": self.field_transaction_audit_id,
            "output_field_id": self.output_field_id,
            "updated_core_state_manifest_ids": list(
                self.updated_core_state_manifest_ids
            ),
            "projection_id": self.projection_id,
            "tool_request_ids": list(self.tool_request_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.to_canonical_dict()
        value["commit_id"] = self.commit_id
        return value


@dataclass(frozen=True, slots=True)
class RuntimeHead:
    generation: int
    tick_seq: int
    snapshot: Any
    role_index: int
    role_assignment_hash: str
    projection_manifest_id: str
    core_state_manifests: tuple[CoreStateManifest, ...]


__all__ = [
    "ACTIVE_ROLES",
    "RUNTIME_SCHEMA_VERSION",
    "RuntimeContractError",
    "IdentityContractError",
    "RoleContractError",
    "SerializationContractError",
    "StoreContractError",
    "AlreadyInitializedError",
    "NotInitializedError",
    "StaleHeadError",
    "RoleOwnershipError",
    "CoreStateUpdateError",
    "JournalIntegrityError",
    "IdempotencyConflictError",
    "CoreIdentity",
    "validate_identity_population",
    "RoleAssignment",
    "CoreStateManifest",
    "ProposalRecord",
    "ConsolidationDecision",
    "ProjectionManifest",
    "ToolRequestRecord",
    "ToolResultRecord",
    "SourceEvent",
    "SpanLifecycleEvent",
    "TickCommitRecord",
    "RuntimeHead",
]
