"""Governed activation and rollback contracts for Trainer-owned parameter generations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from runtime.field import canonical_sha256

ACTIVE_GENERATION_POINTER_SCHEMA = "axon-trainer-active-generation-pointer-v1"
GENERATION_SNAPSHOT_SCHEMA = "axon-trainer-generation-snapshot-v1"
PARAMETER_ACTIVATION_SCHEMA = "axon-trainer-parameter-activation-v1"
PARAMETER_ROLLBACK_SCHEMA = "axon-trainer-parameter-rollback-v1"


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class GenerationSnapshotRecord:
    module_id: str
    generation_id: str
    parameter_manifest_id: str
    artifact_relpath: str
    artifact_sha256: str
    artifact_bytes: int
    source_inventory_id: str
    previous_pointer_id: str | None = None
    snapshot_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "module_id",
            "generation_id",
            "parameter_manifest_id",
            "artifact_relpath",
            "artifact_sha256",
            "source_inventory_id",
        ):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        if len(self.artifact_sha256) != 64:
            raise ValueError("artifact_sha256 must be a SHA256 hex digest")
        object.__setattr__(self, "artifact_sha256", self.artifact_sha256.lower())
        if isinstance(self.artifact_bytes, bool) or not isinstance(self.artifact_bytes, int) or self.artifact_bytes <= 0:
            raise ValueError("artifact_bytes must be positive")
        if self.previous_pointer_id is not None:
            object.__setattr__(self, "previous_pointer_id", _required(self.previous_pointer_id, "previous_pointer_id"))
        object.__setattr__(self, "snapshot_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": GENERATION_SNAPSHOT_SCHEMA,
            "module_id": self.module_id,
            "generation_id": self.generation_id,
            "parameter_manifest_id": self.parameter_manifest_id,
            "artifact_relpath": self.artifact_relpath,
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
            "source_inventory_id": self.source_inventory_id,
            "previous_pointer_id": self.previous_pointer_id,
        }
        if include_id:
            value["snapshot_id"] = self.snapshot_id
        return value


@dataclass(frozen=True, slots=True)
class ActiveGenerationPointer:
    module_id: str
    generation_id: str
    parameter_manifest_id: str
    artifact_relpath: str
    artifact_sha256: str
    source_kind: str
    source_artifact_id: str
    previous_generation_id: str | None = None
    previous_pointer_id: str | None = None
    rollback_snapshot_id: str | None = None
    pointer_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "module_id",
            "generation_id",
            "parameter_manifest_id",
            "artifact_relpath",
            "artifact_sha256",
            "source_kind",
            "source_artifact_id",
        ):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        if len(self.artifact_sha256) != 64:
            raise ValueError("artifact_sha256 must be a SHA256 hex digest")
        object.__setattr__(self, "artifact_sha256", self.artifact_sha256.lower())
        for label in ("previous_generation_id", "previous_pointer_id", "rollback_snapshot_id"):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(self, label, _required(value, label))
        object.__setattr__(self, "pointer_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": ACTIVE_GENERATION_POINTER_SCHEMA,
            "module_id": self.module_id,
            "generation_id": self.generation_id,
            "parameter_manifest_id": self.parameter_manifest_id,
            "artifact_relpath": self.artifact_relpath,
            "artifact_sha256": self.artifact_sha256,
            "source_kind": self.source_kind,
            "source_artifact_id": self.source_artifact_id,
            "previous_generation_id": self.previous_generation_id,
            "previous_pointer_id": self.previous_pointer_id,
            "rollback_snapshot_id": self.rollback_snapshot_id,
        }
        if include_id:
            value["pointer_id"] = self.pointer_id
        return value


@dataclass(frozen=True, slots=True)
class ParameterActivationReceipt:
    module_id: str
    base_generation_id: str
    activated_generation_id: str
    base_inventory_id: str
    proposal_id: str
    gate_decision_id: str
    checkpoint_id: str
    rollback_snapshot_id: str
    active_pointer_id: str
    before_manifest_id: str
    after_manifest_id: str
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "module_id",
            "base_generation_id",
            "activated_generation_id",
            "base_inventory_id",
            "proposal_id",
            "gate_decision_id",
            "checkpoint_id",
            "rollback_snapshot_id",
            "active_pointer_id",
            "before_manifest_id",
            "after_manifest_id",
        ):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        if self.base_generation_id == self.activated_generation_id:
            raise ValueError("activation must advance generation")
        object.__setattr__(self, "receipt_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PARAMETER_ACTIVATION_SCHEMA,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "activated_generation_id": self.activated_generation_id,
            "base_inventory_id": self.base_inventory_id,
            "proposal_id": self.proposal_id,
            "gate_decision_id": self.gate_decision_id,
            "checkpoint_id": self.checkpoint_id,
            "rollback_snapshot_id": self.rollback_snapshot_id,
            "active_pointer_id": self.active_pointer_id,
            "before_manifest_id": self.before_manifest_id,
            "after_manifest_id": self.after_manifest_id,
        }
        if include_id:
            value["receipt_id"] = self.receipt_id
        return value


@dataclass(frozen=True, slots=True)
class ParameterRollbackReceipt:
    module_id: str
    from_generation_id: str
    restored_generation_id: str
    from_pointer_id: str
    rollback_snapshot_id: str
    active_pointer_id: str
    before_manifest_id: str
    after_manifest_id: str
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "module_id",
            "from_generation_id",
            "restored_generation_id",
            "from_pointer_id",
            "rollback_snapshot_id",
            "active_pointer_id",
            "before_manifest_id",
            "after_manifest_id",
        ):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        if self.from_generation_id == self.restored_generation_id:
            raise ValueError("rollback must change generation")
        object.__setattr__(self, "receipt_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PARAMETER_ROLLBACK_SCHEMA,
            "module_id": self.module_id,
            "from_generation_id": self.from_generation_id,
            "restored_generation_id": self.restored_generation_id,
            "from_pointer_id": self.from_pointer_id,
            "rollback_snapshot_id": self.rollback_snapshot_id,
            "active_pointer_id": self.active_pointer_id,
            "before_manifest_id": self.before_manifest_id,
            "after_manifest_id": self.after_manifest_id,
        }
        if include_id:
            value["receipt_id"] = self.receipt_id
        return value


__all__ = [
    "ACTIVE_GENERATION_POINTER_SCHEMA",
    "GENERATION_SNAPSHOT_SCHEMA",
    "PARAMETER_ACTIVATION_SCHEMA",
    "PARAMETER_ROLLBACK_SCHEMA",
    "GenerationSnapshotRecord",
    "ActiveGenerationPointer",
    "ParameterActivationReceipt",
    "ParameterRollbackReceipt",
]
