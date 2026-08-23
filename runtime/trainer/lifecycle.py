"""Candidate parameter-generation lifecycle for Axon's Trainer organ.

Candidate generations are isolated from the live organism.  The lifecycle is
append-only: status changes are events, never in-place rewrites of history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from runtime.field import canonical_sha256

CANDIDATE_LIFECYCLE_SCHEMA = "axon-trainer-candidate-lifecycle-event-v1"
CANDIDATE_CHECKPOINT_SCHEMA = "axon-trainer-candidate-checkpoint-v1"
OPTIMIZATION_STEP_SCHEMA = "axon-trainer-optimization-step-v1"


class CandidateStatus(str, Enum):
    PREPARED = "prepared"
    RUNNING = "running"
    COMPLETED = "completed"
    REJECTED = "rejected"
    GATE_PASSED = "gate_passed"
    PROMOTION_PROPOSED = "promotion_proposed"
    PROMOTED = "promoted"
    RETIRED = "retired"


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class CandidateLifecycleEvent:
    module_id: str
    base_generation_id: str
    candidate_generation_id: str
    plan_id: str
    authorization_id: str
    status: CandidateStatus | str
    step: int
    previous_event_id: str | None = None
    checkpoint_id: str | None = None
    gate_decision_id: str | None = None
    reason: str | None = None
    event_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "module_id",
            "base_generation_id",
            "candidate_generation_id",
            "plan_id",
            "authorization_id",
        ):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        if self.base_generation_id == self.candidate_generation_id:
            raise ValueError("candidate generation must differ from base generation")
        status = self.status if isinstance(self.status, CandidateStatus) else CandidateStatus(self.status)
        object.__setattr__(self, "status", status)
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        for label in ("previous_event_id", "checkpoint_id", "gate_decision_id"):
            value = getattr(self, label)
            if value is not None:
                object.__setattr__(self, label, _required(value, label))
        if self.reason is not None:
            object.__setattr__(self, "reason", _required(self.reason, "reason"))
        object.__setattr__(self, "event_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": CANDIDATE_LIFECYCLE_SCHEMA,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "plan_id": self.plan_id,
            "authorization_id": self.authorization_id,
            "status": self.status.value,
            "step": self.step,
            "previous_event_id": self.previous_event_id,
            "checkpoint_id": self.checkpoint_id,
            "gate_decision_id": self.gate_decision_id,
            "reason": self.reason,
        }
        if include_id:
            value["event_id"] = self.event_id
        return value


@dataclass(frozen=True, slots=True)
class CandidateCheckpointRecord:
    module_id: str
    base_generation_id: str
    candidate_generation_id: str
    plan_id: str
    authorization_id: str
    step: int
    parameter_manifest_id: str
    artifact_relpath: str
    artifact_sha256: str
    artifact_bytes: int
    optimizer_included: bool
    previous_checkpoint_id: str | None = None
    checkpoint_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "module_id",
            "base_generation_id",
            "candidate_generation_id",
            "plan_id",
            "authorization_id",
            "parameter_manifest_id",
            "artifact_relpath",
            "artifact_sha256",
        ):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        if self.base_generation_id == self.candidate_generation_id:
            raise ValueError("candidate generation must differ from base generation")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        if len(self.artifact_sha256) != 64:
            raise ValueError("artifact_sha256 must be a SHA256 hex digest")
        object.__setattr__(self, "artifact_sha256", self.artifact_sha256.lower())
        if isinstance(self.artifact_bytes, bool) or not isinstance(self.artifact_bytes, int) or self.artifact_bytes <= 0:
            raise ValueError("artifact_bytes must be positive")
        if self.previous_checkpoint_id is not None:
            object.__setattr__(self, "previous_checkpoint_id", _required(self.previous_checkpoint_id, "previous_checkpoint_id"))
        object.__setattr__(self, "checkpoint_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": CANDIDATE_CHECKPOINT_SCHEMA,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "plan_id": self.plan_id,
            "authorization_id": self.authorization_id,
            "step": self.step,
            "parameter_manifest_id": self.parameter_manifest_id,
            "artifact_relpath": self.artifact_relpath,
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
            "optimizer_included": bool(self.optimizer_included),
            "previous_checkpoint_id": self.previous_checkpoint_id,
        }
        if include_id:
            value["checkpoint_id"] = self.checkpoint_id
        return value


@dataclass(frozen=True, slots=True)
class OptimizationStepReceipt:
    module_id: str
    candidate_generation_id: str
    plan_id: str
    authorization_id: str
    step: int
    loss: float
    gradient_l2: float
    gradient_clip_norm: float | None
    telemetry_frame_id: str
    changed_tensor_names: tuple[str, ...]
    unchanged_tensor_names: tuple[str, ...]
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in ("module_id", "candidate_generation_id", "plan_id", "authorization_id", "telemetry_frame_id"):
            object.__setattr__(self, label, _required(getattr(self, label), label))
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step <= 0:
            raise ValueError("step must be a positive integer")
        loss = float(self.loss)
        grad = float(self.gradient_l2)
        if not (-float("inf") < loss < float("inf")):
            raise ValueError("loss must be finite")
        if not (0.0 <= grad < float("inf")):
            raise ValueError("gradient_l2 must be finite and non-negative")
        object.__setattr__(self, "loss", loss)
        object.__setattr__(self, "gradient_l2", grad)
        if self.gradient_clip_norm is not None:
            clip = float(self.gradient_clip_norm)
            if not (0.0 < clip < float("inf")):
                raise ValueError("gradient_clip_norm must be positive and finite")
            object.__setattr__(self, "gradient_clip_norm", clip)
        changed = tuple(sorted(set(_required(str(item), "changed tensor name") for item in self.changed_tensor_names)))
        unchanged = tuple(sorted(set(_required(str(item), "unchanged tensor name") for item in self.unchanged_tensor_names)))
        if set(changed) & set(unchanged):
            raise ValueError("changed and unchanged tensor names must be disjoint")
        object.__setattr__(self, "changed_tensor_names", changed)
        object.__setattr__(self, "unchanged_tensor_names", unchanged)
        object.__setattr__(self, "receipt_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": OPTIMIZATION_STEP_SCHEMA,
            "module_id": self.module_id,
            "candidate_generation_id": self.candidate_generation_id,
            "plan_id": self.plan_id,
            "authorization_id": self.authorization_id,
            "step": self.step,
            "loss": self.loss,
            "gradient_l2": self.gradient_l2,
            "gradient_clip_norm": self.gradient_clip_norm,
            "telemetry_frame_id": self.telemetry_frame_id,
            "changed_tensor_names": list(self.changed_tensor_names),
            "unchanged_tensor_names": list(self.unchanged_tensor_names),
        }
        if include_id:
            value["receipt_id"] = self.receipt_id
        return value


__all__ = [
    "CANDIDATE_LIFECYCLE_SCHEMA",
    "CANDIDATE_CHECKPOINT_SCHEMA",
    "OPTIMIZATION_STEP_SCHEMA",
    "CandidateStatus",
    "CandidateLifecycleEvent",
    "CandidateCheckpointRecord",
    "OptimizationStepReceipt",
]
