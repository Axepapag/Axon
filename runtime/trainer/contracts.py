"""Governed parameter-state contracts for Axon's Trainer organ.

The Trainer is to parameter state what the Heart is to canonical Shared Field
state: one deterministic authority boundary around mutation, lineage,
observability, and promotion.  Transformer advisers may live inside the Trainer
later, but they never gain direct tensor-write authority from these contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from runtime.field import canonical_sha256

TRAINER_PARAMETER_SCHEMA = "axon-trainer-parameter-control-v1"
PARAMETER_MODULE_SCHEMA = "axon-parameter-module-descriptor-v1"
PARAMETER_TENSOR_SCHEMA = "axon-parameter-tensor-record-v1"
PARAMETER_MANIFEST_SCHEMA = "axon-parameter-module-manifest-v1"
PARAMETER_INVENTORY_SCHEMA = "axon-parameter-inventory-v1"
PARAMETER_GRANT_SCHEMA = "axon-parameter-mutation-grant-v1"
PARAMETER_PLAN_SCHEMA = "axon-parameter-mutation-plan-v1"
PARAMETER_PLAN_V2_SCHEMA = "axon-parameter-mutation-plan-v2"
PARAMETER_PROMOTION_SCHEMA = "axon-parameter-promotion-proposal-v1"


class OrganKind(str, Enum):
    """Parameter-bearing organism roles known to the Trainer."""

    REASONING_CORE = "reasoning_core"
    SEMANTIC_CORE = "semantic_core"
    TRAINER_CORE = "trainer_core"
    HEART_TRANSLATION_CORE = "heart_translation_core"
    REASONING_ADAPTER = "reasoning_adapter"
    SEMANTIC_ADAPTER = "semantic_adapter"
    TRAINER_ADAPTER = "trainer_adapter"
    HEART_TRANSLATION_ADAPTER = "heart_translation_adapter"
    OTHER = "other"


class ParameterMutationPolicy(str, Enum):
    """Maximum mutation authority of one grant."""

    SEALED = "sealed"
    FULL = "full"
    ADAPTER_ONLY = "adapter_only"
    EXPLICIT_NAMES = "explicit_names"


class TrainerCoreRole(str, Enum):
    """Potential advisory roles inside a future Trainer transformer ensemble.

    These are advisory roles, not authority classes.  The deterministic Trainer
    control plane remains the only parameter writer.
    """

    CURRICULUM = "curriculum"
    OPTIMIZER = "optimizer"
    EVALUATOR = "evaluator"
    FORGETTING_AUDITOR = "forgetting_auditor"
    PROMOTION_CRITIC = "promotion_critic"


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ParameterModuleDescriptor:
    module_id: str
    organ_kind: OrganKind | str
    generation_id: str
    architecture: str
    d_model: int | None = None
    trainer_core_role: TrainerCoreRole | str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "module_id", _nonempty(self.module_id, "module_id"))
        object.__setattr__(self, "generation_id", _nonempty(self.generation_id, "generation_id"))
        object.__setattr__(self, "architecture", _nonempty(self.architecture, "architecture"))
        kind = self.organ_kind if isinstance(self.organ_kind, OrganKind) else OrganKind(self.organ_kind)
        object.__setattr__(self, "organ_kind", kind)
        if self.d_model is not None and (
            isinstance(self.d_model, bool)
            or not isinstance(self.d_model, int)
            or self.d_model <= 0
        ):
            raise ValueError("d_model must be a positive integer or None")
        role = self.trainer_core_role
        if role is not None and not isinstance(role, TrainerCoreRole):
            role = TrainerCoreRole(role)
        if role is not None and kind not in {OrganKind.TRAINER_CORE, OrganKind.TRAINER_ADAPTER}:
            raise ValueError("trainer_core_role is valid only for Trainer cores/adapters")
        object.__setattr__(self, "trainer_core_role", role)
        tags = tuple(sorted(set(_nonempty(str(tag), "tag") for tag in self.tags)))
        object.__setattr__(self, "tags", tags)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": PARAMETER_MODULE_SCHEMA,
            "module_id": self.module_id,
            "organ_kind": self.organ_kind.value,
            "generation_id": self.generation_id,
            "architecture": self.architecture,
            "d_model": self.d_model,
            "trainer_core_role": None if self.trainer_core_role is None else self.trainer_core_role.value,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class ParameterTensorRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    requires_grad: bool
    value_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "tensor name"))
        shape = tuple(int(item) for item in self.shape)
        if any(item < 0 for item in shape):
            raise ValueError("tensor shape values must be non-negative")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", _nonempty(self.dtype, "dtype"))
        if isinstance(self.numel, bool) or not isinstance(self.numel, int) or self.numel < 0:
            raise ValueError("numel must be a non-negative integer")
        if self.value_sha256 is not None:
            digest = _nonempty(self.value_sha256, "value_sha256")
            if len(digest) != 64:
                raise ValueError("value_sha256 must be a 64-character SHA256 hex digest")
            object.__setattr__(self, "value_sha256", digest.lower())

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": PARAMETER_TENSOR_SCHEMA,
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "numel": self.numel,
            "requires_grad": bool(self.requires_grad),
            "value_sha256": self.value_sha256,
        }


@dataclass(frozen=True, slots=True)
class ParameterModuleManifest:
    descriptor: ParameterModuleDescriptor
    tensors: tuple[ParameterTensorRecord, ...]
    buffers: tuple[ParameterTensorRecord, ...] = ()
    parameter_count: int = field(init=False)
    trainable_parameter_count: int = field(init=False)
    buffer_count: int = field(init=False)
    buffer_numel: int = field(init=False)
    manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, ParameterModuleDescriptor):
            raise TypeError("descriptor must be ParameterModuleDescriptor")
        tensors = tuple(sorted(tuple(self.tensors), key=lambda item: item.name))
        if not all(isinstance(item, ParameterTensorRecord) for item in tensors):
            raise TypeError("tensors must contain only ParameterTensorRecord values")
        names = [item.name for item in tensors]
        if len(names) != len(set(names)):
            raise ValueError("duplicate parameter tensor name")
        object.__setattr__(self, "tensors", tensors)
        buffers = tuple(sorted(tuple(self.buffers), key=lambda item: item.name))
        if not all(isinstance(item, ParameterTensorRecord) for item in buffers):
            raise TypeError("buffers must contain only ParameterTensorRecord values")
        buffer_names = [item.name for item in buffers]
        if len(buffer_names) != len(set(buffer_names)):
            raise ValueError("duplicate persistent buffer name")
        if set(names) & set(buffer_names):
            raise ValueError("parameter and buffer names must be disjoint")
        if any(item.requires_grad for item in buffers):
            raise ValueError("persistent buffers cannot be marked trainable parameters")
        object.__setattr__(self, "buffers", buffers)
        total = sum(item.numel for item in tensors)
        trainable = sum(item.numel for item in tensors if item.requires_grad)
        object.__setattr__(self, "parameter_count", total)
        object.__setattr__(self, "trainable_parameter_count", trainable)
        object.__setattr__(self, "buffer_count", len(buffers))
        object.__setattr__(self, "buffer_numel", sum(item.numel for item in buffers))
        object.__setattr__(self, "manifest_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PARAMETER_MANIFEST_SCHEMA,
            "descriptor": self.descriptor.to_canonical_dict(),
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "buffer_count": self.buffer_count,
            "buffer_numel": self.buffer_numel,
            "tensors": [item.to_canonical_dict() for item in self.tensors],
            "buffers": [item.to_canonical_dict() for item in self.buffers],
        }
        if include_id:
            value["manifest_id"] = self.manifest_id
        return value


@dataclass(frozen=True, slots=True)
class ParameterInventory:
    manifests: tuple[ParameterModuleManifest, ...]
    expected_module_ids: tuple[str, ...]
    missing_module_ids: tuple[str, ...]
    inventory_id: str = field(init=False)

    def __post_init__(self) -> None:
        manifests = tuple(sorted(tuple(self.manifests), key=lambda item: item.descriptor.module_id))
        if not all(isinstance(item, ParameterModuleManifest) for item in manifests):
            raise TypeError("manifests must contain only ParameterModuleManifest values")
        ids = [item.descriptor.module_id for item in manifests]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate module manifest")
        expected = tuple(sorted(set(_nonempty(str(item), "expected module id") for item in self.expected_module_ids)))
        missing = tuple(sorted(set(_nonempty(str(item), "missing module id") for item in self.missing_module_ids)))
        if any(item not in expected for item in missing):
            raise ValueError("missing_module_ids must be a subset of expected_module_ids")
        object.__setattr__(self, "manifests", manifests)
        object.__setattr__(self, "expected_module_ids", expected)
        object.__setattr__(self, "missing_module_ids", missing)
        object.__setattr__(self, "inventory_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def complete(self) -> bool:
        return not self.missing_module_ids

    @property
    def parameter_count(self) -> int:
        return sum(item.parameter_count for item in self.manifests)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(item.trainable_parameter_count for item in self.manifests)

    @property
    def buffer_numel(self) -> int:
        return sum(item.buffer_numel for item in self.manifests)

    @property
    def state_numel(self) -> int:
        return self.parameter_count + self.buffer_numel

    def module(self, module_id: str) -> ParameterModuleManifest:
        module_id = _nonempty(module_id, "module_id")
        for manifest in self.manifests:
            if manifest.descriptor.module_id == module_id:
                return manifest
        raise KeyError(module_id)

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PARAMETER_INVENTORY_SCHEMA,
            "expected_module_ids": list(self.expected_module_ids),
            "missing_module_ids": list(self.missing_module_ids),
            "complete": self.complete,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "buffer_numel": self.buffer_numel,
            "state_numel": self.state_numel,
            "manifests": [item.to_canonical_dict() for item in self.manifests],
        }
        if include_id:
            value["inventory_id"] = self.inventory_id
        return value


@dataclass(frozen=True, slots=True)
class ParameterMutationGrant:
    grant_id: str
    module_id: str
    generation_id: str
    policy: ParameterMutationPolicy | str
    allowed_names: tuple[str, ...] = ()
    allowed_prefixes: tuple[str, ...] = ()
    max_trainable_parameters: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "grant_id", _nonempty(self.grant_id, "grant_id"))
        object.__setattr__(self, "module_id", _nonempty(self.module_id, "module_id"))
        object.__setattr__(self, "generation_id", _nonempty(self.generation_id, "generation_id"))
        policy = self.policy if isinstance(self.policy, ParameterMutationPolicy) else ParameterMutationPolicy(self.policy)
        object.__setattr__(self, "policy", policy)
        names = tuple(sorted(set(_nonempty(str(item), "allowed parameter name") for item in self.allowed_names)))
        prefixes = tuple(sorted(set(_nonempty(str(item), "allowed parameter prefix") for item in self.allowed_prefixes)))
        object.__setattr__(self, "allowed_names", names)
        object.__setattr__(self, "allowed_prefixes", prefixes)
        if self.max_trainable_parameters is not None and (
            isinstance(self.max_trainable_parameters, bool)
            or not isinstance(self.max_trainable_parameters, int)
            or self.max_trainable_parameters <= 0
        ):
            raise ValueError("max_trainable_parameters must be a positive integer or None")
        if policy is ParameterMutationPolicy.EXPLICIT_NAMES and not names:
            raise ValueError("explicit_names policy requires allowed_names")
        if policy is ParameterMutationPolicy.ADAPTER_ONLY and not prefixes:
            raise ValueError("adapter_only policy requires allowed_prefixes")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": PARAMETER_GRANT_SCHEMA,
            "grant_id": self.grant_id,
            "module_id": self.module_id,
            "generation_id": self.generation_id,
            "policy": self.policy.value,
            "allowed_names": list(self.allowed_names),
            "allowed_prefixes": list(self.allowed_prefixes),
            "max_trainable_parameters": self.max_trainable_parameters,
        }


@dataclass(frozen=True, slots=True)
class ParameterMutationPlan:
    base_inventory_id: str
    module_id: str
    base_generation_id: str
    candidate_generation_id: str
    tensor_names: tuple[str, ...]
    optimizer_name: str
    learning_rate: float
    max_steps: int
    source_manifest_ids: tuple[str, ...]
    holdout_manifest_ids: tuple[str, ...]
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_inventory_id", _nonempty(self.base_inventory_id, "base_inventory_id"))
        object.__setattr__(self, "module_id", _nonempty(self.module_id, "module_id"))
        object.__setattr__(self, "base_generation_id", _nonempty(self.base_generation_id, "base_generation_id"))
        object.__setattr__(self, "candidate_generation_id", _nonempty(self.candidate_generation_id, "candidate_generation_id"))
        if self.candidate_generation_id == self.base_generation_id:
            raise ValueError("candidate_generation_id must differ from base_generation_id")
        names = tuple(sorted(set(_nonempty(str(item), "tensor name") for item in self.tensor_names)))
        if not names:
            raise ValueError("tensor_names cannot be empty")
        object.__setattr__(self, "tensor_names", names)
        object.__setattr__(self, "optimizer_name", _nonempty(self.optimizer_name, "optimizer_name"))
        learning_rate = float(self.learning_rate)
        if not 0.0 < learning_rate < float("inf"):
            raise ValueError("learning_rate must be positive and finite")
        object.__setattr__(self, "learning_rate", learning_rate)
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        sources = tuple(sorted(set(_nonempty(str(item), "source manifest id") for item in self.source_manifest_ids)))
        holdouts = tuple(sorted(set(_nonempty(str(item), "holdout manifest id") for item in self.holdout_manifest_ids)))
        if not sources:
            raise ValueError("source_manifest_ids cannot be empty")
        if not holdouts:
            raise ValueError("holdout_manifest_ids cannot be empty")
        if set(sources) & set(holdouts):
            raise ValueError("source and holdout manifests must be disjoint")
        object.__setattr__(self, "source_manifest_ids", sources)
        object.__setattr__(self, "holdout_manifest_ids", holdouts)
        object.__setattr__(self, "plan_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PARAMETER_PLAN_SCHEMA,
            "base_inventory_id": self.base_inventory_id,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "tensor_names": list(self.tensor_names),
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "source_manifest_ids": list(self.source_manifest_ids),
            "holdout_manifest_ids": list(self.holdout_manifest_ids),
        }
        if include_id:
            value["plan_id"] = self.plan_id
        return value


@dataclass(frozen=True, slots=True)
class ParameterMutationPlanV2:
    """Resource-independent mutation scope for permanent Trainer tissue.

    The historical v1 plan couples optimizer, learning rate, and ``max_steps``
    to plan identity.  V2 moves learning behavior into
    :class:`GovernedLearningPolicy` and bounded execution into a renewable
    :class:`ResourceTranche`, so granting more compute cannot rename or restart
    the candidate.
    """

    base_inventory_id: str
    module_id: str
    base_generation_id: str
    candidate_generation_id: str
    tensor_names: tuple[str, ...]
    source_manifest_ids: tuple[str, ...]
    holdout_manifest_ids: tuple[str, ...]
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_inventory_id", _nonempty(self.base_inventory_id, "base_inventory_id"))
        object.__setattr__(self, "module_id", _nonempty(self.module_id, "module_id"))
        object.__setattr__(self, "base_generation_id", _nonempty(self.base_generation_id, "base_generation_id"))
        object.__setattr__(self, "candidate_generation_id", _nonempty(self.candidate_generation_id, "candidate_generation_id"))
        if self.candidate_generation_id == self.base_generation_id:
            raise ValueError("candidate_generation_id must differ from base_generation_id")
        names = tuple(sorted(set(_nonempty(str(item), "tensor name") for item in self.tensor_names)))
        if not names:
            raise ValueError("tensor_names cannot be empty")
        object.__setattr__(self, "tensor_names", names)
        sources = tuple(sorted(set(_nonempty(str(item), "source manifest id") for item in self.source_manifest_ids)))
        holdouts = tuple(sorted(set(_nonempty(str(item), "holdout manifest id") for item in self.holdout_manifest_ids)))
        if not sources:
            raise ValueError("source_manifest_ids cannot be empty")
        if not holdouts:
            raise ValueError("holdout_manifest_ids cannot be empty")
        if set(sources) & set(holdouts):
            raise ValueError("source and holdout manifests must be disjoint")
        object.__setattr__(self, "source_manifest_ids", sources)
        object.__setattr__(self, "holdout_manifest_ids", holdouts)
        object.__setattr__(self, "plan_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PARAMETER_PLAN_V2_SCHEMA,
            "base_inventory_id": self.base_inventory_id,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "tensor_names": list(self.tensor_names),
            "source_manifest_ids": list(self.source_manifest_ids),
            "holdout_manifest_ids": list(self.holdout_manifest_ids),
        }
        if include_id:
            value["plan_id"] = self.plan_id
        return value


ParameterMutationPlanLike = ParameterMutationPlan | ParameterMutationPlanV2


def is_parameter_mutation_plan(value: object) -> bool:
    return isinstance(value, (ParameterMutationPlan, ParameterMutationPlanV2))


@dataclass(frozen=True, slots=True)
class ParameterPromotionProposal:
    base_inventory_id: str
    module_id: str
    base_generation_id: str
    candidate_generation_id: str
    candidate_artifact_id: str
    lineage_manifest_id: str
    evaluation_ids: tuple[str, ...]
    proposal_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "base_inventory_id",
            "module_id",
            "base_generation_id",
            "candidate_generation_id",
            "candidate_artifact_id",
            "lineage_manifest_id",
        ):
            object.__setattr__(self, label, _nonempty(getattr(self, label), label))
        if self.base_generation_id == self.candidate_generation_id:
            raise ValueError("promotion candidate must be a new generation")
        evaluations = tuple(sorted(set(_nonempty(str(item), "evaluation id") for item in self.evaluation_ids)))
        if not evaluations:
            raise ValueError("promotion requires at least one evaluation id")
        object.__setattr__(self, "evaluation_ids", evaluations)
        object.__setattr__(self, "proposal_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PARAMETER_PROMOTION_SCHEMA,
            "base_inventory_id": self.base_inventory_id,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "lineage_manifest_id": self.lineage_manifest_id,
            "evaluation_ids": list(self.evaluation_ids),
        }
        if include_id:
            value["proposal_id"] = self.proposal_id
        return value


__all__ = [
    "PARAMETER_GRANT_SCHEMA",
    "PARAMETER_INVENTORY_SCHEMA",
    "PARAMETER_MANIFEST_SCHEMA",
    "PARAMETER_MODULE_SCHEMA",
    "PARAMETER_PLAN_SCHEMA",
    "PARAMETER_PLAN_V2_SCHEMA",
    "PARAMETER_PROMOTION_SCHEMA",
    "PARAMETER_TENSOR_SCHEMA",
    "TRAINER_PARAMETER_SCHEMA",
    "OrganKind",
    "ParameterInventory",
    "ParameterModuleDescriptor",
    "ParameterModuleManifest",
    "ParameterMutationGrant",
    "ParameterMutationPlan",
    "ParameterMutationPlanLike",
    "ParameterMutationPlanV2",
    "ParameterMutationPolicy",
    "ParameterPromotionProposal",
    "ParameterTensorRecord",
    "TrainerCoreRole",
    "is_parameter_mutation_plan",
]
