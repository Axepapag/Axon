"""Machine-enforced training launch contracts and preflight receipts.

The Trainer may govern finite work, but it may not authorize parameter mutation
against disposable or silently truncated anatomy.  A receipt is deterministic,
content-addressed evidence that the exact candidate plan passed the declared
capacity contract before an optimizer can exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from runtime.field import canonical_sha256

from .contracts import ParameterInventory, ParameterMutationPlan


TRAINING_CAPACITY_CONTRACT_SCHEMA = "axon-training-capacity-contract-v1"
TRAINING_BOUND_SCHEMA = "axon-training-declared-bound-v1"
TRAINING_PREFLIGHT_EVIDENCE_SCHEMA = "axon-training-preflight-evidence-v1"
TRAINING_PREFLIGHT_RECEIPT_SCHEMA = "axon-training-preflight-receipt-v1"


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


class TrainingBoundCategory(str, Enum):
    PHYSICAL_PROCESSING_UNIT = "physical_processing_unit"
    COMPUTE_BUDGET = "compute_budget"
    OPTIMIZATION_BUDGET = "optimization_budget"
    ADMISSION_BUDGET = "admission_budget"
    RESULT_COUNT_POLICY = "result_count_policy"


class PreflightEvidenceKind(str, Enum):
    STATIC_CAPACITY_SCAN = "static_capacity_scan"
    ARCHITECTURE_CAPACITY = "architecture_capacity"
    CURRICULUM_DISTRIBUTION = "curriculum_distribution"
    BOUNDARY_COVERAGE = "boundary_coverage"
    COUNTERFACTUAL_DEPENDENCE = "counterfactual_dependence"
    CHECKPOINT_COMPATIBILITY = "checkpoint_compatibility"


REQUIRED_PREFLIGHT_EVIDENCE = frozenset(PreflightEvidenceKind)


@dataclass(frozen=True, slots=True)
class DeclaredTrainingBound:
    name: str
    category: TrainingBoundCategory | str
    value: int | float
    source_preserved: bool
    continuation_or_failure: str
    affects_checkpoint_capacity: bool = False
    bound_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "bound name"))
        category = self.category if isinstance(self.category, TrainingBoundCategory) else TrainingBoundCategory(self.category)
        object.__setattr__(self, "category", category)
        value = self.value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) < float("inf"):
            raise ValueError("declared bound value must be positive and finite")
        if not self.source_preserved:
            raise ValueError("a permitted training bound must preserve authoritative source")
        if self.affects_checkpoint_capacity:
            raise ValueError("a permitted training bound cannot affect checkpoint capacity")
        object.__setattr__(
            self,
            "continuation_or_failure",
            _nonempty(self.continuation_or_failure, "continuation_or_failure"),
        )
        object.__setattr__(self, "bound_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TRAINING_BOUND_SCHEMA,
            "name": self.name,
            "category": self.category.value,
            "value": self.value,
            "source_preserved": self.source_preserved,
            "continuation_or_failure": self.continuation_or_failure,
            "affects_checkpoint_capacity": self.affects_checkpoint_capacity,
        }
        if include_id:
            value["bound_id"] = self.bound_id
        return value


@dataclass(frozen=True, slots=True)
class CompleteFieldTrainingContract:
    module_id: str
    organ_kind: str
    architecture: str
    architecture_config_id: str
    compiler_schema_ids: tuple[str, ...]
    source_position_scheme: str
    target_position_scheme: str
    declared_bounds: tuple[DeclaredTrainingBound, ...]
    exact_source_required: bool = True
    complete_field_required: bool = True
    contract_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in (
            "module_id",
            "organ_kind",
            "architecture",
            "architecture_config_id",
            "source_position_scheme",
            "target_position_scheme",
        ):
            object.__setattr__(self, label, _nonempty(getattr(self, label), label))
        compiler_ids = tuple(sorted(set(_nonempty(str(item), "compiler schema id") for item in self.compiler_schema_ids)))
        if not compiler_ids:
            raise ValueError("capacity contract requires at least one compiler schema id")
        object.__setattr__(self, "compiler_schema_ids", compiler_ids)
        bounds = tuple(sorted(tuple(self.declared_bounds), key=lambda item: item.name))
        if not all(isinstance(item, DeclaredTrainingBound) for item in bounds):
            raise TypeError("declared_bounds must contain DeclaredTrainingBound values")
        names = [item.name for item in bounds]
        if len(names) != len(set(names)):
            raise ValueError("capacity contract contains duplicate declared bound names")
        if not bounds:
            raise ValueError("capacity contract must declare its finite work controls")
        object.__setattr__(self, "declared_bounds", bounds)
        if not self.exact_source_required or not self.complete_field_required:
            raise ValueError("Trainer contracts must require exact complete-field source processing")
        object.__setattr__(self, "contract_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TRAINING_CAPACITY_CONTRACT_SCHEMA,
            "module_id": self.module_id,
            "organ_kind": self.organ_kind,
            "architecture": self.architecture,
            "architecture_config_id": self.architecture_config_id,
            "compiler_schema_ids": list(self.compiler_schema_ids),
            "source_position_scheme": self.source_position_scheme,
            "target_position_scheme": self.target_position_scheme,
            "exact_source_required": self.exact_source_required,
            "complete_field_required": self.complete_field_required,
            "declared_bounds": [item.to_canonical_dict() for item in self.declared_bounds],
        }
        if include_id:
            value["contract_id"] = self.contract_id
        return value


@dataclass(frozen=True, slots=True)
class TrainingPreflightEvidence:
    kind: PreflightEvidenceKind | str
    artifact_id: str
    summary: str
    passed: bool
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, PreflightEvidenceKind) else PreflightEvidenceKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "artifact_id", _nonempty(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "summary", _nonempty(self.summary, "evidence summary"))
        object.__setattr__(self, "evidence_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @classmethod
    def from_payload(
        cls,
        *,
        kind: PreflightEvidenceKind | str,
        payload: Any,
        summary: str,
        passed: bool = True,
    ) -> "TrainingPreflightEvidence":
        return cls(kind=kind, artifact_id=canonical_sha256(payload), summary=summary, passed=passed)

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TRAINING_PREFLIGHT_EVIDENCE_SCHEMA,
            "kind": self.kind.value,
            "artifact_id": self.artifact_id,
            "summary": self.summary,
            "passed": self.passed,
        }
        if include_id:
            value["evidence_id"] = self.evidence_id
        return value


@dataclass(frozen=True, slots=True)
class TrainingPreflightReceipt:
    contract: CompleteFieldTrainingContract
    inventory_id: str
    parameter_manifest_id: str
    plan_id: str
    module_id: str
    base_generation_id: str
    candidate_generation_id: str
    source_manifest_ids: tuple[str, ...]
    holdout_manifest_ids: tuple[str, ...]
    evidence: tuple[TrainingPreflightEvidence, ...]
    passed: bool = field(init=False)
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.contract, CompleteFieldTrainingContract):
            raise TypeError("contract must be CompleteFieldTrainingContract")
        for label in (
            "inventory_id",
            "parameter_manifest_id",
            "plan_id",
            "module_id",
            "base_generation_id",
            "candidate_generation_id",
        ):
            object.__setattr__(self, label, _nonempty(getattr(self, label), label))
        sources = tuple(sorted(set(_nonempty(str(item), "source manifest id") for item in self.source_manifest_ids)))
        holdouts = tuple(sorted(set(_nonempty(str(item), "holdout manifest id") for item in self.holdout_manifest_ids)))
        if not sources or not holdouts or set(sources) & set(holdouts):
            raise ValueError("preflight source and holdout manifests must be non-empty and disjoint")
        object.__setattr__(self, "source_manifest_ids", sources)
        object.__setattr__(self, "holdout_manifest_ids", holdouts)
        evidence = tuple(sorted(tuple(self.evidence), key=lambda item: item.kind.value))
        if not all(isinstance(item, TrainingPreflightEvidence) for item in evidence):
            raise TypeError("evidence must contain TrainingPreflightEvidence values")
        kinds = [item.kind for item in evidence]
        if len(kinds) != len(set(kinds)):
            raise ValueError("preflight receipt contains duplicate evidence kinds")
        missing = REQUIRED_PREFLIGHT_EVIDENCE - set(kinds)
        if missing:
            labels = ", ".join(sorted(item.value for item in missing))
            raise ValueError(f"preflight receipt lacks required evidence: {labels}")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "passed", all(item.passed for item in evidence))
        object.__setattr__(self, "receipt_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def assert_authorizes(self, inventory: ParameterInventory, plan: ParameterMutationPlan) -> None:
        if not isinstance(inventory, ParameterInventory) or not isinstance(plan, ParameterMutationPlan):
            raise TypeError("preflight authorization requires ParameterInventory and ParameterMutationPlan")
        manifest = inventory.module(plan.module_id)
        descriptor = manifest.descriptor
        mismatches: list[str] = []
        expected = {
            "inventory_id": inventory.inventory_id,
            "parameter_manifest_id": manifest.manifest_id,
            "plan_id": plan.plan_id,
            "module_id": plan.module_id,
            "base_generation_id": plan.base_generation_id,
            "candidate_generation_id": plan.candidate_generation_id,
            "source_manifest_ids": tuple(plan.source_manifest_ids),
            "holdout_manifest_ids": tuple(plan.holdout_manifest_ids),
        }
        for label, value in expected.items():
            if getattr(self, label) != value:
                mismatches.append(label)
        if self.contract.module_id != descriptor.module_id:
            mismatches.append("contract.module_id")
        if self.contract.organ_kind != descriptor.organ_kind.value:
            mismatches.append("contract.organ_kind")
        if self.contract.architecture != descriptor.architecture:
            mismatches.append("contract.architecture")
        if not self.passed:
            mismatches.append("passed")
        if mismatches:
            raise ValueError("training preflight does not authorize this candidate: " + ", ".join(mismatches))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TRAINING_PREFLIGHT_RECEIPT_SCHEMA,
            "contract": self.contract.to_canonical_dict(),
            "inventory_id": self.inventory_id,
            "parameter_manifest_id": self.parameter_manifest_id,
            "plan_id": self.plan_id,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "source_manifest_ids": list(self.source_manifest_ids),
            "holdout_manifest_ids": list(self.holdout_manifest_ids),
            "evidence": [item.to_canonical_dict() for item in self.evidence],
            "passed": self.passed,
        }
        if include_id:
            value["receipt_id"] = self.receipt_id
        return value


def build_training_preflight_receipt(
    *,
    contract: CompleteFieldTrainingContract,
    inventory: ParameterInventory,
    plan: ParameterMutationPlan,
    evidence: Iterable[TrainingPreflightEvidence],
) -> TrainingPreflightReceipt:
    manifest = inventory.module(plan.module_id)
    receipt = TrainingPreflightReceipt(
        contract=contract,
        inventory_id=inventory.inventory_id,
        parameter_manifest_id=manifest.manifest_id,
        plan_id=plan.plan_id,
        module_id=plan.module_id,
        base_generation_id=plan.base_generation_id,
        candidate_generation_id=plan.candidate_generation_id,
        source_manifest_ids=plan.source_manifest_ids,
        holdout_manifest_ids=plan.holdout_manifest_ids,
        evidence=tuple(evidence),
    )
    receipt.assert_authorizes(inventory, plan)
    return receipt


__all__ = [
    "TRAINING_CAPACITY_CONTRACT_SCHEMA",
    "TRAINING_BOUND_SCHEMA",
    "TRAINING_PREFLIGHT_EVIDENCE_SCHEMA",
    "TRAINING_PREFLIGHT_RECEIPT_SCHEMA",
    "TrainingBoundCategory",
    "PreflightEvidenceKind",
    "REQUIRED_PREFLIGHT_EVIDENCE",
    "DeclaredTrainingBound",
    "CompleteFieldTrainingContract",
    "TrainingPreflightEvidence",
    "TrainingPreflightReceipt",
    "build_training_preflight_receipt",
]
