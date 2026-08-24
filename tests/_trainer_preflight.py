"""Deterministic complete-field preflight fixtures for Trainer unit tests."""
from __future__ import annotations

from runtime.field import canonical_sha256
from runtime.trainer import (
    CompleteFieldTrainingContract,
    DeclaredTrainingBound,
    ParameterInventory,
    ParameterMutationPlan,
    PreflightEvidenceKind,
    TrainingBoundCategory,
    TrainingPreflightEvidence,
    TrainingPreflightReceipt,
    build_training_preflight_receipt,
)


def unit_preflight_receipt(
    inventory: ParameterInventory,
    plan: ParameterMutationPlan,
) -> TrainingPreflightReceipt:
    """Create explicit launch evidence for a synthetic authority-path test.

    These fixtures test Trainer enforcement, not neural capability.  Production
    organs must use their organ-specific executable preflight.
    """

    descriptor = inventory.module(plan.module_id).descriptor
    contract = CompleteFieldTrainingContract(
        module_id=descriptor.module_id,
        organ_kind=descriptor.organ_kind.value,
        architecture=descriptor.architecture,
        architecture_config_id=canonical_sha256(
            {"unit_test_architecture": descriptor.to_canonical_dict()}
        ),
        compiler_schema_ids=("unit-test-exact-complete-field-v1",),
        source_position_scheme="unit-test-dynamic-position-v1",
        target_position_scheme="unit-test-dynamic-position-v1",
        declared_bounds=(
            DeclaredTrainingBound(
                name="unit_test_optimizer_steps",
                category=TrainingBoundCategory.OPTIMIZATION_BUDGET,
                value=plan.max_steps,
                source_preserved=True,
                continuation_or_failure="Synthetic test ends explicitly without a serving claim.",
            ),
        ),
    )
    evidence = tuple(
        TrainingPreflightEvidence.from_payload(
            kind=kind,
            payload={"fixture": "trainer-unit-preflight-v1", "kind": kind.value, "plan_id": plan.plan_id},
            summary=f"Synthetic Trainer enforcement fixture: {kind.value}",
        )
        for kind in PreflightEvidenceKind
    )
    return build_training_preflight_receipt(
        contract=contract,
        inventory=inventory,
        plan=plan,
        evidence=evidence,
    )


__all__ = ["unit_preflight_receipt"]
