"""Fail-closed parameter mutation authority for Axon's Trainer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from runtime.field import canonical_sha256

from .contracts import (
    ParameterInventory,
    ParameterMutationGrant,
    ParameterMutationPlanLike,
    ParameterMutationPolicy,
    is_parameter_mutation_plan,
)
from .preflight import TrainingPreflightReceipt

AUTHORIZED_MUTATION_SCHEMA = "axon-authorized-parameter-mutation-v2"


class ParameterAuthorityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AuthorizedParameterMutation:
    grant_id: str
    plan_id: str
    inventory_id: str
    module_id: str
    base_generation_id: str
    candidate_generation_id: str
    tensor_names: tuple[str, ...]
    parameter_count: int
    preflight_receipt_id: str
    authorization_id: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": AUTHORIZED_MUTATION_SCHEMA,
            "grant_id": self.grant_id,
            "plan_id": self.plan_id,
            "inventory_id": self.inventory_id,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "tensor_names": list(self.tensor_names),
            "parameter_count": self.parameter_count,
            "preflight_receipt_id": self.preflight_receipt_id,
            "authorization_id": self.authorization_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AuthorizedParameterMutation":
        item = dict(value)
        if item.pop("schema", None) != AUTHORIZED_MUTATION_SCHEMA:
            raise ParameterAuthorityError("unsupported authorized mutation schema")
        try:
            item["tensor_names"] = tuple(item["tensor_names"])
            authorization = cls(**item)
        except (KeyError, TypeError, ValueError) as exc:
            raise ParameterAuthorityError("authorized mutation record is invalid") from exc
        core = authorization.to_canonical_dict()
        core.pop("authorization_id")
        if canonical_sha256(core) != authorization.authorization_id:
            raise ParameterAuthorityError("authorized mutation identity mismatch")
        return authorization

    def resume_scope(self) -> dict[str, Any]:
        """Stable mutation authority excluding refreshable preflight evidence."""

        return {
            "grant_id": self.grant_id,
            "plan_id": self.plan_id,
            "inventory_id": self.inventory_id,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "tensor_names": self.tensor_names,
            "parameter_count": self.parameter_count,
        }


def authorize_parameter_mutation(
    inventory: ParameterInventory,
    grant: ParameterMutationGrant,
    plan: ParameterMutationPlanLike,
    preflight_receipt: TrainingPreflightReceipt,
) -> AuthorizedParameterMutation:
    """Validate one proposed training mutation against exact current lineage.

    This function grants permission only.  It does not create an optimizer,
    execute backpropagation, alter ``requires_grad``, mutate a tensor, or promote
    a generation.
    """

    if not isinstance(inventory, ParameterInventory):
        raise TypeError("inventory must be ParameterInventory")
    if not isinstance(grant, ParameterMutationGrant):
        raise TypeError("grant must be ParameterMutationGrant")
    if not is_parameter_mutation_plan(plan):
        raise TypeError("plan must be a governed parameter mutation plan")
    if not isinstance(preflight_receipt, TrainingPreflightReceipt):
        raise TypeError("parameter mutation authority requires a TrainingPreflightReceipt")
    try:
        preflight_receipt.assert_authorizes(inventory, plan)
    except ValueError as exc:
        raise ParameterAuthorityError(f"training preflight is stale or mismatched: {exc}") from exc
    if not inventory.complete:
        raise ParameterAuthorityError("incomplete parameter inventory cannot authorize mutation")
    if plan.base_inventory_id != inventory.inventory_id:
        raise ParameterAuthorityError("training plan is stale relative to the current parameter inventory")
    if plan.module_id != grant.module_id:
        raise ParameterAuthorityError("training plan module does not match mutation grant")

    try:
        manifest = inventory.module(plan.module_id)
    except KeyError as exc:
        raise ParameterAuthorityError(f"unknown target module {plan.module_id!r}") from exc

    descriptor = manifest.descriptor
    if descriptor.generation_id != plan.base_generation_id:
        raise ParameterAuthorityError("training plan base generation does not match current module generation")
    if descriptor.generation_id != grant.generation_id:
        raise ParameterAuthorityError("mutation grant generation does not match current module generation")

    records = {item.name: item for item in manifest.tensors}
    unknown = sorted(set(plan.tensor_names) - set(records))
    if unknown:
        raise ParameterAuthorityError(f"training plan names unknown parameters: {unknown}")
    frozen = sorted(name for name in plan.tensor_names if not records[name].requires_grad)
    if frozen:
        raise ParameterAuthorityError(
            "training plan attempts to mutate parameters currently marked non-trainable: " + ", ".join(frozen)
        )

    if grant.policy is ParameterMutationPolicy.SEALED:
        raise ParameterAuthorityError("sealed grant authorizes no parameter mutation")
    if grant.policy is ParameterMutationPolicy.EXPLICIT_NAMES:
        unauthorized = sorted(set(plan.tensor_names) - set(grant.allowed_names))
        if unauthorized:
            raise ParameterAuthorityError(f"training plan exceeds explicit parameter grant: {unauthorized}")
    if grant.policy is ParameterMutationPolicy.ADAPTER_ONLY:
        unauthorized = sorted(
            name
            for name in plan.tensor_names
            if not any(name.startswith(prefix) for prefix in grant.allowed_prefixes)
        )
        if unauthorized:
            raise ParameterAuthorityError(f"adapter-only grant rejected non-adapter parameters: {unauthorized}")

    parameter_count = sum(records[name].numel for name in plan.tensor_names)
    if grant.max_trainable_parameters is not None and parameter_count > grant.max_trainable_parameters:
        raise ParameterAuthorityError(
            f"training plan requests {parameter_count} parameters above grant limit "
            f"{grant.max_trainable_parameters}"
        )

    core = {
        "schema": AUTHORIZED_MUTATION_SCHEMA,
        "grant_id": grant.grant_id,
        "plan_id": plan.plan_id,
        "inventory_id": inventory.inventory_id,
        "module_id": plan.module_id,
        "base_generation_id": plan.base_generation_id,
        "candidate_generation_id": plan.candidate_generation_id,
        "tensor_names": list(plan.tensor_names),
        "parameter_count": parameter_count,
        "preflight_receipt_id": preflight_receipt.receipt_id,
    }
    return AuthorizedParameterMutation(
        grant_id=grant.grant_id,
        plan_id=plan.plan_id,
        inventory_id=inventory.inventory_id,
        module_id=plan.module_id,
        base_generation_id=plan.base_generation_id,
        candidate_generation_id=plan.candidate_generation_id,
        tensor_names=plan.tensor_names,
        parameter_count=parameter_count,
        preflight_receipt_id=preflight_receipt.receipt_id,
        authorization_id=canonical_sha256(core),
    )


__all__ = [
    "AUTHORIZED_MUTATION_SCHEMA",
    "AuthorizedParameterMutation",
    "ParameterAuthorityError",
    "authorize_parameter_mutation",
]
