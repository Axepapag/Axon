"""Non-training control-plane host for Axon's Trainer organ.

This host deliberately does not execute optimization yet.  It establishes the
sovereign parameter inventory, telemetry, authorization, and durable lineage
boundary that future offline and online learning must pass through.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from torch import nn

from .authority import AuthorizedParameterMutation, authorize_parameter_mutation
from .contracts import (
    ParameterInventory,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterPromotionProposal,
)
from .registry import ParameterRegistry
from .store import TrainerStateStore
from .telemetry import ParameterTelemetryFrame, capture_parameter_telemetry


@dataclass(slots=True)
class TrainerControlPlane:
    registry: ParameterRegistry
    store: TrainerStateStore

    @classmethod
    def active(cls, *, state_root: Path | str = Path(r"D:\Axon\State")) -> "TrainerControlPlane":
        return cls(registry=ParameterRegistry(), store=TrainerStateStore.active(state_root=state_root))

    def declare_expected(self, descriptors: Iterable[ParameterModuleDescriptor]) -> None:
        self.registry.declare_expected(descriptors)

    def register(self, descriptor: ParameterModuleDescriptor, module: nn.Module) -> None:
        self.registry.register(descriptor, module)

    def snapshot_inventory(
        self,
        *,
        exact_value_hashes: bool = True,
        require_complete: bool = True,
    ) -> ParameterInventory:
        inventory = self.registry.capture_inventory(
            exact_value_hashes=exact_value_hashes,
            require_complete=require_complete,
        )
        self.store.write_inventory(inventory)
        return inventory

    def record_telemetry(
        self,
        module_id: str,
        *,
        step: int,
        inventory_id: str | None = None,
    ) -> ParameterTelemetryFrame:
        descriptor = self.registry.descriptor(module_id)
        frame = capture_parameter_telemetry(
            self.registry.module(module_id),
            module_id=module_id,
            generation_id=descriptor.generation_id,
            step=step,
            inventory_id=inventory_id,
        )
        self.store.append_telemetry(frame)
        return frame

    def authorize(
        self,
        inventory: ParameterInventory,
        grant: ParameterMutationGrant,
        plan: ParameterMutationPlan,
    ) -> AuthorizedParameterMutation:
        authorization = authorize_parameter_mutation(inventory, grant, plan)
        self.store.write_plan(plan)
        self.store.write_authorization(authorization)
        return authorization

    def record_promotion_proposal(self, proposal: ParameterPromotionProposal) -> Path:
        """Persist a proposal only; this does not activate parameters."""

        return self.store.write_promotion_proposal(proposal)


__all__ = ["TrainerControlPlane"]
