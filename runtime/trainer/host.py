"""Sovereign parameter-control host for Axon's Trainer organ.

The host owns inventory, telemetry, authorization, isolated candidate execution,
checkpoint lineage, evaluation gates, and promotion proposals.  It never trains
a live registered module in place and a proposal cannot activate itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from torch import nn

from .authority import AuthorizedParameterMutation, ParameterAuthorityError, authorize_parameter_mutation
from .execution import CandidateOptimizationSession, OptimizerExecutionPolicy
from .gates import EvaluationObservation, PromotionGate, PromotionGateDecision, evaluate_promotion_gate
from .contracts import (
    ParameterInventory,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterPromotionProposal,
)
from .lease import TrainerLeaseDeniedError, TrainerWriterLease
from .registry import ParameterRegistry
from .store import TrainerStateStore
from .telemetry import ParameterTelemetryFrame, capture_parameter_telemetry


@dataclass(slots=True)
class TrainerControlPlane:
    registry: ParameterRegistry
    store: TrainerStateStore
    lease: TrainerWriterLease | None = None

    @classmethod
    def active(
        cls,
        *,
        state_root: Path | str = Path(r"D:\Axon\State"),
        acquire_lease: bool = True,
    ) -> "TrainerControlPlane":
        lease = TrainerWriterLease(state_root)
        if acquire_lease:
            lease.acquire()
        return cls(
            registry=ParameterRegistry(),
            store=TrainerStateStore.active(state_root=state_root),
            lease=lease,
        )

    def _require_writer_authority(self) -> None:
        if self.lease is not None and not self.lease.is_held_by_us():
            raise TrainerLeaseDeniedError("Trainer parameter-writer lease is not held")

    def close(self) -> None:
        if self.lease is not None:
            self.lease.release()

    def __enter__(self) -> "TrainerControlPlane":
        self._require_writer_authority()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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
        self._require_writer_authority()
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
        self._require_writer_authority()
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
        self._require_writer_authority()
        authorization = authorize_parameter_mutation(inventory, grant, plan)
        self.store.write_plan(plan)
        self.store.write_authorization(authorization)
        return authorization

    def begin_candidate(
        self,
        inventory: ParameterInventory,
        grant: ParameterMutationGrant,
        plan: ParameterMutationPlan,
        *,
        policy: OptimizerExecutionPolicy | None = None,
    ) -> CandidateOptimizationSession:
        """Authorize and create an isolated candidate; the live module stays sealed."""

        authorization = self.authorize(inventory, grant, plan)
        return CandidateOptimizationSession(
            live_module=self.registry.module(plan.module_id),
            base_descriptor=self.registry.descriptor(plan.module_id),
            base_inventory=inventory,
            plan=plan,
            authorization=authorization,
            store=self.store,
            policy=policy,
        )

    def evaluate_gate(
        self,
        gate: PromotionGate,
        observations: Iterable[EvaluationObservation],
    ) -> PromotionGateDecision:
        self._require_writer_authority()
        rows = tuple(observations)
        for observation in rows:
            self.store.write_evaluation_observation(observation)
        decision = evaluate_promotion_gate(gate, rows)
        self.store.write_gate_decision(decision)
        return decision

    def record_promotion_proposal(
        self,
        proposal: ParameterPromotionProposal,
        *,
        gate_decision: PromotionGateDecision,
    ) -> Path:
        """Persist a promotion proposal only after a deterministic gate passes.

        This still does not swap a live module or activate candidate parameters.
        """

        self._require_writer_authority()
        if not isinstance(gate_decision, PromotionGateDecision):
            raise TypeError("gate_decision must be PromotionGateDecision")
        if not gate_decision.passed:
            raise ParameterAuthorityError("failed promotion gate cannot authorize a promotion proposal")
        if proposal.module_id != gate_decision.module_id:
            raise ParameterAuthorityError("promotion proposal module differs from gate decision")
        if proposal.candidate_generation_id != gate_decision.candidate_generation_id:
            raise ParameterAuthorityError("promotion proposal generation differs from gate decision")
        if gate_decision.decision_id not in proposal.evaluation_ids:
            raise ParameterAuthorityError("promotion proposal must cite the deterministic gate decision id")
        return self.store.write_promotion_proposal(proposal)


__all__ = ["TrainerControlPlane"]
