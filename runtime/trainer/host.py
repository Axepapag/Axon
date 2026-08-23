"""Sovereign parameter-control host for Axon's Trainer organ.

The host owns inventory, telemetry, authorization, isolated candidate execution,
checkpoint lineage, evaluation gates, and promotion proposals.  It never trains
a live registered module in place and a proposal cannot activate itself.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from torch import nn

from .activation import (
    ActiveGenerationPointer,
    ParameterActivationReceipt,
    ParameterRollbackReceipt,
)
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
from .lifecycle import CandidateCheckpointRecord
from .registry import (
    ParameterRegistry,
    capture_module_manifest,
    descriptors_share_anatomy,
    parameter_value_sha256,
)
from .store import TrainerStateStore
from .telemetry import ParameterTelemetryFrame, capture_parameter_telemetry


def _descriptor_from_payload(value: dict) -> ParameterModuleDescriptor:
    return ParameterModuleDescriptor(
        module_id=value["module_id"],
        organ_kind=value["organ_kind"],
        generation_id=value["generation_id"],
        architecture=value["architecture"],
        d_model=value.get("d_model"),
        trainer_core_role=value.get("trainer_core_role"),
        tags=tuple(value.get("tags", ())),
    )


def _verify_module_values_against_manifest(module: nn.Module, manifest: dict) -> None:
    """Verify exact parameter/buffer values while intentionally ignoring requires_grad policy."""

    parameters = dict(module.named_parameters(recurse=True))
    expected_parameters = {item["name"]: item for item in manifest.get("tensors", ())}
    if set(parameters) != set(expected_parameters):
        raise ParameterAuthorityError("module parameter surface differs from checkpoint manifest")
    for name, parameter in parameters.items():
        record = expected_parameters[name]
        if list(parameter.shape) != list(record["shape"]) or str(parameter.dtype) != record["dtype"]:
            raise ParameterAuthorityError(f"parameter structure differs from checkpoint manifest: {name}")
        if int(parameter.numel()) != int(record["numel"]):
            raise ParameterAuthorityError(f"parameter size differs from checkpoint manifest: {name}")
        if parameter_value_sha256(parameter) != record["value_sha256"]:
            raise ParameterAuthorityError(f"parameter value differs from checkpoint manifest: {name}")

    buffers = dict(module.named_buffers(recurse=True))
    expected_buffers = {item["name"]: item for item in manifest.get("buffers", ())}
    if set(buffers) != set(expected_buffers):
        raise ParameterAuthorityError("module persistent-buffer surface differs from checkpoint manifest")
    for name, buffer in buffers.items():
        record = expected_buffers[name]
        if list(buffer.shape) != list(record["shape"]) or str(buffer.dtype) != record["dtype"]:
            raise ParameterAuthorityError(f"persistent-buffer structure differs from checkpoint manifest: {name}")
        if int(buffer.numel()) != int(record["numel"]):
            raise ParameterAuthorityError(f"persistent-buffer size differs from checkpoint manifest: {name}")
        if parameter_value_sha256(buffer) != record["value_sha256"]:
            raise ParameterAuthorityError(f"persistent-buffer value differs from checkpoint manifest: {name}")


@dataclass(slots=True)
class TrainerControlPlane:
    registry: ParameterRegistry
    store: TrainerStateStore
    lease: TrainerWriterLease

    def __post_init__(self) -> None:
        if not isinstance(self.lease, TrainerWriterLease):
            raise TypeError("TrainerControlPlane requires a TrainerWriterLease")
        if self.lease.trainer_dir.resolve(strict=False) != self.store.root.resolve(strict=False):
            raise TrainerLeaseDeniedError("Trainer writer lease does not govern this Trainer state store")

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
        if not self.lease.is_held_by_us():
            raise TrainerLeaseDeniedError("Trainer parameter-writer lease is not held")

    def close(self) -> None:
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

    def activate_candidate(
        self,
        base_inventory: ParameterInventory,
        proposal: ParameterPromotionProposal,
        *,
        gate_decision: PromotionGateDecision,
        checkpoint: CandidateCheckpointRecord,
    ) -> ParameterActivationReceipt:
        """Atomically advance one live module to a fully gated candidate generation."""

        self._require_writer_authority()
        if not isinstance(base_inventory, ParameterInventory):
            raise TypeError("base_inventory must be ParameterInventory")
        if not isinstance(proposal, ParameterPromotionProposal):
            raise TypeError("proposal must be ParameterPromotionProposal")
        if not isinstance(gate_decision, PromotionGateDecision):
            raise TypeError("gate_decision must be PromotionGateDecision")
        if not isinstance(checkpoint, CandidateCheckpointRecord):
            raise TypeError("checkpoint must be CandidateCheckpointRecord")
        if not gate_decision.passed:
            raise ParameterAuthorityError("failed promotion gate cannot activate parameters")

        current_inventory = self.registry.capture_inventory(exact_value_hashes=True, require_complete=True)
        if current_inventory.inventory_id != base_inventory.inventory_id:
            raise ParameterAuthorityError("activation base inventory is stale")
        if proposal.base_inventory_id != base_inventory.inventory_id:
            raise ParameterAuthorityError("promotion proposal base inventory is stale")
        current_descriptor = self.registry.descriptor(proposal.module_id)
        if current_descriptor.generation_id != proposal.base_generation_id:
            raise ParameterAuthorityError("live module generation changed before activation")
        if proposal.module_id != gate_decision.module_id or proposal.candidate_generation_id != gate_decision.candidate_generation_id:
            raise ParameterAuthorityError("promotion gate lineage differs from proposal")
        if gate_decision.decision_id not in proposal.evaluation_ids:
            raise ParameterAuthorityError("promotion proposal does not cite the passed gate decision")
        if checkpoint.module_id != proposal.module_id:
            raise ParameterAuthorityError("candidate checkpoint module differs from promotion proposal")
        if checkpoint.base_generation_id != proposal.base_generation_id:
            raise ParameterAuthorityError("candidate checkpoint base generation differs from proposal")
        if checkpoint.candidate_generation_id != proposal.candidate_generation_id:
            raise ParameterAuthorityError("candidate checkpoint generation differs from proposal")
        if checkpoint.checkpoint_id != proposal.candidate_artifact_id:
            raise ParameterAuthorityError("promotion proposal does not identify the supplied candidate checkpoint")
        if checkpoint.parameter_manifest_id != proposal.lineage_manifest_id:
            raise ParameterAuthorityError("promotion proposal manifest differs from candidate checkpoint")

        payload = self.store.load_verified_candidate_checkpoint(checkpoint)
        candidate_descriptor = _descriptor_from_payload(dict(payload["descriptor"]))
        if not descriptors_share_anatomy(current_descriptor, candidate_descriptor):
            raise ParameterAuthorityError("candidate checkpoint attempts to change live organ anatomy")
        if candidate_descriptor.generation_id != proposal.candidate_generation_id:
            raise ParameterAuthorityError("candidate descriptor generation differs from promotion proposal")

        live_module = self.registry.module(proposal.module_id)
        probe = copy.deepcopy(live_module)
        probe.load_state_dict(payload["module_state_dict"], strict=True)
        _verify_module_values_against_manifest(probe, dict(payload["parameter_manifest"]))

        previous_pointer = self.store.read_active_pointer(proposal.module_id)
        if previous_pointer is not None and previous_pointer.generation_id != current_descriptor.generation_id:
            raise ParameterAuthorityError("active-generation pointer disagrees with the registered live generation")
        before_manifest = current_inventory.module(proposal.module_id)
        rollback_snapshot = self.store.save_generation_snapshot(
            module=live_module,
            descriptor=current_descriptor,
            source_inventory_id=base_inventory.inventory_id,
            previous_pointer_id=None if previous_pointer is None else previous_pointer.pointer_id,
        )
        rollback_payload = self.store.load_verified_generation_snapshot(rollback_snapshot)
        committed = False
        registry_transitioned = False
        try:
            live_module.load_state_dict(payload["module_state_dict"], strict=True)
            _verify_module_values_against_manifest(live_module, dict(payload["parameter_manifest"]))
            after_manifest = capture_module_manifest(candidate_descriptor, live_module, exact_value_hashes=True)
            self.registry.transition_generation(
                proposal.module_id,
                expected_generation_id=current_descriptor.generation_id,
                new_descriptor=candidate_descriptor,
            )
            registry_transitioned = True
            active_snapshot = self.store.save_generation_snapshot(
                module=live_module,
                descriptor=candidate_descriptor,
                source_inventory_id=base_inventory.inventory_id,
                previous_pointer_id=None if previous_pointer is None else previous_pointer.pointer_id,
            )
            pointer = ActiveGenerationPointer(
                module_id=proposal.module_id,
                generation_id=candidate_descriptor.generation_id,
                parameter_manifest_id=active_snapshot.parameter_manifest_id,
                artifact_relpath=active_snapshot.artifact_relpath,
                artifact_sha256=active_snapshot.artifact_sha256,
                source_kind="generation_snapshot",
                source_artifact_id=active_snapshot.snapshot_id,
                previous_generation_id=current_descriptor.generation_id,
                previous_pointer_id=None if previous_pointer is None else previous_pointer.pointer_id,
                rollback_snapshot_id=rollback_snapshot.snapshot_id,
            )
            self.store.publish_active_pointer(pointer)
            committed = True
        except Exception:
            if not committed:
                live_module.load_state_dict(rollback_payload["module_state_dict"], strict=True)
                if registry_transitioned:
                    self.registry.transition_generation(
                        proposal.module_id,
                        expected_generation_id=candidate_descriptor.generation_id,
                        new_descriptor=current_descriptor,
                    )
            raise

        receipt = ParameterActivationReceipt(
            module_id=proposal.module_id,
            base_generation_id=current_descriptor.generation_id,
            activated_generation_id=candidate_descriptor.generation_id,
            base_inventory_id=base_inventory.inventory_id,
            proposal_id=proposal.proposal_id,
            gate_decision_id=gate_decision.decision_id,
            checkpoint_id=checkpoint.checkpoint_id,
            rollback_snapshot_id=rollback_snapshot.snapshot_id,
            active_pointer_id=pointer.pointer_id,
            before_manifest_id=before_manifest.manifest_id,
            after_manifest_id=after_manifest.manifest_id,
        )
        self.store.write_activation_receipt(receipt)
        return receipt

    def rollback_active_generation(self, module_id: str) -> ParameterRollbackReceipt:
        """Restore the previous exact generation referenced by the active pointer."""

        self._require_writer_authority()
        pointer = self.store.read_active_pointer(module_id)
        if pointer is None:
            raise ParameterAuthorityError("no active-generation pointer exists for rollback")
        if pointer.rollback_snapshot_id is None or pointer.previous_generation_id is None:
            raise ParameterAuthorityError("active generation has no governed rollback target")
        current_descriptor = self.registry.descriptor(module_id)
        if current_descriptor.generation_id != pointer.generation_id:
            raise ParameterAuthorityError("registered live generation disagrees with active pointer")
        current_inventory = self.registry.capture_inventory(exact_value_hashes=True, require_complete=True)
        before_manifest = current_inventory.module(module_id)
        if before_manifest.manifest_id != pointer.parameter_manifest_id:
            raise ParameterAuthorityError("live module manifest disagrees with active pointer")

        rollback_snapshot = self.store.read_generation_snapshot_record(
            module_id,
            pointer.previous_generation_id,
            pointer.rollback_snapshot_id,
        )
        rollback_payload = self.store.load_verified_generation_snapshot(rollback_snapshot)
        rollback_descriptor = _descriptor_from_payload(dict(rollback_payload["descriptor"]))
        if not descriptors_share_anatomy(current_descriptor, rollback_descriptor):
            raise ParameterAuthorityError("rollback snapshot does not match live organ anatomy")
        if rollback_descriptor.generation_id != pointer.previous_generation_id:
            raise ParameterAuthorityError("rollback snapshot generation disagrees with active pointer")

        live_module = self.registry.module(module_id)
        probe = copy.deepcopy(live_module)
        probe.load_state_dict(rollback_payload["module_state_dict"], strict=True)
        expected_after = capture_module_manifest(rollback_descriptor, probe, exact_value_hashes=True)
        if expected_after.manifest_id != rollback_snapshot.parameter_manifest_id:
            raise ParameterAuthorityError("rollback snapshot does not reproduce its declared manifest")

        forward_snapshot = self.store.save_generation_snapshot(
            module=live_module,
            descriptor=current_descriptor,
            source_inventory_id=current_inventory.inventory_id,
            previous_pointer_id=pointer.pointer_id,
        )
        forward_payload = self.store.load_verified_generation_snapshot(forward_snapshot)
        committed = False
        registry_transitioned = False
        try:
            live_module.load_state_dict(rollback_payload["module_state_dict"], strict=True)
            after_manifest = capture_module_manifest(rollback_descriptor, live_module, exact_value_hashes=True)
            if after_manifest.manifest_id != rollback_snapshot.parameter_manifest_id:
                raise ParameterAuthorityError("live module failed exact verification after rollback load")
            self.registry.transition_generation(
                module_id,
                expected_generation_id=current_descriptor.generation_id,
                new_descriptor=rollback_descriptor,
            )
            registry_transitioned = True
            restored_pointer = ActiveGenerationPointer(
                module_id=module_id,
                generation_id=rollback_descriptor.generation_id,
                parameter_manifest_id=rollback_snapshot.parameter_manifest_id,
                artifact_relpath=rollback_snapshot.artifact_relpath,
                artifact_sha256=rollback_snapshot.artifact_sha256,
                source_kind="generation_snapshot",
                source_artifact_id=rollback_snapshot.snapshot_id,
                previous_generation_id=current_descriptor.generation_id,
                previous_pointer_id=pointer.pointer_id,
                rollback_snapshot_id=forward_snapshot.snapshot_id,
            )
            self.store.publish_active_pointer(restored_pointer)
            committed = True
        except Exception:
            if not committed:
                live_module.load_state_dict(forward_payload["module_state_dict"], strict=True)
                if registry_transitioned:
                    self.registry.transition_generation(
                        module_id,
                        expected_generation_id=rollback_descriptor.generation_id,
                        new_descriptor=current_descriptor,
                    )
            raise

        receipt = ParameterRollbackReceipt(
            module_id=module_id,
            from_generation_id=current_descriptor.generation_id,
            restored_generation_id=rollback_descriptor.generation_id,
            from_pointer_id=pointer.pointer_id,
            rollback_snapshot_id=rollback_snapshot.snapshot_id,
            active_pointer_id=restored_pointer.pointer_id,
            before_manifest_id=before_manifest.manifest_id,
            after_manifest_id=after_manifest.manifest_id,
        )
        self.store.write_rollback_receipt(receipt)
        return receipt

    def hydrate_active_generation(self, module_id: str) -> ActiveGenerationPointer | None:
        """Rehydrate a freshly registered organ from the durable active pointer."""

        self._require_writer_authority()
        pointer = self.store.read_active_pointer(module_id)
        if pointer is None:
            return None
        snapshot = self.store.read_generation_snapshot_record(
            module_id,
            pointer.generation_id,
            pointer.source_artifact_id,
        )
        if (
            snapshot.parameter_manifest_id != pointer.parameter_manifest_id
            or snapshot.artifact_relpath != pointer.artifact_relpath
            or snapshot.artifact_sha256 != pointer.artifact_sha256
        ):
            raise ParameterAuthorityError("active pointer disagrees with its generation snapshot")
        payload = self.store.load_verified_generation_snapshot(snapshot)
        active_descriptor = _descriptor_from_payload(dict(payload["descriptor"]))
        current_descriptor = self.registry.descriptor(module_id)
        if not descriptors_share_anatomy(current_descriptor, active_descriptor):
            raise ParameterAuthorityError("active generation is incompatible with registered organ anatomy")
        live_module = self.registry.module(module_id)
        probe = copy.deepcopy(live_module)
        probe.load_state_dict(payload["module_state_dict"], strict=True)
        probe_manifest = capture_module_manifest(active_descriptor, probe, exact_value_hashes=True)
        if probe_manifest.manifest_id != pointer.parameter_manifest_id:
            raise ParameterAuthorityError("active generation snapshot fails exact manifest verification")
        original_state = copy.deepcopy(live_module.state_dict())
        original_descriptor = current_descriptor
        try:
            live_module.load_state_dict(payload["module_state_dict"], strict=True)
            hydrated_manifest = capture_module_manifest(active_descriptor, live_module, exact_value_hashes=True)
            if hydrated_manifest.manifest_id != pointer.parameter_manifest_id:
                raise ParameterAuthorityError("hydrated live module differs from active generation pointer")
            self.registry.transition_generation(
                module_id,
                expected_generation_id=current_descriptor.generation_id,
                new_descriptor=active_descriptor,
            )
        except Exception:
            live_module.load_state_dict(original_state, strict=True)
            if self.registry.descriptor(module_id).generation_id != original_descriptor.generation_id:
                self.registry.transition_generation(
                    module_id,
                    expected_generation_id=active_descriptor.generation_id,
                    new_descriptor=original_descriptor,
                )
            raise
        return pointer


__all__ = ["TrainerControlPlane"]
