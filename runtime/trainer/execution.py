"""Isolated governed optimizer execution for Axon's Trainer organ.

The live registered module is never optimized in place.  An authorized plan
creates a candidate clone, freezes every tensor outside the authorized scope,
and checks after each step that both the live base and unauthorized candidate
tensors remain byte-identical.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .authority import AuthorizedParameterMutation, ParameterAuthorityError
from .contracts import ParameterInventory, ParameterModuleDescriptor, ParameterMutationPlan
from .lifecycle import CandidateCheckpointRecord, CandidateLifecycleEvent, CandidateStatus, OptimizationStepReceipt
from .registry import capture_module_manifest, parameter_value_sha256
from .store import TrainerStateStore
from .telemetry import capture_parameter_telemetry


class TrainerExecutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class OptimizerExecutionPolicy:
    gradient_clip_norm: float | None = 1.0
    exact_scope_verification_each_step: bool = True
    full_parameter_telemetry_each_step: bool = True

    def __post_init__(self) -> None:
        if self.gradient_clip_norm is not None:
            clip = float(self.gradient_clip_norm)
            if not (0.0 < clip < float("inf")):
                raise ValueError("gradient_clip_norm must be positive and finite")
            object.__setattr__(self, "gradient_clip_norm", clip)


def _candidate_descriptor(base: ParameterModuleDescriptor, candidate_generation_id: str) -> ParameterModuleDescriptor:
    return ParameterModuleDescriptor(
        module_id=base.module_id,
        organ_kind=base.organ_kind,
        generation_id=candidate_generation_id,
        architecture=base.architecture,
        d_model=base.d_model,
        trainer_core_role=base.trainer_core_role,
        tags=base.tags,
    )


def _optimizer_for(plan: ParameterMutationPlan, parameters: list[nn.Parameter]) -> torch.optim.Optimizer:
    name = plan.optimizer_name.strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=plan.learning_rate)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=plan.learning_rate)
    raise TrainerExecutionError(f"unsupported governed optimizer {plan.optimizer_name!r}; supported: AdamW, SGD")


class CandidateOptimizationSession:
    """One isolated parameter candidate governed by an authorized mutation plan."""

    def __init__(
        self,
        *,
        live_module: nn.Module,
        base_descriptor: ParameterModuleDescriptor,
        base_inventory: ParameterInventory,
        plan: ParameterMutationPlan,
        authorization: AuthorizedParameterMutation,
        store: TrainerStateStore,
        policy: OptimizerExecutionPolicy | None = None,
    ) -> None:
        if not isinstance(live_module, nn.Module):
            raise TypeError("live_module must be torch.nn.Module")
        if not isinstance(base_descriptor, ParameterModuleDescriptor):
            raise TypeError("base_descriptor must be ParameterModuleDescriptor")
        if not isinstance(base_inventory, ParameterInventory):
            raise TypeError("base_inventory must be ParameterInventory")
        if not isinstance(plan, ParameterMutationPlan):
            raise TypeError("plan must be ParameterMutationPlan")
        if not isinstance(authorization, AuthorizedParameterMutation):
            raise TypeError("authorization must be AuthorizedParameterMutation")
        if not isinstance(store, TrainerStateStore):
            raise TypeError("store must be TrainerStateStore")
        if not base_inventory.complete:
            raise TrainerExecutionError("candidate session requires a complete base inventory")
        if authorization.inventory_id != base_inventory.inventory_id or plan.base_inventory_id != base_inventory.inventory_id:
            raise TrainerExecutionError("candidate session inventory lineage mismatch")
        if authorization.plan_id != plan.plan_id:
            raise TrainerExecutionError("candidate session authorization does not match plan")
        if authorization.module_id != base_descriptor.module_id or plan.module_id != base_descriptor.module_id:
            raise TrainerExecutionError("candidate session module lineage mismatch")
        if authorization.base_generation_id != base_descriptor.generation_id or plan.base_generation_id != base_descriptor.generation_id:
            raise TrainerExecutionError("candidate session base generation mismatch")
        if authorization.candidate_generation_id != plan.candidate_generation_id:
            raise TrainerExecutionError("candidate session candidate generation mismatch")
        if tuple(authorization.tensor_names) != tuple(plan.tensor_names):
            raise TrainerExecutionError("candidate session authorized tensor scope differs from plan")

        base_manifest = base_inventory.module(base_descriptor.module_id)
        if base_manifest.descriptor.to_canonical_dict() != base_descriptor.to_canonical_dict():
            raise TrainerExecutionError("base descriptor disagrees with inventory")
        if any(item.value_sha256 is None for item in base_manifest.tensors):
            raise TrainerExecutionError("candidate session requires exact value hashes in base inventory")

        self.live_module = live_module
        self.base_descriptor = base_descriptor
        self.candidate_descriptor = _candidate_descriptor(base_descriptor, plan.candidate_generation_id)
        self.base_inventory = base_inventory
        self.plan = plan
        self.authorization = authorization
        self.store = store
        self.policy = policy or OptimizerExecutionPolicy()
        self.candidate_module = copy.deepcopy(live_module)
        self.step_index = 0
        self._closed = False
        self._previous_lifecycle_event_id: str | None = None
        self._previous_checkpoint_id: str | None = None

        self._base_hashes = {item.name: item.value_sha256 for item in base_manifest.tensors}
        self._base_buffer_hashes = {item.name: item.value_sha256 for item in base_manifest.buffers}
        if any(value is None for value in self._base_buffer_hashes.values()):
            raise TrainerExecutionError("candidate session requires exact persistent-buffer hashes in base inventory")
        live_names = {name for name, _ in live_module.named_parameters(recurse=True)}
        if live_names != set(self._base_hashes):
            raise TrainerExecutionError("live module parameter names disagree with base inventory")
        live_buffer_names = {name for name, _ in live_module.named_buffers(recurse=True)}
        if live_buffer_names != set(self._base_buffer_hashes):
            raise TrainerExecutionError("live module persistent-buffer names disagree with base inventory")
        candidate_named = dict(self.candidate_module.named_parameters(recurse=True))
        if set(candidate_named) != set(self._base_hashes):
            raise TrainerExecutionError("candidate clone parameter names disagree with base inventory")
        candidate_buffer_names = {name for name, _ in self.candidate_module.named_buffers(recurse=True)}
        if candidate_buffer_names != set(self._base_buffer_hashes):
            raise TrainerExecutionError("candidate clone persistent-buffer names disagree with base inventory")

        authorized_names = set(authorization.tensor_names)
        self._unauthorized_names = tuple(sorted(set(candidate_named) - authorized_names))
        for name, parameter in candidate_named.items():
            parameter.requires_grad_(name in authorized_names)
        selected = [candidate_named[name] for name in authorization.tensor_names]
        if not selected:
            raise TrainerExecutionError("authorized candidate contains no trainable parameters")
        self.optimizer = _optimizer_for(plan, selected)
        self._emit_lifecycle(CandidateStatus.PREPARED, step=0, reason="isolated candidate clone prepared")

    @property
    def closed(self) -> bool:
        return self._closed

    def _assert_open(self) -> None:
        if self._closed:
            raise TrainerExecutionError("candidate session is closed")

    def _assert_live_base_unchanged(self) -> None:
        current = dict(self.live_module.named_parameters(recurse=True))
        if set(current) != set(self._base_hashes):
            raise TrainerExecutionError("live module parameter surface changed during candidate session")
        changed = [name for name, parameter in current.items() if parameter_value_sha256(parameter) != self._base_hashes[name]]
        if changed:
            raise TrainerExecutionError(f"live module changed during isolated candidate training: {changed}")
        buffers = dict(self.live_module.named_buffers(recurse=True))
        if set(buffers) != set(self._base_buffer_hashes):
            raise TrainerExecutionError("live module persistent-buffer surface changed during candidate session")
        changed_buffers = [
            name for name, buffer in buffers.items() if parameter_value_sha256(buffer) != self._base_buffer_hashes[name]
        ]
        if changed_buffers:
            raise TrainerExecutionError(f"live persistent buffers changed during candidate training: {changed_buffers}")

    def _hash_candidate(self) -> dict[str, str]:
        return {name: parameter_value_sha256(parameter) for name, parameter in self.candidate_module.named_parameters(recurse=True)}

    def _hash_candidate_buffers(self) -> dict[str, str]:
        return {name: parameter_value_sha256(buffer) for name, buffer in self.candidate_module.named_buffers(recurse=True)}

    def _emit_lifecycle(
        self,
        status: CandidateStatus,
        *,
        step: int,
        checkpoint_id: str | None = None,
        gate_decision_id: str | None = None,
        reason: str | None = None,
    ) -> CandidateLifecycleEvent:
        event = CandidateLifecycleEvent(
            module_id=self.base_descriptor.module_id,
            base_generation_id=self.base_descriptor.generation_id,
            candidate_generation_id=self.candidate_descriptor.generation_id,
            plan_id=self.plan.plan_id,
            authorization_id=self.authorization.authorization_id,
            status=status,
            step=step,
            previous_event_id=self._previous_lifecycle_event_id,
            checkpoint_id=checkpoint_id,
            gate_decision_id=gate_decision_id,
            reason=reason,
        )
        self.store.append_candidate_lifecycle(event)
        self._previous_lifecycle_event_id = event.event_id
        return event

    def step(self, loss_fn: Callable[[nn.Module], torch.Tensor]) -> OptimizationStepReceipt:
        """Run one governed optimizer step on the isolated candidate only."""

        self._assert_open()
        if self.step_index >= self.plan.max_steps:
            raise TrainerExecutionError("candidate session reached its authorized max_steps")
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        self._assert_live_base_unchanged()
        before_hashes = self._hash_candidate() if self.policy.exact_scope_verification_each_step else {}
        before_buffer_hashes = self._hash_candidate_buffers() if self.policy.exact_scope_verification_each_step else {}

        self.optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(self.candidate_module)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise TrainerExecutionError("loss_fn must return one scalar torch.Tensor")
        loss_value = float(loss.detach().item())
        if not math.isfinite(loss_value):
            self.optimizer.zero_grad(set_to_none=True)
            self._emit_lifecycle(CandidateStatus.REJECTED, step=self.step_index, reason="non-finite loss")
            self._closed = True
            raise TrainerExecutionError("non-finite loss rejected before backpropagation")
        loss.backward()
        if self.policy.exact_scope_verification_each_step:
            forward_buffer_hashes = self._hash_candidate_buffers()
            changed_buffers = [
                name for name in forward_buffer_hashes if forward_buffer_hashes[name] != before_buffer_hashes[name]
            ]
            if changed_buffers:
                self.optimizer.zero_grad(set_to_none=True)
                self._emit_lifecycle(
                    CandidateStatus.REJECTED,
                    step=self.step_index,
                    reason="forward/backward mutated ungranted persistent buffers",
                )
                self._closed = True
                raise ParameterAuthorityError(
                    f"candidate changed persistent buffers without an explicit buffer-state grant: {changed_buffers}"
                )

        selected = [dict(self.candidate_module.named_parameters(recurse=True))[name] for name in self.authorization.tensor_names]
        gradients = [parameter.grad for parameter in selected if parameter.grad is not None]
        if any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
            self.optimizer.zero_grad(set_to_none=True)
            self._emit_lifecycle(CandidateStatus.REJECTED, step=self.step_index, reason="non-finite gradient")
            self._closed = True
            raise TrainerExecutionError("non-finite gradient rejected before optimizer step")
        grad_l2 = float(
            math.sqrt(sum(float(torch.sum(gradient.detach().float() ** 2).item()) for gradient in gradients))
        ) if gradients else 0.0

        if self.policy.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(selected, self.policy.gradient_clip_norm)
        self.optimizer.step()
        self.step_index += 1

        after_hashes = self._hash_candidate() if self.policy.exact_scope_verification_each_step else {}
        after_buffer_hashes = self._hash_candidate_buffers() if self.policy.exact_scope_verification_each_step else {}
        if self.policy.exact_scope_verification_each_step:
            changed_buffers = [
                name for name in after_buffer_hashes if after_buffer_hashes[name] != before_buffer_hashes[name]
            ]
            if changed_buffers:
                self._emit_lifecycle(
                    CandidateStatus.REJECTED,
                    step=self.step_index,
                    reason="ungranted persistent buffers changed",
                )
                self._closed = True
                raise ParameterAuthorityError(
                    f"candidate changed persistent buffers without an explicit buffer-state grant: {changed_buffers}"
                )
            unauthorized_changed = [name for name in self._unauthorized_names if after_hashes[name] != before_hashes[name]]
            if unauthorized_changed:
                self._emit_lifecycle(
                    CandidateStatus.REJECTED,
                    step=self.step_index,
                    reason="unauthorized candidate parameters changed",
                )
                self._closed = True
                raise ParameterAuthorityError(
                    f"optimizer changed parameters outside authorized scope: {unauthorized_changed}"
                )
            changed = tuple(sorted(name for name in self.authorization.tensor_names if after_hashes[name] != before_hashes[name]))
            unchanged = tuple(sorted(name for name in after_hashes if after_hashes[name] == before_hashes[name]))
        else:
            changed = tuple()
            unchanged = tuple()

        self._assert_live_base_unchanged()
        frame = capture_parameter_telemetry(
            self.candidate_module,
            module_id=self.base_descriptor.module_id,
            generation_id=self.candidate_descriptor.generation_id,
            step=self.step_index,
            inventory_id=self.base_inventory.inventory_id,
        )
        payload = frame.to_canonical_dict()
        if not payload["all_values_finite"] or not payload["all_present_gradients_finite"]:
            self._emit_lifecycle(CandidateStatus.REJECTED, step=self.step_index, reason="non-finite post-step telemetry")
            self._closed = True
            raise TrainerExecutionError("candidate telemetry became non-finite")
        if self.policy.full_parameter_telemetry_each_step:
            self.store.append_telemetry(frame)

        receipt = OptimizationStepReceipt(
            module_id=self.base_descriptor.module_id,
            candidate_generation_id=self.candidate_descriptor.generation_id,
            plan_id=self.plan.plan_id,
            authorization_id=self.authorization.authorization_id,
            step=self.step_index,
            loss=loss_value,
            gradient_l2=grad_l2,
            gradient_clip_norm=self.policy.gradient_clip_norm,
            telemetry_frame_id=frame.frame_id,
            changed_tensor_names=changed,
            unchanged_tensor_names=unchanged,
        )
        self.store.append_optimization_step(receipt)
        self._emit_lifecycle(CandidateStatus.RUNNING, step=self.step_index, reason="governed optimizer step completed")
        return receipt

    def checkpoint(self, *, include_optimizer: bool = True) -> CandidateCheckpointRecord:
        self._assert_open()
        self._assert_live_base_unchanged()
        record = self.store.save_candidate_checkpoint(
            module=self.candidate_module,
            descriptor=self.candidate_descriptor,
            base_generation_id=self.base_descriptor.generation_id,
            plan_id=self.plan.plan_id,
            authorization_id=self.authorization.authorization_id,
            step=self.step_index,
            optimizer=self.optimizer if include_optimizer else None,
            previous_checkpoint_id=self._previous_checkpoint_id,
        )
        self._previous_checkpoint_id = record.checkpoint_id
        return record

    def restore_checkpoint(self, record: CandidateCheckpointRecord) -> None:
        self._assert_open()
        if record.module_id != self.base_descriptor.module_id or record.candidate_generation_id != self.candidate_descriptor.generation_id:
            raise TrainerExecutionError("checkpoint belongs to another candidate generation")
        if record.plan_id != self.plan.plan_id or record.authorization_id != self.authorization.authorization_id:
            raise TrainerExecutionError("checkpoint plan/authorization lineage mismatch")
        payload = self.store.load_verified_candidate_checkpoint(record)
        self.candidate_module.load_state_dict(payload["module_state_dict"], strict=True)
        if record.optimizer_included and payload["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.step_index = record.step
        self._previous_checkpoint_id = record.checkpoint_id
        self._assert_live_base_unchanged()

    def complete(self, *, reason: str = "candidate optimization completed") -> CandidateLifecycleEvent:
        self._assert_open()
        self._assert_live_base_unchanged()
        event = self._emit_lifecycle(CandidateStatus.COMPLETED, step=self.step_index, reason=reason)
        self._closed = True
        return event

    def reject(self, *, reason: str) -> CandidateLifecycleEvent:
        self._assert_open()
        self._assert_live_base_unchanged()
        event = self._emit_lifecycle(CandidateStatus.REJECTED, step=self.step_index, reason=reason)
        self._closed = True
        return event

    def candidate_manifest(self):
        return capture_module_manifest(self.candidate_descriptor, self.candidate_module, exact_value_hashes=True)


__all__ = [
    "TrainerExecutionError",
    "OptimizerExecutionPolicy",
    "CandidateOptimizationSession",
]
