"""Isolated governed optimizer execution for Axon's Trainer organ.

The live registered module is never optimized in place. An authorized plan
creates a candidate clone, freezes every tensor outside the authorized scope,
and executes only through an immutable governed learning policy. Optimizer
updates, gradient accumulation, precision, clipping/budgets, checkpoint state,
and telemetry are explicit rather than ad-hoc trainer arguments.
"""
from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import Callable

import torch
from torch import nn

from .authority import AuthorizedParameterMutation, ParameterAuthorityError
from .contracts import (
    ParameterInventory,
    ParameterModuleDescriptor,
    ParameterMutationPlan,
    ParameterMutationPlanLike,
    ParameterMutationPlanV2,
    is_parameter_mutation_plan,
)
from .learning import GovernedLearningPolicy, OptimizerKind, PrecisionMode
from .lifecycle import (
    CandidateCheckpointRecord,
    CandidateLifecycleEvent,
    CandidateStatus,
    LearningMicrostepReceipt,
    OptimizationStepReceipt,
)
from .registry import capture_module_manifest, parameter_value_sha256
from .store import TrainerStateStore
from .telemetry import capture_parameter_telemetry
from .tranche import ResourceTranche


class TrainerExecutionError(RuntimeError):
    pass


# Backward-compatible public name retained while the permanent policy contract
# moves into runtime/trainer/learning.py.
OptimizerExecutionPolicy = GovernedLearningPolicy


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


def _optimizer_for(policy: GovernedLearningPolicy, parameters: list[nn.Parameter], initial_lr: float) -> torch.optim.Optimizer:
    if policy.optimizer is OptimizerKind.ADAMW:
        return torch.optim.AdamW(
            parameters,
            lr=initial_lr,
            weight_decay=policy.weight_decay,
            betas=(policy.adam_beta1, policy.adam_beta2),
            eps=policy.adam_eps,
        )
    if policy.optimizer is OptimizerKind.SGD:
        return torch.optim.SGD(
            parameters,
            lr=initial_lr,
            weight_decay=policy.weight_decay,
            momentum=policy.sgd_momentum,
        )
    raise TrainerExecutionError(f"unsupported governed optimizer {policy.optimizer.value!r}")


class CandidateOptimizationSession:
    """One isolated parameter candidate governed by an authorized mutation plan."""

    def __init__(
        self,
        *,
        live_module: nn.Module,
        base_descriptor: ParameterModuleDescriptor,
        base_inventory: ParameterInventory,
        plan: ParameterMutationPlanLike,
        authorization: AuthorizedParameterMutation,
        store: TrainerStateStore,
        policy: GovernedLearningPolicy | None = None,
        tranche: ResourceTranche | None = None,
    ) -> None:
        if not isinstance(live_module, nn.Module):
            raise TypeError("live_module must be torch.nn.Module")
        if not isinstance(base_descriptor, ParameterModuleDescriptor):
            raise TypeError("base_descriptor must be ParameterModuleDescriptor")
        if not isinstance(base_inventory, ParameterInventory):
            raise TypeError("base_inventory must be ParameterInventory")
        if not is_parameter_mutation_plan(plan):
            raise TypeError("plan must be a governed parameter mutation plan")
        if not isinstance(authorization, AuthorizedParameterMutation):
            raise TypeError("authorization must be AuthorizedParameterMutation")
        if not isinstance(store, TrainerStateStore):
            raise TypeError("store must be TrainerStateStore")
        if tranche is not None and not isinstance(tranche, ResourceTranche):
            raise TypeError("tranche must be ResourceTranche or None")
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

        if isinstance(plan, ParameterMutationPlanV2) and policy is None:
            raise TrainerExecutionError(
                "resource-independent v2 plans require an explicit learning policy"
            )
        learning_policy = policy or GovernedLearningPolicy.from_plan(plan)
        if not isinstance(learning_policy, GovernedLearningPolicy):
            raise TypeError("policy must be GovernedLearningPolicy")
        try:
            learning_policy.assert_plan_compatible(plan)
        except ValueError as exc:
            raise TrainerExecutionError(str(exc)) from exc

        self.live_module = live_module
        self.base_descriptor = base_descriptor
        self.candidate_descriptor = _candidate_descriptor(base_descriptor, plan.candidate_generation_id)
        self.base_inventory = base_inventory
        self.plan = plan
        self.authorization = authorization
        self.store = store
        self.policy = learning_policy
        # Renewable resource tranche (v2 law): execution allowance only.  A
        # tranche never participates in plan/policy/candidate identity.  It
        # may lawfully extend execution beyond the plan's historical envelope
        # when the session was restored from an exact accepted parent bundle.
        if tranche is not None and tranche.module_id != plan.module_id:
            raise TrainerExecutionError("tranche module differs from mutation plan")
        if tranche is not None and tranche.candidate_generation_id != plan.candidate_generation_id:
            raise TrainerExecutionError("tranche candidate generation differs from mutation plan")
        if tranche is not None and tranche.plan_id != plan.plan_id:
            raise TrainerExecutionError("tranche plan differs from mutation plan")
        if tranche is not None and tranche.learning_policy_id != learning_policy.policy_id:
            raise TrainerExecutionError("tranche learning policy differs from candidate session")
        self.tranche = tranche
        self.candidate_module = copy.deepcopy(live_module)
        self.step_index = 0  # completed optimizer updates
        self.micro_step_index = 0
        self.accumulation_index = 0
        self.accumulated_loss_sum = 0.0
        self._closed = False
        self._previous_lifecycle_event_id: str | None = None
        self._previous_checkpoint_id: str | None = None
        self._restored_parent_record: CandidateCheckpointRecord | None = None

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
        self._selected_names = tuple(authorization.tensor_names)
        self._selected = [candidate_named[name] for name in self._selected_names]
        if not self._selected:
            raise TrainerExecutionError("authorized candidate contains no trainable parameters")

        device_types = {parameter.device.type for parameter in self._selected}
        if len(device_types) != 1:
            raise TrainerExecutionError("first-form governed learning requires authorized parameters on one device type")
        self._device_type = next(iter(device_types))
        if self.policy.precision is PrecisionMode.FP16 and self._device_type != "cuda":
            raise TrainerExecutionError("fp16 governed precision currently requires CUDA")
        if self.policy.precision is PrecisionMode.BF16 and self._device_type not in {"cpu", "cuda"}:
            raise TrainerExecutionError("bf16 governed precision currently supports CPU or CUDA only")

        legacy_horizon = self.plan.max_steps if isinstance(self.plan, ParameterMutationPlan) else None
        initial_lr = self.policy.learning_rate_for_step(0, legacy_horizon)
        self.optimizer = _optimizer_for(self.policy, self._selected, initial_lr)
        self._scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
            if self.policy.precision is PrecisionMode.FP16
            else None
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.store.write_learning_policy(self.policy)
        self._emit_lifecycle(CandidateStatus.PREPARED, step=0, reason="isolated candidate clone prepared under governed learning policy")

    @property
    def closed(self) -> bool:
        return self._closed

    def _restored_global_step(self) -> int:
        """The accepted global step this session was restored from, or -1."""

        record = self._restored_parent_record
        if record is None:
            return -1
        return record.step

    def _assert_tranche_admits(self) -> None:
        if self.tranche is not None and not self.tranche.admits_step(self.step_index + 1):
            raise TrainerExecutionError(
                "resource tranche does not admit the next optimizer step; "
                "checkpoint and pause for a later tranche"
            )

    @property
    def current_learning_rate(self) -> float:
        if not self.optimizer.param_groups:
            raise TrainerExecutionError("optimizer has no parameter groups")
        return float(self.optimizer.param_groups[0]["lr"])

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
        changed_buffers = [name for name, buffer in buffers.items() if parameter_value_sha256(buffer) != self._base_buffer_hashes[name]]
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

    def _autocast_context(self):
        if self.policy.precision is PrecisionMode.FP32:
            return nullcontext()
        dtype = torch.float16 if self.policy.precision is PrecisionMode.FP16 else torch.bfloat16
        return torch.autocast(device_type=self._device_type, dtype=dtype)

    def _set_lr_for_next_update(self) -> float:
        legacy_horizon = self.plan.max_steps if isinstance(self.plan, ParameterMutationPlan) else None
        lr = self.policy.learning_rate_for_step(self.step_index, legacy_horizon)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _gradient_state(self) -> dict[str, torch.Tensor | None]:
        named = dict(self.candidate_module.named_parameters(recurse=True))
        return {
            name: None if named[name].grad is None else named[name].grad.detach().clone()
            for name in self._selected_names
        }

    def _restore_gradient_state(self, state: dict[str, torch.Tensor | None]) -> None:
        named = dict(self.candidate_module.named_parameters(recurse=True))
        if set(state) != set(self._selected_names):
            raise TrainerExecutionError("checkpoint gradient-state surface differs from authorized tensor scope")
        for name in self._selected_names:
            value = state[name]
            if value is None:
                named[name].grad = None
            else:
                named[name].grad = value.to(device=named[name].device, dtype=named[name].dtype).clone()

    def _reject(self, reason: str, *, step: int | None = None) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self._emit_lifecycle(CandidateStatus.REJECTED, step=self.step_index if step is None else step, reason=reason)
        self._closed = True

    def micro_step(
        self,
        loss_fn: Callable[[nn.Module], torch.Tensor],
    ) -> tuple[LearningMicrostepReceipt, OptimizationStepReceipt | None]:
        """Accumulate one governed microbatch; update only at the policy boundary."""

        self._assert_open()
        if self.tranche is not None:
            if self.tranche.base_global_step > 0 and self._restored_global_step() != self.tranche.base_global_step:
                raise TrainerExecutionError(
                    "nonzero-base tranche requires an exact parent checkpoint restore "
                    f"at global step {self.tranche.base_global_step}"
                )
            self._assert_tranche_admits()
        elif isinstance(self.plan, ParameterMutationPlanV2):
            raise TrainerExecutionError(
                "resource-independent v2 plans require a renewable resource tranche"
            )
        elif self.step_index >= self.plan.max_steps:
            raise TrainerExecutionError("candidate session reached its authorized max_steps")
        if not callable(loss_fn):
            raise TypeError("loss_fn must be callable")
        self._assert_live_base_unchanged()

        if self.accumulation_index == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self.accumulated_loss_sum = 0.0
        before_hashes = self._hash_candidate() if self.policy.exact_scope_verification_each_step else {}
        before_buffer_hashes = self._hash_candidate_buffers() if self.policy.exact_scope_verification_each_step else {}

        with self._autocast_context():
            loss = loss_fn(self.candidate_module)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise TrainerExecutionError("loss_fn must return one scalar torch.Tensor")
        loss_value = float(loss.detach().float().item())
        if not math.isfinite(loss_value):
            self._reject("non-finite loss")
            raise TrainerExecutionError("non-finite loss rejected before backpropagation")
        scaled_loss = loss / float(self.policy.gradient_accumulation_steps)
        if self._scaler is not None:
            self._scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if self.policy.exact_scope_verification_each_step:
            forward_hashes = self._hash_candidate()
            changed_parameters = [name for name in forward_hashes if forward_hashes[name] != before_hashes[name]]
            if changed_parameters:
                self._reject("forward/backward mutated parameter values before optimizer step")
                raise ParameterAuthorityError(f"candidate forward/backward mutated parameter values: {changed_parameters}")
            forward_buffer_hashes = self._hash_candidate_buffers()
            changed_buffers = [name for name in forward_buffer_hashes if forward_buffer_hashes[name] != before_buffer_hashes[name]]
            if changed_buffers:
                self._reject("forward/backward mutated ungranted persistent buffers")
                raise ParameterAuthorityError(
                    f"candidate changed persistent buffers without an explicit buffer-state grant: {changed_buffers}"
                )

        self.micro_step_index += 1
        self.accumulation_index += 1
        self.accumulated_loss_sum += loss_value
        micro_receipt = LearningMicrostepReceipt(
            module_id=self.base_descriptor.module_id,
            candidate_generation_id=self.candidate_descriptor.generation_id,
            plan_id=self.plan.plan_id,
            authorization_id=self.authorization.authorization_id,
            learning_policy_id=self.policy.policy_id,
            micro_step=self.micro_step_index,
            optimizer_step_before=self.step_index,
            accumulation_index=self.accumulation_index,
            accumulation_target=self.policy.gradient_accumulation_steps,
            loss=loss_value,
            scaled_loss=float(loss_value / float(self.policy.gradient_accumulation_steps)),
            precision_mode=self.policy.precision.value,
        )
        self.store.append_learning_microstep(micro_receipt)

        if self.accumulation_index < self.policy.gradient_accumulation_steps:
            self._emit_lifecycle(
                CandidateStatus.RUNNING,
                step=self.step_index,
                reason=f"accumulated microstep {self.accumulation_index}/{self.policy.gradient_accumulation_steps}",
            )
            return micro_receipt, None

        return micro_receipt, self._apply_optimizer_step()

    def _apply_optimizer_step(self) -> OptimizationStepReceipt:
        if self._scaler is not None:
            self._scaler.unscale_(self.optimizer)
        gradients = [parameter.grad for parameter in self._selected if parameter.grad is not None]
        if any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
            self._reject("non-finite gradient")
            raise TrainerExecutionError("non-finite gradient rejected before optimizer step")
        grad_l2 = float(math.sqrt(sum(float(torch.sum(gradient.detach().float() ** 2).item()) for gradient in gradients))) if gradients else 0.0
        if self.policy.max_gradient_l2 is not None and grad_l2 > self.policy.max_gradient_l2:
            self._reject("gradient L2 exceeded governed budget")
            raise ParameterAuthorityError(
                f"gradient L2 {grad_l2} exceeds governed maximum {self.policy.max_gradient_l2}"
            )
        if self.policy.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._selected, self.policy.gradient_clip_norm)

        lr = self._set_lr_for_next_update()
        before_hashes = self._hash_candidate() if self.policy.exact_scope_verification_each_step else {}
        before_buffer_hashes = self._hash_candidate_buffers() if self.policy.exact_scope_verification_each_step else {}
        update_before = None
        if self.policy.max_update_l2 is not None:
            update_before = [parameter.detach().clone() for parameter in self._selected]

        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()
        self.step_index += 1

        update_l2: float | None = None
        if update_before is not None:
            squared = 0.0
            for previous, parameter in zip(update_before, self._selected, strict=True):
                squared += float(torch.sum((parameter.detach().float() - previous.detach().float()) ** 2).item())
            update_l2 = float(math.sqrt(squared))
            if update_l2 > float(self.policy.max_update_l2):
                with torch.no_grad():
                    for previous, parameter in zip(
                        update_before,
                        self._selected,
                        strict=True,
                    ):
                        parameter.copy_(previous)
                self._reject("parameter update L2 exceeded governed budget", step=self.step_index)
                raise ParameterAuthorityError(
                    f"parameter update L2 {update_l2} exceeds governed maximum {self.policy.max_update_l2}"
                )

        if self.policy.exact_scope_verification_each_step:
            after_hashes = self._hash_candidate()
            after_buffer_hashes = self._hash_candidate_buffers()
            changed_buffers = [name for name in after_buffer_hashes if after_buffer_hashes[name] != before_buffer_hashes[name]]
            if changed_buffers:
                self._reject("optimizer mutated ungranted persistent buffers", step=self.step_index)
                raise ParameterAuthorityError(
                    f"candidate changed persistent buffers without an explicit buffer-state grant: {changed_buffers}"
                )
            unauthorized_changed = [name for name in self._unauthorized_names if after_hashes[name] != before_hashes[name]]
            if unauthorized_changed:
                self._reject("optimizer changed parameters outside authorized scope", step=self.step_index)
                raise ParameterAuthorityError(f"optimizer changed parameters outside authorized scope: {unauthorized_changed}")
            changed = tuple(sorted(name for name in self._selected_names if after_hashes[name] != before_hashes[name]))
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
            self._reject("non-finite post-step telemetry", step=self.step_index)
            raise TrainerExecutionError("candidate telemetry became non-finite")
        if self.policy.full_parameter_telemetry_each_step:
            self.store.append_telemetry(frame)

        mean_loss = self.accumulated_loss_sum / float(self.policy.gradient_accumulation_steps)
        receipt = OptimizationStepReceipt(
            module_id=self.base_descriptor.module_id,
            candidate_generation_id=self.candidate_descriptor.generation_id,
            plan_id=self.plan.plan_id,
            authorization_id=self.authorization.authorization_id,
            learning_policy_id=self.policy.policy_id,
            step=self.step_index,
            micro_step=self.micro_step_index,
            microbatches_accumulated=self.policy.gradient_accumulation_steps,
            loss=mean_loss,
            gradient_l2=grad_l2,
            gradient_clip_norm=self.policy.gradient_clip_norm,
            learning_rate=lr,
            weight_decay=self.policy.weight_decay,
            precision_mode=self.policy.precision.value,
            update_l2=update_l2,
            telemetry_frame_id=frame.frame_id,
            changed_tensor_names=changed,
            unchanged_tensor_names=unchanged,
        )
        self.store.append_optimization_step(receipt)
        self._emit_lifecycle(CandidateStatus.RUNNING, step=self.step_index, reason="governed optimizer step completed")
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulation_index = 0
        self.accumulated_loss_sum = 0.0
        return receipt

    def step(self, loss_fn: Callable[[nn.Module], torch.Tensor]) -> OptimizationStepReceipt:
        """Backward-compatible one-call optimizer step for accumulation=1 policies."""

        if self.policy.gradient_accumulation_steps != 1:
            raise TrainerExecutionError("step() requires gradient_accumulation_steps=1; use micro_step() for accumulation")
        _micro, optimizer_receipt = self.micro_step(loss_fn)
        if optimizer_receipt is None:
            raise TrainerExecutionError("internal error: accumulation=1 did not produce an optimizer step")
        return optimizer_receipt

    def checkpoint(self, *, include_optimizer: bool = True) -> CandidateCheckpointRecord:
        self._assert_open()
        self._assert_live_base_unchanged()
        if self.accumulation_index and not include_optimizer:
            raise TrainerExecutionError("mid-accumulation checkpoints require optimizer state for exact resume")
        scaler_state = None if self._scaler is None else self._scaler.state_dict()
        record = self.store.save_candidate_checkpoint(
            module=self.candidate_module,
            descriptor=self.candidate_descriptor,
            base_generation_id=self.base_descriptor.generation_id,
            plan_id=self.plan.plan_id,
            authorization_id=self.authorization.authorization_id,
            learning_policy=self.policy,
            step=self.step_index,
            micro_step=self.micro_step_index,
            accumulation_index=self.accumulation_index,
            current_learning_rate=self.current_learning_rate,
            accumulated_loss_sum=self.accumulated_loss_sum,
            optimizer=self.optimizer if include_optimizer else None,
            scaler_state=scaler_state,
            gradient_state=self._gradient_state(),
            previous_checkpoint_id=self._previous_checkpoint_id,
        )
        self._previous_checkpoint_id = record.checkpoint_id
        self.store.prune_candidate_checkpoints(record.module_id, record.candidate_generation_id)
        return record

    def restore_checkpoint(self, record: CandidateCheckpointRecord) -> None:
        self._assert_open()
        if record.module_id != self.base_descriptor.module_id or record.candidate_generation_id != self.candidate_descriptor.generation_id:
            raise TrainerExecutionError("checkpoint belongs to another candidate generation")
        if record.plan_id != self.plan.plan_id:
            raise TrainerExecutionError("checkpoint plan lineage mismatch")
        if record.authorization_id != self.authorization.authorization_id:
            prior_authorization = self.store.read_authorization(record.authorization_id)
            if prior_authorization.resume_scope() != self.authorization.resume_scope():
                raise TrainerExecutionError("checkpoint authorization resume scope mismatch")
        if record.learning_policy_id != self.policy.policy_id:
            raise TrainerExecutionError("checkpoint learning-policy lineage mismatch")
        if (record.step > 0 or record.accumulation_index > 0) and not record.optimizer_included:
            raise TrainerExecutionError("exact resume after learning has begun requires optimizer state")
        if record.accumulation_index > 0 and not record.gradient_state_included:
            raise TrainerExecutionError("mid-accumulation resume requires pending gradient state")
        if self._scaler is not None and not record.scaler_included:
            raise TrainerExecutionError("fp16 exact resume requires AMP scaler state")

        payload = self.store.load_verified_candidate_checkpoint(record)
        self.candidate_module.load_state_dict(payload["module_state_dict"], strict=True)
        if record.optimizer_included and payload["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        gradients = payload.get("gradient_state_dict")
        if gradients is not None:
            self._restore_gradient_state(dict(gradients))
        else:
            self.optimizer.zero_grad(set_to_none=True)
        if self._scaler is not None and payload.get("scaler_state_dict") is not None:
            self._scaler.load_state_dict(payload["scaler_state_dict"])
        self.step_index = record.step
        self.micro_step_index = record.micro_step
        self.accumulation_index = record.accumulation_index
        self.accumulated_loss_sum = record.accumulated_loss_sum
        self._previous_checkpoint_id = record.checkpoint_id
        self._restored_parent_record = record
        self._assert_live_base_unchanged()

    def complete(self, *, reason: str = "candidate optimization completed") -> CandidateLifecycleEvent:
        self._assert_open()
        self._assert_live_base_unchanged()
        if self.accumulation_index:
            raise TrainerExecutionError("cannot complete candidate with uncommitted gradient accumulation")
        event = self._emit_lifecycle(CandidateStatus.COMPLETED, step=self.step_index, reason=reason)
        self._closed = True
        return event

    def pause(
        self,
        *,
        reason: str = "candidate optimization paused at an exact checkpoint",
        checkpoint_id: str | None = None,
    ) -> CandidateLifecycleEvent:
        """Close one execution segment without completing the candidate stage."""

        self._assert_open()
        self._assert_live_base_unchanged()
        if self.accumulation_index:
            raise TrainerExecutionError("cannot pause candidate with uncommitted gradient accumulation")
        event = self._emit_lifecycle(
            CandidateStatus.PAUSED,
            step=self.step_index,
            checkpoint_id=checkpoint_id,
            reason=reason,
        )
        self._closed = True
        return event

    def reject(self, *, reason: str) -> CandidateLifecycleEvent:
        self._assert_open()
        self._assert_live_base_unchanged()
        self.optimizer.zero_grad(set_to_none=True)
        event = self._emit_lifecycle(CandidateStatus.REJECTED, step=self.step_index, reason=reason)
        self._closed = True
        return event

    def candidate_manifest(self):
        return capture_module_manifest(self.candidate_descriptor, self.candidate_module, exact_value_hashes=True)


__all__ = [
    "CandidateOptimizationSession",
    "OptimizerExecutionPolicy",
    "TrainerExecutionError",
]
