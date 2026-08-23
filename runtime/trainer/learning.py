"""Immutable governed learning-policy contracts for Axon's Trainer organ."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from runtime.field import canonical_sha256

LEARNING_POLICY_SCHEMA = "axon-trainer-learning-policy-v1"


class OptimizerKind(str, Enum):
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerKind(str, Enum):
    CONSTANT = "constant"
    WARMUP_COSINE = "warmup_cosine"


class PrecisionMode(str, Enum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"


def _finite_nonnegative(value: float, label: str) -> float:
    value = float(value)
    if not (0.0 <= value < float("inf")):
        raise ValueError(f"{label} must be finite and non-negative")
    return value


def _finite_positive(value: float, label: str) -> float:
    value = float(value)
    if not (0.0 < value < float("inf")):
        raise ValueError(f"{label} must be finite and positive")
    return value


@dataclass(frozen=True, slots=True)
class GovernedLearningPolicy:
    """Complete deterministic policy for one candidate optimizer lifecycle.

    The mutation plan still declares *what* may learn and its maximum optimizer
    steps.  This policy declares *how* those authorized parameters learn.  Its
    content hash is persisted and bound into every candidate checkpoint/step.
    """

    optimizer: OptimizerKind | str
    learning_rate: float
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    sgd_momentum: float = 0.0
    scheduler: SchedulerKind | str = SchedulerKind.CONSTANT
    warmup_steps: int = 0
    min_lr_ratio: float = 0.0
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float | None = 1.0
    max_gradient_l2: float | None = None
    max_update_l2: float | None = None
    precision: PrecisionMode | str = PrecisionMode.FP32
    exact_scope_verification_each_step: bool = True
    full_parameter_telemetry_each_step: bool = True
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        optimizer = self.optimizer if isinstance(self.optimizer, OptimizerKind) else OptimizerKind(str(self.optimizer).lower())
        scheduler = self.scheduler if isinstance(self.scheduler, SchedulerKind) else SchedulerKind(str(self.scheduler).lower())
        precision = self.precision if isinstance(self.precision, PrecisionMode) else PrecisionMode(str(self.precision).lower())
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "scheduler", scheduler)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "learning_rate", _finite_positive(self.learning_rate, "learning_rate"))
        object.__setattr__(self, "weight_decay", _finite_nonnegative(self.weight_decay, "weight_decay"))

        beta1 = float(self.adam_beta1)
        beta2 = float(self.adam_beta2)
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError("Adam betas must be in [0, 1)")
        object.__setattr__(self, "adam_beta1", beta1)
        object.__setattr__(self, "adam_beta2", beta2)
        object.__setattr__(self, "adam_eps", _finite_positive(self.adam_eps, "adam_eps"))

        momentum = float(self.sgd_momentum)
        if not (0.0 <= momentum < 1.0):
            raise ValueError("sgd_momentum must be in [0, 1)")
        object.__setattr__(self, "sgd_momentum", momentum)

        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")
        ratio = float(self.min_lr_ratio)
        if not (0.0 <= ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in [0, 1]")
        object.__setattr__(self, "min_lr_ratio", ratio)
        if scheduler is SchedulerKind.CONSTANT and self.warmup_steps:
            raise ValueError("constant scheduler cannot declare warmup_steps")

        accumulation = self.gradient_accumulation_steps
        if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if self.gradient_clip_norm is not None:
            object.__setattr__(self, "gradient_clip_norm", _finite_positive(self.gradient_clip_norm, "gradient_clip_norm"))
        if self.max_gradient_l2 is not None:
            object.__setattr__(self, "max_gradient_l2", _finite_positive(self.max_gradient_l2, "max_gradient_l2"))
        if self.max_update_l2 is not None:
            object.__setattr__(self, "max_update_l2", _finite_positive(self.max_update_l2, "max_update_l2"))

        object.__setattr__(self, "policy_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @classmethod
    def from_plan(cls, plan: Any) -> "GovernedLearningPolicy":
        return cls(optimizer=plan.optimizer_name, learning_rate=plan.learning_rate)

    def assert_plan_compatible(self, plan: Any) -> None:
        if str(plan.optimizer_name).strip().lower() != self.optimizer.value:
            raise ValueError("learning policy optimizer differs from mutation plan")
        if not math.isclose(float(plan.learning_rate), self.learning_rate, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("learning policy base learning rate differs from mutation plan")
        if self.warmup_steps > int(plan.max_steps):
            raise ValueError("learning policy warmup exceeds authorized optimizer-step budget")

    def learning_rate_for_step(self, completed_optimizer_steps: int, max_optimizer_steps: int) -> float:
        if isinstance(completed_optimizer_steps, bool) or not isinstance(completed_optimizer_steps, int) or completed_optimizer_steps < 0:
            raise ValueError("completed_optimizer_steps must be a non-negative integer")
        if isinstance(max_optimizer_steps, bool) or not isinstance(max_optimizer_steps, int) or max_optimizer_steps <= 0:
            raise ValueError("max_optimizer_steps must be a positive integer")
        if completed_optimizer_steps >= max_optimizer_steps:
            return self.learning_rate * self.min_lr_ratio if self.scheduler is SchedulerKind.WARMUP_COSINE else self.learning_rate
        if self.scheduler is SchedulerKind.CONSTANT:
            return self.learning_rate
        if self.warmup_steps and completed_optimizer_steps < self.warmup_steps:
            return self.learning_rate * float(completed_optimizer_steps + 1) / float(self.warmup_steps)
        remaining = max_optimizer_steps - self.warmup_steps
        if remaining <= 1:
            progress = 1.0
        else:
            post_warmup_index = completed_optimizer_steps - self.warmup_steps
            progress = min(1.0, max(0.0, float(post_warmup_index + 1) / float(remaining)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        return self.learning_rate * factor

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": LEARNING_POLICY_SCHEMA,
            "optimizer": self.optimizer.value,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_eps": self.adam_eps,
            "sgd_momentum": self.sgd_momentum,
            "scheduler": self.scheduler.value,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clip_norm": self.gradient_clip_norm,
            "max_gradient_l2": self.max_gradient_l2,
            "max_update_l2": self.max_update_l2,
            "precision": self.precision.value,
            "exact_scope_verification_each_step": bool(self.exact_scope_verification_each_step),
            "full_parameter_telemetry_each_step": bool(self.full_parameter_telemetry_each_step),
        }
        if include_id:
            value["policy_id"] = self.policy_id
        return value


__all__ = [
    "LEARNING_POLICY_SCHEMA",
    "OptimizerKind",
    "SchedulerKind",
    "PrecisionMode",
    "GovernedLearningPolicy",
]
