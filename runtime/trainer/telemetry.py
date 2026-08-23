"""Transparent per-parameter telemetry for Axon's Trainer organ."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from runtime.field import canonical_sha256

PARAMETER_TELEMETRY_SCHEMA = "axon-parameter-telemetry-frame-v1"
PARAMETER_STAT_SCHEMA = "axon-parameter-telemetry-stat-v1"


@dataclass(frozen=True, slots=True)
class ParameterStat:
    name: str
    numel: int
    value_l2: float
    value_rms: float
    value_abs_max: float
    value_zero_fraction: float
    value_finite_fraction: float
    grad_present: bool
    grad_l2: float | None
    grad_rms: float | None
    grad_abs_max: float | None
    grad_zero_fraction: float | None
    grad_finite_fraction: float | None

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": PARAMETER_STAT_SCHEMA,
            "name": self.name,
            "numel": self.numel,
            "value_l2": self.value_l2,
            "value_rms": self.value_rms,
            "value_abs_max": self.value_abs_max,
            "value_zero_fraction": self.value_zero_fraction,
            "value_finite_fraction": self.value_finite_fraction,
            "grad_present": self.grad_present,
            "grad_l2": self.grad_l2,
            "grad_rms": self.grad_rms,
            "grad_abs_max": self.grad_abs_max,
            "grad_zero_fraction": self.grad_zero_fraction,
            "grad_finite_fraction": self.grad_finite_fraction,
        }


@dataclass(frozen=True, slots=True)
class ParameterTelemetryFrame:
    module_id: str
    generation_id: str
    step: int
    stats: tuple[ParameterStat, ...]
    inventory_id: str | None = None
    frame_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.module_id:
            raise ValueError("module_id must be non-empty")
        if not self.generation_id:
            raise ValueError("generation_id must be non-empty")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        stats = tuple(sorted(tuple(self.stats), key=lambda item: item.name))
        if not all(isinstance(item, ParameterStat) for item in stats):
            raise TypeError("stats must contain only ParameterStat values")
        names = [item.name for item in stats]
        if len(names) != len(set(names)):
            raise ValueError("duplicate parameter telemetry name")
        object.__setattr__(self, "stats", stats)
        object.__setattr__(self, "frame_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PARAMETER_TELEMETRY_SCHEMA,
            "module_id": self.module_id,
            "generation_id": self.generation_id,
            "step": self.step,
            "inventory_id": self.inventory_id,
            "parameter_count": sum(item.numel for item in self.stats),
            "tensor_count": len(self.stats),
            "all_values_finite": all(item.value_finite_fraction == 1.0 for item in self.stats),
            "all_present_gradients_finite": all(
                (not item.grad_present) or item.grad_finite_fraction == 1.0 for item in self.stats
            ),
            "stats": [item.to_canonical_dict() for item in self.stats],
        }
        if include_id:
            value["frame_id"] = self.frame_id
        return value


def _stats(tensor: torch.Tensor) -> tuple[float, float, float, float, float]:
    detached = tensor.detach()
    if detached.layout is not torch.strided:
        raise ValueError("parameter telemetry currently requires dense strided tensors")
    if detached.numel() == 0:
        return 0.0, 0.0, 0.0, 1.0, 1.0
    values = detached.to(dtype=torch.float32)
    finite = torch.isfinite(values)
    finite_fraction = float(finite.float().mean().item())
    safe = torch.where(finite, values, torch.zeros_like(values))
    l2 = float(torch.linalg.vector_norm(safe).item())
    rms = float(torch.sqrt(torch.mean(safe * safe)).item())
    abs_max = float(torch.max(torch.abs(safe)).item())
    zero_fraction = float((safe == 0).float().mean().item())
    return l2, rms, abs_max, zero_fraction, finite_fraction


def capture_parameter_telemetry(
    module: nn.Module,
    *,
    module_id: str,
    generation_id: str,
    step: int,
    inventory_id: str | None = None,
) -> ParameterTelemetryFrame:
    """Capture value/gradient health for every named parameter in ``module``."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be torch.nn.Module")
    rows: list[ParameterStat] = []
    for name, parameter in module.named_parameters(recurse=True):
        value_l2, value_rms, value_abs_max, value_zero, value_finite = _stats(parameter)
        gradient = parameter.grad
        if gradient is None:
            grad_l2 = grad_rms = grad_abs_max = grad_zero = grad_finite = None
        else:
            grad_l2, grad_rms, grad_abs_max, grad_zero, grad_finite = _stats(gradient)
        rows.append(
            ParameterStat(
                name=name,
                numel=int(parameter.numel()),
                value_l2=value_l2,
                value_rms=value_rms,
                value_abs_max=value_abs_max,
                value_zero_fraction=value_zero,
                value_finite_fraction=value_finite,
                grad_present=gradient is not None,
                grad_l2=grad_l2,
                grad_rms=grad_rms,
                grad_abs_max=grad_abs_max,
                grad_zero_fraction=grad_zero,
                grad_finite_fraction=grad_finite,
            )
        )
    return ParameterTelemetryFrame(
        module_id=module_id,
        generation_id=generation_id,
        step=step,
        inventory_id=inventory_id,
        stats=tuple(rows),
    )


__all__ = [
    "PARAMETER_TELEMETRY_SCHEMA",
    "PARAMETER_STAT_SCHEMA",
    "ParameterStat",
    "ParameterTelemetryFrame",
    "capture_parameter_telemetry",
]
