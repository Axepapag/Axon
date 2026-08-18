"""Additive 168-row differentiable writer for write-delay-recall pilots.

The existing core writer assumes a different row partition and the external
SoulManager mutates under ``no_grad``.  This module is deliberately separate:
it writes only the authoritative 128 hot rows, preserves all 32 warm and 8
cold rows exactly, and lets tick-B recall loss backpropagate through the tick-A
write.  It is a pilot mechanism, not a promoted replacement for SoulManagerV2.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cores.core import AxonCore
from runtime.field import (
    PROPOSAL_END,
    PROPOSAL_START,
    FieldView,
)
from substrate import assert_supported_text, get_letter_bank


@dataclass(frozen=True, slots=True)
class SoulWriterConfig:
    d_model: int
    hot_rows: int = 128
    warm_rows: int = 32
    cold_rows: int = 8
    gate_init: float = 0.05

    @property
    def total_rows(self) -> int:
        return self.hot_rows + self.warm_rows + self.cold_rows

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if (self.hot_rows, self.warm_rows, self.cold_rows) != (128, 32, 8):
            raise ValueError(
                "pilot requires the authoritative 128-hot/32-warm/8-cold layout"
            )
        if not math.isfinite(self.gate_init) or self.gate_init <= 0.0:
            raise ValueError("gate_init must be finite and positive")

    def to_dict(self) -> dict[str, int | float]:
        value = asdict(self)
        value["total_rows"] = self.total_rows
        return value


@dataclass(frozen=True)
class SoulWriteResult:
    soul: torch.Tensor
    strengths: torch.Tensor
    hot_delta: torch.Tensor


class DifferentiableSoulWriter(nn.Module):
    """Content-addressed residual writer over exactly the hot tier."""

    def __init__(self, cfg: SoulWriterConfig):
        super().__init__()
        self.cfg = cfg
        self.query_norm = nn.LayerNorm(cfg.d_model)
        self.key_norm = nn.LayerNorm(cfg.d_model)
        self.keys = nn.Parameter(torch.empty(cfg.hot_rows, cfg.d_model))
        self.value_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(cfg.gate_init)))
        nn.init.normal_(self.keys, mean=0.0, std=0.02)
        # A small nonzero start keeps every declared writer role capable of
        # receiving gradient in the first coverage smoke.
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.01)

    def forward(
        self,
        field_out: torch.Tensor,
        soul_in: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        field_mask: torch.Tensor | None = None,
    ) -> SoulWriteResult:
        if field_out.ndim != 3 or field_out.shape[-1] != self.cfg.d_model:
            raise ValueError("field_out must have shape (B,N,d_model)")
        if soul_in.ndim != 3 or tuple(soul_in.shape[1:]) != (
            self.cfg.total_rows,
            self.cfg.d_model,
        ):
            raise ValueError(
                "soul_in must have shape "
                f"(B,{self.cfg.total_rows},{self.cfg.d_model})"
            )
        if soul_in.shape[0] != field_out.shape[0]:
            raise ValueError("field and soul batch sizes differ")

        if active_mask.ndim == 1:
            active_mask = active_mask.unsqueeze(0)
        if tuple(active_mask.shape) != (
            soul_in.shape[0],
            self.cfg.total_rows,
        ):
            raise ValueError(
                f"active_mask must have shape (B,{self.cfg.total_rows})"
            )
        active_mask = active_mask.to(device=soul_in.device, dtype=torch.bool)
        hot_active = active_mask[:, : self.cfg.hot_rows]
        if (~hot_active.any(dim=1)).any():
            # Give the caller a precise contract failure instead of softmax
            # over an all-masked tier.
            bad = (~hot_active.any(dim=1)).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"no active hot soul rows for batches {bad}")

        if field_mask is None:
            pooled = field_out.mean(dim=1)
        else:
            if field_mask.ndim == 1:
                field_mask = field_mask.unsqueeze(0)
            if tuple(field_mask.shape) != tuple(field_out.shape[:2]):
                raise ValueError("field_mask must have shape (B,N)")
            field_mask = field_mask.to(device=field_out.device, dtype=torch.bool)
            counts = field_mask.sum(dim=1, keepdim=True)
            if (counts == 0).any():
                raise ValueError("field_mask must expose at least one slot per batch")
            pooled = (
                field_out * field_mask.unsqueeze(-1).to(field_out.dtype)
            ).sum(dim=1) / counts.to(field_out.dtype)

        q = self.query_norm(pooled)
        k = self.key_norm(self.keys)
        logits = (q @ k.T) / math.sqrt(self.cfg.d_model)
        logits = logits.masked_fill(~hot_active, float("-inf"))
        strengths = F.softmax(logits, dim=-1)
        value = self.value_proj(pooled)
        hot_delta = (
            self.gate
            * strengths.unsqueeze(-1)
            * value.unsqueeze(1)
            * hot_active.unsqueeze(-1).to(value.dtype)
        )
        hot_out = soul_in[:, : self.cfg.hot_rows, :] + hot_delta
        # Concatenation is differentiable and makes the preservation boundary
        # explicit; warm/cold never pass through a learned operation.
        soul_out = torch.cat(
            (hot_out, soul_in[:, self.cfg.hot_rows :, :]),
            dim=1,
        )
        return SoulWriteResult(
            soul=soul_out,
            strengths=strengths,
            hot_delta=hot_delta,
        )


@dataclass(frozen=True, slots=True)
class PilotParameterPolicy:
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]


def writer_pilot_reader_parameter_names(core: AxonCore) -> tuple[str, ...]:
    """Return the exact core parameter allowlist for the soul read path.

    Names use the same ``core.`` prefix as :class:`PilotParameterPolicy`.  The
    allowlist is derived from concrete module/parameter identities instead of
    prefix matching so architecture drift cannot silently widen it.
    """

    if core.cfg.soul_mode != "act_reflect_v2":
        raise ValueError(
            "writer pilot reader allowlist requires soul_mode=act_reflect_v2"
        )
    names_by_id = {
        id(parameter): f"core.{name}"
        for name, parameter in core.named_parameters()
    }
    selected: set[str] = set()

    def add_module(module: object, *, label: str) -> None:
        if not isinstance(module, nn.Module):
            raise ValueError(f"writer pilot reader module {label} is missing")
        parameters = tuple(module.parameters())
        if not parameters:
            raise ValueError(f"writer pilot reader module {label} has no parameters")
        for parameter in parameters:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError(
                    f"writer pilot reader parameter in {label} is not owned by core"
                )
            selected.add(name)

    add_module(getattr(core, "soul_ingest", None), label="soul_ingest")
    for layer_index, layer in enumerate(core.layers):
        for module_name in (
            "cross_q_norm",
            "cross_ctx_norm",
            "soul_cross",
        ):
            add_module(
                getattr(layer, module_name, None),
                label=f"layers.{layer_index}.{module_name}",
            )
        gate = getattr(layer, "soul_cross_gate", None)
        if not isinstance(gate, nn.Parameter):
            raise ValueError(
                "writer pilot reader parameter "
                f"layers.{layer_index}.soul_cross_gate is missing"
            )
        name = names_by_id.get(id(gate))
        if name is None:
            raise ValueError(
                "writer pilot reader gate "
                f"layers.{layer_index}.soul_cross_gate is not owned by core"
            )
        selected.add(name)

    if not selected:
        raise ValueError("writer pilot reader parameter allowlist is empty")
    return tuple(sorted(selected))


def configure_writer_pilot_parameters(
    core: AxonCore,
    writer: DifferentiableSoulWriter,
    *,
    train_read_path: bool = True,
) -> PilotParameterPolicy:
    """Freeze the conflicting core writer and declare an honest trainable set."""

    if core.cfg.d_model != writer.cfg.d_model:
        raise ValueError("core and writer d_model differ")
    core.soul_readonly = True
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    for parameter in writer.parameters():
        parameter.requires_grad_(True)

    if train_read_path:
        reader_names = set(writer_pilot_reader_parameter_names(core))
        for name, parameter in core.named_parameters():
            if f"core.{name}" in reader_names:
                parameter.requires_grad_(True)

    named = [
        (f"core.{name}", parameter)
        for name, parameter in core.named_parameters()
    ] + [
        (f"writer.{name}", parameter)
        for name, parameter in writer.named_parameters()
    ]
    trainable = tuple(
        sorted(name for name, parameter in named if parameter.requires_grad)
    )
    frozen = tuple(
        sorted(name for name, parameter in named if not parameter.requires_grad)
    )
    if train_read_path:
        observed_reader_names = {
            name for name in trainable if name.startswith("core.")
        }
        expected_reader_names = set(writer_pilot_reader_parameter_names(core))
        if observed_reader_names != expected_reader_names:
            raise ValueError(
                "writer pilot reader trainable set differs from exact allowlist"
            )
    return PilotParameterPolicy(trainable_names=trainable, frozen_names=frozen)


@dataclass(frozen=True)
class WriteDelayRecallOutput:
    logits: torch.Tensor
    soul_after_write: torch.Tensor
    hot_delta: torch.Tensor
    strengths: torch.Tensor


def _view_tensors(
    view: FieldView,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    field16 = torch.from_numpy(np.array(view.field16, copy=True)).unsqueeze(0).to(
        device=device,
        dtype=dtype,
    )
    roles = torch.from_numpy(np.array(view.role_ids, copy=True)).unsqueeze(0).to(
        device=device,
        dtype=torch.long,
    )
    mask = torch.from_numpy(
        np.array(view.attention_mask, copy=True)
    ).unsqueeze(0).to(device=device, dtype=torch.bool)
    return field16, roles, mask


def visible_view_text(view: FieldView) -> str:
    return "".join(
        ref.rendered_char or ""
        for ref, active in zip(
            view.slot_refs,
            view.attention_mask.tolist(),
            strict=True,
        )
        if active
    )


def write_delay_recall_forward(
    *,
    core: AxonCore,
    writer: DifferentiableSoulWriter,
    tick_a_view: FieldView,
    tick_b_view: FieldView,
    soul_before: torch.Tensor,
    soul_mask: torch.Tensor,
    fact_text: str,
) -> WriteDelayRecallOutput:
    """Run tick A -> differentiable hot write -> fact-free tick B recall."""

    if not core.cfg.char_slot_mode or core.cfg.char_n_regions != 3:
        raise ValueError("core must use the checkpoint-compatible char-slot path")
    if core.cfg.d_model != writer.cfg.d_model:
        raise ValueError("core and writer d_model differ")
    if tick_a_view.proposal_region.value != "response_draft":
        raise ValueError("tick A pilot view must target response_draft")
    if tick_b_view.proposal_region.value != "response_draft":
        raise ValueError("tick B pilot view must target response_draft")
    if not isinstance(fact_text, str) or not fact_text:
        raise ValueError("fact_text must be non-empty")
    if fact_text in visible_view_text(tick_b_view):
        raise ValueError("tick-B visible field leaks the expected fact")
    if tuple(soul_before.shape) != (
        1,
        writer.cfg.total_rows,
        writer.cfg.d_model,
    ):
        raise ValueError("soul_before has the wrong authoritative layout")
    if tuple(soul_mask.shape) != (1, writer.cfg.total_rows):
        raise ValueError("soul_mask has the wrong authoritative layout")

    parameter = next(core.parameters())
    device, dtype = parameter.device, parameter.dtype
    soul0 = soul_before.clone().to(device=device, dtype=dtype)
    soul_mask_local = soul_mask.clone().to(device=device, dtype=torch.bool)
    field_a, roles_a, mask_a = _view_tensors(
        tick_a_view,
        device=device,
        dtype=dtype,
    )
    field_b, roles_b, mask_b = _view_tensors(
        tick_b_view,
        device=device,
        dtype=dtype,
    )

    out_a = core.forward_charslot(
        field_a,
        roles_a,
        soul0,
        mask=mask_a,
        soul_mask=soul_mask_local,
        response_slice=slice(PROPOSAL_START, PROPOSAL_END),
    )
    write = writer(
        out_a["field"],
        soul0,
        soul_mask_local,
        field_mask=mask_a,
    )
    out_b = core.forward_charslot(
        field_b,
        roles_b,
        write.soul,
        mask=mask_b,
        soul_mask=soul_mask_local,
        response_slice=slice(PROPOSAL_START, PROPOSAL_END),
    )
    bank = get_letter_bank()
    bank_unit = torch.from_numpy(bank.vecs_unit.copy()).to(
        device=device,
        dtype=dtype,
    )
    logits = core.charslot_logits(out_b["response_delta_16"], bank_unit)
    return WriteDelayRecallOutput(
        logits=logits,
        soul_after_write=write.soul,
        hot_delta=write.hot_delta,
        strengths=write.strengths,
    )


def padded_target_indices(
    target: str,
    *,
    device: torch.device,
    width: int = PROPOSAL_END - PROPOSAL_START,
) -> torch.Tensor:
    assert_supported_text(target)
    if len(target) > width:
        raise ValueError(f"target length {len(target)} exceeds width {width}")
    bank = get_letter_bank()
    index = {char: position for position, char in enumerate(bank.chars)}
    values = torch.full(
        (1, width),
        bank.empty_index,
        dtype=torch.long,
        device=device,
    )
    for position, char in enumerate(target):
        values[0, position] = index[char]
    return values


def write_delay_recall_loss(
    output: WriteDelayRecallOutput,
    target: str,
) -> torch.Tensor:
    targets = padded_target_indices(target, device=output.logits.device)
    return F.cross_entropy(
        output.logits.reshape(-1, output.logits.shape[-1]),
        targets.reshape(-1),
    )


def named_gradient_coverage(
    modules: Iterable[tuple[str, nn.Module]],
) -> dict[str, object]:
    declared: list[str] = []
    covered: list[str] = []
    missing: list[str] = []
    nonfinite: list[str] = []
    for prefix, module in modules:
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            full_name = f"{prefix}.{name}"
            declared.append(full_name)
            if parameter.grad is None or parameter.grad.numel() == 0:
                missing.append(full_name)
            elif not torch.isfinite(parameter.grad).all():
                nonfinite.append(full_name)
            elif parameter.grad.detach().abs().sum().item() == 0.0:
                missing.append(full_name)
            else:
                covered.append(full_name)
    return {
        "declared": tuple(sorted(declared)),
        "covered": tuple(sorted(covered)),
        "missing_or_zero": tuple(sorted(missing)),
        "nonfinite": tuple(sorted(nonfinite)),
        "coverage": 0.0 if not declared else len(covered) / len(declared),
    }


__all__ = [
    "SoulWriterConfig",
    "SoulWriteResult",
    "DifferentiableSoulWriter",
    "PilotParameterPolicy",
    "writer_pilot_reader_parameter_names",
    "configure_writer_pilot_parameters",
    "WriteDelayRecallOutput",
    "visible_view_text",
    "write_delay_recall_forward",
    "padded_target_indices",
    "write_delay_recall_loss",
    "named_gradient_coverage",
]
