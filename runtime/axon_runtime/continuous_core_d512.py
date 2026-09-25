"""Fresh D16-native continuous reasoning Core baseline.

This module deliberately contains no attention stack, tokenizer embedding table, or
width-specific Heart rail. Exact registered D16 transport cells enter one learned
D512 recurrent chamber. Public symbol emission is categorical over the 351
registered transport tokens plus one private EOS control category; emitted transport
categories map mechanically back to the frozen exact D16 bank.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from runtime.field import canonical_sha256
from substrate import TRANSPORT_VOCAB_SIZE, unicode_transport_bank

CONTINUOUS_CORE_D512_ARCHITECTURE = "axon-continuous-core-d512-gru-v1"
CONTINUOUS_CORE_D512_OUTPUT_SCHEMA = "axon-continuous-core-categorical-output-v1"


@dataclass(frozen=True, slots=True)
class ContinuousCoreD512Config:
    input_dim: int = 16
    d_model: int = 512
    transport_vocab_size: int = TRANSPORT_VOCAB_SIZE
    eos_id: int = TRANSPORT_VOCAB_SIZE
    config_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.input_dim != 16:
            raise ValueError("continuous Core input_dim is fixed to exact D16")
        if self.d_model != 512:
            raise ValueError("B4 baseline d_model is fixed to 512")
        if self.transport_vocab_size != TRANSPORT_VOCAB_SIZE:
            raise ValueError("transport vocabulary must match the registered Axon bank")
        if self.eos_id != self.transport_vocab_size:
            raise ValueError("EOS must be the private category immediately above transport")
        object.__setattr__(self, "config_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def output_classes(self) -> int:
        return self.transport_vocab_size + 1

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": CONTINUOUS_CORE_D512_ARCHITECTURE,
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "transport_vocab_size": self.transport_vocab_size,
            "eos_id": self.eos_id,
            "output_classes": self.output_classes,
        }
        if include_id:
            value["config_id"] = self.config_id
        return value


class ContinuousCoreD512(nn.Module):
    """One persistent D512 GRU chamber with exact D16 ingress and categorical egress."""

    def __init__(self, cfg: ContinuousCoreD512Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ContinuousCoreD512Config()
        bank = np.array(unicode_transport_bank(), dtype=np.float32, order="C", copy=True)
        if bank.shape != (self.cfg.transport_vocab_size, self.cfg.input_dim):
            raise ValueError(f"unexpected registered D16 transport bank shape {bank.shape}")
        self.register_buffer("bank16", torch.from_numpy(bank), persistent=True)
        self.in_proj = nn.Linear(self.cfg.input_dim, self.cfg.d_model)
        self.cell = nn.GRUCell(self.cfg.d_model, self.cfg.d_model)
        self.readout = nn.Linear(self.cfg.d_model, self.cfg.output_classes)

    @property
    def eos_id(self) -> int:
        return self.cfg.eos_id

    def zero_state(self, batch_size: int, *, device: torch.device | None = None) -> torch.Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if device is None:
            device = self.in_proj.weight.device
        return torch.zeros(batch_size, self.cfg.d_model, device=device, dtype=self.in_proj.weight.dtype)

    def ingest_cells(
        self,
        cells16: torch.Tensor,
        state: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Consume exact D16 occurrences sequentially while preserving masked state."""

        if cells16.ndim == 2:
            cells16 = cells16.unsqueeze(0)
        if cells16.ndim != 3 or cells16.shape[-1] != self.cfg.input_dim:
            raise ValueError("cells16 must have shape [batch, sequence, 16] or [sequence, 16]")
        batch, steps, _ = cells16.shape
        if state is None:
            state = self.zero_state(batch, device=cells16.device)
        if state.shape != (batch, self.cfg.d_model):
            raise ValueError("state shape does not match [batch, d_model]")
        if mask is not None:
            if mask.shape != (batch, steps):
                raise ValueError("mask must have shape [batch, sequence]")
            mask = mask.to(device=cells16.device, dtype=torch.bool)

        h = state
        for index in range(steps):
            x = torch.tanh(self.in_proj(cells16[:, index].to(dtype=self.in_proj.weight.dtype)))
            candidate = self.cell(x, h)
            if mask is None:
                h = candidate
            else:
                h = torch.where(mask[:, index].unsqueeze(1), candidate, h)
        return h

    def teacher_forced_logits(
        self,
        source_cells16: torch.Tensor,
        target_token_ids: torch.Tensor,
        *,
        source_mask: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict each target token before seeing it, then predict private EOS.

        Every teacher-forced feedback symbol is re-entered through its exact frozen D16
        transport cell. The returned logits have shape [batch, target_len + 1, 352].
        """

        if target_token_ids.ndim == 1:
            target_token_ids = target_token_ids.unsqueeze(0)
        if target_token_ids.ndim != 2:
            raise ValueError("target_token_ids must have shape [batch, target_len]")
        if target_token_ids.numel() and (
            int(target_token_ids.min()) < 0
            or int(target_token_ids.max()) >= self.cfg.transport_vocab_size
        ):
            raise ValueError("target token id is outside registered transport vocabulary")

        h = self.ingest_cells(source_cells16, initial_state, mask=source_mask)
        batch, target_len = target_token_ids.shape
        if h.shape[0] != batch:
            raise ValueError("source and target batch sizes differ")
        outputs: list[torch.Tensor] = []
        for index in range(target_len):
            outputs.append(self.readout(h))
            token_ids = target_token_ids[:, index].to(device=h.device, dtype=torch.long)
            cells = self.bank16.index_select(0, token_ids).to(dtype=self.in_proj.weight.dtype)
            h = self.cell(torch.tanh(self.in_proj(cells)), h)
        outputs.append(self.readout(h))
        return torch.stack(outputs, dim=1), h

    @torch.no_grad()
    def greedy_generate(
        self,
        source_cells16: torch.Tensor,
        *,
        max_steps: int = 64,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[tuple[int, ...], torch.Tensor, bool]:
        """Free-run one example until private EOS or the explicit generation guard."""

        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if source_cells16.ndim == 3 and source_cells16.shape[0] != 1:
            raise ValueError("greedy_generate currently evaluates one example at a time")
        h = self.ingest_cells(source_cells16, initial_state)
        emitted: list[int] = []
        terminated = False
        for _ in range(max_steps):
            token_id = int(self.readout(h).argmax(dim=-1).item())
            if token_id == self.eos_id:
                terminated = True
                break
            emitted.append(token_id)
            cell = self.bank16[token_id].unsqueeze(0).to(dtype=self.in_proj.weight.dtype)
            h = self.cell(torch.tanh(self.in_proj(cell)), h)
        return tuple(emitted), h, terminated

    def exact_output_cells(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map categorical transport decisions back to the exact frozen D16 cells."""

        ids = token_ids.to(device=self.bank16.device, dtype=torch.long)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.cfg.transport_vocab_size):
            raise ValueError("EOS/control categories do not serialize onto the D16 bus")
        return self.bank16.index_select(0, ids.reshape(-1)).reshape(*ids.shape, self.cfg.input_dim)


__all__ = [
    "CONTINUOUS_CORE_D512_ARCHITECTURE",
    "CONTINUOUS_CORE_D512_OUTPUT_SCHEMA",
    "ContinuousCoreD512",
    "ContinuousCoreD512Config",
]
