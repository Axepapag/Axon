"""Adapter from a checkpoint-compatible ``AxonCore`` to typed field proposals."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from cores.core import AxonCore
from cores.soul_v2 import SoulState
from runtime.field import (
    PROPOSAL_END,
    PROPOSAL_START,
    FieldView,
    LogicalRegion,
    SharedFieldSnapshot,
)
from runtime.multi_tick_refiner import RegionProposal
from substrate import get_letter_bank


class AxonCoreRegionProposer:
    """Run one real char-slot core without mutating its field or soul inputs.

    This adapter deliberately emits a proposal only.  Validation, commit, later
    materialization, and exhale-after-outcome remain the refiner/runtime's
    responsibilities.
    """

    def __init__(
        self,
        core: AxonCore,
        *,
        soul_state: SoulState | None = None,
        max_output_chars: int = PROPOSAL_END - PROPOSAL_START,
    ) -> None:
        if not isinstance(core, AxonCore):
            raise TypeError("core must be AxonCore")
        if not core.cfg.char_slot_mode:
            raise ValueError("core must use char_slot_mode")
        if core.cfg.char_n_regions != 3:
            raise ValueError("checkpoint-compatible proposer requires 3 physical roles")
        if core.cfg.char_slot_max_slots < PROPOSAL_END:
            raise ValueError("core char_slot_max_slots is smaller than 384")
        if not 1 <= max_output_chars <= PROPOSAL_END - PROPOSAL_START:
            raise ValueError("max_output_chars must be in [1, 64]")
        if soul_state is not None and soul_state.d_model != core.cfg.d_model:
            raise ValueError("soul d_model does not match core d_model")
        self.core = core
        self.soul_state = soul_state
        self.max_output_chars = max_output_chars

    def _device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.core.parameters())
        return parameter.device, parameter.dtype

    def _soul_inputs(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.soul_state is None:
            return (
                torch.zeros(
                    (1, 0, self.core.cfg.d_model),
                    device=device,
                    dtype=dtype,
                ),
                torch.zeros((1, 0), device=device, dtype=torch.bool),
            )
        # Clone both model-visible values.  The external SoulState and all of
        # its metadata remain byte-identical across the forward pass.
        return (
            self.soul_state.tensor.detach().clone().unsqueeze(0).to(
                device=device,
                dtype=dtype,
            ),
            self.soul_state.active.detach().clone().unsqueeze(0).to(
                device=device,
                dtype=torch.bool,
            ),
        )

    def __call__(
        self,
        *,
        snapshot: SharedFieldSnapshot,
        view: FieldView,
        tick_index: int,
        target_region: LogicalRegion,
    ) -> RegionProposal:
        if not isinstance(snapshot, SharedFieldSnapshot):
            raise TypeError("snapshot must be SharedFieldSnapshot")
        if not isinstance(view, FieldView):
            raise TypeError("view must be runtime.field.FieldView")
        if snapshot.field_id != view.source_field_id:
            raise ValueError("field view does not belong to the supplied snapshot")
        if view.proposal_region is not target_region:
            raise ValueError("field view proposal region does not match target")
        if view.field16.shape != (384, 16):
            raise ValueError("field view must have shape (384, 16)")
        if view.role_ids.shape != (384,):
            raise ValueError("physical role IDs must have shape (384,)")
        if view.attention_mask.shape != (384,):
            raise ValueError("attention mask must have shape (384,)")
        if set(np.unique(view.role_ids).tolist()) - {0, 1, 2}:
            raise ValueError("physical role IDs must remain in {0,1,2}")

        device, dtype = self._device_dtype()
        field16 = torch.from_numpy(
            np.array(view.field16, copy=True)
        ).unsqueeze(0).to(device=device, dtype=dtype)
        roles = torch.from_numpy(
            np.array(view.role_ids, copy=True)
        ).unsqueeze(0).to(device=device, dtype=torch.long)
        mask = torch.from_numpy(
            np.array(view.attention_mask, copy=True)
        ).unsqueeze(0).to(device=device, dtype=torch.bool)
        soul, soul_mask = self._soul_inputs(device=device, dtype=dtype)

        module_modes = tuple((module, module.training) for module in self.core.modules())
        bank = get_letter_bank()
        bank_unit = torch.from_numpy(bank.vecs_unit.copy()).to(
            device=device,
            dtype=dtype,
        )
        try:
            self.core.eval()
            with torch.inference_mode():
                output = self.core.forward_charslot(
                    field16,
                    roles,
                    soul,
                    mask=mask,
                    soul_mask=soul_mask,
                    response_slice=slice(PROPOSAL_START, PROPOSAL_END),
                )
                delta16 = output.get("response_delta_16")
                if not isinstance(delta16, torch.Tensor):
                    raise RuntimeError("AxonCore did not emit response_delta_16")
                logits = self.core.charslot_logits(delta16, bank_unit)
                indices = logits.argmax(dim=-1)[0].detach().cpu().tolist()
        finally:
            for module, was_training in module_modes:
                module.training = was_training

        decoded: list[str] = []
        for index in indices[: self.max_output_chars]:
            if index == bank.empty_index:
                break
            if not 0 <= index < len(bank.chars):
                raise RuntimeError(f"core emitted invalid alphabet index {index}")
            decoded.append(bank.chars[index])
        return RegionProposal(
            target_region=target_region,
            text="".join(decoded),
            provenance=(
                f"axon_core:{self.core.cfg.d_model}d:"
                f"tick:{tick_index}:view:{view.view_hash}"
            ),
        )


__all__ = ["AxonCoreRegionProposer"]
