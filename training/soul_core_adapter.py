"""Real :class:`AxonCore` adapter for controlled causal-soul evaluation.

Only the tensors named by :class:`FieldView` are passed to the model. Scoring
targets are consumed after the forward pass and condition labels are neither
accepted nor synthesized, preventing the adapter from leaking the causal
condition into the core-visible field.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from cores.core import AxonCore, CHAR_SLOT_DIM
from cores.soul_v2 import SoulState
from substrate import assert_supported_text, get_letter_bank
from training.soul_load_bearing import (
    SoulResponderOutput,
    soul_state_sha256,
    validate_soul_state,
)


_FIELD_VIEW_KEYS = (
    "field16",
    "physical_role_ids",
    "attention_mask",
    "response_start",
    "response_stop",
    "score_targets",
)
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True)
class FieldView(Mapping[str, Any]):
    """Condition-free char-slot input plus post-forward scoring targets.

    ``field16``, ``physical_role_ids``, and ``attention_mask`` are the only
    model-visible values. ``response_start``/``response_stop`` select proposal
    slots. ``score_targets`` are used solely to calculate post-forward NLL.

    Implementing ``Mapping`` makes this view directly hashable by the fixed
    causal-suite machinery without adding a dependency in that module.
    """

    field16: torch.Tensor
    physical_role_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_start: int
    response_stop: int
    score_targets: tuple[str, ...]

    def __getitem__(self, key: str) -> Any:
        if key not in _FIELD_VIEW_KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_FIELD_VIEW_KEYS)

    def __len__(self) -> int:
        return len(_FIELD_VIEW_KEYS)


@dataclass(frozen=True)
class SoulCoreAdapterResult:
    """Normalized causal response plus the actual proposal logits."""

    response: SoulResponderOutput
    logits: torch.Tensor


@dataclass(frozen=True)
class _PreparedFieldView:
    field16: torch.Tensor
    physical_role_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_slice: slice
    score_targets: tuple[str, ...]


def _coerce_field_view(value: FieldView | Mapping[str, Any]) -> FieldView:
    if isinstance(value, FieldView):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("visible input must be a FieldView or tensor mapping")

    supplied = set(value)
    expected = set(_FIELD_VIEW_KEYS)
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    if missing:
        raise ValueError(f"field view is missing keys: {', '.join(missing)}")
    if unexpected:
        raise ValueError(
            f"field view has unexpected keys: {', '.join(unexpected)}"
        )
    score_targets = value["score_targets"]
    if isinstance(score_targets, str) or not isinstance(score_targets, Sequence):
        raise ValueError("score_targets must be a sequence of strings")
    return FieldView(
        field16=value["field16"],
        physical_role_ids=value["physical_role_ids"],
        attention_mask=value["attention_mask"],
        response_start=value["response_start"],
        response_stop=value["response_stop"],
        score_targets=tuple(score_targets),
    )


class AxonCoreSoulAdapter:
    """Adapt one char-slot ``AxonCore`` to the generic causal evaluator."""

    def __init__(self, core: AxonCore):
        if not isinstance(core, AxonCore):
            raise TypeError("core must be an AxonCore")
        if not core.cfg.char_slot_mode:
            raise ValueError("causal soul adapter requires char_slot_mode=True")
        if core.cfg.soul_mode != "act_reflect_v2":
            raise ValueError(
                "causal soul adapter requires soul_mode='act_reflect_v2'"
            )
        if core.cfg.d_model < CHAR_SLOT_DIM:
            raise ValueError(
                f"core d_model must be at least {CHAR_SLOT_DIM} for char slots"
            )
        if tuple(core.char_lift.shape) != (CHAR_SLOT_DIM, core.cfg.d_model):
            raise ValueError("core char_lift shape does not match its config")

        reference = next(core.parameters(), None)
        if reference is None:
            raise ValueError("core has no parameters")
        self.core = core
        self.device = reference.device
        self.dtype = reference.dtype

        bank = get_letter_bank()
        self._chars = tuple(bank.chars)
        self._empty_index = int(bank.empty_index)
        self._char_to_index = {
            character: index
            for index, character in enumerate(self._chars)
            if index != self._empty_index
        }
        self._bank_unit = torch.from_numpy(bank.vecs_unit.copy()).to(
            self.device,
            self.dtype,
        )
        if tuple(self._bank_unit.shape) != (len(self._chars), CHAR_SLOT_DIM):
            raise ValueError("letter-bank tensor has an invalid shape")
        if not torch.isfinite(self._bank_unit).all().item():
            raise ValueError("letter-bank tensor contains non-finite values")

    def _prepare(
        self,
        visible_input: FieldView | Mapping[str, Any],
    ) -> _PreparedFieldView:
        view = _coerce_field_view(visible_input)
        field16 = view.field16
        role_ids = view.physical_role_ids
        attention_mask = view.attention_mask

        if not isinstance(field16, torch.Tensor):
            raise TypeError("field16 must be a tensor")
        if field16.ndim != 3 or field16.shape[0] != 1:
            raise ValueError("field16 must have shape (1, n_slots, 16)")
        if field16.shape[-1] != CHAR_SLOT_DIM:
            raise ValueError(f"field16 final dimension must be {CHAR_SLOT_DIM}")
        if not field16.dtype.is_floating_point:
            raise ValueError("field16 must use a floating-point dtype")
        if not torch.isfinite(field16).all().item():
            raise ValueError("field16 contains non-finite values")

        n_slots = int(field16.shape[1])
        if n_slots <= 0:
            raise ValueError("field16 must contain at least one slot")
        if n_slots > int(self.core.cfg.char_slot_max_slots):
            raise ValueError(
                f"field slot count {n_slots} exceeds core maximum "
                f"{self.core.cfg.char_slot_max_slots}"
            )

        if not isinstance(role_ids, torch.Tensor):
            raise TypeError("physical_role_ids must be a tensor")
        if tuple(role_ids.shape) != (1, n_slots):
            raise ValueError(
                "physical_role_ids must have shape (1, n_slots)"
            )
        if role_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("physical_role_ids must use an integer dtype")
        if (
            (role_ids < 0)
            | (role_ids >= int(self.core.cfg.char_n_regions))
        ).any().item():
            raise ValueError("physical_role_ids contain an out-of-range role")

        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("attention_mask must be a tensor")
        if tuple(attention_mask.shape) != (1, n_slots):
            raise ValueError("attention_mask must have shape (1, n_slots)")
        if attention_mask.dtype != torch.bool:
            raise ValueError("attention_mask must use bool dtype")
        if not attention_mask.any().item():
            raise ValueError("attention_mask must expose at least one field slot")

        if isinstance(view.response_start, bool) or not isinstance(
            view.response_start, int
        ):
            raise ValueError("response_start must be an integer")
        if isinstance(view.response_stop, bool) or not isinstance(
            view.response_stop, int
        ):
            raise ValueError("response_stop must be an integer")
        if not 0 <= view.response_start < view.response_stop <= n_slots:
            raise ValueError("response slot bounds are outside the field")
        response_width = view.response_stop - view.response_start

        targets = tuple(view.score_targets)
        if not targets:
            raise ValueError("score_targets must not be empty")
        if not all(isinstance(target, str) for target in targets):
            raise ValueError("score_targets must contain only strings")
        if len(set(targets)) != len(targets):
            raise ValueError("score_targets must be unique")
        for target in targets:
            assert_supported_text(target)
            if len(target) > response_width:
                raise ValueError(
                    f"target length {len(target)} exceeds response slot width "
                    f"{response_width}"
                )

        # Clone every model-visible tensor before changing device or dtype.
        return _PreparedFieldView(
            field16=field16.detach().clone().to(self.device, self.dtype),
            physical_role_ids=role_ids.detach().clone().to(
                self.device,
                torch.long,
            ),
            attention_mask=attention_mask.detach().clone().to(
                self.device,
                torch.bool,
            ),
            response_slice=slice(view.response_start, view.response_stop),
            score_targets=targets,
        )

    def _target_indices(
        self,
        target: str,
        *,
        response_width: int,
    ) -> torch.Tensor:
        indices = torch.full(
            (1, response_width),
            self._empty_index,
            device=self.device,
            dtype=torch.long,
        )
        for position, character in enumerate(target):
            try:
                indices[0, position] = self._char_to_index[character]
            except KeyError as exc:  # Defensive; assert_supported_text ran first.
                raise ValueError(
                    f"target contains unsupported character {character!r}"
                ) from exc
        return indices

    def evaluate(
        self,
        visible_input: FieldView | Mapping[str, Any],
        soul_state: SoulState,
    ) -> SoulCoreAdapterResult:
        """Run one immutable real-core evaluation and retain proposal logits."""

        prepared = self._prepare(visible_input)
        validate_soul_state(soul_state)
        if int(soul_state.cfg.d_model) != int(self.core.cfg.d_model):
            raise ValueError(
                f"soul d_model {soul_state.cfg.d_model} does not match core "
                f"d_model {self.core.cfg.d_model}"
            )
        if int(soul_state.tensor.shape[-1]) != int(self.core.cfg.d_model):
            raise ValueError("soul tensor width does not match core d_model")

        soul_before = soul_state_sha256(soul_state)
        soul = soul_state.tensor.detach().clone().unsqueeze(0).to(
            self.device,
            self.dtype,
        )
        soul_mask = soul_state.active.detach().clone().unsqueeze(0).to(
            self.device,
            torch.bool,
        )
        module_modes = {
            module: bool(module.training)
            for module in self.core.modules()
        }

        try:
            self.core.eval()
            with torch.inference_mode():
                output = self.core.forward_charslot(
                    field16=prepared.field16,
                    region_ids=prepared.physical_role_ids,
                    soul=soul,
                    mask=prepared.attention_mask,
                    soul_mask=soul_mask,
                    response_slice=prepared.response_slice,
                )
                delta = output.get("response_delta_16")
                if not isinstance(delta, torch.Tensor):
                    raise RuntimeError(
                        "AxonCore did not return response_delta_16"
                    )
                logits = self.core.charslot_logits(delta, self._bank_unit)
                response_width = (
                    prepared.response_slice.stop
                    - prepared.response_slice.start
                )
                expected_shape = (
                    1,
                    response_width,
                    len(self._chars),
                )
                if tuple(logits.shape) != expected_shape:
                    raise RuntimeError(
                        f"proposal logits shape {tuple(logits.shape)} does not "
                        f"match expected {expected_shape}"
                    )
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError("proposal logits contain non-finite values")

                predicted_indices = logits.argmax(dim=-1)[0].tolist()
                prediction = "".join(
                    ""
                    if index == self._empty_index
                    else self._chars[index]
                    for index in predicted_indices
                )
                log_probability = F.log_softmax(logits.float(), dim=-1)
                target_nll: dict[str, float] = {}
                for target in prepared.score_targets:
                    target_indices = self._target_indices(
                        target,
                        response_width=response_width,
                    )
                    nll = -log_probability.gather(
                        -1,
                        target_indices.unsqueeze(-1),
                    ).squeeze(-1).mean()
                    value = float(nll.item())
                    if not math.isfinite(value) or value < 0.0:
                        raise RuntimeError(
                            f"target NLL is invalid for {target!r}"
                        )
                    target_nll[target] = value
                logits_for_report = logits.detach().float().cpu().clone()
        finally:
            # Preserve even intentionally mixed train/eval submodule modes.
            for module, was_training in module_modes.items():
                module.training = was_training

        if soul_state_sha256(soul_state) != soul_before:
            raise RuntimeError("AxonCore causal evaluation mutated soul state")
        return SoulCoreAdapterResult(
            response=SoulResponderOutput(
                prediction=prediction,
                target_nll=target_nll,
            ),
            logits=logits_for_report,
        )

    def __call__(
        self,
        visible_input: FieldView | Mapping[str, Any],
        soul_state: SoulState,
    ) -> SoulResponderOutput:
        """Implement the generic ``SoulResponder`` callback contract."""

        return self.evaluate(visible_input, soul_state).response
