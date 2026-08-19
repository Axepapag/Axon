"""Runtime-safe exact 16D character-field materialization utilities.

This code was extracted from the legacy trainer so inference and council
runtime no longer import a training program.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from substrate import SLOT_DIM as SUBSTRATE_SLOT_DIM
from substrate import (
    assert_supported_text,
    char_to_slot,
    default_alphabet,
    get_letter_bank,
    roundtrip_check,
)


CHARSLOT_REGION_HISTORY = 0
CHARSLOT_REGION_USER = 1
CHARSLOT_REGION_RESPONSE = 2


def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


class CharSlotFieldBuilder:
    """Checkpoint-compatible fixed view used by the bootstrap runtime only."""

    def __init__(
        self,
        history_chars: int,
        user_chars: int,
        resp_chars: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.history_chars = history_chars
        self.user_chars = user_chars
        self.resp_chars = resp_chars
        self.n_slots = history_chars + user_chars + resp_chars
        self.resp_slice = slice(history_chars + user_chars, self.n_slots)
        self.device = device
        self.dtype = dtype
        self.bank = get_letter_bank()
        self.char_index = {char: index for index, char in enumerate(self.bank.chars)}
        self.empty_index = self.bank.empty_index
        self.bank_unit = torch.from_numpy(self.bank.vecs_unit.copy()).to(device, dtype)
        region = np.zeros((self.n_slots,), dtype=np.int64)
        region[history_chars : history_chars + user_chars] = CHARSLOT_REGION_USER
        region[self.resp_slice] = CHARSLOT_REGION_RESPONSE
        self._region = torch.from_numpy(region).unsqueeze(0).to(device)

    def _write_block(self, out: np.ndarray, text: str, offset: int, width: int) -> None:
        clipped = text[:width]
        assert_supported_text(clipped)
        for index, char in enumerate(clipped):
            out[offset + index] = char_to_slot(char)

    def _char_targets(self, answer: str, loss_prefix_mask: int = 0) -> np.ndarray:
        targets = np.full((self.resp_chars,), self.empty_index, dtype=np.int64)
        clipped = answer[: self.resp_chars]
        assert_supported_text(clipped)
        for index, char in enumerate(clipped):
            targets[index] = self.char_index[char]
        if loss_prefix_mask > 0:
            targets[: min(loss_prefix_mask, self.resp_chars)] = -100
        return targets

    def build(
        self,
        history: str,
        user_input: str,
        answer: str,
        draft_text: str,
        loss_prefix_mask: int = 0,
    ) -> dict[str, torch.Tensor]:
        field16 = np.zeros((self.n_slots, SUBSTRATE_SLOT_DIM), dtype=np.float32)
        self._write_block(field16, history[-self.history_chars :], 0, self.history_chars)
        self._write_block(field16, user_input, self.history_chars, self.user_chars)
        self._write_block(field16, draft_text, self.resp_slice.start, self.resp_chars)
        return {
            "field16": torch.from_numpy(field16).unsqueeze(0).to(self.device, self.dtype),
            "region": self._region,
            "targets": torch.from_numpy(
                self._char_targets(answer, loss_prefix_mask)
            ).unsqueeze(0).to(self.device),
        }

    def decode(self, logits: torch.Tensor, n_chars: int) -> str:
        indices = logits.argmax(dim=-1)[0].tolist()
        return "".join(
            "" if index == self.empty_index else self.bank.chars[index]
            for index in indices[:n_chars]
        )

    def verify_roundtrip(self, text: str) -> str:
        assert_supported_text(text)
        if len(text) > self.n_slots:
            raise ValueError(
                f"roundtrip input length {len(text)} exceeds builder slots {self.n_slots}"
            )
        field16 = np.zeros((self.n_slots, SUBSTRATE_SLOT_DIM), dtype=np.float32)
        self._write_block(field16, text, 0, self.n_slots)
        field = torch.from_numpy(field16).to(self.device, self.dtype)
        logits = (field @ self.bank_unit.T).unsqueeze(0)
        decoded = self.decode(logits, len(text))
        if decoded != text:
            raise RuntimeError(f"16D charslot roundtrip mismatch: {text!r} -> {decoded!r}")
        return decoded


def run_charslot_roundtrip_gate(builder: CharSlotFieldBuilder) -> None:
    if SUBSTRATE_SLOT_DIM != 16:
        raise RuntimeError(
            f"substrate slot width changed: expected 16, got {SUBSTRATE_SLOT_DIM}"
        )
    total, failures, details = roundtrip_check()
    if failures:
        raise RuntimeError(f"substrate roundtrip failed for {failures}/{total}: {details!r}")
    alphabet = "".join(default_alphabet())
    for start in range(0, len(alphabet), builder.n_slots):
        builder.verify_roundtrip(alphabet[start : start + builder.n_slots])
    builder.verify_roundtrip("axon 16D exact\nround trip: Aa0!?[]{}")
    log("GATE", f"exact 16D charslot roundtrip passed for all {total} supported characters")


__all__ = [
    "CHARSLOT_REGION_HISTORY",
    "CHARSLOT_REGION_USER",
    "CHARSLOT_REGION_RESPONSE",
    "CharSlotFieldBuilder",
    "run_charslot_roundtrip_gate",
    "log",
]
