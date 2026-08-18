"""Launch-blocking tests for the exact 16D character substrate."""

import pytest
import torch

from substrate import SLOT_DIM, default_alphabet
from training.trainer_slot import CharSlotFieldBuilder, run_charslot_roundtrip_gate


def _builder() -> CharSlotFieldBuilder:
    return CharSlotFieldBuilder(256, 64, 64, torch.device("cpu"), torch.float32)


def test_charslot_builder_roundtrips_every_supported_character_exactly() -> None:
    assert SLOT_DIM == 16
    text = "".join(default_alphabet())
    assert _builder().verify_roundtrip(text) == text


def test_charslot_startup_gate_passes() -> None:
    run_charslot_roundtrip_gate(_builder())


def test_charslot_builder_rejects_unsupported_input() -> None:
    with pytest.raises(ValueError, match="unsupported substrate characters"):
        _builder().build("history", "caf\u00e9", "answer", "draft")


def test_charslot_roundtrip_refuses_silent_truncation() -> None:
    builder = CharSlotFieldBuilder(1, 1, 1, torch.device("cpu"), torch.float32)
    with pytest.raises(ValueError, match="exceeds builder slots"):
        builder.verify_roundtrip("abcd")
