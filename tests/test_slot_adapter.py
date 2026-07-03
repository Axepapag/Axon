"""Tests for adapters/slot_adapter.py frozen gates + char-prototype table."""
from __future__ import annotations

import numpy as np
import pytest

from adapters.slot_adapter import (
    ADAPTER_D_MODELS,
    CharPrototypeTable,
    SlotAdapter,
    get_adapter,
    get_char_prototype_table,
    run_char_prototype_check,
)
from substrate import default_alphabet


@pytest.mark.parametrize("d_model", ADAPTER_D_MODELS)
def test_adapter_snap_idempotence(d_model: int) -> None:
    adapter = SlotAdapter(d_model=d_model)
    report = adapter.check_snap_idempotence(n_test_slots=20)
    assert report.passed, f"snap-idempotence failed at d={d_model}"


@pytest.mark.parametrize("d_model", ADAPTER_D_MODELS)
def test_adapter_separability(d_model: int) -> None:
    adapter = SlotAdapter(d_model=d_model)
    report = adapter.check_separability(n_pairs=50)
    assert report.passed, f"separability failed at d={d_model}"


@pytest.mark.parametrize("d_model", ADAPTER_D_MODELS)
def test_char_prototype_roundtrip(d_model: int) -> None:
    """Every alphabet character must round-trip exactly through encode->decode."""
    table = CharPrototypeTable(d_model=d_model)
    ok, fails = table.check_exact()
    assert ok, f"char-prototype round-trip failed at d={d_model}: {fails[:5]}"


@pytest.mark.parametrize("d_model", ADAPTER_D_MODELS)
def test_char_prototype_batch_decode(d_model: int) -> None:
    table = CharPrototypeTable(d_model=d_model)
    chars = default_alphabet()
    vecs = np.stack([table.encode(c) for c in chars], axis=0)
    decoded = table.decode_batch(vecs)
    assert len(decoded) == 1
    assert decoded[0] == chars


@pytest.mark.parametrize("d_model", ADAPTER_D_MODELS)
def test_char_prototype_logits_shape(d_model: int) -> None:
    table = CharPrototypeTable(d_model=d_model)
    import torch
    vecs = torch.randn(2, 10, d_model)
    logits = table.logits(vecs)
    assert logits.shape == (2, 10, len(default_alphabet()))


def test_get_char_prototype_table_registry() -> None:
    a = get_char_prototype_table(64)
    b = get_char_prototype_table(64)
    assert a is b


def test_run_char_prototype_check() -> None:
    assert run_char_prototype_check(verbose=False)
