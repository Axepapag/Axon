from __future__ import annotations

import torch

from runtime.axon_runtime import ContinuousCoreD512, ContinuousCoreD512Config
from substrate import TRANSPORT_VOCAB_SIZE, encode_unicode_text, unicode_transport_bank
from training.continuous_core_d512 import (
    build_d512_copy_curriculum,
    collate_d512_copy_cases,
    d512_copy_loss,
)


def test_d512_anatomy_is_one_gru_no_attention_and_exact_bank() -> None:
    model = ContinuousCoreD512()
    assert model.cfg.d_model == 512
    assert model.cfg.output_classes == TRANSPORT_VOCAB_SIZE + 1
    assert model.bank16.shape == (TRANSPORT_VOCAB_SIZE, 16)
    assert torch.equal(model.bank16, torch.tensor(unicode_transport_bank()))
    assert isinstance(model.cell, torch.nn.GRUCell)
    names = {type(module).__name__ for module in model.modules()}
    assert not any("Attention" in name or "Transformer" in name for name in names)


def test_output_categories_map_back_to_exact_registered_d16() -> None:
    model = ContinuousCoreD512()
    ids = torch.tensor(encode_unicode_text("CAT λ🙂"), dtype=torch.long)
    cells = model.exact_output_cells(ids)
    assert torch.equal(cells, model.bank16.index_select(0, ids))
    try:
        model.exact_output_cells(torch.tensor([model.eos_id]))
    except ValueError as exc:
        assert "do not serialize" in str(exc)
    else:
        raise AssertionError("private EOS must never serialize as a D16 transport cell")


def test_variable_length_teacher_forced_output_includes_private_eos() -> None:
    curriculum = build_d512_copy_curriculum()
    model = ContinuousCoreD512()
    one = next(case for case in curriculum.train_cases if case.transport_length == 1)
    longer = next(case for case in curriculum.train_cases if case.transport_length >= 6)
    for case in (one, longer):
        cells, targets = collate_d512_copy_cases((case,), device=torch.device("cpu"))
        logits, state = model.teacher_forced_logits(cells, targets)
        assert logits.shape == (1, case.transport_length + 1, TRANSPORT_VOCAB_SIZE + 1)
        assert state.shape == (1, 512)
        loss = d512_copy_loss(model, cells, targets)
        assert torch.isfinite(loss)
        loss.backward()
        model.zero_grad(set_to_none=True)


def test_copy_curriculum_is_exact_variable_length_disjoint_and_native_complete() -> None:
    first = build_d512_copy_curriculum()
    second = build_d512_copy_curriculum()
    assert first.curriculum_id == second.curriculum_id
    assert first.train_manifest_id == second.train_manifest_id
    assert first.heldout_manifest_id == second.heldout_manifest_id
    assert {case.text for case in first.train_cases}.isdisjoint({case.text for case in first.heldout_cases})
    assert min(case.transport_length for case in first.train_cases) == 1
    assert max(case.transport_length for case in first.train_cases) > 8
    train_ids = {token for case in first.train_cases for token in case.token_ids}
    assert set(range(95)) <= train_ids
