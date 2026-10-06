from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from core.e0_two_state import (
    CHECKPOINT_FORMAT,
    EMPTY_ID,
    E0TwoStateCore,
    load_checkpoint,
    save_checkpoint,
)
from substrate.substrate_1024 import lane_bank

BATCH = 4
WIDTH = 512
HIDDEN = 512


def make_core() -> E0TwoStateCore:
    torch.manual_seed(20261006)
    return E0TwoStateCore(width=WIDTH, hidden_size=HIDDEN, max_chunk=64)


def test_native_cells_buffer_is_the_frozen_bank() -> None:
    core = make_core()
    reference = torch.tensor(lane_bank(), dtype=torch.float32)

    assert torch.equal(core.native_cells, reference)
    assert core.native_cells.requires_grad is False
    assert "native_cells" in dict(core.named_buffers())
    assert all(param is not core.native_cells for param in core.parameters())
    assert core.native_cells.shape == (96, 16)


def test_step_shapes() -> None:
    core = make_core().eval()
    char_ids = torch.tensor([0, 94, 95, 65], dtype=torch.int64)
    reasoning, response = core.initial_states(BATCH)

    assert reasoning.shape == (BATCH, HIDDEN)
    assert response.shape == (BATCH, HIDDEN)

    new_reasoning, new_response, char_logits, control_logits = core.step(
        char_ids, reasoning, response
    )

    assert new_reasoning.shape == (BATCH, HIDDEN)
    assert new_response.shape == (BATCH, HIDDEN)
    assert char_logits.shape == (BATCH, 96)
    assert control_logits.shape == (BATCH, 3)


def test_ids_outside_the_frozen_alphabet_fail_closed() -> None:
    core = make_core().eval()
    reasoning, response = core.initial_states(BATCH)

    for bad_id in (96, -1):
        bad = torch.full((BATCH,), bad_id, dtype=torch.int64)
        with pytest.raises(ValueError):
            core.step(bad, reasoning, response)


def test_checkpoint_roundtrip_reproduces_outputs(tmp_path: Path) -> None:
    core = make_core().eval()
    reasoning, response = core.initial_states(BATCH)
    path = tmp_path / "nest" / "e0.pt"

    for tick in range(3):
        char_ids = torch.tensor([(tick * 7 + offset * 13) % 96 for offset in range(BATCH)])
        reasoning, response, _, _ = core.step(char_ids, reasoning, response)

    save_checkpoint(path, core, reasoning, response, core.config())
    assert path.is_file()
    sidecar = Path(str(path) + ".sha256")
    assert sidecar.is_file()
    digest, name = sidecar.read_text(encoding="utf-8").split()
    assert len(digest) == 64 and name == path.name

    restored, restored_reasoning, restored_response, config = load_checkpoint(path)
    restored.eval()
    assert config == {"width": WIDTH, "hidden_size": HIDDEN, "max_chunk": 64}
    assert torch.equal(restored.native_cells, core.native_cells)
    for original, loaded in zip(core.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(original, loaded)

    next_ids = torch.tensor([3, 17, 94, 0], dtype=torch.int64)
    with torch.no_grad():
        continued = core.step(next_ids, reasoning, response)
        reloaded = restored.step(next_ids, restored_reasoning, restored_response)
    for original, loaded in zip(continued, reloaded):
        assert torch.equal(original, loaded)

    tampered = tmp_path / "tampered.pt"
    shutil.copyfile(path, tampered)
    shutil.copyfile(sidecar, Path(str(tampered) + ".sha256"))
    blob = bytearray(tampered.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    tampered.write_bytes(bytes(blob))
    with pytest.raises(ValueError):
        load_checkpoint(tampered)

    orphan = tmp_path / "orphan.pt"
    shutil.copyfile(path, orphan)
    with pytest.raises(ValueError):
        load_checkpoint(orphan)


def test_step_shapes_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda:0")
    core = make_core().to(device).eval()
    char_ids = torch.tensor([0, 94, 95, 65], dtype=torch.int64, device=device)
    reasoning, response = core.initial_states(BATCH, device=device)

    new_reasoning, new_response, char_logits, control_logits = core.step(
        char_ids, reasoning, response
    )

    assert core.native_cells.device == device
    assert torch.equal(
        core.surface_from_ids(char_ids)[:, :16].cpu(), core.native_cells.cpu()[char_ids.cpu()]
    )
    assert new_reasoning.shape == (BATCH, HIDDEN)
    assert new_response.shape == (BATCH, HIDDEN)
    assert char_logits.shape == (BATCH, 96)
    assert control_logits.shape == (BATCH, 3)
    assert new_reasoning.device == device
    assert new_response.device == device
    assert char_logits.device == device
    assert control_logits.device == device


def test_parameter_budget_fits_a_4gib_card() -> None:
    core = make_core()
    total = sum(param.numel() for param in core.parameters())
    assert total < 8_000_000, f"E0 core has {total} parameters"


def test_surface_is_exact_rung_zero_and_shape_mismatch_fails_closed() -> None:
    core = make_core().eval()
    ids = torch.tensor([0, 94, 95, 65], dtype=torch.int64)

    surface = core.surface_from_ids(ids).reshape(BATCH, core.lanes, 16)
    cells = core.native_cells
    for row, char_id in enumerate(ids.tolist()):
        assert torch.equal(surface[row, 0], cells[char_id])
        assert torch.equal(surface[row, 1:], cells[EMPTY_ID].expand(core.lanes - 1, 16))
    assert surface.shape == (BATCH, WIDTH // 16, 16)

    with pytest.raises(ValueError):
        core.surface_from_ids(torch.tensor([[0, 1]], dtype=torch.int64))
    with pytest.raises(ValueError):
        core.surface_from_ids(torch.tensor([1.5, 2.5], dtype=torch.float32))

    good_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    reasoning, response = core.initial_states(BATCH)
    with pytest.raises(ValueError):
        core.step(good_ids, reasoning[:2], response)
    with pytest.raises(ValueError):
        core.step(good_ids, reasoning, response[:, : HIDDEN - 1])
    with pytest.raises(ValueError):
        core.step(good_ids, reasoning.to(torch.float64), response)
    with pytest.raises(ValueError):
        E0TwoStateCore(width=100, hidden_size=HIDDEN)
    assert CHECKPOINT_FORMAT == "axon-e0-checkpoint-v1"
