from __future__ import annotations

from pathlib import Path

import pytest
import torch

from runtime.field import canonical_sha256
from runtime.trainer.attempt_workspace import AttemptWorkspaceError, AttemptWorkspaceStore


def _save(store: AttemptWorkspaceStore, model, optimizer, index: int):
    return store.save(
        assignment_id="a" * 64,
        core_id="core-alpha",
        attempt_id=canonical_sha256({"attempt": index}),
        attempt_index=index,
        parameter_generation=canonical_sha256({"parameters": index}),
        optimizer_generation=canonical_sha256({"optimizer": index}),
        model=model,
        optimizer=optimizer,
    )


def test_attempt_workspace_retains_three_and_loads_exact_latest(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    store = AttemptWorkspaceStore(tmp_path, retain=3)
    latest = None
    for index in range(4):
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 4)).sum().backward()
        optimizer.step()
        latest = _save(store, model, optimizer, index)

    recovered = store.load_latest(assignment_id="a" * 64, core_id="core-alpha")
    assert recovered is not None
    record, payload = recovered
    assert record == latest
    assert len(list(store.root.rglob("????????.json"))) == 3
    assert len(list(store.root.rglob("????????.pt"))) == 3
    for name, tensor in model.state_dict().items():
        assert torch.equal(payload["module_state_dict"][name], tensor)


def test_attempt_workspace_rejects_corrupt_payload(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    store = AttemptWorkspaceStore(tmp_path)
    _save(store, model, optimizer, 0)
    payload_path = next(store.root.rglob("????????.pt"))
    payload_path.write_bytes(payload_path.read_bytes() + b"corrupt")

    with pytest.raises(AttemptWorkspaceError, match="payload hash mismatch"):
        store.load_latest(assignment_id="a" * 64, core_id="core-alpha")
