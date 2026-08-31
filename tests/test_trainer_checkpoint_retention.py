from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from runtime.trainer import (
    OrganKind,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterMutationPolicy,
    TrainerControlPlane,
    TrainerStoreError,
)
from runtime.trainer.store import CHECKPOINT_RETENTION
from tests._trainer_preflight import unit_preflight_receipt


class TinyCore(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.body = nn.Linear(width, width, bias=False)
        self.adapter = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.adapter(x)


def _setup(tmp_path: Path, *, max_steps: int = 8):
    torch.manual_seed(13)
    module = TinyCore()
    descriptor = ParameterModuleDescriptor(
        module_id="semantic-core",
        organ_kind=OrganKind.SEMANTIC_CORE,
        generation_id="semantic-g0",
        architecture="tiny-test-core",
        d_model=4,
    )
    control = TrainerControlPlane.active(state_root=tmp_path)
    control.declare_expected((descriptor,))
    control.register(descriptor, module)
    inventory = control.snapshot_inventory(exact_value_hashes=True)
    grant = ParameterMutationGrant(
        grant_id="adapter-grant",
        module_id=descriptor.module_id,
        generation_id=descriptor.generation_id,
        policy=ParameterMutationPolicy.ADAPTER_ONLY,
        allowed_prefixes=("adapter.",),
        max_trainable_parameters=16,
    )
    plan = ParameterMutationPlan(
        base_inventory_id=inventory.inventory_id,
        module_id=descriptor.module_id,
        base_generation_id=descriptor.generation_id,
        candidate_generation_id="semantic-g1-candidate",
        tensor_names=("adapter.weight",),
        optimizer_name="SGD",
        learning_rate=0.05,
        max_steps=max_steps,
        source_manifest_ids=("study-source",),
        holdout_manifest_ids=("holdout",),
    )
    return control, module, descriptor, inventory, grant, plan


def test_session_checkpoint_retention_keeps_newest_payloads_and_all_records(
    tmp_path: Path,
) -> None:
    control, _live, descriptor, inventory, grant, plan = _setup(tmp_path)
    session = control.begin_candidate(
        inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan)
    )
    x = torch.ones(1, 4)
    records = []
    for _ in range(CHECKPOINT_RETENTION + 2):
        session.step(lambda candidate: candidate(x).square().mean())
        records.append(session.checkpoint(include_optimizer=True))

    candidate_root = (
        tmp_path
        / "training"
        / "trainer"
        / "candidates"
        / descriptor.module_id
        / plan.candidate_generation_id
    )
    artifacts = sorted((candidate_root / "checkpoints").glob("*.pt"))
    durable_records = sorted((candidate_root / "checkpoint_records").glob("*.json"))

    assert len(records) == CHECKPOINT_RETENTION + 2
    assert len(artifacts) == CHECKPOINT_RETENTION
    assert len(durable_records) == CHECKPOINT_RETENTION + 2
    assert {path.name for path in artifacts} == {
        f"{record.artifact_sha256}.pt" for record in records[-CHECKPOINT_RETENTION:]
    }

    control.store.load_verified_candidate_checkpoint(records[-1])
    with pytest.raises(TrainerStoreError, match="missing"):
        control.store.load_verified_candidate_checkpoint(records[0])


def test_prune_is_idempotent_and_preserves_latest_pointer(tmp_path: Path) -> None:
    control, _live, descriptor, inventory, grant, plan = _setup(tmp_path)
    session = control.begin_candidate(
        inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan)
    )
    x = torch.ones(1, 4)
    for _ in range(2):
        session.step(lambda candidate: candidate(x).square().mean())
        session.checkpoint(include_optimizer=True)

    first = control.store.prune_candidate_checkpoints(
        descriptor.module_id, plan.candidate_generation_id
    )
    second = control.store.prune_candidate_checkpoints(
        descriptor.module_id, plan.candidate_generation_id
    )
    assert first == ()
    assert second == ()
    candidate_root = (
        tmp_path
        / "training"
        / "trainer"
        / "candidates"
        / descriptor.module_id
        / plan.candidate_generation_id
    )
    assert len(list((candidate_root / "checkpoints").glob("*.pt"))) == 2


def test_prune_rejects_non_positive_keep(tmp_path: Path) -> None:
    control, _live, descriptor, inventory, grant, plan = _setup(tmp_path)
    control.begin_candidate(
        inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan)
    )
    with pytest.raises(ValueError, match="keep"):
        control.store.prune_candidate_checkpoints(
            descriptor.module_id, plan.candidate_generation_id, keep=0
        )
