from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from runtime.trainer import (
    IncompleteParameterInventoryError,
    OrganKind,
    ParameterAuthorityError,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterMutationPolicy,
    ParameterRegistry,
    TrainerControlPlane,
    TrainerCoreRole,
    TrainerStateStore,
    authorize_parameter_mutation,
    capture_parameter_telemetry,
)


class TinyCore(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.body = nn.Linear(width, width, bias=False)
        self.adapter = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.adapter(x)


def descriptor(
    module_id: str,
    *,
    kind: OrganKind,
    width: int,
    generation: str = "g0",
    role: TrainerCoreRole | None = None,
) -> ParameterModuleDescriptor:
    return ParameterModuleDescriptor(
        module_id=module_id,
        organ_kind=kind,
        generation_id=generation,
        architecture="tiny-test-transformer",
        d_model=width,
        trainer_core_role=role,
    )


def test_registry_requires_complete_heterogeneous_organism_inventory() -> None:
    registry = ParameterRegistry()
    expected = (
        descriptor("reasoner-64", kind=OrganKind.REASONING_CORE, width=64),
        descriptor("semantic-256", kind=OrganKind.SEMANTIC_CORE, width=256),
        descriptor(
            "trainer-evaluator-128",
            kind=OrganKind.TRAINER_CORE,
            width=128,
            role=TrainerCoreRole.EVALUATOR,
        ),
    )
    registry.declare_expected(expected)
    registry.register(expected[0], TinyCore(64))
    registry.register(expected[1], TinyCore(256))

    with pytest.raises(IncompleteParameterInventoryError, match="trainer-evaluator-128"):
        registry.capture_inventory(exact_value_hashes=False)

    registry.register(expected[2], TinyCore(128))
    inventory = registry.capture_inventory(exact_value_hashes=False)
    assert inventory.complete
    assert [item.descriptor.d_model for item in inventory.manifests] == [64, 256, 128]
    assert {item.descriptor.organ_kind for item in inventory.manifests} == {
        OrganKind.REASONING_CORE,
        OrganKind.SEMANTIC_CORE,
        OrganKind.TRAINER_CORE,
    }


def test_exact_inventory_hash_changes_when_one_parameter_changes() -> None:
    registry = ParameterRegistry()
    desc = descriptor("reasoner", kind=OrganKind.REASONING_CORE, width=8)
    module = TinyCore(8)
    registry.declare_expected((desc,))
    registry.register(desc, module)
    before = registry.capture_inventory(exact_value_hashes=True)

    with torch.no_grad():
        module.body.weight[0, 0] += 1.0

    after = registry.capture_inventory(exact_value_hashes=True)
    assert before.inventory_id != after.inventory_id
    before_hashes = {item.name: item.value_sha256 for item in before.module("reasoner").tensors}
    after_hashes = {item.name: item.value_sha256 for item in after.module("reasoner").tensors}
    assert before_hashes["body.weight"] != after_hashes["body.weight"]
    assert before_hashes["adapter.weight"] == after_hashes["adapter.weight"]


def test_parameter_authority_is_stale_and_scope_fail_closed() -> None:
    registry = ParameterRegistry()
    desc = descriptor("semantic", kind=OrganKind.SEMANTIC_CORE, width=8)
    module = TinyCore(8)
    registry.declare_expected((desc,))
    registry.register(desc, module)
    inventory = registry.capture_inventory(exact_value_hashes=True)

    adapter_grant = ParameterMutationGrant(
        grant_id="adapter-only",
        module_id="semantic",
        generation_id="g0",
        policy=ParameterMutationPolicy.ADAPTER_ONLY,
        allowed_prefixes=("adapter.",),
        max_trainable_parameters=64,
    )
    allowed = ParameterMutationPlan(
        base_inventory_id=inventory.inventory_id,
        module_id="semantic",
        base_generation_id="g0",
        candidate_generation_id="g1",
        tensor_names=("adapter.weight",),
        optimizer_name="AdamW",
        learning_rate=1e-4,
        max_steps=10,
        source_manifest_ids=("study-source",),
        holdout_manifest_ids=("heldout",),
    )
    receipt = authorize_parameter_mutation(inventory, adapter_grant, allowed)
    assert receipt.parameter_count == 64

    body_plan = ParameterMutationPlan(
        base_inventory_id=inventory.inventory_id,
        module_id="semantic",
        base_generation_id="g0",
        candidate_generation_id="g2",
        tensor_names=("body.weight",),
        optimizer_name="AdamW",
        learning_rate=1e-4,
        max_steps=10,
        source_manifest_ids=("study-source",),
        holdout_manifest_ids=("heldout",),
    )
    with pytest.raises(ParameterAuthorityError, match="adapter-only"):
        authorize_parameter_mutation(inventory, adapter_grant, body_plan)

    with torch.no_grad():
        module.adapter.weight[0, 0] += 0.25
    newer = registry.capture_inventory(exact_value_hashes=True)
    assert newer.inventory_id != inventory.inventory_id
    with pytest.raises(ParameterAuthorityError, match="stale"):
        authorize_parameter_mutation(newer, adapter_grant, allowed)


def test_telemetry_covers_every_parameter_and_present_gradient() -> None:
    module = TinyCore(4)
    x = torch.ones(1, 4)
    loss = module(x).square().mean()
    loss.backward()

    frame = capture_parameter_telemetry(
        module,
        module_id="reasoner",
        generation_id="g0",
        step=1,
        inventory_id="inventory",
    )
    assert {item.name for item in frame.stats} == {"adapter.weight", "body.weight"}
    assert all(item.grad_present for item in frame.stats)
    payload = frame.to_canonical_dict()
    assert payload["all_values_finite"] is True
    assert payload["all_present_gradients_finite"] is True
    assert payload["parameter_count"] == sum(parameter.numel() for parameter in module.parameters())


def test_control_plane_persists_inventory_plan_authorization_and_telemetry(tmp_path: Path) -> None:
    desc = descriptor("trainer-core", kind=OrganKind.TRAINER_CORE, width=4, role=TrainerCoreRole.OPTIMIZER)
    module = TinyCore(4)
    control = TrainerControlPlane(registry=ParameterRegistry(), store=TrainerStateStore(tmp_path / "trainer"))
    control.declare_expected((desc,))
    control.register(desc, module)
    inventory = control.snapshot_inventory(exact_value_hashes=True)

    plan = ParameterMutationPlan(
        base_inventory_id=inventory.inventory_id,
        module_id="trainer-core",
        base_generation_id="g0",
        candidate_generation_id="g1",
        tensor_names=("body.weight",),
        optimizer_name="AdamW",
        learning_rate=1e-4,
        max_steps=3,
        source_manifest_ids=("source-a",),
        holdout_manifest_ids=("eval-a",),
    )
    grant = ParameterMutationGrant(
        grant_id="explicit",
        module_id="trainer-core",
        generation_id="g0",
        policy=ParameterMutationPolicy.EXPLICIT_NAMES,
        allowed_names=("body.weight",),
    )
    authorization = control.authorize(inventory, grant, plan)
    frame = control.record_telemetry("trainer-core", step=0, inventory_id=inventory.inventory_id)

    assert (tmp_path / "trainer" / "inventories" / f"{inventory.inventory_id}.json").exists()
    assert (tmp_path / "trainer" / "plans" / f"{plan.plan_id}.json").exists()
    assert (tmp_path / "trainer" / "authorizations" / f"{authorization.authorization_id}.json").exists()
    latest = json.loads((tmp_path / "trainer" / "latest_telemetry.json").read_text(encoding="utf-8"))
    assert latest["frame_id"] == frame.frame_id
    assert len((tmp_path / "trainer" / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()) == 1
