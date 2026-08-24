from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from runtime.trainer import (
    EvaluationObservation,
    EvaluationRequirement,
    MetricComparison,
    OrganKind,
    ParameterAuthorityError,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterMutationPolicy,
    ParameterPromotionProposal,
    ParameterRegistry,
    PromotionGate,
    TrainerControlPlane,
    TrainerExecutionError,
    TrainerStateStore,
    TrainerStoreError,
    inspect_trainer_state,
    parameter_value_sha256,
)
from tests._trainer_preflight import unit_preflight_receipt


class TinyCore(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.body = nn.Linear(width, width, bias=False)
        self.adapter = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.adapter(x)


def _setup(tmp_path: Path, *, max_steps: int = 3):
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


def test_candidate_optimizer_changes_only_authorized_clone_and_never_live_module(tmp_path: Path) -> None:
    control, live, _descriptor, inventory, grant, plan = _setup(tmp_path)
    live_before = {name: parameter_value_sha256(parameter) for name, parameter in live.named_parameters()}
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan))

    x = torch.tensor([[1.0, -0.5, 0.25, 2.0]])
    target = torch.zeros_like(x)
    receipt = session.step(lambda candidate: (candidate(x) - target).square().mean())

    assert receipt.changed_tensor_names == ("adapter.weight",)
    assert "body.weight" in receipt.unchanged_tensor_names
    assert {name: parameter_value_sha256(parameter) for name, parameter in live.named_parameters()} == live_before
    assert parameter_value_sha256(session.candidate_module.body.weight) == live_before["body.weight"]
    assert parameter_value_sha256(session.candidate_module.adapter.weight) != live_before["adapter.weight"]
    assert live.adapter.weight.data_ptr() != session.candidate_module.adapter.weight.data_ptr()

    lifecycle = (tmp_path / "training" / "trainer" / "candidate_lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
    steps = (tmp_path / "training" / "trainer" / "optimization_steps.jsonl").read_text(encoding="utf-8").splitlines()
    telemetry = (tmp_path / "training" / "trainer" / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lifecycle) == 2
    assert len(steps) == 1
    assert len(telemetry) == 1


def test_candidate_checkpoint_is_hash_verified_and_restorable(tmp_path: Path) -> None:
    control, live, _descriptor, inventory, grant, plan = _setup(tmp_path)
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan))
    x = torch.ones(1, 4)
    session.step(lambda candidate: candidate(x).square().mean())
    record = session.checkpoint(include_optimizer=True)
    checkpoint_manifest = session.candidate_manifest().manifest_id
    assert checkpoint_manifest == record.parameter_manifest_id

    session.step(lambda candidate: candidate(x).square().mean())
    assert session.candidate_manifest().manifest_id != checkpoint_manifest
    session.restore_checkpoint(record)
    assert session.step_index == 1
    assert session.candidate_manifest().manifest_id == checkpoint_manifest
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in session.candidate_module.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in live.parameters())

    artifact = tmp_path / "training" / "trainer" / record.artifact_relpath
    data = bytearray(artifact.read_bytes())
    data[-1] ^= 0x01
    artifact.write_bytes(data)
    with pytest.raises(TrainerStoreError, match="SHA256"):
        control.store.load_verified_candidate_checkpoint(record)


def test_nonfinite_loss_rejects_candidate_before_optimizer_step(tmp_path: Path) -> None:
    control, live, _descriptor, inventory, grant, plan = _setup(tmp_path)
    live_before = {name: parameter_value_sha256(parameter) for name, parameter in live.named_parameters()}
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan))
    candidate_before = {name: parameter_value_sha256(parameter) for name, parameter in session.candidate_module.named_parameters()}

    with pytest.raises(TrainerExecutionError, match="non-finite loss"):
        session.step(lambda candidate: candidate(torch.ones(1, 4)).sum() * torch.tensor(float("nan")))

    assert session.closed
    assert {name: parameter_value_sha256(parameter) for name, parameter in live.named_parameters()} == live_before
    assert {name: parameter_value_sha256(parameter) for name, parameter in session.candidate_module.named_parameters()} == candidate_before


def test_authorized_max_steps_is_hard_boundary(tmp_path: Path) -> None:
    control, _live, _descriptor, inventory, grant, plan = _setup(tmp_path, max_steps=1)
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan))
    x = torch.ones(1, 4)
    session.step(lambda candidate: candidate(x).square().mean())
    with pytest.raises(TrainerExecutionError, match="max_steps"):
        session.step(lambda candidate: candidate(x).square().mean())


def test_promotion_gate_requires_every_declared_capability_and_regression_suite(tmp_path: Path) -> None:
    control, _live, _descriptor, inventory, grant, plan = _setup(tmp_path)
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan))
    session.step(lambda candidate: candidate(torch.ones(1, 4)).square().mean())
    checkpoint = session.checkpoint(include_optimizer=False)

    gate = PromotionGate(
        gate_id="semantic-core-promotion-v1",
        module_id=plan.module_id,
        candidate_generation_id=plan.candidate_generation_id,
        required_suite_ids=("capability", "counterfactual", "forgetting"),
        requirements=(
            EvaluationRequirement(
                suite_id="capability",
                metric_name="accuracy",
                comparison=MetricComparison.GREATER_OR_EQUAL,
                threshold=0.80,
                label="held-out capability",
            ),
            EvaluationRequirement(
                suite_id="counterfactual",
                metric_name="input_use_delta",
                comparison=MetricComparison.GREATER_OR_EQUAL,
                threshold=0.10,
                label="causal input use",
            ),
            EvaluationRequirement(
                suite_id="forgetting",
                metric_name="retained_accuracy",
                comparison=MetricComparison.GREATER_OR_EQUAL,
                threshold=0.95,
                label="catastrophic forgetting guard",
            ),
        ),
    )

    capability = EvaluationObservation.from_mapping(
        suite_id="capability",
        module_id=plan.module_id,
        candidate_generation_id=plan.candidate_generation_id,
        metrics={"accuracy": 0.85},
    )
    missing = control.evaluate_gate(gate, (capability,))
    assert not missing.passed
    assert missing.missing_suite_ids == ("counterfactual", "forgetting")

    counterfactual = EvaluationObservation.from_mapping(
        suite_id="counterfactual",
        module_id=plan.module_id,
        candidate_generation_id=plan.candidate_generation_id,
        metrics={"input_use_delta": 0.15},
    )
    forgetting_fail = EvaluationObservation.from_mapping(
        suite_id="forgetting",
        module_id=plan.module_id,
        candidate_generation_id=plan.candidate_generation_id,
        metrics={"retained_accuracy": 0.90},
    )
    failed = control.evaluate_gate(gate, (capability, counterfactual, forgetting_fail))
    assert not failed.passed
    assert any("catastrophic forgetting guard" in item for item in failed.failed_requirements)

    forgetting_pass = EvaluationObservation.from_mapping(
        suite_id="forgetting",
        module_id=plan.module_id,
        candidate_generation_id=plan.candidate_generation_id,
        metrics={"retained_accuracy": 0.97},
    )
    passed = control.evaluate_gate(gate, (capability, counterfactual, forgetting_pass))
    assert passed.passed

    proposal = ParameterPromotionProposal(
        base_inventory_id=inventory.inventory_id,
        module_id=plan.module_id,
        base_generation_id=plan.base_generation_id,
        candidate_generation_id=plan.candidate_generation_id,
        candidate_artifact_id=checkpoint.checkpoint_id,
        lineage_manifest_id=checkpoint.parameter_manifest_id,
        evaluation_ids=(passed.decision_id,),
    )
    path = control.record_promotion_proposal(proposal, gate_decision=passed)
    assert path.exists()
    with pytest.raises(ParameterAuthorityError, match="failed promotion gate"):
        control.record_promotion_proposal(proposal, gate_decision=failed)


def test_gate_rejects_observation_from_another_candidate(tmp_path: Path) -> None:
    control, _live, _descriptor, _inventory, _grant, plan = _setup(tmp_path)
    gate = PromotionGate(
        gate_id="gate",
        module_id=plan.module_id,
        candidate_generation_id=plan.candidate_generation_id,
        requirements=(
            EvaluationRequirement(
                suite_id="heldout",
                metric_name="score",
                comparison=MetricComparison.GREATER_OR_EQUAL,
                threshold=1.0,
                label="heldout",
            ),
        ),
    )
    wrong = EvaluationObservation.from_mapping(
        suite_id="heldout",
        module_id=plan.module_id,
        candidate_generation_id="another-generation",
        metrics={"score": 1.0},
    )
    with pytest.raises(ValueError, match="another module or candidate generation"):
        control.evaluate_gate(gate, (wrong,))


class BufferMutatingCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(4, 4, bias=False)
        self.adapter = nn.Linear(4, 4, bias=False)
        self.register_buffer("running_signal", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.running_signal.add_(1.0)
        return self.body(x) + self.adapter(x)


def test_persistent_buffers_are_in_inventory_and_ungranted_mutation_fails_closed(tmp_path: Path) -> None:
    torch.manual_seed(17)
    live = BufferMutatingCore()
    descriptor = ParameterModuleDescriptor(
        module_id="buffer-core",
        organ_kind=OrganKind.REASONING_CORE,
        generation_id="g0",
        architecture="buffer-mutating-test",
        d_model=4,
    )
    control = TrainerControlPlane.active(state_root=tmp_path)
    control.declare_expected((descriptor,))
    control.register(descriptor, live)
    inventory = control.snapshot_inventory(exact_value_hashes=True)
    manifest = inventory.module("buffer-core")
    assert [item.name for item in manifest.buffers] == ["running_signal"]
    assert manifest.buffer_numel == 1
    assert inventory.state_numel == inventory.parameter_count + 1

    grant = ParameterMutationGrant(
        grant_id="adapter-only",
        module_id="buffer-core",
        generation_id="g0",
        policy=ParameterMutationPolicy.ADAPTER_ONLY,
        allowed_prefixes=("adapter.",),
    )
    plan = ParameterMutationPlan(
        base_inventory_id=inventory.inventory_id,
        module_id="buffer-core",
        base_generation_id="g0",
        candidate_generation_id="g1",
        tensor_names=("adapter.weight",),
        optimizer_name="SGD",
        learning_rate=0.01,
        max_steps=1,
        source_manifest_ids=("source",),
        holdout_manifest_ids=("holdout",),
    )
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan))
    live_buffer_before = live.running_signal.clone()
    adapter_before = session.candidate_module.adapter.weight.detach().clone()
    with pytest.raises(ParameterAuthorityError, match="persistent buffers"):
        session.step(lambda candidate: candidate(torch.ones(1, 4)).square().mean())
    assert session.closed
    assert torch.equal(live.running_signal, live_buffer_before)
    assert torch.equal(session.candidate_module.adapter.weight, adapter_before)


def test_read_only_trainer_inspection_exposes_current_candidate_state(tmp_path: Path) -> None:
    torch.manual_seed(23)
    module = TinyCore()
    descriptor = ParameterModuleDescriptor(
        module_id="inspect-core",
        organ_kind=OrganKind.TRAINER_CORE,
        generation_id="g0",
        architecture="inspect-test",
        d_model=4,
    )
    control = TrainerControlPlane.active(state_root=tmp_path)
    control.declare_expected((descriptor,))
    control.register(descriptor, module)
    inventory = control.snapshot_inventory(exact_value_hashes=True)
    grant = ParameterMutationGrant(
        grant_id="inspect-grant",
        module_id="inspect-core",
        generation_id="g0",
        policy=ParameterMutationPolicy.EXPLICIT_NAMES,
        allowed_names=("adapter.weight",),
    )
    plan = ParameterMutationPlan(
        base_inventory_id=inventory.inventory_id,
        module_id="inspect-core",
        base_generation_id="g0",
        candidate_generation_id="g1",
        tensor_names=("adapter.weight",),
        optimizer_name="SGD",
        learning_rate=0.01,
        max_steps=1,
        source_manifest_ids=("source",),
        holdout_manifest_ids=("holdout",),
    )
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan))
    session.step(lambda candidate: candidate(torch.ones(1, 4)).square().mean())
    checkpoint = session.checkpoint(include_optimizer=False)

    snapshot = inspect_trainer_state(state_root=tmp_path)
    payload = snapshot.to_canonical_dict()
    assert payload["writer_lease_present"] is True
    assert payload["latest_lifecycle"]["candidate_generation_id"] == plan.candidate_generation_id
    assert payload["latest_step"]["step"] == 1
    assert payload["latest_telemetry"]["generation_id"] == plan.candidate_generation_id
    assert payload["latest_checkpoints"][0]["checkpoint_id"] == checkpoint.checkpoint_id
    assert payload["artifact_counts"]["inventories"] == 1
    assert payload["artifact_counts"]["authorizations"] == 1
