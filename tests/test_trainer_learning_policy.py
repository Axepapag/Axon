from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from runtime.trainer import (
    GovernedLearningPolicy,
    OptimizerKind,
    OrganKind,
    ParameterAuthorityError,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterMutationPolicy,
    PrecisionMode,
    SchedulerKind,
    TrainerControlPlane,
    TrainerExecutionError,
    inspect_trainer_state,
    parameter_value_sha256,
)
from tests._trainer_preflight import unit_preflight_receipt


class TinyPolicyCore(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.body = nn.Linear(width, width, bias=False)
        self.adapter = nn.Linear(width, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.adapter(x)


def _setup(
    tmp_path: Path,
    *,
    optimizer: str = "AdamW",
    learning_rate: float = 0.01,
    max_steps: int = 4,
    device: str = "cpu",
):
    torch.manual_seed(101)
    module = TinyPolicyCore().to(device)
    descriptor = ParameterModuleDescriptor(
        module_id="policy-core",
        organ_kind=OrganKind.REASONING_CORE,
        generation_id="g0",
        architecture="tiny-policy-core",
        d_model=4,
    )
    control = TrainerControlPlane.active(state_root=tmp_path)
    control.declare_expected((descriptor,))
    control.register(descriptor, module)
    inventory = control.snapshot_inventory(exact_value_hashes=True)
    grant = ParameterMutationGrant(
        grant_id="adapter-only",
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
        candidate_generation_id="g1",
        tensor_names=("adapter.weight",),
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        max_steps=max_steps,
        source_manifest_ids=("source",),
        holdout_manifest_ids=("holdout",),
    )
    return control, module, inventory, grant, plan


def _loss(x: torch.Tensor):
    target = torch.zeros_like(x)
    return lambda candidate: (candidate(x) - target).float().square().mean()


def test_learning_policy_is_content_addressed_and_plan_bound(tmp_path: Path) -> None:
    control, _live, inventory, grant, plan = _setup(tmp_path, optimizer="AdamW", learning_rate=0.02, max_steps=5)
    policy = GovernedLearningPolicy(
        optimizer=OptimizerKind.ADAMW,
        learning_rate=0.02,
        weight_decay=0.1,
        scheduler=SchedulerKind.WARMUP_COSINE,
        warmup_steps=2,
        min_lr_ratio=0.2,
        gradient_accumulation_steps=2,
        precision=PrecisionMode.FP32,
    )
    same = GovernedLearningPolicy(**{k: v for k, v in policy.to_canonical_dict(include_id=False).items() if k != "schema"})
    assert same.policy_id == policy.policy_id
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=policy)
    assert session.policy.policy_id == policy.policy_id
    assert (tmp_path / "training" / "trainer" / "learning_policies" / f"{policy.policy_id}.json").exists()

    wrong = GovernedLearningPolicy(optimizer="sgd", learning_rate=0.02)
    with pytest.raises(TrainerExecutionError, match="optimizer differs"):
        control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=wrong)
    control.close()


def test_gradient_accumulation_updates_only_at_declared_boundary(tmp_path: Path) -> None:
    control, live, inventory, grant, plan = _setup(tmp_path, optimizer="SGD", learning_rate=0.05)
    policy = GovernedLearningPolicy(
        optimizer="sgd",
        learning_rate=0.05,
        gradient_accumulation_steps=2,
        gradient_clip_norm=None,
    )
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=policy)
    x1 = torch.tensor([[1.0, -0.5, 0.25, 2.0]])
    x2 = torch.tensor([[-0.5, 1.5, 0.75, -1.0]])
    candidate_before = parameter_value_sha256(session.candidate_module.adapter.weight)
    live_before = parameter_value_sha256(live.adapter.weight)

    micro1, step1 = session.micro_step(_loss(x1))
    assert step1 is None
    assert micro1.accumulation_index == 1
    assert session.step_index == 0
    assert session.accumulation_index == 1
    assert parameter_value_sha256(session.candidate_module.adapter.weight) == candidate_before

    micro2, step2 = session.micro_step(_loss(x2))
    assert step2 is not None
    assert micro2.accumulation_index == 2
    assert step2.step == 1
    assert step2.microbatches_accumulated == 2
    assert session.step_index == 1
    assert session.accumulation_index == 0
    assert parameter_value_sha256(session.candidate_module.adapter.weight) != candidate_before
    assert parameter_value_sha256(live.adapter.weight) == live_before

    root = tmp_path / "training" / "trainer"
    assert len((root / "learning_microsteps.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len((root / "optimization_steps.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    control.close()


def test_mid_accumulation_checkpoint_restores_gradients_optimizer_and_loss_state(tmp_path: Path) -> None:
    control, _live, inventory, grant, plan = _setup(tmp_path, optimizer="AdamW", learning_rate=0.01)
    policy = GovernedLearningPolicy(
        optimizer="adamw",
        learning_rate=0.01,
        weight_decay=0.03,
        gradient_accumulation_steps=2,
        gradient_clip_norm=1.0,
    )
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=policy)
    x1 = torch.tensor([[0.25, -1.0, 2.0, 0.5]])
    x2 = torch.tensor([[1.5, 0.25, -0.75, 1.0]])
    session.micro_step(_loss(x1))
    checkpoint = session.checkpoint(include_optimizer=True)
    assert checkpoint.step == 0
    assert checkpoint.micro_step == 1
    assert checkpoint.accumulation_index == 1
    assert checkpoint.gradient_state_included
    assert checkpoint.learning_policy_id == policy.policy_id

    _micro, first_receipt = session.micro_step(_loss(x2))
    assert first_receipt is not None
    first_hash = parameter_value_sha256(session.candidate_module.adapter.weight)
    first_loss = first_receipt.loss
    first_grad = first_receipt.gradient_l2

    session.restore_checkpoint(checkpoint)
    assert session.step_index == 0
    assert session.micro_step_index == 1
    assert session.accumulation_index == 1
    _micro, second_receipt = session.micro_step(_loss(x2))
    assert second_receipt is not None
    assert parameter_value_sha256(session.candidate_module.adapter.weight) == first_hash
    assert second_receipt.loss == pytest.approx(first_loss, rel=0.0, abs=0.0)
    assert second_receipt.gradient_l2 == pytest.approx(first_grad, rel=0.0, abs=0.0)
    control.close()


def test_warmup_cosine_scheduler_and_weight_decay_are_receipted(tmp_path: Path) -> None:
    control, _live, inventory, grant, plan = _setup(tmp_path, optimizer="SGD", learning_rate=0.1, max_steps=4)
    policy = GovernedLearningPolicy(
        optimizer="sgd",
        learning_rate=0.1,
        weight_decay=0.02,
        scheduler="warmup_cosine",
        warmup_steps=1,
        min_lr_ratio=0.1,
        gradient_clip_norm=None,
    )
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=policy)
    x = torch.ones(1, 4)
    receipts = [session.step(_loss(x)) for _ in range(4)]
    expected = [policy.learning_rate_for_step(index, plan.max_steps) for index in range(4)]
    assert [receipt.learning_rate for receipt in receipts] == pytest.approx(expected)
    assert all(receipt.weight_decay == pytest.approx(0.02) for receipt in receipts)
    assert receipts[-1].learning_rate < receipts[0].learning_rate
    control.close()


def test_gradient_budget_rejects_before_parameter_update(tmp_path: Path) -> None:
    control, _live, inventory, grant, plan = _setup(tmp_path, optimizer="SGD", learning_rate=0.1)
    policy = GovernedLearningPolicy(
        optimizer="sgd",
        learning_rate=0.1,
        max_gradient_l2=1e-12,
        gradient_clip_norm=None,
    )
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=policy)
    before = parameter_value_sha256(session.candidate_module.adapter.weight)
    with pytest.raises(ParameterAuthorityError, match="gradient L2"):
        session.step(_loss(torch.ones(1, 4) * 10.0))
    assert session.closed
    assert parameter_value_sha256(session.candidate_module.adapter.weight) == before
    control.close()


def test_update_budget_restores_candidate_before_rejection(tmp_path: Path) -> None:
    control, _live, inventory, grant, plan = _setup(tmp_path, optimizer="SGD", learning_rate=0.5)
    policy = GovernedLearningPolicy(
        optimizer="sgd",
        learning_rate=0.5,
        max_update_l2=1e-12,
        gradient_clip_norm=None,
    )
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=policy)
    before = parameter_value_sha256(session.candidate_module.adapter.weight)
    with pytest.raises(ParameterAuthorityError, match="update L2"):
        session.step(_loss(torch.ones(1, 4)))
    assert session.closed
    assert parameter_value_sha256(session.candidate_module.adapter.weight) == before
    control.close()


def test_checkpoint_policy_lineage_mismatch_fails_closed(tmp_path: Path) -> None:
    control, _live, inventory, grant, plan = _setup(tmp_path, optimizer="SGD", learning_rate=0.01)
    policy = GovernedLearningPolicy(optimizer="sgd", learning_rate=0.01, weight_decay=0.0)
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=policy)
    session.step(_loss(torch.ones(1, 4)))
    checkpoint = session.checkpoint(include_optimizer=True)

    other_policy = GovernedLearningPolicy(optimizer="sgd", learning_rate=0.01, weight_decay=0.1)
    other_session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=other_policy)
    with pytest.raises(TrainerExecutionError, match="learning-policy lineage"):
        other_session.restore_checkpoint(checkpoint)
    control.close()


def test_cpu_bf16_autocast_is_governed_and_inspectable(tmp_path: Path) -> None:
    control, _live, inventory, grant, plan = _setup(tmp_path, optimizer="SGD", learning_rate=0.01)
    policy = GovernedLearningPolicy(
        optimizer="sgd",
        learning_rate=0.01,
        precision="bf16",
        gradient_clip_norm=1.0,
    )
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=policy)
    receipt = session.step(_loss(torch.ones(1, 4)))
    assert receipt.precision_mode == "bf16"
    snapshot = inspect_trainer_state(state_root=tmp_path).to_canonical_dict()
    assert snapshot["latest_learning_policy"]["policy_id"] == policy.policy_id
    assert snapshot["latest_microstep"]["precision_mode"] == "bf16"
    assert snapshot["artifact_counts"]["learning_policies"] == 1
    control.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for governed fp16 scaler proof")
def test_cuda_fp16_grad_scaler_path_executes_on_isolated_candidate(tmp_path: Path) -> None:
    control, live, inventory, grant, plan = _setup(
        tmp_path,
        optimizer="AdamW",
        learning_rate=0.001,
        device="cuda",
    )
    policy = GovernedLearningPolicy(
        optimizer="adamw",
        learning_rate=0.001,
        precision="fp16",
        gradient_clip_norm=1.0,
    )
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan), policy=policy)
    live_before = parameter_value_sha256(live.adapter.weight)
    receipt = session.step(_loss(torch.ones(1, 4, device="cuda")))
    assert receipt.precision_mode == "fp16"
    assert receipt.step == 1
    assert session._scaler is not None
    checkpoint = session.checkpoint(include_optimizer=True)
    assert checkpoint.scaler_included
    assert parameter_value_sha256(live.adapter.weight) == live_before
    control.close()
