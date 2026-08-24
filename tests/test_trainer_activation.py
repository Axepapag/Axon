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
    ParameterRegistryError,
    PromotionGate,
    TrainerControlPlane,
    TrainerStateStore,
    capture_module_manifest,
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


def _prepared(tmp_path: Path):
    torch.manual_seed(101)
    live = TinyCore()
    descriptor = ParameterModuleDescriptor(
        module_id="reasoner",
        organ_kind=OrganKind.REASONING_CORE,
        generation_id="g0",
        architecture="tiny-reasoner",
        d_model=4,
    )
    control = TrainerControlPlane.active(state_root=tmp_path)
    control.declare_expected((descriptor,))
    control.register(descriptor, live)
    inventory = control.snapshot_inventory(exact_value_hashes=True)
    grant = ParameterMutationGrant(
        grant_id="adapter-only",
        module_id="reasoner",
        generation_id="g0",
        policy=ParameterMutationPolicy.ADAPTER_ONLY,
        allowed_prefixes=("adapter.",),
    )
    plan = ParameterMutationPlan(
        base_inventory_id=inventory.inventory_id,
        module_id="reasoner",
        base_generation_id="g0",
        candidate_generation_id="g1",
        tensor_names=("adapter.weight",),
        optimizer_name="SGD",
        learning_rate=0.05,
        max_steps=2,
        source_manifest_ids=("source",),
        holdout_manifest_ids=("holdout",),
    )
    session = control.begin_candidate(inventory, grant, plan, preflight_receipt=unit_preflight_receipt(inventory, plan))
    x = torch.ones(1, 4)
    session.step(lambda candidate: candidate(x).square().mean())
    checkpoint = session.checkpoint(include_optimizer=False)
    gate = PromotionGate(
        gate_id="reasoner-gate",
        module_id="reasoner",
        candidate_generation_id="g1",
        requirements=(
            EvaluationRequirement(
                suite_id="capability",
                metric_name="score",
                comparison=MetricComparison.GREATER_OR_EQUAL,
                threshold=0.8,
                label="capability",
            ),
        ),
    )
    observation = EvaluationObservation.from_mapping(
        suite_id="capability",
        module_id="reasoner",
        candidate_generation_id="g1",
        metrics={"score": 0.9},
    )
    decision = control.evaluate_gate(gate, (observation,))
    assert decision.passed
    proposal = ParameterPromotionProposal(
        base_inventory_id=inventory.inventory_id,
        module_id="reasoner",
        base_generation_id="g0",
        candidate_generation_id="g1",
        candidate_artifact_id=checkpoint.checkpoint_id,
        lineage_manifest_id=checkpoint.parameter_manifest_id,
        evaluation_ids=(decision.decision_id,),
    )
    control.record_promotion_proposal(proposal, gate_decision=decision)
    return control, live, descriptor, inventory, plan, checkpoint, decision, proposal


def _hashes(module: nn.Module) -> dict[str, str]:
    return {name: parameter_value_sha256(value) for name, value in module.named_parameters()}


def test_activation_advances_exact_generation_and_rollback_restores_previous_state(tmp_path: Path) -> None:
    control, live, descriptor, inventory, _plan, checkpoint, decision, proposal = _prepared(tmp_path)
    before = _hashes(live)

    receipt = control.activate_candidate(
        inventory,
        proposal,
        gate_decision=decision,
        checkpoint=checkpoint,
    )

    assert receipt.base_generation_id == "g0"
    assert receipt.activated_generation_id == "g1"
    assert control.registry.descriptor("reasoner").generation_id == "g1"
    pointer = control.store.read_active_pointer("reasoner")
    assert pointer is not None
    assert pointer.generation_id == "g1"
    assert pointer.rollback_snapshot_id == receipt.rollback_snapshot_id
    active_manifest_id = capture_module_manifest(control.registry.descriptor("reasoner"), live).manifest_id
    assert active_manifest_id == receipt.after_manifest_id
    assert active_manifest_id == pointer.parameter_manifest_id
    assert active_manifest_id != checkpoint.parameter_manifest_id  # candidate freeze flags are not runtime policy
    assert _hashes(live)["body.weight"] == before["body.weight"]
    assert _hashes(live)["adapter.weight"] != before["adapter.weight"]

    rollback = control.rollback_active_generation("reasoner")
    assert rollback.from_generation_id == "g1"
    assert rollback.restored_generation_id == "g0"
    assert control.registry.descriptor("reasoner").to_canonical_dict() == descriptor.to_canonical_dict()
    assert _hashes(live) == before
    restored_pointer = control.store.read_active_pointer("reasoner")
    assert restored_pointer is not None and restored_pointer.generation_id == "g0"
    assert restored_pointer.rollback_snapshot_id is not None


def test_activation_rejects_stale_live_inventory_before_any_swap(tmp_path: Path) -> None:
    control, live, _descriptor, inventory, _plan, checkpoint, decision, proposal = _prepared(tmp_path)
    with torch.no_grad():
        live.body.weight[0, 0] += 0.5
    stale_state = _hashes(live)

    with pytest.raises(ParameterAuthorityError, match="stale"):
        control.activate_candidate(inventory, proposal, gate_decision=decision, checkpoint=checkpoint)

    assert control.registry.descriptor("reasoner").generation_id == "g0"
    assert control.store.read_active_pointer("reasoner") is None
    assert _hashes(live) == stale_state


def test_failed_gate_can_never_activate_candidate(tmp_path: Path) -> None:
    control, live, _descriptor, inventory, _plan, checkpoint, _decision, proposal = _prepared(tmp_path)
    gate = PromotionGate(
        gate_id="fail",
        module_id="reasoner",
        candidate_generation_id="g1",
        requirements=(
            EvaluationRequirement(
                suite_id="capability",
                metric_name="score",
                comparison=MetricComparison.GREATER_OR_EQUAL,
                threshold=1.0,
                label="capability",
            ),
        ),
    )
    observation = EvaluationObservation.from_mapping(
        suite_id="capability",
        module_id="reasoner",
        candidate_generation_id="g1",
        metrics={"score": 0.2},
    )
    failed = control.evaluate_gate(gate, (observation,))
    before = _hashes(live)
    with pytest.raises(ParameterAuthorityError, match="failed promotion gate"):
        control.activate_candidate(inventory, proposal, gate_decision=failed, checkpoint=checkpoint)
    assert _hashes(live) == before
    assert control.store.read_active_pointer("reasoner") is None


def test_restart_hydration_reproduces_durable_active_generation(tmp_path: Path) -> None:
    control, live, descriptor, inventory, _plan, checkpoint, decision, proposal = _prepared(tmp_path)
    control.activate_candidate(inventory, proposal, gate_decision=decision, checkpoint=checkpoint)
    active_hashes = _hashes(live)
    pointer = control.store.read_active_pointer("reasoner")
    assert pointer is not None

    control.close()
    torch.manual_seed(999)
    restarted_live = TinyCore()
    restarted = TrainerControlPlane.active(state_root=tmp_path)
    restarted.declare_expected((descriptor,))
    restarted.register(descriptor, restarted_live)
    hydrated = restarted.hydrate_active_generation("reasoner")

    assert hydrated is not None and hydrated.pointer_id == pointer.pointer_id
    assert restarted.registry.descriptor("reasoner").generation_id == "g1"
    assert _hashes(restarted_live) == active_hashes
    assert capture_module_manifest(restarted.registry.descriptor("reasoner"), restarted_live).manifest_id == pointer.parameter_manifest_id


def test_registry_generation_transition_rejects_anatomy_change() -> None:
    registry = ParameterRegistry()
    descriptor = ParameterModuleDescriptor(
        module_id="core",
        organ_kind=OrganKind.REASONING_CORE,
        generation_id="g0",
        architecture="tiny",
        d_model=4,
    )
    registry.declare_expected((descriptor,))
    registry.register(descriptor, TinyCore(4))
    incompatible = ParameterModuleDescriptor(
        module_id="core",
        organ_kind=OrganKind.REASONING_CORE,
        generation_id="g1",
        architecture="tiny",
        d_model=8,
    )
    with pytest.raises(ParameterRegistryError, match="anatomy"):
        registry.transition_generation("core", expected_generation_id="g0", new_descriptor=incompatible)


def test_activation_failure_before_pointer_publication_restores_live_generation(tmp_path: Path, monkeypatch) -> None:
    control, live, descriptor, inventory, _plan, checkpoint, decision, proposal = _prepared(tmp_path)
    before = _hashes(live)

    def fail_publish(_pointer):
        raise RuntimeError("simulated pointer publication failure")

    monkeypatch.setattr(control.store, "publish_active_pointer", fail_publish)
    with pytest.raises(RuntimeError, match="simulated pointer publication failure"):
        control.activate_candidate(inventory, proposal, gate_decision=decision, checkpoint=checkpoint)

    assert _hashes(live) == before
    assert control.registry.descriptor("reasoner").to_canonical_dict() == descriptor.to_canonical_dict()
    assert control.store.read_active_pointer("reasoner") is None
