"""Renewable resource tranches: identity separation and additive continuation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from runtime.field import canonical_sha256
from runtime.trainer import (
    AuthorizedParameterMutation,
    CandidateStatus,
    OrganKind,
    ParameterModuleDescriptor,
    ParameterMutationPlan,
    ParameterMutationPlanV2,
    ResourceTranche,
    TrainerControlPlane,
    TrancheContinuation,
    TrancheError,
    TrancheStore,
)
from runtime.trainer.execution import CandidateOptimizationSession
from runtime.trainer.learning import GovernedLearningPolicy
from runtime.trainer.store import TrainerStateStore

from ._short_tmp import short_state_root


class _TinyCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).sum()


def _descriptor(generation: str) -> ParameterModuleDescriptor:
    return ParameterModuleDescriptor(
        module_id="tranche-test-core",
        organ_kind=OrganKind.REASONING_CORE,
        generation_id=generation,
        architecture="tiny-test-v1",
        d_model=8,
    )


def _session_parts(state_root: Path, max_steps: int, generation: str):
    """Register a tiny module and return everything needed for a session."""

    module = _TinyCore()
    base = _descriptor("tiny-base-v1")
    control = TrainerControlPlane.active(state_root=state_root)
    try:
        control.declare_expected((base,))
        control.register(base, module)
        inventory = control.snapshot_inventory(exact_value_hashes=True)
    finally:
        control.close()
    manifest = inventory.module(base.module_id)
    plan = ParameterMutationPlan(
        base_inventory_id=inventory.inventory_id,
        module_id=base.module_id,
        base_generation_id=base.generation_id,
        candidate_generation_id=generation,
        tensor_names=tuple(item.name for item in manifest.tensors if item.requires_grad),
        optimizer_name="AdamW",
        learning_rate=1e-3,
        max_steps=max_steps,
        source_manifest_ids=("tranche-test-source",),
        holdout_manifest_ids=("tranche-test-holdout",),
    )
    authorization = AuthorizedParameterMutation(
        grant_id="tranche-test-grant",
        plan_id=plan.plan_id,
        inventory_id=inventory.inventory_id,
        module_id=plan.module_id,
        base_generation_id=plan.base_generation_id,
        candidate_generation_id=plan.candidate_generation_id,
        tensor_names=plan.tensor_names,
        parameter_count=4 * 8 + 4,
        preflight_receipt_id="0" * 64,
        authorization_id=canonical_sha256(
            {
                "schema": "axon-authorized-parameter-mutation-v2",
                "grant_id": "tranche-test-grant",
                "plan_id": plan.plan_id,
                "inventory_id": inventory.inventory_id,
                "module_id": plan.module_id,
                "base_generation_id": plan.base_generation_id,
                "candidate_generation_id": plan.candidate_generation_id,
                "tensor_names": list(plan.tensor_names),
                "parameter_count": 4 * 8 + 4,
                "preflight_receipt_id": "0" * 64,
            }
        ),
    )
    store = TrainerStateStore.active(state_root=state_root)
    return module, base, inventory, plan, authorization, store


def _loss_fn(module: nn.Module) -> torch.Tensor:
    return module.linear(torch.ones(1, 8)).square().mean()


# --- tranche identity law -------------------------------------------------


def test_tranche_size_never_changes_any_identity() -> None:
    """Codex acceptance gate 1: allowance changes no identity."""

    small = ResourceTranche(
        module_id="m",
        candidate_generation_id="g",
        plan_id="a" * 64,
        learning_policy_id="b" * 64,
        base_global_step=16,
        steps=1,
        parent_bundle_id="c" * 64,
    )
    large = ResourceTranche(
        module_id="m",
        candidate_generation_id="g",
        plan_id="a" * 64,
        learning_policy_id="b" * 64,
        base_global_step=16,
        steps=10_000,
        parent_bundle_id="c" * 64,
    )
    # Different tranches have different tranche ids (they are distinct
    # allowances), but neither participates in any other identity: the
    # module/candidate names they reference are identical.
    assert small.module_id == large.module_id
    assert small.candidate_generation_id == large.candidate_generation_id
    assert small.tranche_id != large.tranche_id


def test_tranche_admission_bounds_are_exact() -> None:
    tranche = ResourceTranche(
        module_id="m",
        candidate_generation_id="g",
        plan_id="a" * 64,
        learning_policy_id="b" * 64,
        base_global_step=16,
        steps=16,
        parent_bundle_id="c" * 64,
    )
    assert tranche.final_global_step == 32
    assert tranche.admits_step(16) is False  # parent step itself is not new work
    assert tranche.admits_step(17) is True
    assert tranche.admits_step(32) is True
    assert tranche.admits_step(33) is False


def test_tranche_rejects_degenerate_shapes() -> None:
    with pytest.raises(TrancheError):
        ResourceTranche(
            module_id="m",
            candidate_generation_id="g",
            plan_id="a" * 64,
            learning_policy_id="b" * 64,
            base_global_step=-1,
            steps=5,
        )
    with pytest.raises(TrancheError):
        ResourceTranche(
            module_id="m",
            candidate_generation_id="g",
            plan_id="a" * 64,
            learning_policy_id="b" * 64,
            base_global_step=0,
            steps=0,
        )


def test_tranche_store_roundtrip_and_immutability(tmp_path) -> None:
    store = TrancheStore(tmp_path)
    tranche = ResourceTranche(
        module_id="m",
        candidate_generation_id="g",
        plan_id="a" * 64,
        learning_policy_id="b" * 64,
        base_global_step=16,
        steps=8,
        parent_bundle_id="c" * 64,
    )
    path = store.write_tranche(tranche)
    assert path.is_file()
    assert store.read_tranche(tranche.tranche_id) == tranche
    # Immutable: rewriting the SAME tranche id with different content fails
    # closed.  A tampered record with a mismatching observed id is rejected
    # on read.
    tampered = tranche.to_canonical_dict()
    tampered["steps"] = 9
    tampered["final_global_step"] = 25
    (tmp_path / "tranches" / f"{tranche.tranche_id}.json").write_text(
        json.dumps(tampered), encoding="utf-8"
    )
    with pytest.raises(TrancheError):
        store.read_tranche(tranche.tranche_id)
    path.write_text(json.dumps(tranche.to_canonical_dict()), encoding="utf-8")
    # Continuation receipts roundtrip and are immutable.
    continuation = TrancheContinuation(
        tranche_id=tranche.tranche_id,
        module_id="m",
        candidate_generation_id="g",
        plan_id=tranche.plan_id,
        learning_policy_id=tranche.learning_policy_id,
        parent_bundle_id=tranche.parent_bundle_id,
        parent_checkpoint_id="a" * 64,
        parent_optimizer_receipt_id="d" * 64,
        parent_soul_id="e" * 64,
        parent_global_step=16,
        prior_tranche_id=None,
    )
    store.write_continuation(continuation)
    assert store.read_continuation(continuation.continuation_id) == continuation
    assert store.continuations_for("m", "g") == (continuation,)
    assert store.tranches_for("m", "g") == (tranche,)


# --- additive continuation: N -> N+1 across tranches -----------------------


def test_session_continues_past_plan_envelope_under_tranche(tmp_path) -> None:
    """Codex acceptance gates 3/4: tranche pauses at bound; next tranche continues N->N+1."""

    state_root = short_state_root("tranche") / "State"
    state_root.mkdir(parents=True, exist_ok=True)
    module, base, inventory, plan, authorization, store = _session_parts(
        state_root, max_steps=2, generation="tiny-candidate-v1"
    )
    policy = GovernedLearningPolicy(optimizer="adamw", learning_rate=1e-3)

    # Segment 1 (v1 envelope): steps 1..2.
    session = CandidateOptimizationSession(
        live_module=module,
        base_descriptor=base,
        base_inventory=inventory,
        plan=plan,
        authorization=authorization,
        store=store,
        policy=policy,
    )
    session.step(_loss_fn)
    session.step(_loss_fn)
    checkpoint = session.checkpoint(include_optimizer=True)
    assert checkpoint.step == 2
    with pytest.raises(RuntimeError, match="authorized max_steps"):
        session.step(_loss_fn)  # envelope reached: v1 behavior preserved
    session.complete(reason="v1 envelope exhausted")

    # v1 law preserved WITHOUT a tranche: a no-tranche session on the same
    # plan still refuses step 3 even after restore.
    session_v1 = CandidateOptimizationSession(
        live_module=module,
        base_descriptor=base,
        base_inventory=inventory,
        plan=plan,
        authorization=authorization,
        store=store,
        policy=policy,
    )
    session_v1.restore_checkpoint(checkpoint)
    with pytest.raises(RuntimeError, match="authorized max_steps"):
        session_v1.step(_loss_fn)

    # Segment 2 (renewable tranche): continues the SAME lineage to steps 3..4
    # under a fresh session with an exact parent restore.
    tranche = ResourceTranche(
        module_id=plan.module_id,
        candidate_generation_id=plan.candidate_generation_id,
        plan_id=plan.plan_id,
        learning_policy_id=policy.policy_id,
        base_global_step=2,
        steps=2,
        parent_bundle_id="a" * 64,
        purpose="test continuation tranche",
    )
    session2 = CandidateOptimizationSession(
        live_module=module,
        base_descriptor=base,
        base_inventory=inventory,
        plan=plan,  # SAME plan identity: no restart, no new generation
        authorization=authorization,
        store=store,
        policy=policy,
        tranche=tranche,
    )
    # A nonzero-base tranche without an exact parent restore fails closed.
    with pytest.raises(RuntimeError, match="parent checkpoint restore"):
        session2.step(_loss_fn)
    session2.restore_checkpoint(checkpoint)
    receipt3 = session2.step(_loss_fn)  # global step 3: N -> N+1
    assert receipt3.step == 3
    receipt4 = session2.step(_loss_fn)
    assert receipt4.step == 4
    checkpoint2 = session2.checkpoint(include_optimizer=True)
    # Tranche bound reached: the next step is refused, work pauses, nothing
    # is lost or relabeled.
    with pytest.raises(RuntimeError, match="later tranche"):
        session2.step(_loss_fn)
    pause = session2.pause(
        reason="tranche bound reached; paused for next tranche",
        checkpoint_id=checkpoint2.checkpoint_id,
    )
    assert pause.status is CandidateStatus.PAUSED
    assert checkpoint2.step == 4

    # A wrong-parent tranche (base 1 against restored step 2) fails closed.
    bad_tranche = ResourceTranche(
        module_id=plan.module_id,
        candidate_generation_id=plan.candidate_generation_id,
        plan_id=plan.plan_id,
        learning_policy_id=policy.policy_id,
        base_global_step=1,
        steps=2,
        parent_bundle_id="b" * 64,
    )
    session_bad = CandidateOptimizationSession(
        live_module=module,
        base_descriptor=base,
        base_inventory=inventory,
        plan=plan,
        authorization=authorization,
        store=store,
        policy=policy,
        tranche=bad_tranche,
    )
    session_bad.restore_checkpoint(checkpoint2)
    with pytest.raises(RuntimeError, match="parent checkpoint restore"):
        session_bad.step(_loss_fn)


def test_tranche_module_or_generation_mismatch_fails_closed(tmp_path) -> None:
    state_root = short_state_root("tranche2") / "State"
    state_root.mkdir(parents=True, exist_ok=True)
    module, base, inventory, plan, authorization, store = _session_parts(
        state_root, max_steps=4, generation="tiny-candidate-v2"
    )
    policy = GovernedLearningPolicy(optimizer="adamw", learning_rate=1e-3)
    wrong_module = ResourceTranche(
        module_id="other-module",
        candidate_generation_id=plan.candidate_generation_id,
        plan_id=plan.plan_id,
        learning_policy_id=policy.policy_id,
        base_global_step=0,
        steps=2,
    )
    with pytest.raises(RuntimeError, match="tranche module differs"):
        CandidateOptimizationSession(
            live_module=module,
            base_descriptor=base,
            base_inventory=inventory,
            plan=plan,
            authorization=authorization,
            store=store,
            policy=policy,
            tranche=wrong_module,
        )
    wrong_generation = ResourceTranche(
        module_id=plan.module_id,
        candidate_generation_id="other-generation",
        plan_id=plan.plan_id,
        learning_policy_id=policy.policy_id,
        base_global_step=0,
        steps=2,
    )
    with pytest.raises(RuntimeError, match="candidate generation differs"):
        CandidateOptimizationSession(
            live_module=module,
            base_descriptor=base,
            base_inventory=inventory,
            plan=plan,
            authorization=authorization,
            store=store,
            policy=policy,
            tranche=wrong_generation,
        )


def test_v2_plan_has_no_resource_or_learning_recipe_ceiling() -> None:
    plan = ParameterMutationPlanV2(
        base_inventory_id="inventory",
        module_id="m",
        base_generation_id="base",
        candidate_generation_id="candidate",
        tensor_names=("weight",),
        source_manifest_ids=("train",),
        holdout_manifest_ids=("heldout",),
    )
    body = plan.to_canonical_dict()
    assert body["schema"] == "axon-parameter-mutation-plan-v2"
    assert "max_steps" not in body
    assert "optimizer_name" not in body
    assert "learning_rate" not in body


def test_v2_session_requires_tranche_and_keeps_schedule_out_of_tranche(tmp_path) -> None:
    state_root = short_state_root("tranchev2") / "State"
    state_root.mkdir(parents=True, exist_ok=True)
    module, base, inventory, old_plan, _authorization, store = _session_parts(
        state_root, max_steps=1, generation="tiny-candidate-v3"
    )
    plan = ParameterMutationPlanV2(
        base_inventory_id=old_plan.base_inventory_id,
        module_id=old_plan.module_id,
        base_generation_id=old_plan.base_generation_id,
        candidate_generation_id=old_plan.candidate_generation_id,
        tensor_names=old_plan.tensor_names,
        source_manifest_ids=old_plan.source_manifest_ids,
        holdout_manifest_ids=old_plan.holdout_manifest_ids,
    )
    policy = GovernedLearningPolicy(
        optimizer="adamw",
        learning_rate=1e-3,
        scheduler="warmup_cosine",
        warmup_steps=2,
        schedule_steps=100,
    )
    authorization = AuthorizedParameterMutation(
        grant_id="tranche-test-grant",
        plan_id=plan.plan_id,
        inventory_id=inventory.inventory_id,
        module_id=plan.module_id,
        base_generation_id=plan.base_generation_id,
        candidate_generation_id=plan.candidate_generation_id,
        tensor_names=plan.tensor_names,
        parameter_count=4 * 8 + 4,
        preflight_receipt_id="0" * 64,
        authorization_id=canonical_sha256(
            {
                "schema": "axon-authorized-parameter-mutation-v2",
                "grant_id": "tranche-test-grant",
                "plan_id": plan.plan_id,
                "inventory_id": inventory.inventory_id,
                "module_id": plan.module_id,
                "base_generation_id": plan.base_generation_id,
                "candidate_generation_id": plan.candidate_generation_id,
                "tensor_names": list(plan.tensor_names),
                "parameter_count": 4 * 8 + 4,
                "preflight_receipt_id": "0" * 64,
            }
        ),
    )
    without_tranche = CandidateOptimizationSession(
        live_module=module,
        base_descriptor=base,
        base_inventory=inventory,
        plan=plan,
        authorization=authorization,
        store=store,
        policy=policy,
    )
    with pytest.raises(RuntimeError, match="renewable resource tranche"):
        without_tranche.step(_loss_fn)

    small = ResourceTranche(
        module_id=plan.module_id,
        candidate_generation_id=plan.candidate_generation_id,
        plan_id=plan.plan_id,
        learning_policy_id=policy.policy_id,
        base_global_step=0,
        steps=1,
    )
    large = ResourceTranche(
        module_id=plan.module_id,
        candidate_generation_id=plan.candidate_generation_id,
        plan_id=plan.plan_id,
        learning_policy_id=policy.policy_id,
        base_global_step=0,
        steps=10_000,
    )
    assert small.tranche_id != large.tranche_id
    assert "lr_schedule_horizon" not in small.to_canonical_dict()
    assert "schedule_steps" not in small.to_canonical_dict()
    session = CandidateOptimizationSession(
        live_module=module,
        base_descriptor=base,
        base_inventory=inventory,
        plan=plan,
        authorization=authorization,
        store=store,
        policy=policy,
        tranche=small,
    )
    receipt = session.step(_loss_fn)
    assert receipt.step == 1
    with pytest.raises(RuntimeError, match="later tranche"):
        session.step(_loss_fn)
