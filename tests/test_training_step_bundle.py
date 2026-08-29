from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from runtime.soul import SoulStore
from runtime.trainer import (
    CandidateSoulWorkspace,
    CandidateStepBundleCoordinator,
    OrganKind,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterMutationPolicy,
    TrainerControlPlane,
    TrainerStoreError,
)
from training import (
    LivingReasoningCoreConfig,
    LivingReasoningCoreD64,
    build_living_reasoning_preflight,
    build_living_reasoning_smoke_curriculum,
    living_episode_objective,
)

ROOT = Path(__file__).resolve().parent.parent


def _step_artifacts(state_root: Path, *, optimizer_steps: int = 1) -> dict[str, Any]:
    torch.manual_seed(314)
    model = LivingReasoningCoreD64(
        LivingReasoningCoreConfig(
            n_heads=1,
            n_layers=2,
            ffn_dim=128,
            state_tokens=2,
            page_size=2,
            dropout=0.0,
            inference_budget_transport_units=32,
        )
    )
    curriculum = build_living_reasoning_smoke_curriculum()
    module_id = "step-bundle-core"
    base_generation = "g0"
    candidate_generation = "g1"
    descriptor = ParameterModuleDescriptor(
        module_id=module_id,
        organ_kind=OrganKind.REASONING_CORE,
        generation_id=base_generation,
        architecture=model.architecture_id,
        d_model=64,
    )
    with TrainerControlPlane.active(state_root=state_root) as control:
        control.declare_expected((descriptor,))
        control.register(descriptor, model)
        inventory = control.snapshot_inventory(exact_value_hashes=True)
        manifest = inventory.module(module_id)
        plan = ParameterMutationPlan(
            base_inventory_id=inventory.inventory_id,
            module_id=module_id,
            base_generation_id=base_generation,
            candidate_generation_id=candidate_generation,
            tensor_names=tuple(item.name for item in manifest.tensors if item.requires_grad),
            optimizer_name="AdamW",
            learning_rate=1e-4,
            max_steps=2,
            source_manifest_ids=(curriculum.train_manifest_id,),
            holdout_manifest_ids=(curriculum.heldout_manifest_id,),
        )
        preflight = build_living_reasoning_preflight(
            model=model,
            curriculum=curriculum,
            inventory=inventory,
            plan=plan,
            state_root=state_root,
            repo_root=ROOT,
        )
        grant = ParameterMutationGrant(
            grant_id="step-bundle-test",
            module_id=module_id,
            generation_id=base_generation,
            policy=ParameterMutationPolicy.FULL,
            max_trainable_parameters=manifest.trainable_parameter_count,
        )
        session = control.begin_candidate(
            inventory,
            grant,
            plan,
            preflight_receipt=preflight,
        )
        SoulStore.active(state_root).ensure_core(
            core_id=module_id,
            architecture_id=model.architecture_id,
            parameter_generation=base_generation,
        )
        soul_workspace = CandidateSoulWorkspace(state_root)
        soul_manifest = soul_workspace.prepare(
            candidate_id=candidate_generation,
            core_id=module_id,
            runtime_episode_session_id=f"synthetic:{curriculum.curriculum_id}",
            whole_episode_split="train",
            soul_trajectory_ids=tuple(item.episode_id for item in curriculum.split("train")),
            candidate_parameter_generation=candidate_generation,
        )
        branch = soul_workspace.branch(candidate_generation, module_id)
        captured: dict[str, Any] = {}
        episode = curriculum.split("train")[0]
        ephemeral_soul = branch.load_head()
        transitions = []

        for _ in range(optimizer_steps):
            def loss_fn(
                candidate: torch.nn.Module,
                soul=ephemeral_soul,
            ) -> torch.Tensor:
                assert isinstance(candidate, LivingReasoningCoreD64)
                loss, unroll, _metrics = living_episode_objective(
                    candidate,
                    episode,
                    soul,
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                )
                captured["unroll"] = unroll
                return loss

            receipt = session.step(loss_fn)
            ephemeral_soul = captured["unroll"].souls[-1]
            transitions.extend(captured["unroll"].transitions)
        checkpoint = session.checkpoint(include_optimizer=True)
        session.complete(reason="step-bundle fixture complete")
        return {
            "module_id": module_id,
            "candidate_generation": candidate_generation,
            "receipt": receipt,
            "checkpoint": checkpoint,
            "soul_manifest": soul_manifest,
            "transitions": tuple(transitions),
            "soul_workspace": soul_workspace,
        }


def test_parameter_checkpoint_and_private_soul_publish_as_one_accepted_step(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "State"
    item = _step_artifacts(state_root)
    coordinator = CandidateStepBundleCoordinator(state_root)
    bundle = coordinator.accept_step(
        optimization_receipt=item["receipt"],
        checkpoint=item["checkpoint"],
        soul_manifest=item["soul_manifest"],
        transitions=item["transitions"],
    )
    pointer = coordinator.latest_pointer(item["module_id"], item["candidate_generation"])
    assert pointer is not None
    assert pointer.current_bundle_id == bundle.bundle_id
    assert pointer.current_step == 1
    assert pointer.rolling_bundle_ids == (bundle.bundle_id,)
    assert coordinator.checkpoint_for_bundle(bundle).checkpoint_id == item["checkpoint"].checkpoint_id
    branch = item["soul_workspace"].branch(item["candidate_generation"], item["module_id"])
    assert branch.load_head().soul_id == bundle.after_soul_id
    assert len(bundle.soul_receipt_ids) == 3


def test_pending_step_recovers_after_only_first_soul_phase_committed(tmp_path: Path) -> None:
    state_root = tmp_path / "State"
    item = _step_artifacts(state_root)
    coordinator = CandidateStepBundleCoordinator(state_root)
    intent = coordinator.prepare_intent(
        optimization_receipt=item["receipt"],
        checkpoint=item["checkpoint"],
        soul_manifest=item["soul_manifest"],
        transitions=item["transitions"],
    )
    branch = item["soul_workspace"].branch(item["candidate_generation"], item["module_id"])
    first = item["transitions"][0]
    branch.commit_transition(first, commit_binding=f"trainer-step-intent:{intent.intent_id}")

    recovered = coordinator.recover_pending(item["module_id"], item["candidate_generation"])
    assert len(recovered) == 1
    assert recovered[0].after_soul_id == intent.expected_after_soul_id
    assert branch.load_head().soul_id == intent.expected_after_soul_id
    assert coordinator.recover_pending(item["module_id"], item["candidate_generation"]) == ()


def test_checkpoint_segment_accepts_multiple_complete_optimizer_soul_steps(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "State"
    item = _step_artifacts(state_root, optimizer_steps=2)
    coordinator = CandidateStepBundleCoordinator(state_root)
    bundle = coordinator.accept_step(
        optimization_receipt=item["receipt"],
        checkpoint=item["checkpoint"],
        soul_manifest=item["soul_manifest"],
        transitions=item["transitions"],
    )
    assert bundle.step == 2
    assert len(bundle.soul_receipt_ids) == 6
    assert coordinator.latest_pointer(item["module_id"], item["candidate_generation"]).current_step == 2


def test_pointer_sentinel_damage_is_preserved_and_rebuilt_from_immutable_bundle(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "State"
    item = _step_artifacts(state_root)
    coordinator = CandidateStepBundleCoordinator(state_root)
    bundle = coordinator.accept_step(
        optimization_receipt=item["receipt"],
        checkpoint=item["checkpoint"],
        soul_manifest=item["soul_manifest"],
        transitions=item["transitions"],
    )
    root = (
        state_root
        / "training"
        / "trainer"
        / "candidates"
        / item["module_id"]
        / item["candidate_generation"]
        / "accepted_steps"
    )
    (root / "checkpoint_done.json").write_text("{}\n", encoding="utf-8")
    try:
        coordinator.latest_pointer(item["module_id"], item["candidate_generation"])
    except TrainerStoreError:
        pass
    else:
        raise AssertionError("damaged accepted-step sentinel did not fail closed")

    assert coordinator.recover_pending(item["module_id"], item["candidate_generation"]) == ()
    pointer = coordinator.latest_pointer(item["module_id"], item["candidate_generation"])
    assert pointer is not None and pointer.current_bundle_id == bundle.bundle_id
    recovery = tuple((root / "recovery_evidence").glob("*.json"))
    assert len(recovery) == 1
    evidence = json.loads(recovery[0].read_text(encoding="utf-8"))
    assert evidence["sentinel"] == "{}\n"
