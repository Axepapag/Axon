"""Run the ratified direct-address pointer-motor tranche on the local GTX."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from runtime.field import canonical_sha256
from runtime.soul import SoulStore, apply_soul_transition
from runtime.trainer import (
    CandidateSoulWorkspace,
    CandidateStepBundleCoordinator,
    GovernedLearningPolicy,
    OrganKind,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlanV2,
    ParameterMutationPolicy,
    ResourceTranche,
    TrainerControlPlane,
    TrancheContinuation,
    TrancheStore,
)
from training import (
    LivingReasoningCoreConfig,
    LivingReasoningCoreD64,
    POINTER_MOTOR_REQUIRED_TAGS,
    build_living_reasoning_preflight,
    build_pointer_motor_curriculum,
    decide_pointer_motor_mastery,
    evaluate_pointer_motor,
    initialize_pointer_scaffold_from_donor,
    pointer_motor_episode_objective,
    rebind_d64_soul_layers,
)

ROOT = Path(__file__).resolve().parent.parent
MODULE_ID = "r64-english-reasoning"
DONOR_GENERATION = "english-candidate-1c991f8c911f79394e91"
DONOR_BUNDLE_ID = "1d77a96509993005e03366066fdd43f1143adb8895ba470b62f1a300aff7aeb5"
DONOR_SOUL_ID = "308ff2a2b5de6337e904733a04da001af613faa7faa364a984150b3837753a0a"
DONOR_CHECKPOINT_ID = "ef373220713565efa9da7b6c7877ffb5de75ab4522513ce679945102690ced3e"
OBJECTIVE_PROGRAM_ID = canonical_sha256(
    {
        "schema": "axon-pointer-motor-objective-program-v1",
        "primitive": "canonical-address-to-existing-position-key",
        "loss": "cross_entropy_exact_source_memory_index",
        "trainable_surface": "pointer_address_query_only",
        "soul": "first_inhale_exhale_training_transition",
        "eos": "excluded",
        "english": "excluded",
    }
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _print_json(value: Mapping[str, Any]) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    stream = getattr(sys.stdout, "buffer", None)
    if stream is not None:
        stream.write(data.encode("utf-8"))
        stream.flush()
    else:
        sys.stdout.write(data)
        sys.stdout.flush()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--tranche-steps", type=int, default=8)
    parser.add_argument("--experiences-per-step", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _args()
    if args.tranche_steps < 1 or args.experiences_per_step < 1:
        raise ValueError("tranche steps and experiences per step must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    _seed(args.seed)
    device = _device(args.device)
    state_root = args.state_root.resolve()
    curriculum = build_pointer_motor_curriculum()
    config = LivingReasoningCoreConfig(
        n_heads=1,
        n_layers=2,
        ffn_dim=16_384,
        state_tokens=4,
        page_size=32,
        pointer_address_scaffold_version="query-scaffold-v1",
    )
    model = LivingReasoningCoreD64(config).to(device)

    donor_coordinator = CandidateStepBundleCoordinator(state_root)
    donor_bundle = donor_coordinator.load_bundle(MODULE_ID, DONOR_GENERATION, DONOR_BUNDLE_ID)
    if donor_bundle.step != 24 or donor_bundle.checkpoint_id != DONOR_CHECKPOINT_ID:
        raise RuntimeError("the pinned step-24 donor bundle/checkpoint identity changed")
    donor_checkpoint = donor_coordinator.checkpoint_for_bundle(donor_bundle)
    donor_payload = donor_coordinator.trainer_store.load_verified_candidate_checkpoint(donor_checkpoint)
    donor_state = donor_payload["module_state_dict"]
    if donor_payload["descriptor"].get("architecture") != "living-d64-english-023f4b5e7d43d59968c9b133":
        raise RuntimeError("the pinned donor architecture identity changed")

    base_generation = "pointer-scaffold-base-" + canonical_sha256(
        {
            "source_checkpoint_id": donor_checkpoint.checkpoint_id,
            "source_bundle_id": donor_bundle.bundle_id,
            "target_architecture_id": model.architecture_id,
            "objective_program_id": OBJECTIVE_PROGRAM_ID,
        }
    )[:20]
    migration = initialize_pointer_scaffold_from_donor(
        model,
        donor_state,
        source_architecture_id=str(donor_payload["descriptor"]["architecture"]),
        source_parameter_generation=str(donor_checkpoint.base_generation_id),
        target_parameter_generation=base_generation,
        source_checkpoint_id=donor_checkpoint.checkpoint_id,
        source_bundle_id=donor_bundle.bundle_id,
        source_soul_id=DONOR_SOUL_ID,
    )
    migration_path = (
        state_root / "training" / "reasoning" / "pointer_motor" / "migrations"
        / f"{migration.receipt_id}.json"
    )
    _atomic_json(migration_path, migration.to_canonical_dict())

    # The governed plan is deliberately limited to the query scaffold.  The
    # old reader, memory keys, decoder, Soul projections, and all buffers are
    # frozen before inventory capture.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_names = tuple(
        name
        for name, parameter in model.named_parameters()
        if name.startswith("pointer_address_query.")
    )
    for name, parameter in model.named_parameters():
        if name in trainable_names:
            parameter.requires_grad_(True)
    if set(trainable_names) != {
        "pointer_address_query.0.weight",
        "pointer_address_query.0.bias",
        "pointer_address_query.2.weight",
        "pointer_address_query.2.bias",
    }:
        raise RuntimeError("pointer scaffold trainable surface is not exactly four tensors")

    policy = GovernedLearningPolicy(
        optimizer="adamw",
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        objective_program_id=OBJECTIVE_PROGRAM_ID,
    )
    candidate_generation = "pointer-motor-candidate-" + canonical_sha256(
        {
            "module_id": MODULE_ID,
            "base_generation": base_generation,
            "architecture_id": model.architecture_id,
            "curriculum_id": curriculum.curriculum_id,
            "learning_policy_id": policy.policy_id,
        }
    )[:20]
    descriptor = ParameterModuleDescriptor(
        module_id=MODULE_ID,
        organ_kind=OrganKind.REASONING_CORE,
        generation_id=base_generation,
        architecture=model.architecture_id,
        d_model=64,
    )
    step_bundles = CandidateStepBundleCoordinator(state_root)
    tranche_store = TrancheStore(state_root / "training" / "trainer")
    latest = step_bundles.latest_bundle(MODULE_ID, candidate_generation)
    if latest is not None and not args.resume:
        raise RuntimeError(
            "pointer-motor candidate already has accepted work; pass --resume for the exact continuation"
        )
    if latest is None and args.resume:
        raise RuntimeError("--resume was requested but no accepted pointer-motor bundle exists")
    report: dict[str, Any] = {
        "schema": "axon-pointer-motor-scaffold-trainer-v1",
        "module_id": MODULE_ID,
        "architecture_id": model.architecture_id,
        "base_generation": base_generation,
        "candidate_generation": candidate_generation,
        "curriculum_id": curriculum.curriculum_id,
        "objective_program_id": OBJECTIVE_PROGRAM_ID,
        "learning_policy": policy.to_canonical_dict(),
        "device": str(device),
        "donor": {
            "bundle_id": donor_bundle.bundle_id,
            "checkpoint_id": donor_checkpoint.checkpoint_id,
            "soul_id": DONOR_SOUL_ID,
            "architecture_id": donor_payload["descriptor"]["architecture"],
            "step": donor_bundle.step,
        },
        "migration": migration.to_canonical_dict(),
        "optimizer_steps": [],
        "accepted_bundles": [],
    }

    with TrainerControlPlane.active(state_root=state_root) as control:
        control.declare_expected((descriptor,))
        control.register(descriptor, model)
        inventory = control.snapshot_inventory(exact_value_hashes=True)
        manifest = inventory.module(MODULE_ID)
        if {item.name for item in manifest.tensors if item.requires_grad} != set(trainable_names):
            raise RuntimeError("inventory trainable surface disagrees with scaffold")
        plan = ParameterMutationPlanV2(
            base_inventory_id=inventory.inventory_id,
            module_id=MODULE_ID,
            base_generation_id=base_generation,
            candidate_generation_id=candidate_generation,
            tensor_names=tuple(sorted(trainable_names)),
            source_manifest_ids=(curriculum.train_manifest_id,),
            holdout_manifest_ids=(curriculum.heldout_manifest_id,),
        )
        tranche = ResourceTranche(
            module_id=MODULE_ID,
            candidate_generation_id=candidate_generation,
            plan_id=plan.plan_id,
            learning_policy_id=policy.policy_id,
            base_global_step=0 if latest is None else latest.step,
            steps=args.tranche_steps,
            parent_bundle_id=None if latest is None else latest.bundle_id,
            purpose="bounded GTX direct canonical-address pointer-motor tranche",
        )
        preflight = build_living_reasoning_preflight(
            model=model,
            curriculum=curriculum,
            inventory=inventory,
            plan=plan,
            state_root=state_root,
            repo_root=ROOT,
            resource_tranche=tranche,
            required_curriculum_tags=POINTER_MOTOR_REQUIRED_TAGS,
        )
        report["plan"] = plan.to_canonical_dict()
        report["tranche"] = tranche.to_canonical_dict()
        report["preflight"] = preflight.to_canonical_dict()
        if not preflight.passed:
            report["status"] = "preflight_failed"
            report_path = args.report or (
                state_root / "training" / "reasoning" / "pointer_motor" / candidate_generation / "report.json"
            )
            _atomic_json(report_path.resolve(), report)
            _print_json(report)
            return 2
        tranche_store.write_tranche(tranche)
        continuation = None
        if latest is not None:
            prior = tuple(
                item
                for item in tranche_store.tranches_for(MODULE_ID, candidate_generation)
                if item.tranche_id != tranche.tranche_id
                and item.final_global_step == latest.step
            )
            if len(prior) != 1:
                raise RuntimeError("pointer-motor continuation requires exactly one prior tranche")
            continuation = TrancheContinuation(
                tranche_id=tranche.tranche_id,
                module_id=MODULE_ID,
                candidate_generation_id=candidate_generation,
                plan_id=plan.plan_id,
                learning_policy_id=policy.policy_id,
                parent_bundle_id=latest.bundle_id,
                parent_checkpoint_id=latest.checkpoint_id,
                parent_optimizer_receipt_id=latest.optimization_receipt_id,
                parent_soul_id=latest.after_soul_id,
                parent_global_step=latest.step,
                prior_tranche_id=prior[0].tranche_id,
            )
            tranche_store.write_continuation(continuation)
        report["continuation"] = None if continuation is None else continuation.to_canonical_dict()
        grant = ParameterMutationGrant(
            grant_id=f"pointer-motor:{plan.plan_id}",
            module_id=MODULE_ID,
            generation_id=base_generation,
            policy=ParameterMutationPolicy.EXPLICIT_NAMES,
            allowed_names=tuple(sorted(trainable_names)),
            max_trainable_parameters=manifest.trainable_parameter_count,
        )
        session = control.begin_candidate(
            inventory,
            grant,
            plan,
            preflight_receipt=preflight,
            policy=policy,
            tranche=tranche,
        )
        session.candidate_module.train()

        soul_workspace = CandidateSoulWorkspace(state_root)
        donor_soul = soul_workspace.branch(DONOR_GENERATION, MODULE_ID).load_snapshot(DONOR_SOUL_ID)
        soul_manifest = soul_workspace.prepare_rebound(
            candidate_id=candidate_generation,
            core_id=MODULE_ID,
            runtime_episode_session_id=f"pointer-motor:{curriculum.curriculum_id}",
            whole_episode_split="train",
            soul_trajectory_ids=tuple(item.episode_id for item in curriculum.split("train")),
            source_soul=donor_soul,
            target_architecture_id=model.architecture_id,
            target_parameter_generation=candidate_generation,
            rebound_layers=rebind_d64_soul_layers(donor_soul, config),
        )
        report["soul_manifest"] = soul_manifest.to_canonical_dict()
        soul_branch = soul_workspace.branch(candidate_generation, MODULE_ID)
        if latest is not None:
            session.restore_checkpoint(step_bundles.checkpoint_for_bundle(latest))
            if soul_branch.load_head().soul_id != latest.after_soul_id:
                raise RuntimeError("pointer-motor checkpoint and candidate Soul HEAD disagree")
        train = curriculum.split("train")
        for _ in range(args.tranche_steps):
            before_soul = soul_branch.load_head()
            captured: dict[str, Any] = {}
            start = session.step_index * args.experiences_per_step
            experiences = tuple(
                train[(start + offset) % len(train)]
                for offset in range(args.experiences_per_step)
            )

            def loss_fn(
                candidate: torch.nn.Module,
                experiences=experiences,
                before_soul=before_soul,
                captured=captured,
            ) -> torch.Tensor:
                if not isinstance(candidate, LivingReasoningCoreD64):
                    raise TypeError("governed candidate is not LivingReasoningCoreD64")
                candidate.train()
                soul = before_soul
                losses = []
                transitions = []
                metrics = []
                for episode in experiences:
                    loss, episode_transitions, row = pointer_motor_episode_objective(
                        candidate,
                        episode,
                        soul,
                        core_id=MODULE_ID,
                        parameter_generation=candidate_generation,
                    )
                    losses.append(loss)
                    transitions.extend(episode_transitions)
                    metrics.append(row)
                    for transition in episode_transitions:
                        soul = apply_soul_transition(soul, transition)
                captured["transitions"] = tuple(transitions)
                captured["metrics"] = tuple(metrics)
                captured["experience_ids"] = tuple(item.episode_id for item in experiences)
                return torch.stack(losses).mean()

            optimization = session.step(loss_fn)
            checkpoint = session.checkpoint(include_optimizer=True)
            bundle = step_bundles.accept_step(
                optimization_receipt=optimization,
                checkpoint=checkpoint,
                soul_manifest=soul_manifest,
                transitions=captured["transitions"],
            )
            report["optimizer_steps"].append(
                {
                    **optimization.to_canonical_dict(),
                    "experience_ids": list(captured["experience_ids"]),
                    "mean_pointer_target_probability": sum(
                        item["target_probability"] for item in captured["metrics"]
                    ) / len(captured["metrics"]),
                    "mean_pointer_margin": sum(item["margin"] for item in captured["metrics"])
                    / len(captured["metrics"]),
                }
            )
            report["accepted_bundles"].append(bundle.to_canonical_dict())

        evaluation = evaluate_pointer_motor(
            session.candidate_module,
            curriculum.split("heldout"),
            soul_branch.load_head(),
            core_id=MODULE_ID,
            parameter_generation=candidate_generation,
        )
        gate = decide_pointer_motor_mastery(evaluation)
        report["heldout"] = evaluation
        report["mastery_gate"] = gate
        report["assignment_completed"] = bool(gate["passed"])
        report["status"] = "completed"
        report["final_global_step"] = session.step_index
        session.complete(reason="direct canonical-address pointer-motor tranche complete")
        report_path = args.report or (
            state_root / "training" / "reasoning" / "pointer_motor" / candidate_generation / "report.json"
        )
        _atomic_json(report_path.resolve(), report)
        _print_json(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
