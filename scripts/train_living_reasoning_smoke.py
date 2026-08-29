"""Governed bounded smoke campaign for the first Soul-conditioned D64 core."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from runtime.field import canonical_sha256
from runtime.soul import SoulStore
from runtime.trainer import (
    CandidateSoulWorkspace,
    OrganKind,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterMutationPolicy,
    TrainerControlPlane,
)
from training import (
    LivingReasoningCoreD64,
    build_living_reasoning_preflight,
    build_living_reasoning_smoke_curriculum,
    candidate_a_config,
    living_episode_objective,
)

ROOT = Path(__file__).resolve().parent.parent


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--ffn-dim", type=int, default=131_072)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = _arguments()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    _seed_everything(args.seed)
    device = _device(args.device)
    config = candidate_a_config(
        n_heads=args.heads,
        n_layers=args.layers,
        ffn_dim=args.ffn_dim,
        page_size=args.page_size,
        dropout=0.0,
    )
    model = LivingReasoningCoreD64(config).to(device)
    curriculum = build_living_reasoning_smoke_curriculum()
    module_id = "reasoning-d64-candidate-a"
    base_generation = "reasoning-d64-untrained-base-v1"
    candidate_generation = "r64a-smoke-" + canonical_sha256(
        {
            "architecture_id": config.architecture_id,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "curriculum_id": curriculum.curriculum_id,
        }
    )[:16]
    descriptor = ParameterModuleDescriptor(
        module_id=module_id,
        organ_kind=OrganKind.REASONING_CORE,
        generation_id=base_generation,
        architecture=config.architecture_id,
        d_model=64,
        tags=("living", "private-soul", "complete-field", "candidate-a"),
    )

    report: dict[str, Any] = {
        "schema": "axon-living-reasoning-smoke-report-v1",
        "device": str(device),
        "seed": args.seed,
        "preflight_only": bool(args.preflight_only),
        "architecture": model.architecture_report(),
        "curriculum_id": curriculum.curriculum_id,
        "train_manifest_id": curriculum.train_manifest_id,
        "heldout_manifest_id": curriculum.heldout_manifest_id,
    }
    with TrainerControlPlane.active(state_root=args.state_root) as control:
        control.declare_expected((descriptor,))
        control.register(descriptor, model)
        inventory = control.snapshot_inventory(exact_value_hashes=True)
        manifest = inventory.module(module_id)
        tensor_names = tuple(item.name for item in manifest.tensors if item.requires_grad)
        plan = ParameterMutationPlan(
            base_inventory_id=inventory.inventory_id,
            module_id=module_id,
            base_generation_id=base_generation,
            candidate_generation_id=candidate_generation,
            tensor_names=tensor_names,
            optimizer_name="AdamW",
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            source_manifest_ids=(curriculum.train_manifest_id,),
            holdout_manifest_ids=(curriculum.heldout_manifest_id,),
        )
        preflight = build_living_reasoning_preflight(
            model=model,
            curriculum=curriculum,
            inventory=inventory,
            plan=plan,
            state_root=args.state_root,
            repo_root=ROOT,
        )
        report.update(
            {
                "inventory_id": inventory.inventory_id,
                "plan_id": plan.plan_id,
                "preflight_receipt_id": preflight.receipt_id,
                "preflight_passed": preflight.passed,
            }
        )
        if not preflight.passed:
            raise RuntimeError("living reasoning preflight failed; optimizer creation denied")
        if args.preflight_only:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
            return 0

        grant = ParameterMutationGrant(
            grant_id=f"candidate-a-smoke:{plan.plan_id}",
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

        live_souls = SoulStore.active(args.state_root)
        live_souls.ensure_core(
            core_id=module_id,
            architecture_id=config.architecture_id,
            parameter_generation=base_generation,
        )
        soul_workspace = CandidateSoulWorkspace(args.state_root)
        soul_manifest = soul_workspace.prepare(
            candidate_id=candidate_generation,
            core_id=module_id,
            runtime_episode_session_id=f"synthetic-mechanism:{curriculum.curriculum_id}",
            whole_episode_split="train",
            soul_trajectory_ids=tuple(item.episode_id for item in curriculum.split("train")),
            candidate_parameter_generation=candidate_generation,
        )
        soul_branch = soul_workspace.branch(candidate_generation, module_id)
        steps: list[dict[str, Any]] = []
        checkpoints = []
        train_episodes = curriculum.split("train")
        for step in range(args.max_steps):
            episode = train_episodes[step % len(train_episodes)]
            captured: dict[str, Any] = {}

            def loss_fn(
                candidate: torch.nn.Module,
                episode=episode,
                captured=captured,
            ) -> torch.Tensor:
                if not isinstance(candidate, LivingReasoningCoreD64):
                    raise TypeError("governed candidate clone has the wrong architecture")
                candidate.train()
                loss, unroll, phase_metrics = living_episode_objective(
                    candidate,
                    episode,
                    soul_branch.load_head(),
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                )
                captured.update(
                    {
                        "loss": float(loss.detach().item()),
                        "unroll": unroll,
                        "phase_metrics": phase_metrics,
                    }
                )
                return loss

            optimizer_receipt = session.step(loss_fn)
            unroll = captured["unroll"]
            soul_receipts = tuple(
                soul_branch.commit_transition(
                    transition,
                    commit_binding=f"trainer-optimizer-step:{optimizer_receipt.receipt_id}",
                )
                for transition in unroll.transitions
            )
            checkpoint = session.checkpoint(include_optimizer=True)
            checkpoints.append(checkpoint)
            steps.append(
                {
                    "step": step + 1,
                    "episode_id": episode.episode_id,
                    "loss": captured["loss"],
                    "optimization_receipt_id": optimizer_receipt.receipt_id,
                    "soul_receipt_ids": [item.receipt_id for item in soul_receipts],
                    "soul_id": soul_branch.load_head().soul_id,
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "phase_metrics": captured["phase_metrics"],
                }
            )
        session.complete(reason="bounded Candidate-A mechanism smoke completed")

        heldout_losses = []
        session.candidate_module.eval()
        with torch.no_grad():
            for episode in curriculum.split("heldout"):
                loss, _unroll, _metrics = living_episode_objective(
                    session.candidate_module,
                    episode,
                    soul_branch.load_head(),
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                )
                heldout_losses.append(float(loss.item()))
        promotion_plan = soul_workspace.promotion_plan(soul_manifest)
        report.update(
            {
                "candidate_generation_id": candidate_generation,
                "soul_candidate_manifest_id": soul_manifest.manifest_id,
                "soul_promotion_plan": promotion_plan.to_canonical_dict(),
                "steps": steps,
                "final_checkpoint_id": checkpoints[-1].checkpoint_id,
                "heldout_mean_loss": sum(heldout_losses) / len(heldout_losses),
                "serving_promotion_claimed": False,
            }
        )

    report["report_id"] = canonical_sha256(report)
    report_path = (
        args.state_root.resolve()
        / "training"
        / "reasoning"
        / candidate_generation
        / "smoke_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "report_path": str(report_path)}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
