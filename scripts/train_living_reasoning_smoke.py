"""Governed bounded smoke campaign for the first Soul-conditioned D64 core."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from runtime.field import canonical_sha256
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
)
from training import (
    LivingReasoningCoreD64,
    build_living_reasoning_preflight,
    build_living_reasoning_smoke_curriculum,
    candidate_a_config,
    evaluate_living_episode,
    living_episode_objective,
    living_source_counterfactuals,
)

ROOT = Path(__file__).resolve().parent.parent


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument(
        "--run-steps",
        type=int,
        default=None,
        help="bounded steps to execute this invocation within the predeclared max-steps campaign",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--ffn-dim", type=int, default=131_072)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1,
        help="accepted parameter+Soul checkpoint segment length in optimizer steps",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest accepted parameter+Soul step bundle",
    )
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
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be positive")
    if args.run_steps is not None and args.run_steps < 1:
        raise ValueError("--run-steps must be positive when supplied")
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
            "checkpoint_interval": args.checkpoint_interval,
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
        step_bundles = CandidateStepBundleCoordinator(args.state_root)
        recovered_bundles = step_bundles.recover_pending(module_id, candidate_generation)
        latest_bundle = step_bundles.latest_bundle(module_id, candidate_generation)
        if latest_bundle is not None:
            if not args.resume:
                raise RuntimeError(
                    "candidate already has accepted work; pass --resume or choose a different governed campaign"
                )
            session.restore_checkpoint(step_bundles.checkpoint_for_bundle(latest_bundle))
            if soul_branch.load_head().soul_id != latest_bundle.after_soul_id:
                raise RuntimeError("accepted checkpoint and candidate Soul HEAD disagree")
        elif args.resume:
            report["resume_note"] = (
                "no accepted bundle existed; unaccepted optimizer/checkpoint orphans, if any, "
                "remain evidence and the candidate restarts from its governed base"
            )
        steps: list[dict[str, Any]] = []
        checkpoints = []
        train_episodes = curriculum.split("train")
        heldout_episodes = curriculum.split("heldout")

        def evaluate_candidate() -> dict[str, Any]:
            session.candidate_module.eval()
            exact_rows = [
                evaluate_living_episode(
                    session.candidate_module,
                    episode,
                    soul_branch.load_head(),
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                )
                for episode in heldout_episodes
            ]
            losses = []
            with torch.no_grad():
                for episode in heldout_episodes:
                    loss, _unroll, _metrics = living_episode_objective(
                        session.candidate_module,
                        episode,
                        soul_branch.load_head(),
                        core_id=module_id,
                        parameter_generation=candidate_generation,
                    )
                    losses.append(float(loss.item()))
            names = (
                "typed_emission_exact_rate",
                "payload_transport_exact_rate",
                "complete_field_coverage_rate",
            )
            payload_token_count = sum(
                row["payload_teacher_forced_token_count"] for row in exact_rows
            )
            payload_token_correct = sum(
                row["payload_teacher_forced_token_correct"] for row in exact_rows
            )
            payload_target_counts = [
                sum(row["payload_teacher_forced_target_counts"][index] for row in exact_rows)
                for index in range(session.candidate_module.eos_index + 1)
            ]
            return {
                "heldout_mean_loss": sum(losses) / len(losses),
                **{
                    name: sum(row[name] for row in exact_rows) / len(exact_rows)
                    for name in names
                },
                "constant_typed_emission_exact_floor": 0.0,
                "constant_payload_transport_exact_floor": 0.0,
                "payload_teacher_forced_token_accuracy": payload_token_correct
                / max(1, payload_token_count),
                "constant_payload_token_accuracy_floor": max(payload_target_counts)
                / max(1, payload_token_count),
                "counterfactuals": living_source_counterfactuals(
                    session.candidate_module,
                    heldout_episodes[0],
                    soul_branch.load_head(),
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                ),
            }

        initial_evaluation = evaluate_candidate()
        campaign_report_dir = (
            args.state_root.resolve()
            / "training"
            / "reasoning"
            / candidate_generation
        )
        prior_reports = sorted(campaign_report_dir.glob("segment_*.json"))
        campaign_baseline_evaluation = (
            initial_evaluation
            if not prior_reports
            else json.loads(prior_reports[0].read_text(encoding="utf-8"))[
                "initial_evaluation"
            ]
        )
        start_step = 0 if latest_bundle is None else latest_bundle.step
        end_step = min(
            args.max_steps,
            start_step + (args.max_steps if args.run_steps is None else args.run_steps),
        )
        if start_step >= args.max_steps:
            raise RuntimeError("candidate campaign is already complete")
        segment_transitions = []
        segment_receipt_ids = []
        ephemeral_soul = soul_branch.load_head()
        for step in range(start_step, end_step):
            episode = train_episodes[step % len(train_episodes)]
            captured: dict[str, Any] = {}

            def loss_fn(
                candidate: torch.nn.Module,
                episode=episode,
                captured=captured,
                soul=ephemeral_soul,
            ) -> torch.Tensor:
                if not isinstance(candidate, LivingReasoningCoreD64):
                    raise TypeError("governed candidate clone has the wrong architecture")
                candidate.train()
                loss, unroll, phase_metrics = living_episode_objective(
                    candidate,
                    episode,
                    soul,
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

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started_at = time.perf_counter()
            optimizer_receipt = session.step(loss_fn)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall_seconds = time.perf_counter() - started_at
            unroll = captured["unroll"]
            ephemeral_soul = unroll.souls[-1]
            segment_transitions.extend(unroll.transitions)
            segment_receipt_ids.append(optimizer_receipt.receipt_id)
            checkpoint_due = (
                (step + 1) % args.checkpoint_interval == 0
                or step + 1 == end_step
            )
            checkpoint = None
            accepted_bundle = None
            accepted_segment_receipt_ids = []
            if checkpoint_due:
                checkpoint = session.checkpoint(include_optimizer=True)
                checkpoints.append(checkpoint)
                accepted_bundle = step_bundles.accept_step(
                    optimization_receipt=optimizer_receipt,
                    checkpoint=checkpoint,
                    soul_manifest=soul_manifest,
                    transitions=tuple(segment_transitions),
                )
                if soul_branch.load_head().soul_id != ephemeral_soul.soul_id:
                    raise RuntimeError("accepted checkpoint segment and ephemeral Soul disagree")
                accepted_segment_receipt_ids = list(segment_receipt_ids)
                segment_transitions.clear()
                segment_receipt_ids.clear()
            steps.append(
                {
                    "step": step + 1,
                    "episode_id": episode.episode_id,
                    "loss": captured["loss"],
                    "optimization_receipt_id": optimizer_receipt.receipt_id,
                    "accepted_step_bundle_id": (
                        None if accepted_bundle is None else accepted_bundle.bundle_id
                    ),
                    "accepted_segment_optimizer_receipt_ids": accepted_segment_receipt_ids,
                    "soul_receipt_ids": (
                        [] if accepted_bundle is None else list(accepted_bundle.soul_receipt_ids)
                    ),
                    "soul_id": ephemeral_soul.soul_id,
                    "checkpoint_id": None if checkpoint is None else checkpoint.checkpoint_id,
                    "phase_metrics": captured["phase_metrics"],
                    "wall_seconds": wall_seconds,
                    "peak_cuda_bytes": (
                        0
                        if device.type != "cuda"
                        else int(torch.cuda.max_memory_allocated(device))
                    ),
                }
            )
        session.complete(reason="bounded Candidate-A campaign segment completed")

        final_evaluation = evaluate_candidate()
        counterfactuals_passed = all(
            value > 1e-8 for value in final_evaluation["counterfactuals"].values()
        )
        task_gate_passed = (
            final_evaluation["heldout_mean_loss"]
            < campaign_baseline_evaluation["heldout_mean_loss"]
            and final_evaluation["payload_teacher_forced_token_accuracy"]
            > final_evaluation["constant_payload_token_accuracy_floor"]
            and counterfactuals_passed
        )
        exact_gate_passed = (
            final_evaluation["typed_emission_exact_rate"]
            > final_evaluation["constant_typed_emission_exact_floor"]
            and final_evaluation["payload_transport_exact_rate"]
            > final_evaluation["constant_payload_transport_exact_floor"]
        )
        promotion_plan = soul_workspace.promotion_plan(soul_manifest)
        report.update(
            {
                "candidate_generation_id": candidate_generation,
                "soul_candidate_manifest_id": soul_manifest.manifest_id,
                "recovered_bundle_ids": [item.bundle_id for item in recovered_bundles],
                "soul_promotion_plan": promotion_plan.to_canonical_dict(),
                "steps": steps,
                "checkpoint_interval": args.checkpoint_interval,
                "segment_start_step": start_step + 1,
                "segment_end_step": end_step,
                "campaign_max_steps": args.max_steps,
                "campaign_complete": end_step == args.max_steps,
                "initial_evaluation": initial_evaluation,
                "campaign_baseline_evaluation": campaign_baseline_evaluation,
                "segment_heldout_loss_fell": (
                    final_evaluation["heldout_mean_loss"]
                    < initial_evaluation["heldout_mean_loss"]
                ),
                "final_evaluation": final_evaluation,
                "final_checkpoint_id": (
                    checkpoints[-1].checkpoint_id
                    if checkpoints
                    else step_bundles.latest_bundle(module_id, candidate_generation).checkpoint_id
                ),
                "task_gate_passed": task_gate_passed,
                "task_gate_policy": (
                    "heldout loss falls; teacher-forced transport token accuracy exceeds "
                    "the strongest heldout constant-category floor; field/proposal/Soul "
                    "counterfactuals are nonzero"
                ),
                "exact_serving_gate_passed": exact_gate_passed,
                "exact_serving_gate_policy": (
                    "free-running typed emission and complete Unicode payload exact rates "
                    "must both exceed their constant zero floors"
                ),
                "serving_promotion_claimed": False,
            }
        )

    report["report_id"] = canonical_sha256(report)
    report_path = campaign_report_dir / f"segment_{start_step + 1:09d}_{end_step:09d}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**report, "report_path": str(report_path)}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
