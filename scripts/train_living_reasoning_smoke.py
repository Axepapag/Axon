"""Governed English-native D64 reasoning smoke trainer.

This is the active post-2026-09-19 Trainer entrypoint.  Its first developmental
stage is exact substrate literacy over the normal FIRST/REFINED/Soul loop; later
stages train free English reasoning and tagged-region FINAL text.  The retired
DELTA/NO_OP/ABSTAIN, operation, address, and special termination heads are not
present in the candidate topology and no old motor ladder is reachable here.

A pre-amendment checkpoint may be supplied as a governed donor.  Donor loading
is an exact subset migration: all retained tensors must match shape/dtype and
the only discarded tensors may be the explicitly retired typed-output heads.
The resulting English candidate has a new architecture and parameter generation.
"""

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
from runtime.soul import SoulStore
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
    build_living_reasoning_preflight,
    build_living_reasoning_smoke_curriculum,
    build_substrate_literacy_curriculum,
    decide_living_reasoning_mastery,
    decide_substrate_literacy_mastery,
    evaluate_living_episode,
    living_episode_objective,
    migrate_typed_checkpoint_state_to_english_variant,
)

ROOT = Path(__file__).resolve().parent.parent
TRAINER_SCHEMA = "axon-english-reasoning-smoke-trainer-v1"
TEXT_EOS_WEIGHT = 4.0
OBJECTIVE_PROGRAM_ID = canonical_sha256(
    {
        "schema": "axon-english-reasoning-objective-program-v1",
        "public_output_contract": "english-proposal-tagged-final-v1",
        "phases": ["first", "refined", "consolidated"],
        "losses": ["text", "alignment_position", "alignment_copy_gate", "alignment_eos_gate"],
        "text_eos_weight": TEXT_EOS_WEIGHT,
        "retired": ["decision", "operation", "region", "start_address", "end_address"],
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


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish evidence once, or verify the exact preexisting bytes under the lease."""

    data = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        observed = path.read_text(encoding="utf-8")
        if observed != data:
            raise RuntimeError(f"immutable evaluation evidence differs at {path}")
        return
    _atomic_json(path, value)


def _print_json_utf8(value: Mapping[str, Any]) -> None:
    """Emit canonical JSON without depending on the Windows console code page."""

    data = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(data.encode("utf-8"))
        buffer.flush()
        return
    sys.stdout.write(data)
    sys.stdout.flush()


def _recover_resume_boundary(
    step_bundles: CandidateStepBundleCoordinator,
    module_id: str,
    candidate_generation: str,
    *,
    resume: bool,
) -> tuple[tuple[Any, ...], Any | None]:
    """Recover pending atomic work before selecting the resumable boundary."""

    recovered = step_bundles.recover_pending(module_id, candidate_generation)
    latest = step_bundles.latest_bundle(module_id, candidate_generation)
    if latest is not None and not resume:
        raise RuntimeError(
            "English candidate already has accepted work; pass --resume or change its governed lineage"
        )
    if resume and latest is None:
        raise RuntimeError("--resume was requested but no accepted English candidate bundle exists")
    return recovered, latest


def _persist_tranche_lineage(
    tranche_store: TrancheStore,
    tranche: ResourceTranche,
    latest: Any | None,
) -> tuple[Path, Path | None, TrancheContinuation | None]:
    """Publish the tranche and, for resumes, its exact continuation receipt."""

    tranche_path = tranche_store.write_tranche(tranche)
    if latest is None:
        if tranche.base_global_step != 0 or tranche.parent_bundle_id is not None:
            raise RuntimeError("a fresh English tranche must begin at global step zero")
        return tranche_path, None, None

    if (
        tranche.base_global_step != latest.step
        or tranche.parent_bundle_id != latest.bundle_id
    ):
        raise RuntimeError("English continuation tranche does not bind the accepted parent")
    prior = tuple(
        item
        for item in tranche_store.tranches_for(
            tranche.module_id,
            tranche.candidate_generation_id,
        )
        if item.tranche_id != tranche.tranche_id
        and item.final_global_step == tranche.base_global_step
    )
    if len(prior) != 1:
        raise RuntimeError(
            "English continuation requires exactly one durable prior tranche reaching its parent"
        )
    continuation = TrancheContinuation(
        tranche_id=tranche.tranche_id,
        module_id=tranche.module_id,
        candidate_generation_id=tranche.candidate_generation_id,
        plan_id=tranche.plan_id,
        learning_policy_id=tranche.learning_policy_id,
        parent_bundle_id=latest.bundle_id,
        parent_checkpoint_id=latest.checkpoint_id,
        parent_optimizer_receipt_id=latest.optimization_receipt_id,
        parent_soul_id=latest.after_soul_id,
        parent_global_step=latest.step,
        prior_tranche_id=prior[0].tranche_id,
    )
    continuation_path = tranche_store.write_continuation(continuation)
    return tranche_path, continuation_path, continuation


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(name)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--module-id", default="r64-english-reasoning")
    parser.add_argument("--tranche-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--ffn-dim", type=int, default=131_072)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--state-tokens", type=int, default=4)
    parser.add_argument("--generate-gate-bias", type=float, default=0.0)
    parser.add_argument("--experiences-per-step", type=int, default=8)
    parser.add_argument("--curriculum", choices=("substrate", "reasoning"), default="substrate")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--typed-donor-checkpoint",
        type=Path,
        default=None,
        help=(
            "pre-amendment Trainer checkpoint payload used only as an exact donor; "
            "this is never treated as an architecture-compatible resume"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="optional report path; defaults under State/training/reasoning/english",
    )
    return parser.parse_args()


def _load_typed_donor(path: Path) -> tuple[Mapping[str, torch.Tensor], str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("typed donor checkpoint must contain a mapping")
    if isinstance(payload.get("module_state_dict"), Mapping):
        state = payload["module_state_dict"]
        descriptor = payload.get("descriptor")
        if not isinstance(descriptor, Mapping):
            raise ValueError("Trainer donor checkpoint is missing its descriptor")
        architecture_id = str(descriptor.get("architecture", ""))
        generation = str(descriptor.get("generation_id", ""))
    elif isinstance(payload.get("model_state_dict"), Mapping):
        state = payload["model_state_dict"]
        architecture_id = str(payload.get("architecture_id", ""))
        generation = str(payload.get("parameter_generation", ""))
    elif all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise ValueError(
            "raw state_dict donors are ambiguous; use a Trainer checkpoint carrying architecture/generation provenance"
        )
    else:
        raise ValueError("unrecognized typed donor checkpoint payload")
    if not architecture_id or not generation:
        raise ValueError("typed donor checkpoint lacks architecture/generation provenance")
    return state, architecture_id, generation


def _aggregate_heldout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("English reasoning smoke curriculum has no heldout episodes")
    if len(rows) == 1:
        return rows[0]
    counts = {
        "supervised_phase_count": sum(float(row["supervised_phase_count"]) for row in rows),
        "text_exact_count": sum(float(row["text_exact_count"]) for row in rows),
        "text_teacher_forced_content_count": sum(int(row["text_teacher_forced_content_count"]) for row in rows),
        "text_teacher_forced_content_correct": sum(int(row["text_teacher_forced_content_correct"]) for row in rows),
        "text_teacher_forced_eos_count": sum(int(row["text_teacher_forced_eos_count"]) for row in rows),
        "text_teacher_forced_eos_correct": sum(int(row["text_teacher_forced_eos_correct"]) for row in rows),
        "final_verdict_count": sum(float(row["final_verdict_count"]) for row in rows),
        "final_verdict_valid_count": sum(float(row["final_verdict_valid_count"]) for row in rows),
        "phase_output_count": sum(float(row["phase_output_count"]) for row in rows),
        "phase_expected_count": sum(float(row["phase_expected_count"]) for row in rows),
        "complete_field_coverage_count": sum(float(row["complete_field_coverage_count"]) for row in rows),
    }
    histogram: dict[str, int] = {}
    for row in rows:
        for text, count in dict(row["constant_text_target_histogram"]).items():
            histogram[text] = histogram.get(text, 0) + int(count)
    phase_count = max(1.0, counts["supervised_phase_count"])
    final_count = max(1.0, counts["final_verdict_count"])
    expected_phase_count = max(1.0, counts["phase_expected_count"])
    strongest = max(histogram.values(), default=0)
    observability_rows = [
        dict(row.get("decoder_observability", {}))
        for row in rows
        if isinstance(row.get("decoder_observability"), Mapping)
    ]
    observability_samples = sum(
        int(row.get("sample_count", 0)) for row in observability_rows
    )

    def weighted_mean(key: str) -> float | None:
        values = [
            (float(row[key]), int(row.get("sample_count", 0)))
            for row in observability_rows
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and row.get(key) is not None
        ]
        denominator = sum(weight for _value, weight in values)
        return (
            sum(value * weight for value, weight in values) / denominator
            if denominator
            else None
        )

    def observed_min(key: str) -> float | None:
        values = [
            float(row[key])
            for row in observability_rows
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and row.get(key) is not None
        ]
        return min(values) if values else None

    def observed_max(key: str) -> float | None:
        values = [
            float(row[key])
            for row in observability_rows
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and row.get(key) is not None
        ]
        return max(values) if values else None

    decoder_observability = {
        "episode_count": len(observability_rows),
        "sample_count": observability_samples,
    }
    for key in (
        "teacher_forced_terminal_eos_probability_mean",
        "teacher_forced_terminal_eos_rank_mean",
        "teacher_forced_terminal_eos_top1_rate",
        "teacher_forced_generated_eos_probability_mean",
        "teacher_forced_generated_eos_rank_mean",
        "teacher_forced_generated_eos_top1_rate",
        "free_running_termination_rate",
        "free_running_first_slice_termination_rate",
        "free_running_unicode_valid_rate",
        "free_running_decision_count_mean",
        "free_running_eos_probability_first_mean",
        "free_running_eos_probability_max_mean",
        "free_running_generated_eos_probability_first_mean",
        "free_running_generated_eos_rank_first_mean",
        "free_running_invalid_transport_rate",
    ):
        decoder_observability[key] = weighted_mean(key)
    for key in (
        "teacher_forced_terminal_eos_probability_min",
        "teacher_forced_terminal_eos_probability_max",
    ):
        decoder_observability[key] = (
            observed_min(key) if key.endswith("_min") else observed_max(key)
        )
    return {
        **counts,
        "text_exact_rate": counts["text_exact_count"] / phase_count,
        "text_teacher_forced_content_accuracy": counts["text_teacher_forced_content_correct"]
        / max(1, counts["text_teacher_forced_content_count"]),
        "text_teacher_forced_eos_accuracy": counts["text_teacher_forced_eos_correct"]
        / max(1, counts["text_teacher_forced_eos_count"]),
        "final_verdict_valid_rate": counts["final_verdict_valid_count"] / final_count,
        "complete_field_coverage_rate": counts["complete_field_coverage_count"] / expected_phase_count,
        "constant_text_target_histogram": dict(sorted(histogram.items())),
        "constant_text_exact_count": float(strongest),
        "constant_text_exact_floor": strongest / phase_count,
        "decoder_observability": decoder_observability,
        "episodes": rows,
    }


def main() -> int:
    args = _arguments()
    if args.tranche_steps < 1:
        raise ValueError("--tranche-steps must be positive")
    if not (0.0 < args.learning_rate < float("inf")):
        raise ValueError("--learning-rate must be positive and finite")
    if args.experiences_per_step < 1:
        raise ValueError("--experiences-per-step must be positive")
    _seed_everything(args.seed)
    device = _device(args.device)

    config = LivingReasoningCoreConfig(
        n_heads=args.heads,
        n_layers=args.layers,
        ffn_dim=args.ffn_dim,
        state_tokens=args.state_tokens,
        page_size=args.page_size,
        generate_gate_bias=args.generate_gate_bias,
    )
    model = LivingReasoningCoreD64(config).to(device)
    migration = None
    if args.typed_donor_checkpoint is not None:
        donor_state, donor_architecture, donor_generation = _load_typed_donor(
            args.typed_donor_checkpoint.resolve()
        )
        target_generation = "english-donor-" + canonical_sha256(
            {
                "donor_architecture": donor_architecture,
                "donor_generation": donor_generation,
                "target_architecture": model.architecture_id,
            }
        )[:20]
        migration = migrate_typed_checkpoint_state_to_english_variant(
            donor_state,
            model,
            source_architecture_id=donor_architecture,
            source_parameter_generation=donor_generation,
            target_parameter_generation=target_generation,
        )
        base_generation = target_generation
        migration_path = (
            args.state_root.resolve()
            / "training"
            / "reasoning"
            / "english_migrations"
            / f"{migration.receipt_id}.json"
        )
        _atomic_json(migration_path, migration.to_canonical_dict())
    else:
        base_generation = "english-init-" + canonical_sha256(
            {
                "architecture_id": model.architecture_id,
                "seed": args.seed,
                "generate_gate_bias": args.generate_gate_bias,
            }
        )[:20]

    if args.curriculum == "substrate":
        curriculum = build_substrate_literacy_curriculum()
        mastery_decider = decide_substrate_literacy_mastery
    else:
        curriculum = build_living_reasoning_smoke_curriculum()
        mastery_decider = decide_living_reasoning_mastery
    policy = GovernedLearningPolicy(
        optimizer="adamw",
        learning_rate=args.learning_rate,
        objective_program_id=OBJECTIVE_PROGRAM_ID,
    )
    candidate_generation = "english-candidate-" + canonical_sha256(
        {
            "module_id": args.module_id,
            "base_generation": base_generation,
            "architecture_id": model.architecture_id,
            "curriculum_id": curriculum.curriculum_id,
            "learning_policy_id": policy.policy_id,
            "curriculum_kind": args.curriculum,
            "experiences_per_step": args.experiences_per_step,
        }
    )[:20]
    descriptor = ParameterModuleDescriptor(
        module_id=args.module_id,
        organ_kind=OrganKind.REASONING_CORE,
        generation_id=base_generation,
        architecture=model.architecture_id,
        d_model=64,
    )

    state_root = args.state_root.resolve()
    step_bundles = CandidateStepBundleCoordinator(state_root)
    tranche_store = TrancheStore(state_root / "training" / "trainer")

    with TrainerControlPlane.active(state_root=state_root) as control:
        # Recovery is a writer action because it may publish a rebuilt pointer
        # and finalize a pending atomic parameter/Soul step.  Acquire the
        # Trainer lease first, then perform it before choosing tranche lineage
        # or restoring a checkpoint.
        recovered, latest = _recover_resume_boundary(
            step_bundles,
            args.module_id,
            candidate_generation,
            resume=args.resume,
        )
        control.declare_expected((descriptor,))
        control.register(descriptor, model)
        inventory = control.snapshot_inventory(exact_value_hashes=True)
        manifest = inventory.module(args.module_id)
        plan = ParameterMutationPlanV2(
            base_inventory_id=inventory.inventory_id,
            module_id=args.module_id,
            base_generation_id=base_generation,
            candidate_generation_id=candidate_generation,
            tensor_names=tuple(item.name for item in manifest.tensors if item.requires_grad),
            source_manifest_ids=(curriculum.train_manifest_id,),
            holdout_manifest_ids=(curriculum.heldout_manifest_id,),
        )
        if latest is not None:
            latest_checkpoint = step_bundles.checkpoint_for_bundle(latest)
            if latest_checkpoint.plan_id != plan.plan_id:
                raise RuntimeError(
                    "accepted English candidate belongs to a different mutation plan"
                )
            if latest_checkpoint.learning_policy_id != policy.policy_id:
                raise RuntimeError(
                    "accepted English candidate belongs to a different learning policy"
                )
        tranche = ResourceTranche(
            module_id=args.module_id,
            candidate_generation_id=candidate_generation,
            plan_id=plan.plan_id,
            learning_policy_id=policy.policy_id,
            base_global_step=0 if latest is None else latest.step,
            steps=args.tranche_steps,
            parent_bundle_id=None if latest is None else latest.bundle_id,
            purpose=f"bounded English-native {args.curriculum} lived-experience tranche",
        )
        tranche_path, continuation_path, continuation = _persist_tranche_lineage(
            tranche_store,
            tranche,
            latest,
        )
        preflight = build_living_reasoning_preflight(
            model=model,
            curriculum=curriculum,
            inventory=inventory,
            plan=plan,
            state_root=state_root,
            repo_root=ROOT,
            resource_tranche=tranche,
        )

        report: dict[str, Any] = {
            "schema": TRAINER_SCHEMA,
            "module_id": args.module_id,
            "architecture_id": model.architecture_id,
            "base_generation": base_generation,
            "candidate_generation": candidate_generation,
            "curriculum_id": curriculum.curriculum_id,
            "curriculum_kind": args.curriculum,
            "experiences_per_step": args.experiences_per_step,
            "objective_program_id": OBJECTIVE_PROGRAM_ID,
            "text_eos_weight": TEXT_EOS_WEIGHT,
            "learning_policy": policy.to_canonical_dict(),
            "plan": plan.to_canonical_dict(),
            "tranche": tranche.to_canonical_dict(),
            "tranche_artifact_path": str(tranche_path),
            "continuation": None if continuation is None else continuation.to_canonical_dict(),
            "continuation_artifact_path": None if continuation_path is None else str(continuation_path),
            "preflight": preflight.to_canonical_dict(),
            "typed_donor_migration": None if migration is None else migration.to_canonical_dict(),
            "device": str(device),
            "recovered_bundle_ids": [item.bundle_id for item in recovered],
            "recovered_bundle_steps": [item.step for item in recovered],
            "optimizer_steps": [],
            "accepted_bundles": [],
            "promotion_attempted": False,
        }
        report_path = (
            args.report.resolve()
            if args.report is not None
            else state_root
            / "training"
            / "reasoning"
            / "english"
            / candidate_generation
            / "report.json"
        )
        if args.preflight_only:
            report["status"] = "preflight_only"
            _atomic_json(report_path, report)
            _print_json_utf8(report)
            return 0

        grant = ParameterMutationGrant(
            grant_id=f"english-reasoning:{plan.plan_id}",
            module_id=args.module_id,
            generation_id=base_generation,
            policy=ParameterMutationPolicy.FULL,
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

        live_souls = SoulStore.active(state_root)
        live_souls.ensure_core(
            core_id=args.module_id,
            architecture_id=model.architecture_id,
            parameter_generation=base_generation,
        )
        soul_workspace = CandidateSoulWorkspace(state_root)
        soul_manifest = soul_workspace.prepare(
            candidate_id=candidate_generation,
            core_id=args.module_id,
            runtime_episode_session_id=f"english-{args.curriculum}:{curriculum.curriculum_id}",
            whole_episode_split="train",
            soul_trajectory_ids=tuple(item.episode_id for item in curriculum.split("train")),
            candidate_parameter_generation=candidate_generation,
        )
        soul_branch = soul_workspace.branch(candidate_generation, args.module_id)
        if latest is not None:
            session.restore_checkpoint(step_bundles.checkpoint_for_bundle(latest))
            if soul_branch.load_head().soul_id != latest.after_soul_id:
                raise RuntimeError("accepted English checkpoint and candidate Soul HEAD disagree")

        train = curriculum.split("train")
        for _local_index in range(args.tranche_steps):
            before_soul = soul_branch.load_head()
            captured: dict[str, Any] = {}
            start_experience = session.step_index * args.experiences_per_step
            experiences = tuple(
                train[(start_experience + offset) % len(train)]
                for offset in range(args.experiences_per_step)
            )

            def loss_fn(
                candidate: torch.nn.Module,
                experiences=experiences,
                before_soul=before_soul,
                captured=captured,
            ) -> torch.Tensor:
                if not isinstance(candidate, LivingReasoningCoreD64):
                    raise TypeError("governed candidate is not the English-native D64 core")
                candidate.train()
                soul = before_soul
                losses: list[torch.Tensor] = []
                transitions = []
                metrics = []
                for episode in experiences:
                    loss, unroll, phase_metrics = living_episode_objective(
                        candidate,
                        episode,
                        soul,
                        core_id=args.module_id,
                        parameter_generation=candidate_generation,
                        text_eos_weight=TEXT_EOS_WEIGHT,
                    )
                    losses.append(loss)
                    transitions.extend(unroll.transitions)
                    metrics.append(phase_metrics)
                    soul = unroll.souls[-1]
                captured["transitions"] = tuple(transitions)
                captured["phase_metrics"] = tuple(metrics)
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
            step_row = optimization.to_canonical_dict()
            step_row["experience_ids"] = list(captured["experience_ids"])
            report["optimizer_steps"].append(step_row)
            report["accepted_bundles"].append(bundle.to_canonical_dict())

        candidate = session.candidate_module
        candidate.eval()
        evaluation_soul = soul_branch.load_head()
        heldout_rows = [
            evaluate_living_episode(
                candidate,
                episode,
                evaluation_soul,
                core_id=args.module_id,
                parameter_generation=candidate_generation,
                transcript_sink=[],
                transcript_sink_cap=3,
            )
            for episode in curriculum.split("heldout")
        ]
        heldout = _aggregate_heldout(heldout_rows)
        gate = mastery_decider(heldout)
        accepted_boundary = step_bundles.latest_bundle(args.module_id, candidate_generation)
        if accepted_boundary is None:
            raise RuntimeError("English heldout evaluation requires an accepted parameter/Soul boundary")
        candidate_soul_id = soul_branch.load_head().soul_id
        if candidate_soul_id != accepted_boundary.after_soul_id:
            raise RuntimeError("English heldout evaluation Soul disagrees with accepted parameter boundary")
        evaluation_body = {
            "schema": "axon-english-reasoning-heldout-evaluation-v1",
            "module_id": args.module_id,
            "architecture_id": model.architecture_id,
            "candidate_generation": candidate_generation,
            "curriculum_id": curriculum.curriculum_id,
            "curriculum_kind": args.curriculum,
            "plan_id": plan.plan_id,
            "tranche_id": tranche.tranche_id,
            "accepted_bundle_id": accepted_boundary.bundle_id,
            "candidate_soul_id": candidate_soul_id,
            "heldout": heldout,
            "mastery_gate": gate,
            "assignment_completed": bool(gate["passed"]),
        }
        evaluation_id = canonical_sha256(evaluation_body)
        evaluation_path = (
            state_root
            / "training"
            / "reasoning"
            / "english"
            / candidate_generation
            / "evaluations"
            / f"{evaluation_id}.json"
        )
        _immutable_json(evaluation_path, {**evaluation_body, "evaluation_id": evaluation_id})
        mastery_landmark_id = None
        if bool(gate["passed"]):
            mastery_landmark_id = step_bundles.mark_landmark(
                accepted_boundary,
                label="heldout-mastery-passed",
                evidence_ids=(evaluation_id, canonical_sha256(gate)),
            )
        session.complete(reason=f"English-native {args.curriculum} tranche complete; no promotion attempted")
        report.update(
            {
                "status": "completed",
                "final_global_step": int(report["accepted_bundles"][-1]["step"]),
                "candidate_soul_id": candidate_soul_id,
                "heldout": heldout,
                "mastery_gate": gate,
                "assignment_completed": bool(gate["passed"]),
                "heldout_evaluation_id": evaluation_id,
                "heldout_evaluation_path": str(evaluation_path),
                "mastery_landmark_id": mastery_landmark_id,
            }
        )
        _atomic_json(report_path, report)
        _print_json_utf8(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
