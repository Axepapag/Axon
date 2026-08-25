"""Run a bounded governed D64 Heart identity-generalization candidate.

This campaign resumes from a verified non-serving checkpoint, replays the
already-proven mechanism cases, visits a deterministic length-bucketed corpus,
and evaluates unseen content, length extrapolation, diagonal alignment, and
head/middle/tail source counterfactuals. Passing remains non-semantic and never
authorizes Heart activation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch

from runtime.field import canonical_sha256
from runtime.heart.translation_core import (
    HEART_TRANSLATION_ARCHITECTURE,
    HeartTranslationCore,
    HeartTranslationCoreConfig,
)
from runtime.trainer import (
    CandidateCheckpointRecord,
    EvaluationObservation,
    EvaluationRequirement,
    GovernedLearningPolicy,
    MetricComparison,
    OrganKind,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterMutationPolicy,
    PrecisionMode,
    PromotionGate,
    SchedulerKind,
    TrainerControlPlane,
    TrainerStateStore,
)
from training.heart_preflight import build_heart_training_preflight
from training.heart_translation import (
    HeartDecoderGeneralizationObjective,
    build_heart_decoder_generalization_curriculum,
    collate_heart_translation_cases,
    deterministic_length_bucketed_batches,
    evaluate_heart_decoder_diagnostics,
    evaluate_heart_decoder_generalization,
    heart_decoder_generalization_loss,
)


HEART_DECODER_GENERALIZATION_SMOKE_SCHEMA = "axon-heart-decoder-generalization-smoke-v1"
HEART_DECODER_MECHANISM_SMOKE_SCHEMA = "axon-heart-decoder-mechanism-smoke-v1"
HEART_DECODER_GENERALIZATION_RECOVERY_SCHEMA = "axon-heart-decoder-generalization-recovery-v1"
HEART_DECODER_GENERALIZATION_SUITE = "heart-decoder-generalization-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _checkpoint_record(path: Path) -> CandidateCheckpointRecord:
    value = _read_json(path)
    stated_id = value.pop("checkpoint_id", None)
    value.pop("schema", None)
    record = CandidateCheckpointRecord(**value)
    if stated_id != record.checkpoint_id:
        raise ValueError("checkpoint record identity disagrees with canonical content")
    return record


def _atomic_json(path: Path, value: dict[str, Any]) -> Path:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"existing generalization run summary disagrees with content: {path}")
        return path
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return path


def _generalization_gate(module_id: str, candidate_generation_id: str) -> PromotionGate:
    specs = (
        ("teacher_forced_character_accuracy", MetricComparison.GREATER_OR_EQUAL, 0.90, "unseen character accuracy"),
        ("teacher_forced_eos_accuracy", MetricComparison.GREATER_OR_EQUAL, 0.90, "unseen EOS accuracy"),
        ("greedy_exact_rate", MetricComparison.GREATER_OR_EQUAL, 0.50, "unseen free-running exactness"),
        ("greedy_termination_rate", MetricComparison.GREATER_OR_EQUAL, 0.90, "unseen termination"),
        ("greedy_correct_prefix_fraction", MetricComparison.GREATER_OR_EQUAL, 0.75, "unseen correct-prefix fraction"),
        ("diagonal_attention_mass", MetricComparison.GREATER_OR_EQUAL, 0.65, "same-position attention mass"),
        ("diagonal_attention_top1_rate", MetricComparison.GREATER_OR_EQUAL, 0.80, "same-position attention top-one rate"),
        ("counterfactual_pair_exact_rate", MetricComparison.GREATER_OR_EQUAL, 1.0, "head middle tail source dependence"),
        ("minimum_counterfactual_target_margin", MetricComparison.GREATER_OR_EQUAL, 0.10, "counterfactual target margin"),
        ("memorized_train_output_count", MetricComparison.LESS_OR_EQUAL, 0.0, "no heldout output copied from a train target"),
        ("replay_greedy_exact_rate", MetricComparison.GREATER_OR_EQUAL, 1.0, "preserve proven mechanism replay"),
        ("replay_teacher_forced_character_accuracy", MetricComparison.GREATER_OR_EQUAL, 0.99, "preserve replay characters"),
        ("replay_teacher_forced_eos_accuracy", MetricComparison.GREATER_OR_EQUAL, 1.0, "preserve replay EOS"),
        ("replay_greedy_termination_rate", MetricComparison.GREATER_OR_EQUAL, 1.0, "preserve replay termination"),
    )
    requirements = tuple(
        EvaluationRequirement(
            suite_id=HEART_DECODER_GENERALIZATION_SUITE,
            metric_name=metric,
            comparison=comparison,
            threshold=threshold,
            label=label,
        )
        for metric, comparison, threshold, label in specs
    )
    return PromotionGate(
        gate_id="heart-decoder-generalization-floor-v1",
        module_id=module_id,
        candidate_generation_id=candidate_generation_id,
        requirements=requirements,
        required_suite_ids=(HEART_DECODER_GENERALIZATION_SUITE,),
    )


@torch.no_grad()
def _fixed_audit_loss(
    model: HeartTranslationCore,
    cases,
    objective: HeartDecoderGeneralizationObjective,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    try:
        batch = collate_heart_translation_cases(model, cases).to(device)
        loss = heart_decoder_generalization_loss(model, batch, objective)
        return {
            "total": float(loss.total.item()),
            "translation": float(loss.translation.item()),
            "diagonal_alignment": float(loss.diagonal_alignment.item()),
            "copy_route": float(loss.copy_route.item()),
            "eos_route": float(loss.eos_route.item()),
            "semantic": float(loss.semantic.item()),
            "pointers": float(loss.pointers.item()),
        }
    finally:
        model.train(was_training)


def _generalization_metrics(diagnostic, evidence, replay_diagnostic) -> dict[str, float]:
    return {
        "teacher_forced_character_accuracy": diagnostic.teacher_forced_character_accuracy,
        "teacher_forced_eos_accuracy": diagnostic.teacher_forced_eos_accuracy,
        "greedy_exact_rate": diagnostic.greedy_exact_rate,
        "greedy_termination_rate": diagnostic.greedy_termination_rate,
        "greedy_correct_prefix_fraction": diagnostic.mean_greedy_correct_prefix_fraction,
        "diagonal_attention_mass": evidence.mean_diagonal_attention_mass,
        "diagonal_attention_top1_rate": evidence.diagonal_attention_top1_rate,
        "counterfactual_pair_exact_rate": evidence.counterfactual_pair_exact_rate,
        "minimum_counterfactual_target_margin": evidence.minimum_counterfactual_target_margin,
        "memorized_train_output_count": float(evidence.memorized_train_output_count),
        "replay_greedy_exact_rate": replay_diagnostic.greedy_exact_rate,
        "replay_teacher_forced_character_accuracy": replay_diagnostic.teacher_forced_character_accuracy,
        "replay_teacher_forced_eos_accuracy": replay_diagnostic.teacher_forced_eos_accuracy,
        "replay_greedy_termination_rate": replay_diagnostic.greedy_termination_rate,
    }


def run_generalization_smoke(
    *,
    state_root: Path,
    repo_root: Path,
    checkpoint_record_path: Path,
    run_summary_path: Path,
    steps: int,
    batch_size: int,
    checkpoint_every: int,
    replay_every: int,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], Path]:
    if steps < replay_every or batch_size < 1 or checkpoint_every < 1 or replay_every < 2:
        raise ValueError("steps>=replay_every, positive batch/checkpoint, and replay_every>=2 are required")
    root = state_root.resolve(strict=False)
    record = _checkpoint_record(checkpoint_record_path)
    source_summary = _read_json(run_summary_path)
    if source_summary.get("candidate_generation_id") != record.candidate_generation_id:
        raise ValueError("source summary and checkpoint name different candidate generations")
    architecture_value = source_summary.get("architecture_config")
    if not isinstance(architecture_value, dict):
        raise ValueError("source run summary has no architecture_config")

    curriculum = build_heart_decoder_generalization_curriculum()
    source_schema = source_summary.get("schema")
    if source_schema == HEART_DECODER_MECHANISM_SMOKE_SCHEMA:
        # The checkpoint record itself supplies exact parameter lineage. The
        # separate source curriculum remains preserved in its summary.
        pass
    elif source_schema in (
        HEART_DECODER_GENERALIZATION_SMOKE_SCHEMA,
        HEART_DECODER_GENERALIZATION_RECOVERY_SCHEMA,
    ):
        if source_summary.get("curriculum_id") != curriculum.curriculum_id:
            raise ValueError("source checkpoint used another generalization curriculum")
    else:
        raise ValueError(f"unsupported source run summary schema {source_schema!r}")
    curriculum.write(root)
    objective = HeartDecoderGeneralizationObjective()
    objective.write(root)
    learning_policy = GovernedLearningPolicy(
        optimizer="adamw",
        learning_rate=1e-4,
        weight_decay=0.0,
        scheduler=SchedulerKind.WARMUP_COSINE,
        warmup_steps=min(8, steps),
        min_lr_ratio=0.1,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision=PrecisionMode.FP32,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = torch.device(device)
    model = HeartTranslationCore(HeartTranslationCoreConfig(**architecture_value)).to(resolved_device)
    store = TrainerStateStore.active(state_root=root)
    source_checkpoint = store.load_verified_candidate_checkpoint(record)
    if source_checkpoint.get("descriptor", {}).get("architecture") != HEART_TRANSLATION_ARCHITECTURE:
        raise ValueError("source checkpoint is not the permanent governed D64 Heart architecture")
    model.load_state_dict(source_checkpoint["module_state_dict"], strict=True)

    replay_ids = set(curriculum.replay_case_ids)
    replay_cases = tuple(case for case in curriculum.train_cases if case.case_id in replay_ids)
    novel_cases = tuple(case for case in curriculum.train_cases if case.case_id not in replay_ids)
    audit_novel = tuple(
        min(novel_cases, key=lambda case: abs(len(case.source_text) - target_length))
        for target_length in (32, 128, 257, 448)
    )
    audit_cases = tuple(dict.fromkeys((*audit_novel, replay_cases[0], replay_cases[-1])))
    replay_step_count = steps // replay_every
    novel_step_count = steps - replay_step_count
    novel_batches = iter(deterministic_length_bucketed_batches(
        novel_cases,
        batch_size=batch_size,
        steps=novel_step_count,
        seed=seed,
        page_chars=model.cfg.source_page_chars,
    ))
    replay_batches = iter(deterministic_length_bucketed_batches(
        replay_cases,
        batch_size=batch_size,
        steps=replay_step_count,
        seed=seed + 1,
        page_chars=model.cfg.source_page_chars,
    ))
    training_batches = tuple(
        next(replay_batches) if step_index % replay_every == 0 else next(novel_batches)
        for step_index in range(1, steps + 1)
    )
    module_id = record.module_id
    base_generation_id = record.candidate_generation_id
    experiment_identity = {
        "schema": HEART_DECODER_GENERALIZATION_SMOKE_SCHEMA,
        "source_checkpoint_id": record.checkpoint_id,
        "curriculum_id": curriculum.curriculum_id,
        "objective_id": objective.objective_id,
        "learning_policy_id": learning_policy.policy_id,
        "steps": steps,
        "batch_size": batch_size,
        "checkpoint_every": checkpoint_every,
        "replay_every": replay_every,
        "seed": seed,
    }
    candidate_generation_id = "h64g-" + canonical_sha256(experiment_identity)[:12]
    base_descriptor = ParameterModuleDescriptor(
        module_id=module_id,
        organ_kind=OrganKind.HEART_TRANSLATION_CORE,
        generation_id=base_generation_id,
        architecture=HEART_TRANSLATION_ARCHITECTURE,
        d_model=model.cfg.d_model,
        tags=("heart", "translator", "decoder-generalization-source", "non-serving"),
    )

    control = TrainerControlPlane.active(state_root=root)
    session = None
    try:
        control.declare_expected((base_descriptor,))
        control.register(base_descriptor, model)
        inventory = control.snapshot_inventory(exact_value_hashes=True)
        manifest = inventory.module(module_id)
        grant = ParameterMutationGrant(
            grant_id="heart-decoder-generalization-full-v1",
            module_id=module_id,
            generation_id=base_generation_id,
            policy=ParameterMutationPolicy.FULL,
            max_trainable_parameters=manifest.trainable_parameter_count,
        )
        plan = ParameterMutationPlan(
            base_inventory_id=inventory.inventory_id,
            module_id=module_id,
            base_generation_id=base_generation_id,
            candidate_generation_id=candidate_generation_id,
            tensor_names=tuple(item.name for item in manifest.tensors if item.requires_grad),
            optimizer_name="AdamW",
            learning_rate=learning_policy.learning_rate,
            max_steps=steps,
            source_manifest_ids=(curriculum.train_manifest_id, objective.objective_id),
            holdout_manifest_ids=(curriculum.heldout_manifest_id,),
        )
        preflight = build_heart_training_preflight(
            model=model,
            curriculum=curriculum,
            inventory=inventory,
            plan=plan,
            batch_size=batch_size,
            state_root=root,
            repo_root=repo_root,
        )
        session = control.begin_candidate(
            inventory,
            grant,
            plan,
            preflight_receipt=preflight,
            policy=learning_policy,
        )

        baseline_diagnostic, baseline_evidence = evaluate_heart_decoder_generalization(
            model,
            curriculum,
            descriptor_id=record.parameter_manifest_id,
            checkpoint_id=record.checkpoint_id,
            split="decoder-generalization:heldout:baseline",
        )
        baseline_diagnostic.write(root)
        baseline_evidence.write(root)
        baseline_replay = evaluate_heart_decoder_diagnostics(
            model,
            replay_cases,
            descriptor_id=record.parameter_manifest_id,
            checkpoint_id=record.checkpoint_id,
            curriculum_id=curriculum.curriculum_id,
            split="decoder-generalization:replay:baseline",
        )
        baseline_replay.write(root)
        baseline_audit_loss = _fixed_audit_loss(model, audit_cases, objective, resolved_device)

        losses: list[float] = []
        first_components: dict[str, float] | None = None
        final_components: dict[str, float] | None = None
        checkpoints: list[CandidateCheckpointRecord] = []
        session.candidate_module.train()
        for step_index, cases in enumerate(training_batches, start=1):
            batch = collate_heart_translation_cases(session.candidate_module, cases).to(resolved_device)
            components: dict[str, float] = {}

            def closure(candidate: HeartTranslationCore) -> torch.Tensor:
                loss = heart_decoder_generalization_loss(candidate, batch, objective)
                components.update(
                    {
                        "total": float(loss.total.detach().item()),
                        "translation": float(loss.translation.detach().item()),
                        "diagonal_alignment": float(loss.diagonal_alignment.detach().item()),
                        "copy_route": float(loss.copy_route.detach().item()),
                        "eos_route": float(loss.eos_route.detach().item()),
                        "semantic": float(loss.semantic.detach().item()),
                        "pointers": float(loss.pointers.detach().item()),
                    }
                )
                return loss.total

            receipt = session.step(closure)
            losses.append(receipt.loss)
            if first_components is None:
                first_components = dict(components)
            final_components = dict(components)
            if step_index % checkpoint_every == 0 or step_index == steps:
                checkpoints.append(session.checkpoint(include_optimizer=True))

        if not checkpoints:
            raise RuntimeError("generalization candidate produced no checkpoint")
        checkpoint = checkpoints[-1]
        session.candidate_module.eval()
        final_diagnostic, final_evidence = evaluate_heart_decoder_generalization(
            session.candidate_module,
            curriculum,
            descriptor_id=checkpoint.parameter_manifest_id,
            checkpoint_id=checkpoint.checkpoint_id,
            split="decoder-generalization:heldout:final",
        )
        final_diagnostic.write(root)
        final_evidence.write(root)
        final_replay = evaluate_heart_decoder_diagnostics(
            session.candidate_module,
            replay_cases,
            descriptor_id=checkpoint.parameter_manifest_id,
            checkpoint_id=checkpoint.checkpoint_id,
            curriculum_id=curriculum.curriculum_id,
            split="decoder-generalization:replay:final",
        )
        final_replay.write(root)
        final_audit_loss = _fixed_audit_loss(
            session.candidate_module,
            audit_cases,
            objective,
            resolved_device,
        )

        metrics = _generalization_metrics(final_diagnostic, final_evidence, final_replay)
        observation = EvaluationObservation.from_mapping(
            suite_id=HEART_DECODER_GENERALIZATION_SUITE,
            module_id=module_id,
            candidate_generation_id=candidate_generation_id,
            metrics=metrics,
            artifact_id=final_evidence.evidence_id,
        )
        decision = control.evaluate_gate(
            _generalization_gate(module_id, candidate_generation_id),
            (observation,),
        )
        if decision.passed:
            session.complete(
                reason="decoder identity generalization floor passed; semantic activation intentionally forbidden"
            )
            candidate_status = "generalization_passed_not_activated"
        else:
            session.reject(reason="decoder identity generalization floor failed")
            candidate_status = "generalization_rejected_not_activated"

        run_core = {
            **experiment_identity,
            "module_id": module_id,
            "base_generation_id": base_generation_id,
            "candidate_generation_id": candidate_generation_id,
            "curriculum_id": curriculum.curriculum_id,
            "inventory_id": inventory.inventory_id,
            "preflight_receipt_id": preflight.receipt_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_ids": [item.checkpoint_id for item in checkpoints],
            "baseline_diagnostic_id": baseline_diagnostic.diagnostic_id,
            "baseline_evidence_id": baseline_evidence.evidence_id,
            "final_diagnostic_id": final_diagnostic.diagnostic_id,
            "final_evidence_id": final_evidence.evidence_id,
            "gate_decision_id": decision.decision_id,
            "candidate_status": candidate_status,
            "device": str(resolved_device),
        }
        run_id = canonical_sha256(run_core)
        summary = {
            **run_core,
            "run_id": run_id,
            "architecture_config": model.cfg.to_canonical_dict(),
            "curriculum": curriculum.to_canonical_dict(),
            "training_objective": objective.to_canonical_dict(),
            "learning_policy": learning_policy.to_canonical_dict(),
            "train_case_count": len(curriculum.train_cases),
            "heldout_case_count": len(curriculum.heldout_cases),
            "replay_case_count": len(replay_cases),
            "audit_case_ids": [case.case_id for case in audit_cases],
            "novel_training_step_count": novel_step_count,
            "replay_training_step_count": replay_step_count,
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "loss_improved": final_audit_loss["total"] < baseline_audit_loss["total"],
            "baseline_audit_loss": baseline_audit_loss,
            "final_audit_loss": final_audit_loss,
            "first_loss_components": first_components,
            "final_loss_components": final_components,
            "baseline_diagnostic": baseline_diagnostic.to_canonical_dict(),
            "baseline_evidence": baseline_evidence.to_canonical_dict(),
            "baseline_replay_diagnostic": baseline_replay.to_canonical_dict(),
            "final_diagnostic": final_diagnostic.to_canonical_dict(),
            "final_evidence": final_evidence.to_canonical_dict(),
            "final_replay_diagnostic": final_replay.to_canonical_dict(),
            "gate_decision": decision.to_canonical_dict(),
        }
        path = root / "training" / "heart" / "generalization_runs" / run_id / "summary.json"
        _atomic_json(path, summary)
        return summary, path
    except Exception:
        if session is not None and not session.closed:
            session.reject(reason="decoder generalization smoke aborted by exception")
        raise
    finally:
        control.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=Path("State"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--checkpoint-record", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--replay-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary, path = run_generalization_smoke(
        state_root=args.state_root,
        repo_root=args.repo_root,
        checkpoint_record_path=args.checkpoint_record,
        run_summary_path=args.run_summary,
        steps=args.steps,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        replay_every=args.replay_every,
        seed=args.seed,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "summary": str(path),
                "candidate_status": summary["candidate_status"],
                "first_loss": summary["first_loss"],
                "final_loss": summary["final_loss"],
                "final_diagnostic_id": summary["final_diagnostic_id"],
                "final_evidence_id": summary["final_evidence_id"],
                "gate_decision_id": summary["gate_decision_id"],
                "failed_requirements": summary["gate_decision"]["failed_requirements"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
