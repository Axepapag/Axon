"""Run a bounded, Trainer-governed Heart decoder mechanism overfit shot.

The experiment resumes from a verified rejected Heart checkpoint, mutates only
an isolated candidate, and proves or falsifies exact copy/alignment on a tiny
subset.  Passing this diagnostic never authorizes Heart activation or claims
semantic competence.
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
    HeartTranslationTrainingObjective,
    build_heart_decoder_mechanism_curriculum,
    build_heart_translation_curriculum,
    collate_heart_translation_cases,
    evaluate_heart_decoder_diagnostics,
    heart_translation_loss,
)


HEART_DECODER_MECHANISM_SMOKE_SCHEMA = "axon-heart-decoder-mechanism-smoke-v1"
HEART_TRANSLATION_SMOKE_SCHEMA = "axon-heart-translation-smoke-v4"
HEART_DECODER_MECHANISM_SUITE = "heart-decoder-mechanism-overfit-v1"


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
            raise RuntimeError(f"existing mechanism run summary disagrees with content: {path}")
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


def _mechanism_gate(module_id: str, candidate_generation_id: str) -> PromotionGate:
    requirements = tuple(
        EvaluationRequirement(
            suite_id=HEART_DECODER_MECHANISM_SUITE,
            metric_name=metric,
            comparison=MetricComparison.GREATER_OR_EQUAL,
            threshold=threshold,
            label=label,
        )
        for metric, threshold, label in (
            ("teacher_forced_character_accuracy", 1.0, "exact teacher-forced characters"),
            ("teacher_forced_eos_accuracy", 1.0, "exact teacher-forced EOS"),
            ("teacher_forced_sequence_exact_rate", 1.0, "exact teacher-forced sequence"),
            ("greedy_exact_rate", 1.0, "exact free-running sequence"),
            ("greedy_termination_rate", 1.0, "complete free-running termination"),
            ("copy_route_fraction", 0.5, "copy route carries at least half the mixture"),
            ("target_character_attention_mass", 0.8, "attention aligns to target character"),
        )
    )
    return PromotionGate(
        gate_id="heart-decoder-mechanism-overfit-floor-v1",
        module_id=module_id,
        candidate_generation_id=candidate_generation_id,
        requirements=requirements,
        required_suite_ids=(HEART_DECODER_MECHANISM_SUITE,),
    )


def run_mechanism_smoke(
    *,
    state_root: Path,
    repo_root: Path,
    checkpoint_record_path: Path,
    run_summary_path: Path,
    steps: int,
    train_case_count: int,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], Path]:
    if steps < 1 or train_case_count < 1:
        raise ValueError("steps and train_case_count must be positive")
    root = state_root.resolve(strict=False)
    record = _checkpoint_record(checkpoint_record_path)
    source_summary = _read_json(run_summary_path)
    if source_summary.get("candidate_generation_id") != record.candidate_generation_id:
        raise ValueError("source run summary and checkpoint name different candidate generations")
    architecture_value = source_summary.get("architecture_config")
    if not isinstance(architecture_value, dict):
        raise ValueError("source run summary has no architecture_config")
    curriculum = build_heart_decoder_mechanism_curriculum()
    source_schema = source_summary.get("schema")
    if source_schema == HEART_TRANSLATION_SMOKE_SCHEMA:
        semantic_curriculum = build_heart_translation_curriculum()
        if source_summary.get("curriculum_id") != semantic_curriculum.curriculum_id:
            raise ValueError("source checkpoint did not train on the current governed semantic curriculum")
    elif source_schema == HEART_DECODER_MECHANISM_SMOKE_SCHEMA:
        if source_summary.get("curriculum_id") != curriculum.curriculum_id:
            raise ValueError("source checkpoint used another decoder mechanism curriculum")
    else:
        raise ValueError(f"unsupported source run summary schema {source_schema!r}")
    if train_case_count > len(curriculum.train_cases):
        raise ValueError("train_case_count exceeds the mechanism curriculum")
    curriculum.write(root)
    selected_cases = curriculum.train_cases[:train_case_count]
    objective = HeartTranslationTrainingObjective(
        translation_weight=1.0,
        semantic_weight=0.05,
        pointer_weight=0.05,
    )
    objective.write(root)
    learning_policy = GovernedLearningPolicy(
        optimizer="adamw",
        learning_rate=1e-3,
        weight_decay=0.0,
        scheduler=SchedulerKind.WARMUP_COSINE,
        warmup_steps=min(4, steps),
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

    module_id = record.module_id
    base_generation_id = record.candidate_generation_id
    experiment_identity = {
        "schema": HEART_DECODER_MECHANISM_SMOKE_SCHEMA,
        "source_checkpoint_id": record.checkpoint_id,
        "curriculum_id": curriculum.curriculum_id,
        "selected_case_ids": [case.case_id for case in selected_cases],
        "objective_id": objective.objective_id,
        "learning_policy_id": learning_policy.policy_id,
        "steps": steps,
        "seed": seed,
    }
    candidate_generation_id = "h64m-" + canonical_sha256(experiment_identity)[:12]
    base_descriptor = ParameterModuleDescriptor(
        module_id=module_id,
        organ_kind=OrganKind.HEART_TRANSLATION_CORE,
        generation_id=base_generation_id,
        architecture=HEART_TRANSLATION_ARCHITECTURE,
        d_model=model.cfg.d_model,
        tags=("heart", "translator", "decoder-mechanism-source", "non-serving"),
    )

    control = TrainerControlPlane.active(state_root=root)
    session = None
    try:
        control.declare_expected((base_descriptor,))
        control.register(base_descriptor, model)
        inventory = control.snapshot_inventory(exact_value_hashes=True)
        manifest = inventory.module(module_id)
        tensor_names = tuple(item.name for item in manifest.tensors if item.requires_grad)
        grant = ParameterMutationGrant(
            grant_id="heart-decoder-mechanism-full-v1",
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
            tensor_names=tensor_names,
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
            batch_size=train_case_count,
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
        baseline = evaluate_heart_decoder_diagnostics(
            model,
            selected_cases,
            descriptor_id=record.parameter_manifest_id,
            checkpoint_id=record.checkpoint_id,
            curriculum_id=curriculum.curriculum_id,
            split=f"decoder-mechanism:train:first-{train_case_count}:baseline",
        )
        baseline.write(root)

        batch = collate_heart_translation_cases(session.candidate_module, selected_cases).to(resolved_device)
        session.candidate_module.train()
        losses: list[float] = []
        for _ in range(steps):
            receipt = session.step(
                lambda candidate: heart_translation_loss(candidate, batch, objective).total
            )
            losses.append(receipt.loss)
        checkpoint = session.checkpoint(include_optimizer=True)
        session.candidate_module.eval()
        final_train = evaluate_heart_decoder_diagnostics(
            session.candidate_module,
            selected_cases,
            descriptor_id=checkpoint.parameter_manifest_id,
            checkpoint_id=checkpoint.checkpoint_id,
            curriculum_id=curriculum.curriculum_id,
            split=f"decoder-mechanism:train:first-{train_case_count}:final",
        )
        final_train.write(root)
        final_heldout = evaluate_heart_decoder_diagnostics(
            session.candidate_module,
            curriculum.heldout_cases,
            descriptor_id=checkpoint.parameter_manifest_id,
            checkpoint_id=checkpoint.checkpoint_id,
            curriculum_id=curriculum.curriculum_id,
            split="decoder-mechanism:heldout:final",
        )
        final_heldout.write(root)

        if final_train.mean_copyable_generation_gate is None:
            raise RuntimeError("exact-copy training diagnostic has no copyable characters")
        metrics = {
            "teacher_forced_character_accuracy": final_train.teacher_forced_character_accuracy,
            "teacher_forced_eos_accuracy": final_train.teacher_forced_eos_accuracy,
            "teacher_forced_sequence_exact_rate": final_train.teacher_forced_sequence_exact_rate,
            "greedy_exact_rate": final_train.greedy_exact_rate,
            "greedy_termination_rate": final_train.greedy_termination_rate,
            "copy_route_fraction": 1.0 - final_train.mean_copyable_generation_gate,
            "target_character_attention_mass": final_train.mean_target_character_attention_mass or 0.0,
        }
        observation = EvaluationObservation.from_mapping(
            suite_id=HEART_DECODER_MECHANISM_SUITE,
            module_id=module_id,
            candidate_generation_id=candidate_generation_id,
            metrics=metrics,
            artifact_id=final_train.diagnostic_id,
        )
        decision = control.evaluate_gate(
            _mechanism_gate(module_id, candidate_generation_id),
            (observation,),
        )
        if decision.passed:
            session.complete(
                reason="decoder mechanism overfit floor passed; semantic activation intentionally forbidden"
            )
            candidate_status = "mechanism_passed_not_activated"
        else:
            session.reject(reason="decoder mechanism overfit floor failed")
            candidate_status = "mechanism_rejected_not_activated"

        run_core = {
            **experiment_identity,
            "module_id": module_id,
            "base_generation_id": base_generation_id,
            "candidate_generation_id": candidate_generation_id,
            "inventory_id": inventory.inventory_id,
            "preflight_receipt_id": preflight.receipt_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "baseline_diagnostic_id": baseline.diagnostic_id,
            "final_train_diagnostic_id": final_train.diagnostic_id,
            "final_heldout_diagnostic_id": final_heldout.diagnostic_id,
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
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "loss_improved": losses[-1] < losses[0],
            "baseline_diagnostic": baseline.to_canonical_dict(),
            "final_train_diagnostic": final_train.to_canonical_dict(),
            "final_heldout_diagnostic": final_heldout.to_canonical_dict(),
            "gate_decision": decision.to_canonical_dict(),
        }
        path = root / "training" / "heart" / "mechanism_runs" / run_id / "summary.json"
        _atomic_json(path, summary)
        return summary, path
    except Exception:
        if session is not None and not session.closed:
            session.reject(reason="decoder mechanism smoke aborted by exception")
        raise
    finally:
        control.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=Path("State"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--checkpoint-record", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--train-case-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary, path = run_mechanism_smoke(
        state_root=args.state_root,
        repo_root=args.repo_root,
        checkpoint_record_path=args.checkpoint_record,
        run_summary_path=args.run_summary,
        steps=args.steps,
        train_case_count=args.train_case_count,
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
                "final_train_diagnostic_id": summary["final_train_diagnostic_id"],
                "final_heldout_diagnostic_id": summary["final_heldout_diagnostic_id"],
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
