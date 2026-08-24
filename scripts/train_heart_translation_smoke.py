"""Run one bounded, Trainer-governed Heart translation candidate smoke.

The live Heart runtime is never modified by this script.  It registers an
untrained non-serving Heart translation base, trains only an isolated Trainer
candidate generation, evaluates that candidate against the disjoint Heart
heldout/counterfactual suites, checkpoints it, and rejects or completes the
candidate according to the cardiac fidelity floor.  It never activates a Heart
translator or opens a cognitive valve.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from runtime.field import canonical_sha256
from runtime.heart import (
    CRITICAL_SEMANTIC_CLASSES,
    HeartEnsemblePolicy,
    HeartTranslatorDescriptor,
    HeartTranslatorRole,
    HeartTranslatorState,
    evaluate_heart_translator_promotion,
)
from runtime.heart.translation_core import (
    HEART_TRANSLATION_ARCHITECTURE,
    HeartTranslationCore,
    HeartTranslationCoreConfig,
    heart_translation_architecture_id,
)
from runtime.trainer import (
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
)
from training.heart_translation import (
    DIALECTS,
    HeartTranslationEvaluationReport,
    HeartTranslationTrainingObjective,
    build_heart_translation_curriculum,
    collate_heart_translation_cases,
    deterministic_training_batches,
    evaluate_heart_translation_model,
    heart_translation_loss,
)
from training.heart_preflight import build_heart_training_preflight


HEART_SMOKE_SCHEMA = "axon-heart-translation-smoke-v3"
HEART_EVALUATION_SUITE = "heart-translation-heldout-v1"


@dataclass(frozen=True, slots=True)
class HeartSmokeResult:
    run_id: str
    state_root: str
    device: str
    steps: int
    batch_size: int
    curriculum_id: str
    architecture_config_id: str
    training_objective_id: str
    train_manifest_id: str
    heldout_manifest_id: str
    module_id: str
    base_generation_id: str
    candidate_generation_id: str
    inventory_id: str
    learning_policy_id: str
    preflight_receipt_id: str
    checkpoint_ids: tuple[str, ...]
    first_loss: float
    final_loss: float
    loss_improved: bool
    baseline_evaluation_id: str
    final_evaluation_id: str
    baseline_grounded_roundtrip: float
    final_grounded_roundtrip: float
    baseline_semantic_fidelity: float
    final_semantic_fidelity: float
    semantic_metric_improved: bool
    heart_promotion_passed: bool
    generic_trainer_gate_passed: bool
    candidate_status: str
    summary_path: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": HEART_SMOKE_SCHEMA,
            "run_id": self.run_id,
            "state_root": self.state_root,
            "device": self.device,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "curriculum_id": self.curriculum_id,
            "architecture_config_id": self.architecture_config_id,
            "training_objective_id": self.training_objective_id,
            "train_manifest_id": self.train_manifest_id,
            "heldout_manifest_id": self.heldout_manifest_id,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "inventory_id": self.inventory_id,
            "learning_policy_id": self.learning_policy_id,
            "preflight_receipt_id": self.preflight_receipt_id,
            "checkpoint_ids": list(self.checkpoint_ids),
            "first_loss": self.first_loss,
            "final_loss": self.final_loss,
            "loss_improved": self.loss_improved,
            "baseline_evaluation_id": self.baseline_evaluation_id,
            "final_evaluation_id": self.final_evaluation_id,
            "baseline_grounded_roundtrip": self.baseline_grounded_roundtrip,
            "final_grounded_roundtrip": self.final_grounded_roundtrip,
            "baseline_semantic_fidelity": self.baseline_semantic_fidelity,
            "final_semantic_fidelity": self.final_semantic_fidelity,
            "semantic_metric_improved": self.semantic_metric_improved,
            "heart_promotion_passed": self.heart_promotion_passed,
            "generic_trainer_gate_passed": self.generic_trainer_gate_passed,
            "candidate_status": self.candidate_status,
            "summary_path": self.summary_path,
        }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
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


def _write_immutable_evaluation(state_root: Path, report: HeartTranslationEvaluationReport) -> Path:
    path = state_root / "training" / "heart" / "evaluations" / f"{report.evaluation_id}.json"
    payload = json.dumps(report.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError("existing Heart evaluation artifact disagrees with immutable content")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _heart_metrics(report: HeartTranslationEvaluationReport) -> dict[str, float]:
    metrics = {
        "grounded_roundtrip": report.evidence.grounded_roundtrip_rate,
        "aggregate_semantic_fidelity": report.evidence.aggregate_semantic_fidelity,
        "counterfactual_use": 1.0 if report.evidence.counterfactual_use_proven else 0.0,
        "regression_failures": float(report.evidence.regression_failures),
        "translation_exact": report.translation_exact_rate,
        "translation_termination": report.translation_termination_rate,
        "source_semantic_exact": report.source_semantic_exact_rate,
        "reverse_semantic_exact": report.reverse_semantic_exact_rate,
        "referent_pointer": report.referent_pointer_rate,
        "grounding_pointer": report.grounding_pointer_rate,
    }
    for name, rate in report.evidence.critical_class_rates:
        metrics[f"critical_{name}"] = rate
    return metrics


def _generic_heart_gate(module_id: str, candidate_generation_id: str, policy: HeartEnsemblePolicy) -> PromotionGate:
    requirements = [
        EvaluationRequirement(
            suite_id=HEART_EVALUATION_SUITE,
            metric_name="grounded_roundtrip",
            comparison=MetricComparison.GREATER_OR_EQUAL,
            threshold=policy.minimum_grounded_roundtrip_rate,
            label="heart grounded semantic roundtrip floor",
        ),
        EvaluationRequirement(
            suite_id=HEART_EVALUATION_SUITE,
            metric_name="aggregate_semantic_fidelity",
            comparison=MetricComparison.GREATER_OR_EQUAL,
            threshold=policy.minimum_aggregate_semantic_fidelity,
            label="heart aggregate semantic fidelity floor",
        ),
        EvaluationRequirement(
            suite_id=HEART_EVALUATION_SUITE,
            metric_name="counterfactual_use",
            comparison=MetricComparison.GREATER_OR_EQUAL,
            threshold=1.0,
            label="heart counterfactual input-use proof",
        ),
        EvaluationRequirement(
            suite_id=HEART_EVALUATION_SUITE,
            metric_name="regression_failures",
            comparison=MetricComparison.LESS_OR_EQUAL,
            threshold=0.0,
            label="heart zero regression failures",
        ),
        EvaluationRequirement(
            suite_id=HEART_EVALUATION_SUITE,
            metric_name="translation_termination",
            comparison=MetricComparison.GREATER_OR_EQUAL,
            threshold=1.0,
            label="heart complete translation termination",
        ),
    ]
    requirements.extend(
        EvaluationRequirement(
            suite_id=HEART_EVALUATION_SUITE,
            metric_name=f"critical_{name}",
            comparison=MetricComparison.GREATER_OR_EQUAL,
            threshold=1.0,
            label=f"heart critical semantic class {name}",
        )
        for name in CRITICAL_SEMANTIC_CLASSES
    )
    return PromotionGate(
        gate_id="heart-translation-cardiac-floor-v1",
        module_id=module_id,
        candidate_generation_id=candidate_generation_id,
        requirements=tuple(requirements),
        required_suite_ids=(HEART_EVALUATION_SUITE,),
    )


def _resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for Heart smoke but torch.cuda.is_available() is false")
    return device


def run_heart_translation_smoke(
    *,
    state_root: Path | str,
    steps: int = 12,
    batch_size: int = 8,
    seed: int = 20260824,
    device: str = "auto",
    model_config: HeartTranslationCoreConfig | None = None,
    training_objective: HeartTranslationTrainingObjective | None = None,
) -> HeartSmokeResult:
    if steps < 2:
        raise ValueError("Heart smoke requires at least two optimizer steps")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    root = Path(state_root)
    resolved_device = _resolve_device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    curriculum = build_heart_translation_curriculum()
    curriculum.write(root)
    objective = HeartTranslationTrainingObjective() if training_objective is None else training_objective
    objective.write(root)
    cfg = HeartTranslationCoreConfig() if model_config is None else model_config
    architecture = heart_translation_architecture_id(cfg)
    architecture_config_id = canonical_sha256(
        {"architecture": architecture, "config": cfg.to_canonical_dict()}
    )
    learning_policy = GovernedLearningPolicy(
        optimizer="adamw",
        learning_rate=3e-4,
        weight_decay=0.01,
        scheduler=SchedulerKind.WARMUP_COSINE,
        warmup_steps=min(2, steps),
        min_lr_ratio=0.2,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision=PrecisionMode.FP32,
    )
    model = HeartTranslationCore(cfg).to(resolved_device)
    # Keep path-facing labels compact for Windows; full lineage is content-addressed.
    module_id = "heart64a"
    base_generation_id = "h64b-" + canonical_sha256(
        {"architecture_config_id": architecture_config_id, "initialization_seed": seed}
    )[:12]
    candidate_generation_id = "h64c-" + canonical_sha256(
        {
            "base_generation_id": base_generation_id,
            "curriculum_id": curriculum.curriculum_id,
            "training_objective_id": objective.objective_id,
            "learning_policy_id": learning_policy.policy_id,
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
        }
    )[:12]
    base_descriptor = ParameterModuleDescriptor(
        module_id=module_id,
        organ_kind=OrganKind.HEART_TRANSLATION_CORE,
        generation_id=base_generation_id,
        architecture=architecture,
        d_model=cfg.d_model,
        tags=("heart", "translator", "non-serving-base"),
    )
    base_heart_descriptor = HeartTranslatorDescriptor(
        translator_id=module_id,
        generation_id=base_generation_id,
        role=HeartTranslatorRole.TRANSLATOR,
        architecture=architecture,
        d_model=cfg.d_model,
        source_dialects=DIALECTS,
        destination_dialects=DIALECTS,
        state=HeartTranslatorState.OFFLINE,
    )
    candidate_heart_descriptor = HeartTranslatorDescriptor(
        translator_id=module_id,
        generation_id=candidate_generation_id,
        role=HeartTranslatorRole.TRANSLATOR,
        architecture=architecture,
        d_model=cfg.d_model,
        source_dialects=DIALECTS,
        destination_dialects=DIALECTS,
        state=HeartTranslatorState.CANDIDATE,
    )

    control = TrainerControlPlane.active(state_root=root)
    session = None
    try:
        control.declare_expected((base_descriptor,))
        control.register(base_descriptor, model)
        inventory = control.snapshot_inventory(exact_value_hashes=True)
        baseline = evaluate_heart_translation_model(
            model,
            curriculum,
            descriptor_id=base_heart_descriptor.descriptor_id,
        )
        _write_immutable_evaluation(root, baseline)

        manifest = inventory.module(module_id)
        tensor_names = tuple(item.name for item in manifest.tensors if item.requires_grad)
        grant = ParameterMutationGrant(
            grant_id="heart-translation-full-smoke-v1",
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
            learning_rate=3e-4,
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
            repo_root=Path(__file__).resolve().parent.parent,
        )
        # learning_policy is constructed before generation identity so the
        # candidate generation is bound to its exact optimizer/schedule recipe.
        session = control.begin_candidate(
            inventory,
            grant,
            plan,
            preflight_receipt=preflight,
            policy=learning_policy,
        )
        # Baseline evaluation intentionally leaves the source model in eval mode;
        # the isolated candidate must explicitly enter training mode before any
        # cuDNN-backed recurrent/attention backward path is exercised.
        session.candidate_module.train()
        batches = deterministic_training_batches(curriculum, batch_size=batch_size, steps=steps, seed=seed)
        losses: list[float] = []
        checkpoint_ids: list[str] = []
        checkpoint_steps = {max(1, steps // 3), max(1, (2 * steps) // 3), steps}
        for step_index, cases in enumerate(batches, start=1):
            batch = collate_heart_translation_cases(session.candidate_module, cases).to(resolved_device)
            receipt = session.step(
                lambda candidate, batch=batch: heart_translation_loss(candidate, batch, objective).total
            )
            losses.append(receipt.loss)
            if step_index in checkpoint_steps:
                checkpoint_ids.append(session.checkpoint(include_optimizer=True).checkpoint_id)

        final_report = evaluate_heart_translation_model(
            session.candidate_module,
            curriculum,
            descriptor_id=candidate_heart_descriptor.descriptor_id,
        )
        evaluation_path = _write_immutable_evaluation(root, final_report)
        heart_policy = HeartEnsemblePolicy()
        heart_decision = evaluate_heart_translator_promotion(
            candidate_heart_descriptor,
            final_report.evidence,
            heart_policy,
        )
        observation = EvaluationObservation.from_mapping(
            suite_id=HEART_EVALUATION_SUITE,
            module_id=module_id,
            candidate_generation_id=candidate_generation_id,
            metrics=_heart_metrics(final_report),
            artifact_id=final_report.evaluation_id,
        )
        generic_decision = control.evaluate_gate(
            _generic_heart_gate(module_id, candidate_generation_id, heart_policy),
            (observation,),
        )
        if generic_decision.passed != heart_decision.passed:
            session.reject(reason="Heart-specific and generic Trainer gates disagreed")
            raise RuntimeError("Heart-specific and generic Trainer promotion decisions disagree")

        if heart_decision.passed:
            session.complete(reason="Heart smoke met cardiac fidelity floor; activation intentionally withheld")
            candidate_status = "completed_not_activated"
        else:
            session.reject(reason="Heart smoke candidate did not meet cardiac fidelity floor")
            candidate_status = "rejected_not_activated"

        semantic_metric_improved = (
            final_report.evidence.aggregate_semantic_fidelity
            > baseline.evidence.aggregate_semantic_fidelity
            or final_report.source_semantic_exact_rate > baseline.source_semantic_exact_rate
        )
        run_core = {
            "schema": HEART_SMOKE_SCHEMA,
            "curriculum_id": curriculum.curriculum_id,
            "architecture_config_id": architecture_config_id,
            "training_objective_id": objective.objective_id,
            "module_id": module_id,
            "base_generation_id": base_generation_id,
            "candidate_generation_id": candidate_generation_id,
            "inventory_id": inventory.inventory_id,
            "learning_policy_id": learning_policy.policy_id,
            "preflight_receipt_id": preflight.receipt_id,
            "steps": steps,
            "batch_size": batch_size,
            "seed": seed,
            "device": str(resolved_device),
            "checkpoint_ids": checkpoint_ids,
            "baseline_evaluation_id": baseline.evaluation_id,
            "final_evaluation_id": final_report.evaluation_id,
            "heart_promotion_passed": heart_decision.passed,
            "generic_trainer_gate_passed": generic_decision.passed,
            "candidate_status": candidate_status,
        }
        run_id = canonical_sha256(run_core)
        summary_path = root / "training" / "heart" / "runs" / run_id / "summary.json"
        result = HeartSmokeResult(
            run_id=run_id,
            state_root=str(root.resolve()),
            device=str(resolved_device),
            steps=steps,
            batch_size=batch_size,
            curriculum_id=curriculum.curriculum_id,
            architecture_config_id=architecture_config_id,
            training_objective_id=objective.objective_id,
            train_manifest_id=curriculum.train_manifest_id,
            heldout_manifest_id=curriculum.heldout_manifest_id,
            module_id=module_id,
            base_generation_id=base_generation_id,
            candidate_generation_id=candidate_generation_id,
            inventory_id=inventory.inventory_id,
            learning_policy_id=learning_policy.policy_id,
            preflight_receipt_id=preflight.receipt_id,
            checkpoint_ids=tuple(checkpoint_ids),
            first_loss=losses[0],
            final_loss=losses[-1],
            loss_improved=losses[-1] < losses[0],
            baseline_evaluation_id=baseline.evaluation_id,
            final_evaluation_id=final_report.evaluation_id,
            baseline_grounded_roundtrip=baseline.evidence.grounded_roundtrip_rate,
            final_grounded_roundtrip=final_report.evidence.grounded_roundtrip_rate,
            baseline_semantic_fidelity=baseline.evidence.aggregate_semantic_fidelity,
            final_semantic_fidelity=final_report.evidence.aggregate_semantic_fidelity,
            semantic_metric_improved=semantic_metric_improved,
            heart_promotion_passed=heart_decision.passed,
            generic_trainer_gate_passed=generic_decision.passed,
            candidate_status=candidate_status,
            summary_path=str(summary_path.resolve()),
        )
        _atomic_json(
            summary_path,
            {
                **result.to_canonical_dict(),
                "architecture_config": cfg.to_canonical_dict(),
                "training_objective": objective.to_canonical_dict(),
                "baseline_evaluation": baseline.to_canonical_dict(),
                "final_evaluation": final_report.to_canonical_dict(),
                "heart_promotion_decision": heart_decision.to_canonical_dict(),
                "trainer_gate_decision": generic_decision.to_canonical_dict(),
                "evaluation_artifact": str(evaluation_path.resolve()),
                "losses": losses,
            },
        )
        return result
    except Exception as exc:
        if session is not None and not session.closed:
            try:
                session.reject(reason=f"Heart smoke aborted by exception: {type(exc).__name__}")
            except Exception:
                # Preserve the original failure; an inability to append the rejection
                # receipt is itself visible in durable Trainer state.
                pass
        raise
    finally:
        control.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_heart_translation_smoke(
        state_root=args.state_root,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(result.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
