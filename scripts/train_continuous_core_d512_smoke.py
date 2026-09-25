"""Governed local B4 smoke for the fresh D16-native D512 continuous Core."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from runtime.axon_runtime import CONTINUOUS_CORE_D512_ARCHITECTURE, ContinuousCoreD512
from runtime.field import canonical_json_bytes, canonical_sha256
from runtime.trainer import (
    CandidateCheckpointRecord,
    GovernedLearningPolicy,
    OrganKind,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlanV2,
    ParameterMutationPolicy,
    PrecisionMode,
    ResourceTranche,
    TrainerControlPlane,
    TrancheStore,
)
from training.continuous_core_d512 import (
    build_d512_copy_curriculum,
    build_d512_training_preflight,
    collate_d512_copy_cases,
    d512_copy_loss,
    d512_copy_objective_id,
    deterministic_copy_batches,
    evaluate_d512_copy,
)

D512_SMOKE_SCHEMA = "axon-continuous-core-d512-smoke-v2"


@dataclass(frozen=True, slots=True)
class D512SmokeResult:
    module_id: str
    base_generation_id: str
    candidate_generation_id: str
    plan_id: str
    learning_policy_id: str
    tranche_id: str
    preflight_receipt_id: str
    checkpoint_id: str
    base_global_step: int
    parent_checkpoint_id: str | None
    device: str
    steps: int
    batch_size: int
    parameter_count: int
    baseline: dict[str, Any]
    final: dict[str, Any]
    losses: tuple[float, ...]
    peak_cuda_bytes: int | None
    summary_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": D512_SMOKE_SCHEMA,
            "module_id": self.module_id,
            "base_generation_id": self.base_generation_id,
            "candidate_generation_id": self.candidate_generation_id,
            "plan_id": self.plan_id,
            "learning_policy_id": self.learning_policy_id,
            "tranche_id": self.tranche_id,
            "preflight_receipt_id": self.preflight_receipt_id,
            "checkpoint_id": self.checkpoint_id,
            "base_global_step": self.base_global_step,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "device": self.device,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "parameter_count": self.parameter_count,
            "baseline": self.baseline,
            "final": self.final,
            "losses": list(self.losses),
            "peak_cuda_bytes": self.peak_cuda_bytes,
        }
        if include_id:
            value["summary_id"] = self.summary_id
        return value


def _resolve_device(value: str) -> torch.device:
    value = value.strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return device


def _write_immutable(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() != body:
            raise RuntimeError(f"immutable D512 smoke artifact collision at {path}")
        return path
    path.write_bytes(body)
    return path


def run_d512_smoke(
    *,
    state_root: Path | str = Path(r"D:\Axon\State"),
    steps: int = 24,
    batch_size: int = 24,
    seed: int = 20260924,
    learning_rate: float = 8e-4,
    device: str = "auto",
    resume_latest: bool = False,
) -> D512SmokeResult:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    root = Path(state_root).resolve()
    resolved_device = _resolve_device(device)
    random.seed(seed)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(resolved_device)

    curriculum = build_d512_copy_curriculum()
    curriculum.write(root)
    objective_id = d512_copy_objective_id()
    model = ContinuousCoreD512().to(resolved_device)
    policy = GovernedLearningPolicy(
        optimizer="adamw",
        learning_rate=learning_rate,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        precision=PrecisionMode.FP32,
        objective_program_id=objective_id,
    )
    module_id = "reason512a"
    base_generation_id = "r512b-" + canonical_sha256(
        {
            "architecture": CONTINUOUS_CORE_D512_ARCHITECTURE,
            "config_id": model.cfg.config_id,
            "seed": seed,
        }
    )[:12]
    candidate_generation_id = "r512c-" + canonical_sha256(
        {
            "base_generation_id": base_generation_id,
            "curriculum_id": curriculum.curriculum_id,
            "policy_id": policy.policy_id,
            "objective_id": objective_id,
        }
    )[:12]
    descriptor = ParameterModuleDescriptor(
        module_id=module_id,
        organ_kind=OrganKind.REASONING_CORE,
        generation_id=base_generation_id,
        architecture=CONTINUOUS_CORE_D512_ARCHITECTURE,
        d_model=512,
        tags=("continuous", "d16-bus", "gru", "no-attention", "non-serving-base"),
    )

    control = TrainerControlPlane.active(state_root=root)
    session = None
    try:
        control.declare_expected((descriptor,))
        control.register(descriptor, model)
        inventory = control.snapshot_inventory(exact_value_hashes=True)
        manifest = inventory.module(module_id)
        parameter_count = manifest.parameter_count
        tensor_names = tuple(item.name for item in manifest.tensors if item.requires_grad)
        grant = ParameterMutationGrant(
            grant_id="d512-continuous-full-b4-v1",
            module_id=module_id,
            generation_id=base_generation_id,
            policy=ParameterMutationPolicy.FULL,
            max_trainable_parameters=manifest.trainable_parameter_count,
        )
        plan = ParameterMutationPlanV2(
            base_inventory_id=inventory.inventory_id,
            module_id=module_id,
            base_generation_id=base_generation_id,
            candidate_generation_id=candidate_generation_id,
            tensor_names=tensor_names,
            source_manifest_ids=(curriculum.train_manifest_id, objective_id),
            holdout_manifest_ids=(curriculum.heldout_manifest_id,),
        )
        preflight = build_d512_training_preflight(
            model=model,
            curriculum=curriculum,
            inventory=inventory,
            plan=plan,
            batch_size=batch_size,
            state_root=root,
        )
        parent_checkpoint = None
        base_global_step = 0
        if resume_latest:
            latest_path = (
                root
                / "training"
                / "trainer"
                / "candidates"
                / module_id
                / candidate_generation_id
                / "latest_checkpoint.json"
            )
            if not latest_path.is_file():
                raise RuntimeError(f"no latest checkpoint exists for resume at {latest_path}")
            parent_checkpoint = CandidateCheckpointRecord.from_mapping(
                json.loads(latest_path.read_text(encoding="utf-8"))
            )
            if (
                parent_checkpoint.module_id != module_id
                or parent_checkpoint.candidate_generation_id != candidate_generation_id
            ):
                raise RuntimeError("latest checkpoint belongs to another D512 candidate")
            base_global_step = parent_checkpoint.step

        tranche = ResourceTranche(
            module_id=module_id,
            candidate_generation_id=candidate_generation_id,
            plan_id=plan.plan_id,
            learning_policy_id=policy.policy_id,
            base_global_step=base_global_step,
            steps=steps,
            purpose=(
                "B4 resumed D512 local substrate-literacy tranche"
                if parent_checkpoint is not None
                else "B4 fresh D512 local substrate-literacy smoke"
            ),
        )
        TrancheStore(root / "training" / "trainer").write_tranche(tranche)

        session = control.begin_candidate(
            inventory,
            grant,
            plan,
            preflight_receipt=preflight,
            policy=policy,
            tranche=tranche,
        )
        if parent_checkpoint is not None:
            session.restore_checkpoint(parent_checkpoint)

        session.candidate_module.eval()
        baseline = evaluate_d512_copy(
            session.candidate_module,
            curriculum.heldout_cases,
            device=resolved_device,
        )
        _write_immutable(
            root / "training" / "continuous_core_d512" / "evaluations" / f"{baseline.evaluation_id}.json",
            baseline.to_canonical_dict(),
        )
        print(
            "BASELINE",
            f"base_step={base_global_step}",
            f"loss={baseline.mean_loss:.6f}",
            f"content={baseline.teacher_content_accuracy:.6f}",
            f"eos={baseline.teacher_eos_accuracy:.6f}",
            f"exact={baseline.free_exact_accuracy:.6f}",
            f"content_floor={baseline.strongest_content_constant_floor:.6f}",
            f"all_label_floor={baseline.strongest_constant_floor:.6f}",
            flush=True,
        )

        session.candidate_module.train()
        batches = deterministic_copy_batches(
            curriculum,
            batch_size=batch_size,
            steps=steps,
            seed=seed + 17 + base_global_step,
        )
        losses: list[float] = []
        for step_index, cases in enumerate(batches, start=1):
            cells, targets = collate_d512_copy_cases(cases, device=resolved_device)
            receipt = session.step(lambda candidate, c=cells, t=targets: d512_copy_loss(candidate, c, t))
            losses.append(receipt.loss)
            if step_index == 1 or step_index % 5 == 0 or step_index == steps:
                print(
                    f"STEP {step_index}/{steps}",
                    f"loss={receipt.loss:.6f}",
                    f"lr={receipt.learning_rate:.8f}",
                    flush=True,
                )

        checkpoint = session.checkpoint(include_optimizer=True)
        session.pause(
            reason="B4 opening tranche completed; paused for heldout evaluation",
            checkpoint_id=checkpoint.checkpoint_id,
        )
        session.candidate_module.eval()
        final = evaluate_d512_copy(
            session.candidate_module,
            curriculum.heldout_cases,
            device=resolved_device,
        )
        _write_immutable(
            root / "training" / "continuous_core_d512" / "evaluations" / f"{final.evaluation_id}.json",
            final.to_canonical_dict(),
        )
        peak = None
        if resolved_device.type == "cuda":
            peak = int(torch.cuda.max_memory_allocated(resolved_device))
        result = D512SmokeResult(
            module_id=module_id,
            base_generation_id=base_generation_id,
            candidate_generation_id=candidate_generation_id,
            plan_id=plan.plan_id,
            learning_policy_id=policy.policy_id,
            tranche_id=tranche.tranche_id,
            preflight_receipt_id=preflight.receipt_id,
            checkpoint_id=checkpoint.checkpoint_id,
            base_global_step=base_global_step,
            parent_checkpoint_id=(None if parent_checkpoint is None else parent_checkpoint.checkpoint_id),
            device=str(resolved_device),
            steps=steps,
            batch_size=batch_size,
            parameter_count=parameter_count,
            baseline=baseline.to_canonical_dict(),
            final=final.to_canonical_dict(),
            losses=tuple(losses),
            peak_cuda_bytes=peak,
        )
        summary_path = root / "training" / "continuous_core_d512" / "smokes" / f"{result.summary_id}.json"
        _write_immutable(summary_path, result.to_canonical_dict())
        print(
            "FINAL",
            f"loss={final.mean_loss:.6f}",
            f"content={final.teacher_content_accuracy:.6f}",
            f"eos={final.teacher_eos_accuracy:.6f}",
            f"exact={final.free_exact_accuracy:.6f}",
            f"termination={final.termination_accuracy:.6f}",
            f"valid_unicode={final.valid_unicode_accuracy:.6f}",
            f"content_floor={final.strongest_content_constant_floor:.6f}",
            f"all_label_floor={final.strongest_constant_floor:.6f}",
            f"checkpoint={checkpoint.checkpoint_id}",
            f"summary={summary_path}",
            flush=True,
        )
        if peak is not None:
            print(f"PEAK_CUDA_BYTES {peak}", flush=True)
        return result
    except Exception:
        if session is not None and not session.closed:
            try:
                session.reject(reason="B4 smoke aborted by exception")
            except Exception:
                pass
        raise
    finally:
        control.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", default=r"D:\Axon\State")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-latest", action="store_true")
    args = parser.parse_args()
    result = run_d512_smoke(
        state_root=Path(args.state_root),
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        device=args.device,
        resume_latest=args.resume_latest,
    )
    print(json.dumps(result.to_canonical_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
