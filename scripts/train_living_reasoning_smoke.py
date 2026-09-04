"""Governed bounded smoke campaign for the first Soul-conditioned D64 core."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from runtime.field import CanonicalStateBranch, LogicalRegion, canonical_sha256
from runtime.soul import SoulStore
from runtime.trainer import (
    CandidateSoulWorkspace,
    CandidateStepBundleCoordinator,
    GovernedLearningPolicy,
    OrganKind,
    ParameterModuleDescriptor,
    ParameterMutationGrant,
    ParameterMutationPlan,
    ParameterMutationPlanV2,
    ParameterMutationPolicy,
    ResourceTranche,
    TrainerControlPlane,
    TrainingProgressJournal,
    TrancheContinuation,
    TrancheStore,
)
from training import (
    FOUNDATION_SEQUENCE_GATE_POLICY_ID,
    LivingReasoningCoreD64,
    LivingReasoningCurriculum,
    TeachingEligibility,
    build_living_reasoning_preflight,
    build_living_reasoning_smoke_curriculum,
    candidate_a_config,
    d64_tournament_metric_computation,
    decide_foundation_sequence_mastery,
    evaluate_living_episode,
    evaluate_sequential_case,
    foundation_sequence_probe,
    is_foundation_sequence_episode,
    living_episode_objective,
    living_source_counterfactuals,
    load_first_form_curriculum,
    load_sequential_first_form,
    sequential_living_objective,
)

ROOT = Path(__file__).resolve().parent.parent


def _write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    """Publish one immutable JSON artifact or verify an identical replay."""

    body = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != body:
            raise RuntimeError(f"immutable report artifact disagrees at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prior_consumed_tranche_id(
    *,
    campaign_report_dir: Path,
    module_id: str,
    candidate_generation_id: str,
    latest_bundle: Any,
    tranche_store: TrancheStore,
    plan_id: str,
    learning_policy_id: str,
) -> str | None:
    """Prove which resource tranche, if any, produced an accepted parent.

    A tranche record is only an issued allowance. It becomes lineage only when
    an immutable segment report binds it to the exact final checkpoint and
    accepted step bundle. This prevents an abandoned allowance from being
    mistaken for consumed history.
    """

    matching_issued = tuple(
        item
        for item in tranche_store.tranches_for(module_id, candidate_generation_id)
        if item.plan_id == plan_id
        and item.learning_policy_id == learning_policy_id
        and item.final_global_step == latest_bundle.step
    )
    consumed: set[str] = set()
    for path in sorted(campaign_report_dir.glob("segment_*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        observed_report_id = body.get("report_id")
        if (
            observed_report_id is None
            or canonical_sha256({key: value for key, value in body.items() if key != "report_id"}) != observed_report_id
        ):
            raise RuntimeError(f"immutable segment report identity mismatch: {path}")
        if (
            body.get("evaluation_only")
            or body.get("candidate_generation_id") != candidate_generation_id
            or body.get("segment_end_step") != latest_bundle.step
            or body.get("final_checkpoint_id") != latest_bundle.checkpoint_id
        ):
            continue
        accepted_bundle_ids = tuple(
            item.get("accepted_step_bundle_id")
            for item in body.get("steps", ())
            if item.get("accepted_step_bundle_id") is not None
        )
        if not accepted_bundle_ids or accepted_bundle_ids[-1] != latest_bundle.bundle_id:
            continue
        resource = body.get("resource_tranche")
        if resource is None:
            continue
        reported = ResourceTranche.from_mapping(resource)
        durable = tranche_store.read_tranche(reported.tranche_id)
        if durable.to_canonical_dict() != reported.to_canonical_dict():
            raise RuntimeError("segment report resource tranche differs from durable record")
        consumed.add(reported.tranche_id)
    if len(consumed) > 1:
        raise RuntimeError("multiple consumed tranches claim the accepted parent bundle")
    if consumed:
        return next(iter(consumed))
    if matching_issued:
        raise RuntimeError(
            "accepted parent coincides with an issued but unclosed resource tranche; "
            "regenerate its evaluation/report before issuing a continuation"
        )
    return None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2,
        help="historical v1 plan envelope; ignored by resource-independent v2 identity",
    )
    parser.add_argument(
        "--legacy-plan-v1",
        action="store_true",
        help="preserve or resume a historical max_steps-bound candidate; new tissue defaults to v2",
    )
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
        "--candidate-label",
        default=None,
        help=(
            "stable core identity for an isolated tournament lane; omission preserves "
            "the pre-tournament synthetic-smoke lineage"
        ),
    )
    parser.add_argument(
        "--curriculum-manifest",
        type=Path,
        action="append",
        help=(
            "immutable FFCS manifest (standard or sequential schema); repeatable; "
            "omit only for the synthetic mechanism smoke"
        ),
    )
    parser.add_argument(
        "--evaluation-case-limit",
        type=int,
        default=None,
        help="bounded complete heldout cases for this diagnostic; deferred cases are counted",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest accepted parameter+Soul step bundle",
    )
    parser.add_argument(
        "--tranche-steps",
        type=int,
        default=None,
        help=(
            "renewable resource tranche: optimizer steps granted to this segment "
            "beyond its exact base step. Fresh v2 tissue starts at base zero; "
            "continuation requires --resume with an accepted parent bundle. "
            "The allowance never changes plan or candidate identity."
        ),
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "evaluation/report regeneration only: no optimizer mutation, no Soul "
            "transition, no accepted step. Requires an existing accepted bundle."
        ),
    )
    parser.add_argument(
        "--progress-dir",
        type=Path,
        default=None,
        help="durable JSONL/current.json observability directory (also mirrors to stdout)",
    )
    parser.add_argument(
        "--external-job-id",
        default=None,
        help="provider-neutral durable job identity used only for progress correlation",
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


def _training_lanes(
    mechanism_curriculum: LivingReasoningCurriculum,
    standard_ffcs: list[Any],
    sequential_ffcs: list[Any],
) -> tuple[tuple[str, tuple[tuple[str, Any, str], ...]], ...]:
    """Build deterministic family lanes without flattening sequential cases."""

    lanes: list[tuple[str, tuple[tuple[str, Any, str], ...]]] = [
        (
            "mechanism",
            tuple(
                ("episode", episode, mechanism_curriculum.train_manifest_id)
                for episode in mechanism_curriculum.split("train")
            ),
        )
    ]
    for item in standard_ffcs:
        teaching_manifest_id = item.teaching_living_curriculum.train_manifest_id
        for family, _counts in item.requested_family_split_counts:
            lanes.append(
                (
                    f"ffcs-{family}",
                    tuple(
                        (
                            "first_form_case",
                            case,
                            teaching_manifest_id,
                        )
                        for case in item.teaching_cases
                        if case.family == family and case.episode.split == "train"
                    ),
                )
            )
    for item in sequential_ffcs:
        lanes.append(
            (
                "ffcs-E",
                tuple(("sequential", case, item.train_manifest_id) for case in item.split("train")),
            )
        )
    return tuple((name, rows) for name, rows in lanes if rows)


def _scheduled_material(
    lanes: tuple[tuple[str, tuple[tuple[str, Any, str], ...]], ...],
    step: int,
) -> tuple[str, str, Any, str]:
    """Select one material item by global-step family round robin."""

    if not lanes:
        raise ValueError("governed campaign has no supervised training material")
    lane_index = step % len(lanes)
    lane_name, lane = lanes[lane_index]
    lane_cycle = step // len(lanes)
    kind, material, source_manifest_id = lane[lane_cycle % len(lane)]
    return lane_name, kind, material, source_manifest_id


def _publish_campaign_curriculum(
    state_root: Path,
    body: dict[str, Any],
) -> tuple[str, Path]:
    """Publish one immutable identity for the complete mixed curriculum."""

    manifest_id = canonical_sha256(body)
    path = state_root.resolve() / "training" / "reasoning" / "campaign_curricula" / manifest_id / "manifest.json"
    document = {**body, "manifest_id": manifest_id}
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != document:
            raise RuntimeError("campaign curriculum manifest identity collision")
        return manifest_id, path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"manifest.{os.getpid()}.tmp")
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return manifest_id, path


def _publish_campaign_split_scope(
    campaign_path: Path,
    campaign_curriculum_id: str,
    split: str,
) -> tuple[str, Path]:
    body = {
        "schema": "axon-d64-reasoning-campaign-split-v1",
        "campaign_curriculum_id": campaign_curriculum_id,
        "split": split,
    }
    scope_id = canonical_sha256(body)
    path = campaign_path.parent / "splits" / f"{split}.json"
    document = {**body, "manifest_id": scope_id}
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != document:
            raise RuntimeError("campaign split manifest identity collision")
        return scope_id, path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{split}.{os.getpid()}.tmp")
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return scope_id, path


def _material_objective(
    candidate: LivingReasoningCoreD64,
    *,
    kind: str,
    material: Any,
    soul: Any,
    core_id: str,
    parameter_generation: str,
) -> tuple[torch.Tensor, Any, tuple[dict[str, float], ...], tuple[Any, ...]]:
    """Run one scheduled lesson and return its complete Soul lineage."""

    if kind == "sequential":
        loss, unrolls, _final_soul = sequential_living_objective(
            candidate,
            material,
            soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
        )
        return (
            loss,
            unrolls[-1],
            (),
            tuple(transition for tick_unroll in unrolls for transition in tick_unroll.transitions),
        )
    if kind == "first_form_case":
        if material.eligibility is not TeachingEligibility.VERIFIED_TARGET:
            raise ValueError(
                "only VERIFIED_TARGET first-form cases may enter optimizer loss"
            )
        material = material.episode
        kind = "episode"
    if kind != "episode":
        raise ValueError(f"unsupported scheduled material kind {kind!r}")
    loss, unroll, phase_metrics = living_episode_objective(
        candidate,
        material,
        soul,
        core_id=core_id,
        parameter_generation=parameter_generation,
    )
    return loss, unroll, phase_metrics, unroll.transitions


def main() -> int:
    args = _arguments()
    if args.external_job_id is not None and not str(args.external_job_id).strip():
        raise ValueError("--external-job-id must be non-empty when supplied")
    progress = (
        None
        if args.progress_dir is None
        else TrainingProgressJournal(
            args.progress_dir,
            job_id=args.external_job_id or f"local-{os.getpid()}",
        )
    )
    if args.legacy_plan_v1 and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be positive")
    if args.run_steps is not None and args.run_steps < 1:
        raise ValueError("--run-steps must be positive when supplied")
    if args.evaluation_case_limit is not None and args.evaluation_case_limit < 1:
        raise ValueError("--evaluation-case-limit must be positive when supplied")
    if args.tranche_steps is not None and args.tranche_steps < 1:
        raise ValueError("--tranche-steps must be positive when supplied")
    if args.legacy_plan_v1 and args.tranche_steps is not None and not args.resume:
        raise ValueError("--tranche-steps requires --resume with an accepted parent bundle")
    if args.tranche_steps is not None and args.run_steps is not None:
        raise ValueError(
            "--tranche-steps is already the complete segment allowance; do not combine it with --run-steps"
        )
    if args.evaluate_only and not args.resume:
        raise ValueError("--evaluate-only requires --resume with an accepted bundle")
    if args.evaluate_only and args.tranche_steps is not None:
        raise ValueError("--evaluate-only cannot be combined with --tranche-steps")
    if not args.legacy_plan_v1 and not args.evaluate_only and args.tranche_steps is None:
        raise ValueError(
            "resource-independent v2 training requires --tranche-steps; "
            "resource allowance is never part of candidate identity"
        )
    if args.candidate_label is not None and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", args.candidate_label) is None:
        raise ValueError("--candidate-label must be a short lowercase slug")
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
    standard_ffcs = []
    sequential_ffcs = []
    for manifest_path in args.curriculum_manifest or ():
        manifest_body = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_schema = manifest_body.get("schema")
        if manifest_schema == "axon-first-form-curriculum-v1":
            standard_ffcs.append(load_first_form_curriculum(manifest_path))
        elif manifest_schema == "axon-first-form-sequential-curriculum-v1":
            sequential_ffcs.append(load_sequential_first_form(manifest_path))
        else:
            raise RuntimeError(
                f"unsupported curriculum manifest schema {manifest_schema!r}; expected "
                "axon-first-form-curriculum-v1 or "
                "axon-first-form-sequential-curriculum-v1"
            )
    standard_ffcs = sorted(standard_ffcs, key=lambda item: item.manifest_id)
    sequential_ffcs = sorted(sequential_ffcs, key=lambda item: item.manifest_id)
    all_ffcs = [*standard_ffcs, *sequential_ffcs]
    ffcs_manifest_ids = tuple(item.manifest_id for item in all_ffcs)
    if len(set(ffcs_manifest_ids)) != len(ffcs_manifest_ids):
        raise RuntimeError("duplicate FFCS manifest identity")
    manifest_identity_hashes = tuple(item.identity_text_sha256 for item in all_ffcs)
    mechanism_curriculum = build_living_reasoning_smoke_curriculum()
    curriculum = mechanism_curriculum
    teaching_eligibility = [
        {
            "manifest_id": item.manifest_id,
            "total_case_count": len(item.cases),
            "teaching_case_count": len(item.teaching_cases),
            "excluded_from_exact_supervision_count": (
                len(item.cases) - len(item.teaching_cases)
            ),
            "eligibility_counts": dict(item.eligibility_counts),
            "teaching_living_curriculum_id": (
                item.teaching_living_curriculum.curriculum_id
            ),
        }
        for item in standard_ffcs
    ]
    evaluation_family_by_episode = {
        case.episode.episode_id: case.family
        for item in standard_ffcs
        for case in item.teaching_cases
    }
    evaluation_manifest_by_episode = {
        case.episode.episode_id: item.manifest_id
        for item in standard_ffcs
        for case in item.teaching_cases
    }
    foundation_sequence_enabled = any(
        is_foundation_sequence_episode(case.episode)
        for item in standard_ffcs
        for case in item.teaching_cases
    )
    if all_ffcs:
        active = CanonicalStateBranch.active_runtime(state_root=args.state_root).load_head()
        active_identity = active.region(LogicalRegion.IDENTITY).text
        active_hash = hashlib.sha256(active_identity.encode("utf-8")).hexdigest()
        if any(observed != active_hash for observed in manifest_identity_hashes):
            raise RuntimeError("FFCS manifest Identity is stale for active canonical state")
        curriculum = LivingReasoningCurriculum(
            tuple(
                episode
                for item in standard_ffcs
                for episode in item.teaching_living_curriculum.episodes
            )
            + mechanism_curriculum.episodes
        )
    primary_curriculum = (
        LivingReasoningCurriculum(
            tuple(
                episode
                for item in standard_ffcs
                for episode in item.teaching_living_curriculum.episodes
            )
        )
        if standard_ffcs
        else mechanism_curriculum
    )
    campaign_curriculum_id, campaign_curriculum_path = _publish_campaign_curriculum(
        args.state_root,
        {
            "schema": "axon-d64-reasoning-campaign-curriculum-v1",
            "standard_ffcs_manifest_ids": [item.manifest_id for item in standard_ffcs],
            "sequential_ffcs_manifest_ids": [item.manifest_id for item in sequential_ffcs],
            "teaching_views": teaching_eligibility,
            "primary_evaluation_curriculum_id": primary_curriculum.curriculum_id,
            "primary_evaluation_scope": (
                "verified_target_standard_curricula_only"
                if standard_ffcs
                else "synthetic_mechanism_only"
            ),
            "mechanism_curriculum_id": mechanism_curriculum.curriculum_id,
            "mechanism_train_manifest_id": mechanism_curriculum.train_manifest_id,
            "mechanism_heldout_manifest_id": mechanism_curriculum.heldout_manifest_id,
            "scheduler": "family-round-robin-v1",
            "foundation_sequence_gate_policy_id": (
                FOUNDATION_SEQUENCE_GATE_POLICY_ID if foundation_sequence_enabled else None
            ),
            "sequential_cases_remain_grouped": True,
            "content_limit": None,
        },
    )
    campaign_train_scope_id, campaign_train_scope_path = _publish_campaign_split_scope(
        campaign_curriculum_path,
        campaign_curriculum_id,
        "train",
    )
    campaign_heldout_scope_id, campaign_heldout_scope_path = _publish_campaign_split_scope(
        campaign_curriculum_path,
        campaign_curriculum_id,
        "heldout",
    )
    if args.candidate_label is None:
        module_id = "reasoning-d64-candidate-a"
        base_generation = "reasoning-d64-untrained-base-v1"
        candidate_label = "legacy-candidate-a"
    else:
        candidate_label = args.candidate_label
        module_id = f"reasoning-d64-{candidate_label}"
        base_generation = f"{module_id}-untrained-base-v1"
    candidate_identity = {
        "candidate_label": candidate_label,
        "architecture_id": config.architecture_id,
        "seed": args.seed,
        "curriculum_id": curriculum.curriculum_id,
        "campaign_curriculum_id": campaign_curriculum_id,
    }
    if args.legacy_plan_v1:
        legacy_identity = dict(candidate_identity)
        if args.candidate_label is None:
            legacy_identity.pop("candidate_label")
        candidate_generation = ("r64a-smoke-" if args.candidate_label is None else "r64t-") + canonical_sha256(
            {
                **legacy_identity,
                "max_steps": args.max_steps,
                "learning_rate": args.learning_rate,
                "checkpoint_interval": args.checkpoint_interval,
            }
        )[:16]
    else:
        candidate_generation = "r64v2-" + canonical_sha256(candidate_identity)[:16]
    if progress is not None:
        progress.emit(
            "starting",
            candidate_generation_id=candidate_generation,
            candidate_label=candidate_label,
            device=str(device),
            accelerator=device.type,
            tranche_steps=args.tranche_steps,
            evaluate_only=bool(args.evaluate_only),
            curriculum_manifest_count=len(all_ffcs),
        )
    descriptor = ParameterModuleDescriptor(
        module_id=module_id,
        organ_kind=OrganKind.REASONING_CORE,
        generation_id=base_generation,
        architecture=config.architecture_id,
        d_model=64,
        tags=("living", "private-soul", "complete-field", candidate_label),
    )

    report: dict[str, Any] = {
        "schema": "axon-living-reasoning-smoke-report-v1",
        "device": str(device),
        "seed": args.seed,
        "preflight_only": bool(args.preflight_only),
        "architecture": model.architecture_report(),
        "curriculum_id": curriculum.curriculum_id,
        "campaign_curriculum_id": campaign_curriculum_id,
        "campaign_curriculum_path": str(campaign_curriculum_path),
        "campaign_train_scope_id": campaign_train_scope_id,
        "campaign_heldout_scope_id": campaign_heldout_scope_id,
        "campaign_train_scope_path": str(campaign_train_scope_path),
        "campaign_heldout_scope_path": str(campaign_heldout_scope_path),
        "train_manifest_id": curriculum.train_manifest_id,
        "heldout_manifest_id": curriculum.heldout_manifest_id,
        "candidate_label": candidate_label,
        "ffcs_manifest_ids": list(ffcs_manifest_ids),
        "standard_ffcs_manifest_ids": [item.manifest_id for item in standard_ffcs],
        "sequential_ffcs_manifest_ids": [item.manifest_id for item in sequential_ffcs],
        "ffcs_manifest_id": (standard_ffcs[0].manifest_id if len(standard_ffcs) == 1 else None),
        "sequential_ffcs_manifest_id": (sequential_ffcs[0].manifest_id if len(sequential_ffcs) == 1 else None),
        "sequential_case_count": sum(len(item.cases) for item in sequential_ffcs),
        "teaching_eligibility": teaching_eligibility,
        "primary_evaluation_curriculum_id": primary_curriculum.curriculum_id,
        "primary_evaluation_scope": (
            "verified_target_standard_curricula_only"
            if standard_ffcs
            else "synthetic_mechanism_only"
        ),
        "mechanism_curriculum_id": mechanism_curriculum.curriculum_id,
        "curriculum_composition": (
            "synthetic_mechanism_only" if not all_ffcs else "governed_ffcs_campaign_plus_synthetic_mechanism"
        ),
    }
    with TrainerControlPlane.active(state_root=args.state_root) as control:
        control.declare_expected((descriptor,))
        control.register(descriptor, model)
        inventory = control.snapshot_inventory(exact_value_hashes=True)
        manifest = inventory.module(module_id)
        tensor_names = tuple(item.name for item in manifest.tensors if item.requires_grad)
        source_manifest_ids = tuple(
            [campaign_train_scope_id, mechanism_curriculum.train_manifest_id]
            + [
                item.teaching_living_curriculum.train_manifest_id
                for item in standard_ffcs
            ]
            + [item.train_manifest_id for item in sequential_ffcs]
        )
        holdout_manifest_ids = tuple(
            [campaign_heldout_scope_id, mechanism_curriculum.heldout_manifest_id]
            + [
                item.teaching_living_curriculum.heldout_manifest_id
                for item in standard_ffcs
            ]
            + [item.heldout_manifest_id for item in sequential_ffcs]
        )
        if args.legacy_plan_v1:
            plan = ParameterMutationPlan(
                base_inventory_id=inventory.inventory_id,
                module_id=module_id,
                base_generation_id=base_generation,
                candidate_generation_id=candidate_generation,
                tensor_names=tensor_names,
                optimizer_name="AdamW",
                learning_rate=args.learning_rate,
                max_steps=args.max_steps,
                source_manifest_ids=source_manifest_ids,
                holdout_manifest_ids=holdout_manifest_ids,
            )
        else:
            plan = ParameterMutationPlanV2(
                base_inventory_id=inventory.inventory_id,
                module_id=module_id,
                base_generation_id=base_generation,
                candidate_generation_id=candidate_generation,
                tensor_names=tensor_names,
                source_manifest_ids=source_manifest_ids,
                holdout_manifest_ids=holdout_manifest_ids,
            )
        policy = GovernedLearningPolicy(
            optimizer="adamw",
            learning_rate=args.learning_rate,
        )
        step_bundles = CandidateStepBundleCoordinator(args.state_root)
        latest_bundle = step_bundles.latest_bundle(module_id, candidate_generation)
        campaign_report_dir = args.state_root.resolve() / "training" / "reasoning" / candidate_generation
        tranche_store = TrancheStore(args.state_root / "training" / "trainer")
        tranche = None
        prior_tranche_id = None
        if args.tranche_steps is not None:
            if latest_bundle is not None and not args.resume:
                raise RuntimeError("candidate already has accepted work; continuation requires --resume")
            if latest_bundle is None and args.resume:
                raise RuntimeError("--resume requested but this candidate has no accepted parent bundle")
            if latest_bundle is not None:
                prior_tranche_id = _prior_consumed_tranche_id(
                    campaign_report_dir=campaign_report_dir,
                    module_id=module_id,
                    candidate_generation_id=candidate_generation,
                    latest_bundle=latest_bundle,
                    tranche_store=tranche_store,
                    plan_id=plan.plan_id,
                    learning_policy_id=policy.policy_id,
                )
            base_global_step = 0 if latest_bundle is None else latest_bundle.step
            parent_bundle_id = None if latest_bundle is None else latest_bundle.bundle_id
            tranche = ResourceTranche(
                module_id=module_id,
                candidate_generation_id=candidate_generation,
                plan_id=plan.plan_id,
                learning_policy_id=policy.policy_id,
                base_global_step=base_global_step,
                steps=args.tranche_steps,
                parent_bundle_id=parent_bundle_id,
                purpose=(
                    "renewable base-zero training tranche"
                    if parent_bundle_id is None
                    else f"renewable continuation tranche (parent bundle {parent_bundle_id[:16]})"
                ),
            )
            tranche_store.write_tranche(tranche)
            report["resource_tranche"] = tranche.to_canonical_dict()
        elif args.evaluate_only and latest_bundle is None:
            raise RuntimeError("--evaluate-only requires an existing accepted bundle; none exists")
        preflight = build_living_reasoning_preflight(
            model=model,
            curriculum=curriculum,
            inventory=inventory,
            plan=plan,
            state_root=args.state_root,
            repo_root=ROOT,
            sequential_curricula=sequential_ffcs,
            resource_tranche=tranche,
            evaluation_only=args.evaluate_only,
        )
        report.update(
            {
                "inventory_id": inventory.inventory_id,
                "plan_id": plan.plan_id,
                "plan_schema": plan.to_canonical_dict()["schema"],
                "learning_policy_id": policy.policy_id,
                "source_manifest_ids": list(plan.source_manifest_ids),
                "holdout_manifest_ids": list(plan.holdout_manifest_ids),
                "preflight_receipt_id": preflight.receipt_id,
                "preflight_passed": preflight.passed,
            }
        )
        if not preflight.passed:
            raise RuntimeError("living reasoning preflight failed; optimizer creation denied")
        if args.preflight_only:
            if progress is not None:
                progress.emit(
                    "completed",
                    preflight_only=True,
                    preflight_receipt_id=preflight.receipt_id,
                )
            print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
            return 0

        if (
            args.legacy_plan_v1
            and latest_bundle is not None
            and latest_bundle.step >= args.max_steps
            and tranche is None
            and not args.evaluate_only
        ):
            if not args.resume:
                raise RuntimeError("candidate campaign is complete; pass --resume for idempotent report recovery")
            prior_reports = sorted(campaign_report_dir.glob("segment_*.json"))
            if not prior_reports:
                raise RuntimeError("complete candidate has no immutable segment report")
            prior_path = prior_reports[-1]
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            observed_report_id = prior.get("report_id")
            report_body = {key: value for key, value in prior.items() if key != "report_id"}
            if (
                canonical_sha256(report_body) != observed_report_id
                or observed_report_id is None
                or prior.get("candidate_generation_id") != candidate_generation
                or prior.get("final_checkpoint_id") != latest_bundle.checkpoint_id
                or prior.get("segment_end_step") != latest_bundle.step
            ):
                raise RuntimeError("complete candidate report does not bind the accepted bundle")
            print(
                json.dumps(
                    {**prior, "report_path": str(prior_path)},
                    ensure_ascii=True,
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0

        grant = ParameterMutationGrant(
            grant_id=f"living-reasoning-smoke:{plan.plan_id}",
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
            policy=policy,
            tranche=tranche,
        )

        live_souls = SoulStore.active(args.state_root)
        live_souls.ensure_core(
            core_id=module_id,
            architecture_id=config.architecture_id,
            parameter_generation=base_generation,
        )
        soul_workspace = CandidateSoulWorkspace(args.state_root)
        sequential_train_cases = tuple(case for item in sequential_ffcs for case in item.split("train"))
        sequential_heldout_cases = tuple(case for item in sequential_ffcs for case in item.split("heldout"))
        sequential_regression_cases = tuple(case for item in sequential_ffcs for case in item.split("regression"))
        all_regression_episodes = primary_curriculum.split("regression")
        regression_episodes = (
            all_regression_episodes
            if args.evaluation_case_limit is None
            else all_regression_episodes[: args.evaluation_case_limit]
        )
        all_sequential_regression = sequential_regression_cases
        sequential_regression = (
            all_sequential_regression
            if args.evaluation_case_limit is None
            else all_sequential_regression[: args.evaluation_case_limit]
        )
        sequential_trajectory_ids = tuple(item.case_id for item in sequential_train_cases)
        soul_manifest = soul_workspace.prepare(
            candidate_id=candidate_generation,
            core_id=module_id,
            runtime_episode_session_id=(f"living-campaign-curriculum:{campaign_curriculum_id}"),
            whole_episode_split="train",
            soul_trajectory_ids=tuple(item.episode_id for item in curriculum.split("train"))
            + sequential_trajectory_ids,
            candidate_parameter_generation=candidate_generation,
        )
        soul_branch = soul_workspace.branch(candidate_generation, module_id)
        recovered_bundles = step_bundles.recover_pending(module_id, candidate_generation)
        latest_bundle = step_bundles.latest_bundle(module_id, candidate_generation)
        if latest_bundle is not None:
            if not args.resume:
                raise RuntimeError(
                    "candidate already has accepted work; pass --resume or choose a different governed campaign"
                )
            if tranche is not None and latest_bundle.bundle_id != tranche.parent_bundle_id:
                raise RuntimeError("accepted parent advanced during recovery; issue a fresh resource tranche")
            parent_checkpoint = step_bundles.checkpoint_for_bundle(latest_bundle)
            session.restore_checkpoint(parent_checkpoint)
            if soul_branch.load_head().soul_id != latest_bundle.after_soul_id:
                raise RuntimeError("accepted checkpoint and candidate Soul HEAD disagree")
            if tranche is not None:
                continuation = TrancheContinuation(
                    tranche_id=tranche.tranche_id,
                    module_id=module_id,
                    candidate_generation_id=candidate_generation,
                    plan_id=plan.plan_id,
                    learning_policy_id=policy.policy_id,
                    parent_bundle_id=latest_bundle.bundle_id,
                    parent_checkpoint_id=latest_bundle.checkpoint_id,
                    parent_optimizer_receipt_id=latest_bundle.optimization_receipt_id,
                    parent_soul_id=latest_bundle.after_soul_id,
                    parent_global_step=latest_bundle.step,
                    prior_tranche_id=prior_tranche_id,
                )
                tranche_store.write_continuation(continuation)
                report["tranche_continuation"] = continuation.to_canonical_dict()
        elif args.resume:
            report["resume_note"] = (
                "no accepted bundle existed; unaccepted optimizer/checkpoint orphans, if any, "
                "remain evidence and the candidate restarts from its governed base"
            )
        steps: list[dict[str, Any]] = []
        checkpoints = []
        # Learned-capability gates use the evidence-qualified campaign surface.
        # Synthetic mechanism material remains a training/regression lane, but
        # cannot inflate or contaminate C1/Language heldout claims.
        all_heldout_episodes = primary_curriculum.split("heldout")
        heldout_episodes = (
            all_heldout_episodes
            if args.evaluation_case_limit is None
            else all_heldout_episodes[: args.evaluation_case_limit]
        )
        all_sequential_heldout = sequential_heldout_cases
        sequential_heldout = (
            all_sequential_heldout
            if args.evaluation_case_limit is None
            else all_sequential_heldout[: args.evaluation_case_limit]
        )
        complete_heldout_evaluation = len(heldout_episodes) == len(all_heldout_episodes) and len(
            sequential_heldout
        ) == len(all_sequential_heldout)
        complete_regression_evaluation = len(regression_episodes) == len(all_regression_episodes) and len(
            sequential_regression
        ) == len(all_sequential_regression)

        def evaluate_candidate() -> dict[str, Any]:
            session.candidate_module.eval()
            qa_rows: list[dict[str, Any]] = []
            exact_rows = [
                evaluate_living_episode(
                    session.candidate_module,
                    episode,
                    soul_branch.load_head(),
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                    transcript_sink=qa_rows,
                )
                for episode in heldout_episodes
            ]
            sequential_rows = [
                evaluate_sequential_case(
                    session.candidate_module,
                    case,
                    soul_branch.load_head(),
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                )
                for case in sequential_heldout
            ]
            exact_losses = []
            sequential_losses = []
            with torch.no_grad():
                for episode in heldout_episodes:
                    loss, _unroll, _metrics = living_episode_objective(
                        session.candidate_module,
                        episode,
                        soul_branch.load_head(),
                        core_id=module_id,
                        parameter_generation=candidate_generation,
                    )
                    exact_losses.append(float(loss.item()))
                for case in sequential_heldout:
                    loss, _unrolls, _soul = sequential_living_objective(
                        session.candidate_module,
                        case,
                        soul_branch.load_head(),
                        core_id=module_id,
                        parameter_generation=candidate_generation,
                    )
                    sequential_losses.append(float(loss.item()))

            foundation_regression_episodes = tuple(
                episode
                for episode in regression_episodes
                if is_foundation_sequence_episode(episode)
            )
            foundation_regression_rows = [
                evaluate_living_episode(
                    session.candidate_module,
                    episode,
                    soul_branch.load_head(),
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                )
                for episode in foundation_regression_episodes
            ]

            def aggregate(
                rows: list[dict[str, Any]],
                losses: list[float],
            ) -> dict[str, Any]:
                supervised_phase_count = sum(
                    row["supervised_phase_count"] for row in rows
                )
                payload_supervised_phase_count = sum(
                    row["payload_supervised_phase_count"] for row in rows
                )
                phase_output_count = sum(row["phase_output_count"] for row in rows)
                payload_token_count = sum(
                    row["payload_teacher_forced_token_count"] for row in rows
                )
                payload_token_correct = sum(
                    row["payload_teacher_forced_token_correct"] for row in rows
                )
                payload_target_counts = [
                    sum(row["payload_teacher_forced_target_counts"][index] for row in rows)
                    for index in range(session.candidate_module.eos_index + 1)
                ]
                return {
                    "evaluated_case_count": len(rows),
                    "heldout_mean_loss": sum(losses) / max(1, len(losses)),
                    "typed_emission_exact_rate": sum(
                        row["typed_emission_exact_count"] for row in rows
                    )
                    / max(1.0, supervised_phase_count),
                    "payload_transport_exact_rate": sum(
                        row["payload_transport_exact_count"] for row in rows
                    )
                    / max(1.0, payload_supervised_phase_count),
                    "complete_field_coverage_rate": sum(
                        row["complete_field_coverage_count"] for row in rows
                    )
                    / max(1.0, phase_output_count),
                    "supervised_phase_count": supervised_phase_count,
                    "payload_supervised_phase_count": payload_supervised_phase_count,
                    "phase_output_count": phase_output_count,
                    "constant_typed_emission_exact_floor": 0.0,
                    "constant_payload_transport_exact_floor": 0.0,
                    "payload_teacher_forced_token_accuracy": (
                        payload_token_correct / max(1, payload_token_count)
                    ),
                    "constant_payload_token_accuracy_floor": (
                        max(payload_target_counts) / max(1, payload_token_count)
                    ),
                }

            combined_rows = exact_rows + sequential_rows
            result = aggregate(
                combined_rows,
                exact_losses + sequential_losses,
            )
            result["counterfactuals"] = living_source_counterfactuals(
                session.candidate_module,
                heldout_episodes[0],
                soul_branch.load_head(),
                core_id=module_id,
                parameter_generation=candidate_generation,
            )
            result["foundation_sequence_heldout_probe"] = foundation_sequence_probe(
                heldout_episodes,
                exact_rows,
            )
            result["foundation_sequence_regression_probe"] = foundation_sequence_probe(
                foundation_regression_episodes,
                foundation_regression_rows,
            )
            result["isolated_family_evaluations"] = {}
            for family in sorted(set(evaluation_family_by_episode.values())):
                indexes = [
                    index
                    for index, episode in enumerate(heldout_episodes)
                    if evaluation_family_by_episode.get(episode.episode_id) == family
                ]
                if indexes:
                    result["isolated_family_evaluations"][family] = aggregate(
                        [exact_rows[index] for index in indexes],
                        [exact_losses[index] for index in indexes],
                    )
            result["isolated_manifest_evaluations"] = {}
            for manifest_id in sorted(set(evaluation_manifest_by_episode.values())):
                indexes = [
                    index
                    for index, episode in enumerate(heldout_episodes)
                    if evaluation_manifest_by_episode.get(episode.episode_id) == manifest_id
                ]
                if indexes:
                    result["isolated_manifest_evaluations"][manifest_id] = aggregate(
                        [exact_rows[index] for index in indexes],
                        [exact_losses[index] for index in indexes],
                    )
            result["qa_transcripts"] = qa_rows[:12]
            return result

        if progress is not None:
            progress.emit(
                "evaluating", phase="initial", global_step=(0 if latest_bundle is None else latest_bundle.step)
            )
        initial_evaluation = evaluate_candidate()
        if progress is not None:
            progress.emit(
                "evaluated",
                phase="initial",
                global_step=(0 if latest_bundle is None else latest_bundle.step),
                heldout_mean_loss=initial_evaluation["heldout_mean_loss"],
                typed_emission_exact_rate=initial_evaluation["typed_emission_exact_rate"],
                payload_transport_exact_rate=initial_evaluation["payload_transport_exact_rate"],
                payload_teacher_forced_token_accuracy=initial_evaluation[
                    "payload_teacher_forced_token_accuracy"
                ],
                constant_payload_token_accuracy_floor=initial_evaluation[
                    "constant_payload_token_accuracy_floor"
                ],
                evaluated_case_count=initial_evaluation["evaluated_case_count"],
                qa_transcripts=initial_evaluation.get("qa_transcripts", [])[:8],
            )
        prior_reports = sorted(campaign_report_dir.glob("segment_*.json"))
        campaign_baseline_evaluation = (
            initial_evaluation
            if not prior_reports
            else json.loads(prior_reports[0].read_text(encoding="utf-8"))["initial_evaluation"]
        )
        start_step = 0 if latest_bundle is None else latest_bundle.step
        if args.evaluate_only:
            end_step = start_step
        elif tranche is not None:
            end_step = tranche.final_global_step
        else:
            end_step = min(
                args.max_steps,
                start_step + (args.max_steps if args.run_steps is None else args.run_steps),
            )
        if start_step >= end_step and tranche is None and not args.evaluate_only:
            raise RuntimeError("candidate campaign is already complete")
        segment_transitions = []
        segment_receipt_ids = []
        ephemeral_soul = soul_branch.load_head()
        segment_start_soul = ephemeral_soul
        curriculum_lanes = _training_lanes(
            mechanism_curriculum,
            standard_ffcs,
            sequential_ffcs,
        )
        if not curriculum_lanes:  # pragma: no cover - mechanism always supplies train
            raise RuntimeError("governed campaign has no supervised training material")
        for step in range(start_step, end_step):
            lane_name, kind, material, source_manifest_id = _scheduled_material(curriculum_lanes, step)
            captured: dict[str, Any] = {}

            def loss_fn(
                candidate: torch.nn.Module,
                kind=kind,
                material=material,
                captured=captured,
                soul=ephemeral_soul,
            ) -> torch.Tensor:
                if not isinstance(candidate, LivingReasoningCoreD64):
                    raise TypeError("governed candidate clone has the wrong architecture")
                candidate.train()
                loss, unroll, phase_metrics, transitions = _material_objective(
                    candidate,
                    kind=kind,
                    material=material,
                    soul=soul,
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                )
                captured.update(
                    {
                        "loss": float(loss.detach().item()),
                        "unroll": unroll,
                        "phase_metrics": phase_metrics,
                        "transitions": transitions,
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
            segment_transitions.extend(captured["transitions"])
            segment_receipt_ids.append(optimizer_receipt.receipt_id)
            checkpoint_due = (step + 1) % args.checkpoint_interval == 0 or step + 1 == end_step
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
                    "material_kind": kind,
                    "curriculum_lane": lane_name,
                    "source_manifest_id": source_manifest_id,
                    "material_id": (material.episode_id if kind == "episode" else material.case_id),
                    "loss": captured["loss"],
                    "optimization_receipt_id": optimizer_receipt.receipt_id,
                    "accepted_step_bundle_id": (None if accepted_bundle is None else accepted_bundle.bundle_id),
                    "accepted_segment_optimizer_receipt_ids": accepted_segment_receipt_ids,
                    "soul_receipt_ids": ([] if accepted_bundle is None else list(accepted_bundle.soul_receipt_ids)),
                    "soul_id": ephemeral_soul.soul_id,
                    "checkpoint_id": None if checkpoint is None else checkpoint.checkpoint_id,
                    "phase_metrics": captured["phase_metrics"],
                    "wall_seconds": wall_seconds,
                    "peak_cuda_bytes": (0 if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))),
                }
            )
            if progress is not None:
                progress.emit(
                    "training",
                    global_step=step + 1,
                    segment_start_step=start_step + 1,
                    segment_end_step=end_step,
                    loss=captured["loss"],
                    learning_rate=args.learning_rate,
                    material_kind=kind,
                    curriculum_lane=lane_name,
                    wall_seconds=wall_seconds,
                    checkpoint_id=None if checkpoint is None else checkpoint.checkpoint_id,
                    accepted_step_bundle_id=(None if accepted_bundle is None else accepted_bundle.bundle_id),
                )
        if progress is not None:
            progress.emit("evaluating", phase="final", global_step=end_step)
        final_evaluation = evaluate_candidate()
        if progress is not None:
            progress.emit(
                "evaluated",
                phase="final",
                global_step=end_step,
                heldout_mean_loss=final_evaluation["heldout_mean_loss"],
                typed_emission_exact_rate=final_evaluation["typed_emission_exact_rate"],
                payload_transport_exact_rate=final_evaluation["payload_transport_exact_rate"],
                payload_teacher_forced_token_accuracy=final_evaluation[
                    "payload_teacher_forced_token_accuracy"
                ],
                constant_payload_token_accuracy_floor=final_evaluation[
                    "constant_payload_token_accuracy_floor"
                ],
                evaluated_case_count=final_evaluation["evaluated_case_count"],
                qa_transcripts=final_evaluation.get("qa_transcripts", [])[:8],
            )
        counterfactuals_passed = all(value > 1e-8 for value in final_evaluation["counterfactuals"].values())
        task_gate_passed = (
            complete_heldout_evaluation
            and final_evaluation["heldout_mean_loss"] < campaign_baseline_evaluation["heldout_mean_loss"]
            and final_evaluation["payload_teacher_forced_token_accuracy"]
            > final_evaluation["constant_payload_token_accuracy_floor"]
            and counterfactuals_passed
        )
        exact_gate_passed = (
            final_evaluation["typed_emission_exact_rate"] > final_evaluation["constant_typed_emission_exact_floor"]
            and final_evaluation["payload_transport_exact_rate"]
            > final_evaluation["constant_payload_transport_exact_floor"]
        )
        promotion_plan = soul_workspace.promotion_plan(soul_manifest)
        tournament_metric_computation = d64_tournament_metric_computation(
            session.candidate_module,
            episodes=heldout_episodes,
            sequential_cases=sequential_heldout,
            initial_soul=soul_branch.load_head(),
            regression_episodes=regression_episodes,
            regression_sequential_cases=sequential_regression,
            stale_soul=(segment_start_soul if segment_start_soul.soul_id != soul_branch.load_head().soul_id else None),
            core_id=module_id,
            parameter_generation=candidate_generation,
            heldout_surface_complete=complete_heldout_evaluation,
            regression_surface_complete=complete_regression_evaluation,
        )
        tournament_metrics = tournament_metric_computation.metric_mapping
        metric_surface_complete = tournament_metric_computation.complete
        foundation_sequence_gate = decide_foundation_sequence_mastery(
            heldout_probe=final_evaluation.get("foundation_sequence_heldout_probe"),
            regression_probe=final_evaluation.get("foundation_sequence_regression_probe"),
            evaluation=final_evaluation,
            complete_heldout=complete_heldout_evaluation,
            complete_regression=complete_regression_evaluation,
        )
        curriculum_stage_complete = (
            task_gate_passed
            and exact_gate_passed
            and (
                bool(foundation_sequence_gate["passed"])
                if foundation_sequence_gate is not None
                else metric_surface_complete
            )
        )
        final_checkpoint_id = (
            checkpoints[-1].checkpoint_id
            if checkpoints
            else step_bundles.latest_bundle(module_id, candidate_generation).checkpoint_id
        )
        if curriculum_stage_complete:
            lifecycle_event = session.complete(
                reason="complete curriculum-stage gate passed; serving activation remains separate"
            )
        else:
            lifecycle_event = session.pause(
                reason=(
                    "execution segment ended at an exact accepted checkpoint; "
                    "curriculum stage remains open for a later renewable tranche"
                ),
                checkpoint_id=final_checkpoint_id,
            )
        report.update(
            {
                "candidate_generation_id": candidate_generation,
                "soul_candidate_manifest_id": soul_manifest.manifest_id,
                "recovered_bundle_ids": [item.bundle_id for item in recovered_bundles],
                "soul_promotion_plan": promotion_plan.to_canonical_dict(),
                "steps": steps,
                "checkpoint_interval": args.checkpoint_interval,
                "segment_start_step": (start_step if args.evaluate_only else start_step + 1),
                "segment_end_step": end_step,
                "campaign_max_steps": (args.max_steps if args.legacy_plan_v1 else None),
                "evaluation_only": bool(args.evaluate_only),
                "campaign_complete": curriculum_stage_complete,
                "curriculum_stage_complete": curriculum_stage_complete,
                "legacy_plan_envelope_exhausted": (end_step >= args.max_steps if args.legacy_plan_v1 else None),
                "resource_tranche_consumed": (tranche is not None and end_step == tranche.final_global_step),
                "paused_for_next_tranche": not curriculum_stage_complete,
                "resource_tranche": (None if tranche is None else tranche.to_canonical_dict()),
                "heldout_case_count": len(all_heldout_episodes),
                "evaluated_heldout_case_count": len(heldout_episodes),
                "deferred_heldout_case_count": (len(all_heldout_episodes) - len(heldout_episodes)),
                "sequential_heldout_case_count": len(all_sequential_heldout),
                "evaluated_sequential_heldout_case_count": len(sequential_heldout),
                "deferred_sequential_heldout_case_count": (len(all_sequential_heldout) - len(sequential_heldout)),
                "regression_case_count": len(all_regression_episodes),
                "evaluated_regression_case_count": len(regression_episodes),
                "deferred_regression_case_count": (len(all_regression_episodes) - len(regression_episodes)),
                "sequential_regression_case_count": len(all_sequential_regression),
                "evaluated_sequential_regression_case_count": len(sequential_regression),
                "deferred_sequential_regression_case_count": (
                    len(all_sequential_regression) - len(sequential_regression)
                ),
                "complete_heldout_evaluation": complete_heldout_evaluation,
                "complete_regression_evaluation": complete_regression_evaluation,
                "initial_evaluation": initial_evaluation,
                "campaign_baseline_evaluation": campaign_baseline_evaluation,
                "segment_heldout_loss_fell": (
                    final_evaluation["heldout_mean_loss"] < initial_evaluation["heldout_mean_loss"]
                ),
                "final_evaluation": final_evaluation,
                "final_checkpoint_id": final_checkpoint_id,
                "lifecycle_event_id": lifecycle_event.event_id,
                "lifecycle_status": lifecycle_event.status.value,
                "task_gate_passed": task_gate_passed,
                "task_gate_policy": (
                    (
                        "on the isolated VERIFIED_TARGET standard-curriculum surface: "
                        if standard_ffcs
                        else "on the synthetic mechanism surface: "
                    )
                    + "heldout loss falls; teacher-forced transport token accuracy exceeds "
                    "that surface's strongest constant-category floor; field/proposal/Soul "
                    "counterfactuals are nonzero; every evidence-qualified heldout case is evaluated"
                ),
                "exact_serving_gate_passed": exact_gate_passed,
                "exact_serving_gate_policy": (
                    "free-running typed emission and complete Unicode payload exact rates "
                    "must both exceed their constant zero floors"
                ),
                "serving_promotion_claimed": False,
                "tournament_metrics": tournament_metrics,
                "tournament_metric_computation": (tournament_metric_computation.to_canonical_dict()),
                "missing_tournament_metrics": list(tournament_metric_computation.missing_metrics),
                "tournament_metric_surface_complete": metric_surface_complete,
                "foundation_sequence_gate": foundation_sequence_gate,
            }
        )

    report["report_id"] = canonical_sha256(report)
    report_path = (
        campaign_report_dir / f"segment_{start_step:09d}_{end_step:09d}_eval.json"
        if args.evaluate_only
        else campaign_report_dir / f"segment_{start_step + 1:09d}_{end_step:09d}.json"
    )
    _write_immutable_json(report_path, report)
    if progress is not None:
        progress.emit(
            "completed" if report["curriculum_stage_complete"] else "paused",
            global_step=report["segment_end_step"],
            curriculum_stage_complete=report["curriculum_stage_complete"],
            task_gate_passed=report["task_gate_passed"],
            exact_serving_gate_passed=report["exact_serving_gate_passed"],
            heldout_mean_loss=report["final_evaluation"]["heldout_mean_loss"],
            report_path=str(report_path),
            report_id=report["report_id"],
        )
    print(json.dumps({**report, "report_path": str(report_path)}, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
