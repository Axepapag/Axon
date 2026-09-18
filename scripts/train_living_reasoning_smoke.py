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
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from runtime.field import CanonicalStateBranch, LogicalRegion, canonical_sha256
from runtime.heart import ReasoningDecision, ReasoningOperationKind
from runtime.soul import SoulStore
from runtime.trainer import (
    CandidateCheckpointRecord,
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
    cloud_bundle,
    resolve_base_module,
)
from training import (
    COPY_ALIGNMENT_MULTICELL_TEACH,
    COPY_ALIGNMENT_MULTICELL_TEACH_ID,
    FOUNDATION_MOTOR_GATE_POLICY_ID,
    FOUNDATION_MOTOR_V2_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_ID,
    FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2_ID,
    FOUNDATION_MOTOR_V2_STAGE_ORDER,
    FOUNDATION_SEQUENCE_GATE_POLICY_ID,
    RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    RECEIPT_TERMINATION_HEAD_PROFILES,
    RECEIPT_TEACHING_PROFILES,
    LivingReasoningCoreD64,
    LivingReasoningCurriculum,
    TeachingEligibility,
    apply_copy_alignment_multicell_teach_weights,
    apply_receipt_continuation_teach_weights,
    build_living_reasoning_preflight,
    build_living_reasoning_smoke_curriculum,
    candidate_a_config,
    constant_baseline_floors,
    d64_tournament_metric_computation,
    decide_foundation_motor_mastery,
    decide_foundation_motor_v2_checkpoint_retention,
    decide_foundation_motor_v2_checkpoint_retention_v2,
    decide_foundation_motor_v2_stage,
    decide_foundation_sequence_mastery,
    resolve_retention_action,
    evaluate_living_episode,
    evaluate_sequential_case,
    foundation_motor_probe,
    foundation_motor_v2_action,
    foundation_motor_v2_objective_program_id,
    foundation_motor_v2_probe,
    foundation_motor_v2_stage_policy,
    foundation_motor_v2_termination_route_rejection,
    RECEIPT_GENERATE_HEAD_PROFILES,
    foundation_sequence_probe,
    is_foundation_motor_episode,
    is_foundation_motor_v2_episode,
    is_foundation_sequence_episode,
    living_episode_objective,
    living_source_counterfactuals,
    load_first_form_curriculum,
    load_sequential_first_form,
    oversample_multicell_copy_cases,
    receipt_continuation_teach_profile,
    sequential_living_objective,
)

ROOT = Path(__file__).resolve().parent.parent


def _compact_motor_v2_probes(evaluation: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "foundation_motor_v2_heldout_probe",
        "foundation_motor_v2_regression_probe",
    ):
        probe = evaluation.get(key)
        if not isinstance(probe, dict):
            continue
        pair = probe.get("pair_exact_rates") or {}
        compact[key] = {
            "case_count": probe.get("case_count"),
            "alignment_copy_gate_accuracy": probe.get("alignment_copy_gate_accuracy"),
            "alignment_position_accuracy": probe.get("alignment_position_accuracy"),
            # The termination repair is judged on exactly this rate: the legacy
            # route held it at 0.3125 because EOS won every argmax.
            "alignment_eos_gate_accuracy": probe.get("alignment_eos_gate_accuracy"),
            "pair_copy_gate": pair.get("copy_gate") if isinstance(pair, dict) else None,
            "pair_position": pair.get("position") if isinstance(pair, dict) else None,
        }
    return compact


def _continuation_step_telemetry(phase_metrics: Any) -> dict[str, Any]:
    """Per-step view of the termination objective, for the monitor only.

    Nothing here feeds the loss: positions are summed and the continuation loss
    is averaged over the phases that actually evaluated a stop=0 term, so a step
    with no supervised anchor reports unavailable instead of a fabricated value.
    """

    positions = 0
    losses: list[float] = []
    eos_gate_accuracies: list[float] = []
    for phase in phase_metrics or ():
        if not isinstance(phase, Mapping):
            continue
        counted = phase.get("termination_continue_positions")
        if counted is not None:
            positions += int(counted)
        loss = phase.get("termination_continue_loss")
        if loss is not None:
            losses.append(float(loss))
        accuracy = phase.get("alignment_eos_gate_accuracy")
        if accuracy is not None:
            eos_gate_accuracies.append(float(accuracy))
    return {
        "termination_continue_positions": positions,
        "termination_continue_loss": (sum(losses) / len(losses)) if losses else None,
        "training_alignment_eos_gate_accuracy": (
            sum(eos_gate_accuracies) / len(eos_gate_accuracies)
            if eos_gate_accuracies
            else None
        ),
    }


def _histogram_items(value: Any) -> Any:
    """Yield ``(label, count)`` candidates from one serialised histogram value.

    All three surface assemblers store ``constant_*_target_histogram`` as a
    ``dict[label, count]``; list-of-pairs and ``{"key": ..., "count": ...}`` stay
    supported because they are what a serialised/reloaded report round-trips to.
    """

    if isinstance(value, Mapping):
        candidate = {str(key) for key in value}
        if candidate == {"key", "count"}:
            yield value
            return
        yield from value.items()
        return
    yield from value or ()


def merge_surface_histogram(rows: Any, name: str) -> dict[str, int]:
    """Merge per-case answer histograms into one surface histogram.

    Per-case rates cannot be averaged into a constant-emitter baseline: the
    strongest fixed answer has to be found across the whole surface, not inside
    each evaluated case.

    Iterating a mapping yields its *keys*, so a ``dict`` row must be unwrapped
    before the generic iteration.  The smoke script previously iterated
    ``row.get(name)`` directly, which matched neither branch for the shape
    ``evaluate_living_episode`` actually returns, skipped every entry, merged to
    ``{}``, and collapsed both constant-emitter floors to ``0.0`` -- the same
    vacuous-floor trap the constant-baseline work removed, one shape down.  It is
    invisible in the report because ``typed > 0.0`` still renders as "beat the
    floor", so an empty merge is raised rather than returned.
    """

    merged: dict[str, int] = {}
    for row in rows or ():
        for item in _histogram_items(row.get(name)):
            if isinstance(item, Mapping):
                label, count = str(item["key"]), int(item["count"])
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                label, count = str(item[0]), int(item[1])
            else:
                continue
            merged[label] = merged.get(label, 0) + count
    if not merged and rows:
        raise RuntimeError(
            "constant-emitter baseline histogram merged to empty for "
            f"{name} over {len(rows)} evaluated cases; the constant floors would "
            "be vacuous zeroes and every beat-the-floor verdict would be a "
            "comparison against nothing"
        )
    return merged


def nonzero_exact_output_observed(evaluation: dict[str, Any]) -> bool:
    """Report weak behavioral progress without implying serving readiness."""

    return bool(
        evaluation["typed_emission_exact_rate"]
        > evaluation["constant_typed_emission_exact_floor"]
        and evaluation["payload_transport_exact_rate"]
        > evaluation["constant_payload_transport_exact_floor"]
    )


PROBATION_SIDECAR_SCHEMA = "axon-motor-retention-probation-v1"


def select_qa_transcript_rows(
    rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Pick a deterministic, family-balanced transcript sample.

    The panel used to take the first rows of a sink that stopped after the third
    payload phase of each episode, so every sample came from the same few early
    cases and the panel read as a wall of identical failures with no successes
    to compare against.  Sample across action families instead, taking failures
    before exact matches inside each family, so one screen shows what is wrong
    and what is right.  Selection is display-only; it changes no metric.
    """

    if limit < 1:
        raise ValueError("qa transcript limit must be positive")
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("family") or "unknown")
        buckets.setdefault(key, []).append(row)
    order = sorted(buckets)
    for key in order:
        # Stable partition: a failing row is the actionable row, so it leads.
        buckets[key].sort(key=lambda item: bool(item.get("exact_match")))
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < limit and any(index < len(buckets[key]) for key in order):
        for key in order:
            if len(selected) >= limit:
                break
            if index < len(buckets[key]):
                selected.append(buckets[key][index])
        index += 1
    return selected


def _probation_sidecar_path(campaign_report_dir: Path) -> Path:
    return campaign_report_dir / "retention_probation.json"


def _write_probation_sidecar(path: Path, body: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _clear_probation_sidecar(path: Path) -> None:
    if path.exists():
        path.unlink()


def _load_probation_sidecar(
    *,
    path: Path,
    candidate_generation: str,
    learning_policy: dict[str, Any],
    effective_objective_program_id: str,
    architecture_id: str,
    standard_ffcs_manifest_ids: list[str],
    max_plateau_probation: int,
) -> dict[str, Any] | None:
    """Load and validate a probation sidecar; fail closed on any mismatch.

    Returns None when no sidecar exists.  A malformed or identity-mismatched
    sidecar raises: silently ignoring it could resume the wrong optimizer
    state under a different recipe.
    """

    if not path.exists():
        return None
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("schema") != PROBATION_SIDECAR_SCHEMA:
        raise RuntimeError(f"probation sidecar schema mismatch: {path}")
    identity_checks = {
        "candidate_generation_id": candidate_generation,
        "learning_policy": learning_policy,
        "effective_objective_program_id": effective_objective_program_id,
        "architecture_id": architecture_id,
        "standard_ffcs_manifest_ids": standard_ffcs_manifest_ids,
        "max_plateau_probation": max_plateau_probation,
    }
    for key, expected in identity_checks.items():
        if body.get(key) != expected:
            raise RuntimeError(f"probation sidecar identity mismatch on {key}: {path}")
    for key in (
        "probationary_checkpoint",
        "probationary_step",
        "probation_count",
        "reference_evaluation",
        "confirmed_checkpoint",
    ):
        if key not in body:
            raise RuntimeError(f"probation sidecar is missing {key}: {path}")
    if (
        isinstance(body["probation_count"], bool)
        or not isinstance(body["probation_count"], int)
        or body["probation_count"] < 1
    ):
        raise RuntimeError(f"probation sidecar has an invalid count: {path}")
    if body["probation_count"] >= max_plateau_probation:
        raise RuntimeError(
            "probation sidecar meets or exceeds its allowance; the branch "
            f"should have been abandoned and cannot be resumed: {path}"
        )
    # Validate the embedded checkpoint records deserialize and are self-consistent.
    CandidateCheckpointRecord.from_mapping(body["probationary_checkpoint"])
    CandidateCheckpointRecord.from_mapping(body["confirmed_checkpoint"])
    return body


def exact_serving_gate_passed(
    evaluation: dict[str, Any],
    *,
    curriculum_stage_complete: bool,
    complete_heldout: bool,
    complete_regression: bool,
    tournament_metric_surface_complete: bool,
) -> bool:
    """Fail closed unless the complete serving evidence surface is exact."""

    return bool(
        curriculum_stage_complete
        and complete_heldout
        and complete_regression
        and tournament_metric_surface_complete
        and evaluation["typed_emission_exact_rate"] == 1.0
        and evaluation["payload_transport_exact_rate"] == 1.0
    )


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
    probation_sidecar: dict | None = None,
) -> str | None:
    """Prove which resource tranche, if any, produced the resuming parent.

    A tranche record is only an issued allowance. It becomes lineage only when
    an immutable segment report binds it to the exact final checkpoint and
    accepted step bundle. This prevents an abandoned allowance from being
    mistaken for consumed history.

    Under a plateau probation the resuming parent is the probationary
    checkpoint, not the confirmed accepted bundle, so the proof targets the
    probation anchor: the segment report reaching the probationary step whose
    final checkpoint is the sidecar's probationary checkpoint.
    """

    if probation_sidecar is not None:
        anchor_step = int(probation_sidecar["probationary_step"])
        anchor_checkpoint = str(probation_sidecar["probationary_checkpoint"]["checkpoint_id"])
        target_step = anchor_step
        target_checkpoint_id = anchor_checkpoint
        target_bundle_id = None
    else:
        target_step = int(latest_bundle.step)
        target_checkpoint_id = str(latest_bundle.checkpoint_id)
        target_bundle_id = str(latest_bundle.bundle_id)

    matching_issued = tuple(
        item
        for item in tranche_store.tranches_for(module_id, candidate_generation_id)
        if item.plan_id == plan_id
        and item.learning_policy_id == learning_policy_id
        and item.final_global_step == target_step
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
            or body.get("segment_end_step") != target_step
            or body.get("final_checkpoint_id") != target_checkpoint_id
        ):
            continue
        if target_bundle_id is not None:
            accepted_bundle_ids = tuple(
                item.get("accepted_step_bundle_id")
                for item in body.get("steps", ())
                if item.get("accepted_step_bundle_id") is not None
            )
            if accepted_bundle_ids:
                if accepted_bundle_ids[-1] != target_bundle_id:
                    continue
            else:
                # Promotion path: a stage-gate promotion accepts at segment
                # end, after per-step records are built, so no step carries
                # the bundle id. The binding above already requires the
                # report's final checkpoint to equal the accepted chain
                # head's checkpoint (coordinator-verified bundle<->checkpoint
                # identity), which an abandoned tranche can never satisfy —
                # abandonment by definition never advances the accepted head
                # to the tentative state.
                pass
        resource = body.get("resource_tranche")
        if resource is None:
            continue
        reported = ResourceTranche.from_mapping(resource)
        durable = tranche_store.read_tranche(reported.tranche_id)
        if durable.to_canonical_dict() != reported.to_canonical_dict():
            raise RuntimeError("segment report resource tranche differs from durable record")
        consumed.add(reported.tranche_id)
    if len(consumed) > 1:
        raise RuntimeError("multiple consumed tranches claim the resuming parent state")
    if consumed:
        return next(iter(consumed))
    if matching_issued:
        raise RuntimeError(
            "resuming parent coincides with an issued but unclosed resource tranche; "
            "regenerate its evaluation/report before issuing a continuation"
        )
    return None


def _continuation_parent_global_step(latest_bundle_step: int, probation_sidecar) -> int:
    """Parent step recorded on a tranche continuation receipt.

    Must equal the leased tranche's ``base_global_step``. Under a plateau
    probation the lease anchors at the probationary step while the accepted
    chain stays at the confirmed parent, so the receipt must follow the
    anchor rather than the accepted bundle's step. The probationary
    checkpoint's own provenance lives in the probation sidecar.
    """
    if probation_sidecar is not None:
        return int(probation_sidecar["probationary_step"])
    return int(latest_bundle_step)


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
        "--generate-gate-bias",
        type=float,
        default=1.5,
        help=(
            "initial copy/generate-gate bias; 1.5 is Candidate A (~82%% generate). "
            "0.0 is a fair coin. Initialization only; not part of architecture identity."
        ),
    )
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
    parser.add_argument(
        "--teach-multicell-copy",
        action="store_true",
        help=(
            "copy_alignment teaching overlay: sum position loss, stop paying "
            "copy-gate 4x, and oversample authored multi-cell train letters. "
            "Does not change program_id, architecture, or the heldout/regression exam."
        ),
    )
    parser.add_argument(
        "--receipt-continuation",
        action="store_true",
        help=(
            "opt into the ratified receipt-governed intra-scalar conduit, its new "
            "architecture/state/objective identities, continuation-loss masking, and "
            "same-stage EOS co-supervision; legacy mode remains the default"
        ),
    )
    parser.add_argument(
        "--receipt-teaching-profile",
        choices=RECEIPT_TEACHING_PROFILES,
        default=RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
        help=(
            "content-addressed receipt objective profile; changing this value creates "
            "a new candidate lineage and is never a checkpoint resume"
        ),
    )
    parser.add_argument(
        "--eos-generate-head-route",
        action="store_true",
        help=(
            "use generated-head EOS termination; requires receipt continuation "
            "and the generate_head_eos_v3 objective profile"
        ),
    )
    parser.add_argument(
        "--termination-head-route",
        action="store_true",
        help=(
            "use a dedicated scalar termination head; requires receipt "
            "continuation and a termination-head objective profile "
            "(termination_head_v5 or the ratified termination_head_balanced_v6); "
            "mutually exclusive with --eos-generate-head-route"
        ),
    )
    parser.add_argument(
        "--motor-retention-guard",
        action="store_true",
        help=(
            "evaluate one tentative end-of-tranche checkpoint before publishing it; "
            "restore the accepted parameter/optimizer/Soul parent if complete-surface "
            "closed-loop motor behavior regresses. Requires one checkpoint at the "
            "tranche boundary and forbids --evaluation-case-limit."
        ),
    )
    parser.add_argument(
        "--motor-retention-plateau-probation",
        type=int,
        default=0,
        help=(
            "three-state guard: allow this many consecutive plateau tranches to "
            "continue their exact optimizer state without promotion (reference "
            "stays the last confirmed accepted parent; regression rolls all the "
            "way back; exhaustion abandons the branch). 0 keeps the legacy "
            "two-state contract where plateau rejects immediately."
        ),
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
    *,
    foundation_motor_v2_training_stage: str | None = None,
    teach_multicell_copy: bool = False,
) -> tuple[tuple[str, tuple[tuple[str, Any, str], ...]], ...]:
    """Build deterministic family lanes without flattening sequential cases."""

    lanes: list[tuple[str, tuple[tuple[str, Any, str], ...]]] = []
    if foundation_motor_v2_training_stage is None:
        lanes.append(
            (
                "mechanism",
                tuple(
                    ("episode", episode, mechanism_curriculum.train_manifest_id)
                    for episode in mechanism_curriculum.split("train")
                ),
            )
        )
    else:
        eligible = set(
            foundation_motor_v2_stage_policy(foundation_motor_v2_training_stage)[
                "eligible_actions"
            ]
        )
    for item in standard_ffcs:
        teaching_manifest_id = item.teaching_living_curriculum.train_manifest_id
        for family, _counts in item.requested_family_split_counts:
            cases = tuple(
                case
                for case in item.teaching_cases
                if case.family == family and case.episode.split == "train"
            )
            if foundation_motor_v2_training_stage is not None:
                cases = tuple(
                    case
                    for case in cases
                    if is_foundation_motor_v2_episode(case.episode)
                    and foundation_motor_v2_action(case.episode) in eligible
                )
                if teach_multicell_copy and foundation_motor_v2_training_stage == "copy_alignment":
                    cases = oversample_multicell_copy_cases(cases)
            lanes.append(
                (
                    (
                        f"ffcs-{family}-{foundation_motor_v2_training_stage}"
                        if foundation_motor_v2_training_stage is not None
                        else f"ffcs-{family}"
                    ),
                    tuple(
                        (
                            "first_form_case",
                            case,
                            teaching_manifest_id,
                        )
                        for case in cases
                    ),
                )
            )
    for item in (
        sequential_ffcs if foundation_motor_v2_training_stage is None else ()
    ):
        lanes.append(
            (
                "ffcs-E",
                tuple(("sequential", case, item.train_manifest_id) for case in item.split("train")),
            )
        )
    return tuple((name, rows) for name, rows in lanes if rows)


def _foundation_motor_v2_stage_from_reports(
    campaign_report_dir: Path,
) -> tuple[str, bool]:
    """Derive the next lesson solely from immutable prior stage-gate evidence."""

    stage_index = 0
    program_complete = False
    for path in sorted(campaign_report_dir.glob("segment_*.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        observed_report_id = body.get("report_id")
        if observed_report_id is None or canonical_sha256(
            {key: value for key, value in body.items() if key != "report_id"}
        ) != observed_report_id:
            raise RuntimeError(f"foundation motor v2 report identity mismatch: {path}")
        if body.get("foundation_motor_v2_program_id") != FOUNDATION_MOTOR_V2_PROGRAM_ID:
            continue
        if body.get("evaluation_only"):
            continue
        expected_stage = FOUNDATION_MOTOR_V2_STAGE_ORDER[stage_index]
        if body.get("foundation_motor_v2_training_stage") != expected_stage:
            raise RuntimeError("foundation motor v2 stage history is not contiguous")
        gate = body.get("foundation_motor_v2_stage_gate")
        if not isinstance(gate, dict) or gate.get("training_stage") != expected_stage:
            raise RuntimeError("foundation motor v2 report lacks its exact stage gate")
        if bool(gate.get("passed")):
            if stage_index == len(FOUNDATION_MOTOR_V2_STAGE_ORDER) - 1:
                program_complete = True
                break
            stage_index += 1
    return FOUNDATION_MOTOR_V2_STAGE_ORDER[stage_index], program_complete


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
    component_weights: dict[str, float] | None = None,
    alignment_position_reduction: str = "mean",
    payload_eos_weight: float = 4.0,
    termination_continue_supervision: bool = False,
) -> tuple[torch.Tensor, Any, tuple[dict[str, float], ...], tuple[Any, ...]]:
    """Run one scheduled lesson and return its complete Soul lineage."""

    if kind == "sequential":
        if component_weights is not None:
            raise ValueError("staged component weights do not apply to sequential material")
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
        component_weights=component_weights,
        alignment_position_reduction=alignment_position_reduction,
        payload_eos_weight=payload_eos_weight,
        termination_continue_supervision=termination_continue_supervision,
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
    # Opt-in mid-run artifact sync (ratified 2026-09-04 proposal). Without the
    # packet-injected AXON_SYNC_* environment this is a no-op with one journal
    # note; it never blocks or fails training.
    sync_hook = cloud_bundle.MidRunSyncHook.from_environment(
        job_id=args.external_job_id or f"local-{os.getpid()}",
        state_root=args.state_root,
        staging_root=args.state_root.parent / "axon_sync_staging",
        receipt_log=(args.progress_dir / "sync_receipts.jsonl") if args.progress_dir is not None else None,
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
    if (
        not args.receipt_continuation
        and args.receipt_teaching_profile
        != RECEIPT_TEACHING_PROFILE_CONTINUATION_V1
    ):
        raise ValueError(
            "a non-default --receipt-teaching-profile requires --receipt-continuation"
        )
    if args.eos_generate_head_route != (
        args.receipt_teaching_profile in RECEIPT_GENERATE_HEAD_PROFILES
    ):
        raise ValueError(
            "--eos-generate-head-route and --receipt-teaching-profile "
            "a generated-head EOS profile must be selected together"
        )
    if args.termination_head_route != (
        args.receipt_teaching_profile in RECEIPT_TERMINATION_HEAD_PROFILES
    ):
        raise ValueError(
            "--termination-head-route and --receipt-teaching-profile "
            "a termination-head profile must be selected together"
        )
    if args.motor_retention_plateau_probation < 0:
        raise ValueError("--motor-retention-plateau-probation must be non-negative")
    if args.motor_retention_plateau_probation > 0 and not args.motor_retention_guard:
        raise ValueError(
            "--motor-retention-plateau-probation requires --motor-retention-guard"
        )
    if args.motor_retention_guard and not args.evaluate_only:
        if args.evaluation_case_limit is not None:
            raise ValueError(
                "--motor-retention-guard requires the complete heldout and regression "
                "surfaces; --evaluation-case-limit is a debug probe and cannot accept "
                "a checkpoint"
            )
        if args.tranche_steps is None or args.checkpoint_interval != args.tranche_steps:
            raise ValueError(
                "--motor-retention-guard requires checkpoint_interval == tranche_steps "
                "so no tentative checkpoint can advance the accepted pointer before the guard"
            )
    _seed_everything(args.seed)
    device = _device(args.device)
    config = candidate_a_config(
        n_heads=args.heads,
        n_layers=args.layers,
        ffn_dim=args.ffn_dim,
        page_size=args.page_size,
        dropout=0.0,
        generate_gate_bias=args.generate_gate_bias,
        receipt_continuation=bool(args.receipt_continuation),
        eos_generate_head_route=bool(args.eos_generate_head_route),
        termination_head_route=bool(args.termination_head_route),
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
    evaluation_case_id_by_episode = {
        case.episode.episode_id: case.case_id
        for item in standard_ffcs
        for case in item.teaching_cases
    }
    # The training lanes tag every step with the living-curriculum train manifest
    # id, so an evaluation transcript must use the same identifier or the two
    # surfaces disagree about which curriculum they are describing.
    evaluation_train_manifest_by_episode = {
        case.episode.episode_id: item.teaching_living_curriculum.train_manifest_id
        for item in standard_ffcs
        for case in item.teaching_cases
    }
    foundation_sequence_enabled = any(
        is_foundation_sequence_episode(case.episode)
        for item in standard_ffcs
        for case in item.teaching_cases
    )
    foundation_motor_enabled = any(
        is_foundation_motor_episode(case.episode)
        for item in standard_ffcs
        for case in item.teaching_cases
    )
    foundation_motor_v2_enabled = any(
        is_foundation_motor_v2_episode(case.episode)
        for item in standard_ffcs
        for case in item.teaching_cases
    )
    if foundation_motor_enabled and foundation_motor_v2_enabled:
        raise RuntimeError("one campaign cannot mix motor-v1 and motor-v2 teaching")
    if args.teach_multicell_copy and not foundation_motor_v2_enabled:
        raise RuntimeError("--teach-multicell-copy requires a motor-v2 curriculum")
    if args.receipt_continuation and not foundation_motor_v2_enabled:
        raise RuntimeError("--receipt-continuation requires a motor-v2 curriculum")
    if args.motor_retention_guard and not foundation_motor_v2_enabled:
        raise RuntimeError("--motor-retention-guard requires a motor-v2 curriculum")
    receipt_teach = (
        receipt_continuation_teach_profile(args.receipt_teaching_profile)
        if args.receipt_continuation
        else None
    )
    receipt_teach_id = (
        canonical_sha256(receipt_teach) if receipt_teach is not None else None
    )
    effective_objective_program_id = (
        foundation_motor_v2_objective_program_id(
            teach_multicell_copy=bool(args.teach_multicell_copy),
            receipt_continuation=bool(args.receipt_continuation),
            receipt_teaching_profile=args.receipt_teaching_profile,
        )
        if foundation_motor_v2_enabled
        else None
    )
    if foundation_motor_v2_enabled and not args.evaluate_only:
        # Fail closed before any compute: nine tranches were spent re-testing a
        # termination objective the autopsy had already rejected, because the
        # rejected route was reachable by simply omitting flags.  A rejected
        # route cannot produce information, so refusing to start is the only
        # outcome that does not waste allowance.  Read-only evaluation of an
        # existing bundle stays permitted, which is how a historical run is
        # still reproduced and judged.
        termination_route_rejection = foundation_motor_v2_termination_route_rejection(
            teach_multicell_copy=bool(args.teach_multicell_copy),
            receipt_continuation=bool(args.receipt_continuation),
            receipt_teaching_profile=args.receipt_teaching_profile,
        )
        if termination_route_rejection is not None:
            raise RuntimeError(termination_route_rejection)
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
            **(
                {"foundation_sequence_gate_policy_id": FOUNDATION_SEQUENCE_GATE_POLICY_ID}
                if foundation_sequence_enabled
                else {}
            ),
            **(
                {"foundation_motor_gate_policy_id": FOUNDATION_MOTOR_GATE_POLICY_ID}
                if foundation_motor_enabled
                else {}
            ),
            **(
                {"foundation_motor_v2_program_id": FOUNDATION_MOTOR_V2_PROGRAM_ID}
                if foundation_motor_v2_enabled
                else {}
            ),
            **(
                {
                    # Governance (guard/probation) is acceptance policy, not
                    # optimizer recipe: the contract family stays constant so
                    # enabling probation never forks candidate identity.  The
                    # contract actually used is recorded per decision.
                    "foundation_motor_v2_retention_contract_id": (
                        FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_ID
                    )
                }
                if args.motor_retention_guard
                else {}
            ),
            **(
                {
                    "effective_objective_program_id": effective_objective_program_id,
                    "teaching_overlay_ids": [
                        (
                            receipt_teach_id
                            if args.receipt_continuation
                            else COPY_ALIGNMENT_MULTICELL_TEACH_ID
                        )
                    ],
                    "receipt_continuation": bool(args.receipt_continuation),
                    "receipt_teaching_profile": (
                        args.receipt_teaching_profile
                        if args.receipt_continuation
                        else None
                    ),
                }
                if args.teach_multicell_copy or args.receipt_continuation
                else {}
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
        legacy_candidate_generation = "r64v2-" + canonical_sha256(candidate_identity)[:16]
        candidate_initialization = {
            "schema": "axon-reasoning-candidate-initialization-v1",
            "architecture_id": config.architecture_id,
            "seed": args.seed,
            "generate_gate_bias": args.generate_gate_bias,
        }
        candidate_initialization_id = canonical_sha256(candidate_initialization)
        candidate_identity_v3 = {
            **candidate_identity,
            "initialization_id": candidate_initialization_id,
        }
        candidate_generation_v3 = "r64v3-" + canonical_sha256(candidate_identity_v3)[:16]
        step_bundles = CandidateStepBundleCoordinator(args.state_root)
        if args.resume and step_bundles.latest_bundle(module_id, candidate_generation_v3):
            candidate_generation = candidate_generation_v3
            candidate_identity_version = "v3"
        elif args.resume and step_bundles.latest_bundle(module_id, legacy_candidate_generation):
            candidate_generation = legacy_candidate_generation
            candidate_identity_version = "legacy-v2"
        else:
            candidate_generation = candidate_generation_v3
            candidate_identity_version = "v3"
    descriptor = ParameterModuleDescriptor(
        module_id=module_id,
        organ_kind=OrganKind.REASONING_CORE,
        generation_id=base_generation,
        architecture=config.architecture_id,
        d_model=64,
        tags=("living", "private-soul", "complete-field", candidate_label),
    )
    base_identity = resolve_base_module(
        model,
        state_root=args.state_root,
        descriptor=descriptor,
        candidate_generation_id=candidate_generation,
    )
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
            base_module_identity=base_identity.to_canonical_dict(),
        )

    report: dict[str, Any] = {
        "schema": "axon-living-reasoning-smoke-report-v1",
        "device": str(device),
        "seed": args.seed,
        "preflight_only": bool(args.preflight_only),
        "architecture": model.architecture_report(),
        "motor_retention_guard_enabled": bool(args.motor_retention_guard),
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
        "candidate_identity_version": (
            "legacy-v1" if args.legacy_plan_v1 else candidate_identity_version
        ),
        "base_module_identity": base_identity.to_canonical_dict(),
        "candidate_initialization": (
            None
            if args.legacy_plan_v1
            else {
                **candidate_initialization,
                "initialization_id": candidate_initialization_id,
            }
        ),
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
            objective_program_id=effective_objective_program_id,
        )
        report["learning_policy"] = policy.to_canonical_dict()
        if args.motor_retention_guard:
            report["foundation_motor_v2_retention_contract_id"] = (
                FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_ID
            )
        if args.legacy_plan_v1:
            step_bundles = CandidateStepBundleCoordinator(args.state_root)
        latest_bundle = step_bundles.latest_bundle(module_id, candidate_generation)
        campaign_report_dir = args.state_root.resolve() / "training" / "reasoning" / candidate_generation
        probation_sidecar: dict[str, Any] | None = None
        probation_count = 0
        sidecar_path = _probation_sidecar_path(campaign_report_dir)
        if (
            args.motor_retention_guard
            and args.motor_retention_plateau_probation > 0
            and not args.evaluate_only
        ):
            probation_sidecar = _load_probation_sidecar(
                path=sidecar_path,
                candidate_generation=candidate_generation,
                learning_policy=policy.to_canonical_dict(),
                effective_objective_program_id=effective_objective_program_id,
                architecture_id=config.architecture_id,
                standard_ffcs_manifest_ids=[item.manifest_id for item in standard_ffcs],
                max_plateau_probation=args.motor_retention_plateau_probation,
            )
            if probation_sidecar is not None:
                probation_count = int(probation_sidecar["probation_count"])
        report["motor_retention_plateau_probation"] = args.motor_retention_plateau_probation
        if args.motor_retention_plateau_probation > 0:
            report["foundation_motor_v2_retention_contract_id"] = (
                FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2_ID
            )
        if not args.legacy_plan_v1:
            _write_immutable_json(
                campaign_report_dir / "candidate_initialization.json",
                {
                    **candidate_initialization,
                    "initialization_id": candidate_initialization_id,
                    "candidate_generation_id": candidate_generation,
                    "candidate_identity_version": candidate_identity_version,
                },
            )
        foundation_motor_v2_training_stage = None
        foundation_motor_v2_program_complete_before_run = False
        if foundation_motor_v2_enabled:
            (
                foundation_motor_v2_training_stage,
                foundation_motor_v2_program_complete_before_run,
            ) = _foundation_motor_v2_stage_from_reports(campaign_report_dir)
            report.update(
                {
                    "foundation_motor_v2_program_id": FOUNDATION_MOTOR_V2_PROGRAM_ID,
                    "effective_objective_program_id": effective_objective_program_id,
                    "foundation_motor_v2_training_stage": foundation_motor_v2_training_stage,
                    "foundation_motor_v2_stage_policy": dict(
                        foundation_motor_v2_stage_policy(
                            foundation_motor_v2_training_stage
                        )
                    ),
                    "teach_multicell_copy": {
                        "enabled": bool(args.teach_multicell_copy or args.receipt_continuation),
                        **(
                            dict(
                                receipt_teach
                                if args.receipt_continuation
                                else COPY_ALIGNMENT_MULTICELL_TEACH
                            )
                            if (args.teach_multicell_copy or args.receipt_continuation)
                            and foundation_motor_v2_training_stage == "copy_alignment"
                            else {}
                        ),
                    },
                    "receipt_continuation": bool(args.receipt_continuation),
                    "receipt_teaching_profile": (
                        args.receipt_teaching_profile
                        if args.receipt_continuation
                        else None
                    ),
                }
            )
            if foundation_motor_v2_program_complete_before_run and not args.evaluate_only:
                raise RuntimeError(
                    "foundation motor v2 program is already complete; new optimizer work denied"
                )
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
                    probation_sidecar=probation_sidecar,
                )
            anchor_step = (
                int(probation_sidecar["probationary_step"])
                if probation_sidecar is not None
                else (0 if latest_bundle is None else latest_bundle.step)
            )
            base_global_step = anchor_step
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
                    (
                        f"foundation motor v2 stage {foundation_motor_v2_training_stage}; "
                        "renewable base-zero training tranche"
                    )
                    if foundation_motor_v2_training_stage is not None
                    and parent_bundle_id is None
                    else (
                        f"foundation motor v2 stage {foundation_motor_v2_training_stage}; "
                        f"renewable continuation tranche (parent bundle {parent_bundle_id[:16]})"
                    )
                    if foundation_motor_v2_training_stage is not None
                    else "renewable base-zero training tranche"
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
            if probation_sidecar is not None:
                # Continue the probationary branch: the accepted chain (and
                # its Soul HEAD) stays at the confirmed parent, while
                # parameters/optimizer resume from the held probationary
                # checkpoint.  Verify the payload survived store pruning
                # before committing to the branch.
                probationary_record = CandidateCheckpointRecord.from_mapping(
                    probation_sidecar["probationary_checkpoint"]
                )
                step_bundles.trainer_store.load_verified_candidate_checkpoint(
                    probationary_record
                )
                session.restore_checkpoint(probationary_record)
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
                    parent_global_step=_continuation_parent_global_step(
                        latest_bundle.step, probation_sidecar
                    ),
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

        # Training and evaluation must score the same effective objective.
        # Using the bare stage policy here made copy_alignment loss blind to
        # payload collapse even though the v3 teaching overlay trained payload
        # content and generated-head EOS.
        train_component_weights = (
            None
            if foundation_motor_v2_training_stage is None
            else dict(
                foundation_motor_v2_stage_policy(foundation_motor_v2_training_stage)[
                    "component_weights"
                ]
            )
        )
        train_position_reduction = "mean"
        if (args.teach_multicell_copy or args.receipt_continuation) and foundation_motor_v2_training_stage is not None:
            train_component_weights = (
                apply_receipt_continuation_teach_weights(
                    train_component_weights or {},
                    training_stage=foundation_motor_v2_training_stage,
                    receipt_teaching_profile=args.receipt_teaching_profile,
                )
                if args.receipt_continuation
                else apply_copy_alignment_multicell_teach_weights(
                    train_component_weights or {},
                    training_stage=foundation_motor_v2_training_stage,
                )
            )
            if foundation_motor_v2_training_stage == "copy_alignment":
                train_position_reduction = str(
                    (
                        receipt_teach
                        if args.receipt_continuation
                        else COPY_ALIGNMENT_MULTICELL_TEACH
                    )["alignment_position_reduction"]
                )
        report["evaluation_component_weights"] = train_component_weights
        report["evaluation_alignment_position_reduction"] = train_position_reduction
        payload_eos_weight = float(
            4.0 if receipt_teach is None else receipt_teach.get("payload_eos_weight", 4.0)
        )
        report["payload_eos_weight"] = payload_eos_weight
        # The v6 balanced-termination fix is profile-carried so training and
        # evaluation score the same effective objective and the campaign
        # identity changes exactly when the supervision semantics change.
        termination_continue_supervision = bool(
            receipt_teach is not None
            and receipt_teach.get("termination_continue_supervision", False)
        )
        report["termination_continue_supervision"] = termination_continue_supervision

        def evaluate_candidate(evaluation_soul: Any | None = None) -> dict[str, Any]:
            session.candidate_module.eval()
            observed_soul = (
                soul_branch.load_head() if evaluation_soul is None else evaluation_soul
            )
            qa_rows: list[dict[str, Any]] = []
            exact_rows = [
                evaluate_living_episode(
                    session.candidate_module,
                    episode,
                    observed_soul,
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                    transcript_sink=qa_rows,
                    transcript_sink_cap=None,
                )
                for episode in heldout_episodes
            ]
            sequential_rows = [
                evaluate_sequential_case(
                    session.candidate_module,
                    case,
                    observed_soul,
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
                        observed_soul,
                        core_id=module_id,
                        parameter_generation=candidate_generation,
                        component_weights=(
                            train_component_weights
                        ),
                        alignment_position_reduction=train_position_reduction,
                        payload_eos_weight=payload_eos_weight,
                        termination_continue_supervision=termination_continue_supervision,
                    )
                    exact_losses.append(float(loss.item()))
                for case in sequential_heldout:
                    loss, _unrolls, _soul = sequential_living_objective(
                        session.candidate_module,
                        case,
                        observed_soul,
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
                    observed_soul,
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                )
                for episode in foundation_regression_episodes
            ]
            motor_regression_episodes = tuple(
                episode
                for episode in regression_episodes
                if is_foundation_motor_episode(episode)
            )
            motor_regression_rows = [
                evaluate_living_episode(
                    session.candidate_module,
                    episode,
                    observed_soul,
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                )
                for episode in motor_regression_episodes
            ]
            motor_v2_regression_episodes = tuple(
                episode
                for episode in regression_episodes
                if is_foundation_motor_v2_episode(episode)
            )
            motor_v2_regression_rows = [
                evaluate_living_episode(
                    session.candidate_module,
                    episode,
                    observed_soul,
                    core_id=module_id,
                    parameter_generation=candidate_generation,
                )
                for episode in motor_v2_regression_episodes
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
                def total(name: str) -> float:
                    return sum(float(row.get(name, 0.0)) for row in rows)

                def merged_histogram(name: str) -> dict[str, int]:
                    """Merge per-case answer histograms into one surface histogram.

                    Per-case rates cannot be averaged into a constant-emitter
                    baseline: the strongest fixed answer must be found across the
                    whole surface, not inside each evaluated case.
                    """

                    return merge_surface_histogram(rows, name)

                typed_target_histogram = merged_histogram(
                    "constant_typed_emission_target_histogram"
                )
                payload_target_histogram = merged_histogram(
                    "constant_payload_transport_target_histogram"
                )
                constant_floors = constant_baseline_floors(
                    typed_target_histogram,
                    payload_target_histogram,
                    supervised_phase_count=supervised_phase_count,
                    payload_supervised_phase_count=payload_supervised_phase_count,
                )

                payload_content_count = total("payload_teacher_forced_content_count")
                payload_content_correct = total("payload_teacher_forced_content_correct")
                payload_eos_count = total("payload_teacher_forced_eos_count")
                payload_eos_correct = total("payload_teacher_forced_eos_correct")
                alignment_position_count = total("alignment_position_count")
                alignment_copy_gate_count = total("alignment_copy_gate_count")
                alignment_eos_gate_count = total("alignment_eos_gate_count")
                decision_target_counts = [
                    sum(int(row.get("decision_target_counts", [0] * len(ReasoningDecision))[index]) for row in rows)
                    for index in range(len(ReasoningDecision))
                ]
                decision_correct_by_target = [
                    sum(int(row.get("decision_correct_by_target", [0] * len(ReasoningDecision))[index]) for row in rows)
                    for index in range(len(ReasoningDecision))
                ]
                operation_target_counts = [
                    sum(int(row.get("operation_target_counts", [0] * len(ReasoningOperationKind))[index]) for row in rows)
                    for index in range(len(ReasoningOperationKind))
                ]
                operation_correct_by_target = [
                    sum(int(row.get("operation_correct_by_target", [0] * len(ReasoningOperationKind))[index]) for row in rows)
                    for index in range(len(ReasoningOperationKind))
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
                    "constant_typed_emission_target_histogram": dict(
                        sorted(typed_target_histogram.items())
                    ),
                    "constant_payload_transport_target_histogram": dict(
                        sorted(payload_target_histogram.items())
                    ),
                    **constant_floors,
                    "payload_teacher_forced_token_accuracy": (
                        payload_token_correct / max(1, payload_token_count)
                    ),
                    "constant_payload_token_accuracy_floor": (
                        max(payload_target_counts) / max(1, payload_token_count)
                    ),
                    "payload_teacher_forced_content_accuracy": (
                        payload_content_correct / max(1.0, payload_content_count)
                    ),
                    "payload_teacher_forced_content_count": payload_content_count,
                    "constant_payload_content_accuracy_floor": (
                        max(payload_target_counts[:-1])
                        / max(1.0, payload_content_count)
                    ),
                    "payload_teacher_forced_eos_accuracy": (
                        payload_eos_correct / max(1.0, payload_eos_count)
                    ),
                    "payload_teacher_forced_eos_count": payload_eos_count,
                    "alignment_position_accuracy": total("alignment_position_correct")
                    / max(1.0, alignment_position_count),
                    "alignment_position_count": alignment_position_count,
                    "alignment_copy_gate_accuracy": total("alignment_copy_gate_correct")
                    / max(1.0, alignment_copy_gate_count),
                    "alignment_copy_gate_count": alignment_copy_gate_count,
                    "alignment_eos_gate_accuracy": total("alignment_eos_gate_correct")
                    / max(1.0, alignment_eos_gate_count),
                    "alignment_eos_gate_count": alignment_eos_gate_count,
                    "decision_accuracy": total("decision_correct")
                    / max(1.0, supervised_phase_count),
                    "decision_target_counts": decision_target_counts,
                    "decision_correct_by_target": decision_correct_by_target,
                    "operation_accuracy": total("operation_correct")
                    / max(1.0, total("operation_count")),
                    "operation_target_counts": operation_target_counts,
                    "operation_correct_by_target": operation_correct_by_target,
                    "region_accuracy": total("region_correct")
                    / max(1.0, total("region_count")),
                    "start_accuracy": total("start_correct")
                    / max(1.0, total("start_count")),
                    "end_accuracy": total("end_correct")
                    / max(1.0, total("end_count")),
                }

            combined_rows = exact_rows + sequential_rows
            result = aggregate(
                combined_rows,
                exact_losses + sequential_losses,
            )
            result["counterfactuals"] = living_source_counterfactuals(
                session.candidate_module,
                heldout_episodes[0],
                observed_soul,
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
            result["foundation_motor_heldout_probe"] = foundation_motor_probe(
                heldout_episodes,
                exact_rows,
            )
            result["foundation_motor_regression_probe"] = foundation_motor_probe(
                motor_regression_episodes,
                motor_regression_rows,
            )
            result["foundation_motor_v2_heldout_probe"] = foundation_motor_v2_probe(
                heldout_episodes,
                exact_rows,
                training_stage=foundation_motor_v2_training_stage,
            )
            result["foundation_motor_v2_regression_probe"] = foundation_motor_v2_probe(
                motor_v2_regression_episodes,
                motor_v2_regression_rows,
                training_stage=foundation_motor_v2_training_stage,
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
            # Attribute every transcript to its own curriculum and case. The
            # aggregate `isolated_manifest_evaluations` table can collapse to a
            # single manifest when the evaluated surface is limited, and nothing
            # else in the live surface would reveal that.
            qa_transcripts: list[dict[str, Any]] = []
            qa_enriched: list[dict[str, Any]] = []
            for row in qa_rows:
                row_episode_id = str(row.get("episode_id") or "")
                qa_enriched.append(
                    {
                        **row,
                        "family": evaluation_family_by_episode.get(row_episode_id),
                        "case_id": evaluation_case_id_by_episode.get(row_episode_id),
                        "source_manifest_id": evaluation_train_manifest_by_episode.get(
                            row_episode_id
                        ),
                        "manifest_id": evaluation_manifest_by_episode.get(row_episode_id),
                    }
                )
            qa_transcripts = select_qa_transcript_rows(qa_enriched, limit=6)
            result["qa_transcripts"] = qa_transcripts
            result["qa_transcript_coverage"] = {
                "payload_phase_count": len(qa_enriched),
                "selected_count": len(qa_transcripts),
                "families": sorted(
                    {str(row.get("family") or "unknown") for row in qa_transcripts}
                ),
            }
            return result

        cached_final_evaluation: dict[str, Any] | None = None
        accepted_parent_evaluation: dict[str, Any] | None = None
        accepted_parent_evaluation_report_id: str | None = None

        def _find_report_evaluation(
            *, step: int, checkpoint_id: str, soul_id: str
        ) -> tuple[dict[str, Any] | None, str | None]:
            for prior_path in sorted(
                campaign_report_dir.glob("segment_*.json"), reverse=True
            ):
                prior = json.loads(prior_path.read_text(encoding="utf-8"))
                prior_report_id = prior.get("report_id")
                if prior_report_id is None or canonical_sha256(
                    {key: value for key, value in prior.items() if key != "report_id"}
                ) != prior_report_id:
                    raise RuntimeError(f"campaign report identity mismatch: {prior_path}")
                if (
                    prior.get("candidate_generation_id") == candidate_generation
                    and prior.get("segment_end_step") == step
                    and prior.get("final_checkpoint_id") == checkpoint_id
                    and prior.get("complete_heldout_evaluation") is True
                    and prior.get("complete_regression_evaluation") is True
                    and prior.get("effective_objective_program_id")
                    == effective_objective_program_id
                    and prior.get("learning_policy") == policy.to_canonical_dict()
                    and prior.get("evaluation_component_weights")
                    == train_component_weights
                    and prior.get("evaluation_alignment_position_reduction")
                    == train_position_reduction
                    and prior.get("payload_eos_weight") == payload_eos_weight
                    and prior.get("termination_continue_supervision", False)
                    == termination_continue_supervision
                    and prior.get("standard_ffcs_manifest_ids")
                    == list(item.manifest_id for item in standard_ffcs)
                    and prior.get("architecture", {}).get("architecture_id")
                    == config.architecture_id
                    and prior.get("soul_promotion_plan", {}).get("candidate_soul_id")
                    == soul_id
                ):
                    evaluation = prior.get("final_evaluation")
                    if not isinstance(evaluation, dict):
                        raise RuntimeError(
                            f"campaign report evaluation is malformed: {prior_path}"
                        )
                    return evaluation, prior_report_id
            return None, None

        if latest_bundle is not None:
            (
                accepted_parent_evaluation,
                accepted_parent_evaluation_report_id,
            ) = _find_report_evaluation(
                step=latest_bundle.step,
                checkpoint_id=latest_bundle.checkpoint_id,
                soul_id=latest_bundle.after_soul_id,
            )
        probationary_evaluation: dict[str, Any] | None = None
        if probation_sidecar is not None:
            probationary_record = CandidateCheckpointRecord.from_mapping(
                probation_sidecar["probationary_checkpoint"]
            )
            # The probationary report describes the held state; its Soul is
            # the confirmed parent's HEAD because probation never promotes.
            probationary_evaluation, _ = _find_report_evaluation(
                step=probationary_record.step,
                checkpoint_id=probationary_record.checkpoint_id,
                soul_id=latest_bundle.after_soul_id,
            )
            if probationary_evaluation is None:
                raise RuntimeError(
                    "probation sidecar is active but its probationary report is "
                    "missing or fails identity checks; refusing to guess"
                )
        report["accepted_parent_evaluation_report_id"] = (
            accepted_parent_evaluation_report_id
        )
        if progress is not None:
            progress.emit(
                "evaluating", phase="initial", global_step=(0 if latest_bundle is None else latest_bundle.step)
            )
        initial_evaluation = (
            probationary_evaluation
            if probationary_evaluation is not None
            else accepted_parent_evaluation
            if accepted_parent_evaluation is not None
            else evaluate_candidate()
        )
        if args.evaluate_only:
            # The accepted checkpoint cannot change during a read-only run;
            # reuse the exact result instead of paying for the full surface a
            # second time.
            cached_final_evaluation = initial_evaluation
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
                constant_typed_emission_exact_floor=initial_evaluation[
                    "constant_typed_emission_exact_floor"
                ],
                constant_payload_transport_exact_floor=initial_evaluation[
                    "constant_payload_transport_exact_floor"
                ],
                evaluated_case_count=initial_evaluation["evaluated_case_count"],
                qa_transcripts=initial_evaluation.get("qa_transcripts", [])[:6],
                qa_transcript_coverage=initial_evaluation.get("qa_transcript_coverage"),
                **_compact_motor_v2_probes(initial_evaluation),
            )
        prior_reports = sorted(campaign_report_dir.glob("segment_*.json"))
        campaign_baseline_evaluation = (
            initial_evaluation
            if not prior_reports
            else json.loads(prior_reports[0].read_text(encoding="utf-8"))["initial_evaluation"]
        )
        start_step = (
            int(probation_sidecar["probationary_step"])
            if probation_sidecar is not None
            else (0 if latest_bundle is None else latest_bundle.step)
        )
        rollback_checkpoint = (
            (
                CandidateCheckpointRecord.from_mapping(
                    probation_sidecar["confirmed_checkpoint"]
                )
                if probation_sidecar is not None
                else step_bundles.checkpoint_for_bundle(latest_bundle)
                if latest_bundle is not None
                else session.checkpoint(include_optimizer=True)
            )
            if args.motor_retention_guard
            else None
        )
        accepted_checkpoint = (
            step_bundles.checkpoint_for_bundle(latest_bundle)
            if latest_bundle is not None
            else None
        )
        # The guard's comparison reference is always the last CONFIRMED
        # accepted parent; during probation that is the sidecar reference,
        # not the probationary state's own (behaviorally identical) eval.
        retained_evaluation = (
            probation_sidecar["reference_evaluation"]
            if probation_sidecar is not None
            else initial_evaluation
        )
        retention_decisions: list[dict[str, Any]] = []
        retention_guard_stop: dict[str, Any] | None = None
        retention_probation_held = False
        sync_hook.set_base_step(start_step)
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
            foundation_motor_v2_training_stage=foundation_motor_v2_training_stage,
            teach_multicell_copy=bool(
                args.teach_multicell_copy or args.receipt_continuation
            ),
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
                    component_weights=train_component_weights,
                    alignment_position_reduction=train_position_reduction,
                    payload_eos_weight=payload_eos_weight,
                    termination_continue_supervision=termination_continue_supervision,
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
            retention_decision = None
            probation_pending = False
            if checkpoint_due:
                checkpoint = session.checkpoint(include_optimizer=True)
                checkpoints.append(checkpoint)
                if args.motor_retention_guard:
                    tentative_evaluation = evaluate_candidate(ephemeral_soul)
                    if args.motor_retention_plateau_probation > 0:
                        retention_decision = decide_foundation_motor_v2_checkpoint_retention_v2(
                            training_stage=foundation_motor_v2_training_stage,
                            accepted_heldout_probe=retained_evaluation.get(
                                "foundation_motor_v2_heldout_probe"
                            ),
                            accepted_regression_probe=retained_evaluation.get(
                                "foundation_motor_v2_regression_probe"
                            ),
                            candidate_heldout_probe=tentative_evaluation.get(
                                "foundation_motor_v2_heldout_probe"
                            ),
                            candidate_regression_probe=tentative_evaluation.get(
                                "foundation_motor_v2_regression_probe"
                            ),
                            complete_heldout=complete_heldout_evaluation,
                            complete_regression=complete_regression_evaluation,
                        )
                        retention_action = resolve_retention_action(
                            retention_decision,
                            probation_count=probation_count,
                            max_plateau_probation=args.motor_retention_plateau_probation,
                        )
                    else:
                        retention_decision = decide_foundation_motor_v2_checkpoint_retention(
                            training_stage=foundation_motor_v2_training_stage,
                            accepted_heldout_probe=retained_evaluation.get(
                                "foundation_motor_v2_heldout_probe"
                            ),
                            accepted_regression_probe=retained_evaluation.get(
                                "foundation_motor_v2_regression_probe"
                            ),
                            candidate_heldout_probe=tentative_evaluation.get(
                                "foundation_motor_v2_heldout_probe"
                            ),
                            candidate_regression_probe=tentative_evaluation.get(
                                "foundation_motor_v2_regression_probe"
                            ),
                            complete_heldout=complete_heldout_evaluation,
                            complete_regression=complete_regression_evaluation,
                        )
                        retention_action = (
                            "accept" if retention_decision["passed"] else "rollback"
                        )
                    retention_decisions.append(
                        {**retention_decision, "action": retention_action}
                    )
                    if retention_action == "accept":
                        cached_final_evaluation = tentative_evaluation
                    elif retention_action == "probate":
                        # Hold this exact optimizer state without promotion.
                        # The sidecar - not the accepted bundle chain - carries
                        # the resumable probationary branch; the chain and its
                        # Soul HEAD stay at the confirmed parent.
                        # The tentative evaluation already measured this exact
                        # state at the boundary and the session is unchanged;
                        # reuse it instead of re-paying the complete surface.
                        cached_final_evaluation = tentative_evaluation
                        probation_pending = True
                        retention_probation_held = True
                        _write_probation_sidecar(
                            sidecar_path,
                            {
                                "schema": PROBATION_SIDECAR_SCHEMA,
                                "candidate_generation_id": candidate_generation,
                                "confirmed_checkpoint": rollback_checkpoint.to_canonical_dict(),
                                "confirmed_step": rollback_checkpoint.step,
                                "probationary_checkpoint": checkpoint.to_canonical_dict(),
                                "probationary_step": step + 1,
                                "probation_count": probation_count + 1,
                                "max_plateau_probation": args.motor_retention_plateau_probation,
                                "reference_evaluation": retained_evaluation,
                                "reference_report_id": accepted_parent_evaluation_report_id,
                                "learning_policy": policy.to_canonical_dict(),
                                "effective_objective_program_id": effective_objective_program_id,
                                "architecture_id": config.architecture_id,
                                "standard_ffcs_manifest_ids": [
                                    item.manifest_id for item in standard_ffcs
                                ],
                            },
                        )
                    else:
                        if rollback_checkpoint is None:  # pragma: no cover - guarded setup
                            raise RuntimeError("motor retention guard has no rollback checkpoint")
                        session.restore_checkpoint(rollback_checkpoint)
                        ephemeral_soul = soul_branch.load_head()
                        cached_final_evaluation = retained_evaluation
                        _clear_probation_sidecar(sidecar_path)
                        retention_guard_stop = {
                            "attempted_step": step + 1,
                            "tentative_checkpoint_id": checkpoint.checkpoint_id,
                            "decision": retention_decision,
                            "action": retention_action,
                            "probation_count": probation_count,
                        }
                if retention_guard_stop is None and not probation_pending:
                    accepted_bundle = step_bundles.accept_step(
                        optimization_receipt=optimizer_receipt,
                        checkpoint=checkpoint,
                        soul_manifest=soul_manifest,
                        transitions=tuple(segment_transitions),
                    )
                    accepted_checkpoint = checkpoint
                    if args.motor_retention_guard:
                        rollback_checkpoint = checkpoint
                        retained_evaluation = tentative_evaluation
                        if probation_sidecar is not None:
                            # An improvement after probation confirms the branch:
                            # the chain pointer now carries the probationary
                            # lineage forward; the sidecar is done.
                            _clear_probation_sidecar(sidecar_path)
                            probation_sidecar = None
                            probation_count = 0
                    if soul_branch.load_head().soul_id != ephemeral_soul.soul_id:
                        raise RuntimeError("accepted checkpoint segment and ephemeral Soul disagree")
                    accepted_segment_receipt_ids = list(segment_receipt_ids)
                    segment_transitions.clear()
                    segment_receipt_ids.clear()
                    # Accepted checkpoint boundary: hand the new artifacts to the
                    # daemon sync worker; GPU compute continues immediately.
                    sync_hook.boundary(step + 1)
            material_id = material.episode_id if kind == "episode" else material.case_id
            material_label = getattr(material, "label", None) or getattr(
                getattr(material, "episode", None), "label", None
            )
            steps.append(
                {
                    "step": step + 1,
                    "material_kind": kind,
                    "curriculum_lane": lane_name,
                    "source_manifest_id": source_manifest_id,
                    "material_id": material_id,
                    "loss": captured["loss"],
                    "optimization_receipt_id": optimizer_receipt.receipt_id,
                    "accepted_step_bundle_id": (None if accepted_bundle is None else accepted_bundle.bundle_id),
                    "accepted_segment_optimizer_receipt_ids": accepted_segment_receipt_ids,
                    "soul_receipt_ids": ([] if accepted_bundle is None else list(accepted_bundle.soul_receipt_ids)),
                    "soul_id": ephemeral_soul.soul_id,
                    "checkpoint_id": None if checkpoint is None else checkpoint.checkpoint_id,
                    "retention_decision": retention_decision,
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
                    material_id=material_id,
                    material_label=material_label,
                    source_manifest_id=source_manifest_id,
                    wall_seconds=wall_seconds,
                    checkpoint_id=None if checkpoint is None else checkpoint.checkpoint_id,
                    accepted_step_bundle_id=(None if accepted_bundle is None else accepted_bundle.bundle_id),
                    **_continuation_step_telemetry(captured["phase_metrics"]),
                )
            if retention_guard_stop is not None:
                break
        accepted_end_step = session.step_index
        attempted_end_step = steps[-1]["step"] if steps else start_step
        if progress is not None:
            progress.emit("evaluating", phase="final", global_step=accepted_end_step)
        final_evaluation = (
            cached_final_evaluation
            if cached_final_evaluation is not None
            else evaluate_candidate()
        )
        if progress is not None:
            progress.emit(
                "evaluated",
                phase="final",
                global_step=accepted_end_step,
                heldout_mean_loss=final_evaluation["heldout_mean_loss"],
                typed_emission_exact_rate=final_evaluation["typed_emission_exact_rate"],
                payload_transport_exact_rate=final_evaluation["payload_transport_exact_rate"],
                payload_teacher_forced_token_accuracy=final_evaluation[
                    "payload_teacher_forced_token_accuracy"
                ],
                constant_payload_token_accuracy_floor=final_evaluation[
                    "constant_payload_token_accuracy_floor"
                ],
                constant_typed_emission_exact_floor=final_evaluation[
                    "constant_typed_emission_exact_floor"
                ],
                constant_payload_transport_exact_floor=final_evaluation[
                    "constant_payload_transport_exact_floor"
                ],
                evaluated_case_count=final_evaluation["evaluated_case_count"],
                qa_transcripts=final_evaluation.get("qa_transcripts", [])[:6],
                qa_transcript_coverage=final_evaluation.get("qa_transcript_coverage"),
                **_compact_motor_v2_probes(final_evaluation),
            )
        counterfactuals_passed = all(value > 1e-8 for value in final_evaluation["counterfactuals"].values())
        task_gate_passed = (
            complete_heldout_evaluation
            and final_evaluation["heldout_mean_loss"] < campaign_baseline_evaluation["heldout_mean_loss"]
            and final_evaluation["payload_teacher_forced_token_accuracy"]
            > final_evaluation["constant_payload_token_accuracy_floor"]
            and counterfactuals_passed
        )
        nonzero_exact_output = nonzero_exact_output_observed(final_evaluation)
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
        foundation_motor_gate = decide_foundation_motor_mastery(
            heldout_probe=final_evaluation.get("foundation_motor_heldout_probe"),
            regression_probe=final_evaluation.get("foundation_motor_regression_probe"),
            evaluation=final_evaluation,
            complete_heldout=complete_heldout_evaluation,
            complete_regression=complete_regression_evaluation,
        )
        foundation_motor_v2_stage_gate = (
            None
            if foundation_motor_v2_training_stage is None
            else decide_foundation_motor_v2_stage(
                training_stage=foundation_motor_v2_training_stage,
                heldout_probe=final_evaluation.get(
                    "foundation_motor_v2_heldout_probe"
                ),
                regression_probe=final_evaluation.get(
                    "foundation_motor_v2_regression_probe"
                ),
                complete_heldout=complete_heldout_evaluation,
                complete_regression=complete_regression_evaluation,
                receipt_continuation=bool(args.receipt_continuation),
                receipt_teaching_profile=args.receipt_teaching_profile,
                eos_generate_head_route=bool(args.eos_generate_head_route),
                termination_head_route=bool(args.termination_head_route),
            )
        )
        foundation_motor_v2_program_complete = bool(
            foundation_motor_v2_stage_gate is not None
            and foundation_motor_v2_stage_gate["passed"]
            and foundation_motor_v2_training_stage
            == FOUNDATION_MOTOR_V2_STAGE_ORDER[-1]
        )
        foundation_gates = tuple(
            gate
            for gate in (foundation_motor_gate, foundation_sequence_gate)
            if gate is not None
        )
        curriculum_stage_complete = (
            task_gate_passed
            and nonzero_exact_output
            and (
                foundation_motor_v2_program_complete
                if foundation_motor_v2_enabled
                else True
            )
            and (
                all(bool(gate["passed"]) for gate in foundation_gates)
                if foundation_gates
                else metric_surface_complete
            )
        )
        exact_serving_gate = exact_serving_gate_passed(
            final_evaluation,
            curriculum_stage_complete=curriculum_stage_complete,
            complete_heldout=complete_heldout_evaluation,
            complete_regression=complete_regression_evaluation,
            tournament_metric_surface_complete=metric_surface_complete,
        )
        if retention_probation_held:
            # The accepted chain stays at the confirmed parent; the report
            # describes the held probationary state so the next launch can
            # reuse this evaluation instead of re-paying the surface.
            final_checkpoint = CandidateCheckpointRecord.from_mapping(
                json.loads(sidecar_path.read_text(encoding="utf-8"))[
                    "probationary_checkpoint"
                ]
            )
            if (
                foundation_motor_v2_stage_gate is not None
                and foundation_motor_v2_stage_gate["passed"]
            ):
                # A passed stage gate on a probation-held tentative is the
                # probation design's "behavior improved" event: promote the
                # probationary checkpoint onto the accepted chain so the
                # landmark and the report bind to accepted state. Without
                # this the segment cannot publish (the landmark requires an
                # accepted boundary) and a passing metric that the retention
                # contract does not track would plateau-loop forever.
                if int(final_checkpoint.step) != int(attempted_end_step):
                    raise RuntimeError(
                        "stage-gate promotion: probationary step does not reach the segment end"
                    )
                promoted_bundle = step_bundles.accept_step(
                    optimization_receipt=optimizer_receipt,
                    checkpoint=final_checkpoint,
                    soul_manifest=soul_manifest,
                    transitions=tuple(segment_transitions),
                )
                if soul_branch.load_head().soul_id != ephemeral_soul.soul_id:
                    raise RuntimeError(
                        "stage-gate promotion: accepted checkpoint segment and ephemeral Soul disagree"
                    )
                # Stamp the final step with the promoted bundle so the
                # tranche-consumption proof in _prior_consumed_tranche_id can
                # bind this segment's resource tranche to the accepted chain
                # (the acceptance happened at segment end, after the per-step
                # records were built).
                steps[-1]["accepted_step_bundle_id"] = promoted_bundle.bundle_id
                steps[-1]["accepted_segment_optimizer_receipt_ids"] = list(
                    segment_receipt_ids
                )
                steps[-1]["soul_receipt_ids"] = list(promoted_bundle.soul_receipt_ids)
                accepted_checkpoint = final_checkpoint
                rollback_checkpoint = final_checkpoint
                retained_evaluation = final_evaluation
                _clear_probation_sidecar(sidecar_path)
                retention_probation_held = False
                sync_hook.boundary(int(attempted_end_step))
        else:
            final_checkpoint = accepted_checkpoint or rollback_checkpoint
        if final_checkpoint is None:
            latest_final_bundle = step_bundles.latest_bundle(module_id, candidate_generation)
            if latest_final_bundle is None:
                raise RuntimeError("training segment has no recoverable final checkpoint")
            final_checkpoint = step_bundles.checkpoint_for_bundle(latest_final_bundle)
        final_checkpoint_id = final_checkpoint.checkpoint_id
        foundation_motor_v2_landmark_id = None
        if (
            foundation_motor_v2_stage_gate is not None
            and foundation_motor_v2_stage_gate["passed"]
        ):
            landmark_bundle = step_bundles.latest_bundle(module_id, candidate_generation)
            if (
                landmark_bundle is None
                or landmark_bundle.checkpoint_id != final_checkpoint_id
            ):
                raise RuntimeError(
                    "passed motor stage has no matching accepted checkpoint+Soul boundary"
                )
            foundation_motor_v2_landmark_id = step_bundles.mark_landmark(
                landmark_bundle,
                label=f"foundation-motor-v2:{foundation_motor_v2_training_stage}:passed",
                evidence_ids=(
                    foundation_motor_v2_stage_gate["decision_id"],
                    effective_objective_program_id,
                ),
            )
        if curriculum_stage_complete:
            lifecycle_event = session.complete(
                reason="complete curriculum-stage gate passed; serving activation remains separate"
            )
        else:
            lifecycle_event = session.pause(
                reason=(
                    "tentative checkpoint failed the motor retention contract; "
                    "candidate restored to its prior recoverable boundary"
                    if retention_guard_stop is not None
                    else "execution segment ended at an exact accepted checkpoint; "
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
                "segment_end_step": accepted_end_step,
                "attempted_segment_end_step": attempted_end_step,
                "motor_retention_decisions": retention_decisions,
                "motor_retention_guard_stop": retention_guard_stop,
                "motor_retention_probation_held": retention_probation_held,
                "motor_retention_probation": (
                    json.loads(sidecar_path.read_text(encoding="utf-8"))
                    if sidecar_path.exists()
                    else None
                ),
                "campaign_max_steps": (args.max_steps if args.legacy_plan_v1 else None),
                "evaluation_only": bool(args.evaluate_only),
                "campaign_complete": curriculum_stage_complete,
                "curriculum_stage_complete": curriculum_stage_complete,
                "legacy_plan_envelope_exhausted": (accepted_end_step >= args.max_steps if args.legacy_plan_v1 else None),
                "resource_tranche_consumed": (
                    tranche is not None and attempted_end_step == tranche.final_global_step
                ),
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
                "nonzero_exact_output_observed": nonzero_exact_output,
                "nonzero_exact_output_policy": (
                    "free-running typed emission and complete Unicode payload exact rates "
                    "both exceed their constant zero floors; progress signal only"
                ),
                "exact_serving_gate_passed": exact_serving_gate,
                "exact_serving_gate_policy": (
                    "curriculum stage complete; heldout, regression, and tournament metric "
                    "surfaces complete; free-running typed emission and complete Unicode "
                    "payload transport both exactly 1.0"
                ),
                "serving_promotion_claimed": False,
                "tournament_metrics": tournament_metrics,
                "tournament_metric_computation": (tournament_metric_computation.to_canonical_dict()),
                "missing_tournament_metrics": list(tournament_metric_computation.missing_metrics),
                "tournament_metric_surface_complete": metric_surface_complete,
                "foundation_sequence_gate": foundation_sequence_gate,
                "foundation_motor_gate": foundation_motor_gate,
                "foundation_motor_v2_stage_gate": foundation_motor_v2_stage_gate,
                "foundation_motor_v2_landmark_id": foundation_motor_v2_landmark_id,
                "foundation_motor_v2_program_complete": foundation_motor_v2_program_complete,
            }
        )

    report["report_id"] = canonical_sha256(report)
    if retention_guard_stop is not None:
        stop_action = retention_guard_stop.get("action")
        stop_state = (retention_guard_stop.get("decision") or {}).get("state")
        suffix = (
            "probation_exhausted"
            if stop_action == "rollback" and stop_state == "plateau"
            else "retention_rejected"
        )
        report_path = campaign_report_dir / (
            f"attempt_{start_step + 1:09d}_{attempted_end_step:09d}_{suffix}.json"
        )
    elif args.evaluate_only:
        report_path = campaign_report_dir / (
            f"segment_{start_step:09d}_{accepted_end_step:09d}_eval.json"
        )
    elif retention_probation_held:
        report_path = campaign_report_dir / (
            f"segment_{start_step + 1:09d}_{accepted_end_step:09d}_probation.json"
        )
    else:
        report_path = campaign_report_dir / (
            f"segment_{start_step + 1:09d}_{accepted_end_step:09d}.json"
        )
    _write_immutable_json(report_path, report)
    if progress is not None:
        progress.emit(
            "completed" if report["curriculum_stage_complete"] else "paused",
            global_step=report["segment_end_step"],
            curriculum_stage_complete=report["curriculum_stage_complete"],
            task_gate_passed=report["task_gate_passed"],
            nonzero_exact_output_observed=report["nonzero_exact_output_observed"],
            exact_serving_gate_passed=report["exact_serving_gate_passed"],
            heldout_mean_loss=report["final_evaluation"]["heldout_mean_loss"],
            report_path=str(report_path),
            report_id=report["report_id"],
        )
    print(json.dumps({**report, "report_path": str(report_path)}, ensure_ascii=True, sort_keys=True, indent=2))
    sync_hook.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
