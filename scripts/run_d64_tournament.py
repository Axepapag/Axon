"""Launch identical governed opening segments for the D64 head tournament."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from runtime.field import canonical_sha256
from runtime.trainer.progress import TrainingProgressJournal
from training import (
    D64TournamentResult,
    assert_same_gate_surface,
    d64_architecture_campaign,
    d64_architecture_screening_tournament,
    d64_head_geometry_tournament,
    load_first_form_curriculum,
    load_sequential_first_form,
)

ROOT = Path(__file__).resolve().parent.parent
LAUNCH_SCHEMA = "axon-d64-tournament-launch-v2"
OBSERVATION_SCHEMA = "axon-d64-tournament-opening-observation-v2"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument(
        "--profile",
        choices=("head-geometry", "architecture-screen"),
        default="head-geometry",
        help="candidate family to execute on the identical governed surface",
    )
    parser.add_argument(
        "--candidate-label",
        action="append",
        help="execute only these declared candidates; repeatable (diagnostic slices remain non-promoting)",
    )
    parser.add_argument(
        "--curriculum-manifest",
        type=Path,
        action="append",
        required=True,
        help="immutable FFCS manifest (standard or sequential schema); repeatable",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--legacy-plan-v1",
        action="store_true",
        help="address the historical max_steps-bound candidate generations",
    )
    parser.add_argument("--run-steps", type=int, default=None)
    parser.add_argument("--evaluation-case-limit", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--progress-dir",
        type=Path,
        default=None,
        help="durable parent progress directory; candidate journals are nested beneath it",
    )
    parser.add_argument(
        "--external-job-id",
        default=None,
        help="provider-neutral job identity used only for progress correlation",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--tranche-steps",
        type=int,
        default=None,
        help="renewable steps per candidate beyond each exact restored parent",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="evaluate existing accepted candidates without optimizer or Soul mutation",
    )
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _command(args: argparse.Namespace, *, candidate: Any) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_living_reasoning_smoke.py"),
        "--state-root",
        str(args.state_root.resolve()),
        "--candidate-label",
        candidate.label,
        "--heads",
        str(candidate.config.n_heads),
        "--layers",
        str(candidate.config.n_layers),
        "--ffn-dim",
        str(candidate.config.ffn_dim),
        "--page-size",
        str(candidate.config.page_size),
        "--device",
        args.device,
        "--evaluation-case-limit",
        str(args.evaluation_case_limit),
        "--learning-rate",
        repr(args.learning_rate),
        "--seed",
        str(args.seed),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
    ]
    if candidate.config.receipt_continuation:
        command.append("--receipt-continuation")
    if args.legacy_plan_v1:
        command.extend(("--max-steps", str(args.max_steps)))
    if args.run_steps is not None and args.tranche_steps is None and not args.evaluate_only:
        command.extend(("--run-steps", str(args.run_steps)))
    for manifest in args.curriculum_manifest:
        command.extend(("--curriculum-manifest", str(manifest.resolve())))
    if args.preflight_only:
        command.append("--preflight-only")
    if args.resume:
        command.append("--resume")
    if args.tranche_steps is not None:
        command.extend(("--tranche-steps", str(args.tranche_steps)))
    if args.evaluate_only:
        command.append("--evaluate-only")
    if args.legacy_plan_v1:
        command.append("--legacy-plan-v1")
    if args.progress_dir is not None:
        command.extend(
            (
                "--progress-dir",
                str((args.progress_dir / "candidates" / candidate.label).resolve()),
            )
        )
    if args.external_job_id:
        command.extend(("--external-job-id", args.external_job_id))
    return command


def _failure_class(stderr: str) -> str | None:
    for line in reversed(stderr.splitlines()):
        match = re.match(r"^(?:[\w.]+\.)?([A-Za-z]+Error):", line.strip())
        if match is not None:
            return match.group(1)
    return None


def _load_candidate_report(
    completed: subprocess.CompletedProcess[str],
    *,
    state_root: Path,
    progress_dir: Path | None,
    candidate_label: str,
) -> dict[str, Any]:
    """Load a child's report from stdout or its durable progress receipt."""

    if completed.stdout.strip():
        report = json.loads(completed.stdout)
        if not isinstance(report, dict):
            raise ValueError("candidate stdout report must be a JSON object")
        return report
    if progress_dir is None:
        raise ValueError("candidate returned no stdout report and has no progress receipt")

    current_path = progress_dir / "candidates" / candidate_label / "current.json"
    current = json.loads(current_path.read_text(encoding="utf-8"))
    details = current.get("details")
    if not isinstance(details, dict) or not isinstance(details.get("report_path"), str):
        raise ValueError("candidate progress receipt does not name a report")

    report_path = Path(details["report_path"]).resolve()
    state_root = state_root.resolve()
    if not report_path.is_relative_to(state_root):
        raise ValueError("candidate report path escapes the governed State root")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("candidate report artifact must be a JSON object")
    report_id = report.get("report_id")
    unsigned = dict(report)
    unsigned.pop("report_id", None)
    if not isinstance(report_id, str) or canonical_sha256(unsigned) != report_id:
        raise ValueError("candidate report artifact failed its content-address check")
    if details.get("report_id") != report_id:
        raise ValueError("candidate progress receipt names a different report id")
    return {**report, "report_path": str(report_path)}


def main() -> int:
    args = _arguments()
    for name in ("evaluation_case_limit", "checkpoint_interval"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.legacy_plan_v1 and args.max_steps < 1:
        raise ValueError("--max-steps must be positive for legacy v1 plans")
    if args.run_steps is not None and args.run_steps < 1:
        raise ValueError("--run-steps must be positive when supplied")
    if args.tranche_steps is not None and args.tranche_steps < 1:
        raise ValueError("--tranche-steps must be positive when supplied")
    if args.evaluate_only and not args.resume:
        raise ValueError("evaluation-only operation requires --resume")
    if args.legacy_plan_v1 and args.tranche_steps is not None and not args.resume:
        raise ValueError("legacy renewal requires --resume")
    if args.tranche_steps is not None and args.evaluate_only:
        raise ValueError("--tranche-steps and --evaluate-only are mutually exclusive")
    if not args.legacy_plan_v1 and not args.evaluate_only and args.tranche_steps is None:
        raise ValueError("v2 tournament training requires --tranche-steps")
    standard_curricula = []
    sequential_curricula = []
    for manifest_path in args.curriculum_manifest:
        body = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = body.get("schema")
        if schema == "axon-first-form-curriculum-v1":
            standard_curricula.append(load_first_form_curriculum(manifest_path))
        elif schema == "axon-first-form-sequential-curriculum-v1":
            sequential_curricula.append(load_sequential_first_form(manifest_path))
        else:
            raise ValueError(
                f"unsupported curriculum manifest schema {schema!r} at {manifest_path}"
            )
    if not standard_curricula and not sequential_curricula:
        raise ValueError("at least one curriculum manifest is required")
    tournament = (
        d64_architecture_screening_tournament()
        if args.profile == "architecture-screen"
        else d64_head_geometry_tournament()
    )
    campaign = d64_architecture_campaign() if args.profile == "architecture-screen" else None
    known_labels = {item.label for item in tournament.candidates}
    selected_labels = (
        known_labels
        if not args.candidate_label
        else set(args.candidate_label)
    )
    unknown_labels = selected_labels - known_labels
    if unknown_labels:
        raise ValueError(
            "unknown tournament candidate labels: " + ", ".join(sorted(unknown_labels))
        )
    selected_candidates = tuple(
        item for item in tournament.candidates if item.label in selected_labels
    )
    progress = (
        None
        if args.progress_dir is None
        else TrainingProgressJournal(
            args.progress_dir,
            job_id=args.external_job_id or f"tournament:{tournament.tournament_id}",
        )
    )
    standard_curricula.sort(key=lambda item: item.manifest_id)
    sequential_curricula.sort(key=lambda item: item.manifest_id)
    ffcs_manifest_ids = [item.manifest_id for item in standard_curricula] + [
        item.manifest_id for item in sequential_curricula
    ]
    if len(set(ffcs_manifest_ids)) != len(ffcs_manifest_ids):
        raise ValueError("duplicate FFCS manifest identity")
    launch = {
        "schema": LAUNCH_SCHEMA,
        "profile": args.profile,
        "tournament": tournament.to_canonical_dict(),
        "campaign": None if campaign is None else campaign.to_canonical_dict(),
        "selected_candidate_labels": [item.label for item in selected_candidates],
        "ffcs_manifest_ids": ffcs_manifest_ids,
        "settings": {
            "device": args.device,
            "max_steps": args.max_steps if args.legacy_plan_v1 else None,
            "run_steps": args.run_steps,
            "evaluation_case_limit": args.evaluation_case_limit,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "checkpoint_interval": args.checkpoint_interval,
            "preflight_only": bool(args.preflight_only),
            "resume": bool(args.resume),
            "tranche_steps": args.tranche_steps,
            "evaluate_only": bool(args.evaluate_only),
            "legacy_plan_v1": bool(args.legacy_plan_v1),
        },
        "claim_boundary": (
            "opening diagnostic only; incomplete heldout and missing tournament metrics "
            "cannot select, promote, or serve a winner"
        ),
    }
    launch_id = canonical_sha256(launch)
    launch["launch_id"] = launch_id
    root = args.state_root.resolve() / "training" / "reasoning" / "tournaments" / tournament.tournament_id
    launch_path = root / "launches" / f"{launch_id}.json"
    if launch_path.exists():
        if json.loads(launch_path.read_text(encoding="utf-8")) != launch:
            raise RuntimeError("immutable tournament launch artifact mismatch")
    else:
        _atomic_json(launch_path, launch)

    observations = []
    failed = False
    for candidate_index, candidate in enumerate(selected_candidates, start=1):
        if progress is not None:
            progress.emit(
                "starting",
                phase="candidate",
                candidate_index=candidate_index,
                candidate_count=len(selected_candidates),
                candidate_label=candidate.label,
                architecture_id=candidate.config.architecture_id,
            )
        command = _command(args, candidate=candidate)
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        report = None
        failure_class = None
        if completed.returncode == 0:
            try:
                report = _load_candidate_report(
                    completed,
                    state_root=args.state_root,
                    progress_dir=args.progress_dir,
                    candidate_label=candidate.label,
                )
                architecture = report.get("architecture")
                architecture_id = (
                    architecture.get("architecture_id")
                    if isinstance(architecture, dict)
                    else None
                )
                report_valid = (
                    report.get("candidate_label") == candidate.label
                    and architecture_id == candidate.config.architecture_id
                    and report.get("ffcs_manifest_ids") == ffcs_manifest_ids
                    and bool(report.get("campaign_curriculum_id"))
                    and report.get("seed") == args.seed
                )
            except (OSError, ValueError, TypeError):
                report = None
                report_valid = False
            if not report_valid:
                failed = True
                failure_class = "ReportContractError"
        else:
            failed = True
            failure_class = _failure_class(completed.stderr)
        child_metrics = None if report is None else report.get("tournament_metrics")
        child_missing_metrics = (
            list(tournament.required_metrics)
            if report is None
            else report.get(
                "missing_tournament_metrics", list(tournament.required_metrics)
            )
        )
        observations.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_label": candidate.label,
                "architecture_id": candidate.config.architecture_id,
                "returncode": completed.returncode,
                "failure_class": failure_class,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
                "report_id": None if report is None else report.get("report_id"),
                "report_path": None if report is None else report.get("report_path"),
                "campaign_curriculum_id": (
                    None if report is None else report.get("campaign_curriculum_id")
                ),
                "preflight_passed": None if report is None else report.get("preflight_passed"),
                "task_gate_passed": None if report is None else report.get("task_gate_passed"),
                "exact_output_gate_passed": (
                    None if report is None else report.get("exact_serving_gate_passed")
                ),
                "tournament_metrics": child_metrics,
                "missing_tournament_metrics": child_missing_metrics,
                "tournament_metric_surface_complete": (
                    None
                    if report is None
                    else report.get("tournament_metric_surface_complete")
                ),
            }
        )
        if progress is not None:
            progress.emit(
                "completed" if completed.returncode == 0 else "failed",
                phase="candidate",
                candidate_index=candidate_index,
                candidate_count=len(selected_candidates),
                candidate_label=candidate.label,
                returncode=completed.returncode,
                failure_class=failure_class,
            )
    candidate_missing_metrics = {
        item["candidate_id"]: list(item["missing_tournament_metrics"])
        for item in observations
    }
    metric_surface_complete = bool(observations) and all(
        not missing for missing in candidate_missing_metrics.values()
    )
    missing_required_metrics = sorted(
        {name for missing in candidate_missing_metrics.values() for name in missing}
    )
    heldout_complete_flags = []
    for item in observations:
        if item["report_path"] is None:
            heldout_complete_flags.append(False)
            continue
        try:
            child_report = json.loads(
                Path(item["report_path"]).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            heldout_complete_flags.append(False)
            continue
        heldout_complete_flags.append(
            bool(child_report.get("complete_heldout_evaluation"))
        )
    heldout_evaluation_complete = all(heldout_complete_flags) and bool(
        heldout_complete_flags
    )
    campaign_curriculum_ids = sorted(
        {
            row["campaign_curriculum_id"]
            for row in observations
            if row["campaign_curriculum_id"] is not None
        }
    )
    campaign_identity_consistent = len(campaign_curriculum_ids) == 1
    if not campaign_identity_consistent:
        failed = True
    comparable_results = []
    for candidate, observation in zip(selected_candidates, observations, strict=True):
        metrics = observation["tournament_metrics"] or {}
        if observation["missing_tournament_metrics"] or set(metrics) != set(
            tournament.required_metrics
        ):
            observation["tournament_result"] = None
            continue
        result = D64TournamentResult.from_mapping(
            tournament,
            candidate,
            metrics,
            gate_passed=bool(
                observation["task_gate_passed"]
                and observation["exact_output_gate_passed"]
                and heldout_evaluation_complete
            ),
        )
        comparable_results.append(result)
        observation["tournament_result"] = result.to_canonical_dict()
    comparable_result_set_complete = len(comparable_results) == len(
        tournament.candidates
    ) and len(selected_candidates) == len(tournament.candidates)
    if comparable_result_set_complete:
        assert_same_gate_surface(tournament, comparable_results)
    body = {
        "schema": OBSERVATION_SCHEMA,
        "profile": args.profile,
        "selected_candidate_labels": [item.label for item in selected_candidates],
        "launch_id": launch_id,
        "tournament_id": tournament.tournament_id,
        "ffcs_manifest_ids": ffcs_manifest_ids,
        "campaign_curriculum_ids": campaign_curriculum_ids,
        "campaign_identity_consistent": campaign_identity_consistent,
        "status": (
            "failed"
            if failed
            else "preflight_complete"
            if args.preflight_only
            else "opening_segment_complete"
        ),
        "candidate_observations": observations,
        "required_metrics": list(tournament.required_metrics),
        "missing_required_metrics": missing_required_metrics,
        "candidate_missing_metrics": candidate_missing_metrics,
        "tournament_metric_surface_complete": metric_surface_complete,
        "heldout_evaluation_complete": heldout_evaluation_complete,
        "comparable_result_set_complete": comparable_result_set_complete,
        "winner_selected": False,
        "promotion_claimed": False,
    }
    observation_id = canonical_sha256(body)
    body["observation_id"] = observation_id
    observation_path = root / "observations" / f"{observation_id}.json"
    if observation_path.exists():
        if json.loads(observation_path.read_text(encoding="utf-8")) != body:
            raise RuntimeError("immutable tournament observation artifact mismatch")
    else:
        _atomic_json(observation_path, body)
    if progress is not None:
        progress.emit(
            "failed" if failed else "completed",
            phase="tournament",
            candidate_count=len(selected_candidates),
            observation_id=observation_id,
            observation_path=str(observation_path),
            winner_selected=False,
        )
    print(
        json.dumps(
            {
                **body,
                "launch_path": str(launch_path),
                "observation_path": str(observation_path),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
