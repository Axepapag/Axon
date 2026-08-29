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
from training import d64_head_geometry_tournament, load_first_form_curriculum

ROOT = Path(__file__).resolve().parent.parent
LAUNCH_SCHEMA = "axon-d64-tournament-launch-v1"
OBSERVATION_SCHEMA = "axon-d64-tournament-opening-observation-v1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--run-steps", type=int, default=1)
    parser.add_argument("--evaluation-case-limit", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
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


def _command(args: argparse.Namespace, *, label: str, heads: int) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_living_reasoning_smoke.py"),
        "--state-root",
        str(args.state_root.resolve()),
        "--curriculum-manifest",
        str(args.curriculum_manifest.resolve()),
        "--candidate-label",
        label,
        "--heads",
        str(heads),
        "--device",
        args.device,
        "--max-steps",
        str(args.max_steps),
        "--run-steps",
        str(args.run_steps),
        "--evaluation-case-limit",
        str(args.evaluation_case_limit),
        "--learning-rate",
        repr(args.learning_rate),
        "--seed",
        str(args.seed),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
    ]
    if args.preflight_only:
        command.append("--preflight-only")
    if args.resume:
        command.append("--resume")
    return command


def _failure_class(stderr: str) -> str | None:
    for line in reversed(stderr.splitlines()):
        match = re.match(r"^(?:[\w.]+\.)?([A-Za-z]+Error):", line.strip())
        if match is not None:
            return match.group(1)
    return None


def main() -> int:
    args = _arguments()
    for name in ("max_steps", "run_steps", "evaluation_case_limit", "checkpoint_interval"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    curriculum = load_first_form_curriculum(args.curriculum_manifest)
    tournament = d64_head_geometry_tournament()
    launch = {
        "schema": LAUNCH_SCHEMA,
        "tournament": tournament.to_canonical_dict(),
        "ffcs_manifest_id": curriculum.manifest_id,
        "settings": {
            "device": args.device,
            "max_steps": args.max_steps,
            "run_steps": args.run_steps,
            "evaluation_case_limit": args.evaluation_case_limit,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "checkpoint_interval": args.checkpoint_interval,
            "preflight_only": bool(args.preflight_only),
            "resume": bool(args.resume),
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
    for candidate in tournament.candidates:
        command = _command(
            args,
            label=candidate.label,
            heads=candidate.config.n_heads,
        )
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
            report = json.loads(completed.stdout)
            architecture_id = report.get("architecture", {}).get("architecture_id")
            if (
                report.get("candidate_label") != candidate.label
                or architecture_id != candidate.config.architecture_id
                or report.get("ffcs_manifest_id") != curriculum.manifest_id
                or report.get("seed") != args.seed
            ):
                report = None
                failed = True
                failure_class = "ReportContractError"
        else:
            failed = True
            failure_class = _failure_class(completed.stderr)
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
                "preflight_passed": None if report is None else report.get("preflight_passed"),
                "task_gate_passed": None if report is None else report.get("task_gate_passed"),
                "exact_output_gate_passed": (
                    None if report is None else report.get("exact_serving_gate_passed")
                ),
            }
        )
    body = {
        "schema": OBSERVATION_SCHEMA,
        "launch_id": launch_id,
        "tournament_id": tournament.tournament_id,
        "ffcs_manifest_id": curriculum.manifest_id,
        "status": "failed" if failed else "preflight_complete" if args.preflight_only else "opening_segment_complete",
        "candidate_observations": observations,
        "required_metrics": list(tournament.required_metrics),
        "missing_required_metrics": list(tournament.required_metrics),
        "heldout_evaluation_complete": False,
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
