"""Candidate report handoff tests for the D64 tournament launcher."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from runtime.field import canonical_sha256
from scripts.run_d64_tournament import _command, _load_candidate_report
from training.reasoning_tournament import d64_architecture_search_space


def _write_progress_report(tmp_path: Path) -> tuple[Path, Path, dict]:
    state_root = tmp_path / "State"
    report = {"candidate_label": "d64-l2-h1-f4096", "seed": 7}
    report["report_id"] = canonical_sha256(report)
    report_path = state_root / "training" / "reasoning" / "campaign" / "segment.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    progress_dir = tmp_path / "progress"
    current_path = progress_dir / "candidates" / report["candidate_label"] / "current.json"
    current_path.parent.mkdir(parents=True)
    current_path.write_text(
        json.dumps(
            {
                "status": "paused",
                "details": {
                    "report_id": report["report_id"],
                    "report_path": str(report_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return state_root, progress_dir, report


def test_noisy_child_stdout_uses_content_addressed_progress_report(tmp_path: Path) -> None:
    state_root, progress_dir, report = _write_progress_report(tmp_path)
    completed = subprocess.CompletedProcess(
        ["python"],
        0,
        stdout='AXON_PROGRESS {"status":"training"}\n{not one JSON document}\n',
        stderr="",
    )

    loaded = _load_candidate_report(
        completed,
        state_root=state_root,
        progress_dir=progress_dir,
        candidate_label=report["candidate_label"],
    )

    assert loaded["report_id"] == report["report_id"]
    assert Path(loaded["report_path"]).is_file()


def test_plain_json_stdout_remains_supported_without_progress_journal() -> None:
    report = {"candidate_label": "local-candidate", "report_id": "local-report"}
    loaded = _load_candidate_report(
        subprocess.CompletedProcess(
            ["python"], 0, stdout=json.dumps(report), stderr=""
        ),
        state_root=Path("State"),
        progress_dir=None,
        candidate_label="local-candidate",
    )
    assert loaded == report


def test_progress_report_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    state_root, progress_dir, report = _write_progress_report(tmp_path)
    report_path = Path(
        json.loads(
            (progress_dir / "candidates" / report["candidate_label"] / "current.json").read_text(
                encoding="utf-8"
            )
        )["details"]["report_path"]
    )
    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["seed"] = 8
    report_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(ValueError, match="content-address"):
        _load_candidate_report(
            subprocess.CompletedProcess(["python"], 0, stdout="", stderr=""),
            state_root=state_root,
            progress_dir=progress_dir,
            candidate_label=report["candidate_label"],
        )


def test_receipt_candidate_receives_profile_and_optional_gate_bias() -> None:
    import argparse

    from training.foundation_motor_curriculum import (
        RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
    )

    candidate = next(
        item for item in d64_architecture_search_space() if item.config.receipt_continuation
    )
    args = argparse.Namespace(
        state_root=Path("State"),
        device="cuda",
        evaluation_case_limit=4,
        learning_rate=1e-4,
        seed=20260912,
        checkpoint_interval=16,
        legacy_plan_v1=False,
        run_steps=None,
        tranche_steps=32,
        evaluate_only=False,
        curriculum_manifest=[Path("manifest.json")],
        preflight_only=False,
        resume=False,
        progress_dir=None,
        external_job_id=None,
        # Only the ratified objective is trainable; the route flag is derived
        # from the profile so a launcher cannot select one without the other.
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
        generate_gate_bias=None,
    )
    command = _command(args, candidate=candidate)
    profile_index = command.index("--receipt-teaching-profile")
    assert (
        command[profile_index + 1] == RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6
    )
    assert "--termination-head-route" in command
    assert "--eos-generate-head-route" not in command
    assert "--generate-gate-bias" not in command

    args.generate_gate_bias = 0.0
    command = _command(args, candidate=candidate)
    bias_index = command.index("--generate-gate-bias")
    assert float(command[bias_index + 1]) == 0.0


def test_a_rejected_termination_route_cannot_build_a_candidate_command() -> None:
    """The tournament refuses a rejected route before spawning a candidate."""

    import argparse

    from training.foundation_motor_curriculum import (
        RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
    )

    candidate = next(
        item for item in d64_architecture_search_space() if item.config.receipt_continuation
    )
    args = argparse.Namespace(
        state_root=Path("State"),
        device="cuda",
        evaluation_case_limit=4,
        learning_rate=1e-4,
        seed=20260912,
        checkpoint_interval=16,
        legacy_plan_v1=False,
        run_steps=None,
        tranche_steps=32,
        evaluate_only=False,
        curriculum_manifest=[Path("manifest.json")],
        preflight_only=False,
        resume=False,
        progress_dir=None,
        external_job_id=None,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
        generate_gate_bias=None,
    )
    with pytest.raises(ValueError, match="refusing to train"):
        _command(args, candidate=candidate)

    # Read-only evaluation of an existing bundle stays permitted.
    args.evaluate_only = True
    assert "--receipt-teaching-profile" in _command(args, candidate=candidate)
