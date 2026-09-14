"""Candidate report handoff tests for the D64 tournament launcher."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from runtime.field import canonical_sha256
from scripts.run_d64_tournament import _load_candidate_report


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
