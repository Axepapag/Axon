from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from tools.training_preflight import ROOT, SUBSTRATE_PROBE, run_preflight


def repository(tmp_path: Path) -> Path:
    for name in ("substrate.py", "native.py", "substrate_1024.py", "d1024_lane_bank.npy"):
        path = tmp_path / "substrate" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture")
    return tmp_path


def passed(command, **kwargs):
    output = json.dumps({"probe": "test evidence"}) if "-c" in command else ""
    return subprocess.CompletedProcess(command, 0, output, "")


def test_missing_repo_never_runs_or_claims_ready(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not run probes against a missing repository")
    report = run_preflight(tmp_path, runner=forbidden)
    assert not report["foundation_passed"]
    assert not report["training_authorized"]
    assert report["checks"][0]["details"]["missing"]


def test_green_foundation_does_not_authorize_training(tmp_path):
    report = run_preflight(repository(tmp_path), runner=passed)
    assert report["foundation_passed"]
    assert report["training_authorized"] is False
    assert all(item["status"] == "not_checked" for item in report["unverified_requirements"])
    assert {item["id"] for item in report["unverified_requirements"]} >= {
        "heart_runtime", "checkpoint_resume", "operator_controls", "backup_restore"}


def test_one_failed_probe_blocks_foundation_and_keeps_other_results(tmp_path):
    def runner(command, **kwargs):
        if command[-1] == "cuda:0":
            return subprocess.CompletedProcess(command, 1, "", "CUDA unavailable")
        return passed(command, **kwargs)
    report = run_preflight(repository(tmp_path), device="cuda:0", runner=runner)
    assert not report["foundation_passed"]
    assert report["checks"][-1]["message"] == "CUDA unavailable"
    assert sum(check["status"] == "pass" for check in report["checks"]) == 4


def test_timeout_becomes_visible_failure(tmp_path):
    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    report = run_preflight(repository(tmp_path), timeout=2, runner=runner)
    assert not report["foundation_passed"]
    assert all("exceeded 2 seconds" in check["message"] for check in report["checks"][1:])


def test_zero_exit_with_malformed_probe_payload_is_failure(tmp_path):
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "not JSON" if "-c" in command else "", "")
    report = run_preflight(repository(tmp_path), runner=runner)
    assert not report["foundation_passed"]
    assert report["checks"][-1]["status"] == "fail"


def test_probe_subprocesses_are_read_only_and_target_selected_root(tmp_path):
    observed = []
    def runner(command, **kwargs):
        observed.append((command, kwargs))
        return passed(command, **kwargs)
    root = repository(tmp_path)
    run_preflight(root, device="cpu", timeout=7, runner=runner)
    assert len(observed) == 4
    assert all("-B" in command for command, _ in observed)
    assert all(kwargs["cwd"] == str(root.resolve()) for _, kwargs in observed)
    assert all(kwargs["timeout"] == 7 for _, kwargs in observed)
    assert observed[-1][0][-1] == "cpu"
    assert sorted(path.name for path in root.iterdir()) == ["substrate"]


def test_real_native_probe_checks_current_codec_and_seal():
    result = subprocess.run([sys.executable, "-B", "-c", SUBSTRATE_PROBE],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["native_count"] == 95
    assert payload["round_trip"] and payload["outsider_rejection"]
    assert len(payload["native_bank_sha256"]) == 64
