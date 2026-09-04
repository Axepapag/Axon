"""End-to-end wiring: sequential FFCS-E manifests flow through the governed harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from runtime.dormant import DormantExperienceStore
from runtime.heart import HeartHost
from training import compile_sequential_first_form, publish_sequential_first_form
from training.first_form_curriculum import (
    FirstFormCurriculumCompiler,
    publish_first_form_curriculum,
)

from ._short_tmp import short_state_root
from .test_first_form_curriculum import _experience_store

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_TEXT = "Axon is Axon. Every brother attends the complete exact field."


@pytest.fixture()
def sequential_state():
    """A short-path State root with canonical Identity applied and FFCS-E published."""

    state_root = short_state_root("axwseq") / "State"
    _experience_store(state_root)
    with HeartHost(state_root=state_root) as host:
        host.amend_identity(
            IDENTITY_TEXT,
            amendment_id="wiring-test-identity-v1",
            evidence_ids=("wiring-test-evidence",),
            provenance="tests/test_sequential_wiring.py",
        )
    store = DormantExperienceStore(state_root)
    curriculum = compile_sequential_first_form(
        store,
        identity_text=IDENTITY_TEXT,
        requested_split_counts=(1, 1, 1),
    )
    manifest_path = publish_sequential_first_form(curriculum, state_root=state_root)
    return state_root, manifest_path


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=1200,
    )


def _smoke_command(state_root: Path, manifest_path: Path, label: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "train_living_reasoning_smoke.py"),
        "--state-root",
        str(state_root),
        "--legacy-plan-v1",
        "--curriculum-manifest",
        str(manifest_path),
        "--candidate-label",
        label,
        "--device",
        "cpu",
        "--max-steps",
        "1",
        "--run-steps",
        "1",
        "--evaluation-case-limit",
        "1",
        "--learning-rate",
        "1e-4",
        "--seed",
        "20260830",
        "--ffn-dim",
        "128",
        "--layers",
        "1",
        "--page-size",
        "32",
    ]


def test_smoke_script_trains_sequential_manifest_and_reports_metrics(sequential_state) -> None:
    state_root, manifest_path = sequential_state
    completed = _run(_smoke_command(state_root, manifest_path, "wiring-sequential"))
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout)
    assert report["sequential_ffcs_manifest_id"] is not None
    assert report["sequential_case_count"] == 3
    assert report["curriculum_composition"] == (
        "governed_ffcs_campaign_plus_synthetic_mechanism"
    )
    assert report["steps"][0]["material_kind"] in ("episode", "sequential")
    from training import d64_head_geometry_tournament

    assert set(report["tournament_metrics"]).issubset(
        d64_head_geometry_tournament().required_metrics
    )
    assert report["tournament_metric_surface_complete"] is False
    assert "current_field_override_rate" in report["missing_tournament_metrics"]
    assert "proposal_refinement_gain" in report["missing_tournament_metrics"]
    assert report["tournament_metrics"]["swapped_soul_rejection_rate"] == 1.0
    assert report["serving_promotion_claimed"] is False


def test_smoke_script_renews_legacy_candidate_without_restart(sequential_state) -> None:
    state_root, manifest_path = sequential_state
    label = "wiring-renewable"
    first = _run(_smoke_command(state_root, manifest_path, label))
    assert first.returncode == 0, first.stderr[-4000:]
    first_report = json.loads(first.stdout)
    assert first_report["segment_end_step"] == 1
    assert first_report["lifecycle_status"] == "paused"

    continuation_command = _smoke_command(state_root, manifest_path, label)
    run_index = continuation_command.index("--run-steps")
    del continuation_command[run_index : run_index + 2]
    continuation_command.extend(("--resume", "--tranche-steps", "1"))
    continued = _run(continuation_command)
    assert continued.returncode == 0, continued.stderr[-4000:]
    report = json.loads(continued.stdout)
    assert report["candidate_generation_id"] == first_report["candidate_generation_id"]
    assert report["plan_id"] == first_report["plan_id"]
    assert report["segment_start_step"] == 2
    assert report["segment_end_step"] == 2
    assert report["legacy_plan_envelope_exhausted"] is True
    assert report["campaign_complete"] is False
    assert report["paused_for_next_tranche"] is True
    assert report["resource_tranche_consumed"] is True
    continuation = report["tranche_continuation"]
    assert continuation["parent_global_step"] == 1
    assert continuation["parent_bundle_id"] == first_report["steps"][-1][
        "accepted_step_bundle_id"
    ]

    evaluation_command = _smoke_command(state_root, manifest_path, label)
    run_index = evaluation_command.index("--run-steps")
    del evaluation_command[run_index : run_index + 2]
    evaluation_command.extend(("--resume", "--evaluate-only"))
    evaluated = _run(evaluation_command)
    assert evaluated.returncode == 0, evaluated.stderr[-4000:]
    evaluation = json.loads(evaluated.stdout)
    assert evaluation["evaluation_only"] is True
    assert evaluation["steps"] == []
    assert evaluation["segment_end_step"] == 2
    assert evaluation["final_checkpoint_id"] == report["final_checkpoint_id"]


def test_smoke_script_v2_renews_without_resource_identity_poison(sequential_state) -> None:
    state_root, manifest_path = sequential_state
    label = "wiring-v2-renewable"
    first_command = _smoke_command(state_root, manifest_path, label)
    first_command.remove("--legacy-plan-v1")
    run_index = first_command.index("--run-steps")
    del first_command[run_index : run_index + 2]
    first_command.extend(("--tranche-steps", "1"))
    progress_dir = state_root / "test-cloud-progress"
    first_command.extend(("--progress-dir", str(progress_dir), "--external-job-id", "wiring-v2-cloud"))
    first = _run(first_command)
    assert first.returncode == 0, first.stderr[-4000:]
    first_report = json.loads("\n".join(
        line for line in first.stdout.splitlines() if not line.startswith("AXON_PROGRESS ")
    ))
    assert first_report["plan_schema"] == "axon-parameter-mutation-plan-v2"
    assert first_report["campaign_max_steps"] is None
    assert first_report["legacy_plan_envelope_exhausted"] is None
    assert first_report["segment_end_step"] == 1
    assert first_report["resource_tranche"]["base_global_step"] == 0
    assert first_report["resource_tranche"]["parent_bundle_id"] is None

    continuation_command = list(first_command)
    max_index = continuation_command.index("--max-steps")
    continuation_command[max_index + 1] = "999"
    continuation_command.extend(("--checkpoint-interval", "2", "--resume"))
    continued = _run(continuation_command)
    assert continued.returncode == 0, continued.stderr[-4000:]
    report = json.loads("\n".join(
        line for line in continued.stdout.splitlines() if not line.startswith("AXON_PROGRESS ")
    ))
    assert report["candidate_generation_id"] == first_report["candidate_generation_id"]
    assert report["plan_id"] == first_report["plan_id"]
    assert report["learning_policy_id"] == first_report["learning_policy_id"]
    assert report["segment_start_step"] == 2
    assert report["segment_end_step"] == 2
    assert report["tranche_continuation"]["parent_global_step"] == 1
    assert report["tranche_continuation"]["prior_tranche_id"] == first_report[
        "resource_tranche"
    ]["tranche_id"]
    events = [json.loads(line) for line in (progress_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    evaluations = [event["details"] for event in events if event["status"] == "evaluated"]
    assert [(row["phase"], row["global_step"]) for row in evaluations] == [
        ("initial", 0), ("final", 1), ("initial", 1), ("final", 2),
    ]
    assert all("qa_transcripts" in row for row in evaluations)
    assert len({event["event_id"] for event in events}) == len(events)


def test_launcher_runs_all_candidates_on_sequential_manifest(sequential_state) -> None:
    """Preflight-only launch: proves child wiring, contracts, and artifacts cheaply.

    Full-step launcher segments run on the real State under CUDA as governed
    campaign work (see the tournament launcher), not inside the unit suite.
    """

    state_root, sequential_manifest = sequential_state
    completed = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_d64_tournament.py"),
            "--state-root",
            str(state_root),
            "--legacy-plan-v1",
            "--curriculum-manifest",
            str(sequential_manifest),
            "--device",
            "cpu",
            "--max-steps",
            "1",
            "--run-steps",
            "1",
            "--evaluation-case-limit",
            "1",
            "--seed",
            "20260830",
            "--preflight-only",
        ]
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    observation = json.loads(completed.stdout)
    assert observation["schema"] == "axon-d64-tournament-opening-observation-v2"
    assert observation["status"] == "preflight_complete"
    assert len(observation["candidate_observations"]) == 3
    for child in observation["candidate_observations"]:
        assert child["returncode"] == 0
        assert child["preflight_passed"] is True
    assert observation["winner_selected"] is False
    assert observation["promotion_claimed"] is False
    assert observation["comparable_result_set_complete"] is False
    assert observation["campaign_identity_consistent"] is True
    assert Path(observation["launch_path"]).is_file()
    assert Path(observation["observation_path"]).is_file()


def test_smoke_script_rejects_stale_identity_manifest(sequential_state) -> None:
    state_root, manifest_path = sequential_state
    with HeartHost(state_root=state_root) as host:
        host.amend_identity(
            "A different ratified identity replaces the earlier one.",
            amendment_id="wiring-test-identity-v2",
            evidence_ids=("wiring-test-evidence-2",),
            provenance="tests/test_sequential_wiring.py",
        )
    completed = _run(_smoke_command(state_root, manifest_path, "wiring-stale"))
    assert completed.returncode != 0
    assert "stale" in completed.stderr.lower()


def test_launcher_rejects_unknown_manifest_schema() -> None:
    base = short_state_root("axwbogus")
    bogus = base / "bogus_manifest.json"
    bogus.write_text(json.dumps({"schema": "axon-not-a-real-manifest-v1"}), encoding="utf-8")
    completed = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_d64_tournament.py"),
            "--state-root",
            str(base / "State"),
            "--curriculum-manifest",
            str(bogus),
            # v2 tournament training requires --tranche-steps; use the legacy
            # v1 path so this probe reaches manifest-schema validation itself.
            "--legacy-plan-v1",
            "--max-steps",
            "1",
        ]
    )
    assert completed.returncode != 0
    assert "unsupported curriculum manifest schema" in completed.stderr


def test_smoke_preflight_binds_multiple_standard_and_sequential_manifests(
    sequential_state,
) -> None:
    state_root, sequential_manifest = sequential_state
    compiler = FirstFormCurriculumCompiler(DormantExperienceStore(state_root))
    abc = compiler.compile_abc(
        identity_text=IDENTITY_TEXT,
        requested_counts=(("A", (1, 1, 1)), ("B", (1, 1, 1)), ("C", (1, 1, 1))),
    )
    df = compiler.compile_df(
        identity_text=IDENTITY_TEXT,
        requested_counts=(("D", (1, 1, 1)), ("F", (1, 1, 1))),
    )
    abc_path = publish_first_form_curriculum(abc, state_root=state_root)
    df_path = publish_first_form_curriculum(df, state_root=state_root)
    command = _smoke_command(state_root, abc_path, "wiring-multi-manifest")
    command.extend(
        [
            "--curriculum-manifest",
            str(df_path),
            "--curriculum-manifest",
            str(sequential_manifest),
            "--preflight-only",
        ]
    )
    completed = _run(command)
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout)
    expected = [*sorted((abc.manifest_id, df.manifest_id)), report["sequential_ffcs_manifest_id"]]
    assert report["ffcs_manifest_ids"] == expected
    assert report["curriculum_composition"] == (
        "governed_ffcs_campaign_plus_synthetic_mechanism"
    )
    assert len(report["source_manifest_ids"]) == 5
    assert len(report["holdout_manifest_ids"]) == 5
    assert Path(report["campaign_curriculum_path"]).is_file()
