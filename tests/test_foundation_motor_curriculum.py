from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from runtime.heart import HeartHost
from training.first_form_curriculum import publish_first_form_curriculum
from training.foundation_motor_curriculum import (
    FOUNDATION_MOTOR_ACTIONS,
    FOUNDATION_MOTOR_GATE_POLICY_ID,
    FOUNDATION_MOTOR_STAGE,
    compile_foundation_motor,
    decide_foundation_motor_mastery,
    foundation_motor_probe,
    verify_foundation_motor_curriculum,
)

from ._short_tmp import short_state_root

IDENTITY = "Axon is Axon. Every brother attends the complete exact field."
ROOT = Path(__file__).resolve().parents[1]


def _curriculum():
    return compile_foundation_motor(
        identity_text=IDENTITY,
        requested_counts=(("F0", (24, 12, 12)),),
    )


def _tag(case, prefix: str) -> str:
    return next(tag[len(prefix) :] for tag in case.episode.mechanism_tags if tag.startswith(prefix))


def _row(*, exact: bool = True, payload: bool = True) -> dict[str, float]:
    return {
        "typed_emission_exact_count": 1.0 if exact else 0.0,
        "supervised_phase_count": 1.0,
        "payload_transport_exact_count": 1.0 if exact and payload else 0.0,
        "payload_supervised_phase_count": 1.0 if payload else 0.0,
        "complete_field_coverage_count": 3.0,
        "phase_output_count": 3.0,
    }


def _rows(episodes, *, broken: int | None = None):
    return [
        _row(
            exact=index != broken,
            payload=bool(episode.targets[-1].payload),
        )
        for index, episode in enumerate(episodes)
    ]


def test_motor_curriculum_is_deterministic_complete_and_split_disjoint() -> None:
    first = _curriculum()
    second = _curriculum()
    assert first.manifest_id == second.manifest_id
    assert first.actual_family_split_counts == (("F0", (24, 12, 12)),)
    verify_foundation_motor_curriculum(first)
    sources: dict[str, set[str]] = defaultdict(set)
    actions: dict[str, set[str]] = defaultdict(set)
    for case in first.cases:
        assert f"foundation_stage:{FOUNDATION_MOTOR_STAGE}" in case.episode.mechanism_tags
        sources[case.episode.split].add(_tag(case, "transfer_item:"))
        actions[case.episode.split].add(_tag(case, "foundation_motor_action:"))
    assert all(actions[split] == set(FOUNDATION_MOTOR_ACTIONS) for split in actions)
    assert sources["train"].isdisjoint(sources["heldout"])
    assert sources["train"].isdisjoint(sources["regression"])
    assert sources["heldout"].isdisjoint(sources["regression"])


def test_motor_payload_actions_bind_exact_copy_alignment() -> None:
    for case in _curriculum().cases:
        target = case.episode.targets[-1]
        action = _tag(case, "foundation_motor_action:")
        if action in {"copy", "insert", "replace"}:
            assert len(target.payload) == 1
            assert target.payload_alignment is not None
            assert len(target.payload_alignment["segments"]) == 1
        else:
            assert target.payload_alignment is None


def test_motor_compile_rejects_a_split_without_all_actions() -> None:
    with pytest.raises(ValueError, match="at least 12"):
        compile_foundation_motor(
            identity_text=IDENTITY,
            requested_counts=(("F0", (10, 12, 12)),),
        )


def test_motor_probe_and_advancement_gate_fail_closed_per_action() -> None:
    curriculum = _curriculum()
    heldout = tuple(case.episode for case in curriculum.cases if case.episode.split == "heldout")
    regression = tuple(case.episode for case in curriculum.cases if case.episode.split == "regression")
    heldout_probe = foundation_motor_probe(heldout, _rows(heldout))
    regression_probe = foundation_motor_probe(regression, _rows(regression))
    evaluation = {
        "payload_teacher_forced_token_accuracy": 0.75,
        "constant_payload_token_accuracy_floor": 0.25,
    }
    passed = decide_foundation_motor_mastery(
        heldout_probe=heldout_probe,
        regression_probe=regression_probe,
        evaluation=evaluation,
        complete_heldout=True,
        complete_regression=True,
    )
    assert passed is not None and passed["passed"] is True
    assert passed["policy_id"] == FOUNDATION_MOTOR_GATE_POLICY_ID

    failed_probe = foundation_motor_probe(heldout, _rows(heldout, broken=0))
    failed = decide_foundation_motor_mastery(
        heldout_probe=failed_probe,
        regression_probe=regression_probe,
        evaluation=evaluation,
        complete_heldout=True,
        complete_regression=True,
    )
    assert failed is not None and failed["passed"] is False
    assert any("per-action" in reason for reason in failed["failures"])


def test_governed_harness_trains_alignment_and_reports_motor_gate() -> None:
    state_root = short_state_root("foundation-motor-wire") / "State"
    with HeartHost(state_root=state_root) as host:
        host.amend_identity(
            IDENTITY,
            amendment_id="foundation-motor-test-identity-v1",
            evidence_ids=("foundation-motor-test-evidence",),
            provenance="tests/test_foundation_motor_curriculum.py",
        )
    curriculum = compile_foundation_motor(
        identity_text=IDENTITY,
        requested_counts=(("F0", (12, 12, 12)),),
    )
    manifest = publish_first_form_curriculum(curriculum, state_root=state_root)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_living_reasoning_smoke.py"),
            "--state-root",
            str(state_root),
            "--curriculum-manifest",
            str(manifest),
            "--candidate-label",
            "foundation-motor-wire",
            "--device",
            "cpu",
            "--tranche-steps",
            "2",
            "--checkpoint-interval",
            "1",
            "--ffn-dim",
            "128",
            "--layers",
            "1",
            "--seed",
            "20260906",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=1200,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    report = json.loads(completed.stdout)
    gate = report["foundation_motor_gate"]
    assert gate is not None
    assert gate["policy_id"] == FOUNDATION_MOTOR_GATE_POLICY_ID
    assert report["foundation_sequence_gate"] is None
    assert report["serving_promotion_claimed"] is False
    motor_step = next(step for step in report["steps"] if step["curriculum_lane"] == "ffcs-F0")
    consolidated = motor_step["phase_metrics"][-1]
    assert consolidated["alignment_copy_positions"] >= 1.0
