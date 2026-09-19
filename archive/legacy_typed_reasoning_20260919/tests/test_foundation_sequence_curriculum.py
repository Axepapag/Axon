from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from runtime.heart import HeartHost
from training.first_form_curriculum import publish_first_form_curriculum
from training.foundation_sequence_curriculum import (
    FOUNDATION_SEQUENCE_GATE_POLICY_ID,
    FOUNDATION_SEQUENCE_STAGE,
    compile_foundation_sequence,
    decide_foundation_sequence_mastery,
    foundation_sequence_probe,
    verify_foundation_sequence_curriculum,
)

from ._short_tmp import short_state_root

IDENTITY = "Axon is Axon. Every brother attends the complete exact field."
ROOT = Path(__file__).resolve().parents[1]


def _curriculum():
    return compile_foundation_sequence(
        identity_text=IDENTITY,
        requested_counts=(("ABC", (16, 8, 8)),),
    )


def _tag(case, prefix: str) -> str:
    return next(tag[len(prefix) :] for tag in case.episode.mechanism_tags if tag.startswith(prefix))


def _row(*, exact: bool = True) -> dict[str, float]:
    return {
        "typed_emission_exact_count": 1.0 if exact else 0.0,
        "supervised_phase_count": 1.0,
        "payload_transport_exact_count": 1.0 if exact else 0.0,
        "payload_supervised_phase_count": 1.0,
        "complete_field_coverage_count": 3.0,
        "phase_output_count": 3.0,
    }


def test_foundation_sequence_is_deterministic_and_runtime_faithful() -> None:
    first = _curriculum()
    second = _curriculum()
    assert first.manifest_id == second.manifest_id
    assert first.actual_family_split_counts == (("ABC", (16, 8, 8)),)
    verify_foundation_sequence_curriculum(first)
    assert all(
        f"foundation_stage:{FOUNDATION_SEQUENCE_STAGE}" in case.episode.mechanism_tags
        for case in first.cases
    )
    assert all(case.episode.targets[-1].payload for case in first.cases)
    assert all(
        case.episode.targets[-1].payload_alignment is not None
        for case in first.cases
    )
    assert all(
        len(case.episode.targets[-1].payload_alignment["segments"])
        == len(case.episode.targets[-1].payload)
        for case in first.cases
    )


def test_foundation_sources_are_disjoint_and_pairs_change_the_answer() -> None:
    curriculum = _curriculum()
    sources: dict[str, set[str]] = defaultdict(set)
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case in curriculum.cases:
        split = case.episode.split
        sources[split].add(_tag(case, "transfer_item:"))
        pairs[(split, _tag(case, "foundation_pair:"))].append(
            case.episode.targets[-1].payload
        )
    assert sources["train"].isdisjoint(sources["heldout"])
    assert sources["train"].isdisjoint(sources["regression"])
    assert sources["heldout"].isdisjoint(sources["regression"])
    assert all(len(payloads) == 2 and payloads[0] != payloads[1] for payloads in pairs.values())


def test_foundation_compile_rejects_incomplete_pair_budget() -> None:
    with pytest.raises(ValueError, match="positive even"):
        compile_foundation_sequence(
            identity_text=IDENTITY,
            requested_counts=(("ABC", (15, 8, 8)),),
        )


def test_foundation_pair_probe_and_advancement_gate() -> None:
    heldout = tuple(case.episode for case in _curriculum().cases if case.episode.split == "heldout")
    regression = tuple(
        case.episode for case in _curriculum().cases if case.episode.split == "regression"
    )
    heldout_probe = foundation_sequence_probe(heldout, [_row() for _ in heldout])
    regression_probe = foundation_sequence_probe(regression, [_row() for _ in regression])
    evaluation = {
        "payload_teacher_forced_token_accuracy": 0.75,
        "constant_payload_token_accuracy_floor": 0.25,
    }
    decision = decide_foundation_sequence_mastery(
        heldout_probe=heldout_probe,
        regression_probe=regression_probe,
        evaluation=evaluation,
        complete_heldout=True,
        complete_regression=True,
    )
    assert decision is not None and decision["passed"] is True
    assert decision["policy_id"] == FOUNDATION_SEQUENCE_GATE_POLICY_ID
    assert decision["scope"] == "curriculum_advancement_only_not_serving_or_promotion"

    broken_rows = [_row() for _ in heldout]
    broken_rows[0] = _row(exact=False)
    failed_probe = foundation_sequence_probe(heldout, broken_rows)
    failed = decide_foundation_sequence_mastery(
        heldout_probe=failed_probe,
        regression_probe=regression_probe,
        evaluation=evaluation,
        complete_heldout=True,
        complete_regression=True,
    )
    assert failed is not None and failed["passed"] is False
    assert any("changed-source" in reason for reason in failed["failures"])


def test_governed_harness_reports_foundation_gate_without_serving_claim() -> None:
    state_root = short_state_root("foundation-wire") / "State"
    with HeartHost(state_root=state_root) as host:
        host.amend_identity(
            IDENTITY,
            amendment_id="foundation-test-identity-v1",
            evidence_ids=("foundation-test-evidence",),
            provenance="tests/test_foundation_sequence_curriculum.py",
        )
    curriculum = compile_foundation_sequence(
        identity_text=IDENTITY,
        requested_counts=(("ABC", (4, 2, 2)),),
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
            "foundation-wire",
            "--device",
            "cpu",
            "--tranche-steps",
            "1",
            "--checkpoint-interval",
            "1",
            "--ffn-dim",
            "128",
            "--layers",
            "1",
            "--seed",
            "20260904",
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
    gate = report["foundation_sequence_gate"]
    assert gate is not None
    assert gate["policy_id"] == FOUNDATION_SEQUENCE_GATE_POLICY_ID
    assert gate["scope"] == "curriculum_advancement_only_not_serving_or_promotion"
    assert report["serving_promotion_claimed"] is False
    assert not report["curriculum_stage_complete"] or gate["passed"]
