from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from runtime.field import canonical_sha256
from runtime.heart import HeartHost
from training.foundation_motor_curriculum import (
    COPY_ALIGNMENT_MULTICELL_TEACH,
    FOUNDATION_MOTOR_ACTIONS,
    FOUNDATION_MOTOR_V2_PROGRAM,
    FOUNDATION_MOTOR_V2_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_STAGE,
    apply_copy_alignment_multicell_teach_weights,
    compile_foundation_motor_v2,
    foundation_motor_payload_transport_cells,
    foundation_motor_v2_action,
    foundation_motor_v2_stage_policy,
    oversample_multicell_copy_cases,
    verify_foundation_motor_v2_curriculum,
)

from ._short_tmp import short_state_root

IDENTITY = "Axon is Axon. Every brother attends the complete exact field."
ROOT = Path(__file__).resolve().parents[1]


def _curriculum():
    return compile_foundation_motor_v2(
        identity_text=IDENTITY,
        requested_counts=(("F0", (36, 36, 36)),),
    )


def test_motor_v2_curriculum_is_deterministic_complete_and_split_disjoint() -> None:
    first = _curriculum()
    second = _curriculum()
    assert first.manifest_id == second.manifest_id
    assert first.actual_family_split_counts == (("F0", (36, 36, 36)),)
    verify_foundation_motor_v2_curriculum(first)
    sources: dict[str, set[str]] = defaultdict(set)
    actions: dict[str, set[str]] = defaultdict(set)
    for case in first.cases:
        assert f"foundation_stage:{FOUNDATION_MOTOR_V2_STAGE}" in case.episode.mechanism_tags
        assert f"foundation_program:{FOUNDATION_MOTOR_V2_PROGRAM_ID}" in case.episode.mechanism_tags
        sources[case.episode.split].add(
            next(
                tag[len("transfer_item:") :]
                for tag in case.episode.mechanism_tags
                if tag.startswith("transfer_item:")
            )
        )
        actions[case.episode.split].add(
            next(
                tag[len("foundation_motor_action:") :]
                for tag in case.episode.mechanism_tags
                if tag.startswith("foundation_motor_action:")
            )
        )
    assert all(actions[split] == set(FOUNDATION_MOTOR_ACTIONS) for split in actions)
    assert sources["train"].isdisjoint(sources["heldout"])
    assert sources["train"].isdisjoint(sources["regression"])
    assert sources["heldout"].isdisjoint(sources["regression"])


def test_compile_script_program_v2_publishes() -> None:
    state_root = short_state_root("motor-v2") / "State"
    with HeartHost(state_root=state_root) as host:
        host.amend_identity(
            IDENTITY,
            amendment_id="foundation-motor-v2-test-identity-v1",
            evidence_ids=("foundation-motor-v2-test-evidence",),
            provenance="tests/test_foundation_motor_v2_curriculum.py",
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compile_foundation_motor.py"),
            "--state-root",
            str(state_root),
            "--program",
            "v2",
            "--train",
            "36",
            "--heldout",
            "36",
            "--regression",
            "36",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout)
    assert result["program"] == "v2"
    assert result["case_count"] == 108
    assert Path(result["manifest_path"]).is_file()


def test_multicell_teach_overlay_does_not_change_program_identity() -> None:
    policy = foundation_motor_v2_stage_policy("copy_alignment")
    taught = apply_copy_alignment_multicell_teach_weights(
        policy["component_weights"],
        training_stage="copy_alignment",
    )
    assert taught["alignment_position"] == 4.0
    assert taught["alignment_copy_gate"] == 0.25
    assert policy["component_weights"]["alignment_copy_gate"] == 4.0
    later = apply_copy_alignment_multicell_teach_weights(
        foundation_motor_v2_stage_policy("transport_eos")["component_weights"],
        training_stage="transport_eos",
    )
    assert later == dict(foundation_motor_v2_stage_policy("transport_eos")["component_weights"])
    assert COPY_ALIGNMENT_MULTICELL_TEACH["alignment_position_reduction"] == "sum"
    assert canonical_sha256(FOUNDATION_MOTOR_V2_PROGRAM) == FOUNDATION_MOTOR_V2_PROGRAM_ID


def test_oversample_multicell_copy_cases_repeats_authored_multibyte_letters() -> None:
    curriculum = compile_foundation_motor_v2(
        identity_text=IDENTITY,
        requested_counts=(("F0", (72, 36, 36)),),
    )
    eligible = set(foundation_motor_v2_stage_policy("copy_alignment")["eligible_actions"])
    train = tuple(
        case
        for case in curriculum.cases
        if case.episode.split == "train"
        and foundation_motor_v2_action(case.episode) in eligible
    )
    multi = tuple(
        case
        for case in train
        if foundation_motor_payload_transport_cells(case.episode) > 1
    )
    assert len(multi) >= 1
    taught = oversample_multicell_copy_cases(train)
    assert all(case in taught for case in train)
    assert taught.count(multi[0]) > train.count(multi[0])
    assert sum(
        1 for case in taught if foundation_motor_payload_transport_cells(case.episode) > 1
    ) >= len(train) - len(multi)
