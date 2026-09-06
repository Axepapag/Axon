from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from runtime.heart import HeartHost
from training.foundation_motor_curriculum import (
    FOUNDATION_MOTOR_ACTIONS,
    FOUNDATION_MOTOR_V2_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_STAGE,
    compile_foundation_motor_v2,
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
