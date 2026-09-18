"""A ratified objective repair must be reachable from a launcher, not just defined.

The failure this guards against: the v6 termination repair was ratified, defined,
unit-tested, and then never executed, because no job config selected it.  Every
tranche launched after ratification re-tested the legacy route the autopsy had
already named as a broken fixed point, and the run's own numbers reproduced that
prediction exactly (payload_teacher_forced_eos_accuracy 0.2917 with
payload_content_accuracy 0.8710, and termination_continue_positions 0.0 for all
600 steps).  Defining a repair is not shipping it.
"""

from __future__ import annotations

import json
from pathlib import Path

from training import (
    FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_BALANCED_PROGRAM_ID,
    RECEIPT_TERMINATION_HEAD_PROFILES,
    RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
    RECEIPT_TEACHING_PROFILES,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs" / "kaggle"

GENERATE_HEAD_PROFILES = {
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4,
}


def _configs() -> list[tuple[Path, dict]]:
    found: list[tuple[Path, dict]] = []
    for path in sorted(CONFIGS.rglob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(config, dict) and "entrypoint_argv" in config:
            found.append((path, config))
    return found


def _option(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def test_a_launched_config_selects_the_ratified_v6_termination_repair() -> None:
    selected = [
        path
        for path, config in _configs()
        if _option(config["entrypoint_argv"], "--receipt-teaching-profile")
        == RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6
    ]
    assert selected, (
        "the ratified v6 termination-balanced objective is not selected by any "
        "configs/kaggle launcher, so the repair cannot be executed by a tranche"
    )

    for path in selected:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        argv = config["entrypoint_argv"]
        assert "--receipt-continuation" in argv, path
        assert "--termination-head-route" in argv, path
        assert FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_BALANCED_PROGRAM_ID in config[
            "notes"
        ], path


def test_no_launcher_selects_an_objective_profile_it_cannot_execute() -> None:
    for path, config in _configs():
        argv = config["entrypoint_argv"]
        if "--receipt-teaching-profile" not in argv:
            continue
        profile = _option(argv, "--receipt-teaching-profile")
        assert profile in RECEIPT_TEACHING_PROFILES, (path, profile)
        if profile != RECEIPT_TEACHING_PROFILE_CONTINUATION_V1:
            assert "--receipt-continuation" in argv, (path, profile)
        assert ("--termination-head-route" in argv) == (
            profile in RECEIPT_TERMINATION_HEAD_PROFILES
        ), (path, profile)
        assert ("--eos-generate-head-route" in argv) == (
            profile in GENERATE_HEAD_PROFILES
        ), (path, profile)
        assert not (
            "--termination-head-route" in argv and "--eos-generate-head-route" in argv
        ), path
