"""A ratified objective repair must be reachable from a launcher, not just defined.

The failure this guards against: the v6 termination repair was ratified, defined,
unit-tested, and then never executed, because no job config selected it.  Every
tranche launched after ratification re-tested the legacy route the autopsy had
already named as a broken fixed point, and the run's own numbers reproduced that
prediction exactly (payload_teacher_forced_eos_accuracy 0.2917 with
payload_content_accuracy 0.8710, and termination_continue_positions 0.0 for all
600 steps).  Defining a repair is not shipping it.

The second failure this guards against: the rejected route stayed reachable by
simply omitting the flags, because --receipt-continuation defaulted to absent and
--receipt-teaching-profile defaulted to continuation_v1.  Documenting a rejection
is not enforcing it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from runtime.heart import HeartHost
from runtime.trainer.cloud_jobs import CloudPacketError
from scripts.axon_kaggle import audit_termination_route
from training import (
    FOUNDATION_MOTOR_V2_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_BALANCED_PROGRAM_ID,
    RECEIPT_TERMINATION_HEAD_PROFILES,
    RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
    RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
    RECEIPT_TEACHING_PROFILES,
    foundation_motor_v2_ratified_termination_profiles,
    foundation_motor_v2_termination_objective_declares_continue_supervision,
    foundation_motor_v2_termination_route_rejection,
)

from ._short_tmp import short_state_root

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs" / "kaggle"
IDENTITY = "Axon is Axon. Every brother attends the complete exact field."

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


def test_the_ratified_termination_set_is_derived_from_the_objectives() -> None:
    """Membership must come from each objective's own declaration, not a list.

    A hand-maintained list of "good" profiles is fail-OPEN: a newly added
    profile is silently trainable until someone remembers to classify it.  The
    ratified set is computed from the objectives themselves, so a new profile
    that does not declare explicit symmetric continue supervision is refused by
    default.
    """

    ratified = foundation_motor_v2_ratified_termination_profiles()
    assert ratified == (RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,)
    assert set(ratified).issubset(RECEIPT_TEACHING_PROFILES)
    for profile in RECEIPT_TEACHING_PROFILES:
        # The set is exactly the profiles whose own teach dict declares it.
        assert foundation_motor_v2_termination_objective_declares_continue_supervision(
            profile
        ) == (profile in ratified)


def test_a_rejected_termination_route_has_no_accepted_reason() -> None:
    """Every legacy entry point into motor-v2 must be refused with a reason."""

    cases = [
        ("pre-receipt base", dict(teach_multicell_copy=False)),
        ("pre-receipt multicell", dict(teach_multicell_copy=True)),
        (
            "continuation_v1",
            dict(teach_multicell_copy=False, receipt_continuation=True, receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_CONTINUATION_V1),
        ),
        (
            "route_eos_balanced_v2",
            dict(teach_multicell_copy=False, receipt_continuation=True, receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2),
        ),
        (
            "generate_head_eos_v3",
            dict(teach_multicell_copy=False, receipt_continuation=True, receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3),
        ),
        (
            "generate_head_balanced_v4",
            dict(teach_multicell_copy=False, receipt_continuation=True, receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4),
        ),
        (
            "termination_head_v5",
            dict(teach_multicell_copy=False, receipt_continuation=True, receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5),
        ),
    ]
    for label, kwargs in cases:
        reason = foundation_motor_v2_termination_route_rejection(**kwargs)
        assert reason, label
        assert "refusing to train" in reason, label

    # The emission rung's own program id must be named, so the receipt of what
    # actually ran is visible in the refusal.
    base_reason = foundation_motor_v2_termination_route_rejection(teach_multicell_copy=False)
    assert FOUNDATION_MOTOR_V2_PROGRAM_ID in base_reason

    # Only the ratified profile is trainable.
    assert (
        foundation_motor_v2_termination_route_rejection(
            teach_multicell_copy=False,
            receipt_continuation=True,
            receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
        )
        is None
    )


def test_no_packetable_launcher_can_upload_a_rejected_route() -> None:
    """The rejected set among configs/kaggle is exactly the measured legacy set.

    These are the launchers that ran or would run a motor-v2 termination
    objective with no explicit continue supervision.  They are receipts of what
    ran, so they stay in the repository; the gate is what stops them from being
    packed and uploaded again.  If a new launcher is added on a rejected route,
    this fails and forces the choice to be explicit.
    """

    refused = []
    for path in sorted(CONFIGS.glob("*.json")):
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        if "entrypoint_argv" not in config:
            continue
        try:
            audit_termination_route(
                config["entrypoint_argv"], state_root=ROOT / "State", source=path.name
            )
        except CloudPacketError as exc:
            assert "refusing to train" in str(exc), path
            refused.append(path.name)
    assert set(refused) == {
        "axon_d64_emission_rung_first_tranche.json",
        "axon_d64_mixer_4l_ffn256_h1_copy_alignment_multicell_teach.json",
        "axon_d64_mixer_4l_ffn256_h1_copy_alignment_renewal.json",
        "axon_d64_mixer_4l_ffn256_h1_copy_alignment_smoke.json",
        "axon_d64_mixer_4l_ffn256_h1_multicell_v2_smoke.json",
        "axon_d64_mixer_4l_ffn256_h1_receipt_ablation_v1.json",
        "axon_d64_mixer_4l_ffn256_h1_receipt_route_eos_balanced_v2.json",
        "axon_d64_mixer_4l_ffn256_h1_unicode_walk_v3.json",
        "axon_d64_mixer_4l_ffn512_h1_copy_alignment_smoke.json",
    }


def test_the_ratified_launcher_is_still_packetable() -> None:
    """A gate that also blocks the repair would be worse than no gate."""

    config = json.loads(
        (CONFIGS / "axon_d64_emission_rung_v6_termination_balanced.json").read_text(
            encoding="utf-8-sig"
        )
    )
    audit_termination_route(
        config["entrypoint_argv"], state_root=ROOT / "State", source="v6 launcher"
    )
    # Read-only evaluation of a historical bundle is still permitted.
    audit_termination_route(
        [
            "python",
            "scripts/train_living_reasoning_smoke.py",
            "--curriculum-manifest",
            "State/training/curricula/ffcs_v1/a872278fd0e8ef926370e1712d01dcf0a277672c4a0088af44aef483d8417740/manifest.json",
            "--evaluate-only",
        ],
        state_root=ROOT / "State",
        source="historical evaluation",
    )


@pytest.fixture(scope="module")
def motor_v2_manifest():
    """A real motor-v2 campaign, so the gate is proven against the real trainer."""

    from training.first_form_curriculum import publish_first_form_curriculum
    from training.foundation_motor_curriculum import compile_foundation_motor_v2

    state_root = short_state_root("routegate") / "State"
    with HeartHost(state_root=state_root) as host:
        host.amend_identity(
            IDENTITY,
            amendment_id="termination-route-gate-identity-v1",
            evidence_ids=("termination-route-gate-evidence",),
            provenance="tests/test_termination_repair_is_launched.py",
        )
    curriculum = compile_foundation_motor_v2(
        identity_text=IDENTITY, requested_counts=(("F0", (36, 36, 36)),)
    )
    return state_root, publish_first_form_curriculum(curriculum, state_root=state_root)


def _trainer_argv(state_root: Path, manifest: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "train_living_reasoning_smoke.py"),
        "--state-root",
        str(state_root),
        "--device",
        "cpu",
        "--curriculum-manifest",
        str(manifest),
        "--candidate-label",
        "termination-route-gate",
        "--heads",
        "1",
        "--layers",
        "1",
        "--ffn-dim",
        "128",
        "--page-size",
        "32",
        "--tranche-steps",
        "1",
        "--checkpoint-interval",
        "1",
        "--evaluation-case-limit",
        "2",
        "--seed",
        "1",
        *extra,
    ]


def test_the_trainer_refuses_every_rejected_route_before_any_compute(
    motor_v2_manifest,
) -> None:
    """Omitting the flags must not silently restore the rejected route.

    This is the loophole that cost nine tranches: --receipt-continuation
    defaulted to absent and --receipt-teaching-profile defaulted to
    continuation_v1, so a hand-written argv ran the rejected objective without
    ever naming it.  The refusal happens before any compute, so it costs a
    manifest load and nothing else.
    """

    state_root, manifest = motor_v2_manifest
    rejected = (
        ("legacy base", ()),
        (
            "termination_head_v5",
            (
                "--receipt-continuation",
                "--receipt-teaching-profile",
                RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
                "--termination-head-route",
            ),
        ),
        (
            "route_eos_balanced_v2",
            (
                "--receipt-continuation",
                "--receipt-teaching-profile",
                RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
            ),
        ),
    )
    for label, extra in rejected:
        completed = subprocess.run(
            _trainer_argv(state_root, manifest, *extra),
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=1800,
            check=False,
        )
        assert completed.returncode != 0, label
        assert "refusing to train" in completed.stderr, (label, completed.stderr[-2000:])


def test_the_trainer_still_runs_the_ratified_route(motor_v2_manifest) -> None:
    """A refusal that also blocked the repair would leave nothing launchable."""

    state_root, manifest = motor_v2_manifest
    completed = subprocess.run(
        _trainer_argv(
            state_root,
            manifest,
            "--receipt-continuation",
            "--receipt-teaching-profile",
            RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
            "--termination-head-route",
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, f"{completed.stdout[-2000:]}\n{completed.stderr[-4000:]}"
