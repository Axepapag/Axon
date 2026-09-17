import pytest

from runtime.field import canonical_sha256
from training.foundation_motor_curriculum import (
    COPY_ALIGNMENT_MULTICELL_TEACH,
    COPY_ALIGNMENT_MULTICELL_TEACH_ID,
    FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM,
    FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM,
    FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM,
    FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM,
    FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM,
    FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RETENTION_CONTRACT,
    FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_ID,
    FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2,
    FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2_ID,
    FOUNDATION_MOTOR_V2_STAGE_ORDER,
    RECEIPT_CONTINUATION_TEACH,
    RECEIPT_CONTINUATION_TEACH_ID,
    RECEIPT_GENERATE_HEAD_BALANCED_TEACH,
    RECEIPT_GENERATE_HEAD_BALANCED_TEACH_ID,
    RECEIPT_GENERATE_HEAD_EOS_TEACH,
    RECEIPT_GENERATE_HEAD_EOS_TEACH_ID,
    RECEIPT_ROUTE_EOS_BALANCED_TEACH,
    RECEIPT_ROUTE_EOS_BALANCED_TEACH_ID,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
    RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
    apply_receipt_continuation_teach_weights,
    decide_foundation_motor_v2_checkpoint_retention,
    decide_foundation_motor_v2_checkpoint_retention_v2,
    decide_foundation_motor_v2_stage,
    foundation_motor_v2_objective_program_id,
    foundation_motor_v2_stage_policy,
    foundation_motor_v2_unreachable_gate_requirements,
    resolve_retention_action,
)


def test_multicell_overlay_has_distinct_effective_objective_identity() -> None:
    assert canonical_sha256(COPY_ALIGNMENT_MULTICELL_TEACH) == (
        COPY_ALIGNMENT_MULTICELL_TEACH_ID
    )
    assert canonical_sha256(FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM) == (
        FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID
    )
    assert FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID != FOUNDATION_MOTOR_V2_PROGRAM_ID
    assert foundation_motor_v2_objective_program_id(teach_multicell_copy=True) == (
        FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID
    )
    assert foundation_motor_v2_objective_program_id(teach_multicell_copy=False) == (
        FOUNDATION_MOTOR_V2_PROGRAM_ID
    )


def test_receipt_conduit_has_new_eos_protected_objective_identity() -> None:
    assert canonical_sha256(RECEIPT_CONTINUATION_TEACH) == RECEIPT_CONTINUATION_TEACH_ID
    assert canonical_sha256(FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM) == (
        FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM_ID
    )
    assert FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM_ID not in {
        FOUNDATION_MOTOR_V2_PROGRAM_ID,
        FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID,
    }
    assert foundation_motor_v2_objective_program_id(
        teach_multicell_copy=True,
        receipt_continuation=True,
    ) == FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM_ID
    weights = RECEIPT_CONTINUATION_TEACH["component_weight_overrides"]
    assert weights["alignment_eos_gate"] > 0.0
    assert RECEIPT_CONTINUATION_TEACH["same_stage_eos_retention_required"] is True


def test_receipt_copy_alignment_gate_requires_same_stage_eos_retention() -> None:
    perfect = {
        "complete_field_coverage_rate": 1.0,
        "alignment_position_accuracy": 1.0,
        "alignment_copy_gate_accuracy": 1.0,
        "alignment_eos_gate_accuracy": 1.0,
        "payload_eos_accuracy": 1.0,
        "payload_content_accuracy": 1.0,
        "payload_content_constant_floor": 0.0,
        "payload_transport_exact_rate": 1.0,
        "typed_emission_exact_rate": 1.0,
        "constant_typed_emission_exact_floor": 1.0 / 3.0,
        "constant_payload_transport_exact_floor": 1.0 / 3.0,
        "pair_exact_rates": {
            "position": 1.0,
            "copy_gate": 1.0,
            "eos_gate": 1.0,
            "content": 1.0,
        },
    }
    passed = decide_foundation_motor_v2_stage(
        training_stage="copy_alignment",
        heldout_probe=perfect,
        regression_probe=perfect,
        complete_heldout=True,
        complete_regression=True,
        receipt_continuation=True,
    )
    assert passed["passed"] is True
    assert passed["effective_objective_program_id"] == (
        FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM_ID
    )

    regressed = {**perfect, "payload_eos_accuracy": 0.0}
    failed = decide_foundation_motor_v2_stage(
        training_stage="copy_alignment",
        heldout_probe=regressed,
        regression_probe=perfect,
        complete_heldout=True,
        complete_regression=True,
        receipt_continuation=True,
    )
    assert failed["passed"] is False
    assert any("payload_eos_accuracy" in reason for reason in failed["failures"])


def test_route_eos_balanced_receipt_profile_is_a_distinct_fresh_objective() -> None:
    assert canonical_sha256(RECEIPT_ROUTE_EOS_BALANCED_TEACH) == (
        RECEIPT_ROUTE_EOS_BALANCED_TEACH_ID
    )
    assert canonical_sha256(
        FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM
    ) == FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM_ID
    assert FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM_ID not in {
        FOUNDATION_MOTOR_V2_PROGRAM_ID,
        FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID,
        FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM_ID,
    }
    assert foundation_motor_v2_objective_program_id(
        teach_multicell_copy=False,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
    ) == FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM_ID

    weights = apply_receipt_continuation_teach_weights(
        {
            "payload": 1.0,
            "alignment_position": 4.0,
            "alignment_copy_gate": 0.25,
            "alignment_eos_gate": 1.0,
        },
        training_stage="copy_alignment",
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
    )
    assert weights["alignment_position"] == 1.0
    assert weights["alignment_copy_gate"] == 4.0
    assert weights["alignment_eos_gate"] == 2.0


def test_route_eos_balanced_stage_gate_reports_exact_objective_identity() -> None:
    perfect = {
        "complete_field_coverage_rate": 1.0,
        "alignment_position_accuracy": 1.0,
        "alignment_copy_gate_accuracy": 1.0,
        "alignment_eos_gate_accuracy": 1.0,
        "payload_eos_accuracy": 1.0,
        "payload_content_accuracy": 1.0,
        "payload_content_constant_floor": 0.0,
        "payload_transport_exact_rate": 1.0,
        "typed_emission_exact_rate": 1.0,
        "constant_typed_emission_exact_floor": 1.0 / 3.0,
        "constant_payload_transport_exact_floor": 1.0 / 3.0,
        "pair_exact_rates": {
            "position": 1.0,
            "copy_gate": 1.0,
            "eos_gate": 1.0,
            "content": 1.0,
        },
    }
    decision = decide_foundation_motor_v2_stage(
        training_stage="copy_alignment",
        heldout_probe=perfect,
        regression_probe=perfect,
        complete_heldout=True,
        complete_regression=True,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
    )
    assert decision["passed"] is True
    assert decision["effective_objective_program_id"] == (
        FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM_ID
    )
    assert decision["receipt_teaching_profile"] == (
        RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2
    )


def test_generate_head_eos_profile_has_distinct_objective_and_matching_gate() -> None:
    assert canonical_sha256(RECEIPT_GENERATE_HEAD_EOS_TEACH) == (
        RECEIPT_GENERATE_HEAD_EOS_TEACH_ID
    )
    assert canonical_sha256(
        FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM
    ) == FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID
    assert RECEIPT_GENERATE_HEAD_EOS_TEACH["component_weight_overrides"][
        "alignment_eos_gate"
    ] == 0.0
    assert foundation_motor_v2_objective_program_id(
        teach_multicell_copy=False,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
    ) == FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID

    exact_without_legacy_eos_gate = {
        "complete_field_coverage_rate": 1.0,
        "alignment_position_accuracy": 1.0,
        "alignment_copy_gate_accuracy": 1.0,
        "alignment_eos_gate_accuracy": 0.0,
        "payload_eos_accuracy": 1.0,
        "payload_content_accuracy": 1.0,
        "payload_content_constant_floor": 0.0,
        "payload_transport_exact_rate": 1.0,
        "typed_emission_exact_rate": 1.0,
        "constant_typed_emission_exact_floor": 1.0 / 3.0,
        "constant_payload_transport_exact_floor": 1.0 / 3.0,
        "pair_exact_rates": {
            "position": 1.0,
            "copy_gate": 1.0,
            "eos_gate": 0.0,
            "content": 1.0,
        },
    }
    decision = decide_foundation_motor_v2_stage(
        training_stage="copy_alignment",
        heldout_probe=exact_without_legacy_eos_gate,
        regression_probe=exact_without_legacy_eos_gate,
        complete_heldout=True,
        complete_regression=True,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
        eos_generate_head_route=True,
    )
    assert decision["passed"] is True
    assert decision["effective_objective_program_id"] == (
        FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID
    )

    with pytest.raises(ValueError, match="must be selected together"):
        decide_foundation_motor_v2_stage(
            training_stage="copy_alignment",
            heldout_probe=exact_without_legacy_eos_gate,
            regression_probe=exact_without_legacy_eos_gate,
            complete_heldout=True,
            complete_regression=True,
            receipt_continuation=True,
            receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
        )


def test_copy_alignment_teaches_emission_before_termination() -> None:
    """Ratified 2026-09-17: stage 0 supervises payload content, no stop pressure.

    Before this, copy_alignment declared payload=0.0, so it certified a core
    that emits nothing, and transport_eos then demanded emission and termination
    in the same stage step against a still-saturated copy gate.  For an
    untrained emitter the cheapest loss reduction was to emit nothing and stop.
    """

    weights = foundation_motor_v2_stage_policy("copy_alignment")["component_weights"]
    assert weights["payload"] == 1.0
    assert weights["alignment_eos_gate"] == 0.0
    assert weights["alignment_copy_gate"] == 4.0
    assert weights["alignment_position"] == 1.0

    # No component may jump 0.0 -> 1.0 across a stage boundary any more: the
    # emission lesson lands in stage 0 and every later stage retains it.
    for index in range(len(FOUNDATION_MOTOR_V2_STAGE_ORDER) - 1):
        later = FOUNDATION_MOTOR_V2_STAGE_ORDER[index + 1]
        after = foundation_motor_v2_stage_policy(later)["component_weights"]
        assert after["payload"] > 0.0, f"{later} must keep supervising payload"

    # Exactly the emit-nothing dead state: every case is "correct" only where
    # the right answer is to emit nothing.  The floors are the real ones measured
    # off the 72-case heldout surface, so typed exactness sitting at the floor is
    # scored as the failure it is rather than as 33.3% of progress.
    empty_emitter = {
        "complete_field_coverage_rate": 1.0,
        "alignment_position_accuracy": 1.0,
        "alignment_copy_gate_accuracy": 1.0,
        "payload_content_accuracy": 0.0,
        "payload_content_constant_floor": 0.0,
        "payload_transport_exact_rate": 0.0,
        "typed_emission_exact_rate": 1.0 / 3.0,
        "constant_typed_emission_exact_floor": 1.0 / 3.0,
        "constant_payload_transport_exact_floor": 1.0 / 3.0,
        "pair_exact_rates": {
            "position": 1.0,
            "copy_gate": 1.0,
            "eos_gate": 1.0,
            "content": 0.0,
        },
    }
    decision = decide_foundation_motor_v2_stage(
        training_stage="copy_alignment",
        heldout_probe=empty_emitter,
        regression_probe=empty_emitter,
        complete_heldout=True,
        complete_regression=True,
    )
    assert decision["passed"] is False
    assert any("payload_content_accuracy" in reason for reason in decision["failures"])
    assert any("payload_transport_exact_rate" in reason for reason in decision["failures"])
    assert any(
        "payload content does not beat constant floor" in reason
        for reason in decision["failures"]
    )
    # Typed exactness is a DELTA conjunction whose decision/operation/region/
    # start/end components are weighted 0.0 at this stage, so gating on it here
    # would be unsatisfiable by construction.  The dead state is rejected by the
    # content and transport requirements instead, and the floor comparison is
    # asserted at the stages that do carry those weights.
    assert not any(
        "typed_emission_exact_rate" in reason for reason in decision["failures"]
    )
    assert foundation_motor_v2_unreachable_gate_requirements() == []
    assert any(
        "changed-source content below" in reason for reason in decision["failures"]
    )

    emitting = {
        **empty_emitter,
        "payload_content_accuracy": 1.0,
        "payload_transport_exact_rate": 1.0,
        "typed_emission_exact_rate": 1.0,
        "pair_exact_rates": {**empty_emitter["pair_exact_rates"], "content": 1.0},
    }
    assert (
        decide_foundation_motor_v2_stage(
            training_stage="copy_alignment",
            heldout_probe=emitting,
            regression_probe=emitting,
            complete_heldout=True,
            complete_regression=True,
        )["passed"]
        is True
    )


def test_reference_motor_retention_contract_rejects_step_96_collapse() -> None:
    assert canonical_sha256(FOUNDATION_MOTOR_V2_RETENTION_CONTRACT) == (
        FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_ID
    )
    step_64 = {
        "alignment_position_accuracy": 1.0,
        "alignment_copy_gate_accuracy": 1.0,
        "payload_content_accuracy": 0.875,
        "payload_eos_accuracy": 0.75,
        "payload_transport_exact_rate": 2.0 / 3.0,
        "pair_exact_rates": {"position": 1.0, "copy_gate": 1.0, "content": 0.75},
    }
    step_96 = {
        "alignment_position_accuracy": 1.0,
        "alignment_copy_gate_accuracy": 1.0,
        "payload_content_accuracy": 0.0,
        "payload_eos_accuracy": 1.0,
        "payload_transport_exact_rate": 0.0,
        "pair_exact_rates": {"position": 1.0, "copy_gate": 1.0, "content": 0.0},
    }
    decision = decide_foundation_motor_v2_checkpoint_retention(
        training_stage="copy_alignment",
        accepted_heldout_probe=step_64,
        accepted_regression_probe=step_64,
        candidate_heldout_probe=step_96,
        candidate_regression_probe=step_96,
        complete_heldout=True,
        complete_regression=True,
    )
    assert decision["passed"] is False
    assert any("payload_transport_exact_rate regressed" in item for item in decision["failures"])
    assert any("payload_content_accuracy regressed" in item for item in decision["failures"])


def test_reference_motor_retention_contract_accepts_non_regressing_checkpoint() -> None:
    before = {
        "alignment_position_accuracy": 0.5,
        "alignment_copy_gate_accuracy": 0.5,
        "payload_content_accuracy": 0.25,
        "payload_eos_accuracy": 0.5,
        "payload_transport_exact_rate": 0.25,
        "pair_exact_rates": {"position": 0.5, "copy_gate": 0.5, "content": 0.0},
    }
    after = {
        "alignment_position_accuracy": 1.0,
        "alignment_copy_gate_accuracy": 1.0,
        "payload_content_accuracy": 0.5,
        "payload_eos_accuracy": 0.75,
        "payload_transport_exact_rate": 0.5,
        "pair_exact_rates": {"position": 1.0, "copy_gate": 1.0, "content": 0.5},
    }
    decision = decide_foundation_motor_v2_checkpoint_retention(
        training_stage="copy_alignment",
        accepted_heldout_probe=before,
        accepted_regression_probe=before,
        candidate_heldout_probe=after,
        candidate_regression_probe=after,
        complete_heldout=True,
        complete_regression=True,
    )
    assert decision["passed"] is True
    assert decision["surface_kind"] == "complete"
    assert decision["improvements"]


def test_reference_motor_retention_contract_refuses_bounded_or_stalled_probe() -> None:
    probe = {
        "alignment_position_accuracy": 0.5,
        "alignment_copy_gate_accuracy": 0.5,
        "payload_content_accuracy": 0.25,
        "payload_eos_accuracy": 0.5,
        "payload_transport_exact_rate": 0.25,
        "pair_exact_rates": {"position": 0.5, "copy_gate": 0.5, "content": 0.0},
    }
    bounded = decide_foundation_motor_v2_checkpoint_retention(
        training_stage="copy_alignment",
        accepted_heldout_probe=probe,
        accepted_regression_probe=probe,
        candidate_heldout_probe=probe,
        candidate_regression_probe=probe,
        complete_heldout=False,
        complete_regression=False,
    )
    assert bounded["passed"] is False
    assert "heldout retention surface is incomplete" in bounded["failures"]
    assert "no protected motor behavior improved" in bounded["failures"]


def test_generate_head_balanced_v4_has_fresh_objective_and_unit_eos_weight() -> None:
    assert RECEIPT_GENERATE_HEAD_BALANCED_TEACH["payload_eos_weight"] == 1.0
    assert canonical_sha256(RECEIPT_GENERATE_HEAD_BALANCED_TEACH) == (
        RECEIPT_GENERATE_HEAD_BALANCED_TEACH_ID
    )
    assert canonical_sha256(
        FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM
    ) == FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM_ID
    assert foundation_motor_v2_objective_program_id(
        teach_multicell_copy=False,
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4,
    ) == FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM_ID


_PROBE_BASE = {
    "alignment_position_accuracy": 1.0,
    "alignment_copy_gate_accuracy": 1.0,
    "payload_content_accuracy": 1.0,
    "payload_eos_accuracy": 0.0,
    "payload_transport_exact_rate": 0.0,
    "pair_exact_rates": {"position": 1.0, "copy_gate": 1.0, "content": 1.0},
}


def _probe(**overrides):
    probe = dict(_PROBE_BASE)
    probe["pair_exact_rates"] = dict(_PROBE_BASE["pair_exact_rates"])
    probe.update(overrides)
    return probe


def test_retention_contract_v2_classifies_three_states() -> None:
    assert canonical_sha256(FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2) == (
        FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2_ID
    )
    assert FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2_ID != FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_ID

    improved = decide_foundation_motor_v2_checkpoint_retention_v2(
        training_stage="copy_alignment",
        accepted_heldout_probe=_probe(),
        accepted_regression_probe=_probe(),
        candidate_heldout_probe=_probe(payload_eos_accuracy=1.0),
        candidate_regression_probe=_probe(payload_eos_accuracy=1.0),
        complete_heldout=True,
        complete_regression=True,
    )
    assert improved["state"] == "improvement"
    assert improved["passed"] is True

    regressed = decide_foundation_motor_v2_checkpoint_retention_v2(
        training_stage="copy_alignment",
        accepted_heldout_probe=_probe(),
        accepted_regression_probe=_probe(),
        candidate_heldout_probe=_probe(payload_content_accuracy=0.0),
        candidate_regression_probe=_probe(),
        complete_heldout=True,
        complete_regression=True,
    )
    assert regressed["state"] == "regression"
    assert regressed["passed"] is False
    assert any("payload_content_accuracy regressed" in f for f in regressed["failures"])

    plateau = decide_foundation_motor_v2_checkpoint_retention_v2(
        training_stage="copy_alignment",
        accepted_heldout_probe=_probe(),
        accepted_regression_probe=_probe(),
        candidate_heldout_probe=_probe(),
        candidate_regression_probe=_probe(),
        complete_heldout=True,
        complete_regression=True,
    )
    # Plateau is its own state, not a failure: no "nothing improved" clause.
    assert plateau["state"] == "plateau"
    assert plateau["passed"] is False
    assert plateau["failures"] == []
    assert plateau["improvements"] == []

    incomplete = decide_foundation_motor_v2_checkpoint_retention_v2(
        training_stage="copy_alignment",
        accepted_heldout_probe=_probe(),
        accepted_regression_probe=_probe(),
        candidate_heldout_probe=_probe(),
        candidate_regression_probe=_probe(),
        complete_heldout=False,
        complete_regression=True,
    )
    assert incomplete["state"] == "regression"  # fail closed
    assert any("incomplete" in f for f in incomplete["failures"])


def test_resolve_retention_action_probation_budget() -> None:
    improvement = {"state": "improvement"}
    plateau = {"state": "plateau"}
    regression = {"state": "regression"}
    assert resolve_retention_action(improvement, probation_count=2, max_plateau_probation=3) == "accept"
    assert resolve_retention_action(plateau, probation_count=0, max_plateau_probation=3) == "probate"
    assert resolve_retention_action(plateau, probation_count=2, max_plateau_probation=3) == "probate"
    assert resolve_retention_action(plateau, probation_count=3, max_plateau_probation=3) == "rollback"
    assert resolve_retention_action(regression, probation_count=0, max_plateau_probation=3) == "rollback"
    # Legacy two-state behavior: zero allowance abandons a plateau immediately.
    assert resolve_retention_action(plateau, probation_count=0, max_plateau_probation=0) == "rollback"


def test_probation_sidecar_round_trip_and_identity_checks(tmp_path) -> None:
    from runtime.trainer import CandidateCheckpointRecord
    from scripts.train_living_reasoning_smoke import (
        PROBATION_SIDECAR_SCHEMA,
        _load_probation_sidecar,
        _probation_sidecar_path,
        _write_probation_sidecar,
    )

    def record(step: int) -> dict:
        return CandidateCheckpointRecord(
            module_id="reasoning-d64-test",
            base_generation_id="g0",
            candidate_generation_id="g1",
            plan_id="a1" * 32,
            authorization_id="a" * 64,
            learning_policy_id="b2" * 32,
            step=step,
            micro_step=0,
            accumulation_index=0,
            parameter_manifest_id="m" * 64,
            artifact_relpath="checkpoints/x.pt",
            artifact_sha256="ab" * 32,
            artifact_bytes=128,
            optimizer_included=True,
            gradient_state_included=True,
            scaler_included=False,
            current_learning_rate=1e-4,
            accumulated_loss_sum=1.0,
        ).to_canonical_dict()

    def load_sidecar(**overrides):
        kwargs = {
            "path": path,
            "candidate_generation": "g1",
            "learning_policy": {"optimizer": "adamw", "learning_rate": 1e-4},
            "effective_objective_program_id": "o" * 64,
            "architecture_id": "arch-1",
            "standard_ffcs_manifest_ids": ["f" * 64],
            "max_plateau_probation": 3,
        }
        kwargs.update(overrides)
        return _load_probation_sidecar(**kwargs)

    path = _probation_sidecar_path(tmp_path)
    assert load_sidecar() is None

    _write_probation_sidecar(
        path,
        {
            "schema": PROBATION_SIDECAR_SCHEMA,
            "candidate_generation_id": "g1",
            "learning_policy": {"optimizer": "adamw", "learning_rate": 1e-4},
            "effective_objective_program_id": "o" * 64,
            "architecture_id": "arch-1",
            "standard_ffcs_manifest_ids": ["f" * 64],
            "max_plateau_probation": 3,
            "confirmed_checkpoint": record(24),
            "confirmed_step": 24,
            "probationary_checkpoint": record(32),
            "probationary_step": 32,
            "probation_count": 1,
            "reference_evaluation": {"heldout_mean_loss": 1.0},
            "reference_report_id": "r" * 64,
        },
    )
    loaded = load_sidecar()
    assert loaded is not None and loaded["probation_count"] == 1

    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_sidecar(architecture_id="arch-2")

    _write_probation_sidecar(path, {**loaded, "probation_count": 3})
    with pytest.raises(RuntimeError, match="allowance"):
        load_sidecar()


def test_probation_continuation_parent_step_matches_tranche_base(tmp_path) -> None:
    """Regression: a probation-anchored tranche's continuation receipt must
    record the probationary anchor step, not the accepted bundle's step.

    Live failure on d64-reference-termhead-v1 (2026-09-16): tranche 4 leased
    with base_global_step=24 (probationary step) but the receipt carried
    parent_global_step=16 (confirmed accepted parent), so
    TrancheStore.write_continuation fail-closed with 'continuation receipt
    disagrees with its resource tranche' and the probation branch could not
    renew. The receipt binds the lease, and the lease anchors at the step the
    optimizer actually resumes from; the probationary checkpoint's own
    provenance lives in the probation sidecar.
    """
    from runtime.trainer import (
        ResourceTranche,
        TrancheContinuation,
        TrancheStore,
    )
    from scripts.train_living_reasoning_smoke import (
        _continuation_parent_global_step,
    )

    assert _continuation_parent_global_step(16, None) == 16
    sidecar = {"probationary_step": 24}
    assert _continuation_parent_global_step(16, sidecar) == 24

    store = TrancheStore(tmp_path / "trainer")
    prior = ResourceTranche(
        module_id="reasoning-d64-test",
        candidate_generation_id="g1",
        plan_id="a1" * 32,
        learning_policy_id="b2" * 32,
        base_global_step=16,
        steps=8,
        parent_bundle_id="c3" * 32,
        purpose="prior tranche",
    )
    store.write_tranche(prior)
    tranche = ResourceTranche(
        module_id="reasoning-d64-test",
        candidate_generation_id="g1",
        plan_id="a1" * 32,
        learning_policy_id="b2" * 32,
        base_global_step=24,
        steps=8,
        parent_bundle_id="c3" * 32,
        purpose="probation continuation",
    )
    store.write_tranche(tranche)

    stale = TrancheContinuation(
        tranche_id=tranche.tranche_id,
        module_id="reasoning-d64-test",
        candidate_generation_id="g1",
        plan_id="a1" * 32,
        learning_policy_id="b2" * 32,
        parent_bundle_id="c3" * 32,
        parent_checkpoint_id="d4" * 32,
        parent_optimizer_receipt_id="e5" * 32,
        parent_soul_id="f6" * 32,
        parent_global_step=16,
        prior_tranche_id=prior.tranche_id,
    )
    with pytest.raises(Exception, match="disagrees with its resource tranche"):
        store.write_continuation(stale)

    fixed = TrancheContinuation(
        tranche_id=tranche.tranche_id,
        module_id="reasoning-d64-test",
        candidate_generation_id="g1",
        plan_id="a1" * 32,
        learning_policy_id="b2" * 32,
        parent_bundle_id="c3" * 32,
        parent_checkpoint_id="d4" * 32,
        parent_optimizer_receipt_id="e5" * 32,
        parent_soul_id="f6" * 32,
        parent_global_step=_continuation_parent_global_step(16, sidecar),
        prior_tranche_id=prior.tranche_id,
    )
    store.write_continuation(fixed)
    assert fixed.parent_global_step == 24


def test_probation_prior_tranche_proofs_probationary_parent(tmp_path) -> None:
    """Regression: under probation the receipt's prior tranche must be the
    tranche that produced the probationary parent, not the tranche consumed
    at the confirmed accepted bundle.

    Live failure on d64-reference-termhead-v1 (2026-09-16), second layer: with
    the parent step fixed to the probationary anchor (24), the receipt still
    cited the accepted-chain tranche (final step 16), so
    TrancheStore.write_continuation fail-closed with 'prior tranche does not
    reach this continuation parent'. The proof must target the probation
    anchor recorded in the probated segment report.
    """
    import json as _json

    from runtime.field import canonical_sha256
    from runtime.trainer import (
        ResourceTranche,
        TrancheContinuation,
        TrancheStore,
    )
    from scripts.train_living_reasoning_smoke import (
        _continuation_parent_global_step,
        _prior_consumed_tranche_id,
    )

    store = TrancheStore(tmp_path / "trainer")
    confirmed_tranche = ResourceTranche(
        module_id="reasoning-d64-test",
        candidate_generation_id="g1",
        plan_id="a1" * 32,
        learning_policy_id="b2" * 32,
        base_global_step=16,
        steps=8,
        parent_bundle_id="c3" * 32,
        purpose="confirmed parent tranche",
    )
    store.write_tranche(confirmed_tranche)
    probation_tranche = ResourceTranche(
        module_id="reasoning-d64-test",
        candidate_generation_id="g1",
        plan_id="a1" * 32,
        learning_policy_id="b2" * 32,
        base_global_step=16,
        steps=8,
        parent_bundle_id="c3" * 32,
        purpose="probated tranche",
    )
    store.write_tranche(probation_tranche)
    continuation_tranche = ResourceTranche(
        module_id="reasoning-d64-test",
        candidate_generation_id="g1",
        plan_id="a1" * 32,
        learning_policy_id="b2" * 32,
        base_global_step=24,
        steps=8,
        parent_bundle_id="c3" * 32,
        purpose="probation continuation",
    )
    store.write_tranche(continuation_tranche)

    campaign_report_dir = tmp_path / "reports"
    campaign_report_dir.mkdir()
    probated_report = {
        "report_id": None,
        "candidate_generation_id": "g1",
        "segment_end_step": 24,
        "final_checkpoint_id": "d4" * 32,
        "steps": [],
        "resource_tranche": probation_tranche.to_canonical_dict(),
    }
    probated_report["report_id"] = canonical_sha256(
        {key: value for key, value in probated_report.items() if key != "report_id"}
    )
    (campaign_report_dir / "segment_000000017_000000024_probation.json").write_text(
        _json.dumps(probated_report), encoding="utf-8"
    )

    sidecar = {
        "probationary_step": 24,
        "probationary_checkpoint": {"checkpoint_id": "d4" * 32},
    }

    # Without the sidecar the proof targets the confirmed accepted bundle and
    # finds no consumed tranche bound to it here.
    latest_bundle = type("Bundle", (), {"step": 16, "checkpoint_id": "c9" * 32, "bundle_id": "c3" * 32})()
    assert (
        _prior_consumed_tranche_id(
            campaign_report_dir=campaign_report_dir,
            module_id="reasoning-d64-test",
            candidate_generation_id="g1",
            latest_bundle=latest_bundle,
            tranche_store=store,
            plan_id="a1" * 32,
            learning_policy_id="b2" * 32,
        )
        is None
    )

    # With the sidecar the proof targets the probationary parent and returns
    # the probated tranche that produced it.
    proven_prior = _prior_consumed_tranche_id(
        campaign_report_dir=campaign_report_dir,
        module_id="reasoning-d64-test",
        candidate_generation_id="g1",
        latest_bundle=latest_bundle,
        tranche_store=store,
        plan_id="a1" * 32,
        learning_policy_id="b2" * 32,
        probation_sidecar=sidecar,
    )
    assert proven_prior == probation_tranche.tranche_id

    receipt = TrancheContinuation(
        tranche_id=continuation_tranche.tranche_id,
        module_id="reasoning-d64-test",
        candidate_generation_id="g1",
        plan_id="a1" * 32,
        learning_policy_id="b2" * 32,
        parent_bundle_id="c3" * 32,
        parent_checkpoint_id="d4" * 32,
        parent_optimizer_receipt_id="e5" * 32,
        parent_soul_id="f6" * 32,
        parent_global_step=_continuation_parent_global_step(16, sidecar),
        prior_tranche_id=proven_prior,
    )
    store.write_continuation(receipt)
    assert receipt.parent_global_step == 24
