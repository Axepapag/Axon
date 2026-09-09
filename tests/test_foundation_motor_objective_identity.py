from runtime.field import canonical_sha256
from training.foundation_motor_curriculum import (
    COPY_ALIGNMENT_MULTICELL_TEACH,
    COPY_ALIGNMENT_MULTICELL_TEACH_ID,
    FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM,
    FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM,
    FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM,
    FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM_ID,
    RECEIPT_CONTINUATION_TEACH,
    RECEIPT_CONTINUATION_TEACH_ID,
    RECEIPT_ROUTE_EOS_BALANCED_TEACH,
    RECEIPT_ROUTE_EOS_BALANCED_TEACH_ID,
    RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
    apply_receipt_continuation_teach_weights,
    decide_foundation_motor_v2_stage,
    foundation_motor_v2_objective_program_id,
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
        "pair_exact_rates": {
            "position": 1.0,
            "copy_gate": 1.0,
            "eos_gate": 1.0,
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
        "pair_exact_rates": {
            "position": 1.0,
            "copy_gate": 1.0,
            "eos_gate": 1.0,
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
