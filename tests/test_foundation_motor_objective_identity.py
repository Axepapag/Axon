from runtime.field import canonical_sha256
from training.foundation_motor_curriculum import (
    COPY_ALIGNMENT_MULTICELL_TEACH,
    COPY_ALIGNMENT_MULTICELL_TEACH_ID,
    FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM,
    FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID,
    FOUNDATION_MOTOR_V2_PROGRAM_ID,
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
