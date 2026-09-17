from scripts.train_living_reasoning_smoke import (
    exact_serving_gate_passed,
    nonzero_exact_output_observed,
)


def _evaluation(
    *,
    typed: float,
    payload: float,
    typed_floor: float = 1.0 / 3.0,
    payload_floor: float = 1.0 / 3.0,
) -> dict[str, float]:
    # Defaults are the floors actually measured over the 72-case heldout
    # surface (24/72 typed, 8/24 transport).  A zero floor here would certify
    # the emit-nothing answer as progress, which is the trap this test guards.
    return {
        "typed_emission_exact_rate": typed,
        "payload_transport_exact_rate": payload,
        "constant_typed_emission_exact_floor": typed_floor,
        "constant_payload_transport_exact_floor": payload_floor,
    }


def test_matching_the_constant_answer_is_not_progress() -> None:
    evaluation = _evaluation(typed=1.0 / 3.0, payload=1.0 / 3.0)
    assert nonzero_exact_output_observed(evaluation) is False


def test_nonzero_exact_output_is_progress_not_serving_readiness() -> None:
    evaluation = _evaluation(typed=2.0 / 3.0, payload=2.0 / 3.0)
    assert nonzero_exact_output_observed(evaluation) is True
    assert exact_serving_gate_passed(
        evaluation,
        curriculum_stage_complete=True,
        complete_heldout=True,
        complete_regression=True,
        tournament_metric_surface_complete=True,
    ) is False


def test_exact_serving_gate_fails_closed_on_every_incomplete_surface() -> None:
    evaluation = _evaluation(typed=1.0, payload=1.0)
    assert exact_serving_gate_passed(
        evaluation,
        curriculum_stage_complete=True,
        complete_heldout=True,
        complete_regression=True,
        tournament_metric_surface_complete=True,
    ) is True
    assert exact_serving_gate_passed(
        evaluation,
        curriculum_stage_complete=True,
        complete_heldout=True,
        complete_regression=True,
        tournament_metric_surface_complete=False,
    ) is False
    assert exact_serving_gate_passed(
        evaluation,
        curriculum_stage_complete=False,
        complete_heldout=True,
        complete_regression=True,
        tournament_metric_surface_complete=True,
    ) is False
