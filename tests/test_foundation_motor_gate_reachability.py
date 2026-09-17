"""A stage may only gate on a metric its own objective can actually move.

The 2026-09-17 emission-rung tranche exposed the general defect: a stage gate
required ``typed_emission_exact_rate`` while every component that metric depends
on (decision, operation, region, start, end) was weighted 0.0 in that stage.
A weight of 0.0 means exactly zero gradient, so the requirement could never be
satisfied at any step: the "plateau" it reported was a mathematical
impossibility, not a failure to learn.  The same class of defect explains the
termhead-v1 probation exhausting on a transport-exactness plateau.

These tests make that class unrepresentable: any future edit that gates a stage
on a metric it cannot train fails here.
"""

from __future__ import annotations

from training import foundation_motor_curriculum as module
from training.foundation_motor_curriculum import (
    FOUNDATION_MOTOR_V2_METRIC_COMPONENTS,
    FOUNDATION_MOTOR_V2_PROGRAM,
    FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS,
    FOUNDATION_MOTOR_V2_STAGE_ORDER,
    RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    RECEIPT_TEACHING_PROFILES,
    foundation_motor_v2_component_weights,
    foundation_motor_v2_first_reachable_stage,
    foundation_motor_v2_unreachable_gate_requirements,
)


def test_no_stage_gates_on_a_metric_it_cannot_train() -> None:
    assert foundation_motor_v2_unreachable_gate_requirements() == []


def test_no_teaching_profile_reintroduces_an_unreachable_gate() -> None:
    for profile in RECEIPT_TEACHING_PROFILES:
        assert (
            foundation_motor_v2_unreachable_gate_requirements(
                receipt_continuation=True, receipt_teaching_profile=profile
            )
            == []
        ), profile
    assert foundation_motor_v2_unreachable_gate_requirements(teach_multicell_copy=True) == []


def test_typed_exactness_is_unreachable_before_the_address_stage() -> None:
    # The exact load-bearing fact: typed exactness is a DELTA conjunction, so
    # it cannot be a requirement at copy_alignment or transport_eos even though
    # the emission rung made those stages responsible for emitting at all.
    assert foundation_motor_v2_first_reachable_stage("typed_emission_exact_rate") == "address"
    for stage in ("copy_alignment", "transport_eos", "decision", "operation"):
        weights = foundation_motor_v2_component_weights(stage)
        assert weights["decision"] == 0.0 or weights["region"] == 0.0
    assert (
        "typed_emission_exact_rate"
        not in FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS["copy_alignment"]
    )
    assert (
        "typed_emission_exact_rate"
        not in FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS["transport_eos"]
    )


def test_the_invariant_catches_the_defect_it_was_written_for(monkeypatch) -> None:
    """Negative control: the guard must actually fire on the 2026-09-17 defect."""

    doctored = dict(FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS)
    doctored["copy_alignment"] = (
        *FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS["copy_alignment"],
        "typed_emission_exact_rate",
    )
    monkeypatch.setattr(
        module, "FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS", doctored, raising=True
    )
    violations = module.foundation_motor_v2_unreachable_gate_requirements()
    assert len(violations) == 1
    assert "copy_alignment gates on typed_emission_exact_rate" in violations[0]
    assert "decision" in violations[0]


def test_the_invariant_is_declared_for_every_gated_metric() -> None:
    # A metric with no declared components would silently bypass the invariant.
    for stage in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        for metric in FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS.get(stage, ()):
            assert metric in FOUNDATION_MOTOR_V2_METRIC_COMPONENTS, (stage, metric)


def test_declared_gate_metrics_match_the_stage_weight_tables() -> None:
    program_stages = {str(stage["name"]) for stage in FOUNDATION_MOTOR_V2_PROGRAM["stages"]}
    assert set(FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS) == program_stages
    assert set(FOUNDATION_MOTOR_V2_STAGE_ORDER) == program_stages


def test_component_weights_are_zero_by_default() -> None:
    # _weights() defaults every unlisted component to 0.0; if that default ever
    # became nonzero this contract would need to be re-derived rather than
    # trusted.
    joint = foundation_motor_v2_component_weights("joint")
    assert all(value > 0.0 for value in joint.values())
    copy_alignment = foundation_motor_v2_component_weights("copy_alignment")
    assert copy_alignment["decision"] == 0.0
    assert copy_alignment["alignment_eos_gate"] == 0.0
    # The receipt overlay is the only thing that gives copy_alignment a stop
    # signal at all.
    overlaid = foundation_motor_v2_component_weights(
        "copy_alignment",
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    )
    assert overlaid["alignment_eos_gate"] > 0.0
