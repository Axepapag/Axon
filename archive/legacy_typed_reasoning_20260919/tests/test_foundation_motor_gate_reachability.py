"""A stage may only gate on a metric the stage it is running can satisfy.

A stage gate can be impossible to satisfy for two independent reasons, and the
first guard written here (845bf8b) modelled only one of them:

``gradient`` -- the metric's components carry weight 0.0 in that stage.  A
weight of 0.0 means exactly zero gradient, so no optimizer step can move the
metric.  This is what the 2026-09-17 emission-rung tranche exposed: a gate
required ``typed_emission_exact_rate`` while decision, operation, region, start
and end were all weighted 0.0, so the "plateau" it reported was a mathematical
impossibility rather than a failure to learn.

``data`` -- the metric's denominator spans action families the stage excludes
from teaching through its ``eligible_actions`` contract.  This is what blocked
the 2026-09-18 v6 campaign: ``copy_alignment`` teaches copy/insert/replace, its
gate required ``payload_transport_exact_rate`` at 0.95, and that rate was
computed over all four DELTA families -- including the eight delete phases the
stage never teaches.  The arithmetic ceiling was 16/24 = 0.6667, so the gate
could never pass however well the lineage learned.  See
evt-20260918T140000Z-copilot-v6-checkpoint-preserved-and-stage0-gate-unreachable.

These tests pin one reproduced instance of each axis, the coupling that makes
the data axis real (the probe must actually narrow the transport rates to the
stage surface, or declaring them "stage_eligible" would be a lie), and the
refusal coupling that keeps the invariant meaningful.  They deliberately do not
claim the class is unrepresentable: an absolute literal compared against a
meaning-shifting denominator is a design choice a future author can repeat.
"""

from __future__ import annotations

from typing import Any

from training import foundation_motor_curriculum as module
from training.foundation_motor_curriculum import (
    FOUNDATION_MOTOR_V2_DELTA_ACTIONS,
    FOUNDATION_MOTOR_V2_METRIC_COMPONENTS,
    FOUNDATION_MOTOR_V2_METRIC_DENOMINATOR_ACTIONS,
    FOUNDATION_MOTOR_V2_PROGRAM,
    FOUNDATION_MOTOR_V2_PROBE_STAGE_SCOPED_METRICS,
    FOUNDATION_MOTOR_V2_STAGE,
    FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS,
    FOUNDATION_MOTOR_V2_STAGE_GATE_PLAN,
    FOUNDATION_MOTOR_V2_STAGE_ORDER,
    RECEIPT_GENERATE_HEAD_PROFILES,
    RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
    RECEIPT_TEACHING_PROFILES,
    foundation_motor_v2_component_weights,
    foundation_motor_v2_first_reachable_stage,
    foundation_motor_v2_probe,
    foundation_motor_v2_ratified_termination_profiles,
    foundation_motor_v2_stage_gate_metrics,
    foundation_motor_v2_termination_route_rejection,
    foundation_motor_v2_unreachable_gate_requirements,
)


class _Episode:
    """Duck-typed stand-in carrying only the mechanism tags the probe reads."""

    def __init__(self, action: str, pair: str) -> None:
        self.mechanism_tags = (
            f"foundation_stage:{FOUNDATION_MOTOR_V2_STAGE}",
            f"foundation_motor_action:{action}",
            f"foundation_pair:{pair}",
        )


def _surface(
    specs: tuple[tuple[str, str, bool, str], ...],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Build a synthetic motor-v2 evaluation surface.

    Each spec is ``(action, pair_id, exact, constant_target)``.  Every pair id
    is used exactly twice because the probe requires complete changed-source
    pairs.  ``constant_target`` is the payload a fixed emitter would have to
    produce for that case, which is what the constant-baseline floors are
    derived from -- the excluded delete cases share one target, which is why a
    constant emitter wins exactly the excluded family.
    """

    episodes: list[Any] = []
    rows: list[dict[str, Any]] = []
    for action, pair, exact, constant_target in specs:
        exact_flag = int(bool(exact))
        episodes.append(_Episode(action, pair))
        rows.append(
            {
                "phase_diagnostics": [
                    {
                        "target_decision": "emit" if exact else "no_op",
                        "decision_exact": bool(exact),
                        "target_operation": action,
                        "operation_exact": bool(exact),
                        "region_exact": bool(exact),
                        "start_exact": bool(exact),
                        "end_exact": bool(exact),
                        "payload_content_exact": bool(exact),
                        "alignment_position_exact": True,
                        "alignment_copy_gate_exact": True,
                        "alignment_eos_gate_exact": True,
                    }
                ],
                "typed_emission_exact_count": exact_flag,
                "supervised_phase_count": 1,
                "payload_transport_exact_count": exact_flag,
                "payload_supervised_phase_count": 1,
                "alignment_position_correct": 1,
                "alignment_position_count": 1,
                "alignment_copy_gate_correct": 1,
                "alignment_copy_gate_count": 1,
                "alignment_eos_gate_correct": 1,
                "alignment_eos_gate_count": 1,
                "payload_teacher_forced_content_correct": exact_flag,
                "payload_teacher_forced_content_count": 1,
                "payload_teacher_forced_eos_correct": 1,
                "payload_teacher_forced_eos_count": 1,
                "payload_teacher_forced_target_counts": [1, 1, 0],
                "decision_correct": exact_flag,
                "operation_correct": exact_flag,
                "operation_count": 1,
                "region_correct": exact_flag,
                "region_count": 1,
                "start_correct": exact_flag,
                "start_count": 1,
                "end_correct": exact_flag,
                "end_count": 1,
                "complete_field_coverage_count": 1,
                "phase_output_count": 1,
                "constant_typed_emission_target_histogram": {constant_target: 1},
                "constant_payload_transport_target_histogram": {constant_target: 1},
            }
        )
    return episodes, rows


def _v6_route_kwargs(profile: str) -> dict[str, Any]:
    return {
        "receipt_continuation": True,
        "receipt_teaching_profile": profile,
        "teach_multicell_copy": True,
        "eos_generate_head_route": profile in RECEIPT_GENERATE_HEAD_PROFILES,
    }


def test_no_stage_gates_on_a_metric_it_cannot_train() -> None:
    assert foundation_motor_v2_unreachable_gate_requirements() == []


def test_no_launchable_route_carries_an_unreachable_gate() -> None:
    """The coupling that keeps this invariant useful: only launchable routes matter.

    A route that is refused at launch never trains, so an unreachable gate on it
    cannot block a campaign.  Every profile that CAN launch must be reachable on
    both axes, and the set of launchable profiles must stay exactly the ratified
    one -- otherwise a new route could slip past this test.
    """

    launchable = [
        profile
        for profile in RECEIPT_TEACHING_PROFILES
        if foundation_motor_v2_termination_route_rejection(
            teach_multicell_copy=True,
            receipt_continuation=True,
            receipt_teaching_profile=profile,
        )
        is None
    ]
    assert launchable == [RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6]
    assert foundation_motor_v2_ratified_termination_profiles() == (
        RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
    )
    for profile in launchable:
        assert foundation_motor_v2_unreachable_gate_requirements(
            **_v6_route_kwargs(profile)
        ) == [], profile
    assert foundation_motor_v2_unreachable_gate_requirements(teach_multicell_copy=True) == []


def test_the_completed_declaration_exposes_the_v5_defect_it_used_to_hide() -> None:
    """The old declaration omitted overlay metrics that the gate body required.

    ``termination_head_v5`` runs copy_alignment with its eos-head overlay active
    while weighting ``alignment_eos_gate`` at 0.0, so its gate required a metric
    it could not move.  Omitting that metric from the declaration hid the
    problem.  It is harmless today only because v5 is refused at launch; this
    assertion is the reminder that the requirement must be cleared before that
    refusal could ever be lifted.
    """

    weights = foundation_motor_v2_component_weights(
        "copy_alignment",
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
    )
    assert weights["alignment_eos_gate"] == 0.0
    assert "alignment_eos_gate_accuracy" in foundation_motor_v2_stage_gate_metrics(
        "copy_alignment", receipt_continuation=True, eos_generate_head_route=False
    )
    violations = foundation_motor_v2_unreachable_gate_requirements(
        receipt_continuation=True,
        receipt_teaching_profile=RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
        teach_multicell_copy=True,
        eos_generate_head_route=False,
    )
    assert violations == [
        "copy_alignment gates on alignment_eos_gate_accuracy while weighting "
        "['alignment_eos_gate'] at 0.0"
    ]


def test_the_eos_head_overlay_is_per_stage() -> None:
    """copy_alignment applies its eos-head overlay only inside the continuation branch.

    This is the historic gate-body semantics (``if receipt_continuation: ... if
    not eos_generate_head_route: require(alignment_eos_gate_accuracy)``) and it
    is why a plain copy_alignment run weights ``alignment_eos_gate`` at 0.0.  The
    body and the reachability declaration must both read it from one place; a
    body that applied the overlay unconditionally would require a metric the
    stage cannot move.
    """

    active = module.foundation_motor_v2_stage_eos_head_active
    assert active("copy_alignment", receipt_continuation=False, eos_generate_head_route=False) is False
    assert active("copy_alignment", receipt_continuation=True, eos_generate_head_route=False) is True
    assert active("copy_alignment", receipt_continuation=True, eos_generate_head_route=True) is False
    assert active("transport_eos", receipt_continuation=False, eos_generate_head_route=False) is True
    assert active("transport_eos", receipt_continuation=True, eos_generate_head_route=False) is True
    assert active("transport_eos", receipt_continuation=True, eos_generate_head_route=True) is False
    for stage in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        for receipt_continuation in (False, True):
            for eos_generate_head_route in (False, True):
                declared = foundation_motor_v2_stage_gate_metrics(
                    stage,
                    receipt_continuation=receipt_continuation,
                    eos_generate_head_route=eos_generate_head_route,
                )
                plan = FOUNDATION_MOTOR_V2_STAGE_GATE_PLAN[stage]
                head_metrics = set(plan["eos_head_metrics"])
                if not head_metrics:
                    continue
                observed = head_metrics.issubset(set(declared))
                assert observed is active(
                    stage,
                    receipt_continuation=receipt_continuation,
                    eos_generate_head_route=eos_generate_head_route,
                ), (stage, receipt_continuation, eos_generate_head_route)


def test_an_unreachable_gate_is_never_reported_as_passed(monkeypatch) -> None:
    """The verdict is what would have distinguished "unsatisfiable" from "failed".

    On 2026-09-18 the v6 gate reported a plain threshold failure, which read as
    "the lineage is not good enough" when the requirement was in fact
    unsatisfiable.  Reconstructed here: a perfect copy_alignment surface with the
    transport rate re-widened to every DELTA family.  No threshold failure
    remains, so ``passed`` can only be False because the gate is unreachable.
    """

    episodes, rows = _surface(
        (
            ("insert", "p-i1", True, "A"),
            ("insert", "p-i1", True, "B"),
            ("insert", "p-i2", True, "C"),
            ("insert", "p-i2", True, "D"),
            ("insert", "p-i3", True, "E"),
            ("insert", "p-i3", True, "F"),
        )
    )
    for row in rows:
        row["payload_teacher_forced_target_counts"] = [0, 0, 0]
    rows[0]["payload_teacher_forced_target_counts"] = [1, 1, 0]
    probe = foundation_motor_v2_probe(episodes, rows, training_stage="copy_alignment")
    assert probe is not None
    assert probe["payload_transport_exact_rate"] == 1.0
    assert (
        probe["payload_content_accuracy"] > probe["payload_content_constant_floor"]
    )

    route = {
        "training_stage": "copy_alignment",
        "heldout_probe": probe,
        "regression_probe": probe,
        "complete_heldout": True,
        "complete_regression": True,
        "receipt_continuation": True,
        "receipt_teaching_profile": RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
        "termination_head_route": True,
    }
    healthy = module.decide_foundation_motor_v2_stage(**route)
    assert healthy["verdict"] == "passed"
    assert healthy["passed"] is True
    assert healthy["failures"] == []
    assert healthy["unreachable_requirements"] == []

    doctored = dict(FOUNDATION_MOTOR_V2_METRIC_DENOMINATOR_ACTIONS)
    doctored["payload_transport_exact_rate"] = FOUNDATION_MOTOR_V2_DELTA_ACTIONS
    monkeypatch.setattr(
        module,
        "FOUNDATION_MOTOR_V2_METRIC_DENOMINATOR_ACTIONS",
        doctored,
        raising=True,
    )
    blocked = module.decide_foundation_motor_v2_stage(**route)
    assert blocked["verdict"] == "unreachable"
    assert blocked["passed"] is False
    assert blocked["failures"] == []
    assert blocked["unreachable_requirements"] == [
        "copy_alignment gates on payload_transport_exact_rate over actions "
        "['delete'] it does not teach"
    ]


def test_a_threshold_miss_is_reported_as_below_threshold() -> None:
    episodes, rows = _surface(
        (
            ("insert", "p-i1", True, "A"),
            ("insert", "p-i1", True, "B"),
        )
    )
    rows[1]["phase_diagnostics"][0]["alignment_copy_gate_exact"] = False
    rows[1]["alignment_copy_gate_correct"] = 0
    probe = foundation_motor_v2_probe(episodes, rows, training_stage="copy_alignment")
    assert probe is not None
    decision = module.decide_foundation_motor_v2_stage(
        training_stage="copy_alignment",
        heldout_probe=probe,
        regression_probe=probe,
        complete_heldout=True,
        complete_regression=True,
    )
    assert decision["passed"] is False
    assert decision["verdict"] == "below_threshold"
    assert decision["unreachable_requirements"] == []
    assert any("alignment_copy_gate_accuracy" in item for item in decision["failures"])


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


def test_the_gradient_axis_catches_the_defect_it_was_written_for(monkeypatch) -> None:
    """Negative control: the guard must fire on the 2026-09-17 gradient defect.

    Re-gating copy_alignment on ``typed_emission_exact_rate`` is a double fault:
    the metric's components are weighted 0.0 there (gradient) and it is a DELTA
    conjunction while copy_alignment teaches only three families (data).  Both
    axes must fire, in plan order.
    """

    doctored = {
        stage: {**plan, "metrics": tuple(plan["metrics"])}
        for stage, plan in FOUNDATION_MOTOR_V2_STAGE_GATE_PLAN.items()
    }
    doctored["copy_alignment"]["metrics"] = (
        *doctored["copy_alignment"]["metrics"],
        "typed_emission_exact_rate",
    )
    monkeypatch.setattr(
        module, "FOUNDATION_MOTOR_V2_STAGE_GATE_PLAN", doctored, raising=True
    )
    findings = module.foundation_motor_v2_unreachable_gate_findings()
    assert [(f["stage"], f["metric"], f["axis"]) for f in findings] == [
        ("copy_alignment", "typed_emission_exact_rate", "gradient"),
        ("copy_alignment", "typed_emission_exact_rate", "data"),
    ]
    assert "decision" in findings[0]["detail"]
    assert "'delete'" in findings[1]["detail"]


def test_the_data_axis_catches_the_delete_defect(monkeypatch) -> None:
    """Negative control: the guard must fire on the 2026-09-18 data defect.

    This reconstructs the pre-fix configuration exactly: the transport rate
    computed over every DELTA family while copy_alignment and transport_eos
    teach only three of them.
    """

    doctored = dict(FOUNDATION_MOTOR_V2_METRIC_DENOMINATOR_ACTIONS)
    doctored["payload_transport_exact_rate"] = FOUNDATION_MOTOR_V2_DELTA_ACTIONS
    monkeypatch.setattr(
        module,
        "FOUNDATION_MOTOR_V2_METRIC_DENOMINATOR_ACTIONS",
        doctored,
        raising=True,
    )
    findings = module.foundation_motor_v2_unreachable_gate_findings(
        **_v6_route_kwargs(RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6)
    )
    assert [(f["stage"], f["metric"], f["axis"]) for f in findings] == [
        ("copy_alignment", "payload_transport_exact_rate", "data"),
        ("transport_eos", "payload_transport_exact_rate", "data"),
    ]
    assert "'delete'" in findings[0]["detail"]


def test_the_probe_narrows_the_transport_rates_to_the_stage_surface() -> None:
    """The coupling behind the ``stage_eligible`` declaration.

    This surface reproduces the campaign arithmetic in miniature.  Four insert
    cases are exact and two delete cases are not, so the whole-surface transport
    rate is 4/6 = 0.6667 -- the number that ceilinged v6 -- while the rate over
    the actions copy_alignment actually teaches is 4/4 = 1.0.  The constant
    "emit nothing and stop" answer wins exactly the two excluded delete cases,
    which is why the whole-surface floor is 2/6 = 0.3333 (the floor the v6 probe
    reported) and the scoped floor is strictly weaker for a fabricator.
    """

    episodes, rows = _surface(
        (
            ("insert", "p-i1", True, "A"),
            ("insert", "p-i1", True, "B"),
            ("insert", "p-i2", True, "C"),
            ("insert", "p-i2", True, "D"),
            ("delete", "p-d", False, ""),
            ("delete", "p-d", False, ""),
        )
    )
    unscoped = foundation_motor_v2_probe(episodes, rows)
    scoped = foundation_motor_v2_probe(episodes, rows, training_stage="copy_alignment")
    assert unscoped is not None and scoped is not None

    assert unscoped["payload_scope"]["basis"] == "whole_surface"
    assert unscoped["payload_scope"]["scope_metrics"] == []
    assert unscoped["payload_scope"]["training_stage"] is None
    assert unscoped["payload_transport_exact_rate"] == 4 / 6
    assert unscoped["constant_payload_transport_exact_floor"] == 2 / 6

    assert scoped["payload_scope"]["basis"] == "stage_eligible_actions"
    assert scoped["payload_scope"]["training_stage"] == "copy_alignment"
    assert scoped["payload_scope"]["eligible_actions"] == ["copy", "insert", "replace"]
    assert scoped["payload_scope"]["eligible_case_count"] == 4
    assert scoped["payload_scope"]["excluded_case_count"] == 2
    assert scoped["payload_scope"]["excluded_actions"] == ["delete"]
    assert scoped["payload_scope"]["scope_metrics"] == sorted(
        FOUNDATION_MOTOR_V2_PROBE_STAGE_SCOPED_METRICS
    )

    assert scoped["payload_transport_exact_rate"] == 1.0
    assert scoped["scoped_constant_payload_transport_exact_floor"] == 1 / 4
    assert (
        scoped["scoped_constant_payload_transport_exact_floor"]
        < unscoped["constant_payload_transport_exact_floor"]
    )

    # The narrowing hides nothing: the whole-surface number survives, and the
    # metrics the declaration does not list as scoped are untouched.
    assert scoped["whole_surface_payload_transport_exact_rate"] == 4 / 6
    assert scoped["whole_surface_payload_eos_accuracy"] == unscoped["payload_eos_accuracy"]
    assert scoped["typed_emission_exact_rate"] == unscoped["typed_emission_exact_rate"]
    assert scoped["decision_accuracy"] == unscoped["decision_accuracy"]
    assert scoped["pair_exact_rates"] == unscoped["pair_exact_rates"]

    # Per-family detail is what makes the failure shape legible.
    assert scoped["per_action_joint_exact_rate"] == {"delete": 0.0, "insert": 1.0}
    assert scoped["per_operation_accuracy"] == {"delete": 0.0, "insert": 1.0}
    assert scoped["pair_exact_rates"]["joint"] == 2 / 3


def test_the_probe_refuses_an_unknown_stage() -> None:
    episodes, rows = _surface((("insert", "p-i1", True, "A"),))
    try:
        foundation_motor_v2_probe(episodes, rows, training_stage="not_a_stage")
    except ValueError:
        return
    raise AssertionError("the probe accepted an unknown training stage")


def test_the_invariant_is_declared_for_every_gated_metric() -> None:
    # A metric with no declared components would silently bypass the weight
    # axis; one with no declared action denominator would silently bypass the
    # data axis.  Check the unconditional view, every overlay combination, and
    # the any-of requirement groups the gate body also enforces.
    for stage in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        plan = FOUNDATION_MOTOR_V2_STAGE_GATE_PLAN[stage]
        views = [plan["metrics"], plan["any_checks"]]
        for overlay_flag in (True, False):
            views.append(
                foundation_motor_v2_stage_gate_metrics(
                    stage,
                    receipt_continuation=overlay_flag,
                    eos_generate_head_route=not overlay_flag,
                )
            )
        for metric in dict.fromkeys(metric for view in views for metric in view):
            assert metric in FOUNDATION_MOTOR_V2_METRIC_COMPONENTS, (stage, metric)
            assert metric in FOUNDATION_MOTOR_V2_METRIC_DENOMINATOR_ACTIONS, (
                stage,
                metric,
            )
    # Every metric the invariant calls "stage_eligible" must be one the probe
    # actually narrows, or the declaration would be asserting a narrowing that
    # does not happen.
    for metric, demand in FOUNDATION_MOTOR_V2_METRIC_DENOMINATOR_ACTIONS.items():
        if demand == "stage_eligible":
            continue
        assert demand is FOUNDATION_MOTOR_V2_DELTA_ACTIONS, metric
        assert metric in ("typed_emission_exact_rate", "per_action_joint_exact_rate")
    assert set(FOUNDATION_MOTOR_V2_PROBE_STAGE_SCOPED_METRICS) == {
        "payload_transport_exact_rate",
        "payload_eos_accuracy",
    }


def test_only_joint_gates_on_a_delta_conjunction() -> None:
    """The two DELTA-demanding metrics may only be required where every family is taught.

    ``typed_emission_exact_rate`` and ``per_action_joint_exact_rate`` are
    conjunctions over the whole DELTA action set, so they demand the complete
    family list no matter which stage is running.  Only ``joint`` teaches every
    family.  If a future edit requires either one earlier, the invariant fires --
    and this test says why.
    """

    delta_demanding = {
        metric
        for metric, demand in FOUNDATION_MOTOR_V2_METRIC_DENOMINATOR_ACTIONS.items()
        if demand == FOUNDATION_MOTOR_V2_DELTA_ACTIONS
    }
    assert delta_demanding == {
        "typed_emission_exact_rate",
        "per_action_joint_exact_rate",
    }
    for stage in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        plan = FOUNDATION_MOTOR_V2_STAGE_GATE_PLAN[stage]
        required = {
            *plan["metrics"],
            *plan["any_checks"],
            *(metric for metric, _floor, _message in plan["floor_beats"]),
            *plan["continuation_metrics"],
            *plan["eos_head_metrics"],
        }
        offenders = required & delta_demanding
        if stage == "joint":
            assert offenders == delta_demanding
            continue
        assert offenders == set(), (stage, sorted(offenders))


def test_the_two_transport_rates_are_the_only_scoped_metrics() -> None:
    for metric in FOUNDATION_MOTOR_V2_PROBE_STAGE_SCOPED_METRICS:
        assert FOUNDATION_MOTOR_V2_METRIC_DENOMINATOR_ACTIONS[metric] == "stage_eligible"
    assert foundation_motor_v2_stage_gate_metrics("copy_alignment") == (
        "alignment_position_accuracy",
        "alignment_copy_gate_accuracy",
        "payload_content_accuracy",
        "payload_transport_exact_rate",
    )


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
