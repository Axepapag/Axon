"""Foundations Stage 0: exact typed-delta motor control.

The core learns the smallest real movements of Axon's body before sequence or
language work: copy one current-field scalar, insert, replace, delete, no-op,
and abstain.  Every case traverses the real Shared Field, D64 reader, private
Soul, typed reasoning output, exact address, and EOS contracts.

The `typed_motor_v2` ladder teaches one competency per stage and teaches
emission before termination: `copy_alignment` supervises payload content with
`alignment_eos_gate` at exactly 0.0, so content is learned where nothing
competes with emitting, and `transport_eos` only adds stop supervision on top of
an already-emitting core.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

from runtime.field import D64FieldCompiler, LogicalRegion, SharedFieldSnapshot, canonical_sha256
from runtime.heart import ProposalPass, ReasoningDecision, ReasoningOperationKind
from substrate import encode_unicode_text

from .first_form_curriculum import FirstFormCase, FirstFormCurriculum, TeachingEligibility, _workspace
from .living_reasoning_curriculum import (
    LivingReasoningEpisode,
    LivingReasoningTarget,
    constant_baseline_floors,
)

FOUNDATION_MOTOR_STAGE = "typed_motor_v1"
FOUNDATION_MOTOR_SOURCE_ID = canonical_sha256(
    {"schema": "axon-foundation-motor-authored-generator-v1"}
)
FOUNDATION_MOTOR_ACTIONS = ("copy", "insert", "replace", "delete", "no_op", "abstain")
FOUNDATION_MOTOR_GATE_POLICY = {
    "schema": "axon-foundation-stage-gate-policy-v1",
    "stage": FOUNDATION_MOTOR_STAGE,
    "scope": "curriculum_advancement_only_not_serving_or_promotion",
    "free_running_case_exact_rate": 0.95,
    "changed_source_pair_exact_rate": 0.95,
    "payload_changed_source_pair_exact_rate": 0.95,
    "per_action_exact_rate": 0.95,
    "complete_field_coverage_rate": 1.0,
    "teacher_forced_must_exceed_constant_floor": True,
    "capacity_law": (
        "competency gate only; failure pauses for diagnosis or a renewable tranche "
        "and never limits tissue, field, output, curriculum, or lifetime steps"
    ),
}
FOUNDATION_MOTOR_GATE_POLICY_ID = canonical_sha256(FOUNDATION_MOTOR_GATE_POLICY)
DEFAULT_FOUNDATION_MOTOR_SPLIT_COUNTS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("F0", (72, 24, 24)),
)

FOUNDATION_MOTOR_V2_STAGE = "typed_motor_v2"
FOUNDATION_MOTOR_V2_SOURCE_ID = canonical_sha256(
    {"schema": "axon-foundation-motor-authored-generator-v2"}
)
FOUNDATION_MOTOR_V2_UNICODE_WALK_SOURCE_ID = canonical_sha256(
    {"schema": "axon-foundation-motor-unicode-walk-generator-v1"}
)
FOUNDATION_MOTOR_V2_STAGE_ORDER = (
    "copy_alignment",
    "transport_eos",
    "decision",
    "operation",
    "address",
    "joint",
)


def _weights(**overrides: float) -> dict[str, float]:
    names = (
        "decision",
        "operation",
        "region",
        "start",
        "end",
        "payload",
        "alignment_position",
        "alignment_copy_gate",
        "alignment_eos_gate",
    )
    return {name: float(overrides.get(name, 0.0)) for name in names}


FOUNDATION_MOTOR_V2_PROGRAM = {
    "schema": "axon-foundation-motor-teaching-program-v1",
    "foundation_stage": FOUNDATION_MOTOR_V2_STAGE,
    "scope": "curriculum_advancement_only_not_serving_or_promotion",
    "stage_selection": "advance_only_after_complete_heldout_and_regression_gate",
    "resource_law": "optimizer steps are renewable work, never a stage or tissue ceiling",
    "stage_order": list(FOUNDATION_MOTOR_V2_STAGE_ORDER),
    "stages": [
        {
            "name": "copy_alignment",
            "eligible_actions": ["copy", "insert", "replace"],
            "component_weights": _weights(
                # Ratified 2026-09-17 from the emission autopsy
                # (evt-20260917T100000Z-copilot-stuck-diagnosis-stage-cliff-and-emission-rung):
                # this stage was silent about payload content while the retention
                # contract already evaluated it, so copy_alignment certified cores
                # that emit nothing and the payload weight then jumped 0.0 -> 1.0
                # in one step at transport_eos.  Emission is taught here, where
                # alignment_eos_gate is still exactly 0.0 and nothing competes
                # with emitting; transport_eos keeps this table and only adds stop
                # supervision.  No component jumps 0.0 -> 1.0 across a stage
                # boundary any more.
                payload=1.0,
                alignment_position=1.0,
                alignment_copy_gate=4.0,
            ),
        },
        {
            "name": "transport_eos",
            "eligible_actions": ["copy", "insert", "replace"],
            "component_weights": _weights(
                payload=1.0,
                alignment_position=1.0,
                alignment_copy_gate=4.0,
                alignment_eos_gate=1.0,
            ),
        },
        {
            "name": "decision",
            "eligible_actions": list(FOUNDATION_MOTOR_ACTIONS),
            "component_weights": _weights(
                decision=1.0,
                payload=0.25,
                alignment_position=0.25,
                alignment_copy_gate=1.0,
                alignment_eos_gate=0.25,
            ),
        },
        {
            "name": "operation",
            "eligible_actions": ["copy", "insert", "replace", "delete"],
            "component_weights": _weights(
                decision=0.25,
                operation=1.0,
                payload=0.25,
                alignment_position=0.25,
                alignment_copy_gate=1.0,
                alignment_eos_gate=0.25,
            ),
        },
        {
            "name": "address",
            "eligible_actions": ["copy", "insert", "replace", "delete"],
            "component_weights": _weights(
                decision=0.25,
                operation=0.25,
                region=1.0,
                start=1.0,
                end=1.0,
                payload=0.25,
                alignment_position=0.25,
                alignment_copy_gate=1.0,
                alignment_eos_gate=0.25,
            ),
        },
        {
            "name": "joint",
            "eligible_actions": list(FOUNDATION_MOTOR_ACTIONS),
            "component_weights": _weights(
                decision=1.0,
                operation=1.0,
                region=1.0,
                start=1.0,
                end=1.0,
                payload=1.0,
                alignment_position=1.0,
                alignment_copy_gate=4.0,
                alignment_eos_gate=1.0,
            ),
        },
    ],
    "gate_threshold": 0.95,
    "complete_field_coverage_rate": 1.0,
}
FOUNDATION_MOTOR_V2_PROGRAM_ID = canonical_sha256(FOUNDATION_MOTOR_V2_PROGRAM)

# Same exam, same gates, same architecture; this overlay changes optimizer
# pressure and therefore MUST participate in the effective objective identity.
COPY_ALIGNMENT_MULTICELL_TEACH = {
    "schema": "axon-foundation-motor-copy-alignment-multicell-teach-v1",
    "alignment_position_reduction": "sum",
    "component_weight_overrides": {
        "alignment_position": 4.0,
        "alignment_copy_gate": 0.25,
    },
    "oversample_multicell": True,
}
COPY_ALIGNMENT_MULTICELL_TEACH_ID = canonical_sha256(COPY_ALIGNMENT_MULTICELL_TEACH)
FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM = {
    "schema": "axon-foundation-motor-teaching-program-variant-v1",
    "base_program_id": FOUNDATION_MOTOR_V2_PROGRAM_ID,
    "teaching_overlay_ids": [COPY_ALIGNMENT_MULTICELL_TEACH_ID],
}
FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID = canonical_sha256(
    FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM
)
RECEIPT_CONTINUATION_TEACH = {
    "schema": "axon-foundation-motor-receipt-continuation-teach-v1",
    "requires_architecture_feature": "receipt_continuation",
    "deterministic_continuation_losses_masked": [
        "payload",
        "alignment_position",
        "alignment_copy_gate",
    ],
    "learned_decisions_retained": [
        "payload_anchor_category",
        "copy_generate_route",
        "source_anchor",
        "eos",
    ],
    "component_weight_overrides": {
        "payload": 1.0,
        "alignment_position": 4.0,
        "alignment_copy_gate": 0.25,
        "alignment_eos_gate": 1.0,
    },
    "alignment_position_reduction": "sum",
    "oversample_multicell": True,
    "same_stage_eos_retention_required": True,
}
RECEIPT_CONTINUATION_TEACH_ID = canonical_sha256(RECEIPT_CONTINUATION_TEACH)
FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM = {
    "schema": "axon-foundation-motor-teaching-program-variant-v2",
    "base_program_id": FOUNDATION_MOTOR_V2_PROGRAM_ID,
    "teaching_overlay_ids": [RECEIPT_CONTINUATION_TEACH_ID],
    "layer_13_resolution": "deterministic_receipt_continuation_is_categorical_transport",
}
FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM_ID = canonical_sha256(
    FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM
)

# The first receipt objective proved exact position and EOS retention but drove
# the learned copy/generate route to zero.  Preserve that objective and its
# evidence immutably; this separately identified profile restores strong route
# supervision while keeping deterministic continuation outside learned loss.
RECEIPT_ROUTE_EOS_BALANCED_TEACH = {
    "schema": "axon-foundation-motor-receipt-continuation-teach-v2",
    "profile": "route_eos_balanced_v2",
    "requires_architecture_feature": "receipt_continuation",
    "deterministic_continuation_losses_masked": [
        "payload",
        "alignment_position",
        "alignment_copy_gate",
    ],
    "learned_decisions_retained": [
        "payload_anchor_category",
        "copy_generate_route",
        "source_anchor",
        "eos",
    ],
    "component_weight_overrides": {
        "payload": 1.0,
        "alignment_position": 1.0,
        "alignment_copy_gate": 4.0,
        "alignment_eos_gate": 2.0,
    },
    "alignment_position_reduction": "mean",
    "oversample_multicell": True,
    "same_stage_eos_retention_required": True,
    "experimental_target": (
        "retain exact source position and EOS while recovering the learned "
        "copy/generate route"
    ),
}
RECEIPT_ROUTE_EOS_BALANCED_TEACH_ID = canonical_sha256(
    RECEIPT_ROUTE_EOS_BALANCED_TEACH
)
FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM = {
    "schema": "axon-foundation-motor-teaching-program-variant-v3",
    "base_program_id": FOUNDATION_MOTOR_V2_PROGRAM_ID,
    "teaching_overlay_ids": [RECEIPT_ROUTE_EOS_BALANCED_TEACH_ID],
    "layer_13_resolution": "deterministic_receipt_continuation_is_categorical_transport",
}
FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM_ID = canonical_sha256(
    FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM
)

# Receipt continuation with an explicit generated-head termination route uses
# a hierarchical decoder distribution: generated-head EOS probability decides
# termination and the copy/generate gate arbitrates non-EOS content. The old
# EOS gate loss is therefore both redundant and contradictory. Preserve all
# prior profiles and identify this objective separately.
RECEIPT_GENERATE_HEAD_EOS_TEACH = {
    "schema": "axon-foundation-motor-receipt-continuation-teach-v3",
    "profile": "generate_head_eos_v3",
    "requires_architecture_features": [
        "receipt_continuation",
        "eos_generate_head_route",
    ],
    "deterministic_continuation_losses_masked": [
        "payload",
        "alignment_position",
        "alignment_copy_gate",
    ],
    "learned_decisions_retained": [
        "payload_anchor_category",
        "copy_generate_route",
        "source_anchor",
        "generate_head_eos",
    ],
    "component_weight_overrides": {
        "payload": 1.0,
        "alignment_position": 1.0,
        "alignment_copy_gate": 4.0,
        "alignment_eos_gate": 0.0,
    },
    "alignment_position_reduction": "mean",
    "oversample_multicell": True,
    "same_stage_eos_retention_required": True,
    "termination_objective": "hierarchical_generated_head_eos_probability",
}
RECEIPT_GENERATE_HEAD_EOS_TEACH_ID = canonical_sha256(
    RECEIPT_GENERATE_HEAD_EOS_TEACH
)
FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM = {
    "schema": "axon-foundation-motor-teaching-program-variant-v4",
    "base_program_id": FOUNDATION_MOTOR_V2_PROGRAM_ID,
    "teaching_overlay_ids": [RECEIPT_GENERATE_HEAD_EOS_TEACH_ID],
    "layer_13_resolution": "deterministic_receipt_continuation_is_categorical_transport",
    "termination_resolution": "generate_head_eos_is_independent_of_content_route",
}
FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID = canonical_sha256(
    FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM
)

# The v3 route fixed the inference/training distribution mismatch, but retained
# the legacy sequence loss's 4x EOS token weight.  Complete-surface and guarded
# runs showed that this still rewards the immediate-EOS basin.  Preserve v3 as
# evidence and identify the balanced payload objective separately.
RECEIPT_GENERATE_HEAD_BALANCED_TEACH = {
    **RECEIPT_GENERATE_HEAD_EOS_TEACH,
    "schema": "axon-foundation-motor-receipt-continuation-teach-v4",
    "profile": "generate_head_balanced_v4",
    "payload_eos_weight": 1.0,
    "termination_objective": "balanced_hierarchical_generated_head_eos_probability",
}
RECEIPT_GENERATE_HEAD_BALANCED_TEACH_ID = canonical_sha256(
    RECEIPT_GENERATE_HEAD_BALANCED_TEACH
)
FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM = {
    "schema": "axon-foundation-motor-teaching-program-variant-v5",
    "base_program_id": FOUNDATION_MOTOR_V2_PROGRAM_ID,
    "teaching_overlay_ids": [RECEIPT_GENERATE_HEAD_BALANCED_TEACH_ID],
    "layer_13_resolution": "deterministic_receipt_continuation_is_categorical_transport",
    "termination_resolution": "generate_head_eos_is_independent_and_not_overweighted",
}
FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM_ID = canonical_sha256(
    FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM
)

# The content<->EOS route swap survived every optimizer-side control (lr 3e-4
# and 1e-4, 4x and 1x EOS weight, both gate-bias inits): content logit growth
# mechanically depresses softmax EOS probability because both live in the one
# generated softmax.  This profile is the controlled architecture variable: a
# dedicated scalar termination head owns the stop decision
# (sigmoid(termination_output(fused))) while the hierarchical content
# distribution is otherwise unchanged.  All optimizer pressures stay exactly
# on the v3 recipe; only the termination equation is replaced.  Preserve all
# prior profiles and identify this objective separately.
RECEIPT_TERMINATION_HEAD_TEACH = {
    **RECEIPT_GENERATE_HEAD_EOS_TEACH,
    "schema": "axon-foundation-motor-receipt-continuation-teach-v5",
    "profile": "termination_head_v5",
    "requires_architecture_features": [
        "receipt_continuation",
        "termination_head_route",
    ],
    "learned_decisions_retained": [
        "payload_anchor_category",
        "copy_generate_route",
        "source_anchor",
        "termination_head",
    ],
    "payload_eos_weight": 4.0,
    "termination_objective": "dedicated_termination_head_sigmoid",
}
RECEIPT_TERMINATION_HEAD_TEACH_ID = canonical_sha256(
    RECEIPT_TERMINATION_HEAD_TEACH
)
FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_PROGRAM = {
    "schema": "axon-foundation-motor-teaching-program-variant-v6",
    "base_program_id": FOUNDATION_MOTOR_V2_PROGRAM_ID,
    "teaching_overlay_ids": [RECEIPT_TERMINATION_HEAD_TEACH_ID],
    "layer_13_resolution": "deterministic_receipt_continuation_is_categorical_transport",
    "termination_resolution": "dedicated_termination_head_is_independent_of_content_softmax",
}
FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_PROGRAM_ID = canonical_sha256(
    FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_PROGRAM
)

# The transport_eos probation autopsy (canonical events
# evt-20260917T013100Z-copilot-content-signal-audit and
# evt-20260917T013451Z-kimi-training-wrongness-audit) verified from the
# step-48 report and the loss source that v5 supervises stop=1 twice
# (payload CE at the restored 4x EOS weight plus the EOS-position gate BCE)
# while stop=0 receives only the diluted implicit log(1-stop) term inside
# payload CE, leaving a canceling-gradient fixed point where EOS wins every
# argmax and free-running transport emits nothing.  v6 is the single
# controlled correction, ratified by Jeff on 2026-09-17: restore the
# v4-balanced 1x payload EOS weight and add explicit stop=0 BCE supervision
# at every learned content anchor so the dedicated stop head is supervised
# symmetrically at both decision classes.  Component weights are pinned
# explicitly here because v5 inherited the v3 table silently; the executed
# per-stage tables are now pinned by tests.
RECEIPT_TERMINATION_HEAD_BALANCED_TEACH = {
    **RECEIPT_TERMINATION_HEAD_TEACH,
    "schema": "axon-foundation-motor-receipt-continuation-teach-v6",
    "profile": "termination_head_balanced_v6",
    "component_weight_overrides": {
        "payload": 1.0,
        "alignment_position": 1.0,
        "alignment_copy_gate": 4.0,
        "alignment_eos_gate": 1.0,
    },
    "payload_eos_weight": 1.0,
    "termination_continue_supervision": True,
    "termination_objective": (
        "dedicated_termination_head_sigmoid_with_explicit_continue_supervision"
    ),
}
RECEIPT_TERMINATION_HEAD_BALANCED_TEACH_ID = canonical_sha256(
    RECEIPT_TERMINATION_HEAD_BALANCED_TEACH
)
FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_BALANCED_PROGRAM = {
    "schema": "axon-foundation-motor-teaching-program-variant-v7",
    "base_program_id": FOUNDATION_MOTOR_V2_PROGRAM_ID,
    "teaching_overlay_ids": [RECEIPT_TERMINATION_HEAD_BALANCED_TEACH_ID],
    "layer_13_resolution": "deterministic_receipt_continuation_is_categorical_transport",
    "termination_resolution": (
        "dedicated_termination_head_is_supervised_symmetrically_at_anchors_and_eos"
    ),
}
FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_BALANCED_PROGRAM_ID = canonical_sha256(
    FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_BALANCED_PROGRAM
)

RECEIPT_TEACHING_PROFILE_CONTINUATION_V1 = "continuation_v1"
RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2 = "route_eos_balanced_v2"
RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3 = "generate_head_eos_v3"
RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4 = "generate_head_balanced_v4"
RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5 = "termination_head_v5"
RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6 = "termination_head_balanced_v6"
RECEIPT_TERMINATION_HEAD_PROFILES = (
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
)
RECEIPT_GENERATE_HEAD_PROFILES = (
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4,
)
RECEIPT_TEACHING_PROFILES = (
    RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
    RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4,
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5,
    RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6,
)


def receipt_teaching_profile_route_flag(profile: str) -> str | None:
    """The route flag a launcher must pass alongside this receipt profile.

    The trainer requires the route flag and the profile to be selected together
    (scripts/train_living_reasoning_smoke.py flag validation), so every launcher
    that emits --receipt-teaching-profile must also emit the matching flag.
    Deriving it here keeps launchers from drifting away from the trainer's own
    contract.
    """

    if profile in RECEIPT_TERMINATION_HEAD_PROFILES:
        return "--termination-head-route"
    if profile in RECEIPT_GENERATE_HEAD_PROFILES:
        return "--eos-generate-head-route"
    return None


# ---------------------------------------------------------------------------
# Rejected termination routes (fail-closed).
#
# Nine motor-v2 tranches spent their allowance on a termination objective that
# the transport_eos probation autopsy had already rejected from the loss source
# (canonical events evt-20260917T013100Z-copilot-content-signal-audit and
# evt-20260917T013451Z-kimi-training-wrongness-audit): on those routes stop=1 is
# pushed up twice -- once by the payload CE at the EOS position and once by the
# EOS-position gate -- while stop=0 is never supervised directly.  The two
# pressures cancel at sigma* = (n+8)/(2n+8) ~ 0.8-0.9, so EOS wins every argmax
# and free-running transport emits nothing.  The 600-step emission rung (job
# 389df54d, revision 1a4bc416) re-tested that route a ninth time and reported
# exactly the plateau it was predicted to report.
#
# The property that separates a repaired objective from a rejected one is
# explicit symmetric continue supervision, which the objective declares itself
# as `termination_continue_supervision`.  Membership of the rejected set is
# therefore DERIVED from each objective's own declaration rather than kept in a
# hand-maintained list of profile names, so an objective that does not declare
# the symmetry is refused by default until it is ratified.  Defining a repair is
# not shipping it; this is the mechanism that makes the rejected route
# unlaunchable instead of merely documented.
FOUNDATION_MOTOR_V2_CONTINUE_SUPERVISION_KEY = "termination_continue_supervision"


def foundation_motor_v2_termination_objective_declares_continue_supervision(
    profile: str,
) -> bool:
    """Whether this receipt profile supervises the continue class explicitly."""

    return (
        receipt_continuation_teach_profile(profile).get(
            FOUNDATION_MOTOR_V2_CONTINUE_SUPERVISION_KEY
        )
        is True
    )


def foundation_motor_v2_ratified_termination_profiles() -> tuple[str, ...]:
    """The receipt profiles that carry explicit symmetric stop supervision."""

    return tuple(
        profile
        for profile in RECEIPT_TEACHING_PROFILES
        if foundation_motor_v2_termination_objective_declares_continue_supervision(
            profile
        )
    )


def foundation_motor_v2_termination_route_rejection(
    *,
    teach_multicell_copy: bool,
    receipt_continuation: bool = False,
    receipt_teaching_profile: str = RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
) -> str | None:
    """Why this motor-v2 run must not execute, or None when the route is ratified.

    Returning a reason is a refusal to train, not a warning: the rejected routes
    are known-broken objectives whose plateaus are already measured, so a
    tranche that runs one cannot produce information and only spends allowance.
    """

    program_id = foundation_motor_v2_objective_program_id(
        teach_multicell_copy=teach_multicell_copy,
        receipt_continuation=receipt_continuation,
        receipt_teaching_profile=receipt_teaching_profile,
    )
    if not receipt_continuation:
        return (
            f"refusing to train the pre-receipt-continuation motor-v2 termination "
            f"route (objective program {program_id}): the stop decision shares one "
            f"softmax with content, so content-logit growth mechanically depresses "
            f"EOS probability, and stop=1 is supervised twice while stop=0 is never "
            f"supervised directly, leaving a canceling-gradient fixed point where "
            f"EOS wins every argmax and free-running transport emits nothing "
            f"(the emission rung reported payload_content_accuracy 0.8710 with "
            f"payload_transport_exact_rate 0.1667). Rejected by the transport_eos "
            f"probation autopsy evt-20260917T013451Z-kimi-training-wrongness-audit. "
            f"Train the ratified route instead: --receipt-continuation "
            f"--receipt-teaching-profile "
            f"{RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6} "
            f"--termination-head-route"
        )
    if not foundation_motor_v2_termination_objective_declares_continue_supervision(
        receipt_teaching_profile
    ):
        return (
            f"refusing to train receipt objective profile "
            f"{receipt_teaching_profile!r} (objective program {program_id}): it does "
            f"not declare {FOUNDATION_MOTOR_V2_CONTINUE_SUPERVISION_KEY}, so the "
            f"continue class receives no direct supervision and the objective keeps "
            f"the asymmetric stop=1/stop=0 fixed point the autopsy rejected "
            f"(evt-20260917T013451Z-kimi-training-wrongness-audit). Ratified "
            f"termination profiles with explicit continue supervision: "
            f"{', '.join(foundation_motor_v2_ratified_termination_profiles()) or 'none'}"
        )
    return None


# ---------------------------------------------------------------------------
# Gate reachability contract.
#
# A stage gate may only require a metric whose causal components all carry
# nonzero weight in that stage.  A component weighted 0.0 receives exactly zero
# gradient, so a metric that depends on it cannot move during the stage and the
# requirement is unreachable by construction: the stage becomes a gate no
# lineage can ever pass, and the plateaus it reports are mathematical
# impossibilities read as learning failures.
#
# This is not hypothetical.  transport_eos required
# payload_transport_exact_rate while the typed conjunction's inputs were
# weighted 0.0 there, and the termhead-v1 probation "exhausted 3/3" on that
# plateau.  Typed exactness is a DELTA-phase conjunction over decision,
# operation, region, start, end and exact free-running payload transport
# (living_reasoning_curriculum.py:897), so it is unreachable before the address
# stage in every teaching profile.
# ---------------------------------------------------------------------------
FOUNDATION_MOTOR_V2_METRIC_COMPONENTS: dict[str, tuple[str, ...]] = {
    "alignment_position_accuracy": ("alignment_position",),
    "alignment_copy_gate_accuracy": ("alignment_copy_gate",),
    "alignment_eos_gate_accuracy": ("alignment_eos_gate",),
    "region_accuracy": ("region",),
    "start_accuracy": ("start",),
    "end_accuracy": ("end",),
    "payload_content_accuracy": ("payload",),
    "payload_eos_accuracy": ("payload",),
    "payload_transport_exact_rate": ("payload",),
    "typed_emission_exact_rate": (
        "decision",
        "operation",
        "region",
        "start",
        "end",
        "payload",
    ),
}

# The metric requirements each stage's gate actually enforces.  Kept beside the
# component map so the invariant in
# foundation_motor_v2_unreachable_gate_requirements() can be checked in a test
# instead of being re-derived by reading the gate body.
FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS: dict[str, tuple[str, ...]] = {
    "copy_alignment": (
        "alignment_position_accuracy",
        "alignment_copy_gate_accuracy",
        "payload_content_accuracy",
        "payload_transport_exact_rate",
    ),
    "transport_eos": (
        "alignment_position_accuracy",
        "alignment_copy_gate_accuracy",
        "payload_content_accuracy",
        "payload_eos_accuracy",
        "payload_transport_exact_rate",
    ),
    "decision": (),
    "operation": (),
    "address": ("region_accuracy", "start_accuracy", "end_accuracy"),
    "joint": ("typed_emission_exact_rate", "payload_transport_exact_rate"),
}


def foundation_motor_v2_component_weights(
    training_stage: str,
    *,
    receipt_continuation: bool = False,
    receipt_teaching_profile: str = RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    teach_multicell_copy: bool = False,
) -> dict[str, float]:
    """Effective objective weight per component for one stage and teaching profile."""

    if training_stage not in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        raise ValueError(f"unknown foundation motor v2 training stage {training_stage!r}")
    weights = dict(foundation_motor_v2_stage_policy(training_stage)["component_weights"])
    if receipt_continuation:
        weights = apply_receipt_continuation_teach_weights(
            weights,
            training_stage=training_stage,
            receipt_teaching_profile=receipt_teaching_profile,
        )
    elif teach_multicell_copy:
        weights = apply_copy_alignment_multicell_teach_weights(
            weights, training_stage=training_stage
        )
    return {name: float(value) for name, value in weights.items()}


def foundation_motor_v2_first_reachable_stage(metric: str) -> str | None:
    """First stage, in program order, at which ``metric`` can actually move.

    ``None`` means no stage weights every component the metric depends on, so
    the metric may never be a gate requirement.
    """

    components = FOUNDATION_MOTOR_V2_METRIC_COMPONENTS[metric]
    for stage in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        weights = foundation_motor_v2_component_weights(stage)
        if all(weights.get(name, 0.0) > 0.0 for name in components):
            return stage
    return None


def foundation_motor_v2_unreachable_gate_requirements(
    *,
    receipt_continuation: bool = False,
    receipt_teaching_profile: str = RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    teach_multicell_copy: bool = False,
) -> list[str]:
    """Stage/metric pairs that are unmet-able by construction; empty when healthy."""

    violations: list[str] = []
    for stage in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        weights = foundation_motor_v2_component_weights(
            stage,
            receipt_continuation=receipt_continuation,
            receipt_teaching_profile=receipt_teaching_profile,
            teach_multicell_copy=teach_multicell_copy,
        )
        for metric in FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS.get(stage, ()):
            components = FOUNDATION_MOTOR_V2_METRIC_COMPONENTS.get(metric)
            if components is None:
                continue
            dead = [name for name in components if weights.get(name, 0.0) == 0.0]
            if dead:
                violations.append(
                    f"{stage} gates on {metric} while weighting {dead} at 0.0"
                )
    return violations

FOUNDATION_MOTOR_V2_RETENTION_CONTRACT = {
    "schema": "axon-foundation-motor-v2-retention-contract-v1",
    "scope": "checkpoint_lineage_acceptance_not_curriculum_promotion",
    "acceptance_surface": "complete_heldout_and_regression",
    "evaluation_loss": "effective_training_component_weights",
    "comparison": "candidate_metrics_must_not_regress_from_accepted_parent",
    "progress_requirement": "at_least_one_protected_behavior_must_strictly_improve",
    "copy_alignment_metrics": [
        "alignment_position_accuracy",
        "alignment_copy_gate_accuracy",
        "payload_content_accuracy",
        "payload_eos_accuracy",
        "payload_transport_exact_rate",
    ],
    "copy_alignment_pair_metrics": ["position", "copy_gate", "content"],
}
FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_ID = canonical_sha256(
    FOUNDATION_MOTOR_V2_RETENTION_CONTRACT
)
def receipt_continuation_teach_profile(profile: str) -> Mapping[str, Any]:
    if profile == RECEIPT_TEACHING_PROFILE_CONTINUATION_V1:
        return RECEIPT_CONTINUATION_TEACH
    if profile == RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2:
        return RECEIPT_ROUTE_EOS_BALANCED_TEACH
    if profile == RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3:
        return RECEIPT_GENERATE_HEAD_EOS_TEACH
    if profile == RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4:
        return RECEIPT_GENERATE_HEAD_BALANCED_TEACH
    if profile == RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5:
        return RECEIPT_TERMINATION_HEAD_TEACH
    if profile == RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6:
        return RECEIPT_TERMINATION_HEAD_BALANCED_TEACH
    raise ValueError(f"unknown receipt teaching profile {profile!r}")


def foundation_motor_v2_objective_program_id(
    *,
    teach_multicell_copy: bool,
    receipt_continuation: bool = False,
    receipt_teaching_profile: str = RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
) -> str:
    """Return the exact optimizer objective identity for this campaign."""

    if receipt_continuation:
        if receipt_teaching_profile == RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_BALANCED_V6:
            return FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_BALANCED_PROGRAM_ID
        if receipt_teaching_profile == RECEIPT_TEACHING_PROFILE_TERMINATION_HEAD_V5:
            return FOUNDATION_MOTOR_V2_RECEIPT_TERMINATION_HEAD_PROGRAM_ID
        if receipt_teaching_profile == RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4:
            return FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM_ID
        if receipt_teaching_profile == RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3:
            return FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID
        if receipt_teaching_profile == RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2:
            return FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM_ID
        if receipt_teaching_profile != RECEIPT_TEACHING_PROFILE_CONTINUATION_V1:
            raise ValueError(
                f"unknown receipt teaching profile {receipt_teaching_profile!r}"
            )
        return FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM_ID
    if receipt_teaching_profile != RECEIPT_TEACHING_PROFILE_CONTINUATION_V1:
        raise ValueError(
            "a non-default receipt teaching profile requires receipt_continuation=True"
        )
    if teach_multicell_copy:
        return FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID
    return FOUNDATION_MOTOR_V2_PROGRAM_ID
DEFAULT_FOUNDATION_MOTOR_V2_SPLIT_COUNTS: tuple[
    tuple[str, tuple[int, int, int]], ...
] = (("F0", (72, 36, 36)),)

_SPLITS = ("train", "heldout", "regression")
_SYMBOLS = tuple(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?.:,;+-=*/_()[]{}"
    "λφΩЖя東亰終端心脳éïüñøå✓∞∑∆🙂🧠🚀"
)

# The original randomized v2 surface happened to place no multi-cell scalar in
# heldout, while its small regression surface exposed only three.  This second,
# content-addressed fixture generator does not replace or mutate that exam.  It
# adds a deliberately split-disjoint transport-walk surface where every aligned
# payload is two, three, or four UTF-8 cells and every split covers all three
# widths.  The characters are assigned and human-readable; their disjointness
# is part of the verifier below rather than an accident of a hash modulo.
_UNICODE_WALK_SYMBOLS: dict[str, dict[int, tuple[str, ...]]] = {
    "train": {
        2: tuple("ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏ"),
        3: tuple("あいうえおかきくけこさしすせそた"),
        4: tuple("😀😁😂😃😄😅😆😇😈😉😊😋😌😍😎😏"),
    },
    "heldout": {
        2: tuple("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠ"),
        3: tuple("アイウエオカキクケコサシスセソタ"),
        4: tuple("😐😑😒😓😔😕😖😗😘😙😚😛😜😝😞😟"),
    },
    "regression": {
        2: tuple("АБВГДЕЖЗИЙКЛМНОП"),
        3: tuple("天地玄黄宇宙洪荒日月盈昃辰宿列張"),
        4: tuple("🤐🤑🤒🤓🤔🤕🤖🤗🤘🤙🤚🤛🤜🤝🤞🤟"),
    },
}


def _unicode_walk_symbol(split: str, aligned_ordinal: int) -> str:
    if split not in _UNICODE_WALK_SYMBOLS:
        raise ValueError(f"unsupported Unicode-walk split {split!r}")
    width = (2, 3, 4)[aligned_ordinal % 3]
    pool = _UNICODE_WALK_SYMBOLS[split][width]
    symbol = pool[(aligned_ordinal // 3) % len(pool)]
    if len(encode_unicode_text(symbol)) != width:
        raise AssertionError("Unicode-walk fixture has the wrong transport width")
    return symbol


def _symbol(split: str, pair_index: int, variant: int) -> str:
    digest = hashlib.sha256(
        f"{FOUNDATION_MOTOR_STAGE}:{split}:{pair_index}:{variant}".encode("utf-8")
    ).digest()
    index = int.from_bytes(digest[:4], "big") % len(_SYMBOLS)
    if variant:
        first = _symbol(split, pair_index, 0)
        while _SYMBOLS[index] == first:
            index = (index + 1) % len(_SYMBOLS)
    return _SYMBOLS[index]


def _payload_alignment(*, symbol: str, source_start: int) -> dict[str, Any]:
    return {
        "schema": "axon-r0-target-alignment-v1",
        "segments": [
            {
                "target_start": 0,
                "target_end": 1,
                "source_region": LogicalRegion.CORTEX.value,
                "source_start": source_start,
                "source_end": source_start + 1,
                "text_sha256": hashlib.sha256(symbol.encode("utf-8")).hexdigest(),
                "authority": "exact_current_shared_field",
            }
        ],
        "supervise_eos_generate": True,
    }


def _episode(
    *,
    identity_text: str,
    split: str,
    pair_id: str,
    pair_index: int,
    variant: int,
    stage: str = FOUNDATION_MOTOR_STAGE,
    action_override: str | None = None,
    source_manifest_id: str = FOUNDATION_MOTOR_SOURCE_ID,
    source_schema: str = "axon-foundation-motor-source-v1",
    example_schema: str = "axon-foundation-motor-example-v1",
    label_prefix: str = "foundation-motor",
    extra_tags: tuple[str, ...] = (),
    symbol_override: str | None = None,
) -> LivingReasoningEpisode:
    action = (
        FOUNDATION_MOTOR_ACTIONS[pair_index % len(FOUNDATION_MOTOR_ACTIONS)]
        if action_override is None
        else action_override
    )
    if action not in FOUNDATION_MOTOR_ACTIONS:
        raise ValueError(f"unsupported foundation motor action {action!r}")
    symbol = (
        _symbol(split, pair_index, variant)
        if symbol_override is None
        else symbol_override
    )
    if len(symbol) != 1:
        raise ValueError("foundation motor symbol must be exactly one Unicode scalar")
    source_identity = canonical_sha256(
        {
            "schema": source_schema,
            "split": split,
            "pair_index": pair_index,
            "variant": variant,
            "symbol": symbol,
        }
    )
    prefix = f"SOURCE_SYMBOL[{source_identity[:12]}]="
    response = ""
    if action == "copy":
        prompt = "Copy the current SOURCE_SYMBOL exactly into the empty response draft."
        target = LivingReasoningTarget(
            phase="consolidated",
            decision=ReasoningDecision.DELTA,
            operation=ReasoningOperationKind.REPLACE,
            region=LogicalRegion.RESPONSE_DRAFT,
            start=0,
            end=0,
            payload=symbol,
            payload_alignment=_payload_alignment(symbol=symbol, source_start=len(prefix)),
        )
    elif action == "insert":
        response = "[]"
        prompt = "Insert the current SOURCE_SYMBOL between the brackets at response position 1."
        target = LivingReasoningTarget(
            phase="consolidated",
            decision=ReasoningDecision.DELTA,
            operation=ReasoningOperationKind.INSERT,
            region=LogicalRegion.RESPONSE_DRAFT,
            start=1,
            end=1,
            payload=symbol,
            payload_alignment=_payload_alignment(symbol=symbol, source_start=len(prefix)),
        )
    elif action == "replace":
        response = "____"
        position = pair_index % len(response)
        prompt = f"Replace response position {position} with the current SOURCE_SYMBOL."
        target = LivingReasoningTarget(
            phase="consolidated",
            decision=ReasoningDecision.DELTA,
            operation=ReasoningOperationKind.REPLACE,
            region=LogicalRegion.RESPONSE_DRAFT,
            start=position,
            end=position + 1,
            payload=symbol,
            payload_alignment=_payload_alignment(symbol=symbol, source_start=len(prefix)),
        )
    elif action == "delete":
        response = f"A{symbol}B"
        prompt = "Delete exactly response position 1; emit no replacement payload."
        target = LivingReasoningTarget(
            phase="consolidated",
            decision=ReasoningDecision.DELTA,
            operation=ReasoningOperationKind.DELETE,
            region=LogicalRegion.RESPONSE_DRAFT,
            start=1,
            end=2,
        )
    elif action == "no_op":
        response = f"stable:{symbol}"
        prompt = "The response draft is already correct. Make no change."
        target = LivingReasoningTarget(
            phase="consolidated", decision=ReasoningDecision.NO_OP
        )
    else:
        response = f"untrusted:{symbol}"
        prompt = "No response-draft operation is authorized. Abstain."
        target = LivingReasoningTarget(
            phase="consolidated", decision=ReasoningDecision.ABSTAIN
        )

    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: identity_text,
            LogicalRegion.USER_INPUT: prompt,
            LogicalRegion.CORTEX: prefix + symbol,
            LogicalRegion.SCRATCH: "",
            LogicalRegion.RESPONSE_DRAFT: response,
        },
        source_manifest_ids=(source_manifest_id,),
    )
    return LivingReasoningEpisode(
        label=f"{label_prefix}-{split}-{action}-{pair_index:03d}-{variant}",
        split=split,
        snapshot=snapshot,
        first_workspace_text=_workspace(
            ProposalPass.FIRST,
            "Identify the exact authorized decision, operation, region, address, and payload.",
        ),
        refined_workspace_text=_workspace(
            ProposalPass.REFINED,
            "Recheck the current field, exact address, Unicode transport, and EOS.",
        ),
        targets=(
            LivingReasoningTarget(
                phase="first", decision=ReasoningDecision.NO_OP, supervision_weight=0.0
            ),
            LivingReasoningTarget(
                phase="refined", decision=ReasoningDecision.NO_OP, supervision_weight=0.0
            ),
            target,
        ),
        mechanism_tags=(
            "foundation_motor",
            f"foundation_stage:{stage}",
            f"foundation_motor_action:{action}",
            f"foundation_pair:{pair_id}",
            f"foundation_variant:{variant}",
            f"transfer_item:{source_identity}",
            "target_validator:exact_typed_delta",
            *extra_tags,
        ),
        outcome_quality="authored_objectively_checkable_target",
        source_example_id=canonical_sha256(
            {
                "schema": example_schema,
                "source_identity": source_identity,
                "action": action,
                "target_id": target.target_id,
            }
        ),
        target_basis="deterministic exact typed operation over the current Shared Field",
    )


def compile_foundation_motor(
    *,
    identity_text: str,
    requested_counts: tuple[
        tuple[str, tuple[int, int, int]], ...
    ] = DEFAULT_FOUNDATION_MOTOR_SPLIT_COUNTS,
) -> FirstFormCurriculum:
    if not identity_text:
        raise ValueError("foundation motor curriculum requires canonical Identity")
    if tuple(family for family, _counts in requested_counts) != ("F0",):
        raise ValueError("foundation motor curriculum uses the registered F0 family")
    counts = tuple(int(value) for value in requested_counts[0][1])
    minimum = 2 * len(FOUNDATION_MOTOR_ACTIONS)
    if len(counts) != 3 or any(value < minimum or value % 2 for value in counts):
        raise ValueError(f"each motor split count must be even and at least {minimum}")

    cases: list[FirstFormCase] = []
    for split, count in zip(_SPLITS, counts, strict=True):
        for pair_index in range(count // 2):
            pair_id = canonical_sha256(
                {
                    "schema": "axon-foundation-motor-pair-v1",
                    "split": split,
                    "pair_index": pair_index,
                    "action": FOUNDATION_MOTOR_ACTIONS[
                        pair_index % len(FOUNDATION_MOTOR_ACTIONS)
                    ],
                }
            )
            for variant in (0, 1):
                episode = _episode(
                    identity_text=identity_text,
                    split=split,
                    pair_id=pair_id,
                    pair_index=pair_index,
                    variant=variant,
                )
                compiled = D64FieldCompiler().compile(episode.snapshot)
                compiled.verify_roundtrip(episode.snapshot)
                action = FOUNDATION_MOTOR_ACTIONS[pair_index % len(FOUNDATION_MOTOR_ACTIONS)]
                cases.append(
                    FirstFormCase(
                        family="F0",
                        competency=f"foundation_motor_{action}",
                        eligibility=TeachingEligibility.VERIFIED_TARGET,
                        lineage_id=f"foundation-motor-v1:{split}:{pair_index:04d}",
                        source_record_ids=(),
                        episode=episode,
                        transport_pages=len(tuple(compiled.iter_character_pages(32))),
                        procedural_depth=1,
                        derived=True,
                    )
                )
    curriculum = FirstFormCurriculum(
        cases=tuple(cases),
        source_import_ids=(FOUNDATION_MOTOR_SOURCE_ID,),
        requested_family_split_counts=requested_counts,
        excluded_counts=(),
        identity_text_sha256=hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
    )
    verify_foundation_motor_curriculum(curriculum)
    return curriculum


def _tag_value(episode: LivingReasoningEpisode, prefix: str) -> str | None:
    matches = tuple(tag[len(prefix) :] for tag in episode.mechanism_tags if tag.startswith(prefix))
    if len(matches) > 1:
        raise ValueError(f"foundation motor episode repeats tag prefix {prefix!r}")
    return matches[0] if matches else None


def is_foundation_motor_episode(episode: LivingReasoningEpisode) -> bool:
    return f"foundation_stage:{FOUNDATION_MOTOR_STAGE}" in episode.mechanism_tags


def verify_foundation_motor_curriculum(curriculum: FirstFormCurriculum) -> None:
    split_sources: dict[str, set[str]] = defaultdict(set)
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    actions: dict[str, set[str]] = defaultdict(set)
    for case in curriculum.cases:
        if case.family != "F0" or not is_foundation_motor_episode(case.episode):
            raise ValueError("foundation motor manifest contains a foreign case")
        pair_id = _tag_value(case.episode, "foundation_pair:")
        variant = _tag_value(case.episode, "foundation_variant:")
        transfer = _tag_value(case.episode, "transfer_item:")
        action = _tag_value(case.episode, "foundation_motor_action:")
        if (
            pair_id is None
            or variant not in {"0", "1"}
            or transfer is None
            or action not in FOUNDATION_MOTOR_ACTIONS
        ):
            raise ValueError("foundation motor case lacks governed pair/action identity")
        split = case.episode.split
        if transfer in split_sources[split]:
            raise ValueError("foundation motor source repeats within a split")
        split_sources[split].add(transfer)
        pairs[(split, pair_id)].append(variant)
        actions[split].add(action)
    for split_a in _SPLITS:
        if actions[split_a] != set(FOUNDATION_MOTOR_ACTIONS):
            raise ValueError("every motor split must cover every typed action")
        for split_b in _SPLITS:
            if split_a < split_b and not split_sources[split_a].isdisjoint(split_sources[split_b]):
                raise ValueError("foundation motor source crosses data splits")
    if any(sorted(variants) != ["0", "1"] for variants in pairs.values()):
        raise ValueError("every foundation motor pair requires variants zero and one")


def _merge_row_histogram(
    selected: list[Any], name: str
) -> dict[str, int]:
    """Merge per-case answer histograms into the whole-probe surface histogram."""

    merged: dict[str, int] = {}
    for _episode, row, _diagnostic in selected:
        for label, count in (row.get(name) or {}).items():
            merged[str(label)] = merged.get(str(label), 0) + int(count)
    return merged


def foundation_motor_probe(
    episodes: Sequence[LivingReasoningEpisode], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    if len(episodes) != len(rows):
        raise ValueError("foundation motor probe episodes and rows must align")
    paired: dict[str, list[bool]] = defaultdict(list)
    payload_paired: dict[str, list[bool]] = defaultdict(list)
    action_results: dict[str, list[bool]] = defaultdict(list)
    selected = exact_cases = 0
    complete_outputs = phase_outputs = 0.0
    for episode, row in zip(episodes, rows, strict=True):
        if not is_foundation_motor_episode(episode):
            continue
        pair_id = _tag_value(episode, "foundation_pair:")
        action = _tag_value(episode, "foundation_motor_action:")
        if pair_id is None or action is None:
            raise ValueError("foundation motor evaluation case lacks pair/action identity")
        exact = (
            row["typed_emission_exact_count"] == row["supervised_phase_count"]
            and row["payload_transport_exact_count"] == row["payload_supervised_phase_count"]
        )
        selected += 1
        exact_cases += int(exact)
        paired[pair_id].append(bool(exact))
        action_results[action].append(bool(exact))
        if episode.targets[-1].payload:
            payload_paired[pair_id].append(bool(exact))
        complete_outputs += float(row["complete_field_coverage_count"])
        phase_outputs += float(row["phase_output_count"])
    if not selected:
        return None
    if any(len(values) != 2 for values in paired.values()) or any(
        len(values) != 2 for values in payload_paired.values()
    ):
        raise ValueError("foundation motor evaluation contains an incomplete source pair")
    return {
        "schema": "axon-foundation-motor-probe-v1",
        "stage": FOUNDATION_MOTOR_STAGE,
        "case_count": selected,
        "pair_count": len(paired),
        "payload_pair_count": len(payload_paired),
        "free_running_case_exact_rate": exact_cases / selected,
        "changed_source_pair_exact_rate": sum(all(v) for v in paired.values()) / len(paired),
        "payload_changed_source_pair_exact_rate": (
            sum(all(v) for v in payload_paired.values()) / len(payload_paired)
        ),
        "per_action_exact_rate": {
            action: sum(values) / len(values) for action, values in sorted(action_results.items())
        },
        "complete_field_coverage_rate": complete_outputs / max(1.0, phase_outputs),
    }


def decide_foundation_motor_mastery(
    *,
    heldout_probe: Mapping[str, Any] | None,
    regression_probe: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    complete_heldout: bool,
    complete_regression: bool,
) -> dict[str, Any] | None:
    if heldout_probe is None and regression_probe is None:
        return None
    failures: list[str] = []
    if heldout_probe is None or regression_probe is None:
        failures.append("missing heldout or regression motor probe")
    else:
        for label, probe in (("heldout", heldout_probe), ("regression", regression_probe)):
            if probe["free_running_case_exact_rate"] < 0.95:
                failures.append(f"{label} exact rate below 0.95")
            if probe["changed_source_pair_exact_rate"] < 0.95:
                failures.append(f"{label} changed-source pair exact rate below 0.95")
            if probe["payload_changed_source_pair_exact_rate"] < 0.95:
                failures.append(f"{label} payload source-pair exact rate below 0.95")
            if any(rate < 0.95 for rate in probe["per_action_exact_rate"].values()):
                failures.append(f"{label} per-action exact rate below 0.95")
            if probe["complete_field_coverage_rate"] != 1.0:
                failures.append(f"{label} complete-field coverage is not exact")
    if not complete_heldout:
        failures.append("heldout surface is incomplete")
    if not complete_regression:
        failures.append("regression surface is incomplete")
    if not (
        float(evaluation["payload_teacher_forced_token_accuracy"])
        > float(evaluation["constant_payload_token_accuracy_floor"])
    ):
        failures.append("teacher-forced token accuracy does not exceed the constant floor")
    body = {
        "schema": "axon-foundation-stage-gate-v1",
        "stage": FOUNDATION_MOTOR_STAGE,
        "scope": "curriculum_advancement_only_not_serving_or_promotion",
        "passed": not failures,
        "failures": failures,
        "heldout_probe": None if heldout_probe is None else dict(heldout_probe),
        "regression_probe": None if regression_probe is None else dict(regression_probe),
        "policy_id": FOUNDATION_MOTOR_GATE_POLICY_ID,
        "policy": FOUNDATION_MOTOR_GATE_POLICY,
    }
    return {**body, "decision_id": canonical_sha256(body)}


def _v2_pair_actions(case_count: int) -> tuple[str, ...]:
    """Balance decision and operation heads simultaneously.

    Each pair contributes two changed-source cases. One third of pairs are
    DELTA, NO_OP and ABSTAIN respectively. Within DELTA, INSERT, DELETE and
    REPLACE are balanced; the REPLACE share is split equally between explicit
    replace and empty-draft copy lessons.
    """

    if case_count < 36 or case_count % 36:
        raise ValueError("each v2 motor split count must be divisible by 36")
    decision_pairs = case_count // 6
    operation_pairs = decision_pairs // 3
    replace_half = operation_pairs // 2
    delta_actions = (
        ["insert"] * operation_pairs
        + ["delete"] * operation_pairs
        + ["copy"] * replace_half
        + ["replace"] * replace_half
    )
    actions: list[str] = []
    for delta_action in delta_actions:
        actions.extend((delta_action, "no_op", "abstain"))
    if len(actions) != case_count // 2:
        raise AssertionError("v2 motor balance arithmetic is inconsistent")
    return tuple(actions)


def compile_foundation_motor_v2(
    *,
    identity_text: str,
    requested_counts: tuple[
        tuple[str, tuple[int, int, int]], ...
    ] = DEFAULT_FOUNDATION_MOTOR_V2_SPLIT_COUNTS,
) -> FirstFormCurriculum:
    if not identity_text:
        raise ValueError("foundation motor v2 curriculum requires canonical Identity")
    if tuple(family for family, _counts in requested_counts) != ("F0",):
        raise ValueError("foundation motor v2 uses the registered F0 family")
    counts = tuple(int(value) for value in requested_counts[0][1])
    if len(counts) != 3:
        raise ValueError("foundation motor v2 requires train, heldout and regression counts")

    cases: list[FirstFormCase] = []
    for split, count in zip(_SPLITS, counts, strict=True):
        actions = _v2_pair_actions(count)
        for pair_index, action in enumerate(actions):
            pair_id = canonical_sha256(
                {
                    "schema": "axon-foundation-motor-pair-v2",
                    "split": split,
                    "pair_index": pair_index,
                    "action": action,
                }
            )
            for variant in (0, 1):
                episode = _episode(
                    identity_text=identity_text,
                    split=split,
                    pair_id=pair_id,
                    pair_index=pair_index,
                    variant=variant,
                    stage=FOUNDATION_MOTOR_V2_STAGE,
                    action_override=action,
                    source_manifest_id=FOUNDATION_MOTOR_V2_SOURCE_ID,
                    source_schema="axon-foundation-motor-source-v2",
                    example_schema="axon-foundation-motor-example-v2",
                    label_prefix="foundation-motor-v2",
                    extra_tags=(
                        "foundation_motor_v2",
                        f"foundation_program:{FOUNDATION_MOTOR_V2_PROGRAM_ID}",
                    ),
                )
                compiled = D64FieldCompiler().compile(episode.snapshot)
                compiled.verify_roundtrip(episode.snapshot)
                cases.append(
                    FirstFormCase(
                        family="F0",
                        competency=f"foundation_motor_v2_{action}",
                        eligibility=TeachingEligibility.VERIFIED_TARGET,
                        lineage_id=f"foundation-motor-v2:{split}:{pair_index:04d}",
                        source_record_ids=(),
                        episode=episode,
                        transport_pages=len(tuple(compiled.iter_character_pages(32))),
                        procedural_depth=1,
                        derived=True,
                    )
                )
    curriculum = FirstFormCurriculum(
        cases=tuple(cases),
        source_import_ids=(FOUNDATION_MOTOR_V2_SOURCE_ID,),
        requested_family_split_counts=requested_counts,
        excluded_counts=(),
        identity_text_sha256=hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
    )
    verify_foundation_motor_v2_curriculum(curriculum)
    return curriculum


def compile_foundation_motor_v2_unicode_walk(
    *,
    identity_text: str,
    requested_counts: tuple[
        tuple[str, tuple[int, int, int]], ...
    ] = DEFAULT_FOUNDATION_MOTOR_V2_SPLIT_COUNTS,
) -> FirstFormCurriculum:
    """Compile the honest multi-cell pointer-walk lesson and exam.

    This keeps motor-v2 actions, targets, and gates intact while replacing
    accidental symbol sampling with explicit 2/3/4-cell Unicode strata.  Its
    distinct source identity and resulting manifest identity prevent it from
    being confused with the historical randomized v2 curriculum.
    """

    if not identity_text:
        raise ValueError("foundation motor v2 Unicode walk requires canonical Identity")
    if tuple(family for family, _counts in requested_counts) != ("F0",):
        raise ValueError("foundation motor v2 Unicode walk uses the registered F0 family")
    counts = tuple(int(value) for value in requested_counts[0][1])
    if len(counts) != 3:
        raise ValueError("foundation motor v2 Unicode walk requires train, heldout and regression counts")

    cases: list[FirstFormCase] = []
    aligned_actions = {"copy", "insert", "replace"}
    for split, count in zip(_SPLITS, counts, strict=True):
        actions = _v2_pair_actions(count)
        aligned_ordinal = 0
        for pair_index, action in enumerate(actions):
            pair_id = canonical_sha256(
                {
                    "schema": "axon-foundation-motor-v2-unicode-walk-pair-v1",
                    "split": split,
                    "pair_index": pair_index,
                    "action": action,
                }
            )
            for variant in (0, 1):
                symbol = None
                if action in aligned_actions:
                    symbol = _unicode_walk_symbol(split, aligned_ordinal)
                    aligned_ordinal += 1
                episode = _episode(
                    identity_text=identity_text,
                    split=split,
                    pair_id=pair_id,
                    pair_index=pair_index,
                    variant=variant,
                    stage=FOUNDATION_MOTOR_V2_STAGE,
                    action_override=action,
                    source_manifest_id=FOUNDATION_MOTOR_V2_UNICODE_WALK_SOURCE_ID,
                    source_schema="axon-foundation-motor-v2-unicode-walk-source-v1",
                    example_schema="axon-foundation-motor-v2-unicode-walk-example-v1",
                    label_prefix="foundation-motor-v2-unicode-walk",
                    extra_tags=(
                        "foundation_motor_v2",
                        "foundation_unicode_walk",
                        f"foundation_program:{FOUNDATION_MOTOR_V2_PROGRAM_ID}",
                    ),
                    symbol_override=symbol,
                )
                compiled = D64FieldCompiler().compile(episode.snapshot)
                compiled.verify_roundtrip(episode.snapshot)
                cases.append(
                    FirstFormCase(
                        family="F0",
                        competency=f"foundation_motor_v2_unicode_walk_{action}",
                        eligibility=TeachingEligibility.VERIFIED_TARGET,
                        lineage_id=f"foundation-motor-v2-unicode-walk:{split}:{pair_index:04d}",
                        source_record_ids=(),
                        episode=episode,
                        transport_pages=len(tuple(compiled.iter_character_pages(32))),
                        procedural_depth=1,
                        derived=True,
                    )
                )
    curriculum = FirstFormCurriculum(
        cases=tuple(cases),
        source_import_ids=(FOUNDATION_MOTOR_V2_UNICODE_WALK_SOURCE_ID,),
        requested_family_split_counts=requested_counts,
        excluded_counts=(),
        identity_text_sha256=hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
    )
    verify_foundation_motor_v2_unicode_walk_curriculum(curriculum)
    return curriculum


def is_foundation_motor_v2_episode(episode: LivingReasoningEpisode) -> bool:
    return f"foundation_stage:{FOUNDATION_MOTOR_V2_STAGE}" in episode.mechanism_tags


def foundation_motor_v2_action(episode: LivingReasoningEpisode) -> str | None:
    if not is_foundation_motor_v2_episode(episode):
        return None
    return _tag_value(episode, "foundation_motor_action:")


def foundation_motor_v2_stage_policy(stage: str) -> Mapping[str, Any]:
    for item in FOUNDATION_MOTOR_V2_PROGRAM["stages"]:
        if item["name"] == stage:
            return item
    raise KeyError(stage)


def foundation_motor_payload_transport_cells(episode: LivingReasoningEpisode) -> int:
    payload = episode.targets[-1].payload or ""
    return len(encode_unicode_text(payload)) if payload else 0


def apply_copy_alignment_multicell_teach_weights(
    weights: Mapping[str, float],
    *,
    training_stage: str,
) -> dict[str, float]:
    result = {key: float(value) for key, value in weights.items()}
    if training_stage != "copy_alignment":
        return result
    result.update(COPY_ALIGNMENT_MULTICELL_TEACH["component_weight_overrides"])
    return result


def apply_receipt_continuation_teach_weights(
    weights: Mapping[str, float],
    *,
    training_stage: str,
    receipt_teaching_profile: str = RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
) -> dict[str, float]:
    result = {key: float(value) for key, value in weights.items()}
    if training_stage != "copy_alignment":
        return result
    overlay = receipt_continuation_teach_profile(receipt_teaching_profile)
    result.update(overlay["component_weight_overrides"])
    return result


def oversample_multicell_copy_cases(
    cases: Sequence[FirstFormCase],
) -> tuple[FirstFormCase, ...]:
    """Repeat authored multi-cell letters until they fill half the training lane."""

    ordered = tuple(cases)
    single: list[FirstFormCase] = []
    multi: list[FirstFormCase] = []
    for case in ordered:
        if foundation_motor_payload_transport_cells(case.episode) > 1:
            multi.append(case)
        else:
            single.append(case)
    if not multi:
        return ordered
    repeated = list(multi)
    while len(repeated) < max(len(single), len(multi)):
        repeated.extend(multi)
    return tuple(single + repeated)


def verify_foundation_motor_v2_curriculum(curriculum: FirstFormCurriculum) -> None:
    split_sources: dict[str, set[str]] = defaultdict(set)
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    actions: dict[str, list[str]] = defaultdict(list)
    for case in curriculum.cases:
        episode = case.episode
        if case.family != "F0" or not is_foundation_motor_v2_episode(episode):
            raise ValueError("foundation motor v2 manifest contains a foreign case")
        if f"foundation_program:{FOUNDATION_MOTOR_V2_PROGRAM_ID}" not in episode.mechanism_tags:
            raise ValueError("foundation motor v2 case lacks exact teaching-program identity")
        pair_id = _tag_value(episode, "foundation_pair:")
        variant = _tag_value(episode, "foundation_variant:")
        transfer = _tag_value(episode, "transfer_item:")
        action = foundation_motor_v2_action(episode)
        if (
            pair_id is None
            or variant not in {"0", "1"}
            or transfer is None
            or action not in FOUNDATION_MOTOR_ACTIONS
        ):
            raise ValueError("foundation motor v2 case lacks governed pair/action identity")
        split = episode.split
        if transfer in split_sources[split]:
            raise ValueError("foundation motor v2 source repeats within a split")
        split_sources[split].add(transfer)
        pairs[(split, pair_id)].append(variant)
        actions[split].append(action)

    if any(sorted(variants) != ["0", "1"] for variants in pairs.values()):
        raise ValueError("every foundation motor v2 pair requires variants zero and one")
    for split_a in _SPLITS:
        counts = {action: actions[split_a].count(action) for action in FOUNDATION_MOTOR_ACTIONS}
        delta = counts["copy"] + counts["insert"] + counts["replace"] + counts["delete"]
        if not (delta == counts["no_op"] == counts["abstain"]):
            raise ValueError("v2 decision teaching mass is not balanced")
        if not (counts["insert"] == counts["delete"] == counts["copy"] + counts["replace"]):
            raise ValueError("v2 operation teaching mass is not balanced")
        if counts["copy"] != counts["replace"]:
            raise ValueError("v2 REPLACE teaching must balance copy and explicit replace")
        for split_b in _SPLITS:
            if split_a < split_b and not split_sources[split_a].isdisjoint(split_sources[split_b]):
                raise ValueError("foundation motor v2 source crosses data splits")


def verify_foundation_motor_v2_unicode_walk_curriculum(
    curriculum: FirstFormCurriculum,
) -> None:
    verify_foundation_motor_v2_curriculum(curriculum)
    symbols: dict[str, set[str]] = defaultdict(set)
    widths: dict[str, set[int]] = defaultdict(set)
    aligned_counts: dict[str, int] = defaultdict(int)
    for case in curriculum.cases:
        episode = case.episode
        if "foundation_unicode_walk" not in episode.mechanism_tags:
            raise ValueError("Unicode-walk curriculum contains a foreign motor-v2 case")
        target = episode.targets[-1]
        if target.payload_alignment is None:
            continue
        payload = target.payload or ""
        width = len(encode_unicode_text(payload))
        if width not in (2, 3, 4):
            raise ValueError("Unicode-walk aligned payload is not multi-cell")
        symbols[episode.split].add(payload)
        widths[episode.split].add(width)
        aligned_counts[episode.split] += 1
    for split in _SPLITS:
        if widths[split] != {2, 3, 4} or aligned_counts[split] < 6:
            raise ValueError("every Unicode-walk split must cover 2/3/4-cell aligned payloads")
    for split_a in _SPLITS:
        for split_b in _SPLITS:
            if split_a < split_b and not symbols[split_a].isdisjoint(symbols[split_b]):
                raise ValueError("Unicode-walk scalar crosses data splits")


def foundation_motor_v2_probe(
    episodes: Sequence[LivingReasoningEpisode], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    if len(episodes) != len(rows):
        raise ValueError("foundation motor v2 probe episodes and rows must align")
    selected: list[tuple[LivingReasoningEpisode, Mapping[str, Any], Mapping[str, Any]]] = []
    for episode, row in zip(episodes, rows, strict=True):
        if not is_foundation_motor_v2_episode(episode):
            continue
        diagnostics = row.get("phase_diagnostics", ())
        if len(diagnostics) != 1:
            raise ValueError("foundation motor v2 case requires one supervised phase diagnostic")
        selected.append((episode, row, diagnostics[0]))
    if not selected:
        return None

    def rate(correct_key: str, count_key: str) -> float:
        count = sum(int(row[count_key]) for _episode, row, _diag in selected)
        return sum(int(row[correct_key]) for _episode, row, _diag in selected) / max(1, count)

    pair_components: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_action: dict[str, list[bool]] = defaultdict(list)
    per_decision: dict[str, list[bool]] = defaultdict(list)
    per_operation: dict[str, list[bool]] = defaultdict(list)
    for episode, row, diagnostic in selected:
        pair_id = _tag_value(episode, "foundation_pair:")
        action = foundation_motor_v2_action(episode)
        if pair_id is None or action is None:
            raise ValueError("foundation motor v2 evaluation lacks pair/action identity")
        full_exact = (
            row["typed_emission_exact_count"] == row["supervised_phase_count"]
            and row["payload_transport_exact_count"]
            == row["payload_supervised_phase_count"]
        )
        per_action[action].append(bool(full_exact))
        decision_name = str(diagnostic["target_decision"])
        per_decision[decision_name].append(bool(diagnostic["decision_exact"]))
        pair_components[pair_id]["decision"].append(bool(diagnostic["decision_exact"]))
        pair_components[pair_id]["joint"].append(bool(full_exact))
        if diagnostic.get("target_operation") is not None:
            operation_name = str(diagnostic["target_operation"])
            per_operation[operation_name].append(bool(diagnostic["operation_exact"]))
            pair_components[pair_id]["operation"].append(
                bool(diagnostic["operation_exact"])
            )
            pair_components[pair_id]["address"].append(
                all(
                    bool(diagnostic[name])
                    for name in ("region_exact", "start_exact", "end_exact")
                )
            )
        if diagnostic.get("payload_content_exact") is not None:
            pair_components[pair_id]["content"].append(
                bool(diagnostic["payload_content_exact"])
            )
        for component, name in (
            ("position", "alignment_position_exact"),
            ("copy_gate", "alignment_copy_gate_exact"),
            ("eos_gate", "alignment_eos_gate_exact"),
        ):
            if diagnostic.get(name) is not None:
                pair_components[pair_id][component].append(bool(diagnostic[name]))

    pair_rates: dict[str, float] = {}
    for component in (
        "position",
        "copy_gate",
        "eos_gate",
        "content",
        "decision",
        "operation",
        "address",
        "joint",
    ):
        relevant = [
            values[component]
            for values in pair_components.values()
            if component in values
        ]
        if any(len(values) != 2 for values in relevant):
            raise ValueError(f"foundation motor v2 {component} pair is incomplete")
        pair_rates[component] = sum(all(values) for values in relevant) / max(1, len(relevant))

    target_counts = [
        sum(int(row["payload_teacher_forced_target_counts"][index]) for _e, row, _d in selected)
        for index in range(len(selected[0][1]["payload_teacher_forced_target_counts"]) - 1)
    ]
    content_count = sum(
        int(row["payload_teacher_forced_content_count"]) for _e, row, _d in selected
    )
    merged_floors = constant_baseline_floors(
        _merge_row_histogram(selected, "constant_typed_emission_target_histogram"),
        _merge_row_histogram(selected, "constant_payload_transport_target_histogram"),
        supervised_phase_count=sum(
            float(row["supervised_phase_count"]) for _e, row, _d in selected
        ),
        payload_supervised_phase_count=sum(
            float(row["payload_supervised_phase_count"]) for _e, row, _d in selected
        ),
    )
    return {
        "schema": "axon-foundation-motor-v2-probe-v1",
        "stage": FOUNDATION_MOTOR_V2_STAGE,
        "case_count": len(selected),
        "complete_field_coverage_rate": sum(
            float(row["complete_field_coverage_count"]) for _e, row, _d in selected
        )
        / max(1.0, sum(float(row["phase_output_count"]) for _e, row, _d in selected)),
        "alignment_position_accuracy": rate(
            "alignment_position_correct", "alignment_position_count"
        ),
        "alignment_copy_gate_accuracy": rate(
            "alignment_copy_gate_correct", "alignment_copy_gate_count"
        ),
        "alignment_eos_gate_accuracy": rate(
            "alignment_eos_gate_correct", "alignment_eos_gate_count"
        ),
        "payload_content_accuracy": rate(
            "payload_teacher_forced_content_correct",
            "payload_teacher_forced_content_count",
        ),
        "payload_content_constant_floor": max(target_counts) / max(1, content_count),
        "typed_emission_exact_rate": rate(
            "typed_emission_exact_count", "supervised_phase_count"
        ),
        "constant_typed_emission_exact_floor": merged_floors[
            "constant_typed_emission_exact_floor"
        ],
        "constant_payload_transport_exact_floor": merged_floors[
            "constant_payload_transport_exact_floor"
        ],
        "payload_eos_accuracy": rate(
            "payload_teacher_forced_eos_correct", "payload_teacher_forced_eos_count"
        ),
        "payload_transport_exact_rate": rate(
            "payload_transport_exact_count", "payload_supervised_phase_count"
        ),
        "decision_accuracy": rate("decision_correct", "supervised_phase_count"),
        "operation_accuracy": rate("operation_correct", "operation_count"),
        "region_accuracy": rate("region_correct", "region_count"),
        "start_accuracy": rate("start_correct", "start_count"),
        "end_accuracy": rate("end_correct", "end_count"),
        "pair_exact_rates": pair_rates,
        "per_decision_accuracy": {
            name: sum(values) / len(values) for name, values in sorted(per_decision.items())
        },
        "per_operation_accuracy": {
            name: sum(values) / len(values) for name, values in sorted(per_operation.items())
        },
        "per_action_joint_exact_rate": {
            name: sum(values) / len(values) for name, values in sorted(per_action.items())
        },
    }


def decide_foundation_motor_v2_checkpoint_retention(
    *,
    training_stage: str,
    accepted_heldout_probe: Mapping[str, Any] | None,
    accepted_regression_probe: Mapping[str, Any] | None,
    candidate_heldout_probe: Mapping[str, Any] | None,
    candidate_regression_probe: Mapping[str, Any] | None,
    complete_heldout: bool,
    complete_regression: bool,
) -> dict[str, Any]:
    """Reject a tentative checkpoint that forgets behavior its parent retained.

    This gate selects the recoverable optimizer lineage.  It does not complete
    a homework assignment or promote a core.  Exact stage gates remain a
    separate decision over complete heldout and regression surfaces.
    """

    if training_stage not in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        raise ValueError(f"unknown foundation motor v2 training stage {training_stage!r}")
    failures: list[str] = []
    improvements: list[str] = []
    if not complete_heldout:
        failures.append("heldout retention surface is incomplete")
    if not complete_regression:
        failures.append("regression retention surface is incomplete")
    metric_names = tuple(
        FOUNDATION_MOTOR_V2_RETENTION_CONTRACT["copy_alignment_metrics"]
    )
    pair_names = tuple(
        FOUNDATION_MOTOR_V2_RETENTION_CONTRACT["copy_alignment_pair_metrics"]
    )
    for label, accepted, candidate in (
        ("heldout", accepted_heldout_probe, candidate_heldout_probe),
        ("regression", accepted_regression_probe, candidate_regression_probe),
    ):
        if accepted is None or candidate is None:
            failures.append(f"{label} retention probe is missing")
            continue
        for metric in metric_names:
            before = float(accepted[metric])
            after = float(candidate[metric])
            if after + 1e-12 < before:
                failures.append(f"{label} {metric} regressed {before:.12g} -> {after:.12g}")
            elif after > before + 1e-12:
                improvements.append(
                    f"{label} {metric} improved {before:.12g} -> {after:.12g}"
                )
        accepted_pairs = accepted["pair_exact_rates"]
        candidate_pairs = candidate["pair_exact_rates"]
        for metric in pair_names:
            before = float(accepted_pairs[metric])
            after = float(candidate_pairs[metric])
            if after + 1e-12 < before:
                failures.append(
                    f"{label} changed-source {metric} regressed {before:.12g} -> {after:.12g}"
                )
            elif after > before + 1e-12:
                improvements.append(
                    f"{label} changed-source {metric} improved {before:.12g} -> {after:.12g}"
                )
    if not improvements:
        failures.append("no protected motor behavior improved")
    decision = {
        "schema": "axon-foundation-motor-v2-retention-decision-v1",
        "contract_id": FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_ID,
        "training_stage": training_stage,
        "complete_heldout": bool(complete_heldout),
        "complete_regression": bool(complete_regression),
        "surface_kind": (
            "complete" if complete_heldout and complete_regression else "bounded_debug_probe"
        ),
        "passed": not failures,
        "failures": failures,
        "improvements": improvements,
    }
    decision["decision_id"] = canonical_sha256(decision)
    return decision


# Three-state successor to the v1 contract.  v1 treated plateau as failure,
# discarding optimizer state that may be traversing a flat region before a
# categorical behavior flips.  v2 separates the three observable outcomes and
# leaves the plateau policy (bounded probation) to the launcher's guard:
#   improvement -> accept (no regression, at least one protected metric up)
#   regression  -> reject and roll back (absolute; never loosened)
#   plateau     -> neither regressed nor improved; probation is a launch policy
_FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2_BASE = {
    key: value
    for key, value in FOUNDATION_MOTOR_V2_RETENTION_CONTRACT.items()
    if key != "progress_requirement"
}
FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2 = {
    **_FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2_BASE,
    "schema": "axon-foundation-motor-v2-retention-contract-v2",
    "states": {
        "improvement": (
            "accept: no protected metric regressed and at least one "
            "protected metric strictly improved; becomes the accepted parent"
        ),
        "regression": (
            "reject_and_rollback: any protected behavioral metric fell; "
            "restore the last confirmed accepted parent unconditionally"
        ),
        "plateau": (
            "probation_candidate: no protected metric changed; the launch "
            "policy may continue this exact optimizer state without "
            "promotion for up to max_consecutive_plateau_tranches"
        ),
    },
    "plateau_handling": "launcher_policy_bounded_probation",
}
FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2_ID = canonical_sha256(
    FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2
)


def decide_foundation_motor_v2_checkpoint_retention_v2(
    *,
    training_stage: str,
    accepted_heldout_probe: Mapping[str, Any] | None,
    accepted_regression_probe: Mapping[str, Any] | None,
    candidate_heldout_probe: Mapping[str, Any] | None,
    candidate_regression_probe: Mapping[str, Any] | None,
    complete_heldout: bool,
    complete_regression: bool,
) -> dict[str, Any]:
    """Classify a tentative checkpoint as improvement, regression, or plateau.

    Unlike the v1 contract, stasis is not a failure: plateau is its own
    state so the caller can hold the optimizer state on probation instead
    of destroying it.  Structural problems (incomplete surfaces, missing
    probes) fail closed as regression.
    """

    if training_stage not in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        raise ValueError(f"unknown foundation motor v2 training stage {training_stage!r}")
    structural: list[str] = []
    if not complete_heldout:
        structural.append("heldout retention surface is incomplete")
    if not complete_regression:
        structural.append("regression retention surface is incomplete")
    regressions: list[str] = []
    improvements: list[str] = []
    for label, accepted, candidate in (
        ("heldout", accepted_heldout_probe, candidate_heldout_probe),
        ("regression", accepted_regression_probe, candidate_regression_probe),
    ):
        if accepted is None or candidate is None:
            structural.append(f"{label} retention probe is missing")
            continue
        for metric in FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2["copy_alignment_metrics"]:
            before = float(accepted[metric])
            after = float(candidate[metric])
            if after + 1e-12 < before:
                regressions.append(f"{label} {metric} regressed {before:.12g} -> {after:.12g}")
            elif after > before + 1e-12:
                improvements.append(
                    f"{label} {metric} improved {before:.12g} -> {after:.12g}"
                )
        accepted_pairs = accepted["pair_exact_rates"]
        candidate_pairs = candidate["pair_exact_rates"]
        for metric in FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2["copy_alignment_pair_metrics"]:
            before = float(accepted_pairs[metric])
            after = float(candidate_pairs[metric])
            if after + 1e-12 < before:
                regressions.append(
                    f"{label} changed-source {metric} regressed {before:.12g} -> {after:.12g}"
                )
            elif after > before + 1e-12:
                improvements.append(
                    f"{label} changed-source {metric} improved {before:.12g} -> {after:.12g}"
                )
    if structural:
        state = "regression"
        failures = structural + regressions
    elif regressions:
        state = "regression"
        failures = regressions
    elif improvements:
        state = "improvement"
        failures = []
    else:
        state = "plateau"
        failures = []
    decision = {
        "schema": "axon-foundation-motor-v2-retention-decision-v2",
        "contract_id": FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2_ID,
        "training_stage": training_stage,
        "state": state,
        "complete_heldout": bool(complete_heldout),
        "complete_regression": bool(complete_regression),
        "surface_kind": (
            "complete" if complete_heldout and complete_regression else "bounded_debug_probe"
        ),
        "passed": state == "improvement",
        "failures": failures,
        "improvements": improvements,
    }
    decision["decision_id"] = canonical_sha256(decision)
    return decision


def resolve_retention_action(
    decision: Mapping[str, Any],
    *,
    probation_count: int,
    max_plateau_probation: int,
) -> str:
    """Map a v2 retention decision plus the probation budget to a guard action.

    Returns one of:
      "accept"   - promote the tentative checkpoint as the confirmed parent
      "probate"  - hold this exact optimizer state without promotion
      "rollback" - restore the confirmed parent and stop (regression, or
                   probation allowance exhausted, or structural failure)
    """
    state = str(decision.get("state", ""))
    if state == "improvement":
        return "accept"
    if state == "plateau":
        if probation_count < max_plateau_probation:
            return "probate"
        return "rollback"
    return "rollback"


def decide_foundation_motor_v2_stage(
    *,
    training_stage: str,
    heldout_probe: Mapping[str, Any] | None,
    regression_probe: Mapping[str, Any] | None,
    complete_heldout: bool,
    complete_regression: bool,
    receipt_continuation: bool = False,
    receipt_teaching_profile: str = RECEIPT_TEACHING_PROFILE_CONTINUATION_V1,
    eos_generate_head_route: bool = False,
    termination_head_route: bool = False,
) -> dict[str, Any]:
    if training_stage not in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        raise ValueError(f"unknown foundation motor v2 training stage {training_stage!r}")
    generate_head_profile = receipt_teaching_profile in {
        RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3,
        RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4,
    }
    if eos_generate_head_route != generate_head_profile:
        raise ValueError(
            "eos_generate_head_route and generate_head_eos_v3 must be selected together"
        )
    if termination_head_route != (
        receipt_teaching_profile in RECEIPT_TERMINATION_HEAD_PROFILES
    ):
        raise ValueError(
            "termination_head_route and a termination-head teaching profile "
            "must be selected together"
        )
    failures: list[str] = []
    if heldout_probe is None or regression_probe is None:
        failures.append("missing heldout or regression motor-v2 probe")
    else:
        for label, probe in (("heldout", heldout_probe), ("regression", regression_probe)):
            threshold = float(FOUNDATION_MOTOR_V2_PROGRAM["gate_threshold"])
            if probe["complete_field_coverage_rate"] != 1.0:
                failures.append(f"{label} complete-field coverage is not exact")

            def require(
                metric: str,
                *,
                observed_probe: Mapping[str, Any] = probe,
                observed_threshold: float = threshold,
                observed_label: str = label,
            ) -> None:
                if float(observed_probe[metric]) < observed_threshold:
                    failures.append(
                        f"{observed_label} {metric} below {observed_threshold}"
                    )

            def require_pair(
                component: str,
                *,
                observed_probe: Mapping[str, Any] = probe,
                observed_threshold: float = threshold,
                observed_label: str = label,
            ) -> None:
                if (
                    float(observed_probe["pair_exact_rates"][component])
                    < observed_threshold
                ):
                    failures.append(
                        f"{observed_label} changed-source {component} below "
                        f"{observed_threshold}"
                    )

            def beat_floor(
                rate_metric: str,
                floor_metric: str,
                *,
                observed_probe: Mapping[str, Any] = probe,
                observed_label: str = label,
            ) -> None:
                # An absent metric must never read as a pass.  Failing closed
                # with a message keeps the verdict honest; raising KeyError here
                # would turn a gate decision into a crash.
                if rate_metric not in observed_probe or floor_metric not in observed_probe:
                    failures.append(
                        f"{observed_label} cannot attest {rate_metric} against "
                        f"{floor_metric}: the probe does not carry it"
                    )
                    return
                if not (
                    float(observed_probe[rate_metric]) > float(observed_probe[floor_metric])
                ):
                    failures.append(
                        f"{observed_label} {rate_metric} does not beat the "
                        f"constant-answer floor {floor_metric}"
                    )

            if training_stage == "copy_alignment":
                require("alignment_position_accuracy")
                require("alignment_copy_gate_accuracy")
                # Ratified 2026-09-17 with the emission rung: this stage now
                # supervises payload content, so it must also require it.  A
                # lineage that emits nothing scores exactly the empty-payload
                # floor and can never satisfy these requirements, which is the
                # only way the next stage can inherit an emitter.
                require("payload_content_accuracy")
                require("payload_transport_exact_rate")
                require_pair("position")
                require_pair("copy_gate")
                require_pair("content")
                if not (
                    float(probe["payload_content_accuracy"])
                    > float(probe["payload_content_constant_floor"])
                ):
                    failures.append(f"{label} payload content does not beat constant floor")
                # The two exactness rates are deliberately NOT blockers here.
                # This stage weights decision/operation/region/start/end at 0.0
                # and alignment_eos_gate at 0.0, while typed_emission_exact_rate
                # demands a DELTA-phase conjunction over decision, operation,
                # region, start, end and exact free-running payload transport
                # (living_reasoning_curriculum.py:897).  Requiring it here would
                # be a gate no lineage can ever pass: the maximum reachable
                # value is 0.  The emission rung is proven non-vacuous by
                # payload_content_accuracy above, which an emit-nothing core
                # scores 0.0 on.  The floors are asserted at the stages that do
                # carry those weights (joint) and by
                # tests/test_foundation_motor_gate_reachability.py.
                if receipt_continuation:
                    require("payload_eos_accuracy")
                    if not eos_generate_head_route:
                        require("alignment_eos_gate_accuracy")
                        require_pair("eos_gate")
            elif training_stage == "transport_eos":
                required_metrics = [
                    "alignment_position_accuracy",
                    "alignment_copy_gate_accuracy",
                    "payload_content_accuracy",
                    "payload_eos_accuracy",
                    # Ratified 2026-09-17 with the emission rung: the metric that
                    # exposes the emit-nothing dead state.  Sixteen of the 24
                    # heldout transport cases require a non-empty payload, so a
                    # lineage that always stops immediately cannot reach 0.95.
                    "payload_transport_exact_rate",
                ]
                required_pairs = ["position", "copy_gate", "content"]
                if not eos_generate_head_route:
                    required_metrics.append("alignment_eos_gate_accuracy")
                    required_pairs.append("eos_gate")
                for metric in required_metrics:
                    require(metric)
                for component in required_pairs:
                    require_pair(component)
                if not (
                    float(probe["payload_content_accuracy"])
                    > float(probe["payload_content_constant_floor"])
                ):
                    failures.append(f"{label} payload content does not beat constant floor")
                # typed_emission_exact_rate is unreachable in this stage for the
                # same reason as at copy_alignment: decision, operation, region,
                # start and end all carry weight 0.0 here, and the typed
                # conjunction demands them for every DELTA phase.  It is
                # asserted at joint, the only stage that weights all of them.
            elif training_stage == "decision":
                if any(float(value) < threshold for value in probe["per_decision_accuracy"].values()):
                    failures.append(f"{label} per-decision accuracy below {threshold}")
                require_pair("decision")
            elif training_stage == "operation":
                if any(float(value) < threshold for value in probe["per_operation_accuracy"].values()):
                    failures.append(f"{label} per-operation accuracy below {threshold}")
                require_pair("operation")
            elif training_stage == "address":
                for metric in ("region_accuracy", "start_accuracy", "end_accuracy"):
                    require(metric)
                require_pair("address")
            else:
                if any(
                    float(value) < threshold
                    for value in probe["per_action_joint_exact_rate"].values()
                ):
                    failures.append(f"{label} per-action joint exact rate below {threshold}")
                require_pair("joint")
                require_pair("content")
                # joint is the only stage that weights decision, operation,
                # region, start and end (all 1.0), so it is the first stage at
                # which typed exactness is reachable at all.  Compare against
                # the strongest constant answer rather than an assumed zero.
                beat_floor(
                    "typed_emission_exact_rate", "constant_typed_emission_exact_floor"
                )
                beat_floor(
                    "payload_transport_exact_rate",
                    "constant_payload_transport_exact_floor",
                )
    if not complete_heldout:
        failures.append("heldout surface is incomplete")
    if not complete_regression:
        failures.append("regression surface is incomplete")
    body = {
        "schema": "axon-foundation-motor-v2-stage-gate-v1",
        "foundation_stage": FOUNDATION_MOTOR_V2_STAGE,
        "training_stage": training_stage,
        "scope": "curriculum_advancement_only_not_serving_or_promotion",
        "passed": not failures,
        "failures": failures,
        "heldout_probe": None if heldout_probe is None else dict(heldout_probe),
        "regression_probe": None if regression_probe is None else dict(regression_probe),
        "program_id": FOUNDATION_MOTOR_V2_PROGRAM_ID,
        "effective_objective_program_id": foundation_motor_v2_objective_program_id(
            teach_multicell_copy=receipt_continuation,
            receipt_continuation=receipt_continuation,
            receipt_teaching_profile=receipt_teaching_profile,
        ),
        "receipt_continuation": receipt_continuation,
        "eos_generate_head_route": eos_generate_head_route,
        "receipt_teaching_profile": (
            receipt_teaching_profile if receipt_continuation else None
        ),
    }
    return {**body, "decision_id": canonical_sha256(body)}


__all__ = [
    "COPY_ALIGNMENT_MULTICELL_TEACH",
    "COPY_ALIGNMENT_MULTICELL_TEACH_ID",
    "DEFAULT_FOUNDATION_MOTOR_SPLIT_COUNTS",
    "DEFAULT_FOUNDATION_MOTOR_V2_SPLIT_COUNTS",
    "FOUNDATION_MOTOR_ACTIONS",
    "FOUNDATION_MOTOR_GATE_POLICY",
    "FOUNDATION_MOTOR_GATE_POLICY_ID",
    "FOUNDATION_MOTOR_SOURCE_ID",
    "FOUNDATION_MOTOR_STAGE",
    "FOUNDATION_MOTOR_V2_METRIC_COMPONENTS",
    "FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM",
    "FOUNDATION_MOTOR_V2_MULTICELL_PROGRAM_ID",
    "FOUNDATION_MOTOR_V2_PROGRAM",
    "FOUNDATION_MOTOR_V2_PROGRAM_ID",
    "FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM",
    "FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_BALANCED_PROGRAM_ID",
    "FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM",
    "FOUNDATION_MOTOR_V2_RECEIPT_GENERATE_HEAD_EOS_PROGRAM_ID",
    "FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM",
    "FOUNDATION_MOTOR_V2_RECEIPT_PROGRAM_ID",
    "FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM",
    "FOUNDATION_MOTOR_V2_RECEIPT_ROUTE_EOS_BALANCED_PROGRAM_ID",
    "FOUNDATION_MOTOR_V2_RETENTION_CONTRACT",
    "FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_ID",
    "FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2",
    "FOUNDATION_MOTOR_V2_RETENTION_CONTRACT_V2_ID",
    "FOUNDATION_MOTOR_V2_SOURCE_ID",
    "FOUNDATION_MOTOR_V2_STAGE",
    "FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS",
    "FOUNDATION_MOTOR_V2_STAGE_ORDER",
    "FOUNDATION_MOTOR_V2_UNICODE_WALK_SOURCE_ID",
    "RECEIPT_CONTINUATION_TEACH",
    "RECEIPT_CONTINUATION_TEACH_ID",
    "RECEIPT_GENERATE_HEAD_BALANCED_TEACH",
    "RECEIPT_GENERATE_HEAD_BALANCED_TEACH_ID",
    "RECEIPT_GENERATE_HEAD_EOS_TEACH",
    "RECEIPT_GENERATE_HEAD_EOS_TEACH_ID",
    "RECEIPT_ROUTE_EOS_BALANCED_TEACH",
    "RECEIPT_ROUTE_EOS_BALANCED_TEACH_ID",
    "RECEIPT_TEACHING_PROFILES",
    "RECEIPT_TEACHING_PROFILE_CONTINUATION_V1",
    "RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_BALANCED_V4",
    "RECEIPT_TEACHING_PROFILE_GENERATE_HEAD_EOS_V3",
    "RECEIPT_TEACHING_PROFILE_ROUTE_EOS_BALANCED_V2",
    "apply_copy_alignment_multicell_teach_weights",
    "apply_receipt_continuation_teach_weights",
    "compile_foundation_motor",
    "compile_foundation_motor_v2",
    "compile_foundation_motor_v2_unicode_walk",
    "decide_foundation_motor_mastery",
    "decide_foundation_motor_v2_checkpoint_retention",
    "decide_foundation_motor_v2_checkpoint_retention_v2",
    "resolve_retention_action",
    "decide_foundation_motor_v2_stage",
    "foundation_motor_payload_transport_cells",
    "foundation_motor_probe",
    "foundation_motor_v2_action",
    "foundation_motor_v2_component_weights",
    "foundation_motor_v2_first_reachable_stage",
    "foundation_motor_v2_objective_program_id",
    "foundation_motor_v2_probe",
    "foundation_motor_v2_stage_policy",
    "foundation_motor_v2_unreachable_gate_requirements",
    "is_foundation_motor_episode",
    "is_foundation_motor_v2_episode",
    "oversample_multicell_copy_cases",
    "receipt_continuation_teach_profile",
    "verify_foundation_motor_curriculum",
    "verify_foundation_motor_v2_curriculum",
    "verify_foundation_motor_v2_unicode_walk_curriculum",
]
