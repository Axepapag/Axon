"""Foundations Stage 0: exact typed-delta motor control.

The core learns the smallest real movements of Axon's body before sequence or
language work: copy one current-field scalar, insert, replace, delete, no-op,
and abstain.  Every case traverses the real Shared Field, D64 reader, private
Soul, typed reasoning output, exact address, and EOS contracts.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

from runtime.field import D64FieldCompiler, LogicalRegion, SharedFieldSnapshot, canonical_sha256
from runtime.heart import ProposalPass, ReasoningDecision, ReasoningOperationKind
from substrate import encode_unicode_text

from .first_form_curriculum import FirstFormCase, FirstFormCurriculum, TeachingEligibility, _workspace
from .living_reasoning_curriculum import LivingReasoningEpisode, LivingReasoningTarget

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
# Teaching overlay only. Not part of FOUNDATION_MOTOR_V2_PROGRAM identity.
# Same exam, same gates, same architecture; changes optimizer pressure.
COPY_ALIGNMENT_MULTICELL_TEACH = {
    "schema": "axon-foundation-motor-copy-alignment-multicell-teach-v1",
    "alignment_position_reduction": "sum",
    "component_weight_overrides": {
        "alignment_position": 4.0,
        "alignment_copy_gate": 0.25,
    },
    "oversample_multicell": True,
}
DEFAULT_FOUNDATION_MOTOR_V2_SPLIT_COUNTS: tuple[
    tuple[str, tuple[int, int, int]], ...
] = (("F0", (72, 36, 36)),)

_SPLITS = ("train", "heldout", "regression")
_SYMBOLS = tuple(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?.:,;+-=*/_()[]{}"
    "λφΩЖя東亰終端心脳éïüñøå✓∞∑∆🙂🧠🚀"
)


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
) -> LivingReasoningEpisode:
    action = (
        FOUNDATION_MOTOR_ACTIONS[pair_index % len(FOUNDATION_MOTOR_ACTIONS)]
        if action_override is None
        else action_override
    )
    if action not in FOUNDATION_MOTOR_ACTIONS:
        raise ValueError(f"unsupported foundation motor action {action!r}")
    symbol = _symbol(split, pair_index, variant)
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
        "payload_eos_accuracy": rate(
            "payload_teacher_forced_eos_correct", "payload_teacher_forced_eos_count"
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


def decide_foundation_motor_v2_stage(
    *,
    training_stage: str,
    heldout_probe: Mapping[str, Any] | None,
    regression_probe: Mapping[str, Any] | None,
    complete_heldout: bool,
    complete_regression: bool,
) -> dict[str, Any]:
    if training_stage not in FOUNDATION_MOTOR_V2_STAGE_ORDER:
        raise ValueError(f"unknown foundation motor v2 training stage {training_stage!r}")
    failures: list[str] = []
    if heldout_probe is None or regression_probe is None:
        failures.append("missing heldout or regression motor-v2 probe")
    else:
        for label, probe in (("heldout", heldout_probe), ("regression", regression_probe)):
            threshold = float(FOUNDATION_MOTOR_V2_PROGRAM["gate_threshold"])
            if probe["complete_field_coverage_rate"] != 1.0:
                failures.append(f"{label} complete-field coverage is not exact")

            def require(metric: str) -> None:
                if float(probe[metric]) < threshold:
                    failures.append(f"{label} {metric} below {threshold}")

            def require_pair(component: str) -> None:
                if float(probe["pair_exact_rates"][component]) < threshold:
                    failures.append(f"{label} changed-source {component} below {threshold}")

            if training_stage == "copy_alignment":
                require("alignment_position_accuracy")
                require("alignment_copy_gate_accuracy")
                require_pair("position")
                require_pair("copy_gate")
            elif training_stage == "transport_eos":
                for metric in (
                    "alignment_position_accuracy",
                    "alignment_copy_gate_accuracy",
                    "alignment_eos_gate_accuracy",
                    "payload_content_accuracy",
                    "payload_eos_accuracy",
                ):
                    require(metric)
                for component in ("position", "copy_gate", "eos_gate", "content"):
                    require_pair(component)
                if not (
                    float(probe["payload_content_accuracy"])
                    > float(probe["payload_content_constant_floor"])
                ):
                    failures.append(f"{label} payload content does not beat constant floor")
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
    }
    return {**body, "decision_id": canonical_sha256(body)}


__all__ = [
    "DEFAULT_FOUNDATION_MOTOR_SPLIT_COUNTS",
    "DEFAULT_FOUNDATION_MOTOR_V2_SPLIT_COUNTS",
    "FOUNDATION_MOTOR_ACTIONS",
    "FOUNDATION_MOTOR_GATE_POLICY",
    "FOUNDATION_MOTOR_GATE_POLICY_ID",
    "FOUNDATION_MOTOR_SOURCE_ID",
    "FOUNDATION_MOTOR_STAGE",
    "COPY_ALIGNMENT_MULTICELL_TEACH",
    "FOUNDATION_MOTOR_V2_PROGRAM",
    "FOUNDATION_MOTOR_V2_PROGRAM_ID",
    "FOUNDATION_MOTOR_V2_SOURCE_ID",
    "FOUNDATION_MOTOR_V2_STAGE",
    "FOUNDATION_MOTOR_V2_STAGE_ORDER",
    "apply_copy_alignment_multicell_teach_weights",
    "compile_foundation_motor",
    "compile_foundation_motor_v2",
    "decide_foundation_motor_mastery",
    "decide_foundation_motor_v2_stage",
    "foundation_motor_probe",
    "foundation_motor_v2_action",
    "foundation_motor_v2_probe",
    "foundation_motor_payload_transport_cells",
    "foundation_motor_v2_stage_policy",
    "is_foundation_motor_episode",
    "oversample_multicell_copy_cases",
    "is_foundation_motor_v2_episode",
    "verify_foundation_motor_curriculum",
    "verify_foundation_motor_v2_curriculum",
]
