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
) -> LivingReasoningEpisode:
    action = FOUNDATION_MOTOR_ACTIONS[pair_index % len(FOUNDATION_MOTOR_ACTIONS)]
    symbol = _symbol(split, pair_index, variant)
    source_identity = canonical_sha256(
        {
            "schema": "axon-foundation-motor-source-v1",
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
        source_manifest_ids=(FOUNDATION_MOTOR_SOURCE_ID,),
    )
    return LivingReasoningEpisode(
        label=f"foundation-motor-{split}-{action}-{pair_index:03d}-{variant}",
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
            f"foundation_stage:{FOUNDATION_MOTOR_STAGE}",
            f"foundation_motor_action:{action}",
            f"foundation_pair:{pair_id}",
            f"foundation_variant:{variant}",
            f"transfer_item:{source_identity}",
            "target_validator:exact_typed_delta",
        ),
        outcome_quality="authored_objectively_checkable_target",
        source_example_id=canonical_sha256(
            {
                "schema": "axon-foundation-motor-example-v1",
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


__all__ = [
    "DEFAULT_FOUNDATION_MOTOR_SPLIT_COUNTS",
    "FOUNDATION_MOTOR_ACTIONS",
    "FOUNDATION_MOTOR_GATE_POLICY",
    "FOUNDATION_MOTOR_GATE_POLICY_ID",
    "FOUNDATION_MOTOR_SOURCE_ID",
    "FOUNDATION_MOTOR_STAGE",
    "compile_foundation_motor",
    "decide_foundation_motor_mastery",
    "foundation_motor_probe",
    "is_foundation_motor_episode",
    "verify_foundation_motor_curriculum",
]
