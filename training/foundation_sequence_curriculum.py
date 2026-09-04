"""Foundations Stage 1: split-disjoint symbol and sequence control.

The frozen substrate already represents characters exactly.  This curriculum
therefore teaches a living D64 core to *use* those symbols through the real
Shared Field -> private Soul -> typed response-delta path.  Every evaluation
example is paired with a changed source under the same operation.  A core must
answer both variants exactly; memorizing the ABC sequence cannot pass that
probe.

Case counts are revisable campaign budgets.  They select complete cases and
never limit field, output, curriculum, or tissue capacity.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from runtime.field import D64FieldCompiler, LogicalRegion, SharedFieldSnapshot, canonical_sha256
from runtime.heart import ProposalPass, ReasoningDecision, ReasoningOperationKind

from .first_form_curriculum import (
    FirstFormCase,
    FirstFormCurriculum,
    TeachingEligibility,
    _workspace,
)
from .living_reasoning_curriculum import LivingReasoningEpisode, LivingReasoningTarget

FOUNDATION_SEQUENCE_STAGE = "sequence_transport_v1"
FOUNDATION_SEQUENCE_SOURCE_ID = canonical_sha256(
    {"schema": "axon-foundation-sequence-authored-generator-v1"}
)
FOUNDATION_SEQUENCE_GATE_POLICY = {
    "schema": "axon-foundation-stage-gate-policy-v1",
    "stage": FOUNDATION_SEQUENCE_STAGE,
    "scope": "curriculum_advancement_only_not_serving_or_promotion",
    "free_running_case_exact_rate": 0.95,
    "changed_source_pair_exact_rate": 0.95,
    "complete_field_coverage_rate": 1.0,
    "teacher_forced_must_exceed_constant_floor": True,
    "capacity_law": (
        "competency gate only; failure pauses for diagnosis or a renewable tranche "
        "and never limits tissue, field, output, curriculum, or lifetime steps"
    ),
}
FOUNDATION_SEQUENCE_GATE_POLICY_ID = canonical_sha256(FOUNDATION_SEQUENCE_GATE_POLICY)
DEFAULT_FOUNDATION_SEQUENCE_SPLIT_COUNTS: tuple[
    tuple[str, tuple[int, int, int]], ...
] = (("ABC", (64, 16, 16)),)

_SPLITS = ("train", "heldout", "regression")
_NATIVE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?.:,;+-=*/_()[]{}"
_UNICODE = tuple("λφΩЖя東亰終端心脳éïüñøå✓∞∑∆🙂🧠🚀")
_OPERATIONS = ("forward", "reverse", "every_other", "middle_span")


def _generated_source(split: str, pair_index: int, variant: int) -> str:
    """Return a deterministic source that never crosses split or pair identity."""

    if split == "train" and pair_index == 0:
        return (
            "abcdefghijklmnopqrstuvwxyz"
            if variant == 0
            else "qwertyuiopasdfghjklzxcvbnm"
        )
    if split == "train" and pair_index == 1:
        return (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if variant == 0
            else "MNBVCXZLKJHGFDSAPOIUYTREWQ"
        )
    if split == "train" and pair_index == 2:
        return "0123456789" if variant == 0 else "2419068357"

    seed_text = f"{FOUNDATION_SEQUENCE_STAGE}:{split}:{pair_index}:{variant}"
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    include_unicode = (pair_index + variant) % 3 == 0
    population: Sequence[str] = tuple(_NATIVE) + (_UNICODE if include_unicode else ())
    length = 5 + seed % 20
    prefix = {"train": "T", "heldout": "H", "regression": "R"}[split]
    body = "".join(rng.choice(population) for _ in range(length))
    return f"{prefix}{pair_index:02x}{variant}{body}"


def _operation_target(source: str, operation: str) -> tuple[str, str]:
    if operation == "forward":
        return "Return the displayed sequence from first character to last.", source
    if operation == "reverse":
        return "Return the displayed sequence from last character to first.", source[::-1]
    if operation == "every_other":
        return "Return characters at zero-based positions 0, 2, 4, and so on.", source[::2]
    if operation == "middle_span":
        start = max(1, len(source) // 4)
        end = max(start + 1, len(source) - start)
        return (
            f"Return the characters at zero-based positions {start} through {end - 1}, inclusive.",
            source[start:end],
        )
    raise ValueError(f"unsupported foundation sequence operation: {operation}")


def _episode(
    *,
    identity_text: str,
    split: str,
    pair_id: str,
    variant: int,
    operation: str,
    source: str,
) -> LivingReasoningEpisode:
    prompt, answer = _operation_target(source, operation)
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: identity_text,
            LogicalRegion.USER_INPUT: prompt,
            LogicalRegion.CORTEX: f"SOURCE_SEQUENCE[{source_hash[:12]}]={source}",
            LogicalRegion.SCRATCH: "",
            LogicalRegion.RESPONSE_DRAFT: "",
        },
        source_manifest_ids=(FOUNDATION_SEQUENCE_SOURCE_ID,),
    )
    return LivingReasoningEpisode(
        label=f"foundation-sequence-{split}-{pair_id[-4:]}-{variant}",
        split=split,
        snapshot=snapshot,
        first_workspace_text=_workspace(
            ProposalPass.FIRST,
            "Locate the complete current SOURCE_SEQUENCE and the requested operation.",
        ),
        refined_workspace_text=_workspace(
            ProposalPass.REFINED,
            "Recheck order, exact characters, response-draft authority, and EOS.",
        ),
        targets=(
            LivingReasoningTarget(
                phase="first", decision=ReasoningDecision.NO_OP, supervision_weight=0.0
            ),
            LivingReasoningTarget(
                phase="refined", decision=ReasoningDecision.NO_OP, supervision_weight=0.0
            ),
            LivingReasoningTarget(
                phase="consolidated",
                decision=ReasoningDecision.DELTA,
                operation=ReasoningOperationKind.REPLACE,
                region=LogicalRegion.RESPONSE_DRAFT,
                start=0,
                end=0,
                payload=answer,
            ),
        ),
        mechanism_tags=(
            "foundation_sequence",
            f"foundation_stage:{FOUNDATION_SEQUENCE_STAGE}",
            f"foundation_operation:{operation}",
            f"foundation_pair:{pair_id}",
            f"foundation_variant:{variant}",
            f"transfer_item:{source_hash}",
            "target_validator:exact_transport",
            "visibility:visible_by_design",
        ),
        outcome_quality="authored_objectively_checkable_target",
        source_example_id=canonical_sha256(
            {
                "schema": "axon-foundation-sequence-example-v1",
                "split": split,
                "pair_id": pair_id,
                "variant": variant,
                "operation": operation,
                "source_sha256": source_hash,
                "target": answer,
            }
        ),
        target_basis=(
            "deterministic transformation of the exact current SOURCE_SEQUENCE; "
            "paired changed-source variant must also be exact"
        ),
    )


def compile_foundation_sequence(
    *,
    identity_text: str,
    requested_counts: tuple[
        tuple[str, tuple[int, int, int]], ...
    ] = DEFAULT_FOUNDATION_SEQUENCE_SPLIT_COUNTS,
) -> FirstFormCurriculum:
    """Compile a Stage-1 manifest with whole, split-disjoint source pairs."""

    if not identity_text:
        raise ValueError("foundation sequence curriculum requires canonical Identity")
    if tuple(family for family, _counts in requested_counts) != ("ABC",):
        raise ValueError("foundation sequence curriculum uses the registered ABC family")
    counts = tuple(int(value) for value in requested_counts[0][1])
    if len(counts) != 3 or any(value < 2 or value % 2 for value in counts):
        raise ValueError("each foundation sequence split count must be a positive even number")

    cases: list[FirstFormCase] = []
    for split, count in zip(_SPLITS, counts, strict=True):
        for pair_index in range(count // 2):
            operation = _OPERATIONS[pair_index % len(_OPERATIONS)]
            pair_id = canonical_sha256(
                {
                    "schema": "axon-foundation-sequence-pair-v1",
                    "split": split,
                    "pair_index": pair_index,
                    "operation": operation,
                }
            )
            for variant in (0, 1):
                source = _generated_source(split, pair_index, variant)
                episode = _episode(
                    identity_text=identity_text,
                    split=split,
                    pair_id=pair_id,
                    variant=variant,
                    operation=operation,
                    source=source,
                )
                compiled = D64FieldCompiler().compile(episode.snapshot)
                compiled.verify_roundtrip(episode.snapshot)
                cases.append(
                    FirstFormCase(
                        family="ABC",
                        competency=f"foundation_sequence_{operation}",
                        eligibility=TeachingEligibility.VERIFIED_TARGET,
                        lineage_id=f"foundation-sequence-v1:{split}:{pair_index:04d}",
                        source_record_ids=(),
                        episode=episode,
                        transport_pages=len(tuple(compiled.iter_character_pages(32))),
                        procedural_depth=(2 if operation != "forward" else 1),
                        derived=True,
                    )
                )

    curriculum = FirstFormCurriculum(
        cases=tuple(cases),
        source_import_ids=(FOUNDATION_SEQUENCE_SOURCE_ID,),
        requested_family_split_counts=requested_counts,
        excluded_counts=(),
        identity_text_sha256=hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
    )
    verify_foundation_sequence_curriculum(curriculum)
    return curriculum


def _tag_value(episode: LivingReasoningEpisode, prefix: str) -> str | None:
    matches = tuple(tag[len(prefix) :] for tag in episode.mechanism_tags if tag.startswith(prefix))
    if len(matches) > 1:
        raise ValueError(f"foundation episode repeats tag prefix {prefix!r}")
    return matches[0] if matches else None


def is_foundation_sequence_episode(episode: LivingReasoningEpisode) -> bool:
    return f"foundation_stage:{FOUNDATION_SEQUENCE_STAGE}" in episode.mechanism_tags


def verify_foundation_sequence_curriculum(curriculum: FirstFormCurriculum) -> None:
    """Fail closed on leakage, incomplete pairs, or malformed stage metadata."""

    if not curriculum.cases:
        raise ValueError("foundation sequence curriculum is empty")
    split_sources: dict[str, set[str]] = defaultdict(set)
    pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for case in curriculum.cases:
        if case.family != "ABC" or not is_foundation_sequence_episode(case.episode):
            raise ValueError("foundation sequence manifest contains a foreign case")
        pair_id = _tag_value(case.episode, "foundation_pair:")
        variant = _tag_value(case.episode, "foundation_variant:")
        transfer = _tag_value(case.episode, "transfer_item:")
        if pair_id is None or variant not in {"0", "1"} or transfer is None:
            raise ValueError("foundation sequence case lacks pair/variant/transfer identity")
        split = case.episode.split
        if transfer in split_sources[split]:
            raise ValueError("foundation sequence source repeats within a split")
        split_sources[split].add(transfer)
        pairs[(split, pair_id)].append(variant)
    for split_a in _SPLITS:
        for split_b in _SPLITS:
            if split_a < split_b and not split_sources[split_a].isdisjoint(split_sources[split_b]):
                raise ValueError("foundation sequence source crosses data splits")
    if any(sorted(variants) != ["0", "1"] for variants in pairs.values()):
        raise ValueError("every foundation source-change pair requires variants zero and one")


def foundation_sequence_probe(
    episodes: Sequence[LivingReasoningEpisode],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Measure paired free-running behavior from aligned evaluation rows."""

    if len(episodes) != len(rows):
        raise ValueError("foundation probe episodes and evaluation rows must align")
    paired: dict[str, list[bool]] = defaultdict(list)
    selected = 0
    exact_cases = 0
    complete_outputs = 0.0
    phase_outputs = 0.0
    for episode, row in zip(episodes, rows, strict=True):
        if not is_foundation_sequence_episode(episode):
            continue
        pair_id = _tag_value(episode, "foundation_pair:")
        if pair_id is None:
            raise ValueError("foundation evaluation episode lacks pair identity")
        selected += 1
        exact = (
            row["typed_emission_exact_count"] == row["supervised_phase_count"]
            and row["payload_transport_exact_count"]
            == row["payload_supervised_phase_count"]
        )
        exact_cases += int(exact)
        paired[pair_id].append(bool(exact))
        complete_outputs += float(row["complete_field_coverage_count"])
        phase_outputs += float(row["phase_output_count"])
    if not selected:
        return None
    if any(len(values) != 2 for values in paired.values()):
        raise ValueError("foundation evaluation split contains an incomplete source-change pair")
    paired_exact = sum(all(values) for values in paired.values())
    return {
        "schema": "axon-foundation-sequence-probe-v1",
        "stage": FOUNDATION_SEQUENCE_STAGE,
        "case_count": selected,
        "pair_count": len(paired),
        "free_running_case_exact_rate": exact_cases / selected,
        "changed_source_pair_exact_rate": paired_exact / len(paired),
        "complete_field_coverage_rate": complete_outputs / max(1.0, phase_outputs),
    }


def decide_foundation_sequence_mastery(
    *,
    heldout_probe: Mapping[str, Any] | None,
    regression_probe: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    complete_heldout: bool,
    complete_regression: bool,
) -> dict[str, Any] | None:
    """Return an explicit stage-advancement decision, never a serving decision."""

    if heldout_probe is None and regression_probe is None:
        return None
    failures: list[str] = []
    if heldout_probe is None or regression_probe is None:
        failures.append("missing heldout or regression foundation probe")
    else:
        requirements = (
            (heldout_probe["free_running_case_exact_rate"] >= 0.95, "heldout exact rate below 0.95"),
            (
                heldout_probe["changed_source_pair_exact_rate"] >= 0.95,
                "heldout changed-source pair exact rate below 0.95",
            ),
            (
                regression_probe["free_running_case_exact_rate"] >= 0.95,
                "regression retention exact rate below 0.95",
            ),
            (
                regression_probe["changed_source_pair_exact_rate"] >= 0.95,
                "regression changed-source pair exact rate below 0.95",
            ),
            (
                heldout_probe["complete_field_coverage_rate"] == 1.0
                and regression_probe["complete_field_coverage_rate"] == 1.0,
                "complete-field coverage is not exact",
            ),
        )
        failures.extend(message for passed, message in requirements if not passed)
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
        "stage": FOUNDATION_SEQUENCE_STAGE,
        "scope": "curriculum_advancement_only_not_serving_or_promotion",
        "passed": not failures,
        "failures": failures,
        "heldout_probe": None if heldout_probe is None else dict(heldout_probe),
        "regression_probe": None if regression_probe is None else dict(regression_probe),
        "policy_id": FOUNDATION_SEQUENCE_GATE_POLICY_ID,
        "policy": FOUNDATION_SEQUENCE_GATE_POLICY,
    }
    return {**body, "decision_id": canonical_sha256(body)}


__all__ = [
    "DEFAULT_FOUNDATION_SEQUENCE_SPLIT_COUNTS",
    "FOUNDATION_SEQUENCE_GATE_POLICY",
    "FOUNDATION_SEQUENCE_GATE_POLICY_ID",
    "FOUNDATION_SEQUENCE_SOURCE_ID",
    "FOUNDATION_SEQUENCE_STAGE",
    "compile_foundation_sequence",
    "decide_foundation_sequence_mastery",
    "foundation_sequence_probe",
    "is_foundation_sequence_episode",
    "verify_foundation_sequence_curriculum",
]
