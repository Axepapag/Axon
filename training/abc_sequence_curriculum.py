"""ABC Sequence Curriculum — the first learned-sequence test for the living cores.

Teaches the core to recite the alphabet (and digits) forward, backward, and
round-trip from any starting position, at graduated difficulty.  Every case
is VERIFIED_TARGET with an exact string payload compiled through the same
D64FieldCompiler and LivingReasoningEpisode contract as FFCS/C1.

The design principle: this is the simplest possible sequence-learning task
that still exercises the full runtime contract (SharedFieldSnapshot, Soul
inhale/exhale, three-phase unroll, categorical transport emission, EOS).
No hidden targets, no conversation, no ambiguity — just ordered character
sequences that are fully present in the Cortex region.

All characters are in the native 95-character frozen alphabet; no Unicode
byte transport is needed.  The full lowercase alphabet fits in one 32-char
page.
"""
from __future__ import annotations

import hashlib
from typing import Any

from runtime.field import (
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_sha256,
)
from runtime.heart import (
    ProposalPass,
    ReasoningDecision,
    ReasoningOperationKind,
)

from .first_form_curriculum import (
    FirstFormCase,
    FirstFormCurriculum,
    _case,
    _workspace,
)
from .living_reasoning_curriculum import (
    LivingReasoningEpisode,
    LivingReasoningTarget,
)

ABC_SEQUENCE_MANIFEST_SCHEMA = "axon-first-form-curriculum-v1"

LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

DEFAULT_ABC_SPLIT_COUNTS: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("ABC", (64, 8, 8)),
)

_SPLITS = ("train", "heldout", "regression")


def _alpha_sequence(
    chars: str,
    *,
    direction: str = "forward",
    start: int = 0,
    end: int | None = None,
    step: int = 1,
) -> str:
    """Extract a subsequence from a character sequence."""
    seq = list(chars)
    if direction == "backward":
        seq = seq[::-1]
    if end is None:
        end = len(seq)
    return "".join(seq[start:end:step])


def _make_abc_case(
    *,
    label: str,
    split: str,
    identity_text: str,
    user_text: str,
    cortex_text: str,
    answer: str,
    competency: str,
    lineage_id: str,
    index: int,
    direction: str,
    difficulty: str,
) -> FirstFormCase:
    """Build one ABC sequence case in the standard runtime-faithful shape."""

    example_id = canonical_sha256(
        {"schema": "axon-abc-sequence-source-v1", "label": label, "answer": answer}
    )
    episode = _copy_episode_abc(
        label=label,
        split=split,
        identity_text=identity_text,
        user_text=user_text,
        evidence_text=cortex_text,
        answer=answer,
        source_example_id=example_id,
        tags=(
            "abc_sequence",
            f"direction_{direction}",
            f"difficulty_{difficulty}",
            competency,
        ),
    )
    return _case(
        family="ABC",
        competency=competency,
        lineage_id=lineage_id,
        source_record_ids=(),
        episode=episode,
        derived=True,
        procedural_depth=1,
    )


def _copy_episode_abc(
    *,
    label: str,
    split: str,
    identity_text: str,
    user_text: str,
    evidence_text: str,
    answer: str,
    source_example_id: str,
    tags: tuple[str, ...],
) -> LivingReasoningEpisode:
    """Build a standard three-phase episode for one ABC sequence case."""

    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: identity_text,
            LogicalRegion.USER_INPUT: user_text,
            LogicalRegion.CORTEX: evidence_text,
            LogicalRegion.SCRATCH: "",
            LogicalRegion.RESPONSE_DRAFT: "",
        }
    )
    return LivingReasoningEpisode(
        label=label,
        split=split,
        snapshot=snapshot,
        first_workspace_text=_workspace(
            ProposalPass.FIRST,
            "Locate the character sequence in the Cortex region.",
        ),
        refined_workspace_text=_workspace(
            ProposalPass.REFINED,
            "Confirm the requested direction and starting position.",
        ),
        targets=(
            LivingReasoningTarget(
                phase="first",
                decision=ReasoningDecision.NO_OP,
                supervision_weight=0.0,
            ),
            LivingReasoningTarget(
                phase="refined",
                decision=ReasoningDecision.NO_OP,
                supervision_weight=0.0,
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
        mechanism_tags=tags,
        outcome_quality="verified_derived_abc_target",
        source_example_id=source_example_id,
        target_basis=(
            "exact sequence reproduction from visible canonical evidence; "
            "the answer is the exact requested character sequence"
        ),
    )


def compile_abc_sequence(
    *,
    identity_text: str,
    requested_counts: tuple[
        tuple[str, tuple[int, int, int]], ...
    ] = DEFAULT_ABC_SPLIT_COUNTS,
) -> FirstFormCurriculum:
    """Compile the ABC sequence curriculum: forward, backward, round-trip,
    uppercase, digits, skip-pattern, and subsequence variants.

    Every case is VERIFIED_TARGET.  The Cortex always contains the full
    alphabet (the "cheat sheet"), so the core learns to *navigate* it,
    not to memorize it from nothing.  The response is the exact requested
    subsequence emitted as an exact REPLACE delta into response_draft.
    """

    if not identity_text:
        raise ValueError("ABC curriculum requires the ratified canonical Identity")
    budgets = {
        family: dict(zip(_SPLITS, counts, strict=True))
        for family, counts in requested_counts
    }
    if set(budgets) != {"ABC"}:
        raise ValueError("compile_abc_sequence requires exact ABC budgets")

    lowercase = "abcdefghijklmnopqrstuvwxyz"
    uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"

    # Build the full case specification list, then distribute across splits
    # deterministically (round-robin across variants, cycling by index).
    specs: list[dict[str, Any]] = []

    # --- Forward from 'a' (the ABC song) ---
    specs.append({
        "user": "Recite the alphabet forward from 'a'.",
        "cortex": f"alphabet: {lowercase}",
        "answer": lowercase,
        "competency": "abc_forward_full",
        "direction": "forward",
        "difficulty": "basic",
        "chars": lowercase, "start": 0, "end": None, "step": 1, "direction_op": "forward",
    })

    # --- Forward from each starting position (sampled) ---
    for start_idx in (2, 5, 8, 12, 16, 20, 24):  # c, f, i, m, q, u, y
        ch = lowercase[start_idx]
        specs.append({
            "user": f"Recite the alphabet forward from '{ch}'.",
            "cortex": f"alphabet: {lowercase}",
            "answer": lowercase[start_idx:],
            "competency": "abc_forward_from_position",
            "direction": "forward",
            "difficulty": "easy",
            "chars": lowercase, "start": start_idx, "step": 1, "direction_op": "forward",
        })

    # --- Backward from 'z' (the reversed ABC song) ---
    specs.append({
        "user": "Recite the alphabet backward from 'z'.",
        "cortex": f"alphabet: {lowercase}",
        "answer": lowercase[::-1],
        "competency": "abc_backward_full",
        "direction": "backward",
        "difficulty": "medium",
        "chars": lowercase, "start": 0, "step": 1, "direction_op": "backward",
    })

    # --- Backward from each starting position (sampled) ---
    for start_idx in (3, 7, 11, 15, 19, 23):  # d, h, l, p, t, x (reversed positions)
        lowercase[::-1]
        # In reversed sequence, starting from char at original position start_idx
        # means taking the tail of the reversed string
        len(lowercase) - 1 - start_idx
        ch = lowercase[start_idx]
        specs.append({
            "user": f"Recite the alphabet backward starting from '{ch}'.",
            "cortex": f"alphabet: {lowercase}",
            "answer": lowercase[start_idx::-1],
            "competency": "abc_backward_from_position",
            "direction": "backward",
            "difficulty": "medium",
            "chars": lowercase, "start": start_idx, "step": 1, "direction_op": "backward_from",
        })

    # --- Round-trip: forward from 'a' to 'z', then backward from 'z' to 'a' ---
    specs.append({
        "user": "Recite the alphabet forward from 'a', then backward from 'z' back to 'a'.",
        "cortex": f"alphabet: {lowercase}",
        "answer": lowercase + lowercase[::-1],
        "competency": "abc_roundtrip_full",
        "direction": "roundtrip",
        "difficulty": "hard",
        "chars": lowercase, "start": 0, "step": 1, "direction_op": "roundtrip",
    })

    # --- Round-trip from a middle position ---
    for start_idx in (3, 10, 18):  # d, k, s
        ch = lowercase[start_idx]
        forward_part = lowercase[start_idx:]
        lowercase[start_idx::-1]
        specs.append({
            "user": f"Recite the alphabet forward from '{ch}' to the end, then backward from '{ch}' back to the start.",
            "cortex": f"alphabet: {lowercase}",
            "answer": forward_part + ch + lowercase[start_idx::-1][1:],
            "competency": "abc_roundtrip_from_position",
            "direction": "roundtrip",
            "difficulty": "hard",
            "chars": lowercase, "start": start_idx, "step": 1, "direction_op": "roundtrip_from",
        })

    # --- Uppercase forward and backward ---
    specs.append({
        "user": "Recite the uppercase alphabet forward from 'A'.",
        "cortex": f"uppercase: {uppercase}",
        "answer": uppercase,
        "competency": "abc_uppercase_forward",
        "direction": "forward",
        "difficulty": "easy",
        "chars": uppercase, "start": 0, "step": 1, "direction_op": "forward",
    })
    specs.append({
        "user": "Recite the uppercase alphabet backward from 'Z'.",
        "cortex": f"uppercase: {uppercase}",
        "answer": uppercase[::-1],
        "competency": "abc_uppercase_backward",
        "direction": "backward",
        "difficulty": "medium",
        "chars": uppercase, "start": 0, "step": 1, "direction_op": "backward",
    })

    # --- Digits forward and backward ---
    specs.append({
        "user": "Recite the digits forward from '0'.",
        "cortex": f"digits: {digits}",
        "answer": digits,
        "competency": "digits_forward",
        "direction": "forward",
        "difficulty": "easy",
        "chars": digits, "start": 0, "step": 1, "direction_op": "forward",
    })
    specs.append({
        "user": "Recite the digits backward from '9'.",
        "cortex": f"digits: {digits}",
        "answer": digits[::-1],
        "competency": "digits_backward",
        "direction": "backward",
        "difficulty": "medium",
        "chars": digits, "start": 0, "step": 1, "direction_op": "backward",
    })

    # --- Skip patterns (every other letter) ---
    specs.append({
        "user": "Recite every other letter of the alphabet starting from 'a'.",
        "cortex": f"alphabet: {lowercase}",
        "answer": lowercase[::2],
        "competency": "abc_skip_pattern",
        "direction": "forward",
        "difficulty": "hard",
        "chars": lowercase, "start": 0, "step": 2, "direction_op": "forward",
    })

    # --- Subsequence extraction ---
    for lo, hi in ((3, 12), (8, 20), (0, 10), (15, 26)):
        seq = lowercase[lo:hi]
        specs.append({
            "user": f"Recite the alphabet from '{lowercase[lo]}' to '{lowercase[hi-1]}' inclusive.",
            "cortex": f"alphabet: {lowercase}",
            "answer": seq,
            "competency": "abc_subsequence",
            "direction": "forward",
            "difficulty": "medium",
            "chars": lowercase, "start": lo, "end": hi, "step": 1, "direction_op": "forward",
        })

    # --- Full uppercase+lowercase combined forward ---
    specs.append({
        "user": "Recite both alphabets: lowercase forward then uppercase forward.",
        "cortex": f"alphabet: {lowercase}\nuppercase: {uppercase}",
        "answer": lowercase + uppercase,
        "competency": "abc_both_cases_forward",
        "direction": "forward",
        "difficulty": "hard",
        "chars": lowercase + uppercase, "start": 0, "step": 1, "direction_op": "forward",
    })

    # --- Full 95-character native alphabet forward ---
    from substrate.substrate import default_alphabet
    full_bank = "".join(default_alphabet())
    specs.append({
        "user": "Recite the complete native character bank forward.",
        "cortex": f"full native alphabet ({len(full_bank)} characters): {full_bank}",
        "answer": full_bank,
        "competency": "full_bank_forward",
        "direction": "forward",
        "difficulty": "hard",
        "chars": full_bank, "start": 0, "step": 1, "direction_op": "forward",
    })

    # --- Reverse full bank ---
    specs.append({
        "user": "Recite the complete native character bank backward.",
        "cortex": f"full native alphabet ({len(full_bank)} characters): {full_bank}",
        "answer": full_bank[::-1],
        "competency": "full_bank_backward",
        "direction": "backward",
        "difficulty": "hard",
        "chars": full_bank, "start": 0, "step": 1, "direction_op": "backward",
    })

    # Now distribute specs across splits and generate cases
    cases: list[FirstFormCase] = []
    for split in _SPLITS:
        for i in range(budgets["ABC"][split]):
            # Deterministic round-robin: each case gets a different spec,
            # cycling through all specs per split
            global_index = sum(budgets["ABC"][s] for s in _SPLITS[:_SPLITS.index(split)]) + i
            spec = specs[global_index % len(specs)]
            label = f"abc-{split}-{i:03d}"
            lineage = f"abc-sequence:{split}:{global_index // 8:04d}"

            answer = spec["answer"]
            user = spec["user"]
            competency = spec["competency"]
            cortex = spec["cortex"]

            case = _make_abc_case(
                label=label,
                split=split,
                identity_text=identity_text,
                user_text=user,
                cortex_text=cortex,
                answer=answer,
                competency=competency,
                lineage_id=lineage,
                index=global_index,
                direction=spec["direction"],
                difficulty=spec["difficulty"],
            )
            cases.append(case)

    # Verify budgets
    actual = {
        split: sum(1 for c in cases if c.episode.split == split)
        for split in _SPLITS
    }
    for split in _SPLITS:
        if actual[split] != budgets["ABC"][split]:
            raise ValueError(
                f"ABC budget mismatch: requested {budgets['ABC'][split]} {split}, "
                f"got {actual[split]}"
            )

    # We don't have real D00 imports, but the compiler requires source_import_ids.
    # Use a synthetic content-addressed identity for this curriculum source.
    source_id = canonical_sha256({"schema": "axon-abc-sequence-source-v1", "compiled": True})
    return FirstFormCurriculum(
        cases=tuple(cases),
        source_import_ids=(source_id,),
        requested_family_split_counts=requested_counts,
        excluded_counts=(),
        identity_text_sha256=hashlib.sha256(identity_text.encode("utf-8")).hexdigest(),
    )


# The _copy_episode_abc function is a variant of the standard _copy_episode
# that uses our specific tags and outcome_quality.  We reuse the FFCS module's
# _case and _workspace helpers for full consistency.