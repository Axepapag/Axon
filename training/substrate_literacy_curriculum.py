"""Stage-0 substrate literacy for a newborn English-native reasoning Core.

The Core is not asked for opinions yet.  It learns that the frozen 16D
transport cells, lifted deterministically onto its D64 rail, have stable symbol
identity and order.  FIRST and REFINED therefore practice exact substrate
reading while the normal three-pass runtime/Soul loop remains intact.
Consolidated output is deliberately unsupervised at this stage.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from runtime.field import LogicalRegion, SharedFieldSnapshot
from substrate import default_alphabet

from .living_reasoning_curriculum import (
    LivingReasoningCurriculum,
    LivingReasoningEpisode,
    LivingReasoningTarget,
)

SUBSTRATE_LITERACY_SCHEMA = "axon-substrate-literacy-curriculum-v1"
SUBSTRATE_LITERACY_SOURCE_ID = "axon-substrate-literacy-authored-v1"
SUBSTRATE_LITERACY_GATE_REQUIREMENTS: Mapping[str, float] = {
    "text_exact_rate": 1.0,
    "text_teacher_forced_content_accuracy": 1.0,
    "text_teacher_forced_eos_accuracy": 1.0,
    "complete_field_coverage_rate": 1.0,
}


def _alignment(text: str, *, source_start: int = 0) -> dict[str, Any]:
    return {
        "schema": "axon-r0-target-alignment-v1",
        "segments": [
            {
                "target_start": 0,
                "target_end": len(text),
                "source_region": LogicalRegion.CORTEX.value,
                "source_start": source_start,
                "source_end": source_start + len(text),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "authority": "exact_current_shared_field",
            }
        ],
        "supervise_eos_generate": True,
    }


def _episode(
    *,
    label: str,
    split: str,
    source: str,
    target: str,
    source_start: int = 0,
    prompt: str = "Return the exact requested substrate text from Cortex.",
) -> LivingReasoningEpisode:
    if source[source_start : source_start + len(target)] != target:
        raise ValueError("substrate target must be an exact contiguous source fragment")
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: "I am Axon.",
            LogicalRegion.USER_INPUT: prompt,
            LogicalRegion.CORTEX: source,
            LogicalRegion.RESPONSE_DRAFT: "",
            LogicalRegion.SCRATCH: "",
        },
        source_manifest_ids=(SUBSTRATE_LITERACY_SOURCE_ID,),
    )
    alignment = _alignment(target, source_start=source_start)
    return LivingReasoningEpisode(
        label=label,
        split=split,
        snapshot=snapshot,
        first_workspace_text=target,
        refined_workspace_text=target,
        targets=(
            LivingReasoningTarget(
                phase="first",
                text=target,
                text_alignment=alignment,
                supervision_weight=1.0,
            ),
            LivingReasoningTarget(
                phase="refined",
                text=target,
                text_alignment=alignment,
                supervision_weight=1.0,
            ),
            LivingReasoningTarget(
                phase="consolidated",
                text="#scratch# Substrate exercise complete.",
                supervision_weight=0.0,
            ),
        ),
        mechanism_tags=(
            "english",
            "proposal",
            "refinement",
            "tagged_final",
            "unicode",
            "substrate",
            "exact_copy",
            "persistent_soul",
        ),
        target_basis=SUBSTRATE_LITERACY_SCHEMA,
    )


def _native_identity_episodes() -> list[LivingReasoningEpisode]:
    episodes: list[LivingReasoningEpisode] = []
    for index, character in enumerate(default_alphabet()):
        if character == " ":
            source = target = "A A"
            suffix = "space"
        elif character == "\n":
            source = target = "A\nB"
            suffix = "newline"
        else:
            source = target = character
            suffix = f"u{ord(character):04x}"
        episodes.append(
            _episode(
                label=f"substrate-native-{index:03d}-{suffix}",
                split="train",
                source=source,
                target=target,
                prompt="Read Cortex exactly. Preserve every substrate symbol and stop after the sample.",
            )
        )
    return episodes


def _relation_episodes() -> list[LivingReasoningEpisode]:
    episodes: list[LivingReasoningEpisode] = []
    families = (
        ("lower", "abcdefghijklmnopqrstuvwxyz"),
        ("upper", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        ("digit", "0123456789"),
    )
    for family, source in families:
        for index in range(len(source) - 1):
            target = source[index + 1]
            episodes.append(
                _episode(
                    label=f"substrate-after-{family}-{index:02d}",
                    split="train",
                    source=source,
                    target=target,
                    source_start=index + 1,
                    prompt=f"In the ordered {family} sequence in Cortex, return the symbol immediately after position {index}.",
                )
            )
            target = source[index]
            episodes.append(
                _episode(
                    label=f"substrate-before-{family}-{index + 1:02d}",
                    split="train",
                    source=source,
                    target=target,
                    source_start=index,
                    prompt=f"In the ordered {family} sequence in Cortex, return the symbol immediately before position {index + 1}.",
                )
            )
    return episodes


def _sequence_episodes() -> list[LivingReasoningEpisode]:
    samples = (
        "abcXYZ09",
        "Aa0 Zz9",
        "()[]{}<>",
        "+-=*/%",
        "#:@&_~^$",
        "hello world.",
        "Axon 64D!",
        "a\nb c",
        "lambda ?",
        "brain ??",
        "???",
        "na?ve caf?",
        "? ? ? ? ?",
        "??????",
        "A?9$??Z",
    )
    return [
        _episode(
            label=f"substrate-sequence-{index:02d}",
            split="train",
            source=sample,
            target=sample,
            prompt="Return the exact Cortex sequence. Preserve order, case, spaces, punctuation, and Unicode.",
        )
        for index, sample in enumerate(samples)
    ]


def _heldout_episodes() -> list[LivingReasoningEpisode]:
    samples = (
        "zA7$?",
        "Q0???",
        "{}[] 19",
        "Axon\n???",
        "caf? ?",
        "????",
        "a1B2c3D4",
        "<tag>#value#",
    )
    episodes = [
        _episode(
            label=f"substrate-heldout-sequence-{index:02d}",
            split="heldout",
            source=sample,
            target=sample,
            prompt="Return the exact unseen Cortex sequence with no normalization.",
        )
        for index, sample in enumerate(samples)
    ]
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for index in (2, 7, 13, 20):
        episodes.append(
            _episode(
                label=f"substrate-heldout-relation-{index:02d}",
                split="heldout",
                source=alphabet,
                target=alphabet[index + 1],
                source_start=index + 1,
                prompt=f"Read the lowercase sequence in Cortex and return the character one place to the right of index {index}.",
            )
        )
    return episodes


def build_substrate_literacy_curriculum() -> LivingReasoningCurriculum:
    """Complete first-stage symbol/sequence literacy with held-out compositions."""

    return LivingReasoningCurriculum(
        episodes=tuple(
            [
                *_native_identity_episodes(),
                *_relation_episodes(),
                *_sequence_episodes(),
                *_heldout_episodes(),
            ]
        )
    )


def decide_substrate_literacy_mastery(report: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on exact substrate reading without requiring FINAL mastery yet."""

    failures: list[str] = []
    observed: dict[str, float] = {}
    for metric, threshold in SUBSTRATE_LITERACY_GATE_REQUIREMENTS.items():
        value = float(report.get(metric, 0.0))
        observed[metric] = value
        if value < threshold:
            failures.append(f"{metric}={value:.6f} < {threshold:.6f}")
    exact = observed.get("text_exact_rate", 0.0)
    floor = float(report.get("constant_text_exact_floor", 1.0))
    if exact <= floor:
        failures.append(f"text_exact_rate={exact:.6f} did not beat constant floor {floor:.6f}")
    return {
        "schema": "axon-substrate-literacy-gate-v1",
        "passed": not failures,
        "requirements": dict(SUBSTRATE_LITERACY_GATE_REQUIREMENTS),
        "observed": observed,
        "constant_text_exact_floor": floor,
        "failures": failures,
    }


__all__ = [
    "SUBSTRATE_LITERACY_GATE_REQUIREMENTS",
    "SUBSTRATE_LITERACY_SCHEMA",
    "SUBSTRATE_LITERACY_SOURCE_ID",
    "build_substrate_literacy_curriculum",
    "decide_substrate_literacy_mastery",
]
