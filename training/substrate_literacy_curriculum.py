"""Stage-0A exact substrate literacy for a newborn English-native Core.

The Core is not asked to reason about symbol relations yet.  It first learns
that the frozen 16D transport cells, lifted deterministically onto its D64
rail, have stable symbol identity, order, and exact variable-length sequence
transport.  FIRST and REFINED practice lossless Cortex -> English transport
while the normal three-pass runtime/Soul loop remains intact.  Consolidated
output is deliberately unsupervised at this stage.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, Mapping

from runtime.field import LogicalRegion, SharedFieldSnapshot
from substrate import default_alphabet

from .living_reasoning_curriculum import (
    LivingReasoningCurriculum,
    LivingReasoningEpisode,
    LivingReasoningTarget,
)

SUBSTRATE_LITERACY_SCHEMA = "axon-substrate-literacy-curriculum-v2"
SUBSTRATE_LITERACY_SOURCE_ID = "axon-substrate-literacy-authored-v2"
SUBSTRATE_LITERACY_GATE_REQUIREMENTS: Mapping[str, float] = {
    "text_exact_rate": 1.0,
    "text_teacher_forced_content_accuracy": 1.0,
    "text_teacher_forced_eos_accuracy": 1.0,
    "complete_field_coverage_rate": 1.0,
}

_COPY_PROMPT = (
    "Copy the complete Cortex text exactly. Preserve every substrate symbol, "
    "order, case, whitespace, punctuation, and Unicode; then stop."
)


def _alignment(text: str) -> dict[str, Any]:
    return {
        "schema": "axon-r0-target-alignment-v1",
        "segments": [
            {
                "target_start": 0,
                "target_end": len(text),
                "source_region": LogicalRegion.CORTEX.value,
                "source_start": 0,
                "source_end": len(text),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "authority": "exact_current_shared_field",
            }
        ],
        "supervise_eos_generate": True,
    }


def _copy_episode(*, label: str, split: str, text: str) -> LivingReasoningEpisode:
    if not text:
        raise ValueError("substrate copy text must be nonempty")
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: "I am Axon.",
            LogicalRegion.USER_INPUT: _COPY_PROMPT,
            LogicalRegion.CORTEX: text,
            LogicalRegion.RESPONSE_DRAFT: "",
            LogicalRegion.SCRATCH: "",
        },
        source_manifest_ids=(SUBSTRATE_LITERACY_SOURCE_ID,),
    )
    alignment = _alignment(text)
    return LivingReasoningEpisode(
        label=label,
        split=split,
        snapshot=snapshot,
        first_workspace_text=text,
        refined_workspace_text=text,
        targets=(
            LivingReasoningTarget(
                phase="first",
                text=text,
                text_alignment=alignment,
                supervision_weight=1.0,
            ),
            LivingReasoningTarget(
                phase="refined",
                text=text,
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
            "substrate_exact_copy",
            "persistent_soul",
        ),
        target_basis=SUBSTRATE_LITERACY_SCHEMA,
    )


def _native_identity_episodes() -> list[LivingReasoningEpisode]:
    episodes: list[LivingReasoningEpisode] = []
    for index, character in enumerate(default_alphabet()):
        # Proposal text must be substantively nonempty, so teach whitespace in
        # a minimal exact context instead of pretending a blank proposal is a
        # successful reasoning emission.
        if character == " ":
            text = "A A"
        elif character == "\n":
            text = "A\nB"
        else:
            text = character
        episodes.append(
            _copy_episode(
                label=f"substrate-copy-native-{index:03d}-u{ord(character):04x}",
                split="train",
                text=text,
            )
        )
    return episodes


def _native_sequences(
    *,
    count: int,
    split: str,
    label_prefix: str,
    offset: int,
    strides: tuple[int, ...],
    lengths: tuple[int, ...],
) -> list[LivingReasoningEpisode]:
    alphabet = tuple(default_alphabet())
    sequences: list[str] = []
    seen: set[str] = set()
    candidate = 0
    while len(sequences) < count:
        length = lengths[candidate % len(lengths)]
        stride = strides[(candidate // len(lengths)) % len(strides)]
        start = (offset + candidate * 17) % len(alphabet)
        value = "".join(
            alphabet[(start + step * stride) % len(alphabet)] for step in range(length)
        )
        candidate += 1
        if value in seen:
            continue
        seen.add(value)
        sequences.append(value)
    return [
        _copy_episode(
            label=f"{label_prefix}-{index:03d}",
            split=split,
            text=value,
        )
        for index, value in enumerate(sequences)
    ]


def _unicode_train_episodes() -> list[LivingReasoningEpisode]:
    samples = (
        "lambda \u03bb",
        "brain \U0001f9e0",
        "\u03bb\U0001f9e0",
        "na\u00efve caf\u00e9",
        "\u4e16\u754c Axon",
        "\u0394=\u03bb+1",
        "A\u0301 B\u0308",
        "\U0001f680 64D",
        "line1\nline2",
        "tabs\tstay\texact",
        "#scratch# text",
        "<tag>#value#",
    )
    return [
        _copy_episode(
            label=f"substrate-copy-unicode-train-{index:02d}",
            split="train",
            text=value,
        )
        for index, value in enumerate(samples)
    ]


def _unicode_heldout_episodes() -> list[LivingReasoningEpisode]:
    samples = (
        "zA7$\U0001f9e0",
        "Q0\u03bb\u03a9\U0001f680",
        "caf\u00e9 \u4e16\u754c",
        "Axon\n\u03bb\U0001f9e0",
        "\U0001f642\U0001f643\U0001f642\U0001f643",
        "a1B2c3D4\u03bb",
        "{\u03bb}[\U0001f9e0] 19",
        "<final>#\u4e16\u754c#",
    )
    return [
        _copy_episode(
            label=f"substrate-copy-unicode-heldout-{index:02d}",
            split="heldout",
            text=value,
        )
        for index, value in enumerate(samples)
    ]


def _stable_mix(episodes: Iterable[LivingReasoningEpisode]) -> list[LivingReasoningEpisode]:
    """Deterministically mix lengths/symbol families instead of training in blocks."""

    return sorted(
        episodes,
        key=lambda item: hashlib.sha256(
            f"substrate-v2-train-order:{item.label}".encode("utf-8")
        ).hexdigest(),
    )


def build_substrate_literacy_curriculum() -> LivingReasoningCurriculum:
    """Teach exact symbol/sequence transport before any before/after reasoning."""

    train = _stable_mix(
        [
            *_native_identity_episodes(),
            *_native_sequences(
                count=160,
                split="train",
                label_prefix="substrate-copy-native-sequence-train",
                offset=3,
                strides=(1, 3, 7, 11),
                lengths=(2, 3, 4, 6, 8, 12, 16),
            ),
            *_unicode_train_episodes(),
        ]
    )
    heldout = [
        *_native_sequences(
            count=24,
            split="heldout",
            label_prefix="substrate-copy-native-sequence-heldout",
            offset=41,
            strides=(13, 17),
            lengths=(5, 7, 9, 13, 15),
        ),
        *_unicode_heldout_episodes(),
    ]
    train_text = {item.targets[0].text for item in train}
    heldout_text = {item.targets[0].text for item in heldout}
    overlap = train_text & heldout_text
    if overlap:
        raise RuntimeError(f"substrate heldout exact-text leakage: {sorted(overlap)!r}")
    return LivingReasoningCurriculum(episodes=tuple([*train, *heldout]))


def decide_substrate_literacy_mastery(report: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on exact substrate reading without requiring FINAL mastery yet."""

    failures: list[str] = []
    observed: dict[str, float | None] = {}
    for metric, threshold in SUBSTRATE_LITERACY_GATE_REQUIREMENTS.items():
        raw_value = report.get(metric)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            observed[metric] = None
            failures.append(f"{metric} is not a numeric probability")
            continue
        value = float(raw_value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            observed[metric] = None
            failures.append(f"{metric} is not a finite probability in [0, 1]")
            continue
        observed[metric] = value
        if value < threshold:
            failures.append(f"{metric}={value:.6f} < {threshold:.6f}")
    exact = observed.get("text_exact_rate")
    raw_floor = report.get("constant_text_exact_floor")
    if isinstance(raw_floor, bool) or not isinstance(raw_floor, (int, float)):
        floor: float | None = None
        failures.append("constant_text_exact_floor is not a numeric probability")
    else:
        candidate_floor = float(raw_floor)
        if not math.isfinite(candidate_floor) or not 0.0 <= candidate_floor <= 1.0:
            floor = None
            failures.append("constant_text_exact_floor is not a finite probability in [0, 1]")
        else:
            floor = candidate_floor
    if exact is not None and floor is not None and exact <= floor:
        failures.append(f"text_exact_rate={exact:.6f} did not beat constant floor {floor:.6f}")
    return {
        "schema": "axon-substrate-literacy-gate-v2",
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
