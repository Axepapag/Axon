"""Exact native one-cell source-address practice before sequence copying.

The public FIRST/REFINED surface accepts exact Unicode text, so this first
pointer stage uses only native one-cell scalars.  A later complete-scalar stage
will exercise non-native UTF-8 transport without ever treating a raw byte as a
public response.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

from runtime.field import LogicalRegion, SharedFieldSnapshot
from substrate import default_alphabet

from .living_reasoning_curriculum import (
    LivingReasoningCurriculum,
    LivingReasoningEpisode,
    LivingReasoningTarget,
)

POINTER_BOOTSTRAP_SCHEMA = "axon-pointer-bootstrap-native-v1"
POINTER_BOOTSTRAP_SOURCE_ID = "axon-pointer-bootstrap-native-authored-v1"
POINTER_BOOTSTRAP_GATE_REQUIREMENTS: Mapping[str, float] = {
    "pointer_first_source_top1_rate": 1.0,
    "pointer_first_source_probability_mean": 1.0,
    "free_running_first_transport_accuracy": 1.0,
    "free_running_nonempty_valid_unicode_rate": 1.0,
    "complete_field_coverage_rate": 1.0,
}

_POSITIONS = (0, 1, 7, 15, 31, 32, 33, 47, 63)
_FILLER = tuple(character for character in default_alphabet() if character not in {" ", "\n"})


def _alignment(*, character: str, position: int) -> dict[str, Any]:
    return {
        "schema": "axon-r0-target-alignment-v1",
        "segments": [{
            "target_start": 0,
            "target_end": 1,
            "source_region": LogicalRegion.CORTEX.value,
            "source_start": position,
            "source_end": position + 1,
            "text_sha256": hashlib.sha256(character.encode("utf-8")).hexdigest(),
            "authority": "exact_designated_cortex_scalar_address",
        }],
        "supervise_eos_generate": True,
    }


def _filler(length: int, *, seed: int) -> str:
    return "".join(_FILLER[(seed + index * 17) % len(_FILLER)] for index in range(length))


def _episode(*, label: str, split: str, character: str, position: int, ordinal: int) -> LivingReasoningEpisode:
    cortex = _filler(position, seed=ordinal * 11) + character + _filler(
        67 - position, seed=ordinal * 23 + 5
    )
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: "I am Axon.",
            LogicalRegion.USER_INPUT: (
                f"Return exactly the Cortex character at canonical position {position}; stop."
            ),
            LogicalRegion.CORTEX: cortex,
            LogicalRegion.RESPONSE_DRAFT: "",
            LogicalRegion.SCRATCH: _filler((ordinal % 19) + 1, seed=ordinal * 31),
        },
        source_manifest_ids=(POINTER_BOOTSTRAP_SOURCE_ID,),
    )
    alignment = _alignment(character=character, position=position)
    return LivingReasoningEpisode(
        label=label,
        split=split,
        snapshot=snapshot,
        first_workspace_text=character,
        refined_workspace_text=character,
        targets=(
            LivingReasoningTarget("first", character, text_alignment=alignment),
            LivingReasoningTarget("refined", character, text_alignment=alignment),
            LivingReasoningTarget("consolidated", "#scratch# Pointer exercise complete.", supervision_weight=0.0),
        ),
        mechanism_tags=(
            "english", "proposal", "refinement", "tagged_final",
            "pointer_bootstrap", "native_one_cell", "exact_address", "persistent_soul",
        ),
        target_basis=POINTER_BOOTSTRAP_SCHEMA,
    )


def build_pointer_bootstrap_curriculum() -> LivingReasoningCurriculum:
    """Build disjoint exact-address native scalar practice across page edges."""

    characters = _FILLER
    edge_indices = tuple(range(len(_POSITIONS)))
    heldout_characters = tuple(characters[index] for index in range(0, len(characters), 4))
    train_characters = tuple(character for character in characters if character not in heldout_characters)
    episodes: list[LivingReasoningEpisode] = []
    for ordinal, character in enumerate(train_characters):
        for repeat in range(2):
            edge_index = (ordinal * 3 + repeat) % len(edge_indices)
            position = _POSITIONS[edge_indices[edge_index]]
            episodes.append(_episode(
                label=f"pointer-native-train-{ordinal:03d}-{repeat}", split="train",
                character=character, position=position, ordinal=ordinal * 2 + repeat,
            ))
    for ordinal, character in enumerate(heldout_characters):
        edge_index = (ordinal * 5 + 1) % len(edge_indices)
        position = _POSITIONS[edge_indices[edge_index]]
        episodes.append(_episode(
            label=f"pointer-native-heldout-{ordinal:03d}", split="heldout",
            character=character, position=position, ordinal=10_000 + ordinal,
        ))
    train = [item for item in episodes if item.split == "train"]
    heldout = [item for item in episodes if item.split == "heldout"]
    if {item.targets[0].text for item in train} & {item.targets[0].text for item in heldout}:
        raise RuntimeError("pointer bootstrap heldout target leakage")
    return LivingReasoningCurriculum(episodes=tuple(episodes))


def decide_pointer_bootstrap_mastery(report: Mapping[str, Any]) -> dict[str, Any]:
    """Require exact heldout address selection and valid free-running output."""

    failures: list[str] = []
    observed: dict[str, float | None] = {}
    for metric, threshold in POINTER_BOOTSTRAP_GATE_REQUIREMENTS.items():
        raw = report.get(metric)
        value = None if isinstance(raw, bool) or not isinstance(raw, (int, float)) else float(raw)
        if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            observed[metric] = None
            failures.append(f"{metric} is not a finite probability in [0, 1]")
        else:
            observed[metric] = value
            if value < threshold:
                failures.append(f"{metric}={value:.6f} < {threshold:.6f}")
    return {
        "schema": "axon-pointer-bootstrap-native-gate-v1",
        "passed": not failures,
        "requirements": dict(POINTER_BOOTSTRAP_GATE_REQUIREMENTS),
        "observed": observed,
        "failures": failures,
    }


__all__ = [
    "POINTER_BOOTSTRAP_GATE_REQUIREMENTS",
    "POINTER_BOOTSTRAP_SCHEMA",
    "POINTER_BOOTSTRAP_SOURCE_ID",
    "build_pointer_bootstrap_curriculum",
    "decide_pointer_bootstrap_mastery",
]
