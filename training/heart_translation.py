"""Provenance-bound curriculum, loss, and evaluation for Heart translation tissue.

The first curriculum is deliberately controlled and synthetic.  It is not claimed
as natural-language mastery.  It exercises the permanent Heart model path across
multiple dialect renderings while making polarity, modality, quantification,
time, causality, speech act, referent identity, and grounding explicitly
measurable.  Every case is content-addressed and labeled synthetic in provenance.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from runtime.field import LogicalRegion, SharedFieldSnapshot, canonical_sha256
from runtime.heart.d64_codec import D64HeartCodec
from runtime.heart.intelligence import (
    CRITICAL_SEMANTIC_CLASSES,
    HeartSemanticFidelityEvidence,
)
from runtime.heart.translation_core import HEART_SEMANTIC_LABELS, HeartTranslationCore
from substrate import assert_supported_text, default_alphabet

HEART_CURRICULUM_SCHEMA = "axon-heart-translation-curriculum-v3"
HEART_CASE_SCHEMA = "axon-heart-translation-case-v3"
HEART_EVALUATION_SCHEMA = "axon-heart-translation-evaluation-v4"
HEART_DECODER_DIAGNOSTIC_SCHEMA = "axon-heart-decoder-diagnostic-v2"
HEART_DECODER_CASE_DIAGNOSTIC_SCHEMA = "axon-heart-decoder-case-diagnostic-v1"
HEART_DECODER_MECHANISM_CURRICULUM_SCHEMA = "axon-heart-decoder-mechanism-curriculum-v1"
HEART_DECODER_GENERALIZATION_CURRICULUM_SCHEMA = "axon-heart-decoder-generalization-curriculum-v1"
HEART_DECODER_GENERALIZATION_OBJECTIVE_SCHEMA = "axon-heart-decoder-generalization-objective-v2"
HEART_DECODER_GENERALIZATION_EVIDENCE_SCHEMA = "axon-heart-decoder-generalization-evidence-v1"
HEART_DECODER_COUNTERFACTUAL_SCHEMA = "axon-heart-decoder-counterfactual-pair-v1"
HEART_TRAINING_OBJECTIVE_SCHEMA = "axon-heart-translation-training-objective-v1"
HEART_TRAINING_RECIPE_SCHEMA = "axon-heart-translation-training-recipe-v1"

DIALECTS: tuple[str, ...] = (
    "canonical_english_v1",
    "heart_explicit_v1",
    "structured_proposition_v1",
)
DIALECT_TO_ID = {name: index for index, name in enumerate(DIALECTS)}
TRAIN_DIRECTIONS: tuple[tuple[str, str], ...] = tuple(
    (left, right) for left in DIALECTS for right in DIALECTS if left != right
)
EVAL_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("canonical_english_v1", "heart_explicit_v1"),
    ("heart_explicit_v1", "structured_proposition_v1"),
    ("structured_proposition_v1", "canonical_english_v1"),
)


@dataclass(frozen=True, slots=True)
class HeartMeaningSignature:
    polarity_negation: str = "positive"
    modality: str = "asserted"
    quantification: str = "singular"
    temporal_relation: str = "atemporal"
    causal_relation: str = "none"
    speech_act: str = "assertion"

    def __post_init__(self) -> None:
        for name, labels in HEART_SEMANTIC_LABELS.items():
            value = getattr(self, name)
            if value not in labels:
                raise ValueError(f"invalid {name} label {value!r}")

    def to_canonical_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in HEART_SEMANTIC_LABELS}


@dataclass(frozen=True, slots=True)
class _MeaningSpec:
    spec_id: str
    renderings: tuple[tuple[str, str], ...]
    cues: tuple[tuple[str, str], ...]
    referent_text: str
    signature: HeartMeaningSignature
    critical_classes: tuple[str, ...]

    def rendering(self, dialect: str) -> str:
        return dict(self.renderings)[dialect]

    def cue(self, dialect: str) -> str:
        return dict(self.cues)[dialect]


@dataclass(frozen=True, slots=True)
class HeartTranslationCase:
    split: str
    source_dialect: str
    destination_dialect: str
    source_text: str
    target_text: str
    referent_text: str
    referent_start: int
    referent_end: int
    grounding_start: int
    grounding_end: int
    signature: HeartMeaningSignature
    critical_classes: tuple[str, ...]
    provenance: str
    spec_id: str
    case_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.split not in {"train", "heldout", "regression"}:
            raise ValueError("invalid Heart translation split")
        if self.source_dialect not in DIALECT_TO_ID or self.destination_dialect not in DIALECT_TO_ID:
            raise ValueError("unknown Heart translation dialect")
        if self.source_dialect == self.destination_dialect:
            raise ValueError("source and destination dialects must differ")
        assert_supported_text(self.source_text)
        assert_supported_text(self.target_text)
        if not (0 <= self.referent_start < self.referent_end <= len(self.source_text)):
            raise ValueError("invalid referent span")
        if self.source_text[self.referent_start : self.referent_end] != self.referent_text:
            raise ValueError("referent span does not match referent_text")
        if not (0 <= self.grounding_start < self.grounding_end <= len(self.source_text)):
            raise ValueError("invalid grounding span")
        classes = tuple(sorted(set(self.critical_classes)))
        if not classes or any(item not in CRITICAL_SEMANTIC_CLASSES for item in classes):
            raise ValueError("critical_classes must be non-empty required semantic classes")
        object.__setattr__(self, "critical_classes", classes)
        object.__setattr__(self, "case_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_CASE_SCHEMA,
            "split": self.split,
            "source_dialect": self.source_dialect,
            "destination_dialect": self.destination_dialect,
            "source_text": self.source_text,
            "target_text": self.target_text,
            "referent_text": self.referent_text,
            "referent_start": self.referent_start,
            "referent_end": self.referent_end,
            "grounding_start": self.grounding_start,
            "grounding_end": self.grounding_end,
            "signature": self.signature.to_canonical_dict(),
            "critical_classes": list(self.critical_classes),
            "provenance": self.provenance,
            "spec_id": self.spec_id,
        }
        if include_id:
            value["case_id"] = self.case_id
        return value


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    semantic_class: str
    left_case_id: str
    right_case_id: str

    def __post_init__(self) -> None:
        if self.semantic_class not in CRITICAL_SEMANTIC_CLASSES:
            raise ValueError("counterfactual pair uses an unknown semantic class")
        if self.left_case_id == self.right_case_id:
            raise ValueError("counterfactual pair must contain distinct cases")

    def to_canonical_dict(self) -> dict[str, str]:
        return {
            "semantic_class": self.semantic_class,
            "left_case_id": self.left_case_id,
            "right_case_id": self.right_case_id,
        }


@dataclass(frozen=True, slots=True)
class HeartDecoderCounterfactualPair:
    """Two held-out identity cases differing at exactly one source position."""

    position_kind: str
    left_case_id: str
    right_case_id: str
    changed_position: int
    left_character: str
    right_character: str
    probe_label: str | None = None

    def __post_init__(self) -> None:
        if self.position_kind not in {"head", "middle", "tail"}:
            raise ValueError("identity counterfactual position_kind must be head, middle, or tail")
        if not self.left_case_id or not self.right_case_id or self.left_case_id == self.right_case_id:
            raise ValueError("identity counterfactual pair requires two distinct case ids")
        if isinstance(self.changed_position, bool) or not isinstance(self.changed_position, int) or self.changed_position < 0:
            raise ValueError("identity counterfactual changed_position must be non-negative")
        if len(self.left_character) != 1 or len(self.right_character) != 1:
            raise ValueError("identity counterfactual characters must be single characters")
        if self.left_character == self.right_character:
            raise ValueError("identity counterfactual characters must differ")
        if self.probe_label is not None:
            label = self.probe_label.strip()
            if not label:
                raise ValueError("identity counterfactual probe_label must be non-empty when present")
            object.__setattr__(self, "probe_label", label)
        assert_supported_text(self.left_character + self.right_character)

    def to_canonical_dict(self) -> dict[str, Any]:
        value = {
            "schema": HEART_DECODER_COUNTERFACTUAL_SCHEMA,
            "position_kind": self.position_kind,
            "left_case_id": self.left_case_id,
            "right_case_id": self.right_case_id,
            "changed_position": self.changed_position,
            "left_character": self.left_character,
            "right_character": self.right_character,
        }
        if self.probe_label is not None:
            value["probe_label"] = self.probe_label
        return value


@dataclass(frozen=True, slots=True)
class HeartDecoderGeneralizationObjective:
    """Discrete identity loss plus explicit monotonic copy auxiliaries."""

    translation_weight: float = 1.0
    diagonal_alignment_weight: float = 1.0
    copy_route_weight: float = 0.1
    eos_route_weight: float = 0.25
    semantic_weight: float = 0.01
    pointer_weight: float = 0.01
    objective_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "translation_weight",
            "diagonal_alignment_weight",
            "copy_route_weight",
            "eos_route_weight",
            "semantic_weight",
            "pointer_weight",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value < float("inf"):
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "objective_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_DECODER_GENERALIZATION_OBJECTIVE_SCHEMA,
            "translation_weight": self.translation_weight,
            "diagonal_alignment_weight": self.diagonal_alignment_weight,
            "copy_route_weight": self.copy_route_weight,
            "eos_route_weight": self.eos_route_weight,
            "semantic_weight": self.semantic_weight,
            "pointer_weight": self.pointer_weight,
        }
        if include_id:
            value["objective_id"] = self.objective_id
        return value

    def write(self, state_root: Path | str) -> Path:
        root = Path(state_root) / "training" / "heart" / "objectives"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.objective_id}.json"
        payload = json.dumps(self.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("existing decoder generalization objective disagrees with immutable content")
            return path
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path


@dataclass(frozen=True, slots=True)
class HeartTranslationTrainingObjective:
    """Immutable, content-addressed task-loss recipe for Heart learning."""

    translation_weight: float = 1.0
    semantic_weight: float = 0.5
    pointer_weight: float = 0.25
    objective_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("translation_weight", "semantic_weight", "pointer_weight"):
            value = float(getattr(self, name))
            if not 0.0 < value < float("inf"):
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "objective_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_TRAINING_OBJECTIVE_SCHEMA,
            "translation_weight": self.translation_weight,
            "semantic_weight": self.semantic_weight,
            "pointer_weight": self.pointer_weight,
        }
        if include_id:
            value["objective_id"] = self.objective_id
        return value

    def write(self, state_root: Path | str) -> Path:
        root = Path(state_root) / "training" / "heart" / "objectives"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.objective_id}.json"
        payload = json.dumps(self.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("existing Heart training objective disagrees with immutable content")
            return path
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path


@dataclass(frozen=True, slots=True)
class HeartTranslationTrainingRecipe:
    """Immutable identity for curriculum traversal outside optimizer policy."""

    batch_sampler: str = "deterministic_shuffled_epoch_v1"
    recipe_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.batch_sampler != "deterministic_shuffled_epoch_v1":
            raise ValueError("unsupported Heart batch sampler")
        object.__setattr__(
            self,
            "recipe_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_TRAINING_RECIPE_SCHEMA,
            "batch_sampler": self.batch_sampler,
        }
        if include_id:
            value["recipe_id"] = self.recipe_id
        return value

    def write(self, state_root: Path | str) -> Path:
        root = Path(state_root) / "training" / "heart" / "recipes"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.recipe_id}.json"
        payload = json.dumps(self.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("existing Heart training recipe disagrees with immutable content")
            return path
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path


@dataclass(frozen=True, slots=True)
class HeartTranslationCurriculum:
    train_cases: tuple[HeartTranslationCase, ...]
    heldout_cases: tuple[HeartTranslationCase, ...]
    regression_cases: tuple[HeartTranslationCase, ...]
    counterfactual_pairs: tuple[CounterfactualPair, ...]
    curriculum_id: str = field(init=False)
    train_manifest_id: str = field(init=False)
    heldout_manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        all_cases = self.train_cases + self.heldout_cases + self.regression_cases
        ids = [case.case_id for case in all_cases]
        if len(ids) != len(set(ids)):
            raise ValueError("Heart curriculum contains duplicate case ids")
        if not self.train_cases or not self.heldout_cases or not self.regression_cases:
            raise ValueError("Heart curriculum requires train, heldout, and regression cases")
        train_specs = {case.spec_id for case in self.train_cases}
        heldout_specs = {case.spec_id for case in self.heldout_cases + self.regression_cases}
        if train_specs & heldout_specs:
            raise ValueError("Heart train and holdout semantic specs must be disjoint")
        heldout_ids = {case.case_id for case in self.heldout_cases}
        coverage = {pair.semantic_class for pair in self.counterfactual_pairs}
        if coverage != set(CRITICAL_SEMANTIC_CLASSES):
            raise ValueError("counterfactual pairs must cover every critical semantic class")
        for pair in self.counterfactual_pairs:
            if pair.left_case_id not in heldout_ids or pair.right_case_id not in heldout_ids:
                raise ValueError("counterfactual pairs must bind heldout cases")
        object.__setattr__(
            self,
            "train_manifest_id",
            canonical_sha256({"schema": HEART_CURRICULUM_SCHEMA, "split": "train", "case_ids": sorted(case.case_id for case in self.train_cases)}),
        )
        object.__setattr__(
            self,
            "heldout_manifest_id",
            canonical_sha256(
                {
                    "schema": HEART_CURRICULUM_SCHEMA,
                    "split": "heldout+regression",
                    "case_ids": sorted(case.case_id for case in self.heldout_cases + self.regression_cases),
                    "counterfactual_pairs": [pair.to_canonical_dict() for pair in self.counterfactual_pairs],
                }
            ),
        )
        object.__setattr__(self, "curriculum_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_CURRICULUM_SCHEMA,
            "dialects": list(DIALECTS),
            "train_manifest_id": self.train_manifest_id,
            "heldout_manifest_id": self.heldout_manifest_id,
            "train_cases": [case.to_canonical_dict() for case in self.train_cases],
            "heldout_cases": [case.to_canonical_dict() for case in self.heldout_cases],
            "regression_cases": [case.to_canonical_dict() for case in self.regression_cases],
            "counterfactual_pairs": [pair.to_canonical_dict() for pair in self.counterfactual_pairs],
        }
        if include_id:
            value["curriculum_id"] = self.curriculum_id
        return value

    def write(self, state_root: Path | str) -> Path:
        root = Path(state_root) / "training" / "heart" / "curricula"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.curriculum_id}.json"
        payload = json.dumps(self.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("existing Heart curriculum artifact disagrees with immutable content")
            return path
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path


@dataclass(frozen=True, slots=True)
class HeartDecoderMechanismCurriculum:
    """Small, separate copy/alignment gate; never substitutes for semantic curriculum."""

    train_cases: tuple[HeartTranslationCase, ...]
    heldout_cases: tuple[HeartTranslationCase, ...]
    curriculum_id: str = field(init=False)
    train_manifest_id: str = field(init=False)
    heldout_manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.train_cases or not self.heldout_cases:
            raise ValueError("decoder mechanism curriculum requires train and heldout cases")
        all_cases = self.train_cases + self.heldout_cases
        if len({case.case_id for case in all_cases}) != len(all_cases):
            raise ValueError("decoder mechanism curriculum contains duplicate cases")
        if any(case.source_text != case.target_text for case in all_cases):
            raise ValueError("decoder mechanism cases must be exact copy targets")
        train_specs = {case.spec_id for case in self.train_cases}
        heldout_specs = {case.spec_id for case in self.heldout_cases}
        if train_specs & heldout_specs:
            raise ValueError("decoder mechanism train and heldout specs must be disjoint")
        object.__setattr__(
            self,
            "train_manifest_id",
            canonical_sha256(
                {
                    "schema": HEART_DECODER_MECHANISM_CURRICULUM_SCHEMA,
                    "split": "train",
                    "case_ids": sorted(case.case_id for case in self.train_cases),
                }
            ),
        )
        object.__setattr__(
            self,
            "heldout_manifest_id",
            canonical_sha256(
                {
                    "schema": HEART_DECODER_MECHANISM_CURRICULUM_SCHEMA,
                    "split": "heldout",
                    "case_ids": sorted(case.case_id for case in self.heldout_cases),
                }
            ),
        )
        object.__setattr__(
            self,
            "curriculum_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_DECODER_MECHANISM_CURRICULUM_SCHEMA,
            "train_manifest_id": self.train_manifest_id,
            "heldout_manifest_id": self.heldout_manifest_id,
            "train_cases": [case.to_canonical_dict() for case in self.train_cases],
            "heldout_cases": [case.to_canonical_dict() for case in self.heldout_cases],
        }
        if include_id:
            value["curriculum_id"] = self.curriculum_id
        return value

    def write(self, state_root: Path | str) -> Path:
        root = Path(state_root) / "training" / "heart" / "decoder_curricula"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.curriculum_id}.json"
        payload = json.dumps(self.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("existing decoder mechanism curriculum disagrees with immutable content")
            return path
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path


@dataclass(frozen=True, slots=True)
class HeartDecoderGeneralizationCurriculum:
    """Diverse identity-copy curriculum with replay and held-out source probes."""

    train_cases: tuple[HeartTranslationCase, ...]
    heldout_cases: tuple[HeartTranslationCase, ...]
    replay_case_ids: tuple[str, ...]
    counterfactual_pairs: tuple[HeartDecoderCounterfactualPair, ...]
    curriculum_id: str = field(init=False)
    train_manifest_id: str = field(init=False)
    heldout_manifest_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.train_cases or not self.heldout_cases:
            raise ValueError("decoder generalization curriculum requires train and heldout cases")
        all_cases = self.train_cases + self.heldout_cases
        ids = [case.case_id for case in all_cases]
        if len(ids) != len(set(ids)):
            raise ValueError("decoder generalization curriculum contains duplicate cases")
        if any(case.source_text != case.target_text for case in all_cases):
            raise ValueError("decoder generalization cases must be exact identity targets")
        train_specs = {case.spec_id for case in self.train_cases}
        heldout_specs = {case.spec_id for case in self.heldout_cases}
        if train_specs & heldout_specs:
            raise ValueError("decoder generalization train and heldout specs must be disjoint")
        train_texts = {case.source_text for case in self.train_cases}
        heldout_texts = {case.source_text for case in self.heldout_cases}
        if train_texts & heldout_texts:
            raise ValueError("decoder generalization heldout text must be unseen")
        replay = tuple(self.replay_case_ids)
        train_ids = {case.case_id for case in self.train_cases}
        if not replay or len(replay) != len(set(replay)) or not set(replay) <= train_ids:
            raise ValueError("replay_case_ids must be a unique non-empty subset of train cases")
        object.__setattr__(self, "replay_case_ids", replay)

        heldout_by_id = {case.case_id: case for case in self.heldout_cases}
        kinds = {pair.position_kind for pair in self.counterfactual_pairs}
        if kinds != {"head", "middle", "tail"}:
            raise ValueError("identity counterfactual pairs must cover head, middle, and tail")
        for pair in self.counterfactual_pairs:
            try:
                left = heldout_by_id[pair.left_case_id]
                right = heldout_by_id[pair.right_case_id]
            except KeyError as exc:
                raise ValueError("identity counterfactual pairs must bind heldout cases") from exc
            if len(left.source_text) != len(right.source_text):
                raise ValueError("identity counterfactual sources must have equal length")
            differences = [
                index
                for index, (left_char, right_char) in enumerate(
                    zip(left.source_text, right.source_text, strict=True)
                )
                if left_char != right_char
            ]
            if differences != [pair.changed_position]:
                raise ValueError("identity counterfactual sources must differ only at the declared position")
            if (
                left.source_text[pair.changed_position] != pair.left_character
                or right.source_text[pair.changed_position] != pair.right_character
            ):
                raise ValueError("identity counterfactual characters disagree with their cases")

        alphabet = set(default_alphabet())
        covered = {character for case in self.train_cases for character in case.source_text}
        if covered != alphabet:
            missing = "".join(sorted(alphabet - covered))
            raise ValueError(f"decoder generalization train split misses substrate characters: {missing!r}")
        if max(len(case.source_text) for case in self.train_cases) <= 256:
            raise ValueError("decoder generalization train split requires multi-page examples")
        if max(len(case.source_text) for case in self.heldout_cases) <= max(
            len(case.source_text) for case in self.train_cases
        ):
            raise ValueError("decoder generalization heldout split requires length extrapolation")

        object.__setattr__(
            self,
            "train_manifest_id",
            canonical_sha256(
                {
                    "schema": HEART_DECODER_GENERALIZATION_CURRICULUM_SCHEMA,
                    "split": "train",
                    "case_ids": sorted(case.case_id for case in self.train_cases),
                    "replay_case_ids": list(replay),
                }
            ),
        )
        object.__setattr__(
            self,
            "heldout_manifest_id",
            canonical_sha256(
                {
                    "schema": HEART_DECODER_GENERALIZATION_CURRICULUM_SCHEMA,
                    "split": "heldout",
                    "case_ids": sorted(case.case_id for case in self.heldout_cases),
                    "counterfactual_pairs": [pair.to_canonical_dict() for pair in self.counterfactual_pairs],
                }
            ),
        )
        object.__setattr__(
            self,
            "curriculum_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_DECODER_GENERALIZATION_CURRICULUM_SCHEMA,
            "train_manifest_id": self.train_manifest_id,
            "heldout_manifest_id": self.heldout_manifest_id,
            "replay_case_ids": list(self.replay_case_ids),
            "train_cases": [case.to_canonical_dict() for case in self.train_cases],
            "heldout_cases": [case.to_canonical_dict() for case in self.heldout_cases],
            "counterfactual_pairs": [pair.to_canonical_dict() for pair in self.counterfactual_pairs],
        }
        if include_id:
            value["curriculum_id"] = self.curriculum_id
        return value

    def write(self, state_root: Path | str) -> Path:
        root = Path(state_root) / "training" / "heart" / "decoder_curricula"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.curriculum_id}.json"
        payload = json.dumps(self.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("existing decoder generalization curriculum disagrees with immutable content")
            return path
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path


@dataclass(slots=True)
class HeartTranslationBatch:
    source_indices: torch.Tensor
    source_cells16: torch.Tensor
    source_positions: torch.Tensor
    source_mask: torch.Tensor
    source_frame_ids: tuple[str, ...]
    source_field_ids: tuple[str, ...]
    source_rail_ids: tuple[str, ...]
    source_dialect_ids: torch.Tensor
    destination_dialect_ids: torch.Tensor
    decoder_input_ids: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    semantic_targets: dict[str, torch.Tensor]
    referent_start: torch.Tensor
    referent_end: torch.Tensor
    grounding_start: torch.Tensor
    grounding_end: torch.Tensor
    cases: tuple[HeartTranslationCase, ...]

    def to(self, device: torch.device | str) -> "HeartTranslationBatch":
        return HeartTranslationBatch(
            source_indices=self.source_indices.to(device),
            source_cells16=self.source_cells16.to(device),
            source_positions=self.source_positions.to(device),
            source_mask=self.source_mask.to(device),
            source_frame_ids=self.source_frame_ids,
            source_field_ids=self.source_field_ids,
            source_rail_ids=self.source_rail_ids,
            source_dialect_ids=self.source_dialect_ids.to(device),
            destination_dialect_ids=self.destination_dialect_ids.to(device),
            decoder_input_ids=self.decoder_input_ids.to(device),
            target_ids=self.target_ids.to(device),
            target_mask=self.target_mask.to(device),
            semantic_targets={name: value.to(device) for name, value in self.semantic_targets.items()},
            referent_start=self.referent_start.to(device),
            referent_end=self.referent_end.to(device),
            grounding_start=self.grounding_start.to(device),
            grounding_end=self.grounding_end.to(device),
            cases=self.cases,
        )


@dataclass(frozen=True, slots=True)
class HeartTranslationLoss:
    total: torch.Tensor
    translation: torch.Tensor
    semantic: torch.Tensor
    pointers: torch.Tensor


@dataclass(frozen=True, slots=True)
class HeartDecoderGeneralizationLoss:
    total: torch.Tensor
    translation: torch.Tensor
    diagonal_alignment: torch.Tensor
    copy_route: torch.Tensor
    eos_route: torch.Tensor
    semantic: torch.Tensor
    pointers: torch.Tensor


@dataclass(frozen=True, slots=True)
class HeartTranslationCaseResult:
    case_id: str
    generated_text: str
    generated_characters: int
    terminated: bool
    target_exact: bool
    source_semantic_exact: bool
    reverse_semantic_exact: bool
    referent_pointer_exact: bool
    grounding_pointer_exact: bool
    referent_preserved: bool
    grounded_roundtrip: bool
    aggregate_semantic_fidelity: bool
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if not isinstance(self.generated_text, str):
            raise TypeError("generated_text must be a string")
        if self.generated_characters != len(self.generated_text):
            raise ValueError("generated character count disagrees with exact text")
        object.__setattr__(
            self,
            "result_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "case_id": self.case_id,
            "generated_text": self.generated_text,
            "generated_characters": self.generated_characters,
            "terminated": self.terminated,
            "target_exact": self.target_exact,
            "source_semantic_exact": self.source_semantic_exact,
            "reverse_semantic_exact": self.reverse_semantic_exact,
            "referent_pointer_exact": self.referent_pointer_exact,
            "grounding_pointer_exact": self.grounding_pointer_exact,
            "referent_preserved": self.referent_preserved,
            "grounded_roundtrip": self.grounded_roundtrip,
            "aggregate_semantic_fidelity": self.aggregate_semantic_fidelity,
        }
        if include_id:
            value["result_id"] = self.result_id
        return value


@dataclass(frozen=True, slots=True)
class HeartTranslationEvaluationReport:
    descriptor_id: str
    curriculum_id: str
    evaluation_id: str
    heldout_case_count: int
    translation_exact_rate: float
    translation_termination_rate: float
    source_semantic_exact_rate: float
    reverse_semantic_exact_rate: float
    referent_pointer_rate: float
    grounding_pointer_rate: float
    evidence: HeartSemanticFidelityEvidence
    case_results: tuple[HeartTranslationCaseResult, ...]

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": HEART_EVALUATION_SCHEMA,
            "descriptor_id": self.descriptor_id,
            "curriculum_id": self.curriculum_id,
            "evaluation_id": self.evaluation_id,
            "heldout_case_count": self.heldout_case_count,
            "translation_exact_rate": self.translation_exact_rate,
            "translation_termination_rate": self.translation_termination_rate,
            "source_semantic_exact_rate": self.source_semantic_exact_rate,
            "reverse_semantic_exact_rate": self.reverse_semantic_exact_rate,
            "referent_pointer_rate": self.referent_pointer_rate,
            "grounding_pointer_rate": self.grounding_pointer_rate,
            "evidence": self.evidence.to_canonical_dict(),
            "case_results": [item.to_canonical_dict() for item in self.case_results],
        }


@dataclass(frozen=True, slots=True)
class HeartDecoderCaseDiagnostic:
    case_id: str
    target_characters: int
    teacher_forced_characters_correct: int
    teacher_forced_character_accuracy: float
    teacher_forced_eos_correct: bool
    teacher_forced_sequence_exact: bool
    greedy_text: str
    greedy_terminated: bool
    greedy_exact: bool
    greedy_first_divergence: int | None
    greedy_correct_prefix_characters: int
    mean_generation_gate: float
    mean_copyable_generation_gate: float | None
    mean_noncopyable_generation_gate: float | None
    mean_target_character_attention_mass: float | None
    mean_attention_peak: float
    diagnostic_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.target_characters < 1:
            raise ValueError("decoder diagnostics require a non-empty target")
        if not 0 <= self.teacher_forced_characters_correct <= self.target_characters:
            raise ValueError("teacher-forced correct count is outside the target")
        if not 0 <= self.greedy_correct_prefix_characters <= self.target_characters:
            raise ValueError("greedy correct-prefix count is outside the target")
        if self.greedy_first_divergence is not None and not (
            0 <= self.greedy_first_divergence <= self.target_characters
        ):
            raise ValueError("greedy first divergence is outside the target")
        object.__setattr__(
            self,
            "diagnostic_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_DECODER_CASE_DIAGNOSTIC_SCHEMA,
            "case_id": self.case_id,
            "target_characters": self.target_characters,
            "teacher_forced_characters_correct": self.teacher_forced_characters_correct,
            "teacher_forced_character_accuracy": self.teacher_forced_character_accuracy,
            "teacher_forced_eos_correct": self.teacher_forced_eos_correct,
            "teacher_forced_sequence_exact": self.teacher_forced_sequence_exact,
            "greedy_text": self.greedy_text,
            "greedy_terminated": self.greedy_terminated,
            "greedy_exact": self.greedy_exact,
            "greedy_first_divergence": self.greedy_first_divergence,
            "greedy_correct_prefix_characters": self.greedy_correct_prefix_characters,
            "mean_generation_gate": self.mean_generation_gate,
            "mean_copyable_generation_gate": self.mean_copyable_generation_gate,
            "mean_noncopyable_generation_gate": self.mean_noncopyable_generation_gate,
            "mean_target_character_attention_mass": self.mean_target_character_attention_mass,
            "mean_attention_peak": self.mean_attention_peak,
        }
        if include_id:
            value["diagnostic_id"] = self.diagnostic_id
        return value


@dataclass(frozen=True, slots=True)
class HeartDecoderDiagnosticReport:
    descriptor_id: str
    checkpoint_id: str
    curriculum_id: str
    split: str
    execution_device: str
    case_count: int
    teacher_forced_character_accuracy: float
    teacher_forced_eos_accuracy: float
    teacher_forced_sequence_exact_rate: float
    greedy_exact_rate: float
    greedy_termination_rate: float
    mean_greedy_correct_prefix_fraction: float
    mean_generation_gate: float
    mean_copyable_generation_gate: float | None
    mean_noncopyable_generation_gate: float | None
    mean_target_character_attention_mass: float | None
    mean_attention_peak: float
    position_accuracy: tuple[tuple[int, int, float], ...]
    case_diagnostics: tuple[HeartDecoderCaseDiagnostic, ...]
    diagnostic_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not all(
            (self.descriptor_id, self.checkpoint_id, self.curriculum_id, self.split, self.execution_device)
        ):
            raise ValueError("decoder diagnostic identity must be non-empty")
        if self.case_count != len(self.case_diagnostics) or self.case_count < 1:
            raise ValueError("decoder diagnostic case count disagrees with evidence")
        object.__setattr__(
            self,
            "diagnostic_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_DECODER_DIAGNOSTIC_SCHEMA,
            "descriptor_id": self.descriptor_id,
            "checkpoint_id": self.checkpoint_id,
            "curriculum_id": self.curriculum_id,
            "split": self.split,
            "execution_device": self.execution_device,
            "case_count": self.case_count,
            "teacher_forced_character_accuracy": self.teacher_forced_character_accuracy,
            "teacher_forced_eos_accuracy": self.teacher_forced_eos_accuracy,
            "teacher_forced_sequence_exact_rate": self.teacher_forced_sequence_exact_rate,
            "greedy_exact_rate": self.greedy_exact_rate,
            "greedy_termination_rate": self.greedy_termination_rate,
            "mean_greedy_correct_prefix_fraction": self.mean_greedy_correct_prefix_fraction,
            "mean_generation_gate": self.mean_generation_gate,
            "mean_copyable_generation_gate": self.mean_copyable_generation_gate,
            "mean_noncopyable_generation_gate": self.mean_noncopyable_generation_gate,
            "mean_target_character_attention_mass": self.mean_target_character_attention_mass,
            "mean_attention_peak": self.mean_attention_peak,
            "position_accuracy": [
                {"position": position, "support": support, "accuracy": accuracy}
                for position, support, accuracy in self.position_accuracy
            ],
            "case_diagnostics": [item.to_canonical_dict() for item in self.case_diagnostics],
        }
        if include_id:
            value["diagnostic_id"] = self.diagnostic_id
        return value

    def write(self, state_root: Path | str) -> Path:
        root = Path(state_root) / "training" / "heart" / "decoder_diagnostics"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.diagnostic_id}.json"
        payload = json.dumps(self.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("existing Heart decoder diagnostic disagrees with immutable content")
            return path
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path


@dataclass(frozen=True, slots=True)
class HeartDecoderGeneralizationEvidence:
    descriptor_id: str
    checkpoint_id: str
    curriculum_id: str
    split: str
    decoder_diagnostic_id: str
    mean_diagonal_attention_mass: float
    diagonal_attention_top1_rate: float
    counterfactual_pair_exact_rate: float
    minimum_counterfactual_target_margin: float
    memorized_train_output_count: int
    length_bucket_metrics: tuple[tuple[str, float, float, int], ...]
    counterfactual_results: tuple[dict[str, Any], ...]
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not all(
            (
                self.descriptor_id,
                self.checkpoint_id,
                self.curriculum_id,
                self.split,
                self.decoder_diagnostic_id,
            )
        ):
            raise ValueError("decoder generalization evidence identity must be non-empty")
        for name in (
            "mean_diagonal_attention_mass",
            "diagonal_attention_top1_rate",
            "counterfactual_pair_exact_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        margin = float(self.minimum_counterfactual_target_margin)
        if not -float("inf") < margin < float("inf"):
            raise ValueError("minimum counterfactual margin must be finite")
        object.__setattr__(self, "minimum_counterfactual_target_margin", margin)
        if self.memorized_train_output_count < 0:
            raise ValueError("memorized_train_output_count must be non-negative")
        object.__setattr__(
            self,
            "evidence_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_DECODER_GENERALIZATION_EVIDENCE_SCHEMA,
            "descriptor_id": self.descriptor_id,
            "checkpoint_id": self.checkpoint_id,
            "curriculum_id": self.curriculum_id,
            "split": self.split,
            "decoder_diagnostic_id": self.decoder_diagnostic_id,
            "mean_diagonal_attention_mass": self.mean_diagonal_attention_mass,
            "diagonal_attention_top1_rate": self.diagonal_attention_top1_rate,
            "counterfactual_pair_exact_rate": self.counterfactual_pair_exact_rate,
            "minimum_counterfactual_target_margin": self.minimum_counterfactual_target_margin,
            "memorized_train_output_count": self.memorized_train_output_count,
            "length_bucket_metrics": [
                {
                    "bucket": bucket,
                    "teacher_forced_character_accuracy": teacher_accuracy,
                    "greedy_exact_rate": greedy_rate,
                    "case_count": case_count,
                }
                for bucket, teacher_accuracy, greedy_rate, case_count in self.length_bucket_metrics
            ],
            "counterfactual_results": list(self.counterfactual_results),
        }
        if include_id:
            value["evidence_id"] = self.evidence_id
        return value

    def write(self, state_root: Path | str) -> Path:
        root = Path(state_root) / "training" / "heart" / "decoder_generalization"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{self.evidence_id}.json"
        payload = json.dumps(self.to_canonical_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("existing decoder generalization evidence disagrees with immutable content")
            return path
        fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path


def _spec(
    spec_id: str,
    renderings: dict[str, str],
    cues: dict[str, str],
    referent: str,
    signature: HeartMeaningSignature,
    *critical: str,
) -> _MeaningSpec:
    if set(renderings) != set(DIALECTS) or set(cues) != set(DIALECTS):
        raise ValueError("meaning spec must render every Heart dialect")
    for dialect in DIALECTS:
        assert_supported_text(renderings[dialect])
        if referent not in renderings[dialect]:
            raise ValueError("referent missing from rendering")
        if cues[dialect] not in renderings[dialect]:
            raise ValueError("grounding cue missing from rendering")
    return _MeaningSpec(
        spec_id=spec_id,
        renderings=tuple(sorted(renderings.items())),
        cues=tuple(sorted(cues.items())),
        referent_text=referent,
        signature=signature,
        critical_classes=tuple(critical),
    )


def _modality_specs(subjects: Sequence[str], verbs: Sequence[str], prefix: str) -> list[_MeaningSpec]:
    specs: list[_MeaningSpec] = []
    forms = {
        "asserted": ("{s} will {v}.", "Assertion: {s} will {v}.", "State that {s} will {v}.", "will", "Assertion", "State"),
        "capability": ("{s} can {v}.", "Capability: {s} is able to {v}.", "{s} has ability to {v}.", "can", "Capability", "ability"),
        "obligation": ("{s} must {v}.", "Obligation: {s} is required to {v}.", "{s} has duty to {v}.", "must", "Obligation", "duty"),
        "possibility": ("{s} might {v}.", "Possibility: {s} may {v}.", "{s} could {v}.", "might", "Possibility", "could"),
    }
    for subject in subjects:
        for verb in verbs:
            for modality, (a, b, c, ca, cb, cc) in forms.items():
                specs.append(
                    _spec(
                        f"{prefix}:modality:{subject}:{verb}:{modality}",
                        {
                            DIALECTS[0]: a.format(s=subject, v=verb),
                            DIALECTS[1]: b.format(s=subject, v=verb),
                            DIALECTS[2]: c.format(s=subject, v=verb),
                        },
                        {DIALECTS[0]: ca, DIALECTS[1]: cb, DIALECTS[2]: cc},
                        subject,
                        HeartMeaningSignature(modality=modality, temporal_relation="future" if modality == "asserted" else "atemporal"),
                        "modality",
                        "referent_identity",
                        "grounding_provenance",
                    )
                )
    return specs


def _polarity_specs(subjects: Sequence[str], verbs: Sequence[str], prefix: str) -> list[_MeaningSpec]:
    specs: list[_MeaningSpec] = []
    for subject in subjects:
        for verb in verbs:
            for negative in (False, True):
                if negative:
                    renderings = {
                        DIALECTS[0]: f"{subject} does not {verb}.",
                        DIALECTS[1]: f"Negative assertion: {subject} does not {verb}.",
                        DIALECTS[2]: f"{subject} is marked not {verb}.",
                    }
                    cues = {DIALECTS[0]: "not", DIALECTS[1]: "Negative", DIALECTS[2]: "not"}
                    polarity = "negative"
                else:
                    renderings = {
                        DIALECTS[0]: f"{subject} does {verb}.",
                        DIALECTS[1]: f"Positive assertion: {subject} does {verb}.",
                        DIALECTS[2]: f"{subject} is marked yes {verb}.",
                    }
                    cues = {DIALECTS[0]: "does", DIALECTS[1]: "Positive", DIALECTS[2]: "yes"}
                    polarity = "positive"
                specs.append(
                    _spec(
                        f"{prefix}:polarity:{subject}:{verb}:{polarity}",
                        renderings,
                        cues,
                        subject,
                        HeartMeaningSignature(polarity_negation=polarity),
                        "polarity_negation",
                        "referent_identity",
                        "grounding_provenance",
                    )
                )
    return specs


def _temporal_specs(subjects: Sequence[str], verb: str, past: str, third: str, prefix: str) -> list[_MeaningSpec]:
    specs: list[_MeaningSpec] = []
    forms = {
        "past": (f"{{s}} {past} yesterday.", f"Past event: yesterday {{s}} {past}.", f"Previously {{s}} {past}.", "yesterday", "Past", "Previously"),
        "present": (f"{{s}} {third} today.", f"Present event: today {{s}} {third}.", f"Currently {{s}} {third}.", "today", "Present", "Currently"),
        "future": (f"{{s}} will {verb} tomorrow.", f"Future event: tomorrow {{s}} will {verb}.", f"Later {{s}} will {verb}.", "tomorrow", "Future", "Later"),
        "atemporal": (f"{{s}} can {verb}.", f"Atemporal capability: {{s}} can {verb}.", f"General ability: {{s}} can {verb}.", "can", "Atemporal", "General"),
    }
    for subject in subjects:
        for temporal, (a, b, c, ca, cb, cc) in forms.items():
            modality = "capability" if temporal == "atemporal" else "asserted"
            specs.append(
                _spec(
                    f"{prefix}:temporal:{subject}:{temporal}",
                    {DIALECTS[0]: a.format(s=subject), DIALECTS[1]: b.format(s=subject), DIALECTS[2]: c.format(s=subject)},
                    {DIALECTS[0]: ca, DIALECTS[1]: cb, DIALECTS[2]: cc},
                    subject,
                    HeartMeaningSignature(modality=modality, temporal_relation=temporal),
                    "temporal_relation",
                    "referent_identity",
                    "grounding_provenance",
                )
            )
    return specs


def _quantification_specs(prefix: str, verb: str = "run") -> list[_MeaningSpec]:
    forms = {
        "singular": (f"The dog can {verb}.", f"Single dog: the dog can {verb}.", f"One dog has ability to {verb}.", "dog", "The", "Single", "One"),
        "generic": (f"A dog can {verb} in general.", f"Generic dog claim: a dog can {verb}.", f"General dog ability allows {verb}.", "dog", "general", "Generic", "General"),
        "universal": (f"Every dog can {verb}.", f"Universal dog claim: every dog can {verb}.", f"All dog members can {verb}.", "dog", "Every", "Universal", "All"),
        "existential": (f"Some dog can {verb}.", f"Existential dog claim: some dog can {verb}.", f"At least one dog can {verb}.", "dog", "Some", "Existential", "least"),
        "zero": (f"No dog can {verb}.", f"Zero dog claim: no dog can {verb}.", f"Not one dog can {verb}.", "dog", "No", "Zero", "Not"),
    }
    specs: list[_MeaningSpec] = []
    for quant, (a, b, c, referent, ca, cb, cc) in forms.items():
        specs.append(
            _spec(
                f"{prefix}:quantification:{quant}:{verb}",
                {DIALECTS[0]: a, DIALECTS[1]: b, DIALECTS[2]: c},
                {DIALECTS[0]: ca, DIALECTS[1]: cb, DIALECTS[2]: cc},
                referent,
                HeartMeaningSignature(modality="capability", quantification=quant),
                "quantification",
                "referent_identity",
                "grounding_provenance",
            )
        )
    return specs


def _causal_specs(subjects: Sequence[str], prefix: str) -> list[_MeaningSpec]:
    specs: list[_MeaningSpec] = []
    for subject in subjects:
        specs.extend(
            [
                _spec(
                    f"{prefix}:causal:{subject}:effect_first",
                    {
                        DIALECTS[0]: f"{subject} slipped because the floor was wet.",
                        DIALECTS[1]: f"Effect then cause: {subject} slipped because the floor was wet.",
                        DIALECTS[2]: f"{subject} slipping was caused by the wet floor.",
                    },
                    {DIALECTS[0]: "because", DIALECTS[1]: "because", DIALECTS[2]: "caused"},
                    subject,
                    HeartMeaningSignature(causal_relation="effect_because_cause"),
                    "causal_relation",
                    "referent_identity",
                    "grounding_provenance",
                ),
                _spec(
                    f"{prefix}:causal:{subject}:cause_first",
                    {
                        DIALECTS[0]: f"Because the floor was wet, {subject} slipped.",
                        DIALECTS[1]: f"Cause then effect: the wet floor caused {subject} to slip.",
                        DIALECTS[2]: f"Wet floor caused {subject} to slip afterward.",
                    },
                    {DIALECTS[0]: "Because", DIALECTS[1]: "Cause", DIALECTS[2]: "caused"},
                    subject,
                    HeartMeaningSignature(causal_relation="cause_before_effect"),
                    "causal_relation",
                    "referent_identity",
                    "grounding_provenance",
                ),
            ]
        )
    return specs


def _speech_specs(subjects: Sequence[str], prefix: str) -> list[_MeaningSpec]:
    specs: list[_MeaningSpec] = []
    for subject in subjects:
        variants = {
            "assertion": (
                f"{subject} opened the door.",
                f"Assertion: {subject} opened the door.",
                f"State that {subject} opened the door.",
                "opened",
                "Assertion",
                "State",
                "past",
            ),
            "question": (
                f"Did {subject} open the door?",
                f"Question: did {subject} open the door?",
                f"Ask whether {subject} opened the door.",
                "Did",
                "Question",
                "Ask",
                "past",
            ),
            "request": (
                f"{subject}, open the door.",
                f"Request: {subject} should open the door.",
                f"Requested action for {subject}: open the door.",
                "open",
                "Request",
                "Requested",
                "future",
            ),
            "warning": (
                f"Warning: {subject} must not touch the wire.",
                f"Danger warning: {subject} should not touch the wire.",
                f"Warn {subject}: do not touch the wire.",
                "Warning",
                "warning",
                "Warn",
                "future",
            ),
        }
        for speech, (a, b, c, ca, cb, cc, temporal) in variants.items():
            polarity = "negative" if speech == "warning" else "positive"
            modality = "obligation" if speech == "warning" else "asserted"
            specs.append(
                _spec(
                    f"{prefix}:speech:{subject}:{speech}",
                    {DIALECTS[0]: a, DIALECTS[1]: b, DIALECTS[2]: c},
                    {DIALECTS[0]: ca, DIALECTS[1]: cb, DIALECTS[2]: cc},
                    subject,
                    HeartMeaningSignature(
                        polarity_negation=polarity,
                        modality=modality,
                        temporal_relation=temporal,
                        speech_act=speech,
                    ),
                    "speech_act",
                    "referent_identity",
                    "grounding_provenance",
                )
            )
    return specs


def _referent_specs(prefix: str, pairs: Sequence[tuple[str, str]]) -> list[_MeaningSpec]:
    specs: list[_MeaningSpec] = []
    for first, second in pairs:
        for chosen in (first, second):
            other = second if chosen == first else first
            specs.append(
                _spec(
                    f"{prefix}:referent:{first}:{second}:{chosen}",
                    {
                        DIALECTS[0]: f"{first} told {second} that {chosen} will wait.",
                        DIALECTS[1]: f"Referent: the future waiter is {chosen}; {first} spoke to {second}.",
                        DIALECTS[2]: f"Waiter identity equals {chosen}, not {other}.",
                    },
                    {DIALECTS[0]: chosen, DIALECTS[1]: "Referent", DIALECTS[2]: "identity"},
                    chosen,
                    HeartMeaningSignature(temporal_relation="future"),
                    "referent_identity",
                    "grounding_provenance",
                )
            )
    return specs


def _case_from_spec(spec: _MeaningSpec, split: str, source: str, destination: str) -> HeartTranslationCase:
    source_text = spec.rendering(source)
    target_text = spec.rendering(destination)
    referent_start = source_text.index(spec.referent_text)
    cue = spec.cue(source)
    grounding_start = source_text.index(cue)
    return HeartTranslationCase(
        split=split,
        source_dialect=source,
        destination_dialect=destination,
        source_text=source_text,
        target_text=target_text,
        referent_text=spec.referent_text,
        referent_start=referent_start,
        referent_end=referent_start + len(spec.referent_text),
        grounding_start=grounding_start,
        grounding_end=grounding_start + len(cue),
        signature=spec.signature,
        critical_classes=spec.critical_classes,
        provenance=f"synthetic:heart-translation-v1:{spec.spec_id}",
        spec_id=spec.spec_id,
    )


def _cases(specs: Iterable[_MeaningSpec], split: str, directions: Sequence[tuple[str, str]]) -> tuple[HeartTranslationCase, ...]:
    return tuple(
        _case_from_spec(spec, split, source, destination)
        for spec in specs
        for source, destination in directions
    )


_COMPLETE_FIELD_CONTEXT = "Neutral context page carries filler words only. " * 7


def _complete_field_variant(
    case: HeartTranslationCase,
    semantic_class: str,
) -> HeartTranslationCase:
    """Place the grounded clause beyond one page without changing its meaning."""

    prefix = _COMPLETE_FIELD_CONTEXT
    return HeartTranslationCase(
        split=case.split,
        source_dialect=case.source_dialect,
        destination_dialect=case.destination_dialect,
        source_text=prefix + case.source_text,
        target_text=case.target_text,
        referent_text=case.referent_text,
        referent_start=len(prefix) + case.referent_start,
        referent_end=len(prefix) + case.referent_end,
        grounding_start=len(prefix) + case.grounding_start,
        grounding_end=len(prefix) + case.grounding_end,
        signature=case.signature,
        critical_classes=case.critical_classes,
        provenance=f"synthetic:heart-translation-v3:complete-field:{case.spec_id}",
        spec_id=f"{case.spec_id}:complete-field:{semantic_class}",
    )


def _complete_field_coverage_cases(
    cases: Sequence[HeartTranslationCase],
) -> tuple[HeartTranslationCase, ...]:
    variants: list[HeartTranslationCase] = []
    for semantic_class in CRITICAL_SEMANTIC_CLASSES:
        source = next(case for case in cases if semantic_class in case.critical_classes)
        variants.append(_complete_field_variant(source, semantic_class))
    return tuple(variants)


def build_heart_translation_curriculum() -> HeartTranslationCurriculum:
    train_subjects = ("Mira", "Tessa", "Nolan", "Aria")
    train_verbs = ("run", "wait")
    train_specs = (
        _modality_specs(train_subjects, train_verbs, "train")
        + _polarity_specs(train_subjects, train_verbs, "train")
        + _temporal_specs(train_subjects, "run", "ran", "runs", "train")
        + _quantification_specs("train", "run")
        + _causal_specs(train_subjects, "train")
        + _speech_specs(train_subjects, "train")
        + _referent_specs("train", (("Mira", "Tessa"), ("Nolan", "Aria")))
    )
    heldout_specs = (
        _modality_specs(("Lena",), ("jump",), "heldout")
        + _polarity_specs(("Omar",), ("wait",), "heldout")
        + _temporal_specs(("Lena",), "jump", "jumped", "jumps", "heldout")
        + _quantification_specs("heldout", "jump")
        + _causal_specs(("Omar",), "heldout")
        + _speech_specs(("Lena",), "heldout")
        + _referent_specs("heldout", (("Lena", "Omar"),))
    )
    regression_specs = (
        _modality_specs(("Rhea",), ("wait",), "regression")[:2]
        + _polarity_specs(("Rhea",), ("wait",), "regression")
        + _causal_specs(("Rhea",), "regression")[:1]
        + _speech_specs(("Rhea",), "regression")[:2]
    )
    train_cases = _cases(train_specs, "train", TRAIN_DIRECTIONS)
    heldout_cases = _cases(heldout_specs, "heldout", EVAL_DIRECTIONS)
    regression_cases = _cases(regression_specs, "regression", EVAL_DIRECTIONS)

    direction = EVAL_DIRECTIONS[0]

    def pick(fragment: str) -> HeartTranslationCase:
        matches = [
            case
            for case in heldout_cases
            if fragment in case.spec_id
            and case.source_dialect == direction[0]
            and case.destination_dialect == direction[1]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"counterfactual selector {fragment!r} matched {len(matches)} cases")
        return matches[0]

    modality_left = pick("heldout:modality:Lena:jump:capability")
    modality_right = pick("heldout:modality:Lena:jump:obligation")
    polarity_left = pick("heldout:polarity:Omar:wait:positive")
    polarity_right = pick("heldout:polarity:Omar:wait:negative")
    temporal_left = pick("heldout:temporal:Lena:past")
    temporal_right = pick("heldout:temporal:Lena:future")
    quant_left = pick("heldout:quantification:universal:jump")
    quant_right = pick("heldout:quantification:zero:jump")
    causal_left = pick("heldout:causal:Omar:effect_first")
    causal_right = pick("heldout:causal:Omar:cause_first")
    speech_left = pick("heldout:speech:Lena:question")
    speech_right = pick("heldout:speech:Lena:request")
    referent_left = pick("heldout:referent:Lena:Omar:Lena")
    referent_right = pick("heldout:referent:Lena:Omar:Omar")
    grounding_left = modality_left
    grounding_right = causal_left

    pairs = (
        CounterfactualPair("modality", modality_left.case_id, modality_right.case_id),
        CounterfactualPair("polarity_negation", polarity_left.case_id, polarity_right.case_id),
        CounterfactualPair("temporal_relation", temporal_left.case_id, temporal_right.case_id),
        CounterfactualPair("quantification", quant_left.case_id, quant_right.case_id),
        CounterfactualPair("causal_relation", causal_left.case_id, causal_right.case_id),
        CounterfactualPair("speech_act", speech_left.case_id, speech_right.case_id),
        CounterfactualPair("referent_identity", referent_left.case_id, referent_right.case_id),
        CounterfactualPair("grounding_provenance", grounding_left.case_id, grounding_right.case_id),
    )
    train_cases = train_cases + _complete_field_coverage_cases(train_cases)
    heldout_cases = heldout_cases + _complete_field_coverage_cases(heldout_cases)
    return HeartTranslationCurriculum(
        train_cases=train_cases,
        heldout_cases=heldout_cases,
        regression_cases=regression_cases,
        counterfactual_pairs=pairs,
    )


def _decoder_copy_case(
    *,
    split: str,
    name: str,
    text: str,
    referent_text: str,
    source_dialect: str,
    destination_dialect: str,
    referent_start: int | None = None,
    provenance_family: str = "heart-decoder-mechanism-v1",
    spec_family: str = "decoder-mechanism",
) -> HeartTranslationCase:
    resolved_referent_start = text.index(referent_text) if referent_start is None else referent_start
    if text[resolved_referent_start : resolved_referent_start + len(referent_text)] != referent_text:
        raise ValueError("decoder copy referent_start does not identify referent_text")
    return HeartTranslationCase(
        split=split,
        source_dialect=source_dialect,
        destination_dialect=destination_dialect,
        source_text=text,
        target_text=text,
        referent_text=referent_text,
        referent_start=resolved_referent_start,
        referent_end=resolved_referent_start + len(referent_text),
        grounding_start=0,
        grounding_end=len(text),
        signature=HeartMeaningSignature(),
        critical_classes=("grounding_provenance", "referent_identity"),
        provenance=f"synthetic:{provenance_family}:{split}:{name}",
        spec_id=f"{spec_family}:{split}:{name}",
    )


def build_heart_decoder_mechanism_curriculum() -> HeartDecoderMechanismCurriculum:
    """Build exact-copy cases from tiny through beyond-page complete-field spans."""

    train_rows = (
        ("minimal", "Mira.", "Mira", DIALECTS[0], DIALECTS[1]),
        ("capability", "Mira can run.", "Mira", DIALECTS[1], DIALECTS[2]),
        ("negation", "Tessa cannot wait.", "Tessa", DIALECTS[2], DIALECTS[0]),
        ("temporal", "Nolan will run tomorrow.", "Nolan", DIALECTS[0], DIALECTS[2]),
        ("causal", "Aria waits because Mira rests.", "Aria", DIALECTS[2], DIALECTS[1]),
        (
            "complete-field",
            "preserved context remains addressable. " * 9 + "Mira closes the record.",
            "Mira",
            DIALECTS[1],
            DIALECTS[0],
        ),
    )
    heldout_rows = (
        ("minimal", "Lena.", "Lena", DIALECTS[0], DIALECTS[2]),
        ("negation", "Omar cannot jump.", "Omar", DIALECTS[2], DIALECTS[1]),
        ("causal", "Rhea waits because Lena runs.", "Rhea", DIALECTS[1], DIALECTS[0]),
        (
            "complete-field",
            "dormant evidence remains preserved and restorable. " * 7 + "Lena opens the record.",
            "Lena",
            DIALECTS[0],
            DIALECTS[1],
        ),
    )
    return HeartDecoderMechanismCurriculum(
        train_cases=tuple(
            _decoder_copy_case(
                split="train",
                name=name,
                text=text,
                referent_text=referent,
                source_dialect=source,
                destination_dialect=destination,
            )
            for name, text, referent, source, destination in train_rows
        ),
        heldout_cases=tuple(
            _decoder_copy_case(
                split="heldout",
                name=name,
                text=text,
                referent_text=referent,
                source_dialect=source,
                destination_dialect=destination,
            )
            for name, text, referent, source, destination in heldout_rows
        ),
    )


def _identity_length_bucket(length: int, page_chars: int = 256) -> str:
    if length <= 32:
        return "tiny_001_032"
    if length <= 64:
        return "short_033_064"
    if length <= 128:
        return "medium_065_128"
    if length < page_chars:
        return "long_129_255"
    if length == page_chars:
        return "boundary_256"
    if length <= page_chars * 2:
        return "multipage_257_512"
    return "extrapolation_513_plus"


def _deterministic_identity_text(length: int, seed: int) -> str:
    if length < 1:
        raise ValueError("identity text length must be positive")
    alphabet = default_alphabet()
    rng = random.Random(seed)
    # A position-dependent permutation prevents a repeated phrase from becoming
    # a shortcut while keeping every artifact reproducible and inspectable.
    return "".join(alphabet[(rng.randrange(len(alphabet)) + index * 17) % len(alphabet)] for index in range(length))


def _generated_identity_case(
    *,
    split: str,
    name: str,
    text: str,
    dialect_index: int,
    provenance_family: str = "heart-decoder-generalization-v1",
    spec_family: str = "decoder-generalization",
) -> HeartTranslationCase:
    referent_start = min(max(0, len(text) // 2), max(0, len(text) - 4))
    referent_end = min(len(text), referent_start + 4)
    referent_text = "".join(text[index] for index in range(referent_start, referent_end))
    return _decoder_copy_case(
        split=split,
        name=name,
        text=text,
        referent_text=referent_text,
        referent_start=referent_start,
        source_dialect=DIALECTS[dialect_index % len(DIALECTS)],
        destination_dialect=DIALECTS[(dialect_index + 1) % len(DIALECTS)],
        provenance_family=provenance_family,
        spec_family=spec_family,
    )


def build_heart_decoder_generalization_curriculum() -> HeartDecoderGeneralizationCurriculum:
    """Build diverse unseen identity cases, replay, and head/middle/tail probes."""

    mechanism = build_heart_decoder_mechanism_curriculum()
    train_cases: list[HeartTranslationCase] = list(mechanism.train_cases)
    train_lengths = (8, 16, 24, 32, 48, 64, 96, 128, 192, 255, 256, 257, 320, 384, 448)
    for variant in range(2):
        for index, length in enumerate(train_lengths):
            train_cases.append(
                _generated_identity_case(
                    split="train",
                    name=f"generated-v{variant}-n{length}",
                    text=_deterministic_identity_text(length, 2026082500 + variant * 100 + index),
                    dialect_index=index + variant,
                )
            )
    alphabet_text = "".join(default_alphabet())
    train_cases.append(
        _generated_identity_case(
            split="train",
            name="complete-substrate-alphabet",
            text=alphabet_text,
            dialect_index=2,
        )
    )

    heldout_cases: list[HeartTranslationCase] = list(mechanism.heldout_cases)
    heldout_lengths = (10, 20, 40, 72, 112, 176, 240, 258, 352, 512, 521)
    for index, length in enumerate(heldout_lengths):
        heldout_cases.append(
            _generated_identity_case(
                split="heldout",
                name=f"unseen-n{length}",
                text=_deterministic_identity_text(length, 2026082600 + index),
                dialect_index=index + 1,
            )
        )

    pairs: list[HeartDecoderCounterfactualPair] = []
    for pair_index, (kind, length, position) in enumerate(
        (
            ("head", 73, 0),
            ("middle", 269, 134),
            ("tail", 521, 520),
        )
    ):
        left_text = _deterministic_identity_text(length, 2026082700 + pair_index)
        alphabet = default_alphabet()
        left_character = left_text[position]
        right_character = alphabet[(alphabet.index(left_character) + 1) % len(alphabet)]
        right_characters = list(left_text)
        right_characters[position] = right_character
        right_text = "".join(right_characters)
        left = _generated_identity_case(
            split="heldout",
            name=f"counterfactual-{kind}-left",
            text=left_text,
            dialect_index=pair_index,
        )
        right = _generated_identity_case(
            split="heldout",
            name=f"counterfactual-{kind}-right",
            text=right_text,
            dialect_index=pair_index,
        )
        heldout_cases.extend((left, right))
        pairs.append(
            HeartDecoderCounterfactualPair(
                position_kind=kind,
                left_case_id=left.case_id,
                right_case_id=right.case_id,
                changed_position=position,
                left_character=left_character,
                right_character=right_character,
            )
        )

    return HeartDecoderGeneralizationCurriculum(
        train_cases=tuple(train_cases),
        heldout_cases=tuple(heldout_cases),
        replay_case_ids=tuple(case.case_id for case in mechanism.train_cases),
        counterfactual_pairs=tuple(pairs),
    )


def _position_mutation_cases(
    *,
    split: str,
    name: str,
    length: int,
    position: int,
    seed: int,
    dialect_index: int,
    probe_label: str,
) -> tuple[HeartTranslationCase, HeartTranslationCase, HeartDecoderCounterfactualPair]:
    if position >= length:
        raise ValueError("position mutation must address a character inside its complete source")
    left_text = _deterministic_identity_text(length, seed)
    alphabet = default_alphabet()
    left_character = left_text[position]
    right_character = alphabet[(alphabet.index(left_character) + 1) % len(alphabet)]
    right_characters = list(left_text)
    right_characters[position] = right_character
    right_text = "".join(right_characters)
    left = _generated_identity_case(
        split=split,
        name=f"{name}-left",
        text=left_text,
        dialect_index=dialect_index,
        provenance_family="heart-decoder-long-position-v1",
        spec_family="decoder-long-position",
    )
    right = _generated_identity_case(
        split=split,
        name=f"{name}-right",
        text=right_text,
        dialect_index=dialect_index,
        provenance_family="heart-decoder-long-position-v1",
        spec_family="decoder-long-position",
    )
    relative_position = position / max(1, length - 1)
    position_kind = "head" if relative_position < 0.25 else "tail" if relative_position >= 0.75 else "middle"
    return (
        left,
        right,
        HeartDecoderCounterfactualPair(
            position_kind=position_kind,
            left_case_id=left.case_id,
            right_case_id=right.case_id,
            changed_position=position,
            left_character=left_character,
            right_character=right_character,
            probe_label=probe_label,
        ),
    )


def build_heart_decoder_long_position_curriculum() -> HeartDecoderGeneralizationCurriculum:
    """Add page-boundary and late-tail identity work without creating a model ceiling."""

    base = build_heart_decoder_generalization_curriculum()
    train_cases = list(base.train_cases)
    heldout_cases = list(base.heldout_cases)
    heldout_pairs = list(base.counterfactual_pairs)

    for index, length in enumerate((255, 256, 257, 384, 511, 512, 513, 576, 640)):
        train_cases.append(
            _generated_identity_case(
                split="train",
                name=f"long-position-n{length}",
                text=_deterministic_identity_text(length, 2026082800 + index),
                dialect_index=index,
                provenance_family="heart-decoder-long-position-v1",
                spec_family="decoder-long-position",
            )
        )

    train_probes = (
        (320, 254, "before-first-page-end"),
        (320, 255, "first-page-last"),
        (320, 256, "second-page-first"),
        (320, 257, "after-second-page-start"),
        (640, 510, "before-second-page-end"),
        (640, 511, "second-page-last"),
        (640, 512, "third-page-first"),
        (640, 513, "after-third-page-start"),
        (640, 639, "late-train-tail"),
    )
    for index, (length, position, label) in enumerate(train_probes):
        left, right, _ = _position_mutation_cases(
            split="train",
            name=f"train-probe-{label}",
            length=length,
            position=position,
            seed=2026082900 + index,
            dialect_index=index,
            probe_label=label,
        )
        train_cases.extend((left, right))

    for index, length in enumerate((641, 700, 769)):
        heldout_cases.append(
            _generated_identity_case(
                split="heldout",
                name=f"unseen-long-position-n{length}",
                text=_deterministic_identity_text(length, 2026083000 + index),
                dialect_index=index + 1,
                provenance_family="heart-decoder-long-position-v1",
                spec_family="decoder-long-position",
            )
        )

    heldout_probes = (
        (641, 0, "long-head"),
        (385, 254, "heldout-before-first-page-end"),
        (385, 255, "heldout-first-page-last"),
        (385, 256, "heldout-second-page-first"),
        (385, 257, "heldout-after-second-page-start"),
        (641, 510, "heldout-before-second-page-end"),
        (641, 511, "heldout-second-page-last"),
        (641, 512, "heldout-third-page-first"),
        (641, 513, "heldout-after-third-page-start"),
        (700, 699, "heldout-late-tail"),
        (769, 768, "heldout-extrapolated-tail"),
    )
    for index, (length, position, label) in enumerate(heldout_probes):
        left, right, pair = _position_mutation_cases(
            split="heldout",
            name=label,
            length=length,
            position=position,
            seed=2026083100 + index,
            dialect_index=index + 2,
            probe_label=label,
        )
        heldout_cases.extend((left, right))
        heldout_pairs.append(pair)

    return HeartDecoderGeneralizationCurriculum(
        train_cases=tuple(train_cases),
        heldout_cases=tuple(heldout_cases),
        replay_case_ids=tuple(case.case_id for case in base.train_cases),
        counterfactual_pairs=tuple(heldout_pairs),
    )


def _label_index(name: str, value: str) -> int:
    return HEART_SEMANTIC_LABELS[name].index(value)


def collate_heart_translation_cases(
    model: HeartTranslationCore,
    cases: Sequence[HeartTranslationCase],
) -> HeartTranslationBatch:
    rows = tuple(cases)
    if not rows:
        raise ValueError("cannot collate an empty Heart translation batch")
    max_target = max(len(case.target_text) for case in rows) + 1
    batch = len(rows)
    codec = D64HeartCodec()
    frames = tuple(
        codec.compile(
            SharedFieldSnapshot.from_texts(
                {LogicalRegion.USER_INPUT: case.source_text},
                tick_id=0,
                source_manifest_ids=(case.case_id,),
                source="heart_translation_curriculum",
                provenance=case.provenance,
            )
        )
        for case in rows
    )
    cardiac = codec.batch_for_model(model, frames)
    decoder_input = torch.full((batch, max_target), model.decoder_pad_index, dtype=torch.long)
    target_ids = torch.zeros((batch, max_target), dtype=torch.long)
    target_mask = torch.zeros((batch, max_target), dtype=torch.bool)
    source_dialects = torch.empty(batch, dtype=torch.long)
    destination_dialects = torch.empty(batch, dtype=torch.long)
    semantic_targets = {name: torch.empty(batch, dtype=torch.long) for name in HEART_SEMANTIC_LABELS}
    referent_start = torch.empty(batch, dtype=torch.long)
    referent_end = torch.empty(batch, dtype=torch.long)
    grounding_start = torch.empty(batch, dtype=torch.long)
    grounding_end = torch.empty(batch, dtype=torch.long)

    for row, case in enumerate(rows):
        target = [model.char_to_index[char] for char in case.target_text]
        decoder_input[row, 0] = model.bos_index
        if target:
            decoder_input[row, 1 : len(target) + 1] = torch.tensor(target, dtype=torch.long)
            target_ids[row, : len(target)] = torch.tensor(target, dtype=torch.long)
        target_ids[row, len(target)] = model.eos_index
        target_mask[row, : len(target) + 1] = True
        source_dialects[row] = DIALECT_TO_ID[case.source_dialect]
        destination_dialects[row] = DIALECT_TO_ID[case.destination_dialect]
        for name in HEART_SEMANTIC_LABELS:
            semantic_targets[name][row] = _label_index(name, getattr(case.signature, name))
        referent_start[row] = case.referent_start
        referent_end[row] = case.referent_end - 1
        grounding_start[row] = case.grounding_start
        grounding_end[row] = case.grounding_end - 1

    return HeartTranslationBatch(
        source_indices=cardiac.source_indices,
        source_cells16=cardiac.source_cells16,
        source_positions=cardiac.source_positions,
        source_mask=cardiac.source_mask,
        source_frame_ids=cardiac.frame_ids,
        source_field_ids=cardiac.field_ids,
        source_rail_ids=cardiac.rail_ids,
        source_dialect_ids=source_dialects,
        destination_dialect_ids=destination_dialects,
        decoder_input_ids=decoder_input,
        target_ids=target_ids,
        target_mask=target_mask,
        semantic_targets=semantic_targets,
        referent_start=referent_start,
        referent_end=referent_end,
        grounding_start=grounding_start,
        grounding_end=grounding_end,
        cases=rows,
    )


def heart_translation_loss(
    model: HeartTranslationCore,
    batch: HeartTranslationBatch,
    objective: HeartTranslationTrainingObjective | None = None,
) -> HeartTranslationLoss:
    resolved_objective = objective or HeartTranslationTrainingObjective()
    output = model(
        batch.source_indices,
        batch.source_mask,
        batch.source_dialect_ids,
        batch.destination_dialect_ids,
        batch.decoder_input_ids,
        source_cells16=batch.source_cells16,
        source_positions=batch.source_positions,
    )
    gathered = output.target_log_probs.gather(2, batch.target_ids.unsqueeze(-1)).squeeze(-1)
    translation = -(gathered * batch.target_mask.to(gathered.dtype)).sum() / batch.target_mask.sum().clamp_min(1)
    semantic_terms = [
        F.cross_entropy(output.semantic_logits[name], batch.semantic_targets[name])
        for name in HEART_SEMANTIC_LABELS
    ]
    semantic = torch.stack(semantic_terms).mean()
    pointer_terms = (
        F.cross_entropy(output.referent_start_logits, batch.referent_start),
        F.cross_entropy(output.referent_end_logits, batch.referent_end),
        F.cross_entropy(output.grounding_start_logits, batch.grounding_start),
        F.cross_entropy(output.grounding_end_logits, batch.grounding_end),
    )
    pointers = torch.stack(pointer_terms).mean()
    total = (
        resolved_objective.translation_weight * translation
        + resolved_objective.semantic_weight * semantic
        + resolved_objective.pointer_weight * pointers
    )
    return HeartTranslationLoss(total=total, translation=translation, semantic=semantic, pointers=pointers)


def heart_decoder_generalization_loss(
    model: HeartTranslationCore,
    batch: HeartTranslationBatch,
    objective: HeartDecoderGeneralizationObjective | None = None,
) -> HeartDecoderGeneralizationLoss:
    resolved_objective = objective or HeartDecoderGeneralizationObjective()
    if any(case.source_text != case.target_text for case in batch.cases):
        raise ValueError("decoder generalization loss requires exact identity cases")
    output = model(
        batch.source_indices,
        batch.source_mask,
        batch.source_dialect_ids,
        batch.destination_dialect_ids,
        batch.decoder_input_ids,
        source_cells16=batch.source_cells16,
        source_positions=batch.source_positions,
    )
    gathered = output.target_log_probs.gather(2, batch.target_ids.unsqueeze(-1)).squeeze(-1)
    translation = -(gathered * batch.target_mask.to(gathered.dtype)).sum() / batch.target_mask.sum().clamp_min(1)

    character_mask = batch.target_mask & batch.target_ids.ne(model.eos_index)
    positions = torch.arange(batch.target_ids.shape[1], device=batch.target_ids.device)
    source_width = batch.source_indices.shape[1]
    diagonal_indices = positions.clamp(max=max(0, source_width - 1)).view(1, -1, 1).expand(
        batch.target_ids.shape[0], -1, 1
    )
    diagonal_mass = output.decoder_trace.memory_attention.gather(2, diagonal_indices).squeeze(-1)
    diagonal_alignment = -(
        diagonal_mass.clamp_min(torch.finfo(diagonal_mass.dtype).tiny).log()
        * character_mask.to(diagonal_mass.dtype)
    ).sum() / character_mask.sum().clamp_min(1)

    generation_gate = output.decoder_trace.generation_gate.squeeze(-1)
    copy_probability = (1.0 - generation_gate).clamp_min(torch.finfo(generation_gate.dtype).tiny)
    copy_route = -(
        copy_probability.log() * character_mask.to(copy_probability.dtype)
    ).sum() / character_mask.sum().clamp_min(1)
    eos_mask = batch.target_mask & batch.target_ids.eq(model.eos_index)
    eos_route = -(
        generation_gate.clamp_min(torch.finfo(generation_gate.dtype).tiny).log()
        * eos_mask.to(generation_gate.dtype)
    ).sum() / eos_mask.sum().clamp_min(1)

    semantic_terms = [
        F.cross_entropy(output.semantic_logits[name], batch.semantic_targets[name])
        for name in HEART_SEMANTIC_LABELS
    ]
    semantic = torch.stack(semantic_terms).mean()
    pointer_terms = (
        F.cross_entropy(output.referent_start_logits, batch.referent_start),
        F.cross_entropy(output.referent_end_logits, batch.referent_end),
        F.cross_entropy(output.grounding_start_logits, batch.grounding_start),
        F.cross_entropy(output.grounding_end_logits, batch.grounding_end),
    )
    pointers = torch.stack(pointer_terms).mean()
    total = (
        resolved_objective.translation_weight * translation
        + resolved_objective.diagonal_alignment_weight * diagonal_alignment
        + resolved_objective.copy_route_weight * copy_route
        + resolved_objective.eos_route_weight * eos_route
        + resolved_objective.semantic_weight * semantic
        + resolved_objective.pointer_weight * pointers
    )
    return HeartDecoderGeneralizationLoss(
        total=total,
        translation=translation,
        diagonal_alignment=diagonal_alignment,
        copy_route=copy_route,
        eos_route=eos_route,
        semantic=semantic,
        pointers=pointers,
    )


def deterministic_length_bucketed_batches(
    cases: Sequence[HeartTranslationCase],
    *,
    batch_size: int,
    steps: int,
    seed: int,
    page_chars: int = 256,
) -> tuple[tuple[HeartTranslationCase, ...], ...]:
    """Visit every case once per epoch while never padding across length buckets."""

    rows = tuple(cases)
    if not rows or batch_size < 1 or steps < 1:
        raise ValueError("length-bucketed batches require cases and positive batch/step counts")
    rng = random.Random(seed)
    by_bucket: dict[str, list[HeartTranslationCase]] = {}
    for case in rows:
        by_bucket.setdefault(_identity_length_bucket(len(case.source_text), page_chars), []).append(case)
    bucket_names = tuple(sorted(by_bucket))
    batches: list[tuple[HeartTranslationCase, ...]] = []
    while len(batches) < steps:
        epoch_chunks: dict[str, list[tuple[HeartTranslationCase, ...]]] = {}
        for bucket in bucket_names:
            shuffled = list(by_bucket[bucket])
            rng.shuffle(shuffled)
            epoch_chunks[bucket] = [
                tuple(shuffled[start : start + batch_size])
                for start in range(0, len(shuffled), batch_size)
            ]
        while any(epoch_chunks.values()) and len(batches) < steps:
            for bucket in bucket_names:
                if epoch_chunks[bucket] and len(batches) < steps:
                    batches.append(epoch_chunks[bucket].pop(0))
    return tuple(batches)


def deterministic_training_batches(
    curriculum: HeartTranslationCurriculum,
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> tuple[tuple[HeartTranslationCase, ...], ...]:
    if batch_size < 1 or steps < 1:
        raise ValueError("batch_size and steps must be positive")
    rng = random.Random(seed)
    cases = list(curriculum.train_cases)
    epoch: list[HeartTranslationCase] = []
    batches: list[tuple[HeartTranslationCase, ...]] = []
    for _ in range(steps):
        batch: list[HeartTranslationCase] = []
        while len(batch) < batch_size:
            if not epoch:
                epoch = list(cases)
                rng.shuffle(epoch)
            take = min(batch_size - len(batch), len(epoch))
            batch.extend(epoch[:take])
            del epoch[:take]
        batches.append(tuple(batch))
    return tuple(batches)


def _semantic_predictions(output) -> dict[str, list[int]]:
    return {name: output.semantic_logits[name].argmax(dim=-1).tolist() for name in HEART_SEMANTIC_LABELS}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float | None:
    support = int(mask.sum().item())
    if support == 0:
        return None
    return float(values[mask].mean().item())


def _greedy_divergence(
    generated_text: str,
    target_text: str,
    terminated: bool,
) -> tuple[int | None, int]:
    for index, (generated_char, target_char) in enumerate(
        zip(generated_text, target_text, strict=False)
    ):
        if generated_char != target_char:
            return index, index
    shared = min(len(generated_text), len(target_text))
    if generated_text == target_text and terminated:
        return None, len(target_text)
    return shared, shared


def evaluate_heart_decoder_diagnostics(
    model: HeartTranslationCore,
    cases: Sequence[HeartTranslationCase],
    *,
    descriptor_id: str,
    checkpoint_id: str,
    curriculum_id: str,
    split: str,
) -> HeartDecoderDiagnosticReport:
    """Localize optimization, EOS, exposure, copy-gate, and alignment failure."""

    rows = tuple(cases)
    if not rows:
        raise ValueError("decoder diagnostics require at least one case")
    if not descriptor_id or not checkpoint_id or not curriculum_id or not split:
        raise ValueError("decoder diagnostic identity must be non-empty")
    model.eval()
    batch = collate_heart_translation_cases(model, rows).to(model.device)
    with torch.no_grad():
        output = model(
            batch.source_indices,
            batch.source_mask,
            batch.source_dialect_ids,
            batch.destination_dialect_ids,
            batch.decoder_input_ids,
            source_cells16=batch.source_cells16,
            source_positions=batch.source_positions,
        )
        generated = model.greedy_translate(
            batch.source_indices,
            batch.source_mask,
            batch.source_dialect_ids,
            batch.destination_dialect_ids,
            max_chars=max(len(case.target_text) for case in rows) + 12,
            source_cells16=batch.source_cells16,
            source_positions=batch.source_positions,
        )

    predictions = output.target_log_probs.argmax(dim=-1)
    character_mask = batch.target_mask & batch.target_ids.ne(model.eos_index)
    token_correct = predictions.eq(batch.target_ids)
    character_correct = token_correct & character_mask
    sequence_exact = (token_correct | ~batch.target_mask).all(dim=1)
    eos_correct = torch.tensor(
        [
            bool(predictions[row, len(case.target_text)].item() == model.eos_index)
            for row, case in enumerate(rows)
        ],
        dtype=torch.bool,
        device=predictions.device,
    )

    attention = output.decoder_trace.memory_attention
    generation_gate = output.decoder_trace.generation_gate.squeeze(-1)
    source_matches_target = (
        batch.source_indices.unsqueeze(1).eq(batch.target_ids.unsqueeze(-1))
        & batch.source_mask.unsqueeze(1)
    )
    copyable = source_matches_target.any(dim=-1) & character_mask
    noncopyable = ~source_matches_target.any(dim=-1) & character_mask
    target_attention_mass = (
        attention * source_matches_target.to(attention.dtype)
    ).sum(dim=-1)
    attention_peak = attention.max(dim=-1).values

    case_diagnostics: list[HeartDecoderCaseDiagnostic] = []
    greedy_exact: list[bool] = []
    prefix_fractions: list[float] = []
    for row, (case, generated_result) in enumerate(zip(rows, generated, strict=True)):
        target_characters = len(case.target_text)
        correct_characters = int(character_correct[row].sum().item())
        divergence, correct_prefix = _greedy_divergence(
            generated_result.text,
            case.target_text,
            generated_result.terminated,
        )
        exact = generated_result.terminated and generated_result.text == case.target_text
        greedy_exact.append(exact)
        prefix_fractions.append(correct_prefix / target_characters)
        row_target_mask = batch.target_mask[row]
        row_character_mask = character_mask[row]
        mean_gate = _masked_mean(generation_gate[row], row_target_mask)
        mean_peak = _masked_mean(attention_peak[row], row_character_mask)
        if mean_gate is None or mean_peak is None:
            raise RuntimeError("decoder diagnostic target unexpectedly has no characters")
        case_diagnostics.append(
            HeartDecoderCaseDiagnostic(
                case_id=case.case_id,
                target_characters=target_characters,
                teacher_forced_characters_correct=correct_characters,
                teacher_forced_character_accuracy=correct_characters / target_characters,
                teacher_forced_eos_correct=bool(eos_correct[row].item()),
                teacher_forced_sequence_exact=bool(sequence_exact[row].item()),
                greedy_text=generated_result.text,
                greedy_terminated=generated_result.terminated,
                greedy_exact=exact,
                greedy_first_divergence=divergence,
                greedy_correct_prefix_characters=correct_prefix,
                mean_generation_gate=mean_gate,
                mean_copyable_generation_gate=_masked_mean(generation_gate[row], copyable[row]),
                mean_noncopyable_generation_gate=_masked_mean(generation_gate[row], noncopyable[row]),
                mean_target_character_attention_mass=_masked_mean(
                    target_attention_mass[row], copyable[row]
                ),
                mean_attention_peak=mean_peak,
            )
        )

    position_accuracy: list[tuple[int, int, float]] = []
    for position in range(character_mask.shape[1]):
        support = int(character_mask[:, position].sum().item())
        if support:
            correct = int(character_correct[:, position].sum().item())
            position_accuracy.append((position, support, correct / support))

    total_characters = int(character_mask.sum().item())
    return HeartDecoderDiagnosticReport(
        descriptor_id=descriptor_id,
        checkpoint_id=checkpoint_id,
        curriculum_id=curriculum_id,
        split=split,
        execution_device=str(model.device),
        case_count=len(rows),
        teacher_forced_character_accuracy=int(character_correct.sum().item()) / total_characters,
        teacher_forced_eos_accuracy=float(eos_correct.float().mean().item()),
        teacher_forced_sequence_exact_rate=float(sequence_exact.float().mean().item()),
        greedy_exact_rate=sum(greedy_exact) / len(rows),
        greedy_termination_rate=sum(item.terminated for item in generated) / len(rows),
        mean_greedy_correct_prefix_fraction=sum(prefix_fractions) / len(prefix_fractions),
        mean_generation_gate=_masked_mean(generation_gate, batch.target_mask) or 0.0,
        mean_copyable_generation_gate=_masked_mean(generation_gate, copyable),
        mean_noncopyable_generation_gate=_masked_mean(generation_gate, noncopyable),
        mean_target_character_attention_mass=_masked_mean(target_attention_mass, copyable),
        mean_attention_peak=_masked_mean(attention_peak, character_mask) or 0.0,
        position_accuracy=tuple(position_accuracy),
        case_diagnostics=tuple(case_diagnostics),
    )


def evaluate_heart_decoder_generalization(
    model: HeartTranslationCore,
    curriculum: HeartDecoderGeneralizationCurriculum,
    *,
    descriptor_id: str,
    checkpoint_id: str,
    split: str,
) -> tuple[HeartDecoderDiagnosticReport, HeartDecoderGeneralizationEvidence]:
    """Measure unseen identity copying, diagonal alignment, and source dependence."""

    cases = curriculum.heldout_cases
    diagnostic = evaluate_heart_decoder_diagnostics(
        model,
        cases,
        descriptor_id=descriptor_id,
        checkpoint_id=checkpoint_id,
        curriculum_id=curriculum.curriculum_id,
        split=split,
    )
    batch = collate_heart_translation_cases(model, cases).to(model.device)
    with torch.no_grad():
        output = model(
            batch.source_indices,
            batch.source_mask,
            batch.source_dialect_ids,
            batch.destination_dialect_ids,
            batch.decoder_input_ids,
            source_cells16=batch.source_cells16,
            source_positions=batch.source_positions,
        )
    attention = output.decoder_trace.memory_attention
    character_mask = batch.target_mask & batch.target_ids.ne(model.eos_index)
    positions = torch.arange(batch.target_ids.shape[1], device=batch.target_ids.device)
    diagonal_indices = positions.clamp(max=max(0, batch.source_indices.shape[1] - 1)).view(1, -1, 1).expand(
        batch.target_ids.shape[0], -1, 1
    )
    diagonal_mass = attention.gather(2, diagonal_indices).squeeze(-1)
    mean_diagonal = _masked_mean(diagonal_mass, character_mask)
    if mean_diagonal is None:
        raise RuntimeError("decoder generalization evidence has no identity characters")
    diagonal_top1 = attention.argmax(dim=-1).eq(positions.view(1, -1)) & character_mask
    diagonal_top1_rate = float(diagonal_top1.sum().item()) / int(character_mask.sum().item())

    case_index = {case.case_id: index for index, case in enumerate(cases)}
    predictions = output.target_log_probs.argmax(dim=-1)
    counterfactual_results: list[dict[str, Any]] = []
    pair_passes: list[bool] = []
    margins: list[float] = []
    for pair in curriculum.counterfactual_pairs:
        left_row = case_index[pair.left_case_id]
        right_row = case_index[pair.right_case_id]
        position = pair.changed_position
        left_index = model.char_to_index[pair.left_character]
        right_index = model.char_to_index[pair.right_character]
        left_margin = float(
            (
                output.target_log_probs[left_row, position, left_index]
                - output.target_log_probs[left_row, position, right_index]
            ).item()
        )
        right_margin = float(
            (
                output.target_log_probs[right_row, position, right_index]
                - output.target_log_probs[right_row, position, left_index]
            ).item()
        )
        distribution_delta = float(
            (
                output.target_log_probs[left_row, position]
                - output.target_log_probs[right_row, position]
            ).abs().max().item()
        )
        left_correct = int(predictions[left_row, position].item()) == left_index
        right_correct = int(predictions[right_row, position].item()) == right_index
        passed = left_correct and right_correct and left_margin > 0.0 and right_margin > 0.0 and distribution_delta > 0.0
        pair_passes.append(passed)
        margins.extend((left_margin, right_margin))
        counterfactual_results.append(
            {
                **pair.to_canonical_dict(),
                "left_prediction_correct": left_correct,
                "right_prediction_correct": right_correct,
                "left_target_margin": left_margin,
                "right_target_margin": right_margin,
                "distribution_max_abs_delta": distribution_delta,
                "passed": passed,
            }
        )

    diagnostic_by_id = {item.case_id: item for item in diagnostic.case_diagnostics}
    train_targets = {case.target_text for case in curriculum.train_cases}
    memorized_train_outputs = sum(
        item.greedy_text in train_targets and item.greedy_text != case.source_text
        for case in cases
        for item in (diagnostic_by_id[case.case_id],)
    )
    bucket_metrics: list[tuple[str, float, float, int]] = []
    buckets = sorted({_identity_length_bucket(len(case.source_text)) for case in cases})
    for bucket in buckets:
        bucket_cases = [case for case in cases if _identity_length_bucket(len(case.source_text)) == bucket]
        bucket_diagnostics = [diagnostic_by_id[case.case_id] for case in bucket_cases]
        characters = sum(item.target_characters for item in bucket_diagnostics)
        correct = sum(item.teacher_forced_characters_correct for item in bucket_diagnostics)
        bucket_metrics.append(
            (
                bucket,
                correct / characters,
                sum(item.greedy_exact for item in bucket_diagnostics) / len(bucket_diagnostics),
                len(bucket_diagnostics),
            )
        )

    evidence = HeartDecoderGeneralizationEvidence(
        descriptor_id=descriptor_id,
        checkpoint_id=checkpoint_id,
        curriculum_id=curriculum.curriculum_id,
        split=split,
        decoder_diagnostic_id=diagnostic.diagnostic_id,
        mean_diagonal_attention_mass=mean_diagonal,
        diagonal_attention_top1_rate=diagonal_top1_rate,
        counterfactual_pair_exact_rate=sum(pair_passes) / len(pair_passes),
        minimum_counterfactual_target_margin=min(margins),
        memorized_train_output_count=memorized_train_outputs,
        length_bucket_metrics=tuple(bucket_metrics),
        counterfactual_results=tuple(counterfactual_results),
    )
    return diagnostic, evidence


def _analyze_batch(model: HeartTranslationCore, batch: HeartTranslationBatch):
    bos = torch.full(
        (batch.source_indices.shape[0], 1),
        model.bos_index,
        dtype=torch.long,
        device=batch.source_indices.device,
    )
    return model(
        batch.source_indices,
        batch.source_mask,
        batch.source_dialect_ids,
        batch.destination_dialect_ids,
        bos,
        source_cells16=batch.source_cells16,
        source_positions=batch.source_positions,
    )


def evaluate_heart_translation_model(
    model: HeartTranslationCore,
    curriculum: HeartTranslationCurriculum,
    *,
    descriptor_id: str,
) -> HeartTranslationEvaluationReport:
    model.eval()
    cases = curriculum.heldout_cases
    device = model.device
    batch = collate_heart_translation_cases(model, cases).to(device)
    with torch.no_grad():
        source_output = _analyze_batch(model, batch)
        generation_results = model.greedy_translate(
            batch.source_indices,
            batch.source_mask,
            batch.source_dialect_ids,
            batch.destination_dialect_ids,
            max_chars=max(len(case.target_text) for case in cases) + 12,
            source_cells16=batch.source_cells16,
            source_positions=batch.source_positions,
        )
    generated = tuple(result.text for result in generation_results)
    terminated = tuple(result.terminated for result in generation_results)
    source_predictions = _semantic_predictions(source_output)
    source_semantic_ok: list[bool] = []
    referent_pointer_ok: list[bool] = []
    grounding_pointer_ok: list[bool] = []
    for index, case in enumerate(cases):
        source_semantic_ok.append(
            all(source_predictions[name][index] == int(batch.semantic_targets[name][index]) for name in HEART_SEMANTIC_LABELS)
        )
        referent_pointer_ok.append(
            int(source_output.referent_start_logits[index].argmax()) == case.referent_start
            and int(source_output.referent_end_logits[index].argmax()) == case.referent_end - 1
        )
        grounding_pointer_ok.append(
            int(source_output.grounding_start_logits[index].argmax()) == case.grounding_start
            and int(source_output.grounding_end_logits[index].argmax()) == case.grounding_end - 1
        )

    reverse_ok = [False] * len(cases)
    reverse_predictions: dict[str, dict[int, int]] = {name: {} for name in HEART_SEMANTIC_LABELS}
    nonempty_indices = [index for index, text in enumerate(generated) if terminated[index] and text]
    if nonempty_indices:
        reverse_cases: list[HeartTranslationCase] = []
        reverse_map: list[int] = []
        for index in nonempty_indices:
            case = cases[index]
            text = generated[index]
            try:
                assert_supported_text(text)
            except Exception:
                continue
            referent = case.referent_text if case.referent_text in text else text[:1]
            if not referent:
                continue
            referent_start = text.index(referent)
            reverse_cases.append(
                HeartTranslationCase(
                    split="heldout",
                    source_dialect=case.destination_dialect,
                    destination_dialect=case.source_dialect,
                    source_text=text,
                    target_text=case.source_text,
                    referent_text=referent,
                    referent_start=referent_start,
                    referent_end=referent_start + len(referent),
                    grounding_start=referent_start,
                    grounding_end=referent_start + len(referent),
                    signature=case.signature,
                    critical_classes=case.critical_classes,
                    provenance=case.provenance + ":generated-roundtrip",
                    spec_id=case.spec_id + ":generated-roundtrip:" + str(index),
                )
            )
            reverse_map.append(index)
        if reverse_cases:
            reverse_batch = collate_heart_translation_cases(model, reverse_cases).to(device)
            with torch.no_grad():
                reverse_output = _analyze_batch(model, reverse_batch)
            preds = _semantic_predictions(reverse_output)
            for row, original_index in enumerate(reverse_map):
                ok = True
                for name in HEART_SEMANTIC_LABELS:
                    expected = _label_index(name, getattr(cases[original_index].signature, name))
                    reverse_predictions[name][original_index] = preds[name][row]
                    ok = ok and preds[name][row] == expected
                reverse_ok[original_index] = ok

    target_exact = [
        terminated[index] and generated[index] == case.target_text
        for index, case in enumerate(cases)
    ]
    referent_preserved = [
        terminated[index] and case.referent_text in generated[index]
        for index, case in enumerate(cases)
    ]
    grounded_roundtrip = [
        source_semantic_ok[index]
        and reverse_ok[index]
        and target_exact[index]
        and referent_pointer_ok[index]
        and grounding_pointer_ok[index]
        for index in range(len(cases))
    ]
    aggregate_semantic = [
        source_semantic_ok[index] and reverse_ok[index] and referent_preserved[index]
        for index in range(len(cases))
    ]
    case_results = tuple(
        HeartTranslationCaseResult(
            case_id=case.case_id,
            generated_text=generated[index],
            generated_characters=generation_results[index].generated_characters,
            terminated=terminated[index],
            target_exact=target_exact[index],
            source_semantic_exact=source_semantic_ok[index],
            reverse_semantic_exact=reverse_ok[index],
            referent_pointer_exact=referent_pointer_ok[index],
            grounding_pointer_exact=grounding_pointer_ok[index],
            referent_preserved=referent_preserved[index],
            grounded_roundtrip=grounded_roundtrip[index],
            aggregate_semantic_fidelity=aggregate_semantic[index],
        )
        for index, case in enumerate(cases)
    )

    critical_rates: dict[str, float] = {}
    for semantic_class in CRITICAL_SEMANTIC_CLASSES:
        relevant = [index for index, case in enumerate(cases) if semantic_class in case.critical_classes]
        if not relevant:
            raise RuntimeError(f"heldout curriculum has no cases for critical class {semantic_class}")
        successes: list[bool] = []
        for index in relevant:
            case = cases[index]
            if semantic_class == "referent_identity":
                successes.append(referent_pointer_ok[index] and referent_preserved[index])
            elif semantic_class == "grounding_provenance":
                successes.append(grounding_pointer_ok[index])
            else:
                expected = _label_index(semantic_class, getattr(case.signature, semantic_class))
                successes.append(
                    source_predictions[semantic_class][index] == expected
                    and reverse_predictions[semantic_class].get(index) == expected
                )
        critical_rates[semantic_class] = sum(successes) / len(successes)

    case_by_id = {case.case_id: (index, case) for index, case in enumerate(cases)}
    pair_coverage: dict[str, bool] = {}
    for pair in curriculum.counterfactual_pairs:
        left_index, left = case_by_id[pair.left_case_id]
        right_index, right = case_by_id[pair.right_case_id]
        if pair.semantic_class == "referent_identity":
            passed = (
                referent_pointer_ok[left_index]
                and referent_pointer_ok[right_index]
                and left.referent_text != right.referent_text
                and left.referent_text in generated[left_index]
                and right.referent_text in generated[right_index]
            )
        elif pair.semantic_class == "grounding_provenance":
            passed = (
                grounding_pointer_ok[left_index]
                and grounding_pointer_ok[right_index]
                and (left.grounding_start, left.grounding_end) != (right.grounding_start, right.grounding_end)
            )
        else:
            expected_left = _label_index(pair.semantic_class, getattr(left.signature, pair.semantic_class))
            expected_right = _label_index(pair.semantic_class, getattr(right.signature, pair.semantic_class))
            pred_left = source_predictions[pair.semantic_class][left_index]
            pred_right = source_predictions[pair.semantic_class][right_index]
            passed = expected_left != expected_right and pred_left == expected_left and pred_right == expected_right and pred_left != pred_right
        pair_coverage[pair.semantic_class] = bool(passed)
    counterfactual_proven = set(pair_coverage) == set(CRITICAL_SEMANTIC_CLASSES) and all(pair_coverage.values())

    regression_failures = 0
    if curriculum.regression_cases:
        regression_batch = collate_heart_translation_cases(model, curriculum.regression_cases).to(device)
        with torch.no_grad():
            regression_output = _analyze_batch(model, regression_batch)
        regression_predictions = _semantic_predictions(regression_output)
        for row, case in enumerate(curriculum.regression_cases):
            if not all(
                regression_predictions[name][row] == _label_index(name, getattr(case.signature, name))
                for name in HEART_SEMANTIC_LABELS
            ):
                regression_failures += 1

    metric_payload = {
        "schema": HEART_EVALUATION_SCHEMA,
        "descriptor_id": descriptor_id,
        "curriculum_id": curriculum.curriculum_id,
        "heldout_case_ids": [case.case_id for case in cases],
        "generated_sha256": canonical_sha256(
            [
                {
                    "text": result.text,
                    "terminated": result.terminated,
                    "generated_characters": result.generated_characters,
                }
                for result in generation_results
            ]
        ),
        "case_results_sha256": canonical_sha256(
            [item.to_canonical_dict() for item in case_results]
        ),
        "translation_termination_rate": sum(terminated) / len(cases),
        "grounded_roundtrip_rate": sum(grounded_roundtrip) / len(cases),
        "aggregate_semantic_fidelity": sum(aggregate_semantic) / len(cases),
        "critical_class_rates": critical_rates,
        "counterfactual_use_proven": counterfactual_proven,
        "regression_failures": regression_failures,
    }
    evaluation_id = canonical_sha256(metric_payload)
    evidence = HeartSemanticFidelityEvidence(
        descriptor_id=descriptor_id,
        evaluation_id=evaluation_id,
        heldout_case_count=len(cases),
        grounded_roundtrip_rate=metric_payload["grounded_roundtrip_rate"],
        aggregate_semantic_fidelity=metric_payload["aggregate_semantic_fidelity"],
        critical_class_rates=tuple(critical_rates.items()),
        counterfactual_use_proven=counterfactual_proven,
        regression_failures=regression_failures,
    )
    return HeartTranslationEvaluationReport(
        descriptor_id=descriptor_id,
        curriculum_id=curriculum.curriculum_id,
        evaluation_id=evaluation_id,
        heldout_case_count=len(cases),
        translation_exact_rate=sum(target_exact) / len(cases),
        translation_termination_rate=metric_payload["translation_termination_rate"],
        source_semantic_exact_rate=sum(source_semantic_ok) / len(cases),
        reverse_semantic_exact_rate=sum(reverse_ok) / len(cases),
        referent_pointer_rate=sum(referent_pointer_ok) / len(cases),
        grounding_pointer_rate=sum(grounding_pointer_ok) / len(cases),
        evidence=evidence,
        case_results=case_results,
    )


__all__ = [
    "DIALECTS",
    "DIALECT_TO_ID",
    "HEART_CASE_SCHEMA",
    "HEART_CURRICULUM_SCHEMA",
    "HEART_DECODER_CASE_DIAGNOSTIC_SCHEMA",
    "HEART_DECODER_COUNTERFACTUAL_SCHEMA",
    "HEART_DECODER_DIAGNOSTIC_SCHEMA",
    "HEART_DECODER_GENERALIZATION_CURRICULUM_SCHEMA",
    "HEART_DECODER_GENERALIZATION_EVIDENCE_SCHEMA",
    "HEART_DECODER_GENERALIZATION_OBJECTIVE_SCHEMA",
    "HEART_DECODER_MECHANISM_CURRICULUM_SCHEMA",
    "HEART_EVALUATION_SCHEMA",
    "HEART_TRAINING_RECIPE_SCHEMA",
    "CounterfactualPair",
    "HeartDecoderCaseDiagnostic",
    "HeartDecoderCounterfactualPair",
    "HeartDecoderDiagnosticReport",
    "HeartDecoderGeneralizationCurriculum",
    "HeartDecoderGeneralizationEvidence",
    "HeartDecoderGeneralizationLoss",
    "HeartDecoderGeneralizationObjective",
    "HeartDecoderMechanismCurriculum",
    "HeartMeaningSignature",
    "HeartTranslationBatch",
    "HeartTranslationCase",
    "HeartTranslationCaseResult",
    "HeartTranslationCurriculum",
    "HeartTranslationEvaluationReport",
    "HeartTranslationLoss",
    "HeartTranslationTrainingObjective",
    "HeartTranslationTrainingRecipe",
    "build_heart_decoder_generalization_curriculum",
    "build_heart_decoder_long_position_curriculum",
    "build_heart_decoder_mechanism_curriculum",
    "build_heart_translation_curriculum",
    "collate_heart_translation_cases",
    "deterministic_length_bucketed_batches",
    "deterministic_training_batches",
    "evaluate_heart_decoder_diagnostics",
    "evaluate_heart_decoder_generalization",
    "evaluate_heart_translation_model",
    "heart_decoder_generalization_loss",
    "heart_translation_loss",
]
