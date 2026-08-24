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

from substrate import assert_supported_text
from runtime.field import canonical_sha256
from runtime.heart.intelligence import (
    CRITICAL_SEMANTIC_CLASSES,
    HeartSemanticFidelityEvidence,
)
from runtime.heart.translation_core import HEART_SEMANTIC_LABELS, HeartTranslationCore


HEART_CURRICULUM_SCHEMA = "axon-heart-translation-curriculum-v1"
HEART_CASE_SCHEMA = "axon-heart-translation-case-v1"
HEART_EVALUATION_SCHEMA = "axon-heart-translation-evaluation-v1"
HEART_TRAINING_OBJECTIVE_SCHEMA = "axon-heart-translation-training-objective-v1"

DIALECTS: tuple[str, ...] = (
    "canonical_english_v1",
    "heart_explicit_v1",
    "rail_d64_v1",
)
DIALECT_TO_ID = {name: index for index, name in enumerate(DIALECTS)}
TRAIN_DIRECTIONS: tuple[tuple[str, str], ...] = tuple(
    (left, right) for left in DIALECTS for right in DIALECTS if left != right
)
EVAL_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("canonical_english_v1", "heart_explicit_v1"),
    ("heart_explicit_v1", "rail_d64_v1"),
    ("rail_d64_v1", "canonical_english_v1"),
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


@dataclass(slots=True)
class HeartTranslationBatch:
    source_indices: torch.Tensor
    source_mask: torch.Tensor
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
            source_mask=self.source_mask.to(device),
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
class HeartTranslationEvaluationReport:
    descriptor_id: str
    curriculum_id: str
    evaluation_id: str
    heldout_case_count: int
    translation_exact_rate: float
    source_semantic_exact_rate: float
    reverse_semantic_exact_rate: float
    referent_pointer_rate: float
    grounding_pointer_rate: float
    evidence: HeartSemanticFidelityEvidence

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": HEART_EVALUATION_SCHEMA,
            "descriptor_id": self.descriptor_id,
            "curriculum_id": self.curriculum_id,
            "evaluation_id": self.evaluation_id,
            "heldout_case_count": self.heldout_case_count,
            "translation_exact_rate": self.translation_exact_rate,
            "source_semantic_exact_rate": self.source_semantic_exact_rate,
            "reverse_semantic_exact_rate": self.reverse_semantic_exact_rate,
            "referent_pointer_rate": self.referent_pointer_rate,
            "grounding_pointer_rate": self.grounding_pointer_rate,
            "evidence": self.evidence.to_canonical_dict(),
        }


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

    by_spec_and_direction = {
        (case.spec_id, case.source_dialect, case.destination_dialect): case
        for case in heldout_cases
    }
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
    return HeartTranslationCurriculum(
        train_cases=train_cases,
        heldout_cases=heldout_cases,
        regression_cases=regression_cases,
        counterfactual_pairs=pairs,
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
    max_source = max(len(case.source_text) for case in rows)
    max_target = max(len(case.target_text) for case in rows) + 1
    if max_source > model.cfg.max_source_chars or max_target > model.cfg.max_target_chars + 1:
        raise ValueError("Heart curriculum case exceeds model character budget")
    batch = len(rows)
    source_indices = torch.full((batch, max_source), model.source_pad_index, dtype=torch.long)
    source_mask = torch.zeros((batch, max_source), dtype=torch.bool)
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
        source = [model.char_to_index[char] for char in case.source_text]
        target = [model.char_to_index[char] for char in case.target_text]
        source_indices[row, : len(source)] = torch.tensor(source, dtype=torch.long)
        source_mask[row, : len(source)] = True
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
        source_indices=source_indices,
        source_mask=source_mask,
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
    objective: HeartTranslationTrainingObjective = HeartTranslationTrainingObjective(),
) -> HeartTranslationLoss:
    output = model(
        batch.source_indices,
        batch.source_mask,
        batch.source_dialect_ids,
        batch.destination_dialect_ids,
        batch.decoder_input_ids,
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
        objective.translation_weight * translation
        + objective.semantic_weight * semantic
        + objective.pointer_weight * pointers
    )
    return HeartTranslationLoss(total=total, translation=translation, semantic=semantic, pointers=pointers)


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
    batches: list[tuple[HeartTranslationCase, ...]] = []
    for _ in range(steps):
        if batch_size <= len(cases):
            batches.append(tuple(rng.sample(cases, batch_size)))
        else:
            batches.append(tuple(rng.choice(cases) for _ in range(batch_size)))
    return tuple(batches)


def _semantic_predictions(output) -> dict[str, list[int]]:
    return {name: output.semantic_logits[name].argmax(dim=-1).tolist() for name in HEART_SEMANTIC_LABELS}


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
        generated = model.greedy_translate(
            batch.source_indices,
            batch.source_mask,
            batch.source_dialect_ids,
            batch.destination_dialect_ids,
            max_chars=max(len(case.target_text) for case in cases) + 12,
        )
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
    nonempty_indices = [index for index, text in enumerate(generated) if text]
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

    target_exact = [generated[index] == case.target_text for index, case in enumerate(cases)]
    referent_preserved = [case.referent_text in generated[index] for index, case in enumerate(cases)]
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
        "generated_sha256": canonical_sha256(list(generated)),
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
        source_semantic_exact_rate=sum(source_semantic_ok) / len(cases),
        reverse_semantic_exact_rate=sum(reverse_ok) / len(cases),
        referent_pointer_rate=sum(referent_pointer_ok) / len(cases),
        grounding_pointer_rate=sum(grounding_pointer_ok) / len(cases),
        evidence=evidence,
    )


__all__ = [
    "HEART_CURRICULUM_SCHEMA",
    "HEART_CASE_SCHEMA",
    "HEART_EVALUATION_SCHEMA",
    "DIALECTS",
    "DIALECT_TO_ID",
    "HeartMeaningSignature",
    "HeartTranslationCase",
    "CounterfactualPair",
    "HeartTranslationCurriculum",
    "HeartTranslationBatch",
    "HeartTranslationLoss",
    "HeartTranslationEvaluationReport",
    "build_heart_translation_curriculum",
    "collate_heart_translation_cases",
    "heart_translation_loss",
    "deterministic_training_batches",
    "evaluate_heart_translation_model",
]
