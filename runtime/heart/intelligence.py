"""Governed contracts for learned Heart translation/conduction tissue.

The canonical Shared Field remains truth.  Learned Heart translators are
non-authoritative senses that decode/encode organ dialects and propose grounded
meaning transport.  Deterministic Heart authority validates fidelity evidence
before a translator generation may serve; translators never gain canonical
write authority from this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from runtime.field import canonical_sha256

HEART_TRANSLATOR_SCHEMA = "axon-heart-translator-descriptor-v1"
HEART_ENSEMBLE_POLICY_SCHEMA = "axon-heart-intelligence-ensemble-policy-v1"
HEART_FIDELITY_EVIDENCE_SCHEMA = "axon-heart-semantic-fidelity-evidence-v1"
HEART_PROMOTION_DECISION_SCHEMA = "axon-heart-translator-promotion-decision-v1"


class HeartTranslatorRole(str, Enum):
    """Learned cardiac roles; none of these are canonical authority classes."""

    TRANSLATOR = "translator"
    TRANSLATION_CRITIC = "translation_critic"
    AMBIGUITY_CRITIC = "ambiguity_critic"


class HeartTranslatorState(str, Enum):
    SERVING = "serving"
    CANDIDATE = "candidate"
    OFFLINE = "offline"
    RETIRED = "retired"


CRITICAL_SEMANTIC_CLASSES: tuple[str, ...] = (
    "referent_identity",
    "polarity_negation",
    "modality",
    "quantification",
    "temporal_relation",
    "causal_relation",
    "speech_act",
    "grounding_provenance",
)


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _rate(value: float, label: str) -> float:
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class HeartTranslatorDescriptor:
    """Immutable identity of one learned Heart translator generation."""

    translator_id: str
    generation_id: str
    role: HeartTranslatorRole | str
    architecture: str
    d_model: int
    source_dialects: tuple[str, ...]
    destination_dialects: tuple[str, ...]
    state: HeartTranslatorState | str = HeartTranslatorState.CANDIDATE
    descriptor_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "translator_id", _nonempty(self.translator_id, "translator_id"))
        object.__setattr__(self, "generation_id", _nonempty(self.generation_id, "generation_id"))
        object.__setattr__(self, "architecture", _nonempty(self.architecture, "architecture"))
        role = self.role if isinstance(self.role, HeartTranslatorRole) else HeartTranslatorRole(self.role)
        state = self.state if isinstance(self.state, HeartTranslatorState) else HeartTranslatorState(self.state)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "state", state)
        if isinstance(self.d_model, bool) or not isinstance(self.d_model, int) or self.d_model < 64:
            raise ValueError("Heart translator d_model must be an integer >= 64")
        sources = tuple(sorted(set(_nonempty(str(item), "source dialect") for item in self.source_dialects)))
        destinations = tuple(sorted(set(_nonempty(str(item), "destination dialect") for item in self.destination_dialects)))
        if not sources or not destinations:
            raise ValueError("Heart translator must declare source and destination dialects")
        object.__setattr__(self, "source_dialects", sources)
        object.__setattr__(self, "destination_dialects", destinations)
        object.__setattr__(self, "descriptor_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_TRANSLATOR_SCHEMA,
            "translator_id": self.translator_id,
            "generation_id": self.generation_id,
            "role": self.role.value,
            "architecture": self.architecture,
            "d_model": self.d_model,
            "source_dialects": list(self.source_dialects),
            "destination_dialects": list(self.destination_dialects),
            "state": self.state.value,
        }
        if include_id:
            value["descriptor_id"] = self.descriptor_id
        return value


@dataclass(frozen=True, slots=True)
class HeartEnsemblePolicy:
    """Mature serving/candidate topology and non-negotiable fidelity floor."""

    minimum_serving_translators: int = 3
    minimum_candidate_lanes: int = 1
    minimum_grounded_roundtrip_rate: float = 0.9999
    minimum_aggregate_semantic_fidelity: float = 0.999
    require_all_critical_classes_perfect: bool = True
    require_counterfactual_use_proof: bool = True
    policy_id: str = field(init=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_serving_translators, bool)
            or not isinstance(self.minimum_serving_translators, int)
            or self.minimum_serving_translators < 3
        ):
            raise ValueError("Heart ensemble requires at least three serving translators")
        if (
            isinstance(self.minimum_candidate_lanes, bool)
            or not isinstance(self.minimum_candidate_lanes, int)
            or self.minimum_candidate_lanes < 1
        ):
            raise ValueError("Heart ensemble requires at least one isolated candidate lane")
        object.__setattr__(
            self,
            "minimum_grounded_roundtrip_rate",
            _rate(self.minimum_grounded_roundtrip_rate, "minimum_grounded_roundtrip_rate"),
        )
        object.__setattr__(
            self,
            "minimum_aggregate_semantic_fidelity",
            _rate(self.minimum_aggregate_semantic_fidelity, "minimum_aggregate_semantic_fidelity"),
        )
        object.__setattr__(self, "policy_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_ENSEMBLE_POLICY_SCHEMA,
            "minimum_serving_translators": self.minimum_serving_translators,
            "minimum_candidate_lanes": self.minimum_candidate_lanes,
            "minimum_grounded_roundtrip_rate": self.minimum_grounded_roundtrip_rate,
            "minimum_aggregate_semantic_fidelity": self.minimum_aggregate_semantic_fidelity,
            "require_all_critical_classes_perfect": self.require_all_critical_classes_perfect,
            "require_counterfactual_use_proof": self.require_counterfactual_use_proof,
            "critical_semantic_classes": list(CRITICAL_SEMANTIC_CLASSES),
        }
        if include_id:
            value["policy_id"] = self.policy_id
        return value


@dataclass(frozen=True, slots=True)
class HeartSemanticFidelityEvidence:
    """Held-out evidence for one candidate translator generation.

    Rates are measured by the evaluation harness; this contract does not infer
    meaning from text.  Critical class scores allow a deterministic promotion
    gate to reject a candidate that hides a catastrophic distinction behind a
    high aggregate average.
    """

    descriptor_id: str
    evaluation_id: str
    heldout_case_count: int
    grounded_roundtrip_rate: float
    aggregate_semantic_fidelity: float
    critical_class_rates: tuple[tuple[str, float], ...]
    counterfactual_use_proven: bool
    regression_failures: int = 0
    evidence_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptor_id", _nonempty(self.descriptor_id, "descriptor_id"))
        object.__setattr__(self, "evaluation_id", _nonempty(self.evaluation_id, "evaluation_id"))
        if isinstance(self.heldout_case_count, bool) or not isinstance(self.heldout_case_count, int) or self.heldout_case_count <= 0:
            raise ValueError("heldout_case_count must be a positive integer")
        if isinstance(self.regression_failures, bool) or not isinstance(self.regression_failures, int) or self.regression_failures < 0:
            raise ValueError("regression_failures must be a non-negative integer")
        object.__setattr__(self, "grounded_roundtrip_rate", _rate(self.grounded_roundtrip_rate, "grounded_roundtrip_rate"))
        object.__setattr__(self, "aggregate_semantic_fidelity", _rate(self.aggregate_semantic_fidelity, "aggregate_semantic_fidelity"))
        items = tuple(sorted(((_nonempty(str(name), "critical class"), _rate(rate, f"critical rate {name}")) for name, rate in self.critical_class_rates)))
        names = [name for name, _ in items]
        if len(names) != len(set(names)):
            raise ValueError("critical_class_rates contains duplicate classes")
        if set(names) != set(CRITICAL_SEMANTIC_CLASSES):
            raise ValueError("critical_class_rates must cover every required critical semantic class exactly once")
        object.__setattr__(self, "critical_class_rates", items)
        object.__setattr__(self, "evidence_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_FIDELITY_EVIDENCE_SCHEMA,
            "descriptor_id": self.descriptor_id,
            "evaluation_id": self.evaluation_id,
            "heldout_case_count": self.heldout_case_count,
            "grounded_roundtrip_rate": self.grounded_roundtrip_rate,
            "aggregate_semantic_fidelity": self.aggregate_semantic_fidelity,
            "critical_class_rates": {name: rate for name, rate in self.critical_class_rates},
            "counterfactual_use_proven": bool(self.counterfactual_use_proven),
            "regression_failures": self.regression_failures,
        }
        if include_id:
            value["evidence_id"] = self.evidence_id
        return value


@dataclass(frozen=True, slots=True)
class HeartTranslationPromotionDecision:
    descriptor_id: str
    policy_id: str
    evidence_id: str
    passed: bool
    reasons: tuple[str, ...]
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptor_id", _nonempty(self.descriptor_id, "descriptor_id"))
        object.__setattr__(self, "policy_id", _nonempty(self.policy_id, "policy_id"))
        object.__setattr__(self, "evidence_id", _nonempty(self.evidence_id, "evidence_id"))
        reasons = tuple(_nonempty(str(item), "promotion reason") for item in self.reasons)
        if self.passed and reasons:
            raise ValueError("passed promotion decisions cannot carry failure reasons")
        if not self.passed and not reasons:
            raise ValueError("failed promotion decisions require at least one reason")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "decision_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": HEART_PROMOTION_DECISION_SCHEMA,
            "descriptor_id": self.descriptor_id,
            "policy_id": self.policy_id,
            "evidence_id": self.evidence_id,
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
        }
        if include_id:
            value["decision_id"] = self.decision_id
        return value


def evaluate_heart_translator_promotion(
    descriptor: HeartTranslatorDescriptor,
    evidence: HeartSemanticFidelityEvidence,
    policy: HeartEnsemblePolicy,
) -> HeartTranslationPromotionDecision:
    """Apply the deterministic Heart promotion floor to held-out evidence."""

    if evidence.descriptor_id != descriptor.descriptor_id:
        raise ValueError("fidelity evidence is bound to a different Heart translator descriptor")
    reasons: list[str] = []
    if evidence.grounded_roundtrip_rate < policy.minimum_grounded_roundtrip_rate:
        reasons.append("grounded_roundtrip_below_floor")
    if evidence.aggregate_semantic_fidelity < policy.minimum_aggregate_semantic_fidelity:
        reasons.append("aggregate_semantic_fidelity_below_floor")
    if policy.require_all_critical_classes_perfect:
        imperfect = [name for name, rate in evidence.critical_class_rates if rate != 1.0]
        if imperfect:
            reasons.append("critical_semantic_class_not_perfect:" + ",".join(imperfect))
    if policy.require_counterfactual_use_proof and not evidence.counterfactual_use_proven:
        reasons.append("counterfactual_use_not_proven")
    if evidence.regression_failures:
        reasons.append(f"regression_failures:{evidence.regression_failures}")
    return HeartTranslationPromotionDecision(
        descriptor_id=descriptor.descriptor_id,
        policy_id=policy.policy_id,
        evidence_id=evidence.evidence_id,
        passed=not reasons,
        reasons=tuple(reasons),
    )
