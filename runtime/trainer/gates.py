"""Deterministic evaluation gates for Trainer candidate generations."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from runtime.field import canonical_sha256

EVALUATION_OBSERVATION_SCHEMA = "axon-trainer-evaluation-observation-v1"
EVALUATION_REQUIREMENT_SCHEMA = "axon-trainer-evaluation-requirement-v1"
PROMOTION_GATE_SCHEMA = "axon-trainer-promotion-gate-v1"
PROMOTION_GATE_DECISION_SCHEMA = "axon-trainer-promotion-gate-decision-v1"


class MetricComparison(str, Enum):
    GREATER_OR_EQUAL = "gte"
    LESS_OR_EQUAL = "lte"


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class EvaluationObservation:
    suite_id: str
    module_id: str
    candidate_generation_id: str
    metrics: tuple[tuple[str, float], ...]
    artifact_id: str | None = None
    observation_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in ("suite_id", "module_id", "candidate_generation_id"):
            object.__setattr__(self, label, _nonempty(getattr(self, label), label))
        items = tuple(sorted(((_nonempty(str(name), "metric name"), float(value)) for name, value in self.metrics)))
        names = [name for name, _ in items]
        if not items:
            raise ValueError("metrics cannot be empty")
        if len(names) != len(set(names)):
            raise ValueError("duplicate evaluation metric")
        if any(not (-float("inf") < value < float("inf")) for _, value in items):
            raise ValueError("evaluation metrics must be finite")
        object.__setattr__(self, "metrics", items)
        if self.artifact_id is not None:
            object.__setattr__(self, "artifact_id", _nonempty(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "observation_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @classmethod
    def from_mapping(
        cls,
        *,
        suite_id: str,
        module_id: str,
        candidate_generation_id: str,
        metrics: Mapping[str, float],
        artifact_id: str | None = None,
    ) -> "EvaluationObservation":
        return cls(
            suite_id=suite_id,
            module_id=module_id,
            candidate_generation_id=candidate_generation_id,
            metrics=tuple(metrics.items()),
            artifact_id=artifact_id,
        )

    def metric(self, name: str) -> float:
        name = _nonempty(name, "metric name")
        for metric_name, value in self.metrics:
            if metric_name == name:
                return value
        raise KeyError(name)

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": EVALUATION_OBSERVATION_SCHEMA,
            "suite_id": self.suite_id,
            "module_id": self.module_id,
            "candidate_generation_id": self.candidate_generation_id,
            "metrics": {name: metric for name, metric in self.metrics},
            "artifact_id": self.artifact_id,
        }
        if include_id:
            value["observation_id"] = self.observation_id
        return value


@dataclass(frozen=True, slots=True)
class EvaluationRequirement:
    suite_id: str
    metric_name: str
    comparison: MetricComparison | str
    threshold: float
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "suite_id", _nonempty(self.suite_id, "suite_id"))
        object.__setattr__(self, "metric_name", _nonempty(self.metric_name, "metric_name"))
        object.__setattr__(self, "label", _nonempty(self.label, "label"))
        comparison = self.comparison if isinstance(self.comparison, MetricComparison) else MetricComparison(self.comparison)
        object.__setattr__(self, "comparison", comparison)
        threshold = float(self.threshold)
        if not (-float("inf") < threshold < float("inf")):
            raise ValueError("threshold must be finite")
        object.__setattr__(self, "threshold", threshold)

    def passes(self, value: float) -> bool:
        if self.comparison is MetricComparison.GREATER_OR_EQUAL:
            return value >= self.threshold
        return value <= self.threshold

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema": EVALUATION_REQUIREMENT_SCHEMA,
            "suite_id": self.suite_id,
            "metric_name": self.metric_name,
            "comparison": self.comparison.value,
            "threshold": self.threshold,
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class PromotionGate:
    gate_id: str
    module_id: str
    candidate_generation_id: str
    requirements: tuple[EvaluationRequirement, ...]
    required_suite_ids: tuple[str, ...] = ()
    gate_spec_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in ("gate_id", "module_id", "candidate_generation_id"):
            object.__setattr__(self, label, _nonempty(getattr(self, label), label))
        requirements = tuple(sorted(tuple(self.requirements), key=lambda item: (item.suite_id, item.metric_name, item.label)))
        if not requirements:
            raise ValueError("promotion gate requires at least one requirement")
        if not all(isinstance(item, EvaluationRequirement) for item in requirements):
            raise TypeError("requirements must contain EvaluationRequirement values")
        object.__setattr__(self, "requirements", requirements)
        suites = set(_nonempty(str(item), "required suite id") for item in self.required_suite_ids)
        suites.update(item.suite_id for item in requirements)
        object.__setattr__(self, "required_suite_ids", tuple(sorted(suites)))
        object.__setattr__(self, "gate_spec_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PROMOTION_GATE_SCHEMA,
            "gate_id": self.gate_id,
            "module_id": self.module_id,
            "candidate_generation_id": self.candidate_generation_id,
            "required_suite_ids": list(self.required_suite_ids),
            "requirements": [item.to_canonical_dict() for item in self.requirements],
        }
        if include_id:
            value["gate_spec_id"] = self.gate_spec_id
        return value


@dataclass(frozen=True, slots=True)
class PromotionGateDecision:
    gate_spec_id: str
    gate_id: str
    module_id: str
    candidate_generation_id: str
    observation_ids: tuple[str, ...]
    missing_suite_ids: tuple[str, ...]
    failed_requirements: tuple[str, ...]
    passed: bool
    decision_id: str = field(init=False)

    def __post_init__(self) -> None:
        for label in ("gate_spec_id", "gate_id", "module_id", "candidate_generation_id"):
            object.__setattr__(self, label, _nonempty(getattr(self, label), label))
        object.__setattr__(self, "observation_ids", tuple(sorted(set(_nonempty(str(item), "observation id") for item in self.observation_ids))))
        object.__setattr__(self, "missing_suite_ids", tuple(sorted(set(_nonempty(str(item), "missing suite id") for item in self.missing_suite_ids))))
        object.__setattr__(self, "failed_requirements", tuple(sorted(set(_nonempty(str(item), "failed requirement") for item in self.failed_requirements))))
        expected = not self.missing_suite_ids and not self.failed_requirements
        if bool(self.passed) != expected:
            raise ValueError("passed must exactly reflect missing suites and failed requirements")
        object.__setattr__(self, "decision_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": PROMOTION_GATE_DECISION_SCHEMA,
            "gate_spec_id": self.gate_spec_id,
            "gate_id": self.gate_id,
            "module_id": self.module_id,
            "candidate_generation_id": self.candidate_generation_id,
            "observation_ids": list(self.observation_ids),
            "missing_suite_ids": list(self.missing_suite_ids),
            "failed_requirements": list(self.failed_requirements),
            "passed": bool(self.passed),
        }
        if include_id:
            value["decision_id"] = self.decision_id
        return value


def evaluate_promotion_gate(
    gate: PromotionGate,
    observations: tuple[EvaluationObservation, ...] | list[EvaluationObservation],
) -> PromotionGateDecision:
    if not isinstance(gate, PromotionGate):
        raise TypeError("gate must be PromotionGate")
    rows = tuple(observations)
    if not all(isinstance(item, EvaluationObservation) for item in rows):
        raise TypeError("observations must contain EvaluationObservation values")
    applicable: dict[str, EvaluationObservation] = {}
    for observation in rows:
        if observation.module_id != gate.module_id or observation.candidate_generation_id != gate.candidate_generation_id:
            raise ValueError("evaluation observation belongs to another module or candidate generation")
        if observation.suite_id in applicable:
            raise ValueError(f"duplicate observation for suite {observation.suite_id!r}")
        applicable[observation.suite_id] = observation

    missing = tuple(sorted(set(gate.required_suite_ids) - set(applicable)))
    failures: list[str] = []
    for requirement in gate.requirements:
        observation = applicable.get(requirement.suite_id)
        if observation is None:
            continue
        try:
            value = observation.metric(requirement.metric_name)
        except KeyError:
            failures.append(f"{requirement.label}:missing_metric:{requirement.suite_id}:{requirement.metric_name}")
            continue
        if not requirement.passes(value):
            failures.append(
                f"{requirement.label}:{requirement.suite_id}:{requirement.metric_name}:"
                f"{value}:{requirement.comparison.value}:{requirement.threshold}"
            )
    return PromotionGateDecision(
        gate_spec_id=gate.gate_spec_id,
        gate_id=gate.gate_id,
        module_id=gate.module_id,
        candidate_generation_id=gate.candidate_generation_id,
        observation_ids=tuple(item.observation_id for item in applicable.values()),
        missing_suite_ids=missing,
        failed_requirements=tuple(failures),
        passed=not missing and not failures,
    )


__all__ = [
    "EVALUATION_OBSERVATION_SCHEMA",
    "EVALUATION_REQUIREMENT_SCHEMA",
    "PROMOTION_GATE_SCHEMA",
    "PROMOTION_GATE_DECISION_SCHEMA",
    "MetricComparison",
    "EvaluationObservation",
    "EvaluationRequirement",
    "PromotionGate",
    "PromotionGateDecision",
    "evaluate_promotion_gate",
]
