"""Stable Semantic Cortex service contracts.

This file is permanent interface anatomy only.  It does not instantiate,
train, or activate a semantic specialist, and it does not open the Heart's
``semantic_cortex`` valve.  Implementations must remain proposal/observation
sources; canonical materialization is Heart authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from runtime.field import canonical_sha256

SEMANTIC_SERVICE_SCHEMA = "axon-semantic-cortex-service-v1"


@dataclass(frozen=True, slots=True)
class ExactEvidenceRef:
    """Pointer back to exact authoritative evidence."""

    source_kind: str
    source_id: str
    provenance: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_kind", "source_id", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str) or len(self.sha256) != 64
        ):
            raise ValueError("sha256 must be None or a 64-character hexadecimal digest")
        if self.sha256 is not None:
            try:
                int(self.sha256, 16)
            except ValueError as exc:
                raise ValueError("sha256 must be hexadecimal") from exc

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "provenance": self.provenance,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticQuery:
    """One grounded query offered to a specialist lane."""

    lane: str
    source_field_id: str
    source_tick_id: int
    text: str
    evidence_refs: tuple[ExactEvidenceRef, ...] = ()
    query_id: str = ""

    def __post_init__(self) -> None:
        for name in ("lane", "source_field_id", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.source_tick_id, bool) or not isinstance(self.source_tick_id, int) or self.source_tick_id < 0:
            raise ValueError("source_tick_id must be a non-negative integer")
        refs = tuple(self.evidence_refs)
        if not all(isinstance(item, ExactEvidenceRef) for item in refs):
            raise TypeError("evidence_refs must contain ExactEvidenceRef values")
        object.__setattr__(self, "evidence_refs", refs)
        if not self.query_id:
            object.__setattr__(self, "query_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SEMANTIC_SERVICE_SCHEMA,
            "kind": "query",
            "lane": self.lane,
            "source_field_id": self.source_field_id,
            "source_tick_id": self.source_tick_id,
            "text": self.text,
            "evidence_refs": [item.to_canonical_dict() for item in self.evidence_refs],
        }
        if include_id:
            payload["query_id"] = self.query_id
        return payload


@dataclass(frozen=True, slots=True)
class SemanticObservation:
    """One specialist's latest noncanonical semantic observation/proposal."""

    lane: str
    source_field_id: str
    source_tick_id: int
    microtick_sequence: int
    model_generation: str
    subject: str
    relation: str
    object: str
    confidence: float
    relevance: float
    novelty: float
    evidence_refs: tuple[ExactEvidenceRef, ...]
    retrieval_candidate_ids: tuple[str, ...] = ()
    observation_id: str = ""

    def __post_init__(self) -> None:
        for name in ("lane", "source_field_id", "model_generation", "subject", "relation", "object"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("source_tick_id", "microtick_sequence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("confidence", "relevance", "novelty"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)
        refs = tuple(self.evidence_refs)
        if not refs:
            raise ValueError("SemanticObservation requires exact evidence_refs")
        if not all(isinstance(item, ExactEvidenceRef) for item in refs):
            raise TypeError("evidence_refs must contain ExactEvidenceRef values")
        object.__setattr__(self, "evidence_refs", refs)
        candidate_ids = tuple(str(item) for item in self.retrieval_candidate_ids)
        if any(not item for item in candidate_ids):
            raise ValueError("retrieval_candidate_ids may not contain empty identifiers")
        object.__setattr__(self, "retrieval_candidate_ids", candidate_ids)
        if not self.observation_id:
            object.__setattr__(
                self,
                "observation_id",
                canonical_sha256(self.to_canonical_dict(include_id=False)),
            )

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SEMANTIC_SERVICE_SCHEMA,
            "kind": "observation",
            "lane": self.lane,
            "source_field_id": self.source_field_id,
            "source_tick_id": self.source_tick_id,
            "microtick_sequence": self.microtick_sequence,
            "model_generation": self.model_generation,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "novelty": self.novelty,
            "evidence_refs": [item.to_canonical_dict() for item in self.evidence_refs],
            "retrieval_candidate_ids": list(self.retrieval_candidate_ids),
        }
        if include_id:
            payload["observation_id"] = self.observation_id
        return payload


@runtime_checkable
class SemanticSpecialist(Protocol):
    """Width/architecture-agnostic specialist service interface."""

    @property
    def lane(self) -> str: ...

    @property
    def model_generation(self) -> str: ...

    def observe(self, query: SemanticQuery) -> SemanticObservation: ...


__all__ = [
    "SEMANTIC_SERVICE_SCHEMA",
    "ExactEvidenceRef",
    "SemanticQuery",
    "SemanticObservation",
    "SemanticSpecialist",
]
