"""Grounded Semantic Cortex service contracts.

Build D.2 tightens the Cortex boundary around the accepted D64 dual surface.
A specialist may observe only an explicitly identified derived semantic
projection whose slots remain bound to exact D64 lanes and canonical source
spans.  Specialist outputs remain noncanonical observations/proposals; this
module grants no Heart authority and does not open the ``semantic_cortex``
valve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

from runtime.field import (
    D64SemanticSlot,
    D64SemanticSurface,
    canonical_sha256,
)

SEMANTIC_SERVICE_SCHEMA = "axon-semantic-cortex-service-v2"
SEMANTIC_PROJECTION_SCHEMA = "axon-semantic-projection-ref-v1"
SEMANTIC_SLOT_REF_SCHEMA = "axon-semantic-slot-ref-v1"


def _require_digest(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-character hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc
    return value


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
        if self.sha256 is not None:
            _require_digest("sha256", self.sha256)

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "provenance": self.provenance,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class SemanticSlotRef:
    """Exact grounding receipt for one selected D64 semantic slot."""

    slot_id: str
    text_sha256: str
    exact_lane_refs: tuple[int, ...]
    source_span_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_digest("slot_id", self.slot_id)
        _require_digest("text_sha256", self.text_sha256)
        lane_refs = tuple(int(item) for item in self.exact_lane_refs)
        if not lane_refs:
            raise ValueError("SemanticSlotRef.exact_lane_refs must be non-empty")
        if any(item < 0 for item in lane_refs):
            raise ValueError("SemanticSlotRef.exact_lane_refs must be non-negative")
        if any(right <= left for left, right in zip(lane_refs, lane_refs[1:])):
            raise ValueError("SemanticSlotRef.exact_lane_refs must be strictly increasing")
        object.__setattr__(self, "exact_lane_refs", lane_refs)
        span_ids = tuple(sorted(set(str(item) for item in self.source_span_ids)))
        if not span_ids or any(not item for item in span_ids):
            raise ValueError("SemanticSlotRef.source_span_ids must contain non-empty IDs")
        object.__setattr__(self, "source_span_ids", span_ids)

    @classmethod
    def from_slot(cls, slot: D64SemanticSlot) -> "SemanticSlotRef":
        if not isinstance(slot, D64SemanticSlot):
            raise TypeError("SemanticSlotRef.from_slot requires D64SemanticSlot")
        return cls(
            slot_id=slot.slot_id,
            text_sha256=slot.text_sha256,
            exact_lane_refs=slot.exact_lane_refs,
            source_span_ids=slot.source_span_ids,
        )

    def assert_matches(self, slot: D64SemanticSlot) -> None:
        if not isinstance(slot, D64SemanticSlot):
            raise TypeError("slot must be D64SemanticSlot")
        if (
            self.slot_id != slot.slot_id
            or self.text_sha256 != slot.text_sha256
            or self.exact_lane_refs != slot.exact_lane_refs
            or self.source_span_ids != slot.source_span_ids
        ):
            raise ValueError("semantic slot reference does not match the grounded D64 slot")

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_SLOT_REF_SCHEMA,
            "slot_id": self.slot_id,
            "text_sha256": self.text_sha256,
            "exact_lane_refs": list(self.exact_lane_refs),
            "source_span_ids": list(self.source_span_ids),
        }


@dataclass(frozen=True, slots=True)
class SemanticProjectionRef:
    """Fail-closed identity for the exact D64 semantic projection observed."""

    source_field_id: str
    source_tick_id: int
    rail_id: str
    semantic_surface_id: str
    feature_generation: str
    slots: tuple[SemanticSlotRef, ...]

    def __post_init__(self) -> None:
        _require_digest("source_field_id", self.source_field_id)
        _require_digest("rail_id", self.rail_id)
        _require_digest("semantic_surface_id", self.semantic_surface_id)
        if isinstance(self.source_tick_id, bool) or not isinstance(self.source_tick_id, int) or self.source_tick_id < 0:
            raise ValueError("source_tick_id must be a non-negative integer")
        if not isinstance(self.feature_generation, str) or not self.feature_generation:
            raise ValueError("feature_generation must be a non-empty string")
        slots = tuple(self.slots)
        if not slots or not all(isinstance(item, SemanticSlotRef) for item in slots):
            raise ValueError("SemanticProjectionRef requires at least one SemanticSlotRef")
        slot_ids = [item.slot_id for item in slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("SemanticProjectionRef may not contain duplicate slot IDs")
        object.__setattr__(self, "slots", slots)

    @classmethod
    def from_surface(
        cls,
        surface: D64SemanticSurface,
        *,
        slot_ids: Iterable[str] | None = None,
    ) -> "SemanticProjectionRef":
        if not isinstance(surface, D64SemanticSurface):
            raise TypeError("SemanticProjectionRef.from_surface requires D64SemanticSurface")
        selected_ids = None if slot_ids is None else tuple(str(item) for item in slot_ids)
        if selected_ids is None:
            selected = surface.slots
        else:
            wanted = set(selected_ids)
            selected = tuple(slot for slot in surface.slots if slot.slot_id in wanted)
            if set(slot.slot_id for slot in selected) != wanted:
                missing = sorted(wanted - {slot.slot_id for slot in selected})
                raise KeyError(f"semantic surface does not contain requested slots: {missing}")
        if not selected:
            raise ValueError("semantic projection cannot be empty")
        return cls(
            source_field_id=surface.source_field_id,
            source_tick_id=surface.source_tick_id,
            rail_id=surface.source_rail_id,
            semantic_surface_id=surface.surface_id,
            feature_generation=surface.feature_generation,
            slots=tuple(SemanticSlotRef.from_slot(slot) for slot in selected),
        )

    def assert_matches(self, surface: D64SemanticSurface) -> None:
        if not isinstance(surface, D64SemanticSurface):
            raise TypeError("surface must be D64SemanticSurface")
        if (
            self.source_field_id != surface.source_field_id
            or self.source_tick_id != surface.source_tick_id
            or self.rail_id != surface.source_rail_id
            or self.semantic_surface_id != surface.surface_id
            or self.feature_generation != surface.feature_generation
        ):
            raise ValueError("semantic projection identity is stale or bound to another D64 surface")
        by_id = {slot.slot_id: slot for slot in surface.slots}
        for slot_ref in self.slots:
            slot = by_id.get(slot_ref.slot_id)
            if slot is None:
                raise ValueError("semantic projection references a slot absent from the surface")
            slot_ref.assert_matches(slot)

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "schema": SEMANTIC_PROJECTION_SCHEMA,
            "source_field_id": self.source_field_id,
            "source_tick_id": self.source_tick_id,
            "rail_id": self.rail_id,
            "semantic_surface_id": self.semantic_surface_id,
            "feature_generation": self.feature_generation,
            "slots": [item.to_canonical_dict() for item in self.slots],
        }


@dataclass(frozen=True, slots=True)
class SemanticQuery:
    """One grounded query offered to a specialist lane."""

    lane: str
    text: str
    projection: SemanticProjectionRef
    evidence_refs: tuple[ExactEvidenceRef, ...] = ()
    query_id: str = ""

    def __post_init__(self) -> None:
        for name in ("lane", "text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.projection, SemanticProjectionRef):
            raise TypeError("projection must be SemanticProjectionRef")
        refs = tuple(self.evidence_refs)
        if not all(isinstance(item, ExactEvidenceRef) for item in refs):
            raise TypeError("evidence_refs must contain ExactEvidenceRef values")
        object.__setattr__(self, "evidence_refs", refs)
        if not self.query_id:
            object.__setattr__(self, "query_id", canonical_sha256(self.to_canonical_dict(include_id=False)))
        else:
            _require_digest("query_id", self.query_id)

    @property
    def source_field_id(self) -> str:
        return self.projection.source_field_id

    @property
    def source_tick_id(self) -> int:
        return self.projection.source_tick_id

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SEMANTIC_SERVICE_SCHEMA,
            "kind": "query",
            "lane": self.lane,
            "text": self.text,
            "projection": self.projection.to_canonical_dict(),
            "evidence_refs": [item.to_canonical_dict() for item in self.evidence_refs],
        }
        if include_id:
            payload["query_id"] = self.query_id
        return payload


@dataclass(frozen=True, slots=True)
class SemanticObservation:
    """One specialist's latest noncanonical semantic observation/proposal."""

    lane: str
    query_id: str
    projection: SemanticProjectionRef
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
        for name in ("lane", "model_generation", "subject", "relation", "object"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        _require_digest("query_id", self.query_id)
        if not isinstance(self.projection, SemanticProjectionRef):
            raise TypeError("projection must be SemanticProjectionRef")
        if isinstance(self.microtick_sequence, bool) or not isinstance(self.microtick_sequence, int) or self.microtick_sequence < 0:
            raise ValueError("microtick_sequence must be a non-negative integer")
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
        else:
            _require_digest("observation_id", self.observation_id)

    @property
    def source_field_id(self) -> str:
        return self.projection.source_field_id

    @property
    def source_tick_id(self) -> int:
        return self.projection.source_tick_id

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SEMANTIC_SERVICE_SCHEMA,
            "kind": "observation",
            "lane": self.lane,
            "query_id": self.query_id,
            "projection": self.projection.to_canonical_dict(),
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
    "SEMANTIC_PROJECTION_SCHEMA",
    "SEMANTIC_SLOT_REF_SCHEMA",
    "ExactEvidenceRef",
    "SemanticSlotRef",
    "SemanticProjectionRef",
    "SemanticQuery",
    "SemanticObservation",
    "SemanticSpecialist",
]
