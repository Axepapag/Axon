"""Deterministic relevance and retention audit for verified dormant evidence.

Build C.1 adds a real semantic/relevance gate without inventing learned
embeddings.  The auditor consumes only evidence that has already been exact-
dereferenced and hash/provenance verified by ``DormantEvidenceIndex``.  It may
rank, retain, reject, or budget that evidence; it never mutates canonical state
and never becomes a memory authority.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from runtime.field import LogicalRegion, SharedFieldSnapshot, canonical_sha256

from .evidence_bridge import VerifiedDormantEvidence

_RELEVANCE_TOKEN_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", flags=re.UNICODE)
AUDITOR_SCHEMA = "axon-dormant-relevance-auditor-v1"


def _normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _terms(value: str) -> frozenset[str]:
    return frozenset(
        _normalize(match.group(0))
        for match in _RELEVANCE_TOKEN_RE.finditer(value)
        if _normalize(match.group(0))
    )


def _string_values(value: object) -> tuple[str, ...]:
    """Collect bounded textual metadata without treating it as authority."""

    found: list[str] = []
    if isinstance(value, str):
        if value:
            found.append(value)
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            item = value[key]
            if isinstance(item, str) and item:
                found.append(item)
            elif isinstance(item, (list, tuple)):
                for child in item[:32]:
                    if isinstance(child, str) and child:
                        found.append(child)
    elif isinstance(value, (list, tuple)):
        for item in value[:32]:
            if isinstance(item, str) and item:
                found.append(item)
    return tuple(found)


@dataclass(frozen=True, slots=True)
class DormantRelevancePolicy:
    """Governed selection/budget policy for one dormant-recall pass."""

    max_items: int = 8
    max_chars: int = 10_000
    max_item_chars: int = 4_096
    min_score: float = 0.0

    def __post_init__(self) -> None:
        for name in ("max_items", "max_chars", "max_item_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        score = float(self.min_score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")
        object.__setattr__(self, "min_score", score)


@dataclass(frozen=True, slots=True)
class DormantRelevanceScore:
    """Explainable score for one exact verified dormant container."""

    container_id: str
    score: float
    retrieval_support: float
    lexical_support: float
    graph_support: float
    relation_support: float
    confidence_support: float
    task_support: float
    novelty_support: float
    already_active: bool
    char_count: int

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "container_id": self.container_id,
            "score": round(self.score, 12),
            "retrieval_support": round(self.retrieval_support, 12),
            "lexical_support": round(self.lexical_support, 12),
            "graph_support": round(self.graph_support, 12),
            "relation_support": round(self.relation_support, 12),
            "confidence_support": round(self.confidence_support, 12),
            "task_support": round(self.task_support, 12),
            "novelty_support": round(self.novelty_support, 12),
            "already_active": self.already_active,
            "char_count": self.char_count,
        }


@dataclass(frozen=True, slots=True)
class DormantRelevanceDecision:
    """Noncanonical receipt for one relevance/retention decision."""

    query: str
    selected: tuple[VerifiedDormantEvidence, ...]
    scores: tuple[DormantRelevanceScore, ...]
    skipped_oversize: tuple[str, ...]
    skipped_budget: tuple[str, ...]
    below_threshold: tuple[str, ...]
    fallback_used: bool
    total_chars: int

    @property
    def selected_container_ids(self) -> tuple[str, ...]:
        return tuple(item.container.container_id for item in self.selected)

    @property
    def decision_id(self) -> str:
        return canonical_sha256(
            {
                "schema": AUDITOR_SCHEMA,
                "query": self.query,
                "selected_container_ids": list(self.selected_container_ids),
                "scores": [item.to_canonical_dict() for item in self.scores],
                "skipped_oversize": list(self.skipped_oversize),
                "skipped_budget": list(self.skipped_budget),
                "below_threshold": list(self.below_threshold),
                "fallback_used": self.fallback_used,
                "total_chars": self.total_chars,
            }
        )


class DormantRelevanceAuditor:
    """Primitive-but-real deterministic semantic relevance auditor.

    The score deliberately combines the signals already owned by Axon:
    recovered lexical/edge/graph evidence, exact container confidence, simple
    task-shape cues, and redundancy/retention against the active canonical
    field.  It does not claim learned semantic similarity.
    """

    WEIGHTS: Mapping[str, float] = MappingProxyType(
        {
            "retrieval": 0.14,
            "lexical": 0.20,
            "graph": 0.06,
            "relation": 0.30,
            "confidence": 0.10,
            "task": 0.12,
            "novelty": 0.08,
        }
    )

    def __init__(self, policy: DormantRelevancePolicy | None = None) -> None:
        self.policy = policy or DormantRelevancePolicy()

    @staticmethod
    def _active_refs(field: SharedFieldSnapshot) -> frozenset[str]:
        refs: set[str] = set()
        for span in field.region(LogicalRegion.STRUCTURED_KNOWLEDGE).spans:
            refs.update(span.container_refs)
        return frozenset(refs)

    @staticmethod
    def _active_terms(field: SharedFieldSnapshot) -> frozenset[str]:
        values: list[str] = []
        for region in field.regions:
            if region.name is LogicalRegion.STRUCTURED_KNOWLEDGE:
                continue
            if region.text:
                values.append(region.text)
        return _terms(" ".join(values))

    @staticmethod
    def _evidence_terms(item: VerifiedDormantEvidence) -> frozenset[str]:
        values: list[str] = [
            item.container.text,
            item.container.normalized_text,
            item.container.kind,
            *_string_values(item.container.record.get("metadata", {})),
        ]
        for edge in item.edges:
            values.extend((edge.edge_type, edge.target, edge.source_text))
        return _terms(" ".join(value for value in values if value))

    @staticmethod
    def _task_support(query_terms: frozenset[str], kind: str, evidence_terms: frozenset[str]) -> float:
        if not query_terms:
            return 0.0
        kind_terms = _terms(kind)
        kind_match = len(query_terms & kind_terms) / len(query_terms)
        relation_cues = {"why", "cause", "causes", "because", "relation", "related"}
        identity_cues = {"what", "who", "identity", "definition", "define", "kind", "type"}
        procedure_cues = {"how", "procedure", "steps", "build", "implement", "use"}
        cue_bonus = 0.0
        lowered_kind = _normalize(kind)
        if query_terms & relation_cues and any(token in lowered_kind for token in ("edge", "relation", "cause")):
            cue_bonus = 1.0
        elif query_terms & identity_cues and any(token in lowered_kind for token in ("concept", "definition", "entity")):
            cue_bonus = 1.0
        elif query_terms & procedure_cues and any(token in lowered_kind for token in ("procedure", "instruction", "code", "method")):
            cue_bonus = 1.0
        evidence_overlap = len(query_terms & evidence_terms) / len(query_terms)
        return max(0.0, min(1.0, max(kind_match, cue_bonus, evidence_overlap)))

    def score(
        self,
        query: str,
        item: VerifiedDormantEvidence,
        field: SharedFieldSnapshot,
    ) -> DormantRelevanceScore:
        if not isinstance(field, SharedFieldSnapshot):
            raise TypeError("field must be SharedFieldSnapshot")
        query_terms = _terms(query)
        if not query_terms:
            raise ValueError("query produced no relevance terms")
        evidence_terms = self._evidence_terms(item)
        active_refs = self._active_refs(field)
        active_terms = self._active_terms(field)

        lexical_support = min(1.0, item.candidate.lexical_hits / max(1, len(query_terms)))
        graph_support = min(
            1.0,
            item.candidate.graph_hits / max(1, len(query_terms)),
        )
        relation_support = min(
            1.0,
            int(getattr(item.candidate, "relation_hits", 0)) / max(1, len(query_terms)),
        )
        retrieval_support = float(item.candidate.score) / (float(item.candidate.score) + 20.0)
        confidence_support = max(0.0, min(1.0, float(item.container.confidence)))
        task_support = self._task_support(query_terms, item.container.kind, evidence_terms)

        already_active = item.container.container_id in active_refs
        active_overlap = (
            len(evidence_terms & active_terms) / max(1, len(evidence_terms)) if evidence_terms else 0.0
        )
        # Already-active exact evidence gets a retention floor rather than being
        # discarded merely for being familiar.  New evidence is rewarded for
        # adding nonredundant terms to the rest of the active field.
        novelty_support = max(0.35 if already_active else 0.0, 1.0 - min(1.0, active_overlap))

        components = {
            "retrieval": retrieval_support,
            "lexical": lexical_support,
            "graph": graph_support,
            "relation": relation_support,
            "confidence": confidence_support,
            "task": task_support,
            "novelty": novelty_support,
        }
        total = sum(self.WEIGHTS[name] * components[name] for name in self.WEIGHTS)
        total = max(0.0, min(1.0, total))
        return DormantRelevanceScore(
            container_id=item.container.container_id,
            score=total,
            retrieval_support=retrieval_support,
            lexical_support=lexical_support,
            graph_support=graph_support,
            relation_support=relation_support,
            confidence_support=confidence_support,
            task_support=task_support,
            novelty_support=novelty_support,
            already_active=already_active,
            char_count=len(item.container.text),
        )

    def select(
        self,
        query: str,
        evidence: Sequence[VerifiedDormantEvidence],
        field: SharedFieldSnapshot,
    ) -> DormantRelevanceDecision:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(field, SharedFieldSnapshot):
            raise TypeError("field must be SharedFieldSnapshot")

        pairs = [(self.score(query, item, field), item) for item in evidence]
        pairs.sort(
            key=lambda pair: (
                -pair[0].score,
                -int(getattr(pair[1].candidate, "relation_hits", 0)),
                -pair[1].candidate.lexical_hits,
                -pair[1].candidate.graph_hits,
                pair[0].container_id,
            )
        )

        selected: list[VerifiedDormantEvidence] = []
        oversize: list[str] = []
        budget: list[str] = []
        below: list[str] = []
        total_chars = 0

        for score, item in pairs:
            container_id = score.container_id
            char_count = len(item.container.text)
            if char_count > self.policy.max_item_chars:
                oversize.append(container_id)
                continue
            if score.score < self.policy.min_score:
                below.append(container_id)
                continue
            separator_chars = 1 if selected else 0
            if len(selected) >= self.policy.max_items or total_chars + separator_chars + char_count > self.policy.max_chars:
                budget.append(container_id)
                continue
            selected.append(item)
            total_chars += separator_chars + char_count

        fallback_used = False
        if not selected:
            # Doctrine says the semantic/relevance path fails closed to exact
            # lexical retrieval.  Only a lexical-hit candidate may bypass a
            # relevance threshold, and it still must obey full-item budgets.
            for score, item in pairs:
                if item.candidate.lexical_hits <= 0:
                    continue
                char_count = len(item.container.text)
                if char_count > self.policy.max_item_chars or char_count > self.policy.max_chars:
                    continue
                selected.append(item)
                total_chars = char_count
                fallback_used = True
                break

        return DormantRelevanceDecision(
            query=query.strip(),
            selected=tuple(selected),
            scores=tuple(score for score, _ in pairs),
            skipped_oversize=tuple(oversize),
            skipped_budget=tuple(budget),
            below_threshold=tuple(below),
            fallback_used=fallback_used,
            total_chars=total_chars,
        )


__all__ = [
    "AUDITOR_SCHEMA",
    "DormantRelevancePolicy",
    "DormantRelevanceScore",
    "DormantRelevanceDecision",
    "DormantRelevanceAuditor",
]
