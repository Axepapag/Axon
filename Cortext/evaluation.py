"""Evaluation-first consumer of the accepted grounded D64 dual surface.

This module is deliberately a baseline, not a trained Semantic Cortex organ.
It asks a narrow measurable question: once the existing dormant retriever has
admitted a candidate pool, how well can the current deterministic
``structural-lexical-v1`` D64 semantic surface rerank exact target containers?

Each case is materialized as a real ``SharedFieldSnapshot``, compiled through
the exact D64 rail and the accepted semantic surface, and bound through the
Cortex projection contract.  No canonical branch is mutated and the Heart's
``semantic_cortex`` valve remains closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from runtime.dormant.evaluation import DormantForwardEvaluationCase
from runtime.dormant.evidence_bridge import DormantEvidenceIndex, VerifiedDormantEvidence
from runtime.dormant.relevance import DormantRelevanceAuditor, DormantRelevancePolicy
from runtime.field import (
    D64FieldCompiler,
    D64SemanticSlot,
    D64SemanticSurfaceCompiler,
    FieldSpan,
    LogicalRegion,
    RegionState,
    SharedFieldSnapshot,
    canonical_sha256,
)
from substrate import InvalidUnicodeScalarError, encode_unicode_text

from .contracts import ExactEvidenceRef, SemanticProjectionRef, SemanticQuery

D64_RERANK_EVALUATION_SCHEMA = "axon-d64-specialist-baseline-evaluation-v1"
D64_RERANK_BASELINE_GENERATION = "d64-structural-lexical-cosine-v1"


def _reciprocal_rank(ranking: Sequence[str], expected: Sequence[str]) -> float:
    expected_set = set(expected)
    for index, container_id in enumerate(ranking, start=1):
        if container_id in expected_set:
            return 1.0 / index
    return 0.0


def _case_span_id(case_id: str, role: str, value: str) -> str:
    return f"d2-{role}-{canonical_sha256({'case_id': case_id, 'role': role, 'value': value})[:24]}"


@dataclass(frozen=True, slots=True)
class D64RerankCaseResult:
    case: DormantForwardEvaluationCase
    query_id: str
    evaluation_field_id: str
    rail_id: str
    semantic_surface_id: str
    feature_generation: str
    query_slot_id: str
    pool_container_ids: tuple[str, ...]
    structural_top_k_container_ids: tuple[str, ...]
    audited_top_k_container_ids: tuple[str, ...]
    pool_hit: bool
    structural_reciprocal_rank: float
    audited_reciprocal_rank: float
    exact_character_count: int
    semantic_slot_count: int
    unsupported_candidate_ids: tuple[str, ...] = ()

    @property
    def structural_hit(self) -> bool:
        return self.structural_reciprocal_rank > 0.0

    @property
    def audited_hit(self) -> bool:
        return self.audited_reciprocal_rank > 0.0

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            **self.case.to_canonical_dict(),
            "query_id": self.query_id,
            "evaluation_field_id": self.evaluation_field_id,
            "rail_id": self.rail_id,
            "semantic_surface_id": self.semantic_surface_id,
            "feature_generation": self.feature_generation,
            "query_slot_id": self.query_slot_id,
            "pool_container_ids": list(self.pool_container_ids),
            "structural_top_k_container_ids": list(self.structural_top_k_container_ids),
            "audited_top_k_container_ids": list(self.audited_top_k_container_ids),
            "pool_hit": self.pool_hit,
            "structural_reciprocal_rank": self.structural_reciprocal_rank,
            "structural_hit": self.structural_hit,
            "audited_reciprocal_rank": self.audited_reciprocal_rank,
            "audited_hit": self.audited_hit,
            "exact_character_count": self.exact_character_count,
            "semantic_slot_count": self.semantic_slot_count,
            "unsupported_candidate_ids": list(self.unsupported_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class D64RerankEvaluationResult:
    index_id: str
    k: int
    pool_limit: int
    baseline_generation: str
    feature_generation: str
    cases: tuple[D64RerankCaseResult, ...]
    pool_recall: float
    structural_hit_at_k: float
    structural_mean_reciprocal_rank: float
    audited_hit_at_k: float
    audited_mean_reciprocal_rank: float
    unsupported_candidate_count: int

    @property
    def evaluation_id(self) -> str:
        return canonical_sha256(
            {
                "schema": D64_RERANK_EVALUATION_SCHEMA,
                "index_id": self.index_id,
                "k": self.k,
                "pool_limit": self.pool_limit,
                "baseline_generation": self.baseline_generation,
                "feature_generation": self.feature_generation,
                "cases": [item.to_canonical_dict() for item in self.cases],
            }
        )

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "schema": D64_RERANK_EVALUATION_SCHEMA,
            "evaluation_id": self.evaluation_id,
            "index_id": self.index_id,
            "k": self.k,
            "pool_limit": self.pool_limit,
            "baseline_generation": self.baseline_generation,
            "feature_generation": self.feature_generation,
            "case_count": len(self.cases),
            "pool_recall": self.pool_recall,
            "structural_hit_at_k": self.structural_hit_at_k,
            "structural_mean_reciprocal_rank": self.structural_mean_reciprocal_rank,
            "audited_hit_at_k": self.audited_hit_at_k,
            "audited_mean_reciprocal_rank": self.audited_mean_reciprocal_rank,
            "unsupported_candidate_count": self.unsupported_candidate_count,
            "cases": [item.to_canonical_dict() for item in self.cases],
        }


class StructuralLexicalD64Reranker:
    """Transparent no-training baseline over grounded D64 field-span slots."""

    generation = D64_RERANK_BASELINE_GENERATION

    def __init__(self) -> None:
        self._exact_compiler = D64FieldCompiler()
        self._semantic_compiler = D64SemanticSurfaceCompiler()

    @staticmethod
    def _materialize_snapshot(
        case: DormantForwardEvaluationCase,
        evidence: Sequence[VerifiedDormantEvidence],
    ) -> tuple[SharedFieldSnapshot, str, dict[str, str]]:
        query_span_id = _case_span_id(case.case_id, "query", case.query)
        query_region = RegionState(
            name=LogicalRegion.USER_INPUT,
            spans=(
                FieldSpan(
                    span_id=query_span_id,
                    text=case.query,
                    kind="semantic_eval_text",
                    source="d2-held-out-query",
                    provenance=f"d2:{case.case_id}:query",
                    container_refs=(case.source_container_id,),
                ),
            ),
        )
        knowledge_spans: list[FieldSpan] = []
        candidate_span_ids: dict[str, str] = {}
        for position, item in enumerate(evidence):
            container = item.container
            span_id = _case_span_id(case.case_id, "candidate", container.container_id)
            candidate_span_ids[container.container_id] = span_id
            if position:
                knowledge_spans.append(
                    FieldSpan(
                        span_id=_case_span_id(case.case_id, "separator", str(position)),
                        text="\n\n",
                        kind="evidence_separator",
                        source="d2-evaluation",
                        provenance=f"d2:{case.case_id}:separator:{position}",
                    )
                )
            knowledge_spans.append(
                FieldSpan(
                    span_id=span_id,
                    text=container.text,
                    kind="semantic_eval_text",
                    source=container.source,
                    provenance=container.provenance,
                    confidence=container.confidence,
                    container_refs=(container.container_id,),
                    edge_refs=tuple(edge.edge_id for edge in item.edges),
                )
            )
        snapshot = SharedFieldSnapshot(
            tick_id=0,
            regions=(
                query_region,
                RegionState(
                    name=LogicalRegion.CORTEX,
                    spans=tuple(knowledge_spans),
                ),
            ),
            source_manifest_ids=(f"d2-held-out:{case.case_id}",),
        )
        return snapshot, query_span_id, candidate_span_ids

    @staticmethod
    def _field_span_by_source_id(
        slots: Sequence[D64SemanticSlot],
        source_span_id: str,
    ) -> D64SemanticSlot:
        matches = [
            slot for slot in slots if slot.slot_kind == "field_span" and slot.source_span_ids == (source_span_id,)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one grounded field_span semantic slot for {source_span_id!r}; found {len(matches)}"
            )
        return matches[0]

    def rank_case(
        self,
        case: DormantForwardEvaluationCase,
        evidence: Sequence[VerifiedDormantEvidence],
        *,
        k: int,
    ) -> tuple[SemanticQuery, tuple[str, ...], str, str, int, int, tuple[str, ...]]:
        if not evidence:
            raise ValueError("D64 reranking requires a non-empty exact candidate pool")
        # Every valid Unicode scalar is representable through the exact 16D
        # UTF-8 transport.  Only invalid surrogate text remains inaccessible.
        encode_unicode_text(case.query)
        supported_evidence: list[VerifiedDormantEvidence] = []
        unsupported_candidate_ids: list[str] = []
        for item in evidence:
            try:
                encode_unicode_text(item.container.text)
            except InvalidUnicodeScalarError:
                unsupported_candidate_ids.append(item.container.container_id)
            else:
                supported_evidence.append(item)

        snapshot, query_span_id, candidate_span_ids = self._materialize_snapshot(
            case,
            supported_evidence,
        )
        exact = self._exact_compiler.compile(snapshot)
        semantic = self._semantic_compiler.compile(snapshot, exact)
        semantic.verify_grounding(snapshot, exact)

        query_slot = self._field_span_by_source_id(semantic.slots, query_span_id)
        projection = SemanticProjectionRef.from_surface(
            semantic,
            slot_ids=(query_slot.slot_id,),
        )
        projection.assert_matches(semantic)
        source_container = next(
            (item.container for item in evidence if item.container.container_id == case.source_container_id),
            None,
        )
        evidence_refs: tuple[ExactEvidenceRef, ...] = ()
        if source_container is not None:
            evidence_refs = (
                ExactEvidenceRef(
                    source_kind="dormant_container",
                    source_id=source_container.container_id,
                    provenance=source_container.provenance,
                    sha256=source_container.raw_sha256,
                ),
            )
        query = SemanticQuery(
            lane="semantic_edge_rerank",
            text=case.query,
            projection=projection,
            evidence_refs=evidence_refs,
        )

        scored: list[tuple[float, str]] = []
        for item in supported_evidence:
            container_id = item.container.container_id
            candidate_slot = self._field_span_by_source_id(
                semantic.slots,
                candidate_span_ids[container_id],
            )
            # D.1 structural features are L2-normalized; their dot product is a
            # transparent cosine baseline.  No learned weights or hidden graph
            # labels enter this ranking.
            score = float(np.dot(query_slot.features64, candidate_slot.features64))
            scored.append((score, container_id))
        scored.extend((float("-inf"), container_id) for container_id in unsupported_candidate_ids)
        scored.sort(key=lambda item: (-item[0], item[1]))
        ranking = tuple(container_id for _, container_id in scored[:k])
        return (
            query,
            ranking,
            exact.rail_id,
            semantic.surface_id,
            exact.coverage.compiled_active_characters,
            semantic.slot_count,
            tuple(sorted(unsupported_candidate_ids)),
        )


def evaluate_d64_reranking(
    index: DormantEvidenceIndex,
    cases: Sequence[DormantForwardEvaluationCase],
    *,
    k: int = 8,
    candidate_multiplier: int = 8,
    auditor: DormantRelevanceAuditor | None = None,
    reranker: StructuralLexicalD64Reranker | None = None,
) -> D64RerankEvaluationResult:
    """Compare D.1 structural reranking with the accepted C.1 relevance auditor."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(candidate_multiplier, bool) or not isinstance(candidate_multiplier, int) or candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be a positive integer")
    if not cases:
        raise ValueError("cases must be non-empty")

    active_auditor = auditor or DormantRelevanceAuditor(
        DormantRelevancePolicy(items_per_materialization=k, target_chars=1_000_000)
    )
    active_reranker = reranker or StructuralLexicalD64Reranker()
    empty_field = SharedFieldSnapshot.empty(tick_id=0)
    pool_limit = max(k, k * candidate_multiplier)
    results: list[D64RerankCaseResult] = []

    for case in cases:
        evidence = index.retrieve(case.query, limit=pool_limit, include_graph=True)
        if not evidence:
            raise ValueError(f"held-out case {case.case_id} produced an empty exact candidate pool")
        pool_ids = tuple(item.container.container_id for item in evidence)
        query, structural_ranking, rail_id, surface_id, exact_chars, slot_count, unsupported_candidate_ids = (
            active_reranker.rank_case(
                case,
                evidence,
                k=k,
            )
        )
        audited = active_auditor.select(case.query, evidence, empty_field)
        audited_ranking = tuple(audited.selected_container_ids[:k])
        expected = case.expected_target_container_ids
        results.append(
            D64RerankCaseResult(
                case=case,
                query_id=query.query_id,
                evaluation_field_id=query.source_field_id,
                rail_id=rail_id,
                semantic_surface_id=surface_id,
                feature_generation=query.projection.feature_generation,
                query_slot_id=query.projection.slots[0].slot_id,
                pool_container_ids=pool_ids,
                structural_top_k_container_ids=structural_ranking,
                audited_top_k_container_ids=audited_ranking,
                pool_hit=bool(set(pool_ids) & set(expected)),
                structural_reciprocal_rank=_reciprocal_rank(structural_ranking, expected),
                audited_reciprocal_rank=_reciprocal_rank(audited_ranking, expected),
                exact_character_count=exact_chars,
                semantic_slot_count=slot_count,
                unsupported_candidate_ids=unsupported_candidate_ids,
            )
        )

    count = len(results)
    feature_generations = {item.feature_generation for item in results}
    if len(feature_generations) != 1:
        raise ValueError("held-out D64 evaluation mixed semantic feature generations")
    return D64RerankEvaluationResult(
        index_id=index.index_id,
        k=k,
        pool_limit=pool_limit,
        baseline_generation=active_reranker.generation,
        feature_generation=next(iter(feature_generations)),
        cases=tuple(results),
        pool_recall=sum(1 for item in results if item.pool_hit) / count,
        structural_hit_at_k=sum(1 for item in results if item.structural_hit) / count,
        structural_mean_reciprocal_rank=sum(item.structural_reciprocal_rank for item in results) / count,
        audited_hit_at_k=sum(1 for item in results if item.audited_hit) / count,
        audited_mean_reciprocal_rank=sum(item.audited_reciprocal_rank for item in results) / count,
        unsupported_candidate_count=sum(len(item.unsupported_candidate_ids) for item in results),
    )


__all__ = [
    "D64_RERANK_BASELINE_GENERATION",
    "D64_RERANK_EVALUATION_SCHEMA",
    "D64RerankCaseResult",
    "D64RerankEvaluationResult",
    "StructuralLexicalD64Reranker",
    "evaluate_d64_reranking",
]
