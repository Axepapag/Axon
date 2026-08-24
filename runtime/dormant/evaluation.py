"""Held-out deterministic evaluation for dormant retrieval/relevance.

Build C keeps two different semantic-edge evaluations on purpose.

``semantic_edge_cases`` is the original inverse-association stress test:
relation + target -> one source container.  It is retained because its failures
are useful evidence, but the label can be underdetermined when many sources
share the same relation/target.

``forward_semantic_edge_cases`` is the stronger primary quality gate:
exact source-container text + relation -> an exact target container resolved
from the recovered edge target.  It reports raw candidate-pool recall, raw
P0 top-k, and post-auditor top-k separately so candidate generation and
reranking cannot hide each other's failures.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from runtime.field import SharedFieldSnapshot, canonical_sha256

from .evidence_bridge import DormantEvidenceIndex
from .relevance import DormantRelevanceAuditor, DormantRelevancePolicy

EVALUATION_SCHEMA = "axon-dormant-relevance-evaluation-v1"
FORWARD_EVALUATION_SCHEMA = "axon-dormant-forward-relevance-evaluation-v1"
_EVAL_TOKEN_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", flags=re.UNICODE)


def _normalize(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _token_count(value: str) -> int:
    return len({_normalize(match.group(0)) for match in _EVAL_TOKEN_RE.finditer(value)})


@dataclass(frozen=True, slots=True)
class DormantEvaluationCase:
    case_id: str
    query: str
    expected_container_id: str
    edge_type: str
    target: str
    edge_raw_sha256: str

    def to_canonical_dict(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "expected_container_id": self.expected_container_id,
            "edge_type": self.edge_type,
            "target": self.target,
            "edge_raw_sha256": self.edge_raw_sha256,
        }


@dataclass(frozen=True, slots=True)
class DormantEvaluationCaseResult:
    case: DormantEvaluationCase
    ranked_container_ids: tuple[str, ...]
    reciprocal_rank: float
    hit: bool


@dataclass(frozen=True, slots=True)
class DormantEvaluationResult:
    index_id: str
    k: int
    cases: tuple[DormantEvaluationCaseResult, ...]
    hit_at_k: float
    mean_reciprocal_rank: float

    @property
    def evaluation_id(self) -> str:
        return canonical_sha256(
            {
                "schema": EVALUATION_SCHEMA,
                "index_id": self.index_id,
                "k": self.k,
                "cases": [item.case.to_canonical_dict() for item in self.cases],
                "rankings": [list(item.ranked_container_ids) for item in self.cases],
            }
        )

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "schema": EVALUATION_SCHEMA,
            "evaluation_id": self.evaluation_id,
            "index_id": self.index_id,
            "k": self.k,
            "case_count": len(self.cases),
            "hit_at_k": self.hit_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "cases": [
                {
                    **item.case.to_canonical_dict(),
                    "ranked_container_ids": list(item.ranked_container_ids),
                    "reciprocal_rank": item.reciprocal_rank,
                    "hit": item.hit,
                }
                for item in self.cases
            ],
        }


@dataclass(frozen=True, slots=True)
class DormantForwardEvaluationCase:
    """Grounded source+relation -> target retrieval case."""

    case_id: str
    query: str
    source_container_id: str
    expected_target_container_ids: tuple[str, ...]
    edge_type: str
    target: str
    edge_raw_sha256: str

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "source_container_id": self.source_container_id,
            "expected_target_container_ids": list(self.expected_target_container_ids),
            "edge_type": self.edge_type,
            "target": self.target,
            "edge_raw_sha256": self.edge_raw_sha256,
        }


@dataclass(frozen=True, slots=True)
class DormantForwardEvaluationCaseResult:
    case: DormantForwardEvaluationCase
    pool_container_ids: tuple[str, ...]
    raw_top_k_container_ids: tuple[str, ...]
    audited_top_k_container_ids: tuple[str, ...]
    pool_hit: bool
    raw_reciprocal_rank: float
    audited_reciprocal_rank: float

    @property
    def raw_hit(self) -> bool:
        return self.raw_reciprocal_rank > 0.0

    @property
    def audited_hit(self) -> bool:
        return self.audited_reciprocal_rank > 0.0


@dataclass(frozen=True, slots=True)
class DormantForwardEvaluationResult:
    index_id: str
    k: int
    pool_limit: int
    cases: tuple[DormantForwardEvaluationCaseResult, ...]
    pool_recall: float
    raw_hit_at_k: float
    raw_mean_reciprocal_rank: float
    audited_hit_at_k: float
    audited_mean_reciprocal_rank: float

    @property
    def evaluation_id(self) -> str:
        return canonical_sha256(
            {
                "schema": FORWARD_EVALUATION_SCHEMA,
                "index_id": self.index_id,
                "k": self.k,
                "pool_limit": self.pool_limit,
                "cases": [item.case.to_canonical_dict() for item in self.cases],
                "pool_rankings": [list(item.pool_container_ids) for item in self.cases],
                "raw_rankings": [list(item.raw_top_k_container_ids) for item in self.cases],
                "audited_rankings": [list(item.audited_top_k_container_ids) for item in self.cases],
            }
        )

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "schema": FORWARD_EVALUATION_SCHEMA,
            "evaluation_id": self.evaluation_id,
            "index_id": self.index_id,
            "k": self.k,
            "pool_limit": self.pool_limit,
            "case_count": len(self.cases),
            "pool_recall": self.pool_recall,
            "raw_hit_at_k": self.raw_hit_at_k,
            "raw_mean_reciprocal_rank": self.raw_mean_reciprocal_rank,
            "audited_hit_at_k": self.audited_hit_at_k,
            "audited_mean_reciprocal_rank": self.audited_mean_reciprocal_rank,
            "cases": [
                {
                    **item.case.to_canonical_dict(),
                    "pool_container_ids": list(item.pool_container_ids),
                    "raw_top_k_container_ids": list(item.raw_top_k_container_ids),
                    "audited_top_k_container_ids": list(item.audited_top_k_container_ids),
                    "pool_hit": item.pool_hit,
                    "raw_reciprocal_rank": item.raw_reciprocal_rank,
                    "raw_hit": item.raw_hit,
                    "audited_reciprocal_rank": item.audited_reciprocal_rank,
                    "audited_hit": item.audited_hit,
                }
                for item in self.cases
            ],
        }


def semantic_edge_cases(
    semantic_edges_path: str | Path,
    *,
    sample_size: int = 64,
    seed: str = "axon-build-c1",
) -> tuple[DormantEvaluationCase, ...]:
    """Select a deterministic hash-min sample for the inverse stress test."""

    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    path = Path(semantic_edges_path)
    selected: list[tuple[int, str, DormantEvaluationCase]] = []
    with path.open("rb") as handle:
        offset = 0
        for raw in handle:
            raw_offset = offset
            offset += len(raw)
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            source_id = str(record.get("source_container_id", "")).strip()
            edge_type = str(record.get("edge_type", "")).strip()
            target = str(record.get("target", "")).strip()
            if not source_id or not edge_type or not target:
                continue
            query = f"{edge_type} {target}".strip()
            raw_hash = hashlib.sha256(raw).hexdigest()
            case_id = canonical_sha256(
                {
                    "source_container_id": source_id,
                    "edge_type": edge_type,
                    "target": target,
                    "raw_sha256": raw_hash,
                    "offset": raw_offset,
                }
            )
            case = DormantEvaluationCase(
                case_id=case_id,
                query=query,
                expected_container_id=source_id,
                edge_type=edge_type,
                target=target,
                edge_raw_sha256=raw_hash,
            )
            key_int = int.from_bytes(
                hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).digest(),
                "big",
            )
            heap_item = (-key_int, case_id, case)
            if len(selected) < sample_size:
                heapq.heappush(selected, heap_item)
            elif heap_item[0] > selected[0][0]:
                heapq.heapreplace(selected, heap_item)
    selected.sort(key=lambda item: (-item[0], item[1]))
    return tuple(case for _, _, case in selected)


def _forward_edge_seed_sample(
    index: DormantEvidenceIndex,
    *,
    candidate_count: int,
    seed: str,
) -> tuple[tuple[str, str], ...]:
    """Hash-min sample resolved edge/source IDs from the derived graph sense."""

    selected: list[tuple[int, str, tuple[str, str]]] = []
    for edge_id, source_id in index.iter_edge_sources():
        seed_id = canonical_sha256(
            {
                "edge_id": edge_id,
                "source_container_id": source_id,
            }
        )
        key_int = int.from_bytes(
            hashlib.sha256(f"{seed}:{seed_id}".encode("utf-8")).digest(),
            "big",
        )
        payload = (edge_id, source_id)
        heap_item = (-key_int, seed_id, payload)
        if len(selected) < candidate_count:
            heapq.heappush(selected, heap_item)
        elif heap_item[0] > selected[0][0]:
            heapq.heapreplace(selected, heap_item)
    selected.sort(key=lambda item: (-item[0], item[1]))
    return tuple(payload for _, _, payload in selected)


def forward_semantic_edge_cases(
    index: DormantEvidenceIndex,
    *,
    sample_size: int = 64,
    seed: str = "axon-build-c1-forward",
) -> tuple[DormantForwardEvaluationCase, ...]:
    """Create grounded source+relation -> target cases with no target leakage.

    The recovered edge target must resolve through the derived normalized-key
    map to at least one actual container.  Source and target IDs are then exact-
    dereferenced before the case is admitted. Source length never excludes an
    otherwise valid grounded case.
    """

    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")

    candidate_count = min(32_768, max(sample_size * 128, sample_size))
    seeds = _forward_edge_seed_sample(
        index,
        candidate_count=candidate_count,
        seed=seed,
    )
    cases: list[DormantForwardEvaluationCase] = []
    for edge_id, source_id in seeds:
        try:
            edge = index.dereference_edge(edge_id)
            source = index.dereference_container(source_id)
        except KeyError:
            continue
        if edge.source_container_id != source_id:
            continue
        edge_type = edge.edge_type.strip()
        target = edge.target.strip()
        if not edge_type or not target:
            continue
        expected_ids = index.resolved_target_container_ids(edge_id, limit=16)
        if not expected_ids:
            continue
        raw_hash = edge.raw_sha256
        raw_offset = edge.byte_offset
        if source.status not in {"dormant", "active"}:
            continue
        source_text = source.text.strip()
        if not source_text:
            continue
        # Do not admit a case if the expected target text is already literally
        # present in the source query; that would test lexical leakage, not graph
        # relation retrieval.
        normalized_source = _normalize(source_text)
        normalized_target = _normalize(target)
        if normalized_target and normalized_target in normalized_source:
            continue
        verified_target_ids: list[str] = []
        for container_id in expected_ids:
            if container_id == source_id:
                continue
            try:
                target_container = index.dereference_container(container_id)
            except KeyError:
                continue
            if target_container.status not in {"dormant", "active"}:
                continue
            if _normalize(target_container.normalized_text or target_container.text) == normalized_target:
                verified_target_ids.append(container_id)
        if not verified_target_ids:
            continue
        query = f"{source_text} {edge_type}".strip()
        if _token_count(query) > 128:
            continue
        case_id = canonical_sha256(
            {
                "direction": "source_relation_to_target",
                "source_container_id": source_id,
                "expected_target_container_ids": sorted(verified_target_ids),
                "edge_type": edge_type,
                "target": target,
                "raw_sha256": raw_hash,
                "offset": raw_offset,
            }
        )
        cases.append(
            DormantForwardEvaluationCase(
                case_id=case_id,
                query=query,
                source_container_id=source_id,
                expected_target_container_ids=tuple(sorted(verified_target_ids)),
                edge_type=edge_type,
                target=target,
                edge_raw_sha256=raw_hash,
            )
        )
        if len(cases) >= sample_size:
            break
    if len(cases) < sample_size:
        raise ValueError(
            f"only {len(cases)} usable forward semantic-edge cases were found; requested {sample_size}"
        )
    return tuple(cases)


def _reciprocal_rank(ranking: Sequence[str], expected: Sequence[str]) -> float:
    expected_set = set(expected)
    for index, container_id in enumerate(ranking, start=1):
        if container_id in expected_set:
            return 1.0 / index
    return 0.0


def evaluate_relevance(
    index: DormantEvidenceIndex,
    cases: Sequence[DormantEvaluationCase],
    *,
    k: int = 8,
    candidate_multiplier: int = 4,
    auditor: DormantRelevanceAuditor | None = None,
) -> DormantEvaluationResult:
    """Run the retained inverse-association stress test."""

    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 128:
        raise ValueError("k must be an integer in [1, 128]")
    if (
        isinstance(candidate_multiplier, bool)
        or not isinstance(candidate_multiplier, int)
        or candidate_multiplier < 1
    ):
        raise ValueError("candidate_multiplier must be a positive integer")
    if not cases:
        raise ValueError("cases must be non-empty")

    active_auditor = auditor or DormantRelevanceAuditor(
        DormantRelevancePolicy(max_items=k, max_chars=1_000_000, max_item_chars=1_000_000)
    )
    empty_field = SharedFieldSnapshot.empty(tick_id=0)
    results: list[DormantEvaluationCaseResult] = []
    pool_limit = min(128, max(k, k * candidate_multiplier))
    for case in cases:
        evidence = index.retrieve(case.query, limit=pool_limit, include_graph=True)
        decision = active_auditor.select(case.query, evidence, empty_field)
        ranking = decision.selected_container_ids[:k]
        reciprocal_rank = _reciprocal_rank(ranking, (case.expected_container_id,))
        results.append(
            DormantEvaluationCaseResult(
                case=case,
                ranked_container_ids=tuple(ranking),
                reciprocal_rank=reciprocal_rank,
                hit=reciprocal_rank > 0.0,
            )
        )
    hit_at_k = sum(1 for item in results if item.hit) / len(results)
    mean_reciprocal_rank = sum(item.reciprocal_rank for item in results) / len(results)
    return DormantEvaluationResult(
        index_id=index.index_id,
        k=k,
        cases=tuple(results),
        hit_at_k=hit_at_k,
        mean_reciprocal_rank=mean_reciprocal_rank,
    )


def evaluate_forward_relevance(
    index: DormantEvidenceIndex,
    cases: Sequence[DormantForwardEvaluationCase],
    *,
    k: int = 8,
    candidate_multiplier: int = 8,
    auditor: DormantRelevanceAuditor | None = None,
) -> DormantForwardEvaluationResult:
    """Measure candidate generation and relevance reranking independently."""

    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 128:
        raise ValueError("k must be an integer in [1, 128]")
    if (
        isinstance(candidate_multiplier, bool)
        or not isinstance(candidate_multiplier, int)
        or candidate_multiplier < 1
    ):
        raise ValueError("candidate_multiplier must be a positive integer")
    if not cases:
        raise ValueError("cases must be non-empty")

    active_auditor = auditor or DormantRelevanceAuditor(
        DormantRelevancePolicy(max_items=k, max_chars=1_000_000, max_item_chars=1_000_000)
    )
    empty_field = SharedFieldSnapshot.empty(tick_id=0)
    pool_limit = min(128, max(k, k * candidate_multiplier))
    results: list[DormantForwardEvaluationCaseResult] = []
    for case in cases:
        evidence = index.retrieve(case.query, limit=pool_limit, include_graph=True)
        pool_ids = tuple(item.container.container_id for item in evidence)
        raw_top_k = pool_ids[:k]
        decision = active_auditor.select(case.query, evidence, empty_field)
        audited_top_k = decision.selected_container_ids[:k]
        expected = case.expected_target_container_ids
        results.append(
            DormantForwardEvaluationCaseResult(
                case=case,
                pool_container_ids=pool_ids,
                raw_top_k_container_ids=raw_top_k,
                audited_top_k_container_ids=tuple(audited_top_k),
                pool_hit=bool(set(pool_ids) & set(expected)),
                raw_reciprocal_rank=_reciprocal_rank(raw_top_k, expected),
                audited_reciprocal_rank=_reciprocal_rank(audited_top_k, expected),
            )
        )

    count = len(results)
    return DormantForwardEvaluationResult(
        index_id=index.index_id,
        k=k,
        pool_limit=pool_limit,
        cases=tuple(results),
        pool_recall=sum(1 for item in results if item.pool_hit) / count,
        raw_hit_at_k=sum(1 for item in results if item.raw_hit) / count,
        raw_mean_reciprocal_rank=sum(item.raw_reciprocal_rank for item in results) / count,
        audited_hit_at_k=sum(1 for item in results if item.audited_hit) / count,
        audited_mean_reciprocal_rank=sum(item.audited_reciprocal_rank for item in results) / count,
    )


__all__ = [
    "EVALUATION_SCHEMA",
    "FORWARD_EVALUATION_SCHEMA",
    "DormantEvaluationCase",
    "DormantEvaluationCaseResult",
    "DormantEvaluationResult",
    "DormantForwardEvaluationCase",
    "DormantForwardEvaluationCaseResult",
    "DormantForwardEvaluationResult",
    "semantic_edge_cases",
    "forward_semantic_edge_cases",
    "evaluate_relevance",
    "evaluate_forward_relevance",
]
