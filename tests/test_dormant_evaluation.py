from __future__ import annotations

from pathlib import Path

from runtime.dormant import (
    DormantEvidenceIndex,
    evaluate_forward_relevance,
    evaluate_relevance,
    forward_semantic_edge_cases,
    semantic_edge_cases,
)
from tests.test_dormant_evidence_bridge import _write_fixture_state


def test_semantic_edge_cases_are_deterministic_and_grounded(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    edges = state_root / "dormant" / "semantic_edges.jsonl"
    first = semantic_edge_cases(edges, sample_size=1, seed="fixed")
    second = semantic_edge_cases(edges, sample_size=1, seed="fixed")
    assert first == second
    assert len(first) == 1
    assert first[0].expected_container_id == "c-axon"
    assert first[0].query == "uses System"
    assert len(first[0].edge_raw_sha256) == 64


def test_relevance_evaluation_hits_real_fixture_semantic_edge(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    cases = semantic_edge_cases(state_root / "dormant" / "semantic_edges.jsonl", sample_size=1)
    with DormantEvidenceIndex.build(state_root) as index:
        result = evaluate_relevance(index, cases, k=2, candidate_multiplier=2)
        assert result.hit_at_k == 1.0
        assert result.mean_reciprocal_rank > 0.0
        assert result.cases[0].case.expected_container_id in result.cases[0].ranked_container_ids
        assert len(result.evaluation_id) == 64


def test_forward_semantic_evaluation_measures_source_relation_to_exact_target(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    edges = state_root / "dormant" / "semantic_edges.jsonl"
    with DormantEvidenceIndex.build(state_root) as index:
        first = forward_semantic_edge_cases(index, sample_size=1, seed="fixed-forward")
        second = forward_semantic_edge_cases(index, sample_size=1, seed="fixed-forward")
        assert first == second
        assert first[0].source_container_id == "c-axon"
        assert first[0].expected_target_container_ids == ("c-system",)
        assert first[0].query == "Axon uses"
        assert "System" not in first[0].query

        result = evaluate_forward_relevance(index, first, k=2, candidate_multiplier=2)
        assert result.pool_recall == 1.0
        assert result.raw_hit_at_k == 1.0
        assert result.audited_hit_at_k == 1.0
        assert result.cases[0].raw_reciprocal_rank > 0.0
        assert result.cases[0].audited_reciprocal_rank > 0.0
        assert "c-system" in result.cases[0].pool_container_ids
        assert len(result.evaluation_id) == 64
