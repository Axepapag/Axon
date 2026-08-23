from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from Cortext import D64_RERANK_BASELINE_GENERATION, evaluate_d64_reranking
from runtime.dormant import DormantEvidenceIndex, forward_semantic_edge_cases
from runtime.field import D64_STRUCTURAL_FEATURE_GENERATION
from tests.test_dormant_evidence_bridge import _write_fixture_state


def test_d64_reranking_evaluation_uses_real_dual_surface_and_exact_candidates(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    with DormantEvidenceIndex.build(state_root) as index:
        cases = forward_semantic_edge_cases(index, sample_size=1, seed="d2-fixed")
        result = evaluate_d64_reranking(index, cases, k=2, candidate_multiplier=2)

    assert result.baseline_generation == D64_RERANK_BASELINE_GENERATION
    assert result.feature_generation == D64_STRUCTURAL_FEATURE_GENERATION
    assert result.pool_recall == 1.0
    assert result.structural_hit_at_k == 1.0
    assert result.audited_hit_at_k == 1.0
    assert len(result.evaluation_id) == 64

    case = result.cases[0]
    assert len(case.query_id) == 64
    assert len(case.evaluation_field_id) == 64
    assert len(case.rail_id) == 64
    assert len(case.semantic_surface_id) == 64
    assert len(case.query_slot_id) == 64
    assert case.exact_character_count > 0
    assert case.semantic_slot_count > 0
    assert "c-system" in case.pool_container_ids
    assert "c-system" in case.structural_top_k_container_ids


def test_d64_reranking_counts_unsupported_candidates_without_normalizing_or_truncating(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    dormant = state_root / "dormant"
    unsupported = {
        "container_id": "c-mu",
        "kind": "concept",
        "text": "Axon μ",
        "normalized_text": "axon μ",
        "letters": "Axon μ",
        "edges": [],
        "source": "fixture.db:concepts:3",
        "provenance": "fixture:container:mu",
        "confidence": 0.7,
        "status": "dormant",
        "metadata": {},
    }
    with (dormant / "containers.jsonl").open("ab") as handle:
        handle.write(
            (json.dumps(unsupported, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        )
    manifest_path = dormant / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_counts"]["containers"] = 3
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )

    with DormantEvidenceIndex.build(state_root) as index:
        cases = forward_semantic_edge_cases(index, sample_size=1, seed="d2-fixed")
        result = evaluate_d64_reranking(index, cases, k=3, candidate_multiplier=3)

    case = result.cases[0]
    assert "c-mu" in case.pool_container_ids
    assert case.unsupported_candidate_ids == ("c-mu",)
    assert result.unsupported_candidate_count == 1
    assert case.structural_top_k_container_ids[-1] == "c-mu"


def test_d64_reranking_cli_imports_repo_packages_from_documented_invocation() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_d64_specialist_baseline.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Evaluate Axon's grounded D64 structural-lexical reranking baseline" in completed.stdout
