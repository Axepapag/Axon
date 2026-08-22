from __future__ import annotations

import json
from pathlib import Path

from runtime.dormant import DormantEvidenceIndex
from tests.test_dormant_evidence_bridge import _jsonl_line, _write_fixture_state


def test_relation_targets_get_bounded_best_edge_support_without_additive_flood(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    dormant = state_root / "dormant"

    base_edge = {
        "source_container_id": "c-axon",
        "source_text": "Axon uses",
        "edge_type": "uses",
        "target": "System",
        "provenance": "fixture:edge:bounded",
        "confidence": 0.9,
        "status": "dormant",
    }
    # Repetition is intentionally noisy. The derived graph sense may count
    # bounded topology support, but repeated relations must not add unbounded
    # score or manufacture certainty.
    edges = []
    for index in range(32):
        edge = dict(base_edge)
        edge["provenance"] = f"fixture:edge:bounded:{index}"
        edges.append(edge)
    (dormant / "semantic_edges.jsonl").write_bytes(b"".join(_jsonl_line(edge) for edge in edges))

    # Keep the recovered manifest internally truthful for this fixture.
    manifest_path = dormant / "corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_counts"]["semantic_edges"] = len(edges)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    with DormantEvidenceIndex.build(state_root) as index:
        candidates = index.query_candidates("Axon uses", limit=8, include_graph=True)
        by_id = {candidate.container_id: candidate for candidate in candidates}
        assert "c-axon" in by_id
        assert "c-system" in by_id
        source = by_id["c-axon"]
        target = by_id["c-system"]
        assert target.graph_hits <= 8
        assert target.score <= source.score
        assert target.edge_ids


def test_relation_aware_graph_expansion_retrieves_target_without_target_text(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    with DormantEvidenceIndex.build(state_root) as index:
        candidates = index.query_candidates("Axon uses", limit=8, include_graph=True)
        system = next(candidate for candidate in candidates if candidate.container_id == "c-system")
        assert system.lexical_hits == 0
        assert system.graph_hits > 0
        assert system.edge_ids
