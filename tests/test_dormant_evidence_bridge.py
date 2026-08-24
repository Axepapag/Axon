from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from runtime.dormant import (
    DormantEvidenceBridge,
    DormantEvidenceError,
    DormantEvidenceIndex,
    StaleDormantIndexError,
)
from runtime.field import LogicalRegion, SharedFieldSnapshot


def _jsonl_line(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _write_fixture_state(root: Path) -> Path:
    state_root = root / "State"
    dormant = state_root / "dormant"
    dormant.mkdir(parents=True)

    axon = {
        "container_id": "c-axon",
        "kind": "concept",
        "text": "Axon",
        "normalized_text": "axon",
        "letters": "Axon",
        "edges": [
            {
                "edge_type": "uses",
                "target": "System",
                "confidence": 0.9,
                "provenance": "fixture:edge:embedded",
                "status": "dormant",
            }
        ],
        "source": "fixture.db:concepts:1",
        "provenance": "fixture:container:axon",
        "confidence": 0.95,
        "status": "dormant",
        "metadata": {"definition": "stateful architecture"},
    }
    system = {
        "container_id": "c-system",
        "kind": "concept",
        "text": "System",
        "normalized_text": "system",
        "letters": "System",
        "edges": [],
        "source": "fixture.db:concepts:2",
        "provenance": "fixture:container:system",
        "confidence": 0.8,
        "status": "dormant",
        "metadata": {},
    }
    (dormant / "containers.jsonl").write_bytes(_jsonl_line(axon) + _jsonl_line(system))

    edge = {
        "source_container_id": "c-axon",
        "source_text": "Axon uses System",
        "edge_type": "uses",
        "target": "System",
        "provenance": "fixture:edge:1",
        "confidence": 0.9,
        "status": "dormant",
    }
    (dormant / "semantic_edges.jsonl").write_bytes(_jsonl_line(edge))
    for name in ("kg_cache_50k.jsonl", "layout_groups.jsonl", "symbol_registry.jsonl"):
        (dormant / name).write_bytes(b"")

    manifest = {
        "kind": "axon_recovered_corpus_manifest",
        "version": 1,
        "source_files": [
            {
                "path": "fixture.db",
                "size_bytes": 123,
                "sha256": hashlib.sha256(b"fixture-source").hexdigest(),
                "included": True,
            }
        ],
        "output_counts": {"containers": 2, "semantic_edges": 1},
    }
    (dormant / "corpus_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return state_root


def test_build_query_exact_dereference_graph_surface_and_d64_roundtrip(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    index = DormantEvidenceIndex.build(state_root)
    try:
        candidates = index.query_candidates("stateful architecture", limit=4)
        assert candidates[0].container_id == "c-axon"
        assert any(item.container_id == "c-system" and item.graph_hits > 0 for item in candidates)

        exact = index.dereference_container("c-axon")
        assert exact.text == "Axon"
        assert exact.source == "fixture.db:concepts:1"
        assert exact.provenance == "fixture:container:axon"
        assert exact.text_sha256 == hashlib.sha256(b"Axon").hexdigest()

        evidence = index.retrieve("stateful architecture", limit=2)
        assert evidence[0].container.container_id == "c-axon"
        assert evidence[0].container.text == "Axon"

        bridge = DormantEvidenceBridge(index)
        result = bridge.query_surface_compile(
            SharedFieldSnapshot(tick_id=0),
            "stateful architecture",
            limit=2,
        )
        structured = result.snapshot.region(LogicalRegion.CORTEX)
        assert structured.spans[0].text == "Axon"
        assert structured.spans[0].container_refs == ("c-axon",)
        assert "fixture:container:axon" in structured.spans[0].provenance
        assert result.snapshot.parent_field_id is not None
        assert result.snapshot.tick_id == 1
        assert result.compiled.coverage.complete is True
        assert (
            result.compiled.region_text(LogicalRegion.CORTEX)
            == structured.text
        )
        result.compiled.verify_roundtrip(result.snapshot)
    finally:
        index.close()


def test_query_pages_every_unique_term_instead_of_rejecting_after_128(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    index = DormantEvidenceIndex.build(state_root)
    try:
        query = " ".join([*(f"term{index}" for index in range(150)), "stateful"])
        candidates = index.query_candidates(query, limit=4, include_graph=False)
        assert candidates[0].container_id == "c-axon"
    finally:
        index.close()


def test_index_schema_contains_lookup_metadata_not_authoritative_text_columns(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    index = DormantEvidenceIndex.build(state_root)
    index_path = index.path
    index.close()

    connection = sqlite3.connect(index_path)
    try:
        container_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(containers)").fetchall()
        }
        edge_columns = {row[1] for row in connection.execute("PRAGMA table_info(edges)").fetchall()}
        assert "text" not in container_columns
        assert "source" not in container_columns
        assert "provenance" not in container_columns
        assert "target" not in edge_columns
        assert "source_text" not in edge_columns
        assert "provenance" not in edge_columns
        assert {"byte_offset", "byte_length", "raw_sha256", "text_sha256"} <= container_columns
        assert {"source_container_id", "target_key", "raw_sha256"} <= edge_columns
    finally:
        connection.close()


def test_open_fails_closed_when_authoritative_corpus_changes(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    index = DormantEvidenceIndex.build(state_root)
    index_path = index.path
    index.close()

    containers = state_root / "dormant" / "containers.jsonl"
    with containers.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(StaleDormantIndexError):
        DormantEvidenceIndex.open(state_root, index_path=index_path, verify_binding=True)


def test_index_uses_compact_hashed_postings_and_integer_graph_refs(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    index = DormantEvidenceIndex.build(state_root)
    index_path = index.path
    index.close()

    connection = sqlite3.connect(index_path)
    try:
        container_term_columns = {
            row[1]: row[2] for row in connection.execute("PRAGMA table_info(container_terms)").fetchall()
        }
        edge_term_columns = {
            row[1]: row[2] for row in connection.execute("PRAGMA table_info(edge_terms)").fetchall()
        }
        graph_columns = {
            row[1]: row[2] for row in connection.execute("PRAGMA table_info(graph_neighbors)").fetchall()
        }
        assert container_term_columns == {"term_hash": "BLOB", "container_rowid": "INTEGER"}
        assert edge_term_columns == {"term_hash": "BLOB", "edge_rowid": "INTEGER"}
        assert graph_columns == {
            "source_container_rowid": "INTEGER",
            "target_container_rowid": "INTEGER",
            "edge_rowid": "INTEGER",
        }
        assert connection.execute("SELECT typeof(term_hash) FROM container_terms LIMIT 1").fetchone()[0] == "blob"
        assert connection.execute("SELECT length(term_hash) FROM container_terms LIMIT 1").fetchone()[0] == 32
    finally:
        connection.close()


def test_verified_index_fails_closed_if_any_authoritative_file_changes_after_open(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    index = DormantEvidenceIndex.build(state_root)
    try:
        support_file = state_root / "dormant" / "kg_cache_50k.jsonl"
        support_file.write_bytes(b"changed\n")
        with pytest.raises(StaleDormantIndexError):
            index.query_candidates("stateful architecture")
    finally:
        index.close()


def test_rebuild_has_stable_binding_and_index_identity(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    first_path = state_root / "dormant" / ".derived" / "first.sqlite3"
    second_path = state_root / "dormant" / ".derived" / "second.sqlite3"
    first = DormantEvidenceIndex.build(state_root, index_path=first_path)
    try:
        first_id = first.index_id
        first_binding = first.binding.binding_id
        first_candidates = first.query_candidates("stateful architecture", limit=4)
    finally:
        first.close()

    second = DormantEvidenceIndex.build(state_root, index_path=second_path)
    try:
        assert second.index_id == first_id
        assert second.binding.binding_id == first_binding
        assert second.query_candidates("stateful architecture", limit=4) == first_candidates
        assert first_path.with_name(first_path.name + ".manifest.json").is_file()
        assert second_path.with_name(second_path.name + ".manifest.json").is_file()
    finally:
        second.close()


def test_open_rejects_index_identity_tampering(tmp_path: Path) -> None:
    state_root = _write_fixture_state(tmp_path)
    index = DormantEvidenceIndex.build(state_root)
    index_path = index.path
    index.close()

    connection = sqlite3.connect(index_path)
    try:
        connection.execute("UPDATE meta SET value = ? WHERE key = 'index_id'", ("0" * 64,))
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(DormantEvidenceError):
        DormantEvidenceIndex.open(state_root, index_path=index_path, verify_binding=False)
