#!/usr/bin/env python3
"""Tests for curator/recovered_corpus_builder.py.

Covers:
- Source DB schema mapping (actual recovered schemas)
- Container + spelled-out edge emission
- Artifact writing and manifest contents
- Source file hashes in manifest
- Personal-log exclusion by default
- Limit/max-items budgets
- No source DB mutation
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recovered_corpus_builder import (
    DEFAULT_OUT_DIR,
    RecoveredCorpusBuilder,
    build_diary_container,
    build_entity_container,
    build_fact_container,
    build_procedure_container,
    build_relation_container,
    clean_text,
    json_source_pointer,
    sha256_file,
    source_pointer,
)
from container_schema import ContainerStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_path_obj():
    d = tempfile.mkdtemp(prefix="recovered_corpus_test_")
    yield Path(d)
    import shutil
    if os.path.exists(d):
        shutil.rmtree(d)


@pytest.fixture
def semantic_db(tmp_path_obj: Path):
    path = str(tmp_path_obj / "semantic.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE extracted_entities (
            id INTEGER PRIMARY KEY,
            name TEXT,
            type TEXT,
            attributes TEXT,
            source TEXT,
            created_at TEXT
        );
        INSERT INTO extracted_entities VALUES
            (1, 'Jeffrey', 'person', '{"role": "architect"}', 'memory_agent', '2026-03-15'),
            (2, 'Axon Voice Bridge', 'project', '{"purpose": "voice"}', 'memory_agent', '2026-03-15');

        CREATE TABLE extracted_facts (
            id INTEGER PRIMARY KEY,
            key TEXT,
            value TEXT,
            source TEXT,
            created_at TEXT
        );
        INSERT INTO extracted_facts VALUES
            (1, 'phone_capability', 'uses Twilio', 'memory_agent', '2026-03-15');

        CREATE TABLE extracted_relations (
            id INTEGER PRIMARY KEY,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            confidence REAL,
            source TEXT,
            created_at TEXT
        );
        INSERT INTO extracted_relations VALUES
            (1, 'phone.make_call', 'currently_uses', 'Twilio', 1.0, 'memory_agent', '2026-03-15');

        CREATE TABLE extracted_procedures (
            id INTEGER PRIMARY KEY,
            name TEXT,
            trigger TEXT,
            steps_json TEXT,
            outcome TEXT,
            applicability TEXT,
            source TEXT,
            created_at TEXT
        );
        INSERT INTO extracted_procedures VALUES
            (1, 'Start Axon', 'boot needed',
             '["cd Axon", "python axon.py"]', 'running', 'local', 'memory_agent', '2026-03-15');
    """)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def episodic_db(tmp_path_obj: Path):
    path = str(tmp_path_obj / "episodic.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE episodes (
            id INTEGER PRIMARY KEY,
            episode_type TEXT,
            turns_json TEXT,
            summary TEXT,
            extracted_json TEXT,
            source TEXT,
            created_at TEXT,
            payload_json TEXT
        );
        INSERT INTO episodes VALUES
            (1, 'conversation_summary', NULL,
             'Jeffrey asked for real-time phone conversation.',
             '{"relations": [{"predicate": "requested", "object": "voice feature"}]}',
             'memory_agent', '2026-03-14T22:16:54', NULL);
    """)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def backlog_db(tmp_path_obj: Path):
    path = str(tmp_path_obj / "backlog.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE backlog_jobs (
            id INTEGER PRIMARY KEY,
            operation TEXT,
            payload_json TEXT,
            budget INTEGER,
            status TEXT,
            attempts INTEGER,
            last_error TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        INSERT INTO backlog_jobs VALUES
            (1, 'summarize', '{"turns": [{"user": "u1", "assistant": "a1"}]}',
             5000, 'done', 0, NULL, '2026-03-12', '2026-03-15');
    """)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def personal_log(tmp_path_obj: Path):
    path = str(tmp_path_obj / "personal_log.json")
    data = {
        "version": "1.0.0",
        "created_at": "2026-03-15T00:00:00Z",
        "about": "Test diary",
        "entries": [
            {"timestamp": "2026-03-15T00:00:00Z", "type": "reflection", "content": "Today I learned about substrate slots."},
        ],
        "last_updated": "2026-03-15T00:00:00Z",
        "entry_count": 1,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def test_clean_text_keeps_alnum_and_spaces():
    assert clean_text("dog->animal_42\n") == "dog animal 42"


def test_clean_text_truncates():
    assert clean_text("abcdef", max_chars=3) == "abc"


def test_source_pointer():
    assert source_pointer("D:\\00\\a.db", "t", 7) == "D:\\00\\a.db:t:7"


def test_json_source_pointer():
    assert json_source_pointer("D:\\00\\log.json", 3) == "D:\\00\\log.json:entry:3"


def test_build_entity_container_has_is_a_edge():
    row = {"id": 1, "name": "Jeffrey", "type": "person", "attributes": None}
    c = build_entity_container(row, "extracted_entities", "a.db", [])
    assert c.kind == "person"
    assert c.text == "Jeffrey"
    assert any(e.edge_type == "is a" and e.target == "person" for e in c.edges)
    assert c.status == ContainerStatus.DORMANT


def test_build_fact_container():
    row = {"id": 1, "key": "capability", "value": "uses Twilio"}
    c, edges = build_fact_container(row, "extracted_facts", "a.db", [])
    assert c.kind == "fact"
    assert "capability" in c.text
    assert len(edges) == 1
    assert edges[0].edge_type == "capability"
    assert edges[0].target == "uses Twilio"


def test_build_procedure_container():
    row = {
        "id": 1,
        "name": "Start Axon",
        "trigger": "boot",
        "steps_json": '["step1", "step2"]',
        "outcome": "running",
        "applicability": "local",
    }
    c = build_procedure_container(row, "extracted_procedures", "a.db", [])
    assert c.kind == "procedure"
    assert c.metadata["steps"] == ["step1", "step2"]


def test_build_diary_container_redaction():
    entry = {"timestamp": "t", "type": "reflection", "content": "Hello world"}
    c = build_diary_container(entry, 0, "log.json", [])
    assert c.kind == "diary"
    assert c.metadata["redaction"] == "diary_only"


def test_sha256_file(tmp_path_obj: Path):
    p = tmp_path_obj / "x.txt"
    p.write_text("hello", encoding="utf-8")
    assert sha256_file(str(p)) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


# ---------------------------------------------------------------------------
# Builder pipeline tests
# ---------------------------------------------------------------------------


class TestRecoveredCorpusBuilder:
    def test_smoke_run_writes_artifacts(self, tmp_path_obj, semantic_db, episodic_db, backlog_db, personal_log):
        out_dir = str(tmp_path_obj / "out")
        builder = RecoveredCorpusBuilder(
            semantic_db=semantic_db,
            episodic_db=episodic_db,
            backlog_db=backlog_db,
            personal_log=personal_log,
            out_dir=out_dir,
            limit=10,
            max_items=50,
            no_personal_log=True,
            no_vectors=True,
        )
        stats = builder.run()
        assert stats.container_count > 0
        assert (tmp_path_obj / "out" / "containers.jsonl").exists()
        assert (tmp_path_obj / "out" / "symbol_registry.jsonl").exists()
        assert (tmp_path_obj / "out" / "semantic_edges.jsonl").exists()
        assert (tmp_path_obj / "out" / "layout_groups.jsonl").exists()
        assert (tmp_path_obj / "out" / "corpus_manifest.json").exists()

    def test_manifest_contains_hashes(self, tmp_path_obj, semantic_db, episodic_db, backlog_db, personal_log):
        out_dir = str(tmp_path_obj / "out")
        builder = RecoveredCorpusBuilder(
            semantic_db=semantic_db,
            episodic_db=episodic_db,
            backlog_db=backlog_db,
            personal_log=personal_log,
            out_dir=out_dir,
            limit=10,
            no_personal_log=True,
            no_vectors=True,
        )
        builder.run()
        with open(tmp_path_obj / "out" / "corpus_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert "source_files" in manifest
        hashes = {sf["path"]: sf["sha256"] for sf in manifest["source_files"]}
        assert hashes[os.path.abspath(semantic_db)]
        assert manifest["output_counts"]["containers"] > 0

    def test_personal_log_excluded_by_default(self, tmp_path_obj, semantic_db, episodic_db, backlog_db, personal_log):
        out_dir = str(tmp_path_obj / "out")
        builder = RecoveredCorpusBuilder(
            semantic_db=semantic_db,
            episodic_db=episodic_db,
            backlog_db=backlog_db,
            personal_log=personal_log,
            out_dir=out_dir,
            limit=10,
            no_personal_log=True,
            no_vectors=True,
        )
        builder.run()
        containers = list(load_jsonl(tmp_path_obj / "out" / "containers.jsonl"))
        assert not any(c["kind"] == "diary" for c in containers)

    def test_personal_log_included_when_requested(self, tmp_path_obj, semantic_db, episodic_db, backlog_db, personal_log):
        out_dir = str(tmp_path_obj / "out")
        builder = RecoveredCorpusBuilder(
            semantic_db=semantic_db,
            episodic_db=episodic_db,
            backlog_db=backlog_db,
            personal_log=personal_log,
            out_dir=out_dir,
            limit=10,
            no_personal_log=False,
            no_vectors=True,
        )
        builder.run()
        containers = list(load_jsonl(tmp_path_obj / "out" / "containers.jsonl"))
        assert any(c["kind"] == "diary" for c in containers)

    def test_max_items_stops_early(self, tmp_path_obj, semantic_db):
        out_dir = str(tmp_path_obj / "out")
        builder = RecoveredCorpusBuilder(
            semantic_db=semantic_db,
            episodic_db="missing.db",
            backlog_db="missing.db",
            personal_log="missing.json",
            out_dir=out_dir,
            max_items=2,
            no_vectors=True,
        )
        stats = builder.run()
        assert stats.container_count <= 2

    def test_semantic_edges_do_not_emit_symbols(self, tmp_path_obj, semantic_db):
        out_dir = str(tmp_path_obj / "out")
        builder = RecoveredCorpusBuilder(
            semantic_db=semantic_db,
            episodic_db="missing.db",
            backlog_db="missing.db",
            personal_log="missing.json",
            out_dir=out_dir,
            no_vectors=True,
        )
        builder.run()
        for obj in load_jsonl(tmp_path_obj / "out" / "containers.jsonl"):
            assert obj["symbols"] == []
            for edge in obj["edges"]:
                assert edge.get("symbol") in (None, "")

        for edge in load_jsonl(tmp_path_obj / "out" / "semantic_edges.jsonl"):
            assert "symbol" not in edge

    def test_same_edge_type_different_targets_stay_spelled_out(self, tmp_path_obj, semantic_db):
        conn = sqlite3.connect(semantic_db)
        conn.execute(
            "INSERT INTO extracted_relations VALUES (?, ?, ?, ?, ?, ?, ?)",
            (2, "dog", "is_a", "animal", 1.0, "memory_agent", "2026-03-15"),
        )
        conn.execute(
            "INSERT INTO extracted_relations VALUES (?, ?, ?, ?, ?, ?, ?)",
            (3, "dog", "is_a", "tool", 1.0, "memory_agent", "2026-03-15"),
        )
        conn.commit()
        conn.close()

        out_dir = str(tmp_path_obj / "out")
        builder = RecoveredCorpusBuilder(
            semantic_db=semantic_db,
            episodic_db="missing.db",
            backlog_db="missing.db",
            personal_log="missing.json",
            out_dir=out_dir,
            no_vectors=True,
        )
        builder.run()
        edges = list(load_jsonl(tmp_path_obj / "out" / "semantic_edges.jsonl"))
        targets = {
            e["target"]
            for e in edges
            if e["source_text"].startswith("dog") and e["edge_type"] == "is a"
        }
        assert {"animal", "tool"}.issubset(targets)

    def test_relation_confidence_is_clamped(self):
        c, edges = build_relation_container(
            {
                "id": 1,
                "subject": "dog",
                "predicate": "is_a",
                "object": "animal",
                "confidence": 1.9,
            },
            "extracted_relations",
            "a.db",
            [],
        )
        assert c.confidence == 1.0
        assert edges[0].confidence == 1.0

    def test_readonly_does_not_mutate_source(self, semantic_db):
        import hashlib
        with open(semantic_db, "rb") as f:
            before = hashlib.md5(f.read()).hexdigest()
        builder = RecoveredCorpusBuilder(
            semantic_db=semantic_db,
            episodic_db="missing.db",
            backlog_db="missing.db",
            personal_log="missing.json",
            out_dir=tempfile.mkdtemp(),
            limit=5,
            no_vectors=True,
            dry_run=True,
        )
        builder.run()
        with open(semantic_db, "rb") as f:
            after = hashlib.md5(f.read()).hexdigest()
        assert before == after


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
