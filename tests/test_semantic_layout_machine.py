#!/usr/bin/env python3
"""Tests for semantic_layout_machine.py

Tests cover:
- Symbol generation: deterministic and substrate-safe
- Broad kind classification
- Temp sqlite ingestion for entities/facts/relations/procedures
- Artifact writing counts and basic schema keys
- No source DB mutation
- Vector blob decoding (optional)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path so we can import the module directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_layout_machine import (
    BROAD_KINDS,
    SYMBOL_ALPHABET,
    SYMBOL_ALPHABET_SET,
    SYMBOL_BASE,
    SemanticLayoutMachine,
    SymbolRegistry,
    classify_broad_kind,
    decode_vector_blob,
    generate_symbol,
    int_to_symbol,
    is_substrate_safe,
    lexical_bucket_key,
    open_readonly,
)
from container_schema import Container, ContainerStatus


# ---------------------------------------------------------------------------
# Fixtures: temp SQLite DBs
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path():
    """Create a temp .db file path. Cleaned up after the test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


def create_semantic_db(db_path: str) -> None:
    """Create a small artificial axon_semantic_memory.db with known tables."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE extracted_entities (
            id INTEGER PRIMARY KEY,
            name TEXT,
            entity_type TEXT,
            definition TEXT,
            edges TEXT
        );
        INSERT INTO extracted_entities VALUES
            (1, 'Jeffrey', 'person', 'Jeffrey is the architect of Axon.', NULL),
            (2, 'D:\\Axon', 'location', 'Axon project root directory.', NULL),
            (3, 'Ollama', 'tool', 'Ollama runs models locally.', NULL),
            (4, 'Axon', 'entity', 'Axon is a forever-ticking agent.', NULL),
            (5, 'Cloudflare', 'organization', 'Cloudflare provides DNS.', NULL);

        CREATE TABLE extracted_facts (
            id INTEGER PRIMARY KEY,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            definition TEXT
        );
        INSERT INTO extracted_facts VALUES
            (1, 'Axon', 'uses', '16D substrate', 'Axon uses a 16D substrate.'),
            (2, 'Jeffrey', 'is a', 'person', 'Jeffrey is a person.');

        CREATE TABLE extracted_relations (
            id INTEGER PRIMARY KEY,
            source TEXT,
            relation TEXT,
            target TEXT
        );
        INSERT INTO extracted_relations VALUES
            (1, 'Axon', 'collaborates with', 'Dexter'),
            (2, 'Axon', 'depends on', 'Ollama');

        CREATE TABLE extracted_procedures (
            id INTEGER PRIMARY KEY,
            name TEXT,
            steps TEXT,
            outcome TEXT,
            applicability TEXT,
            description TEXT
        );
        INSERT INTO extracted_procedures VALUES
            (1, 'Start Axon', '["cd Axon","python axon.py"]', 'Axon running', 'When Axon needs to start', 'Boot procedure');

        CREATE TABLE knowledge_vectors (
            id INTEGER PRIMARY KEY,
            name TEXT,
            vector BLOB
        );
    """)
    # Insert vector blobs as raw float32
    import struct
    vectors = [
        ([1.0, 0.0, 0.0, 0.0], "alpha"),
        ([0.0, 1.0, 0.0, 0.0], "beta"),
        ([1.0, 1.0, 0.0, 0.0], "gamma"),
        ([0.5, 0.5, 0.5, 0.5], "delta"),
    ]
    for i, (vec, name) in enumerate(vectors):
        blob = struct.pack(f"<{len(vec)}f", *vec)
        conn.execute("INSERT INTO knowledge_vectors VALUES (?, ?, ?)", (i, name, blob))
    conn.commit()
    conn.close()


@pytest.fixture
def semantic_db(tmp_db_path):
    """Create a populated semantic DB and return its path."""
    create_semantic_db(tmp_db_path)
    return tmp_db_path


@pytest.fixture
def tmp_out_dir():
    """Create a temp output directory."""
    d = tempfile.mkdtemp(prefix="semantic_layout_test_")
    yield d
    # Cleanup
    import shutil
    if os.path.exists(d):
        shutil.rmtree(d)


# ---------------------------------------------------------------------------
# Symbol generation tests
# ---------------------------------------------------------------------------


class TestSymbolGeneration:
    """Tests for deterministic, substrate-safe symbol generation."""

    def test_int_to_symbol_zero(self):
        assert int_to_symbol(0) == "A"

    def test_int_to_symbol_one(self):
        assert int_to_symbol(1) == "B"

    def test_int_to_symbol_twentyfive(self):
        assert int_to_symbol(25) == "Z"

    def test_int_to_symbol_twentysix(self):
        assert int_to_symbol(26) == "a"

    def test_int_to_symbol_fiftyone(self):
        assert int_to_symbol(51) == "z"

    def test_int_to_symbol_fiftytwo(self):
        assert int_to_symbol(52) == "0"

    def test_int_to_symbol_sixtyone(self):
        assert int_to_symbol(61) == "9"

    def test_int_to_symbol_sixtytwo_wraps(self):
        # 62 -> BA (two-char)
        assert int_to_symbol(62) == "BA"

    def test_int_to_symbol_deterministic(self):
        """Same integer always maps to the same symbol."""
        for n in range(200):
            assert int_to_symbol(n) == int_to_symbol(n)

    def test_int_to_symbol_unique(self):
        """No two integers map to the same symbol."""
        symbols = [int_to_symbol(n) for n in range(500)]
        assert len(set(symbols)) == len(symbols)

    def test_int_to_symbol_negative_raises(self):
        with pytest.raises(ValueError):
            int_to_symbol(-1)

    def test_generate_symbol_level0_padded(self):
        """Level 0 symbols are at least 2 chars."""
        assert generate_symbol(0, level=0) == "AA"
        assert generate_symbol(1, level=0) == "AB"

    def test_generate_symbol_level1_prepends_parent(self):
        """Level 1+ symbols prepend parent_symbol."""
        assert generate_symbol(0, level=1, parent_symbol="AA") == "AAA"
        assert generate_symbol(1, level=1, parent_symbol="AA") == "AAB"

    def test_is_substrate_safe_valid(self):
        assert is_substrate_safe("AA")
        assert is_substrate_safe("AB")
        assert is_substrate_safe("MKIj")
        assert is_substrate_safe("Zeg")

    def test_is_substrate_safe_invalid(self):
        assert not is_substrate_safe("")
        assert not is_substrate_safe("A A")  # space
        assert not is_substrate_safe("A-A")  # hyphen
        assert not is_substrate_safe("A.A")  # period
        assert not is_substrate_safe("A_B")  # underscore

    def test_symbol_alphabet_contains_all_safe_chars(self):
        """The symbol alphabet has exactly A-Z, a-z, 0-9."""
        assert len(SYMBOL_ALPHABET) == 62
        assert "A" in SYMBOL_ALPHABET
        assert "Z" in SYMBOL_ALPHABET
        assert "a" in SYMBOL_ALPHABET
        assert "z" in SYMBOL_ALPHABET
        assert "0" in SYMBOL_ALPHABET
        assert "9" in SYMBOL_ALPHABET
        assert " " not in SYMBOL_ALPHABET
        assert "." not in SYMBOL_ALPHABET
        assert "-" not in SYMBOL_ALPHABET


# ---------------------------------------------------------------------------
# Broad kind classification tests
# ---------------------------------------------------------------------------


class TestBroadKindClassification:
    """Tests for broad kind classification heuristics."""

    def test_entity_type_person(self):
        assert classify_broad_kind("Jeffrey", entity_type="person") == "person"

    def test_entity_type_location(self):
        assert classify_broad_kind("D:\\Axon", entity_type="location") == "place"

    def test_entity_type_tool(self):
        assert classify_broad_kind("Ollama", entity_type="tool") == "tool"

    def test_entity_type_organization(self):
        assert classify_broad_kind("Cloudflare", entity_type="organization") == "organization"

    def test_table_source_procedure(self):
        assert classify_broad_kind("something", table_source="extracted_procedures") == "procedure"

    def test_table_source_fact(self):
        assert classify_broad_kind("something", table_source="extracted_facts") == "fact"

    def test_table_source_relation(self):
        assert classify_broad_kind("something", table_source="extracted_relations") == "relation"

    def test_text_heuristic_person(self):
        assert classify_broad_kind("Jeffrey is a user") == "person"

    def test_text_heuristic_place(self):
        assert classify_broad_kind("Server location directory") == "place"

    def test_text_heuristic_tool(self):
        assert classify_broad_kind("This is a daemon script") == "tool"

    def test_text_heuristic_concept(self):
        assert classify_broad_kind("This is a methodology framework") == "concept"

    def test_text_heuristic_fact(self):
        assert classify_broad_kind("The definition of this attribute") == "fact"

    def test_fallback_entity(self):
        """Non-empty text with no heuristic match falls back to 'entity'."""
        assert classify_broad_kind("xyz123") == "entity"

    def test_fallback_unknown(self):
        """Empty text falls back to 'unknown'."""
        assert classify_broad_kind("") == "unknown"

    def test_all_broad_kinds_exist(self):
        """All required broad kinds are present."""
        expected = {"person", "place", "organization", "tool", "procedure", "fact", "relation", "concept", "entity", "unknown"}
        assert expected == set(BROAD_KINDS)


# ---------------------------------------------------------------------------
# SymbolRegistry tests
# ---------------------------------------------------------------------------


class TestSymbolRegistry:
    """Tests for the SymbolRegistry."""

    def test_broad_symbol_creation(self):
        reg = SymbolRegistry()
        entry = reg.get_or_create_broad("person")
        assert entry.symbol == "AA"
        assert entry.level == 0
        assert entry.broad_kind == "person"
        assert entry.item_count == 0

    def test_broad_symbol_deterministic(self):
        """Same broad kind returns the same entry."""
        reg = SymbolRegistry()
        e1 = reg.get_or_create_broad("person")
        e2 = reg.get_or_create_broad("person")
        assert e1.symbol == e2.symbol
        assert e1 is e2

    def test_broad_symbol_increment(self):
        reg = SymbolRegistry()
        e1 = reg.get_or_create_broad("person")
        e1.item_count += 1
        e1.item_count += 1
        assert e1.item_count == 2

    def test_child_symbol_creation(self):
        reg = SymbolRegistry()
        broad = reg.get_or_create_broad("person")
        child = reg.get_or_create_child("lex_abc", broad.symbol, level=1, broad_kind="person")
        assert child.parent_symbol == broad.symbol
        assert child.level == 1
        assert child.symbol == broad.symbol + "A"  # first child

    def test_child_symbol_in_parent_children(self):
        reg = SymbolRegistry()
        broad = reg.get_or_create_broad("person")
        child = reg.get_or_create_child("lex_abc", broad.symbol, level=1, broad_kind="person")
        assert child.symbol in broad.child_symbols

    def test_child_symbol_deterministic(self):
        """Same child name under same parent returns same entry."""
        reg = SymbolRegistry()
        broad = reg.get_or_create_broad("person")
        c1 = reg.get_or_create_child("lex_abc", broad.symbol, level=1)
        c2 = reg.get_or_create_child("lex_abc", broad.symbol, level=1)
        assert c1.symbol == c2.symbol
        assert c1 is c2

    def test_all_entries_sorted_by_level(self):
        reg = SymbolRegistry()
        reg.get_or_create_broad("person")
        reg.get_or_create_broad("tool")
        broad = reg.get_or_create_broad("concept")
        reg.get_or_create_child("lex_a", broad.symbol, level=1)
        entries = reg.all_entries()
        assert entries[0].level == 0
        assert entries[-1].level >= 1

    def test_registry_jsonl(self):
        reg = SymbolRegistry()
        reg.get_or_create_broad("person")
        reg.get_or_create_child("lex_a", reg.get_or_create_broad("person").symbol, level=1)
        jsonl = reg.to_jsonl()
        lines = jsonl.strip().split("\n")
        assert len(lines) >= 2
        for line in lines:
            obj = json.loads(line)
            assert "symbol" in obj
            assert "name" in obj
            assert "level" in obj


# ---------------------------------------------------------------------------
# Temp SQLite ingestion tests
# ---------------------------------------------------------------------------


class TestSQLiteIngestion:
    """Tests for reading from temp SQLite DBs."""

    def test_open_readonly(self, semantic_db):
        """Can open the DB read-only."""
        conn = open_readonly(semantic_db)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t[0] for t in tables]
        assert "extracted_entities" in table_names
        assert "extracted_facts" in table_names
        assert "extracted_relations" in table_names
        assert "extracted_procedures" in table_names
        assert "knowledge_vectors" in table_names
        conn.close()

    def test_readonly_does_not_mutate(self, semantic_db):
        """Opening read-only should not change the file. Verify by checking
        that a write attempt fails and the file is unchanged."""
        import hashlib
        with open(semantic_db, "rb") as f:
            hash_before = hashlib.md5(f.read()).hexdigest()

        conn = open_readonly(semantic_db)
        # Attempting a write should fail because the DB is opened read-only
        try:
            conn.execute("INSERT INTO extracted_entities (name) VALUES ('hack')")
            conn.commit()
            # If we get here, the DB wasn't read-only — fail the test
            assert False, "Write should have failed on read-only DB"
        except sqlite3.OperationalError:
            pass  # expected
        finally:
            conn.close()

        with open(semantic_db, "rb") as f:
            hash_after = hashlib.md5(f.read()).hexdigest()
        assert hash_before == hash_after, "Source DB was mutated!"

    def test_ingest_entities(self, semantic_db, tmp_out_dir):
        """Full pipeline processes entity rows into containers."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
        )
        stats = machine.run()
        # 5 entities + 2 facts + 2 relations + 1 procedure = 10 containers
        assert stats.container_count == 10
        assert stats.edge_count > 0
        assert stats.symbol_count > 0

    def test_ingest_entities_only_with_limit(self, semantic_db, tmp_out_dir):
        """Limit restricts rows per table."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
            limit=2,
        )
        stats = machine.run()
        # 2 entities + 2 facts + 2 relations + 1 procedure = 7 (procedures table has 1 row)
        assert stats.container_count == 7

    def test_ingest_with_max_items(self, semantic_db, tmp_out_dir):
        """Max items stops processing early."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
            max_items=3,
        )
        stats = machine.run()
        assert stats.container_count <= 3


# ---------------------------------------------------------------------------
# Artifact writing tests
# ---------------------------------------------------------------------------


class TestArtifactWriting:
    """Tests for output artifact schema and counts."""

    def test_artifacts_written(self, semantic_db, tmp_out_dir):
        """All 5 required artifacts are written."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
        )
        machine.run()
        out = Path(tmp_out_dir)
        assert (out / "containers.jsonl").exists()
        assert (out / "symbol_registry.jsonl").exists()
        assert (out / "semantic_edges.jsonl").exists()
        assert (out / "layout_groups.jsonl").exists()
        assert (out / "corpus_manifest.json").exists()

    def test_containers_jsonl_schema(self, semantic_db, tmp_out_dir):
        """Each container line has required schema keys."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
        )
        machine.run()
        with open(Path(tmp_out_dir) / "containers.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                assert "container_id" in obj
                assert "kind" in obj
                assert "text" in obj
                assert "normalized_text" in obj
                assert "letters" in obj
                assert "edges" in obj
                assert "symbols" in obj
                assert "source" in obj
                assert "provenance" in obj
                assert "status" in obj
                assert obj["status"] == "dormant"
                assert obj["letters"] == obj["text"]
                assert "layout_symbols" in obj["metadata"]

    def test_symbol_registry_jsonl_schema(self, semantic_db, tmp_out_dir):
        """Each symbol registry line has required keys."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
        )
        machine.run()
        with open(Path(tmp_out_dir) / "symbol_registry.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                assert "symbol" in obj
                assert "name" in obj
                assert "level" in obj
                assert "parent_symbol" in obj
                assert "group_path" in obj
                assert "item_count" in obj
                assert "source" in obj
                assert "provenance" in obj
                assert "status" in obj
                # Symbols must be substrate-safe
                assert is_substrate_safe(obj["symbol"])

    def test_semantic_edges_jsonl_schema(self, semantic_db, tmp_out_dir):
        """Each semantic edge line has required keys."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
        )
        machine.run()
        edges_path = Path(tmp_out_dir) / "semantic_edges.jsonl"
        with open(edges_path, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) > 0
        for line in lines:
            obj = json.loads(line)
            assert "edge_type" in obj
            assert "target" in obj
            assert "confidence" in obj
            assert "provenance" in obj
            assert "status" in obj
            assert 0.0 <= obj["confidence"] <= 1.0

    def test_layout_groups_jsonl_schema(self, semantic_db, tmp_out_dir):
        """Each layout group line has required keys."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
        )
        machine.run()
        with open(Path(tmp_out_dir) / "layout_groups.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                assert "symbol" in obj
                assert "level" in obj
                assert "name" in obj
                assert "parent_symbol" in obj
                assert "broad_kind" in obj
                assert "item_count" in obj
                assert "child_symbols" in obj

    def test_corpus_manifest_schema(self, semantic_db, tmp_out_dir):
        """The corpus manifest has required keys."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
        )
        machine.run()
        with open(Path(tmp_out_dir) / "corpus_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        assert "source_dbs" in manifest
        assert "table_counts" in manifest
        assert "output_counts" in manifest
        assert "options" in manifest
        assert "created_at" in manifest
        assert "notes" in manifest
        assert manifest["output_counts"]["containers"] > 0

    def test_dry_run_does_not_write(self, semantic_db, tmp_out_dir):
        """Dry run processes but does not write artifacts."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
            dry_run=True,
        )
        stats = machine.run()
        assert stats.container_count > 0
        assert not Path(tmp_out_dir, "containers.jsonl").exists()

    def test_container_symbols_stay_empty_for_spelled_out_edges(self, semantic_db, tmp_out_dir):
        """Container symbols are not used for semantic edge meaning."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
        )
        machine.run()
        with open(Path(tmp_out_dir) / "containers.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                assert obj["symbols"] == []
                for edge in obj["edges"]:
                    assert edge.get("symbol") in (None, "")

    def test_levels_control_grouping_depth(self, semantic_db, tmp_out_dir):
        """Levels control layout metadata depth, not ensemble-visible symbols."""
        machine1 = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
            levels=1,
        )
        machine1.run()
        with open(Path(tmp_out_dir) / "containers.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                assert len(obj["metadata"]["layout_symbols"]) == 1

        # Wipe and run with levels=2
        import shutil
        shutil.rmtree(tmp_out_dir)
        os.makedirs(tmp_out_dir)

        machine2 = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
            levels=2,
        )
        machine2.run()
        with open(Path(tmp_out_dir) / "containers.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                assert len(obj["metadata"]["layout_symbols"]) >= 2

    def test_levels_three_adds_deeper_symbols(self, semantic_db, tmp_out_dir):
        """Levels=3 gives deeper layout metadata symbols."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
            levels=3,
        )
        machine.run()
        with open(Path(tmp_out_dir) / "containers.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                assert len(obj["metadata"]["layout_symbols"]) >= 3
                for sym in obj["metadata"]["layout_symbols"]:
                    assert is_substrate_safe(sym)

    def test_edges_are_exported_without_symbols(self, semantic_db, tmp_out_dir):
        """Semantic edges are exported as English edge_type + target only."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
        )
        machine.run()
        with open(Path(tmp_out_dir) / "semantic_edges.jsonl", encoding="utf-8") as f:
            edges = [json.loads(line.strip()) for line in f if line.strip()]
        assert edges
        for edge in edges:
            assert "symbol" not in edge
            assert edge["edge_type"]
            assert edge["target"]


# ---------------------------------------------------------------------------
# Vector blob decoding tests
# ---------------------------------------------------------------------------


class TestVectorBlobDecoding:
    """Tests for vector blob decoding."""

    def test_decode_float32_blob(self):
        """Can decode a raw float32 blob."""
        import struct
        vec = [1.0, 2.0, 3.0, 4.0]
        blob = struct.pack(f"<{len(vec)}f", *vec)
        decoded = decode_vector_blob(blob)
        assert decoded is not None
        assert len(decoded) == 4
        assert decoded[0] == pytest.approx(1.0)
        assert decoded[3] == pytest.approx(4.0)

    def test_decode_json_string(self):
        """Can decode a JSON list string."""
        vec = [0.1, 0.2, 0.3]
        decoded = decode_vector_blob(json.dumps(vec))
        assert decoded == vec

    def test_decode_none(self):
        """None input returns None."""
        assert decode_vector_blob(None) is None

    def test_decode_garbage(self):
        """Garbage bytes return None (or a best-effort list), not crash."""
        decoded = decode_vector_blob(b"\x00\x01\x02")
        # Could be None or a short list — just ensure no crash
        assert decoded is None or isinstance(decoded, list)

    def test_decode_list_passthrough(self):
        """A list passes through."""
        decoded = decode_vector_blob([1.0, 2.0])
        assert decoded == [1.0, 2.0]

    def test_vector_grouping_in_machine(self, semantic_db, tmp_out_dir):
        """The machine processes vector tables and records samples."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
            max_vector_samples=4,
        )
        stats = machine.run()
        assert stats.vector_sample_count == 4

    def test_no_vectors_flag_skips_vectors(self, semantic_db, tmp_out_dir):
        """--no-vectors skips vector processing."""
        machine = SemanticLayoutMachine(
            semantic_db=semantic_db,
            out_dir=tmp_out_dir,
            no_vectors=True,
        )
        stats = machine.run()
        assert stats.vector_sample_count == 0


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Tests that the machine is deterministic across runs."""

    def test_same_input_same_output(self, semantic_db, tmp_out_dir):
        """Two runs with the same input produce the same symbol counts."""
        dir1 = os.path.join(tmp_out_dir, "run1")
        dir2 = os.path.join(tmp_out_dir, "run2")
        os.makedirs(dir1, exist_ok=True)
        os.makedirs(dir2, exist_ok=True)

        m1 = SemanticLayoutMachine(semantic_db=semantic_db, out_dir=dir1)
        s1 = m1.run()

        m2 = SemanticLayoutMachine(semantic_db=semantic_db, out_dir=dir2)
        s2 = m2.run()

        assert s1.container_count == s2.container_count
        assert s1.symbol_count == s2.symbol_count
        assert s1.broad_kind_counts == s2.broad_kind_counts

        # Compare symbol registries
        with open(Path(dir1) / "symbol_registry.jsonl") as f1, \
             open(Path(dir2) / "symbol_registry.jsonl") as f2:
            assert f1.read() == f2.read()


# ---------------------------------------------------------------------------
# Lexical bucket tests
# ---------------------------------------------------------------------------


class TestLexicalBucket:
    """Tests for lexical bucket key generation."""

    def test_basic_bucket(self):
        assert lexical_bucket_key("dog") == "dog"

    def test_truncates_to_max_chars(self):
        assert lexical_bucket_key("elephant") == "ele"

    def test_strips_non_alphanumeric(self):
        assert lexical_bucket_key("D:\\Axon") == "dax"

    def test_empty_returns_underscore(self):
        assert lexical_bucket_key("") == "_"

    def test_only_punctuation_returns_underscore(self):
        assert lexical_bucket_key("---...") == "_"

    def test_case_insensitive(self):
        assert lexical_bucket_key("Dog") == lexical_bucket_key("dog")
