#!/usr/bin/env python3
"""semantic_layout_machine.py - bootstrap semantic layout from recovered Axon DBs.

This is the first semantic layout machine for Axon. It reads recovered SQLite
memory DBs in streaming fashion, converts extracted entities/facts/relations/
procedures into container_schema-compatible Container records, assigns
substrate-safe semantic symbols via a deterministic base registry, and emits
JSONL/JSON artifacts that become dormant structured knowledge / container
curriculum.

Architecture alignment (SOURCE_OF_TRUTH.md):
  - Does NOT wait for a trained semantic core. It bootstraps a deterministic
    base registry + dormant state from recovered DBs first.
  - The semantic core later trains as curator/builder from this ground truth.
  - Rails are not involved here (shared state infrastructure, not this importer).
  - Input goes through the shared field conceptually; the soul is not involved
    in this importer.
  - Output is dormant structured knowledge / container curriculum.

Design constraints:
  - Streaming SQLite reads. Never loads all rows into memory.
  - Never mutates source DBs (opens read-only).
  - No heavy dependencies. stdlib + numpy only (numpy optional for vectors).
  - Substrate-safe symbols: A-Z, a-z, 0-9 only.
  - Layout hierarchy is metadata/indexing only. Container.symbols contains
    symbols created for actual SemanticEdge records only.
  - Deterministic symbol generation. Same input -> same symbols every run.

CLI:
    python semantic_layout_machine.py --semantic-db D:\\00\\axon_semantic_memory.db \\
        --out-dir datasets\\recovered\\semantic_layout --limit 1000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# --- container_schema integration (same repo, dependency-free) ---
from container_schema import Container, ContainerStatus, EdgeType, SemanticEdge, normalize_container

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEMANTIC_DB = r"D:\00\axon_semantic_memory.db"
DEFAULT_OLD_DB = r"D:\00\axon_memory.db"
DEFAULT_EPISODIC_DB = r"D:\00\axon_episodic_memory.db"
DEFAULT_OUT_DIR = os.path.join("datasets", "recovered", "semantic_layout")

# Substrate-safe alphabet: exactly what the 16D LetterBank supports for
# symbol codes (uppercase, lowercase, digits). No spaces, no punctuation.
SYMBOL_ALPHABET = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [chr(c) for c in range(ord("0"), ord("9") + 1)]
)
SYMBOL_ALPHABET_SET = set(SYMBOL_ALPHABET)
SYMBOL_BASE = len(SYMBOL_ALPHABET)  # 62

# Broad kinds (level 0)
BROAD_KINDS: List[str] = [
    "person",
    "place",
    "organization",
    "tool",
    "procedure",
    "fact",
    "relation",
    "concept",
    "entity",
    "unknown",
]

# Broad-kind heuristic patterns (checked in order; first match wins).
# These are deliberately conservative — they classify based on entity type
# field, table source, and simple text patterns. They do not overclaim.
_PERSON_HINTS = re.compile(
    r"\b(person|user|author|admin|engineer|manager|analyst|developer|jeff|jeffrey)\b",
    re.IGNORECASE,
)
_PLACE_HINTS = re.compile(
    r"\b(place|location|directory|path|folder|city|country|server|host|localhost|ip|port|address)\b",
    re.IGNORECASE,
)
_ORG_HINTS = re.compile(
    r"\b(organization|company|corporation|team|agency|association|nvidia|microsoft|amazon|heroku|salesforce)\b",
    re.IGNORECASE,
)
_TOOL_HINTS = re.compile(
    r"\b(tool|script|daemon|service|server|module|library|api|endpoint|function|process|executable|binary)\b",
    re.IGNORECASE,
)
_PROCEDURE_HINTS = re.compile(
    r"\b(procedure|step|workflow|process|method|algorithm|protocol|routine|task)\b",
    re.IGNORECASE,
)
_FACT_HINTS = re.compile(
    r"\b(fact|property|attribute|definition|parameter|setting|config|status|state)\b",
    re.IGNORECASE,
)
_CONCEPT_HINTS = re.compile(
    r"\b(concept|theory|principle|idea|methodology|framework|model|pattern)\b",
    re.IGNORECASE,
)

# Entity type -> broad kind map (used when DB row has an entity_type column).
ENTITY_TYPE_MAP: Dict[str, str] = {
    "person": "person",
    "user": "person",
    "people": "person",
    "place": "place",
    "location": "place",
    "address": "place",
    "organization": "organization",
    "company": "organization",
    "org": "organization",
    "team": "organization",
    "tool": "tool",
    "software": "tool",
    "application": "tool",
    "service": "tool",
    "daemon": "tool",
    "process": "tool",
    "procedure": "procedure",
    "workflow": "procedure",
    "fact": "fact",
    "property": "fact",
    "attribute": "fact",
    "relation": "relation",
    "relationship": "relation",
    "concept": "concept",
    "theory": "concept",
    "entity": "entity",
}

# ---------------------------------------------------------------------------
# Symbol generation
# ---------------------------------------------------------------------------


def int_to_symbol(n: int) -> str:
    """Convert a non-negative integer to a substrate-safe symbol string.

    Uses base-62 encoding with SYMBOL_ALPHABET (A-Z, a-z, 0-9).
    0 -> 'A', 1 -> 'B', ..., 25 -> 'Z', 26 -> 'a', ..., 51 -> 'z', 52 -> '0',
    53 -> '1', ..., 61 -> '9', 62 -> 'BA', etc.

    This is deterministic and unique. Same integer always maps to the same
    symbol. No two integers map to the same symbol.
    """
    if n < 0:
        raise ValueError(f"int_to_symbol requires n >= 0, got {n}")
    if n == 0:
        return SYMBOL_ALPHABET[0]  # 'A'
    digits: List[str] = []
    while n > 0:
        n, r = divmod(n, SYMBOL_BASE)
        digits.append(SYMBOL_ALPHABET[r])
    digits.reverse()
    return "".join(digits)


def is_substrate_safe(s: str) -> bool:
    """Check that every character in s is in the substrate-safe symbol set."""
    return len(s) > 0 and all(c in SYMBOL_ALPHABET_SET for c in s)


def generate_symbol(index: int, level: int = 0, parent_symbol: str = "") -> str:
    """Generate a deterministic, unique, substrate-safe symbol for a group.

    At level 0 (broad): symbols are AA, AB, AC, ... (two-char prefix + index).
    At deeper levels: parent_symbol + child_index encoded.
    """
    sym = int_to_symbol(index)
    if level == 0:
        # Broad groups: pad to at least 2 chars for readability.
        return sym.rjust(2, SYMBOL_ALPHABET[0])
    else:
        return parent_symbol + sym


# ---------------------------------------------------------------------------
# SymbolRegistry
# ---------------------------------------------------------------------------


@dataclass
class SymbolEntry:
    """A registered semantic symbol record."""

    symbol: str
    name: str
    level: int
    parent_symbol: str = ""
    group_path: str = ""
    item_count: int = 0
    source: str = ""
    provenance: str = ""
    status: str = "dormant"
    broad_kind: str = ""
    child_symbols: List[str] = field(default_factory=list)
    centroid: Optional[List[float]] = None  # for vector-based groups

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "symbol": self.symbol,
            "name": self.name,
            "level": self.level,
            "parent_symbol": self.parent_symbol,
            "group_path": self.group_path,
            "item_count": self.item_count,
            "source": self.source,
            "provenance": self.provenance,
            "status": self.status,
            "broad_kind": self.broad_kind,
            "child_symbols": list(self.child_symbols),
        }
        if self.centroid is not None:
            d["centroid"] = self.centroid
        return d


class SymbolRegistry:
    """Deterministic symbol registry. Assigns unique substrate-safe symbols."""

    def __init__(self, source: str = "semantic_layout_machine"):
        self.source = source
        self._entries: Dict[str, SymbolEntry] = {}
        self._symbol_index: Dict[str, SymbolEntry] = {}
        self._broad_index: Dict[str, int] = {}  # broad_kind -> next index
        self._child_index: Dict[str, int] = {}  # parent_symbol -> next index

    def get_or_create_broad(self, broad_kind: str) -> SymbolEntry:
        """Get or create the level-0 symbol for a broad kind."""
        if broad_kind in self._entries:
            return self._entries[broad_kind]
        # Assign index based on order of first appearance
        idx = len(self._broad_index)
        self._broad_index[broad_kind] = idx
        symbol = generate_symbol(idx, level=0)
        entry = SymbolEntry(
            symbol=symbol,
            name=broad_kind,
            level=0,
            parent_symbol="",
            group_path=broad_kind,
            item_count=0,
            source=self.source,
            provenance="broad_kind_classification",
            status="dormant",
            broad_kind=broad_kind,
        )
        self._entries[broad_kind] = entry
        self._symbol_index[symbol] = entry
        return entry

    def get_or_create_child(
        self,
        name: str,
        parent_symbol: str,
        level: int,
        broad_kind: str = "",
        centroid: Optional[List[float]] = None,
    ) -> SymbolEntry:
        """Get or create a child group symbol under a parent."""
        key = f"{parent_symbol}::{name}"
        if key in self._entries:
            entry = self._entries[key]
            parent = self._symbol_index.get(parent_symbol)
            if parent and entry.symbol not in parent.child_symbols:
                parent.child_symbols.append(entry.symbol)
            return entry
        idx = self._child_index.get(parent_symbol, 0)
        self._child_index[parent_symbol] = idx + 1
        symbol = generate_symbol(idx, level=level, parent_symbol=parent_symbol)
        parent = self._symbol_index.get(parent_symbol)
        parent_path = parent.group_path if parent else parent_symbol
        group_path = f"{parent_path}/{name}"
        entry = SymbolEntry(
            symbol=symbol,
            name=name,
            level=level,
            parent_symbol=parent_symbol,
            group_path=group_path,
            item_count=0,
            source=self.source,
            provenance="lexical_bucket" if centroid is None else "vector_cluster",
            status="dormant",
            broad_kind=broad_kind,
            centroid=centroid,
        )
        self._entries[key] = entry
        self._symbol_index[symbol] = entry
        if parent:
            if symbol not in parent.child_symbols:
                parent.child_symbols.append(symbol)
        return entry

    def all_entries(self) -> List[SymbolEntry]:
        """Return all entries ordered by level, then symbol."""
        return sorted(self._entries.values(), key=lambda e: (e.level, e.symbol))

    def to_jsonl(self) -> str:
        lines = [json.dumps(e.to_dict(), ensure_ascii=False, sort_keys=True) for e in self.all_entries()]
        return "\n".join(lines)

    def count(self) -> int:
        return len(self._entries)


# ---------------------------------------------------------------------------
# LayoutGroup
# ---------------------------------------------------------------------------


@dataclass
class LayoutGroup:
    """A group in the semantic layout hierarchy."""

    symbol: str
    level: int
    name: str
    parent_symbol: str = ""
    broad_kind: str = ""
    item_count: int = 0
    child_symbols: List[str] = field(default_factory=list)
    centroid: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "symbol": self.symbol,
            "level": self.level,
            "name": self.name,
            "parent_symbol": self.parent_symbol,
            "broad_kind": self.broad_kind,
            "item_count": self.item_count,
            "child_symbols": list(self.child_symbols),
        }
        if self.centroid is not None:
            d["centroid"] = self.centroid
        return d


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_broad_kind(
    text: str,
    entity_type: str = "",
    table_source: str = "",
    definition: str = "",
) -> str:
    """Classify a text item into a broad kind.

    Priority:
    1. entity_type column if present and mappable.
    2. table_source if it directly indicates a kind (procedures table -> procedure, etc.).
    3. text/definition heuristic regex.
    4. fallback 'entity' if text is non-empty, else 'unknown'.

    This is deliberately conservative. It does not overclaim quality.
    """
    # 1. entity_type
    if entity_type:
        et = entity_type.strip().lower()
        if et in ENTITY_TYPE_MAP:
            return ENTITY_TYPE_MAP[et]
        # Try partial match
        for key, val in ENTITY_TYPE_MAP.items():
            if key in et:
                return val

    # 2. table_source
    if table_source:
        ts = table_source.lower()
        if "procedure" in ts:
            return "procedure"
        if "fact" in ts or "propert" in ts:
            return "fact"
        if "relation" in ts:
            return "relation"
        if "entit" in ts:
            pass  # entity is the fallback, don't force it here

    # 3. text/definition heuristics
    combined = f"{text} {definition}"
    if _PROCEDURE_HINTS.search(combined):
        return "procedure"
    if _ORG_HINTS.search(combined):
        return "organization"
    if _PERSON_HINTS.search(combined):
        return "person"
    if _PLACE_HINTS.search(combined):
        return "place"
    if _TOOL_HINTS.search(combined):
        return "tool"
    if _FACT_HINTS.search(combined):
        return "fact"
    if _CONCEPT_HINTS.search(combined):
        return "concept"

    # 4. fallback
    if text and text.strip():
        return "entity"
    return "unknown"


# ---------------------------------------------------------------------------
# Lexical bucketing (deterministic finer grouping)
# ---------------------------------------------------------------------------


def lexical_bucket_key(text: str, max_chars: int = 3) -> str:
    """Generate a deterministic lexical bucket key from text.

    Uses the first few alphanumeric characters of the normalized text.
    Items that share the same bucket key are grouped together.
    """
    if not text:
        return "_"
    normalized = re.sub(r"[^A-Za-z0-9]", "", text.lower())
    if not normalized:
        return "_"
    return normalized[:max_chars]


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def open_readonly(db_path: str) -> sqlite3.Connection:
    """Open a SQLite DB in read-only mode. Never mutates the source."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    uri_path = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri_path, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_tables(conn: sqlite3.Connection) -> List[str]:
    """List all table names in the DB."""
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [row[0] for row in cur.fetchall()]


def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """Get column names for a table."""
    cur = conn.execute(f'PRAGMA table_info("{table}")')
    return [row[1] for row in cur.fetchall()]


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    """Count rows in a table safely."""
    cur = conn.execute(f'SELECT COUNT(*) FROM "{table}"')
    return cur.fetchone()[0]


def find_column(columns: List[str], candidates: Sequence[str]) -> Optional[str]:
    """Find the first matching column name from candidates (case-insensitive)."""
    col_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        cl = cand.lower()
        if cl in col_lower:
            return col_lower[cl]
        # partial match
        for col_l, col_real in col_lower.items():
            if cand.lower() in col_l:
                return col_real
    return None


def decode_vector_blob(blob: bytes) -> Optional[Any]:
    """Decode a vector blob from SQLite. Tries numpy first, falls back to JSON.

    Supports:
    - numpy array bytes (via np.frombuffer)
    - JSON list string
    - raw float32 little-endian sequence
    """
    if blob is None:
        return None
    if isinstance(blob, (list, tuple)):
        return list(blob)
    if isinstance(blob, str):
        try:
            return json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            return None
    if isinstance(blob, bytes):
        # Try numpy
        try:
            import numpy as np
            arr = np.frombuffer(blob, dtype=np.float32)
            if len(arr) > 0:
                return arr.tolist()
        except Exception:
            pass
        # Try JSON
        try:
            return json.loads(blob.decode("utf-8", errors="ignore"))
        except (json.JSONDecodeError, ValueError):
            pass
        # Try raw float32
        try:
            import struct
            n = len(blob) // 4
            return list(struct.unpack(f"<{n}f", blob[: n * 4]))
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Streaming row readers
# ---------------------------------------------------------------------------


def stream_rows(
    conn: sqlite3.Connection,
    table: str,
    columns: List[str],
    limit: int = -1,
) -> Iterator[sqlite3.Row]:
    """Stream rows from a table one at a time. Never loads all into memory."""
    col_list = ", ".join('"%s"' % c for c in columns)
    if limit > 0:
        cur = conn.execute('SELECT %s FROM "%s" LIMIT %d' % (col_list, table, limit))
    else:
        cur = conn.execute('SELECT %s FROM "%s"' % (col_list, table))
    while True:
        row = cur.fetchone()
        if row is None:
            break
        yield row
    cur.close()


def row_to_dict(row: sqlite3.Row, columns: List[str]) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a dict with only the given columns."""
    return {col: row[col] for col in columns if col in row.keys()}


# ---------------------------------------------------------------------------
# Container builders
# ---------------------------------------------------------------------------


def build_entity_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Container:
    """Build a Container from an extracted_entities row."""
    text = str(row_dict.get("name") or row_dict.get("entity") or row_dict.get("text") or row_dict.get("word") or "").strip()
    entity_type = str(row_dict.get("entity_type") or row_dict.get("type") or "").strip()
    definition = str(row_dict.get("definition") or row_dict.get("description") or row_dict.get("summary") or "").strip()

    broad_kind = classify_broad_kind(text, entity_type, table_source, definition)

    # Build edges from available columns
    edges: List[SemanticEdge] = []
    # Some DBs store edges as JSON
    raw_edges = row_dict.get("edges") or row_dict.get("relations")
    if raw_edges:
        if isinstance(raw_edges, str):
            try:
                raw_edges = json.loads(raw_edges)
            except (json.JSONDecodeError, ValueError):
                raw_edges = []
        if isinstance(raw_edges, list):
            for e in raw_edges:
                if isinstance(e, dict):
                    edges.append(SemanticEdge.from_dict(e))
                elif isinstance(e, (list, tuple)) and len(e) == 2:
                    edges.append(SemanticEdge(edge_type=str(e[0]), target=str(e[1])))

    c = Container(
        kind=broad_kind,
        text=text,
        source=f"{source_db}:{table_source}",
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.8,  # deterministic import confidence
        provenance="semantic_layout_machine:entity_import",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "entity_type": entity_type,
            "definition": definition,
            "source_table": table_source,
        },
    )
    # Attach category in metadata for compatibility
    if entity_type:
        c.metadata["category"] = entity_type
    return c


def build_fact_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Tuple[Container, List[SemanticEdge]]:
    """Build a Container from an extracted_facts row, plus extracted edges."""
    subject = str(row_dict.get("subject") or row_dict.get("entity") or row_dict.get("name") or "").strip()
    predicate = str(row_dict.get("predicate") or row_dict.get("relation") or row_dict.get("property") or "").strip()
    obj = str(row_dict.get("object") or row_dict.get("value") or row_dict.get("target") or "").strip()
    definition = str(row_dict.get("definition") or row_dict.get("description") or "").strip()

    # The container text is the fact statement
    if subject and predicate and obj:
        text = f"{subject} {predicate} {obj}"
    elif subject and obj:
        text = f"{subject}: {obj}"
    elif subject:
        text = subject
    else:
        text = str(row_dict.get("fact") or row_dict.get("text") or "").strip()

    edges: List[SemanticEdge] = []
    if predicate and obj:
        # Normalize edge type
        et = normalize_edge_type(predicate)
        edges.append(SemanticEdge(
            edge_type=et,
            target=obj,
            confidence=0.8,
            provenance=f"{source_db}:{table_source}",
            status="dormant",
        ))

    c = Container(
        kind="fact",
        text=text,
        source=f"{source_db}:{table_source}",
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.8,
        provenance="semantic_layout_machine:fact_import",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "definition": definition,
            "source_table": table_source,
        },
    )
    return c, edges


def build_relation_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Tuple[Container, List[SemanticEdge]]:
    """Build a Container from an extracted_relations row, plus edges."""
    source_text = str(row_dict.get("source") or row_dict.get("subject") or row_dict.get("entity") or "").strip()
    relation = str(row_dict.get("relation") or row_dict.get("predicate") or row_dict.get("type") or "").strip()
    target = str(row_dict.get("target") or row_dict.get("object") or row_dict.get("destination") or "").strip()

    if source_text and relation and target:
        text = f"{source_text} {relation} {target}"
    elif source_text and target:
        text = f"{source_text} -> {target}"
    elif source_text:
        text = source_text
    else:
        text = str(row_dict.get("text") or row_dict.get("name") or "").strip()

    edges: List[SemanticEdge] = []
    if relation and target:
        et = normalize_edge_type(relation)
        edges.append(SemanticEdge(
            edge_type=et,
            target=target,
            confidence=0.8,
            provenance=f"{source_db}:{table_source}",
            status="dormant",
        ))

    c = Container(
        kind="relation",
        text=text,
        source=f"{source_db}:{table_source}",
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.8,
        provenance="semantic_layout_machine:relation_import",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "source_entity": source_text,
            "relation": relation,
            "target_entity": target,
            "source_table": table_source,
        },
    )
    return c, edges


def build_procedure_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Container:
    """Build a Container from an extracted_procedures row."""
    name = str(row_dict.get("name") or row_dict.get("procedure") or row_dict.get("title") or row_dict.get("text") or "").strip()
    steps = row_dict.get("steps") or row_dict.get("instructions") or ""
    outcome = str(row_dict.get("outcome") or row_dict.get("result") or "").strip()
    applicability = str(row_dict.get("applicability") or row_dict.get("scope") or row_dict.get("context") or "").strip()
    description = str(row_dict.get("description") or row_dict.get("summary") or "").strip()

    # Parse steps if JSON
    if isinstance(steps, str) and steps:
        try:
            steps_parsed = json.loads(steps)
            if isinstance(steps_parsed, list):
                steps = steps_parsed
        except (json.JSONDecodeError, ValueError):
            pass

    c = Container(
        kind="procedure",
        text=name,
        source=f"{source_db}:{table_source}",
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.8,
        provenance="semantic_layout_machine:procedure_import",
        status=ContainerStatus.DORMANT,
        edges=[],
        symbols=list(symbols),
        metadata={
            "steps": steps,
            "outcome": outcome,
            "applicability": applicability,
            "description": description,
            "source_table": table_source,
        },
    )
    return c


def normalize_edge_type(raw: str) -> str:
    """Normalize a raw relation string to a canonical edge type if possible."""
    raw_lower = raw.lower().strip()
    # Check against known EdgeType values
    for et in EdgeType:
        if raw_lower == et.value.lower():
            return et.value
    # Common synonyms
    synonyms = {
        "is": "is a",
        "isa": "is a",
        "is_an": "is a",
        "part_of": "part of",
        "located_at": "located in",
        "located_in": "located in",
        "has": "contains",
        "contains_component": "contains",
        "uses_tool": "uses",
        "used_by": "uses",
        "configured_by": "configures",
    }
    if raw_lower in synonyms:
        et_val = synonyms[raw_lower]
        # Verify it's a valid EdgeType
        for et in EdgeType:
            if et.value == et_val:
                return et_val
    return raw  # keep as-is (CUSTOM)


# ---------------------------------------------------------------------------
# Semantic layout machine
# ---------------------------------------------------------------------------


@dataclass
class LayoutStats:
    """Accumulated statistics from a layout run."""

    source_dbs: List[str] = field(default_factory=list)
    table_counts: Dict[str, int] = field(default_factory=dict)
    container_count: int = 0
    edge_count: int = 0
    symbol_count: int = 0
    group_count: int = 0
    broad_kind_counts: Dict[str, int] = field(default_factory=dict)
    vector_sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_dbs": list(self.source_dbs),
            "table_counts": dict(self.table_counts),
            "container_count": self.container_count,
            "edge_count": self.edge_count,
            "symbol_count": self.symbol_count,
            "group_count": self.group_count,
            "broad_kind_counts": dict(self.broad_kind_counts),
            "vector_sample_count": self.vector_sample_count,
        }


class SemanticLayoutMachine:
    """The semantic layout machine.

    Reads recovered Axon DBs, classifies items into broad then finer groups,
    assigns substrate-safe symbols, and emits JSONL/JSON artifacts.

    Usage:
        machine = SemanticLayoutMachine(
            semantic_db="D:\\00\\axon_semantic_memory.db",
            out_dir="datasets/recovered/semantic_layout",
        )
        stats = machine.run()
    """

    def __init__(
        self,
        semantic_db: str = DEFAULT_SEMANTIC_DB,
        out_dir: str = DEFAULT_OUT_DIR,
        include_old_db: bool = False,
        old_db: str = DEFAULT_OLD_DB,
        include_episodic: bool = False,
        episodic_db: str = DEFAULT_EPISODIC_DB,
        limit: int = -1,
        max_items: int = -1,
        max_vector_samples: int = 5000,
        levels: int = 2,
        dry_run: bool = False,
        no_vectors: bool = False,
        seed: int = 42,
    ):
        self.semantic_db = semantic_db
        self.out_dir = out_dir
        self.include_old_db = include_old_db
        self.old_db = old_db
        self.include_episodic = include_episodic
        self.episodic_db = episodic_db
        self.limit = limit
        self.max_items = max_items
        self.max_vector_samples = max_vector_samples
        self.levels = max(1, levels)
        self.dry_run = dry_run
        self.no_vectors = no_vectors
        self.seed = seed

        self.registry = SymbolRegistry(source="semantic_layout_machine")
        self.stats = LayoutStats()
        self._all_edges: List[Dict[str, Any]] = []
        self._all_groups: List[LayoutGroup] = []
        self._containers: List[Container] = []

    # -- public API --

    def run(self) -> LayoutStats:
        """Execute the full layout pipeline and write artifacts."""
        dbs_to_process = self._collect_dbs()
        self.stats.source_dbs = [db for db, _ in dbs_to_process]

        item_count = 0
        for db_path, db_role in dbs_to_process:
            item_count = self._process_db(db_path, db_role, item_count)

        # Write artifacts
        if not self.dry_run:
            self._write_artifacts()
        else:
            self.stats.symbol_count = self.registry.count()
            self.stats.group_count = len(self._all_groups)

        return self.stats

    # -- internal: DB discovery --

    def _collect_dbs(self) -> List[Tuple[str, str]]:
        """Collect (db_path, role) pairs to process."""
        dbs: List[Tuple[str, str]] = []
        if Path(self.semantic_db).exists():
            dbs.append((self.semantic_db, "semantic"))
        if self.include_old_db and Path(self.old_db).exists():
            dbs.append((self.old_db, "old"))
        if self.include_episodic and Path(self.episodic_db).exists():
            dbs.append((self.episodic_db, "episodic"))
        return dbs

    # -- internal: DB processing --

    def _process_db(self, db_path: str, db_role: str, item_count: int) -> int:
        """Process a single DB. Returns updated item_count."""
        conn = open_readonly(db_path)
        try:
            tables = get_tables(conn)
            self.stats.table_counts[f"{db_role}:{db_path}"] = len(tables)

            # Map known tables
            entity_tables = [t for t in tables if "entit" in t.lower()]
            fact_tables = [t for t in tables if "fact" in t.lower() or "propert" in t.lower()]
            relation_tables = [t for t in tables if "relation" in t.lower()]
            procedure_tables = [t for t in tables if "procedure" in t.lower() or "workflow" in t.lower()]
            vector_tables = [t for t in tables if "vector" in t.lower() or "embedding" in t.lower()]

            # Process entities
            for table in entity_tables:
                item_count = self._process_entities(conn, db_path, db_role, table, item_count)

            # Process facts
            for table in fact_tables:
                item_count = self._process_facts(conn, db_path, db_role, table, item_count)

            # Process relations
            for table in relation_tables:
                item_count = self._process_relations(conn, db_path, db_role, table, item_count)

            # Process procedures
            for table in procedure_tables:
                item_count = self._process_procedures(conn, db_path, db_role, table, item_count)

            # Process vectors (optional, for grouping only)
            if not self.no_vectors and vector_tables:
                self._process_vectors(conn, db_path, db_role, vector_tables)

            # If no known tables were found, try generic extraction from any table
            if not entity_tables and not fact_tables and not relation_tables and not procedure_tables:
                for table in tables:
                    if table.startswith("sqlite_"):
                        continue
                    item_count = self._process_generic(conn, db_path, db_role, table, item_count)

            return item_count
        finally:
            conn.close()

    def _check_max_items(self, item_count: int) -> bool:
        """Return True if we should stop (max_items reached)."""
        if self.max_items > 0 and item_count >= self.max_items:
            return True
        return False

    def _process_entities(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        db_role: str,
        table: str,
        item_count: int,
    ) -> int:
        """Process an entity table, streaming rows."""
        columns = get_columns(conn, table)
        row_count = count_rows(conn, table)

        # Identify relevant columns
        text_col = find_column(columns, ["name", "entity", "text", "word", "label", "title"])
        type_col = find_column(columns, ["entity_type", "type", "category", "kind"])
        def_col = find_column(columns, ["definition", "description", "summary"])
        edges_col = find_column(columns, ["edges", "relations"])

        if text_col is None:
            # Can't build a container without a text column
            return item_count

        select_cols = [c for c in [text_col, type_col, def_col, edges_col] if c is not None]
        # Also include any id column
        id_col = find_column(columns, ["id", "rowid", "uid"])
        if id_col and id_col not in select_cols:
            select_cols.insert(0, id_col)

        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            text = str(row_dict.get(text_col, "")).strip()
            if not text:
                continue
            entity_type = str(row_dict.get(type_col, "")).strip() if type_col else ""
            definition = str(row_dict.get(def_col, "")).strip() if def_col else ""

            broad_kind = classify_broad_kind(text, entity_type, table, definition)
            layout_symbols = self._assign_group_symbols(broad_kind, text, prefix="lex")

            container = build_entity_container(
                row_dict=row_dict,
                table_source=table,
                source_db=db_path,
                symbols=[],
            )
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            item_count += 1

        return item_count

    def _process_facts(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        db_role: str,
        table: str,
        item_count: int,
    ) -> int:
        """Process a facts table, streaming rows."""
        columns = get_columns(conn, table)
        subj_col = find_column(columns, ["subject", "entity", "name"])
        pred_col = find_column(columns, ["predicate", "relation", "property", "type"])
        obj_col = find_column(columns, ["object", "value", "target"])
        def_col = find_column(columns, ["definition", "description"])

        if subj_col is None and pred_col is None:
            return item_count

        select_cols = [c for c in [subj_col, pred_col, obj_col, def_col] if c is not None]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            subject = str(row_dict.get(subj_col, "")).strip() if subj_col else ""
            if not subject and not str(row_dict.get(pred_col, "")).strip():
                continue

            broad_kind = "fact"
            layout_symbols = self._assign_group_symbols(broad_kind, subject, prefix="fact")

            container, _edges = build_fact_container(
                row_dict=row_dict,
                table_source=table,
                source_db=db_path,
                symbols=[],
            )
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            for e in container.edges:
                self._emit_edge(container.container_id, container.text, e)
            item_count += 1

        return item_count

    def _process_relations(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        db_role: str,
        table: str,
        item_count: int,
    ) -> int:
        """Process a relations table, streaming rows."""
        columns = get_columns(conn, table)
        src_col = find_column(columns, ["source", "subject", "entity", "name"])
        rel_col = find_column(columns, ["relation", "predicate", "type"])
        tgt_col = find_column(columns, ["target", "object", "destination"])

        if src_col is None and rel_col is None:
            return item_count

        select_cols = [c for c in [src_col, rel_col, tgt_col] if c is not None]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            source_text = str(row_dict.get(src_col, "")).strip() if src_col else ""
            relation = str(row_dict.get(rel_col, "")).strip() if rel_col else ""
            if not source_text and not relation:
                continue

            broad_kind = "relation"
            layout_symbols = self._assign_group_symbols(broad_kind, source_text, prefix="rel")

            container, _edges = build_relation_container(
                row_dict=row_dict,
                table_source=table,
                source_db=db_path,
                symbols=[],
            )
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            for e in container.edges:
                self._emit_edge(container.container_id, container.text, e)
            item_count += 1

        return item_count

    def _process_procedures(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        db_role: str,
        table: str,
        item_count: int,
    ) -> int:
        """Process a procedures table, streaming rows."""
        columns = get_columns(conn, table)
        name_col = find_column(columns, ["name", "procedure", "title", "text"])
        steps_col = find_column(columns, ["steps", "instructions"])
        outcome_col = find_column(columns, ["outcome", "result"])
        applic_col = find_column(columns, ["applicability", "scope", "context"])
        desc_col = find_column(columns, ["description", "summary"])

        if name_col is None:
            return item_count

        select_cols = [c for c in [name_col, steps_col, outcome_col, applic_col, desc_col] if c is not None]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            name = str(row_dict.get(name_col, "")).strip()
            if not name:
                continue

            broad_kind = "procedure"
            layout_symbols = self._assign_group_symbols(broad_kind, name, prefix="proc")

            container = build_procedure_container(
                row_dict=row_dict,
                table_source=table,
                source_db=db_path,
                symbols=[],
            )
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            item_count += 1

        return item_count

    def _process_vectors(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        db_role: str,
        vector_tables: List[str],
    ) -> None:
        """Process vector tables for optional semantic grouping.

        This is a sample-based approach: reads up to max_vector_samples rows,
        decodes vectors, and records centroid info in group entries.
        Does NOT load all vectors into memory.
        """
        import random
        rng = random.Random(self.seed)

        for table in vector_tables:
            columns = get_columns(conn, table)
            vec_col = find_column(columns, ["vector", "embedding", "vector_blob", "embed"])
            text_col = find_column(columns, ["name", "entity", "text", "word", "label"])
            if vec_col is None:
                continue

            row_count = count_rows(conn, table)
            if row_count == 0:
                continue

            # Sample-based: determine sample indices
            sample_size = min(self.max_vector_samples, row_count)
            if sample_size < row_count:
                sample_indices = set(rng.sample(range(row_count), sample_size))
            else:
                sample_indices = set(range(row_count))

            select_cols = [c for c in [vec_col, text_col] if c is not None]
            vectors: List[List[float]] = []
            texts: List[str] = []

            row_idx = 0
            for row in stream_rows(conn, table, select_cols, limit=row_count):
                if row_idx in sample_indices:
                    row_dict = row_to_dict(row, select_cols)
                    blob = row_dict.get(vec_col)
                    vec = decode_vector_blob(blob) if blob else None
                    if vec is not None and len(vec) > 0:
                        vectors.append(vec)
                        txt = str(row_dict.get(text_col, "")).strip() if text_col else ""
                        texts.append(txt)
                row_idx += 1
                if len(vectors) >= sample_size:
                    break

            self.stats.vector_sample_count += len(vectors)

            # Simple deterministic grouping: compute centroid of sampled vectors
            if vectors:
                try:
                    import numpy as np
                    arr = np.array(vectors, dtype=np.float32)
                    centroid = arr.mean(axis=0).tolist()
                    # Attach centroid to the broad 'concept' or 'entity' group
                    broad_kind = "concept"
                    broad_entry = self.registry.get_or_create_broad(broad_kind)
                    child_name = f"vec_cluster_{table}"
                    level1_entry = self.registry.get_or_create_child(
                        name=child_name,
                        parent_symbol=broad_entry.symbol,
                        level=1,
                        broad_kind=broad_kind,
                        centroid=centroid,
                    )
                except Exception:
                    pass  # numpy not available or vector decode failed

    def _process_generic(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        db_role: str,
        table: str,
        item_count: int,
    ) -> int:
        """Generic table processing when no known table patterns are found.

        Treats each row as a potential entity, using the first text-like column.
        """
        columns = get_columns(conn, table)
        text_col = find_column(columns, ["name", "text", "word", "title", "label", "entity", "content", "description"])
        if text_col is None:
            return item_count

        type_col = find_column(columns, ["type", "category", "kind", "entity_type"])
        def_col = find_column(columns, ["definition", "description", "summary"])
        select_cols = [c for c in [text_col, type_col, def_col] if c is not None]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            text = str(row_dict.get(text_col, "")).strip()
            if not text:
                continue
            entity_type = str(row_dict.get(type_col, "")).strip() if type_col else ""
            definition = str(row_dict.get(def_col, "")).strip() if def_col else ""

            broad_kind = classify_broad_kind(text, entity_type, table, definition)
            layout_symbols = self._assign_group_symbols(broad_kind, text, prefix="gen")

            container = build_entity_container(
                row_dict=row_dict,
                table_source=table,
                source_db=db_path,
                symbols=[],
            )
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            item_count += 1

        return item_count

    # -- internal: symbol assignment --

    def _assign_group_symbols(self, broad_kind: str, text: str, prefix: str) -> List[str]:
        """Assign broad plus progressively finer symbols for one container.

        Level 0 is the broad kind. Levels 1+ form a deterministic hierarchy
        under that kind using increasingly longer lexical buckets. This gives
        a stable first dormant-state layout before the semantic core learns to
        refine and merge groups.
        """
        broad_entry = self.registry.get_or_create_broad(broad_kind)
        broad_entry.item_count += 1

        symbols = [broad_entry.symbol]
        parent_symbol = broad_entry.symbol
        for level in range(1, self.levels):
            bucket_chars = min(3 * level, 24)
            bucket_key = lexical_bucket_key(text, max_chars=bucket_chars)
            child_name = f"{prefix}_{bucket_key}" if level == 1 else f"{prefix}{level}_{bucket_key}"
            child = self.registry.get_or_create_child(
                name=child_name,
                parent_symbol=parent_symbol,
                level=level,
                broad_kind=broad_kind,
            )
            child.item_count += 1
            symbols.append(child.symbol)
            parent_symbol = child.symbol

        return symbols

    def _assign_edge_symbol(self, edge_type: str) -> str:
        """Assign a registered symbol to an edge type."""
        relation_entry = self.registry.get_or_create_broad("relation")
        edge_key = lexical_bucket_key(edge_type or "edge", max_chars=24)
        edge_entry = self.registry.get_or_create_child(
            name=f"edge_{edge_key}",
            parent_symbol=relation_entry.symbol,
            level=1,
            broad_kind="relation",
        )
        edge_entry.item_count += 1
        return edge_entry.symbol

    # -- internal: emission --

    def _emit_container(self, container: Container) -> None:
        """Record a container for later writing."""
        edge_symbols: List[str] = []
        if container.edges:
            symbolized_edges: List[SemanticEdge] = []
            for edge in container.edges:
                symbol = edge.symbol or self._assign_edge_symbol(edge.edge_type)
                edge_symbols.append(symbol)
                symbolized_edges.append(SemanticEdge(
                    edge_type=edge.edge_type,
                    target=edge.target,
                    symbol=symbol,
                    confidence=edge.confidence,
                    provenance=edge.provenance,
                    status=edge.status,
                ))
            container.edges = symbolized_edges
        container.symbols = sorted(set(edge_symbols))
        self.stats.container_count += 1
        kind = container.kind
        self.stats.broad_kind_counts[kind] = self.stats.broad_kind_counts.get(kind, 0) + 1
        self._containers.append(container)

    def _emit_edge(self, source_container_id: str, source_text: str, edge: SemanticEdge) -> None:
        """Record an edge for later writing."""
        symbol = edge.symbol or self._assign_edge_symbol(edge.edge_type)
        self.stats.edge_count += 1
        self._all_edges.append({
            "edge_type": edge.edge_type,
            "source_container_id": source_container_id,
            "source_text": source_text,
            "target": edge.target,
            "symbol": symbol,
            "confidence": edge.confidence,
            "provenance": edge.provenance,
            "status": edge.status,
        })

    # -- internal: artifact writing --

    def _write_artifacts(self) -> None:
        """Write all artifacts to the output directory."""
        out_path = Path(self.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # containers.jsonl
        containers_path = out_path / "containers.jsonl"
        with open(containers_path, "w", encoding="utf-8") as f:
            for c in self._containers:
                f.write(c.to_json() + "\n")

        # symbol_registry.jsonl
        symbols_path = out_path / "symbol_registry.jsonl"
        with open(symbols_path, "w", encoding="utf-8") as f:
            for entry in self.registry.all_entries():
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self.stats.symbol_count = self.registry.count()

        # semantic_edges.jsonl
        edges_path = out_path / "semantic_edges.jsonl"
        with open(edges_path, "w", encoding="utf-8") as f:
            for e in self._all_edges:
                f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")

        # layout_groups.jsonl
        groups_path = out_path / "layout_groups.jsonl"
        groups = self._build_layout_groups()
        with open(groups_path, "w", encoding="utf-8") as f:
            for g in groups:
                f.write(json.dumps(g.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self.stats.group_count = len(groups)
        self._all_groups = groups

        # corpus_manifest.json
        manifest = self._build_manifest()
        manifest_path = out_path / "corpus_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _build_layout_groups(self) -> List[LayoutGroup]:
        """Build LayoutGroup records from the symbol registry."""
        groups: List[LayoutGroup] = []
        for entry in self.registry.all_entries():
            groups.append(LayoutGroup(
                symbol=entry.symbol,
                level=entry.level,
                name=entry.name,
                parent_symbol=entry.parent_symbol,
                broad_kind=entry.broad_kind,
                item_count=entry.item_count,
                child_symbols=list(entry.child_symbols),
                centroid=entry.centroid,
            ))
        return groups

    def _build_manifest(self) -> Dict[str, Any]:
        """Build the corpus manifest."""
        return {
            "source_dbs": self.stats.source_dbs,
            "table_counts": self.stats.table_counts,
            "output_counts": {
                "containers": self.stats.container_count,
                "semantic_edges": self.stats.edge_count,
                "symbols": self.stats.symbol_count,
                "layout_groups": self.stats.group_count,
                "vector_samples": self.stats.vector_sample_count,
            },
            "broad_kind_counts": self.stats.broad_kind_counts,
            "options": {
                "limit": self.limit,
                "max_items": self.max_items,
                "max_vector_samples": self.max_vector_samples,
                "levels": self.levels,
                "dry_run": self.dry_run,
                "no_vectors": self.no_vectors,
                "seed": self.seed,
                "include_old_db": self.include_old_db,
                "include_episodic": self.include_episodic,
            },
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": "Bootstrap deterministic base registry + dormant state from recovered DBs. "
            "Semantic core trains later as curator/builder from this ground truth. "
            "Symbols are substrate-safe (A-Z, a-z, 0-9). Layout hierarchy is metadata only; "
            "ensemble-visible container symbols are edge-derived. "
            "Broad grouping is deterministic: entity type + table source + text heuristics. "
            "Finer grouping is lexical buckets with optional vector sample centroid assignment.",
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Semantic Layout Machine - bootstrap semantic layout from recovered Axon DBs."
    )
    parser.add_argument("--semantic-db", default=DEFAULT_SEMANTIC_DB, help="Path to axon_semantic_memory.db")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for artifacts")
    parser.add_argument("--include-old-db", action="store_true", help="Also read axon_memory.db")
    parser.add_argument("--old-db", default=DEFAULT_OLD_DB, help="Path to axon_memory.db")
    parser.add_argument("--include-episodic", action="store_true", help="Also read episodic DB")
    parser.add_argument("--episodic-db", default=DEFAULT_EPISODIC_DB, help="Path to episodic DB")
    parser.add_argument("--limit", type=int, default=-1, help="Limit rows per table (-1 = no limit)")
    parser.add_argument("--max-items", type=int, default=-1, help="Max total items to process (-1 = no limit)")
    parser.add_argument("--max-vector-samples", type=int, default=5000, help="Max vector samples for grouping")
    parser.add_argument("--levels", type=int, default=2, help="Grouping depth (1=broad only, 2=lexical)")
    parser.add_argument("--dry-run", action="store_true", help="Process but do not write artifacts")
    parser.add_argument("--no-vectors", action="store_true", help="Skip vector table processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for vector sampling")
    args = parser.parse_args(argv)

    machine = SemanticLayoutMachine(
        semantic_db=args.semantic_db,
        out_dir=args.out_dir,
        include_old_db=args.include_old_db,
        old_db=args.old_db,
        include_episodic=args.include_episodic,
        episodic_db=args.episodic_db,
        limit=args.limit,
        max_items=args.max_items,
        max_vector_samples=args.max_vector_samples,
        levels=args.levels,
        dry_run=args.dry_run,
        no_vectors=args.no_vectors,
        seed=args.seed,
    )
    stats = machine.run()

    print(f"Semantic layout complete.")
    print(f"  Containers:     {stats.container_count}")
    print(f"  Edges:          {stats.edge_count}")
    print(f"  Symbols:        {stats.symbol_count}")
    print(f"  Groups:         {stats.group_count}")
    print(f"  Vector samples: {stats.vector_sample_count}")
    print(f"  Broad kinds:    {json.dumps(stats.broad_kind_counts)}")
    if not args.dry_run:
        print(f"  Output dir:     {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
