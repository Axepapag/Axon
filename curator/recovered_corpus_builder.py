#!/usr/bin/env python3
r"""recovered_corpus_builder.py - build dormant containers from recovered Axon DBs.

This module reads recovered source files in D:\00 in streaming fashion,
converts every record into a container_schema.Container, assigns deterministic
layout metadata for search/curation, and emits the dormant-state artifacts
required by the rest of the Axon pipeline:

    containers.jsonl
    symbol_registry.jsonl
    semantic_edges.jsonl
    layout_groups.jsonl
    corpus_manifest.json

Architecture alignment (SOURCE_OF_TRUTH.md):
  - Source DBs/JSON are opened read-only and never mutated.
  - Raw SQL/JSON rows become structured Container records before any trainer
    or core sees them.
  - Semantic edges are spelled out in English: edge_type + target.
  - No semantic-edge symbols are emitted; Container.symbols stays empty in this
    recovered path.
  - Layout/index symbols are metadata only and never ensemble-visible edge
    meaning.
  - Every container carries a source pointer and provenance.
  - Personal-log entries are excluded by default.

CLI:
    python curator/recovered_corpus_builder.py --smoke
    python curator/recovered_corpus_builder.py \
        --semantic-db D:\00\axon_semantic_memory.db \
        --episodic-db D:\00\axon_episodic_memory.db \
        --old-db D:\00\axon_memory.db \
        --backlog-db D:\00\axon_memory_backlog.db \
        --personal-log D:\00\axon_personal_log.json \
        --out-dir D:\Axon\datasets\recovered\dormant_state_v1 \
        --limit 1000 --max-items 5000 --no-personal-log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# Ensure repo root and the curator directory are importable
_ROOT = str(Path(__file__).resolve().parent.parent)
_CURATOR = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _CURATOR not in sys.path:
    sys.path.insert(0, _CURATOR)

from container_schema import Container, ContainerStatus, SemanticEdge, normalize_container
from semantic_layout_machine import (
    SymbolRegistry,
    classify_broad_kind,
    decode_vector_blob,
    find_column,
    get_columns,
    get_tables,
    lexical_bucket_key,
    open_readonly,
    stream_rows,
    row_to_dict,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SEMANTIC_DB = r"D:\00\axon_semantic_memory.db"
DEFAULT_EPISODIC_DB = r"D:\00\axon_episodic_memory.db"
DEFAULT_OLD_DB = r"D:\00\axon_memory.db"
DEFAULT_BACKLOG_DB = r"D:\00\axon_memory_backlog.db"
DEFAULT_PERSONAL_LOG = r"D:\00\axon_personal_log.json"
DEFAULT_OUT_DIR = os.path.join("datasets", "recovered", "dormant_state_v1")

TRIM_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    """Normalize arbitrary text to a conservative substrate-friendly string.

    Keeps alphanumerics and spaces. Other characters become spaces. Collapses
    whitespace. This is the same conservative policy used by
    dormant_materializer.native_text, but without requiring numpy.
    """
    raw = "" if value is None else str(value)
    out: list[str] = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(" ")
    text = TRIM_RE.sub(" ", "".join(out)).strip()
    return text


def safe_json(value: Any) -> Any:
    """Best-effort parse a JSON string; return input if it is not a string."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def clamp_confidence(value: Any, default: float = 0.8) -> float:
    """Coerce recovered confidence values into the schema's [0, 1] range."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


# ---------------------------------------------------------------------------
# Source pointer and provenance
# ---------------------------------------------------------------------------


def source_pointer(db_path: str, table: str, record_id: Any) -> str:
    """Build a stable source pointer for a SQLite record."""
    return f"{db_path}:{table}:{record_id}"


def json_source_pointer(json_path: str, index: int) -> str:
    """Build a stable source pointer for a JSON list entry."""
    return f"{json_path}:entry:{index}"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class BuilderStats:
    """Accumulated statistics from a builder run."""

    source_files: List[Dict[str, Any]] = field(default_factory=list)
    table_counts: Dict[str, int] = field(default_factory=dict)
    container_count: int = 0
    edge_count: int = 0
    symbol_count: int = 0
    group_count: int = 0
    broad_kind_counts: Dict[str, int] = field(default_factory=dict)
    skipped_counts: Dict[str, int] = field(default_factory=dict)
    vector_sample_count: int = 0

    def add_skip(self, reason: str, n: int = 1) -> None:
        self.skipped_counts[reason] = self.skipped_counts.get(reason, 0) + n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_files": list(self.source_files),
            "table_counts": dict(self.table_counts),
            "container_count": self.container_count,
            "edge_count": self.edge_count,
            "symbol_count": self.symbol_count,
            "group_count": self.group_count,
            "broad_kind_counts": dict(self.broad_kind_counts),
            "skipped_counts": dict(self.skipped_counts),
            "vector_sample_count": self.vector_sample_count,
        }


# ---------------------------------------------------------------------------
# Container builders per source table
# ---------------------------------------------------------------------------


def _edges_from_attributes(attributes: Any) -> List[SemanticEdge]:
    """Parse an attributes dict/JSON for explicit edge declarations."""
    attrs = safe_json(attributes)
    if not isinstance(attrs, dict):
        return []
    edges: List[SemanticEdge] = []
    # Some attributes contain an 'edges' list
    raw_edges = attrs.get("edges")
    if isinstance(raw_edges, list):
        for e in raw_edges:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                edges.append(SemanticEdge(edge_type=str(e[0]), target=str(e[1])))
            elif isinstance(e, dict):
                edges.append(SemanticEdge.from_dict(e))
    # Treat remaining scalar attributes as descriptive facts, not edges, to avoid
    # flooding the edge graph with loose metadata.
    return edges


def build_entity_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Container:
    """Build a Container from an extracted_entities row."""
    text = clean_text(row_dict.get("name") or row_dict.get("entity") or row_dict.get("text") or row_dict.get("word"))
    entity_type = clean_text(row_dict.get("type") or row_dict.get("entity_type"))
    attributes = row_dict.get("attributes")
    definition = ""
    if isinstance(attributes, str):
        try:
            attrs = json.loads(attributes)
            if isinstance(attrs, dict):
                definition = clean_text(attrs.get("description") or attrs.get("summary") or attrs.get("purpose"))
        except (json.JSONDecodeError, ValueError):
            pass

    broad_kind = classify_broad_kind(text, entity_type, table_source, definition)

    edges = _edges_from_attributes(attributes)
    if entity_type and text:
        edges.append(SemanticEdge(
            edge_type="is a",
            target=entity_type,
            confidence=0.8,
            provenance=source_pointer(source_db, table_source, row_dict.get("id")),
            status="dormant",
        ))

    return Container(
        kind=broad_kind,
        text=text,
        source=source_pointer(source_db, table_source, row_dict.get("id")),
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.8,
        provenance="recovered_corpus_builder:entity_import:v1",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "entity_type": entity_type,
            "attributes": attributes if isinstance(attributes, str) else json.dumps(attributes) if attributes is not None else "",
            "definition": definition,
            "source_table": table_source,
            "source_id": row_dict.get("id"),
            "origin_family": "semantic_entity",
        },
    )


def build_fact_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Tuple[Container, List[SemanticEdge]]:
    """Build a Container from an extracted_facts row."""
    key = clean_text(row_dict.get("key"))
    value = clean_text(row_dict.get("value") or row_dict.get("text"))

    if key and value:
        text = f"{key}: {value}"
    elif key:
        text = key
    elif value:
        text = value
    else:
        text = ""

    edges: List[SemanticEdge] = []
    if key and value:
        edges.append(SemanticEdge(
            edge_type=key,
            target=value,
            confidence=0.8,
            provenance=source_pointer(source_db, table_source, row_dict.get("id")),
            status="dormant",
        ))

    container = Container(
        kind="fact",
        text=text,
        source=source_pointer(source_db, table_source, row_dict.get("id")),
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.8,
        provenance="recovered_corpus_builder:fact_import:v1",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "key": key,
            "value": value,
            "source_table": table_source,
            "source_id": row_dict.get("id"),
            "origin_family": "semantic_fact",
        },
    )
    return container, edges


def build_relation_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Tuple[Container, List[SemanticEdge]]:
    """Build a Container from an extracted_relations row."""
    subject = clean_text(row_dict.get("subject"))
    predicate = clean_text(row_dict.get("predicate") or row_dict.get("relation"))
    obj = clean_text(row_dict.get("object") or row_dict.get("target"))

    if subject and predicate and obj:
        text = f"{subject} {predicate} {obj}"
    elif subject and obj:
        text = f"{subject}: {obj}"
    elif subject:
        text = subject
    else:
        text = ""

    edges: List[SemanticEdge] = []
    confidence = clamp_confidence(row_dict.get("confidence", 0.8), default=0.8)

    if predicate and obj:
        edges.append(SemanticEdge(
            edge_type=predicate,
            target=obj,
            confidence=confidence,
            provenance=source_pointer(source_db, table_source, row_dict.get("id")),
            status="dormant",
        ))

    container = Container(
        kind="relation",
        text=text,
        source=source_pointer(source_db, table_source, row_dict.get("id")),
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=confidence,
        provenance="recovered_corpus_builder:relation_import:v1",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "source_table": table_source,
            "source_id": row_dict.get("id"),
            "origin_family": "semantic_relation",
        },
    )
    return container, edges


def build_procedure_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Container:
    """Build a Container from an extracted_procedures row."""
    name = clean_text(row_dict.get("name") or row_dict.get("procedure") or row_dict.get("title"))
    trigger = clean_text(row_dict.get("trigger"))
    steps = safe_json(row_dict.get("steps_json") or row_dict.get("steps") or "[]")
    if not isinstance(steps, list):
        steps = []
    outcome = clean_text(row_dict.get("outcome"))
    applicability = clean_text(row_dict.get("applicability"))
    description = clean_text(row_dict.get("description") or row_dict.get("summary"))

    return Container(
        kind="procedure",
        text=name,
        source=source_pointer(source_db, table_source, row_dict.get("id")),
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.8,
        provenance="recovered_corpus_builder:procedure_import:v1",
        status=ContainerStatus.DORMANT,
        edges=[],
        symbols=list(symbols),
        metadata={
            "trigger": trigger,
            "steps": steps,
            "outcome": outcome,
            "applicability": applicability,
            "description": description,
            "source_table": table_source,
            "source_id": row_dict.get("id"),
            "origin_family": "semantic_procedure",
        },
    )


def build_episode_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Tuple[Container, List[SemanticEdge]]:
    """Build a Container from an episodes row.

    Episodes are treated as potentially-sensitive and kept in their full
    normalized form. Redaction policy controls access; it never destroys text.
    """
    episode_type = clean_text(row_dict.get("episode_type"))
    summary = clean_text(row_dict.get("summary"))
    text = summary or episode_type or f"episode-{row_dict.get('id')}"

    extracted = safe_json(row_dict.get("extracted_json"))
    payload = safe_json(row_dict.get("payload_json"))

    edges: List[SemanticEdge] = []
    # If extracted_json contains relations/facts, turn them into edges.
    if isinstance(extracted, dict):
        for rel in extracted.get("relations") or []:
            if isinstance(rel, dict):
                pred = clean_text(rel.get("predicate") or rel.get("relation"))
                tgt = clean_text(rel.get("object") or rel.get("target"))
                if pred and tgt:
                    edges.append(SemanticEdge(
                        edge_type=pred,
                        target=tgt,
                        confidence=0.7,
                        provenance=source_pointer(source_db, table_source, row_dict.get("id")),
                        status="dormant",
                    ))

    container = Container(
        kind="episode",
        text=text,
        source=source_pointer(source_db, table_source, row_dict.get("id")),
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.7,
        provenance="recovered_corpus_builder:episode_import:v1",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "episode_type": episode_type,
            "summary": summary,
            "extracted_json": extracted,
            "payload_json": payload,
            "source_table": table_source,
            "source_id": row_dict.get("id"),
            "origin_family": "episodic_memory",
            "redaction": "episodic_filter",
        },
    )
    return container, edges


def build_backlog_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Container:
    """Build a Container from a backlog_jobs row."""
    operation = clean_text(row_dict.get("operation"))
    payload = safe_json(row_dict.get("payload_json"))
    payload_summary = ""
    if isinstance(payload, dict):
        payload_summary = clean_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    text = operation or f"backlog-{row_dict.get('id')}"

    edges: List[SemanticEdge] = []
    if operation and payload_summary:
        edges.append(SemanticEdge(
            edge_type=operation,
            target=payload_summary,
            confidence=0.6,
            provenance=source_pointer(source_db, table_source, row_dict.get("id")),
            status="dormant",
        ))

    return Container(
        kind="backlog_job",
        text=text,
        source=source_pointer(source_db, table_source, row_dict.get("id")),
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.6,
        provenance="recovered_corpus_builder:backlog_import:v1",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "operation": operation,
            "payload_json": payload,
            "budget": row_dict.get("budget"),
            "status": row_dict.get("status"),
            "attempts": row_dict.get("attempts"),
            "last_error": clean_text(row_dict.get("last_error")),
            "source_table": table_source,
            "source_id": row_dict.get("id"),
            "origin_family": "memory_backlog",
        },
    )


def build_message_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Container:
    """Build a dormant container from a legacy messages row."""
    role = clean_text(row_dict.get("role"))
    content = clean_text(row_dict.get("content"))
    timestamp = clean_text(row_dict.get("timestamp"))
    metadata = safe_json(row_dict.get("metadata"))

    edges: List[SemanticEdge] = []
    if role:
        edges.append(SemanticEdge(
            edge_type="message role",
            target=role,
            confidence=0.7,
            provenance=source_pointer(source_db, table_source, row_dict.get("id")),
            status="dormant",
        ))

    return Container(
        kind="message",
        text=content,
        source=source_pointer(source_db, table_source, row_dict.get("id")),
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.65,
        provenance="recovered_corpus_builder:old_message_import:v1",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "role": role,
            "timestamp": timestamp,
            "metadata": metadata,
            "source_table": table_source,
            "source_id": row_dict.get("id"),
            "origin_family": "old_memory_message",
            "redaction": "conversation_history",
        },
    )


def build_mission_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Container:
    """Build a dormant container from a legacy missions row."""
    goal = clean_text(row_dict.get("goal"))
    status = clean_text(row_dict.get("status"))

    edges: List[SemanticEdge] = []
    if status:
        edges.append(SemanticEdge(
            edge_type="has status",
            target=status,
            confidence=0.7,
            provenance=source_pointer(source_db, table_source, row_dict.get("id")),
            status="dormant",
        ))

    return Container(
        kind="mission",
        text=goal,
        source=source_pointer(source_db, table_source, row_dict.get("id")),
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.7,
        provenance="recovered_corpus_builder:old_mission_import:v1",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "status": status,
            "created_at": clean_text(row_dict.get("created_at")),
            "completed_at": clean_text(row_dict.get("completed_at")),
            "source_table": table_source,
            "source_id": row_dict.get("id"),
            "origin_family": "old_memory_mission",
        },
    )


def build_objective_container(
    row_dict: Dict[str, Any],
    table_source: str,
    source_db: str,
    symbols: List[str],
    tick: int = -1,
) -> Container:
    """Build a dormant container from a legacy objectives row."""
    description = clean_text(row_dict.get("description"))
    status = clean_text(row_dict.get("status"))
    mission_id = clean_text(row_dict.get("mission_id"))
    reason = clean_text(row_dict.get("reason"))

    edges: List[SemanticEdge] = []
    if mission_id:
        edges.append(SemanticEdge(
            edge_type="belongs to mission",
            target=mission_id,
            confidence=0.75,
            provenance=source_pointer(source_db, table_source, row_dict.get("id")),
            status="dormant",
        ))
    if status:
        edges.append(SemanticEdge(
            edge_type="has status",
            target=status,
            confidence=0.7,
            provenance=source_pointer(source_db, table_source, row_dict.get("id")),
            status="dormant",
        ))

    return Container(
        kind="objective",
        text=description,
        source=source_pointer(source_db, table_source, row_dict.get("id")),
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=0.7,
        provenance="recovered_corpus_builder:old_objective_import:v1",
        status=ContainerStatus.DORMANT,
        edges=edges,
        symbols=list(symbols),
        metadata={
            "mission_id": mission_id,
            "parent_id": clean_text(row_dict.get("parent_id")),
            "status": status,
            "reason": reason,
            "order_index": row_dict.get("order_index"),
            "source_table": table_source,
            "source_id": row_dict.get("id"),
            "origin_family": "old_memory_objective",
        },
    )


def build_diary_container(
    entry: Dict[str, Any],
    index: int,
    json_path: str,
    symbols: List[str],
    tick: int = -1,
) -> Container:
    """Build a Container from a personal-log entry.

    Diary entries are the most sensitive source. They are isolated to the diary
    curriculum family and tagged with redaction metadata.
    """
    entry_type = clean_text(entry.get("type"))
    content = clean_text(entry.get("content"))
    timestamp = clean_text(entry.get("timestamp"))
    text = content or entry_type or f"diary-{index}"

    return Container(
        kind="diary",
        text=text,
        source=json_source_pointer(json_path, index),
        source_tick=tick,
        created_tick=tick,
        updated_tick=tick,
        confidence=1.0,
        provenance="recovered_corpus_builder:diary_import:v1",
        status=ContainerStatus.DORMANT,
        edges=[],
        symbols=list(symbols),
        metadata={
            "entry_type": entry_type,
            "timestamp": timestamp,
            "content": content,
            "redaction": "diary_only",
            "origin_family": "personal_log",
        },
    )


# ---------------------------------------------------------------------------
# Symbol assignment helpers
# ---------------------------------------------------------------------------


def _assign_group_symbols(registry: SymbolRegistry, broad_kind: str, text: str, levels: int) -> List[str]:
    """Assign broad + lexical layout symbols. These are metadata only."""
    broad_entry = registry.get_or_create_broad(broad_kind)
    broad_entry.item_count += 1

    symbols = [broad_entry.symbol]
    parent_symbol = broad_entry.symbol
    for level in range(1, levels):
        bucket_chars = min(3 * level, 24)
        bucket_key = lexical_bucket_key(text, max_chars=bucket_chars)
        child_name = f"lex_{bucket_key}" if level == 1 else f"lex{level}_{bucket_key}"
        child = registry.get_or_create_child(
            name=child_name,
            parent_symbol=parent_symbol,
            level=level,
            broad_kind=broad_kind,
        )
        child.item_count += 1
        symbols.append(child.symbol)
        parent_symbol = child.symbol
    return symbols


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    """Return the SHA-256 digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# RecoveredCorpusBuilder
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


class RecoveredCorpusBuilder:
    """Build dormant containers and symbol registry from recovered Axon sources.

    Usage:
        builder = RecoveredCorpusBuilder(
            semantic_db="D:\\00\\axon_semantic_memory.db",
            episodic_db="D:\\00\\axon_episodic_memory.db",
            old_db="D:\\00\\axon_memory.db",
            backlog_db="D:\\00\\axon_memory_backlog.db",
            personal_log="D:\\00\\axon_personal_log.json",
            out_dir="datasets/recovered/dormant_state_v1",
            limit=1000,
            no_personal_log=True,
        )
        stats = builder.run()
    """

    def __init__(
        self,
        semantic_db: str = DEFAULT_SEMANTIC_DB,
        episodic_db: str = DEFAULT_EPISODIC_DB,
        old_db: str = "",
        backlog_db: str = DEFAULT_BACKLOG_DB,
        personal_log: str = DEFAULT_PERSONAL_LOG,
        out_dir: str = DEFAULT_OUT_DIR,
        limit: int = -1,
        max_items: int = -1,
        levels: int = 2,
        no_personal_log: bool = True,
        no_vectors: bool = False,
        seed: int = 42,
        dry_run: bool = False,
    ):
        self.semantic_db = semantic_db
        self.episodic_db = episodic_db
        self.old_db = old_db
        self.backlog_db = backlog_db
        self.personal_log = personal_log
        self.out_dir = out_dir
        self.limit = limit
        self.max_items = max_items
        self.levels = max(1, levels)
        self.no_personal_log = no_personal_log
        self.no_vectors = no_vectors
        self.seed = seed
        self.dry_run = dry_run

        self.registry = SymbolRegistry(source="recovered_corpus_builder")
        self.stats = BuilderStats()
        self._containers: List[Container] = []
        self._all_edges: List[Dict[str, Any]] = []
        self._all_groups: List[LayoutGroup] = []

    # -- public API --

    def run(self) -> BuilderStats:
        """Execute the full builder pipeline and write artifacts."""
        self._record_source_files()

        item_count = 0
        item_count = self._process_semantic_db(item_count)
        item_count = self._process_old_memory_db(item_count)
        item_count = self._process_episodic_db(item_count)
        item_count = self._process_backlog_db(item_count)
        if not self.no_personal_log:
            item_count = self._process_personal_log(item_count)

        if not self.dry_run:
            self._write_artifacts()
        else:
            self.stats.symbol_count = self.registry.count()
            self.stats.group_count = len(self._all_groups)

        return self.stats

    # -- source file registration --

    def _record_source_files(self) -> None:
        for path in [self.semantic_db, self.old_db, self.episodic_db, self.backlog_db, self.personal_log]:
            if not path:
                continue
            if os.path.exists(path):
                self.stats.source_files.append({
                    "path": os.path.abspath(path),
                    "sha256": sha256_file(path),
                    "size_bytes": os.path.getsize(path),
                    "included": not (self.no_personal_log and path == self.personal_log),
                })
            else:
                self.stats.source_files.append({
                    "path": os.path.abspath(path),
                    "sha256": "",
                    "size_bytes": 0,
                    "included": False,
                    "note": "file not found",
                })

    # -- max-items guard --

    def _check_max_items(self, item_count: int) -> bool:
        if self.max_items > 0 and item_count >= self.max_items:
            return True
        return False

    # -- semantic DB --

    def _process_semantic_db(self, item_count: int) -> int:
        if not os.path.exists(self.semantic_db):
            self.stats.add_skip("semantic_db_missing")
            return item_count
        conn = open_readonly(self.semantic_db)
        try:
            tables = get_tables(conn)
            self.stats.table_counts[f"semantic:{self.semantic_db}"] = len([t for t in tables if t != "sqlite_sequence"])

            for table, processor in [
                ("extracted_entities", self._process_entities),
                ("extracted_facts", self._process_facts),
                ("extracted_relations", self._process_relations),
                ("extracted_procedures", self._process_procedures),
            ]:
                if table in tables:
                    item_count = processor(conn, self.semantic_db, table, item_count)

            if not self.no_vectors and "knowledge_vectors" in tables:
                self._process_knowledge_vectors(conn, self.semantic_db, "knowledge_vectors")

            return item_count
        finally:
            conn.close()

    def _process_entities(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        table: str,
        item_count: int,
    ) -> int:
        columns = get_columns(conn, table)
        select_cols = [c for c in ["id", "name", "type", "attributes", "source", "created_at"] if c in columns]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            text = clean_text(row_dict.get("name"))
            if not text:
                self.stats.add_skip("entity_empty_text")
                continue

            entity_type = clean_text(row_dict.get("type"))
            broad_kind = classify_broad_kind(text, entity_type, table)
            layout_symbols = _assign_group_symbols(self.registry, broad_kind, text, self.levels)

            container = build_entity_container(row_dict, table, db_path, symbols=[])
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            item_count += 1

        return item_count

    def _process_facts(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        table: str,
        item_count: int,
    ) -> int:
        columns = get_columns(conn, table)
        select_cols = [c for c in ["id", "key", "value", "source", "created_at"] if c in columns]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            key = clean_text(row_dict.get("key"))
            if not key:
                self.stats.add_skip("fact_empty_key")
                continue

            broad_kind = "fact"
            layout_symbols = _assign_group_symbols(self.registry, broad_kind, key, self.levels)

            container, _ = build_fact_container(row_dict, table, db_path, symbols=[])
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
        table: str,
        item_count: int,
    ) -> int:
        columns = get_columns(conn, table)
        select_cols = [c for c in ["id", "subject", "predicate", "object", "confidence", "source", "created_at"] if c in columns]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            subject = clean_text(row_dict.get("subject"))
            if not subject:
                self.stats.add_skip("relation_empty_subject")
                continue

            broad_kind = "relation"
            layout_symbols = _assign_group_symbols(self.registry, broad_kind, subject, self.levels)

            container, _ = build_relation_container(row_dict, table, db_path, symbols=[])
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
        table: str,
        item_count: int,
    ) -> int:
        columns = get_columns(conn, table)
        select_cols = [c for c in ["id", "name", "trigger", "steps_json", "outcome", "applicability", "source", "created_at"] if c in columns]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            name = clean_text(row_dict.get("name"))
            if not name:
                self.stats.add_skip("procedure_empty_name")
                continue

            broad_kind = "procedure"
            layout_symbols = _assign_group_symbols(self.registry, broad_kind, name, self.levels)

            container = build_procedure_container(row_dict, table, db_path, symbols=[])
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            item_count += 1

        return item_count

    def _process_knowledge_vectors(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        table: str,
    ) -> None:
        columns = get_columns(conn, table)
        vec_col = find_column(columns, ["vector"])
        if vec_col is None:
            return
        select_cols = [c for c in ["id", "item_type", "item_id", vec_col] if c in columns]
        # Vector processing is optional and sampled; default cap is small.
        sample_limit = 5000 if self.limit < 0 else min(self.limit, 5000)

        vectors: List[List[float]] = []
        for row in stream_rows(conn, table, select_cols, limit=sample_limit):
            blob = row[vec_col] if vec_col in [c for c in columns] else None
            vec = decode_vector_blob(blob) if blob else None
            if vec:
                vectors.append(vec)
            if len(vectors) >= sample_limit:
                break

        self.stats.vector_sample_count += len(vectors)
        if vectors:
            try:
                import numpy as np
                arr = np.array(vectors, dtype=np.float32)
                centroid = arr.mean(axis=0).tolist()
                broad_entry = self.registry.get_or_create_broad("concept")
                child = self.registry.get_or_create_child(
                    name="vec_cluster_knowledge_vectors",
                    parent_symbol=broad_entry.symbol,
                    level=1,
                    broad_kind="concept",
                    centroid=centroid,
                )
                child.item_count += len(vectors)
            except Exception:
                pass

    # -- old mixed memory DB --

    def _process_old_memory_db(self, item_count: int) -> int:
        if not self.old_db:
            return item_count
        if not os.path.exists(self.old_db):
            self.stats.add_skip("old_db_missing")
            return item_count
        conn = open_readonly(self.old_db)
        try:
            tables = get_tables(conn)
            self.stats.table_counts[f"old_memory:{self.old_db}"] = len([t for t in tables if t != "sqlite_sequence"])

            for table, processor in [
                ("extracted_entities", self._process_entities),
                ("extracted_facts", self._process_facts),
                ("extracted_relations", self._process_relations),
            ]:
                if table in tables:
                    item_count = processor(conn, self.old_db, table, item_count)

            if "messages" in tables:
                item_count = self._process_old_messages(conn, self.old_db, "messages", item_count)
            if "missions" in tables:
                item_count = self._process_old_missions(conn, self.old_db, "missions", item_count)
            if "objectives" in tables:
                item_count = self._process_old_objectives(conn, self.old_db, "objectives", item_count)

            if not self.no_vectors and "knowledge_vectors" in tables:
                self._process_knowledge_vectors(conn, self.old_db, "knowledge_vectors")

            return item_count
        finally:
            conn.close()

    def _process_old_messages(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        table: str,
        item_count: int,
    ) -> int:
        columns = get_columns(conn, table)
        select_cols = [c for c in ["id", "role", "content", "timestamp", "metadata"] if c in columns]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            content = clean_text(row_dict.get("content"))
            if not content:
                self.stats.add_skip("old_message_empty_content")
                continue

            broad_kind = "message"
            layout_symbols = _assign_group_symbols(self.registry, broad_kind, content, self.levels)

            container = build_message_container(row_dict, table, db_path, symbols=[])
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            for e in container.edges:
                self._emit_edge(container.container_id, container.text, e)
            item_count += 1

        return item_count

    def _process_old_missions(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        table: str,
        item_count: int,
    ) -> int:
        columns = get_columns(conn, table)
        select_cols = [c for c in ["id", "goal", "status", "created_at", "completed_at"] if c in columns]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            goal = clean_text(row_dict.get("goal"))
            if not goal:
                self.stats.add_skip("old_mission_empty_goal")
                continue

            broad_kind = "mission"
            layout_symbols = _assign_group_symbols(self.registry, broad_kind, goal, self.levels)

            container = build_mission_container(row_dict, table, db_path, symbols=[])
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            for e in container.edges:
                self._emit_edge(container.container_id, container.text, e)
            item_count += 1

        return item_count

    def _process_old_objectives(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        table: str,
        item_count: int,
    ) -> int:
        columns = get_columns(conn, table)
        select_cols = [c for c in ["id", "mission_id", "parent_id", "description", "status", "reason", "order_index"] if c in columns]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            description = clean_text(row_dict.get("description"))
            if not description:
                self.stats.add_skip("old_objective_empty_description")
                continue

            broad_kind = "objective"
            layout_symbols = _assign_group_symbols(self.registry, broad_kind, description, self.levels)

            container = build_objective_container(row_dict, table, db_path, symbols=[])
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            for e in container.edges:
                self._emit_edge(container.container_id, container.text, e)
            item_count += 1

        return item_count

    # -- episodic DB --

    def _process_episodic_db(self, item_count: int) -> int:
        if not os.path.exists(self.episodic_db):
            self.stats.add_skip("episodic_db_missing")
            return item_count
        conn = open_readonly(self.episodic_db)
        try:
            tables = get_tables(conn)
            self.stats.table_counts[f"episodic:{self.episodic_db}"] = len([t for t in tables if t != "sqlite_sequence"])

            if "episodes" in tables:
                item_count = self._process_episodes(conn, self.episodic_db, "episodes", item_count)

            return item_count
        finally:
            conn.close()

    def _process_episodes(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        table: str,
        item_count: int,
    ) -> int:
        columns = get_columns(conn, table)
        select_cols = [c for c in ["id", "episode_type", "turns_json", "summary", "extracted_json", "source", "created_at", "payload_json"] if c in columns]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            summary = clean_text(row_dict.get("summary"))
            if not summary:
                self.stats.add_skip("episode_empty_summary")
                continue

            broad_kind = "episode"
            layout_symbols = _assign_group_symbols(self.registry, broad_kind, summary, self.levels)

            container, _ = build_episode_container(row_dict, table, db_path, symbols=[])
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            for e in container.edges:
                self._emit_edge(container.container_id, container.text, e)
            item_count += 1

        return item_count

    # -- backlog DB --

    def _process_backlog_db(self, item_count: int) -> int:
        if not os.path.exists(self.backlog_db):
            self.stats.add_skip("backlog_db_missing")
            return item_count
        conn = open_readonly(self.backlog_db)
        try:
            tables = get_tables(conn)
            self.stats.table_counts[f"backlog:{self.backlog_db}"] = len([t for t in tables if t != "sqlite_sequence"])

            if "backlog_jobs" in tables:
                item_count = self._process_backlog_jobs(conn, self.backlog_db, "backlog_jobs", item_count)

            return item_count
        finally:
            conn.close()

    def _process_backlog_jobs(
        self,
        conn: sqlite3.Connection,
        db_path: str,
        table: str,
        item_count: int,
    ) -> int:
        columns = get_columns(conn, table)
        select_cols = [c for c in ["id", "operation", "payload_json", "budget", "status", "attempts", "last_error", "created_at", "updated_at"] if c in columns]
        effective_limit = self.limit if self.limit > 0 else -1

        for row in stream_rows(conn, table, select_cols, limit=effective_limit):
            if self._check_max_items(item_count):
                break
            row_dict = row_to_dict(row, select_cols)
            operation = clean_text(row_dict.get("operation"))
            if not operation:
                self.stats.add_skip("backlog_empty_operation")
                continue

            broad_kind = "backlog_job"
            layout_symbols = _assign_group_symbols(self.registry, broad_kind, operation, self.levels)

            container = build_backlog_container(row_dict, table, db_path, symbols=[])
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            for e in container.edges:
                self._emit_edge(container.container_id, container.text, e)
            item_count += 1

        return item_count

    # -- personal log --

    def _process_personal_log(self, item_count: int) -> int:
        if not os.path.exists(self.personal_log):
            self.stats.add_skip("personal_log_missing")
            return item_count

        try:
            with open(self.personal_log, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.stats.add_skip(f"personal_log_parse_error:{e}")
            return item_count

        entries = data.get("entries", []) if isinstance(data, dict) else []
        self.stats.table_counts[f"personal_log:{self.personal_log}"] = len(entries)

        effective_limit = self.limit if self.limit > 0 else len(entries)
        for index, entry in enumerate(entries[:effective_limit]):
            if self._check_max_items(item_count):
                break
            if not isinstance(entry, dict):
                self.stats.add_skip("diary_non_dict_entry")
                continue
            content = clean_text(entry.get("content"))
            if not content:
                self.stats.add_skip("diary_empty_content")
                continue

            broad_kind = "diary"
            layout_symbols = _assign_group_symbols(self.registry, broad_kind, content, self.levels)

            container = build_diary_container(entry, index, self.personal_log, symbols=[])
            container.metadata["layout_symbols"] = layout_symbols
            self._emit_container(container)
            item_count += 1

        return item_count

    # -- emission --

    def _emit_container(self, container: Container) -> None:
        if container.edges:
            unsymbolized: List[SemanticEdge] = []
            for edge in container.edges:
                unsymbolized.append(SemanticEdge(
                    edge_type=edge.edge_type,
                    target=edge.target,
                    symbol=None,
                    confidence=edge.confidence,
                    provenance=edge.provenance,
                    status=edge.status,
                ))
            container.edges = unsymbolized
        container.symbols = []
        self.stats.container_count += 1
        kind = container.kind
        self.stats.broad_kind_counts[kind] = self.stats.broad_kind_counts.get(kind, 0) + 1
        self._containers.append(container)

    def _emit_edge(self, source_container_id: str, source_text: str, edge: SemanticEdge) -> None:
        self.stats.edge_count += 1
        self._all_edges.append({
            "edge_type": edge.edge_type,
            "source_container_id": source_container_id,
            "source_text": source_text,
            "target": edge.target,
            "confidence": edge.confidence,
            "provenance": edge.provenance,
            "status": edge.status,
        })

    # -- artifact writing --

    def _write_artifacts(self) -> None:
        out_path = Path(self.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        containers_path = out_path / "containers.jsonl"
        with open(containers_path, "w", encoding="utf-8", newline="\n") as f:
            for c in self._containers:
                f.write(c.to_json() + "\n")

        symbols_path = out_path / "symbol_registry.jsonl"
        with open(symbols_path, "w", encoding="utf-8", newline="\n") as f:
            for entry in self.registry.all_entries():
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self.stats.symbol_count = self.registry.count()

        edges_path = out_path / "semantic_edges.jsonl"
        with open(edges_path, "w", encoding="utf-8", newline="\n") as f:
            for e in self._all_edges:
                f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")

        groups_path = out_path / "layout_groups.jsonl"
        groups = self._build_layout_groups()
        with open(groups_path, "w", encoding="utf-8", newline="\n") as f:
            for g in groups:
                f.write(json.dumps(g.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self.stats.group_count = len(groups)
        self._all_groups = groups

        manifest = self._build_manifest()
        manifest_path = out_path / "corpus_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _build_layout_groups(self) -> List[LayoutGroup]:
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
        return {
            "kind": "axon_recovered_corpus_manifest",
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_files": self.stats.source_files,
            "table_counts": self.stats.table_counts,
            "output_counts": {
                "containers": self.stats.container_count,
                "semantic_edges": self.stats.edge_count,
                "layout_ids": self.stats.symbol_count,
                "layout_groups": self.stats.group_count,
                "vector_samples": self.stats.vector_sample_count,
            },
            "broad_kind_counts": self.stats.broad_kind_counts,
            "skipped_counts": self.stats.skipped_counts,
            "options": {
                "limit": self.limit,
                "max_items": self.max_items,
                "levels": self.levels,
                "no_personal_log": self.no_personal_log,
                "no_vectors": self.no_vectors,
                "seed": self.seed,
                "dry_run": self.dry_run,
            },
            "notes": "Bootstrap dormant state from recovered DBs. Semantic edges are spelled out. "
            "Layout IDs are metadata only. Personal log is included only when explicitly requested.",
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build dormant container corpus from recovered Axon memory DBs."
    )
    ap.add_argument("--semantic-db", default=DEFAULT_SEMANTIC_DB, help="Path to axon_semantic_memory.db")
    ap.add_argument("--episodic-db", default=DEFAULT_EPISODIC_DB, help="Path to axon_episodic_memory.db")
    ap.add_argument("--old-db", default=DEFAULT_OLD_DB, help="Path to axon_memory.db")
    ap.add_argument("--no-old-db", action="store_true", help="Do not import axon_memory.db")
    ap.add_argument("--backlog-db", default=DEFAULT_BACKLOG_DB, help="Path to axon_memory_backlog.db")
    ap.add_argument("--personal-log", default=DEFAULT_PERSONAL_LOG, help="Path to axon_personal_log.json")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    ap.add_argument("--limit", type=int, default=-1, help="Max rows per table (-1 = all)")
    ap.add_argument("--max-items", type=int, default=-1, help="Max total containers (-1 = all)")
    ap.add_argument("--levels", type=int, default=2, help="Layout grouping depth")
    ap.add_argument("--no-personal-log", action="store_true", help="Exclude personal log (default)")
    ap.add_argument("--include-personal-log", action="store_true", help="Include personal log")
    ap.add_argument("--no-vectors", action="store_true", help="Skip vector table processing")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--dry-run", action="store_true", help="Count only, do not write")
    ap.add_argument("--smoke", action="store_true", help="Smoke mode: limit=100, max-items=500, no vectors, no personal log")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.smoke:
        args.limit = 100 if args.limit < 0 else args.limit
        args.max_items = 500 if args.max_items < 0 else args.max_items
        args.no_vectors = True
        args.no_personal_log = True
        args.no_old_db = True

    no_personal_log = args.no_personal_log or not args.include_personal_log

    builder = RecoveredCorpusBuilder(
        semantic_db=args.semantic_db,
        episodic_db=args.episodic_db,
        old_db="" if args.no_old_db else args.old_db,
        backlog_db=args.backlog_db,
        personal_log=args.personal_log,
        out_dir=args.out_dir,
        limit=args.limit,
        max_items=args.max_items,
        levels=args.levels,
        no_personal_log=no_personal_log,
        no_vectors=args.no_vectors,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    stats = builder.run()

    print("Recovered corpus build complete.")
    print(f"  Containers: {stats.container_count}")
    print(f"  Edges:      {stats.edge_count}")
    print(f"  Layout IDs: {stats.symbol_count}")
    print(f"  Groups:     {stats.group_count}")
    print(f"  Broad kinds: {json.dumps(stats.broad_kind_counts)}")
    if stats.skipped_counts:
        print(f"  Skipped:    {json.dumps(stats.skipped_counts)}")
    if not args.dry_run:
        print(f"  Output dir: {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
