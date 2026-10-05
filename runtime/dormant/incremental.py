"""Transactional incremental maintenance for Axon's derived dormant evidence index.

Exact JSONL under ``State/dormant`` remains the sole memory authority.  This
module updates only the disposable SQLite lookup sense.  The fast path supports
append-only growth plus *layout-preserving* record updates: an existing JSONL
record may change only when its byte offset and byte length remain unchanged.
That lets existing exact offsets remain trustworthy while changed postings,
metadata, relation links, and hashes are repaired transactionally.

Truncation, deletion, insertion into the indexed prefix, variable-length edits,
or container-ID replacement fail closed to the isolated full-generation rebuild
path.  This is deliberate: an incremental organ may be primitive, but it may not
silently guess how authoritative bytes moved.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from runtime.field import canonical_json_bytes, canonical_sha256

from .evidence_bridge import (
    AUTHORITATIVE_FILES,
    INDEX_MANIFEST_SCHEMA,
    INDEX_SCHEMA,
    CorpusFileBinding,
    DormantCorpus,
    DormantCorpusBinding,
    DormantCorpusError,
    DormantEvidenceError,
    DormantEvidenceIndex,
    _container_normalized_text,
    _container_terms,
    _container_text,
    _edge_id,
    _edge_terms,
    _graph_key,
    _hash_file,
    _record_provenance,
    _record_source,
    _sha256_digest,
    _sha256_text_digest,
    _term_key,
)
from .generations import (
    GENERATION_MANIFEST_SCHEMA,
    DormantEvidenceGenerationStore,
    DormantGenerationError,
    DormantIndexGeneration,
)

INCREMENTAL_PLAN_SCHEMA = "axon-dormant-incremental-plan-v1"
INCREMENTAL_RESULT_SCHEMA = "axon-dormant-incremental-result-v1"


class DormantIncrementalError(DormantEvidenceError):
    """Base class for incremental dormant-index maintenance failures."""


class DormantIncrementalFallbackRequired(DormantIncrementalError):
    """The authoritative mutation is valid but not safe for the incremental path."""


class DormantIncrementalRecoveryRequired(DormantIncrementalError):
    """Index bytes committed but logical generation publication did not finish."""


@dataclass(frozen=True, slots=True)
class DormantRecordMutation:
    record_type: str
    operation: str
    rowid: int | None
    byte_offset: int
    byte_length: int
    raw_sha256: str
    identity_before: str | None
    identity_after: str

    def __post_init__(self) -> None:
        if self.record_type not in {"container", "edge"}:
            raise ValueError("record_type must be container or edge")
        if self.operation not in {"update", "append"}:
            raise ValueError("operation must be update or append")
        if self.operation == "update" and (self.rowid is None or self.rowid < 1):
            raise ValueError("update mutation requires a positive rowid")
        if self.operation == "append" and self.rowid is not None:
            raise ValueError("append mutation may not carry a rowid")
        if self.byte_offset < 0 or self.byte_length < 1:
            raise ValueError("mutation byte range is invalid")
        if len(self.raw_sha256) != 64:
            raise ValueError("mutation raw_sha256 must be a SHA256 hex digest")
        if not self.identity_after:
            raise ValueError("mutation identity_after must be non-empty")

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "operation": self.operation,
            "rowid": self.rowid,
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length,
            "raw_sha256": self.raw_sha256,
            "identity_before": self.identity_before,
            "identity_after": self.identity_after,
        }


@dataclass(frozen=True, slots=True)
class DormantIncrementalPlan:
    predecessor_generation: DormantIndexGeneration
    predecessor_index_id: str
    predecessor_binding_id: str
    relative_index_path: str
    new_binding: DormantCorpusBinding
    corpus_stat_guard: tuple[tuple[str, int, int], ...]
    container_mutations: tuple[DormantRecordMutation, ...]
    edge_mutations: tuple[DormantRecordMutation, ...]

    def __post_init__(self) -> None:
        if self.predecessor_generation.index_id != self.predecessor_index_id:
            raise ValueError("predecessor generation/index identity mismatch")
        if self.predecessor_generation.binding_id != self.predecessor_binding_id:
            raise ValueError("predecessor generation/binding identity mismatch")
        if not self.relative_index_path:
            raise ValueError("relative_index_path must be non-empty")

    @property
    def new_index_id(self) -> str:
        return canonical_sha256({"schema": INDEX_SCHEMA, "binding": self.new_binding.to_canonical_dict()})

    @property
    def is_noop(self) -> bool:
        return self.new_binding.binding_id == self.predecessor_binding_id

    @property
    def container_appends(self) -> int:
        return sum(item.operation == "append" for item in self.container_mutations)

    @property
    def container_updates(self) -> int:
        return sum(item.operation == "update" for item in self.container_mutations)

    @property
    def edge_appends(self) -> int:
        return sum(item.operation == "append" for item in self.edge_mutations)

    @property
    def edge_updates(self) -> int:
        return sum(item.operation == "update" for item in self.edge_mutations)

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "schema": INCREMENTAL_PLAN_SCHEMA,
            "predecessor_generation_id": self.predecessor_generation.generation_id,
            "predecessor_index_id": self.predecessor_index_id,
            "predecessor_binding_id": self.predecessor_binding_id,
            "relative_index_path": self.relative_index_path,
            "new_binding_id": self.new_binding.binding_id,
            "new_index_id": self.new_index_id,
            "container_mutations": [item.to_canonical_dict() for item in self.container_mutations],
            "edge_mutations": [item.to_canonical_dict() for item in self.edge_mutations],
        }

    @property
    def plan_id(self) -> str:
        return canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class DormantIncrementalResult:
    descriptor: DormantIndexGeneration
    plan_id: str
    container_appends: int
    container_updates: int
    edge_appends: int
    edge_updates: int
    metadata_only: bool
    full_rebuild_fallback: bool = False

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "schema": INCREMENTAL_RESULT_SCHEMA,
            "generation": self.descriptor.to_canonical_dict(),
            "plan_id": self.plan_id,
            "container_appends": self.container_appends,
            "container_updates": self.container_updates,
            "edge_appends": self.edge_appends,
            "edge_updates": self.edge_updates,
            "metadata_only": self.metadata_only,
            "full_rebuild_fallback": self.full_rebuild_fallback,
        }


class DormantIncrementalMaintainer:
    """Plan, apply, verify, and publish one incremental derived-index generation."""

    def __init__(self, store: DormantEvidenceGenerationStore) -> None:
        if not isinstance(store, DormantEvidenceGenerationStore):
            raise TypeError("store must be DormantEvidenceGenerationStore")
        self.store = store
        self.state_root = store.state_root

    @staticmethod
    def _read_index_meta(index_path: Path) -> tuple[DormantCorpusBinding, str, int, int]:
        connection = sqlite3.connect(index_path)
        try:
            meta = dict(connection.execute("SELECT key, value FROM meta"))
        except sqlite3.DatabaseError as exc:
            raise DormantIncrementalError(f"cannot read incremental predecessor metadata: {index_path}") from exc
        finally:
            connection.close()
        try:
            binding = DormantCorpusBinding.from_mapping(json.loads(meta["binding_json"]))
            index_id = str(meta["index_id"])
            container_count = int(meta["container_count"])
            edge_count = int(meta["edge_count"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DormantIncrementalError("incremental predecessor metadata is malformed") from exc
        expected = canonical_sha256({"schema": INDEX_SCHEMA, "binding": binding.to_canonical_dict()})
        if meta.get("schema") != INDEX_SCHEMA or index_id != expected:
            raise DormantIncrementalError("incremental predecessor identity is invalid")
        return binding, index_id, container_count, edge_count

    @staticmethod
    def _json_record(line: bytes, *, label: str, offset: int) -> Mapping[str, Any]:
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DormantCorpusError(f"invalid {label} JSON at byte offset {offset}") from exc
        if not isinstance(record, Mapping):
            raise DormantCorpusError(f"{label} record at byte offset {offset} is not an object")
        return record

    @staticmethod
    def _binding_file(binding: DormantCorpusBinding, name: str) -> CorpusFileBinding:
        for item in binding.files:
            if item.name == name:
                return item
        raise DormantIncrementalError(f"predecessor binding is missing {name}")

    def _scan_containers(
        self,
        corpus: DormantCorpus,
        connection: sqlite3.Connection,
        old_file: CorpusFileBinding,
    ) -> tuple[str, int, tuple[DormantRecordMutation, ...]]:
        path = corpus.containers_path
        current_size = path.stat().st_size
        if current_size < old_file.size_bytes:
            raise DormantIncrementalFallbackRequired("containers.jsonl was truncated; full generation rebuild required")

        rows = iter(
            connection.execute(
                "SELECT container_rowid, container_id, byte_offset, byte_length, raw_sha256 "
                "FROM containers ORDER BY byte_offset ASC"
            )
        )
        current = next(rows, None)
        digest = hashlib.sha256()
        offset = 0
        mutations: list[DormantRecordMutation] = []
        appended_ids: set[str] = set()
        with path.open("rb") as handle:
            for line in handle:
                line_offset = offset
                offset += len(line)
                digest.update(line)
                if line_offset < old_file.size_bytes:
                    if offset > old_file.size_bytes:
                        raise DormantIncrementalFallbackRequired(
                            "containers.jsonl changed across the predecessor EOF boundary"
                        )
                    if not line.strip():
                        if current is not None and int(current["byte_offset"]) == line_offset:
                            raise DormantIncrementalFallbackRequired(
                                "an indexed container record became blank/deleted; full rebuild required"
                            )
                        continue
                    if current is None:
                        raise DormantIncrementalFallbackRequired(
                            "containers.jsonl inserted a record into the indexed prefix"
                        )
                    row_offset = int(current["byte_offset"])
                    row_length = int(current["byte_length"])
                    if row_offset != line_offset or row_length != len(line):
                        raise DormantIncrementalFallbackRequired(
                            "container record layout shifted; only equal-length in-place updates are incremental"
                        )
                    raw_sha = _sha256_digest(line)
                    if raw_sha != bytes(current["raw_sha256"]):
                        record = self._json_record(line, label="container", offset=line_offset)
                        container_id = str(record.get("container_id", ""))
                        prior_id = str(current["container_id"])
                        if not container_id or container_id != prior_id:
                            raise DormantIncrementalFallbackRequired(
                                "container stable identity changed in place; full rebuild required"
                            )
                        mutations.append(
                            DormantRecordMutation(
                                record_type="container",
                                operation="update",
                                rowid=int(current["container_rowid"]),
                                byte_offset=line_offset,
                                byte_length=len(line),
                                raw_sha256=raw_sha.hex(),
                                identity_before=prior_id,
                                identity_after=container_id,
                            )
                        )
                    current = next(rows, None)
                    continue

                if not line.strip():
                    continue
                record = self._json_record(line, label="container", offset=line_offset)
                container_id = str(record.get("container_id", ""))
                if not container_id:
                    raise DormantCorpusError(f"container missing container_id at byte offset {line_offset}")
                if container_id in appended_ids:
                    raise DormantCorpusError(f"duplicate appended container_id {container_id!r}")
                exists = connection.execute(
                    "SELECT 1 FROM containers WHERE container_id = ?", (container_id,)
                ).fetchone()
                if exists is not None:
                    raise DormantIncrementalFallbackRequired(
                        f"appended container reuses existing container_id {container_id!r}; full rebuild required"
                    )
                appended_ids.add(container_id)
                mutations.append(
                    DormantRecordMutation(
                        record_type="container",
                        operation="append",
                        rowid=None,
                        byte_offset=line_offset,
                        byte_length=len(line),
                        raw_sha256=_sha256_digest(line).hex(),
                        identity_before=None,
                        identity_after=container_id,
                    )
                )
        if current is not None:
            raise DormantIncrementalFallbackRequired(
                "containers.jsonl no longer contains every predecessor record; full rebuild required"
            )
        if offset != current_size:
            raise DormantCorpusError("containers.jsonl changed size while scanning")
        return digest.hexdigest(), offset, tuple(mutations)

    def _scan_edges(
        self,
        corpus: DormantCorpus,
        connection: sqlite3.Connection,
        old_file: CorpusFileBinding,
    ) -> tuple[str, int, tuple[DormantRecordMutation, ...]]:
        path = corpus.semantic_edges_path
        current_size = path.stat().st_size
        if current_size < old_file.size_bytes:
            raise DormantIncrementalFallbackRequired("semantic_edges.jsonl was truncated; full rebuild required")

        rows = iter(
            connection.execute(
                "SELECT edge_rowid, edge_id, byte_offset, byte_length, raw_sha256 "
                "FROM edges ORDER BY byte_offset ASC"
            )
        )
        current = next(rows, None)
        digest = hashlib.sha256()
        offset = 0
        mutations: list[DormantRecordMutation] = []
        with path.open("rb") as handle:
            for line in handle:
                line_offset = offset
                offset += len(line)
                digest.update(line)
                if line_offset < old_file.size_bytes:
                    if offset > old_file.size_bytes:
                        raise DormantIncrementalFallbackRequired(
                            "semantic_edges.jsonl changed across the predecessor EOF boundary"
                        )
                    if not line.strip():
                        if current is not None and int(current["byte_offset"]) == line_offset:
                            raise DormantIncrementalFallbackRequired(
                                "an indexed semantic edge became blank/deleted; full rebuild required"
                            )
                        continue
                    if current is None:
                        raise DormantIncrementalFallbackRequired(
                            "semantic_edges.jsonl inserted a record into the indexed prefix"
                        )
                    row_offset = int(current["byte_offset"])
                    row_length = int(current["byte_length"])
                    if row_offset != line_offset or row_length != len(line):
                        raise DormantIncrementalFallbackRequired(
                            "semantic-edge record layout shifted; only equal-length in-place updates are incremental"
                        )
                    raw_sha = _sha256_digest(line)
                    prior_id = str(current["edge_id"])
                    next_id = _edge_id(line_offset, raw_sha.hex())
                    if raw_sha != bytes(current["raw_sha256"]):
                        self._json_record(line, label="semantic edge", offset=line_offset)
                        mutations.append(
                            DormantRecordMutation(
                                record_type="edge",
                                operation="update",
                                rowid=int(current["edge_rowid"]),
                                byte_offset=line_offset,
                                byte_length=len(line),
                                raw_sha256=raw_sha.hex(),
                                identity_before=prior_id,
                                identity_after=next_id,
                            )
                        )
                    current = next(rows, None)
                    continue

                if not line.strip():
                    continue
                self._json_record(line, label="semantic edge", offset=line_offset)
                raw_sha = _sha256_digest(line)
                mutations.append(
                    DormantRecordMutation(
                        record_type="edge",
                        operation="append",
                        rowid=None,
                        byte_offset=line_offset,
                        byte_length=len(line),
                        raw_sha256=raw_sha.hex(),
                        identity_before=None,
                        identity_after=_edge_id(line_offset, raw_sha.hex()),
                    )
                )
        if current is not None:
            raise DormantIncrementalFallbackRequired(
                "semantic_edges.jsonl no longer contains every predecessor edge; full rebuild required"
            )
        if offset != current_size:
            raise DormantCorpusError("semantic_edges.jsonl changed size while scanning")
        return digest.hexdigest(), offset, tuple(mutations)

    def plan(self) -> DormantIncrementalPlan:
        predecessor = self.store.active_descriptor()
        if not self.store.pointer_path.is_file():
            # A logical pointer is required before authoritative bytes change so
            # crash recovery always has an explicit predecessor identity.
            predecessor = self.store.bootstrap_legacy()
        index_path = self.store._safe_index_path(predecessor.relative_index_path)
        old_binding, old_index_id, _, _ = self._read_index_meta(index_path)
        if old_index_id != predecessor.index_id or old_binding.binding_id != predecessor.binding_id:
            raise DormantIncrementalError("incremental predecessor changed before planning")

        corpus = DormantCorpus.from_state_root(self.state_root)
        guard_before = corpus.stat_guard()
        connection = sqlite3.connect(index_path)
        connection.row_factory = sqlite3.Row
        try:
            containers_hash, containers_size, container_mutations = self._scan_containers(
                corpus,
                connection,
                self._binding_file(old_binding, "containers.jsonl"),
            )
            edges_hash, edges_size, edge_mutations = self._scan_edges(
                corpus,
                connection,
                self._binding_file(old_binding, "semantic_edges.jsonl"),
            )
        finally:
            connection.close()

        file_hashes = {
            "containers.jsonl": containers_hash,
            "semantic_edges.jsonl": edges_hash,
        }
        file_sizes = {
            "containers.jsonl": containers_size,
            "semantic_edges.jsonl": edges_size,
        }
        for name in AUTHORITATIVE_FILES:
            if name in file_hashes:
                continue
            path = corpus.root / name
            file_hashes[name] = _hash_file(path)
            file_sizes[name] = path.stat().st_size

        guard_after = corpus.stat_guard()
        if guard_after != guard_before:
            raise DormantCorpusError("authoritative dormant corpus changed while planning incremental maintenance")
        bindings = tuple(
            CorpusFileBinding(name=name, size_bytes=file_sizes[name], sha256=file_hashes[name])
            for name in AUTHORITATIVE_FILES
        )
        manifest_hash = file_hashes["corpus_manifest.json"]
        new_binding = DormantCorpusBinding(
            corpus_manifest_sha256=manifest_hash,
            files=bindings,
            recovered_sources=corpus.recovered_sources(),
        )
        return DormantIncrementalPlan(
            predecessor_generation=predecessor,
            predecessor_index_id=old_index_id,
            predecessor_binding_id=old_binding.binding_id,
            relative_index_path=predecessor.relative_index_path,
            new_binding=new_binding,
            corpus_stat_guard=guard_after,
            container_mutations=container_mutations,
            edge_mutations=edge_mutations,
        )

    @staticmethod
    def _read_exact_mutation(handle: Any, mutation: DormantRecordMutation) -> bytes:
        handle.seek(mutation.byte_offset)
        raw = handle.read(mutation.byte_length)
        if len(raw) != mutation.byte_length or _sha256_digest(raw).hex() != mutation.raw_sha256:
            raise DormantCorpusError(
                f"{mutation.record_type} authoritative bytes changed after incremental planning"
            )
        return raw

    @staticmethod
    def _container_values(record: Mapping[str, Any], raw: bytes, byte_offset: int) -> tuple[object, ...]:
        container_id = str(record.get("container_id", ""))
        if not container_id:
            raise DormantCorpusError(f"container missing container_id at byte offset {byte_offset}")
        text = _container_text(record)
        normalized = _container_normalized_text(record)
        return (
            container_id,
            byte_offset,
            len(raw),
            _sha256_digest(raw),
            _sha256_text_digest(text),
            _sha256_text_digest(_record_source(record)),
            _sha256_text_digest(_record_provenance(record)),
            _graph_key(normalized),
            str(record.get("kind", record.get("category", ""))),
            str(record.get("status", "")),
            float(record.get("confidence", 1.0)),
        )

    @staticmethod
    def _edge_values(record: Mapping[str, Any], raw: bytes, byte_offset: int) -> tuple[object, ...]:
        raw_digest = _sha256_digest(raw)
        edge_target = str(record.get("target", ""))
        return (
            _edge_id(byte_offset, raw_digest.hex()),
            byte_offset,
            len(raw),
            raw_digest,
            str(record.get("source_container_id", "")),
            _graph_key(edge_target),
            _sha256_text_digest(edge_target),
            _sha256_text_digest(str(record.get("source_text", ""))),
            _sha256_text_digest(_record_provenance(record)),
            str(record.get("status", "")),
            float(record.get("confidence", 1.0)),
        )

    @staticmethod
    def _chunks(values: Sequence[Any], size: int = 700) -> Iterable[Sequence[Any]]:
        for start in range(0, len(values), size):
            yield values[start : start + size]

    def _apply_plan(self, plan: DormantIncrementalPlan) -> tuple[int, int]:
        corpus = DormantCorpus.from_state_root(self.state_root)
        if corpus.stat_guard() != plan.corpus_stat_guard:
            raise DormantCorpusError("authoritative dormant corpus changed after incremental planning")
        index_path = self.store._safe_index_path(plan.relative_index_path)
        connection = sqlite3.connect(index_path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA journal_mode=WAL")
        target_refresh: list[tuple[int, bytes]] = []
        source_refresh: list[tuple[int, str]] = []
        edge_refresh_rowids: set[int] = set()
        old_edge_graph_rows: set[tuple[int, int, int]] = set()
        try:
            connection.execute("BEGIN IMMEDIATE")
            meta = dict(connection.execute("SELECT key, value FROM meta"))
            if meta.get("index_id") != plan.predecessor_index_id:
                raise DormantIncrementalError("incremental predecessor index changed before transaction")
            try:
                observed_binding = DormantCorpusBinding.from_mapping(json.loads(meta["binding_json"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DormantIncrementalError("incremental predecessor binding metadata is malformed") from exc
            if observed_binding.binding_id != plan.predecessor_binding_id:
                raise DormantIncrementalError("incremental predecessor binding changed before transaction")

            with corpus.containers_path.open("rb") as container_handle:
                for mutation in plan.container_mutations:
                    raw = self._read_exact_mutation(container_handle, mutation)
                    record = self._json_record(raw, label="container", offset=mutation.byte_offset)
                    values = self._container_values(record, raw, mutation.byte_offset)
                    container_id = str(values[0])
                    new_key = bytes(values[7])
                    terms = _container_terms(record)
                    if mutation.operation == "update":
                        prior = connection.execute(
                            "SELECT container_id, normalized_key FROM containers WHERE container_rowid = ?",
                            (mutation.rowid,),
                        ).fetchone()
                        if prior is None or str(prior["container_id"]) != mutation.identity_before:
                            raise DormantIncrementalError("container predecessor row changed during transaction")
                        old_key = bytes(prior["normalized_key"])
                        if container_id != mutation.identity_after:
                            raise DormantIncrementalError("container identity changed after planning")
                        connection.execute(
                            """
                            UPDATE containers SET
                                container_id=?, byte_offset=?, byte_length=?, raw_sha256=?,
                                text_sha256=?, source_sha256=?, provenance_sha256=?,
                                normalized_key=?, kind=?, status=?, confidence=?
                            WHERE container_rowid=?
                            """,
                            (*values, mutation.rowid),
                        )
                        connection.execute(
                            "DELETE FROM container_terms WHERE container_rowid = ?", (mutation.rowid,)
                        )
                        if terms:
                            connection.executemany(
                                "INSERT OR IGNORE INTO container_terms(term_hash, container_rowid) VALUES (?, ?)",
                                ((_term_key(term), mutation.rowid) for term in terms),
                            )
                        if old_key != new_key:
                            target_refresh.append((int(mutation.rowid), new_key))
                    else:
                        cursor = connection.execute(
                            """
                            INSERT INTO containers (
                                container_id, byte_offset, byte_length, raw_sha256,
                                text_sha256, source_sha256, provenance_sha256,
                                normalized_key, kind, status, confidence
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            values,
                        )
                        rowid = int(cursor.lastrowid)
                        if terms:
                            connection.executemany(
                                "INSERT OR IGNORE INTO container_terms(term_hash, container_rowid) VALUES (?, ?)",
                                ((_term_key(term), rowid) for term in terms),
                            )
                        target_refresh.append((rowid, new_key))
                        source_refresh.append((rowid, container_id))

            with corpus.semantic_edges_path.open("rb") as edge_handle:
                for mutation in plan.edge_mutations:
                    raw = self._read_exact_mutation(edge_handle, mutation)
                    record = self._json_record(raw, label="semantic edge", offset=mutation.byte_offset)
                    values = self._edge_values(record, raw, mutation.byte_offset)
                    next_edge_id = str(values[0])
                    terms = _edge_terms(record)
                    if mutation.operation == "update":
                        prior = connection.execute(
                            "SELECT edge_id, source_container_id FROM edges WHERE edge_rowid = ?",
                            (mutation.rowid,),
                        ).fetchone()
                        if prior is None or str(prior["edge_id"]) != mutation.identity_before:
                            raise DormantIncrementalError("semantic-edge predecessor row changed during transaction")
                        old_source = connection.execute(
                            "SELECT container_rowid FROM containers WHERE container_id = ?",
                            (str(prior["source_container_id"]),),
                        ).fetchone()
                        if old_source is not None:
                            old_source_rowid = int(old_source["container_rowid"])
                            rows = connection.execute(
                                "SELECT target_container_rowid FROM graph_neighbors "
                                "WHERE source_container_rowid = ? AND edge_rowid = ?",
                                (old_source_rowid, mutation.rowid),
                            )
                            old_edge_graph_rows.update(
                                (old_source_rowid, int(row["target_container_rowid"]), int(mutation.rowid))
                                for row in rows
                            )
                        if next_edge_id != mutation.identity_after:
                            raise DormantIncrementalError("semantic-edge identity changed after planning")
                        connection.execute(
                            """
                            UPDATE edges SET
                                edge_id=?, byte_offset=?, byte_length=?, raw_sha256=?,
                                source_container_id=?, target_key=?, target_text_sha256=?,
                                source_text_sha256=?, provenance_sha256=?, status=?, confidence=?
                            WHERE edge_rowid=?
                            """,
                            (*values, mutation.rowid),
                        )
                        connection.execute("DELETE FROM edge_terms WHERE edge_rowid = ?", (mutation.rowid,))
                        if terms:
                            connection.executemany(
                                "INSERT OR IGNORE INTO edge_terms(term_hash, edge_rowid) VALUES (?, ?)",
                                ((_term_key(term), mutation.rowid) for term in terms),
                            )
                        edge_refresh_rowids.add(int(mutation.rowid))
                    else:
                        cursor = connection.execute(
                            """
                            INSERT INTO edges (
                                edge_id, byte_offset, byte_length, raw_sha256,
                                source_container_id, target_key, target_text_sha256,
                                source_text_sha256, provenance_sha256, status, confidence
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            values,
                        )
                        edge_rowid = int(cursor.lastrowid)
                        if terms:
                            connection.executemany(
                                "INSERT OR IGNORE INTO edge_terms(term_hash, edge_rowid) VALUES (?, ?)",
                                ((_term_key(term), edge_rowid) for term in terms),
                            )
                        edge_refresh_rowids.add(edge_rowid)

            # Remove exact graph rows belonging to updated edges using the
            # table's existing (source, target, edge) primary key. This avoids a
            # full graph scan by edge_rowid on legacy indexes that do not carry
            # an edge-first graph index.
            if old_edge_graph_rows:
                connection.executemany(
                    "DELETE FROM graph_neighbors WHERE source_container_rowid = ? "
                    "AND target_container_rowid = ? AND edge_rowid = ?",
                    sorted(old_edge_graph_rows),
                )

            # If a container's normalized key changed (or a new target container
            # appeared), only links *to that target row* need rebuilding.
            # idx_graph_target makes deletion bounded, while idx_edges_target_key
            # finds the exact set of edges that now resolve to the new key.
            for target_rowid, target_key in target_refresh:
                connection.execute(
                    "DELETE FROM graph_neighbors WHERE target_container_rowid = ?",
                    (target_rowid,),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO graph_neighbors(
                        source_container_rowid, target_container_rowid, edge_rowid
                    )
                    SELECT source.container_rowid, ?, e.edge_rowid
                    FROM edges e
                    JOIN containers source ON source.container_id = e.source_container_id
                    WHERE e.target_key = ?
                    """,
                    (target_rowid, target_key),
                )

            # A newly appended source container can make previously unresolved
            # source->target edges resolvable without touching any old target.
            for source_rowid, source_container_id in source_refresh:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO graph_neighbors(
                        source_container_rowid, target_container_rowid, edge_rowid
                    )
                    SELECT ?, target.container_rowid, e.edge_rowid
                    FROM edges e
                    JOIN containers target ON target.normalized_key = e.target_key
                    WHERE e.source_container_id = ?
                    """,
                    (source_rowid, source_container_id),
                )

            # Changed/appended edges are few by construction on the incremental
            # path. Their new relation links can be rebuilt directly from the
            # edge primary key and existing container indexes.
            for edge_rowid in sorted(edge_refresh_rowids):
                connection.execute(
                    """
                    INSERT OR IGNORE INTO graph_neighbors(
                        source_container_rowid, target_container_rowid, edge_rowid
                    )
                    SELECT source.container_rowid, target.container_rowid, e.edge_rowid
                    FROM edges e
                    JOIN containers source ON source.container_id = e.source_container_id
                    JOIN containers target ON target.normalized_key = e.target_key
                    WHERE e.edge_rowid = ?
                    """,
                    (edge_rowid,),
                )

            if corpus.stat_guard() != plan.corpus_stat_guard:
                raise DormantCorpusError("authoritative dormant corpus changed during incremental transaction")
            container_count = int(connection.execute("SELECT COUNT(*) FROM containers").fetchone()[0])
            edge_count = int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
            binding_json = canonical_json_bytes(plan.new_binding.to_canonical_dict()).decode("utf-8")
            values = (
                ("schema", INDEX_SCHEMA),
                ("index_id", plan.new_index_id),
                ("binding_json", binding_json),
                ("container_count", str(container_count)),
                ("edge_count", str(edge_count)),
                ("maintenance_mode", "incremental_update"),
                ("predecessor_index_id", plan.predecessor_index_id),
                ("incremental_plan_id", plan.plan_id),
            )
            connection.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                values,
            )
            connection.commit()
        except Exception:
            try:
                connection.rollback()
            except sqlite3.DatabaseError:
                pass
            raise
        finally:
            connection.close()
        return container_count, edge_count

    def _generation_descriptor(self, plan: DormantIncrementalPlan) -> DormantIndexGeneration:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return DormantIndexGeneration(
            generation_id=f"inc-{stamp}-{plan.new_index_id[:16]}",
            index_id=plan.new_index_id,
            binding_id=plan.new_binding.binding_id,
            relative_index_path=plan.relative_index_path,
            promoted_at=datetime.now(timezone.utc).isoformat(),
            mode="incremental_update",
            predecessor_generation_id=plan.predecessor_generation.generation_id,
        )

    def _write_prepared_manifest(
        self,
        descriptor: DormantIndexGeneration,
        *,
        plan: DormantIncrementalPlan,
    ) -> Path:
        """Fsync recovery evidence before any derived-index bytes mutate."""

        generation_dir = self.store.generations_root / descriptor.generation_id
        generation_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": GENERATION_MANIFEST_SCHEMA,
            "descriptor_id": descriptor.descriptor_id,
            "generation": descriptor.to_canonical_dict(),
            "build_mode": "incremental_in_place",
            "publication_state": "prepared",
            "incremental_append_update": True,
            "layout_preserving_updates_only": True,
            "old_index_id": plan.predecessor_index_id,
            "new_index_id": plan.new_index_id,
            "new_binding": plan.new_binding.to_canonical_dict(),
            "plan_id": plan.plan_id,
            "container_appends": plan.container_appends,
            "container_updates": plan.container_updates,
            "edge_appends": plan.edge_appends,
            "edge_updates": plan.edge_updates,
            "recovered_after_pointer_gap": False,
            "authority_note": (
                "Prepared crash-recovery evidence for a derived-index transaction only. "
                "State/dormant exact JSONL remains the sole memory authority."
            ),
        }
        self.store._atomic_write(generation_dir / "generation.json", payload)
        return generation_dir / "generation.json"

    def _write_generation_manifest(
        self,
        descriptor: DormantIndexGeneration,
        *,
        plan: DormantIncrementalPlan,
        container_count: int,
        edge_count: int,
        recovered_after_pointer_gap: bool = False,
    ) -> Path:
        generation_dir = self.store.generations_root / descriptor.generation_id
        generation_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": GENERATION_MANIFEST_SCHEMA,
            "descriptor_id": descriptor.descriptor_id,
            "generation": descriptor.to_canonical_dict(),
            "build_mode": "incremental_in_place",
            "publication_state": "verified",
            "incremental_append_update": True,
            "layout_preserving_updates_only": True,
            "old_index_id": plan.predecessor_index_id,
            "new_index_id": plan.new_index_id,
            "new_binding": plan.new_binding.to_canonical_dict(),
            "plan_id": plan.plan_id,
            "container_count": container_count,
            "edge_count": edge_count,
            "container_appends": plan.container_appends,
            "container_updates": plan.container_updates,
            "edge_appends": plan.edge_appends,
            "edge_updates": plan.edge_updates,
            "recovered_after_pointer_gap": recovered_after_pointer_gap,
            "authority_note": (
                "Logical derived generation only; no authoritative memory bytes are stored here. "
                "State/dormant exact JSONL remains the sole memory authority."
            ),
        }
        self.store._atomic_write(generation_dir / "generation.json", payload)
        return generation_dir / "generation.json"

    def _write_index_sidecar(
        self,
        descriptor: DormantIndexGeneration,
        *,
        plan: DormantIncrementalPlan,
        container_count: int,
        edge_count: int,
    ) -> None:
        index_path = self.store._safe_index_path(descriptor.relative_index_path)
        sidecar = index_path.with_name(index_path.name + ".manifest.json")
        payload = {
            "schema": INDEX_MANIFEST_SCHEMA,
            "index_schema": INDEX_SCHEMA,
            "index_id": descriptor.index_id,
            "built_at": descriptor.promoted_at,
            "index_file": index_path.name,
            "container_count": container_count,
            "edge_count": edge_count,
            "binding": plan.new_binding.to_canonical_dict(),
            "maintenance_mode": "incremental_update",
            "predecessor_index_id": plan.predecessor_index_id,
            "incremental_plan_id": plan.plan_id,
            "authority_note": (
                "Derived lookup metadata only. Exact dormant text remains authoritative only in "
                "State/dormant JSONL and is dereferenced/verified at use time."
            ),
        }
        self.store._atomic_write(sidecar, payload)

    def apply(self, plan: DormantIncrementalPlan) -> DormantIncrementalResult:
        if not isinstance(plan, DormantIncrementalPlan):
            raise TypeError("plan must be DormantIncrementalPlan")
        if plan.is_noop:
            return DormantIncrementalResult(
                descriptor=plan.predecessor_generation,
                plan_id=plan.plan_id,
                container_appends=0,
                container_updates=0,
                edge_appends=0,
                edge_updates=0,
                metadata_only=False,
            )

        descriptor = self._generation_descriptor(plan)
        # Crash recovery evidence must exist before the SQLite transaction can
        # make the predecessor pointer stale.
        self._write_prepared_manifest(descriptor, plan=plan)
        container_count, edge_count = self._apply_plan(plan)
        index_path = self.store._safe_index_path(plan.relative_index_path)
        try:
            with DormantEvidenceIndex.open(
                self.state_root, index_path=index_path, verify_binding=True
            ) as verified:
                if verified.index_id != plan.new_index_id or verified.binding != plan.new_binding:
                    raise DormantIncrementalRecoveryRequired(
                        "incrementally maintained index failed post-transaction identity verification"
                    )
        except DormantIncrementalRecoveryRequired:
            raise
        except Exception as exc:
            raise DormantIncrementalRecoveryRequired(
                "incremental index transaction committed but authoritative verification failed"
            ) from exc

        try:
            self._write_generation_manifest(
                descriptor,
                plan=plan,
                container_count=container_count,
                edge_count=edge_count,
            )
            self._write_index_sidecar(
                descriptor,
                plan=plan,
                container_count=container_count,
                edge_count=edge_count,
            )
            self.store._write_pointer(descriptor)
        except Exception as exc:
            raise DormantIncrementalRecoveryRequired(
                "incremental index committed but generation pointer publication is incomplete"
            ) from exc

        return DormantIncrementalResult(
            descriptor=descriptor,
            plan_id=plan.plan_id,
            container_appends=plan.container_appends,
            container_updates=plan.container_updates,
            edge_appends=plan.edge_appends,
            edge_updates=plan.edge_updates,
            metadata_only=not plan.container_mutations and not plan.edge_mutations,
        )

    def maintain(self, *, fallback_full: bool = False) -> DormantIncrementalResult:
        try:
            plan = self.plan()
            return self.apply(plan)
        except DormantIncrementalFallbackRequired as exc:
            if not fallback_full:
                raise
            descriptor = self.store.build_generation(promote=True)
            return DormantIncrementalResult(
                descriptor=descriptor,
                plan_id=canonical_sha256(
                    {
                        "schema": INCREMENTAL_PLAN_SCHEMA,
                        "fallback": "full_generation",
                        "reason": str(exc),
                        "index_id": descriptor.index_id,
                    }
                ),
                container_appends=0,
                container_updates=0,
                edge_appends=0,
                edge_updates=0,
                metadata_only=False,
                full_rebuild_fallback=True,
            )

    def recover_pointer(self) -> DormantIndexGeneration:
        """Finish publication after a crash between SQLite commit and pointer swap.

        Recovery never fabricates a generation from SQLite metadata alone.  It
        requires a previously fsynced incremental generation manifest whose
        descriptor matches the now-current index bytes and exact corpus binding.
        """

        predecessor = self.store.pointer_descriptor_unverified()
        if predecessor is None:
            raise DormantIncrementalRecoveryRequired("no predecessor generation pointer exists")
        index_path = self.store._safe_index_path(predecessor.relative_index_path)
        current_index_id, current_binding_id = self.store._read_index_identity(index_path)
        if (
            current_index_id == predecessor.index_id
            and current_binding_id == predecessor.binding_id
        ):
            return self.store.active_descriptor()

        candidates: list[tuple[DormantIndexGeneration, Path, dict[str, Any]]] = []
        if self.store.generations_root.is_dir():
            for manifest_path in self.store.generations_root.glob("inc-*/generation.json"):
                try:
                    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if raw.get("schema") != GENERATION_MANIFEST_SCHEMA:
                        continue
                    descriptor = DormantIndexGeneration.from_mapping(raw["generation"])
                    if descriptor.descriptor_id != str(raw["descriptor_id"]):
                        continue
                    if descriptor.predecessor_generation_id != predecessor.generation_id:
                        continue
                    if descriptor.relative_index_path != predecessor.relative_index_path:
                        continue
                    if descriptor.index_id != current_index_id or descriptor.binding_id != current_binding_id:
                        continue
                    candidates.append((descriptor, manifest_path, dict(raw)))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
        if len(candidates) != 1:
            raise DormantIncrementalRecoveryRequired(
                "cannot uniquely recover incremental pointer from durable generation evidence"
            )
        descriptor, manifest_path, manifest_payload = candidates[0]
        with DormantEvidenceIndex.open(
            self.state_root, index_path=index_path, verify_binding=True
        ) as verified:
            if verified.index_id != descriptor.index_id or verified.binding.binding_id != descriptor.binding_id:
                raise DormantIncrementalRecoveryRequired(
                    "recovery generation does not verify against authoritative corpus"
                )
            binding = verified.binding

        connection = sqlite3.connect(index_path)
        try:
            meta = dict(connection.execute("SELECT key, value FROM meta"))
            container_count = int(meta["container_count"])
            edge_count = int(meta["edge_count"])
            plan_id = str(meta.get("incremental_plan_id", manifest_payload.get("plan_id", "")))
            predecessor_index_id = str(
                meta.get("predecessor_index_id", manifest_payload.get("old_index_id", predecessor.index_id))
            )
        except (sqlite3.DatabaseError, KeyError, TypeError, ValueError) as exc:
            raise DormantIncrementalRecoveryRequired(
                "recovery index metadata is incomplete"
            ) from exc
        finally:
            connection.close()

        # Finish the sidecar and durable generation evidence before making the
        # logical pointer visible again.
        sidecar = index_path.with_name(index_path.name + ".manifest.json")
        self.store._atomic_write(
            sidecar,
            {
                "schema": INDEX_MANIFEST_SCHEMA,
                "index_schema": INDEX_SCHEMA,
                "index_id": descriptor.index_id,
                "built_at": descriptor.promoted_at,
                "index_file": index_path.name,
                "container_count": container_count,
                "edge_count": edge_count,
                "binding": binding.to_canonical_dict(),
                "maintenance_mode": "incremental_update_recovered",
                "predecessor_index_id": predecessor_index_id,
                "incremental_plan_id": plan_id,
                "authority_note": (
                    "Recovered derived lookup metadata only. Exact dormant JSONL remains memory authority."
                ),
            },
        )
        manifest_payload["publication_state"] = "recovered"
        manifest_payload["recovered_after_pointer_gap"] = True
        manifest_payload["container_count"] = container_count
        manifest_payload["edge_count"] = edge_count
        self.store._atomic_write(manifest_path, manifest_payload)
        self.store._write_pointer(descriptor)
        return descriptor


__all__ = [
    "INCREMENTAL_PLAN_SCHEMA",
    "INCREMENTAL_RESULT_SCHEMA",
    "DormantIncrementalError",
    "DormantIncrementalFallbackRequired",
    "DormantIncrementalRecoveryRequired",
    "DormantRecordMutation",
    "DormantIncrementalPlan",
    "DormantIncrementalResult",
    "DormantIncrementalMaintainer",
]
