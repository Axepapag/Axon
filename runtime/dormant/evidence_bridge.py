"""Read-only dormant evidence retrieval and canonical surfacing.

The JSONL corpus beneath ``State/dormant`` remains the sole dormant-memory
authority.  This module builds a disposable SQLite *sense* containing only
lookup metadata (IDs, byte ranges, hashes, lexical postings, filters and graph
neighbor IDs).  Exact container/edge text is never copied into the index.
Every selected record is dereferenced from the authoritative JSONL and verified
again before it may become a canonical ``FieldSpan``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from runtime.field import (
    D64FieldCompiler,
    FieldSpan,
    LogicalRegion,
    RegionState,
    RegionVisibility,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
)
from runtime.source_of_truth import capacity_policy

INDEX_SCHEMA = "axon-dormant-evidence-index-v1"
INDEX_MANIFEST_SCHEMA = "axon-dormant-evidence-index-manifest-v1"
DEFAULT_INDEX_RELATIVE = Path("dormant") / ".derived" / "evidence_v1" / "index.sqlite3"
AUTHORITATIVE_FILES = (
    "containers.jsonl",
    "semantic_edges.jsonl",
    "kg_cache_50k.jsonl",
    "layout_groups.jsonl",
    "symbol_registry.jsonl",
    "corpus_manifest.json",
)
_TOKEN_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*", flags=re.UNICODE)


class DormantEvidenceError(RuntimeError):
    """Base class for dormant evidence bridge failures."""


class DormantCorpusError(DormantEvidenceError):
    """The authoritative dormant corpus is absent or malformed."""


class StaleDormantIndexError(DormantEvidenceError):
    """A derived index is not bound to the current authoritative corpus."""


class DormantRecordIntegrityError(DormantEvidenceError):
    """An indexed record failed exact dereference verification."""


class DormantQueryError(DormantEvidenceError):
    """A retrieval request is invalid."""


def _sha256_digest(value: bytes) -> bytes:
    return hashlib.sha256(value).digest()


def _sha256_bytes(value: bytes) -> str:
    return _sha256_digest(value).hex()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_text_digest(value: str) -> bytes:
    return _sha256_digest(value.encode("utf-8"))


def _normalized_graph_text(value: str) -> str:
    return " ".join(str(value).casefold().split())


def _graph_key(value: str) -> bytes:
    return _sha256_text_digest(_normalized_graph_text(value))


def _term_key(value: str) -> bytes:
    return _sha256_text_digest(_normalized_graph_text(value))


def _edge_id(byte_offset: int, raw_sha256: str) -> str:
    return f"edge:{byte_offset}:{raw_sha256}"


def _tokenize(value: str) -> tuple[str, ...]:
    terms = {_normalized_graph_text(match.group(0)) for match in _TOKEN_RE.finditer(value)}
    terms.discard("")
    return tuple(sorted(terms))


def _hash_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _record_source(record: Mapping[str, Any]) -> str:
    return str(record.get("source", ""))


def _record_provenance(record: Mapping[str, Any]) -> str:
    return str(record.get("provenance", ""))


def _container_text(record: Mapping[str, Any]) -> str:
    return str(record.get("text", record.get("word", "")))


def _container_normalized_text(record: Mapping[str, Any]) -> str:
    raw = record.get("normalized_text")
    if raw is None or str(raw) == "":
        raw = _container_text(record)
    return _normalized_graph_text(str(raw))


def _container_terms(record: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("text", "normalized_text", "letters", "word"):
        value = record.get(key)
        if isinstance(value, str) and value:
            values.append(value)

    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        for key in (
            "definition",
            "summary",
            "description",
            "title",
            "name",
            "key",
            "value",
            "entity_type",
            "category",
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                values.append(value)

    edges = record.get("edges")
    if isinstance(edges, Sequence) and not isinstance(edges, (str, bytes, bytearray)):
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            for key in ("edge_type", "target"):
                value = edge.get(key)
                if isinstance(value, str) and value:
                    values.append(value)

    terms: set[str] = set()
    normalized_values = {
        _normalized_graph_text(value) for value in values if isinstance(value, str) and value
    }
    for value in normalized_values:
        terms.update(_tokenize(value))
    return tuple(sorted(terms))


def _edge_terms(record: Mapping[str, Any]) -> tuple[str, ...]:
    values = [
        str(record.get("edge_type", "")),
        str(record.get("target", "")),
        str(record.get("source_text", "")),
    ]
    terms: set[str] = set()
    for value in {_normalized_graph_text(item) for item in values if item}:
        terms.update(_tokenize(value))
    return tuple(sorted(terms))


@dataclass(frozen=True, slots=True)
class CorpusFileBinding:
    name: str
    size_bytes: int
    sha256: str

    def to_canonical_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class DormantCorpusBinding:
    corpus_manifest_sha256: str
    files: tuple[CorpusFileBinding, ...]
    recovered_sources: tuple[tuple[str, int, str, bool], ...]

    @property
    def binding_id(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "files": [item.to_canonical_dict() for item in self.files],
            "recovered_sources": [list(item) for item in self.recovered_sources],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DormantCorpusBinding":
        files = tuple(
            CorpusFileBinding(
                name=str(item["name"]),
                size_bytes=int(item["size_bytes"]),
                sha256=str(item["sha256"]),
            )
            for item in value.get("files", [])
        )
        recovered = tuple(
            (str(item[0]), int(item[1]), str(item[2]), bool(item[3]))
            for item in value.get("recovered_sources", [])
        )
        return cls(
            corpus_manifest_sha256=str(value["corpus_manifest_sha256"]),
            files=files,
            recovered_sources=recovered,
        )


class DormantCorpus:
    """Read-only locator and hash authority for ``State/dormant``."""

    def __init__(self, dormant_root: str | os.PathLike[str]) -> None:
        self.root = Path(dormant_root).resolve()
        if not self.root.is_dir():
            raise DormantCorpusError(f"dormant root does not exist: {self.root}")
        for name in AUTHORITATIVE_FILES:
            path = self.root / name
            if not path.is_file():
                raise DormantCorpusError(f"missing authoritative dormant file: {path}")
        try:
            self.manifest = json.loads((self.root / "corpus_manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DormantCorpusError("corpus_manifest.json is unreadable or invalid") from exc
        if self.manifest.get("kind") != "axon_recovered_corpus_manifest":
            raise DormantCorpusError("unexpected dormant corpus manifest kind")

    @property
    def containers_path(self) -> Path:
        return self.root / "containers.jsonl"

    @property
    def semantic_edges_path(self) -> Path:
        return self.root / "semantic_edges.jsonl"

    @classmethod
    def from_state_root(cls, state_root: str | os.PathLike[str]) -> "DormantCorpus":
        return cls(Path(state_root) / "dormant")

    def recovered_sources(self) -> tuple[tuple[str, int, str, bool], ...]:
        items: list[tuple[str, int, str, bool]] = []
        for source in self.manifest.get("source_files", []):
            if not isinstance(source, Mapping):
                continue
            items.append(
                (
                    str(source.get("path", "")),
                    int(source.get("size_bytes", -1)),
                    str(source.get("sha256", "")),
                    bool(source.get("included", False)),
                )
            )
        return tuple(sorted(items))

    def stat_guard(self) -> tuple[tuple[str, int, int], ...]:
        items: list[tuple[str, int, int]] = []
        for name in AUTHORITATIVE_FILES:
            stat = (self.root / name).stat()
            items.append((name, stat.st_size, stat.st_mtime_ns))
        return tuple(items)

    def compute_binding(self) -> DormantCorpusBinding:
        before = self.stat_guard()
        bindings = tuple(
            CorpusFileBinding(
                name=name,
                size_bytes=(self.root / name).stat().st_size,
                sha256=_hash_file(self.root / name),
            )
            for name in AUTHORITATIVE_FILES
        )
        after = self.stat_guard()
        if before != after:
            raise DormantCorpusError("authoritative dormant corpus changed while hashing")
        manifest_hash = next(item.sha256 for item in bindings if item.name == "corpus_manifest.json")
        return DormantCorpusBinding(
            corpus_manifest_sha256=manifest_hash,
            files=bindings,
            recovered_sources=self.recovered_sources(),
        )

    def assert_binding(self, expected: DormantCorpusBinding) -> None:
        observed = self.compute_binding()
        if observed != expected:
            raise StaleDormantIndexError(
                "derived dormant index is stale: authoritative corpus binding changed"
            )


def default_index_path(state_root: str | os.PathLike[str]) -> Path:
    return Path(state_root).resolve() / DEFAULT_INDEX_RELATIVE


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    container_id: str
    score: float
    lexical_hits: int
    graph_hits: int
    edge_ids: tuple[str, ...] = ()
    relation_hits: int = 0


@dataclass(frozen=True, slots=True)
class VerifiedDormantEdge:
    edge_id: str
    source_container_id: str
    edge_type: str
    target: str
    source_text: str
    provenance: str
    confidence: float
    status: str
    byte_offset: int
    byte_length: int
    raw_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedDormantContainer:
    container_id: str
    text: str
    normalized_text: str
    kind: str
    source: str
    provenance: str
    confidence: float
    status: str
    byte_offset: int
    byte_length: int
    raw_sha256: str
    text_sha256: str
    record: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VerifiedDormantEvidence:
    candidate: EvidenceCandidate
    container: VerifiedDormantContainer
    edges: tuple[VerifiedDormantEdge, ...] = ()


@dataclass(frozen=True, slots=True)
class SurfacedDormantResult:
    query: str
    evidence: tuple[VerifiedDormantEvidence, ...]
    snapshot: SharedFieldSnapshot
    compiled: Any


class DormantEvidenceIndex:
    """Disposable metadata-only lexical/graph index over authoritative JSONL."""

    def __init__(
        self,
        corpus: DormantCorpus,
        index_path: str | os.PathLike[str],
        connection: sqlite3.Connection,
        binding: DormantCorpusBinding,
        index_id: str,
        *,
        binding_verified: bool,
    ) -> None:
        self.corpus = corpus
        self.path = Path(index_path).resolve()
        self._connection = connection
        self.binding = binding
        self.index_id = index_id
        self._binding_verified = binding_verified
        self._stat_guard = corpus.stat_guard()
        self._connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "DormantEvidenceIndex":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA foreign_keys=OFF;
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE containers (
                container_rowid INTEGER PRIMARY KEY,
                container_id TEXT NOT NULL UNIQUE,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                raw_sha256 BLOB NOT NULL,
                text_sha256 BLOB NOT NULL,
                source_sha256 BLOB NOT NULL,
                provenance_sha256 BLOB NOT NULL,
                normalized_key BLOB NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL
            );
            CREATE TABLE container_terms (
                term_hash BLOB NOT NULL,
                container_rowid INTEGER NOT NULL,
                PRIMARY KEY (term_hash, container_rowid)
            ) WITHOUT ROWID;
            CREATE TABLE edges (
                edge_rowid INTEGER PRIMARY KEY,
                edge_id TEXT NOT NULL UNIQUE,
                byte_offset INTEGER NOT NULL,
                byte_length INTEGER NOT NULL,
                raw_sha256 BLOB NOT NULL,
                source_container_id TEXT NOT NULL,
                target_key BLOB NOT NULL,
                target_text_sha256 BLOB NOT NULL,
                source_text_sha256 BLOB NOT NULL,
                provenance_sha256 BLOB NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL
            );
            CREATE TABLE edge_terms (
                term_hash BLOB NOT NULL,
                edge_rowid INTEGER NOT NULL,
                PRIMARY KEY (term_hash, edge_rowid)
            ) WITHOUT ROWID;
            CREATE TABLE graph_neighbors (
                source_container_rowid INTEGER NOT NULL,
                target_container_rowid INTEGER NOT NULL,
                edge_rowid INTEGER NOT NULL,
                PRIMARY KEY (source_container_rowid, target_container_rowid, edge_rowid)
            ) WITHOUT ROWID;
            """
        )

    @classmethod
    def build(
        cls,
        state_root: str | os.PathLike[str],
        index_path: str | os.PathLike[str] | None = None,
    ) -> "DormantEvidenceIndex":
        corpus = DormantCorpus.from_state_root(state_root)
        target = Path(index_path).resolve() if index_path is not None else default_index_path(state_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        build_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        temp = target.with_name(f"{target.name}.building.{build_stamp}.{os.getpid()}")
        sidecar = target.with_name(target.name + ".manifest.json")
        sidecar_temp = sidecar.with_name(f"{sidecar.name}.building.{build_stamp}.{os.getpid()}")

        corpus_guard_before = corpus.stat_guard()
        connection = sqlite3.connect(temp)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        cls._create_schema(connection)

        file_hashes: dict[str, str] = {}
        file_sizes: dict[str, int] = {}
        container_count = 0
        edge_count = 0
        try:
            digest = hashlib.sha256()
            offset = 0
            with corpus.containers_path.open("rb") as handle:
                for line in handle:
                    line_offset = offset
                    offset += len(line)
                    digest.update(line)
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise DormantCorpusError(
                            f"invalid container JSON at byte offset {line_offset}"
                        ) from exc
                    container_id = str(record.get("container_id", ""))
                    if not container_id:
                        raise DormantCorpusError(
                            f"container missing container_id at byte offset {line_offset}"
                        )
                    text = _container_text(record)
                    normalized = _container_normalized_text(record)
                    raw_digest = _sha256_digest(line)
                    cursor = connection.execute(
                        """
                        INSERT INTO containers (
                            container_id, byte_offset, byte_length, raw_sha256,
                            text_sha256, source_sha256, provenance_sha256,
                            normalized_key, kind, status, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            container_id,
                            line_offset,
                            len(line),
                            raw_digest,
                            _sha256_text_digest(text),
                            _sha256_text_digest(_record_source(record)),
                            _sha256_text_digest(_record_provenance(record)),
                            _graph_key(normalized),
                            str(record.get("kind", record.get("category", ""))),
                            str(record.get("status", "")),
                            float(record.get("confidence", 1.0)),
                        ),
                    )
                    container_rowid = int(cursor.lastrowid)
                    terms = _container_terms(record)
                    if terms:
                        connection.executemany(
                            "INSERT OR IGNORE INTO container_terms(term_hash, container_rowid) VALUES (?, ?)",
                            ((_term_key(term), container_rowid) for term in terms),
                        )
                    container_count += 1
                    if container_count % 5000 == 0:
                        connection.commit()
            file_hashes["containers.jsonl"] = digest.hexdigest()
            file_sizes["containers.jsonl"] = offset

            digest = hashlib.sha256()
            offset = 0
            with corpus.semantic_edges_path.open("rb") as handle:
                for line in handle:
                    line_offset = offset
                    offset += len(line)
                    digest.update(line)
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise DormantCorpusError(
                            f"invalid semantic edge JSON at byte offset {line_offset}"
                        ) from exc
                    raw_digest = _sha256_digest(line)
                    edge_id = _edge_id(line_offset, raw_digest.hex())
                    source_container_id = str(record.get("source_container_id", ""))
                    edge_target = str(record.get("target", ""))
                    cursor = connection.execute(
                        """
                        INSERT INTO edges (
                            edge_id, byte_offset, byte_length, raw_sha256,
                            source_container_id, target_key, target_text_sha256,
                            source_text_sha256, provenance_sha256, status, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            edge_id,
                            line_offset,
                            len(line),
                            raw_digest,
                            source_container_id,
                            _graph_key(edge_target),
                            _sha256_text_digest(edge_target),
                            _sha256_text_digest(str(record.get("source_text", ""))),
                            _sha256_text_digest(_record_provenance(record)),
                            str(record.get("status", "")),
                            float(record.get("confidence", 1.0)),
                        ),
                    )
                    edge_rowid = int(cursor.lastrowid)
                    terms = _edge_terms(record)
                    if terms:
                        connection.executemany(
                            "INSERT OR IGNORE INTO edge_terms(term_hash, edge_rowid) VALUES (?, ?)",
                            ((_term_key(term), edge_rowid) for term in terms),
                        )
                    edge_count += 1
                    if edge_count % 5000 == 0:
                        connection.commit()
            file_hashes["semantic_edges.jsonl"] = digest.hexdigest()
            file_sizes["semantic_edges.jsonl"] = offset

            for name in AUTHORITATIVE_FILES:
                path = corpus.root / name
                if name not in file_hashes:
                    file_hashes[name] = _hash_file(path)
                    file_sizes[name] = path.stat().st_size

            corpus_guard_after = corpus.stat_guard()
            if corpus_guard_before != corpus_guard_after:
                raise DormantCorpusError("authoritative dormant corpus changed during index build")

            bindings = tuple(
                CorpusFileBinding(name=name, size_bytes=file_sizes[name], sha256=file_hashes[name])
                for name in AUTHORITATIVE_FILES
            )
            manifest_hash = file_hashes["corpus_manifest.json"]
            binding = DormantCorpusBinding(
                corpus_manifest_sha256=manifest_hash,
                files=bindings,
                recovered_sources=corpus.recovered_sources(),
            )
            index_id = canonical_sha256({"schema": INDEX_SCHEMA, "binding": binding.to_canonical_dict()})

            connection.executescript(
                """
                CREATE INDEX idx_containers_normalized_key ON containers(normalized_key);
                CREATE INDEX idx_containers_kind_status ON containers(kind, status, confidence);
                CREATE INDEX idx_container_terms_container ON container_terms(container_rowid);
                CREATE INDEX idx_edges_source ON edges(source_container_id);
                CREATE INDEX idx_edges_target_key ON edges(target_key);
                CREATE INDEX idx_edge_terms_edge ON edge_terms(edge_rowid);
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO graph_neighbors(
                    source_container_rowid, target_container_rowid, edge_rowid
                )
                SELECT source.container_rowid, target.container_rowid, e.edge_rowid
                FROM edges e
                JOIN containers source ON source.container_id = e.source_container_id
                JOIN containers target ON target.normalized_key = e.target_key
                """
            )
            connection.execute(
                "CREATE INDEX idx_graph_target ON graph_neighbors(target_container_rowid)"
            )
            connection.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                (
                    ("schema", INDEX_SCHEMA),
                    ("index_id", index_id),
                    ("binding_json", canonical_json_bytes(binding.to_canonical_dict()).decode("utf-8")),
                    ("container_count", str(container_count)),
                    ("edge_count", str(edge_count)),
                ),
            )
            connection.commit()
            connection.execute("PRAGMA optimize")
            connection.commit()
            connection.close()

            manifest_payload = {
                "schema": INDEX_MANIFEST_SCHEMA,
                "index_schema": INDEX_SCHEMA,
                "index_id": index_id,
                "built_at": datetime.now(timezone.utc).isoformat(),
                "index_file": target.name,
                "container_count": container_count,
                "edge_count": edge_count,
                "binding": binding.to_canonical_dict(),
                "authority_note": (
                    "Derived lookup metadata only. Exact dormant text remains authoritative "
                    "only in State/dormant JSONL and is dereferenced/verified at use time."
                ),
            }
            sidecar_temp.write_bytes(canonical_json_bytes(manifest_payload) + b"\n")
            os.replace(temp, target)
            os.replace(sidecar_temp, sidecar)

            reopened = sqlite3.connect(target)
            return cls(
                corpus=corpus,
                index_path=target,
                connection=reopened,
                binding=binding,
                index_id=index_id,
                binding_verified=True,
            )
        except Exception:
            with suppress(Exception):
                connection.close()
            failed_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            for transient in (temp, sidecar_temp):
                if not transient.exists():
                    continue
                failed_path = transient.with_name(f"{transient.name}.failed.{failed_stamp}")
                with suppress(OSError):
                    os.replace(transient, failed_path)
                # Preserve the transient in place if even a same-volume rename fails.
            raise

    @classmethod
    def open(
        cls,
        state_root: str | os.PathLike[str],
        index_path: str | os.PathLike[str] | None = None,
        *,
        verify_binding: bool = True,
    ) -> "DormantEvidenceIndex":
        corpus = DormantCorpus.from_state_root(state_root)
        target = Path(index_path).resolve() if index_path is not None else default_index_path(state_root)
        if not target.is_file():
            raise DormantEvidenceError(f"derived dormant evidence index does not exist: {target}")
        connection = sqlite3.connect(target)
        connection.row_factory = sqlite3.Row
        meta = dict(connection.execute("SELECT key, value FROM meta"))
        if meta.get("schema") != INDEX_SCHEMA:
            connection.close()
            raise DormantEvidenceError("unsupported dormant evidence index schema")
        try:
            binding = DormantCorpusBinding.from_mapping(json.loads(meta["binding_json"]))
            index_id = str(meta["index_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            connection.close()
            raise DormantEvidenceError("dormant evidence index metadata is invalid") from exc
        expected_index_id = canonical_sha256(
            {"schema": INDEX_SCHEMA, "binding": binding.to_canonical_dict()}
        )
        if index_id != expected_index_id:
            connection.close()
            raise DormantEvidenceError("dormant evidence index identity does not match its binding")
        instance = cls(
            corpus=corpus,
            index_path=target,
            connection=connection,
            binding=binding,
            index_id=index_id,
            binding_verified=False,
        )
        if verify_binding:
            instance.assert_fresh()
        return instance

    def assert_fresh(self) -> None:
        self.corpus.assert_binding(self.binding)
        self._stat_guard = self.corpus.stat_guard()
        self._binding_verified = True

    def _require_verified(self) -> None:
        if not self._binding_verified:
            raise StaleDormantIndexError(
                "index binding has not been verified against the authoritative corpus"
            )
        if self.corpus.stat_guard() != self._stat_guard:
            self._binding_verified = False
            raise StaleDormantIndexError(
                "derived dormant index is stale: authoritative corpus file metadata changed"
            )

    def lookup_container_ids_by_normalized_text(
        self,
        text: str,
        *,
        limit: int = 32,
    ) -> tuple[str, ...]:
        """Resolve an exact normalized-text key to derived container IDs.

        This is lookup metadata only. Callers that need evidence must still
        dereference the returned IDs through ``dereference_container`` before
        treating them as authoritative.
        """

        self._require_verified()
        if not isinstance(text, str) or not text.strip():
            raise DormantQueryError("text must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise DormantQueryError("limit must be a positive integer")
        rows = self._connection.execute(
            """
            SELECT container_id
            FROM containers
            WHERE normalized_key = ?
            ORDER BY container_rowid ASC
            LIMIT ?
            """,
            (_graph_key(text), limit),
        )
        return tuple(str(row["container_id"]) for row in rows)

    def iter_edge_sources(self) -> Iterable[tuple[str, str]]:
        """Yield derived edge/source IDs in stable index order.

        This is a sequential metadata-only sense over the compact edge table.
        It exposes no edge or container text; callers must exact-dereference IDs
        before treating a sampled link as evidence.
        """

        self._require_verified()
        rows = self._connection.execute(
            """
            SELECT edge_id, source_container_id
            FROM edges
            ORDER BY edge_rowid ASC
            """
        )
        for row in rows:
            yield str(row["edge_id"]), str(row["source_container_id"])

    def resolved_target_container_ids(
        self,
        edge_id: str,
        *,
        limit: int = 16,
    ) -> tuple[str, ...]:
        """Resolve one derived edge ID to matching target-container IDs."""

        self._require_verified()
        if not isinstance(edge_id, str) or not edge_id:
            raise DormantQueryError("edge_id must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise DormantQueryError("limit must be a positive integer")
        rows = self._connection.execute(
            """
            SELECT target.container_id
            FROM edges e
            JOIN containers target ON target.normalized_key = e.target_key
            WHERE e.edge_id = ?
            ORDER BY target.container_rowid ASC
            LIMIT ?
            """,
            (edge_id, limit),
        )
        return tuple(str(row["container_id"]) for row in rows)

    def query_candidates(
        self,
        query: str,
        *,
        limit: int = 8,
        kinds: Iterable[str] | None = None,
        statuses: Iterable[str] | None = ("dormant", "active"),
        min_confidence: float = 0.0,
        include_graph: bool = True,
    ) -> tuple[EvidenceCandidate, ...]:
        self._require_verified()
        if not isinstance(query, str) or not query.strip():
            raise DormantQueryError("query must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise DormantQueryError("limit must be a positive integer")
        if not 0.0 <= float(min_confidence) <= 1.0:
            raise DormantQueryError("min_confidence must be in [0, 1]")
        terms = _tokenize(query)
        if not terms:
            raise DormantQueryError("query produced no lexical terms")

        term_hashes = tuple(_term_key(term) for term in terms)
        scores: dict[int, float] = {}
        lexical_hits: dict[int, int] = {}
        graph_hits: dict[int, int] = {}
        relation_hits: dict[int, int] = {}
        edge_refs: dict[int, set[str]] = {}

        # SQLite variable limits are a physical paging concern, not a query
        # context limit. Every unique term is processed in ordered pages and
        # its hits are accumulated before the caller's result-count policy is
        # applied.
        governed = capacity_policy()
        term_page_size = governed.integer("dormant.term_query_page_size")
        lexical_candidate_count = max(
            limit * governed.integer("dormant.lexical_candidate_multiplier"),
            governed.integer("dormant.lexical_candidate_floor"),
        )
        edge_candidate_count = max(
            limit * governed.integer("dormant.edge_candidate_multiplier"),
            governed.integer("dormant.edge_candidate_floor"),
        )
        ranked_edge_hit_map: dict[int, int] = {}
        for start in range(0, len(term_hashes), term_page_size):
            term_page = term_hashes[start : start + term_page_size]
            placeholders = ",".join("?" for _ in term_page)
            rows = self._connection.execute(
                f"""
                SELECT container_rowid, COUNT(*) AS hits
                FROM container_terms
                WHERE term_hash IN ({placeholders})
                GROUP BY container_rowid
                ORDER BY hits DESC, container_rowid ASC
                LIMIT ?
                """,
                (*term_page, lexical_candidate_count),
            )
            for row in rows:
                container_rowid = int(row["container_rowid"])
                hits = int(row["hits"])
                lexical_hits[container_rowid] = lexical_hits.get(container_rowid, 0) + hits
                scores[container_rowid] = scores.get(container_rowid, 0.0) + hits * 10.0

            # Rank edge IDs from the compact postings table first. Joining
            # source metadata during the aggregate is needlessly expensive on
            # the live multi-GB index; hydrate only bounded winners afterward.
            ranked_edge_rows = self._connection.execute(
                f"""
                SELECT edge_rowid, COUNT(*) AS hits
                FROM edge_terms
                WHERE term_hash IN ({placeholders})
                GROUP BY edge_rowid
                ORDER BY hits DESC, edge_rowid ASC
                LIMIT ?
                """,
                (*term_page, edge_candidate_count),
            )
            for row in ranked_edge_rows:
                edge_rowid = int(row["edge_rowid"])
                ranked_edge_hit_map[edge_rowid] = (
                    ranked_edge_hit_map.get(edge_rowid, 0) + int(row["hits"])
                )

        ranked_edge_hits = tuple(
            sorted(
                ranked_edge_hit_map.items(),
                key=lambda item: (-item[1], item[0]),
            )[:edge_candidate_count]
        )
        edge_meta: dict[int, tuple[str, int, bytes]] = {}
        edge_rowids = [edge_rowid for edge_rowid, _ in ranked_edge_hits]
        edge_hydration_page_size = governed.integer("dormant.edge_hydration_page_size")
        for start in range(0, len(edge_rowids), edge_hydration_page_size):
            chunk = edge_rowids[start : start + edge_hydration_page_size]
            chunk_placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                f"""
                SELECT e.edge_rowid, e.edge_id, e.target_key,
                       source.container_rowid AS source_container_rowid
                FROM edges e
                LEFT JOIN containers source ON source.container_id = e.source_container_id
                WHERE e.edge_rowid IN ({chunk_placeholders})
                """,
                chunk,
            )
            for row in rows:
                if row["source_container_rowid"] is None:
                    continue
                edge_meta[int(row["edge_rowid"])] = (
                    str(row["edge_id"]),
                    int(row["source_container_rowid"]),
                    bytes(row["target_key"]),
                )

        matched_edges: list[tuple[int, str, int, int, bytes]] = []
        for edge_rowid, hits in ranked_edge_hits:
            meta = edge_meta.get(edge_rowid)
            if meta is None:
                continue
            edge_id, container_rowid, target_key = meta
            lexical_hits[container_rowid] = lexical_hits.get(container_rowid, 0) + hits
            scores[container_rowid] = scores.get(container_rowid, 0.0) + hits * 6.0
            edge_refs.setdefault(container_rowid, set()).add(edge_id)
            matched_edges.append((edge_rowid, edge_id, container_rowid, hits, target_key))

        if include_graph and matched_edges:
            # Relation-aware expansion: edges whose *own indexed terms* matched
            # the query get a bounded direct path to their resolved targets.
            # Resolve target keys through the existing normalized-key index,
            # keeping at most four IDs per key. This avoids materializing every
            # duplicate target row for common concepts.
            #
            # Support is a bounded best-edge signal, never an additive pile-up:
            # target_support = matched_source_score * edge_query_coverage.
            matched_cap = min(
                len(matched_edges),
                max(
                    limit * governed.integer("dormant.relation_seed_multiplier"),
                    governed.integer("dormant.relation_seed_floor"),
                ),
            )
            matched = matched_edges[:matched_cap]
            resolved_targets: dict[bytes, tuple[int, ...]] = {}
            for target_key in {item[4] for item in matched}:
                rows = self._connection.execute(
                    """
                    SELECT container_rowid
                    FROM containers
                    WHERE normalized_key = ?
                    ORDER BY container_rowid ASC
                    LIMIT ?
                    """,
                    (target_key, governed.integer("dormant.relation_targets_per_key")),
                )
                resolved_targets[target_key] = tuple(int(row["container_rowid"]) for row in rows)

            relation_support: dict[int, float] = {}
            relation_hits = {}
            relation_edge: dict[int, str] = {}
            query_term_count = max(1, len(terms))
            for _, edge_id, source_rowid, hits, target_key in matched:
                coverage = min(1.0, hits / query_term_count)
                support = scores.get(source_rowid, 0.0) * coverage
                if support <= 0.0:
                    continue
                for target_rowid in resolved_targets.get(target_key, ()):
                    if target_rowid == source_rowid:
                        continue
                    if support > relation_support.get(target_rowid, 0.0):
                        relation_support[target_rowid] = support
                        relation_hits[target_rowid] = hits
                        relation_edge[target_rowid] = edge_id

            for target_rowid, support in relation_support.items():
                scores[target_rowid] = max(scores.get(target_rowid, 0.0), support)
                graph_hits[target_rowid] = max(graph_hits.get(target_rowid, 0), 1)
                edge_refs.setdefault(target_rowid, set()).add(relation_edge[target_rowid])

        if include_graph and scores:
            # Topological expansion remains useful when a query identifies a
            # container but does not strongly match one edge.  Traverse a small
            # equal fan-out per seed in both directions rather than one global
            # LIMIT, so a 10k-degree node cannot consume another seed's budget.
            seed_rowids = [
                item[0]
                for item in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[
                    : max(
                        limit * governed.integer("dormant.graph_seed_multiplier"),
                        governed.integer("dormant.graph_seed_floor"),
                    )
                ]
            ]
            seed_scores = {item: scores[item] for item in seed_rowids}
            fanout_per_direction = governed.integer("dormant.graph_fanout_per_direction")
            graph_hit_saturation = governed.integer("dormant.graph_hit_saturation")
            for seed_rowid in seed_rowids:
                outgoing = self._connection.execute(
                    """
                    SELECT g.source_container_rowid, g.target_container_rowid, e.edge_id
                    FROM graph_neighbors g
                    JOIN edges e ON e.edge_rowid = g.edge_rowid
                    WHERE g.source_container_rowid = ?
                    ORDER BY g.edge_rowid ASC, g.target_container_rowid ASC
                    LIMIT ?
                    """,
                    (seed_rowid, fanout_per_direction),
                )
                for row in outgoing:
                    target_rowid = int(row["target_container_rowid"])
                    if target_rowid == seed_rowid:
                        continue
                    edge_id = str(row["edge_id"])
                    prior_graph_hits = graph_hits.get(target_rowid, 0)
                    if prior_graph_hits < graph_hit_saturation:
                        graph_hits[target_rowid] = prior_graph_hits + 1
                    scores[target_rowid] = max(
                        scores.get(target_rowid, 0.0),
                        min(seed_scores[seed_rowid] * 0.10, 3.0),
                    )
                    edge_refs.setdefault(target_rowid, set()).add(edge_id)

                incoming = self._connection.execute(
                    """
                    SELECT g.source_container_rowid, g.target_container_rowid, e.edge_id
                    FROM graph_neighbors g
                    JOIN edges e ON e.edge_rowid = g.edge_rowid
                    WHERE g.target_container_rowid = ?
                    ORDER BY g.edge_rowid ASC, g.source_container_rowid ASC
                    LIMIT ?
                    """,
                    (seed_rowid, fanout_per_direction),
                )
                for row in incoming:
                    source_rowid = int(row["source_container_rowid"])
                    if source_rowid == seed_rowid:
                        continue
                    edge_id = str(row["edge_id"])
                    prior_graph_hits = graph_hits.get(source_rowid, 0)
                    if prior_graph_hits < graph_hit_saturation:
                        graph_hits[source_rowid] = prior_graph_hits + 1
                    scores[source_rowid] = max(
                        scores.get(source_rowid, 0.0),
                        min(seed_scores[seed_rowid] * 0.10, 3.0),
                    )
                    edge_refs.setdefault(source_rowid, set()).add(edge_id)

        if not scores:
            return ()

        candidate_rowids = list(scores)
        allowed_kinds = None if kinds is None else {str(value) for value in kinds}
        allowed_statuses = None if statuses is None else {str(value) for value in statuses}
        filtered: list[EvidenceCandidate] = []
        for start in range(0, len(candidate_rowids), edge_hydration_page_size):
            chunk = candidate_rowids[start : start + edge_hydration_page_size]
            chunk_placeholders = ",".join("?" for _ in chunk)
            metadata_rows = self._connection.execute(
                f"""
                SELECT container_rowid, container_id, kind, status, confidence
                FROM containers
                WHERE container_rowid IN ({chunk_placeholders})
                """,
                chunk,
            )
            for row in metadata_rows:
                container_rowid = int(row["container_rowid"])
                container_id = str(row["container_id"])
                if allowed_kinds is not None and str(row["kind"]) not in allowed_kinds:
                    continue
                if allowed_statuses is not None and str(row["status"]) not in allowed_statuses:
                    continue
                if float(row["confidence"]) < float(min_confidence):
                    continue
                filtered.append(
                    EvidenceCandidate(
                        container_id=container_id,
                        score=float(scores[container_rowid]),
                        lexical_hits=int(lexical_hits.get(container_rowid, 0)),
                        graph_hits=int(graph_hits.get(container_rowid, 0)),
                        edge_ids=tuple(sorted(edge_refs.get(container_rowid, set()))),
                        relation_hits=int(relation_hits.get(container_rowid, 0)),
                    )
                )
        filtered.sort(key=lambda item: (-item.score, -item.lexical_hits, item.container_id))
        return tuple(filtered[:limit])

    def dereference_container(self, container_id: str) -> VerifiedDormantContainer:
        self._require_verified()
        row = self._connection.execute(
            "SELECT * FROM containers WHERE container_id = ?", (container_id,)
        ).fetchone()
        if row is None:
            raise KeyError(container_id)
        with self.corpus.containers_path.open("rb") as handle:
            handle.seek(int(row["byte_offset"]))
            raw = handle.read(int(row["byte_length"]))
        if len(raw) != int(row["byte_length"]) or _sha256_digest(raw) != bytes(row["raw_sha256"]):
            raise DormantRecordIntegrityError(
                f"container {container_id} raw bytes no longer match the derived index"
            )
        try:
            record = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DormantRecordIntegrityError(f"container {container_id} JSON is invalid") from exc
        if str(record.get("container_id", "")) != container_id:
            raise DormantRecordIntegrityError(f"container {container_id} identity mismatch")
        text = _container_text(record)
        source = _record_source(record)
        provenance = _record_provenance(record)
        checks = (
            (_sha256_text_digest(text), bytes(row["text_sha256"]), "text"),
            (_sha256_text_digest(source), bytes(row["source_sha256"]), "source"),
            (_sha256_text_digest(provenance), bytes(row["provenance_sha256"]), "provenance"),
        )
        for observed, expected, label in checks:
            if observed != expected:
                raise DormantRecordIntegrityError(
                    f"container {container_id} {label} hash mismatch"
                )
        return VerifiedDormantContainer(
            container_id=container_id,
            text=text,
            normalized_text=str(record.get("normalized_text", "")),
            kind=str(record.get("kind", record.get("category", ""))),
            source=source,
            provenance=provenance,
            confidence=float(record.get("confidence", 1.0)),
            status=str(record.get("status", "")),
            byte_offset=int(row["byte_offset"]),
            byte_length=int(row["byte_length"]),
            raw_sha256=bytes(row["raw_sha256"]).hex(),
            text_sha256=bytes(row["text_sha256"]).hex(),
            record=record,
        )

    def dereference_edge(self, edge_id: str) -> VerifiedDormantEdge:
        self._require_verified()
        row = self._connection.execute("SELECT * FROM edges WHERE edge_id = ?", (edge_id,)).fetchone()
        if row is None:
            raise KeyError(edge_id)
        with self.corpus.semantic_edges_path.open("rb") as handle:
            handle.seek(int(row["byte_offset"]))
            raw = handle.read(int(row["byte_length"]))
        if len(raw) != int(row["byte_length"]) or _sha256_digest(raw) != bytes(row["raw_sha256"]):
            raise DormantRecordIntegrityError(
                f"semantic edge {edge_id} raw bytes no longer match the derived index"
            )
        if edge_id != _edge_id(int(row["byte_offset"]), _sha256_bytes(raw)):
            raise DormantRecordIntegrityError(f"semantic edge {edge_id} identity mismatch")
        try:
            record = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DormantRecordIntegrityError(f"semantic edge {edge_id} JSON is invalid") from exc
        source_container_id = str(record.get("source_container_id", ""))
        target = str(record.get("target", ""))
        source_text = str(record.get("source_text", ""))
        provenance = _record_provenance(record)
        checks = (
            (_sha256_text_digest(target), bytes(row["target_text_sha256"]), "target"),
            (_sha256_text_digest(source_text), bytes(row["source_text_sha256"]), "source_text"),
            (_sha256_text_digest(provenance), bytes(row["provenance_sha256"]), "provenance"),
        )
        if source_container_id != str(row["source_container_id"]):
            raise DormantRecordIntegrityError(f"semantic edge {edge_id} source identity mismatch")
        for observed, expected, label in checks:
            if observed != expected:
                raise DormantRecordIntegrityError(
                    f"semantic edge {edge_id} {label} hash mismatch"
                )
        return VerifiedDormantEdge(
            edge_id=edge_id,
            source_container_id=source_container_id,
            edge_type=str(record.get("edge_type", "")),
            target=target,
            source_text=source_text,
            provenance=provenance,
            confidence=float(record.get("confidence", 1.0)),
            status=str(record.get("status", "")),
            byte_offset=int(row["byte_offset"]),
            byte_length=int(row["byte_length"]),
            raw_sha256=bytes(row["raw_sha256"]).hex(),
        )

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 8,
        kinds: Iterable[str] | None = None,
        statuses: Iterable[str] | None = ("dormant", "active"),
        min_confidence: float = 0.0,
        include_graph: bool = True,
        edges_per_evidence: int | None = None,
    ) -> tuple[VerifiedDormantEvidence, ...]:
        candidates = self.query_candidates(
            query,
            limit=limit,
            kinds=kinds,
            statuses=statuses,
            min_confidence=min_confidence,
            include_graph=include_graph,
        )
        edge_count = (
            capacity_policy().integer("dormant.edges_per_evidence")
            if edges_per_evidence is None
            else int(edges_per_evidence)
        )
        if edge_count < 1:
            raise DormantQueryError("edges_per_evidence must be positive")
        evidence: list[VerifiedDormantEvidence] = []
        for candidate in candidates:
            container = self.dereference_container(candidate.container_id)
            verified_edges: list[VerifiedDormantEdge] = []
            for edge_id in candidate.edge_ids[:edge_count]:
                edge = self.dereference_edge(edge_id)
                if edge.source_container_id == container.container_id or candidate.graph_hits:
                    verified_edges.append(edge)
            evidence.append(
                VerifiedDormantEvidence(
                    candidate=candidate,
                    container=container,
                    edges=tuple(verified_edges),
                )
            )
        return tuple(evidence)


class DormantEvidenceBridge:
    """Retrieval -> exact dereference -> canonical cortex -> D64."""

    def __init__(self, index: DormantEvidenceIndex) -> None:
        self.index = index

    def surface(
        self,
        base: SharedFieldSnapshot,
        evidence: Sequence[VerifiedDormantEvidence],
        *,
        replace_existing: bool = True,
    ) -> SharedFieldSnapshot:
        if not isinstance(base, SharedFieldSnapshot):
            raise TypeError("base must be SharedFieldSnapshot")
        if not evidence:
            raise DormantQueryError("cannot surface an empty evidence set")

        spans: list[FieldSpan] = []
        if not replace_existing:
            spans.extend(base.region(LogicalRegion.CORTEX).spans)
        for index, item in enumerate(evidence):
            container = item.container
            if not container.text:
                continue
            if spans:
                spans.append(
                    FieldSpan(
                        span_id=f"dormant-separator:{base.field_id[:12]}:{index}",
                        text="\n",
                        kind="evidence_separator",
                        source="runtime.dormant.evidence_bridge",
                        provenance=f"derived-index:{self.index.index_id}",
                    )
                )
            edge_ids = tuple(edge.edge_id for edge in item.edges)
            provenance_payload = {
                "bridge": INDEX_SCHEMA,
                "index_id": self.index.index_id,
                "binding_id": self.index.binding.binding_id,
                "authoritative_file": "containers.jsonl",
                "byte_offset": container.byte_offset,
                "byte_length": container.byte_length,
                "raw_sha256": container.raw_sha256,
                "text_sha256": container.text_sha256,
                "source": container.source,
                "record_provenance": container.provenance,
            }
            spans.append(
                FieldSpan(
                    span_id=f"dormant:{container.container_id}:{container.raw_sha256[:16]}",
                    text=container.text,
                    kind=f"dormant_{container.kind or 'evidence'}",
                    source=container.source,
                    provenance=canonical_json_bytes(provenance_payload).decode("utf-8"),
                    confidence=container.confidence,
                    container_refs=(container.container_id,),
                    edge_refs=edge_ids,
                )
            )
        if not spans:
            raise DormantQueryError("selected evidence had no surfacable exact text")

        replacement = RegionState(
            name=LogicalRegion.CORTEX,
            spans=tuple(spans),
            visibility=RegionVisibility.ATTENDED,
        )
        regions = tuple(
            replacement if region.name is LogicalRegion.CORTEX else region
            for region in base.regions
        )
        manifest_ids = tuple(
            sorted(
                set(base.source_manifest_ids)
                | {
                    f"dormant-binding:{self.index.binding.binding_id}",
                    f"dormant-index:{self.index.index_id}",
                }
            )
        )
        return SharedFieldSnapshot(
            tick_id=base.tick_id + 1,
            parent_field_id=base.field_id,
            source_manifest_ids=manifest_ids,
            regions=regions,
        )

    def query_surface_compile(
        self,
        base: SharedFieldSnapshot,
        query: str,
        *,
        limit: int = 8,
        include_graph: bool = True,
        compiler: D64FieldCompiler | None = None,
    ) -> SurfacedDormantResult:
        evidence = self.index.retrieve(query, limit=limit, include_graph=include_graph)
        if not evidence:
            raise DormantQueryError(f"no dormant evidence candidates for query {query!r}")
        surfaced = self.surface(base, evidence)
        active_compiler = compiler or D64FieldCompiler()
        compiled = active_compiler.compile(surfaced)
        compiled.verify_roundtrip(surfaced)
        return SurfacedDormantResult(
            query=query,
            evidence=evidence,
            snapshot=surfaced,
            compiled=compiled,
        )
