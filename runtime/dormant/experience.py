"""Exact, content-addressed lived-experience storage beneath Dormant State.

The recovered container corpus is useful semantic/retrieval anatomy, but it is
allowed to normalize and derive readable surfaces.  This module supplies the
separate evidence layer that normalization may never replace: immutable batches
of exact record values bound to immutable source snapshots.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from runtime.field import canonical_json_bytes, canonical_sha256


EXPERIENCE_RECORD_SCHEMA = "axon-dormant-experience-record-v1"
EXPERIENCE_IMPORT_SCHEMA = "axon-dormant-experience-import-v1"
INLINE_SOURCE_SNAPSHOT_SCHEMA = "axon-dormant-inline-source-snapshot-v1"
EXPERIENCE_ROOT_NAME = "experience_v1"


class DormantExperienceError(RuntimeError):
    """The exact lived-experience corpus is missing, stale, or malformed."""


def _nonempty(value: str, label: str) -> str:
    value = str(value)
    if not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Reject non-canonical payload values instead of stringifying evidence."""

    try:
        return json.loads(canonical_json_bytes(value).decode("utf-8"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("experience payload must be canonical JSON data") from exc


@dataclass(frozen=True, slots=True)
class RecoveredSourceSnapshot:
    source_name: str
    original_path: str
    size_bytes: int
    sha256: str
    snapshot_relative_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_name", _nonempty(self.source_name, "source_name"))
        object.__setattr__(self, "original_path", _nonempty(self.original_path, "original_path"))
        object.__setattr__(
            self,
            "snapshot_relative_path",
            _nonempty(self.snapshot_relative_path, "snapshot_relative_path").replace("\\", "/"),
        )
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")
        digest = _nonempty(self.sha256, "sha256").lower()
        if len(digest) != 64:
            raise ValueError("sha256 must be a 64-character digest")
        object.__setattr__(self, "sha256", digest)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "original_path": self.original_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "snapshot_relative_path": self.snapshot_relative_path,
        }


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    record_kind: str
    source_name: str
    source_sha256: str
    source_pointer: str
    sequence: int
    exact_text: str
    payload: Mapping[str, Any]
    occurred_at: str = ""
    evidence_class: str = "recovered_lived_evidence"
    lifecycle: str = "recovered"
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("record_kind", "source_name", "source_pointer", "evidence_class", "lifecycle"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        digest = _nonempty(self.source_sha256, "source_sha256").lower()
        if len(digest) != 64:
            raise ValueError("source_sha256 must be a 64-character digest")
        object.__setattr__(self, "source_sha256", digest)
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if not isinstance(self.exact_text, str):
            raise TypeError("exact_text must be a string")
        payload = _json_safe(dict(self.payload))
        if not self.exact_text and not payload:
            raise ValueError("an experience record must preserve exact_text or payload")
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "occurred_at", str(self.occurred_at or ""))
        object.__setattr__(self, "record_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    @property
    def exact_text_sha256(self) -> str:
        return hashlib.sha256(self.exact_text.encode("utf-8")).hexdigest()

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": EXPERIENCE_RECORD_SCHEMA,
            "record_kind": self.record_kind,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "source_pointer": self.source_pointer,
            "sequence": self.sequence,
            "occurred_at": self.occurred_at,
            "exact_text": self.exact_text,
            "exact_text_sha256": self.exact_text_sha256,
            "payload": dict(self.payload),
            "evidence_class": self.evidence_class,
            "lifecycle": self.lifecycle,
        }
        if include_id:
            value["record_id"] = self.record_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperienceRecord":
        if value.get("schema") != EXPERIENCE_RECORD_SCHEMA:
            raise DormantExperienceError("unsupported experience record schema")
        record = cls(
            record_kind=str(value["record_kind"]),
            source_name=str(value["source_name"]),
            source_sha256=str(value["source_sha256"]),
            source_pointer=str(value["source_pointer"]),
            sequence=int(value["sequence"]),
            exact_text=str(value.get("exact_text", "")),
            payload=dict(value.get("payload", {})),
            occurred_at=str(value.get("occurred_at", "")),
            evidence_class=str(value.get("evidence_class", "recovered_lived_evidence")),
            lifecycle=str(value.get("lifecycle", "recovered")),
        )
        if value.get("record_id") != record.record_id:
            raise DormantExperienceError("experience record identity mismatch")
        if value.get("exact_text_sha256") != record.exact_text_sha256:
            raise DormantExperienceError("experience exact-text hash mismatch")
        return record


@dataclass(frozen=True, slots=True)
class ExperienceImportManifest:
    label: str
    sources: tuple[RecoveredSourceSnapshot, ...]
    record_count: int
    record_kind_counts: tuple[tuple[str, int], ...]
    records_sha256: str
    ordered_record_ids_sha256: str
    import_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _nonempty(self.label, "label"))
        sources = tuple(sorted(tuple(self.sources), key=lambda item: item.source_name))
        if not all(isinstance(item, RecoveredSourceSnapshot) for item in sources):
            raise TypeError("sources must contain RecoveredSourceSnapshot values")
        if len({item.source_name for item in sources}) != len(sources):
            raise ValueError("duplicate experience source_name")
        object.__setattr__(self, "sources", sources)
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or self.record_count < 1:
            raise ValueError("record_count must be a positive integer")
        counts = tuple(sorted((str(name), int(count)) for name, count in self.record_kind_counts))
        if any(not name or count < 1 for name, count in counts) or sum(count for _, count in counts) != self.record_count:
            raise ValueError("record_kind_counts must exactly cover record_count")
        object.__setattr__(self, "record_kind_counts", counts)
        for name in ("records_sha256", "ordered_record_ids_sha256"):
            digest = _nonempty(getattr(self, name), name).lower()
            if len(digest) != 64:
                raise ValueError(f"{name} must be a 64-character digest")
            object.__setattr__(self, name, digest)
        object.__setattr__(self, "import_id", canonical_sha256(self.to_canonical_dict(include_id=False)))

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": EXPERIENCE_IMPORT_SCHEMA,
            "label": self.label,
            "sources": [item.to_canonical_dict() for item in self.sources],
            "record_count": self.record_count,
            "record_kind_counts": [[name, count] for name, count in self.record_kind_counts],
            "records_sha256": self.records_sha256,
            "ordered_record_ids_sha256": self.ordered_record_ids_sha256,
        }
        if include_id:
            value["import_id"] = self.import_id
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperienceImportManifest":
        if value.get("schema") != EXPERIENCE_IMPORT_SCHEMA:
            raise DormantExperienceError("unsupported experience import schema")
        manifest = cls(
            label=str(value["label"]),
            sources=tuple(RecoveredSourceSnapshot(**dict(item)) for item in value.get("sources", [])),
            record_count=int(value["record_count"]),
            record_kind_counts=tuple((str(item[0]), int(item[1])) for item in value["record_kind_counts"]),
            records_sha256=str(value["records_sha256"]),
            ordered_record_ids_sha256=str(value["ordered_record_ids_sha256"]),
        )
        if value.get("import_id") != manifest.import_id:
            raise DormantExperienceError("experience import identity mismatch")
        return manifest


class DormantExperienceStore:
    """Publishes and verifies immutable, idempotent experience batches."""

    def __init__(self, state_root: str | os.PathLike[str]) -> None:
        self.state_root = Path(state_root).resolve()
        self.root = self.state_root / "dormant" / EXPERIENCE_ROOT_NAME
        self.imports_root = self.root / "imports"

    def _verify_sources(self, sources: Sequence[RecoveredSourceSnapshot]) -> None:
        for source in sources:
            path = (self.root / source.snapshot_relative_path).resolve()
            try:
                path.relative_to(self.root.resolve())
            except ValueError as exc:
                raise DormantExperienceError("source snapshot path escapes experience root") from exc
            if not path.is_file() or path.stat().st_size != source.size_bytes:
                raise DormantExperienceError(f"source snapshot is missing or wrong-sized: {path}")
            if _sha256_file(path) != source.sha256:
                raise DormantExperienceError(f"source snapshot hash mismatch: {path}")

    def publish_inline_source_snapshot(
        self,
        *,
        source_name: str,
        original_path: str,
        exact_bytes: bytes,
    ) -> RecoveredSourceSnapshot:
        """Publish one immutable exact source supplied by the running organism.

        Recovered databases are copied by their importer.  New lived events do
        not begin as standalone files, so the Heart first freezes their exact
        durable-envelope bytes here.  The resulting binding is subject to the
        same hash and path verification as every recovered source.
        """

        source_name = _nonempty(source_name, "source_name")
        original_path = _nonempty(original_path, "original_path")
        if not isinstance(exact_bytes, bytes) or not exact_bytes:
            raise ValueError("exact_bytes must be non-empty bytes")
        source_sha256 = hashlib.sha256(exact_bytes).hexdigest()
        snapshot_id = canonical_sha256(
            {
                "schema": INLINE_SOURCE_SNAPSHOT_SCHEMA,
                "source_name": source_name,
                "original_path": original_path,
                "size_bytes": len(exact_bytes),
                "sha256": source_sha256,
            }
        )
        binding = RecoveredSourceSnapshot(
            source_name=source_name,
            original_path=original_path,
            size_bytes=len(exact_bytes),
            sha256=source_sha256,
            snapshot_relative_path=f"source_snapshots/{snapshot_id}/files/source.bin",
        )
        final = self.root / "source_snapshots" / snapshot_id
        expected_manifest = {
            "schema": INLINE_SOURCE_SNAPSHOT_SCHEMA,
            "snapshot_id": snapshot_id,
            "sources": [binding.to_canonical_dict()],
        }
        if final.exists():
            try:
                observed = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise DormantExperienceError("inline source snapshot manifest is unreadable") from exc
            if observed != expected_manifest:
                raise DormantExperienceError("inline source snapshot manifest mismatch")
            self._verify_sources((binding,))
            return binding

        snapshots_root = self.root / "source_snapshots"
        snapshots_root.mkdir(parents=True, exist_ok=True)
        building = snapshots_root / f".b-{uuid.uuid4().hex[:8]}"
        files = building / "files"
        files.mkdir(parents=True, exist_ok=False)
        source_path = files / "source.bin"
        with source_path.open("xb") as handle:
            handle.write(exact_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        with (building / "manifest.json").open("xb") as handle:
            handle.write(canonical_json_bytes(expected_manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(building, final)
        self._verify_sources((binding,))
        return binding

    def publish_import(
        self,
        records: Iterable[ExperienceRecord],
        *,
        sources: Sequence[RecoveredSourceSnapshot],
        label: str,
    ) -> ExperienceImportManifest:
        ordered = tuple(records)
        if not ordered:
            raise ValueError("cannot publish an empty experience import")
        if not all(isinstance(item, ExperienceRecord) for item in ordered):
            raise TypeError("records must contain only ExperienceRecord values")
        ids = [item.record_id for item in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("experience import contains duplicate record identities")
        source_map = {item.source_name: item for item in sources}
        if any(item.source_name not in source_map for item in ordered):
            raise ValueError("every recovered experience record must bind a supplied source snapshot")
        if any(source_map[item.source_name].sha256 != item.source_sha256 for item in ordered):
            raise ValueError("experience record/source snapshot hash mismatch")
        self._verify_sources(tuple(sources))

        record_bytes = b"".join(canonical_json_bytes(item.to_canonical_dict()) + b"\n" for item in ordered)
        counts: dict[str, int] = {}
        for item in ordered:
            counts[item.record_kind] = counts.get(item.record_kind, 0) + 1
        manifest = ExperienceImportManifest(
            label=label,
            sources=tuple(sources),
            record_count=len(ordered),
            record_kind_counts=tuple(counts.items()),
            records_sha256=hashlib.sha256(record_bytes).hexdigest(),
            ordered_record_ids_sha256=hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest(),
        )
        final = self.imports_root / manifest.import_id
        if final.exists():
            observed = self.verify_import(manifest.import_id, verify_sources=True)
            if observed != manifest:
                raise DormantExperienceError("existing import directory does not match requested import")
            return observed

        self.imports_root.mkdir(parents=True, exist_ok=True)
        # Keep the transient name short enough for legacy Windows MAX_PATH.
        # The final content-addressed directory still carries the full identity.
        building = self.imports_root / f".b-{uuid.uuid4().hex[:8]}"
        building.mkdir(parents=False, exist_ok=False)
        records_path = building / "records.jsonl"
        manifest_path = building / "manifest.json"
        with records_path.open("xb") as handle:
            handle.write(record_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        with manifest_path.open("xb") as handle:
            handle.write(canonical_json_bytes(manifest.to_canonical_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(building, final)
        return self.verify_import(manifest.import_id, verify_sources=True)

    def verify_import(self, import_id: str, *, verify_sources: bool = False) -> ExperienceImportManifest:
        import_id = _nonempty(import_id, "import_id")
        root = self.imports_root / import_id
        try:
            raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DormantExperienceError(f"experience import manifest is unreadable: {import_id}") from exc
        manifest = ExperienceImportManifest.from_mapping(raw)
        if manifest.import_id != import_id:
            raise DormantExperienceError("experience import directory identity mismatch")
        records_path = root / "records.jsonl"
        if not records_path.is_file() or _sha256_file(records_path) != manifest.records_sha256:
            raise DormantExperienceError("experience import record-file hash mismatch")
        ids: list[str] = []
        counts: dict[str, int] = {}
        with records_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = ExperienceRecord.from_mapping(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise DormantExperienceError(
                        f"invalid experience record {import_id}:{line_number}"
                    ) from exc
                ids.append(record.record_id)
                counts[record.record_kind] = counts.get(record.record_kind, 0) + 1
        if len(ids) != manifest.record_count or len(ids) != len(set(ids)):
            raise DormantExperienceError("experience import record count/identity mismatch")
        if tuple(sorted(counts.items())) != manifest.record_kind_counts:
            raise DormantExperienceError("experience import kind counts mismatch")
        observed_ids = hashlib.sha256("\n".join(ids).encode("ascii")).hexdigest()
        if observed_ids != manifest.ordered_record_ids_sha256:
            raise DormantExperienceError("experience import ordered identity hash mismatch")
        if verify_sources:
            self._verify_sources(manifest.sources)
        return manifest

    def import_manifests(self) -> tuple[ExperienceImportManifest, ...]:
        if not self.imports_root.is_dir():
            return ()
        manifests = [
            self.verify_import(path.name)
            for path in sorted(self.imports_root.iterdir(), key=lambda item: item.name)
            if path.is_dir() and not path.name.startswith(".")
        ]
        return tuple(manifests)

    def iter_records(self, import_ids: Sequence[str] | None = None) -> Iterator[ExperienceRecord]:
        selected = tuple(import_ids) if import_ids is not None else tuple(
            item.import_id for item in self.import_manifests()
        )
        for import_id in selected:
            manifest = self.verify_import(import_id)
            with (self.imports_root / manifest.import_id / "records.jsonl").open(
                "r", encoding="utf-8"
            ) as handle:
                for line in handle:
                    yield ExperienceRecord.from_mapping(json.loads(line))


__all__ = [
    "EXPERIENCE_RECORD_SCHEMA",
    "EXPERIENCE_IMPORT_SCHEMA",
    "EXPERIENCE_ROOT_NAME",
    "INLINE_SOURCE_SNAPSHOT_SCHEMA",
    "DormantExperienceError",
    "RecoveredSourceSnapshot",
    "ExperienceRecord",
    "ExperienceImportManifest",
    "DormantExperienceStore",
]
