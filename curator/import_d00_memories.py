#!/usr/bin/env python3
"""Preserve Axon's recovered ``D:\\00`` memories in exact Dormant State.

Source databases are opened with SQLite ``mode=ro&immutable=1`` and are never
modified.  Every durable source is copied byte-for-byte into a content-addressed
Dormant snapshot before logical experience records are published.  Derived
vector tables remain present in the raw snapshot but are not duplicated as
experience text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from runtime.dormant import (
    EXPERIENCE_ROOT_NAME,
    DormantExperienceStore,
    ExperienceRecord,
    RecoveredSourceSnapshot,
)
from runtime.field import canonical_json_bytes, canonical_sha256


D00_SNAPSHOT_SCHEMA = "axon-d00-memory-source-snapshot-v1"
D00_IMPORT_LABEL = "d00-recovered-autobiography-v1"
SOURCE_NAMES = (
    "axon_memory.db",
    "axon_runtime_state.db",
    "axon_episodic_memory.db",
    "axon_memory_backlog.db",
    "axon_semantic_memory.db",
    "axon_personal_log.json",
    "axon_runtime_state.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_inventory(source_root: Path) -> tuple[dict[str, Any], ...]:
    inventory: list[dict[str, Any]] = []
    for name in SOURCE_NAMES:
        path = source_root / name
        if not path.is_file():
            raise FileNotFoundError(f"required recovered memory source is missing: {path}")
        inventory.append(
            {
                "source_name": name,
                "original_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return tuple(inventory)


def snapshot_d00_sources(
    source_root: Path,
    state_root: Path,
    *,
    inventory: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, tuple[RecoveredSourceSnapshot, ...]]:
    """Copy and verify a content-addressed byte-exact D:\\00 source snapshot."""

    inventory = tuple(inventory or _source_inventory(source_root))
    snapshot_id = canonical_sha256(
        {"schema": D00_SNAPSHOT_SCHEMA, "sources": [dict(item) for item in inventory]}
    )
    experience_root = state_root.resolve() / "dormant" / EXPERIENCE_ROOT_NAME
    snapshots_root = experience_root / "source_snapshots"
    final_root = snapshots_root / snapshot_id
    files_root = final_root / "files"
    manifest_path = final_root / "manifest.json"

    bindings = tuple(
        RecoveredSourceSnapshot(
            source_name=str(item["source_name"]),
            original_path=str(item["original_path"]),
            size_bytes=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
            snapshot_relative_path=f"source_snapshots/{snapshot_id}/files/{item['source_name']}",
        )
        for item in inventory
    )
    expected_manifest = {
        "schema": D00_SNAPSHOT_SCHEMA,
        "snapshot_id": snapshot_id,
        "sources": [item.to_canonical_dict() for item in bindings],
    }

    if final_root.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != expected_manifest:
            raise RuntimeError("existing D:\\00 source snapshot manifest does not match")
        for item in bindings:
            target = experience_root / item.snapshot_relative_path
            if target.stat().st_size != item.size_bytes or _sha256_file(target) != item.sha256:
                raise RuntimeError(f"existing D:\\00 source snapshot is corrupt: {target}")
        return snapshot_id, bindings

    snapshots_root.mkdir(parents=True, exist_ok=True)
    # Keep the transient name short enough for legacy Windows MAX_PATH.
    building = snapshots_root / f".b-{uuid.uuid4().hex[:8]}"
    building_files = building / "files"
    building_files.mkdir(parents=True, exist_ok=False)
    for item in bindings:
        source = Path(item.original_path)
        target = building_files / item.source_name
        with source.open("rb") as input_handle, target.open("xb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if target.stat().st_size != item.size_bytes or _sha256_file(target) != item.sha256:
            raise RuntimeError(f"copied D:\\00 source snapshot failed exact verification: {source}")
    with (building / "manifest.json").open("xb") as handle:
        handle.write(canonical_json_bytes(expected_manifest) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(building, final_root)
    return snapshot_id, bindings


def _open_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _record(
    *,
    kind: str,
    binding: RecoveredSourceSnapshot,
    pointer: str,
    sequence: int,
    exact_text: str,
    payload: Mapping[str, Any],
    occurred_at: str = "",
) -> ExperienceRecord:
    return ExperienceRecord(
        record_kind=kind,
        source_name=binding.source_name,
        source_sha256=binding.sha256,
        source_pointer=pointer,
        sequence=sequence,
        exact_text=exact_text,
        payload=dict(payload),
        occurred_at=occurred_at,
    )


def _rows(connection: sqlite3.Connection, table: str, columns: Sequence[str]) -> Iterator[sqlite3.Row]:
    quoted = ", ".join(f'"{column}"' for column in columns)
    yield from connection.execute(f'SELECT {quoted} FROM "{table}" ORDER BY rowid')


def _message_records(path: Path, binding: RecoveredSourceSnapshot) -> Iterator[ExperienceRecord]:
    connection = _open_readonly(path)
    try:
        columns = ("id", "role", "content", "timestamp", "metadata")
        for sequence, row in enumerate(_rows(connection, "messages", columns)):
            payload = {column: row[column] for column in columns}
            yield _record(
                kind="message",
                binding=binding,
                pointer=f"sqlite:{binding.source_name}:messages:{row['id']}",
                sequence=sequence,
                exact_text=str(row["content"] or ""),
                payload=payload,
                occurred_at=str(row["timestamp"] or ""),
            )
    finally:
        connection.close()


def _mission_records(path: Path, binding: RecoveredSourceSnapshot) -> Iterator[ExperienceRecord]:
    connection = _open_readonly(path)
    try:
        mission_columns = ("id", "goal", "status", "created_at", "completed_at")
        for sequence, row in enumerate(_rows(connection, "missions", mission_columns)):
            payload = {column: row[column] for column in mission_columns}
            yield _record(
                kind="mission",
                binding=binding,
                pointer=f"sqlite:{binding.source_name}:missions:{row['id']}",
                sequence=sequence,
                exact_text=str(row["goal"] or ""),
                payload=payload,
                occurred_at=str(row["created_at"] or ""),
            )
        objective_columns = (
            "id", "mission_id", "parent_id", "description", "status", "reason", "order_index"
        )
        for sequence, row in enumerate(_rows(connection, "objectives", objective_columns)):
            payload = {column: row[column] for column in objective_columns}
            yield _record(
                kind="objective",
                binding=binding,
                pointer=f"sqlite:{binding.source_name}:objectives:{row['id']}",
                sequence=sequence,
                exact_text=str(row["description"] or ""),
                payload=payload,
            )
    finally:
        connection.close()


def _episode_records(path: Path, binding: RecoveredSourceSnapshot) -> Iterator[ExperienceRecord]:
    connection = _open_readonly(path)
    columns = (
        "id", "episode_type", "turns_json", "summary", "extracted_json", "source",
        "created_at", "payload_json",
    )
    try:
        for sequence, row in enumerate(_rows(connection, "episodes", columns)):
            payload = {column: row[column] for column in columns}
            exact_text = str(row["summary"] or row["turns_json"] or row["payload_json"] or "")
            yield _record(
                kind="episode",
                binding=binding,
                pointer=f"sqlite:{binding.source_name}:episodes:{row['id']}",
                sequence=sequence,
                exact_text=exact_text,
                payload=payload,
                occurred_at=str(row["created_at"] or ""),
            )
    finally:
        connection.close()


def _backlog_records(path: Path, binding: RecoveredSourceSnapshot) -> Iterator[ExperienceRecord]:
    connection = _open_readonly(path)
    columns = (
        "id", "operation", "payload_json", "budget", "status", "attempts", "last_error",
        "created_at", "updated_at",
    )
    try:
        for sequence, row in enumerate(_rows(connection, "backlog_jobs", columns)):
            payload = {column: row[column] for column in columns}
            exact_text = str(row["operation"] or row["payload_json"] or row["last_error"] or "")
            yield _record(
                kind="backlog_job",
                binding=binding,
                pointer=f"sqlite:{binding.source_name}:backlog_jobs:{row['id']}",
                sequence=sequence,
                exact_text=exact_text,
                payload=payload,
                occurred_at=str(row["created_at"] or ""),
            )
    finally:
        connection.close()


def _personal_log_records(path: Path, binding: RecoveredSourceSnapshot) -> Iterator[ExperienceRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", []) if isinstance(data, Mapping) else []
    for sequence, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            continue
        yield _record(
            kind="diary",
            binding=binding,
            pointer=f"json:{binding.source_name}:entries:{sequence}",
            sequence=sequence,
            exact_text=str(entry.get("content") or ""),
            payload=dict(entry),
            occurred_at=str(entry.get("timestamp") or ""),
        )


def _runtime_json_records(path: Path, binding: RecoveredSourceSnapshot) -> Iterator[ExperienceRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    conversations = data.get("conversations", []) if isinstance(data, Mapping) else []
    for sequence, entry in enumerate(conversations):
        if not isinstance(entry, Mapping):
            continue
        yield _record(
            kind="runtime_conversation_message",
            binding=binding,
            pointer=f"json:{binding.source_name}:conversations:{sequence}",
            sequence=sequence,
            exact_text=str(entry.get("content") or ""),
            payload=dict(entry),
        )


def collect_d00_experience_records(
    source_root: Path,
    bindings: Sequence[RecoveredSourceSnapshot],
) -> tuple[ExperienceRecord, ...]:
    by_name = {item.source_name: item for item in bindings}
    records: list[ExperienceRecord] = []
    for name in ("axon_memory.db", "axon_runtime_state.db"):
        records.extend(_message_records(source_root / name, by_name[name]))
        records.extend(_mission_records(source_root / name, by_name[name]))
    records.extend(
        _episode_records(source_root / "axon_episodic_memory.db", by_name["axon_episodic_memory.db"])
    )
    records.extend(
        _backlog_records(source_root / "axon_memory_backlog.db", by_name["axon_memory_backlog.db"])
    )
    records.extend(
        _personal_log_records(source_root / "axon_personal_log.json", by_name["axon_personal_log.json"])
    )
    records.extend(
        _runtime_json_records(source_root / "axon_runtime_state.json", by_name["axon_runtime_state.json"])
    )
    return tuple(records)


def import_d00_memories(source_root: Path, state_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    state_root = state_root.resolve()
    inventory = _source_inventory(source_root)
    snapshot_id, bindings = snapshot_d00_sources(
        source_root, state_root, inventory=inventory
    )
    records = collect_d00_experience_records(source_root, bindings)
    manifest = DormantExperienceStore(state_root).publish_import(
        records,
        sources=bindings,
        label=D00_IMPORT_LABEL,
    )
    return {
        "schema": "axon-d00-memory-import-result-v1",
        "snapshot_id": snapshot_id,
        "import_id": manifest.import_id,
        "record_count": manifest.record_count,
        "record_kind_counts": [list(item) for item in manifest.record_kind_counts],
        "source_count": len(bindings),
        "source_bytes": sum(item.size_bytes for item in bindings),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\00"))
    parser.add_argument("--state-root", type=Path, default=Path(r"D:\Axon\State"))
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="hash and report protected sources without writing Dormant State",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.inventory_only:
        inventory = _source_inventory(args.source_root.resolve())
        print(
            json.dumps(
                {
                    "schema": D00_SNAPSHOT_SCHEMA,
                    "snapshot_id": canonical_sha256(
                        {"schema": D00_SNAPSHOT_SCHEMA, "sources": [dict(item) for item in inventory]}
                    ),
                    "source_count": len(inventory),
                    "source_bytes": sum(int(item["size_bytes"]) for item in inventory),
                    "sources": list(inventory),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    print(json.dumps(import_d00_memories(args.source_root, args.state_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "D00_SNAPSHOT_SCHEMA",
    "D00_IMPORT_LABEL",
    "SOURCE_NAMES",
    "snapshot_d00_sources",
    "collect_d00_experience_records",
    "import_d00_memories",
]
