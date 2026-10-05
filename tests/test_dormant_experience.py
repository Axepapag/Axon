from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from curator.import_d00_memories import (
    SOURCE_NAMES,
    collect_d00_experience_records,
    snapshot_d00_sources,
)
from runtime.dormant import DormantExperienceError, DormantExperienceStore


def _write_db(path: Path, statements: list[str]) -> None:
    connection = sqlite3.connect(path)
    try:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def _sources(root: Path) -> None:
    _write_db(
        root / "axon_memory.db",
        [
            "create table messages(id text, role text, content text, timestamp text, metadata text)",
            "insert into messages values ('m1','user','Exact!?\nLine 2','2026-01-01','{\"x\":1}')",
            "insert into messages values ('m2','assistant','Answer — exact','2026-01-02','{}')",
            "create table missions(id text, goal text, status text, created_at text, completed_at text)",
            "insert into missions values ('mission','Preserve all.','done','2026-01-01','2026-01-02')",
            "create table objectives(id text, mission_id text, parent_id text, description text, status text, reason text, order_index integer)",
            "insert into objectives values ('o1','mission',null,'Keep punctuation: yes!','done','evidence',0)",
        ],
    )
    _write_db(
        root / "axon_runtime_state.db",
        [
            "create table messages(id text, role text, content text, timestamp text, metadata text)",
            "insert into messages values ('r1','user','Runtime exact','2026-02-01','{}')",
            "create table missions(id text, goal text, status text, created_at text, completed_at text)",
            "insert into missions values ('rm','Runtime goal','started','2026-02-01',null)",
            "create table objectives(id text, mission_id text, parent_id text, description text, status text, reason text, order_index integer)",
            "insert into objectives values ('ro','rm',null,'Runtime objective','pending',null,0)",
        ],
    )
    _write_db(
        root / "axon_episodic_memory.db",
        [
            "create table episodes(id integer, episode_type text, turns_json text, summary text, extracted_json text, source text, created_at text, payload_json text)",
            "insert into episodes values (1,'turn',null,'Episode: exact!','{}','memory','2026-03-01','{\"raw\":\"yes\"}')",
        ],
    )
    _write_db(
        root / "axon_memory_backlog.db",
        [
            "create table backlog_jobs(id integer, operation text, payload_json text, budget integer, status text, attempts integer, last_error text, created_at text, updated_at text)",
            "insert into backlog_jobs values (1,'learn','{\"topic\":\"heart\"}',10,'done',1,null,'2026-03-01','2026-03-02')",
        ],
    )
    _write_db(root / "axon_semantic_memory.db", ["create table facts(id integer, value text)"])
    (root / "axon_personal_log.json").write_text(
        json.dumps({"entries": [{"timestamp": "2026-03-03", "type": "diary", "content": "I remember?!\nExactly."}]}),
        encoding="utf-8",
    )
    (root / "axon_runtime_state.json").write_text(
        json.dumps({"conversations": [{"role": "user", "content": "JSON exact…"}]}),
        encoding="utf-8",
    )


def test_d00_snapshot_and_experience_import_are_exact_and_idempotent(tmp_path: Path) -> None:
    source_root = tmp_path / "00"
    state_root = tmp_path / "Axon" / "State"
    source_root.mkdir(parents=True)
    _sources(source_root)

    snapshot_id, bindings = snapshot_d00_sources(source_root, state_root)
    assert {item.source_name for item in bindings} == set(SOURCE_NAMES)
    experience_root = state_root / "dormant" / "experience_v1"
    for binding in bindings:
        original = source_root / binding.source_name
        copied = experience_root / binding.snapshot_relative_path
        assert copied.read_bytes() == original.read_bytes()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == binding.sha256

    records = collect_d00_experience_records(source_root, bindings)
    exact = next(item for item in records if item.source_pointer.endswith(":messages:m1"))
    assert exact.exact_text == "Exact!?\nLine 2"
    assert exact.payload["metadata"] == '{"x":1}'
    store = DormantExperienceStore(state_root)
    first = store.publish_import(records, sources=bindings, label="test-d00")
    second = store.publish_import(records, sources=bindings, label="test-d00")
    assert first == second
    assert first.record_count == len(records)
    assert {item.record_id for item in store.iter_records((first.import_id,))} == {
        item.record_id for item in records
    }

    source_stat = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_root.iterdir()}
    snapshot_d00_sources(source_root, state_root)
    assert source_stat == {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_root.iterdir()
    }


def test_experience_verification_fails_closed_on_record_tampering(tmp_path: Path) -> None:
    source_root = tmp_path / "00"
    state_root = tmp_path / "State"
    source_root.mkdir()
    _sources(source_root)
    _, bindings = snapshot_d00_sources(source_root, state_root)
    records = collect_d00_experience_records(source_root, bindings)
    store = DormantExperienceStore(state_root)
    manifest = store.publish_import(records, sources=bindings, label="tamper-test")
    records_path = store.imports_root / manifest.import_id / "records.jsonl"
    with records_path.open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(DormantExperienceError, match="record-file hash"):
        store.verify_import(manifest.import_id)
