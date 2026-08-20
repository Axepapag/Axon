from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest

from training.build_soul_write_delay_curriculum import (
    build_soul_write_delay_curriculum,
    discover_soul_sources,
)
from training.d00_soul_sources import (
    D00_EPISODIC_DATABASE,
    D00_PERSONAL_LOG,
    D00_PRIVATE_ADAPTER,
    D00_SEMANTIC_DATABASE,
    DEFAULT_PRODUCTION_SAMPLE_CAP,
    D00GroundingError,
    allocate_family_counts,
    audit_compiled_field_capacity,
    discover_d00_private_soul_sources,
    is_d00_private_adapter,
    runtime_field_payload_capacities,
    select_grounded_records,
    structured_memory_query,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _semantic_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "create table extracted_facts ("
            "id integer primary key, key text, value text, source text, "
            "created_at text)"
        )
        connection.execute(
            "create table extracted_relations ("
            "id integer primary key, subject text, predicate text, "
            "object text, source text, created_at text)"
        )
        connection.execute(
            "create table extracted_procedures ("
            "id integer primary key, name text, steps_json text, "
            "source text, created_at text)"
        )
        facts = [
            (
                index + 1,
                f"setting {index}",
                f"value {index}x",
                "semantic",
                "2026-01-01",
            )
            for index in range(9)
        ]
        facts.extend(
            [
                (20, "ghost setting", "absent value", "semantic", "2026-01-01"),
                (
                    21,
                    "value leak",
                    "value leak",
                    "semantic",
                    "2026-01-01",
                ),
                (
                    22,
                    "api token",
                    "secret value",
                    "semantic",
                    "2026-01-01",
                ),
                (
                    23,
                    "setting 0",
                    "value 0x",
                    "duplicate",
                    "2026-01-02",
                ),
                (
                    24,
                    "k" * 30,
                    "separate answer",
                    "semantic",
                    "2026-01-01",
                ),
            ]
        )
        connection.executemany(
            "insert into extracted_facts values (?, ?, ?, ?, ?)",
            facts,
        )
        relations = [
            (
                index + 1,
                f"System {index}",
                "uses",
                f"Module {index}x",
                "semantic",
                "2026-01-01",
            )
            for index in range(9)
        ]
        relations.append(
            (
                20,
                "Ghost system",
                "uses",
                "Absent module",
                "semantic",
                "2026-01-01",
            )
        )
        connection.executemany(
            "insert into extracted_relations values (?, ?, ?, ?, ?, ?)",
            relations,
        )
        connection.executemany(
            "insert into extracted_procedures values (?, ?, ?, ?, ?)",
            [
                (
                    1,
                    "Workflow Alpha",
                    json.dumps(["Open panel", "Choose item"]),
                    "semantic",
                    "2026-01-01",
                ),
                (
                    2,
                    "Workflow Beta",
                    json.dumps(["Check status", "Record result"]),
                    "semantic",
                    "2026-01-01",
                ),
                (
                    3,
                    "Ghost workflow",
                    json.dumps(["Missing step"]),
                    "semantic",
                    "2026-01-01",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _episodic_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "create table episodes ("
            "id integer primary key, extracted_json text)"
        )
        payload = {
            "facts": [
                {"key": f"setting {index}", "value": f"value {index}x"}
                for index in range(9)
            ]
            + [
                {"key": "value leak", "value": "value leak"},
                {"key": "api token", "value": "secret value"},
                {"key": "k" * 30, "value": "separate answer"},
            ],
            "relations": [
                {
                    "subject": f"System {index}",
                    "predicate": "uses",
                    "object": f"Module {index}x",
                }
                for index in range(9)
            ],
            "procedures": [
                {"name": "Workflow Alpha", "steps": ["Open panel", "Choose item"]},
                {"name": "Workflow Beta", "steps": ["Check status", "Record result"]},
            ],
        }
        connection.execute(
            "insert into episodes values (?, ?)",
            (7, json.dumps(payload, sort_keys=True)),
        )
        connection.execute(
            "insert into episodes values (?, ?)",
            (8, "{malformed"),
        )
        connection.commit()
    finally:
        connection.close()


def _residual_database(path: Path, *, backlog: bool = False) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "create table extracted_facts ("
            "id integer primary key, key text, value text)"
        )
        connection.execute(
            "insert into extracted_facts values (1, 'old key', 'old value')"
        )
        if not backlog:
            connection.execute(
                "create table extracted_relations ("
                "id integer primary key, subject text, predicate text, object text)"
            )
            connection.execute(
                "insert into extracted_relations "
                "values (1, 'old', 'relates to', 'residual')"
            )
            connection.execute(
                "create table messages (id text primary key, content text)"
            )
            connection.execute(
                "insert into messages values ('m1', 'private raw message')"
            )
        connection.commit()
    finally:
        connection.close()


def _source_root(root: Path) -> list[Path]:
    semantic = root / D00_SEMANTIC_DATABASE
    episodic = root / D00_EPISODIC_DATABASE
    old = root / "axon_memory.db"
    backlog = root / "axon_memory_backlog.db"
    personal = root / D00_PERSONAL_LOG
    _semantic_database(semantic)
    _episodic_database(episodic)
    _residual_database(old)
    _residual_database(backlog, backlog=True)
    personal.write_text(
        json.dumps(
            {
                "entry_count": 2,
                "entries": [
                    {"text": "manual private note one"},
                    {"text": "manual private note two"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return [semantic, episodic, old, backlog, personal]


def test_grounded_adapter_is_bounded_balanced_private_and_read_only(
    tmp_path: Path,
) -> None:
    source_paths = _source_root(tmp_path)
    before = {
        path: (_sha256(path), path.stat().st_mtime_ns) for path in source_paths
    }

    sources = discover_d00_private_soul_sources(tmp_path, limit_per_kind=20)
    assert len(sources) == 1
    assert sources[0].adapter == D00_PRIVATE_ADAPTER
    selection = select_grounded_records(sources[0], seed=7)

    assert len(selection.records) == 20
    assert allocate_family_counts(20) == {
        "fact": 9,
        "relation": 9,
        "procedure": 2,
    }
    kinds = [
        str(record.row["memory_kind"]) for record in selection.records
    ]
    assert {kind: kinds.count(kind) for kind in set(kinds)} == {
        "fact": 9,
        "relation": 9,
        "procedure": 2,
    }
    for record in selection.records:
        row = record.row
        answer_component = str(row["answer_bearing_texts"][1])
        assert row["exact_episodic_grounding"] is True
        assert _normalized(answer_component) not in _normalized(
            str(row["recall_query"])
        )
        assert row["local_only"] is True
        assert row["cloud_export_allowed"] is False
        assert record.source_path.endswith(D00_SEMANTIC_DATABASE)

    audit = selection.audit
    assert audit["canonical_target_store"] == D00_SEMANTIC_DATABASE
    assert audit["grounding"]["passed"] is True
    assert audit["grounding"]["semantic_rows_missing_exact_grounding"] == 3
    assert (
        audit["family_counts"]["fact"][
            "quarantined_compiled_field_capacity"
        ]
        == 1
    )
    assert audit["compiled_field_capacity"]["passed"] is True
    assert (
        audit["compiled_field_capacity"][
            "emitted_rows_exact_compiled_view_passed"
        ]
        == 20
    )
    assert audit["quarantine_counts"]["old_memory_all_rows"] == 4
    assert audit["quarantine_counts"]["old_memory_structured_residuals"] == 3
    assert audit["quarantine_counts"]["personal_log_manual_only_entries"] == 2
    assert audit["personal_log"]["emitted_rows"] == 0
    assert audit["privacy"]["all_emitted_rows_local_only"] is True
    assert len(audit["source_hashes"]) == len(source_paths)
    assert all(item["read_only_verified"] for item in selection.source_manifest)

    for path in source_paths:
        assert (_sha256(path), path.stat().st_mtime_ns) == before[path]


def test_builder_manifest_carries_grounding_hash_quarantine_and_privacy_audits(
    tmp_path: Path,
) -> None:
    source_paths = _source_root(tmp_path)
    sources = discover_d00_private_soul_sources(tmp_path, limit_per_kind=20)
    assert discover_soul_sources(tmp_path) != sources
    # Default discovery carries the production cap; use the explicitly bounded
    # test spec so this fixture need not fabricate 8192 private records.
    escaped_output = tmp_path / "escaped-derived"
    with pytest.raises(D00GroundingError, match="requires.*reject"):
        build_soul_write_delay_curriculum(
            sources,
            escaped_output,
            seed=7,
            unsupported_policy="escape",
        )
    assert not escaped_output.exists()

    manifest = build_soul_write_delay_curriculum(
        sources,
        tmp_path / "derived",
        seed=7,
    )

    audit = manifest["d00_adapter_audit"]
    assert manifest["episodes"] == 20
    assert audit["requested_sample_cap"] == 20
    assert audit["passed"] is True
    assert audit["family_counts"]["fact"]["emitted"] == 9
    assert audit["family_counts"]["relation"]["emitted"] == 9
    assert audit["family_counts"]["procedure"]["emitted"] == 2
    assert all(
        item["sha256_before"] == item["sha256_after"]
        for item in manifest["sources"]
    )
    assert {
        item["path"] for item in manifest["sources"]
    } == {str(path.resolve()) for path in source_paths}
    assert manifest["source_privacy"]["private_d00_sources_present"] is True
    assert manifest["source_privacy"]["cloud_export_allowed"] is False

    episodes = [
        json.loads(line)
        for line in (tmp_path / "derived" / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(episodes) == 20
    assert all(episode["privacy"]["local_only"] for episode in episodes)
    assert all(
        not episode["privacy"]["cloud_export_allowed"]
        for episode in episodes
    )
    assert all(
        episode["provenance"]["d00_grounding"][
            "exact_episodic_grounding"
        ]
        is True
        for episode in episodes
    )
    derived_text = "\n".join(
        path.read_text(
            encoding="ascii" if path.suffix == ".sha256" else "utf-8"
        )
        for path in (tmp_path / "derived").iterdir()
        if path.is_file()
    )
    assert "private raw message" not in derived_text
    assert "manual private note" not in derived_text


def test_discovery_requires_semantic_and_episodic_and_quota_fails_closed(
    tmp_path: Path,
) -> None:
    _semantic_database(tmp_path / D00_SEMANTIC_DATABASE)
    assert discover_d00_private_soul_sources(tmp_path) == []
    _episodic_database(tmp_path / D00_EPISODIC_DATABASE)
    sources = discover_d00_private_soul_sources(tmp_path, limit_per_kind=21)
    with pytest.raises(D00GroundingError, match="requires"):
        select_grounded_records(sources[0], seed=1)


def test_cap_marker_and_private_adapter_validation() -> None:
    query = structured_memory_query(limit_per_kind=20)
    assert "WHERE 0" in query
    assert "d00_sample_cap=20" in query
    assert allocate_family_counts(DEFAULT_PRODUCTION_SAMPLE_CAP) == {
        "fact": 3687,
        "relation": 3686,
        "procedure": 819,
    }
    assert is_d00_private_adapter("d00_private_anything")
    assert not is_d00_private_adapter("exact_v4")
    with pytest.raises(ValueError, match="positive"):
        structured_memory_query(limit_per_kind=0)
    with pytest.raises(ValueError, match="production cap"):
        structured_memory_query(
            limit_per_kind=DEFAULT_PRODUCTION_SAMPLE_CAP + 1
        )


def test_tag_aware_capacities_come_from_exact_runtime_compilation() -> None:
    capacities = runtime_field_payload_capacities()
    assert capacities == {
        "recall_query_user_input": 51,
        "target_response_draft": 47,
        "tick_a_structured_knowledge": 118,
    }

    audit = audit_compiled_field_capacity(
        target="T" * 48,
        evidence="E" * 60,
        recall_query="Q" * 57,
    )
    assert audit["passed"] is False
    assert audit["checks"]["recall_query_user_input"]["required_chars"] == 57
    assert audit["checks"]["recall_query_user_input"]["visible_chars"] == 51
    assert audit["checks"]["target_response_draft"]["visible_chars"] == 47
    assert (
        audit["checks"]["tick_a_structured_knowledge"]["visible_chars"]
        == 118
    )
    assert {
        omission["reason"]
        for check in audit["checks"].values()
        for omission in check["omissions"]
    } == {"user_capacity", "proposal_capacity", "context_capacity"}
