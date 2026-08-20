from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from training.build_multitick_curriculum import (
    CurriculumBuildError,
    MixedPrivacySourceError,
    SourceRecord,
    SourceSpec,
    build_curriculum,
    build_curriculum_from_records,
    build_episode,
    main as build_multitick_main,
)
from training.d00_field_sources import (
    ADAPTER,
    EPISODIC_DATABASE,
    EXCLUDED_OLD_MEMORY_SOURCES,
    FAMILY_SCRATCH,
    FAMILY_STRUCTURED,
    PERSONAL_LOG,
    SEMANTIC_DATABASE,
    audit_actual_stores,
    deterministic_family_quotas,
    extract_d00_field_source_records,
    main as audit_main,
    normalize_for_query_leak_check,
    preflight_user_input,
    query_contains_normalized_answer,
    tagged_user_payload_capacity,
)


def _create_semantic_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE extracted_facts (
                id INTEGER PRIMARY KEY,
                key TEXT,
                value TEXT,
                source TEXT,
                created_at TEXT
            );
            CREATE TABLE extracted_relations (
                id INTEGER PRIMARY KEY,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                confidence REAL,
                source TEXT,
                created_at TEXT
            );
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
            """
        )
        for index in range(12):
            connection.execute(
                """
                INSERT INTO extracted_facts
                    (id, key, value, source, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    index + 1,
                    f"fact key {index}",
                    f"fact answer {index}",
                    "memory_agent",
                    f"2026-01-{index + 1:02d}",
                ),
            )
            connection.execute(
                """
                INSERT INTO extracted_relations
                    (id, subject, predicate, object, confidence, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    index + 1,
                    f"subject {index}",
                    "connects to",
                    f"relation answer {index}",
                    1.0,
                    "memory_agent",
                    f"2026-02-{index + 1:02d}",
                ),
            )
        for index in range(4):
            connection.execute(
                """
                INSERT INTO extracted_procedures
                    (
                        id, name, trigger, steps_json, outcome, applicability,
                        source, created_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    index + 1,
                    f"procedure {index}",
                    f"trigger {index}",
                    json.dumps([f"exact step {index}", f"later step {index}"]),
                    f"exact outcome {index}",
                    f"case {index}",
                    "memory_agent",
                    f"2026-03-{index + 1:02d}",
                ),
            )

        # Three exact-grounding failures must be counted and never selected.
        connection.execute(
            """
            INSERT INTO extracted_facts
                (id, key, value, source, created_at)
            VALUES (100, 'ungrounded key', 'ungrounded answer', 'memory_agent', 'x')
            """
        )
        connection.execute(
            """
            INSERT INTO extracted_relations
                (id, subject, predicate, object, confidence, source, created_at)
            VALUES (
                100, 'ungrounded subject', 'misses', 'ungrounded object',
                1.0, 'memory_agent', 'x'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO extracted_procedures
                (
                    id, name, trigger, steps_json, outcome, applicability,
                    source, created_at
                )
            VALUES (
                100, 'ungrounded procedure', 'ungrounded trigger',
                '["ungrounded step"]', 'ungrounded outcome', 'ungrounded case',
                'memory_agent', 'x'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO extracted_facts
                (id, key, value, source, created_at)
            VALUES (
                200, 'residual key', 'residual answer',
                'imported from axon_memory.db', 'x'
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def _create_episodic_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
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
            """
        )
        episode_id = 1
        for index in range(12):
            extracted = {
                "facts": [
                    {"key": f"fact key {index}", "value": f"fact answer {index}"}
                ],
                "relations": [],
                "procedures": [],
            }
            connection.execute(
                """
                INSERT INTO episodes
                    (
                        id, episode_type, summary, extracted_json, source,
                        created_at, payload_json
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    "tool_result",
                    f"Grounded fact episode {index}.",
                    json.dumps(extracted),
                    "memory_agent",
                    f"2026-01-{index + 1:02d}",
                    "{}",
                ),
            )
            episode_id += 1
        for index in range(12):
            extracted = {
                "facts": [],
                "relations": [
                    {
                        "subject": f"subject {index}",
                        "predicate": "connects to",
                        "object": f"relation answer {index}",
                        "confidence": 1.0,
                    }
                ],
                "procedures": [],
            }
            connection.execute(
                """
                INSERT INTO episodes
                    (
                        id, episode_type, summary, extracted_json, source,
                        created_at, payload_json
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    "tool_result",
                    f"Grounded relation episode {index}.",
                    json.dumps(extracted),
                    "memory_agent",
                    f"2026-02-{index + 1:02d}",
                    "{}",
                ),
            )
            episode_id += 1
        for index in range(4):
            extracted = {
                "facts": [],
                "relations": [],
                "procedures": [
                    {
                        "name": f"procedure {index}",
                        "trigger": f"trigger {index}",
                        "steps": [f"exact step {index}", f"later step {index}"],
                        "outcome": f"exact outcome {index}",
                        "applicability": f"case {index}",
                    }
                ],
            }
            connection.execute(
                """
                INSERT INTO episodes
                    (
                        id, episode_type, summary, extracted_json, source,
                        created_at, payload_json
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    "tool_result",
                    f"Grounded procedure episode {index}.",
                    json.dumps(extracted),
                    "memory_agent",
                    f"2026-03-{index + 1:02d}",
                    "{}",
                ),
            )
            episode_id += 1
        connection.execute(
            """
            INSERT INTO episodes
                (
                    id, episode_type, summary, extracted_json, source,
                    created_at, payload_json
                )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                "tool_result",
                "Grounded but excluded residual episode.",
                json.dumps(
                    {
                        "facts": [
                            {
                                "key": "residual key",
                                "value": "residual answer",
                            }
                        ],
                        "relations": [],
                        "procedures": [],
                    }
                ),
                "memory_agent",
                "2026-04-01",
                "{}",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _fixture_root(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _create_semantic_database(tmp_path / SEMANTIC_DATABASE)
    _create_episodic_database(tmp_path / EPISODIC_DATABASE)
    for name in EXCLUDED_OLD_MEMORY_SOURCES:
        (tmp_path / name).write_bytes(f"excluded {name}".encode("ascii"))
    (tmp_path / PERSONAL_LOG).write_text(
        "TOP SECRET MANUAL ONLY SENTINEL",
        encoding="utf-8",
    )
    return tmp_path


def test_family_quota_is_exact_deterministic_45_45_10() -> None:
    assert deterministic_family_quotas(8192) == {
        "fact": 3687,
        "relation": 3686,
        "procedure": 819,
    }
    assert deterministic_family_quotas(20) == {
        "fact": 9,
        "relation": 9,
        "procedure": 2,
    }
    assert tagged_user_payload_capacity() == 51
    assert preflight_user_input("x" * 51)["fully_visible"] is True
    assert preflight_user_input("x" * 52)["fully_visible"] is False


def test_exact_grounded_extraction_is_bounded_deterministic_and_read_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _fixture_root(tmp_path)
    source_bytes = {
        name: (root / name).read_bytes()
        for name in (SEMANTIC_DATABASE, EPISODIC_DATABASE)
    }
    forbidden_paths = {
        (root / name).resolve()
        for name in (*EXCLUDED_OLD_MEMORY_SOURCES, PERSONAL_LOG)
    }
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.resolve() in forbidden_paths:
            raise AssertionError(f"excluded source was opened: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    first = extract_d00_field_source_records(root, total_limit=20)
    second = extract_d00_field_source_records(root, total_limit=20)

    assert first == second
    assert first.audit["passed"] is True
    assert first.audit["selected_rows"] == 20
    assert first.audit["selected_by_kind"] == {
        "fact": 9,
        "relation": 9,
        "procedure": 2,
    }
    assert first.audit["selected_by_family"] == {
        FAMILY_STRUCTURED: 18,
        FAMILY_SCRATCH: 2,
    }
    assert (
        first.audit["quarantine_counts"]["ungrounded_semantic_candidates"]
        == 3
    )
    assert (
        first.audit["quarantine_counts"][
            "quarantined_old_memory_residual_candidate"
        ]
        == 1
    )
    assert (
        first.audit["quarantine_counts"][
            "old_memory_residual_sources_detected"
        ]
        == 3
    )
    assert (
        first.audit["quarantine_counts"][
            "personal_log_manual_only_sources_detected"
        ]
        == 1
    )
    assert first.audit["quarantine_counts"]["personal_log_rows_read"] == 0
    assert first.audit["quarantine_counts"]["personal_log_rows_included"] == 0
    assert all(source["read_only_verified"] for source in first.audit["sources"])
    assert {
        name: (root / name).read_bytes()
        for name in (SEMANTIC_DATABASE, EPISODIC_DATABASE)
    } == source_bytes

    serialized = json.dumps(first.source_record_kwargs, sort_keys=True)
    assert "TOP SECRET MANUAL ONLY SENTINEL" not in serialized
    for payload in first.source_record_kwargs:
        assert set(payload) == {
            "row",
            "pointer",
            "source_path",
            "source_sha256",
            "row_sha256",
            "adapter",
        }
        assert payload["adapter"] == ADAPTER
        row = payload["row"]
        assert row["local_only"] is True
        assert row["cloud_export_allowed"] is False
        assert row["privacy"]["personal_log_included"] is False
        assert row["privacy"]["old_memory_residual_included"] is False
        assert row["provenance"]["exact_grounding"]["matched"] is True
        assert "source" not in row["provenance"]["semantic"]
        assert "source" not in row["provenance"]["episodic"]
        assert row["provenance"]["semantic"]["source_metadata"]["sha256"]
        assert row["provenance"]["episodic"]["source_metadata"]["sha256"]
        assert row["structured_knowledge"]
        assert row["conversation_history"]
        assert row["user_input"]
        assert row["situation_awareness"]
        assert row["task_state"]
        targets = [
            row[key]
            for key in ("scratch_plan", "response_after")
            if key in row
        ]
        assert targets
        assert all(1 <= len(target) <= 256 for target in targets)
        assert not query_contains_normalized_answer(row["user_input"], targets)
        assert len(row["user_input"]) <= tagged_user_payload_capacity()
        assert row["active_view_preflight"]["user_input"]["fully_visible"]
        assert row["active_view_preflight"]["paging_required"] is False


def test_source_record_kwargs_build_valid_exact_v4_episodes(tmp_path: Path) -> None:
    extraction = extract_d00_field_source_records(
        _fixture_root(tmp_path),
        total_limit=20,
    )
    families = set()
    for payload in extraction.source_record_kwargs:
        record = SourceRecord(**payload)
        episode, reason = build_episode(
            record,
            unsupported_policy="reject",
            include_diary=False,
        )
        assert reason is None
        assert episode is not None
        families.add(episode["family"])
        assert episode["initial_field"]["structured_knowledge"]["text"]
        assert episode["initial_field"]["conversation_history"]["text"]
        assert episode["initial_field"]["situation_awareness"]["text"]
        assert episode["initial_field"]["task_state"]["text"]
        assert episode["initial_field"]["user_input"]["text"]
        assert all(
            len(tick["complete_proposed_region_text"]) <= 256
            for tick in episode["ticks"]
        )
    assert families == {FAMILY_STRUCTURED, FAMILY_SCRATCH}


def test_audit_cli_emits_only_aggregate_contract(
    tmp_path: Path,
    capsys,
) -> None:
    root = _fixture_root(tmp_path)
    assert audit_main(
        ["--source-root", str(root), "--total-limit", "20"]
    ) == 0
    output = capsys.readouterr().out
    audit = json.loads(output)
    assert audit == audit_actual_stores(root, total_limit=20)
    assert audit["passed"] is True
    assert "fact answer" not in output
    assert "relation answer" not in output
    assert "exact outcome" not in output
    assert "TOP SECRET" not in output


def test_query_leak_normalization_is_detection_only() -> None:
    assert normalize_for_query_leak_check("A_B-C") == "a b c"
    assert query_contains_normalized_answer(
        "Recall ALPHA-beta now.",
        ["alpha beta"],
    )
    assert not query_contains_normalized_answer(
        "Use grounded evidence.",
        ["the private answer"],
    )


def test_default_multitick_cli_uses_direct_grounded_private_records(
    tmp_path: Path,
    capsys,
) -> None:
    source_root = _fixture_root(tmp_path / "sources")
    output = tmp_path / "artifact"
    assert build_multitick_main(
        [
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output),
            "--grounded-d00-limit",
            "20",
            "--seed",
            "17",
        ]
    ) == 0
    assert "built 20 episodes" in capsys.readouterr().out

    manifest = json.loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    episodes = [
        json.loads(line)
        for line in (output / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert manifest["source_mode"] == "grounded_d00_direct_records"
    assert manifest["episodes"] == 20
    assert manifest["grounded_d00_adapter_audit"]["passed"] is True
    assert manifest["grounded_d00_adapter_audit"]["selected_by_kind"] == {
        "fact": 9,
        "relation": 9,
        "procedure": 2,
    }
    assert (
        manifest["grounded_d00_adapter_audit"]["record_contract"][
            "user_input_preflight_count"
        ]
        == 20
    )
    assert manifest["privacy"] == {
        "any_rows_local_only": True,
        "all_rows_local_only": True,
        "cloud_export_allowed": False,
        "mixed_private_and_exportable_rows": False,
    }
    assert manifest["diary"]["cloud_export_allowed"] is False
    assert manifest["writable_readback_gate_passed"] is False
    assert manifest["read_coverage_launch_gate"]["passed"] is False
    assert (
        manifest["read_coverage_launch_gate"]["artifact_launch_eligible"]
        is False
    )
    assert {source["role"] for source in manifest["sources"]} == {
        "semantic",
        "episodic_grounding",
    }
    for source in manifest["sources"]:
        assert source["read_only_verified"] is True
        assert source["sha256_before"] == source["sha256_after"]
    assert len(episodes) == 20
    assert all(episode["local_only"] is True for episode in episodes)
    assert all(
        episode["privacy"]["source_declared_local_only"] is True
        and episode["privacy"]["source_declared_cloud_export_allowed"] is False
        and episode["privacy"]["cloud_export_allowed"] is False
        for episode in episodes
    )
    assert all(
        episode["read_coverage_contract"][
            "writable_region_post_commit_readback_guaranteed"
        ]
        is False
        for episode in episodes
    )


def test_direct_grounded_path_rejects_failed_audit_and_source_mixing(
    tmp_path: Path,
) -> None:
    source_root = _fixture_root(tmp_path / "sources")
    extraction = extract_d00_field_source_records(
        source_root,
        total_limit=20,
    )
    records = tuple(
        SourceRecord(**kwargs)
        for kwargs in extraction.source_record_kwargs
    )
    failed_audit = json.loads(json.dumps(extraction.audit))
    failed_audit["passed"] = False
    with pytest.raises(CurriculumBuildError):
        build_curriculum_from_records(
            records,
            tmp_path / "failed",
            source_audit=failed_audit,
        )

    generic = tmp_path / "generic.jsonl"
    generic.write_text(
        json.dumps(
            {
                "family": "no_op_v4",
                "lineage_id": "generic",
                "initial_draft": "generic draft",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(MixedPrivacySourceError):
        build_curriculum(
            [SourceSpec.jsonl(generic)],
            tmp_path / "mixed",
            direct_records=records,
            direct_source_audit=extraction.audit,
        )
