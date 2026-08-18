from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING = REPO_ROOT / "training"
for path in (REPO_ROOT, TRAINING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_phase1_curriculum import ALL_FAMILIES, Phase1CurriculumBuilder


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _runtime_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute("create table messages (id integer primary key, role text, content text, timestamp text, metadata text)")
        con.execute(
            "insert into messages (role, content, timestamp, metadata) values (?, ?, ?, ?)",
            ("user", "How should I inspect the bus?", "1", "{}"),
        )
        con.execute(
            "insert into messages (role, content, timestamp, metadata) values (?, ?, ?, ?)",
            ("assistant", "Inspect the routes, then run the bus tests.", "2", "{}"),
        )
        con.commit()
    finally:
        con.close()


def _episodic_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.execute(
            "create table episodes (id integer primary key, episode_type text, turns_json text, summary text, extracted_json text, source text, created_at text, payload_json text)"
        )
        con.execute(
            "insert into episodes (episode_type, turns_json, summary, extracted_json, source, created_at, payload_json) values (?, ?, ?, ?, ?, ?, ?)",
            (
                "tool_result",
                "[]",
                "The agent fixed a WebSocket route after reading the traceback.",
                json.dumps({"lesson": "Read traceback before editing", "tool": "pytest"}),
                "test",
                "3",
                "{}",
            ),
        )
        con.commit()
    finally:
        con.close()


def _curriculum_dir(path: Path) -> None:
    _write_jsonl(
        path / "field_surfacing_v1.jsonl",
        [
            {
                "container_id": "c1",
                "query": "websocket route",
                "structured_knowledge": "edge: route uses websocket",
                "answer": "The bus route uses WebSocket.",
            }
        ],
    )
    _write_jsonl(
        path / "retrieval_qa_v1.jsonl",
        [
            {
                "container_id": "c2",
                "query": "pytest command",
                "structured_knowledge": "fact: focused tests are useful",
                "answer": "Run the focused tests first.",
            }
        ],
    )
    _write_jsonl(
        path / "procedure_next_step_v1.jsonl",
        [
            {
                "container_id": "p1",
                "procedure_name": "debug server",
                "trigger": "route failed",
                "steps_before": ["read traceback"],
                "answer": "run focused test",
            }
        ],
    )
    _write_jsonl(
        path / "episodic_exhale_filter_v1.jsonl",
        [
            {
                "container_id": "e1",
                "episode_summary": "A prior run found target leakage in phase0.",
                "facts_to_retain": ["Do not put the supervised answer in active response_draft."],
                "structured_knowledge": "lesson: avoid target leakage",
            }
        ],
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_phase1_builder_writes_real_memory_families(tmp_path: Path) -> None:
    runtime_json = tmp_path / "axon_runtime_state.json"
    runtime_json.write_text(
        json.dumps(
            {
                "conversations": [
                    {"role": "user", "content": "Can you inspect the repo?"},
                    {"role": "assistant", "content": "I will inspect files and run focused tests."},
                ]
            }
        ),
        encoding="utf-8",
    )
    runtime_db = tmp_path / "axon_runtime_state.db"
    old_db = tmp_path / "axon_memory.db"
    episodic_db = tmp_path / "axon_episodic_memory.db"
    _runtime_db(runtime_db)
    _runtime_db(old_db)
    _episodic_db(episodic_db)
    curriculum = tmp_path / "curriculum_v1"
    _curriculum_dir(curriculum)
    personal_log = tmp_path / "axon_personal_log.json"
    personal_log.write_text(json.dumps([{"content": "Private continuity note."}]), encoding="utf-8")

    out = tmp_path / "phase0b"
    builder = Phase1CurriculumBuilder(
        runtime_db=str(runtime_db),
        runtime_json=str(runtime_json),
        episodic_db=str(episodic_db),
        old_memory_db=str(old_db),
        personal_log=str(personal_log),
        curriculum_dir=curriculum,
        out_dir=out,
        max_examples=10,
        include_personal_log=True,
    )
    stats = builder.run()

    manifest = json.loads((out / "phase0b_curriculum_manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "axon_phase0b_curriculum_manifest"
    assert set(manifest["families"]) == set(ALL_FAMILIES)
    assert manifest["options"]["include_personal_log"] is True
    for family in ALL_FAMILIES:
        assert (out / f"{family}.jsonl").exists()
        assert stats.counts[family] >= 1

    runtime_rows = _read_jsonl(out / "runtime_response_delta_v1.jsonl")
    assert runtime_rows[0]["active_field"]["response_draft"] == ""
    assert runtime_rows[0]["target_delta"]["region"] == "response_draft"

    scratch_rows = _read_jsonl(out / "scratchpad_delta_v1.jsonl")
    assert scratch_rows[0]["target_delta"]["region"] == "scratch"
    assert scratch_rows[0]["secondary_delta"]["region"] == "response_draft"

    soul_rows = _read_jsonl(out / "soul_exhale_pair_v1.jsonl")
    assert soul_rows[0]["tick_a"]["target_soul_trace"]
    assert soul_rows[0]["probe"]["zero_soul"] == "should_fail_or_be_less_specific"

    written = "\n".join(p.read_text(encoding="utf-8") for p in out.glob("*.jsonl"))
    assert '"symbol"' not in written
    assert "edge-symbol" not in written


def test_phase1_builder_keeps_diary_opt_in(tmp_path: Path) -> None:
    curriculum = tmp_path / "curriculum_v1"
    _curriculum_dir(curriculum)
    personal_log = tmp_path / "axon_personal_log.json"
    personal_log.write_text(json.dumps([{"content": "Private continuity note."}]), encoding="utf-8")

    out = tmp_path / "phase0b"
    builder = Phase1CurriculumBuilder(
        runtime_db=str(tmp_path / "missing_runtime.db"),
        runtime_json=str(tmp_path / "missing_runtime.json"),
        episodic_db=str(tmp_path / "missing_episodic.db"),
        old_memory_db=str(tmp_path / "missing_old.db"),
        personal_log=str(personal_log),
        curriculum_dir=curriculum,
        out_dir=out,
        families=["diary_continuity_delta_v1"],
        max_examples=10,
        include_personal_log=False,
    )
    stats = builder.run()

    assert stats.counts["diary_continuity_delta_v1"] == 0
    assert "personal_log_not_included" in stats.skipped["diary_continuity_delta_v1"]
    assert (out / "diary_continuity_delta_v1.jsonl").read_text(encoding="utf-8") == ""


def test_phase1_builder_records_empty_episode_skips_per_family(tmp_path: Path) -> None:
    curriculum = tmp_path / "curriculum_v1"
    _write_jsonl(
        curriculum / "episodic_exhale_filter_v1.jsonl",
        [{"container_id": "empty", "episode_summary": "", "facts_to_retain": [], "structured_knowledge": ""}],
    )

    out = tmp_path / "phase0b"
    builder = Phase1CurriculumBuilder(
        runtime_db=str(tmp_path / "missing_runtime.db"),
        runtime_json=str(tmp_path / "missing_runtime.json"),
        episodic_db=str(tmp_path / "missing_episodic.db"),
        old_memory_db=str(tmp_path / "missing_old.db"),
        personal_log=str(tmp_path / "missing_personal.json"),
        curriculum_dir=curriculum,
        out_dir=out,
        families=["soul_ablation_probe_v1"],
        max_examples=10,
    )
    stats = builder.run()

    assert stats.counts["soul_ablation_probe_v1"] == 0
    assert "empty_episode" in stats.skipped["soul_ablation_probe_v1"]
    assert "soul_exhale_pair_v1" not in stats.skipped
