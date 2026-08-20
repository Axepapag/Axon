#!/usr/bin/env python3
r"""Build Axon phase0b/phase1 curriculum from recovered runtime memory.

This builder prepares the next rung after phase0 charslot bootstrap. It stays
read-only over Jeff's recovered memory files and writes generated JSONL
datasets under ``datasets/recovered/phase0b_curriculum_v1``.

Doctrine:
  - Exact text remains character material for the frozen 16D substrate.
  - Semantic edges are rendered as readable English, never opaque symbols.
  - Soul examples are probe/training specifications; the soul is private
    core state and is not the canonical knowledge store.
  - Personal diary material is isolated behind ``--include-personal-log``.
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
from typing import Any, Iterable, Iterator, Sequence

ROOT = Path(__file__).resolve().parent.parent
SUBSTRATE = ROOT / "substrate"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SUBSTRATE) not in sys.path:
    sys.path.insert(0, str(SUBSTRATE))

from substrate import default_alphabet

DEFAULT_RUNTIME_DB = r"D:\00\axon_runtime_state.db"
DEFAULT_RUNTIME_JSON = r"D:\00\axon_runtime_state.json"
DEFAULT_EPISODIC_DB = r"D:\00\axon_episodic_memory.db"
DEFAULT_OLD_MEMORY_DB = r"D:\00\axon_memory.db"
DEFAULT_PERSONAL_LOG = r"D:\00\axon_personal_log.json"
DEFAULT_CURRICULUM_DIR = ROOT / "datasets" / "recovered" / "curriculum_v1"
DEFAULT_OUT_DIR = ROOT / "datasets" / "recovered" / "phase0b_curriculum_v1"

ALL_FAMILIES = [
    "runtime_response_delta_v1",
    "structured_knowledge_delta_v1",
    "scratchpad_delta_v1",
    "soul_exhale_pair_v1",
    "soul_ablation_probe_v1",
    "diary_continuity_delta_v1",
]

TRIM_RE = re.compile(r"[ \t\r\f\v]+")
ALPHABET = set(default_alphabet())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text(value: Any, max_chars: int | None = None, *, keep_newlines: bool = False) -> str:
    """Return text safe for the current substrate alphabet."""
    raw = "" if value is None else str(value)
    out: list[str] = []
    for ch in raw:
        if ch in ALPHABET:
            if ch == "\n" and not keep_newlines:
                out.append(" ")
            else:
                out.append(ch)
        elif ch.isspace():
            out.append("\n" if keep_newlines else " ")
        else:
            out.append(" ")
    text = "".join(out)
    if keep_newlines:
        lines = [TRIM_RE.sub(" ", line).strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
    else:
        text = TRIM_RE.sub(" ", text).strip()
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def short_source(path: str | Path, *parts: Any) -> str:
    suffix = ":".join(str(p) for p in parts if p is not None and str(p) != "")
    return f"{path}{':' + suffix if suffix else ''}"


def iter_jsonl(path: Path, limit: int = -1) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    produced = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit >= 0 and produced >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                produced += 1
            except json.JSONDecodeError:
                continue


def connect_ro(path: str) -> sqlite3.Connection | None:
    if not path or not os.path.exists(path):
        return None
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (table,),
    ).fetchone()
    return row is not None


def render_history(turns: Sequence[dict[str, str]], max_chars: int) -> str:
    lines: list[str] = []
    for turn in turns:
        role = clean_text(turn.get("role", ""), 24)
        content = clean_text(turn.get("content", ""), 512)
        if role and content:
            lines.append(f"{role}: {content}")
    return clean_text("\n".join(lines), max_chars, keep_newlines=True)


def delta_example(
    *,
    family: str,
    source: str,
    active_field: dict[str, str],
    target_region: str,
    target_text: str,
    tags: Sequence[str] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "family": family,
        "source": source,
        "active_field": {k: clean_text(v, keep_newlines=True) for k, v in active_field.items()},
        "target_delta": {
            "region": target_region,
            "op": "replace",
            "text": clean_text(target_text, 2048, keep_newlines=True),
        },
        "loss": {
            "type": "char_delta",
            "region": target_region,
        },
        "tags": list(tags),
    }
    if extra:
        row.update(extra)
    return row


@dataclass
class BuildStats:
    counts: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, dict[str, int]] = field(default_factory=dict)
    source_files: list[dict[str, Any]] = field(default_factory=list)

    def add(self, family: str) -> None:
        self.counts[family] = self.counts.get(family, 0) + 1

    def skip(self, family: str, reason: str) -> None:
        bucket = self.skipped.setdefault(family, {})
        bucket[reason] = bucket.get(reason, 0) + 1


class Phase1CurriculumBuilder:
    def __init__(
        self,
        *,
        runtime_db: str = DEFAULT_RUNTIME_DB,
        runtime_json: str = DEFAULT_RUNTIME_JSON,
        episodic_db: str = DEFAULT_EPISODIC_DB,
        old_memory_db: str = DEFAULT_OLD_MEMORY_DB,
        personal_log: str = DEFAULT_PERSONAL_LOG,
        curriculum_dir: Path = DEFAULT_CURRICULUM_DIR,
        out_dir: Path = DEFAULT_OUT_DIR,
        families: Sequence[str] | None = None,
        max_examples: int = 10_000,
        include_personal_log: bool = False,
        hash_sources: bool = False,
        history_chars: int = 768,
        seed: int = 42,
    ):
        self.runtime_db = runtime_db
        self.runtime_json = runtime_json
        self.episodic_db = episodic_db
        self.old_memory_db = old_memory_db
        self.personal_log = personal_log
        self.curriculum_dir = Path(curriculum_dir)
        self.out_dir = Path(out_dir)
        self.families = list(families) if families is not None else list(ALL_FAMILIES)
        self.max_examples = max_examples
        self.include_personal_log = include_personal_log
        self.hash_sources = hash_sources
        self.history_chars = history_chars
        self.seed = seed
        self.stats = BuildStats()

    def run(self) -> BuildStats:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stats.source_files = self._source_file_records()
        for family in self.families:
            rows = self._build_family(family)
            self._write_family(family, rows)
        self._write_manifest()
        return self.stats

    def _source_file_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        paths: list[Path] = [
            Path(self.runtime_db),
            Path(self.runtime_json),
            Path(self.episodic_db),
            Path(self.old_memory_db),
            Path(self.personal_log),
            self.curriculum_dir,
        ]
        for path in paths:
            rec: dict[str, Any] = {
                "path": str(path),
                "exists": path.exists(),
            }
            if path.exists() and path.is_file():
                rec["size_bytes"] = path.stat().st_size
                rec["mtime"] = int(path.stat().st_mtime)
                rec["sha256"] = sha256_file(path) if self.hash_sources else ""
                if not self.hash_sources:
                    rec["sha256_note"] = "not hashed by default to keep builds cheap"
            elif path.exists() and path.is_dir():
                rec["kind"] = "directory"
            records.append(rec)
        return records

    def _build_family(self, family: str) -> Iterator[dict[str, Any]]:
        if family == "runtime_response_delta_v1":
            return self._build_runtime_response_delta()
        if family == "structured_knowledge_delta_v1":
            return self._build_structured_knowledge_delta()
        if family == "scratchpad_delta_v1":
            return self._build_scratchpad_delta()
        if family == "soul_exhale_pair_v1":
            return self._build_soul_exhale_pair()
        if family == "soul_ablation_probe_v1":
            return self._build_soul_ablation_probe()
        if family == "diary_continuity_delta_v1":
            return self._build_diary_continuity_delta()
        raise ValueError(f"unknown phase1 curriculum family: {family}")

    def _write_family(self, family: str, rows: Iterator[dict[str, Any]]) -> None:
        path = self.out_dir / f"{family}.jsonl"
        count = 0
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                if count >= self.max_examples:
                    break
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        self.stats.counts[family] = count

    def _write_manifest(self) -> None:
        manifest = {
            "kind": "axon_phase0b_curriculum_manifest",
            "version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "families": self.families,
            "family_counts": self.stats.counts,
            "family_skipped": self.stats.skipped,
            "source_files": self.stats.source_files,
            "options": {
                "max_examples": self.max_examples,
                "include_personal_log": self.include_personal_log,
                "hash_sources": self.hash_sources,
                "history_chars": self.history_chars,
                "seed": self.seed,
            },
            "notes": (
                "Phase0b/phase1 curriculum for exact character-field runtime. "
                "Rows are JSON specifications for future trainers and probes; "
                "they do not mutate dormant state or the active CPU run."
            ),
        }
        with (self.out_dir / "phase0b_curriculum_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _runtime_turns_from_json(self) -> list[dict[str, str]]:
        path = Path(self.runtime_json)
        if not path.exists():
            return []
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        turns: list[dict[str, str]] = []
        for item in obj.get("conversations", []) if isinstance(obj, dict) else []:
            if not isinstance(item, dict):
                continue
            role = clean_text(item.get("role", ""), 24)
            content = clean_text(item.get("content", ""), 4096)
            if role and content:
                turns.append({"role": role, "content": content, "source": short_source(path, "conversations")})
        return turns

    def _runtime_turns_from_db(self, db_path: str) -> list[dict[str, str]]:
        con = connect_ro(db_path)
        if con is None:
            return []
        try:
            if not table_exists(con, "messages"):
                return []
            rows = con.execute(
                'select id, role, content, timestamp from messages order by timestamp, id limit ?',
                (max(self.max_examples * 4, 1000),),
            ).fetchall()
        finally:
            con.close()
        turns: list[dict[str, str]] = []
        for msg_id, role, content, timestamp in rows:
            role_s = clean_text(role, 24)
            content_s = clean_text(content, 4096)
            if role_s and content_s:
                turns.append({
                    "role": role_s,
                    "content": content_s,
                    "source": short_source(db_path, "messages", msg_id or timestamp),
                })
        return turns

    def _dialogue_pairs(self) -> Iterator[tuple[list[dict[str, str]], dict[str, str], dict[str, str]]]:
        seen: set[tuple[str, str]] = set()
        streams = [
            self._runtime_turns_from_json(),
            self._runtime_turns_from_db(self.runtime_db),
            self._runtime_turns_from_db(self.old_memory_db),
        ]
        for turns in streams:
            history: list[dict[str, str]] = []
            for idx in range(len(turns) - 1):
                cur = turns[idx]
                nxt = turns[idx + 1]
                if cur["role"].lower() not in {"user", "human"}:
                    history.append(cur)
                    continue
                if nxt["role"].lower() not in {"assistant", "axon"}:
                    history.append(cur)
                    continue
                key = (cur["content"][:256], nxt["content"][:256])
                if key in seen:
                    history.append(cur)
                    continue
                seen.add(key)
                yield history[-8:], cur, nxt
                history.extend([cur, nxt])

    def _build_runtime_response_delta(self) -> Iterator[dict[str, Any]]:
        for history_turns, user_turn, assistant_turn in self._dialogue_pairs():
            answer = clean_text(assistant_turn["content"], 2048)
            user_input = clean_text(user_turn["content"], 1024)
            if len(answer) < 4 or len(user_input) < 2:
                self.stats.skip("runtime_response_delta_v1", "too_short")
                continue
            yield delta_example(
                family="runtime_response_delta_v1",
                source=assistant_turn.get("source", ""),
                active_field={
                    "conversation_history": render_history(history_turns, self.history_chars),
                    "user_input": user_input,
                    "structured_knowledge": "",
                    "response_draft": "",
                },
                target_region="response_draft",
                target_text=answer,
                tags=["real_runtime_memory", "response_draft", "phase0b"],
            )
            self.stats.add("runtime_response_delta_v1")

    def _build_structured_knowledge_delta(self) -> Iterator[dict[str, Any]]:
        files = [
            self.curriculum_dir / "field_surfacing_v1.jsonl",
            self.curriculum_dir / "retrieval_qa_v1.jsonl",
        ]
        for file_path in files:
            for row in iter_jsonl(file_path, self.max_examples * 3):
                query = row.get("query") or row.get("source_text") or row.get("context") or ""
                structured = row.get("structured_knowledge") or row.get("context_slots_text") or ""
                answer = row.get("answer") or row.get("target") or ""
                query_s = clean_text(query, 1024)
                structured_s = clean_text(structured, 2048)
                answer_s = clean_text(answer, 512)
                if not query_s or not structured_s or not answer_s:
                    self.stats.skip("structured_knowledge_delta_v1", "missing_field")
                    continue
                yield delta_example(
                    family="structured_knowledge_delta_v1",
                    source=short_source(file_path, row.get("container_id")),
                    active_field={
                        "conversation_history": "",
                        "user_input": query_s,
                        "structured_knowledge": structured_s,
                        "response_draft": "",
                    },
                    target_region="response_draft",
                    target_text=answer_s,
                    tags=["surfaced_dormant", "readable_edges", "response_draft"],
                    extra={"container_id": row.get("container_id", "")},
                )
                self.stats.add("structured_knowledge_delta_v1")

    def _build_scratchpad_delta(self) -> Iterator[dict[str, Any]]:
        path = self.curriculum_dir / "procedure_next_step_v1.jsonl"
        for row in iter_jsonl(path, self.max_examples * 3):
            answer = clean_text(row.get("answer", ""), 512)
            if not answer:
                self.stats.skip("scratchpad_delta_v1", "empty_answer")
                continue
            steps = row.get("steps_before") or row.get("steps") or []
            if not isinstance(steps, list):
                steps = []
            steps_text = clean_text("\n".join(f"step: {s}" for s in steps), 2048, keep_newlines=True)
            procedure = clean_text(row.get("procedure_name", ""), 256)
            trigger = clean_text(row.get("trigger", ""), 512)
            user_input = clean_text(f"{trigger} next step", 512)
            structured = clean_text(f"procedure: {procedure}\n{steps_text}", 2048, keep_newlines=True)
            scratch = clean_text(f"next step: {answer}", 512)
            yield delta_example(
                family="scratchpad_delta_v1",
                source=short_source(path, row.get("container_id"), row.get("sub_family")),
                active_field={
                    "conversation_history": "",
                    "user_input": user_input,
                    "structured_knowledge": structured,
                    "scratch": "",
                    "response_draft": "",
                },
                target_region="scratch",
                target_text=scratch,
                tags=["procedure", "scratch_delta"],
                extra={
                    "secondary_delta": {
                        "region": "response_draft",
                        "op": "replace",
                        "text": answer,
                    },
                    "container_id": row.get("container_id", ""),
                },
            )
            self.stats.add("scratchpad_delta_v1")

    def _episode_rows_from_curriculum(self) -> Iterator[dict[str, Any]]:
        path = self.curriculum_dir / "episodic_exhale_filter_v1.jsonl"
        for row in iter_jsonl(path, self.max_examples * 4):
            yield {**row, "_source": short_source(path, row.get("container_id"), row.get("phase"))}

    def _episode_rows_from_db(self) -> Iterator[dict[str, Any]]:
        con = connect_ro(self.episodic_db)
        if con is None:
            return
        try:
            if not table_exists(con, "episodes"):
                return
            rows = con.execute(
                "select id, episode_type, summary, extracted_json, source, created_at from episodes order by id limit ?",
                (max(self.max_examples * 3, 1000),),
            ).fetchall()
        finally:
            con.close()
        for episode_id, episode_type, summary, extracted_json, source, created_at in rows:
            facts: list[str] = []
            if extracted_json:
                try:
                    obj = json.loads(extracted_json)
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            if isinstance(value, (str, int, float, bool)):
                                facts.append(clean_text(f"{key}: {value}", 256))
                            elif isinstance(value, list):
                                for item in value[:3]:
                                    facts.append(clean_text(f"{key}: {item}", 256))
                except json.JSONDecodeError:
                    pass
            yield {
                "family": "episodic_exhale_filter_v1",
                "phase": "a",
                "container_id": f"episode-{episode_id}",
                "episode_summary": clean_text(summary, 1024),
                "facts_to_retain": [f for f in facts if f][:5],
                "structured_knowledge": clean_text(f"episode type {episode_type} summary {summary}", 2048),
                "_source": short_source(self.episodic_db, "episodes", episode_id or created_at or source),
            }

    def _paired_episode_specs(self, skip_family: str) -> Iterator[dict[str, Any]]:
        seen: set[str] = set()
        for row in list(self._episode_rows_from_curriculum()) + list(self._episode_rows_from_db()):
            summary = clean_text(row.get("episode_summary") or row.get("structured_knowledge") or "", 1024)
            facts_raw = row.get("facts_to_retain") or row.get("expected_answer") or []
            if isinstance(facts_raw, str):
                facts = [clean_text(facts_raw, 512)]
            elif isinstance(facts_raw, list):
                facts = [clean_text(x, 512) for x in facts_raw if clean_text(x, 512)]
            else:
                facts = []
            if not summary and not facts:
                self.stats.skip(skip_family, "empty_episode")
                continue
            key = row.get("container_id") or summary[:128]
            if key in seen:
                continue
            seen.add(str(key))
            retain = facts[:3] or [summary[:256]]
            yield {
                "source": row.get("_source", ""),
                "container_id": row.get("container_id", ""),
                "summary": summary,
                "retain": retain,
                "structured": clean_text(row.get("structured_knowledge") or summary, 2048),
            }

    def _build_soul_exhale_pair(self) -> Iterator[dict[str, Any]]:
        for spec in self._paired_episode_specs("soul_exhale_pair_v1"):
            retain_text = clean_text("\n".join(f"retain: {x}" for x in spec["retain"]), 1024, keep_newlines=True)
            yield {
                "family": "soul_exhale_pair_v1",
                "source": spec["source"],
                "container_id": spec["container_id"],
                "tick_a": {
                    "active_field": {
                        "structured_knowledge": spec["structured"],
                        "user_input": "Read this episode and keep only useful experience in soul.",
                        "response_draft": "",
                    },
                    "target_delta": {
                        "region": "response_draft",
                        "op": "replace",
                        "text": "noted",
                    },
                    "target_soul_trace": retain_text,
                },
                "tick_b": {
                    "active_field": {
                        "structured_knowledge": "",
                        "user_input": "What relevant experience did you retain from the previous tick?",
                        "response_draft": "",
                    },
                    "expected_answer": retain_text,
                },
                "probe": {
                    "correct_soul": "should_answer",
                    "zero_soul": "should_fail_or_be_less_specific",
                    "swapped_soul": "should_fail_or_answer_other_episode",
                },
                "tags": ["soul", "exhale_after_answer", "hot_to_cold_training_seed"],
            }
            self.stats.add("soul_exhale_pair_v1")

    def _build_soul_ablation_probe(self) -> Iterator[dict[str, Any]]:
        for spec in self._paired_episode_specs("soul_ablation_probe_v1"):
            expected = clean_text("\n".join(spec["retain"]), 1024, keep_newlines=True)
            yield {
                "family": "soul_ablation_probe_v1",
                "source": spec["source"],
                "container_id": spec["container_id"],
                "setup_tick": {
                    "active_field": {
                        "structured_knowledge": spec["structured"],
                        "user_input": "Store this episode for a later probe.",
                    },
                    "target_soul_trace": expected,
                },
                "probe_tick": {
                    "active_field": {
                        "structured_knowledge": "",
                        "user_input": "Recall the retained episode detail.",
                        "response_draft": "",
                    },
                    "expected_answer": expected,
                },
                "ablation_matrix": [
                    {"name": "correct_soul", "expected": "match_expected_answer"},
                    {"name": "zero_soul", "expected": "lower_specificity_or_miss"},
                    {"name": "swapped_soul", "expected": "mismatch_expected_answer"},
                ],
                "metrics": ["char_accuracy", "specificity_score", "answer_contains_retained_fact"],
                "tags": ["probe_only", "soul_load_bearing"],
            }
            self.stats.add("soul_ablation_probe_v1")

    def _build_diary_continuity_delta(self) -> Iterator[dict[str, Any]]:
        if not self.include_personal_log:
            self.stats.skip("diary_continuity_delta_v1", "personal_log_not_included")
            return
        path = Path(self.personal_log)
        if not path.exists():
            self.stats.skip("diary_continuity_delta_v1", "missing_personal_log")
            return
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            self.stats.skip("diary_continuity_delta_v1", "bad_json")
            return
        entries = obj if isinstance(obj, list) else obj.get("entries", []) if isinstance(obj, dict) else []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            text = clean_text(
                entry.get("content") or entry.get("text") or entry.get("summary") or entry,
                2048,
                keep_newlines=True,
            )
            if not text:
                self.stats.skip("diary_continuity_delta_v1", "empty_diary")
                continue
            yield delta_example(
                family="diary_continuity_delta_v1",
                source=short_source(path, idx),
                active_field={
                    "diary": text,
                    "user_input": "Extract a continuity note for private diary state.",
                    "response_draft": "",
                },
                target_region="scratch",
                target_text=f"diary continuity: {text[:512]}",
                tags=["diary_only", "private_continuity"],
                extra={"allowed_regions": ["diary", "scratch"], "blocked_regions": ["structured_knowledge"]},
            )
            self.stats.add("diary_continuity_delta_v1")


def parse_families(value: str) -> list[str]:
    if value.lower() == "all":
        return list(ALL_FAMILIES)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build Axon phase0b/phase1 curriculum specs.")
    ap.add_argument("--runtime-db", default=DEFAULT_RUNTIME_DB)
    ap.add_argument("--runtime-json", default=DEFAULT_RUNTIME_JSON)
    ap.add_argument("--episodic-db", default=DEFAULT_EPISODIC_DB)
    ap.add_argument("--old-memory-db", default=DEFAULT_OLD_MEMORY_DB)
    ap.add_argument("--personal-log", default=DEFAULT_PERSONAL_LOG)
    ap.add_argument("--curriculum-dir", default=str(DEFAULT_CURRICULUM_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--families", default="all")
    ap.add_argument("--max-examples", type=int, default=10_000)
    ap.add_argument("--history-chars", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-personal-log", action="store_true")
    ap.add_argument("--hash-sources", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.max_examples = min(args.max_examples, 50)
    builder = Phase1CurriculumBuilder(
        runtime_db=args.runtime_db,
        runtime_json=args.runtime_json,
        episodic_db=args.episodic_db,
        old_memory_db=args.old_memory_db,
        personal_log=args.personal_log,
        curriculum_dir=Path(args.curriculum_dir),
        out_dir=Path(args.out_dir),
        families=parse_families(args.families),
        max_examples=args.max_examples,
        include_personal_log=args.include_personal_log,
        hash_sources=args.hash_sources,
        history_chars=args.history_chars,
        seed=args.seed,
    )
    stats = builder.run()
    print("Phase0b curriculum build complete.")
    for family in builder.families:
        print(f"  {family}: {stats.counts.get(family, 0)}")
    print(f"  Output dir: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
