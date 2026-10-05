#!/usr/bin/env python3
"""Axon v7 state/delta curriculum factory.

This factory builds the curriculum Jeff described:

1. copy a full valid state exactly
2. copy weirder full states exactly
3. make one small response delta while preserving the rest
4. evict or rewrite one non-draft region
5. grow non-draft regions through ++ calls between ticks
6. answer from structured state

It writes normal trainer_v2 JSONL episodes. The vectors are still made by
field_contract.py during training; this script only organizes source material
into Axon's readable state regions.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from field_contract import SHARED_ORDER, SHARED_REGIONS
from trainer_v2 import ALLOWED_CHARS, parse_episode

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "datasets" / "axon7_state_delta_v1.jsonl"

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".log", ".json", ".jsonl", ".csv",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".yaml",
    ".yml", ".toml", ".ini",
}
SQLITE_EXTS = {".db", ".sqlite", ".sqlite3"}

REGION_BUDGETS = {
    "input_window": 180,
    "response_draft": 220,
    "working_memory": 240,
    "rolling_summary": 240,
    "situation_awareness": 240,
    "structured_knowledge": 260,
    "diary": 200,
    "advisor": 180,
    "tool_results": 220,
    "task_state": 160,
}

REGION_PREFIX = {
    "input_window": "Jeff:",
    "response_draft": "Draft:",
    "working_memory": "Memory:",
    "rolling_summary": "Summary:",
    "situation_awareness": "Awareness:",
    "structured_knowledge": "Knowledge:",
    "diary": "Diary:",
    "advisor": "Advisor:",
    "tool_results": "Tool result:",
    "task_state": "Task:",
}

AXON_WRITABLE_REGIONS = (
    "working_memory",
    "rolling_summary",
    "situation_awareness",
    "diary",
    "task_state",
)

PLUS_CODE_BY_REGION = {name: code for name, code, _, _ in SHARED_REGIONS}

DEFAULT_SNIPPETS = [
    "Axon owns the readable state after every tick.",
    "The first lesson is to copy a full state exactly.",
    "Only response_draft has standing buffer rows.",
    "Non-draft regions grow through complete ++ calls between ticks.",
    "Empty rows evict readable state.",
    "Tool results and advisor notes can be cleared after use.",
    "The core should preserve unchanged regions before making changes.",
    "Task state records what Axon is currently doing.",
    "Rolling summary carries conversation context.",
    "Structured knowledge can be brought into active state.",
]

_ASCII_REPAIRS = str.maketrans({
    "\r": "\n",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
})


@dataclass
class SourceBundle:
    snippets: list[str]
    states: list[dict[str, str]]
    files_read: int = 0
    sqlite_tables: int = 0


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).translate(_ASCII_REPAIRS)
    text = "".join(c if c in ALLOWED_CHARS else " " for c in text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_good_snippet(text: str) -> bool:
    text = clean_text(text)
    if len(text) < 8:
        return False
    lower = text.lower()
    if "vector=b'" in lower or "embedding=b'" in lower:
        return False
    if lower.count("\\x") > 2:
        return False
    if sum(ch.isalnum() for ch in text) < max(4, len(text) // 5):
        return False
    return True


def clamp(text: str, limit: int) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rfind(" ")
    if cut < limit // 3:
        cut = limit
    return text[:cut].rstrip(" ,;:-")


def fixed_width(text: str, width: int) -> str:
    text = clean_text(text)
    if len(text) >= width:
        return text[:width]
    return text + ("." * (width - len(text)))


def blank_state() -> dict[str, str]:
    return {region: "" for region in SHARED_ORDER}


def normalize_state(raw: Any) -> dict[str, str]:
    out = blank_state()
    if not isinstance(raw, dict):
        return out
    for region in SHARED_ORDER:
        out[region] = clamp(raw.get(region, ""), REGION_BUDGETS[region])
    return out


def make_tick(before: dict[str, str], after: dict[str, str], input_event: str = "") -> dict[str, Any]:
    return {
        "input_event": clamp(input_event, 220),
        "harness_events": [],
        "state_before": before,
        "state_after": after,
    }


def make_episode(stage: int, idx: int, lesson: str, ticks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_id": f"sd_s{stage:02d}_{idx:05d}",
        "source": "state_curriculum_factory",
        "stage": stage,
        "lesson": lesson,
        "ticks": ticks,
    }


def flatten_json_text(obj: Any, out: list[str], *, limit: int) -> None:
    if len(out) >= limit:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"state_before", "state_after", "ticks"}:
                continue
            if isinstance(value, (dict, list)):
                flatten_json_text(value, out, limit=limit)
            else:
                text = clean_text(value)
                if is_good_snippet(text):
                    out.append(f"{clean_text(key)}: {text}")
                    if len(out) >= limit:
                        return
        return
    if isinstance(obj, list):
        for item in obj:
            flatten_json_text(item, out, limit=limit)
            if len(out) >= limit:
                return
        return
    text = clean_text(obj)
    if is_good_snippet(text):
        out.append(text)


def harvest_axon_states(obj: Any) -> list[dict[str, str]]:
    states: list[dict[str, str]] = []
    if not isinstance(obj, dict):
        return states
    ticks = obj.get("ticks")
    if not isinstance(ticks, list):
        return states
    for tick in ticks:
        if not isinstance(tick, dict):
            continue
        for key in ("state_before", "state_after"):
            raw = tick.get(key)
            if isinstance(raw, dict):
                st = normalize_state(raw)
                if any(st.values()):
                    states.append(st)
    return states


def split_plain_text(text: str, *, chunk_chars: int, limit: int) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    pieces = re.split(r"(?:\n\s*){2,}|(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        piece = clean_text(piece)
        if not piece:
            continue
        if len(piece) > chunk_chars:
            for start in range(0, len(piece), chunk_chars):
                chunks.append(piece[start:start + chunk_chars].strip())
                if len(chunks) >= limit:
                    return chunks
            current = ""
            continue
        if current and len(current) + 1 + len(piece) > chunk_chars:
            chunks.append(current)
            if len(chunks) >= limit:
                return chunks
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current and len(chunks) < limit:
        chunks.append(current)
    return chunks[:limit]


def read_text_limited(path: Path, max_bytes: int) -> str:
    with path.open("rb") as f:
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace")


def iter_jsonl(path: Path, max_lines: int) -> Iterable[Any]:
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if line_no > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_csv_rows(path: Path, limit: int) -> list[str]:
    rows: list[str] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames:
            for row in reader:
                parts = [
                    f"{clean_text(k)}={clean_text(v)}"
                    for k, v in row.items()
                    if clean_text(v)
                ]
                if parts:
                    rows.append(" | ".join(parts))
                if len(rows) >= limit:
                    break
        else:
            f.seek(0)
            raw_reader = csv.reader(f, dialect=dialect)
            for row in raw_reader:
                text = " | ".join(clean_text(v) for v in row if clean_text(v))
                if text:
                    rows.append(text)
                if len(rows) >= limit:
                    break
    return rows


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_sqlite(path: Path, *, max_tables: int, rows_per_table: int) -> tuple[list[str], int]:
    snippets: list[str] = []
    table_count = 0
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return snippets, table_count
    try:
        tables = [
            row[0]
            for row in con.execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%' order by name"
            )
        ]
        for table in tables[:max_tables]:
            table_lower = table.lower()
            if "vector" in table_lower or "embedding" in table_lower:
                continue
            table_count += 1
            try:
                cur = con.execute(f"select * from {quote_ident(table)} limit ?", (rows_per_table,))
            except sqlite3.Error:
                continue
            cols = [desc[0] for desc in cur.description or []]
            for row in cur.fetchall():
                parts = []
                for col, value in zip(cols, row):
                    col_lower = str(col).lower()
                    if "vector" in col_lower or "embedding" in col_lower:
                        continue
                    if isinstance(value, (bytes, bytearray, memoryview)):
                        continue
                    text = clean_text(value)
                    if is_good_snippet(text):
                        parts.append(f"{clean_text(col)}={text}")
                if parts:
                    snippets.append(f"sqlite {path.name}.{table}: " + " | ".join(parts))
    finally:
        con.close()
    return snippets, table_count


def collect_files(source: Path, max_files: int) -> list[Path]:
    if source.is_file():
        return [source]
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTS or suffix in SQLITE_EXTS:
            files.append(path)
            if len(files) >= max_files:
                break
    return files


def load_source(
    source: Path,
    *,
    max_files: int,
    max_items: int,
    chunk_chars: int,
    max_file_bytes: int,
) -> SourceBundle:
    snippets: list[str] = []
    states: list[dict[str, str]] = []
    sqlite_tables = 0
    files = collect_files(source, max_files)
    for path in files:
        if len(snippets) >= max_items and len(states) >= max_items:
            break
        suffix = path.suffix.lower()
        if suffix in SQLITE_EXTS:
            rows, tables = read_sqlite(path, max_tables=8, rows_per_table=64)
            sqlite_tables += tables
            snippets.extend(rows[: max(0, max_items - len(snippets))])
            continue
        if suffix == ".jsonl":
            for obj in iter_jsonl(path, max_lines=2000):
                states.extend(harvest_axon_states(obj))
                flat: list[str] = []
                flatten_json_text(obj, flat, limit=8)
                snippets.extend(flat[: max(0, max_items - len(snippets))])
                if len(snippets) >= max_items and len(states) >= max_items:
                    break
            continue
        if suffix == ".json":
            text = read_text_limited(path, max_file_bytes)
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                snippets.extend(split_plain_text(text, chunk_chars=chunk_chars, limit=32))
                continue
            states.extend(harvest_axon_states(obj))
            flat: list[str] = []
            flatten_json_text(obj, flat, limit=64)
            snippets.extend(flat[: max(0, max_items - len(snippets))])
            continue
        if suffix == ".csv":
            snippets.extend(read_csv_rows(path, max(0, max_items - len(snippets))))
            continue
        text = read_text_limited(path, max_file_bytes)
        snippets.extend(split_plain_text(text, chunk_chars=chunk_chars, limit=64))
        snippets = snippets[:max_items]

    cleaned_snippets = []
    seen = set()
    for snippet in snippets + DEFAULT_SNIPPETS:
        snippet = clamp(snippet, chunk_chars)
        if not is_good_snippet(snippet):
            continue
        key = snippet.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned_snippets.append(snippet)
        if len(cleaned_snippets) >= max_items:
            break

    cleaned_states: list[dict[str, str]] = []
    seen_states = set()
    for state in states:
        norm = normalize_state(state)
        key = json.dumps(norm, sort_keys=True)
        if key in seen_states:
            continue
        seen_states.add(key)
        cleaned_states.append(norm)
        if len(cleaned_states) >= max_items:
            break

    return SourceBundle(
        snippets=cleaned_snippets or DEFAULT_SNIPPETS,
        states=cleaned_states,
        files_read=len(files),
        sqlite_tables=sqlite_tables,
    )


def choose_snippet(rng: random.Random, snippets: list[str], limit: int) -> str:
    return clamp(rng.choice(snippets), limit)


def region_text(region: str, snippet: str, idx: int, *, weird: bool) -> str:
    budget = REGION_BUDGETS[region]
    prefix = REGION_PREFIX[region]
    if weird:
        forms = [
            f"{prefix} [{idx}] {snippet}",
            f"{prefix} slot={idx}; keep={snippet}",
            f"{prefix} {{{idx}}} {snippet} :: preserve",
        ]
    else:
        forms = [
            f"{prefix} {snippet}",
            f"{prefix} {idx}: {snippet}",
        ]
    return clamp(forms[idx % len(forms)], budget)


def random_state(
    rng: random.Random,
    bundle: SourceBundle,
    idx: int,
    *,
    dense: bool = False,
    weird: bool = False,
    prefer_harvested: bool = True,
) -> dict[str, str]:
    if prefer_harvested and bundle.states and rng.random() < 0.35:
        base = dict(rng.choice(bundle.states))
    else:
        base = blank_state()

    fill_prob = 0.75 if dense else 0.42
    for region in SHARED_ORDER:
        if base.get(region) and rng.random() < 0.65:
            continue
        if rng.random() > fill_prob:
            continue
        snippet = choose_snippet(rng, bundle.snippets, REGION_BUDGETS[region] - 32)
        base[region] = region_text(region, snippet, idx, weird=weird)

    if not any(base.values()):
        base["task_state"] = "Task: copy this state exactly."
        base["structured_knowledge"] = choose_snippet(
            rng, bundle.snippets, REGION_BUDGETS["structured_knowledge"])

    if not base["input_window"] and rng.random() < 0.8:
        base["input_window"] = "Jeff: copy the full state exactly."
    return normalize_state(base)


def build_copy_stage(
    stage: int,
    lesson: str,
    count: int,
    rng: random.Random,
    bundle: SourceBundle,
    *,
    dense: bool,
    weird: bool,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        before = random_state(rng, bundle, i, dense=dense, weird=weird)
        after = dict(before)
        episodes.append(make_episode(stage, i, lesson, [
            make_tick(before, after, "copy full state exactly"),
        ]))
    return episodes


def build_response_delta(count: int, rng: random.Random, bundle: SourceBundle) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        before = random_state(rng, bundle, i, dense=False, weird=False)
        snippet = choose_snippet(rng, bundle.snippets, 140)
        before["input_window"] = clamp(f"Jeff: answer from this state item {i}.", REGION_BUDGETS["input_window"])
        if rng.random() < 0.4:
            before["response_draft"] = clamp("Draft: stale partial answer.", 80)
        after = dict(before)
        after["response_draft"] = clamp(f"Source says: {snippet}", REGION_BUDGETS["response_draft"])
        episodes.append(make_episode(3, i, "delta_response_draft", [
            make_tick(before, after, before["input_window"]),
        ]))
    return episodes


def build_evict_delta(count: int, rng: random.Random, bundle: SourceBundle) -> list[dict[str, Any]]:
    regions = [
        "working_memory", "rolling_summary", "situation_awareness",
        "structured_knowledge", "diary", "advisor", "tool_results", "task_state",
    ]
    episodes: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        region = regions[(i - 1) % len(regions)]
        before = random_state(rng, bundle, i, dense=False, weird=True)
        before[region] = region_text(
            region,
            choose_snippet(rng, bundle.snippets, REGION_BUDGETS[region] - 32),
            i,
            weird=True,
        )
        before["input_window"] = clamp(f"Jeff: clear {region}.", REGION_BUDGETS["input_window"])
        after = dict(before)
        after[region] = ""
        after["response_draft"] = clamp(f"Cleared {region}.", REGION_BUDGETS["response_draft"])
        episodes.append(make_episode(4, i, "delta_evict_region", [
            make_tick(before, after, before["input_window"]),
        ]))
    return episodes


def build_rewrite_delta(count: int, rng: random.Random, bundle: SourceBundle) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    regions = list(AXON_WRITABLE_REGIONS)
    for i in range(1, count + 1):
        region = regions[(i - 1) % len(regions)]
        before = random_state(rng, bundle, i, dense=False, weird=False)
        width = min(120, REGION_BUDGETS[region])
        old = f"{REGION_PREFIX[region]} old {choose_snippet(rng, bundle.snippets, 70)}"
        new = f"{REGION_PREFIX[region]} new {choose_snippet(rng, bundle.snippets, 70)}"
        before[region] = fixed_width(old, width)
        before["input_window"] = clamp(f"Jeff: rewrite {region} but preserve the rest.", REGION_BUDGETS["input_window"])
        after = dict(before)
        after[region] = fixed_width(new, len(before[region]))
        episodes.append(make_episode(5, i, "delta_rewrite_same_width", [
            make_tick(before, after, before["input_window"]),
        ]))
    return episodes


def build_plus_growth(count: int, rng: random.Random, bundle: SourceBundle) -> list[dict[str, Any]]:
    regions = list(AXON_WRITABLE_REGIONS)
    episodes: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        region = regions[(i - 1) % len(regions)]
        code = PLUS_CODE_BY_REGION[region]
        payload = choose_snippet(rng, bundle.snippets, 90)
        before1 = random_state(rng, bundle, i, dense=False, weird=False)
        before1[region] = ""
        before1["input_window"] = clamp(f"Jeff: add this to {region}: {payload}", REGION_BUDGETS["input_window"])
        after1 = dict(before1)
        after1["response_draft"] = clamp(f"++{code} {{{payload}}}", REGION_BUDGETS["response_draft"])

        before2 = dict(before1)
        before2[region] = clamp(payload, REGION_BUDGETS[region])
        before2["response_draft"] = ""
        after2 = dict(before2)
        after2["response_draft"] = clamp(f"{region} updated.", REGION_BUDGETS["response_draft"])
        episodes.append(make_episode(6, i, "delta_plus_growth", [
            make_tick(before1, after1, before1["input_window"]),
            make_tick(before2, after2, "harness applied plus call"),
        ]))
    return episodes


def build_semantic_delta(count: int, rng: random.Random, bundle: SourceBundle) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        knowledge = choose_snippet(rng, bundle.snippets, REGION_BUDGETS["structured_knowledge"] - 12)
        before = random_state(rng, bundle, i, dense=False, weird=False)
        before["input_window"] = clamp("Jeff: answer using structured knowledge.", REGION_BUDGETS["input_window"])
        before["structured_knowledge"] = clamp(f"Knowledge: {knowledge}", REGION_BUDGETS["structured_knowledge"])
        if rng.random() < 0.5:
            before["task_state"] = "Task: answer from structured knowledge."
        after = dict(before)
        after["response_draft"] = clamp(f"Knowledge says: {knowledge}", REGION_BUDGETS["response_draft"])
        if rng.random() < 0.35:
            after["tool_results"] = ""
        episodes.append(make_episode(7, i, "delta_semantic_answer", [
            make_tick(before, after, before["input_window"]),
        ]))
    return episodes


def validate_episodes(episodes: list[dict[str, Any]]) -> None:
    for episode in episodes:
        parse_episode(episode, repair=False)


def write_jsonl(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")


def write_stage_files(out: Path, episodes: list[dict[str, Any]]) -> None:
    by_stage: dict[int, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_stage.setdefault(int(episode["stage"]), []).append(episode)
    stem = out.with_suffix("")
    for stage, items in sorted(by_stage.items()):
        write_jsonl(stem.parent / f"{stem.name}_stage{stage}.jsonl", items)


def report(episodes: list[dict[str, Any]], bundle: SourceBundle, out: Path) -> None:
    lessons = Counter(str(ep.get("lesson", "?")) for ep in episodes)
    region_used = Counter()
    total_ticks = 0
    for ep in episodes:
        for tick in ep.get("ticks", []):
            total_ticks += 1
            for region, text in (tick.get("state_after") or {}).items():
                if text:
                    region_used[region] += 1
    print(f"source files read: {bundle.files_read}")
    if bundle.sqlite_tables:
        print(f"sqlite tables read: {bundle.sqlite_tables}")
    print(f"snippets: {len(bundle.snippets)}   harvested Axon states: {len(bundle.states)}")
    print(f"episodes: {len(episodes)}   ticks: {total_ticks}")
    print(f"wrote: {out}")
    print("lesson mix:")
    for lesson, n in lessons.most_common():
        print(f"  {lesson:24s} {n:5d}")
    print("region usage in after-states:")
    for region in SHARED_ORDER:
        print(f"  {region:22s} {region_used[region]:5d}/{total_ticks}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build Axon v7 full-state copy and delta curriculum")
    ap.add_argument("--source", default="D:/00",
                    help="file or folder to harvest; can also be an existing Axon JSONL dataset")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-files", type=int, default=500)
    ap.add_argument("--max-items", type=int, default=2000)
    ap.add_argument("--chunk-chars", type=int, default=220)
    ap.add_argument("--max-file-bytes", type=int, default=262144)
    ap.add_argument("--copy", type=int, default=256,
                    help="stage 1 exact full-state copy lessons")
    ap.add_argument("--weird-copy", type=int, default=256,
                    help="stage 2 dense/weird exact full-state copy lessons")
    ap.add_argument("--response", type=int, default=160,
                    help="stage 3 response_draft delta lessons")
    ap.add_argument("--evict", type=int, default=128,
                    help="stage 4 clear/evict one region lessons")
    ap.add_argument("--rewrite", type=int, default=128,
                    help="stage 5 same-width non-draft rewrite lessons")
    ap.add_argument("--plus", type=int, default=128,
                    help="stage 6 two-tick ++ growth lessons")
    ap.add_argument("--semantic", type=int, default=160,
                    help="stage 7 answer-from-state lessons")
    ap.add_argument("--no-stage-files", action="store_true")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    source = Path(args.source)
    out = Path(args.out)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    if not source.exists():
        print(f"warning: source does not exist, using built-in seed snippets: {source}")
        bundle = SourceBundle(snippets=list(DEFAULT_SNIPPETS), states=[])
    else:
        bundle = load_source(
            source,
            max_files=max(1, int(args.max_files)),
            max_items=max(1, int(args.max_items)),
            chunk_chars=max(64, int(args.chunk_chars)),
            max_file_bytes=max(4096, int(args.max_file_bytes)),
        )

    rng = random.Random(int(args.seed))
    episodes: list[dict[str, Any]] = []
    episodes.extend(build_copy_stage(
        1, "state_copy_full", int(args.copy), rng, bundle,
        dense=False, weird=False))
    episodes.extend(build_copy_stage(
        2, "state_copy_weird", int(args.weird_copy), rng, bundle,
        dense=True, weird=True))
    episodes.extend(build_response_delta(int(args.response), rng, bundle))
    episodes.extend(build_evict_delta(int(args.evict), rng, bundle))
    episodes.extend(build_rewrite_delta(int(args.rewrite), rng, bundle))
    episodes.extend(build_plus_growth(int(args.plus), rng, bundle))
    episodes.extend(build_semantic_delta(int(args.semantic), rng, bundle))

    validate_episodes(episodes)
    write_jsonl(out, episodes)
    if not args.no_stage_files:
        write_stage_files(out, episodes)
    report(episodes, bundle, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
