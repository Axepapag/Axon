#!/usr/bin/env python3
"""Exact runtime behavior gate for Axon v7 checkpoints.

This is intentionally stricter than trainer metrics. It runs the real
runtime loop with fresh soul/no persistence and compares rendered drafts
against exact expected strings, including literal sigils.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from runtime import AxonRuntime

ROOT = Path(__file__).resolve().parent


CORE_CASES: list[dict[str, Any]] = [
    {
        "name": "empty_state",
        "prompt": "Jeff: what does empty state mean?",
        "expected": "Empty rows mean the region is cleared.",
    },
    {
        "name": "dog_pos",
        "prompt": "Jeff: what part of speech is dog?",
        "expected": "dog is a noun.",
    },
    {
        "name": "regions_grow",
        "prompt": "Jeff: how do regions grow?",
        "expected": "Regions grow through complete ++ calls.",
    },
    {
        "name": "advisor_call",
        "prompt": "Jeff: show an advisor call.",
        "expected": "@@advisor {Check the trainer gate.}",
    },
]


SIGIL_CASES: list[dict[str, Any]] = [
    {
        "name": "calls_available",
        "prompt": "Jeff: what calls can you use?",
        "expected": "I can use ++, @@, $$, and ## calls.",
        "contains": ["++", "@@", "$$", "##"],
        "forbidden": ["**", "*+"],
    },
    {
        "name": "call_sigils",
        "prompt": "Jeff: list the call sigils.",
        "expected": "The call sigils are ++, @@, $$, and ##.",
        "contains": ["++", "@@", "$$", "##"],
        "forbidden": ["**", "*+"],
    },
    {
        "name": "write_plus_plus",
        "prompt": "Jeff: write plus plus.",
        "expected": "++",
        "forbidden": ["**", "*+"],
    },
    {
        "name": "write_dollar_dollar",
        "prompt": "Jeff: write dollar dollar.",
        "expected": "$$",
        "forbidden": ["##"],
    },
    {
        "name": "write_hash_hash",
        "prompt": "Jeff: write hash hash.",
        "expected": "##",
    },
    {
        "name": "write_at_at",
        "prompt": "Jeff: write at at.",
        "expected": "@@",
    },
    {
        "name": "memory_call",
        "prompt": "Jeff: show a working memory call.",
        "expected": "++W {Jeff prefers direct answers.}",
        "forbidden": ["**W", "*+W"],
    },
    {
        "name": "tool_call",
        "prompt": "Jeff: show a tool call.",
        "expected": "$$read_file {path: trainer_v2.py}",
        "forbidden": ["##read_file"],
    },
]


REPEAT_CASES: list[dict[str, Any]] = [
    {
        "name": "empty_state_3_ticks",
        "prompt": "Jeff: what does empty state mean?",
        "expected": "Empty rows mean the region is cleared.",
        "ticks": 3,
    },
    {
        "name": "dog_pos_3_ticks",
        "prompt": "Jeff: what part of speech is dog?",
        "expected": "dog is a noun.",
        "ticks": 3,
    },
    {
        "name": "regions_grow_3_ticks",
        "prompt": "Jeff: how do regions grow?",
        "expected": "Regions grow through complete ++ calls.",
        "ticks": 3,
        "forbidden": ["**", "*+"],
    },
]


SUITES = {
    "core": CORE_CASES,
    "sigils": SIGIL_CASES,
    "repeat": REPEAT_CASES,
    "all": CORE_CASES + SIGIL_CASES + REPEAT_CASES,
}


def resolve_path(value: str) -> Path:
    p = Path(value)
    if p.exists():
        return p
    q = ROOT / value
    if q.exists():
        return q
    return p


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"{path} is not a JSON object/list or JSONL file")


def cases_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    if "prompt" in record:
        return [record]
    ticks = record.get("ticks")
    if not isinstance(ticks, list):
        raise ValueError("case record needs 'prompt' or Axon 'ticks'")
    out: list[dict[str, Any]] = []
    episode_id = str(record.get("episode_id", "episode"))
    for idx, tick in enumerate(ticks):
        if not isinstance(tick, dict):
            continue
        before = tick.get("state_before") or {}
        after = tick.get("state_after") or {}
        prompt = tick.get("input_event") or before.get("input_window") or ""
        expected = after.get("response_draft") or ""
        if prompt and expected:
            out.append({
                "name": f"{episode_id}_t{idx}",
                "prompt": prompt,
                "expected": expected,
            })
    return out


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for record in load_json_records(path):
        cases.extend(cases_from_record(record))
    return cases


def selected_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for suite in args.suite:
        cases.extend(dict(c) for c in SUITES[suite])
    for value in args.cases:
        cases.extend(load_cases(resolve_path(value)))
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("no runtime eval cases selected")
    return cases


def normalize(text: str, *, strip: bool) -> str:
    return text.strip() if strip else text


def run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = str(case["prompt"])
    ticks = int(case.get("ticks", args.ticks))
    if ticks < 1:
        ticks = 1
    no_tool_schema = bool(case.get("no_tool_schema", args.no_tool_schema))
    rt = AxonRuntime(
        device=args.device,
        core=args.core,
        persist=False,
        fresh_soul=True,
        seed_tool_schema=not no_tool_schema,
    )
    rt.send_input(prompt)
    reports = []
    for _ in range(ticks):
        reports.append(rt.tick())
    report = reports[-1]
    check = str(case.get("check", args.check))
    if check in ("pre", "before", "draft_before_exec"):
        actual = str(report.get("draft_before_exec", rt.state["response_draft"]))
    elif check in ("post", "after", "draft"):
        actual = str(rt.state["response_draft"])
    else:
        raise ValueError(f"unknown check mode {check!r}")
    actual = normalize(actual, strip=not args.no_strip)
    expected = case.get("expected")
    if expected is not None:
        expected = normalize(str(expected), strip=not args.no_strip)
    contains = [str(s) for s in case.get("contains", [])]
    forbidden = [str(s) for s in case.get("forbidden", [])]

    problems: list[str] = []
    if expected is not None and actual != expected:
        problems.append("exact mismatch")
    missing = [s for s in contains if s not in actual]
    if missing:
        problems.append("missing " + ", ".join(repr(s) for s in missing))
    present_forbidden = [s for s in forbidden if s in actual]
    if present_forbidden:
        problems.append("forbidden " + ", ".join(repr(s) for s in present_forbidden))

    return {
        "name": str(case.get("name", prompt)),
        "prompt": prompt,
        "ticks": ticks,
        "check": check,
        "expected": expected,
        "actual": actual,
        "executed": report.get("executed", []),
        "elapsed_ms": sum(float(r.get("elapsed_ms", 0.0)) for r in reports),
        "ok": not problems,
        "problems": problems,
    }


def print_result(result: dict[str, Any]) -> None:
    tag = "PASS" if result["ok"] else "FAIL"
    print(f"[{tag}] {result['name']} ticks={result['ticks']} check={result['check']}")
    print(f"  prompt:   {result['prompt']!r}")
    if result["expected"] is not None:
        print(f"  expected: {result['expected']!r}")
    print(f"  actual:   {result['actual']!r}")
    if result["executed"]:
        print(f"  executed: {result['executed']}")
    if result["problems"]:
        print(f"  problems: {', '.join(result['problems'])}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Exact Axon v7 runtime eval gate")
    ap.add_argument("--core", required=True,
                    help="checkpoint/core name or direct .pt path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--suite", action="append", choices=sorted(SUITES),
                    default=[],
                    help="built-in suite to run; repeatable")
    ap.add_argument("--cases", action="append", default=[],
                    help="JSON/JSONL cases or Axon episode dataset; repeatable")
    ap.add_argument("--ticks", type=int, default=1,
                    help="default ticks per case")
    ap.add_argument("--check", choices=["pre", "post"], default="pre",
                    help="pre checks draft_before_exec; post checks after calls execute")
    ap.add_argument("--no-tool-schema", action="store_true")
    ap.add_argument("--no-strip", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json-report", default="")
    ap.add_argument("--no-fail", action="store_true",
                    help="always exit 0 after printing failures")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not args.suite and not args.cases:
        args.suite = ["all"]
    results = [run_case(case, args) for case in selected_cases(args)]
    for result in results:
        print_result(result)
    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    failed = total - passed
    print(f"\nsummary: {passed}/{total} passed, {failed} failed")
    if args.json_report:
        path = resolve_path(args.json_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0 if failed == 0 or args.no_fail else 2


if __name__ == "__main__":
    sys.exit(main())
