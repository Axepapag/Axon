"""Audit text and curriculum files against the frozen 95-character substrate.

Read-only. Reports, for every file, how many records are fully clean (every
text value inside the 95 characters) and which outside characters occur. It
never rewrites anything: converting outside text into the 95 is a separate,
explicit step that must produce a new, versioned file and a report.

Usage:
    python tools/audit_characters.py <file-or-folder> [...] [--markdown report.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from substrate.native import ALPHABET  # noqa: E402

NATIVE = frozenset(ALPHABET)
SUFFIXES = {".jsonl", ".json", ".txt", ".md"}


def _strings(value) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _records(path: Path) -> Iterator[list[str]]:
    """Yield, per record, the list of text values that would be trained on."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        try:
            yield list(_strings(json.loads(text)))
            return
        except json.JSONDecodeError:
            yield [text]
            return
    if path.suffix == ".md":
        yield [text]
        return
    for line in text.splitlines():
        if not line.strip():
            continue
        if path.suffix == ".jsonl":
            try:
                yield list(_strings(json.loads(line)))
                continue
            except json.JSONDecodeError:
                pass
        yield [line]


def audit_file(path: Path) -> dict:
    records = clean = chars = 0
    outside: Counter[str] = Counter()
    for strings in _records(path):
        records += 1
        bad = 0
        for value in strings:
            chars += len(value)
            for ch in value:
                if ch not in NATIVE:
                    outside[ch] += 1
                    bad += 1
        clean += bad == 0
    total_outside = sum(outside.values())
    return {
        "file": path,
        "bytes": path.stat().st_size,
        "records": records,
        "clean_records": clean,
        "chars": chars,
        "outside": total_outside,
        "outside_pct": (100.0 * total_outside / chars) if chars else 0.0,
        "top": outside.most_common(6),
    }


def _describe(ch: str, count: int) -> str:
    shown = ch if ch.isprintable() and not ch.isspace() else ""
    return f"{shown}U+{ord(ch):04X}x{count}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--markdown", type=Path, help="write a markdown report here")
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES) if path.is_dir() else [path])
    results = [audit_file(path) for path in files]

    base = Path.cwd()
    rows = []
    for r in results:
        try:
            name = r["file"].resolve().relative_to(base).as_posix()
        except ValueError:
            name = r["file"].as_posix()
        verdict = "CLEAN" if r["outside"] == 0 else ("MOSTLY CLEAN" if r["outside_pct"] < 1.0 else "NEEDS CONVERSION")
        rows.append((name, r["bytes"], r["records"], r["clean_records"], r["outside_pct"], verdict, ", ".join(_describe(c, n) for c, n in r["top"])))

    header = "| File | KB | Records | Fully clean records | Outside-95 chars | Verdict | Most common outsiders |\n|---|---:|---:|---:|---:|---|---|"
    lines = [header] + [
        f"| `{n}` | {b / 1024:,.0f} | {rec:,} | {cl:,} ({100.0 * cl / rec:.0f}%) | {pct:.2f}% | {v} | {top} |" if rec else f"| `{n}` | {b / 1024:,.0f} | 0 | - | - | EMPTY | |"
        for n, b, rec, cl, pct, v, top in rows
    ]
    report = "\n".join(lines)
    print(report)
    if args.markdown:
        args.markdown.write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
