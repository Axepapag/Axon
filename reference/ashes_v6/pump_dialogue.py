#!/usr/bin/env python3
"""Pump the phase-2 authorship curriculum: real dialogue pairs.

Extracts (context -> reply) pairs from the previous Axon's episodic
memory and the schoolhouse episode files into tab-separated lessons:

    D:/ashes/Datasets/dialogue/soul_pairs.txt      (episodic memory)
    D:/ashes/Datasets/dialogue/schoolhouse_pairs.txt (sh_*.jsonl episodes)

Format: one lesson per line, "context<TAB>reply". The trainer turns each
into an authorship field: the context region intact, the reply region
blanked, and the lesson is WRITE THE REPLY - pure span-restoration aimed
at authorship, no prediction, no tokens. This is the bridge from a core
that can spell to a core that answers.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

OUT_DIR = Path("D:/ashes/Datasets/dialogue")
EPISODIC_DB = Path("D:/00/axon_episodic_memory.db")
SH_GLOB = Path("D:/00/Raw/corpus")

NOISE_MARKERS = ("```", "[TASK RESULT", "[SYSTEM", '{"', "Tool:", "Status:",
                 "MSN-", "TASK-", "state_hash", "://", "\\\\")
CTX_MAX, REPLY_MAX = 220, 220
MIN_LEN = 8


def clean(text: str) -> str:
    text = re.sub(r"</?[a-z_]+>", " ", str(text))
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def natural(text: str, max_len: int) -> str | None:
    """Clean and bound one side of a pair; None if it isn't natural text."""
    text = clean(text)
    if len(text) < MIN_LEN:
        return None
    if any(m in text for m in NOISE_MARKERS):
        return None
    if "##" in text or "---" in text or any(ord(c) > 0x2000 for c in text):
        return None  # markdown scaffolding / emoji / box-drawing junk
    letters = sum(1 for c in text if c.isalpha() or c in " .,!?'")
    if letters / len(text) < 0.8:
        return None
    if len(text) > max_len:
        # cut at the last sentence end inside the budget
        cut = max(text.rfind(e, 0, max_len) for e in ".!?")
        text = text[:cut + 1] if cut > max_len // 3 else text[:max_len]
    return text.strip()


def harvest_episodes() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if not EPISODIC_DB.exists():
        return pairs
    con = sqlite3.connect(EPISODIC_DB)
    for (payload,) in con.execute("SELECT payload_json FROM episodes"):
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except Exception:
            continue
        for turn in data.get("turns", []) if isinstance(data, dict) else []:
            if not isinstance(turn, dict):
                continue
            ctx = natural(turn.get("user", ""), CTX_MAX)
            reply = natural(turn.get("assistant", ""), REPLY_MAX)
            if ctx and reply:
                pairs.append((ctx, reply))
    con.close()
    return pairs


def harvest_schoolhouse() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for path in sorted(SH_GLOB.glob("sh_*.jsonl")):
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ctx = natural(obj.get("jeff_voice", ""), CTX_MAX)
                reply = natural(obj.get("axon_response", ""), REPLY_MAX)
                if ctx and reply:
                    pairs.append((ctx, reply))
    return pairs


def write_pairs(path: Path, pairs: list[tuple[str, str]]) -> int:
    seen: set[str] = set()
    lines: list[str] = []
    for ctx, reply in pairs:
        key = (ctx + "\t" + reply).lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{ctx}\t{reply}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_soul = write_pairs(OUT_DIR / "soul_pairs.txt", harvest_episodes())
    n_sh = write_pairs(OUT_DIR / "schoolhouse_pairs.txt", harvest_schoolhouse())
    print(f"dialogue curriculum pumped to {OUT_DIR}")
    print(f"  soul_pairs.txt         {n_soul:>6,} context->reply lessons")
    print(f"  schoolhouse_pairs.txt  {n_sh:>6,} context->reply lessons")


if __name__ == "__main__":
    main()
