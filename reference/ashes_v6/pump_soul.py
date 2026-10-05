#!/usr/bin/env python3
"""Pump the soul curriculum: Axon's episodic memory -> training text.

Extracts the previous Axon's lived experience into plain-text lessons at
D:/ashes/Datasets/soul/ so the schoolhouse can bake it into core weights:

  soul_turns.txt      - real Jeff<->Axon conversational turns (episodes)
  soul_summaries.txt  - the episode summaries the old Axon composed
  soul_journal.txt    - his personal journal entries
  soul_axonm.txt      - clean dialogue recovered from the AxonM era

Tool-call scaffolding, JSON, and machine noise are filtered out; only
natural English survives. Lines are deduplicated and length-bounded.
The output is text - the trainer reads it character by character like
everything else. No vectors are imported (the old external embeddings
are discarded per the no-teacher rule).
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

OUT_DIR = Path("D:/ashes/Datasets/soul")
EPISODIC_DB = Path("D:/00/axon_episodic_memory.db")
PERSONAL_LOG = Path("D:/00/axon_personal_log.json")
AXONM_DB = Path("D:/AxonM/Memory/axon_memory.db")

NOISE_MARKERS = ("```", "[TASK RESULT", "[SYSTEM", '{"', "Tool:", "Status:",
                 "MSN-", "TASK-", "state_hash", "://", "\\\\", "_json",
                 "PRAGMA", "SELECT ", "def ", "import ")
MIN_LEN, MAX_LEN = 12, 400


def is_natural(text: str) -> bool:
    if not text or not (MIN_LEN <= len(text) <= MAX_LEN):
        return False
    if any(m in text for m in NOISE_MARKERS):
        return False
    letters = sum(1 for c in text if c.isalpha() or c == " ")
    return letters / len(text) >= 0.75


def clean_line(text: str) -> str:
    text = re.sub(r"</?[a-z_]+>", " ", text)          # <thought> etc. markup
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)   # strip bold markdown
    text = re.sub(r"^[#\-\*>\s]+", "", text)          # list/heading prefixes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def harvest_episodes() -> tuple[list[str], list[str]]:
    turns: list[str] = []
    summaries: list[str] = []
    con = sqlite3.connect(EPISODIC_DB)
    cur = con.cursor()
    for summary, payload in cur.execute("SELECT summary, payload_json FROM episodes"):
        if summary:
            for line in str(summary).splitlines():
                line = clean_line(line)
                if is_natural(line):
                    summaries.append(line)
        if payload:
            try:
                data = json.loads(payload)
            except Exception:
                continue
            for turn in data.get("turns", []) if isinstance(data, dict) else []:
                if not isinstance(turn, dict):
                    continue
                for role in ("user", "assistant"):
                    raw = turn.get(role)
                    if not isinstance(raw, str):
                        continue
                    for sent in split_sentences(clean_line(raw)):
                        if is_natural(sent):
                            turns.append(sent)
    con.close()
    return turns, summaries


def harvest_journal() -> list[str]:
    out: list[str] = []
    if not PERSONAL_LOG.exists():
        return out
    data = json.loads(PERSONAL_LOG.read_text(encoding="utf-8"))
    for entry in data.get("entries", []):
        content = entry.get("content") if isinstance(entry, dict) else None
        if not isinstance(content, str):
            continue
        for sent in split_sentences(clean_line(content)):
            if is_natural(sent):
                out.append(sent)
    return out


def harvest_axonm() -> list[str]:
    out: list[str] = []
    if not AXONM_DB.exists():
        return out
    con = sqlite3.connect(AXONM_DB)
    for (content,) in con.execute("SELECT content FROM messages"):
        if not isinstance(content, str):
            continue
        if any(m in content for m in NOISE_MARKERS):
            continue
        for sent in split_sentences(clean_line(content)):
            if is_natural(sent):
                out.append(sent)
    con.close()
    return out


def write_dedup(path: Path, lines: list[str]) -> int:
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            kept.append(line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return len(kept)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    turns, summaries = harvest_episodes()
    journal = harvest_journal()
    axonm = harvest_axonm()
    n_turns = write_dedup(OUT_DIR / "soul_turns.txt", turns)
    n_sum = write_dedup(OUT_DIR / "soul_summaries.txt", summaries)
    n_journal = write_dedup(OUT_DIR / "soul_journal.txt", journal)
    n_axonm = write_dedup(OUT_DIR / "soul_axonm.txt", axonm)
    print(f"soul curriculum pumped to {OUT_DIR}")
    print(f"  soul_turns.txt      {n_turns:>7,} lines (episode dialogue)")
    print(f"  soul_summaries.txt  {n_sum:>7,} lines (his own summaries)")
    print(f"  soul_journal.txt    {n_journal:>7,} lines (personal journal)")
    print(f"  soul_axonm.txt      {n_axonm:>7,} lines (AxonM-era dialogue)")


if __name__ == "__main__":
    main()
