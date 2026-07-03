"""Append-only transcript: JSONL + rendered markdown.

Transcripts are append-only.  Any attempt to rewrite an existing file is
refused.  New files may be created; missing files may be initialized.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AppendOnlyError(Exception):
    """Raised when a caller attempts to overwrite existing transcript data."""


TranscriptFileName = "transcript.jsonl"
RenderedFileName = "transcript.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _init_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def append_entries(round_dir: str | Path, entries: list[dict[str, Any]]) -> Path:
    """Append entries to transcript.jsonl and re-render transcript.md.

    Raises AppendOnlyError if the caller supplies fewer entries than already
    exist on disk (a rewrite / truncation attempt).
    """
    d = Path(round_dir)
    jsonl_path = d / TranscriptFileName
    md_path = d / RenderedFileName
    _init_file(jsonl_path)
    _init_file(md_path)

    existing = read_jsonl(jsonl_path)
    # Appending zero entries is allowed; it just re-renders.
    if entries is not existing and len(entries) < len(existing):
        raise AppendOnlyError(
            f"refusing to truncate transcript: {len(existing)} existing entries, "
            f"{len(entries)} supplied"
        )

    # If the caller supplied the full list, append only the tail beyond existing.
    new_count = max(0, len(entries) - len(existing))
    if new_count:
        with jsonl_path.open("a", encoding="utf-8") as f:
            for entry in entries[len(existing):]:
                f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    # Always render from the authoritative jsonl.
    all_entries = read_jsonl(jsonl_path)
    md_path.write_text(render_markdown(all_entries), encoding="utf-8")
    return jsonl_path


def render_markdown(entries: list[dict[str, Any]]) -> str:
    lines = ["# Round Table Transcript\n", f"generated_at: {_now()}\n", "---\n"]
    for idx, entry in enumerate(entries, start=1):
        seat = entry.get("seat", "unknown")
        cycle = entry.get("cycle", "?")
        turn_type = entry.get("type", "turn")
        status = entry.get("status", "completed")
        lines.append(f"## {idx}. {seat} (cycle {cycle}, {turn_type}) — {status}\n")
        for key in ("action", "error", "metadata", "bus_topics"):
            if key in entry:
                lines.append(f"**{key}**:\n```json\n{json.dumps(entry[key], ensure_ascii=False, indent=2)}\n```\n")
        lines.append("\n")
    return "\n".join(lines)


def get_transcript_path(round_dir: str | Path) -> Path:
    return Path(round_dir) / TranscriptFileName


def get_rendered_path(round_dir: str | Path) -> Path:
    return Path(round_dir) / RenderedFileName
