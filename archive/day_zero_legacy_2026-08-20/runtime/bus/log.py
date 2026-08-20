"""Append-only JSONL helpers for durable bus logs."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON object to a JSONL file, flushing to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield every JSON object in a JSONL file."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl_reverse(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    """Return the most recent JSONL entries first."""
    records = list(read_jsonl(path))
    if limit and limit < len(records):
        records = records[-limit:]
    return list(reversed(records))


def read_jsonl_after_id(
    path: Path, after_id: str | None, id_field: str = "msg_id", limit: int = 0
) -> list[dict[str, Any]]:
    """Return entries whose id_field sorts strictly after after_id."""
    records = list(read_jsonl(path))
    if after_id is None:
        result = records
    else:
        result = [r for r in records if r.get(id_field, "") > after_id]
    if limit and limit < len(result):
        return result[-limit:]
    return result


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON with tmp+replace durability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
