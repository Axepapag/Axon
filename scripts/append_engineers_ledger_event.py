#!/usr/bin/env python3
"""Validate and append exactly one immutable engineer-ledger JSONL event."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

SCHEMA = "axon-engineers-ledger-event-v1"
REQUIRED = {
    "schema",
    "event_id",
    "timestamp",
    "agent",
    "turn",
    "actions",
    "files",
    "verification",
    "decisions",
    "flags",
    "next_steps",
    "identity_stamp",
}


def append_event(ledger: Path, event_path: Path) -> str:
    raw = event_path.read_text(encoding="utf-8")
    if "\n" in raw.rstrip("\n"):
        raise ValueError("pending event must contain exactly one physical JSON line")
    event = json.loads(raw)
    missing = REQUIRED - set(event)
    if missing:
        raise ValueError(f"pending event is missing required fields: {sorted(missing)}")
    if event["schema"] != SCHEMA:
        raise ValueError("unsupported engineer-ledger event schema")
    event_id = str(event["event_id"])
    if not event_id:
        raise ValueError("event_id must be non-empty")
    canonical = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    existing = ledger.read_bytes() if ledger.exists() else b""
    if existing and not existing.endswith(b"\n"):
        raise ValueError("canonical ledger does not end at a complete JSONL boundary")
    for line_number, line in enumerate(existing.decode("utf-8").splitlines(), 1):
        if not line.strip():
            # Historical blank lines are immutable too.  Ignore them while
            # validating uniqueness; never normalize or rewrite the ledger.
            continue
        observed = json.loads(line)
        if observed.get("event_id") == event_id:
            if line == canonical:
                return event_id
            raise ValueError(f"event_id already exists with different content at line {line_number}")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("ab") as handle:
        handle.write(canonical.encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    if ledger.read_bytes().splitlines()[-1].decode("utf-8") != canonical:
        raise RuntimeError("canonical ledger append verification failed")
    return event_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event", type=Path)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "roundtable" / "ENGINEERS_LEDGER_CANONICAL.jsonl",
    )
    args = parser.parse_args()
    print(append_event(args.ledger.resolve(), args.event.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
