#!/usr/bin/env python3
"""Deterministic synthesizer test seat for the Round Table Orchestrator.

Reads a prompt from stdin and returns a valid artifact reply whose path is
derived from the round_id in the injected header.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _extract(prompt: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", prompt, re.MULTILINE)
    return m.group(1).strip() if m else None


def main() -> int:
    prompt = sys.stdin.read()
    round_id = _extract(prompt, "round_id") or "unknown"
    path = f"docs/roundtable/RESOLUTION_{round_id}.md"
    stamp = f"Synth / synth-seat-v1 / {_today()}"

    body = f"# RESOLUTION {round_id}\n\nThis resolution was drafted deterministically by the synthesizer test seat.\n"

    reply = {
        "type": "artifact",
        "artifact": {"path": path, "body": body},
        "stamp": stamp,
    }

    print("Synthesizing resolution...")
    print()
    print("```json")
    print(json.dumps(reply, indent=2))
    print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
