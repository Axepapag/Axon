#!/usr/bin/env python3
"""Echo test seat for the Round Table Orchestrator.

Reads a prompt from stdin, extracts the round_id and seat from the injected
header, and returns a deterministic valid reply containing a short text post
and the required identity stamp.
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
    seat = _extract(prompt, "your_seat") or "echo"
    cycle = _extract(prompt, "cycle") or "0"
    stamp = f"Echo / echo-seat-v1 / {_today()}"

    reply = {
        "type": "post",
        "text": f"Echo reply from {seat} for round {round_id} cycle {cycle}.",
        "stamp": stamp,
    }

    # Emit any free text first, then the fenced JSON block as required by the
    # reply contract.
    print(f"This is the echo seat ({seat}) speaking.")
    print()
    print("```json")
    print(json.dumps(reply, indent=2))
    print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
