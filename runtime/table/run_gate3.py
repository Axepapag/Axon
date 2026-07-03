#!/usr/bin/env python3
"""Gate 3 runner: live smoke with the real Dream Team bus and a real kimi seat.

STOP — read before running:
- This script wakes a REAL agent (kimi) and touches the LIVE bus.
- Per ORCHESTRATOR_SPEC.md section 9, gate 3 requires an officer present.
- Do not execute this file in an automated test run. It is intentionally NOT
  collected by pytest (no `test_` prefix) and must be invoked manually.

What it will do when executed:
1. Require BUS_TOKEN in the process environment (never read from files).
2. Open a round `gate3-live-<date>` with brief `tests/fixtures/table/test_brief.md`,
   seats=["echo", "kimi"], synthesizer="kimi", max_cycles=1, offline=False.
3. Connect to ws://127.0.0.1:8765/ws/bus as client_id "table".
4. Run exactly one full cycle (echo turn, then kimi turn), ONE SUBPROCESS AT A TIME.
5. Parse kimi's real reply and publish `agent.kimi.turn.completed` plus the
   relevant committee.* topic to the bus.
6. Read back events from the bus and verify at least one kimi event is present.
7. Render the transcript to markdown.
8. Wake kimi again as synthesizer to draft a resolution artifact under
   `docs/roundtable/RESOLUTION_gate3-live-<date>.md`.
9. Publish `committee.resolution.proposed` and leave the round in status
   `awaiting_convener`.
10. Report success/failure to stdout and exit 0 on success, non-zero on failure.

Run only when:
- An officer (Claude or Codex) is present and has approved the live wake.
- The bus is healthy: `Invoke-RestMethod http://127.0.0.1:8765/health`.
- BUS_TOKEN is exported in the shell that launches this script.

Example:
    $env:BUS_TOKEN = "..."
    python runtime/table/run_gate3.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from runtime.table import buslink, waker


BRIEF = Path("tests/fixtures/table/test_brief.md")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def main() -> int:
    if "BUS_TOKEN" not in os.environ:
        print("[gate3] HALT: BUS_TOKEN must be set in the environment.", file=sys.stderr)
        return 2

    if not BRIEF.exists():
        print(f"[gate3] HALT: brief not found at {BRIEF}", file=sys.stderr)
        return 2

    round_id = f"gate3-live-{_today()}"
    table = waker.RoundTable()

    print(f"[gate3] opening round {round_id}")
    table.open_round(
        round_id=round_id,
        brief_path=str(BRIEF),
        seats=["echo", "kimi"],
        synthesizer="kimi",
        max_cycles=1,
        per_turn_timeout_s=600,
        budgets={"max_invocations": 6, "board_view_events": 40, "board_view_chars": 24000},
        offline=False,
        officers=["codex", "claude"],
    )

    print("[gate3] running one cycle (echo, then live kimi)...")
    result = table.run(round_id)
    print(f"[gate3] run result: {result}")

    if not result.get("halted"):
        print("[gate3] FAIL: round did not halt as expected", file=sys.stderr)
        return 1

    # Verify kimi events were persisted on the bus.
    print("[gate3] reading back bus events...")
    link = buslink.BusLink(offline=False)
    events = asyncio.run(link.replay_events(patterns=["agent.kimi.*", "committee.*"], timeout=5.0))
    kimi_events = [e for e in events if e.get("topic", "").startswith("agent.kimi.")]
    print(f"[gate3] replayed {len(events)} events, {len(kimi_events)} kimi events")

    if not kimi_events:
        print("[gate3] FAIL: no kimi events read back from bus", file=sys.stderr)
        return 1

    resolution_path = result.get("resolution_path")
    if not resolution_path or not Path(resolution_path).exists():
        print("[gate3] FAIL: synthesizer did not produce a resolution file", file=sys.stderr)
        return 1

    print(f"[gate3] SUCCESS: resolution at {resolution_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
