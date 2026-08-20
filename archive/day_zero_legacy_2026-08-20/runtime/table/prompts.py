"""Prompt builder for the Round Table Orchestrator."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ContractPath = Path("docs") / "WORKING_CONTRACT.md"


@dataclass
class BudgetView:
    max_invocations: int
    board_view_events: int
    board_view_chars: int


@dataclass
class PromptContext:
    round_id: str
    cycle: int
    max_cycles: int
    seat: str
    seat_display: str
    standing_question: str
    officers: list[str]
    synthesizer: str
    budgets: BudgetView
    doctrine_stamp: dict[str, str]
    brief_text: str
    transcript_entries: list[dict[str, Any]]
    unread_dms: list[dict[str, Any]]


def load_preamble(contract_path: str | Path | None = None) -> str:
    """Return the Working Contract MISSION PREAMBLE verbatim."""
    path = Path(contract_path) if contract_path else ContractPath
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"## MISSION PREAMBLE.*?\n(?P<preamble>(?:> .*(?:\n|$))+)",
        text,
        re.DOTALL,
    )
    if not match:
        # Fallback: return a short, safe notice rather than fail the round.
        return (
            "> You are working under the Axon Working Contract.\n"
            "> Hierarchy: Jeff > SOURCE_OF_TRUTH > resolutions > mission > judgment.\n"
            "> HALT AND FLAG anything impossible, conflicting, or ambiguous.\n"
        )
    preamble = match.group("preamble")
    # Strip the leading "> " markdown quote markers but keep line structure.
    lines = []
    for line in preamble.splitlines():
        if line.startswith("> "):
            lines.append(line[2:])
        elif line == ">":
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def json_line(entry: dict[str, Any]) -> str:
    return json.dumps(entry, ensure_ascii=False, sort_keys=True)


def _render_digest(entries: list[dict[str, Any]], budget: BudgetView) -> tuple[str, dict[str, Any]]:
    """Render transcript digest within board_view_* budgets.

    Returns the rendered digest plus a metadata dict recording truncation.
    """
    total = len(entries)
    # Determine how many entries fit under the char budget, most recent last.
    selected: list[dict[str, Any]] = []
    used = 0
    header_overhead = 80
    remaining_chars = max(0, budget.board_view_chars - header_overhead)

    for entry in reversed(entries):
        line = json_line(entry)
        if used + len(line) + 1 <= remaining_chars and len(selected) < budget.board_view_events:
            selected.append(entry)
            used += len(line) + 1
        else:
            break

    selected.reverse()
    truncated_count = total - len(selected)

    if not selected:
        digest_text = "(board is empty)"
    else:
        digest_text = "\n".join(json_line(e) for e in selected)

    if truncated_count > 0:
        notice = f"digest truncated: showing last {len(selected)} of {total} entries"
        digest_text = f"{notice}\n{digest_text}"
        truncated = True
    else:
        notice = ""
        truncated = False

    return digest_text, {
        "total_entries": total,
        "shown_entries": len(selected),
        "truncated_count": truncated_count,
        "truncated": truncated,
        "truncation_notice": notice,
        "used_chars": used + header_overhead,
        "budget_chars": budget.board_view_chars,
    }


def build_prompt(ctx: PromptContext) -> tuple[str, dict[str, Any]]:
    """Assemble a turn prompt per the spec.

    Returns (prompt_text, metadata) where metadata records budgets and
    truncation so the waker can append it to the transcript.
    """
    preamble = load_preamble()

    officers_str = ", ".join(ctx.officers) if ctx.officers else "(none designated)"
    budgets_str = (
        f"max_invocations={ctx.budgets.max_invocations}, "
        f"board_view_events={ctx.budgets.board_view_events}, "
        f"board_view_chars={ctx.budgets.board_view_chars}"
    )

    digest_text, digest_meta = _render_digest(ctx.transcript_entries, ctx.budgets)

    dm_text = "(no unread direct messages)"
    if ctx.unread_dms:
        dm_lines = []
        for dm in ctx.unread_dms:
            dm_lines.append(f"- from {dm.get('from_agent', 'unknown')}: {dm.get('content', '')}")
        dm_text = "\n".join(dm_lines)

    prompt = f"""{preamble}

=== ROUND HEADER ===
round_id: {ctx.round_id}
cycle: {ctx.cycle} of {ctx.max_cycles}
your_seat: {ctx.seat}
display_name: {ctx.seat_display}
standing_question: {ctx.standing_question}
officers: {officers_str}
synthesizer: {ctx.synthesizer}
budgets: {budgets_str}
doctrine_stamp: {ctx.doctrine_stamp.get('sot_path', '')} @ {ctx.doctrine_stamp.get('sot_sha256', '')}

=== BRIEF ===
{ctx.brief_text}

=== BOARD DIGEST ===
{digest_text}

=== YOUR UNREAD DMs ===
{dm_text}

=== REPLY CONTRACT ===
End your reply with ONE fenced JSON block matching this schema:

```json
{{
  "type": "post",                     // post | dm | vote | artifact | flag | pass
  "text": "my contribution ...",      // post/dm: the message (markdown ok)
  "to": "kimi",                       // dm only
  "vote": {{"question": "...", "choice": "...", "why": "..."}},   // vote only
  "artifact": {{"path": "docs/roundtable/Agent_delta_architecture.md",
                "body": "full file content..."}},                 // artifact only
  "flag": {{"blocking": true, "item": "...", "problem": "...",
            "evidence": "...", "options": "...", "recommendation": "..."}},
  "stamp": "YourName / model / date"
}}
```

You may include free text before the JSON block. Multiple actions may be sent
as an `actions` array of the above objects. If you cannot produce valid JSON,
write plain text and the orchestrator will preserve it as a post.

End your reply with your identity stamp (name / model / date) inside the JSON block.
"""

    meta = {
        "round_id": ctx.round_id,
        "cycle": ctx.cycle,
        "seat": ctx.seat,
        "digest": digest_meta,
        "unread_dms_count": len(ctx.unread_dms),
        "prompt_chars": len(prompt),
    }
    return prompt, meta
