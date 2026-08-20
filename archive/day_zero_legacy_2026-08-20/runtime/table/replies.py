"""Reply parser: extract the LAST fenced ```json block with plain-text fallback.

A contribution is never lost to a parse error.  Anything unparseable becomes a
plain `post` with `parse_fallback: true`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal


ReplyType = Literal["post", "dm", "vote", "artifact", "flag", "pass"]


def _today() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _fenced_json_blocks(text: str) -> list[str]:
    """Return all fenced json blocks, in order."""
    pattern = r"```json\s*\n(.*?)\n```"
    return re.findall(pattern, text, re.DOTALL)


def _validate_action(action: dict[str, Any]) -> dict[str, Any]:
    """Ensure the action has the minimum required shape."""
    if not isinstance(action, dict):
        raise ValueError("action is not an object")
    t = action.get("type")
    if t not in ("post", "dm", "vote", "artifact", "flag", "pass"):
        raise ValueError(f"invalid reply type: {t!r}")
    out = dict(action)
    if "stamp" not in out or not isinstance(out["stamp"], str):
        out["stamp"] = f"unknown / unknown / {_today()}"
    return out


def _parse_single(text: str) -> dict[str, Any]:
    """Parse one reply text into a single action dict."""
    blocks = _fenced_json_blocks(text)
    if not blocks:
        raise ValueError("no fenced json block found")

    raw = blocks[-1].strip()
    if not raw:
        raise ValueError("empty fenced json block")

    parsed = json.loads(raw)

    # actions array
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("empty actions array")
        validated = [_validate_action(a) for a in parsed]
        return {"actions": validated}

    return _validate_action(parsed)


def parse_reply(stdout: str, stderr: str = "") -> dict[str, Any]:
    """Parse agent stdout/stderr.

    Returns a dict guaranteed to represent the contribution.  On any failure
    the entire stdout is wrapped as a plain post with `parse_fallback: true`.
    """
    combined = stdout
    if stderr:
        combined = f"{stdout}\n--- stderr ---\n{stderr}"

    try:
        result = _parse_single(combined)
    except Exception:
        return {
            "type": "post",
            "text": stdout.strip(),
            "parse_fallback": True,
            "stamp": f"unknown / unknown / {_today()}",
        }

    # If the result is a single action, return it directly for convenience.
    if "actions" not in result:
        return result

    # For an actions array, return the first action plus a marker that there
    # are additional actions; callers that need all actions use parse_reply_all.
    actions = result["actions"]
    first = dict(actions[0])
    first["_additional_actions"] = len(actions) - 1
    return first


def parse_reply_all(stdout: str, stderr: str = "") -> list[dict[str, Any]]:
    """Parse agent output and return every action."""
    combined = stdout
    if stderr:
        combined = f"{stdout}\n--- stderr ---\n{stderr}"

    try:
        result = _parse_single(combined)
    except Exception:
        return [
            {
                "type": "post",
                "text": stdout.strip(),
                "parse_fallback": True,
                "stamp": f"unknown / unknown / {_today()}",
            }
        ]

    if "actions" in result:
        return result["actions"]
    return [result]
