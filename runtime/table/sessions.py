"""Stateful seat session store.

Maps seat id -> durable session metadata so the waker can resume a live
interactive session instead of cold-starting a seat on every turn.

Store path: ``State/table/sessions.json``
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DefaultStateRoot = Path("State") / "table"
SessionsFileName = "sessions.json"


@dataclass
class SeatSession:
    seat_id: str
    session_id: str
    pinned: bool
    updated_at: str


def _sessions_path(state_root: str | Path | None = None) -> Path:
    root = Path(state_root) if state_root else DefaultStateRoot
    return root / SessionsFileName


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_sessions(state_root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the raw sessions mapping. Missing files return an empty dict."""
    path = _sessions_path(state_root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def save_sessions(
    sessions: dict[str, dict[str, Any]],
    state_root: str | Path | None = None,
) -> Path:
    """Atomically write the sessions mapping to disk."""
    path = _sessions_path(state_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sessions, indent=2, sort_keys=True), encoding="utf-8")
    return path


def get_session(state_root: str | Path | None = None, seat_id: str | None = None) -> str | None:
    """Return the current session id for a seat, or None."""
    sessions = load_sessions(state_root)
    entry = sessions.get(seat_id)
    if not isinstance(entry, dict):
        return None
    session_id = entry.get("session_id")
    return session_id if isinstance(session_id, str) and session_id else None


def is_pinned(state_root: str | Path | None = None, seat_id: str | None = None) -> bool:
    """Return whether a seat's session is pinned (immune to capture)."""
    sessions = load_sessions(state_root)
    entry = sessions.get(seat_id)
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("pinned", False))


def pin_session(
    state_root: str | Path | None = None,
    seat_id: str | None = None,
    session_id: str | None = None,
) -> SeatSession:
    """Pin a seat to a session id. If session_id is omitted, pin the existing id."""
    sessions = load_sessions(state_root)
    entry: dict[str, Any] = sessions.get(seat_id) or {}
    if session_id:
        entry["session_id"] = session_id
    if not entry.get("session_id"):
        raise ValueError(f"no session id for seat {seat_id}; supply --session")
    entry["pinned"] = True
    entry["updated_at"] = _now_iso()
    sessions[seat_id] = entry
    save_sessions(sessions, state_root)
    return SeatSession(
        seat_id=seat_id,
        session_id=entry["session_id"],
        pinned=True,
        updated_at=entry["updated_at"],
    )


def unpin_session(
    state_root: str | Path | None = None,
    seat_id: str | None = None,
) -> SeatSession | None:
    """Unpin a seat's session. The session id is kept but may be overwritten by capture."""
    sessions = load_sessions(state_root)
    entry = sessions.get(seat_id)
    if not isinstance(entry, dict):
        return None
    entry["pinned"] = False
    entry["updated_at"] = _now_iso()
    sessions[seat_id] = entry
    save_sessions(sessions, state_root)
    return SeatSession(
        seat_id=seat_id,
        session_id=entry.get("session_id", ""),
        pinned=False,
        updated_at=entry["updated_at"],
    )


def set_session(
    state_root: str | Path | None = None,
    seat_id: str | None = None,
    session_id: str | None = None,
    pinned: bool = False,
) -> SeatSession:
    """Set a seat's session id directly, respecting the pinned flag if already pinned."""
    sessions = load_sessions(state_root)
    entry: dict[str, Any] = sessions.get(seat_id) or {}
    if entry.get("pinned") and not pinned:
        # Caller asked to set without pinning, but seat is pinned: leave pinned id in place.
        return SeatSession(
            seat_id=seat_id,
            session_id=entry.get("session_id", ""),
            pinned=True,
            updated_at=entry.get("updated_at", _now_iso()),
        )
    entry["session_id"] = session_id
    entry["pinned"] = pinned
    entry["updated_at"] = _now_iso()
    sessions[seat_id] = entry
    save_sessions(sessions, state_root)
    return SeatSession(
        seat_id=seat_id,
        session_id=session_id or "",
        pinned=pinned,
        updated_at=entry["updated_at"],
    )


def capture_session(
    state_root: str | Path | None = None,
    seat_id: str | None = None,
    session_id: str | None = None,
) -> SeatSession | None:
    """Update a seat's session id from regex capture only when not pinned."""
    if not session_id:
        return None
    sessions = load_sessions(state_root)
    entry: dict[str, Any] = sessions.get(seat_id) or {}
    if entry.get("pinned"):
        return SeatSession(
            seat_id=seat_id,
            session_id=entry.get("session_id", ""),
            pinned=True,
            updated_at=entry.get("updated_at", _now_iso()),
        )
    entry["session_id"] = session_id
    entry["pinned"] = False
    entry["updated_at"] = _now_iso()
    sessions[seat_id] = entry
    save_sessions(sessions, state_root)
    return SeatSession(
        seat_id=seat_id,
        session_id=session_id,
        pinned=False,
        updated_at=entry["updated_at"],
    )


def clear_session(
    state_root: str | Path | None = None,
    seat_id: str | None = None,
    only_if_pinned: bool = False,
) -> bool:
    """Clear a seat's session id. Respects pinned sessions unless ``only_if_pinned=False``."""
    sessions = load_sessions(state_root)
    entry = sessions.get(seat_id)
    if not isinstance(entry, dict):
        return False
    if entry.get("pinned") and only_if_pinned:
        return False
    entry["session_id"] = ""
    entry["updated_at"] = _now_iso()
    sessions[seat_id] = entry
    save_sessions(sessions, state_root)
    return True


def extract_session_id(stdout: str, stderr: str, parse_regex: str | None) -> str | None:
    """Run ``parse_regex`` over stdout+stderr and return the first captured group."""
    if not parse_regex:
        return None
    try:
        pattern = re.compile(parse_regex)
    except re.error:
        return None
    combined = f"{stdout}\n{stderr}"
    match = pattern.search(combined)
    if match:
        return match.group(1)
    return None
