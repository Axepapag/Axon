"""Terminal Seat bridge: hand a live terminal to the round table.

Jeff runs `python -m runtime.table term --id term-1`, which opens a real
interactive shell (a Windows pseudoconsole owned by this bridge, mirrored to
his own console window). He sets up ANY terminal agent by hand — `ollama run
<model>`, a brand-new CLI, an ssh session — then presses Ctrl+G to hand the
keyboard to the bus. In table mode, `term.<id>.prompt` events are typed into
the terminal, the agent's output is captured until it goes quiet (or a
prompt-pattern matches), stripped of ANSI codes, and published back as
`term.<id>.reply` plus a `committee.message.created` chat message. Ctrl+G
hands the keyboard back at any time; Ctrl+Q exits the bridge.

The waker treats a manifest with `wake.type == "terminal"` as one of these
seats: its turn is publish-prompt / await-reply instead of a subprocess.

BUS_TOKEN is read from the environment only and never logged, echoed,
captured into replies, or written anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .buslink import BusLink

DEFAULT_BUS_DB = r"D:\Dream_Team_Clean\data\dream_team.sqlite"

# CSI sequences, OSC sequences (title sets etc.), other ESC forms, then any
# remaining control characters except tab/newline. Carriage returns are
# normalized separately so captured text keeps its line structure.
_ANSI_CSI = re.compile(r"\x1b\[[0-9;:?]*[ -/]*[@-~]")
_ANSI_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_MISC = re.compile(r"\x1b[@-_=><]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences and stray control chars; keep line structure."""
    text = _ANSI_CSI.sub("", text)
    text = _ANSI_OSC.sub("", text)
    text = _ANSI_MISC.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    return text


def truncate_with_marker(text: str, limit: int = 4000) -> str:
    """Explicit truncation only — never silent (Working Contract rule 7)."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return text[:limit] + f"\n[truncated {dropped} chars]"


def trim_echo(captured: str, prompt_text: str) -> str:
    """Remove the terminal's echo of the typed prompt from the capture.

    Terminals echo what is typed. The echo may be wrapped/redrawn, so match
    on a normalized prefix of the prompt and cut through its last line.
    """
    if not captured or not prompt_text:
        return captured
    probe = " ".join(prompt_text.split())[:80]
    if not probe:
        return captured
    lines = captured.split("\n")
    squashed = ""
    cut_at = 0
    for i, line in enumerate(lines[:40]):  # echo lives near the top
        squashed = " ".join((squashed + " " + line).split())
        if probe in squashed:
            cut_at = i + 1
            break
    return "\n".join(lines[cut_at:]) if cut_at else captured


@dataclass
class QuietCapture:
    """Decide when a terminal agent has finished answering.

    Done when: prompt_pattern matches the captured text, OR at least one
    chunk of output has arrived after the prompt was typed and no further
    output for `quiet_seconds`, OR `max_wait_seconds` elapsed (reported as
    timed_out, capture still returned — explicit, never silent).
    """

    quiet_seconds: float = 8.0
    max_wait_seconds: float = 600.0
    prompt_pattern: str | None = None
    started_at: float = 0.0
    last_chunk_at: float = 0.0
    any_output: bool = False
    chunks: list[str] = field(default_factory=list)
    timed_out: bool = False

    def start(self, now: float) -> None:
        self.started_at = now
        self.last_chunk_at = now
        self.any_output = False
        self.chunks = []
        self.timed_out = False

    def feed(self, chunk: str, now: float) -> None:
        if chunk:
            self.chunks.append(chunk)
            self.any_output = True
            self.last_chunk_at = now

    def text(self) -> str:
        return "".join(self.chunks)

    def done(self, now: float) -> bool:
        if now - self.started_at >= self.max_wait_seconds:
            self.timed_out = True
            return True
        if self.prompt_pattern and re.search(self.prompt_pattern, strip_ansi(self.text())):
            return True
        return self.any_output and (now - self.last_chunk_at) >= self.quiet_seconds


# --------------------------------------------------------------------------- #
# Bus store polling (read-only sqlite; used by the bridge for inbound prompts
# and by the waker to await a terminal seat's reply)
# --------------------------------------------------------------------------- #

def bus_db_path() -> str:
    return os.environ.get("AXON_BUS_DB", DEFAULT_BUS_DB)


def _query_events(db_path: str, topics: list[str], after_event_id: str) -> list[dict[str, Any]]:
    marks = ",".join("?" for _ in topics)
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        rows = con.execute(
            f"""
            SELECT event_id, topic, payload FROM bus_events
            WHERE topic IN ({marks}) AND event_id > ?
            ORDER BY event_id ASC
            """,
            (*topics, after_event_id),
        ).fetchall()
        con.close()
    except sqlite3.Error:
        return []
    out = []
    for event_id, topic, payload in rows:
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            data = {}
        out.append({"event_id": event_id, "topic": topic, "payload": data})
    return out


def latest_event_id(db_path: str) -> str:
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        row = con.execute("SELECT MAX(event_id) FROM bus_events").fetchone()
        con.close()
        return row[0] or ""
    except sqlite3.Error:
        return ""


def poll_reply(
    term_id: str,
    turn_id: str,
    timeout_seconds: float,
    db_path: str | None = None,
    poll_interval: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    after_event_id: str = "",
) -> str | None:
    """Wait for a term.<id>.reply event carrying this turn_id. None on timeout."""
    db = db_path or bus_db_path()
    topic = f"term.{term_id}.reply"
    cursor = after_event_id
    deadline = clock() + timeout_seconds
    while clock() < deadline:
        for ev in _query_events(db, [topic], cursor):
            cursor = max(cursor, ev["event_id"])
            if ev["payload"].get("turn_id") == turn_id:
                return str(ev["payload"].get("text", ""))
        sleep(poll_interval)
    return None


# --------------------------------------------------------------------------- #
# The live bridge (Windows only; winpty imported lazily so this module stays
# importable in tests and by the waker without pywinpty installed)
# --------------------------------------------------------------------------- #

MANUAL = "manual"
TABLE = "table"

_BANNER = "\x1b[7m\x1b[33m"  # inverse yellow
_RESET = "\x1b[0m"


class TermBridge:
    def __init__(
        self,
        term_id: str,
        shell: str = "powershell.exe",
        quiet_seconds: float = 8.0,
        prompt_pattern: str | None = None,
        newline_mode: str = "space",
        max_wait_seconds: float = 600.0,
    ) -> None:
        self.term_id = term_id
        self.shell = shell
        self.newline_mode = newline_mode
        self.capture = QuietCapture(
            quiet_seconds=quiet_seconds,
            max_wait_seconds=max_wait_seconds,
            prompt_pattern=prompt_pattern,
        )
        self.mode = MANUAL
        self.bus = BusLink()
        self.db = bus_db_path()
        self.queued_prompts: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._capturing = False
        self._alive = True
        self._proc = None

    # ---- console helpers -------------------------------------------------
    def _banner(self, text: str) -> None:
        sys.stdout.write(f"\n{_BANNER} [{self.term_id}] {text} {_RESET}\n")
        sys.stdout.flush()

    def _announce_mode(self) -> None:
        self.bus.publish_sync(f"term.{self.term_id}.status", {"mode": self.mode})
        self._banner(
            f"mode: {self.mode.upper()}"
            + (" — the table is driving; Ctrl+G to take the keyboard back"
               if self.mode == TABLE
               else " — your keyboard; Ctrl+G to hand over to the table, Ctrl+Q to exit")
        )

    # ---- prompt normalization -------------------------------------------
    def prepare_input(self, text: str) -> str:
        if self.newline_mode == "space":
            body = " ".join(text.splitlines())
        elif self.newline_mode == "triple":
            body = '"""' + text + '"""'
        else:  # raw
            body = text
        return body + "\r"

    # ---- table-drive path -------------------------------------------------
    def _drive_prompt(self, payload: dict[str, Any]) -> None:
        turn_id = str(payload.get("turn_id", ""))
        text = str(payload.get("text", ""))
        if not text:
            return
        self._banner(f"table turn {turn_id or '(unlabeled)'} — typing prompt")
        now = time.monotonic()
        with self._lock:
            self.capture.start(now)
            self._capturing = True
        self._proc.write(self.prepare_input(text))
        while self._alive:
            time.sleep(0.25)
            with self._lock:
                if self.capture.done(time.monotonic()):
                    self._capturing = False
                    break
        raw = self.capture.text()
        reply = trim_echo(strip_ansi(raw), text).strip()
        if self.capture.timed_out:
            reply += "\n[terminal capture hit max wait — reply may be incomplete]"
        self.bus.publish_sync(
            f"term.{self.term_id}.reply", {"turn_id": turn_id, "text": reply}
        )
        self.bus.publish_sync(
            "committee.message.created",
            {"round_id": "terminal-seat", "from": self.term_id,
             "text": truncate_with_marker(reply)},
        )
        self._banner(f"reply published ({len(reply)} chars)")

    # ---- bus poller thread -------------------------------------------------
    def _poll_bus(self) -> None:
        cursor = latest_event_id(self.db)
        topics = [f"term.{self.term_id}.prompt", f"term.{self.term_id}.mode"]
        while self._alive:
            for ev in _query_events(self.db, topics, cursor):
                cursor = max(cursor, ev["event_id"])
                if ev["topic"].endswith(".mode"):
                    wanted = ev["payload"].get("mode")
                    if wanted in (MANUAL, TABLE) and wanted != self.mode:
                        self.mode = wanted
                        self._announce_mode()
                    continue
                if self.mode == TABLE:
                    self._drive_prompt(ev["payload"])
                else:
                    self.queued_prompts.append(ev["payload"])
                    self._banner(
                        f"table prompt queued ({len(self.queued_prompts)} waiting) — Ctrl+G to hand over"
                    )
            time.sleep(1.5)

    # ---- pty reader thread --------------------------------------------------
    def _read_pty(self) -> None:
        while self._alive and self._proc.isalive():
            try:
                chunk = self._proc.read(4096)
            except (EOFError, ConnectionError, OSError):
                break
            if not chunk:
                continue
            sys.stdout.write(chunk)
            sys.stdout.flush()
            with self._lock:
                if self._capturing:
                    self.capture.feed(chunk, time.monotonic())
        self._alive = False

    # ---- keyboard (main thread) ----------------------------------------------
    _VT_KEYS = {"H": "\x1b[A", "P": "\x1b[B", "M": "\x1b[C", "K": "\x1b[D",
                "G": "\x1b[H", "O": "\x1b[F", "R": "\x1b[2~", "S": "\x1b[3~"}

    def _keyboard_loop(self) -> None:
        import msvcrt

        while self._alive:
            if not msvcrt.kbhit():
                time.sleep(0.03)
                continue
            ch = msvcrt.getwch()
            if ch == "\x07":  # Ctrl+G: hand over / take back
                self.mode = TABLE if self.mode == MANUAL else MANUAL
                self._announce_mode()
                if self.mode == TABLE and self.queued_prompts:
                    for payload in self.queued_prompts:
                        self._drive_prompt(payload)
                    self.queued_prompts.clear()
                continue
            if ch == "\x11":  # Ctrl+Q: exit the bridge
                self._banner("exiting terminal seat")
                self._alive = False
                break
            if self.mode != MANUAL:
                continue  # table is driving; keyboard parked
            if ch in ("\x00", "\xe0"):
                code = msvcrt.getwch()
                self._proc.write(self._VT_KEYS.get(code, ""))
            else:
                self._proc.write("\r" if ch == "\r" else ch)

    # ---- entry ---------------------------------------------------------------
    def run(self) -> int:
        try:
            from winpty import PtyProcess
        except ImportError:
            print("pywinpty is required for terminal seats: pip install pywinpty", file=sys.stderr)
            return 1
        os.system("")  # enable VT processing in this console
        import shutil as _sh

        size = _sh.get_terminal_size((120, 30))
        self._proc = PtyProcess.spawn(self.shell, dimensions=(size.lines, size.columns))
        threading.Thread(target=self._read_pty, daemon=True).start()
        threading.Thread(target=self._poll_bus, daemon=True).start()
        self._announce_mode()
        try:
            self._keyboard_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._alive = False
            if self._proc.isalive():
                self._proc.terminate(force=True)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m runtime.table term")
    parser.add_argument("--id", required=True, help="terminal seat id (bus client id)")
    parser.add_argument("--shell", default="powershell.exe")
    parser.add_argument("--quiet-seconds", type=float, default=8.0)
    parser.add_argument("--max-wait-seconds", type=float, default=600.0)
    parser.add_argument("--prompt-pattern", default=None,
                        help="regex marking end-of-reply (else quiet period)")
    parser.add_argument("--newline-mode", choices=["space", "raw", "triple"], default="space",
                        help="how multi-line prompts are typed into the terminal")
    args = parser.parse_args(argv)
    bridge = TermBridge(
        term_id=args.id,
        shell=args.shell,
        quiet_seconds=args.quiet_seconds,
        prompt_pattern=args.prompt_pattern,
        newline_mode=args.newline_mode,
        max_wait_seconds=args.max_wait_seconds,
    )
    return bridge.run()


if __name__ == "__main__":
    sys.exit(main())
