"""Terminal seat tests: capture logic, ANSI stripping, waker integration.

The live pty path (winpty) is exercised by the officer demo, not pytest;
everything testable without a console is tested here.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from runtime.table import term_seat, waker
from runtime.table.manifests import ManifestDir, load_manifest
from runtime.table.term_seat import (
    QuietCapture,
    poll_reply,
    strip_ansi,
    trim_echo,
    truncate_with_marker,
)

FIXTURES = Path(__file__).parent / "fixtures" / "table"
BRIEF = FIXTURES / "test_brief.md"


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #

def test_strip_ansi_removes_csi_and_osc():
    raw = "\x1b]0;title\x07\x1b[32mgreen\x1b[0m text\r\nline2\x1b[K\x1b[?25h"
    assert strip_ansi(raw) == "green text\nline2"


def test_strip_ansi_keeps_plain_lines():
    assert strip_ansi("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_truncate_with_marker_is_explicit():
    text = "x" * 4100
    out = truncate_with_marker(text, limit=4000)
    assert out.startswith("x" * 4000)
    assert "[truncated 100 chars]" in out
    assert truncate_with_marker("short") == "short"


def test_trim_echo_cuts_prompt_echo():
    prompt = "What is the capital of France?"
    captured = "What is the capital of\nFrance?\nParis is the capital.\n"
    assert trim_echo(captured, prompt).strip() == "Paris is the capital."


def test_trim_echo_no_echo_passthrough():
    captured = "Paris is the capital."
    assert trim_echo(captured, "unrelated prompt text") == captured


# --------------------------------------------------------------------------- #
# quiet-capture state machine (fake clock)
# --------------------------------------------------------------------------- #

def test_quiet_capture_waits_for_first_output():
    cap = QuietCapture(quiet_seconds=5.0, max_wait_seconds=100.0)
    cap.start(now=0.0)
    # No output yet: quiet period must NOT trigger.
    assert not cap.done(now=30.0)
    cap.feed("thinking...", now=31.0)
    assert not cap.done(now=33.0)          # still within quiet window
    assert cap.done(now=36.5)              # quiet for 5.5s after last chunk
    assert cap.text() == "thinking..."
    assert not cap.timed_out


def test_quiet_capture_max_wait_times_out():
    cap = QuietCapture(quiet_seconds=5.0, max_wait_seconds=50.0)
    cap.start(now=0.0)
    assert cap.done(now=51.0)
    assert cap.timed_out


def test_quiet_capture_prompt_pattern_ends_early():
    cap = QuietCapture(quiet_seconds=60.0, max_wait_seconds=600.0, prompt_pattern=r">>>\s*$")
    cap.start(now=0.0)
    cap.feed("answer text\n>>> ", now=1.0)
    assert cap.done(now=1.1)               # pattern beats the quiet timer
    assert not cap.timed_out


# --------------------------------------------------------------------------- #
# bus-store polling
# --------------------------------------------------------------------------- #

def _make_bus_db(path: Path) -> Path:
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE bus_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            topic TEXT NOT NULL,
            source TEXT NOT NULL,
            session_id TEXT,
            payload TEXT NOT NULL,
            segment_file TEXT,
            segment_offset INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    con.commit()
    con.close()
    return path


def _insert_event(db: Path, event_id: str, topic: str, payload: dict) -> None:
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO bus_events (event_id, topic, source, payload, created_at)"
        " VALUES (?, ?, 'test', ?, 'now')",
        (event_id, topic, json.dumps(payload)),
    )
    con.commit()
    con.close()


def test_poll_reply_finds_matching_turn(tmp_path):
    db = _make_bus_db(tmp_path / "bus.sqlite")
    _insert_event(db, "1".zfill(20), "term.t1.reply", {"turn_id": "other", "text": "no"})
    _insert_event(db, "2".zfill(20), "term.t1.reply", {"turn_id": "turn-9", "text": "yes!"})
    ticks = iter([0.0, 0.0, 1.0, 2.0, 3.0])
    out = poll_reply(
        "t1", "turn-9", timeout_seconds=10.0, db_path=str(db),
        sleep=lambda s: None, clock=lambda: next(ticks),
    )
    assert out == "yes!"


def test_poll_reply_times_out(tmp_path):
    db = _make_bus_db(tmp_path / "bus.sqlite")
    t = {"v": 0.0}

    def clock():
        t["v"] += 3.0
        return t["v"]

    out = poll_reply("t1", "missing", timeout_seconds=9.0, db_path=str(db),
                     sleep=lambda s: None, clock=clock)
    assert out is None


def test_poll_reply_respects_after_cursor(tmp_path):
    db = _make_bus_db(tmp_path / "bus.sqlite")
    _insert_event(db, "5".zfill(20), "term.t1.reply", {"turn_id": "turn-1", "text": "stale"})
    _insert_event(db, "9".zfill(20), "term.t1.reply", {"turn_id": "turn-1", "text": "fresh"})
    ticks = iter([0.0, 0.0, 1.0])
    out = poll_reply(
        "t1", "turn-1", timeout_seconds=10.0, db_path=str(db),
        sleep=lambda s: None, clock=lambda: next(ticks),
        after_event_id="7".zfill(20),
    )
    assert out == "fresh"


# --------------------------------------------------------------------------- #
# manifest terminal type
# --------------------------------------------------------------------------- #

def test_terminal_manifest_loads():
    m = load_manifest(ManifestDir / "term-1.json")
    assert m.is_terminal
    assert not m.is_manual                 # empty argv but terminal, not manual
    assert m.wake.term_id == "term-1"
    assert m.enabled is False


def test_terminal_manifest_requires_term_id(tmp_path):
    bad = {
        "client_id": "t", "display_name": "T",
        "wake": {"type": "terminal", "argv": [], "timeout_seconds": 5, "workdir": "."},
        "identity_stamp": "t/t/t",
    }
    p = tmp_path / "t.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="term_id"):
        load_manifest(p)


# --------------------------------------------------------------------------- #
# waker terminal turn (bus mocked via publish capture + temp sqlite)
# --------------------------------------------------------------------------- #

def _terminal_manifests(tmp_path: Path) -> Path:
    d = tmp_path / "manifests"
    d.mkdir()
    for name in ("echo_a.json", "synth.json"):
        shutil.copy(ManifestDir / name, d / name)
    term = {
        "client_id": "tseat",
        "display_name": "Terminal Seat",
        "wake": {"type": "terminal", "term_id": "t9", "argv": [],
                 "timeout_seconds": 5, "workdir": str(tmp_path)},
        "capabilities": ["test"],
        "identity_stamp": "tseat / live / set-at-wake",
        "enabled": True,
        "notes": "",
    }
    (d / "tseat.json").write_text(json.dumps(term), encoding="utf-8")
    return d


def test_waker_terminal_turn_roundtrip(tmp_path, monkeypatch):
    db = _make_bus_db(tmp_path / "bus.sqlite")
    monkeypatch.setenv("AXON_BUS_DB", str(db))

    manifests_dir = _terminal_manifests(tmp_path)
    table = waker.RoundTable(state_root=tmp_path / "State", manifests_dir=manifests_dir)

    published: list[tuple[str, dict]] = []

    def fake_publish(topic: str, payload: dict) -> bool:
        published.append((topic, payload))
        if topic == "term.t9.prompt":
            # The bridge "answers" immediately.
            _insert_event(db, "3".zfill(20), "term.t9.reply",
                          {"turn_id": payload["turn_id"], "text": "terminal says hi"})
        return True

    monkeypatch.setattr(table.bus, "publish_sync", fake_publish)

    table.open_round(
        round_id="term-turn-test",
        brief_path=str(BRIEF),
        seats=["tseat"],
        synthesizer="synth",
        max_cycles=1,
        per_turn_timeout_s=5,
        offline=False,
        officers=[],
    )
    cfg = table._load_config("term-turn-test")
    stdout, stderr, code = table._wake_seat(cfg, "tseat", "hello terminal")
    assert code == 0
    assert stdout == "terminal says hi"
    assert any(t == "term.t9.prompt" for t, _ in published)


def test_waker_terminal_offline_fails_cleanly(tmp_path, monkeypatch):
    db = _make_bus_db(tmp_path / "bus.sqlite")
    monkeypatch.setenv("AXON_BUS_DB", str(db))
    manifests_dir = _terminal_manifests(tmp_path)
    table = waker.RoundTable(state_root=tmp_path / "State", manifests_dir=manifests_dir)
    table.open_round(
        round_id="term-offline-test",
        brief_path=str(BRIEF),
        seats=["tseat"],
        synthesizer="synth",
        max_cycles=1,
        per_turn_timeout_s=5,
        offline=True,
        officers=[],
    )
    cfg = table._load_config("term-offline-test")
    stdout, stderr, code = table._wake_seat(cfg, "tseat", "hello")
    assert code == 1
    assert "offline" in stderr


# --------------------------------------------------------------------------- #
# bridge input preparation (no pty needed)
# --------------------------------------------------------------------------- #

def test_prepare_input_newline_modes():
    b = term_seat.TermBridge.__new__(term_seat.TermBridge)
    b.newline_mode = "space"
    assert b.prepare_input("line1\nline2") == "line1 line2\r"
    b.newline_mode = "triple"
    assert b.prepare_input("a\nb") == '"""a\nb"""\r'
    b.newline_mode = "raw"
    assert b.prepare_input("a\nb") == "a\nb\r"
