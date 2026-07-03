"""CLI for the Round Table Orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import sessions
from .waker import RoundTable


def _split_seats(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def _parse_budgets(args: argparse.Namespace) -> dict[str, int]:
    budgets: dict[str, int] = {}
    if args.max_invocations is not None:
        budgets["max_invocations"] = args.max_invocations
    if args.board_view_events is not None:
        budgets["board_view_events"] = args.board_view_events
    if args.board_view_chars is not None:
        budgets["board_view_chars"] = args.board_view_chars
    return budgets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m runtime.table", description="Round Table Orchestrator v1")
    sub = parser.add_subparsers(dest="command", required=True)

    open_p = sub.add_parser("open", help="open a new round")
    open_p.add_argument("--brief", required=True, help="path to brief markdown file")
    open_p.add_argument("--seats", required=True, help="comma-separated seat IDs")
    open_p.add_argument("--synthesizer", default="claude", help="seat that drafts the resolution")
    open_p.add_argument("--cycles", type=int, default=3, help="max cycles")
    open_p.add_argument("--timeout", type=int, default=600, help="per-turn timeout seconds")
    open_p.add_argument("--round-id", required=True, help="round identifier")
    open_p.add_argument("--offline", action="store_true", help="run without the live bus")
    open_p.add_argument("--max-invocations", type=int, default=None)
    open_p.add_argument("--board-view-events", type=int, default=None)
    open_p.add_argument("--board-view-chars", type=int, default=None)
    open_p.add_argument("--officers", default="codex,claude", help="comma-separated officer IDs")

    run_p = sub.add_parser("run", help="run a round to end condition")
    run_p.add_argument("--round", required=True, help="round ID")

    step_p = sub.add_parser("step", help="execute exactly one turn")
    step_p.add_argument("--round", required=True, help="round ID")

    status_p = sub.add_parser("status", help="show round status")
    status_p.add_argument("--round", required=True, help="round ID")

    pause_p = sub.add_parser("pause", help="pause a round")
    pause_p.add_argument("--round", required=True, help="round ID")

    resume_p = sub.add_parser("resume", help="resume a round")
    resume_p.add_argument("--round", required=True, help="round ID")

    close_p = sub.add_parser("close", help="close a round")
    close_p.add_argument("--round", required=True, help="round ID")
    close_p.add_argument("--reason", default="convener", help="close reason")

    pin_p = sub.add_parser("pin", help="pin a seat to a session id")
    pin_p.add_argument("--seat", required=True, help="seat ID")
    pin_p.add_argument("--session", required=True, help="session id to pin")

    unpin_p = sub.add_parser("unpin", help="unpin a seat's session")
    unpin_p.add_argument("--seat", required=True, help="seat ID")

    seats_p = sub.add_parser("seats", help="list seats and their session state")

    term_p = sub.add_parser("term", help="start a terminal seat bridge (your live terminal, Ctrl+G hands it to the table)")
    term_p.add_argument("--id", required=True, help="terminal seat id (bus client id)")
    term_p.add_argument("--shell", default="powershell.exe")
    term_p.add_argument("--quiet-seconds", type=float, default=8.0)
    term_p.add_argument("--max-wait-seconds", type=float, default=600.0)
    term_p.add_argument("--prompt-pattern", default=None)
    term_p.add_argument("--newline-mode", choices=["space", "raw", "triple"], default="space")

    tsend_p = sub.add_parser("term-send", help="publish one test prompt to a terminal seat (officer smoke)")
    tsend_p.add_argument("--id", required=True, help="terminal seat id")
    tsend_p.add_argument("--text", required=True, help="prompt text to type into the terminal")
    tsend_p.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the reply")

    args = parser.parse_args(argv)

    if args.command == "term":
        from .term_seat import TermBridge

        return TermBridge(
            term_id=args.id,
            shell=args.shell,
            quiet_seconds=args.quiet_seconds,
            prompt_pattern=args.prompt_pattern,
            newline_mode=args.newline_mode,
            max_wait_seconds=args.max_wait_seconds,
        ).run()

    if args.command == "term-send":
        import time as _time

        from . import term_seat
        from .buslink import BusLink

        turn_id = f"term-send-{int(_time.time())}"
        db = term_seat.bus_db_path()
        cursor = term_seat.latest_event_id(db)
        ok = BusLink().publish_sync(
            f"term.{args.id}.prompt", {"turn_id": turn_id, "text": args.text}
        )
        if not ok:
            print("bus rejected/queued the prompt (is the bus up and BUS_TOKEN set?)")
            return 1
        print(f"prompt published (turn {turn_id}); waiting up to {args.wait:.0f}s for reply...")
        reply = term_seat.poll_reply(args.id, turn_id, args.wait, db_path=db, after_event_id=cursor)
        if reply is None:
            print("no reply (is the bridge running and in TABLE mode?)")
            return 1
        print("--- reply ---")
        print(reply)
        return 0

    table = RoundTable()

    if args.command == "open":
        budgets = _parse_budgets(args)
        cfg = table.open_round(
            round_id=args.round_id,
            brief_path=args.brief,
            seats=_split_seats(args.seats),
            synthesizer=args.synthesizer,
            max_cycles=args.cycles,
            per_turn_timeout_s=args.timeout,
            budgets=budgets or None,
            offline=args.offline,
            officers=_split_seats(args.officers),
        )
        print(f"opened round {cfg.round_id} status={cfg.status}")
        return 0

    if args.command == "run":
        result = table.run(args.round)
        print(result)
        return 0

    if args.command == "step":
        result = table.step(args.round)
        print(result)
        return 0

    if args.command == "status":
        print(table.status(args.round))
        return 0

    if args.command == "pause":
        table.pause(args.round)
        print(f"paused {args.round}")
        return 0

    if args.command == "resume":
        table.resume(args.round)
        print(f"resumed {args.round}")
        return 0

    if args.command == "close":
        table.close(args.round, args.reason)
        print(f"closed {args.round}")
        return 0

    if args.command == "pin":
        sessions.pin_session(seat_id=args.seat, session_id=args.session)
        print(f"pinned {args.seat} -> {args.session}")
        return 0

    if args.command == "unpin":
        sessions.unpin_session(seat_id=args.seat)
        print(f"unpinned {args.seat}")
        return 0

    if args.command == "seats":
        for seat in table.list_seats():
            pinned_flag = " (pinned)" if seat["pinned"] else ""
            session_info = seat["session_id"] or "(none)"
            model_info = seat["model"] or "(default)"
            print(
                f"{seat['seat_id']:12} model={model_info:20} session={session_info}{pinned_flag} "
                f"enabled={seat['enabled']} manifest={seat['manifest']}"
            )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
