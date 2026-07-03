"""CLI for the Round Table Orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

    args = parser.parse_args(argv)
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
