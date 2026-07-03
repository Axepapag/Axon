"""Axon Round Table launcher — a menu for everything, zero commands to remember.

Started by double-clicking D:\\Axon\\table.bat (or `python runtime/table/launcher.py`).
Loads BUS_TOKEN automatically from the tunnel launcher's env file; the token
value is never printed or logged.
"""

from __future__ import annotations

import os
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = Path(r"D:\cloudflare-tunnel\.env.local")
CHAT_URL = "http://127.0.0.1:8765/"
DEFAULT_BRIEF = "docs/roundtable/RoundTable_SlotArchitecture.md"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_token() -> bool:
    if os.environ.get("BUS_TOKEN"):
        return True
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            for key in ("BUS_TOKEN", "DREAM_TEAM_WS_TOKEN"):
                if line.startswith(key + "=") and not os.environ.get("BUS_TOKEN"):
                    os.environ["BUS_TOKEN"] = line.split("=", 1)[1].strip()
    except OSError:
        pass
    return bool(os.environ.get("BUS_TOKEN"))


def ask(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {label}{suffix}: ").strip()
    return val or default


def pause() -> None:
    input("\n  (Enter to go back to the menu) ")


def run_cli(args: list[str]) -> None:
    from runtime.table import cli

    try:
        cli.main(args)
    except SystemExit:
        pass
    except Exception as exc:  # menu must survive anything
        print(f"  !! {type(exc).__name__}: {exc}")


def list_rounds() -> list[str]:
    rounds_dir = ROOT / "ops" / "table" / "rounds"
    if not rounds_dir.exists():
        return []
    return sorted(p.name for p in rounds_dir.iterdir() if (p / "round.json").exists())


def pick_round() -> str | None:
    rounds = list_rounds()
    if not rounds:
        print("  (no rounds yet — open one first)")
        return None
    for i, r in enumerate(rounds, 1):
        print(f"   {i}. {r}")
    choice = ask("which round", "1")
    try:
        return rounds[int(choice) - 1]
    except (ValueError, IndexError):
        print("  ?? not a valid choice")
        return None


MENU = """
 ==========================================
   AXON ROUND TABLE
 ==========================================
   1. Terminal seat in THIS window
      (you drive; Ctrl+G hands it to the table; Ctrl+Q exits)
   2. Test a terminal seat (send it one prompt)
   3. Start a round
   4. Round status / continue a round
   5. One turn at a time (step a round)
   6. List seats & warm sessions
   7. Pin a warm session to a seat
   8. Open the chat page
   9. Exit
 ==========================================
"""


def action_terminal_seat() -> None:
    seat = ask("seat name", "term-1")
    mode = ask("multiline prompts as (space/triple/raw)", "space")
    quiet = ask("seconds of quiet = reply done", "8")
    print("\n  Starting your terminal. Set up any agent you like.")
    print("  Ctrl+G = hand keyboard to the table / take it back.  Ctrl+Q = exit.\n")
    time.sleep(1.0)
    from runtime.table.term_seat import TermBridge

    TermBridge(
        term_id=seat,
        quiet_seconds=float(quiet or 8),
        newline_mode=mode if mode in ("space", "triple", "raw") else "space",
    ).run()


def action_test_seat() -> None:
    seat = ask("seat name", "term-1")
    text = ask("prompt to send", "Introduce yourself to the round table.")
    run_cli(["term-send", "--id", seat, "--text", text, "--wait", "90"])
    pause()


def action_start_round() -> None:
    default_id = f"round-{time.strftime('%m%d-%H%M')}"
    rid = ask("round name", default_id)
    brief = ask("brief file", DEFAULT_BRIEF)
    seats = ask("seats (comma separated)", "hermes,kimi")
    cycles = ask("how many cycles", "2")
    run_cli([
        "open", "--round-id", rid, "--brief", brief,
        "--seats", seats, "--cycles", cycles,
    ])
    go = ask("run it now? (y/n)", "y").lower()
    if go.startswith("y"):
        print("  Running — watch the chat page. Ctrl+C here pauses the table.\n")
        run_cli(["run", "--round", rid])
    pause()


def action_status() -> None:
    rid = pick_round()
    if not rid:
        pause()
        return
    run_cli(["status", "--round", rid])
    go = ask("continue running this round? (y/n)", "n").lower()
    if go.startswith("y"):
        run_cli(["resume", "--round", rid])
        run_cli(["run", "--round", rid])
    pause()


def action_step() -> None:
    rid = pick_round()
    if not rid:
        pause()
        return
    run_cli(["step", "--round", rid])
    pause()


def action_pin() -> None:
    seat = ask("seat name", "kimi")
    session = ask("session id (from the agent's 'resume' line)")
    if session:
        run_cli(["pin", "--seat", seat, "--session", session])
    pause()


def main() -> int:
    os.system("")  # enable colors in this console
    if not load_token():
        print("  !! Could not find BUS_TOKEN (checked environment and"
              f" {ENV_FILE}). Bus features will queue offline.")
    while True:
        print(MENU)
        choice = input("  pick a number: ").strip()
        print()
        if choice == "1":
            action_terminal_seat()
        elif choice == "2":
            action_test_seat()
        elif choice == "3":
            action_start_round()
        elif choice == "4":
            action_status()
        elif choice == "5":
            action_step()
        elif choice == "6":
            run_cli(["seats"])
            pause()
        elif choice == "7":
            action_pin()
        elif choice == "8":
            webbrowser.open(CHAT_URL)
        elif choice == "9" or choice.lower() in ("q", "exit"):
            return 0
        else:
            print("  ?? pick 1-9")


if __name__ == "__main__":
    sys.exit(main())
