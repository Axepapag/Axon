#!/usr/bin/env python3
"""Controlled Hermes/GLM cloud one-shot wrapper with injected persistent state.

Promoted from D:\AxonGliksbot\scripts\hermes_glm.py into the Axon runtime.
Archive copy remains read-only; this is the active runtime wrapper.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


STATE_ROOT = Path(r"C:\Users\Jeffg\Documents\Codex\.codex")
HERMES_MEMORY = STATE_ROOT / "hermes-memory.md"
HERMES_TURN_STATE = STATE_ROOT / "hermes-turn-state.md"
HERMES_EXE = Path(
    r"C:\Users\Jeffg\AppData\Local\hermes\hermes-agent\venv\Scripts\hermes.exe"
)
DEFAULT_TOOLSETS = os.environ.get("HERMES_TOOLSETS", "file").strip()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def build_prompt(task: str) -> str:
    return f"""You are Hermes, operating as a cloud agent lane for Jeff and Codex.

Persistent state discipline:
- Treat the two state documents below as your memory and active carryover.
- Use these facts unless the user's latest instruction overrides them.
- If this task changes durable facts or the next-step carryover, update the state files with your tools when available.
- If you cannot update the files directly, end with a compact "State updates for Codex" section.
- Never store secrets, tokens, private keys, recovery codes, or credentials.

Hermes memory path: {HERMES_MEMORY}
Hermes turn-state path: {HERMES_TURN_STATE}

--- HERMES MEMORY ---
{read_text(HERMES_MEMORY)}

--- HERMES TURN STATE ---
{read_text(HERMES_TURN_STATE)}

--- TASK FROM CODEX ---
{task}
"""


def main() -> int:
    task = " ".join(sys.argv[1:]).strip()
    if not task and not sys.stdin.isatty():
        task = sys.stdin.read().strip()
    if not task:
        print("Provide a prompt argument or pipe prompt text.", file=sys.stderr)
        return 2
    if not HERMES_EXE.exists():
        print(f"Hermes executable was not found at {HERMES_EXE}", file=sys.stderr)
        return 2

    args = [str(HERMES_EXE), "-m", "glm-5.2:cloud"]
    if DEFAULT_TOOLSETS:
        args.extend(["--toolsets", DEFAULT_TOOLSETS])
    args.extend(["-z", build_prompt(task)])

    result = subprocess.run(
        args,
        cwd=os.getcwd(),
        text=True,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
