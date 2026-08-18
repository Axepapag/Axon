"""Run one bounded Kimi work packet with durable status and no wall timeout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SCHEMA = "axon-kimi-supervised-job-v1"
LOCK_SCHEMA = "axon-kimi-supervisor-lock-v1"
JOB_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,80}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def continuity_prompt(packet: str) -> str:
    return f"""You are Kimi acting as a bounded, directly supervised sub-agent for Codex on
D:\\Axon. Complete exactly this one work packet and then stop.

CONTINUITY AND SAFETY CONTRACT:
  - Before acting, read the work packet itself,
    D:\\Axon\\roundtable\\Kimmy_update_for_Codex_03August.md,
    D:\\Axon\\codex-turn-state.md,
    C:\\Users\\Jeffg\\Documents\\Codex\\.codex\\codex-turn-state.md,
    and the latest relevant entry in D:\\Kimmy\\kimmy_personal_log.md.
    The historical 28July handoff files may be absent; do not recreate them
    and do not block on their absence.
- Inspect current files before editing. Preserve the dirty worktree and all
  unrelated user/Kimi/Codex changes.
- Never modify protected checkpoints, D:\\Axon\\State\\axon_runtime, Kaggle
  assets, schedules, generated Kimi live_bundle.md, or the verified v2 runtime
  behavior unless the packet explicitly names a narrow source file.
- Do not launch training, Kaggle, an infinite runtime, tools/advisors, or
  checkpoint promotion.
- Stay inside the packet scope. Do not perform side quests.
- Run every test named by the packet. Report exact commands and results.
- At the end, append a dated TURN UPDATE to
  D:\\Kimmy\\kimmy_personal_log.md with request, inputs, exact changed files,
  validation, blockers, and next action. Update compact injection source files
  only if durable facts changed; never edit live_bundle.md directly. If source
  injection files changed, refresh with
  python D:\\Kimmy\\Source\\injection_watcher.py --force.
- Your final response must contain: STATUS, FILES CHANGED, TESTS, RISKS, and
  RECOMMENDED NEXT PACKET. Do not claim success from an exit code alone.

WORK PACKET:
{packet}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()

    if not JOB_ID_RE.fullmatch(args.job_id):
        parser.error("job ID must match [a-z0-9][a-z0-9._-]{2,80}")

    repo = Path(__file__).resolve().parents[1]
    packet_path = Path(args.packet).resolve()
    if not packet_path.is_file() or not inside(packet_path, repo):
        parser.error("packet must be an existing file inside the Axon repository")

    kimi_path = Path(r"C:\Users\Jeffg\.kimi-code\bin\kimi.exe")
    kimmy_root = Path(r"D:\Kimmy")
    if not kimi_path.is_file():
        parser.error(f"Kimi executable is missing: {kimi_path}")

    state_root = repo / "State" / "kimi_supervisor"
    jobs_root = state_root / "jobs"
    job_root = jobs_root / args.job_id
    lock_path = state_root / "active.lock.json"
    status_path = job_root / "status.json"
    stdout_path = job_root / "stdout.log"
    stderr_path = job_root / "stderr.log"
    latest_path = state_root / "latest.json"
    state_root.mkdir(parents=True, exist_ok=True)
    jobs_root.mkdir(parents=True, exist_ok=True)
    try:
        job_root.mkdir()
    except FileExistsError as exc:
        raise SystemExit(f"job directory already exists: {job_root}") from exc

    started_at = utc_now()
    lock_payload = {
        "schema": LOCK_SCHEMA,
        "job_id": args.job_id,
        "wrapper_pid": os.getpid(),
        "packet_path": str(packet_path),
        "started_at": started_at,
    }
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise SystemExit(f"another supervised Kimi packet owns {lock_path}") from exc

    with os.fdopen(lock_fd, "w", encoding="utf-8", newline="\n") as lock:
        json.dump(lock_payload, lock, indent=2, sort_keys=True)
        lock.write("\n")
        lock.flush()
        os.fsync(lock.fileno())

    base_status: dict[str, Any] = {
        "schema": SCHEMA,
        "job_id": args.job_id,
        "status": "RUNNING",
        "wrapper_pid": os.getpid(),
        "packet_path": str(packet_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_at": started_at,
        "completed_at": None,
        "exit_code": None,
        "output_sha256": None,
        "failure": None,
    }
    atomic_json(status_path, base_status)
    atomic_json(latest_path, base_status)

    exit_code = 1
    failure: str | None = None
    terminal_status = "FAILED"
    try:
        prompt = continuity_prompt(packet_path.read_text(encoding="utf-8"))
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                [
                    str(kimi_path),
                    "--add-dir",
                    str(kimmy_root),
                    "-p",
                    prompt,
                ],
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=None,
            )
        exit_code = int(completed.returncode)
        terminal_status = "SUCCEEDED" if exit_code == 0 else "FAILED"
        if exit_code != 0:
            failure = f"Kimi exited with code {exit_code}"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        finished = {
            **base_status,
            "status": terminal_status,
            "completed_at": utc_now(),
            "exit_code": exit_code,
            "output_sha256": sha256_file(stdout_path),
            "failure": failure,
        }
        atomic_json(status_path, finished)
        atomic_json(latest_path, finished)
        try:
            current_lock = json.loads(lock_path.read_text(encoding="utf-8"))
            if (
                current_lock.get("job_id") == args.job_id
                and current_lock.get("wrapper_pid") == os.getpid()
            ):
                lock_path.unlink()
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            # Preserve an unreadable/foreign lock for Codex to investigate.
            pass

    print(json.dumps(finished, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
