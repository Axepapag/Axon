"""Run one bounded Kimi Code job under Axon's canonical one-State root.

ChatGPT or another explicit orchestrator chooses when to run this supervisor.
It is safe to call from recurring automation because it refuses dirty worktrees
by default, holds one exclusive Kimi job lock, records durable evidence under
State/kimi_orchestrator, and never grants Kimi commit/push authority.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterator

SCHEMA = "axon-kimi-roundtable-job-v1"
LOCK_SCHEMA = "axon-kimi-roundtable-lock-v1"
JOB_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,96}$")
DEFAULT_MODEL = "kimi-code/k3"
AGENTS = {
    "audit": "axon-explorer",
    "triage": "axon-triage-lead",
    "plan": "axon-architect",
    "review": "axon-reviewer",
    "implement": "axon-swarm-lead",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "safe.directory=D:/Axon", *args],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@contextmanager
def exclusive_job_lock(state_root: Path, job_id: str) -> Iterator[Path]:
    """Hold the single Axon/Kimi mission lock for the complete job lifecycle."""

    lock_path = state_root / "active.lock.json"
    payload = {
        "schema": LOCK_SCHEMA,
        "job_id": job_id,
        "pid": os.getpid(),
        "started_at": utc_now(),
    }
    state_root.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SystemExit(
            f"another Kimi round-table job owns {lock_path}; inspect it before starting another mission"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield lock_path
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            if current.get("job_id") == job_id and current.get("pid") == os.getpid():
                lock_path.unlink()
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass


def build_prompt(mode: str, mission: str) -> str:
    mode_rules = {
        "audit": "Read-only. Do not modify repository or State files and do not run shell commands.",
        "triage": (
            "Read-only multi-agent triage. Delegate independent questions to bounded read-only "
            "project agents and merge their evidence; do not modify files or execute shell commands."
        ),
        "plan": "Read-only architecture work. Do not modify files or execute shell commands.",
        "review": "Read-only independent review. Do not modify files or execute shell commands.",
        "implement": (
            "A bounded axon-coder sub-agent may edit only files needed for this packet and run bounded tests. "
            "The swarm lead itself must not commit or push."
        ),
    }
    return f"""You are a Kimi Code sub-agent working for ChatGPT on Axon.

Before acting, read D:\\Axon\\AGENTS.md, docs\\WORKING_CONTRACT.md,
docs\\SOURCE_OF_TRUTH.md, roundtable\\ENGINEERS_LEDGER_PROTOCOL.md, and
roundtable\\ENGINEERS_LEDGER.md. Reconcile the current ledger before relying on
old packet text. For canonical-ledger reconciliation, read the rolling summary's
current-through event ID and inspect only the last few physical JSONL lines with
a negative Read offset; do not repeatedly Grep the entire canonical ledger.

MODE: {mode}
MODE RULE: {mode_rules[mode]}

Permanent boundaries:
- one living State root: D:\\Axon\\State;
- no fake/truncated core-facing anatomy;
- exact canonical 16D text and provenance stay authoritative;
- compiler has no reasoning or commit authority;
- never modify D:\\00, D:\\axon7, or protected archives;
- never launch long training, promote checkpoints, start persistent services,
  force-push, rewrite history, or delete evidence;
- do not edit the Axon engineer ledger or D:\\ChatGPT_State; ChatGPT owns those;
- do not commit or push. Return work to ChatGPT for independent verification.

MISSION:
{mission}

Your final answer must be evidence-based and identify exact files/tests. If a
material architecture ambiguity exists, FLAG it instead of silently choosing.
"""


def run_locked_job(
    *,
    repo: Path,
    state_root: Path,
    kimi: str,
    mode: str,
    model: str,
    mission: str,
    job_id: str,
    timeout_seconds: int,
) -> int:
    jobs_root = state_root / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    job_root = jobs_root / job_id
    if job_root.exists():
        raise SystemExit(f"job already exists: {job_root}")
    job_root.mkdir()

    prompt_path = job_root / "prompt.md"
    stdout_path = job_root / "stdout.txt"
    stderr_path = job_root / "stderr.txt"
    status_path = job_root / "status.json"
    prompt = build_prompt(mode, mission)
    prompt_path.write_text(prompt, encoding="utf-8")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "job_id": job_id,
        "mode": mode,
        "agent": AGENTS[mode],
        "model": model,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "exit_code": None,
        "prompt_path": str(prompt_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": None,
        "git_head_before": git(repo, "rev-parse", "HEAD").stdout.strip(),
        "failure": None,
    }
    atomic_json(status_path, payload)
    atomic_json(state_root / "latest.json", payload)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    command = [
        kimi,
        "--model",
        model,
        "--agent",
        AGENTS[mode],
        "--prompt",
        prompt,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        payload["exit_code"] = completed.returncode
        payload["status"] = "succeeded" if completed.returncode == 0 else "failed"
        if completed.returncode != 0:
            payload["failure"] = f"Kimi exited with code {completed.returncode}"
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        payload["status"] = "timeout"
        payload["failure"] = f"Kimi exceeded {timeout_seconds} seconds"
        payload["exit_code"] = 124
    finally:
        payload["completed_at"] = utc_now()
        payload["stdout_sha256"] = sha256_file(stdout_path)
        payload["git_head_after"] = git(repo, "rev-parse", "HEAD").stdout.strip()
        payload["git_status_after"] = git(repo, "status", "--porcelain").stdout.splitlines()
        atomic_json(status_path, payload)
        atomic_json(state_root / "latest.json", payload)

    print(json.dumps(payload, indent=2, sort_keys=True))
    if stdout_path.exists():
        print("\n--- KIMI RESULT ---\n")
        print(stdout_path.read_text(encoding="utf-8"))
    return int(payload["exit_code"] or 0)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(AGENTS), default="audit")
    mission_group = parser.add_mutually_exclusive_group(required=True)
    mission_group.add_argument("--mission")
    mission_group.add_argument("--mission-file", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    state_root = repo / "State" / "kimi_orchestrator"

    status = git(repo, "status", "--porcelain")
    if status.returncode != 0:
        raise SystemExit(f"cannot inspect Git status: {status.stderr.strip()}")
    if status.stdout.strip() and not args.allow_dirty:
        raise SystemExit(
            "refusing Kimi job on a dirty Axon worktree; review/publish the current mission first"
        )

    kimi = shutil.which("kimi")
    if not kimi:
        fallback = Path.home() / ".kimi-code" / "bin" / "kimi.exe"
        if fallback.is_file():
            kimi = str(fallback)
        else:
            raise SystemExit("Kimi Code CLI is not installed or not on PATH")

    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")

    if args.mission_file is not None:
        mission_path = args.mission_file.resolve(strict=True)
        try:
            mission_path.relative_to(repo)
        except ValueError as exc:
            raise SystemExit("--mission-file must be inside the Axon repository") from exc
        mission = mission_path.read_text(encoding="utf-8")
    else:
        mission = args.mission or ""

    job_id = args.job_id or datetime.now(timezone.utc).strftime(
        f"%Y%m%dT%H%M%SZ-{args.mode}"
    )
    if not JOB_RE.fullmatch(job_id):
        parser.error("job-id must use lowercase letters, digits, dot, underscore, or dash")

    with exclusive_job_lock(state_root, job_id):
        return run_locked_job(
            repo=repo,
            state_root=state_root,
            kimi=kimi,
            mode=args.mode,
            model=args.model,
            mission=mission,
            job_id=job_id,
            timeout_seconds=args.timeout_seconds,
        )


if __name__ == "__main__":
    raise SystemExit(main())
