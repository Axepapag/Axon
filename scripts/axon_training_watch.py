#!/usr/bin/env python3
"""Axon Training Watch — a live terminal dashboard for cloud training.

Two data sources, chosen automatically:

* a local job's ``outputs/axon_observability/trainer/events.jsonl``
  (fast, offline, works after ``axon_kaggle.py fetch``), or
* Kaggle's live log stream for a running job (``kaggle kernels logs -f``),
  parsed for ``AXON_PROGRESS`` events as they happen.

Examples:
    python scripts/axon_training_watch.py <job-id>              # live follow
    python scripts/axon_training_watch.py <job-id> --local      # replay local file
    python scripts/axon_training_watch.py <job-id> --steps 50   # rolling window of 50 steps
    python scripts/axon_training_watch.py <job-id> --qa         # show Soul Q/A transcripts too
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE = ROOT / "State"
PYTHON = sys.executable or "python"

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
MAGENTA = "\033[35m"

LANE_COLORS = {
    "mechanism": MAGENTA,
    "ffcs-L0": CYAN,
    "ffcs-L1": CYAN,
    "ffcs-L2": CYAN,
    "ffcs-L3": CYAN,
    "ffcs-L4": CYAN,
}

SPARK_CHARS = " ▁▂▃▄▅▆▇█"


def _supports_ansi() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.name == "nt":
        os.system("")  # enables VT processing on Windows terminals
    return True


ANSI = _supports_ansi()


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}" if ANSI else text


def _spark(values: Iterable[float]) -> str:
    nums = [max(0.0, float(v)) for v in values]
    if not nums:
        return ""
    lo, hi = min(nums), max(nums)
    span = (hi - lo) or 1.0
    out = []
    for value in nums:
        level = int((value - lo) / span * (len(SPARK_CHARS) - 1))
        out.append(SPARK_CHARS[level])
    return "".join(out)


def _fmt_loss(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _short(value: Any, width: int = 12) -> str:
    text = str(value or "")
    return text if len(text) <= width else text[: width - 1] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Event model
# ─────────────────────────────────────────────────────────────────────────────


class Watcher:
    """Consumes AXON_PROGRESS / AXON_QA events and renders the dashboard."""

    def __init__(self, *, window: int, show_qa: bool, transcript_lines: int) -> None:
        self.window = window
        self.show_qa = show_qa
        self.transcript_lines = transcript_lines
        self.losses: deque[float] = deque(maxlen=window)
        self.lanes: deque[str] = deque(maxlen=window)
        self.wall: deque[float] = deque(maxlen=window)
        self.step_numbers: deque[int] = deque(maxlen=window)
        self.last_step = 0
        self.segment_end = None
        self.last_lane = None
        self.last_loss = None
        self.checkpoints = 0
        self.bundles = 0
        self.status = "waiting"
        self.started_at = None
        self.last_event_at = None
        self.accelerator = None
        self.candidate = None
        self.tranche_steps = None
        self.gates: dict[str, Any] = {}
        self.eval_summary: dict[str, Any] = {}
        self.eval_history: deque[tuple[str, float, float]] = deque(maxlen=8)
        self.qa_lines: deque[str] = deque(maxlen=transcript_lines)
        self.event_count = 0

    # -- ingestion ------------------------------------------------------------

    def consume(self, payload: dict[str, Any]) -> None:
        self.event_count += 1
        schema = str(payload.get("schema") or "")
        status = str(payload.get("status") or "")
        details = payload.get("details") or {}
        occurred = payload.get("occurred_at")
        if occurred:
            self.last_event_at = occurred
        if schema == "axon-training-progress-event-v1":
            if status == "starting":
                self.status = "starting"
                self.accelerator = details.get("accelerator")
                self.candidate = details.get("candidate_label")
                self.tranche_steps = details.get("tranche_steps")
                self.started_at = occurred
            elif status == "training":
                self.status = "training"
                step = int(details.get("global_step") or 0)
                loss = details.get("loss")
                lane = details.get("curriculum_lane")
                wall = details.get("wall_seconds")
                self.last_step = max(self.last_step, step)
                if "segment_end_step" in details:
                    self.segment_end = int(details["segment_end_step"])
                if loss is not None:
                    self.losses.append(float(loss))
                    self.last_loss = float(loss)
                if lane:
                    self.lanes.append(str(lane))
                    self.last_lane = str(lane)
                if wall is not None:
                    self.wall.append(float(wall))
                self.step_numbers.append(step)
                if details.get("checkpoint_id"):
                    self.checkpoints += 1
                if details.get("accepted_step_bundle_id"):
                    self.bundles += 1
            elif status == "evaluating":
                self.status = f"evaluating({details.get('phase', '?')})"
            elif status == "evaluated":
                self._consume_evaluated(details)
            elif status in {"completed", "paused"}:
                self.status = status
                self.gates = {
                    "task_gate_passed": details.get("task_gate_passed"),
                    "exact_serving_gate_passed": details.get("exact_serving_gate_passed"),
                    "curriculum_stage_complete": details.get("curriculum_stage_complete"),
                }
                if details.get("heldout_mean_loss") is not None:
                    self.eval_summary["heldout_mean_loss"] = details["heldout_mean_loss"]
                if details.get("report_path"):
                    self.eval_summary["report_path"] = details["report_path"]
        elif schema == "axon-training-eval-event-v1":
            self._consume_evaluated(details)
        elif schema == "axon-training-qa-event-v1":
            self.status = self.status if self.status != "waiting" else "training"
            self.render_qa_line(details)

    def _consume_evaluated(self, details: dict[str, Any]) -> None:
        """Detailed evaluation snapshot (loss/accuracy/QA) after a full pass."""
        self.eval_summary = {
            "heldout_mean_loss": details.get("heldout_mean_loss"),
            "typed_emission_exact_rate": details.get("typed_emission_exact_rate"),
            "payload_transport_exact_rate": details.get("payload_transport_exact_rate"),
            "token_accuracy": details.get("payload_teacher_forced_token_accuracy"),
            "token_accuracy_floor": details.get("constant_payload_token_accuracy_floor"),
            "evaluated_case_count": details.get("evaluated_case_count"),
            "phase": details.get("phase"),
            "report_path": self.eval_summary.get("report_path"),
        }
        if details.get("heldout_mean_loss") is not None:
            self.eval_history.append(
                (
                    str(details.get("phase")),
                    float(details["heldout_mean_loss"]),
                    float(details.get("payload_teacher_forced_token_accuracy") or 0.0),
                )
            )
        for row in details.get("qa_transcripts") or []:
            self.qa_lines.append(self._format_qa(row))

    def render_qa_line(self, details: dict[str, Any]) -> None:
        step = details.get("global_step", "?")
        lane = details.get("curriculum_lane", "?")
        question = _one_line(details.get("prompt") or details.get("question") or "")
        answer = _one_line(details.get("response") or details.get("answer") or "")
        correct = details.get("exact_match")
        mark = "?" if correct is None else ("✓" if correct else "✗")
        color = GREEN if correct else (RED if correct is False else YELLOW)
        self.qa_lines.append(
            "  "
            + _color(f"[step {step}]", DIM)
            + f" {lane}: "
            + _color("Q: ", BOLD)
            + question
            + "  "
            + _color("A: ", BOLD)
            + answer
            + " "
            + _color(mark, color)
        )

    def _format_qa(self, row: dict[str, Any]) -> str:
        prompt = _one_line(row.get("prompt") or "?", 46)
        predicted = _one_line(row.get("predicted_payload") or "", 28)
        expected = _one_line(row.get("expected_payload") or "", 28)
        correct = row.get("exact_match")
        mark = "?" if correct is None else ("✓" if correct else "✗")
        color = GREEN if correct else (RED if correct is False else YELLOW)
        return (
            "  "
            + _color("Q:", BOLD)
            + f" {prompt}  "
            + _color("A:", BOLD)
            + f" {predicted!r}"
            + (" (expected " + repr(expected) + ")" if not correct else "")
            + " "
            + _color(mark, color)
        )

    # -- rendering ------------------------------------------------------------

    def render(self) -> str:
        lines: list[str] = []
        title = "AXON TRAINING WATCH"
        lines.append(_color("=" * 74, CYAN))
        lines.append(_color(f"  {title}", BOLD + CYAN))
        lines.append(_color("=" * 74, CYAN))

        accel = self.accelerator or "?"
        candidate = self.candidate or "?"
        status_color = GREEN if self.status in {"training", "starting"} else YELLOW
        tranche = f"/{self.tranche_steps}" if self.tranche_steps else ""
        lines.append(
            f" candidate: {_short(candidate, 28)}  status: {_color(self.status, status_color)}"
            f"  accelerator: {accel}"
        )
        progress_bar = ""
        if self.tranche_steps:
            done = min(self.last_step, int(self.tranche_steps))
            filled = int(34 * done / max(1, int(self.tranche_steps)))
            progress_bar = _color("█" * filled, GREEN) + _color("·" * (34 - filled), DIM)
            lines.append(f" steps: {self.last_step}{tranche}  [{progress_bar}]")
        else:
            lines.append(f" steps: {self.last_step}")

        if self.losses:
            spark = _spark(list(self.losses))
            recent = list(self.losses)[-10:]
            lines.append(
                f" loss: last {_color(f'{self.last_loss:.3f}' if self.last_loss is not None else '-', BOLD)}"
                f"  recent10 mean {_fmt_loss(sum(recent) / len(recent))}"
                f"  window min/max {_fmt_loss(min(self.losses))}/{_fmt_loss(max(self.losses))}"
            )
            lines.append(f"       {spark}")
        lane_counts = Counter(self.lanes)
        if lane_counts:
            mix = "  ".join(f"{lane}:{count}" for lane, count in lane_counts.most_common(8))
            lines.append(f" lanes: {mix}")
        if self.wall:
            avg_wall = sum(self.wall) / len(self.wall)
            remaining = None
            if self.segment_end and self.last_step:
                remaining = max(0, self.segment_end - self.last_step) * avg_wall
            eta = f"  ETA ~{remaining/60:.0f}m" if remaining else ""
            lines.append(f" pace: {avg_wall:.1f}s/step  checkpoints:{self.checkpoints}  bundles:{self.bundles}{eta}")
        if self.eval_summary:
            heldout = self.eval_summary.get("heldout_mean_loss")
            metrics = [f"heldout_loss {_fmt_loss(heldout)}"]
            for label, key in (
                ("typed_exact", "typed_emission_exact_rate"),
                ("payload_exact", "payload_transport_exact_rate"),
                ("token_acc", "token_accuracy"),
                ("floor", "token_accuracy_floor"),
            ):
                value = self.eval_summary.get(key)
                if value is not None:
                    metrics.append(f"{label} {float(value)*100:.1f}%")
            phase = self.eval_summary.get("phase")
            lines.append(" eval[" + str(phase or "?") + "]: " + "  ".join(metrics))
        if self.eval_history:
            history = "  ".join(
                f"{phase}@{loss:.3f}/{acc*100:.0f}%" for phase, loss, acc in list(self.eval_history)[-4:]
            )
            lines.append(f" eval history: {history}")
        if self.gates:
            gate_text = "  ".join(
                f"{key}={_color('PASS' if value else 'FAIL', GREEN if value else RED)}"
                for key, value in self.gates.items()
                if value is not None
            )
            lines.append(f" gates: {gate_text}")
        lines.append(f" events: {self.event_count}  last: {self.last_event_at or '-'}")

        if self.show_qa:
            lines.append(_color("-" * 74, DIM))
            lines.append(_color(" Soul Q/A (live transcript)", BOLD))
            if self.qa_lines:
                lines.extend(list(self.qa_lines)[-self.transcript_lines:])
            else:
                lines.append(_color("  (no qa events in this stream yet)", DIM))

        lines.append(_color("-" * 74, DIM))
        lines.append(_color(" Ctrl+C to stop watching (never stops the cloud job).", DIM))
        return "\n".join(lines)


def _one_line(text: Any, width: int = 60) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= width else value[: width - 1] + "…"


# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────


def _iter_progress_lines(handle) -> Iterable[str]:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if "AXON_PROGRESS" in line or "AXON_QA" in line:
            start = line.find("{")
            if start < 0:
                continue
            candidate = line[start:]
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _local_events_path(job_id: str) -> Path:
    return DEFAULT_STATE / "training" / "cloud" / "jobs" / job_id / "outputs" / (
        "axon_observability/trainer/events.jsonl"
    )


def _follow_local(path: Path, watcher: Watcher, poll_seconds: float) -> int:
    print("source: local events.jsonl (tailing)")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        # skip existing content unless small; we tail from end
        handle.seek(0, 2)
        idle = 0.0
        while True:
            line = handle.readline()
            if line:
                stripped = line.strip()
                if stripped:
                    start = stripped.find("{")
                    if start >= 0:
                        try:
                            payload = json.loads(stripped[start:])
                            if isinstance(payload, dict):
                                watcher.consume(payload)
                        except json.JSONDecodeError:
                            pass
                idle = 0.0
            else:
                idle += poll_seconds
                print("\033[2J\033[H", end="")
                print(watcher.render())
                if idle >= 1.0:
                    time.sleep(poll_seconds)
                    idle = 0.0
                else:
                    time.sleep(0.2)


def _follow_kaggle(kernel_ref: str, watcher: Watcher) -> int:
    print(f"source: kaggle live log stream ({kernel_ref})")
    argv = ["kaggle", "kernels", "logs", "-f", kernel_ref]
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(ROOT),
    )
    assert process.stdout is not None
    for line in process.stdout:
        if "AXON_PROGRESS" in line or "AXON_QA" in line:
            start = line.find("{")
            if start < 0:
                continue
            try:
                payload = json.loads(line[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                watcher.consume(payload)
                print("\033[2J\033[H", end="")
                print(watcher.render())
    return process.wait()


def _replay_local(path: Path, watcher: Watcher, *, follow: bool, poll_seconds: float) -> int:
    print("source: local events.jsonl (replay)")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            start = stripped.find("{")
            if start < 0:
                continue
            try:
                payload = json.loads(stripped[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                watcher.consume(payload)
                rendered = watcher.render()
                print("\033[2J\033[H", end="")
                print(rendered)
    if follow:
        return _follow_local(path, watcher, poll_seconds)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_kernel_ref(job_id: str) -> str:
    record_path = DEFAULT_STATE / "training" / "cloud" / "jobs" / job_id / "job.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        kernel_ref = record.get("kernel_ref")
        if kernel_ref:
            return str(kernel_ref)
    raise SystemExit(f"unknown job id (no kernel_ref in {record_path}); pass --kernel owner/slug")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Axon cloud job id (or use --kernel for owner/slug)")
    parser.add_argument("--kernel", default=None, help="explicit kernel ref owner/slug")
    parser.add_argument("--local", action="store_true", help="read local fetched events instead of Kaggle live log")
    parser.add_argument("--replay", action="store_true", help="replay the whole local file, then follow")
    parser.add_argument("--steps", type=int, default=60, help="rolling window size for loss sparkline (default 60)")
    parser.add_argument("--qa", action="store_true", help="show Soul Q/A transcripts when the trainer emits them")
    parser.add_argument("--qa-lines", type=int, default=12, help="transcript lines to keep on screen")
    parser.add_argument("--poll", type=float, default=2.0, help="local tail poll seconds")
    args = parser.parse_args()

    job_id = args.job_id
    watcher = Watcher(window=max(5, args.steps), show_qa=args.qa, transcript_lines=max(1, args.qa_lines))

    if args.local or args.replay:
        path = _local_events_path(job_id)
        if not path.is_file():
            raise SystemExit(
                f"no local events at {path}; run 'axon_kaggle.py fetch {job_id}' first "
                "or drop --local to follow Kaggle live"
            )
        if args.replay:
            return _replay_local(path, watcher, follow=True, poll_seconds=args.poll)
        return _follow_local(path, watcher, args.poll)
    kernel_ref = args.kernel or _resolve_kernel_ref(job_id)
    return _follow_kaggle(kernel_ref, watcher)


if __name__ == "__main__":
    raise SystemExit(main())
