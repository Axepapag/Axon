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
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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

    def __init__(
        self,
        *,
        window: int,
        show_qa: bool,
        transcript_lines: int,
        job_id: str | None = None,
        job_title: str | None = None,
    ) -> None:
        self.window = window
        self.show_qa = show_qa
        self.transcript_lines = transcript_lines
        self.job_id = job_id
        self.job_title = job_title
        self.losses: deque[float] = deque(maxlen=window)
        self.lanes: deque[str] = deque(maxlen=window)
        self.wall: deque[float] = deque(maxlen=window)
        self.step_numbers: deque[int] = deque(maxlen=window)
        self.recent_steps: deque[str] = deque(maxlen=8)
        self.last_step = 0
        self.segment_start = 0
        self.segment_end = None
        self.last_lane = None
        self.last_loss = None
        self.last_kind = None
        self.last_checkpoint = None
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
        self.motor_v2: dict[str, dict[str, Any]] = {}
        self.runner_lines: deque[str] = deque(maxlen=6)
        self.qa_lines: deque[str] = deque(maxlen=transcript_lines)
        self.event_count = 0
        self.seen_event_ids: set[str] = set()
        self.error = None
        self._lock = threading.Lock()
        self.sync_enabled: bool | None = None
        self.sync_waiting: str | None = None
        self.sync_verified_through: int | None = None
        self.sync_verified_ranges: list[list[int]] = []
        self.sync_released_ranges: list[list[int]] = []
        self.sync_last_pulled = 0
        self.sync_kernel_note: str | None = None
        self.sync_error_type: str | None = None

    # -- ingestion ------------------------------------------------------------

    def consume(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._consume_unlocked(payload)

    def note_sync(
        self,
        *,
        enabled: bool,
        waiting: str | None = None,
        verified_through_step: int | None = None,
        verified_ranges: list[list[int]] | None = None,
        released_ranges: list[list[int]] | None = None,
        pulled: int = 0,
        error_type: str | None = None,
        failed: bool = False,
    ) -> None:
        """Observation-only sync dashboard fields. Never continuation authority."""

        with self._lock:
            self.sync_enabled = enabled
            self.sync_waiting = waiting
            if verified_through_step is not None:
                self.sync_verified_through = verified_through_step
            if verified_ranges is not None:
                self.sync_verified_ranges = list(verified_ranges)
            if released_ranges is not None:
                self.sync_released_ranges = list(released_ranges)
            self.sync_last_pulled = int(pulled)
            self.sync_error_type = error_type
            if failed:
                self.status = "failed"

    def _consume_unlocked(self, payload: dict[str, Any]) -> None:
        # Kaggle's live stream can replay the same event (including on reconnect).
        # Never count replayed checkpoints, samples, or steps as new work.
        event_id = payload.get("event_id")
        if event_id:
            if event_id in self.seen_event_ids:
                return
            self.seen_event_ids.add(event_id)
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
                kind = details.get("material_kind")
                self.last_step = max(self.last_step, step)
                if "segment_end_step" in details:
                    self.segment_end = int(details["segment_end_step"])
                    if self.tranche_steps is not None:
                        self.segment_start = self.segment_end - int(self.tranche_steps)
                if loss is not None:
                    self.losses.append(float(loss))
                    self.last_loss = float(loss)
                if lane:
                    self.lanes.append(str(lane))
                    self.last_lane = str(lane)
                if kind:
                    self.last_kind = str(kind)
                if wall is not None:
                    self.wall.append(float(wall))
                self.step_numbers.append(step)
                checkpoint_id = details.get("checkpoint_id")
                if checkpoint_id:
                    self.checkpoints += 1
                    self.last_checkpoint = str(checkpoint_id)
                if details.get("accepted_step_bundle_id"):
                    self.bundles += 1
                loss_text = _fmt_loss(self.last_loss)
                lane_text = str(lane or self.last_lane or "-")
                kind_text = str(kind or self.last_kind or "-")
                wall_text = f"{float(wall):.1f}s" if wall is not None else "-"
                self.recent_steps.append(
                    f"  {step:>5}  loss {loss_text}  {lane_text}  {kind_text}  {wall_text}"
                )
            elif status == "evaluating":
                self.status = f"evaluating({details.get('phase', '?')})"
                if details.get("phase") == "initial":
                    self.segment_start = int(details.get("global_step") or 0)
                    self.last_step = max(self.last_step, self.segment_start)
            elif status == "evaluated":
                self._consume_evaluated(details)
                self._consume_motor_v2(details)
            elif status in {"completed", "paused"}:
                self.status = status
                self.gates = {
                    "task_gate_passed": details.get("task_gate_passed"),
                    "nonzero_exact_output_observed": details.get(
                        "nonzero_exact_output_observed"
                    ),
                    "exact_serving_gate_passed": details.get("exact_serving_gate_passed"),
                    "curriculum_stage_complete": details.get("curriculum_stage_complete"),
                }
                if details.get("heldout_mean_loss") is not None:
                    self.eval_summary["heldout_mean_loss"] = details["heldout_mean_loss"]
                if details.get("report_path"):
                    self.eval_summary["report_path"] = details["report_path"]
        elif schema == "axon-kaggle-runner-event-v1":
            detail_text = _one_line(
                details.get("error")
                or details.get("selected")
                or details.get("job_id")
                or details.get("accelerator")
                or "",
                48,
            )
            self.runner_lines.append(f"  runner {status}" + (f"  {detail_text}" if detail_text else ""))
            if status == "failed":
                self.status = "failed"
                self.error = details.get("error") or f"training process exited {details.get('returncode', '?')}"
            elif status == "python_selected":
                self.accelerator = (details.get("probe") or {}).get("device")
            elif status in {"starting", "running", "completed"}:
                self.status = f"runner({status})"
        elif schema == "axon-training-eval-event-v1":
            self._consume_evaluated(details)
        elif schema == "axon-training-qa-event-v1":
            self.status = self.status if self.status != "waiting" else "training"
            self.render_qa_line(details)
        elif schema == "axon-mid-run-sync-receipt-v1":
            rng = details.get("step_range")
            range_text = ""
            if isinstance(rng, list) and len(rng) == 2:
                range_text = f" steps {rng[0]}-{rng[1]}"
            self.sync_kernel_note = f"{status or 'sync'}{range_text}"

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

    def _consume_motor_v2(self, details: dict[str, Any]) -> None:
        phase = str(details.get("phase") or "?")
        heldout = details.get("foundation_motor_v2_heldout_probe")
        regression = details.get("foundation_motor_v2_regression_probe")
        if not isinstance(heldout, dict) and not isinstance(regression, dict):
            return
        self.motor_v2[phase] = {
            "heldout": heldout if isinstance(heldout, dict) else {},
            "regression": regression if isinstance(regression, dict) else {},
        }

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
        with self._lock:
            return self._render_unlocked()

    def _render_unlocked(self) -> str:
        lines: list[str] = []
        title = "AXON TRAINING WATCH"
        lines.append(_color("=" * 74, CYAN))
        lines.append(_color(f"  {title}", BOLD + CYAN))
        lines.append(_color("=" * 74, CYAN))
        if self.job_title:
            lines.append(f" job: {self.job_title}")
        if self.job_id:
            lines.append(f" id:  {self.job_id}")

        accel = self.accelerator or "?"
        candidate = self.candidate or "?"
        status_color = GREEN if self.status in {"training", "starting", "runner(running)"} else YELLOW
        lines.append(
            f" candidate: {_short(candidate, 28)}  status: {_color(self.status, status_color)}"
            f"  accelerator: {accel}"
        )
        progress_bar = ""
        if self.tranche_steps:
            done = min(max(0, self.last_step - self.segment_start), int(self.tranche_steps))
            filled = int(34 * done / max(1, int(self.tranche_steps)))
            progress_bar = _color("█" * filled, GREEN) + _color("·" * (34 - filled), DIM)
            lines.append(
                f" tranche: {done}/{self.tranche_steps}  global step: {self.last_step}  [{progress_bar}]"
            )
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
        for phase, probes in self.motor_v2.items():
            for split, probe in probes.items():
                rendered = _motor_v2_line(f"motor v2 {phase}/{split}", probe)
                if rendered:
                    lines.append(rendered)
        if self.recent_steps:
            lines.append(_color(" last steps:", DIM))
            lines.extend(list(self.recent_steps)[-5:])
        if self.last_checkpoint:
            lines.append(f" last checkpoint: {_short(self.last_checkpoint, 16)}")
        if self.sync_enabled is False:
            lines.append(_color(" sync: recipe did not enable mid-run checkpoint uploads", DIM))
        elif self.sync_enabled:
            bits = [" sync:"]
            if self.sync_verified_through is not None:
                bits.append(f"verified through step {self.sync_verified_through}")
                bits.append(f"kept {len(self.sync_verified_ranges)}/3 windows")
            if self.sync_released_ranges:
                bits.append(f"released {len(self.sync_released_ranges)} older payloads")
            if self.sync_last_pulled:
                bits.append(f"new +{self.sync_last_pulled}")
            if self.sync_waiting:
                bits.append(self.sync_waiting)
            if self.sync_kernel_note:
                bits.append(f"kernel {self.sync_kernel_note}")
            if self.sync_error_type:
                bits.append(_color(self.sync_error_type, YELLOW))
            if len(bits) == 1:
                bits.append("waiting for first checkpoint upload")
            lines.append("  ".join(bits))
        elif self.sync_kernel_note:
            lines.append(f" sync kernel: {self.sync_kernel_note}")
        if self.runner_lines:
            lines.append(_color(" runner:", DIM))
            lines.extend(list(self.runner_lines)[-4:])
        if self.gates:
            gate_text = "  ".join(
                f"{key}={_color('PASS' if value else 'FAIL', GREEN if value else RED)}"
                for key, value in self.gates.items()
                if value is not None
            )
            lines.append(f" gates: {gate_text}")
        lines.append(f" events: {self.event_count}  last: {self.last_event_at or '-'}")
        if self.error:
            lines.append(_color(f" ERROR: {self.error}", RED))

        if self.show_qa:
            lines.append(_color("-" * 74, DIM))
            lines.append(_color(" Teacher-forced payload samples (not autonomous conversation)", BOLD))
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


def _rate_text(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _motor_v2_line(label: str, probe: dict[str, Any] | None) -> str | None:
    if not probe:
        return None
    gate = probe.get("alignment_copy_gate_accuracy")
    position = probe.get("alignment_position_accuracy")
    if gate is None and position is None:
        return None
    pair_gate = probe.get("pair_copy_gate")
    pair_position = probe.get("pair_position")
    cases = probe.get("case_count")
    position_color = GREEN if position == 1.0 else (YELLOW if position is not None else DIM)
    parts = [
        f" {label}:",
        f"copy-gate {_rate_text(gate)}",
        _color(f"position {_rate_text(position)}", position_color),
    ]
    if pair_gate is not None or pair_position is not None:
        parts.append(f"pair-gate {_rate_text(pair_gate)}  pair-pos {_rate_text(pair_position)}")
    if cases is not None:
        parts.append(f"n={cases}")
    return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Sources
# ─────────────────────────────────────────────────────────────────────────────


def _iter_progress_lines(handle) -> Iterable[str]:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if any(marker in line for marker in ("AXON_PROGRESS", "AXON_QA", "AXON_KAGGLE", "AXON_SYNC")):
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


def job_enables_mid_run_sync(job_id: str, *, state_root: Path = DEFAULT_STATE) -> bool:
    """True when this job opted into observation-only mid-run checkpoint uploads."""

    record_path = state_root / "training" / "cloud" / "jobs" / job_id / "job.json"
    if record_path.is_file():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
        if isinstance(record, dict) and (record.get("sync_mid_run") or record.get("sync_dataset_ref")):
            return True
    manifest_path = state_root / "training" / "cloud" / "jobs" / job_id / "packet_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        config = manifest.get("config") if isinstance(manifest, dict) else None
        if isinstance(config, dict) and config.get("sync_mid_run"):
            return True
    return False


def apply_sync_pull(
    job_id: str,
    watcher: Watcher,
    *,
    adapter: Any | None = None,
    state_root: Path | None = None,
) -> dict[str, Any]:
    """One observation-only pull + payload retention. Missing dataset is waiting."""

    from runtime.trainer.cloud_jobs import CloudPacketError
    from runtime.trainer.kaggle_adapter import KaggleTrainerAdapter, SyncDatasetUnavailable

    if adapter is not None:
        root = Path(adapter.state_root)
    elif state_root is not None:
        root = Path(state_root)
    else:
        root = DEFAULT_STATE
    if not job_enables_mid_run_sync(job_id, state_root=root):
        watcher.note_sync(
            enabled=False,
            waiting="recipe did not enable mid-run checkpoint uploads",
        )
        return {"skipped": True, "reason": "sync_mid_run_disabled"}
    if adapter is None:
        adapter = KaggleTrainerAdapter(repo_root=ROOT, state_root=root)
    try:
        pulled = adapter.sync_pull(job_id)
        status = adapter.sync_status(job_id, rehash=False)
    except SyncDatasetUnavailable:
        if watcher.sync_verified_through is None:
            watcher.note_sync(enabled=True, waiting="waiting for first checkpoint upload")
        else:
            watcher.note_sync(
                enabled=True,
                waiting="waiting on next Kaggle dataset version",
                verified_through_step=watcher.sync_verified_through,
                verified_ranges=list(watcher.sync_verified_ranges),
                released_ranges=list(watcher.sync_released_ranges),
            )
        return {"waiting": True}
    except CloudPacketError as exc:
        watcher.note_sync(
            enabled=True,
            waiting="sync integrity/provider failure",
            error_type=type(exc).__name__,
            failed=True,
        )
        return {"fatal": True, "error_type": type(exc).__name__, "error": str(exc)}
    except Exception as exc:
        watcher.note_sync(
            enabled=True,
            waiting="sync poll failure",
            error_type=type(exc).__name__,
            failed=True,
        )
        return {"fatal": True, "error_type": type(exc).__name__, "error": str(exc)}
    waiting = None
    if not status.get("verified_ranges"):
        waiting = "waiting for first checkpoint upload"
    watcher.note_sync(
        enabled=True,
        waiting=waiting,
        verified_through_step=status.get("verified_through_step"),
        verified_ranges=list(status.get("verified_ranges") or []),
        released_ranges=list(status.get("released_ranges") or []),
        pulled=len(pulled.get("pulled") or []),
    )
    return pulled


def _sync_poll_loop(
    job_id: str,
    watcher: Watcher,
    interval: float,
    redraw,
    stop: threading.Event,
) -> None:
    from runtime.trainer.kaggle_adapter import KaggleTrainerAdapter

    if not job_enables_mid_run_sync(job_id):
        apply_sync_pull(job_id, watcher)
        redraw()
        return
    adapter = KaggleTrainerAdapter(repo_root=ROOT, state_root=DEFAULT_STATE)
    first = True
    while not stop.is_set():
        if not first and stop.wait(interval):
            break
        first = False
        result = apply_sync_pull(job_id, watcher, adapter=adapter)
        redraw()
        if result.get("fatal"):
            break


def _follow_kaggle(
    kernel_ref: str,
    watcher: Watcher,
    *,
    job_id: str | None = None,
    sync_poll: bool = True,
    sync_interval: float = 30.0,
) -> int:
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
    stop = threading.Event()
    render_lock = threading.Lock()

    def redraw() -> None:
        with render_lock:
            print("\033[2J\033[H", end="")
            print(watcher.render(), flush=True)

    poller = None
    if sync_poll and job_id:
        poller = threading.Thread(
            target=_sync_poll_loop,
            args=(job_id, watcher, max(5.0, float(sync_interval)), redraw, stop),
            daemon=True,
            name="axon-sync-poll",
        )
        poller.start()
    try:
        for line in process.stdout:
            if any(marker in line for marker in ("AXON_PROGRESS", "AXON_QA", "AXON_KAGGLE", "AXON_SYNC")):
                start = line.find("{")
                if start < 0:
                    continue
                try:
                    payload = json.loads(line[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    watcher.consume(payload)
                    redraw()
            else:
                # Provider/authentication/traceback diagnostics must not disappear
                # just because the trainer never managed to emit a progress event.
                print(line, end="", flush=True)
        return process.wait()
    finally:
        stop.set()
        if poller is not None:
            poller.join(timeout=2.0)


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


def _job_title(job_id: str) -> str | None:
    manifest_path = (
        DEFAULT_STATE / "training" / "cloud" / "jobs" / job_id / "packet_manifest.json"
    )
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    config = manifest.get("config") if isinstance(manifest, dict) else None
    if isinstance(config, dict) and config.get("name"):
        return str(config["name"])
    return None


def follow_job(
    job_id: str,
    *,
    kernel: str | None = None,
    qa: bool = False,
    steps: int = 60,
    local: bool = False,
    replay: bool = False,
    poll: float = 2.0,
    qa_lines: int = 12,
    sync_poll: bool = True,
    sync_interval: float = 30.0,
) -> int:
    watcher = Watcher(
        window=max(5, steps),
        show_qa=qa,
        transcript_lines=max(1, qa_lines),
        job_id=job_id,
        job_title=_job_title(job_id),
    )
    if local or replay:
        path = _local_events_path(job_id)
        if not path.is_file():
            raise SystemExit(
                f"no local events at {path}; run 'axon_kaggle.py fetch {job_id}' first "
                "or drop --local to follow Kaggle live"
            )
        if replay:
            return _replay_local(path, watcher, follow=True, poll_seconds=poll)
        return _follow_local(path, watcher, poll)
    kernel_ref = kernel or _resolve_kernel_ref(job_id)
    return _follow_kaggle(
        kernel_ref,
        watcher,
        job_id=job_id,
        sync_poll=sync_poll,
        sync_interval=sync_interval,
    )


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
    parser.add_argument(
        "--no-sync-poll",
        action="store_true",
        help="do not auto-download mid-run checkpoint windows while following",
    )
    parser.add_argument(
        "--sync-interval",
        type=float,
        default=30.0,
        help="seconds between observation-only sync pulls (default 30)",
    )
    args = parser.parse_args()
    return follow_job(
        args.job_id,
        kernel=args.kernel,
        qa=args.qa,
        steps=args.steps,
        local=args.local,
        replay=args.replay,
        poll=args.poll,
        qa_lines=args.qa_lines,
        sync_poll=not args.no_sync_poll,
        sync_interval=args.sync_interval,
    )


if __name__ == "__main__":
    raise SystemExit(main())
