"""Round Table Orchestrator waker — sequential turn runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import buslink, prompts, replies, transcript
from .manifests import AgentManifest, load_all_manifests, load_manifest
from .prompts import BudgetView, PromptContext


StateRoot = Path("State") / "table"
RoundsDir = StateRoot / "rounds"
RoundConfigName = "round.json"
ManualReplyFileName = "manual_reply.json"


@dataclass
class RoundConfig:
    round_id: str
    brief_path: str
    seats: list[str]
    turn_order: str
    synthesizer: str
    max_cycles: int
    per_turn_timeout_s: int
    budgets: dict[str, int]
    end_conditions: list[str]
    doctrine_stamp: dict[str, str]
    status: str
    offline: bool = False
    officers: list[str] | None = None
    current_cycle: int = 1
    current_seat_index: int = 0
    failures: dict[str, int] | None = None
    disabled_seats: list[str] | None = None

    @property
    def budget_view(self) -> BudgetView:
        return BudgetView(
            max_invocations=self.budgets.get("max_invocations", 16),
            board_view_events=self.budgets.get("board_view_events", 40),
            board_view_chars=self.budgets.get("board_view_chars", 24000),
        )


class WakerError(Exception):
    pass


class StaleDoctrineError(WakerError):
    pass


class RoundTable:
    """Sequential round-table orchestrator."""

    def __init__(
        self,
        state_root: str | Path | None = None,
        manifests_dir: str | Path | None = None,
    ) -> None:
        self.state_root = Path(state_root) if state_root else StateRoot
        self.rounds_dir = self.state_root / "rounds"
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.manifests = load_all_manifests(manifests_dir)
        self.bus = buslink.BusLink(state_dir=self.state_root, offline=True)

    # ------------------------------------------------------------------
    # Config IO
    # ------------------------------------------------------------------
    def _round_dir(self, round_id: str) -> Path:
        return self.rounds_dir / round_id

    def _config_path(self, round_id: str) -> Path:
        return self._round_dir(round_id) / RoundConfigName

    def _load_config(self, round_id: str) -> RoundConfig:
        path = self._config_path(round_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return RoundConfig(**data)

    def _save_config(self, cfg: RoundConfig) -> None:
        path = self._config_path(cfg.round_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "round_id": cfg.round_id,
                    "brief_path": cfg.brief_path,
                    "seats": cfg.seats,
                    "turn_order": cfg.turn_order,
                    "synthesizer": cfg.synthesizer,
                    "max_cycles": cfg.max_cycles,
                    "per_turn_timeout_s": cfg.per_turn_timeout_s,
                    "budgets": cfg.budgets,
                    "end_conditions": cfg.end_conditions,
                    "doctrine_stamp": cfg.doctrine_stamp,
                    "status": cfg.status,
                    "offline": cfg.offline,
                    "officers": cfg.officers,
                    "current_cycle": cfg.current_cycle,
                    "current_seat_index": cfg.current_seat_index,
                    "failures": cfg.failures,
                    "disabled_seats": cfg.disabled_seats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _compute_sha256(path: str | Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------
    def open_round(
        self,
        round_id: str,
        brief_path: str,
        seats: list[str],
        synthesizer: str,
        max_cycles: int = 3,
        per_turn_timeout_s: int = 600,
        budgets: dict[str, int] | None = None,
        end_conditions: list[str] | None = None,
        sot_path: str = "docs/SOURCE_OF_TRUTH.md",
        offline: bool = False,
        officers: list[str] | None = None,
    ) -> RoundConfig:
        if self._config_path(round_id).exists():
            raise WakerError(f"round {round_id} already exists")

        brief = Path(brief_path)
        if not brief.exists():
            raise WakerError(f"brief not found: {brief_path}")

        cfg = RoundConfig(
            round_id=round_id,
            brief_path=str(brief.resolve()),
            seats=seats,
            turn_order="as_listed",
            synthesizer=synthesizer,
            max_cycles=max_cycles,
            per_turn_timeout_s=per_turn_timeout_s,
            budgets=budgets
            or {
                "max_invocations": len(seats) * max_cycles + 2,
                "board_view_events": 40,
                "board_view_chars": 24000,
            },
            end_conditions=end_conditions or ["max_cycles", "all_pass", "convener_stop"],
            doctrine_stamp={"sot_path": sot_path, "sot_sha256": self._compute_sha256(sot_path)},
            status="open",
            offline=offline,
            officers=officers or ["codex", "claude"],
            failures={},
            disabled_seats=[],
        )
        self._save_config(cfg)
        # Initialize empty transcript files.
        transcript.append_entries(self._round_dir(round_id), [])
        self.bus.offline = offline
        return cfg

    def status(self, round_id: str) -> dict[str, Any]:
        cfg = self._load_config(round_id)
        entries = transcript.read_jsonl(transcript.get_transcript_path(self._round_dir(round_id)))
        invocation_count = sum(1 for e in entries if e.get("type") == "turn" and e.get("status") == "completed")
        return {
            "round_id": cfg.round_id,
            "status": cfg.status,
            "cycle": cfg.current_cycle,
            "seat_index": cfg.current_seat_index,
            "seats": cfg.seats,
            "disabled_seats": cfg.disabled_seats,
            "invocations": invocation_count,
            "budget_invocations": cfg.budget_view.max_invocations,
            "offline": cfg.offline,
        }

    def pause(self, round_id: str) -> RoundConfig:
        cfg = self._load_config(round_id)
        cfg.status = "paused"
        self._save_config(cfg)
        return cfg

    def resume(self, round_id: str) -> RoundConfig:
        cfg = self._load_config(round_id)
        cfg.status = "open"
        self._save_config(cfg)
        return cfg

    def close(self, round_id: str, reason: str) -> RoundConfig:
        cfg = self._load_config(round_id)
        cfg.status = "closed"
        self._save_config(cfg)
        return cfg

    # ------------------------------------------------------------------
    # Doctrine check
    # ------------------------------------------------------------------
    def _verify_doctrine(self, cfg: RoundConfig) -> None:
        path = Path(cfg.doctrine_stamp["sot_path"])
        if not path.exists():
            raise StaleDoctrineError(f"SOT path missing: {path}")
        current = self._compute_sha256(path)
        expected = cfg.doctrine_stamp.get("sot_sha256", "")
        if current != expected:
            raise StaleDoctrineError(
                f"doctrine_stamp mismatch: expected {expected[:16]}..., got {current[:16]}..."
            )

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    def _build_prompt_for_seat(
        self,
        cfg: RoundConfig,
        seat_id: str,
    ) -> tuple[str, dict[str, Any]]:
        manifest = self.manifests.get(seat_id)
        display_name = manifest.display_name if manifest else seat_id
        brief_text = Path(cfg.brief_path).read_text(encoding="utf-8")
        entries = transcript.read_jsonl(transcript.get_transcript_path(self._round_dir(cfg.round_id)))

        ctx = PromptContext(
            round_id=cfg.round_id,
            cycle=cfg.current_cycle,
            max_cycles=cfg.max_cycles,
            seat=seat_id,
            seat_display=display_name,
            standing_question=brief_text.splitlines()[0] if brief_text else cfg.brief_path,
            officers=cfg.officers or [],
            synthesizer=cfg.synthesizer,
            budgets=cfg.budget_view,
            doctrine_stamp=cfg.doctrine_stamp,
            brief_text=brief_text,
            transcript_entries=entries,
            unread_dms=[],
        )
        return prompts.build_prompt(ctx)

    # ------------------------------------------------------------------
    # Wake / invoke
    # ------------------------------------------------------------------
    def _wake_seat(
        self,
        cfg: RoundConfig,
        seat_id: str,
        prompt_text: str,
    ) -> tuple[str, str, int | None]:
        manifest = self.manifests.get(seat_id)
        if manifest is None:
            raise WakerError(f"no manifest for seat {seat_id}")

        if manifest.is_manual:
            return self._wake_manual(cfg, seat_id, prompt_text)

        timeout = min(cfg.per_turn_timeout_s, manifest.wake.timeout_seconds)
        argv = [arg.replace("{prompt}", prompt_text) for arg in manifest.wake.argv]
        workdir = Path(manifest.wake.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        stdin_payload: str | None = None
        if manifest.wake.prompt_via == "stdin":
            stdin_payload = prompt_text
        elif manifest.wake.prompt_via == "file":
            # Write prompt to a temp file and substitute placeholder if present.
            fd, tmp_path = tempfile.mkstemp(suffix=".txt", dir=workdir)
            try:
                with open(fd, "w", encoding="utf-8") as f:
                    f.write(prompt_text)
                argv = [arg.replace("{prompt_file}", tmp_path) for arg in argv]
            except Exception:
                import os

                os.close(fd)
                raise

        try:
            result = subprocess.run(
                argv,
                input=stdin_payload,
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return stdout, stderr + "\n[timeout]", -1
        finally:
            if manifest.wake.prompt_via == "file":
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _wake_manual(
        self,
        cfg: RoundConfig,
        seat_id: str,
        prompt_text: str,
    ) -> tuple[str, str, int]:
        round_dir = self._round_dir(cfg.round_id)
        prompt_path = round_dir / f"manual_prompt_{seat_id}.md"
        reply_path = round_dir / ManualReplyFileName
        prompt_path.write_text(prompt_text, encoding="utf-8")
        # Wait for the manual reply file to appear.
        print(
            f"[manual turn] reply expected at {reply_path} for seat {seat_id}; "
            f"prompt written to {prompt_path}",
            file=sys.stderr,
        )
        while not reply_path.exists():
            time.sleep(1)
        try:
            data = json.loads(reply_path.read_text(encoding="utf-8"))
            return str(data.get("stdout", "")), str(data.get("stderr", "")), int(data.get("returncode", 0))
        finally:
            reply_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------
    def _run_single_turn(
        self,
        cfg: RoundConfig,
        seat_id: str,
    ) -> dict[str, Any]:
        self._verify_doctrine(cfg)

        prompt_text, prompt_meta = self._build_prompt_for_seat(cfg, seat_id)
        round_dir = self._round_dir(cfg.round_id)

        stdout, stderr, returncode = self._wake_seat(cfg, seat_id, prompt_text)

        if returncode == -1:
            # Timeout
            entry = self._record_failure(cfg, seat_id, "timeout", stdout, stderr, returncode)
            self._publish_turn_event(cfg, seat_id, "agent.{seat}.turn.failed", entry)
            return entry

        if returncode != 0:
            entry = self._record_failure(cfg, seat_id, "nonzero_exit", stdout, stderr, returncode)
            self._publish_turn_event(cfg, seat_id, "agent.{seat}.turn.failed", entry)
            return entry

        actions = replies.parse_reply_all(stdout, stderr)
        entry = self._record_turn(cfg, seat_id, actions, prompt_meta, stdout)

        for action in actions:
            topic = self._topic_for_action(action)
            if topic:
                self._publish_turn_event(cfg, seat_id, topic, entry)
        self._publish_turn_event(cfg, seat_id, "agent.{seat}.turn.completed", entry)

        return entry

    def _topic_for_action(self, action: dict[str, Any]) -> str | None:
        t = action.get("type")
        if t == "post":
            return "committee.message.created"
        if t == "dm":
            return "committee.message.created"
        if t == "vote":
            return "committee.vote.cast"
        if t == "artifact":
            return "committee.delta.created"
        if t == "flag":
            return "committee.resolution.proposed"
        return None

    def _record_turn(
        self,
        cfg: RoundConfig,
        seat_id: str,
        actions: list[dict[str, Any]],
        prompt_meta: dict[str, Any],
        raw_stdout: str,
    ) -> dict[str, Any]:
        entry = {
            "type": "turn",
            "status": "completed",
            "seat": seat_id,
            "cycle": cfg.current_cycle,
            "action": actions[0] if actions else {"type": "pass"},
            "actions": actions,
            "metadata": prompt_meta,
            "timestamp": time.time(),
        }
        self._append_entry(cfg, entry)
        return entry

    def _record_failure(
        self,
        cfg: RoundConfig,
        seat_id: str,
        reason: str,
        stdout: str,
        stderr: str,
        returncode: int | None,
    ) -> dict[str, Any]:
        entry = {
            "type": "turn",
            "status": "failed",
            "seat": seat_id,
            "cycle": cfg.current_cycle,
            "error": {
                "reason": reason,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
            "timestamp": time.time(),
        }
        self._append_entry(cfg, entry)
        # Track consecutive failures for auto-disable.
        cfg.failures = cfg.failures or {}
        cfg.failures[seat_id] = cfg.failures.get(seat_id, 0) + 1
        if cfg.failures[seat_id] >= 2 and seat_id not in cfg.disabled_seats:
            cfg.disabled_seats = (cfg.disabled_seats or []) + [seat_id]
        self._save_config(cfg)
        return entry

    def _append_entry(self, cfg: RoundConfig, entry: dict[str, Any]) -> None:
        round_dir = self._round_dir(cfg.round_id)
        path = transcript.get_transcript_path(round_dir)
        existing = transcript.read_jsonl(path)
        transcript.append_entries(round_dir, existing + [entry])

    def _publish_turn_event(
        self,
        cfg: RoundConfig,
        seat_id: str,
        topic_template: str,
        entry: dict[str, Any],
    ) -> None:
        topic = topic_template.format(seat=seat_id)
        payload = {
            "round_id": cfg.round_id,
            "seat": seat_id,
            "cycle": cfg.current_cycle,
            "entry": entry,
        }
        if cfg.offline:
            self.bus.publish_sync(topic, payload)
        else:
            # Best-effort synchronous HTTP publish; full async flush happens in run_round.
            self.bus.publish_sync(topic, payload)

    # ------------------------------------------------------------------
    # End conditions
    # ------------------------------------------------------------------
    def _check_end(self, cfg: RoundConfig) -> bool:
        if "convener_stop" in cfg.end_conditions and cfg.status == "closed":
            return True
        if "max_cycles" in cfg.end_conditions and cfg.current_cycle > cfg.max_cycles:
            return True
        if "all_pass" in cfg.end_conditions:
            if self._all_passed_last_cycle(cfg):
                return True
        return False

    def _all_passed_last_cycle(self, cfg: RoundConfig) -> bool:
        entries = transcript.read_jsonl(transcript.get_transcript_path(self._round_dir(cfg.round_id)))
        # Look at entries from the most recently completed cycle.
        cycle_entries = [e for e in entries if e.get("cycle") == cfg.current_cycle - 1 and e.get("type") == "turn"]
        if not cycle_entries:
            return False
        seats_in_cycle = {e.get("seat") for e in cycle_entries}
        for seat in cfg.seats:
            if seat in cfg.disabled_seats:
                continue
            if seat not in seats_in_cycle:
                return False
            seat_entries = [e for e in cycle_entries if e.get("seat") == seat]
            last = seat_entries[-1]
            actions = last.get("actions", [last.get("action", {})])
            if not all(a.get("type") == "pass" for a in actions):
                return False
        return True

    # ------------------------------------------------------------------
    # Synthesizer
    # ------------------------------------------------------------------
    def _run_synthesizer(self, cfg: RoundConfig) -> Path | None:
        entries = transcript.read_jsonl(transcript.get_transcript_path(self._round_dir(cfg.round_id)))
        brief_text = Path(cfg.brief_path).read_text(encoding="utf-8")
        transcript_text = transcript.render_markdown(entries)

        prompt = f"""You are the synthesizer for round {cfg.round_id}.

=== ROUND HEADER ===
round_id: {cfg.round_id}

Read the brief and the full transcript below, then draft a resolution document.
Reply with a JSON object of type "artifact" where path is
"docs/roundtable/RESOLUTION_{cfg.round_id}.md" and body is the full markdown
content of the resolution. The resolution should summarize the discussion,
capture any decisions, and note dissenting flags or open questions.

=== BRIEF ===
{brief_text}

=== FULL TRANSCRIPT ===
{transcript_text}

=== REPLY CONTRACT ===
End with a fenced JSON block:
```json
{{
  "type": "artifact",
  "artifact": {{
    "path": "docs/roundtable/RESOLUTION_{cfg.round_id}.md",
    "body": "# RESOLUTION..."
  }},
  "stamp": "Synthesizer / table / date"
}}
```
"""
        stdout, stderr, returncode = self._wake_seat(cfg, cfg.synthesizer, prompt)
        if returncode != 0:
            self._record_failure(cfg, cfg.synthesizer, "synthesizer_failed", stdout, stderr, returncode)
            return None

        actions = replies.parse_reply_all(stdout, stderr)
        artifact = None
        for action in actions:
            if action.get("type") == "artifact" and "artifact" in action:
                artifact = action["artifact"]
                break

        expected = Path("docs/roundtable") / f"RESOLUTION_{cfg.round_id}.md"
        if artifact is None:
            # Tool-using seats (kimi, codex, claude CLIs) often write the
            # resolution file directly instead of inlining a large markdown
            # body in JSON (raw newlines make the inline form invalid JSON).
            # Accept the file on disk as the artifact.
            if expected.is_file() and expected.stat().st_size > 0:
                return expected
            self._record_failure(cfg, cfg.synthesizer, "synthesizer_no_artifact", stdout, stderr, returncode)
            return None

        if not artifact.get("body"):
            # Path-only artifact reply: the seat wrote the file itself.
            claimed = Path(artifact.get("path", expected))
            if claimed.is_file() and claimed.stat().st_size > 0 and str(
                claimed.resolve()
            ).startswith(str(Path("docs/roundtable").resolve())):
                return claimed
            self._record_failure(cfg, cfg.synthesizer, "synthesizer_empty_artifact", stdout, stderr, returncode)
            return None

        path = Path(artifact["path"])
        # Restrict writes to docs/roundtable/.
        if not str(path.resolve()).startswith(str(Path("docs/roundtable").resolve())):
            self._record_failure(
                cfg,
                cfg.synthesizer,
                "synthesizer_bad_path",
                f"resolved path: {path.resolve()}",
                "",
                returncode,
            )
            return None

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(artifact["body"], encoding="utf-8")

        entry = {
            "type": "synthesis",
            "status": "completed",
            "seat": cfg.synthesizer,
            "cycle": cfg.current_cycle,
            "artifact_path": str(path),
            "timestamp": time.time(),
        }
        self._append_entry(cfg, entry)
        self._publish_turn_event(cfg, cfg.synthesizer, "committee.resolution.proposed", entry)

        cfg.status = "awaiting_convener"
        self._save_config(cfg)
        return path

    # ------------------------------------------------------------------
    # Run / step
    # ------------------------------------------------------------------
    def step(self, round_id: str) -> dict[str, Any]:
        try:
            return self._step_unsafe(round_id)
        except StaleDoctrineError as exc:
            cfg = self._load_config(round_id)
            cfg.status = "stale_doctrine"
            self._save_config(cfg)
            self._notify_officers(cfg, f"stale doctrine: {exc}")
            return {"halted": True, "reason": "stale_doctrine", "error": str(exc)}

    def _step_unsafe(self, round_id: str) -> dict[str, Any]:
        cfg = self._load_config(round_id)
        if cfg.status in ("closed", "awaiting_convener", "stale_doctrine"):
            return {"halted": True, "reason": f"round status is {cfg.status}"}

        self.bus.offline = cfg.offline

        if self._check_end(cfg):
            path = self._run_synthesizer(cfg)
            return {"halted": True, "reason": "end_conditions_met", "resolution_path": str(path) if path else None}

        seat_id = self._next_seat(cfg)
        if seat_id is None:
            # Completed a cycle; advance.
            cfg.current_cycle += 1
            cfg.current_seat_index = 0
            self._save_config(cfg)
            if self._check_end(cfg):
                path = self._run_synthesizer(cfg)
                return {"halted": True, "reason": "end_conditions_met", "resolution_path": str(path) if path else None}
            seat_id = self._next_seat(cfg)

        entry = self._run_single_turn(cfg, seat_id)

        # Advance cursor.
        cfg.current_seat_index = (cfg.current_seat_index + 1) % len(cfg.seats)
        if cfg.current_seat_index == 0:
            cfg.current_cycle += 1
        self._save_config(cfg)

        # Check for blocking flag.
        actions = entry.get("actions", [entry.get("action", {})])
        for action in actions:
            if action.get("type") == "flag" and action.get("flag", {}).get("blocking"):
                cfg.status = "paused"
                self._save_config(cfg)
                self._notify_officers(cfg, f"blocking flag from {seat_id}")
                return {"halted": True, "reason": "blocking_flag", "seat": seat_id, "entry": entry}

        return {"halted": False, "seat": seat_id, "entry": entry}

    def _next_seat(self, cfg: RoundConfig) -> str | None:
        seats = cfg.seats
        idx = cfg.current_seat_index
        for _ in range(len(seats)):
            seat = seats[idx % len(seats)]
            if seat not in (cfg.disabled_seats or []):
                return seat
            idx += 1
        return None

    def _notify_officers(self, cfg: RoundConfig, message: str) -> None:
        for officer in cfg.officers or []:
            self.bus.publish_sync(
                "committee.message.created",
                {"round_id": cfg.round_id, "notice": message, "to_officers": officer},
            )

    def run(self, round_id: str) -> dict[str, Any]:
        while True:
            result = self.step(round_id)
            if result.get("halted"):
                return result


# Convenience factory used by CLI.
def create_table(manifests_dir: str | Path | None = None) -> RoundTable:
    return RoundTable(manifests_dir=manifests_dir)
