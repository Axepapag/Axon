#!/usr/bin/env python3
"""Human-readable snapshot of an R0 training run."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def render(state: dict) -> str:
    latest = state.get("latest_evaluation", {})
    lines = [
        f"status={state.get('status')} step={state.get('step')}/{state.get('target_step')}",
        f"progress={100*float(state.get('progress', 0)):.2f}% loss={state.get('mean_recent_loss')}",
        f"rate={float(state.get('steps_per_second', 0)):.3f} steps/s eta={float(state.get('eta_seconds', 0))/3600:.2f}h",
        f"coverage_enforced={state.get('coverage_enforced')} diary_writes={state.get('diary_writes_enabled')}",
        (
            "teacher_accuracy "
            f"scratch={latest.get('scratch_teacher_char_accuracy')} "
            f"response={latest.get('response_teacher_char_accuracy')} "
            f"counterfactual={latest.get('counterfactual_teacher_char_accuracy')}"
        ),
    ]
    for sample in state.get("latest_samples", []):
        lines.extend(
            [
                "",
                f"[{sample.get('family')}] user: {sample.get('user_input', '')}",
                f"gold scratch: {sample.get('gold_scratch', '')}",
                f"axon scratch: {sample.get('predicted_scratch', '')}",
                f"gold response: {sample.get('gold_response', '')}",
                f"axon response: {sample.get('predicted_response', '')}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    live = args.run_dir / "live.json"
    while True:
        if live.exists():
            print("\033[2J\033[H" + render(json.loads(live.read_text(encoding="utf-8"))), flush=True)
        else:
            print(f"waiting for {live}", flush=True)
        if not args.follow:
            return 0
        time.sleep(max(0.5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
