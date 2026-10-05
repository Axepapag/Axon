#!/usr/bin/env python3
"""Dataset validator for Axon v7 curriculum batches.

    python validate_dataset.py datasets\axon7_curriculum_001.jsonl [--samples N]

Runs every record through trainer_v2's parse_episode (the gate
training actually uses), then re-diagnoses failures with reasons, and
reports the quality stats that parse_episode doesn't enforce (harness
continuity, tool follow-through, lesson mix, size variety).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from trainer_v2 import parse_episode, parse_or_repair, ALLOWED_CHARS
from field_contract import SHARED_ORDER, BUFFERED_REGIONS


def diagnose(obj) -> str:
    """Why would parse_episode reject this? Returns '' if it wouldn't."""
    if not isinstance(obj, dict):
        return "record is not a JSON object"
    ticks = obj.get("ticks")
    if not isinstance(ticks, list) or not (1 <= len(ticks) <= 8):
        n = len(ticks) if isinstance(ticks, list) else "?"
        return f"ticks must be a list of 1-8 (got {n})"
    for ti, t in enumerate(ticks):
        if not isinstance(t, dict):
            return f"tick {ti}: not an object"
        for key in ("state_before", "state_after"):
            d = t.get(key) or {}
            if not isinstance(d, dict):
                return f"tick {ti}: {key} is not an object"
            for r in SHARED_ORDER:
                v = d.get(r, "")
                if v is None:
                    continue
                if not isinstance(v, str):
                    return f"tick {ti}: {key}.{r} is not a string"
                bad = sorted({c for c in v if c not in ALLOWED_CHARS})
                if bad:
                    return (f"tick {ti}: {key}.{r} has banned chars: "
                            f"{''.join(bad)!r}")
        sb = {r: (t.get("state_before") or {}).get(r) or "" for r in SHARED_ORDER}
        sa = {r: (t.get("state_after") or {}).get(r) or "" for r in SHARED_ORDER}
        for r in SHARED_ORDER:
            room = len(sb[r]) + BUFFERED_REGIONS.get(r, 0)
            if len(sa[r]) > room:
                return (f"tick {ti}: region '{r}' grows {len(sb[r])} -> "
                        f"{len(sa[r])} (room {room}); growth needs a ++ "
                        f"call between ticks")
        ie = t.get("input_event") or ""
        if not isinstance(ie, str):
            return f"tick {ti}: input_event is not a string"
        bad = sorted({c for c in ie if c not in ALLOWED_CHARS})
        if bad:
            return f"tick {ti}: input_event has banned chars: {''.join(bad)!r}"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()
    path = Path(args.path)

    n_total = n_json_bad = n_valid = n_repaired = 0
    reasons = Counter()
    lessons = Counter()
    tick_counts = Counter()
    sigils = Counter()
    region_chars = Counter()
    region_used = Counter()
    continuity_breaks = 0
    continuity_checked = 0
    ids = Counter()
    samples: list[dict] = []
    draft_growth_max = 0

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_json_bad += 1
                reasons["invalid JSON"] += 1
                continue
            ids[str(obj.get("episode_id", "?"))] += 1
            ep, was_repaired, _ = parse_or_repair(obj)
            if ep is None:
                reasons[diagnose(obj) or "rejected (unknown reason)"] += 1
                continue
            n_valid += 1
            if was_repaired:
                n_repaired += 1
            lessons[ep.lesson] += 1
            tick_counts[len(ep.ticks)] += 1
            if len(samples) < args.samples:
                samples.append(obj)
            prev_after = None
            for t in ep.ticks:
                draft_growth = (len(t.state_after.get("response_draft", ""))
                                - len(t.state_before.get("response_draft", "")))
                draft_growth_max = max(draft_growth_max, draft_growth)
                for r in SHARED_ORDER:
                    n = len(t.state_after.get(r, ""))
                    region_chars[r] += n
                    if n:
                        region_used[r] += 1
                d = t.state_after.get("response_draft", "")
                for sig in ("++", "@@", "$$", "##"):
                    sigils[sig] += d.count(sig + " {") + d.count(sig + "{")
                # ++ with region code, e.g. "++S {"
                for code in "IRWSAKDVTP":
                    sigils["++" + code] += d.count(f"++{code} {{") + d.count(f"++{code}{{")
                if prev_after is not None:
                    continuity_checked += 1
                    # outside the draft (tool consumption) and harness-fed
                    # insertions, before(t+1) should match after(t)
                    for r in SHARED_ORDER:
                        if r in ("response_draft", "input_window",
                                 "tool_results", "advisor",
                                 "structured_knowledge"):
                            continue
                        if t.state_before.get(r, "") != prev_after.get(r, ""):
                            # allow ++ growth: before(t+1) startswith after(t)
                            if not t.state_before.get(r, "").startswith(
                                    prev_after.get(r, "")):
                                continuity_breaks += 1
                                break
                prev_after = t.state_after

    print(f"file: {path}")
    print(f"records: {n_total}   trainable: {n_valid} "
          f"(clean: {n_valid - n_repaired}, auto-repaired: {n_repaired})   "
          f"unrepairable: {n_total - n_valid}   "
          f"({n_valid / max(1, n_total):.1%} trainable)")
    if reasons:
        print("\nunrepairable — reasons:")
        for reason, n in reasons.most_common(15):
            print(f"  {n:5d}  {reason}")
    print("\nlesson mix (valid episodes):")
    for lesson, n in lessons.most_common():
        print(f"  {lesson:14s} {n:5d}  ({n / max(1, n_valid):.0%})")
    print("\nticks per episode:", dict(sorted(tick_counts.items())))
    print(f"max in-tick draft growth: {draft_growth_max} chars (limit 500)")
    print("\ntool sigils in drafts:",
          {k: v for k, v in sorted(sigils.items()) if v})
    print("\nregion usage (ticks with content / avg chars when used):")
    total_ticks = sum(k * v for k, v in tick_counts.items())
    for r in SHARED_ORDER:
        used = region_used[r]
        avg = region_chars[r] / max(1, used)
        print(f"  {r:22s} {used:5d}/{total_ticks}   avg {avg:6.0f} chars")
    dupes = {k: v for k, v in ids.items() if v > 1}
    print(f"\nduplicate episode_ids: {len(dupes)}")
    print(f"harness continuity breaks (informational): "
          f"{continuity_breaks}/{continuity_checked} multi-tick transitions")
    for i, s in enumerate(samples):
        print("\n" + "=" * 70)
        print(f"SAMPLE {i + 1}: {s.get('episode_id')} lesson={s.get('lesson')}")
        for ti, t in enumerate(s.get("ticks", [])):
            print(f"  tick {ti}: input={t.get('input_event', '')!r}")
            sa = t.get("state_after", {})
            for r in ("response_draft", "rolling_summary", "diary"):
                v = sa.get(r, "")
                if v:
                    print(f"    after.{r}: {v[:120]!r}")
    return 0 if n_valid == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
