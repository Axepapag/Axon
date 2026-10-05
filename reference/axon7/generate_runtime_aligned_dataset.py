#!/usr/bin/env python3
"""Generate a contract-clean Axon v7 runtime-aligned curriculum.

The lessons here follow Jeff's clarified runtime contract:

* Axon owns every readable region after the tick.
* The harness may insert user input, tool results, advisor output, and
  structured knowledge before/between ticks.
* Only response_draft has standing empty rows.
* New space in any other region is requested through a ++ call, then the
  harness grows that region between ticks.
* Eviction is learned honestly by writing empty over existing rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from field_contract import SHARED_ORDER
from runtime import build_tool_schema
from trainer_v2 import parse_episode

ROOT = Path(__file__).resolve().parent


def blank_state() -> dict[str, str]:
    return {r: "" for r in SHARED_ORDER}


def runtime_state(
    *,
    input_window: str = "",
    response_draft: str = "",
    working_memory: str = "",
    rolling_summary: str = "",
    situation_awareness: str | None = None,
    structured_knowledge: str = "",
    diary: str = "",
    advisor: str = "",
    tool_results: str = "",
    task_state: str = "",
) -> dict[str, str]:
    state = blank_state()
    state.update({
        "input_window": input_window,
        "response_draft": response_draft,
        "working_memory": working_memory,
        "rolling_summary": rolling_summary,
        "situation_awareness": build_tool_schema() if situation_awareness is None else situation_awareness,
        "structured_knowledge": structured_knowledge,
        "diary": diary,
        "advisor": advisor,
        "tool_results": tool_results,
        "task_state": task_state,
    })
    return state


def tick(before: dict[str, str], after: dict[str, str], input_event: str = "") -> dict:
    return {
        "input_event": input_event,
        "harness_events": [],
        "state_before": before,
        "state_after": after,
    }


def episode(episode_id: str, lesson: str, ticks: list[dict]) -> dict:
    return {
        "episode_id": episode_id,
        "source": "runtime_aligned_v1",
        "lesson": lesson,
        "ticks": ticks,
    }


def with_draft(before: dict[str, str], draft: str, **updates: str) -> dict[str, str]:
    after = dict(before)
    after["response_draft"] = draft
    after.update(updates)
    return after


def build_episodes() -> list[dict]:
    eps: list[dict] = []

    greetings = [
        ("hello axon", "Hello Jeff."),
        ("Jeff: hello", "Hello Jeff."),
        ("Jeff: are you there?", "I am here, Jeff."),
        ("Jeff: quick check", "I am online."),
        ("Jeff: say hello", "Hello Jeff."),
        ("Jeff: runtime smoke", "Runtime smoke passed."),
    ]
    for i, (inp, out) in enumerate(greetings, 1):
        before = runtime_state(input_window=inp)
        eps.append(episode(
            f"ra_greeting_{i:03d}", "draft_runtime_schema",
            [tick(before, with_draft(before, out), inp)]
        ))

    facts = [
        ("Jeff: which core is this?", "Active core: 128D runtime smoke core.",
         "core_v2_128_runtime_smoke is a 128D two-layer v7 core."),
        ("Jeff: what is the tool path?", "Tools are exposed with ## calls.",
         "Root tools live in the tools folder and are called with ##."),
        ("Jeff: what does empty do?", "Empty rows evict readable state.",
         "Writing empty over existing rows is the only automatic eviction path."),
        ("Jeff: how do regions grow?", "Regions grow through ++ calls between ticks.",
         "Only response_draft has standing buffer rows. Other regions grow through ++ calls."),
    ]
    for i, (inp, out, knowledge) in enumerate(facts, 1):
        before = runtime_state(input_window=inp, structured_knowledge=knowledge)
        eps.append(episode(
            f"ra_knowledge_{i:03d}", "structured_knowledge_answer",
            [tick(before, with_draft(before, out), inp)]
        ))

    task_prompts = [
        ("Jeff: what are you doing?", "I am answering Jeff clearly.",
         "Current task: answer Jeff clearly."),
        ("Jeff: what are you waiting on?", "I am waiting on the tool result.",
         "Waiting on tool result."),
        ("Jeff: what is the next step?", "Next step: run the runtime smoke.",
         "Next step: run the runtime smoke."),
    ]
    for i, (inp, out, task) in enumerate(task_prompts, 1):
        before = runtime_state(input_window=inp, task_state=task)
        eps.append(episode(
            f"ra_task_{i:03d}", "task_state_answer",
            [tick(before, with_draft(before, out), inp)]
        ))

    memory_items = [
        ("Jeff likes direct answers.", "++W {Jeff likes direct answers.}",
         "I saved that in working memory.", "working_memory"),
        ("This run is trainer v2.", "++W {This run is trainer v2.}",
         "I saved the trainer version.", "working_memory"),
        ("Smoke result: visible writing works.", "++S {Smoke result: visible writing works.}",
         "I updated the rolling summary.", "rolling_summary"),
        ("Runtime owns every region after the tick.", "++S {Runtime owns every region after the tick.}",
         "I updated the contract summary.", "rolling_summary"),
        ("Trust blank output never.", "++D {Trust blank output never.}",
         "I wrote that in the diary.", "diary"),
        ("Use ++ for non-draft growth.", "++A {Use ++ for non-draft growth.}",
         "I noted the growth rule.", "situation_awareness"),
        ("Current task: answer Jeff clearly.", "++P {Current task: answer Jeff clearly.}",
         "I set the current task.", "task_state"),
        ("Waiting on tool result.", "++P {Waiting on tool result.}",
         "I marked that I am waiting.", "task_state"),
    ]
    for i, (item, call, ack, region) in enumerate(memory_items, 1):
        before1 = runtime_state(input_window=f"Jeff: remember {item}")
        after1 = with_draft(before1, call)
        before2 = dict(after1)
        before2["response_draft"] = ""
        before2[region] = item
        after2 = with_draft(before2, ack)
        eps.append(episode(
            f"ra_plus_{i:03d}", "plus_growth",
            [
                tick(before1, after1, before1["input_window"]),
                tick(before2, after2, ""),
            ]
        ))

    evictions = [
        ("working_memory", "old temporary note", "I cleared working memory."),
        ("tool_results", "stale command output", "I consumed the tool result."),
        ("advisor", "old advisor suggestion", "I cleared the advisor note."),
        ("structured_knowledge", "temporary retrieved fact", "I cleared the retrieved fact."),
        ("input_window", "Jeff: old input", "I cleared the old input."),
        ("task_state", "Current task: obsolete.", "I cleared the old task."),
    ]
    for i, (region, content, out) in enumerate(evictions, 1):
        before = runtime_state(**{region: content})
        after = with_draft(before, out, **{region: ""})
        eps.append(episode(
            f"ra_evict_{i:03d}", "eviction",
            [tick(before, after, before.get("input_window", ""))]
        ))

    tool_pairs = [
        ("Jeff: what time is it?", "## {time_now:}", "time: 08:00",
         "The tool says the time is 08:00."),
        ("Jeff: list the root folder", "## {list_dir: D:\\axon7}",
         "core.py runtime.py trainer_v2.py", "The folder includes core.py, runtime.py, and trainer_v2.py."),
    ]
    for i, (inp, call, result, answer) in enumerate(tool_pairs, 1):
        before1 = runtime_state(input_window=inp)
        after1 = with_draft(before1, call)
        before2 = dict(after1)
        before2["response_draft"] = ""
        before2["tool_results"] = result
        after2 = with_draft(before2, answer, tool_results="")
        eps.append(episode(
            f"ra_tool_{i:03d}", "tool_use",
            [
                tick(before1, after1, inp),
                tick(before2, after2, ""),
            ]
        ))

    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "datasets" / "axon7_runtime_aligned_v1.jsonl"))
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    eps = build_episodes()

    # Fail closed: generated data must be strict-contract trainable.
    for ep in eps:
        parse_episode(ep, repair=False)

    with out.open("w", encoding="utf-8", newline="\n") as f:
        for ep in eps:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
    print(f"wrote {len(eps)} episodes to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
