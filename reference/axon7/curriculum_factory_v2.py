#!/usr/bin/env python3
"""Axon v7 staged curriculum factory.

This generator writes real trainer_v2 episodes, not generic
instruction/response rows. Every record is validated with the same
parse_episode(..., repair=False) gate the trainer uses.

Stages are intentionally small and ordered:

1. visible writing
2. input -> response mapping
3. tail/delete discipline
4. ++ growth calls with follow-up state
5. state preservation and eviction
6. runtime boot/schema
7. tool/advisor/knowledge integration
8. mixed capstone
9. focused symbol and repeat-tick stability
10. literal sigil drill
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from field_contract import SHARED_ORDER
from runtime import build_tool_schema
from trainer_v2 import ALLOWED_CHARS, parse_episode

ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = Path("D:/00/corpus")
DEFAULT_OUT = ROOT / "datasets" / "axon7_curriculum_v2_all.jsonl"

DRAFT_BUDGET = 420
SHORT_BUDGET = 120
MEDIUM_BUDGET = 260
MINIMAL_TOOL_SCHEMA = (
    "TOOLS: ++I/R/W/S/A/K/D/V/T/P {text} add to a region. "
    "@@ {request} ask an advisor. "
    "$$ {command} run a shell command."
)

_ASCII_REPAIRS = str.maketrans({
    "\r": "\n",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
})


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).translate(_ASCII_REPAIRS)
    text = "".join(c if c in ALLOWED_CHARS else " " for c in text)
    text = " ".join(text.split())
    return text.strip()


def truncate_words(text: str, limit: int) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rfind(" ")
    if cut < limit // 3:
        cut = limit
    return text[:cut].rstrip(" ,;:-")


def truncate_sentence(text: str, limit: int) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    chunk = text[:limit]
    best = -1
    for sep in (". ", "! ", "? "):
        best = max(best, chunk.rfind(sep))
    if best > limit // 3:
        return chunk[:best + 1].strip()
    return truncate_words(text, limit)


def blank_state() -> dict[str, str]:
    return {r: "" for r in SHARED_ORDER}


def state(**kwargs: str) -> dict[str, str]:
    s = blank_state()
    for key, value in kwargs.items():
        if key not in s:
            raise KeyError(f"unknown region: {key}")
        s[key] = clean_text(value)
    return s


def with_updates(base: dict[str, str], **kwargs: str) -> dict[str, str]:
    out = dict(base)
    for key, value in kwargs.items():
        if key not in out:
            raise KeyError(f"unknown region: {key}")
        out[key] = clean_text(value)
    return out


def make_tick(before: dict[str, str], after: dict[str, str], input_event: str = "") -> dict:
    return {
        "input_event": clean_text(input_event),
        "harness_events": [],
        "state_before": before,
        "state_after": after,
    }


def make_episode(stage: int, idx: int, lesson: str, ticks: list[dict]) -> dict:
    return {
        "episode_id": f"cv2_s{stage}_{idx:04d}",
        "source": "curriculum_factory_v2",
        "stage": stage,
        "lesson": lesson,
        "ticks": ticks,
    }


def load_grammar(corpus: Path, limit: int) -> list[dict[str, str]]:
    path = corpus / "grammar_clean.jsonl"
    rows: list[dict[str, str]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if len(rows) >= limit:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            surface = clean_text(obj.get("surface", ""))
            target = obj.get("target") or {}
            pos = clean_text(target.get("pos", "word"))
            lesson = clean_text((obj.get("notes") or {}).get("lesson", ""))
            projection = clean_text(obj.get("training_projection", ""))
            if surface and pos:
                rows.append({
                    "surface": surface,
                    "pos": pos,
                    "lesson": lesson or f"{surface} is a {pos}.",
                    "projection": projection,
                })
    return rows


def load_layer_examples(corpus: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(corpus.glob("layer1_batch_*.txt")):
        if len(rows) >= limit:
            break
        block: dict[str, str] = {}
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                if block:
                    word = clean_text(block.get("word", ""))
                    definition = clean_text(block.get("definition", ""))
                    example = clean_text(block.get("example", ""))
                    if word and definition and example:
                        rows.append({
                            "word": word,
                            "definition": definition,
                            "example": example,
                        })
                        if len(rows) >= limit:
                            break
                    block = {}
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            block[key.strip().lower()] = value.strip()
    return rows


def stage1_visible(limit: int) -> list[dict]:
    # Stage 1 is rendering practice. Keep the target observable in the prompt so
    # identical empty states never ask for different drafts.
    responses = [
        "Hello Jeff.",
        "I am here.",
        "I am online.",
        "Runtime smoke passed.",
        "I understand.",
        "I will check.",
        "I will verify first.",
        "No response is not acceptable.",
        "Blank output is a failure.",
        "I will keep it simple.",
        "The next step is validation.",
        "I can write clean text.",
        "I will stop here.",
        "Task state is clear.",
        "Working memory is empty.",
        "Summary updated.",
        "Tool result consumed.",
        "I saved that.",
        "I cleared the old task.",
        "I am ready for the next tick.",
    ]
    eps: list[dict] = []
    idx = 1
    while len(eps) < limit:
        response = responses[(idx - 1) % len(responses)]
        prompt = response
        before = state(input_window=prompt)
        after = with_updates(before, response_draft=response)
        eps.append(make_episode(1, idx, "visible_writing", [
            make_tick(before, after, prompt),
        ]))
        idx += 1
    return eps


def mapping_pairs(grammar: list[dict[str, str]], layer: list[dict[str, str]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = [
        ("Jeff: what should you do before deleting files?", "I should verify the exact path first."),
        ("Jeff: what does empty state mean?", "Empty rows mean the region is cleared."),
        ("Jeff: who owns the readable state?", "Axon owns the readable state after the tick."),
        ("Jeff: how do regions grow?", "Regions grow through complete ++ calls."),
        ("Jeff: what is the response draft for?", "The response draft is where Axon writes the reply."),
    ]
    for row in grammar:
        surface = row["surface"]
        pos = row["pos"]
        pairs.append((f"Jeff: what part of speech is {surface}?", f"{surface} is a {pos}."))
    for row in layer:
        word = row["word"]
        definition = truncate_sentence(row["definition"], 90)
        example = truncate_sentence(row["example"], 90)
        pairs.append((f"Jeff: define {word}.", f"{word}: {definition}."))
        pairs.append((f"Jeff: use {word} in a sentence.", example))
    return pairs


def stage2_mapping(grammar: list[dict[str, str]], layer: list[dict[str, str]], limit: int) -> list[dict]:
    pairs = mapping_pairs(grammar, layer)
    eps: list[dict] = []
    for idx, (prompt, response) in enumerate(pairs[:limit], 1):
        response = truncate_sentence(response, SHORT_BUDGET)
        before = state(input_window=prompt)
        after = with_updates(before, response_draft=response)
        eps.append(make_episode(2, idx, "input_response", [
            make_tick(before, after, prompt),
        ]))
    return eps


def stage3_tail(limit: int) -> list[dict]:
    prompts = [
        "Jeff: answer in three words",
        "Jeff: stop cleanly",
        "Jeff: no rambling",
        "Jeff: give the result only",
        "Jeff: clear the bad draft",
    ]
    clean_responses = [
        "Done.",
        "Verified.",
        "I stopped.",
        "Clean response.",
        "The answer is yes.",
        "The answer is no.",
        "I need context.",
        "Runtime is ready.",
        "Checkpoint loaded.",
        "Gate failed.",
    ]
    noisy = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
    eps: list[dict] = []
    for idx in range(1, limit + 1):
        prompt = prompts[(idx - 1) % len(prompts)]
        response = clean_responses[(idx - 1) % len(clean_responses)]
        if idx % 2:
            before = state(input_window=prompt)
        else:
            before = state(input_window=prompt, response_draft=noisy)
        after = with_updates(before, response_draft=response)
        eps.append(make_episode(3, idx, "tail_discipline", [
            make_tick(before, after, prompt),
        ]))
    return eps


def stage4_growth(limit: int) -> list[dict]:
    items = [
        ("W", "working_memory", "Jeff prefers direct answers.", "I saved that in working memory."),
        ("W", "working_memory", "Verify disk state before destructive actions.", "I saved the safety rule."),
        ("S", "rolling_summary", "The trainer now gates visible draft output.", "Summary updated."),
        ("S", "rolling_summary", "The cold full run failed from repeated glyphs.", "I updated the summary."),
        ("P", "task_state", "Current task: build a clean curriculum.", "Task state updated."),
        ("P", "task_state", "Next step: validate the generated dataset.", "I set the next step."),
        ("D", "diary", "Blank output is not a response.", "I wrote that in the diary."),
        ("A", "situation_awareness", "Only response_draft has standing buffer rows.", "I noted the field rule."),
    ]
    eps: list[dict] = []
    for idx in range(1, limit + 1):
        code, region, payload, ack = items[(idx - 1) % len(items)]
        prompt = f"Jeff: remember this: {payload}"
        call = f"++{code} {{{payload}}}"
        before1 = state(input_window=prompt)
        after1 = with_updates(before1, response_draft=call)
        before2 = with_updates(before1, response_draft="", **{region: payload})
        after2 = with_updates(before2, response_draft=ack)
        eps.append(make_episode(4, idx, "plus_growth", [
            make_tick(before1, after1, prompt),
            make_tick(before2, after2, ""),
        ]))
    return eps


def stage5_state(limit: int) -> list[dict]:
    cases = [
        ("working_memory", "Jeff prefers direct answers.", "I remember that Jeff prefers direct answers."),
        ("rolling_summary", "The last run passed the numeric gate but rendered rough text.", "The last run passed numerically but needs cleaner text."),
        ("task_state", "Current task: make the curriculum valid.", "I am working on the valid curriculum."),
        ("diary", "Trust blank output never.", "I still hold that blank output is a failure."),
    ]
    eps: list[dict] = []
    for idx in range(1, limit + 1):
        region, content, response = cases[(idx - 1) % len(cases)]
        prompt = "Jeff: what do you remember?"
        before = state(input_window=prompt, **{region: content})
        after = with_updates(before, response_draft=response)
        if idx % 5 == 0:
            after = with_updates(after, **{region: ""}, response_draft=f"{response} I cleared that region.")
        eps.append(make_episode(5, idx, "state_preservation", [
            make_tick(before, after, prompt),
        ]))
    return eps


def stage6_runtime(grammar: list[dict[str, str]], layer: list[dict[str, str]], limit: int) -> list[dict]:
    schema_variants = [build_tool_schema(), MINIMAL_TOOL_SCHEMA]
    boot_pairs = [
        ("Jeff: boot check", "Runtime boot schema is loaded."),
        ("Jeff: what calls can you use?", "I can use ++, @@, $$, and ## calls."),
        ("Jeff: how do you add memory?", "I add memory with a complete ++ call."),
        ("Jeff: what should happen after a tool succeeds?", "The successful tool call should be removed from the draft."),
    ]
    prompts = boot_pairs + mapping_pairs(grammar, layer)
    eps: list[dict] = []
    for idx in range(1, limit + 1):
        prompt, response = prompts[(idx - 1) % len(prompts)]
        schema = schema_variants[(idx - 1) % len(schema_variants)]
        before = state(input_window=prompt, situation_awareness=schema)
        after = with_updates(before, response_draft=response)
        eps.append(make_episode(6, idx, "runtime_boot", [
            make_tick(before, after, prompt),
        ]))
    return eps


def stage7_integration(limit: int) -> list[dict]:
    cases = [
        (
            state(input_window="Jeff: what did the tool say?", tool_results="file exists: trainer_v2.py"),
            "The tool result says trainer_v2.py exists.",
            {"tool_results": ""},
        ),
        (
            state(input_window="Jeff: what does the advisor recommend?", advisor="Advisor: train the small curriculum before scaling."),
            "The advisor recommends training the small curriculum before scaling.",
            {"advisor": ""},
        ),
        (
            state(input_window="Jeff: answer from knowledge.", structured_knowledge="Axon v7 uses task_state as region P."),
            "task_state is region P.",
            {},
        ),
        (
            state(input_window="Jeff: what is the current task?", task_state="Current task: validate curriculum v2."),
            "The current task is to validate curriculum v2.",
            {},
        ),
    ]
    eps: list[dict] = []
    for idx in range(1, limit + 1):
        before, response, updates = cases[(idx - 1) % len(cases)]
        after = with_updates(before, response_draft=response, **updates)
        eps.append(make_episode(7, idx, "integration", [
            make_tick(before, after, before.get("input_window", "")),
        ]))
    return eps


def stage8_mixed(limit: int) -> list[dict]:
    eps: list[dict] = []
    for idx in range(1, limit + 1):
        if idx % 2:
            task = "Current task: run a gated smoke test."
            summary = "The small curriculum should train before the full dataset."
            prompt = "Jeff: what is the next move?"
            before = state(input_window=prompt, task_state=task, rolling_summary=summary)
            call = "++W {Run the small curriculum before scaling to 768.}"
            draft = f"Next move: train the small curriculum and gate it. {call}"
            after = with_updates(before, response_draft=draft)
            eps.append(make_episode(8, idx, "mixed", [make_tick(before, after, prompt)]))
        else:
            prompt = "Jeff: the tool finished."
            before1 = state(input_window=prompt, tool_results="validation passed: 120 episodes clean")
            after1 = with_updates(before1, response_draft="Validation passed. ++S {Curriculum v2 validation passed with clean episodes.}", tool_results="")
            before2 = state(input_window=prompt, rolling_summary="Curriculum v2 validation passed with clean episodes.")
            after2 = with_updates(before2, response_draft="Summary now includes the validation result.")
            eps.append(make_episode(8, idx, "mixed", [
                make_tick(before1, after1, prompt),
                make_tick(before2, after2, ""),
            ]))
    return eps


def stage9_symbol_idempotence(limit: int) -> list[dict]:
    schema_variants = [build_tool_schema(), MINIMAL_TOOL_SCHEMA]
    symbol_pairs = [
        ("Jeff: what calls can you use?", "I can use ++, @@, $$, and ## calls."),
        ("Jeff: list the call sigils.", "The call sigils are ++, @@, $$, and ##."),
        ("Jeff: how do regions grow?", "Regions grow through complete ++ calls."),
        ("Jeff: how do you add memory?", "I add memory with a complete ++ call."),
        ("Jeff: show a working memory call.", "++W {Jeff prefers direct answers.}"),
        ("Jeff: show a summary call.", "++S {The runtime smoke passed.}"),
        ("Jeff: show a task call.", "++P {Current task: verify runtime output.}"),
        ("Jeff: show a diary call.", "++D {Blank output is not a response.}"),
        ("Jeff: show an advisor call.", "@@advisor {Check the trainer gate.}"),
        ("Jeff: show a tool call.", "$$read_file {path: trainer_v2.py}"),
        ("Jeff: show a note call.", "##note {Hold the clean draft on repeat ticks.}"),
    ]
    clean_pairs = [
        ("Jeff: boot check", "Runtime boot schema is loaded."),
        ("Jeff: what does empty state mean?", "Empty rows mean the region is cleared."),
        ("Jeff: who owns the readable state?", "Axon owns the readable state after the tick."),
        ("Jeff: what is the response draft for?", "The response draft is where Axon writes the reply."),
        ("Jeff: what should happen after a tool succeeds?", "The successful tool call should be removed from the draft."),
        ("Jeff: what part of speech is dog?", "dog is a noun."),
        ("Jeff: what part of speech is run?", "run is a verb."),
    ]
    noisy_tails = [
        "eeeeeeeeeeeeeeeeeeeeeeee",
        "nnnnnnnnnnnnnnnnnnnnnnnn",
        "........................",
        " argh argh argh",
        "     eeeee     ",
    ]
    cases = symbol_pairs + clean_pairs
    eps: list[dict] = []
    idx = 1
    while len(eps) < limit:
        prompt, response = cases[(idx - 1) % len(cases)]
        schema = schema_variants[(idx - 1) % len(schema_variants)]
        variant = (idx - 1) % 4
        if variant == 0:
            before = state(input_window=prompt, situation_awareness=schema)
            after = with_updates(before, response_draft=response)
            ticks = [make_tick(before, after, prompt)]
            lesson = "symbol_exact"
        elif variant == 1:
            before = state(input_window=prompt, situation_awareness=schema, response_draft=response)
            after = with_updates(before, response_draft=response)
            ticks = [make_tick(before, after, prompt)]
            lesson = "repeat_tick_hold"
        elif variant == 2:
            noisy = truncate_sentence(response + noisy_tails[(idx - 1) % len(noisy_tails)], DRAFT_BUDGET)
            before = state(input_window=prompt, situation_awareness=schema, response_draft=noisy)
            after = with_updates(before, response_draft=response)
            ticks = [make_tick(before, after, prompt)]
            lesson = "draft_tail_cleanup"
        else:
            before1 = state(input_window=prompt, situation_awareness=schema)
            after1 = with_updates(before1, response_draft=response)
            before2 = with_updates(before1, response_draft=response)
            after2 = with_updates(before2, response_draft=response)
            ticks = [
                make_tick(before1, after1, prompt),
                make_tick(before2, after2, ""),
            ]
            lesson = "two_tick_stability"
        eps.append(make_episode(9, idx, lesson, ticks))
        idx += 1
    return eps


def stage10_sigil_drill(limit: int) -> list[dict]:
    schema_variants = [build_tool_schema(), MINIMAL_TOOL_SCHEMA]
    cases = [
        ("Jeff: write plus plus.", "++"),
        ("Jeff: write dollar dollar.", "$$"),
        ("Jeff: write hash hash.", "##"),
        ("Jeff: write at at.", "@@"),
        ("Jeff: copy this exactly: ++", "++"),
        ("Jeff: copy this exactly: $$", "$$"),
        ("Jeff: copy this exactly: ##", "##"),
        ("Jeff: copy this exactly: @@", "@@"),
        ("Jeff: list the call sigils.", "The call sigils are ++, @@, $$, and ##."),
        ("Jeff: what calls can you use?", "I can use ++, @@, $$, and ## calls."),
        ("Jeff: how do regions grow?", "Regions grow through complete ++ calls."),
        ("Jeff: how do you add memory?", "I add memory with a complete ++ call."),
        ("Jeff: show a working memory call.", "++W {Jeff prefers direct answers.}"),
        ("Jeff: show a summary call.", "++S {The runtime smoke passed.}"),
        ("Jeff: show a task call.", "++P {Current task: verify runtime output.}"),
        ("Jeff: show a diary call.", "++D {Blank output is not a response.}"),
        ("Jeff: show an advisor call.", "@@advisor {Check the trainer gate.}"),
        ("Jeff: show a tool call.", "$$read_file {path: trainer_v2.py}"),
        ("Jeff: show a note call.", "##note {Hold the clean draft on repeat ticks.}"),
        ("Jeff: which sigil adds memory?", "++ adds memory."),
        ("Jeff: which sigil runs a shell command?", "$$ runs a shell command."),
        ("Jeff: which sigil asks an advisor?", "@@ asks an advisor."),
        ("Jeff: which sigil is a structured runtime note?", "## is a structured runtime note."),
    ]
    wrong_tails = [
        "**",
        "*+",
        "##",
        "$#",
        " arghargh",
        " eeeee",
    ]
    eps: list[dict] = []
    idx = 1
    while len(eps) < limit:
        prompt, response = cases[(idx - 1) % len(cases)]
        schema = schema_variants[(idx - 1) % len(schema_variants)]
        variant = (idx - 1) % 3
        if variant == 0:
            before = state(input_window=prompt, situation_awareness=schema)
            after = with_updates(before, response_draft=response)
            lesson = "sigil_exact"
        elif variant == 1:
            before = state(input_window=prompt, situation_awareness=schema, response_draft=response)
            after = with_updates(before, response_draft=response)
            lesson = "sigil_hold"
        else:
            noisy = truncate_sentence(response + wrong_tails[(idx - 1) % len(wrong_tails)], DRAFT_BUDGET)
            before = state(input_window=prompt, situation_awareness=schema, response_draft=noisy)
            after = with_updates(before, response_draft=response)
            lesson = "sigil_cleanup"
        eps.append(make_episode(10, idx, lesson, [
            make_tick(before, after, prompt),
        ]))
        idx += 1
    return eps


def build(stage: int | str, corpus: Path, limit_per_stage: int, seed: int) -> list[dict]:
    random.seed(seed)
    grammar = load_grammar(corpus, max(40, limit_per_stage))
    layer = load_layer_examples(corpus, max(20, limit_per_stage // 2))
    builders = {
        1: lambda: stage1_visible(limit_per_stage),
        2: lambda: stage2_mapping(grammar, layer, limit_per_stage),
        3: lambda: stage3_tail(limit_per_stage),
        4: lambda: stage4_growth(limit_per_stage),
        5: lambda: stage5_state(limit_per_stage),
        6: lambda: stage6_runtime(grammar, layer, limit_per_stage),
        7: lambda: stage7_integration(limit_per_stage),
        8: lambda: stage8_mixed(limit_per_stage),
        9: lambda: stage9_symbol_idempotence(limit_per_stage),
        10: lambda: stage10_sigil_drill(limit_per_stage),
    }
    if stage == "all":
        episodes: list[dict] = []
        for n in range(1, 9):
            episodes.extend(builders[n]())
        return episodes
    n = int(stage)
    if n not in builders:
        raise ValueError("stage must be 1-10 or all")
    return builders[n]()


def validate_episodes(episodes: Iterable[dict]) -> Counter:
    stats = Counter()
    ids = Counter()
    for ep in episodes:
        parse_episode(ep, repair=False)
        stats[f"stage_{ep.get('stage')}"] += 1
        stats[f"lesson_{ep.get('lesson')}"] += 1
        ids[ep.get("episode_id")] += 1
    dupes = [k for k, v in ids.items() if v > 1]
    if dupes:
        raise ValueError(f"duplicate episode_id(s): {dupes[:5]}")
    return stats


def write_jsonl(path: Path, episodes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Axon v7 staged curriculum")
    ap.add_argument("--stage", default="all", help="1-10 or all")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--limit-per-stage", type=int, default=48)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--split-stages", action="store_true",
                    help="also write one file per stage beside --out")
    args = ap.parse_args()

    episodes = build(args.stage, Path(args.corpus), args.limit_per_stage, args.seed)
    stats = validate_episodes(episodes)
    out = Path(args.out)
    write_jsonl(out, episodes)
    print(f"wrote {len(episodes)} clean episodes to {out}")

    if args.split_stages:
        by_stage: dict[int, list[dict]] = {}
        for ep in episodes:
            by_stage.setdefault(int(ep["stage"]), []).append(ep)
        for stage, eps in sorted(by_stage.items()):
            stage_path = out.with_name(f"{out.stem}_stage{stage}{out.suffix}")
            write_jsonl(stage_path, eps)
            print(f"wrote stage {stage}: {len(eps)} episodes to {stage_path}")

    print("stats:")
    for key, value in sorted(stats.items()):
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
