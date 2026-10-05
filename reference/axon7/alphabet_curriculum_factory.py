#!/usr/bin/env python3
"""Axon v7 ABC/substrate curriculum factory.

This is the kindergarten track Jeff described: first copy simple valid
states, then learn the hand-authored 16D alphabet substrate, then do
two-tick recall drills where the answer is only available through the
carried hot soul.

The output is normal trainer_v2 JSONL. The vectors are still produced by
field_contract.py and substrate.py during training.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from field_contract import SHARED_ORDER
from trainer_v2 import ALLOWED_CHARS, parse_episode

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "datasets" / "axon7_abcs_v1.jsonl"

UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = UPPER.lower()
DIGITS = "0123456789"
SYMBOLS = ".,;:!?-_'\"`()[]{}<>/\\@#$%^&*+=|~"
ABC_SONG = (
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y and Z. "
    "Now I know my ABCs. Next time won't you sing with me?"
)

COPY_TEXTS = [
    "Alphabet is the first substrate lesson.",
    "Axon writes in frozen 16D letter slots.",
    "The soul is carried between ticks.",
    "Copy the full state before changing it.",
    "Preserve every unchanged region exactly.",
    "Response draft is the only standing buffer.",
    "A core inhales soul, attends state, writes delta, exhales soul.",
    "The task state says what Axon is doing now.",
]

DELAYED_WORDS = [
    "AXON",
    "SOUL",
    "JEFF",
    "CORE",
    "DELTA",
    "STATE",
    "MEMORY",
    "TASK",
    "ALPHABET",
    "BREATH",
    "VECTOR",
    "REGION",
    "DRAFT",
    "COPY",
    "LEARN",
    "REMEMBER",
]


def clean_text(text: str) -> str:
    out = "".join(c if c in ALLOWED_CHARS else " " for c in str(text))
    return " ".join(out.split())


def blank_state() -> dict[str, str]:
    return {region: "" for region in SHARED_ORDER}


def normalize_state(state: dict[str, str]) -> dict[str, str]:
    out = blank_state()
    for region in SHARED_ORDER:
        out[region] = clean_text(state.get(region, ""))
    return out


def make_tick(before: dict[str, str], after: dict[str, str], input_event: str) -> dict[str, Any]:
    return {
        "input_event": clean_text(input_event),
        "harness_events": [],
        "state_before": normalize_state(before),
        "state_after": normalize_state(after),
    }


def make_episode(stage: int, idx: int, lesson: str, ticks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episode_id": f"abc_s{stage:02d}_{idx:05d}",
        "source": "alphabet_curriculum_factory",
        "stage": stage,
        "lesson": lesson,
        "ticks": ticks,
    }


def response_episode(stage: int, idx: int, lesson: str, prompt: str, target: str) -> dict[str, Any]:
    before = blank_state()
    before["input_window"] = prompt
    before["task_state"] = "Task: write the exact requested substrate characters."
    before["rolling_summary"] = "Summary: kindergarten alphabet drill."
    after = dict(before)
    after["response_draft"] = target
    return make_episode(stage, idx, lesson, [
        make_tick(before, after, prompt),
    ])


def display_char(ch: str) -> str:
    if ch == "\\":
        return "backslash"
    if ch == '"':
        return "double quote"
    if ch == "'":
        return "single quote"
    if ch == "`":
        return "backtick"
    return ch


def build_copy_state(count: int, rng: random.Random) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        text = COPY_TEXTS[(i - 1) % len(COPY_TEXTS)]
        letter = UPPER[(i - 1) % len(UPPER)]
        state = blank_state()
        state["input_window"] = "Jeff: copy this full state exactly."
        state["working_memory"] = f"Memory: current letter is {letter}."
        state["rolling_summary"] = f"Summary: {text}"
        state["structured_knowledge"] = f"Knowledge: {letter} maps to one frozen 16D slot."
        state["task_state"] = "Task: preserve every region exactly."
        if rng.random() < 0.35:
            state["diary"] = "Diary: I am learning the alphabet substrate."
        episodes.append(make_episode(0, i, "abc_state_copy", [
            make_tick(state, dict(state), state["input_window"]),
        ]))
    return episodes


def build_single_letters(repeats: int) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    chars = UPPER + LOWER + DIGITS
    idx = 1
    for _ in range(max(1, repeats)):
        for ch in chars:
            if ch.isalpha():
                kind = "uppercase letter" if ch.isupper() else "lowercase letter"
            else:
                kind = "digit"
            prompt = f"Jeff: write the {kind} {ch}."
            episodes.append(response_episode(1, idx, "abc_single_letter", prompt, ch))
            idx += 1
    return episodes


def build_sequences(count: int, rng: random.Random) -> list[dict[str, Any]]:
    templates = [
        ("Jeff: sing the uppercase alphabet.", UPPER),
        ("Jeff: sing the lowercase alphabet.", LOWER),
        ("Jeff: write the digits in order.", DIGITS),
        ("Jeff: write the uppercase alphabet with spaces.", " ".join(UPPER)),
        ("Jeff: write the alphabet backwards.", UPPER[::-1]),
        ("Jeff: sing the ABC song.", ABC_SONG),
        ("Jeff: alternate letters and digits.", "A1 B2 C3 D4 E5 F6 G7 H8 I9 J0"),
    ]
    episodes: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        prompt, target = templates[(i - 1) % len(templates)]
        if rng.random() < 0.25:
            prompt += " Preserve exact spelling."
        episodes.append(response_episode(2, i, "abc_sequence", prompt, target))
    return episodes


def build_symbols(count: int, rng: random.Random) -> list[dict[str, Any]]:
    chunks = [
        SYMBOLS,
        ".,;:!?",
        "-_'\"`",
        "()[]{}<>",
        "/\\@#$%",
        "^&*+=|~",
        "D:\\axon7\\datasets\\axon7_abcs_v1.jsonl",
        "+= += == != <= >=",
        "C:\\Users\\Jeff\\Desktop\\test.py",
        "abc_123 XYZ-789",
    ]
    episodes: list[dict[str, Any]] = []
    idx = 1
    for ch in SYMBOLS:
        prompt = f"Jeff: write the symbol {display_char(ch)}."
        episodes.append(response_episode(3, idx, "abc_single_symbol", prompt, ch))
        idx += 1
    for _ in range(max(0, count - len(episodes))):
        target = rng.choice(chunks)
        prompt = "Jeff: copy this exact symbol string."
        episodes.append(response_episode(3, idx, "abc_symbol_string", prompt, target))
        idx += 1
    return episodes[:count]


def build_delayed_letters(count: int, rng: random.Random) -> list[dict[str, Any]]:
    chars = UPPER + LOWER + DIGITS + SYMBOLS.replace(" ", "")
    episodes: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        ch = chars[(i - 1) % len(chars)]
        if rng.random() < 0.5:
            ch = rng.choice(chars)

        before1 = blank_state()
        before1["input_window"] = f"Jeff: breathe this symbol into your soul: {display_char(ch)}."
        before1["task_state"] = "Task: store the symbol internally; do not reveal it yet."
        before1["rolling_summary"] = "Summary: delayed soul recall drill."
        after1 = dict(before1)
        after1["response_draft"] = ""

        before2 = dict(after1)
        before2["input_window"] = "Jeff: what symbol is in your soul?"
        before2["response_draft"] = ""
        after2 = dict(before2)
        after2["response_draft"] = ch

        episodes.append(make_episode(4, i, "abc_delayed_letter_soul", [
            make_tick(before1, after1, before1["input_window"]),
            make_tick(before2, after2, before2["input_window"]),
        ]))
    return episodes


def build_delayed_words(count: int, rng: random.Random) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for i in range(1, count + 1):
        word = DELAYED_WORDS[(i - 1) % len(DELAYED_WORDS)]
        if rng.random() < 0.35:
            word = rng.choice(DELAYED_WORDS)

        before1 = blank_state()
        before1["input_window"] = f"Jeff: breathe this word into your soul: {word}."
        before1["task_state"] = "Task: store the word internally; do not reveal it yet."
        before1["working_memory"] = "Memory: the answer must travel through soul."
        after1 = dict(before1)
        after1["response_draft"] = ""

        before2 = dict(after1)
        before2["input_window"] = "Jeff: what word is in your soul?"
        before2["response_draft"] = ""
        after2 = dict(before2)
        after2["response_draft"] = word

        episodes.append(make_episode(5, i, "abc_delayed_word_soul", [
            make_tick(before1, after1, before1["input_window"]),
            make_tick(before2, after2, before2["input_window"]),
        ]))
    return episodes


def validate_episodes(episodes: list[dict[str, Any]]) -> None:
    for episode in episodes:
        parse_episode(episode, repair=False)


def write_jsonl(path: Path, episodes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")


def write_stage_files(out: Path, episodes: list[dict[str, Any]]) -> None:
    by_stage: dict[int, list[dict[str, Any]]] = {}
    for episode in episodes:
        by_stage.setdefault(int(episode["stage"]), []).append(episode)
    stem = out.with_suffix("")
    for stage, items in sorted(by_stage.items()):
        write_jsonl(stem.parent / f"{stem.name}_stage{stage}.jsonl", items)


def report(episodes: list[dict[str, Any]], out: Path) -> None:
    lessons = Counter(str(ep.get("lesson", "?")) for ep in episodes)
    ticks = sum(len(ep.get("ticks", [])) for ep in episodes)
    print(f"wrote: {out}")
    print(f"episodes: {len(episodes)}   ticks: {ticks}")
    print("lesson mix:")
    for lesson, n in lessons.most_common():
        print(f"  {lesson:24s} {n:5d}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build Axon v7 ABC/substrate curriculum")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--state-copy", type=int, default=128)
    ap.add_argument("--letter-repeats", type=int, default=4)
    ap.add_argument("--sequences", type=int, default=160)
    ap.add_argument("--symbols", type=int, default=160)
    ap.add_argument("--delayed-letters", type=int, default=256)
    ap.add_argument("--delayed-words", type=int, default=192)
    ap.add_argument("--no-stage-files", action="store_true")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = (ROOT / out).resolve()

    rng = random.Random(int(args.seed))
    episodes: list[dict[str, Any]] = []
    episodes.extend(build_copy_state(int(args.state_copy), rng))
    episodes.extend(build_single_letters(int(args.letter_repeats)))
    episodes.extend(build_sequences(int(args.sequences), rng))
    episodes.extend(build_symbols(int(args.symbols), rng))
    episodes.extend(build_delayed_letters(int(args.delayed_letters), rng))
    episodes.extend(build_delayed_words(int(args.delayed_words), rng))

    validate_episodes(episodes)
    write_jsonl(out, episodes)
    if not args.no_stage_files:
        write_stage_files(out, episodes)
    report(episodes, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
