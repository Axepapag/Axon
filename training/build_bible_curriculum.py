#!/usr/bin/env python3
"""Build a conversational-style curriculum from the King James Bible.

Splits the cleaned text into overlapping (history, target) slices so the core
can train on plain-text continuation. Targets are kept short enough to fit the
64-position char-slot response decoder.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from substrate import assert_supported_text, default_alphabet


def supported_chars() -> set[str]:
    return set(default_alphabet())


def clean_text(text: str, allowed: set[str]) -> str:
    """Keep only substrate-supported characters and collapse whitespace."""
    cleaned = "".join(ch if ch in allowed else " " for ch in text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n+", "\n", cleaned)
    return cleaned.strip()


def build_examples(
    text: str,
    *,
    history_len: int,
    target_len: int,
    stride: int,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    n = len(text)
    if n < history_len + target_len:
        return examples

    count = 0
    for i in range(history_len, n - target_len + 1, stride):
        history = text[i - history_len : i]
        target = text[i : i + target_len].strip()
        if not target:
            continue
        # Trim target at last space to avoid cutting words.
        if " " in target:
            target = target.rsplit(" ", 1)[0]
        target = target.strip()
        if not target:
            continue
        try:
            assert_supported_text(target)
        except Exception:
            continue

        examples.append(
            {
                "example_id": f"bible_{count:07d}",
                "active_field": {
                    "conversation_history": history.strip(),
                    "user_input": "continue",
                },
                "target_delta": {"text": target},
            }
        )
        count += 1
        if max_examples is not None and count >= max_examples:
            break
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Bible curriculum")
    parser.add_argument("--input", type=Path, default=ROOT / "datasets" / "bible" / "kjv_bible.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "bible" / "bible_curriculum.jsonl")
    parser.add_argument("--history-len", type=int, default=128)
    parser.add_argument("--target-len", type=int, default=48)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    raw = args.input.read_text(encoding="utf-8")
    allowed = supported_chars()
    cleaned = clean_text(raw, allowed)
    print(f"cleaned text length: {len(cleaned)} characters")

    examples = build_examples(
        cleaned,
        history_len=args.history_len,
        target_len=args.target_len,
        stride=args.stride,
        max_examples=args.max_examples,
    )
    print(f"built {len(examples)} examples")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(ex) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
