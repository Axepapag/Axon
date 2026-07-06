#!/usr/bin/env python3
"""build_text_corpus.py — convert a raw text corpus (e.g. TinyStories) into a
substrate-safe, STORY-GROUPED sentence file for Phase0 mixing.

Output format (v2): one sentence per line, ONE BLANK LINE between stories.
Story boundaries are load-bearing: the trainer walks sentences sequentially
within a story so conversation_history carries the real story-so-far.
(v1 shuffled sentences flat, which destroyed causality and made blank-mode
continuation underdetermined — the 2026-07-06 curriculum correction.)

Streams line by line (2GB+ inputs fine on small RAM). Splits on sentence
boundaries (handles closing quotes after terminal punctuation: ?" !" .'),
drops chars outside the substrate alphabet (v8: code symbols and common
punctuation survive), collapses whitespace, filters by length.

Usage:
  python3 training/build_text_corpus.py IN.txt OUT.txt [--min-chars 12] [--max-chars 192]
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from substrate import default_alphabet  # noqa: E402

STORY_DELIM = "<|endoftext|>"

# Sentence boundary: terminal punctuation optionally followed by a closing
# quote, then whitespace. Two alternated fixed-width lookbehinds (regex
# lookbehind must be fixed width). Catches: `friend?" The rabbit` and
# `said.' Then` — the v1 splitter missed these, gluing sentences together.
_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"']))\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SPLIT.split(text) if s]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=192)
    ap.add_argument("--report-every", type=int, default=2_000_000)
    args = ap.parse_args(argv)

    alphabet = set(default_alphabet())
    ws = re.compile(r"\s+")
    # Unicode -> substrate equivalents (TinyStoriesV2 is full of curly
    # quotes; dropping them destroys dialogue structure).
    xlat = str.maketrans({
        "“": '"', "”": '"', "‘": "'", "’": "'",
        "–": "-", "—": "-", "…": ".",
    })

    def clean(text: str) -> str:
        text = text.translate(xlat)
        kept = [ch if ch in alphabet else (" " if ch.isspace() else "") for ch in text]
        return ws.sub(" ", "".join(kept)).strip()

    n_in = n_out = n_stories = 0
    buf: list[str] = []
    with open(args.src, "r", encoding="utf-8", errors="replace") as f, \
         open(args.dst, "w", encoding="utf-8") as out:
        def flush_story():
            nonlocal n_out, n_stories
            para = " ".join(buf).translate(xlat)
            buf.clear()
            wrote = 0
            for sent in split_sentences(para):
                s = clean(sent)
                if args.min_chars <= len(s) <= args.max_chars:
                    out.write(s + "\n")
                    wrote += 1
            if wrote:
                out.write("\n")  # story boundary — load-bearing
                n_out += wrote
                n_stories += 1
        for line in f:
            n_in += 1
            line = line.strip()
            if not line or line == STORY_DELIM:
                flush_story()
                continue
            buf.append(line.replace(STORY_DELIM, " "))
            if n_in % args.report_every == 0:
                print(f"[corpus] {n_in:,} lines in -> {n_out:,} sentences / {n_stories:,} stories", flush=True)
        flush_story()
    print(f"[corpus] DONE {n_in:,} lines -> {n_out:,} sentences in {n_stories:,} stories -> {args.dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
