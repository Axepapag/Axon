#!/usr/bin/env python3
"""build_text_corpus.py — convert a raw text corpus (e.g. TinyStories) into a
substrate-safe, one-sentence-per-line file for Phase0 mixing.

Streams line by line (2GB+ inputs fine on small RAM). Splits on sentence
boundaries, drops chars outside the substrate alphabet (v8: code symbols and
common punctuation survive), collapses whitespace, filters by length.

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--min-chars", type=int, default=12)
    ap.add_argument("--max-chars", type=int, default=192)
    ap.add_argument("--report-every", type=int, default=2_000_000)
    args = ap.parse_args(argv)

    alphabet = set(default_alphabet())
    splitter = re.compile(r"(?<=[.!?])\s+")
    ws = re.compile(r"\s+")

    def clean(text: str) -> str:
        kept = [ch if ch in alphabet else (" " if ch.isspace() else "") for ch in text]
        return ws.sub(" ", "".join(kept)).strip()

    n_in = n_out = 0
    buf: list[str] = []
    with open(args.src, "r", encoding="utf-8", errors="replace") as f, \
         open(args.dst, "w", encoding="utf-8") as out:
        def flush():
            nonlocal n_out
            para = " ".join(buf)
            buf.clear()
            for sent in splitter.split(para):
                s = clean(sent)
                if args.min_chars <= len(s) <= args.max_chars:
                    out.write(s + "\n")
                    n_out += 1
        for line in f:
            n_in += 1
            line = line.strip()
            if not line or line == STORY_DELIM:
                flush()
                continue
            buf.append(line.replace(STORY_DELIM, " "))
            if n_in % args.report_every == 0:
                print(f"[corpus] {n_in:,} lines in -> {n_out:,} sentences out", flush=True)
        flush()
    print(f"[corpus] DONE {n_in:,} lines -> {n_out:,} sentences -> {args.dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
