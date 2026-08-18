#!/usr/bin/env python3
"""Train-as-you-live engine for the exact 16D character field.

Every training step:
  1. Inhale the private soul.
  2. Attend the masked shared field as exact 16D character slots.
  3. Emit response-draft deltas as per-position 16D vectors.
  4. Decode through the frozen LetterBank.
  5. Exhale the produced field back into the soul.

The deleted wide-slot lineage is not part of this trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cores.core import AxonCore, CoreConfig
from cores.soul_v2 import SoulManagerV2, SoulV2Config, TierSpec
from substrate import SLOT_DIM as SUBSTRATE_SLOT_DIM
from substrate import assert_supported_text, char_to_slot, default_alphabet, get_letter_bank, roundtrip_check


LOCKED_CORE_PRESETS = {
    "A": "64,2,1,131072",
    "B": "128,2,1,262144",
    "C": "256,2,2,524288",
}

DRAFT_MODES = ("copy", "partial", "blank")


def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def draft_seed_for_mode(answer: str, mode: str, partial_frac: float = 0.5) -> str:
    if mode == "copy":
        return answer
    if mode == "partial":
        keep = max(0, min(len(answer), int(round(len(answer) * partial_frac))))
        return answer[:keep].rstrip()
    if mode == "blank":
        return ""
    raise ValueError(f"unknown draft seed mode: {mode}")


def normalize_allowed_draft_modes(value: Any) -> tuple[str, ...]:
    """Validate a row-level draft policy without silently widening it."""
    if value is None:
        return DRAFT_MODES
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("allowed_draft_modes must be a non-empty list")
    normalized: list[str] = []
    for raw_mode in value:
        mode = str(raw_mode)
        if mode not in DRAFT_MODES:
            raise ValueError(f"unsupported allowed_draft_modes entry: {mode!r}")
        if mode not in normalized:
            normalized.append(mode)
    return tuple(normalized)


def choose_draft_mode(
    example: dict[str, Any],
    mode_weights: list[float] | tuple[float, ...],
    *,
    rng: Any = random,
) -> str:
    """Sample only from modes explicitly allowed by this example."""
    if len(mode_weights) != len(DRAFT_MODES):
        raise ValueError(f"mode_weights must have {len(DRAFT_MODES)} entries")
    allowed = normalize_allowed_draft_modes(example.get("allowed_draft_modes"))
    weights_by_mode = dict(zip(DRAFT_MODES, mode_weights))
    allowed_weights = [float(weights_by_mode[mode]) for mode in allowed]
    if sum(allowed_weights) <= 0:
        raise ValueError(
            "allowed_draft_modes have zero probability under --charslot-mode-weights: "
            f"allowed={list(allowed)!r} weights={list(mode_weights)!r}"
        )
    return str(rng.choices(allowed, weights=allowed_weights)[0])


class Phase0Curriculum:
    """Stream substrate-safe sentences from recovered corpus and story corpus."""

    def __init__(
        self,
        curriculum_dir: str | None = None,
        containers_path: str | None = None,
        max_chars: int = 256,
        history_turns: int = 10,
        rng: random.Random | None = None,
        text_corpus: str | None = None,
        text_corpus_weight: float = 0.7,
        text_corpus_max: int = 1_000_000,
        target_words: int = 0,
    ):
        self.max_chars = max_chars
        self.history_turns = max(0, history_turns)
        self.target_words = max(0, int(target_words))
        self.rng = rng or random.Random(42)
        self.sentences: list[str] = []
        self.eval_sentences: list[str] = []
        self.history: list[str] = []
        self.stories: list[list[str]] = []
        self.story_weight = float(text_corpus_weight) if text_corpus else 0.0
        self.story_eval_examples: list[dict[str, Any]] = []

        self._load(curriculum_dir, containers_path)
        if text_corpus and os.path.exists(text_corpus):
            self._load_story_corpus(text_corpus, text_corpus_max)

        self.rng.shuffle(self.sentences)
        eval_n = min(512, max(32, len(self.sentences) // 100)) if len(self.sentences) >= 64 else len(self.sentences)
        self.eval_sentences = self.sentences[:eval_n]
        self.sentences = self.sentences[eval_n:] or self.eval_sentences[:]

        if self.stories:
            n_eval_stories = min(48, max(8, len(self.stories) // 100))
            held = self.stories[:n_eval_stories]
            self.stories = self.stories[n_eval_stories:] or held[:]
            for sid, story in enumerate(held):
                k = self.rng.randrange(1, len(story))
                ex = self._example_from_sentence(story[k], history_text=" ".join(story[:k]))
                ex["story_id"] = f"eval_s{sid}"
                self.story_eval_examples.append(ex)

        self._pos = 0
        self._story_idx = 0
        self._sent_idx = 0
        log(
            "DATA",
            f"Phase0 sentences loaded: train={len(self.sentences)} "
            f"stories={len(self.stories)} (weight={self.story_weight}) "
            f"eval={len(self.eval_sentences)} story_eval={len(self.story_eval_examples)} "
            f"target_words={self.target_words or 'half'}",
        )

    def _load_story_corpus(self, text_corpus: str, text_corpus_max: int) -> None:
        block: list[str] = []
        n_sent = 0
        with open(text_corpus, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    if len(block) >= 2:
                        self.stories.append(block)
                        n_sent += len(block)
                    block = []
                    if n_sent >= text_corpus_max:
                        break
                    continue
                block.append(line)
        if len(block) >= 2 and n_sent < text_corpus_max:
            self.stories.append(block)
        self.rng.shuffle(self.stories)

    def _substrate_safe(self, text: str) -> str:
        alphabet = set(default_alphabet())
        out = []
        for ch in text:
            if ch in alphabet:
                out.append(ch)
            elif ch.isspace():
                out.append(" ")
        return re.sub(r"\s+", " ", "".join(out)).strip()

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [safe for p in parts if len(safe := self._substrate_safe(p)) >= 8]

    def _load(self, curriculum_dir: str | None, containers_path: str | None) -> None:
        if curriculum_dir and os.path.isdir(curriculum_dir):
            for family in ("field_surfacing_v1", "retrieval_qa_v1", "procedure_next_step_v1"):
                path = pathlib.Path(curriculum_dir) / f"{family}.jsonl"
                if path.exists():
                    self._load_jsonl(path, text_keys=["structured_knowledge", "context_slots_text", "answer"])

        if containers_path and os.path.exists(containers_path):
            self._load_containers(containers_path)

        if not self.sentences:
            self.sentences = [
                "the quick brown fox jumps over the lazy dog",
                "axon is a token free character field agent",
                "the shared field is made of exact frozen character slots",
                "every character maps to a frozen sixteen dimensional vector",
                "souls breathe every step from day one",
            ] * 200

    def _load_jsonl(self, path: pathlib.Path, text_keys: list[str]) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in text_keys:
                    val = obj.get(key)
                    if isinstance(val, str):
                        self.sentences.extend(self._split_sentences(val))
                    elif isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                self.sentences.extend(self._split_sentences(item))

    def _load_containers(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50_000:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") or obj.get("letters") or ""
                if isinstance(text, str):
                    self.sentences.extend(self._split_sentences(text))

    def _example_from_sentence(self, sentence: str, history_text: str = "") -> dict[str, Any]:
        words = sentence.split()
        if self.target_words > 0 and len(words) > self.target_words:
            split = len(words) - self.target_words
        else:
            split = max(1, len(words) // 2)
        user_input = " ".join(words[:split])
        answer = " ".join(words[split:])
        if len(answer) > self.max_chars:
            answer = answer[: self.max_chars].rsplit(" ", 1)[0]
        return {
            "mode": "phase0",
            "context": user_input,
            "conversation_history": history_text,
            "user_input": user_input,
            "answer": answer,
            "active_regions": {"conversation_history", "user_input", "response_draft"},
        }

    def next(self) -> dict[str, Any]:
        if self.stories and self.rng.random() < self.story_weight:
            story = self.stories[self._story_idx]
            if self._sent_idx >= len(story):
                self._story_idx = self.rng.randrange(len(self.stories))
                self._sent_idx = 0
                story = self.stories[self._story_idx]
            k = self._sent_idx
            self._sent_idx += 1
            ex = self._example_from_sentence(story[k], history_text=" ".join(story[:k]))
            ex["story_id"] = f"s{self._story_idx}"
            return ex

        if self._pos >= len(self.sentences):
            self.rng.shuffle(self.sentences)
            self._pos = 0
        sentence = self.sentences[self._pos]
        self._pos += 1
        history_text = "\n".join(self.history[-self.history_turns:])
        ex = self._example_from_sentence(sentence, history_text=history_text)
        self.history.append(f"user {ex['user_input']}\naxon {ex['answer']}")
        return ex

    def eval_examples(self, n: int) -> list[dict[str, Any]]:
        examples: list[dict[str, Any]] = []
        if self.story_eval_examples:
            n_story = min(len(self.story_eval_examples), max(1, int(round(n * self.story_weight))))
            examples.extend(self.story_eval_examples[:n_story])
        source = self.eval_sentences or self.sentences
        history: list[str] = []
        for i in range(n - len(examples)):
            history_text = "\n".join(history[-self.history_turns:])
            ex = self._example_from_sentence(source[i % len(source)], history_text=history_text)
            examples.append(ex)
            history.append(f"user {ex['user_input']}\naxon {ex['answer']}")
        return examples


PHASE0B_DEFAULT_FAMILIES = (
    "runtime_response_delta_v1",
    "structured_knowledge_delta_v1",
    "scratchpad_delta_v1",
)

PHASE1A_DEFAULT_FAMILIES = (
    "runtime_response_chain_v2",
    "structured_knowledge_response_v2",
    "procedure_next_step_response_v2",
)


class Phase0bCurriculum:
    """Consume phase0b JSONL specs through the checkpoint-compatible charslot view.

    This first phase0b bridge intentionally keeps the existing 384-slot
    checkpoint layout: history, user_input, response_draft. Non-response target
    regions such as scratch are exposed as readable target labels and context,
    but the supervised characters are still emitted through the response slice.
    That lets us resume the current 64D core without changing region embeddings
    or slot count.
    """

    def __init__(
        self,
        phase0b_dir: str | None,
        families: str | list[str] | tuple[str, ...],
        max_chars: int = 256,
        history_turns: int = 10,
        rng: random.Random | None = None,
        max_examples: int = 100_000,
        history_chars: int = 256,
        user_chars: int = 64,
        family_sampling: str = "global",
        family_weights: str | list[float] | tuple[float, ...] | None = None,
    ):
        self.phase0b_dir = pathlib.Path(phase0b_dir or "datasets/recovered/phase0b_curriculum_v1")
        self.families = self._parse_families(families)
        self.max_chars = max_chars
        self.history_turns = max(0, history_turns)
        self.rng = rng or random.Random(42)
        self.max_examples = max(1, int(max_examples))
        self.history_chars = int(history_chars)
        self.user_chars = int(user_chars)
        self.family_sampling = str(family_sampling)
        if self.family_sampling not in {"global", "weighted"}:
            raise ValueError(f"unsupported phase0b family sampling policy: {self.family_sampling!r}")
        self.family_weights = self._parse_family_weights(family_weights)
        self.examples: list[dict[str, Any]] = []
        self.eval_cache: list[dict[str, Any]] = []
        self.test_cache: list[dict[str, Any]] = []
        self._pos = 0
        self._load()
        if not self.examples:
            raise ValueError(f"phase0b curriculum produced no examples from {self.phase0b_dir}")
        self.rng.shuffle(self.examples)
        if not self.eval_cache:
            eval_n = min(512, max(32, len(self.examples) // 100)) if len(self.examples) >= 64 else len(self.examples)
            self.eval_cache = self.examples[:eval_n]
            self.examples = self.examples[eval_n:] or self.eval_cache[:]
        for source in (self.examples, self.eval_cache, self.test_cache):
            for example in source:
                example.setdefault("eval_case_id", self._stable_case_id(example))
        self.eval_cache.sort(key=lambda example: str(example["eval_case_id"]))
        self.test_cache.sort(key=lambda example: str(example["eval_case_id"]))
        self._order = list(range(len(self.examples)))
        self.rng.shuffle(self._order)
        self._family_orders: dict[str, list[int]] = {
            family: [index for index, example in enumerate(self.examples) if example.get("family") == family]
            for family in self.families
        }
        if self.family_sampling == "weighted":
            missing = [family for family, indices in self._family_orders.items() if not indices]
            if missing:
                raise ValueError(f"weighted family sampling has no train examples for: {missing!r}")
        for indices in self._family_orders.values():
            self.rng.shuffle(indices)
        self._family_positions = {family: 0 for family in self.families}
        self._family_credits = {family: 0.0 for family in self.families}
        self._family_steps = 0
        log(
            "DATA",
            f"Delta examples loaded: train={len(self.examples)} dev={len(self.eval_cache)} "
            f"test={len(self.test_cache)} "
            f"families={','.join(self.families)} target_chars={self.max_chars}",
        )

    def _parse_families(self, families: str | list[str] | tuple[str, ...]) -> list[str]:
        if isinstance(families, str):
            raw = [x.strip() for x in families.split(",") if x.strip()]
        else:
            raw = [str(x).strip() for x in families if str(x).strip()]
        return raw or list(PHASE0B_DEFAULT_FAMILIES)

    def _parse_family_weights(
        self,
        value: str | list[float] | tuple[float, ...] | None,
    ) -> dict[str, float]:
        if self.family_sampling == "global":
            return {}
        if isinstance(value, str):
            parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
        elif value is None:
            parsed = []
        else:
            parsed = [float(item) for item in value]
        if len(parsed) != len(self.families):
            raise ValueError(
                "weighted phase0b family sampling requires one --phase0b-family-weights value "
                f"per configured family ({len(parsed)} != {len(self.families)})"
            )
        if any(weight < 0 for weight in parsed) or sum(parsed) <= 0:
            raise ValueError("phase0b family weights must be non-negative and not all zero")
        total = sum(parsed)
        return {family: weight / total for family, weight in zip(self.families, parsed)}

    @staticmethod
    def _stable_case_id(example: dict[str, Any]) -> str:
        identity = {
            "family": example.get("family", ""),
            "source": example.get("source", ""),
            "conversation_history": example.get("conversation_history", ""),
            "user_input": example.get("user_input", ""),
            "answer": example.get("answer", ""),
            "allowed_draft_modes": example.get("allowed_draft_modes", list(DRAFT_MODES)),
        }
        encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _substrate_safe(self, text: Any, *, keep_newlines: bool = False) -> str:
        alphabet = set(default_alphabet())
        raw = "" if text is None else str(text)
        out: list[str] = []
        for ch in raw:
            if ch in alphabet:
                out.append(ch if keep_newlines or ch != "\n" else " ")
            elif ch.isspace():
                out.append("\n" if keep_newlines else " ")
            else:
                out.append(" ")
        if keep_newlines:
            lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in "".join(out).splitlines()]
            return "\n".join(line for line in lines if line)
        return re.sub(r"\s+", " ", "".join(out)).strip()

    def _truncate_answer(self, text: str) -> str:
        text = self._substrate_safe(text, keep_newlines=False)
        if len(text) <= self.max_chars:
            return text
        clipped = text[: self.max_chars].rstrip()
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0].rstrip()
        return clipped or text[: self.max_chars].rstrip()

    def _history_from_active_field(self, active: dict[str, Any], target_region: str, family: str) -> str:
        parts: list[str] = [f"family: {family}", f"target region: {target_region}"]
        for key in ("conversation_history", "structured_knowledge", "scratch", "diary"):
            value = self._substrate_safe(active.get(key, ""), keep_newlines=True)
            if value:
                parts.append(f"{key}:\n{value}")
        return "\n".join(parts)

    def _example_from_delta_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        active = row.get("active_field") if isinstance(row.get("active_field"), dict) else {}
        target = row.get("target_delta") if isinstance(row.get("target_delta"), dict) else {}
        target_region = self._substrate_safe(target.get("region", "response_draft")) or "response_draft"
        answer = self._truncate_answer(target.get("text", ""))
        if not answer:
            return None
        user_input = self._substrate_safe(active.get("user_input", ""), keep_newlines=False)
        if target_region != "response_draft":
            user_input = self._substrate_safe(f"write {target_region}: {user_input}", keep_newlines=False)
        return {
            "mode": "phase0b",
            "family": self._substrate_safe(row.get("family", "")) or "phase0b",
            "target_region": target_region,
            "context": user_input,
            "conversation_history": self._history_from_active_field(active, target_region, str(row.get("family", ""))),
            "user_input": user_input,
            "answer": answer,
            "source": str(row.get("source", "")),
            "active_regions": {"conversation_history", "user_input", "response_draft"},
        }

    def _example_from_exact_delta_row(self, row: dict[str, Any]) -> dict[str, Any]:
        schema = str(row.get("schema", ""))
        if schema not in {"axon_phase0b_delta_v2", "axon_phase0b_delta_v3", "axon_phase1_delta_v2"}:
            raise ValueError(f"unsupported exact delta schema {schema!r}")
        mode = "phase0b_exact" if schema.startswith("axon_phase0b_delta_v") else "phase1a"
        split = str(row.get("split", ""))
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"{mode} row has invalid split {split!r}")
        active = row.get("active_field") if isinstance(row.get("active_field"), dict) else {}
        target = row.get("target_delta") if isinstance(row.get("target_delta"), dict) else {}
        if target.get("region") != "response_draft" or target.get("op") != "replace":
            raise ValueError(f"{mode} rows must be response_draft replace deltas")
        history = str(active.get("conversation_history", ""))
        user_input = str(active.get("user_input", ""))
        answer = str(target.get("text", ""))
        assert_supported_text(history)
        assert_supported_text(user_input)
        assert_supported_text(answer)
        if len(history) > self.history_chars:
            raise ValueError(f"{mode} history exceeds explicit budget: {len(history)} > {self.history_chars}")
        if len(user_input) > self.user_chars:
            raise ValueError(f"{mode} user input exceeds explicit budget: {len(user_input)} > {self.user_chars}")
        if not answer or len(answer) > self.max_chars:
            raise ValueError(f"{mode} answer length must be 1..{self.max_chars}; got {len(answer)}")
        audit = row.get("budget_audit") if isinstance(row.get("budget_audit"), dict) else {}
        if audit.get("silent_clips") != 0 or audit.get("unsupported_substitutions") != 0:
            raise ValueError(f"{mode} budget audit is not lossless")
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        allowed_draft_modes = normalize_allowed_draft_modes(row.get("allowed_draft_modes"))
        return {
            "mode": mode,
            "family": str(row.get("family", mode)),
            "target_region": "response_draft",
            "context": user_input,
            "conversation_history": history,
            "user_input": user_input,
            "answer": answer,
            "source": str(source.get("record_id", "")),
            "split": split,
            "chain": row.get("chain", {}),
            "allowed_draft_modes": list(allowed_draft_modes),
            "strict_budget": True,
            "active_regions": {"conversation_history", "user_input", "response_draft"},
        }

    def _example_from_soul_pair(self, row: dict[str, Any]) -> dict[str, Any] | None:
        tick_a = row.get("tick_a") if isinstance(row.get("tick_a"), dict) else {}
        active = tick_a.get("active_field") if isinstance(tick_a.get("active_field"), dict) else {}
        answer = self._truncate_answer(tick_a.get("target_soul_trace", ""))
        if not answer:
            return None
        user_input = self._substrate_safe(active.get("user_input", "retain useful experience"), keep_newlines=False)
        history = self._history_from_active_field(active, "soul_trace", str(row.get("family", "soul_exhale_pair_v1")))
        return {
            "mode": "phase0b",
            "family": self._substrate_safe(row.get("family", "")) or "soul_exhale_pair_v1",
            "target_region": "soul_trace",
            "context": user_input,
            "conversation_history": history,
            "user_input": user_input,
            "answer": answer,
            "source": str(row.get("source", "")),
            "active_regions": {"conversation_history", "user_input", "response_draft"},
        }

    def _example_from_soul_probe(self, row: dict[str, Any]) -> dict[str, Any] | None:
        probe = row.get("probe_tick") if isinstance(row.get("probe_tick"), dict) else {}
        active = probe.get("active_field") if isinstance(probe.get("active_field"), dict) else {}
        answer = self._truncate_answer(probe.get("expected_answer", ""))
        if not answer:
            return None
        user_input = self._substrate_safe(active.get("user_input", "recall retained episode detail"), keep_newlines=False)
        history = self._history_from_active_field(active, "soul_probe", str(row.get("family", "soul_ablation_probe_v1")))
        return {
            "mode": "phase0b",
            "family": self._substrate_safe(row.get("family", "")) or "soul_ablation_probe_v1",
            "target_region": "soul_probe",
            "context": user_input,
            "conversation_history": history,
            "user_input": user_input,
            "answer": answer,
            "source": str(row.get("source", "")),
            "active_regions": {"conversation_history", "user_input", "response_draft"},
        }

    def _row_to_example(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if row.get("schema") in {"axon_phase0b_delta_v2", "axon_phase0b_delta_v3", "axon_phase1_delta_v2"}:
            return self._example_from_exact_delta_row(row)
        family = str(row.get("family", ""))
        if family in {"soul_exhale_pair_v1"}:
            return self._example_from_soul_pair(row)
        if family in {"soul_ablation_probe_v1"}:
            return self._example_from_soul_probe(row)
        return self._example_from_delta_row(row)

    def _load_jsonl(self, path: pathlib.Path) -> int:
        count = 0
        train_count = 0
        train_limit = max(1, self.max_examples // max(1, len(self.families)))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ex = self._row_to_example(row)
                if ex:
                    split = ex.get("split", "train")
                    if split == "train":
                        if train_count >= train_limit:
                            continue
                        self.examples.append(ex)
                        train_count += 1
                    elif split == "dev":
                        self.eval_cache.append(ex)
                    else:
                        self.test_cache.append(ex)
                    count += 1
        return count

    def _load(self) -> None:
        if not self.phase0b_dir.exists():
            raise FileNotFoundError(f"phase0b curriculum dir not found: {self.phase0b_dir}")
        loaded: dict[str, int] = {}
        for family in self.families:
            path = self.phase0b_dir / f"{family}.jsonl"
            if not path.exists():
                log("DATA", f"phase0b family missing: {path}")
                continue
            loaded[family] = self._load_jsonl(path)
        log("DATA", "Phase0b family counts: " + ", ".join(f"{k}={v}" for k, v in loaded.items()))

    def next(self) -> dict[str, Any]:
        if self.family_sampling == "weighted":
            return self._next_weighted()
        if self._pos >= len(self._order):
            self.rng.shuffle(self._order)
            self._pos = 0
        ex = self.examples[self._order[self._pos]]
        self._pos += 1
        return ex

    def _next_weighted(self) -> dict[str, Any]:
        positive_families = [family for family in self.families if self.family_weights[family] > 0]
        for family in positive_families:
            self._family_credits[family] += self.family_weights[family]
        chosen = max(positive_families, key=lambda family: (self._family_credits[family], -self.families.index(family)))
        self._family_credits[chosen] -= sum(self.family_weights[family] for family in positive_families)
        position = self._family_positions[chosen]
        order = self._family_orders[chosen]
        if position >= len(order):
            raise RuntimeError(
                "weighted phase0b family exhausted without replacement: "
                f"family={chosen!r} consumed={position} available={len(order)} "
                f"steps={self._family_steps}; rebuild or explicitly reset with a revised curriculum"
            )
        example = self.examples[order[position]]
        self._family_positions[chosen] = position + 1
        self._family_steps += 1
        self._pos = self._family_steps
        return example

    def eval_examples(self, n: int) -> list[dict[str, Any]]:
        source = self.eval_cache or self.examples
        return self._stratified_examples(source, n)

    def test_examples(self, n: int) -> list[dict[str, Any]]:
        source = self.test_cache or self.eval_cache or self.examples
        return self._stratified_examples(source, n)

    def _stratified_examples(self, source: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        """Return a stable, near-equal family mix without consuming train RNG."""
        if n <= 0:
            return []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for example in source:
            family = str(example.get("family", ""))
            grouped.setdefault(family, []).append(example)
        for examples in grouped.values():
            examples.sort(key=lambda example: str(example["eval_case_id"]))
        family_order = [family for family in self.families if family in grouped]
        family_order.extend(sorted(family for family in grouped if family not in family_order))
        if not family_order:
            return []
        base, remainder = divmod(n, len(family_order))
        allocations = {
            family: base + int(index < remainder)
            for index, family in enumerate(family_order)
        }
        selected: list[dict[str, Any]] = []
        for offset in range(max(allocations.values(), default=0)):
            for family in family_order:
                if offset >= allocations[family]:
                    continue
                bucket = grouped[family]
                selected.append(bucket[offset % len(bucket)])
        return selected

    def state_dict(self) -> dict[str, Any]:
        """Small exact sampler state for checkpoint-to-checkpoint continuity."""
        return {
            "schema": "axon_phase0b_sampler_v1",
            "position": self._pos,
            "order": self._order.copy(),
            "rng_state": self.rng.getstate(),
            "example_count": len(self.examples),
            "families": self.families,
            "family_sampling_policy": self.family_sampling,
            "family_weights": self.family_weights.copy(),
            "family_orders": {family: order.copy() for family, order in self._family_orders.items()},
            "family_positions": self._family_positions.copy(),
            "family_credits": self._family_credits.copy(),
            "family_steps": self._family_steps,
            "family_exhaustion_policy": "error",
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema") != "axon_phase0b_sampler_v1":
            raise ValueError(f"unsupported phase0b sampler state: {state.get('schema')!r}")
        if int(state.get("example_count", -1)) != len(self.examples):
            raise ValueError(
                "phase0b sampler example count changed "
                f"({state.get('example_count')} != {len(self.examples)})"
            )
        if list(state.get("families", [])) != self.families:
            raise ValueError(
                "phase0b sampler families changed "
                f"({state.get('families')!r} != {self.families!r})"
            )
        order = [int(i) for i in state.get("order", [])]
        if len(order) != len(self.examples) or sorted(order) != list(range(len(self.examples))):
            raise ValueError("phase0b sampler order is not a complete permutation")
        position = int(state.get("position", 0))
        if not 0 <= position <= len(order):
            raise ValueError(f"phase0b sampler position out of range: {position}")
        saved_policy = str(state.get("family_sampling_policy", "global"))
        if saved_policy != self.family_sampling:
            raise ValueError(
                f"phase0b sampler family policy changed ({saved_policy!r} != {self.family_sampling!r})"
            )
        if self.family_sampling == "weighted":
            if state.get("family_exhaustion_policy") != "error":
                raise ValueError("weighted phase0b sampler exhaustion policy is missing or incompatible")
            saved_weights = state.get("family_weights")
            if not isinstance(saved_weights, dict) or saved_weights != self.family_weights:
                raise ValueError("weighted phase0b sampler family weights changed")
            saved_orders = state.get("family_orders")
            saved_positions = state.get("family_positions")
            saved_credits = state.get("family_credits")
            if not all(isinstance(value, dict) for value in (saved_orders, saved_positions, saved_credits)):
                raise ValueError("weighted phase0b sampler state is incomplete")
            normalized_orders: dict[str, list[int]] = {}
            for family in self.families:
                family_order = [int(index) for index in saved_orders.get(family, [])]
                expected = sorted(self._family_orders[family])
                if sorted(family_order) != expected:
                    raise ValueError(f"weighted phase0b sampler order is invalid for family {family!r}")
                family_position = int(saved_positions.get(family, -1))
                if not 0 <= family_position <= len(family_order):
                    raise ValueError(f"weighted phase0b sampler position is invalid for family {family!r}")
                normalized_orders[family] = family_order
            family_steps = int(state.get("family_steps", -1))
            if family_steps < 0 or family_steps != sum(int(saved_positions[family]) for family in self.families):
                raise ValueError("weighted phase0b sampler family step count is invalid")
            if position != family_steps:
                raise ValueError("weighted phase0b sampler global position does not match family steps")
            self._family_orders = normalized_orders
            self._family_positions = {family: int(saved_positions[family]) for family in self.families}
            self._family_credits = {family: float(saved_credits[family]) for family in self.families}
            self._family_steps = family_steps
        self._order = order
        self._pos = position
        self.rng.setstate(state["rng_state"])


def capture_training_state(curriculum: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema": "axon_training_state_v1",
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if hasattr(curriculum, "state_dict"):
        state["curriculum_state"] = curriculum.state_dict()
    return state


def _curriculum_state_summary(state: Any) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    return {
        "schema": state.get("schema"),
        "example_count": state.get("example_count"),
        "position": state.get("position"),
        "families": list(state.get("families", [])),
        "family_sampling_policy": state.get("family_sampling_policy", "global"),
        "family_weights": dict(state.get("family_weights", {})),
        "family_positions": dict(state.get("family_positions", {})),
        "family_steps": state.get("family_steps", 0),
        "family_exhaustion_policy": state.get("family_exhaustion_policy"),
    }


def restore_training_state(
    state: dict[str, Any],
    curriculum: Any,
    *,
    curriculum_state_policy: str = "restore",
) -> dict[str, Any]:
    if state.get("schema") != "axon_training_state_v1":
        raise ValueError(f"unsupported training state: {state.get('schema')!r}")
    if curriculum_state_policy not in {"restore", "reset"}:
        raise ValueError(f"unsupported curriculum state policy: {curriculum_state_policy!r}")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    cuda_states = state.get("cuda_rng_state_all")
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    curriculum_state = state.get("curriculum_state")
    if curriculum_state is not None:
        if curriculum_state_policy == "restore":
            if not hasattr(curriculum, "load_state_dict"):
                raise ValueError("checkpoint has curriculum state but active curriculum cannot restore it")
            curriculum.load_state_dict(curriculum_state)
    return {
        "policy": curriculum_state_policy,
        "source_state_present": curriculum_state is not None,
        "restored": curriculum_state is not None and curriculum_state_policy == "restore",
        "intentionally_reset": curriculum_state is not None and curriculum_state_policy == "reset",
        "source": _curriculum_state_summary(curriculum_state),
        "active": _curriculum_state_summary(curriculum.state_dict()) if hasattr(curriculum, "state_dict") else None,
    }


def load_optimizer_state_with_lr_policy(
    optimizer: torch.optim.Optimizer,
    optimizer_state: dict[str, Any],
    *,
    requested_lr: float,
    policy: str,
) -> dict[str, Any]:
    """Restore Adam moments, then explicitly preserve or override loaded LR."""
    if policy not in {"preserve", "override"}:
        raise ValueError(f"unsupported resume LR policy: {policy!r}")
    source_lrs = [float(group.get("lr", 0.0)) for group in optimizer_state.get("param_groups", [])]
    optimizer.load_state_dict(optimizer_state)
    loaded_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if policy == "override":
        for group in optimizer.param_groups:
            group["lr"] = float(requested_lr)
    effective_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    return {
        "schema": "axon_optimizer_resume_v1",
        "state_restored": True,
        "lr_policy": policy,
        "requested_lr": float(requested_lr),
        "source_lrs": source_lrs,
        "loaded_lrs": loaded_lrs,
        "effective_lrs": effective_lrs,
    }


class CheckpointManager:
    """Rolling checkpoints with pointer.json and checkpoint_done.json sentinels."""

    def __init__(self, run_dir: pathlib.Path, keep: int = 3):
        self.run_dir = run_dir
        self.keep = keep
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._slot = 0

    def save(
        self,
        core: AxonCore,
        soul_mgr: SoulManagerV2,
        cfg: CoreConfig,
        soul_cfg: SoulV2Config,
        step: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        curriculum: Any = None,
        include_optimizer: bool = False,
        training_provenance: dict[str, Any] | None = None,
        continuity_provenance: dict[str, Any] | None = None,
    ) -> pathlib.Path:
        active = f"ckpt_{self._slot}.pt"
        self._slot = (self._slot + 1) % self.keep
        path = self.run_dir / active
        payload = {
            "checkpoint_schema": "axon_charslot_checkpoint_v2",
            "step": step,
            "cfg": cfg.to_dict(),
            "soul_cfg": soul_cfg.to_dict(),
            "core_state": core.state_dict(),
            "soul_mgr_state": soul_mgr.state_dict(),
            "soul_state": soul_mgr.state.to_saveable(),
            "training_state": capture_training_state(curriculum),
        }
        if training_provenance is not None:
            payload["training_provenance"] = training_provenance
        if continuity_provenance is not None:
            payload["continuity_provenance"] = continuity_provenance
        if include_optimizer:
            if optimizer is None:
                raise ValueError("include_optimizer=True requires an optimizer")
            payload["optimizer_state"] = optimizer.state_dict()
        torch.save(payload, path)
        (self.run_dir / "pointer.json").write_text(json.dumps({"step": step, "active": active}), encoding="utf-8")
        (self.run_dir / "checkpoint_done.json").write_text(
            json.dumps({"step": step, "time": time.time(), "optimizer_state": include_optimizer}),
            encoding="utf-8",
        )
        return path


CHARSLOT_REGION_HISTORY = 0
CHARSLOT_REGION_USER = 1
CHARSLOT_REGION_RESPONSE = 2


class CharSlotFieldBuilder:
    """Char-granular field: history + user_input + response_draft."""

    def __init__(self, history_chars: int, user_chars: int, resp_chars: int, device: torch.device, dtype: torch.dtype):
        self.history_chars = history_chars
        self.user_chars = user_chars
        self.resp_chars = resp_chars
        self.n_slots = history_chars + user_chars + resp_chars
        self.resp_slice = slice(history_chars + user_chars, self.n_slots)
        self.device = device
        self.dtype = dtype
        self.bank = get_letter_bank()
        self.char_index = {c: i for i, c in enumerate(self.bank.chars)}
        self.empty_index = self.bank.empty_index
        self.bank_unit = torch.from_numpy(self.bank.vecs_unit.copy()).to(device, dtype)
        region = np.zeros((self.n_slots,), dtype=np.int64)
        region[history_chars : history_chars + user_chars] = CHARSLOT_REGION_USER
        region[self.resp_slice] = CHARSLOT_REGION_RESPONSE
        self._region = torch.from_numpy(region).unsqueeze(0).to(device)

    def _write_block(self, out: np.ndarray, text: str, offset: int, width: int) -> None:
        clipped = text[:width]
        assert_supported_text(clipped)
        for i, ch in enumerate(clipped):
            out[offset + i] = char_to_slot(ch)

    def _char_targets(self, answer: str, loss_prefix_mask: int = 0) -> np.ndarray:
        t = np.full((self.resp_chars,), self.empty_index, dtype=np.int64)
        clipped = answer[: self.resp_chars]
        assert_supported_text(clipped)
        for i, ch in enumerate(clipped):
            t[i] = self.char_index[ch]
        if loss_prefix_mask > 0:
            t[: min(loss_prefix_mask, self.resp_chars)] = -100
        return t

    def build(
        self,
        history: str,
        user_input: str,
        answer: str,
        draft_text: str,
        loss_prefix_mask: int = 0,
    ) -> dict[str, torch.Tensor]:
        field16 = np.zeros((self.n_slots, SUBSTRATE_SLOT_DIM), dtype=np.float32)
        self._write_block(field16, history[-self.history_chars :], 0, self.history_chars)
        self._write_block(field16, user_input, self.history_chars, self.user_chars)
        self._write_block(field16, draft_text, self.resp_slice.start, self.resp_chars)
        return {
            "field16": torch.from_numpy(field16).unsqueeze(0).to(self.device, self.dtype),
            "region": self._region,
            "targets": torch.from_numpy(self._char_targets(answer, loss_prefix_mask)).unsqueeze(0).to(self.device),
        }

    def decode(self, logits: torch.Tensor, n_chars: int) -> str:
        idx = logits.argmax(dim=-1)[0].tolist()
        return "".join("" if i == self.empty_index else self.bank.chars[i] for i in idx[:n_chars])

    def verify_roundtrip(self, text: str) -> str:
        """Prove exact encode/decode identity through this builder's 16D path."""
        assert_supported_text(text)
        if len(text) > self.n_slots:
            raise ValueError(f"roundtrip input length {len(text)} exceeds builder slots {self.n_slots}")
        field16 = np.zeros((self.n_slots, SUBSTRATE_SLOT_DIM), dtype=np.float32)
        self._write_block(field16, text, 0, self.n_slots)
        field = torch.from_numpy(field16).to(self.device, self.dtype)
        logits = (field @ self.bank_unit.T).unsqueeze(0)
        decoded = self.decode(logits, len(text))
        if decoded != text:
            raise RuntimeError(f"16D charslot roundtrip mismatch: {text!r} -> {decoded!r}")
        return decoded


def run_charslot_roundtrip_gate(builder: CharSlotFieldBuilder) -> None:
    """Launch-blocking proof for the frozen 16D alphabet and field builder."""
    if SUBSTRATE_SLOT_DIM != 16:
        raise RuntimeError(f"substrate slot width changed: expected 16, got {SUBSTRATE_SLOT_DIM}")
    total, n_fail, failures = roundtrip_check()
    if n_fail:
        raise RuntimeError(f"substrate roundtrip failed for {n_fail}/{total}: {failures!r}")

    alphabet = "".join(default_alphabet())
    for start in range(0, len(alphabet), builder.n_slots):
        builder.verify_roundtrip(alphabet[start : start + builder.n_slots])
    builder.verify_roundtrip("axon 16D exact\nround trip: Aa0!?[]{}")
    log("GATE", f"exact 16D charslot roundtrip passed for all {total} supported characters")


@torch.no_grad()
def evaluate_charslot(
    core: AxonCore,
    soul_mgr: SoulManagerV2,
    builder: CharSlotFieldBuilder,
    examples: list[dict],
    draft_mode: str = "blank",
    partial_frac: float = 0.5,
) -> dict[str, Any]:
    core.eval()
    case_ids = [
        str(ex.get("eval_case_id") or Phase0bCurriculum._stable_case_id(ex))
        for ex in examples
    ]
    case_set_sha256 = hashlib.sha256("\n".join(case_ids).encode("utf-8")).hexdigest()
    family_counts = Counter(str(ex.get("family", "unknown")) for ex in examples)
    exact = chars_total = chars_correct = 0
    sfx_exact = sfx_total = sfx_correct = sfx_n = 0
    all_logits: list[torch.Tensor] = []
    pred_chars_all: list[str] = []
    samples: list[dict[str, str]] = []
    for ex in examples:
        ans = ex.get("answer", "")
        user_input = ex.get("user_input", ex.get("context", ""))
        history = ex.get("conversation_history", "")
        draft_text = draft_seed_for_mode(ans, draft_mode, partial_frac=partial_frac)
        built = builder.build(history, user_input, ans, draft_text)
        soul, soul_mask = soul_mgr.inhale()
        out = core.forward_charslot(
            built["field16"],
            built["region"],
            soul.to(builder.dtype),
            soul_mask=soul_mask.to(builder.device),
            response_slice=builder.resp_slice,
        )
        logits = core.charslot_logits(out["response_delta_16"], builder.bank_unit)
        all_logits.append(logits[0].float().cpu())
        pred = builder.decode(logits, len(ans))
        pred_chars_all.extend(pred)
        if len(samples) < 3:
            samples.append(
                {
                    "story_id": str(ex.get("story_id", "-")),
                    "history": history[-96:],
                    "user_input": user_input[:96],
                    "draft_seed": draft_text[:96],
                    "target": ans[:96],
                    "pred": pred[:96],
                }
            )
        exact += int(pred == ans)
        for a, b in zip(pred, ans):
            chars_total += 1
            chars_correct += int(a == b)
        k = len(draft_text)
        t_sfx, p_sfx = ans[k:], pred[k:]
        if t_sfx:
            sfx_n += 1
            sfx_exact += int(p_sfx == t_sfx)
            for a, b in zip(p_sfx.ljust(len(t_sfx)), t_sfx):
                sfx_total += 1
                sfx_correct += int(a == b)
    core.train()
    t = max(1, len(examples))
    collapse_var = entropy = 0.0
    if all_logits:
        probs = torch.stack([F.softmax(l, dim=-1) for l in all_logits])
        collapse_var = float(probs.mean(dim=0).var().item())
        entropy = float((-(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)).mean().item())
    pred_counts = Counter(pred_chars_all)
    pred_total = max(1, len(pred_chars_all))
    return {
        "exact_fill": exact / t,
        "char_acc": chars_correct / max(1, chars_total),
        "suffix_char_acc": sfx_correct / max(1, sfx_total),
        "suffix_exact": sfx_exact / max(1, sfx_n),
        "collapse_var": collapse_var,
        "pred_entropy": entropy,
        "pred_unique": len(pred_counts),
        "pred_top_frac": max(pred_counts.values(), default=0) / pred_total,
        "samples": samples,
        "n": t,
        "case_ids": case_ids,
        "case_set_sha256": case_set_sha256,
        "family_counts": dict(sorted(family_counts.items())),
    }


def evaluate_charslot_suite(
    core: AxonCore,
    soul_mgr: SoulManagerV2,
    builder: CharSlotFieldBuilder,
    examples: list[dict[str, Any]],
    *,
    partial_frac: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Evaluate the fixed copy/partial/blank suite on one immutable case set."""
    return {
        mode: evaluate_charslot(
            core,
            soul_mgr,
            builder,
            examples,
            draft_mode=mode,
            partial_frac=partial_frac,
        )
        for mode in DRAFT_MODES
    }


def build_fixed_eval_envelope(
    *,
    step: int,
    fixed_suite_sha256: str,
    metrics: dict[str, dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the fail-closed source/final envelope consumed by pilot gates."""
    if not re.fullmatch(r"[0-9a-fA-F]{64}", fixed_suite_sha256):
        raise ValueError("fixed_suite_sha256 must be 64 hexadecimal characters")
    if set(metrics) != set(DRAFT_MODES):
        raise ValueError(f"fixed evaluation requires modes {DRAFT_MODES!r}; got {sorted(metrics)!r}")
    observed_hashes = {mode_metrics.get("case_set_sha256") for mode_metrics in metrics.values()}
    if observed_hashes != {fixed_suite_sha256}:
        raise ValueError(f"fixed evaluation metrics use inconsistent case sets: {observed_hashes!r}")
    envelope: dict[str, Any] = {
        "schema": "axon_charslot_fixed_eval_v1",
        "step": int(step),
        "fixed_suite_sha256": fixed_suite_sha256.lower(),
        "metrics": metrics,
    }
    if metadata:
        envelope.update(metadata)
    return envelope


def _parse_mode_weights(value: str) -> list[float]:
    weights = [float(x.strip()) for x in value.split(",")]
    if len(weights) != 3:
        raise ValueError("--charslot-mode-weights wants 'copy,partial,blank'")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("--charslot-mode-weights must be non-negative and not all zero")
    return weights


def _set_default(args: argparse.Namespace, name: str, value: Any) -> None:
    if not hasattr(args, name):
        setattr(args, name, value)


def _ensure_charslot_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "mode": "phase0",
        "threshold": "charslot",
        "history_chars": 256,
        "user_chars": 64,
        "charslot_mode_weights": "0.3,0.3,0.4",
        "charslot_rung_preset": "manual",
        "charslot_partial_frac": 0.5,
        "phase0_target_words": 0,
        "phase0b_dir": "datasets/recovered/phase0b_curriculum_v1",
        "phase0b_families": ",".join(PHASE0B_DEFAULT_FAMILIES),
        "phase0b_max_examples": 100_000,
        "phase0b_family_sampling": "global",
        "phase0b_family_weights": "",
        "phase1a_dir": "datasets/recovered/phase1a_curriculum_v2",
        "phase1a_families": ",".join(PHASE1A_DEFAULT_FAMILIES),
        "phase1a_max_examples": 100_000,
        "text_corpus": "",
        "text_corpus_weight": 0.7,
        "text_corpus_max": 1_000_000,
        "core_cfg": "A",
        "device": "cpu",
        "fp16": False,
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "grad_checkpoint": False,
        "dropout": 0.0,
        "steps": 10000,
        "smoke": False,
        "smoke_steps": 250,
        "history_turns": 10,
        "curriculum_dir": "datasets/recovered/curriculum_v1",
        "containers_path": "State/dormant/containers.jsonl",
        "run_dir": "runs/charslot",
        "max_response_chars": 64,
        "soul_rows": 64,
        "soul_hot_rows": 64,
        "n_soul_compartments": 6,
        "soul_gate_init": 0.05,
        "hot_rows": 128,
        "warm_rows": 32,
        "cold_rows": 8,
        "router_threshold": 0.5,
        "write_gate_init": 0.1,
        "eval_every": 500,
        "eval_n": 32,
        "eval_samples": 2,
        "eval_partial_frac": 0.5,
        "eval_source_on_resume": True,
        "checkpoint_every": 1000,
        "log_every": 50,
        "collapse_threshold": 1e-6,
        "resume": "",
        "resume_lr_policy": "preserve",
        "resume_curriculum_state_policy": "restore",
        "resume_curriculum_reset_reason": "",
        "seed": 0,
    }
    for key, value in defaults.items():
        _set_default(args, key, value)


def apply_charslot_rung_preset(args: argparse.Namespace) -> None:
    _ensure_charslot_defaults(args)
    preset = getattr(args, "charslot_rung_preset", "manual")
    if preset == "manual":
        return
    if preset == "tight-stories":
        args.charslot_mode_weights = "0.10,0.55,0.35"
        args.charslot_partial_frac = max(float(args.charslot_partial_frac), 0.70)
        args.phase0_target_words = int(args.phase0_target_words or 4)
        args.text_corpus_weight = max(float(args.text_corpus_weight), 0.90)
        args.eval_n = max(int(args.eval_n), 64)
        return
    if preset == "context-stories":
        args.charslot_mode_weights = "0.10,0.50,0.40"
        args.charslot_partial_frac = max(float(args.charslot_partial_frac), 0.75)
        args.phase0_target_words = int(args.phase0_target_words or 4)
        args.text_corpus_weight = max(float(args.text_corpus_weight), 0.90)
        args.history_chars = max(int(args.history_chars), 512)
        args.eval_n = max(int(args.eval_n), 64)
        return
    raise ValueError(f"unknown --charslot-rung-preset {preset!r}")


def parse_core_cfg(value: str) -> tuple[int, int, int, int]:
    if value in LOCKED_CORE_PRESETS:
        value = LOCKED_CORE_PRESETS[value]
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--core-cfg must be d,layers,heads,ffn or a preset A/B/C; got {value}")
    return tuple(parts)  # type: ignore[return-value]


def train_charslot(args: argparse.Namespace) -> None:
    _ensure_charslot_defaults(args)
    if args.mode not in {"phase0", "phase0b", "phase1a"}:
        raise ValueError("the cleaned trainer currently supports --mode phase0, phase0b, or phase1a only")
    if args.threshold != "charslot":
        raise ValueError("the cleaned trainer only supports --threshold charslot")
    if args.resume_lr_policy not in {"preserve", "override"}:
        raise ValueError(f"unsupported --resume-lr-policy {args.resume_lr_policy!r}")
    if args.resume_curriculum_state_policy not in {"restore", "reset"}:
        raise ValueError(
            f"unsupported --resume-curriculum-state-policy {args.resume_curriculum_state_policy!r}"
        )
    if args.resume_curriculum_state_policy == "reset":
        if not args.resume:
            raise ValueError("--resume-curriculum-state-policy reset requires --resume")
        if not str(args.resume_curriculum_reset_reason).strip():
            raise ValueError(
                "--resume-curriculum-state-policy reset requires --resume-curriculum-reset-reason"
            )

    apply_charslot_rung_preset(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if getattr(args, "fp16", False) else torch.float32

    from cores.core import CHAR_SLOT_DIM

    assert CHAR_SLOT_DIM == SUBSTRATE_SLOT_DIM, "core CHAR_SLOT_DIM must match substrate SLOT_DIM"

    d_model, n_layers, n_heads, ffn_dim = parse_core_cfg(args.core_cfg)
    builder = CharSlotFieldBuilder(args.history_chars, args.user_chars, args.max_response_chars, device, dtype)
    run_charslot_roundtrip_gate(builder)
    core_cfg = CoreConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        dropout=args.dropout,
        soul_mode="act_reflect_v2",
        soul_rows=args.soul_rows,
        soul_hot_rows=args.soul_hot_rows,
        soul_write_mode="direct" if args.soul_hot_rows == 0 else "compartments",
        n_soul_compartments=args.n_soul_compartments,
        soul_gate_init=args.soul_gate_init,
        char_slot_mode=True,
        char_slot_max_slots=builder.n_slots,
    )
    core = AxonCore(core_cfg).to(device, dtype)
    core.use_checkpoint = bool(getattr(args, "grad_checkpoint", False))
    core.train()

    soul_cfg = SoulV2Config(
        d_model=d_model,
        tiers=[
            TierSpec(name="hot", max_rows=args.hot_rows, initial_active=0),
            TierSpec(name="warm", max_rows=args.warm_rows, initial_active=0),
            TierSpec(name="cold", max_rows=args.cold_rows, initial_active=0),
        ],
        categories=["episodic", "lessons", "diary", "awareness", "scratch", "tasks"],
        router_threshold=args.router_threshold,
        write_gate_init=args.write_gate_init,
    )
    soul_mgr = SoulManagerV2(soul_cfg, device, torch.float32, n_heads=n_heads)

    start_step = 0
    resume_payload: dict[str, Any] | None = None
    training_provenance: dict[str, Any] = {
        "schema": "axon_training_provenance_v1",
        "source_checkpoint": str(args.resume) if args.resume else None,
        "source_checkpoint_step": None,
        "optimizer": None,
        "curriculum": None,
    }
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        resume_payload = ckpt
        saved_cfg = CoreConfig.from_dict(ckpt["cfg"])
        resume_contract_fields = (
            "d_model",
            "n_layers",
            "n_heads",
            "ffn_dim",
            "soul_rows",
            "soul_hot_rows",
            "soul_mode",
            "soul_write_mode",
            "n_soul_compartments",
            "char_slot_mode",
            "char_n_regions",
        )
        mismatches = [
            f"{name} checkpoint={getattr(saved_cfg, name)!r} requested={getattr(core_cfg, name)!r}"
            for name in resume_contract_fields
            if getattr(saved_cfg, name) != getattr(core_cfg, name)
        ]
        if mismatches:
            raise ValueError("resume checkpoint core config mismatch: " + "; ".join(mismatches))
        if saved_cfg.char_slot_mode and int(saved_cfg.char_slot_max_slots) != int(builder.n_slots):
            raise ValueError(
                "resume checkpoint char_slot_max_slots does not match requested "
                f"builder slots ({saved_cfg.char_slot_max_slots} != {builder.n_slots})"
            )
        missing, unexpected = core.load_state_dict(ckpt["core_state"], strict=False)
        if missing or unexpected:
            log("RESUME", f"state mismatch tolerated missing={len(missing)} unexpected={len(unexpected)}")
        if "soul_mgr_state" in ckpt:
            soul_mgr.load_state_dict(ckpt["soul_mgr_state"])
        if "soul_state" in ckpt:
            from cores.soul_v2 import SoulState

            soul_mgr.state = SoulState.from_saveable(ckpt["soul_state"], device, torch.float32)
        start_step = int(ckpt.get("step", 0) or 0)
        training_provenance["source_checkpoint_step"] = start_step
        log("RESUME", f"loaded checkpoint from {args.resume} at step={start_step}")

    opt = torch.optim.AdamW(
        list(core.parameters()) + list(soul_mgr.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    if resume_payload is not None:
        optimizer_state = resume_payload.get("optimizer_state")
        if optimizer_state is None:
            training_provenance["optimizer"] = {
                "schema": "axon_optimizer_resume_v1",
                "state_restored": False,
                "lr_policy": args.resume_lr_policy,
                "requested_lr": float(args.lr),
                "source_lrs": [],
                "loaded_lrs": [],
                "effective_lrs": [float(group["lr"]) for group in opt.param_groups],
            }
            log("RESUME", "optimizer state absent; AdamW starts fresh for this leg")
        else:
            training_provenance["optimizer"] = load_optimizer_state_with_lr_policy(
                opt,
                optimizer_state,
                requested_lr=args.lr,
                policy=args.resume_lr_policy,
            )
            log(
                "RESUME",
                "optimizer state restored "
                f"lr_policy={args.resume_lr_policy} "
                f"source_lrs={training_provenance['optimizer']['source_lrs']} "
                f"effective_lrs={training_provenance['optimizer']['effective_lrs']}",
            )
        del optimizer_state
    else:
        training_provenance["optimizer"] = {
            "schema": "axon_optimizer_resume_v1",
            "state_restored": False,
            "lr_policy": "fresh",
            "requested_lr": float(args.lr),
            "source_lrs": [],
            "loaded_lrs": [],
            "effective_lrs": [float(group["lr"]) for group in opt.param_groups],
        }

    log(
        "BUILD",
        f"charslot core d={d_model} l={n_layers} h={n_heads} ffn={ffn_dim} "
        f"slots={builder.n_slots} (hist={args.history_chars} user={args.user_chars} "
        f"resp={args.max_response_chars}) params={sum(p.numel() for p in core.parameters()):,} "
        f"soul_params={sum(p.numel() for p in soul_mgr.parameters()):,} "
        f"device={device} dtype={dtype} grad_checkpoint={core.use_checkpoint}",
    )

    if args.mode in {"phase0b", "phase1a"}:
        phase_dir = args.phase1a_dir if args.mode == "phase1a" else args.phase0b_dir
        phase_families = args.phase1a_families if args.mode == "phase1a" else args.phase0b_families
        phase_max_examples = args.phase1a_max_examples if args.mode == "phase1a" else args.phase0b_max_examples
        curriculum = Phase0bCurriculum(
            phase0b_dir=phase_dir,
            families=phase_families,
            max_chars=args.max_response_chars,
            history_turns=args.history_turns,
            rng=random.Random(args.seed),
            max_examples=phase_max_examples,
            history_chars=args.history_chars,
            user_chars=args.user_chars,
            family_sampling=args.phase0b_family_sampling if args.mode == "phase0b" else "global",
            family_weights=args.phase0b_family_weights if args.mode == "phase0b" else None,
        )
    else:
        curriculum = Phase0Curriculum(
            curriculum_dir=args.curriculum_dir,
            containers_path=args.containers_path,
            max_chars=args.max_response_chars,
            history_turns=args.history_turns,
            rng=random.Random(args.seed),
            text_corpus=args.text_corpus or None,
            text_corpus_weight=args.text_corpus_weight,
            text_corpus_max=args.text_corpus_max,
            target_words=args.phase0_target_words,
        )
    if resume_payload is not None:
        training_state = resume_payload.get("training_state")
        if training_state is None:
            if args.resume_curriculum_state_policy == "reset":
                raise ValueError("cannot audit curriculum reset because checkpoint training_state is absent")
            training_provenance["curriculum"] = {
                "policy": args.resume_curriculum_state_policy,
                "source_state_present": False,
                "restored": False,
                "intentionally_reset": False,
                "reset_reason": None,
                "source": None,
                "active": _curriculum_state_summary(curriculum.state_dict())
                if hasattr(curriculum, "state_dict")
                else None,
            }
            log("RESUME", "RNG/sampler state absent; curriculum starts a fresh deterministic stream")
        else:
            curriculum_audit = restore_training_state(
                training_state,
                curriculum,
                curriculum_state_policy=args.resume_curriculum_state_policy,
            )
            curriculum_audit["reset_reason"] = (
                str(args.resume_curriculum_reset_reason).strip()
                if curriculum_audit["intentionally_reset"]
                else None
            )
            training_provenance["curriculum"] = curriculum_audit
            if curriculum_audit["intentionally_reset"]:
                log(
                    "RESUME",
                    "RNG restored; curriculum sampler intentionally reset "
                    f"reason={curriculum_audit['reset_reason']!r} "
                    f"source={curriculum_audit['source']} active={curriculum_audit['active']}",
                )
            else:
                log("RESUME", "RNG and curriculum sampler state restored")
        del training_state
        resume_payload = None
        ckpt = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        training_provenance["curriculum"] = {
            "policy": "fresh",
            "source_state_present": False,
            "restored": False,
            "intentionally_reset": False,
            "reset_reason": None,
            "source": None,
            "active": _curriculum_state_summary(curriculum.state_dict())
            if hasattr(curriculum, "state_dict")
            else None,
        }
    continuity_provenance: dict[str, Any] | None = None
    if args.resume:
        optimizer_audit = training_provenance["optimizer"]
        curriculum_audit = training_provenance["curriculum"]
        effective_lrs = list(optimizer_audit.get("effective_lrs", []))
        if not effective_lrs or any(lr != effective_lrs[0] for lr in effective_lrs[1:]):
            raise ValueError(f"resumed optimizer has ambiguous effective learning rates: {effective_lrs!r}")
        if optimizer_audit["state_restored"]:
            optimizer_policy = (
                "restore_state_override_lr"
                if args.resume_lr_policy == "override"
                else "restore_state_preserve_lr"
            )
        else:
            optimizer_policy = "fresh_state_requested_lr"
        if curriculum_audit["intentionally_reset"]:
            sampler_policy = "reset_exact_v3"
        elif curriculum_audit["restored"]:
            sampler_policy = "restore_exact_sampler"
        else:
            sampler_policy = "fresh_deterministic_sampler"
        active_sampler = curriculum_audit.get("active") or {}
        continuity_provenance = {
            "schema": "axon_continuity_provenance_v1",
            "source_step": start_step,
            "optimizer": {
                "policy": optimizer_policy,
                "state_restored": bool(optimizer_audit["state_restored"]),
                "effective_lr": float(effective_lrs[0]),
            },
            "sampler": {
                "policy": sampler_policy,
                "reset": bool(curriculum_audit["intentionally_reset"]),
                "initial_position": int(active_sampler.get("position", 0) or 0),
                "families": list(active_sampler.get("families", [])),
                "family_sampling_policy": active_sampler.get("family_sampling_policy", "global"),
                "family_weights": dict(active_sampler.get("family_weights", {})),
                "family_exhaustion_policy": active_sampler.get("family_exhaustion_policy"),
            },
        }
        training_provenance["continuity"] = continuity_provenance
    ckpt_mgr = CheckpointManager(pathlib.Path(args.run_dir), keep=3)

    mode_weights = _parse_mode_weights(args.charslot_mode_weights)
    fixed_eval_examples = curriculum.eval_examples(args.eval_n)
    if not fixed_eval_examples:
        raise ValueError("evaluation case set is empty")
    eval_case_set_sha256 = hashlib.sha256(
        "\n".join(str(ex.get("eval_case_id") or Phase0bCurriculum._stable_case_id(ex)) for ex in fixed_eval_examples).encode(
            "utf-8"
        )
    ).hexdigest()
    eval_family_counts = dict(
        sorted(Counter(str(ex.get("family", "unknown")) for ex in fixed_eval_examples).items())
    )
    log(
        "RUNG",
        f"preset={args.charslot_rung_preset} weights=copy/partial/blank:{mode_weights} "
        f"partial_frac={args.charslot_partial_frac:.2f} "
        f"eval_partial_frac={args.eval_partial_frac:.2f} eval_case_sha={eval_case_set_sha256} "
        f"eval_families={eval_family_counts} "
        f"target_words={args.phase0_target_words or 'half'} history_chars={args.history_chars}",
    )

    if args.resume and args.eval_source_on_resume:
        source_metrics = evaluate_charslot_suite(
            core,
            soul_mgr,
            builder,
            fixed_eval_examples,
            partial_frac=args.eval_partial_frac,
        )
        source_case_hashes = {metrics["case_set_sha256"] for metrics in source_metrics.values()}
        if source_case_hashes != {eval_case_set_sha256}:
            raise RuntimeError(f"source evaluation case identity drifted: {sorted(source_case_hashes)}")
        source_eval_path = pathlib.Path(args.run_dir) / "eval_source.json"
        source_eval_payload = build_fixed_eval_envelope(
            step=start_step,
            fixed_suite_sha256=eval_case_set_sha256,
            metrics=source_metrics,
            metadata={
                "role": "pre_update_source",
                "charslot_partial_frac": args.charslot_partial_frac,
                "eval_partial_frac": args.eval_partial_frac,
                "case_set_sha256": eval_case_set_sha256,
                "family_counts": eval_family_counts,
            },
        )
        source_eval_path.write_text(
            json.dumps(source_eval_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        training_provenance["source_evaluation"] = {
            "path": source_eval_path.name,
            "step": start_step,
            "case_set_sha256": eval_case_set_sha256,
            "family_counts": eval_family_counts,
        }
        log("EVAL_SOURCE", f"step={start_step} case_sha={eval_case_set_sha256} -> {source_eval_path}")
    else:
        training_provenance["source_evaluation"] = None

    if args.smoke:
        args.steps = min(args.steps, args.smoke_steps)
        args.eval_every = min(args.eval_every, args.smoke_steps)
        args.checkpoint_every = 10**9

    skipped_overlength = 0
    losses: list[float] = []
    latest_eval: dict[str, Any] = {}
    latest_eval_step: int | None = None
    t0 = time.time()
    step = start_step
    if start_step >= args.steps:
        log("DONE", f"checkpoint step {start_step} already reached target step {args.steps}")
        return

    for step in range(start_step + 1, args.steps + 1):
        ex = curriculum.next()
        ans = ex.get("answer", "")
        if not ans:
            continue
        strict_budget = bool(ex.get("strict_budget"))
        if len(ans) > args.max_response_chars:
            if strict_budget:
                raise RuntimeError(
                    f"exact delta answer escaped loader budget: {len(ans)} > {args.max_response_chars}"
                )
            skipped_overlength += 1
            continue
        raw_user_input = ex.get("user_input", ex.get("context", ""))
        if strict_budget:
            if len(raw_user_input) > args.user_chars:
                raise RuntimeError(
                    f"exact delta user input escaped loader budget: {len(raw_user_input)} > {args.user_chars}"
                )
            user_input = raw_user_input
        else:
            user_input = raw_user_input[: args.user_chars]
        history = ex.get("conversation_history", "")
        draft_stage = choose_draft_mode(ex, mode_weights)
        draft_text = draft_seed_for_mode(ans, draft_stage, partial_frac=args.charslot_partial_frac)
        prefix_mask = len(draft_text) if draft_stage == "partial" else 0
        built = builder.build(history, user_input, ans, draft_text, loss_prefix_mask=prefix_mask)

        soul, soul_mask = soul_mgr.inhale()
        out = core.forward_charslot(
            built["field16"],
            built["region"],
            soul.to(dtype),
            soul_mask=soul_mask.to(device),
            response_slice=builder.resp_slice,
        )
        logits = core.charslot_logits(out["response_delta_16"], builder.bank_unit)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), built["targets"].reshape(-1), ignore_index=-100)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(soul_mgr.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.item()))

        with torch.no_grad():
            soul_mgr.exhale_after_answer(out["field"].detach())
            if step % 10 == 0:
                soul_mgr.maybe_compress()
                soul_mgr.maybe_evict()

        if step % args.log_every == 0 or step == 1:
            trained_steps = step - start_step
            sps = trained_steps / max(1e-6, time.time() - t0)
            recent = np.mean(losses[-args.log_every:]) if losses else 0.0
            log(
                "STEP",
                f"step={step}/{args.steps} mode={args.mode} family={ex.get('family', '-')} "
                f"target={ex.get('target_region', 'response_draft')} draft={draft_stage} "
                f"loss={recent:.4f} char={float(loss.detach().cpu().item()):.4f} "
                f"soul_active={soul_mgr.state.n_active()} skipped={skipped_overlength} {sps:.1f}it/s",
            )

        if step % args.eval_every == 0:
            latest_eval_step = step
            eval_examples = fixed_eval_examples
            stop_for_collapse = False
            for eval_mode in ("copy", "partial", "blank"):
                metrics = evaluate_charslot(
                    core,
                    soul_mgr,
                    builder,
                    eval_examples,
                    draft_mode=eval_mode,
                    partial_frac=args.eval_partial_frac,
                )
                latest_eval[eval_mode] = metrics
                log(
                    f"EVAL_{eval_mode.upper()}",
                    f"step={step} exact_fill={metrics['exact_fill']:.3f} "
                    f"char_acc={metrics['char_acc']:.3f} "
                    f"sfx_acc={metrics['suffix_char_acc']:.3f} sfx_exact={metrics['suffix_exact']:.3f} "
                    f"entropy={metrics['pred_entropy']:.3f} uniq={metrics['pred_unique']} "
                    f"top={metrics['pred_top_frac']:.3f} collapse_var={metrics['collapse_var']:.6f} n={metrics['n']}",
                )
                for i, sample in enumerate(metrics.get("samples", [])[: int(getattr(args, "eval_samples", 0))]):
                    log(
                        "PRED",
                        f"step={step} mode={eval_mode} sample={i} story={sample['story_id']} "
                        f"history=...{sample['history'][-60:]!r} user_input={sample['user_input']!r} "
                        f"draft_seed={sample['draft_seed']!r} target={sample['target']!r} pred={sample['pred']!r}",
                    )
                if metrics["collapse_var"] < args.collapse_threshold and step > args.smoke_steps:
                    log("TRIPWIRE", f"collapse detected at step {step} mode={eval_mode} (var={metrics['collapse_var']:.6f}); halting")
                    stop_for_collapse = True
                    break
            if stop_for_collapse:
                break

        if step % args.checkpoint_every == 0 and not args.smoke:
            path = ckpt_mgr.save(
                core,
                soul_mgr,
                core_cfg,
                soul_cfg,
                step,
                optimizer=opt,
                curriculum=curriculum,
                include_optimizer=True,
                training_provenance=training_provenance,
                continuity_provenance=continuity_provenance,
            )
            log("SAVE", f"checkpoint step={step} -> {path}")

    if not args.smoke:
        if latest_eval_step != step or set(latest_eval) != set(DRAFT_MODES):
            latest_eval = evaluate_charslot_suite(
                core,
                soul_mgr,
                builder,
                fixed_eval_examples,
                partial_frac=args.eval_partial_frac,
            )
            latest_eval_step = step
        final_case_hashes = {metrics["case_set_sha256"] for metrics in latest_eval.values()}
        if final_case_hashes != {eval_case_set_sha256}:
            raise RuntimeError(f"final evaluation case identity drifted: {sorted(final_case_hashes)}")
        path = ckpt_mgr.save(
            core,
            soul_mgr,
            core_cfg,
            soul_cfg,
            step,
            optimizer=opt,
            curriculum=curriculum,
            include_optimizer=True,
            training_provenance=training_provenance,
            continuity_provenance=continuity_provenance,
        )
        log("SAVE", f"final checkpoint step={step} -> {path}")
        eval_result_path = pathlib.Path(args.run_dir) / "eval_final.json"
        final_eval_payload = build_fixed_eval_envelope(
            step=step,
            fixed_suite_sha256=eval_case_set_sha256,
            metrics=latest_eval,
            metadata={
                "role": "post_update_final",
                "eval_step": latest_eval_step,
                "charslot_partial_frac": args.charslot_partial_frac,
                "eval_partial_frac": args.eval_partial_frac,
                "case_set_sha256": eval_case_set_sha256,
                "family_counts": eval_family_counts,
                "training_provenance": training_provenance,
            },
        )
        eval_result_path.write_text(
            json.dumps(final_eval_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        log("SAVE", f"final evaluation metrics -> {eval_result_path}")
    log("DONE", f"charslot training finished at step={step}")


def train(args: argparse.Namespace) -> None:
    """Compatibility entrypoint: the only trainer path is charslot."""
    train_charslot(args)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Axon exact-16D charfield trainer")
    ap.add_argument("--mode", choices=["phase0", "phase0b", "phase1a"], default="phase0")
    ap.add_argument("--threshold", choices=["charslot"], default="charslot")
    ap.add_argument("--history-chars", type=int, default=256)
    ap.add_argument("--user-chars", type=int, default=64)
    ap.add_argument("--charslot-mode-weights", default="0.3,0.3,0.4")
    ap.add_argument("--charslot-rung-preset", choices=["manual", "tight-stories", "context-stories"], default="manual")
    ap.add_argument("--charslot-partial-frac", type=float, default=0.5)
    ap.add_argument("--phase0-target-words", type=int, default=0)
    ap.add_argument("--phase0b-dir", default="datasets/recovered/phase0b_curriculum_v1")
    ap.add_argument("--phase0b-families", default=",".join(PHASE0B_DEFAULT_FAMILIES))
    ap.add_argument("--phase0b-max-examples", type=int, default=100_000)
    ap.add_argument(
        "--phase0b-family-sampling",
        choices=["global", "weighted"],
        default="global",
        help="global permutation or deterministic weighted without-replacement family scheduling",
    )
    ap.add_argument(
        "--phase0b-family-weights",
        default="",
        help="comma-separated weights aligned with --phase0b-families; required for weighted sampling",
    )
    ap.add_argument("--phase1a-dir", default="datasets/recovered/phase1a_curriculum_v2")
    ap.add_argument("--phase1a-families", default=",".join(PHASE1A_DEFAULT_FAMILIES))
    ap.add_argument("--phase1a-max-examples", type=int, default=100_000)
    ap.add_argument("--text-corpus", default="")
    ap.add_argument("--text-corpus-weight", type=float, default=0.7)
    ap.add_argument("--text-corpus-max", type=int, default=1_000_000)
    ap.add_argument("--core-cfg", default="A")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32", action="store_true", help="kept for compatibility; float32 is already default")
    ap.add_argument("--fp16", action="store_true", help="opt-in float16")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--resume-lr-policy",
        choices=["preserve", "override"],
        default="preserve",
        help="preserve checkpoint LR or override it with --lr after restoring optimizer moments",
    )
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=250)
    ap.add_argument("--history-turns", type=int, default=10)
    ap.add_argument("--curriculum-dir", default="datasets/recovered/curriculum_v1")
    ap.add_argument("--containers-path", default="State/dormant/containers.jsonl")
    ap.add_argument("--run-dir", default="runs/charslot")
    ap.add_argument("--max-response-chars", type=int, default=64)
    ap.add_argument("--soul-rows", type=int, default=64)
    ap.add_argument("--soul-hot-rows", type=int, default=64)
    ap.add_argument("--n-soul-compartments", type=int, default=6)
    ap.add_argument("--soul-gate-init", type=float, default=0.05)
    ap.add_argument("--hot-rows", type=int, default=128)
    ap.add_argument("--warm-rows", type=int, default=32)
    ap.add_argument("--cold-rows", type=int, default=8)
    ap.add_argument("--router-threshold", type=float, default=0.5)
    ap.add_argument("--write-gate-init", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=32)
    ap.add_argument("--eval-samples", type=int, default=2)
    ap.add_argument(
        "--eval-partial-frac",
        type=float,
        default=0.5,
        help="fixed partial-draft fraction for comparable source/final evaluation",
    )
    ap.add_argument(
        "--eval-source-on-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="persist eval_source.json before the first resumed update",
    )
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--collapse-threshold", type=float, default=1e-6)
    ap.add_argument("--resume", default="")
    ap.add_argument(
        "--resume-curriculum-state-policy",
        choices=["restore", "reset"],
        default="restore",
        help="restore sampler state or intentionally reset an incompatible curriculum sampler",
    )
    ap.add_argument(
        "--resume-curriculum-reset-reason",
        default="",
        help="required audit reason when --resume-curriculum-state-policy=reset",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.smoke:
        args.steps = min(args.steps, args.smoke_steps)
        if args.mode == "phase0":
            args.max_response_chars = 32
        args.hot_rows = 32
        args.warm_rows = 8
        args.cold_rows = 4
        args.eval_every = args.smoke_steps
        args.log_every = 10
        args.run_dir = "runs/charslot_smoke"

    train_charslot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
