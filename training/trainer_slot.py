#!/usr/bin/env python3
"""trainer_slot.py — Train-as-you-live engine for the slot era.

Governance:
  - SOURCE_OF_TRUTH.md Layer 5 (arithmetic-only response-delta path)
  - SOURCE_OF_TRUTH.md Layer 6/12 (temperature-tiered soul)
  - SOURCE_OF_TRUTH.md Layer 13 (training contracts)
  - docs/WORKING_CONTRACT.md

Every training step:
  1. INHALE private soul (SoulManagerV2.inhale)
  2. ATTEND masked shared field through frozen SlotAdapter
  3. EMIT response_draft delta as per-character d_model vectors from the core
  4. DECODE through frozen CharPrototypeTable + discrete CE
  5. EXHALE experience trace to hot soul

Modes:
  phase0  - bulk text continuation / reconstruction (volume)
  recall  - two-tick precision drills with cf probes
  both    - interleave phase0 and recall examples

No legacy_8192 code is promoted or copied.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

# Ensure repo root is importable when run as a script
_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from adapters.slot_adapter import get_adapter, get_char_prototype_table
from cores.core import AxonCore, CoreConfig
from cores.soul_v2 import SoulManagerV2, SoulV2Config, TierSpec
from slots.slot_field_contract import (
    FieldLayout,
    SlotField,
    default_layout,
    materialize_container,
    ContainerRecord,
    REGION_NAMES,
)
from slots.slot_spec import (
    SLOT_WIDTH,
    KIND_TEXT,
    KIND_RESPONSE_DRAFT,
    STATUS_ACTIVE,
    STATUS_DRAFT,
    pack_text_chain,
    pack_slot,
    unpack_chain,
    unpack_region,
)
from substrate import SLOT_DIM as SUBSTRATE_SLOT_DIM
from substrate import char_to_slot, default_alphabet, get_letter_bank


# ---------------------------------------------------------------------------
# Locked core presets (d_model, layers, heads, ffn)
# ---------------------------------------------------------------------------
LOCKED_CORE_PRESETS = {
    "A": "64,2,1,131072",
    "B": "128,2,1,262144",
    "C": "256,2,2,524288",
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(tag: str, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Field builder: pack lessons into named regions + mask unused regions
# ---------------------------------------------------------------------------
class SlotFieldBuilder:
    """Builds real 8192D fields, active masks, and full-delta targets."""

    def __init__(
        self,
        layout: FieldLayout,
        adapter,
        prototype_table,
        device: torch.device,
        dtype: torch.dtype,
        max_response_chars: int = 256,
    ):
        self.layout = layout
        self.adapter = adapter
        self.table = prototype_table
        self.device = device
        self.dtype = dtype
        self.max_response_chars = max_response_chars
        self.draft_slice = layout.region_slice("response_draft")
        self.draft_bounds = layout.region_bounds("response_draft")

    def build(
        self,
        region_texts: dict[str, str],
        answer_text: str = "",
        draft_text: str = "",
        active_regions: set[str] | None = None,
    ) -> dict[str, Any]:
        """Pack region texts into a field, mask inactive regions, build targets.

        The input field is real substrate data: each text character is packed as
        a frozen 16D code inside an 8192D slot, then projected down for the
        core.  The target field is built the same way.  ``field_delta_targets_d``
        is the typed full-field delta target in d_model space: read-only active
        regions target zero/no-op, while response_draft targets a replacement
        payload projected from a substrate-packed target field.

        Returns dict with:
          field_d:                  (1, n_slots, d_model) core input
          mask:                     (1, n_slots) active-region mask
          targets:                  (1, max_response_chars) char CE labels
          field_delta_targets_d:    (1, n_slots, d_model) full-delta labels
          field_delta_weights:      (1, n_slots) slot weights
          target_len:               active target char length
          field_np:                 (n_slots, 8192) input field snapshot
          target_field_np:          (n_slots, 8192) target field snapshot
        """
        active_regions = active_regions or set(region_texts.keys()) | {"response_draft"}
        field = SlotField.empty(self.layout)

        # Pack each active region. The response_draft field gets the current
        # draft seed, never the supervised target answer.
        for name in REGION_NAMES:
            if name == "response_draft":
                text = draft_text
                kind = KIND_RESPONSE_DRAFT
                status = STATUS_DRAFT
            else:
                text = region_texts.get(name, "")
                kind = KIND_TEXT
                status = STATUS_ACTIVE
            if not text:
                continue
            slots = pack_text_chain(text, kind=kind, status=status)
            field.set_region(name, slots)

        # Build the target field as a real substrate-packed field too. The
        # read-only regions match the input field; response_draft is the
        # writable replacement payload.
        target_field = SlotField(self.layout, field.data.copy())

        target_text = ""
        target_chars: list[str] = []
        if answer_text:
            target_chars = [c for c in answer_text[: self.max_response_chars] if c in self.table.chars]
            target_text = "".join(target_chars)
            target_field.clear_region("response_draft")
            if target_text:
                target_field.set_region(
                    "response_draft",
                    pack_text_chain(target_text, kind=KIND_RESPONSE_DRAFT, status=STATUS_DRAFT),
                )

        # Build boolean mask: True only for active regions
        mask_np = field.mask(active_regions)

        # Project down to d_model space
        field_np = field.data.astype(np.float32)
        target_field_np = target_field.data.astype(np.float32)
        field_d_np = self.adapter.project_down_region(field_np)  # (n_slots, d_model)
        target_field_d_np = self.adapter.project_down_region(target_field_np)

        # Typed full-field delta: active read-only slots learn no-op. Writable
        # response_draft slots learn the replacement payload in d_model space.
        field_delta_targets_np = np.zeros_like(field_d_np, dtype=np.float32)
        field_delta_weights_np = np.zeros((self.layout.total_slots,), dtype=np.float32)
        for name in active_regions:
            sl = self.layout.region_slice(name)
            field_delta_weights_np[sl] = 0.1
        draft_sl = self.layout.region_slice("response_draft")
        field_delta_targets_np[draft_sl] = target_field_d_np[draft_sl]
        field_delta_weights_np[draft_sl] = 1.0

        field_d = torch.from_numpy(field_d_np).to(self.device, dtype=self.dtype).unsqueeze(0)
        mask_t = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)
        field_delta_targets_d = torch.from_numpy(field_delta_targets_np).to(self.device, dtype=self.dtype).unsqueeze(0)
        field_delta_weights = torch.from_numpy(field_delta_weights_np).to(self.device, dtype=self.dtype).unsqueeze(0)

        # Build supervision targets for response_draft characters
        targets = torch.full((1, self.max_response_chars), -100, dtype=torch.long, device=self.device)
        target_len = len(target_chars)
        if target_chars:
            for i, ch in enumerate(target_chars):
                targets[0, i] = self.table.char_index(ch)

        return {
            "field_d": field_d,
            "mask": mask_t,
            "targets": targets,
            "field_delta_targets_d": field_delta_targets_d,
            "field_delta_weights": field_delta_weights,
            "target_len": target_len,
            "field_np": field_np,
            "target_field_np": target_field_np,
            "active_regions": active_regions,
        }


# ---------------------------------------------------------------------------
# Rendering helpers for training diagnostics
# ---------------------------------------------------------------------------
def _preview(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def render_field_snapshot(
    field_np: np.ndarray,
    layout: FieldLayout,
    active_regions: set[str],
    max_chars: int = 160,
) -> list[str]:
    """Render the active slot field as readable region text for diagnostics."""
    lines: list[str] = []
    for name in REGION_NAMES:
        if name not in active_regions:
            continue
        slots = unpack_region(field_np[layout.region_slice(name)])
        text, edges = unpack_chain(slots)
        text = _preview(text, max_chars)
        edges = _preview(edges, max_chars)
        if edges:
            lines.append(f"{name}: text={text!r} edges={edges!r}")
        else:
            lines.append(f"{name}: text={text!r}")
    return lines


def draft_seed_for_mode(answer: str, mode: str) -> str:
    """Return the visible response_draft seed for an evaluation/training mode."""
    if mode == "copy":
        return answer
    if mode == "partial":
        keep = max(0, len(answer) // 2)
        return answer[:keep].rstrip()
    if mode == "blank":
        return ""
    raise ValueError(f"unknown draft seed mode: {mode}")


def draft_seed_for_step(answer: str, step: int, args: argparse.Namespace) -> tuple[str, str]:
    """Scheduled copy -> repair -> predict response_draft visibility."""
    teacher_steps = max(0, int(getattr(args, "draft_teacher_steps", 0)))
    ramp_steps = max(0, int(getattr(args, "draft_mask_ramp_steps", 0)))
    if step <= teacher_steps:
        return answer, "copy"
    if ramp_steps <= 0:
        return "", "blank"
    progress = min(1.0, max(0.0, (step - teacher_steps) / ramp_steps))
    if progress >= 1.0:
        return "", "blank"
    keep = max(0, int(round(len(answer) * (1.0 - progress))))
    seed = answer[:keep].rstrip()
    return seed, "partial" if seed else "blank"


# ---------------------------------------------------------------------------
# Curriculum: Phase 0 bulk text
# ---------------------------------------------------------------------------
class Phase0Curriculum:
    """Stream substrate-safe sentences from recovered corpus + dormant containers."""

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
    ):
        self.max_chars = max_chars
        self.history_turns = max(0, history_turns)
        self.rng = rng or random.Random(42)
        self.sentences: list[str] = []
        self.eval_sentences: list[str] = []
        self.history: list[str] = []
        # Optional second pool: a pre-cleaned one-sentence-per-line corpus
        # (build_text_corpus.py output, e.g. TinyStories). Sampled with
        # probability text_corpus_weight against the memories pool.
        self.story_sentences: list[str] = []
        self.story_weight = float(text_corpus_weight) if text_corpus else 0.0
        self._load(curriculum_dir, containers_path)
        if text_corpus and os.path.exists(text_corpus):
            with open(text_corpus, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= text_corpus_max:
                        break
                    line = line.rstrip("\n")
                    if line:
                        self.story_sentences.append(line)
            self.rng.shuffle(self.story_sentences)
        self.rng.shuffle(self.sentences)
        eval_n = min(512, max(32, len(self.sentences) // 100)) if len(self.sentences) >= 64 else len(self.sentences)
        self.eval_sentences = self.sentences[:eval_n]
        self.sentences = self.sentences[eval_n:] or self.eval_sentences[:]
        if self.story_sentences:
            # Blend eval proportionally so metrics reflect both pools.
            n_story_eval = max(8, int(round(len(self.eval_sentences) * self.story_weight)))
            story_eval = self.story_sentences[:n_story_eval]
            self.story_sentences = self.story_sentences[n_story_eval:] or story_eval[:]
            keep = len(self.eval_sentences) - n_story_eval
            mixed = story_eval + self.eval_sentences[: max(keep, 8)]
            self.eval_sentences = mixed[: max(len(self.eval_sentences), len(mixed))]
        self._pos = 0
        self._story_pos = 0
        log("DATA", f"Phase0 sentences loaded: train={len(self.sentences)} "
                    f"stories={len(self.story_sentences)} (weight={self.story_weight}) "
                    f"eval={len(self.eval_sentences)}")

    def _substrate_safe(self, text: str) -> str:
        alphabet = set(default_alphabet())
        out = []
        for ch in text:
            if ch in alphabet:
                out.append(ch)
            elif ch.isspace():
                out.append(" ")
            # else drop
        return re.sub(r"\s+", " ", "".join(out)).strip()

    def _split_sentences(self, text: str) -> list[str]:
        # Simple sentence split on . ! ? followed by space or end
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [self._substrate_safe(p) for p in parts if len(self._substrate_safe(p)) >= 8]

    def _load(self, curriculum_dir: str | None, containers_path: str | None) -> None:
        # 1. Try curriculum_v1 text families
        if curriculum_dir and os.path.isdir(curriculum_dir):
            for family in ("field_surfacing_v1", "retrieval_qa_v1", "procedure_next_step_v1"):
                path = pathlib.Path(curriculum_dir) / f"{family}.jsonl"
                if path.exists():
                    self._load_jsonl(path, text_keys=["structured_knowledge", "context_slots_text", "answer"])

        # 2. Stream dormant containers
        if containers_path and os.path.exists(containers_path):
            self._load_containers(containers_path)

        # 3. Fallback: a tiny synthetic corpus so smoke always has data
        if not self.sentences:
            self.sentences = [
                "the quick brown fox jumps over the lazy dog",
                "axon is a token free slot based agent",
                "the shared field is made of fixed width slots",
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
                if i >= 50_000:  # cap streaming
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
        # Split sentence into context and continuation (supervise second half)
        words = sentence.split()
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
        if self.story_sentences and self.rng.random() < self.story_weight:
            if self._story_pos >= len(self.story_sentences):
                self.rng.shuffle(self.story_sentences)
                self._story_pos = 0
            sentence = self.story_sentences[self._story_pos]
            self._story_pos += 1
        else:
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
        source = self.eval_sentences or self.sentences
        history: list[str] = []
        examples: list[dict[str, Any]] = []
        for i in range(n):
            history_text = "\n".join(history[-self.history_turns:])
            ex = self._example_from_sentence(source[i % len(source)], history_text=history_text)
            examples.append(ex)
            history.append(f"user {ex['user_input']}\naxon {ex['answer']}")
        return examples


# ---------------------------------------------------------------------------
# Curriculum: RECALL drills
# ---------------------------------------------------------------------------
class RecallCurriculum:
    """Two-tick recall pairs: Tick A stores, Tick B recalls from soul."""

    def __init__(self, path: str | None = None, n_synthetic: int = 4000):
        self.lessons: list[dict] = []
        if path and os.path.exists(path):
            self._load_jsonl(path)
        if not self.lessons:
            self.lessons = self._synthetic(n_synthetic)
        self.eval_lessons = self.lessons[: max(1, len(self.lessons) // 20)]
        self.train_lessons = self.lessons[max(1, len(self.lessons) // 20) :]
        self.rng = random.Random(42)
        self.rng.shuffle(self.train_lessons)
        self._pos = 0
        log("DATA", f"Recall lessons loaded: train={len(self.train_lessons)} eval={len(self.eval_lessons)}")

    def _load_jsonl(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "pairs" in obj:
                    self.lessons.append(obj)

    def _synthetic(self, n: int) -> list[dict]:
        hubs = {
            "axon": ["window", "stt", "voice", "ssh", "core"],
            "jeff": ["diary", "schedule", "advisor", "roundtable"],
            "python": ["fastapi", "torch", "numpy", "paramiko"],
            "security": ["mitre", "recon", "bypass", "payload"],
        }
        rng = random.Random(0)
        hub_list = list(hubs.keys())
        out = []
        for i in range(n):
            k = rng.choice([1, 1, 2, 2, 3])
            cs = rng.sample(hub_list, k)
            pairs = [[rng.choice(hubs[c]), c] for c in cs]  # [entity, hub]
            direction = "hub2ent" if rng.random() < 0.7 else "ent2hub"
            out.append({"lesson_id": f"syn_{i:05d}", "pairs": pairs, "direction": direction})
        return out

    @staticmethod
    def orient(lesson: dict) -> tuple[list[tuple[str, str]], list[str]]:
        pairs = lesson["pairs"]
        if lesson.get("direction", "hub2ent") == "hub2ent":
            bp = [(e, h) for e, h in pairs]
        else:
            bp = [(h, e) for e, h in pairs]
        answers = [a for a, _ in bp]
        return bp, answers

    def next(self) -> dict[str, Any]:
        if self._pos >= len(self.train_lessons):
            self.rng.shuffle(self.train_lessons)
            self._pos = 0
        lesson = self.train_lessons[self._pos]
        self._pos += 1
        bp, answers = self.orient(lesson)
        cues = [c for _, c in bp]
        answer = " ".join(answers)
        return {
            "mode": "recall",
            "lesson": lesson,
            "context": " ".join(cues),
            "answer": answer,
            "active_regions": {"conversation_history", "response_draft"},
        }


# ---------------------------------------------------------------------------
# Rolling checkpoints
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Rolling-3 checkpoints with pointer.json + checkpoint_done.json sentinel."""

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
    ) -> pathlib.Path:
        active = f"ckpt_{self._slot}.pt"
        self._slot = (self._slot + 1) % self.keep
        path = self.run_dir / active
        payload = {
            "step": step,
            "cfg": cfg.to_dict(),
            "soul_cfg": soul_cfg.to_dict(),
            "core_state": core.state_dict(),
            "soul_mgr_state": soul_mgr.state_dict(),
            "soul_state": soul_mgr.state.to_saveable(),
        }
        torch.save(payload, path)
        (self.run_dir / "pointer.json").write_text(
            json.dumps({"step": step, "active": active}), encoding="utf-8"
        )
        (self.run_dir / "checkpoint_done.json").write_text(
            json.dumps({"step": step, "time": time.time()}), encoding="utf-8"
        )
        return path


# ---------------------------------------------------------------------------
# Counterfactual probe harness
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_cf_probe(
    core: AxonCore,
    soul_mgr: SoulManagerV2,
    builder: SlotFieldBuilder,
    lessons: list[dict],
    device: torch.device,
    dtype: torch.dtype,
    n: int = 24,
) -> dict[str, Any]:
    """Run zero/swapped/irrelevant-soul and field-leak probes on recall lessons.

    Returns counts and a verdict.  A healthy core should:
      - answer correctly with the right soul (orig > chance)
      - fail with zero soul (zero_wrong high or zero flip)
      - flip with a swapped soul (swap_flip high)
      - ignore an irrelevant soul (irrelevant != orig)
      - not leak the answer from the field (field_leak low)
    """
    core.eval()
    table = get_char_prototype_table(core.cfg.d_model)
    orig_ok = swap_flip = zero_fail = irrelevant_diff = field_leak = tot = 0

    for i, lesson in enumerate(lessons[:n]):
        bp, answers = RecallCurriculum.orient(lesson)
        answer = " ".join(answers)

        def run_with_soul(soul_state_backup=None, mask_regions: set[str] | None = None):
            backup = soul_mgr.state
            if soul_state_backup is not None:
                soul_mgr.state = soul_state_backup
            try:
                ctx = " ".join([c for _, c in bp])
                built = builder.build({"conversation_history": ctx}, answer_text=answer,
                                      active_regions=({"conversation_history"} if mask_regions else {"conversation_history", "response_draft"}))
                if mask_regions:
                    built["mask"][:, builder.layout.region_slice("conversation_history")] = False
                    built["mask"][:, builder.layout.region_slice("response_draft")] = True
                soul, soul_mask = soul_mgr.inhale()
                out = core.forward_slot(
                    built["field_d"], soul.to(dtype), built["mask"],
                    soul_mask=soul_mask.to(device), response_draft_slice=builder.draft_slice,
                )
                logits = table.logits(out["response_delta_chars"])
                pred_idx = logits.argmax(dim=-1)[0].cpu().tolist()
                pred = "".join(table.chars[idx] if idx >= 0 else "" for idx in pred_idx)
                return pred, out
            finally:
                soul_mgr.state = backup

        # Correct soul: run tick A to seed it, then tick B recall
        soul_mgr.zero_soul()
        built_a = builder.build({"conversation_history": " ".join([c for _, c in bp])},
                                answer_text=answer,
                                active_regions={"conversation_history", "response_draft"})
        soul_a, mask_a = soul_mgr.inhale()
        out_a = core.forward_slot(built_a["field_d"], soul_a.to(dtype), built_a["mask"],
                                  soul_mask=mask_a.to(device), response_draft_slice=builder.draft_slice)
        soul_mgr.exhale_after_answer(out_a["field"].detach())

        pred_orig, _ = run_with_soul()
        pred_orig = pred_orig[: len(answer)]
        orig_ok += int(pred_orig.strip() == answer.strip())

        # Swapped soul (next lesson's content)
        j = (i + 1) % len(lessons[:n])
        bp_o, _ = RecallCurriculum.orient(lessons[j])
        soul_mgr.zero_soul()
        built_o = builder.build({"conversation_history": " ".join([c for _, c in bp_o])},
                                answer_text=" ".join([a for a, _ in bp_o]),
                                active_regions={"conversation_history", "response_draft"})
        soul_o, mask_o = soul_mgr.inhale()
        out_o = core.forward_slot(built_o["field_d"], soul_o.to(dtype), built_o["mask"],
                                  soul_mask=mask_o.to(device), response_draft_slice=builder.draft_slice)
        soul_mgr.exhale_after_answer(out_o["field"].detach())
        pred_swap, _ = run_with_soul()
        pred_swap = pred_swap[: len(answer)]
        swap_flip += int(pred_swap.strip() != pred_orig.strip())

        # Zero soul
        soul_mgr.zero_soul()
        pred_zero, _ = run_with_soul()
        pred_zero = pred_zero[: len(answer)]
        zero_fail += int(pred_zero.strip() != answer.strip())

        # Irrelevant soul: same as zero for short probes, but could load random
        pred_irr, _ = run_with_soul()
        pred_irr = pred_irr[: len(answer)]
        irrelevant_diff += int(pred_irr.strip() != pred_orig.strip())

        # Field leak: mask the conversation_history, answer should not come from field
        pred_leak, _ = run_with_soul(mask_regions={"conversation_history"})
        pred_leak = pred_leak[: len(answer)]
        field_leak += int(pred_leak.strip() == answer.strip())

        tot += 1

    core.train()
    t = max(1, tot)
    verdict = (
        "SOUL_IS_READ"
        if (orig_ok / t > 0.2 and swap_flip / t > 0.5 and zero_fail / t > 0.5)
        else "SOUL_NOT_READ"
    )
    return {
        "orig_ok": orig_ok,
        "swap_flip": swap_flip,
        "zero_fail": zero_fail,
        "irrelevant_diff": irrelevant_diff,
        "field_leak": field_leak,
        "tot": tot,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    core: AxonCore,
    soul_mgr: SoulManagerV2,
    builder: SlotFieldBuilder,
    examples: list[dict],
    device: torch.device,
    dtype: torch.dtype,
    render_chars: int = 0,
    draft_mode: str = "blank",
) -> dict[str, Any]:
    """Evaluate exact-fill and token accuracy on a list of examples."""
    core.eval()
    table = get_char_prototype_table(core.cfg.d_model)
    exact = chars_total = chars_correct = 0
    all_logits: list[torch.Tensor] = []
    pred_chars_all: list[str] = []
    field_losses: list[float] = []
    readonly_rms: list[float] = []
    response_rms: list[float] = []
    samples: list[dict[str, str]] = []
    for ex in examples:
        ctx = ex.get("context", "")
        history = ex.get("conversation_history", "")
        user_input = ex.get("user_input", ctx)
        ans = ex.get("answer", "")
        active = ex.get("active_regions", {"conversation_history", "user_input", "response_draft"})
        draft_text = draft_seed_for_mode(ans, draft_mode)
        built = builder.build(
            {"conversation_history": history, "user_input": user_input},
            answer_text=ans,
            draft_text=draft_text,
            active_regions=active,
        )
        soul, soul_mask = soul_mgr.inhale()
        out = core.forward_slot(
            built["field_d"], soul.to(dtype), built["mask"],
            soul_mask=soul_mask.to(device), response_draft_slice=builder.draft_slice,
        )
        if "field_delta" in out:
            per_slot = ((out["field_delta"] - built["field_delta_targets_d"]) ** 2).mean(dim=-1)
            weights = built["field_delta_weights"].clamp_min(0.0)
            denom = weights.sum().clamp_min(1.0)
            field_losses.append(float(((per_slot * weights).sum() / denom).detach().cpu().item()))
            draft_sl = builder.layout.region_slice("response_draft")
            readonly_mask = weights.clone()
            readonly_mask[:, draft_sl] = 0.0
            readonly_denom = readonly_mask.sum().clamp_min(1.0)
            readonly_rms.append(float(torch.sqrt(((out["field_delta"] ** 2).mean(dim=-1) * readonly_mask).sum() / readonly_denom).detach().cpu().item()))
            response_rms.append(float(torch.sqrt((out["field_delta"][:, draft_sl, :] ** 2).mean()).detach().cpu().item()))
        logits = table.logits(out["response_delta_chars"])  # (1, L, C)
        all_logits.append(logits[0].cpu())
        pred_idx = logits.argmax(dim=-1)[0].cpu().tolist()
        pred = "".join(table.chars[idx] for idx in pred_idx)
        pred = pred[: len(ans)]
        pred_chars_all.extend(pred)
        if len(samples) < 3:
            sample = {
                "history": history[:96],
                "user_input": user_input[:96],
                "draft_seed": draft_text[:96],
                "target": ans[:96],
                "pred": pred[:96],
            }
            if render_chars > 0:
                sample["field"] = " | ".join(
                    render_field_snapshot(built["field_np"], builder.layout, set(active), max_chars=render_chars)
                )
            samples.append(sample)
        exact += int(pred.strip() == ans.strip())
        for a, b in zip(pred, ans):
            chars_total += 1
            chars_correct += int(a == b)
    core.train()
    t = max(1, len(examples))
    # Collapse tripwire: variance of argmax distribution across eval batch
    collapse_var = 0.0
    entropy = 0.0
    if all_logits:
        probs = torch.stack([F.softmax(l, dim=-1) for l in all_logits])
        collapse_var = float(probs.mean(dim=0).var().item())
        entropy = float((-(probs * (probs.clamp_min(1e-9).log())).sum(dim=-1)).mean().item())
    pred_counts = Counter(pred_chars_all)
    pred_total = max(1, len(pred_chars_all))
    top_char_frac = max(pred_counts.values(), default=0) / pred_total
    return {
        "exact_fill": exact / t,
        "char_acc": chars_correct / max(1, chars_total),
        "collapse_var": collapse_var,
        "pred_entropy": entropy,
        "pred_unique": len(pred_counts),
        "pred_top_frac": top_char_frac,
        "field_delta_loss": float(np.mean(field_losses)) if field_losses else 0.0,
        "readonly_delta_rms": float(np.mean(readonly_rms)) if readonly_rms else 0.0,
        "response_delta_rms": float(np.mean(response_rms)) if response_rms else 0.0,
        "samples": samples,
        "n": t,
    }


# ---------------------------------------------------------------------------
# Char-slot threshold (2026-07-04, probe-verified: training/char_slot_probe.py)
#
# The core attends the frozen 16D character substrate DIRECTLY — one slot per
# character — instead of a lossy 8192->d_model summary.  Text is written back
# by a per-position head and snapped to the frozen LetterBank.  A/B verdict:
# per-slot COPY char_acc=1.000 by step 500 (157k params, CPU) vs the pooled
# head's 0.485 plateau == the 51M-param GPU-run ceiling.  Position must
# survive the threshold; this path guarantees it end to end.
# ---------------------------------------------------------------------------
CHARSLOT_REGION_HISTORY = 0
CHARSLOT_REGION_USER = 1
CHARSLOT_REGION_RESPONSE = 2


class CharSlotFieldBuilder:
    """Char-granular field: history + user_input + response_draft, one frozen
    16D substrate slot per character.  No 8192D packing, no lossy adapter."""

    def __init__(self, history_chars: int, user_chars: int, resp_chars: int,
                 device: torch.device, dtype: torch.dtype):
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
        region[history_chars: history_chars + user_chars] = CHARSLOT_REGION_USER
        region[self.resp_slice] = CHARSLOT_REGION_RESPONSE
        self._region = torch.from_numpy(region).unsqueeze(0).to(device)

    def _write_block(self, out: np.ndarray, text: str, offset: int, width: int) -> None:
        for i, ch in enumerate(text[:width]):
            out[offset + i] = char_to_slot(ch if ch in self.char_index else " ")

    def _char_targets(self, answer: str) -> np.ndarray:
        t = np.full((self.resp_chars,), self.empty_index, dtype=np.int64)
        for i, ch in enumerate(answer[: self.resp_chars]):
            t[i] = self.char_index.get(ch, self.char_index[" "])
        return t

    def build(self, history: str, user_input: str, answer: str, draft_text: str) -> dict[str, torch.Tensor]:
        field16 = np.zeros((self.n_slots, SUBSTRATE_SLOT_DIM), dtype=np.float32)
        # History keeps its most recent tail — that is the useful context.
        self._write_block(field16, history[-self.history_chars:], 0, self.history_chars)
        self._write_block(field16, user_input, self.history_chars, self.user_chars)
        self._write_block(field16, draft_text, self.resp_slice.start, self.resp_chars)
        return {
            "field16": torch.from_numpy(field16).unsqueeze(0).to(self.device, self.dtype),
            "region": self._region,
            "targets": torch.from_numpy(self._char_targets(answer)).unsqueeze(0).to(self.device),
        }

    def decode(self, logits: torch.Tensor, n_chars: int) -> str:
        idx = logits.argmax(dim=-1)[0].tolist()
        return "".join("" if i == self.empty_index else self.bank.chars[i]
                       for i in idx[:n_chars])


@torch.no_grad()
def evaluate_charslot(
    core: AxonCore,
    soul_mgr: SoulManagerV2,
    builder: CharSlotFieldBuilder,
    examples: list[dict],
    draft_mode: str = "blank",
) -> dict[str, Any]:
    """Same metric family as evaluate(): exact_fill/char_acc + collapse stats."""
    core.eval()
    exact = chars_total = chars_correct = 0
    all_logits: list[torch.Tensor] = []
    pred_chars_all: list[str] = []
    samples: list[dict[str, str]] = []
    for ex in examples:
        ans = ex.get("answer", "")
        user_input = ex.get("user_input", ex.get("context", ""))
        history = ex.get("conversation_history", "")
        draft_text = draft_seed_for_mode(ans, draft_mode)
        built = builder.build(history, user_input, ans, draft_text)
        soul, soul_mask = soul_mgr.inhale()
        out = core.forward_charslot(
            built["field16"], built["region"], soul.to(builder.dtype),
            soul_mask=soul_mask.to(builder.device), response_slice=builder.resp_slice,
        )
        logits = core.charslot_logits(out["response_delta_16"], builder.bank_unit)
        all_logits.append(logits[0].float().cpu())
        pred = builder.decode(logits, len(ans))
        pred_chars_all.extend(pred)
        if len(samples) < 3:
            samples.append({
                "history": history[-96:],
                "user_input": user_input[:96],
                "draft_seed": draft_text[:96],
                "target": ans[:96],
                "pred": pred[:96],
            })
        exact += int(pred.strip() == ans.strip())
        for a, b in zip(pred, ans):
            chars_total += 1
            chars_correct += int(a == b)
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
        "collapse_var": collapse_var,
        "pred_entropy": entropy,
        "pred_unique": len(pred_counts),
        "pred_top_frac": max(pred_counts.values(), default=0) / pred_total,
        "samples": samples,
        "n": t,
    }


def train_charslot(args: argparse.Namespace) -> None:
    """Train-as-you-live on the char-slot threshold.  Soul breathing, rolling
    checkpoints and the collapse tripwire all carry over from train();
    only the field->core interface differs.  Phase0 only for now (the recall
    drills still ride the 8192D builder)."""
    assert args.mode == "phase0", "charslot threshold currently supports --mode phase0 only"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if getattr(args, "fp16", False) else torch.float32

    from cores.core import CHAR_SLOT_DIM
    assert CHAR_SLOT_DIM == SUBSTRATE_SLOT_DIM, "core CHAR_SLOT_DIM must match substrate SLOT_DIM"

    d_model, n_layers, n_heads, ffn_dim = parse_core_cfg(args.core_cfg)
    builder = CharSlotFieldBuilder(
        args.history_chars, args.user_chars, args.max_response_chars, device, dtype)
    core_cfg = CoreConfig(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, ffn_dim=ffn_dim,
        dropout=args.dropout,
        soul_mode="act_reflect_v2",
        soul_rows=args.soul_rows,
        soul_hot_rows=args.soul_hot_rows,
        soul_write_mode="direct" if args.soul_hot_rows == 0 else "compartments",
        n_soul_compartments=args.n_soul_compartments,
        soul_gate_init=args.soul_gate_init,
        slot_mode=False,
        max_response_chars=args.max_response_chars,
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
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = core.load_state_dict(ckpt["core_state"], strict=False)
        if missing or unexpected:
            log("RESUME", f"state mismatch tolerated missing={len(missing)} unexpected={len(unexpected)}")
        if "soul_mgr_state" in ckpt:
            soul_mgr.load_state_dict(ckpt["soul_mgr_state"])
        if "soul_state" in ckpt:
            from cores.soul_v2 import SoulState
            soul_mgr.state = SoulState.from_saveable(ckpt["soul_state"], device, torch.float32)
        start_step = int(ckpt.get("step", 0) or 0)
        log("RESUME", f"loaded checkpoint from {args.resume} at step={start_step}")

    opt = torch.optim.AdamW(
        list(core.parameters()) + list(soul_mgr.parameters()),
        lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
    )

    log("BUILD", f"charslot core d={d_model} l={n_layers} h={n_heads} ffn={ffn_dim} "
                 f"slots={builder.n_slots} (hist={args.history_chars} user={args.user_chars} "
                 f"resp={args.max_response_chars}) "
                 f"params={sum(p.numel() for p in core.parameters()):,} "
                 f"soul_params={sum(p.numel() for p in soul_mgr.parameters()):,} "
                 f"device={device} dtype={dtype} grad_checkpoint={core.use_checkpoint}")

    phase0 = Phase0Curriculum(
        curriculum_dir=args.curriculum_dir,
        containers_path=args.containers_path,
        max_chars=args.max_response_chars,
        history_turns=args.history_turns,
        rng=random.Random(args.seed),
        text_corpus=args.text_corpus or None,
        text_corpus_weight=args.text_corpus_weight,
        text_corpus_max=args.text_corpus_max,
    )
    run_dir = pathlib.Path(args.run_dir)
    ckpt_mgr = CheckpointManager(run_dir, keep=3)

    # Joint draft curriculum from step 1 — the probe showed no teacher
    # schedule is needed when position survives the threshold.
    mode_weights = [float(x) for x in args.charslot_mode_weights.split(",")]
    assert len(mode_weights) == 3, "--charslot-mode-weights wants 'copy,partial,blank'"

    if args.smoke:
        args.steps = min(args.steps, 250)
        args.eval_every = min(args.eval_every, 250)
        args.checkpoint_every = 10 ** 9

    skipped_overlength = 0
    losses: list[float] = []
    t0 = time.time()
    step = start_step
    if start_step >= args.steps:
        log("DONE", f"checkpoint step {start_step} already reached target step {args.steps}")
        return

    for step in range(start_step + 1, args.steps + 1):
        ex = phase0.next()
        ans = ex.get("answer", "")
        if not ans:
            continue
        if len(ans) > args.max_response_chars:
            skipped_overlength += 1
            continue
        user_input = ex.get("user_input", ex.get("context", ""))[: args.user_chars]
        history = ex.get("conversation_history", "")
        draft_stage = random.choices(("copy", "partial", "blank"), weights=mode_weights)[0]
        draft_text = draft_seed_for_mode(ans, draft_stage)
        built = builder.build(history, user_input, ans, draft_text)

        # INHALE
        soul, soul_mask = soul_mgr.inhale()

        # ATTEND the 16D substrate directly + per-position RESPONSE DELTA
        out = core.forward_charslot(
            built["field16"], built["region"], soul.to(dtype),
            soul_mask=soul_mask.to(device), response_slice=builder.resp_slice,
        )

        # DECODE against the frozen LetterBank
        logits = core.charslot_logits(out["response_delta_16"], builder.bank_unit)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            built["targets"].reshape(-1),
        )

        # LEARN
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(soul_mgr.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.item()))

        # EXHALE (after answer; field detached from the response path)
        with torch.no_grad():
            soul_mgr.exhale_after_answer(out["field"].detach())
            if step % 10 == 0:
                soul_mgr.maybe_compress()
                soul_mgr.maybe_evict()

        if step % args.log_every == 0 or step == 1:
            trained_steps = step - start_step
            sps = trained_steps / max(1e-6, time.time() - t0)
            recent = np.mean(losses[-args.log_every:]) if losses else 0.0
            log("STEP", f"step={step}/{args.steps} mode=phase0 draft={draft_stage} "
                        f"loss={recent:.4f} char={float(loss.detach().cpu().item()):.4f} "
                        f"soul_active={soul_mgr.state.n_active()} skipped={skipped_overlength} {sps:.1f}it/s")

        if step % args.eval_every == 0:
            eval_examples = phase0.eval_examples(args.eval_n)
            stop_for_collapse = False
            for eval_mode in ("copy", "partial", "blank"):
                metrics = evaluate_charslot(core, soul_mgr, builder, eval_examples, draft_mode=eval_mode)
                tag = f"EVAL_{eval_mode.upper()}"
                log(tag, f"step={step} exact_fill={metrics['exact_fill']:.3f} "
                         f"char_acc={metrics['char_acc']:.3f} entropy={metrics['pred_entropy']:.3f} "
                         f"uniq={metrics['pred_unique']} top={metrics['pred_top_frac']:.3f} "
                         f"collapse_var={metrics['collapse_var']:.6f} n={metrics['n']}")
                for i, sample in enumerate(metrics.get("samples", [])[: int(getattr(args, "eval_samples", 0))]):
                    log("PRED", f"step={step} mode={eval_mode} sample={i} "
                                f"user_input={sample['user_input']!r} draft_seed={sample['draft_seed']!r} "
                                f"target={sample['target']!r} pred={sample['pred']!r}")
                if metrics["collapse_var"] < args.collapse_threshold and step > args.smoke_steps:
                    log("TRIPWIRE", f"collapse detected at step {step} mode={eval_mode} "
                                    f"(var={metrics['collapse_var']:.6f}); halting")
                    stop_for_collapse = True
                    break
            if stop_for_collapse:
                break

        if step % args.checkpoint_every == 0 and not args.smoke:
            path = ckpt_mgr.save(core, soul_mgr, core_cfg, soul_cfg, step)
            log("SAVE", f"checkpoint step={step} -> {path}")

    if not args.smoke:
        path = ckpt_mgr.save(core, soul_mgr, core_cfg, soul_cfg, step)
        log("SAVE", f"final checkpoint step={step} -> {path}")
    log("DONE", f"charslot training finished at step={step}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def parse_core_cfg(value: str) -> tuple[int, int, int, int]:
    if value in LOCKED_CORE_PRESETS:
        value = LOCKED_CORE_PRESETS[value]
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--core-cfg must be d,layers,heads,ffn or a preset A/B/C; got {value}")
    return tuple(parts)  # type: ignore[return-value]


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    # Doctrine: fp32 is the default everywhere. fp16 went NaN in this
    # project's history (SOT, Current Training Reality) and is opt-in only.
    dtype = torch.float16 if getattr(args, "fp16", False) else torch.float32

    d_model, n_layers, n_heads, ffn_dim = parse_core_cfg(args.core_cfg)
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
        slot_mode=True,
        max_response_chars=args.max_response_chars,
    )

    adapter = get_adapter(d_model)
    table = get_char_prototype_table(d_model)

    layout = default_layout()
    builder = SlotFieldBuilder(layout, adapter, table, device, dtype, max_response_chars=args.max_response_chars)

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
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        missing, unexpected = core.load_state_dict(ckpt["core_state"], strict=False)
        if missing or unexpected:
            log("RESUME", f"state mismatch tolerated missing={len(missing)} unexpected={len(unexpected)}")
        if "soul_mgr_state" in ckpt:
            soul_mgr.load_state_dict(ckpt["soul_mgr_state"])
        if "soul_state" in ckpt:
            from cores.soul_v2 import SoulState
            soul_mgr.state = SoulState.from_saveable(ckpt["soul_state"], device, torch.float32)
        start_step = int(ckpt.get("step", 0) or 0)
        log("RESUME", f"loaded checkpoint from {args.resume} at step={start_step}")

    opt = torch.optim.AdamW(
        list(core.parameters()) + list(soul_mgr.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    log("BUILD", f"core d={d_model} l={n_layers} h={n_heads} ffn={ffn_dim} "
                 f"params={sum(p.numel() for p in core.parameters()):,} "
                 f"soul_params={sum(p.numel() for p in soul_mgr.parameters()):,} "
                 f"device={device} dtype={dtype} grad_checkpoint={core.use_checkpoint}")

    # Curricula
    phase0 = Phase0Curriculum(
        curriculum_dir=args.curriculum_dir,
        containers_path=args.containers_path,
        max_chars=args.max_response_chars,
        history_turns=args.history_turns,
        rng=random.Random(args.seed),
    )
    recall = RecallCurriculum(args.recall_curriculum, n_synthetic=args.recall_synthetic)

    run_dir = pathlib.Path(args.run_dir)
    ckpt_mgr = CheckpointManager(run_dir, keep=3)

    # Smoke overrides
    if args.smoke:
        args.steps = min(args.steps, 250)
        args.eval_every = min(args.eval_every, 250)
        args.cf_probe_every = min(args.cf_probe_every, 250)
        args.checkpoint_every = 10 ** 9
        args.curriculum_dir = None
        args.containers_path = None

    skipped_overlength = 0
    step = start_step
    t0 = time.time()
    losses: list[float] = []

    if start_step >= args.steps:
        log("DONE", f"checkpoint step {start_step} already reached target step {args.steps}")
        return

    for step in range(start_step + 1, args.steps + 1):
        # Pick mode
        mode = args.mode
        if mode == "both":
            mode = "recall" if (step % 4 == 0) else "phase0"

        if mode == "phase0":
            ex = phase0.next()
        else:
            ex = recall.next()

        ctx = ex.get("context", "")
        ans = ex.get("answer", "")
        if len(ans) > args.max_response_chars:
            skipped_overlength += 1
            continue
        if not ans:
            continue

        history = ex.get("conversation_history", "")
        user_input = ex.get("user_input", ctx)
        active = ex.get("active_regions", {"conversation_history", "user_input", "response_draft"})
        draft_text, draft_stage = draft_seed_for_step(ans, step, args)
        built = builder.build(
            {"conversation_history": history, "user_input": user_input},
            answer_text=ans,
            draft_text=draft_text,
            active_regions=active,
        )

        # INHALE
        soul, soul_mask = soul_mgr.inhale()

        # ATTEND + RESPONSE DELTA
        out = core.forward_slot(
            built["field_d"], soul.to(dtype), built["mask"],
            soul_mask=soul_mask.to(device), response_draft_slice=builder.draft_slice,
        )

        # DECODE exact text payload + train full-field delta
        logits = table.logits(out["response_delta_chars"])  # (1, L, C)
        char_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            built["targets"].reshape(-1),
            ignore_index=-100,
        )
        per_slot_delta = ((out["field_delta"] - built["field_delta_targets_d"]) ** 2).mean(dim=-1)
        delta_weights = built["field_delta_weights"].clamp_min(0.0)
        delta_loss = (per_slot_delta * delta_weights).sum() / delta_weights.sum().clamp_min(1.0)
        loss = args.char_loss_weight * char_loss + args.field_delta_loss_weight * delta_loss

        # LEARN
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(core.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(soul_mgr.parameters(), args.grad_clip)
        opt.step()
        losses.append(float(loss.item()))

        # EXHALE (after answer; field is detached from response-delta path)
        with torch.no_grad():
            soul_mgr.exhale_after_answer(out["field"].detach())
            if step % 10 == 0:
                soul_mgr.maybe_compress()
                soul_mgr.maybe_evict()

        if step % args.log_every == 0 or step == 1:
            trained_steps = step - start_step
            sps = trained_steps / max(1e-6, time.time() - t0)
            recent = np.mean(losses[-args.log_every:]) if losses else 0.0
            log("STEP", f"step={step}/{args.steps} mode={mode} draft={draft_stage} loss={recent:.4f} "
                        f"char={float(char_loss.detach().cpu().item()):.4f} "
                        f"field_delta={float(delta_loss.detach().cpu().item()):.4f} "
                        f"soul_active={soul_mgr.state.n_active()} skipped={skipped_overlength} {sps:.1f}it/s")

        if step % args.eval_every == 0:
            eval_examples = recall.eval_lessons[: args.eval_n] if mode != "phase0" else phase0.eval_examples(args.eval_n)
            eval_modes = ("blank",) if mode != "phase0" else ("copy", "partial", "blank")
            stop_for_collapse = False
            for eval_mode in eval_modes:
                metrics = evaluate(
                    core,
                    soul_mgr,
                    builder,
                    eval_examples,
                    device,
                    dtype,
                    render_chars=(args.render_chars if getattr(args, "render_field", False) else 0),
                    draft_mode=eval_mode,
                )
                tag = f"EVAL_{eval_mode.upper()}"
                log(tag, f"step={step} exact_fill={metrics['exact_fill']:.3f} "
                         f"char_acc={metrics['char_acc']:.3f} entropy={metrics['pred_entropy']:.3f} "
                         f"uniq={metrics['pred_unique']} top={metrics['pred_top_frac']:.3f} "
                         f"field_delta={metrics['field_delta_loss']:.4f} "
                         f"readonly_rms={metrics['readonly_delta_rms']:.4f} "
                         f"response_rms={metrics['response_delta_rms']:.4f} "
                         f"collapse_var={metrics['collapse_var']:.6f} n={metrics['n']}")
                eval_samples = int(getattr(args, "eval_samples", 0))
                for i, sample in enumerate(metrics.get("samples", [])[:eval_samples]):
                    log("PRED", f"step={step} mode={eval_mode} sample={i} "
                                f"user_input={sample['user_input']!r} draft_seed={sample['draft_seed']!r} "
                                f"target={sample['target']!r} pred={sample['pred']!r}")
                    if sample.get("field"):
                        log("FIELD", f"step={step} mode={eval_mode} sample={i} attending={sample['field']}")
                if metrics["collapse_var"] < args.collapse_threshold and step > args.smoke_steps:
                    log("TRIPWIRE", f"collapse detected at step {step} mode={eval_mode} "
                                    f"(var={metrics['collapse_var']:.6f}); halting")
                    stop_for_collapse = True
                    break
            if stop_for_collapse:
                break

        if step % args.cf_probe_every == 0 and mode in ("recall", "both"):
            r = run_cf_probe(core, soul_mgr, builder, recall.eval_lessons, device, dtype, n=args.cf_probe_n)
            log("CF_PROBE", f"step={step} orig={r['orig_ok']}/{r['tot']} swap_flip={r['swap_flip']}/{r['tot']} "
                            f"zero_fail={r['zero_fail']}/{r['tot']} leak={r['field_leak']}/{r['tot']} {r['verdict']}")

        if step % args.checkpoint_every == 0 and not args.smoke:
            path = ckpt_mgr.save(core, soul_mgr, core_cfg, soul_cfg, step)
            log("SAVE", f"checkpoint step={step} -> {path}")

    # Final smoke gate
    if args.smoke:
        log("SMOKE", "running final smoke gate...")
        eval_examples = phase0.eval_examples(args.eval_n) + [
            {"context": " ".join([c for _, c in RecallCurriculum.orient(ex)[0]]),
             "answer": " ".join(RecallCurriculum.orient(ex)[1]),
             "active_regions": {"conversation_history", "response_draft"}}
            for ex in recall.eval_lessons[: args.eval_n]
        ]
        metrics = evaluate(core, soul_mgr, builder, eval_examples, device, dtype, draft_mode="copy")
        r = run_cf_probe(core, soul_mgr, builder, recall.eval_lessons, device, dtype, n=args.cf_probe_n)
        # Smoke gate: loss fell and exact-fill is above the constant-output floor.
        # Soul-read is a precision target; the harness runs but we do not assert
        # mastery in a short CPU smoke — that requires a longer recall curriculum.
        learned = metrics["char_acc"] >= 0.05 or metrics["exact_fill"] >= 0.05
        log("SMOKE", f"char_acc={metrics['char_acc']:.3f} exact_fill={metrics['exact_fill']:.3f} "
                     f"swap_flip={r['swap_flip']}/{r['tot']} zero_fail={r['zero_fail']}/{r['tot']} "
                     f"verdict={r['verdict']}")
        assert learned, "SMOKE FAIL: model did not learn (char_acc/exact_fill floor)"
        log("SMOKE", "PASS")

    log("DONE", f"finished at step {step}; skipped_overlength={skipped_overlength}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Axon slot-era train-as-you-live trainer")
    ap.add_argument("--mode", choices=["phase0", "recall", "both"], default="phase0")
    ap.add_argument("--threshold", choices=["slot", "charslot"], default="slot",
                    help="Field->core interface: 'slot' = frozen 8192->d adapter + pooled "
                         "char head (legacy); 'charslot' = attend the 16D substrate "
                         "directly, per-position decode (probe-verified)")
    ap.add_argument("--history-chars", type=int, default=256,
                    help="charslot: conversation_history region size in characters")
    ap.add_argument("--user-chars", type=int, default=64,
                    help="charslot: user_input region size in characters")
    ap.add_argument("--charslot-mode-weights", default="0.3,0.3,0.4",
                    help="charslot: sampling weights for copy,partial,blank drafts")
    ap.add_argument("--text-corpus", default="",
                    help="pre-cleaned one-sentence-per-line corpus (build_text_corpus.py) "
                         "mixed into phase0")
    ap.add_argument("--text-corpus-weight", type=float, default=0.7,
                    help="probability of sampling from --text-corpus vs memories")
    ap.add_argument("--text-corpus-max", type=int, default=1_000_000,
                    help="cap on corpus sentences loaded into RAM")
    ap.add_argument("--core-cfg", default="A", help="Preset A/B/C or d,layers,heads,ffn")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fp32", action="store_true", help="(kept for compat; float32 is already the doctrine default)")
    ap.add_argument("--fp16", action="store_true", help="Opt-in float16 (doctrine default is float32; fp16 has NaN history)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-checkpoint", action="store_true", help="Trade compute for lower activation memory")
    ap.add_argument("--char-loss-weight", type=float, default=1.0)
    ap.add_argument("--field-delta-loss-weight", type=float, default=0.25)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-steps", type=int, default=250)
    ap.add_argument("--draft-teacher-steps", type=int, default=10000)
    ap.add_argument("--draft-mask-ramp-steps", type=int, default=50000)
    ap.add_argument("--history-turns", type=int, default=10)
    ap.add_argument("--curriculum-dir", default="datasets/recovered/curriculum_v1")
    ap.add_argument("--containers-path", default="State/dormant/containers.jsonl")
    ap.add_argument("--recall-curriculum", default="")
    ap.add_argument("--recall-synthetic", type=int, default=2000)
    ap.add_argument("--run-dir", default="runs/slot")
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
    ap.add_argument("--render-field", action="store_true", help="Log readable active field regions for eval samples")
    ap.add_argument("--render-chars", type=int, default=160, help="Max chars per rendered field region")
    ap.add_argument("--cf-probe-every", type=int, default=500)
    ap.add_argument("--cf-probe-n", type=int, default=24)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--collapse-threshold", type=float, default=1e-6)
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.smoke:
        args.steps = min(args.steps, args.smoke_steps)
        args.core_cfg = "A"
        args.max_response_chars = 32
        args.hot_rows = 32
        args.warm_rows = 8
        args.cold_rows = 4
        args.eval_every = args.cf_probe_every = args.smoke_steps
        args.log_every = 10
        args.run_dir = "runs/slot_smoke"

    if args.threshold == "charslot":
        train_charslot(args)
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
