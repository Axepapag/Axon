#!/usr/bin/env python3
"""Axon v7 trainer v2.

This is a clean trainer for the existing v7 contract:

* frozen 16D letter substrate
* one shared field builder from field_contract.py
* ProjectionBank letter16/region16 heads
* AxonCore transformer with latent soul rows
* runtime-compatible .pt checkpoints

The important change from the deleted schoolhouse is the objective. Empty
buffer rows are no longer allowed to dominate the loss. The main write
signal is nonempty letter targets, especially visible response_draft rows.
Metrics report whether the model can actually write nonempty text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from core import AxonCore, CoreConfig
from field_contract import (
    BUFFERED_REGIONS,
    SHARED_ORDER,
    decode_region,
    materialize,
    selftest as field_selftest,
    target_field,
)
from heads import ProjectionBank, assert_inventory
from substrate import SLOT_DIM, default_alphabet, get_letter_bank, verify_substrate

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
CORE_DIR = ROOT / "Core"

ALLOWED_CHARS = set(default_alphabet())
EMPTY_TOKEN = "<empty>"


class TrainingInterrupted(Exception):
    """Raised from process signal handlers so the trainer can checkpoint."""

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


@dataclass
class EpisodeTick:
    input_event: str
    state_before: dict[str, str]
    state_after: dict[str, str]


@dataclass
class Episode:
    episode_id: str
    lesson: str
    ticks: list[EpisodeTick]


@dataclass
class DatasetReport:
    records: int = 0
    valid: int = 0
    repaired: int = 0
    rejected: int = 0
    reasons: Counter[str] | None = None
    lessons: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = Counter()
        if self.lessons is None:
            self.lessons = Counter()


def now() -> str:
    return time.strftime("%H:%M:%S")


def log(section: str, msg: str) -> None:
    print(f"[{now()}] [{section:8s}] {msg}", flush=True)


def clean_text(value: Any, *, repair: bool) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        if not repair:
            raise ValueError(f"expected string, got {type(value).__name__}")
        value = str(value)
    text = value.translate(_ASCII_REPAIRS)
    bad = sorted({c for c in text if c not in ALLOWED_CHARS})
    if bad and not repair:
        raise ValueError(f"banned chars: {''.join(bad)!r}")
    if repair and bad:
        text = "".join(c if c in ALLOWED_CHARS else " " for c in text)
    return text


def normalize_state(raw: Any, *, repair: bool) -> dict[str, str]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        if not repair:
            raise ValueError("state is not an object")
        raw = {}
    return {r: clean_text(raw.get(r, ""), repair=repair) for r in SHARED_ORDER}


def repair_growth(before: dict[str, str], after: dict[str, str]) -> bool:
    """Keep repaired examples inside the runtime field contract.

    response_draft has its standing buffer. Other regions cannot grow
    inside one tick in the current runtime; they can only rewrite/shrink
    existing rows unless a future curriculum uses ++ calls to grow them
    between ticks. For repaired records, impossible growth is clipped to
    the rows that exist before the tick.
    """
    changed = False
    for region in SHARED_ORDER:
        room = len(before[region]) + BUFFERED_REGIONS.get(region, 0)
        if len(after[region]) > room:
            after[region] = after[region][:room]
            changed = True
    return changed


def parse_episode(obj: Any, *, repair: bool = False) -> Episode:
    if not isinstance(obj, dict):
        raise ValueError("record is not an object")
    ticks_raw = obj.get("ticks")
    if not isinstance(ticks_raw, list) or not (1 <= len(ticks_raw) <= 8):
        n = len(ticks_raw) if isinstance(ticks_raw, list) else "?"
        raise ValueError(f"ticks must be a list of 1-8 (got {n})")

    ticks: list[EpisodeTick] = []
    for i, raw_tick in enumerate(ticks_raw):
        if not isinstance(raw_tick, dict):
            raise ValueError(f"tick {i}: not an object")
        before = normalize_state(raw_tick.get("state_before"), repair=repair)
        after = normalize_state(raw_tick.get("state_after"), repair=repair)
        if repair:
            repair_growth(before, after)
        # Contract check: the actual field builder must be able to lay
        # after-state targets onto before-state rows.
        _, _, row_map = materialize(before)
        target_field(after, row_map)
        ticks.append(EpisodeTick(
            input_event=clean_text(raw_tick.get("input_event", ""), repair=repair),
            state_before=before,
            state_after=after,
        ))

    return Episode(
        episode_id=str(obj.get("episode_id", "?")),
        lesson=str(obj.get("lesson", "?")),
        ticks=ticks,
    )


def parse_or_repair(obj: Any) -> tuple[Episode | None, bool, str]:
    try:
        return parse_episode(obj, repair=False), False, ""
    except Exception as strict_error:
        try:
            return parse_episode(obj, repair=True), True, str(strict_error)
        except Exception as repair_error:
            return None, False, str(repair_error)


def iter_records(path: Path):
    if path.is_dir():
        files = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in (".jsonl", ".json")
        )
    else:
        files = [path]
    for file_path in files:
        if file_path.suffix.lower() == ".json":
            try:
                obj = json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                yield file_path, None, f"invalid JSON: {e}"
                continue
            if isinstance(obj, list):
                for item in obj:
                    yield file_path, item, ""
            else:
                yield file_path, obj, ""
            continue
        with file_path.open(encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield file_path, json.loads(line), ""
                except json.JSONDecodeError as e:
                    yield file_path, None, f"{file_path.name}:{line_no}: invalid JSON: {e}"


def load_episodes(path: Path, *, max_episodes: int = 0) -> tuple[list[Episode], DatasetReport]:
    report = DatasetReport()
    episodes: list[Episode] = []
    for file_path, obj, error in iter_records(path):
        report.records += 1
        if error:
            report.rejected += 1
            report.reasons[error] += 1
            continue
        ep, repaired, reason = parse_or_repair(obj)
        if ep is None:
            report.rejected += 1
            report.reasons[reason or "rejected"] += 1
            continue
        episodes.append(ep)
        report.valid += 1
        report.repaired += int(repaired)
        report.lessons[ep.lesson] += 1
        if max_episodes and len(episodes) >= max_episodes:
            break
    return episodes, report


def resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def choose_device(requested: str) -> torch.device:
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def alphabet_tensor(device: torch.device) -> tuple[list[str], torch.Tensor, int, torch.Tensor]:
    bank = get_letter_bank()
    vecs = torch.tensor(bank.vecs, dtype=torch.float32, device=device)
    vecs = F.normalize(vecs, dim=-1)
    whitespace = torch.tensor([c.isspace() for c in bank.chars],
                              dtype=torch.bool, device=device)
    return bank.chars, vecs, bank.empty_index, whitespace


def make_initial_soul_state(
    cfg: CoreConfig,
    core_id: str,
    device: torch.device,
    seed: int,
    scale: float,
) -> torch.Tensor:
    """Birth a small nonzero private soul so day-one lessons cannot ignore it."""
    if int(cfg.soul_rows) <= 0:
        return torch.zeros((0, int(cfg.d_model)), device=device)
    material = (
        f"{core_id}|{cfg.d_model}|{cfg.n_heads}|{cfg.n_layers}|"
        f"{cfg.ffn_dim}|{cfg.soul_rows}|{seed}"
    ).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    soul_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(soul_seed)
    soul = torch.randn(
        (int(cfg.soul_rows), int(cfg.d_model)),
        generator=generator,
        dtype=torch.float32,
    )
    return (soul * float(scale)).to(device=device)


def coerce_soul_state(raw: Any, cfg: CoreConfig, device: torch.device) -> torch.Tensor | None:
    if raw is None:
        return None
    if not torch.is_tensor(raw):
        return None
    expected = (int(cfg.soul_rows), int(cfg.d_model))
    if tuple(raw.shape) != expected:
        return None
    return raw.detach().to(device=device, dtype=torch.float32).clone()


def sanitize_soul_state(soul: torch.Tensor, clip: float) -> torch.Tensor:
    soul = torch.nan_to_num(soul.detach(), nan=0.0, posinf=float(clip), neginf=-float(clip))
    if float(clip) > 0:
        soul = soul.clamp(min=-float(clip), max=float(clip))
    return soul


def classes_from_slots(slots16: torch.Tensor, alphabet_unit: torch.Tensor) -> torch.Tensor:
    unit = F.normalize(slots16.float(), dim=-1)
    return (unit @ alphabet_unit.T).argmax(dim=-1)


def decode_region_from_classes(
    classes: torch.Tensor,
    row_map: list[tuple[str, int]],
    region: str,
    chars: list[str],
    empty_idx: int,
) -> str:
    out: list[str] = []
    cls = classes.detach().cpu().tolist()
    for r, (name, i) in enumerate(row_map):
        if name == region and i >= 0:
            c = cls[r]
            if c != empty_idx:
                out.append(chars[c])
    return "".join(out)


def core_forward_ticks(
    core: AxonCore,
    field: torch.Tensor,
    soul: torch.Tensor,
    cfg: CoreConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    cold_ticks = max(0, int(cfg.n_ticks) - int(cfg.grad_ticks))
    for tick in range(int(cfg.n_ticks)):
        if tick < cold_ticks and core.training:
            with torch.no_grad():
                out = core.forward_with_soul(field, soul)
                field = out["field"].detach()
                soul = out["soul"].detach()
        else:
            out = core.forward_with_soul(field, soul)
            field = out["field"]
            soul = out["soul"]
    return field, soul


def soul_gate_floor_loss(core: AxonCore, args: argparse.Namespace) -> tuple[torch.Tensor | None, float]:
    gates: list[torch.Tensor] = []
    for name, param in core.named_parameters():
        if "soul_cross_gate" in name or "soul_reflect_gate" in name:
            gates.append(param.float().abs().reshape(()))
    if not gates:
        return None, 0.0
    gate_tensor = torch.stack(gates)
    gate_mean = float(gate_tensor.detach().mean().item())
    floor = float(args.soul_gate_floor)
    loss = F.relu(gate_tensor.new_tensor(floor) - gate_tensor).pow(2).mean()
    return loss, gate_mean


def tick_loss(
    tick: EpisodeTick,
    core: AxonCore,
    bank: ProjectionBank,
    cfg: CoreConfig,
    soul: torch.Tensor,
    alphabet_unit: torch.Tensor,
    chars: list[str],
    empty_idx: int,
    whitespace_lookup: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    *,
    render: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Counter[str], dict[str, str] | None]:
    slots_np, kinds, row_map = materialize(tick.state_before)
    target_np, letter_mask_np = target_field(tick.state_after, row_map)
    slots16 = torch.from_numpy(slots_np).to(device=device, dtype=torch.float32)
    targets16 = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)
    letter_mask = torch.from_numpy(letter_mask_np).to(device=device)

    field = bank.project_in(slots16, kinds)
    shared_h, next_soul_b = core_forward_ticks(
        core, field.unsqueeze(0), soul.unsqueeze(0), cfg)
    shared_h = shared_h.squeeze(0)
    next_soul = next_soul_b.squeeze(0)

    pred16 = bank.egress_letters(shared_h)
    logits = F.normalize(pred16.float(), dim=-1) @ alphabet_unit.T
    logits = logits / max(float(args.temperature), 1.0e-6)

    target_cls = classes_from_slots(targets16, alphabet_unit)
    before_cls = classes_from_slots(slots16, alphabet_unit)
    pred_cls = logits.argmax(dim=-1)

    target_nonempty = letter_mask & (target_cls != empty_idx)
    target_empty = letter_mask & (target_cls == empty_idx)
    before_nonempty = letter_mask & (before_cls != empty_idx)
    delete_empty = target_empty & before_nonempty
    slack_empty = target_empty & ~before_nonempty
    target_whitespace = target_nonempty & whitespace_lookup[target_cls]
    target_visible = target_nonempty & ~whitespace_lookup[target_cls]
    pred_nonempty = letter_mask & (pred_cls != empty_idx)
    pred_whitespace = pred_nonempty & whitespace_lookup[pred_cls]
    pred_visible = pred_nonempty & ~whitespace_lookup[pred_cls]

    draft_mask = torch.tensor(
        [name == "response_draft" and i >= 0 for name, i in row_map],
        device=device,
        dtype=torch.bool,
    )
    draft_empty = draft_mask & target_empty
    draft_tail_empty = draft_empty & ~before_nonempty

    weights = torch.zeros_like(logits[:, 0])
    # A literal space is not a visible response. Keep whitespace learnable for
    # formatting, but make visible glyphs carry the main writing signal.
    weights[target_visible] = float(args.visible_weight)
    weights[target_whitespace] = float(args.whitespace_weight)
    weights[delete_empty] = float(args.delete_empty_weight)
    weights[slack_empty] = float(args.slack_empty_weight)
    weights[draft_tail_empty] = float(args.draft_tail_empty_weight)
    weights[draft_mask & target_visible] *= float(args.draft_boost)

    active = letter_mask & (weights > 0)
    if active.any():
        ce = F.cross_entropy(logits[active], target_cls[active], reduction="none")
        loss = (ce * weights[active]).sum() / weights[active].sum().clamp_min(1.0)
    else:
        loss = logits.sum() * 0.0

    soul_aux = logits.sum() * 0.0
    soul_delta_rms = torch.zeros((), device=device)
    soul_div_rms = torch.zeros((), device=device)
    soul_ablation_rms = torch.zeros((), device=device)
    soul_gate_mean = 0.0
    soul_enabled = (
        bool(getattr(args, "soul_force", True))
        and int(cfg.soul_rows) > 0
        and cfg.soul_mode in ("act_reflect", "act_reflect_v2")
    )
    if soul_enabled:
        delta = next_soul.float() - soul.float()
        soul_delta_rms = delta.pow(2).mean().sqrt()
        soul_aux = soul_aux + float(args.soul_delta_weight) * F.relu(
            soul_delta_rms.new_tensor(float(args.soul_delta_min)) - soul_delta_rms
        ).pow(2)

        if next_soul.shape[0] > 1:
            soul_div_rms = next_soul.float().std(dim=0, unbiased=False).pow(2).mean().sqrt()
        else:
            soul_div_rms = next_soul.float().pow(2).mean().sqrt()
        soul_aux = soul_aux + float(args.soul_diversity_weight) * F.relu(
            soul_div_rms.new_tensor(float(args.soul_diversity_min)) - soul_div_rms
        ).pow(2)

        if float(args.soul_ablation_weight) > 0:
            with torch.no_grad():
                ablated_h_b, _ = core_forward_ticks(
                    core,
                    field.unsqueeze(0),
                    torch.zeros_like(soul).unsqueeze(0),
                    cfg,
                )
            ablated_h = ablated_h_b.squeeze(0)
            soul_ablation_rms = (shared_h.float() - ablated_h.float()).pow(2).mean().sqrt()
            soul_aux = soul_aux + float(args.soul_ablation_weight) * F.relu(
                soul_ablation_rms.new_tensor(float(args.soul_ablation_min)) - soul_ablation_rms
            ).pow(2)

        gate_loss, soul_gate_mean = soul_gate_floor_loss(core, args)
        if gate_loss is not None:
            soul_aux = soul_aux + float(args.soul_gate_weight) * gate_loss

        loss = loss + soul_aux

    stats: Counter[str] = Counter()
    stats["loss_count"] += 1
    stats["soul_count"] += 1
    stats["soul_delta_sum"] += float(soul_delta_rms.detach().item())
    stats["soul_div_sum"] += float(soul_div_rms.detach().item())
    stats["soul_ablation_sum"] += float(soul_ablation_rms.detach().item())
    stats["soul_gate_sum"] += float(soul_gate_mean)
    stats["soul_aux_loss_sum"] += float(soul_aux.detach().item())
    stats["letters"] += int(letter_mask.sum().item())
    stats["target_nonempty"] += int(target_nonempty.sum().item())
    stats["target_visible"] += int(target_visible.sum().item())
    stats["target_whitespace"] += int(target_whitespace.sum().item())
    stats["target_empty"] += int(target_empty.sum().item())
    stats["pred_empty"] += int((letter_mask & (pred_cls == empty_idx)).sum().item())
    stats["pred_visible"] += int(pred_visible.sum().item())
    stats["pred_whitespace"] += int(pred_whitespace.sum().item())
    stats["nonempty_correct"] += int((target_nonempty & (pred_cls == target_cls)).sum().item())
    stats["visible_correct"] += int((target_visible & (pred_cls == target_cls)).sum().item())
    stats["whitespace_correct"] += int((target_whitespace & (pred_cls == target_cls)).sum().item())
    stats["empty_correct"] += int((target_empty & (pred_cls == empty_idx)).sum().item())

    draft_nonempty = draft_mask & target_nonempty
    draft_visible = draft_mask & target_visible
    draft_whitespace = draft_mask & target_whitespace
    draft_overrun_nonempty = draft_tail_empty & pred_nonempty
    draft_overrun_visible = draft_tail_empty & pred_visible
    stats["draft_nonempty"] += int(draft_nonempty.sum().item())
    stats["draft_nonempty_correct"] += int(
        (draft_nonempty & (pred_cls == target_cls)).sum().item())
    stats["draft_visible"] += int(draft_visible.sum().item())
    stats["draft_visible_correct"] += int(
        (draft_visible & (pred_cls == target_cls)).sum().item())
    stats["draft_whitespace"] += int(draft_whitespace.sum().item())
    stats["draft_whitespace_correct"] += int(
        (draft_whitespace & (pred_cls == target_cls)).sum().item())
    stats["draft_tail_empty"] += int(draft_tail_empty.sum().item())
    stats["draft_tail_empty_correct"] += int(
        (draft_tail_empty & (pred_cls == empty_idx)).sum().item())
    stats["draft_overrun_nonempty"] += int(draft_overrun_nonempty.sum().item())
    stats["draft_overrun_visible"] += int(draft_overrun_visible.sum().item())
    draft_letters = draft_mask & letter_mask
    stats["draft_letters"] += int(draft_letters.sum().item())
    stats["draft_pred_nonempty"] += int(
        (draft_letters & pred_nonempty).sum().item())
    stats["draft_pred_visible"] += int(
        (draft_letters & pred_visible).sum().item())
    stats["draft_pred_whitespace"] += int(
        (draft_letters & pred_whitespace).sum().item())
    draft_pred_visible_classes = pred_cls[draft_letters & pred_visible]
    if draft_pred_visible_classes.numel() > 0:
        stats["draft_pred_visible_total"] += int(draft_pred_visible_classes.numel())
        stats["draft_pred_visible_peak"] += int(
            torch.bincount(
                draft_pred_visible_classes,
                minlength=len(chars),
            ).max().item())

    changed_nonempty = target_nonempty & (before_cls != target_cls)
    changed_visible = target_visible & (before_cls != target_cls)
    preserved_nonempty = target_nonempty & before_nonempty & (before_cls == target_cls)
    preserved_visible = target_visible & before_nonempty & (before_cls == target_cls)
    stats["changed_nonempty"] += int(changed_nonempty.sum().item())
    stats["changed_nonempty_correct"] += int(
        (changed_nonempty & (pred_cls == target_cls)).sum().item())
    stats["changed_visible"] += int(changed_visible.sum().item())
    stats["changed_visible_correct"] += int(
        (changed_visible & (pred_cls == target_cls)).sum().item())
    stats["preserved_nonempty"] += int(preserved_nonempty.sum().item())
    stats["preserved_nonempty_correct"] += int(
        (preserved_nonempty & (pred_cls == target_cls)).sum().item())
    stats["preserved_visible"] += int(preserved_visible.sum().item())
    stats["preserved_visible_correct"] += int(
        (preserved_visible & (pred_cls == target_cls)).sum().item())

    render_payload = None
    if render:
        pred_draft = decode_region_from_classes(
            pred_cls, row_map, "response_draft", chars, empty_idx)
        target_draft = decode_region(target_np, row_map, "response_draft")
        render_payload = {
            "target_draft": target_draft,
            "pred_draft": pred_draft,
            "blank_warning": (
                "1" if target_draft.strip() and not pred_draft.strip() else "0"
            ),
        }

    return loss, next_soul, stats, render_payload


def episode_loss(
    episode: Episode,
    core: AxonCore,
    bank: ProjectionBank,
    cfg: CoreConfig,
    alphabet_unit: torch.Tensor,
    chars: list[str],
    empty_idx: int,
    whitespace_lookup: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    *,
    initial_soul: torch.Tensor | None = None,
    render: bool = False,
) -> tuple[torch.Tensor, Counter[str], dict[str, str] | None, torch.Tensor]:
    if initial_soul is None:
        soul = torch.zeros(cfg.soul_rows, cfg.d_model, device=device)
    else:
        soul = initial_soul.detach().to(device=device, dtype=torch.float32)
    losses: list[torch.Tensor] = []
    stats: Counter[str] = Counter()
    render_payload = None
    for i, tick in enumerate(episode.ticks):
        loss, soul, tick_stats, maybe_render = tick_loss(
            tick, core, bank, cfg, soul, alphabet_unit, chars, empty_idx,
            whitespace_lookup,
            args, device, render=render and render_payload is None)
        losses.append(loss)
        stats.update(tick_stats)
        if maybe_render is not None:
            render_payload = maybe_render
        if int(args.bptt_window) > 0 and (i + 1) % int(args.bptt_window) == 0:
            soul = soul.detach()
    if not losses:
        return torch.zeros((), device=device, requires_grad=True), stats, render_payload, soul
    return torch.stack(losses).mean(), stats, render_payload, soul


class ModelEMA:
    def __init__(self, core: AxonCore, bank: ProjectionBank, decay: float):
        self.decay = float(decay)
        self.shadow = self._snapshot(core, bank)

    @staticmethod
    def _snapshot(core: AxonCore, bank: ProjectionBank) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for prefix, module in (("core", core), ("bank", bank)):
            for k, v in module.state_dict().items():
                out[f"{prefix}.{k}"] = v.detach().clone()
        return out

    def update(self, core: AxonCore, bank: ProjectionBank) -> None:
        current = self._snapshot(core, bank)
        for k, v in current.items():
            old = self.shadow.get(k)
            if old is None or not torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
            else:
                self.shadow[k] = old.mul(self.decay).add(v.detach(), alpha=1.0 - self.decay)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone() for k, v in state.items()}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def split(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        core_state: dict[str, torch.Tensor] = {}
        bank_state: dict[str, torch.Tensor] = {}
        for k, v in self.shadow.items():
            if k.startswith("core."):
                core_state[k[5:]] = v.detach().clone()
            elif k.startswith("bank."):
                bank_state[k[5:]] = v.detach().clone()
        return core_state, bank_state


def make_scheduler(optimizer: AdamW, *, start_step: int, total_steps: int, warmup_steps: int):
    def lr_lambda(step_idx: int) -> float:
        absolute_step = start_step + step_idx
        if warmup_steps > 0 and absolute_step < warmup_steps:
            return max(0.01, float(absolute_step + 1) / float(warmup_steps))
        remain = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (absolute_step - warmup_steps) / remain))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def checkpoint_payload(
    core: AxonCore,
    bank: ProjectionBank,
    cfg: CoreConfig,
    *,
    core_id: str,
    step: int,
    optimizer: AdamW | None,
    scheduler: LambdaLR | None,
    ema: ModelEMA | None,
    args: argparse.Namespace,
    metrics: dict[str, float] | None = None,
    soul_state: torch.Tensor | None = None,
    deploy: bool = False,
) -> dict[str, Any]:
    if deploy and ema is not None:
        core_state, bank_state = ema.split()
        ema_applied = True
    else:
        core_state = {k: v.detach().cpu() for k, v in core.state_dict().items()}
        bank_state = {k: v.detach().cpu() for k, v in bank.state_dict().items()}
        ema_applied = False

    payload: dict[str, Any] = {
        "kind": "trained",
        "contract": "axon7",
        "trainer": "trainer_v2",
        "core_id": core_id,
        "step": int(step),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cfg": cfg.to_dict(),
        "core_state": core_state,
        "projection_bank_state": bank_state,
        "head_keys": bank.head_keys(),
        "metrics": metrics or {},
    }
    if soul_state is not None:
        payload["soul_state"] = soul_state.detach().cpu().float().clone()
        payload["soul_state_kind"] = "trainer_hot_carried"
    if deploy:
        payload["ema_weights_applied"] = ema_applied
        payload["train_config"] = slim_args(args)
        return payload
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if ema is not None:
        payload["ema_state"] = ema.state_dict()
        payload["ema_decay"] = ema.decay
    payload["train_config"] = slim_args(args)
    return payload


def slim_args(args: argparse.Namespace) -> dict[str, Any]:
    keep = [
        "episodes", "steps", "lr", "accum", "weight_decay", "warmup_steps",
        "temperature", "nonempty_weight", "visible_weight",
        "whitespace_weight", "draft_tail_empty_weight",
        "delete_empty_weight", "slack_empty_weight",
        "draft_boost", "bptt_window", "ema",
        "ema_decay", "grad_checkpoint", "name", "core_id",
        "soul_force", "soul_delta_weight", "soul_delta_min",
        "soul_diversity_weight", "soul_diversity_min",
        "soul_ablation_weight", "soul_ablation_min",
        "soul_gate_weight", "soul_gate_floor",
        "soul_state_scale", "soul_state_clip",
        "draft_overrun_frac", "repeat_collapse_frac",
        "require_draft_visible_frac", "space_collapse_frac",
        "runtime_eval_suite", "runtime_eval_device", "runtime_eval_check",
        "require_visible_acc", "require_preserve_acc",
    ]
    return {k: getattr(args, k) for k in keep if hasattr(args, k)}


def save_checkpoint(
    path: Path,
    core: AxonCore,
    bank: ProjectionBank,
    cfg: CoreConfig,
    *,
    core_id: str,
    step: int,
    optimizer: AdamW | None,
    scheduler: LambdaLR | None,
    ema: ModelEMA | None,
    args: argparse.Namespace,
    metrics: dict[str, float],
    soul_state: torch.Tensor | None = None,
    deploy: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(
        core, bank, cfg, core_id=core_id, step=step, optimizer=optimizer,
        scheduler=scheduler, ema=ema, args=args, metrics=metrics,
        soul_state=soul_state, deploy=deploy)
    atomic_torch_save(payload, path)
    log("SAVE", str(path))


def safe_filename_part(raw: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(raw))


def checkpoint_filename(cfg: CoreConfig, step: int, suffix: str = "") -> str:
    safe_suffix = safe_filename_part(suffix)
    if safe_suffix and not safe_suffix.startswith("_"):
        safe_suffix = "_" + safe_suffix
    return (
        f"D{int(cfg.d_model)}_heads{int(cfg.n_heads)}_"
        f"layers{int(cfg.n_layers)}_soul{int(cfg.soul_rows)}_"
        f"{int(step):06d}steps{safe_suffix}.pt"
    )


def checkpoint_path(save_dir: Path, cfg: CoreConfig, step: int, suffix: str = "") -> Path:
    return save_dir / checkpoint_filename(cfg, step, suffix)


def unique_emergency_path(save_dir: Path, cfg: CoreConfig, reason: str, step: int) -> Path:
    base = checkpoint_path(save_dir, cfg, step, f"_{reason}")
    if not base.exists():
        return base
    for i in range(2, 1000):
        candidate = checkpoint_path(save_dir, cfg, step, f"_{reason}_{i}")
        if not candidate.exists():
            return candidate
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return checkpoint_path(save_dir, cfg, step, f"_{reason}_{stamp}")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        torch.save(payload, tmp)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def save_emergency_checkpoint(
    save_dir: Path,
    *,
    reason: str,
    exc: BaseException | None,
    current_step: int,
    last_completed_step: int,
    phase: str,
    core: AxonCore,
    bank: ProjectionBank,
    cfg: CoreConfig,
    core_id: str,
    optimizer: AdamW,
    scheduler: LambdaLR,
    ema: ModelEMA | None,
    args: argparse.Namespace,
    metrics: dict[str, float],
    soul_state: torch.Tensor | None,
) -> Path | None:
    """Best-effort salvage checkpoint for interruption/crash paths."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        path = unique_emergency_path(save_dir, cfg, reason, current_step)
        payload = checkpoint_payload(
            core, bank, cfg, core_id=core_id, step=int(current_step),
            optimizer=optimizer, scheduler=scheduler, ema=ema, args=args,
            metrics=metrics, soul_state=soul_state, deploy=False)
        payload.update({
            "emergency_checkpoint": True,
            "emergency_reason": reason,
            "emergency_phase": phase,
            "current_step": int(current_step),
            "last_completed_step": int(last_completed_step),
            "saved_after_exception": exc is not None,
        })
        if exc is not None:
            payload["exception_type"] = type(exc).__name__
            payload["exception_message"] = str(exc)
            payload["exception_traceback"] = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__))
        atomic_torch_save(payload, path)
        log("SAVE", f"emergency {reason}: {path}")
        return path
    except BaseException as save_error:
        log("FAIL", f"emergency checkpoint failed: "
            f"{type(save_error).__name__}: {save_error}")
        return None


def install_interruption_handlers():
    old_handlers: dict[int, Any] = {}

    def _raise_interruption(signum, _frame):
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = str(signum)
        raise TrainingInterrupted(f"received {sig_name}")

    for sig_name in ("SIGTERM", "SIGBREAK"):
        signum = getattr(signal, sig_name, None)
        if signum is None:
            continue
        try:
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _raise_interruption)
        except (OSError, ValueError, RuntimeError):
            pass
    return old_handlers


def restore_interruption_handlers(old_handlers: dict[int, Any]) -> None:
    for signum, handler in old_handlers.items():
        try:
            signal.signal(signum, handler)
        except (OSError, ValueError, RuntimeError):
            pass


def load_training_state(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[AxonCore, ProjectionBank, CoreConfig, str, int, dict[str, Any] | None, dict[str, Any] | None, torch.Tensor]:
    optimizer_state = None
    scheduler_state = None

    if args.resume:
        path = resolve_path(args.resume)
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = CoreConfig.from_dict(ck.get("cfg", {}))
        core_id = str(ck.get("core_id", args.core_id or path.stem))
        core = AxonCore(cfg).to(device)
        bank = ProjectionBank(cfg.d_model).to(device)
        assert_inventory(bank)
        core.load_state_dict(ck["core_state"])
        bank.load_state_dict(ck["projection_bank_state"])
        optimizer_state = ck.get("optimizer_state")
        scheduler_state = ck.get("scheduler_state")
        start_step = int(ck.get("step", 0))
        soul_state = coerce_soul_state(ck.get("soul_state"), cfg, device)
        if soul_state is None:
            soul_state = make_initial_soul_state(
                cfg, core_id, device, int(args.seed), float(args.soul_state_scale))
            log("LOAD", "checkpoint had no compatible soul_state; birthed fresh carried soul")
        else:
            soul_state = sanitize_soul_state(soul_state, float(args.soul_state_clip))
            log("LOAD", "restored carried soul_state")
        log("LOAD", f"resumed {path} at step {start_step}")
        return core, bank, cfg, core_id, start_step, optimizer_state, scheduler_state, soul_state

    if args.warm_start:
        path = resolve_path(args.warm_start)
        ck = torch.load(path, map_location=device, weights_only=False)
        if args.config:
            cfg_path = resolve_path(args.config)
            cfg = CoreConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
        else:
            cfg = CoreConfig.from_dict(ck.get("cfg", {}))
        core_id = args.core_id or f"{path.stem}_v7"
        core = AxonCore(cfg).to(device)
        bank = ProjectionBank(cfg.d_model).to(device)
        missing, unexpected = core.load_state_dict(ck.get("core_state", {}), strict=False)
        old_bank = ck.get("projection_bank_state", {})
        current_bank = bank.state_dict()
        compatible = {
            k: v for k, v in old_bank.items()
            if k in current_bank and tuple(v.shape) == tuple(current_bank[k].shape)
        }
        current_bank.update(compatible)
        bank.load_state_dict(current_bank)
        assert_inventory(bank)
        soul_state = make_initial_soul_state(
            cfg, core_id, device, int(args.seed), float(args.soul_state_scale))
        log("LOAD", f"warm-started {path}; core missing={len(missing)} unexpected={len(unexpected)} bank_keys={len(compatible)}")
        return core, bank, cfg, core_id, 0, None, None, soul_state

    if not args.config:
        raise SystemExit("--config is required for a new core")
    cfg_path = resolve_path(args.config)
    cfg = CoreConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))
    core_id = args.core_id or "core_v7"
    core = AxonCore(cfg).to(device)
    bank = ProjectionBank(cfg.d_model).to(device)
    assert_inventory(bank)
    soul_state = make_initial_soul_state(
        cfg, core_id, device, int(args.seed), float(args.soul_state_scale))
    log("LOAD", f"new core {core_id} cfg={cfg.to_dict()}")
    return core, bank, cfg, core_id, 0, None, None, soul_state


def metric_rates(stats: Counter[str], loss_value: float) -> dict[str, float]:
    letters = max(1, stats["letters"])
    nonempty = max(1, stats["target_nonempty"])
    visible = max(1, stats["target_visible"])
    whitespace = max(1, stats["target_whitespace"])
    empty = max(1, stats["target_empty"])
    draft_nonempty = max(1, stats["draft_nonempty"])
    draft_visible = max(1, stats["draft_visible"])
    draft_whitespace = max(1, stats["draft_whitespace"])
    draft_tail_empty = max(1, stats["draft_tail_empty"])
    changed_nonempty = max(1, stats["changed_nonempty"])
    changed_visible = max(1, stats["changed_visible"])
    preserved_nonempty = max(1, stats["preserved_nonempty"])
    preserved_visible = max(1, stats["preserved_visible"])
    soul_count = max(1, stats["soul_count"])
    return {
        "loss": float(loss_value),
        "nonempty_acc": stats["nonempty_correct"] / nonempty,
        "visible_acc": stats["visible_correct"] / visible,
        "whitespace_acc": stats["whitespace_correct"] / whitespace,
        "draft_nonempty_acc": stats["draft_nonempty_correct"] / draft_nonempty,
        "draft_visible_acc": stats["draft_visible_correct"] / draft_visible,
        "draft_whitespace_acc": stats["draft_whitespace_correct"] / draft_whitespace,
        "draft_tail_empty_acc": stats["draft_tail_empty_correct"] / draft_tail_empty,
        "draft_overrun_nonempty_frac": stats["draft_overrun_nonempty"] / draft_tail_empty,
        "draft_overrun_visible_frac": stats["draft_overrun_visible"] / draft_tail_empty,
        "draft_pred_nonempty_frac": stats["draft_pred_nonempty"] / max(1, stats["draft_letters"]),
        "draft_pred_visible_frac": stats["draft_pred_visible"] / max(1, stats["draft_letters"]),
        "draft_pred_whitespace_frac": stats["draft_pred_whitespace"] / max(1, stats["draft_letters"]),
        "draft_repeat_peak_frac": (
            stats["draft_pred_visible_peak"] / max(1, stats["draft_pred_visible_total"])
        ),
        "changed_nonempty_acc": stats["changed_nonempty_correct"] / changed_nonempty,
        "changed_visible_acc": stats["changed_visible_correct"] / changed_visible,
        "preserved_nonempty_acc": stats["preserved_nonempty_correct"] / preserved_nonempty,
        "preserved_visible_acc": stats["preserved_visible_correct"] / preserved_visible,
        "empty_acc": stats["empty_correct"] / empty,
        "pred_empty_frac": stats["pred_empty"] / letters,
        "pred_visible_frac": stats["pred_visible"] / letters,
        "pred_whitespace_frac": stats["pred_whitespace"] / letters,
        "target_nonempty_rows": float(stats["target_nonempty"]),
        "target_visible_rows": float(stats["target_visible"]),
        "draft_nonempty_rows": float(stats["draft_nonempty"]),
        "draft_visible_rows": float(stats["draft_visible"]),
        "draft_tail_empty_rows": float(stats["draft_tail_empty"]),
        "preserved_visible_rows": float(stats["preserved_visible"]),
        "changed_visible_rows": float(stats["changed_visible"]),
        "soul_delta": stats["soul_delta_sum"] / soul_count,
        "soul_div": stats["soul_div_sum"] / soul_count,
        "soul_ablation": stats["soul_ablation_sum"] / soul_count,
        "soul_gate": stats["soul_gate_sum"] / soul_count,
        "soul_aux_loss": stats["soul_aux_loss_sum"] / soul_count,
    }


@torch.no_grad()
def evaluate(
    episodes: list[Episode],
    core: AxonCore,
    bank: ProjectionBank,
    cfg: CoreConfig,
    alphabet_unit: torch.Tensor,
    chars: list[str],
    empty_idx: int,
    whitespace_lookup: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    *,
    limit: int,
    initial_soul: torch.Tensor | None = None,
) -> dict[str, float]:
    was_training = core.training
    core.eval()
    bank.eval()
    stats: Counter[str] = Counter()
    total_loss = 0.0
    n = min(limit, len(episodes))
    eval_soul = initial_soul.detach().clone() if initial_soul is not None else None
    for ep in episodes[:n]:
        loss, ep_stats, _, eval_soul = episode_loss(
            ep, core, bank, cfg, alphabet_unit, chars, empty_idx,
            whitespace_lookup,
            args, device, initial_soul=eval_soul, render=False)
        if eval_soul is not None:
            eval_soul = sanitize_soul_state(eval_soul, float(args.soul_state_clip))
        total_loss += float(loss.detach().item())
        stats.update(ep_stats)
    if was_training:
        core.train()
        bank.train()
    return metric_rates(stats, total_loss / max(1, n))


def format_metrics(step: int, steps: int, metrics: dict[str, float], lr: float) -> str:
    return (
        f"{step:>6d}/{steps:<6d} "
        f"loss={metrics['loss']:.4f} "
        f"ne_acc={metrics['nonempty_acc']:.3f} "
        f"vis_acc={metrics['visible_acc']:.3f} "
        f"draft_ne={metrics['draft_nonempty_acc']:.3f} "
        f"draft_vis={metrics['draft_visible_acc']:.3f} "
        f"tail_acc={metrics['draft_tail_empty_acc']:.3f} "
        f"draft_fill={metrics['draft_pred_nonempty_frac']:.3f} "
        f"draft_vfill={metrics['draft_pred_visible_frac']:.3f} "
        f"draft_space={metrics['draft_pred_whitespace_frac']:.3f} "
        f"overrun_vis={metrics['draft_overrun_visible_frac']:.3f} "
        f"repeat={metrics['draft_repeat_peak_frac']:.3f} "
        f"chg_ne={metrics['changed_nonempty_acc']:.3f} "
        f"chg_vis={metrics['changed_visible_acc']:.3f} "
        f"pres_vis={metrics['preserved_visible_acc']:.3f} "
        f"pred_empty={metrics['pred_empty_frac']:.3f} "
        f"pred_space={metrics['pred_whitespace_frac']:.3f} "
        f"ne_rows={int(metrics['target_nonempty_rows'])} "
        f"vis_rows={int(metrics['target_visible_rows'])} "
        f"pres_rows={int(metrics['preserved_visible_rows'])} "
        f"tail_rows={int(metrics['draft_tail_empty_rows'])} "
        f"soul_d={metrics['soul_delta']:.4f} "
        f"soul_div={metrics['soul_div']:.4f} "
        f"soul_ab={metrics['soul_ablation']:.4f} "
        f"soul_g={metrics['soul_gate']:.4f} "
        f"soul_aux={metrics['soul_aux_loss']:.2e} "
        f"lr={lr:.2e}"
    )


def run_runtime_eval_gate(
    checkpoint_path: Path,
    args: argparse.Namespace,
    train_device: torch.device,
) -> int:
    suites = list(getattr(args, "runtime_eval_suite", []) or [])
    if not suites:
        return 0
    eval_device = str(getattr(args, "runtime_eval_device", "") or "").strip()
    if not eval_device:
        eval_device = train_device.type
    cmd = [
        sys.executable,
        str(ROOT / "runtime_eval.py"),
        "--core",
        str(checkpoint_path),
        "--device",
        eval_device,
        "--check",
        str(getattr(args, "runtime_eval_check", "pre")),
    ]
    for suite in suites:
        cmd += ["--suite", suite]
    if getattr(args, "runtime_eval_no_fail", False):
        cmd.append("--no-fail")

    log("GATE", "runtime eval: " + " ".join(
        f'"{c}"' if " " in c else c for c in cmd))
    if train_device.type == "cuda":
        torch.cuda.empty_cache()
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode:
        log("FAIL", f"runtime eval failed with exit code {result.returncode}")
        return int(result.returncode)
    log("GATE", "runtime eval passed")
    return 0


def train(args: argparse.Namespace) -> int:
    if not verify_substrate(verbose=False):
        raise SystemExit("substrate geometry failed")
    if not field_selftest(verbose=False):
        raise SystemExit("field contract self-test failed")

    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    episodes_path = resolve_path(args.episodes)
    episodes, report = load_episodes(episodes_path, max_episodes=args.max_episodes)
    if not episodes:
        raise SystemExit(f"no trainable episodes under {episodes_path}")
    if args.overfit_first:
        episodes = episodes[: args.overfit_first]
        log("DATA", f"overfit mode: using first {len(episodes)} episode(s)")
    log("DATA", f"{report.valid}/{report.records} trainable "
        f"(clean {report.valid - report.repaired}, repaired {report.repaired}, "
        f"rejected {report.rejected}) from {episodes_path}")
    if report.reasons:
        for reason, n in report.reasons.most_common(3):
            log("DATA", f"reject {n}: {reason}")
    if report.lessons:
        mix = ", ".join(f"{k}:{v}" for k, v in report.lessons.most_common(6))
        log("DATA", f"lessons: {mix}")

    core, bank, cfg, core_id, start_step, opt_state, sched_state, carried_soul = load_training_state(args, device)
    core.use_checkpoint = bool(args.grad_checkpoint)
    core.train()
    bank.train()

    params = list(core.parameters()) + list(bank.parameters())
    optimizer = AdamW(
        params,
        lr=float(args.lr),
        betas=(0.9, 0.95),
        weight_decay=float(args.weight_decay),
    )
    if opt_state and not args.reset_optimizer:
        try:
            optimizer.load_state_dict(opt_state)
            log("LOAD", "optimizer state restored")
        except Exception as e:
            log("LOAD", f"optimizer reset: {type(e).__name__}: {e}")
    scheduler = make_scheduler(
        optimizer,
        start_step=start_step,
        total_steps=int(args.steps),
        warmup_steps=int(args.warmup_steps),
    )
    if sched_state and not args.reset_optimizer:
        try:
            scheduler.load_state_dict(sched_state)
            log("LOAD", "scheduler state restored")
        except Exception as e:
            log("LOAD", f"scheduler reset: {type(e).__name__}: {e}")

    ema = ModelEMA(core, bank, float(args.ema_decay)) if args.ema else None
    chars, alphabet_unit, empty_idx, whitespace_lookup = alphabet_tensor(device)

    rng = random.Random(args.seed)
    order = list(range(len(episodes)))
    seen = 0
    last_metrics: dict[str, float] = {}
    save_dir = resolve_path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    log("TRAIN", f"device={device} core_id={core_id} params="
        f"{sum(p.numel() for p in params):,} grad_checkpoint={core.use_checkpoint}")
    log("TRAIN", f"soul force={'on' if args.soul_force else 'off'} "
        f"carried_soul={tuple(carried_soul.shape)} "
        f"delta_w={args.soul_delta_weight} ablate_w={args.soul_ablation_weight}")
    log("TRAIN", f"loss weights: visible={args.visible_weight} "
        f"whitespace={args.whitespace_weight} nonempty_legacy={args.nonempty_weight} "
        f"draft_tail_empty={args.draft_tail_empty_weight} "
        f"draft_boost={args.draft_boost} delete_empty={args.delete_empty_weight} "
        f"slack_empty={args.slack_empty_weight}")

    current_step = int(start_step)
    last_completed_step = int(start_step)
    phase = "starting"
    old_signal_handlers = install_interruption_handlers()
    try:
        for step in range(start_step + 1, int(args.steps) + 1):
            current_step = int(step)
            phase = "zero_grad"
            optimizer.zero_grad(set_to_none=True)
            step_stats: Counter[str] = Counter()
            loss_total = 0.0
            render_payload = None
            for micro in range(int(args.accum)):
                phase = f"step {step} micro {micro + 1}/{int(args.accum)}"
                if args.overfit_first:
                    ep = episodes[seen % len(episodes)]
                    seen += 1
                else:
                    if seen % len(order) == 0:
                        rng.shuffle(order)
                    ep = episodes[order[seen % len(order)]]
                    seen += 1
                want_render = (step % int(args.render_every) == 0 and micro == 0)
                loss, stats, maybe_render, next_soul = episode_loss(
                    ep, core, bank, cfg, alphabet_unit, chars, empty_idx,
                    whitespace_lookup,
                    args, device, initial_soul=carried_soul, render=want_render)
                carried_soul = sanitize_soul_state(
                    next_soul, float(args.soul_state_clip))
                phase = f"backward step {step} micro {micro + 1}/{int(args.accum)}"
                (loss / int(args.accum)).backward()
                loss_total += float(loss.detach().item())
                step_stats.update(stats)
                if maybe_render is not None:
                    render_payload = maybe_render

            phase = f"optimizer step {step}"
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.grad_clip))
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(core, bank)
            last_completed_step = int(step)

            phase = f"metrics step {step}"
            last_metrics = metric_rates(step_stats, loss_total / max(1, int(args.accum)))
            if step % int(args.log_every) == 0 or step == 1:
                lr = optimizer.param_groups[0]["lr"]
                log("STEP", format_metrics(step, int(args.steps), last_metrics, lr))
                if (step >= int(args.blank_warn_after)
                        and last_metrics["target_nonempty_rows"] > 0
                        and last_metrics["pred_empty_frac"] >= float(args.blank_warn_frac)
                        and last_metrics["nonempty_acc"] == 0.0):
                    log("WARN", "blank-collapse signature: nonempty targets present, "
                        "predictions still mostly <empty>")

            if render_payload is not None:
                phase = f"render step {step}"
                log("RENDER", f"target: {render_payload['target_draft'][:180]!r}")
                log("RENDER", f"pred:   {render_payload['pred_draft'][:180]!r}")
                if render_payload["blank_warning"] == "1":
                    log("WARN", "target draft is nonempty but prediction decoded blank")

            if int(args.checkpoint_every) > 0 and step % int(args.checkpoint_every) == 0:
                phase = f"checkpoint step {step}"
                ck_path = checkpoint_path(save_dir, cfg, step)
                save_checkpoint(
                    ck_path, core, bank, cfg, core_id=core_id, step=step,
                    optimizer=optimizer, scheduler=scheduler, ema=ema, args=args,
                    metrics=last_metrics, soul_state=carried_soul, deploy=False)

        phase = "final eval"
        current_step = int(args.steps)
        eval_metrics = evaluate(
            episodes, core, bank, cfg, alphabet_unit, chars, empty_idx,
            whitespace_lookup,
            args, device, limit=int(args.eval_episodes),
            initial_soul=carried_soul)
        last_metrics = eval_metrics
        log("EVAL", format_metrics(int(args.steps), int(args.steps), eval_metrics,
                                   optimizer.param_groups[0]["lr"]))

        phase = "final checkpoint"
        final_path = checkpoint_path(save_dir, cfg, int(args.steps))
        save_checkpoint(
            final_path, core, bank, cfg, core_id=core_id, step=int(args.steps),
            optimizer=optimizer, scheduler=scheduler, ema=ema, args=args,
            metrics=eval_metrics, soul_state=carried_soul, deploy=False)

        if args.require_visible:
            if eval_metrics["draft_visible_acc"] < float(args.require_draft_acc):
                log("FAIL", "--require-visible failed: draft_visible_acc "
                    f"{eval_metrics['draft_visible_acc']:.3f} < "
                    f"{float(args.require_draft_acc):.3f}")
                return 2
            if eval_metrics["draft_pred_visible_frac"] < float(args.require_draft_visible_frac):
                log("FAIL", "--require-visible failed: draft visible fill "
                    f"{eval_metrics['draft_pred_visible_frac']:.3f} < "
                    f"{float(args.require_draft_visible_frac):.3f}")
                return 2
            if eval_metrics["draft_pred_whitespace_frac"] >= float(args.space_collapse_frac):
                log("FAIL", "--require-visible failed: draft predictions are mostly whitespace "
                    f"({eval_metrics['draft_pred_whitespace_frac']:.3f})")
                return 2
            if eval_metrics["draft_overrun_visible_frac"] >= float(args.draft_overrun_frac):
                log("FAIL", "--require-visible failed: draft visible overrun "
                    f"{eval_metrics['draft_overrun_visible_frac']:.3f} >= "
                    f"{float(args.draft_overrun_frac):.3f}")
                return 2
            if eval_metrics["draft_repeat_peak_frac"] >= float(args.repeat_collapse_frac):
                log("FAIL", "--require-visible failed: draft repeat-collapse "
                    f"{eval_metrics['draft_repeat_peak_frac']:.3f} >= "
                    f"{float(args.repeat_collapse_frac):.3f}")
                return 2
            if eval_metrics["pred_empty_frac"] >= float(args.blank_warn_frac):
                log("FAIL", "--require-visible failed: predictions are still mostly blank")
                return 2
        if float(args.require_visible_acc) > 0:
            if eval_metrics["visible_acc"] < float(args.require_visible_acc):
                log("FAIL", "--require-visible-acc failed: visible_acc "
                    f"{eval_metrics['visible_acc']:.3f} < "
                    f"{float(args.require_visible_acc):.3f}")
                return 2
        if float(args.require_preserve_acc) > 0:
            if eval_metrics["preserved_visible_acc"] < float(args.require_preserve_acc):
                log("FAIL", "--require-preserve-acc failed: preserved_visible_acc "
                    f"{eval_metrics['preserved_visible_acc']:.3f} < "
                    f"{float(args.require_preserve_acc):.3f}")
                return 2

        phase = "runtime eval gate"
        gate_rc = run_runtime_eval_gate(final_path, args, device)
        if gate_rc:
            return gate_rc

        if not args.no_deploy:
            phase = "deploy checkpoint"
            deploy_path = CORE_DIR / f"{core_id}.pt"
            save_checkpoint(
                deploy_path, core, bank, cfg, core_id=core_id, step=int(args.steps),
                optimizer=None, scheduler=None, ema=ema, args=args,
                metrics=eval_metrics, soul_state=carried_soul, deploy=True)
        return 0
    except (KeyboardInterrupt, TrainingInterrupted) as e:
        log("INTERRUPT", f"{type(e).__name__}: {e}")
        save_emergency_checkpoint(
            save_dir, reason="interrupt", exc=e, current_step=current_step,
            last_completed_step=last_completed_step, phase=phase,
            core=core, bank=bank, cfg=cfg, core_id=core_id,
            optimizer=optimizer, scheduler=scheduler, ema=ema,
            args=args, metrics=last_metrics, soul_state=carried_soul)
        return 130
    except BaseException as e:
        log("CRASH", f"{type(e).__name__}: {e}")
        save_emergency_checkpoint(
            save_dir, reason="crash", exc=e, current_step=current_step,
            last_completed_step=last_completed_step, phase=phase,
            core=core, bank=bank, cfg=cfg, core_id=core_id,
            optimizer=optimizer, scheduler=scheduler, ema=ema,
            args=args, metrics=last_metrics, soul_state=carried_soul)
        raise
    finally:
        restore_interruption_handlers(old_signal_handlers)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Axon v7 trainer v2")
    ap.add_argument("--episodes", required=True,
                    help="JSONL/JSON file or folder. Relative paths resolve from this repo.")
    ap.add_argument("--config", help="new-core CoreConfig JSON")
    ap.add_argument("--resume", help="resume a v7 checkpoint/core")
    ap.add_argument("--warm-start", help="best-effort warm start from a compatible older checkpoint")
    ap.add_argument("--core-id", default="")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1.0e-3)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--nonempty-weight", type=float, default=1.0)
    ap.add_argument("--visible-weight", type=float, default=2.0,
                    help="loss weight for non-whitespace visible target characters")
    ap.add_argument("--whitespace-weight", type=float, default=0.2,
                    help="loss weight for target spaces/newlines/tabs")
    ap.add_argument("--draft-tail-empty-weight", type=float, default=0.2,
                    help="loss weight for empty response_draft buffer rows after the target text")
    ap.add_argument("--delete-empty-weight", type=float, default=0.05)
    ap.add_argument("--slack-empty-weight", type=float, default=0.01)
    ap.add_argument("--draft-boost", type=float, default=2.0)
    ap.add_argument("--bptt-window", type=int, default=0,
                    help="0 = full episode BPTT (default; needed for the soul "
                         "to learn cross-tick recall/carry on short <=3-tick "
                         "episodes). Set >0 only for long episodes where the "
                         "graph must be truncated for memory.")
    ap.add_argument("--soul-force", action="store_true",
                    help="always include soul shaping/causality pressure")
    ap.add_argument("--no-soul-force", dest="soul_force", action="store_false")
    ap.set_defaults(soul_force=True)
    ap.add_argument("--soul-delta-weight", type=float, default=0.02,
                    help="gentle reward pressure for shaping soul each tick")
    ap.add_argument("--soul-delta-min", type=float, default=0.01,
                    help="minimum desired RMS soul change per tick")
    ap.add_argument("--soul-diversity-weight", type=float, default=0.01,
                    help="gentle pressure against identical/dead soul rows")
    ap.add_argument("--soul-diversity-min", type=float, default=0.002,
                    help="minimum desired row-diversity RMS for the soul")
    ap.add_argument("--soul-ablation-weight", type=float, default=0.01,
                    help="force the field action to depend measurably on soul")
    ap.add_argument("--soul-ablation-min", type=float, default=0.002,
                    help="minimum desired output change when soul is ablated")
    ap.add_argument("--soul-gate-weight", type=float, default=0.01,
                    help="prevent soul cross/reflect gates from collapsing to zero")
    ap.add_argument("--soul-gate-floor", type=float, default=0.02,
                    help="minimum absolute soul gate target")
    ap.add_argument("--soul-state-scale", type=float, default=0.02,
                    help="scale for deterministic birth soul on new/legacy checkpoints")
    ap.add_argument("--soul-state-clip", type=float, default=5.0,
                    help="absolute clamp for carried hot soul state after each lesson")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ema", action="store_true", help="track EMA shadow weights")
    ap.add_argument("--no-ema", dest="ema", action="store_false")
    ap.set_defaults(ema=False)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.set_defaults(grad_checkpoint=False)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--render-every", type=int, default=100)
    ap.add_argument("--eval-episodes", type=int, default=32)
    ap.add_argument("--blank-warn-after", type=int, default=10)
    ap.add_argument("--blank-warn-frac", type=float, default=0.98)
    ap.add_argument("--require-visible", action="store_true",
                    help="exit nonzero if final eval cannot write nonempty draft rows")
    ap.add_argument("--require-draft-acc", type=float, default=0.20,
                    help="minimum final draft_visible_acc when --require-visible is used")
    ap.add_argument("--require-draft-visible-frac", type=float, default=0.005,
                    help="minimum final visible glyph fraction in draft predictions")
    ap.add_argument("--space-collapse-frac", type=float, default=0.90,
                    help="fail --require-visible if draft predictions are this much whitespace")
    ap.add_argument("--draft-overrun-frac", type=float, default=0.35,
                    help="fail --require-visible if too many empty draft-tail rows are visible chars")
    ap.add_argument("--repeat-collapse-frac", type=float, default=0.85,
                    help="fail --require-visible if one visible glyph dominates draft predictions")
    ap.add_argument("--require-visible-acc", type=float, default=0.0,
                    help="if >0, fail final eval unless all-region visible_acc reaches this")
    ap.add_argument("--require-preserve-acc", type=float, default=0.0,
                    help="if >0, fail final eval unless unchanged visible rows copy accurately")
    ap.add_argument("--runtime-eval-suite", action="append",
                    choices=["core", "sigils", "repeat", "all"], default=[],
                    help="run exact runtime_eval.py suite after final checkpoint; repeatable")
    ap.add_argument("--runtime-eval-device", default="",
                    help="device for runtime eval gate; default = training device")
    ap.add_argument("--runtime-eval-check", choices=["pre", "post"], default="pre",
                    help="pre checks draft_before_exec before tool-call cleanup")
    ap.add_argument("--runtime-eval-no-fail", action="store_true",
                    help="run runtime eval but do not fail/deploy-block on failures")
    ap.add_argument("--overfit-first", type=int, default=0,
                    help="train repeatedly, cycling through the first N episodes")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--name", default="v7v2")
    ap.add_argument("--save-path", default=str(CHECKPOINT_DIR))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-deploy", action="store_true")
    ap.add_argument("--reset-optimizer", action="store_true")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
