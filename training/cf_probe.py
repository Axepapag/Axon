#!/usr/bin/env python3
"""Counterfactual soul-conditioning probe for Axon v7.

Takes a checkpoint, runs the store tick of hidden-letter episodes, then tests
whether the recall output is *conditioned on the stored soul* by:

  1. original soul -> should produce the original target
  2. partner's soul -> should produce the partner's target
  3. zero soul -> should NOT produce the original target

This is the binary "soul is being read" signal Hermes asked for.

Usage:
    python cf_probe.py --checkpoint checkpoints/w1/D128...pt --device cuda --episodes 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from substrate import SLOT_DIM, get_letter_bank, verify_substrate
from field_contract import SHARED_ORDER, materialize, decode_region, selftest as field_selftest
from heads import ProjectionBank, assert_inventory
from core import AxonCore, CoreConfig

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"

UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")
ALPH = UPPER + DIGITS


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def blank_state() -> dict[str, str]:
    return {r: "" for r in SHARED_ORDER}


def make_store_tick(ch: str) -> dict[str, Any]:
    before = blank_state()
    before["input_window"] = f"remember {ch}"
    before["task_state"] = "hold inside"
    before["rolling_summary"] = "hidden recall"
    after = dict(before)
    after["response_draft"] = ""
    return {"input_event": f"remember {ch}", "harness_events": [],
            "state_before": before, "state_after": after}


def make_recall_tick(ch: str) -> dict[str, Any]:
    before = blank_state()
    before["input_window"] = "answer"
    before["task_state"] = "answer from inside"
    before["rolling_summary"] = "hidden recall"
    after = dict(before)
    after["response_draft"] = ch
    for region, value in before.items():
        if isinstance(value, str) and ch in value:
            raise ValueError(f"recall before {region} contains answer {ch!r}")
    return {"input_event": "answer", "harness_events": [],
            "state_before": before, "state_after": after}


def build_episodes(n: int) -> list[tuple[str, dict, dict]]:
    episodes = []
    for i in range(n):
        ch = ALPH[i % len(ALPH)]
        store = make_store_tick(ch)
        recall = make_recall_tick(ch)
        episodes.append((ch, store, recall))
    return episodes


def load_core(path: Path, device: torch.device) -> tuple[AxonCore, ProjectionBank, CoreConfig]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = CoreConfig.from_dict(ckpt.get("cfg", {}))
    core = AxonCore(cfg).to(device).eval()
    core.load_state_dict(ckpt["core_state"])
    bank = ProjectionBank(cfg.d_model).to(device).eval()
    bank.load_state_dict(ckpt["projection_bank_state"])
    log(f"loaded {path.name}")
    log(f"  cfg: d_model={cfg.d_model} heads={cfg.n_heads} layers={cfg.n_layers} "
        f"ffn={cfg.ffn_dim} soul_rows={cfg.soul_rows} hot={cfg.soul_hot_rows} "
        f"mode={cfg.soul_write_mode} compartments={getattr(cfg, 'n_soul_compartments', 0)}")
    return core, bank, cfg


def run_tick(
    core: AxonCore,
    bank: ProjectionBank,
    cfg: CoreConfig,
    tick: dict,
    soul: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    slots_np, kinds, row_map = materialize(tick["state_before"])
    slots16 = torch.from_numpy(slots_np).to(device, dtype=torch.float32)
    with torch.no_grad():
        field = bank.project_in(slots16, kinds).unsqueeze(0)
        soul_b = soul.unsqueeze(0)
        for _ in range(cfg.n_ticks):
            out = core.forward_with_soul(field, soul_b)
            field = out["field"]
            soul_b = out["soul"]
    return soul_b.squeeze(0)


def decode_draft(
    bank: ProjectionBank,
    field: torch.Tensor,
    row_map: dict,
) -> str:
    pred16 = bank.egress_letters(field.squeeze(0)).detach().cpu().numpy()
    return decode_region(pred16, row_map, "response_draft")


def recall_with_soul(
    core: AxonCore,
    bank: ProjectionBank,
    cfg: CoreConfig,
    recall_tick: dict,
    soul: torch.Tensor,
    device: torch.device,
) -> str:
    slots_np, kinds, row_map = materialize(recall_tick["state_before"])
    slots16 = torch.from_numpy(slots_np).to(device, dtype=torch.float32)
    with torch.no_grad():
        field = bank.project_in(slots16, kinds).unsqueeze(0)
        soul_b = soul.unsqueeze(0)
        for _ in range(cfg.n_ticks):
            out = core.forward_with_soul(field, soul_b)
            field = out["field"]
            soul_b = out["soul"]
    return decode_draft(bank, field, row_map)


def main() -> int:
    ap = argparse.ArgumentParser(description="Counterfactual soul-conditioning probe")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=200,
                    help="number of hidden-letter episodes to generate")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="",
                    help="optional JSON file to write the report")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not verify_substrate(verbose=False):
        print("substrate geometry FAILED")
        return 1
    if not field_selftest(verbose=False):
        print("field contract self-test FAILED")
        return 1

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        # Try repo-relative first, then checkpoints/ subdir.
        repo_path = ROOT / args.checkpoint
        if repo_path.exists():
            ckpt_path = repo_path
        else:
            ckpt_path = CHECKPOINT_DIR / args.checkpoint
    if not ckpt_path.exists():
        print(f"checkpoint not found: {ckpt_path}")
        return 1

    core, bank, cfg = load_core(ckpt_path, device)

    episodes = build_episodes(args.episodes)
    log(f"generated {len(episodes)} hidden-letter episodes ({len(ALPH)} unique targets)")

    # Capture post-store souls.
    total_rows = cfg.total_soul_rows()
    souls = []
    targets = []
    log("running store ticks ...")
    for i, (ch, store_tick, _) in enumerate(episodes):
        soul = torch.zeros(total_rows, cfg.d_model, device=device)
        post_store = run_tick(core, bank, cfg, store_tick, soul, device)
        souls.append(post_store)
        targets.append(ch)
        if (i + 1) % 50 == 0:
            log(f"  store tick {i+1}/{len(episodes)}")

    # Pair episodes with different targets. Use deterministic pairing: i with i+1
    # in a ring, skipping if they share a target (shouldn't happen for n mod 36 != 0).
    pairs = []
    for i in range(len(episodes)):
        j = (i + 1) % len(episodes)
        if targets[i] != targets[j]:
            pairs.append((i, j))

    # Original recall.
    log("running original recalls ...")
    original_correct = 0
    original_preds = [None] * len(episodes)
    for i, (ch, _, recall_tick) in enumerate(episodes):
        pred = recall_with_soul(core, bank, cfg, recall_tick, souls[i], device)
        original_preds[i] = pred
        if pred.strip() == ch:
            original_correct += 1
    original_acc = original_correct / len(episodes)

    # Swap recall.
    log("running swapped-soul recalls ...")
    swap_correct = 0
    swap_flipped = 0
    swap_count = 0
    for i, j in pairs:
        pred_i_with_j = recall_with_soul(core, bank, cfg, episodes[i][2], souls[j], device)
        pred_j_with_i = recall_with_soul(core, bank, cfg, episodes[j][2], souls[i], device)
        if pred_i_with_j.strip() == targets[j]:
            swap_correct += 1
        if pred_j_with_i.strip() == targets[i]:
            swap_correct += 1
        if pred_i_with_j.strip() != original_preds[i].strip():
            swap_flipped += 1
        if pred_j_with_i.strip() != original_preds[j].strip():
            swap_flipped += 1
        swap_count += 2
    swap_acc = swap_correct / max(1, swap_count)
    flip_rate = swap_flipped / max(1, swap_count)

    # Zero-soul recall.
    log("running zero-soul recalls ...")
    zero_wrong = 0
    zero_empty = 0
    zero_count = 0
    zero_soul = torch.zeros(total_rows, cfg.d_model, device=device)
    for i, (ch, _, recall_tick) in enumerate(episodes):
        pred = recall_with_soul(core, bank, cfg, recall_tick, zero_soul, device)
        zero_count += 1
        if pred.strip() == ch:
            zero_wrong += 1
        if pred.strip() == "":
            zero_empty += 1
    zero_wrong_rate = zero_wrong / max(1, zero_count)
    zero_empty_rate = zero_empty / max(1, zero_count)

    report = {
        "checkpoint": str(ckpt_path),
        "episodes": len(episodes),
        "pairs": len(pairs),
        "original_accuracy": round(original_acc, 4),
        "swap_accuracy": round(swap_acc, 4),
        "swap_flip_rate": round(flip_rate, 4),
        "zero_wrong_rate": round(zero_wrong_rate, 4),
        "zero_empty_rate": round(zero_empty_rate, 4),
        "verdict": "SOUL_IS_READ" if swap_acc > 0.5 and zero_wrong_rate < 0.25 else "SOUL_NOT_CONDITIONING",
    }

    log("")
    log("=" * 60)
    log("COUNTERFACTUAL SOUL PROBE REPORT")
    log("=" * 60)
    log(f"  original recall accuracy: {original_acc:.3f}")
    log(f"  swap accuracy:            {swap_acc:.3f}  (pred == partner target)")
    log(f"  swap flip rate:           {flip_rate:.3f}  (output changed after swap)")
    log(f"  zero-soul wrong rate:     {zero_wrong_rate:.3f}  (should be low)")
    log(f"  zero-soul empty rate:     {zero_empty_rate:.3f}")
    log(f"  verdict:                  {report['verdict']}")
    log("=" * 60)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"wrote report to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
