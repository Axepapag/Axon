#!/usr/bin/env python3
"""Minimal CPU conversational-ignition trainer for Axon.

Loads a 64D exact-v4 checkpoint, trains on a conversational curriculum using
char-slot forward and the corrected conversational objective, and saves
checkpoints with rotation.

This is intentionally simpler than trainer_multitick.py: it bypasses the
exact-v4 read-page contract and uses a direct history + user_input +
response_draft field layout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# Allow importing from repo root when run as script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cores.core import AxonCore, CoreConfig
from substrate import get_letter_bank
from training.conversational_objective import (
    DECODER_WIDTH,
    EMPTY_INDEX,
    IGNORE_INDEX,
    conversational_objective,
    decode_parallel_logits,
)
from training.trainer_slot import CharSlotFieldBuilder


DEFAULT_HISTORY_CHARS = 128
DEFAULT_USER_CHARS = 64
DEFAULT_RESP_CHARS = 64


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "core_state" not in payload or "cfg" not in payload:
        raise ValueError(f"checkpoint {path} missing core_state or cfg")
    return payload


def build_core(payload: dict[str, Any], device: torch.device) -> AxonCore:
    cfg = CoreConfig.from_dict(payload["cfg"])
    core = AxonCore(cfg).to(device)
    core.load_state_dict(payload["core_state"])
    core.train()
    return core


def load_soul_state(payload: dict[str, Any], device: torch.device):
    soul_state = payload.get("soul_state")
    if not isinstance(soul_state, dict):
        return None, None
    soul = soul_state.get("soul")
    soul_mask = soul_state.get("soul_mask")
    if soul is None or soul_mask is None:
        return None, None
    return soul.to(device), soul_mask.to(device)


def load_curriculum(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def prepare_example(
    example: dict[str, Any],
    builder: CharSlotFieldBuilder,
) -> dict[str, torch.Tensor]:
    """Convert a conversational curriculum example into trainer tensors."""
    active_field = example.get("active_field", {})
    history = active_field.get("conversation_history", "")
    user_input = active_field.get("user_input", "")
    target_delta = example.get("target_delta", {})
    target = target_delta.get("text", "")
    if not target:
        raise ValueError(
            f"example {example.get('example_id', '?')} has empty target text"
        )
    built = builder.build(history, user_input, target, "")
    return {
        "field16": built["field16"],
        "region": built["region"],
        "targets": built["targets"],
        "target_text": target,
    }


def train_step(
    core: AxonCore,
    builder: CharSlotFieldBuilder,
    example_tensors: dict[str, torch.Tensor],
    soul: torch.Tensor | None,
    soul_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
    """Single example forward/backward returning loss and updated soul."""
    out = core.forward_charslot(
        example_tensors["field16"],
        example_tensors["region"],
        soul,
        soul_mask=soul_mask,
        response_slice=builder.resp_slice,
    )
    logits = core.charslot_logits(out["response_delta_16"], builder.bank_unit)
    target_text = example_tensors["target_text"]
    obj = conversational_objective(
        logits,
        target_text,
        answer_required=True,
        first_position_nonempty_weight=0.1,
    )
    metrics = compute_quick_metrics(logits, obj.targets.indices)
    new_soul = out["soul"].detach() if out.get("soul") is not None else soul
    new_soul_mask = soul_mask
    return obj.loss, new_soul, new_soul_mask, metrics


def compute_quick_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    mask = targets != IGNORE_INDEX
    correct = ((pred == targets) & mask).sum().item()
    total = mask.sum().item()
    nonempty_mask = (targets != EMPTY_INDEX) & mask
    nonempty_total = nonempty_mask.sum().item()
    nonempty_correct = ((pred == targets) & nonempty_mask).sum().item()
    return {
        "char_acc": correct / max(1, total),
        "nonempty_char_acc": nonempty_correct / max(1, nonempty_total),
        "loss": float(F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
        ).item()),
    }


def evaluate(
    core: AxonCore,
    builder: CharSlotFieldBuilder,
    examples: list[dict[str, Any]],
    soul: torch.Tensor | None,
    soul_mask: torch.Tensor | None,
    device: torch.device,
    max_examples: int | None = None,
) -> dict[str, float]:
    eval_examples = examples[:max_examples] if max_examples else examples
    core.eval()
    losses: list[float] = []
    char_accs: list[float] = []
    exact = 0
    n_nonempty_pred = 0
    all_preds: list[str] = []
    with torch.no_grad():
        for ex in eval_examples:
            tensors = prepare_example(ex, builder)
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in tensors.items()}
            out = core.forward_charslot(
                tensors["field16"],
                tensors["region"],
                soul,
                soul_mask=soul_mask,
                response_slice=builder.resp_slice,
            )
            logits = core.charslot_logits(out["response_delta_16"], builder.bank_unit)
            targets = tensors["targets"]
            obj = conversational_objective(
                logits,
                tensors["target_text"],
                answer_required=True,
                first_position_nonempty_weight=0.1,
            )
            losses.append(float(obj.loss.item()))
            metrics = compute_quick_metrics(logits, obj.targets.indices)
            char_accs.append(metrics["char_acc"])
            decoded = decode_parallel_logits(logits)
            pred_text = decoded.texts[0]
            terminated = decoded.terminated[0]
            all_preds.append(pred_text)
            if pred_text.strip():
                n_nonempty_pred += 1
            if pred_text == tensors["target_text"]:
                exact += 1
    core.train()
    n = len(eval_examples)
    return {
        "mean_loss": float(np.mean(losses)),
        "mean_char_acc": float(np.mean(char_accs)),
        "exact_match_rate": exact / max(1, n),
        "nonempty_rate": n_nonempty_pred / max(1, n),
        "n": n,
        "sample_predictions": all_preds[:5],
        "sample_targets": [
            eval_examples[i].get("target_delta", {}).get("text", "")
            for i in range(min(5, n))
        ],
    }


def write_live_state(
    live_path: Path,
    step: int,
    target_step: int,
    train_losses: list[float],
    train_metrics: dict[str, float],
    core: AxonCore,
    builder: CharSlotFieldBuilder,
    live_examples: list[dict[str, Any]],
    soul: torch.Tensor | None,
    soul_mask: torch.Tensor | None,
    device: torch.device,
) -> None:
    """Write a small JSON dashboard file for the monitor GUI."""
    recent = train_losses[-10:] if train_losses else []
    mean_recent_loss = float(np.mean(recent)) if recent else 0.0

    core.eval()
    sample_preds: list[dict[str, str]] = []
    with torch.no_grad():
        for ex in live_examples:
            tensors = prepare_example(ex, builder)
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in tensors.items()}
            out = core.forward_charslot(
                tensors["field16"],
                tensors["region"],
                soul,
                soul_mask=soul_mask,
                response_slice=builder.resp_slice,
            )
            logits = core.charslot_logits(out["response_delta_16"], builder.bank_unit)
            decoded = decode_parallel_logits(logits)
            sample_preds.append(
                {
                    "history": ex.get("active_field", {}).get("conversation_history", ""),
                    "user_input": ex.get("active_field", {}).get("user_input", ""),
                    "target": ex.get("target_delta", {}).get("text", ""),
                    "prediction": decoded.texts[0],
                    "terminated": str(decoded.terminated[0]),
                }
            )
    core.train()

    checkpoints = sorted(
        live_path.parent.glob("ckpt_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    latest_checkpoint = str(checkpoints[-1]) if checkpoints else ""

    state = {
        "status": "running",
        "step": step,
        "target_step": target_step,
        "progress": step / max(1, target_step),
        "mean_recent_loss": round(mean_recent_loss, 4),
        "char_acc": round(train_metrics.get("char_acc", 0.0), 3),
        "nonempty_char_acc": round(train_metrics.get("nonempty_char_acc", 0.0), 3),
        "latest_checkpoint": latest_checkpoint,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "samples": sample_preds,
    }
    tmp_path = live_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp_path.replace(live_path)


def rotate_checkpoints(run_dir: Path, keep: int = 3) -> None:
    checkpoints = sorted(
        run_dir.glob("ckpt_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for old in checkpoints[:-keep]:
        old.unlink()
        print(f"removed old checkpoint: {old.name}")


def save_checkpoint(
    run_dir: Path,
    step: int,
    core: AxonCore,
    optimizer: torch.optim.Optimizer,
    soul: torch.Tensor | None,
    soul_mask: torch.Tensor | None,
    extra: dict[str, Any] | None = None,
    keep: int = 3,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "checkpoint_schema": "axon_conversational_cpu_checkpoint_v1",
        "step": step,
        "cfg": core.cfg.to_dict(),
        "core_state": core.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if soul is not None and soul_mask is not None:
        payload["soul_state"] = {
            "soul": soul.detach().cpu(),
            "soul_mask": soul_mask.detach().cpu(),
        }
    if extra:
        payload.update(extra)
    ckpt_path = run_dir / f"ckpt_{step}.pt"
    tmp_path = run_dir / f"ckpt_{step}.pt.tmp"
    torch.save(payload, tmp_path)
    tmp_path.replace(ckpt_path)
    rotate_checkpoints(run_dir, keep=keep)
    return ckpt_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Conversational CPU trainer")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--curriculum", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-chars", type=int, default=DEFAULT_HISTORY_CHARS)
    parser.add_argument("--user-chars", type=int, default=DEFAULT_USER_CHARS)
    parser.add_argument("--resp-chars", type=int, default=DEFAULT_RESP_CHARS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--live-path", type=Path, default=None)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = load_checkpoint(args.checkpoint, device)
    core = build_core(payload, device)
    soul, soul_mask = load_soul_state(payload, device)
    start_step = int(payload.get("step", 0))
    target_step = start_step + args.steps

    builder = CharSlotFieldBuilder(
        history_chars=args.history_chars,
        user_chars=args.user_chars,
        resp_chars=args.resp_chars,
        device=device,
        dtype=torch.float32,
    )

    examples = load_curriculum(args.curriculum)
    if not examples:
        raise ValueError(f"no examples loaded from {args.curriculum}")
    print(f"loaded {len(examples)} conversational examples")

    live_path = args.live_path or (args.run_dir / "live.json")
    live_examples = examples[:3]

    # Resume from latest checkpoint in run_dir if requested.
    optimizer = torch.optim.AdamW(
        core.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.resume and args.run_dir.exists():
        checkpoints = sorted(
            args.run_dir.glob("ckpt_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if checkpoints:
            resume_path = checkpoints[-1]
            print(f"resuming from {resume_path}")
            resume_payload = load_checkpoint(resume_path, device)
            core.load_state_dict(resume_payload["core_state"])
            start_step = int(resume_payload.get("step", 0))
            target_step = start_step + args.steps
            soul, soul_mask = load_soul_state(resume_payload, device)
            if "optimizer_state" in resume_payload:
                try:
                    optimizer.load_state_dict(resume_payload["optimizer_state"])
                    print("restored optimizer state from resume checkpoint")
                except Exception as exc:
                    print(f"warning: could not restore optimizer state: {exc}")
    elif "optimizer_state" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer_state"])
            print("restored optimizer state from base checkpoint")
        except Exception as exc:
            print(f"warning: could not restore optimizer state: {exc}")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.run_dir / "metrics.jsonl"
    metrics_handle = metrics_path.open("a", encoding="utf-8")

    rng = random.Random(args.seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)
    index = 0
    train_losses: list[float] = []

    print(
        f"starting conversational CPU training at step {start_step} -> {target_step} "
        f"on {device} with lr={args.lr}"
    )

    step = start_step
    try:
        while step < target_step:
            ex = shuffled[index]
            index += 1
            if index >= len(shuffled):
                index = 0
                rng.shuffle(shuffled)

            tensors = prepare_example(ex, builder)
            tensors = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in tensors.items()}

            optimizer.zero_grad(set_to_none=True)
            loss, soul, soul_mask, metrics = train_step(
                core, builder, tensors, soul, soul_mask
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))
            step += 1

            if step % 10 == 0:
                recent = train_losses[-10:]
                print(
                    f"step {step}  loss={float(np.mean(recent)):.4f} "
                    f"char_acc={metrics['char_acc']:.3f} "
                    f"nonempty_char_acc={metrics['nonempty_char_acc']:.3f}"
                )
                if args.sample_every and step % args.sample_every == 0:
                    write_live_state(
                        live_path,
                        step,
                        target_step,
                        train_losses,
                        metrics,
                        core,
                        builder,
                        live_examples,
                        soul,
                        soul_mask,
                        device,
                    )

            if step % args.eval_every == 0:
                eval_metrics = evaluate(
                    core, builder, examples, soul, soul_mask, device,
                    max_examples=args.max_eval_examples,
                )
                print(
                    f"eval at step {step}: loss={eval_metrics['mean_loss']:.4f} "
                    f"char_acc={eval_metrics['mean_char_acc']:.3f} "
                    f"exact={eval_metrics['exact_match_rate']:.3f} "
                    f"nonempty={eval_metrics['nonempty_rate']:.3f}"
                )
                for pred, tgt in zip(
                    eval_metrics["sample_predictions"], eval_metrics["sample_targets"]
                ):
                    print(f"  pred={pred!r} target={tgt!r}")
                metrics_handle.write(
                    json.dumps(
                        {
                            "type": "eval",
                            "step": step,
                            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            **eval_metrics,
                        },
                        default=str,
                    )
                    + "\n"
                )
                metrics_handle.flush()
                core.train()

            if step % args.checkpoint_every == 0:
                ckpt_path = save_checkpoint(
                    args.run_dir, step, core, optimizer, soul, soul_mask,
                    keep=args.keep_checkpoints,
                )
                print(f"checkpoint saved: {ckpt_path}")
                metrics_handle.write(
                    json.dumps(
                        {
                            "type": "checkpoint",
                            "step": step,
                            "path": str(ckpt_path),
                        }
                    )
                    + "\n"
                )
                metrics_handle.flush()

    finally:
        metrics_handle.close()

    final_ckpt = save_checkpoint(
        args.run_dir, step, core, optimizer, soul, soul_mask,
        keep=args.keep_checkpoints,
    )
    pointer = {"step": step, "active": final_ckpt.name, "target_step": target_step}
    (args.run_dir / "pointer.json").write_text(
        json.dumps(pointer, indent=2) + "\n", encoding="utf-8"
    )
    print(f"finished at step {step}; final checkpoint: {final_ckpt}")
    finished_state = {
        "status": "finished",
        "step": step,
        "target_step": target_step,
        "progress": 1.0,
        "mean_recent_loss": round(float(np.mean(train_losses[-10:])) if train_losses else 0.0, 4),
        "latest_checkpoint": str(final_ckpt),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "samples": [],
    }
    finished_path = live_path.with_name("live_finished.json")
    finished_tmp = finished_path.with_suffix(".json.tmp")
    finished_tmp.write_text(json.dumps(finished_state, indent=2), encoding="utf-8")
    finished_tmp.replace(finished_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
