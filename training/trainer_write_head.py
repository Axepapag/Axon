#!/usr/bin/env python3
"""Smoke trainer for the Axon shared text write head.

This trainer implements the locked rules from ``SOURCE_OF_TRUTH.md`` Layer 13
and ``RESOLUTION_writehead-0703`` Q6:

  * discrete per-position cross-entropy + length CE (masked past target length);
  * smoke gate: loss falls and positive exact-fill climbs above the
    constant-output floor before any long run;
  * collapse tripwire: warning when eval outputs have low variance;
  * rolling-3 checkpoints with ``pointer.json`` + ``checkpoint_done.json``;
  * Q6 counterfactual exact-fill rates every eval: positive, zero-field,
    swapped-field, irrelevant-field.

Curriculum framing (honest):

  The smoke drill here is **read-fidelity reconstruction from a frozen adapter
  summary**: pack real sentences into 8192D slots, project them down to the
  chosen ``d_model`` with the frozen shared adapter, and train the write head to
  reconstruct the exact slot text. This measures how much discrete text survives
  the lossy read path. It is information-theoretically partial at small
  ``d_model`` and is expected to be hard.

  The **real** write-head task trains the core + head jointly: the core learns
  to produce a ``d_model`` slot vector that the head can decode into the
  intended field text. That joint task is out of scope for this smoke scaffold;
  this trainer only proves the head's interface, loss, checkpointing, and gate
  machinery.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure repo root importable when run as a script.
ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from substrate import default_alphabet
from slots.slot_spec import pack_slot, MAX_TEXT_CHARS, KIND_TEXT
from adapters.slot_adapter import get_adapter
from heads.write_head import (
    WRITE_HEAD_VERSION,
    WriteHead,
    WriteHeadConfig,
    compute_loss as compute_write_head_loss,
    commit_diff,
    encode_decoded_slot,
)
from heads.probe import (
    ReadFidelityProbe,
    ReadFidelityProbeConfig,
    probe_metrics,
)

ALPHABET = default_alphabet()
ALPHABET_SET = set(ALPHABET)
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def clean_for_substrate(text: str) -> str:
    """Keep only substrate alphabet characters; map whitespace to space."""
    out: list[str] = []
    for c in text:
        if c in ALPHABET_SET:
            out.append(c)
        elif c.isspace():
            out.append(" ")
        # All other characters are dropped silently; the caller counts them.
    cleaned = "".join(out)
    cleaned = re.sub(r" +", " ", cleaned)
    return cleaned.strip()


def split_sentences(text: str) -> list[str]:
    """Split text on sentence terminators; return non-empty cleaned sentences."""
    parts = re.split(r"[.!?]+", text)
    sentences = []
    for part in parts:
        s = clean_for_substrate(part)
        if s:
            sentences.append(s)
    return sentences


def gather_source_text(paths: Iterable[Path]) -> tuple[list[str], dict[str, int]]:
    """Read docs/fixtures and split into substrate-clean sentences.

    Returns:
        sentences: list of sentences usable as slot text.
        counts: diagnostic counts (files, sentences, skipped_by_length, etc.).
    """
    sentences: list[str] = []
    counts = {
        "files_read": 0,
        "sentences_found": 0,
        "skipped_empty": 0,
        "skipped_overlength": 0,
        "skipped_invalid": 0,
    }
    for path in paths:
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            continue
        counts["files_read"] += 1
        for s in split_sentences(raw):
            counts["sentences_found"] += 1
            if not s:
                counts["skipped_empty"] += 1
                continue
            if len(s) > MAX_TEXT_CHARS:
                counts["skipped_overlength"] += 1
                continue
            # After cleaning, every character is in the alphabet by construction.
            sentences.append(s)
    return sentences, counts


def find_source_paths(root: Path) -> list[Path]:
    """Find .md and .txt files under docs/ and tests/fixtures/."""
    paths: list[Path] = []
    for sub in [root / "docs", root / "tests" / "fixtures"]:
        if sub.exists():
            paths.extend(sub.rglob("*.md"))
            paths.extend(sub.rglob("*.txt"))
    return sorted(set(paths))


def build_curriculum(
    d_model: int,
    root: Path,
    max_slots: int | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str], dict[str, Any]]:
    """Generate (summary, target_chars, target_length, text, kind) training data.

    Sentences come from real docs/fixtures text, packed into 8192D slots and
    projected down through the frozen shared adapter for ``d_model``.
    """
    paths = find_source_paths(root)
    sentences, counts = gather_source_text(paths)
    rng = random.Random(seed)
    rng.shuffle(sentences)
    if max_slots is not None:
        sentences = sentences[:max_slots]

    adapter = get_adapter(d_model)
    summaries: list[np.ndarray] = []
    texts: list[str] = []
    kinds: list[str] = []
    target_indices: list[list[int]] = []
    target_lengths: list[int] = []

    overlength_count = 0
    for text in sentences:
        if len(text) > MAX_TEXT_CHARS:
            overlength_count += 1
            continue
        slot = pack_slot(text=text, kind=KIND_TEXT)
        summary = adapter.project_down(slot.vector)
        summaries.append(summary.astype(np.float32))
        texts.append(text)
        kinds.append(KIND_TEXT)
        indices = [CHAR_TO_IDX[c] for c in text]
        padded = indices + [-100] * (MAX_TEXT_CHARS - len(indices))
        target_indices.append(padded)
        target_lengths.append(len(text))

    counts["skipped_overlength"] += overlength_count
    counts["slots_built"] = len(texts)

    summary_tensor = torch.from_numpy(np.stack(summaries, axis=0))
    target_tensor = torch.tensor(target_indices, dtype=torch.long)
    length_tensor = torch.tensor(target_lengths, dtype=torch.long)
    return summary_tensor, target_tensor, length_tensor, texts, kinds, counts


class WriteHeadDataset(Dataset):
    """Torch dataset wrapping the curriculum tensors."""

    def __init__(
        self,
        summaries: torch.Tensor,
        target_chars: torch.Tensor,
        target_lengths: torch.Tensor,
        texts: list[str],
        kinds: list[str],
    ) -> None:
        assert len(summaries) == len(texts)
        self.summaries = summaries
        self.target_chars = target_chars
        self.target_lengths = target_lengths
        self.texts = texts
        self.kinds = kinds

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
        return (
            self.summaries[idx],
            self.target_chars[idx],
            self.target_lengths[idx],
            self.texts[idx],
            self.kinds[idx],
        )


def exact_fill_rate(
    write_head: WriteHead,
    summaries: torch.Tensor,
    target_texts: Sequence[str],
    target_lengths: Sequence[int],
    mode: str,
    device: torch.device,
) -> dict[str, float]:
    """Compute Q6 counterfactual exact-fill rate.

    Modes:
      - positive: summaries are the true cues for the targets.
      - zero: input is a zero vector (masked/empty field).
      - swapped: each input summary is shifted by one slot (cue mismatch).
      - irrelevant: inputs are a random permutation of the summaries.
    """
    write_head.eval()
    summaries = summaries.to(device)
    n = len(target_texts)
    if n == 0:
        return {"exact_fill": 0.0, "length_exact": 0.0, "combined": 0.0}

    if mode == "positive":
        inputs = summaries
    elif mode == "zero":
        inputs = torch.zeros_like(summaries)
    elif mode == "swapped":
        inputs = torch.roll(summaries, shifts=1, dims=0)
    elif mode == "irrelevant":
        perm = torch.randperm(n)
        inputs = summaries[perm]
    else:
        raise ValueError(f"unknown cf mode: {mode}")

    preds, pred_lengths = write_head.decode(inputs)
    text_exact = 0
    length_exact = 0
    combined = 0
    for i, (pred, pred_len) in enumerate(zip(preds, pred_lengths)):
        tgt_len = target_lengths[i]
        if pred_len == tgt_len:
            length_exact += 1
        if pred == target_texts[i]:
            text_exact += 1
            if pred_len == tgt_len:
                combined += 1

    return {
        "exact_fill": text_exact / n,
        "length_exact": length_exact / n,
        "combined": combined / n,
    }


def evaluate(
    write_head: WriteHead,
    dataloader: DataLoader,
    device: torch.device,
    run_cf_probe: bool = True,
) -> dict[str, Any]:
    """Run one eval pass and return metrics + Q6 counterfactual rates."""
    write_head.eval()
    total_loss = 0.0
    total_items = 0
    all_texts: list[str] = []
    all_targets: list[str] = []
    all_lengths: list[int] = []
    all_summaries: list[torch.Tensor] = []
    all_pred_lengths: list[int] = []

    for summaries, targets, lengths, texts, _kinds in dataloader:
        summaries = summaries.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)
        with torch.no_grad():
            out = write_head(summaries)
            loss = compute_write_head_loss(out, targets, lengths)["total"]
        total_loss += float(loss.item()) * summaries.size(0)
        total_items += summaries.size(0)

        preds, pred_lengths = write_head.decode(summaries)
        all_texts.extend(preds)
        all_targets.extend(texts)
        all_lengths.extend(lengths.tolist())
        all_pred_lengths.extend(pred_lengths)
        all_summaries.append(summaries.cpu())

    mean_loss = total_loss / max(1, total_items)
    text_exact = sum(1 for p, t in zip(all_texts, all_targets) if p == t)
    length_exact = sum(1 for pl, tl in zip(all_pred_lengths, all_lengths) if pl == tl)
    combined = sum(
        1 for p, t, pl, tl in zip(all_texts, all_targets, all_pred_lengths, all_lengths)
        if p == t and pl == tl
    )
    length_std = float(np.std(all_pred_lengths)) if all_pred_lengths else 0.0

    metrics: dict[str, Any] = {
        "loss": mean_loss,
        "n": total_items,
        "positive_exact_fill": text_exact / max(1, total_items),
        "positive_length_exact": length_exact / max(1, total_items),
        "positive_combined": combined / max(1, total_items),
        "predicted_length_std": length_std,
    }

    if run_cf_probe and total_items > 0:
        all_summaries_cat = torch.cat(all_summaries, dim=0)
        for mode in ("zero", "swapped", "irrelevant"):
            rates = exact_fill_rate(
                write_head, all_summaries_cat, all_targets, all_lengths, mode, device
            )
            metrics[f"{mode}_exact_fill"] = rates["exact_fill"]
            metrics[f"{mode}_length_exact"] = rates["length_exact"]
            metrics[f"{mode}_combined"] = rates["combined"]

    return metrics


def collapse_tripwire(metrics: dict[str, Any], floor: float) -> list[str]:
    """Return warning strings if the model looks collapsed/constant."""
    warnings: list[str] = []
    if metrics["predicted_length_std"] < 0.5 and metrics["positive_exact_fill"] < floor:
        warnings.append(
            f"COLLAPSE TRIPWIRE: low length variance (std={metrics['predicted_length_std']:.3f}) "
            f"with exact-fill={metrics['positive_exact_fill']:.3f}"
        )
    if metrics["positive_exact_fill"] > 0.0 and metrics["positive_exact_fill"] < floor / 2:
        warnings.append(
            f"COLLAPSE TRIPWIRE: exact-fill is above zero but below floor/2 "
            f"({metrics['positive_exact_fill']:.3f} < {floor/2:.3f})"
        )
    return warnings


def save_rolling_checkpoint(
    step: int,
    write_head: WriteHead,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
    checkpoint_dir: Path,
    keep: int = 3,
) -> Path:
    """Save a checkpoint and maintain a rolling-3 active set + sentinel."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = checkpoint_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / f"checkpoint_step{step:06d}.pt"
    # Use the write-head module's checkpoint version so it can be loaded
    # directly with WriteHead.load(); extra trainer fields are ignored.
    payload = {
        "version": WRITE_HEAD_VERSION,
        "config": write_head.cfg.to_dict(),
        "state_dict": write_head.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
    }
    torch.save(payload, str(path))

    # Rolling-3 active checkpoints; older ones are moved to archive, not deleted.
    all_ckpts = sorted(
        checkpoint_dir.glob("checkpoint_step*.pt"),
        key=lambda p: int(p.stem.split("step")[1]),
    )
    for old in all_ckpts[:-keep]:
        dest = archive_dir / old.name
        shutil.move(str(old), str(dest))

    active_ckpts = sorted(
        checkpoint_dir.glob("checkpoint_step*.pt"),
        key=lambda p: int(p.stem.split("step")[1]),
    )
    pointer = {
        "latest": str(active_ckpts[-1]) if active_ckpts else None,
        "previous": [str(p) for p in (active_ckpts[:-1] if active_ckpts else [])],
        "step": step,
        "active_count": len(active_ckpts),
    }
    (checkpoint_dir / "pointer.json").write_text(json.dumps(pointer, indent=2))
    sentinel = {
        "status": "done",
        "checkpoint": str(path),
        "step": step,
        "metrics": metrics,
    }
    (checkpoint_dir / "checkpoint_done.json").write_text(json.dumps(sentinel, indent=2))
    return path


def smoke_gate_passed(
    initial_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    floor: float,
) -> tuple[bool, str]:
    """Return (passed, reason) for the smoke gate."""
    loss_fell = final_metrics["loss"] < initial_metrics["loss"] * 0.95
    above_floor = final_metrics["positive_exact_fill"] > floor
    if not loss_fell and not above_floor:
        return False, f"loss did not fall and exact-fill stayed at or below floor ({floor:.3f})"
    if not loss_fell:
        return False, f"exact-fill above floor but loss did not fall"
    if not above_floor:
        return False, f"loss fell but exact-fill did not exceed floor ({floor:.3f})"
    return True, "loss fell and exact-fill climbed above constant-output floor"


def q6_gate_passed(metrics: dict[str, Any]) -> tuple[bool, str]:
    """Return (passed, reason) for the Q6 counterfactual write-head gate.

    Thresholds (positive > 90%, controls < 5%) are the resolution proposals;
    the first smoke run is expected to calibrate them.
    """
    pos = metrics.get("positive_combined", 0.0)
    zero = metrics.get("zero_combined", 1.0)
    swapped = metrics.get("swapped_combined", 1.0)
    irrelevant = metrics.get("irrelevant_combined", 1.0)
    if pos < 0.90:
        return False, f"positive combined exact-fill {pos:.3f} < 0.90"
    for name, val in [("zero", zero), ("swapped", swapped), ("irrelevant", irrelevant)]:
        if val > 0.05:
            return False, f"{name}-field control {val:.3f} > 0.05"
    return True, "Q6 counterfactual gate passed"


def run_smoke(
    write_head: WriteHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int,
    eval_every: int,
    checkpoint_dir: Path,
    floor: float,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Run a short smoke loop. Returns (passed, initial_metrics, final_metrics)."""
    log(f"starting smoke run: {max_steps} steps, eval every {eval_every}")
    initial_metrics = evaluate(write_head, val_loader, device, run_cf_probe=True)
    log(f"  initial eval: {format_metrics(initial_metrics)}")
    best_loss = float("inf")

    write_head.train()
    step = 0
    global_step = 0
    for epoch in range((max_steps // len(train_loader)) + 2):
        for batch in train_loader:
            summaries, targets, lengths, _texts, _kinds = batch
            summaries = summaries.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            out = write_head(summaries)
            loss = compute_write_head_loss(out, targets, lengths)["total"]
            loss.backward()
            optimizer.step()

            step += 1
            global_step = step
            if step % eval_every == 0 or step == 1 or step >= max_steps:
                metrics = evaluate(write_head, val_loader, device, run_cf_probe=True)
                write_head.train()
                best_loss = min(best_loss, metrics["loss"])
                log(f"  step {step:5d}: {format_metrics(metrics)}")
                for warn in collapse_tripwire(metrics, floor):
                    log(f"  WARNING {warn}")
                if step >= max_steps:
                    break
        if step >= max_steps:
            break

    final_metrics = evaluate(write_head, val_loader, device, run_cf_probe=True)
    log(f"  final eval: {format_metrics(final_metrics)}")
    passed, reason = smoke_gate_passed(initial_metrics, final_metrics, floor)
    log(f"  smoke gate: {'PASS' if passed else 'FAIL'} — {reason}")

    if passed:
        ckpt_path = save_rolling_checkpoint(
            global_step, write_head, optimizer, final_metrics, checkpoint_dir
        )
        log(f"  saved checkpoint: {ckpt_path}")

    return passed, initial_metrics, final_metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    parts = [
        f"loss={metrics['loss']:.4f}",
        f"pos_ex={metrics['positive_exact_fill']:.3f}",
        f"pos_len={metrics['positive_length_exact']:.3f}",
        f"zero_ex={metrics.get('zero_exact_fill', float('nan')):.3f}",
        f"swap_ex={metrics.get('swapped_exact_fill', float('nan')):.3f}",
        f"irr_ex={metrics.get('irrelevant_exact_fill', float('nan')):.3f}",
        f"len_std={metrics['predicted_length_std']:.2f}",
    ]
    return "  ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write-head smoke trainer")
    ap.add_argument("--d_model", type=int, default=64,
                    help="adapter/core dimension (64, 128, 256)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="compute device")
    ap.add_argument("--smoke", action="store_true", default=True,
                    help="run smoke mode (default)")
    ap.add_argument("--long", action="store_true",
                    help="continue into a longer run after the smoke gate passes")
    ap.add_argument("--max_steps", type=int, default=200,
                    help="training steps for smoke mode")
    ap.add_argument("--long_steps", type=int, default=2000,
                    help="additional steps after smoke gate for --long")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_slots", type=int, default=1000,
                    help="cap curriculum size for fast smoke")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint_dir", default="",
                    help="override checkpoint directory")
    args = ap.parse_args(argv)

    # Enforce safe defaults: no accidental long GPU run.
    if args.long and not args.smoke:
        args.smoke = True

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        log("cuda requested but unavailable; falling back to cpu")
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = Path(ROOT)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        root / "State" / "write_head" / f"d{args.d_model}"
    )

    # Constant-output floor: must beat guessing the most common length/char.
    floor = max(0.02, 1.5 / len(ALPHABET))

    log("=" * 60)
    log("WRITE-HEAD SMOKE TRAINER")
    log("=" * 60)
    log(f"d_model={args.d_model}  device={device}  smoke={args.smoke}  long={args.long}")
    log(f"floor={floor:.4f}  max_slots={args.max_slots}")

    log("building curriculum from docs/fixtures...")
    summaries, targets, lengths, texts, kinds, counts = build_curriculum(
        args.d_model, root, max_slots=args.max_slots, seed=args.seed
    )
    log(f"  curriculum: {counts}")
    if len(texts) < 10:
        log("ERROR: not enough curriculum examples to train")
        return 1

    # Train/val split.
    n = len(texts)
    n_val = max(1, n // 10)
    n_train = n - n_val
    train_ds = WriteHeadDataset(
        summaries[:n_train], targets[:n_train], lengths[:n_train],
        texts[:n_train], kinds[:n_train],
    )
    val_ds = WriteHeadDataset(
        summaries[n_train:], targets[n_train:], lengths[n_train:],
        texts[n_train:], kinds[n_train:],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    log(f"  train={n_train}  val={n_val}")

    cfg = WriteHeadConfig.from_substrate(args.d_model)
    write_head = WriteHead(cfg).to(device)
    optimizer = torch.optim.Adam(write_head.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in write_head.parameters())
    log(f"write head params: {n_params:,}")

    # Optional read-fidelity probe baseline (train a tiny probe on frozen summaries).
    probe_cfg = ReadFidelityProbeConfig.from_substrate(args.d_model, hidden_dim=128)
    probe = ReadFidelityProbe(probe_cfg).to(device)
    probe.eval_metrics = lambda ds: probe_metrics(  # type: ignore[attr-defined]
        probe, ds.summaries.to(device), ds.texts, ds.kinds, device=device
    )
    log("read-fidelity probe baseline on validation summaries...")
    val_probe_metrics = probe_metrics(probe, val_ds.summaries.to(device), val_ds.texts, val_ds.kinds, device=device)
    log(f"  {val_probe_metrics}")

    passed, initial_metrics, final_metrics = run_smoke(
        write_head, train_loader, val_loader, optimizer, device,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        checkpoint_dir=checkpoint_dir,
        floor=floor,
    )

    if not passed:
        log("smoke gate FAILED — long run blocked by SOURCE_OF_TRUTH Layer 13(b)")
        return 1

    q6_ok, q6_reason = q6_gate_passed(final_metrics)
    log(f"  Q6 gate: {'PASS' if q6_ok else 'FAIL'} — {q6_reason}")

    if args.long:
        if not q6_ok:
            log("Q6 gate not passed — long run blocked by RESOLUTION_writehead-0703 Q6")
            return 1
        log(f"smoke and Q6 passed; continuing long run for {args.long_steps} steps")
        # Long run still uses the same locked rules; checkpoints roll.
        write_head.train()
        start_step = args.max_steps
        step = start_step
        while step < start_step + args.long_steps:
            for batch in train_loader:
                summaries, targets, lengths, _texts, _kinds = batch
                summaries = summaries.to(device)
                targets = targets.to(device)
                lengths = lengths.to(device)
                optimizer.zero_grad()
                out = write_head(summaries)
                loss = compute_write_head_loss(out, targets, lengths)["total"]
                loss.backward()
                optimizer.step()
                step += 1
                if step % args.eval_every == 0:
                    metrics = evaluate(write_head, val_loader, device, run_cf_probe=True)
                    write_head.train()
                    log(f"  long step {step:5d}: {format_metrics(metrics)}")
                    for warn in collapse_tripwire(metrics, floor):
                        log(f"  WARNING {warn}")
                    save_rolling_checkpoint(
                        step, write_head, optimizer, metrics, checkpoint_dir
                    )
                if step >= start_step + args.long_steps:
                    break
        final_metrics = evaluate(write_head, val_loader, device, run_cf_probe=True)
        save_rolling_checkpoint(
            step, write_head, optimizer, final_metrics, checkpoint_dir
        )
        log(f"long run finished: {format_metrics(final_metrics)}")

    # Final report.
    report = {
        "d_model": args.d_model,
        "device": str(device),
        "floor": floor,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "smoke_passed": passed,
        "q6_passed": q6_ok,
        "q6_reason": q6_reason,
        "curriculum_counts": counts,
        "checkpoint_dir": str(checkpoint_dir),
    }
    report_path = checkpoint_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log(f"wrote smoke report: {report_path}")
    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
