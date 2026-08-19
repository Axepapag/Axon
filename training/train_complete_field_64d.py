#!/usr/bin/env python3
"""Observable trainer for Axon's complete-field 64D R0 core."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from substrate import assert_supported_text, default_alphabet, roundtrip_check
from training.complete_field_64d import (
    CompleteField64D,
    CompleteFieldPager,
    REGION_ORDER,
    ReaderConfig,
    canonical_field,
    sequence_cross_entropy,
    teacher_char_accuracy,
)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def load_jsonl(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if split is not None and record.get("split") != split:
                continue
            if record.get("schema") != "axon-complete-field-r0-example-v2":
                raise ValueError(f"{path}:{line_number}: counterfactual R0 schema v2 is required")
            field = canonical_field(record["field"])
            targets = record["targets"]
            if set(targets) != {"scratch", "response_draft"}:
                raise ValueError(f"{path}:{line_number}: target regions are not R0 scratch/response")
            assert_supported_text(targets["scratch"])
            assert_supported_text(targets["response_draft"])
            counterfactuals = record.get("response_counterfactuals", [])
            if not isinstance(counterfactuals, list):
                raise ValueError(f"{path}:{line_number}: response_counterfactuals must be a list")
            for counterfactual in counterfactuals:
                if set(counterfactual) != {"variant_id", "scratch", "response_draft"}:
                    raise ValueError(f"{path}:{line_number}: malformed response counterfactual")
                assert_supported_text(counterfactual["variant_id"])
                assert_supported_text(counterfactual["scratch"])
                assert_supported_text(counterfactual["response_draft"])
                if counterfactual["scratch"] == targets["scratch"]:
                    raise ValueError(f"{path}:{line_number}: counterfactual scratch is not an intervention")
                if counterfactual["response_draft"] == targets["response_draft"]:
                    raise ValueError(f"{path}:{line_number}: counterfactual response did not change")
            if record.get("write_authority", {}).get("diary") is not False:
                raise ValueError(f"{path}:{line_number}: diary must be disabled in R0")
            record["field"] = field
            record["response_counterfactuals"] = counterfactuals
            records.append(record)
    if not records:
        raise ValueError(f"no records loaded from {path}")
    return records


def stratified_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deterministic round-robin selection so one sorted family cannot dominate eval."""
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_family.setdefault(record["family"], []).append(record)
    selected: list[dict[str, Any]] = []
    offset = 0
    families = sorted(by_family)
    while len(selected) < min(limit, len(records)):
        progressed = False
        for family in families:
            rows = by_family[family]
            if offset < len(rows):
                selected.append(rows[offset])
                progressed = True
                if len(selected) >= min(limit, len(records)):
                    break
        if not progressed:
            break
        offset += 1
    return selected


def substrate_gate() -> None:
    total, failures, details = roundtrip_check()
    if failures:
        raise RuntimeError(f"exact 16D roundtrip failed for {failures}/{total}: {details}")
    if total != len(default_alphabet()):
        raise RuntimeError("alphabet count changed during launch")


def coverage_gate(records: Iterable[Mapping[str, Any]], page_sizes: tuple[int, ...] = (64, 128, 256)) -> None:
    for record in list(records)[:16]:
        expected = sum(len(record["field"][name]) for name in REGION_ORDER)
        hashes = set()
        for page_size in page_sizes:
            _, manifest = CompleteFieldPager(page_size).paginate(record["field"])
            if not manifest.complete or manifest.expected_characters != expected:
                raise RuntimeError(f"coverage gate failed for {record['example_id']} at page_size={page_size}")
            hashes.add(manifest.field_sha256)
        if len(hashes) != 1:
            raise RuntimeError("logical field identity changed across physical page sizes")


def corruption(text: str) -> str:
    if not text:
        return "No useful scratch was preserved."
    replacement = "x" if text[0] != "x" else "y"
    return replacement + text[1:]


@torch.no_grad()
def evaluate_teacher(
    model: CompleteField64D,
    records: list[dict[str, Any]],
    max_examples: int,
    causal_examples: int = 16,
) -> dict[str, Any]:
    model.eval()
    scratch_losses: list[float] = []
    response_losses: list[float] = []
    scratch_acc: list[float] = []
    response_acc: list[float] = []
    empty_deltas: list[float] = []
    corrupt_deltas: list[float] = []
    counterfactual_losses: list[float] = []
    counterfactual_acc: list[float] = []
    samples = stratified_records(records, max(1, max_examples))
    for index, record in enumerate(samples):
        out = model.forward_transaction(
            record["field"], record["targets"]["scratch"], record["targets"]["response_draft"]
        )
        sl = sequence_cross_entropy(out["scratch_logits"], out["scratch_targets"])
        rl = sequence_cross_entropy(out["response_logits"], out["response_targets"])
        scratch_losses.append(float(sl.item()))
        response_losses.append(float(rl.item()))
        scratch_acc.append(teacher_char_accuracy(out["scratch_logits"], out["scratch_targets"]))
        response_acc.append(teacher_char_accuracy(out["response_logits"], out["response_targets"]))
        if index < causal_examples:
            for variant, collector in (("", empty_deltas), (corruption(record["targets"]["scratch"]), corrupt_deltas)):
                ablated = dict(record["field"])
                ablated["scratch"] = variant
                state, _ = model.read_field(ablated)
                logits, targets = model.decode_teacher(state, record["targets"]["response_draft"], head=1)
                collector.append(float(sequence_cross_entropy(logits, targets).item() - rl.item()))
        for counterfactual in record["response_counterfactuals"]:
            if len(counterfactual_losses) >= causal_examples:
                break
            intervened = dict(record["field"])
            intervened["scratch"] = counterfactual["scratch"]
            state, _ = model.read_field(intervened)
            logits, targets = model.decode_teacher(state, counterfactual["response_draft"], head=1)
            counterfactual_losses.append(float(sequence_cross_entropy(logits, targets).item()))
            counterfactual_acc.append(teacher_char_accuracy(logits, targets))
    result = {
        "examples": len(samples),
        "mean_scratch_loss": float(np.mean(scratch_losses)),
        "mean_response_loss": float(np.mean(response_losses)),
        "mean_total_loss": float(np.mean(scratch_losses) + np.mean(response_losses)),
        "scratch_teacher_char_accuracy": float(np.mean(scratch_acc)),
        "response_teacher_char_accuracy": float(np.mean(response_acc)),
        "causal_empty_scratch_ce_delta": float(np.mean(empty_deltas)) if empty_deltas else 0.0,
        "causal_corrupt_scratch_ce_delta": float(np.mean(corrupt_deltas)) if corrupt_deltas else 0.0,
        "counterfactual_examples": len(counterfactual_losses),
        "counterfactual_teacher_loss": (
            float(np.mean(counterfactual_losses)) if counterfactual_losses else math.inf
        ),
        "counterfactual_teacher_char_accuracy": (
            float(np.mean(counterfactual_acc)) if counterfactual_acc else 0.0
        ),
    }
    model.train()
    return result


@torch.no_grad()
def observable_samples(model: CompleteField64D, records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    model.eval()
    output: list[dict[str, Any]] = []
    for record in stratified_records(records, count):
        transaction = model.run_transaction(record["field"])
        output.append(
            {
                "example_id": record["example_id"],
                "family": record["family"],
                "user_input": record["field"]["user_input"],
                "gold_scratch": record["targets"]["scratch"],
                "predicted_scratch": transaction["scratch"],
                "gold_response": record["targets"]["response_draft"],
                "predicted_response": transaction["response_draft"],
                "scratch_terminated": transaction["scratch_terminated"],
                "response_terminated": transaction["response_terminated"],
                "coverage_tick1": transaction["coverage_tick1"],
                "coverage_tick2": transaction["coverage_tick2"],
                "typed_delta": transaction["typed_delta"],
            }
        )
    model.train()
    return output


@torch.no_grad()
def forced_counterfactual_samples(
    model: CompleteField64D,
    records: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Decode the same field after correct and matched scratch interventions."""
    model.eval()
    causal_records = [record for record in records if record["response_counterfactuals"]]
    output: list[dict[str, Any]] = []
    for record in stratified_records(causal_records, len(causal_records)):
        correct_field = dict(record["field"])
        correct_field["scratch"] = record["targets"]["scratch"]
        correct_state, _ = model.read_field(correct_field)
        correct_response, correct_terminated = model.decode_greedy(correct_state, head=1)
        for counterfactual in record["response_counterfactuals"]:
            intervened = dict(record["field"])
            intervened["scratch"] = counterfactual["scratch"]
            counterfactual_state, _ = model.read_field(intervened)
            counterfactual_response, counterfactual_terminated = model.decode_greedy(
                counterfactual_state, head=1
            )
            output.append(
                {
                    "example_id": record["example_id"],
                    "family": record["family"],
                    "variant_id": counterfactual["variant_id"],
                    "correct_scratch": record["targets"]["scratch"],
                    "counterfactual_scratch": counterfactual["scratch"],
                    "gold_correct_response": record["targets"]["response_draft"],
                    "gold_counterfactual_response": counterfactual["response_draft"],
                    "predicted_correct_response": correct_response,
                    "predicted_counterfactual_response": counterfactual_response,
                    "correct_terminated": correct_terminated,
                    "counterfactual_terminated": counterfactual_terminated,
                    "response_changed": correct_response != counterfactual_response,
                }
            )
            if len(output) >= count:
                model.train()
                return output
    model.train()
    return output


def checkpoint_payload(
    model: CompleteField64D,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config: ReaderConfig,
    args: argparse.Namespace,
    baseline: Mapping[str, Any],
    sampler_state: object,
) -> dict[str, Any]:
    return {
        "schema": "axon-complete-field-r0-checkpoint-v2",
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "reader_config": asdict(config),
        "trainer_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "baseline": dict(baseline),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "sampler": sampler_state,
        },
    }


def save_checkpoint(
    run_dir: Path,
    model: CompleteField64D,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config: ReaderConfig,
    args: argparse.Namespace,
    baseline: Mapping[str, Any],
    sampler_state: object,
    keep: int,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / f"ckpt_{step:09d}.pt"
    temporary = final.with_suffix(".pt.tmp")
    torch.save(
        checkpoint_payload(model, optimizer, scaler, step, config, args, baseline, sampler_state),
        temporary,
    )
    os.replace(temporary, final)
    atomic_json(
        run_dir / "pointer.json",
        {"schema": "axon-checkpoint-pointer-v1", "step": step, "checkpoint": str(final), "time": time.time()},
    )
    atomic_json(
        run_dir / "checkpoint_done.json",
        {"schema": "axon-checkpoint-sentinel-v1", "step": step, "checkpoint": str(final), "time": time.time()},
    )
    checkpoints = sorted(run_dir.glob("ckpt_*.pt"))
    for stale in checkpoints[:-keep]:
        stale.unlink()
    return final


def restore(
    path: Path,
    model: CompleteField64D,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, dict[str, Any], object | None]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "axon-complete-field-r0-checkpoint-v2":
        raise ValueError(f"unsupported checkpoint schema in {path}")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload.get("scaler_state", {}))
    rng = payload.get("rng", {})
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return int(payload["step"]), dict(payload.get("baseline", {})), rng.get("sampler")


def newest_checkpoint(run_dir: Path) -> Path | None:
    paths = sorted(run_dir.glob("ckpt_*.pt")) if run_dir.exists() else []
    return paths[-1] if paths else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the observable complete-field 64D R0 core")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--max-output-chars", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--causal-weight", type=float, default=0.50)
    parser.add_argument("--causal-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=64018)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-if-available", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if args.steps < 1 or args.grad_accum < 1 or args.causal_every < 1:
        raise ValueError("steps, grad_accum, and causal_every must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    substrate_gate()
    train_records = load_jsonl(args.train)
    eval_records = load_jsonl(args.eval)
    coverage_gate(train_records)
    coverage_gate(eval_records)
    causal_train_records = [record for record in train_records if record["response_counterfactuals"]]
    causal_eval_records = [record for record in eval_records if record["response_counterfactuals"]]
    if not causal_train_records or not causal_eval_records:
        raise ValueError("matched response counterfactuals are required in train and eval")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    config = ReaderConfig(page_size=args.page_size, max_output_chars=args.max_output_chars)
    model = CompleteField64D(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_dir / "config.json",
        {
            "schema": "axon-complete-field-r0-run-config-v2",
            "reader": asdict(config),
            "trainer": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "causal_train_records": len(causal_train_records),
            "causal_eval_records": len(causal_eval_records),
        },
    )

    step = 0
    baseline: dict[str, Any] = {}
    sampler_state: object | None = None
    if args.resume or args.resume_if_available:
        checkpoint = newest_checkpoint(args.run_dir)
        if checkpoint is None and args.resume:
            raise FileNotFoundError("--resume requested but no checkpoint exists")
        if checkpoint is not None:
            step, baseline, sampler_state = restore(checkpoint, model, optimizer, scaler, device)
            print(f"resumed {checkpoint} at step {step}", flush=True)
        else:
            print("no checkpoint found; starting a fresh run", flush=True)
    if not baseline:
        baseline = evaluate_teacher(model, eval_records, args.eval_examples)
        baseline["step"] = step
        atomic_json(args.run_dir / "baseline.json", baseline)

    rng = random.Random()
    if sampler_state is None:
        rng.seed(args.seed + step)
    else:
        rng.setstate(sampler_state)
    recent_losses: list[float] = []
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    status = "running"
    try:
        while step < args.steps:
            record = train_records[rng.randrange(len(train_records))]
            autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)
            with autocast:
                out = model.forward_transaction(
                    record["field"],
                    record["targets"]["scratch"],
                    record["targets"]["response_draft"],
                )
                scratch_loss = sequence_cross_entropy(out["scratch_logits"], out["scratch_targets"])
                response_loss = sequence_cross_entropy(out["response_logits"], out["response_targets"])
                causal_loss = torch.zeros((), device=device)
                causal_accuracy = 0.0
                causal_example_id: str | None = None
                causal_variant_id: str | None = None
                if args.causal_weight > 0 and step % args.causal_every == 0:
                    causal_record = causal_train_records[rng.randrange(len(causal_train_records))]
                    counterfactuals = causal_record["response_counterfactuals"]
                    counterfactual = counterfactuals[rng.randrange(len(counterfactuals))]
                    intervened = dict(causal_record["field"])
                    intervened["scratch"] = counterfactual["scratch"]
                    causal_state, _ = model.read_field(intervened)
                    causal_logits, causal_targets = model.decode_teacher(
                        causal_state, counterfactual["response_draft"], head=1
                    )
                    causal_loss = sequence_cross_entropy(causal_logits, causal_targets)
                    causal_accuracy = teacher_char_accuracy(causal_logits.detach(), causal_targets)
                    causal_example_id = causal_record["example_id"]
                    causal_variant_id = counterfactual["variant_id"]
                loss = scratch_loss + response_loss + args.causal_weight * causal_loss
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            next_step = step + 1
            if next_step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad).item())
                if not math.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite gradient norm at step {next_step}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = 0.0
            step = next_step
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"non-finite loss at step {step}")
            recent_losses.append(loss_value)
            recent_losses = recent_losses[-100:]
            elapsed = max(1e-6, time.time() - started)
            metric = {
                "schema": "axon-complete-field-r0-metric-v1",
                "step": step,
                "loss": loss_value,
                "scratch_loss": float(scratch_loss.detach().item()),
                "response_loss": float(response_loss.detach().item()),
                "counterfactual_loss": float(causal_loss.detach().item()),
                "counterfactual_teacher_char_accuracy": causal_accuracy,
                "counterfactual_example_id": causal_example_id,
                "counterfactual_variant_id": causal_variant_id,
                "scratch_teacher_char_accuracy": teacher_char_accuracy(
                    out["scratch_logits"].detach(), out["scratch_targets"]
                ),
                "response_teacher_char_accuracy": teacher_char_accuracy(
                    out["response_logits"].detach(), out["response_targets"]
                ),
                "coverage_tick1_chars": out["coverage_tick1"].observed_characters,
                "coverage_tick1_pages": out["coverage_tick1"].page_count,
                "coverage_tick2_chars": out["coverage_tick2"].observed_characters,
                "coverage_tick2_pages": out["coverage_tick2"].page_count,
                "grad_norm": grad_norm,
                "steps_per_second": step / elapsed,
                "example_id": record["example_id"],
                "family": record["family"],
                "time": time.time(),
            }
            append_jsonl(args.run_dir / "metrics.jsonl", metric)

            evaluation: dict[str, Any] | None = None
            samples: list[dict[str, Any]] | None = None
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                evaluation = evaluate_teacher(model, eval_records, args.eval_examples)
                evaluation.update({"schema": "axon-complete-field-r0-eval-v1", "step": step, "time": time.time()})
                append_jsonl(args.run_dir / "evaluations.jsonl", evaluation)
            if step == 1 or step % args.sample_every == 0 or step == args.steps:
                samples = observable_samples(model, eval_records, args.sample_count)
                append_jsonl(
                    args.run_dir / "samples.jsonl",
                    {"schema": "axon-complete-field-r0-samples-v1", "step": step, "samples": samples, "time": time.time()},
                )
            if step % args.checkpoint_every == 0 or step == args.steps:
                save_checkpoint(
                    args.run_dir,
                    model,
                    optimizer,
                    scaler,
                    step,
                    config,
                    args,
                    baseline,
                    rng.getstate(),
                    args.keep_checkpoints,
                )

            if evaluation is not None or samples is not None or step == 1:
                last_eval = evaluation or {}
                atomic_json(
                    args.run_dir / "live.json",
                    {
                        "schema": "axon-complete-field-r0-live-v1",
                        "status": status,
                        "step": step,
                        "target_step": args.steps,
                        "progress": step / args.steps,
                        "mean_recent_loss": float(np.mean(recent_losses)),
                        "steps_per_second": step / elapsed,
                        "eta_seconds": (args.steps - step) / max(1e-9, step / elapsed),
                        "baseline": baseline,
                        "latest_evaluation": last_eval,
                        "latest_samples": samples or [],
                        "coverage_enforced": True,
                        "diary_writes_enabled": False,
                        "time": time.time(),
                    },
                )
            if step == 1 or step % 25 == 0:
                print(
                    f"step={step} loss={loss_value:.4f} scratch={scratch_loss.item():.4f} "
                    f"response={response_loss.item():.4f} pages={out['coverage_tick1'].page_count} "
                    f"chars={out['coverage_tick1'].observed_characters} rate={step/elapsed:.3f}/s",
                    flush=True,
                )
    except KeyboardInterrupt:
        status = "interrupted"
        print("interrupted; writing recovery checkpoint", flush=True)
    except BaseException:
        status = "failed"
        raise
    finally:
        if step > 0:
            save_checkpoint(
                args.run_dir,
                model,
                optimizer,
                scaler,
                step,
                config,
                args,
                baseline,
                rng.getstate(),
                args.keep_checkpoints,
            )
        live_path = args.run_dir / "live.json"
        live: dict[str, Any] = {}
        if live_path.exists():
            live = json.loads(live_path.read_text(encoding="utf-8"))
        live.update({"status": "completed" if step >= args.steps else status, "step": step, "time": time.time()})
        atomic_json(live_path, live)

    final_eval = evaluate_teacher(model, eval_records, args.eval_examples)
    final_samples = observable_samples(model, eval_records, min(16, len(eval_records)))
    final_counterfactuals = forced_counterfactual_samples(
        model, causal_eval_records, min(16, len(causal_eval_records) * 2)
    )
    atomic_json(
        args.run_dir / "counterfactual_samples.json",
        {
            "schema": "axon-complete-field-r0-counterfactual-samples-v1",
            "step": step,
            "samples": final_counterfactuals,
        },
    )
    predicted_responses = [sample["predicted_response"] for sample in final_samples]
    response_termination_rate = float(np.mean([sample["response_terminated"] for sample in final_samples]))
    scratch_termination_rate = float(np.mean([sample["scratch_terminated"] for sample in final_samples]))
    response_nonblank_rate = float(np.mean([bool(text.strip()) for text in predicted_responses]))
    unique_response_count = len(set(predicted_responses))
    counterfactual_response_change_rate = float(
        np.mean([sample["response_changed"] for sample in final_counterfactuals])
    )
    forced_correct_termination_rate = float(
        np.mean([sample["correct_terminated"] for sample in final_counterfactuals])
    )
    forced_counterfactual_termination_rate = float(
        np.mean([sample["counterfactual_terminated"] for sample in final_counterfactuals])
    )
    loss_improved = final_eval["mean_total_loss"] < float(
        baseline.get("mean_total_loss", math.inf)
    )
    accuracy_improved = final_eval["response_teacher_char_accuracy"] > float(
        baseline.get("response_teacher_char_accuracy", 0.0)
    )
    counterfactual_accuracy_improved = (
        final_eval["counterfactual_teacher_char_accuracy"]
        >= float(baseline.get("counterfactual_teacher_char_accuracy", 0.0)) + 0.10
    )
    causal_passed = (
        final_eval["counterfactual_teacher_char_accuracy"] >= 0.50
        and counterfactual_accuracy_improved
        and counterfactual_response_change_rate >= 0.75
        and forced_correct_termination_rate >= 0.95
        and forced_counterfactual_termination_rate >= 0.95
    )
    free_running_passed = (
        response_termination_rate >= 0.95
        and scratch_termination_rate >= 0.95
        and response_nonblank_rate >= 0.95
        and unique_response_count >= 2
    )
    gate = {
        "schema": "axon-complete-field-r0-run-gate-v2",
        "step": step,
        "target_step": args.steps,
        "baseline_total_loss": baseline.get("mean_total_loss"),
        "final_total_loss": final_eval["mean_total_loss"],
        "loss_improved": loss_improved,
        "response_accuracy_improved": accuracy_improved,
        "coverage_contract_passed": True,
        "diary_writes_observed": 0,
        "diagnostic_empty_scratch_ce_delta": final_eval["causal_empty_scratch_ce_delta"],
        "diagnostic_corrupt_scratch_ce_delta": final_eval["causal_corrupt_scratch_ce_delta"],
        "baseline_counterfactual_teacher_char_accuracy": baseline.get(
            "counterfactual_teacher_char_accuracy"
        ),
        "final_counterfactual_teacher_char_accuracy": final_eval[
            "counterfactual_teacher_char_accuracy"
        ],
        "counterfactual_accuracy_improved": counterfactual_accuracy_improved,
        "counterfactual_response_change_rate": counterfactual_response_change_rate,
        "forced_correct_termination_rate": forced_correct_termination_rate,
        "forced_counterfactual_termination_rate": forced_counterfactual_termination_rate,
        "causal_scratch_gate_passed": causal_passed,
        "scratch_termination_rate": scratch_termination_rate,
        "response_termination_rate": response_termination_rate,
        "response_nonblank_rate": response_nonblank_rate,
        "unique_response_count": unique_response_count,
        "free_running_gate_passed": free_running_passed,
        "promotion_allowed": (
            loss_improved and accuracy_improved and causal_passed and free_running_passed
        ),
    }
    atomic_json(args.run_dir / "gate.json", gate)
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
