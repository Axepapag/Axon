"""Verify byte-exact logical resume for the complete-field R0 v6 CUDA objective."""
from __future__ import annotations

import argparse
import json
import os
import random
from argparse import Namespace
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from training.complete_field_64d import CompleteField64D, ReaderConfig, sequence_cross_entropy
from training.train_complete_field_64d import file_sha256, load_jsonl, restore, save_checkpoint


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    return left == right


def make_state(device: torch.device, config: ReaderConfig, lr: float) -> tuple[CompleteField64D, torch.optim.Optimizer, torch.amp.GradScaler]:
    model = CompleteField64D(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    return model, optimizer, scaler


def train_steps(
    model: CompleteField64D,
    optimizer: torch.optim.Optimizer,
    records: list[dict[str, Any]],
    sampler: random.Random,
    steps: int,
    teacher_forcing_ratio: float,
) -> list[float]:
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        record = records[sampler.randrange(len(records))]
        out = model.forward_transaction(
            record["field"],
            record["targets"]["scratch"],
            record["targets"]["response_draft"],
            teacher_forcing_ratio=teacher_forcing_ratio,
            alignment={
                "schema": record["alignment"]["schema"],
                "scratch": record["alignment"]["scratch"],
                "response_draft": record["alignment"]["response_draft"],
            },
        )
        scratch = sequence_cross_entropy(out["scratch_logits"], out["scratch_targets"])
        response = sequence_cross_entropy(out["response_logits"], out["response_targets"])
        alignment = out["alignment"]["position_loss"] + 0.5 * out["alignment"]["gate_loss"]
        loss = scratch + response + alignment
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad):
            raise FloatingPointError("non-finite gradient during resume verification")
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return losses


def rng_snapshot(sampler: random.Random) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
        "sampler": sampler.getstate(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=86019)
    parser.add_argument("--total-steps", type=int, default=4)
    parser.add_argument("--split-step", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--max-output-chars", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.75)
    parser.add_argument(
        "--legacy-record-direct",
        action="store_true",
        help="explicitly verify the preserved detached-record V6 trainer only",
    )
    args = parser.parse_args()
    if not args.legacy_record_direct:
        raise RuntimeError(
            "detached-record optimizer verification is disabled for canonical Axon "
            "training; use only as an explicitly opted-in legacy mechanism check"
        )
    if not 0 < args.split_step < args.total_steps:
        raise ValueError("split step must be inside total steps")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    records = load_jsonl(args.train)
    dataset = {"train": file_sha256(args.train), "eval": file_sha256(args.train)}
    config = ReaderConfig(page_size=args.page_size, max_output_chars=args.max_output_chars, dropout=0.0)

    seed_all(args.seed)
    full_model, full_optimizer, _ = make_state(device, config, args.lr)
    full_sampler = random.Random(args.seed + 1)
    full_losses = train_steps(
        full_model,
        full_optimizer,
        records,
        full_sampler,
        args.total_steps,
        args.teacher_forcing_ratio,
    )
    full_model_state = {key: value.detach().cpu().clone() for key, value in full_model.state_dict().items()}
    full_optimizer_state = full_optimizer.state_dict()
    full_rng = rng_snapshot(full_sampler)

    seed_all(args.seed)
    split_model, split_optimizer, split_scaler = make_state(device, config, args.lr)
    split_sampler = random.Random(args.seed + 1)
    split_losses = train_steps(
        split_model,
        split_optimizer,
        records,
        split_sampler,
        args.split_step,
        args.teacher_forcing_ratio,
    )
    checkpoint = save_checkpoint(
        args.run_dir,
        split_model,
        split_optimizer,
        split_scaler,
        args.split_step,
        config,
        Namespace(device=str(device), verification="v6-resume"),
        {"verification": "v6-resume"},
        split_sampler.getstate(),
        3,
        dataset,
    )

    resumed_model, resumed_optimizer, resumed_scaler = make_state(device, config, args.lr)
    restored_step, _, sampler_state = restore(
        checkpoint,
        resumed_model,
        resumed_optimizer,
        resumed_scaler,
        device,
        dataset,
    )
    resumed_sampler = random.Random()
    resumed_sampler.setstate(sampler_state)
    resumed_losses = train_steps(
        resumed_model,
        resumed_optimizer,
        records,
        resumed_sampler,
        args.total_steps - restored_step,
        args.teacher_forcing_ratio,
    )
    resumed_model_state = {key: value.detach().cpu().clone() for key, value in resumed_model.state_dict().items()}
    resumed_optimizer_state = resumed_optimizer.state_dict()
    resumed_rng = rng_snapshot(resumed_sampler)

    result = {
        "schema": "axon-complete-field-r0-v6-resume-verification-v1",
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "total_steps": args.total_steps,
        "split_step": args.split_step,
        "checkpoint": str(checkpoint),
        "checkpoint_schema": "axon-complete-field-r0-checkpoint-v6",
        "dataset_sha256": dataset,
        "full_losses": full_losses,
        "resumed_losses": split_losses + resumed_losses,
        "losses_exact": full_losses == split_losses + resumed_losses,
        "model_exact": exact_equal(full_model_state, resumed_model_state),
        "optimizer_exact": exact_equal(full_optimizer_state, resumed_optimizer_state),
        "rng_exact": exact_equal(full_rng, resumed_rng),
    }
    result["passed"] = bool(
        result["losses_exact"]
        and result["model_exact"]
        and result["optimizer_exact"]
        and result["rng_exact"]
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
