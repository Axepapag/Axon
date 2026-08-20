"""CLI driver for exact-v4 multi-tick continuation training.

Loads a core checkpoint (core state only is required), instantiates
``MultiTickTrainer`` with a read-only soul, and trains over the exact-v4
curriculum until ``--steps`` is reached.

Checkpoints written by this script contain enough state to resume the
optimizer and PyTorch RNG, but they are intentionally a new schema because
the exact-v4 trainer does not share the phase0b charslot curriculum sampler.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np
import torch

# Make the repo root importable when run as a script.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cores.core import AxonCore, CoreConfig
from training.soul_compressor import SoulCompressor
from training.trainer_multitick import (
    MultiTickTrainer,
    _prepare_soul,
    verify_finite_nonzero_gradient_coverage,
    verify_optimizer_coverage,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-v4 multi-tick continuation training"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--train-soul",
        action="store_true",
        help="Enable trainable soul-write (requires soul_read_bearing=True).",
    )
    parser.add_argument(
        "--soul-continuity-weight",
        type=float,
        default=0.01,
        help="Weight for the soul continuity regularizer.",
    )
    parser.add_argument(
        "--soul-l2-weight",
        type=float,
        default=1e-5,
        help="Weight for the soul L2 magnitude regularizer.",
    )
    parser.add_argument(
        "--soul-compression-weight",
        type=float,
        default=0.1,
        help="Weight for the Phase-A soul compression loss.",
    )
    parser.add_argument(
        "--soul-compressor-lr",
        type=float,
        default=None,
        help="Learning rate for the SoulCompressor (defaults to --lr).",
    )
    return parser.parse_args(argv)


def load_checkpoint(path: pathlib.Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "core_state" not in payload or "cfg" not in payload:
        raise ValueError(
            f"checkpoint {path} is missing core_state or cfg; "
            "expected an Axon core checkpoint"
        )
    return payload


def build_core(payload: dict[str, Any], device: torch.device) -> AxonCore:
    cfg = CoreConfig.from_dict(payload["cfg"])
    core = AxonCore(cfg).to(device)
    core.load_state_dict(payload["core_state"], strict=True)
    return core


def load_episodes(dataset_dir: pathlib.Path) -> list[dict[str, Any]]:
    episodes_path = dataset_dir / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(episodes_path)
    episodes: list[dict[str, Any]] = []
    with episodes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))
    if not episodes:
        raise ValueError("episode file is empty")
    return episodes


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "python_random_state" in state:
        try:
            random.setstate(state["python_random_state"])
        except Exception as exc:
            print(f"WARNING: could not restore python RNG state: {exc}")
    if "numpy_random_state" in state:
        try:
            np.random.set_state(state["numpy_random_state"])
        except Exception as exc:
            print(f"WARNING: could not restore numpy RNG state: {exc}")
    if "torch_rng_state" in state:
        trs = state["torch_rng_state"]
        try:
            torch.set_rng_state(trs)
        except Exception:
            # PyTorch RNG state format can vary across versions/platforms.
            # Reconstruct a fresh ByteTensor from the raw bytes when possible.
            try:
                if isinstance(trs, torch.Tensor):
                    raw = trs.cpu().numpy().tobytes()
                elif isinstance(trs, (bytes, bytearray)):
                    raw = bytes(trs)
                else:
                    raw = bytes(trs)
                reconstructed = torch.frombuffer(raw, dtype=torch.uint8).clone()
                torch.set_rng_state(reconstructed)
                print("WARNING: reconstructed torch RNG state from raw bytes")
            except Exception as exc2:
                print(f"WARNING: could not restore torch RNG state: {exc2}")
    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
        try:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        except Exception as exc:
            print(f"WARNING: could not restore CUDA RNG state: {exc}")


def rotate_checkpoints(run_dir: pathlib.Path, keep: int = 2) -> None:
    """Keep only the ``keep`` most recent ckpt_*.pt files in run_dir."""
    checkpoints = sorted(
        run_dir.glob("ckpt_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in checkpoints[keep:]:
        try:
            old.unlink()
            print(f"removed old checkpoint: {old.name}")
        except OSError:
            pass


def save_checkpoint(
    run_dir: pathlib.Path,
    step: int,
    cfg: dict[str, Any],
    core: AxonCore,
    optimizer: torch.optim.Optimizer,
    rng_state: dict[str, Any],
    soul_state: dict[str, Any] | None = None,
    soul_compressor_state: dict[str, Any] | None = None,
) -> pathlib.Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"ckpt_{step}.pt"
    payload = {
        "checkpoint_schema": "axon_exact_v4_checkpoint_v1",
        "step": step,
        "cfg": cfg,
        "core_state": core.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": rng_state,
    }
    if soul_state is not None:
        payload["soul_state"] = soul_state
    if soul_compressor_state is not None:
        payload["soul_compressor_state"] = soul_compressor_state
    torch.save(payload, path)
    rotate_checkpoints(run_dir, keep=2)
    return path


def eval_loop(
    trainer: MultiTickTrainer,
    episodes: list[dict[str, Any]],
    eval_count: int,
    rng: random.Random,
    soul: torch.Tensor | None = None,
    soul_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    dev_episodes = [ep for ep in episodes if ep.get("split") == "dev"]
    if not dev_episodes:
        dev_episodes = episodes
    chosen = dev_episodes
    if eval_count < len(dev_episodes):
        chosen = rng.sample(dev_episodes, eval_count)

    trainer.core.eval()
    losses: list[float] = []
    exact_matches: list[int] = []
    with torch.no_grad():
        for episode in chosen:
            result = trainer.forward_episode(
                episode,
                soul=soul,
                soul_mask=soul_mask,
            )
            losses.append(float(result.total_loss.detach().cpu()))
            exact_matches.append(
                1 if result.transition_reconstruction_gate_passed else 0
            )
    trainer.core.train()
    return {
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
        "exact_match_rate": (
            float(np.mean(exact_matches)) if exact_matches else float("nan")
        ),
        "n": len(chosen),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    dataset_dir = pathlib.Path(args.dataset_dir).resolve()
    run_dir = pathlib.Path(args.run_dir).resolve()
    device = torch.device(args.device)
    target_step = int(args.steps)
    batch_size = max(1, int(args.batch_size))

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    payload = load_checkpoint(checkpoint_path, device)
    start_step = int(payload.get("step", 0))
    if target_step <= start_step:
        raise ValueError(
            f"--steps {target_step} must be greater than checkpoint step {start_step}"
        )

    set_seed(args.seed)
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])

    core = build_core(payload, device)

    # Phase-A soul compressor: external module that distills hot rows into warm
    # rows at episode boundaries.  It is separate from AxonCore so legacy core
    # checkpoints load with strict=True unchanged.
    soul_compressor: SoulCompressor | None = None
    if args.train_soul and core.cfg.soul_hot_rows > 0:
        soul_compressor = SoulCompressor(
            d_model=core.cfg.d_model,
            hot_rows=core.cfg.soul_hot_rows,
            warm_rows=core.cfg.soul_rows,
            num_heads=core.cfg.n_heads,
        ).to(device)
        if "soul_compressor_state" in payload:
            soul_compressor.load_state_dict(payload["soul_compressor_state"])
            print("restored soul compressor state from checkpoint")
        else:
            print("initialized fresh soul compressor")

    trainer = MultiTickTrainer(
        core,
        soul_read_bearing=args.train_soul,
        train_soul=args.train_soul,
        soul_continuity_weight=args.soul_continuity_weight,
        soul_l2_weight=args.soul_l2_weight,
        soul_compressor=soul_compressor,
        soul_compression_weight=args.soul_compression_weight,
    )

    compressor_lr = (
        args.soul_compressor_lr
        if args.soul_compressor_lr is not None
        else args.lr
    )
    if soul_compressor is not None and compressor_lr != args.lr:
        core_params = [
            parameter
            for name, parameter in core.named_parameters()
            if name in trainer.core_declared_trainable_names
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": core_params, "lr": args.lr},
                {"params": list(soul_compressor.parameters()), "lr": compressor_lr},
            ]
        )
    else:
        optimizer = torch.optim.AdamW(trainer.optimizer_parameters(), lr=args.lr)
    if "optimizer_state" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer_state"])
        except Exception as exc:
            print(
                f"warning: could not restore optimizer state ({exc}); "
                "starting optimizer from scratch"
            )

    if args.train_soul and batch_size > 1:
        raise ValueError(
            "train_soul=True currently requires batch_size=1; "
            "persistent soul state cannot be carried across parallel episodes"
        )

    # Persistent soul state: carried from episode to episode and saved in
    # checkpoints so training resumes with the same memory state.
    persistent_soul: torch.Tensor | None = None
    persistent_soul_mask: torch.Tensor | None = None
    if trainer.soul_read_bearing:
        if "soul_state" in payload:
            loaded_soul = payload["soul_state"]["soul"].to(device)
            loaded_soul_mask = payload["soul_state"]["soul_mask"].to(device)
            persistent_soul, persistent_soul_mask = _prepare_soul(
                core,
                soul_read_bearing=True,
                train_soul=args.train_soul,
                soul=loaded_soul,
                soul_mask=loaded_soul_mask,
            )
            print(
                f"restored soul state from checkpoint: "
                f"rows={persistent_soul.shape[1]}, "
                f"active={persistent_soul_mask.sum().item()}"
            )
        else:
            persistent_soul, persistent_soul_mask = _prepare_soul(
                core,
                soul_read_bearing=True,
                train_soul=args.train_soul,
                soul=None,
                soul_mask=None,
            )
            print(
                f"initialized default soul state: "
                f"rows={persistent_soul.shape[1]}, "
                f"active={persistent_soul_mask.sum().item()}"
            )

    episodes = load_episodes(dataset_dir)
    rng = random.Random(args.seed)

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_handle = metrics_path.open("a", encoding="utf-8")

    core.train()
    train_losses: list[float] = []
    step = start_step
    epoch = 0
    index = 0
    shuffled = episodes.copy()
    rng_epoch = random.Random(args.seed + epoch)
    rng_epoch.shuffle(shuffled)

    print(
        f"start exact-v4 training at step {start_step} -> {target_step} "
        f"on {device} with {len(episodes)} episodes, lr={args.lr}, "
        f"batch_size={batch_size}"
    )

    try:
        while step < target_step:
            step_losses: list[float] = []
            batch_episodes: list[dict[str, Any]] = []
            for _ in range(batch_size):
                if index >= len(shuffled):
                    epoch += 1
                    index = 0
                    rng_epoch = random.Random(args.seed + epoch)
                    shuffled = episodes.copy()
                    rng_epoch.shuffle(shuffled)
                batch_episodes.append(shuffled[index])
                index += 1

            if batch_size == 1:
                train_result = trainer.train_episode(
                    batch_episodes[0],
                    optimizer,
                    soul=persistent_soul,
                    soul_mask=persistent_soul_mask,
                    verify_gradients=True,
                )
                step_losses.append(
                    float(train_result.forward.total_loss.detach().cpu())
                )
                if train_result.forward.final_soul is not None:
                    persistent_soul = train_result.forward.final_soul
                if train_result.forward.final_soul_mask is not None:
                    persistent_soul_mask = train_result.forward.final_soul_mask
            else:
                extra_modules = (
                    {"soul_compressor": soul_compressor}
                    if soul_compressor is not None
                    else None
                )
                verify_optimizer_coverage(
                    optimizer,
                    core,
                    trainer.declared_trainable_names,
                    extra_modules=extra_modules,
                )
                optimizer.zero_grad(set_to_none=True)
                for episode in batch_episodes:
                    forward = trainer.forward_episode(
                        episode,
                        gradient_scale=1.0 / batch_size,
                    )
                    step_losses.append(float(forward.total_loss.detach().cpu()))
                verify_finite_nonzero_gradient_coverage(
                    core,
                    trainer.declared_trainable_names,
                    raise_on_error=True,
                    extra_modules=extra_modules,
                )
                optimizer.step()

            step += 1
            mean_step_loss = float(np.mean(step_losses))
            train_losses.append(mean_step_loss)

            if step % 50 == 0:
                recent = train_losses[-50:]
                print(
                    f"step {step}  loss={float(np.mean(recent)):.4f} "
                    f"(last {len(recent)})"
                )

            if step % args.eval_every == 0:
                eval_metrics = eval_loop(
                    trainer,
                    episodes,
                    args.eval_episodes,
                    random.Random(args.seed + step),
                    soul=persistent_soul,
                    soul_mask=persistent_soul_mask,
                )
                print(
                    f"eval at step {step}: loss={eval_metrics['mean_loss']:.4f} "
                    f"exact_match={eval_metrics['exact_match_rate']:.3f} "
                    f"n={eval_metrics['n']}"
                )
                metrics_handle.write(
                    json.dumps(
                        {
                            "type": "eval",
                            "step": step,
                            "timestamp_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                            ),
                            **eval_metrics,
                        }
                    )
                    + "\n"
                )
                metrics_handle.flush()
                core.train()

            if step % args.checkpoint_every == 0:
                soul_state = None
                if persistent_soul is not None and persistent_soul_mask is not None:
                    soul_state = {
                        "soul": persistent_soul.detach().cpu(),
                        "soul_mask": persistent_soul_mask.detach().cpu(),
                    }
                soul_compressor_state = (
                    soul_compressor.state_dict()
                    if soul_compressor is not None
                    else None
                )
                ckpt_path = save_checkpoint(
                    run_dir,
                    step,
                    core.cfg.to_dict(),
                    core,
                    optimizer,
                    get_rng_state(),
                    soul_state=soul_state,
                    soul_compressor_state=soul_compressor_state,
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

    # Final checkpoint and pointer.
    final_soul_state = None
    if persistent_soul is not None and persistent_soul_mask is not None:
        final_soul_state = {
            "soul": persistent_soul.detach().cpu(),
            "soul_mask": persistent_soul_mask.detach().cpu(),
        }
    final_soul_compressor_state = (
        soul_compressor.state_dict()
        if soul_compressor is not None
        else None
    )
    final_ckpt = save_checkpoint(
        run_dir,
        step,
        core.cfg.to_dict(),
        core,
        optimizer,
        get_rng_state(),
        soul_state=final_soul_state,
        soul_compressor_state=final_soul_compressor_state,
    )
    pointer = {"step": step, "active": final_ckpt.name, "target_step": target_step}
    (run_dir / "pointer.json").write_text(
        json.dumps(pointer, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "checkpoint_done.json").write_text(
        json.dumps({"step": step, "optimizer_state": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"finished at step {step}; final checkpoint: {final_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
