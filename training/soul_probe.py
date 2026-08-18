#!/usr/bin/env python3
"""Inspect saved Axon soul state without exposing raw vectors.

This is a visibility probe, not a load-bearing proof. It reports soul
occupancy, tier/category distribution, salience, and row norms from a
checkpoint. A real load-bearing test still needs paired zero/swap-soul probes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older torch builds do not expose weights_only. This utility is meant
        # for local trainer checkpoints, not arbitrary downloaded files.
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"checkpoint is not a dict: {path}")
    return obj


def _tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        out = value.detach().cpu()
        return out.to(dtype=dtype) if dtype is not None else out
    out = torch.as_tensor(value)
    return out.to(dtype=dtype) if dtype is not None else out


def _safe_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        value = value.item()
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def _mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return _safe_float(values.float().mean())


def _max(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return _safe_float(values.float().max())


def summarize_checkpoint(path: str | Path, *, top_rows: int = 12) -> dict[str, Any]:
    """Return a compact soul summary for a trainer checkpoint."""
    ckpt_path = Path(path)
    ckpt = _load_checkpoint(ckpt_path)
    soul_state = ckpt.get("soul_state")
    if not isinstance(soul_state, dict):
        raise ValueError(f"checkpoint has no soul_state dict: {ckpt_path}")

    cfg = soul_state.get("cfg") if isinstance(soul_state.get("cfg"), dict) else {}
    if not cfg and isinstance(ckpt.get("soul_cfg"), dict):
        cfg = ckpt["soul_cfg"]

    tensor = _tensor(soul_state.get("tensor"), dtype=torch.float32)
    if tensor is None or tensor.ndim != 2:
        raise ValueError("soul_state.tensor must be a 2D tensor")
    total_rows, d_model = int(tensor.shape[0]), int(tensor.shape[1])

    active = _tensor(soul_state.get("active"), dtype=torch.bool)
    if active is None or active.numel() != total_rows:
        active = tensor.norm(dim=1) > 0
    active = active.reshape(total_rows)

    tier = _tensor(soul_state.get("tier"), dtype=torch.long)
    if tier is None or tier.numel() != total_rows:
        tier = torch.zeros(total_rows, dtype=torch.long)
    tier = tier.reshape(total_rows)

    category = _tensor(soul_state.get("category"), dtype=torch.long)
    if category is None or category.numel() != total_rows:
        category = torch.full((total_rows,), -1, dtype=torch.long)
    category = category.reshape(total_rows)

    salience = _tensor(soul_state.get("salience"), dtype=torch.float32)
    if salience is None or salience.numel() != total_rows:
        salience = torch.zeros(total_rows, dtype=torch.float32)
    salience = salience.reshape(total_rows)

    dormant_for = _tensor(soul_state.get("dormant_for"), dtype=torch.long)
    if dormant_for is None or dormant_for.numel() != total_rows:
        dormant_for = torch.zeros(total_rows, dtype=torch.long)
    dormant_for = dormant_for.reshape(total_rows)

    tick_born = _tensor(soul_state.get("tick_born"), dtype=torch.long)
    if tick_born is None or tick_born.numel() != total_rows:
        tick_born = torch.zeros(total_rows, dtype=torch.long)
    tick_born = tick_born.reshape(total_rows)

    norms = tensor.norm(dim=1)
    active_norms = norms[active]
    active_salience = salience[active]

    tier_specs = cfg.get("tiers", []) if isinstance(cfg, dict) else []
    tier_names = [str(spec.get("name", f"tier_{i}")) for i, spec in enumerate(tier_specs)]
    max_tier_idx = int(tier.max().item()) if tier.numel() else 0
    while len(tier_names) <= max_tier_idx:
        tier_names.append(f"tier_{len(tier_names)}")

    tiers: list[dict[str, Any]] = []
    for idx, name in enumerate(tier_names):
        tier_mask = tier == idx
        tier_active = active & tier_mask
        tier_norms = norms[tier_active]
        tier_salience = salience[tier_active]
        tiers.append(
            {
                "index": idx,
                "name": name,
                "capacity": int(tier_mask.sum().item()),
                "active": int(tier_active.sum().item()),
                "inactive": int((tier_mask & ~active).sum().item()),
                "mean_norm": _mean(tier_norms),
                "max_norm": _max(tier_norms),
                "mean_salience": _mean(tier_salience),
                "max_salience": _max(tier_salience),
                "dormant_rows": int(((dormant_for > 0) & tier_active).sum().item()),
            }
        )

    category_names = cfg.get("categories", []) if isinstance(cfg, dict) else []
    categories: list[dict[str, Any]] = []
    for idx, name in enumerate(category_names):
        mask = active & (category == idx)
        categories.append(
            {
                "index": idx,
                "name": str(name),
                "active": int(mask.sum().item()),
                "mean_salience": _mean(salience[mask]),
                "mean_norm": _mean(norms[mask]),
            }
        )

    top: list[dict[str, Any]] = []
    active_idx = active.nonzero(as_tuple=True)[0]
    if active_idx.numel() and top_rows > 0:
        ranking = torch.argsort(salience[active_idx] + norms[active_idx] * 1e-6, descending=True)
        for row_idx in active_idx[ranking[:top_rows]]:
            idx = int(row_idx.item())
            cat_idx = int(category[idx].item())
            tier_idx = int(tier[idx].item())
            top.append(
                {
                    "row": idx,
                    "tier": tier_names[tier_idx] if 0 <= tier_idx < len(tier_names) else f"tier_{tier_idx}",
                    "category": (
                        str(category_names[cat_idx])
                        if 0 <= cat_idx < len(category_names)
                        else ""
                    ),
                    "norm": _safe_float(norms[idx]),
                    "salience": _safe_float(salience[idx]),
                    "dormant_for": int(dormant_for[idx].item()),
                    "tick_born": int(tick_born[idx].item()),
                }
            )

    core_cfg = ckpt.get("core_cfg") if isinstance(ckpt.get("core_cfg"), dict) else {}
    if not core_cfg and isinstance(ckpt.get("cfg"), dict):
        core_cfg = ckpt["cfg"]
    return {
        "kind": "axon_soul_probe",
        "checkpoint": str(ckpt_path),
        "checkpoint_step": int(ckpt.get("step", 0) or 0),
        "core": {
            "d_model": int(core_cfg.get("d_model", cfg.get("d_model", d_model))),
            "n_layers": int(core_cfg.get("n_layers", 0) or 0),
            "n_heads": int(core_cfg.get("n_heads", 0) or 0),
            "d_ff": int(core_cfg.get("d_ff", core_cfg.get("ffn_dim", 0)) or 0),
        },
        "soul": {
            "d_model": int(cfg.get("d_model", d_model)),
            "tensor_shape": [total_rows, d_model],
            "total_rows": total_rows,
            "active_rows": int(active.sum().item()),
            "current_tick": int(soul_state.get("current_tick", 0) or 0),
            "active_mean_norm": _mean(active_norms),
            "active_max_norm": _max(active_norms),
            "active_mean_salience": _mean(active_salience),
            "active_max_salience": _max(active_salience),
        },
        "tiers": tiers,
        "categories": categories,
        "top_active_rows": top,
        "notes": [
            "This is an occupancy and health probe only.",
            "Load-bearing behavior requires paired correct-soul, zero-soul, and swapped-soul probes.",
            "Raw soul vectors are intentionally not printed.",
        ],
    }


def print_text(summary: dict[str, Any]) -> None:
    soul = summary["soul"]
    core = summary["core"]
    print(f"Soul probe: {summary['checkpoint']}")
    print(f"  checkpoint_step: {summary['checkpoint_step']}")
    print(
        "  core: "
        f"d={core['d_model']} layers={core['n_layers']} heads={core['n_heads']} ffn={core['d_ff']}"
    )
    print(
        "  soul: "
        f"shape={soul['tensor_shape']} active={soul['active_rows']}/{soul['total_rows']} "
        f"tick={soul['current_tick']} mean_norm={soul['active_mean_norm']:.4f} "
        f"mean_salience={soul['active_mean_salience']:.4f}"
    )
    print("  tiers:")
    for tier in summary["tiers"]:
        print(
            f"    {tier['name']}: active={tier['active']}/{tier['capacity']} "
            f"mean_norm={tier['mean_norm']:.4f} mean_salience={tier['mean_salience']:.4f} "
            f"dormant={tier['dormant_rows']}"
        )
    if summary["categories"]:
        print("  categories:")
        for cat in summary["categories"]:
            print(
                f"    {cat['name']}: active={cat['active']} "
                f"mean_norm={cat['mean_norm']:.4f} mean_salience={cat['mean_salience']:.4f}"
            )
    if summary["top_active_rows"]:
        print("  top_active_rows:")
        for row in summary["top_active_rows"]:
            category = f" category={row['category']}" if row["category"] else ""
            print(
                f"    row={row['row']} tier={row['tier']}{category} "
                f"norm={row['norm']:.4f} salience={row['salience']:.4f} "
                f"dormant_for={row['dormant_for']} tick_born={row['tick_born']}"
            )
    print("  note: occupancy only; run zero/swap paired probes before claiming soul load-bearing.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect saved Axon soul state.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--top-rows", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_checkpoint(args.checkpoint, top_rows=args.top_rows)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_text(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
