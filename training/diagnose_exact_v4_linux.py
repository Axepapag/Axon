"""Diagnostic: validate the first exact-v4 episode and print any read-page diff.

This is a standalone script meant for Kaggle/Linux debugging.  It loads the
bundled checkpoint and the first episode, then tries to validate it.  If the
validation fails (especially on read_page), it prints the stored and computed
read_page audit dicts so we can see what differs between the builder and the
runtime.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cores.core import AxonCore, CoreConfig
from runtime.field import compile_next_read_page, FieldViewCursor
from substrate import get_letter_bank
from training.build_multitick_curriculum import (
    _runtime_snapshot_from_builder_field,
    canonical_field_payload,
    field_sha256,
    sha256_text,
)
from training.trainer_multitick import (
    MultiTickTrainer,
    _snapshot_texts,
    _texts_dict,
    validate_exact_v4_episode,
)


def deep_diff(a: Any, b: Any, path: str = "") -> list[str]:
    diffs: list[str] = []
    if type(a) != type(b):
        diffs.append(f"{path}: type {type(a)} != {type(b)} ({a!r} vs {b!r})")
        return diffs
    if isinstance(a, dict):
        keys = set(a.keys()) | set(b.keys())
        for key in sorted(keys):
            if key not in a:
                diffs.append(f"{path}.{key}: missing in computed")
            elif key not in b:
                diffs.append(f"{path}.{key}: missing in stored")
            else:
                diffs.extend(deep_diff(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (av, bv) in enumerate(zip(a, b)):
            diffs.extend(deep_diff(av, bv, f"{path}[{i}]"))
    else:
        if a != b:
            diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def diagnose(checkpoint_path: pathlib.Path, dataset_dir: pathlib.Path) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = CoreConfig.from_dict(payload["cfg"])
    core = AxonCore(cfg).to(device)
    core.load_state_dict(payload["core_state"])
    print(f"loaded checkpoint {checkpoint_path.name}: d_model={cfg.d_model}")

    episodes_path = dataset_dir / "episodes.jsonl"
    episodes: list[dict[str, Any]] = []
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    print(f"loaded {len(episodes)} episodes")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    shuffled = episodes.copy()
    random.Random(0).shuffle(shuffled)
    episode = shuffled[0]
    print(f"first shuffled episode: {episode.get('episode_id')}")

    try:
        validated = validate_exact_v4_episode(episode)
        print("validate_exact_v4_episode passed")
        trainer = MultiTickTrainer(core, soul_read_bearing=False)
        result = trainer.forward_episode(episode)
        print(f"forward_episode passed; loss={result.total_loss.item():.4f}")
        optimizer = torch.optim.AdamW(trainer.optimizer_parameters(), lr=1e-4)
        step = trainer.train_episode(episode, optimizer, verify_gradients=True)
        print(f"train_episode passed; loss={step.forward.total_loss.item():.4f}")
        return 0
    except Exception as exc:
        print(f"VALIDATION/TRAINING FAILED: {exc}")

    # Reconstruct runtime snapshot for tick 2 to compare read pages.
    ticks = episode.get("ticks", [])
    if len(ticks) < 3:
        print("episode has fewer than 3 ticks; cannot diagnose tick 2")
        return 1

    tick = ticks[2]
    path = f"ticks[2]"
    initial_field = episode.get("initial_field", {})
    # Build runtime snapshot from initial field + previous ticks.
    from runtime.field import (
        LogicalRegion,
        RegionState,
        SharedFieldSnapshot,
        apply_delta,
    )
    from training.trainer_multitick import _require_mapping

    # Materialize initial snapshot.
    regions = []
    for region_name in episode.get("canonical_regions", []):
        region_data = initial_field.get(region_name, {"text": "", "spans": []})
        regions.append(
            RegionState(
                logical_region=LogicalRegion(region_name),
                text=region_data.get("text", ""),
                spans=[],
            )
        )
    current = SharedFieldSnapshot(tick_id=0, regions=regions)

    read_cursor = FieldViewCursor()
    for idx, t in enumerate(ticks[:3]):
        target_name = t.get("target_region")
        target = LogicalRegion(target_name)
        before_texts = _require_mapping(
            t.get("field_before_texts"), f"ticks[{idx}].field_before_texts"
        )
        if _snapshot_texts(current) != _texts_dict(before_texts):
            print(f"tick {idx}: field_before mismatch")
            print("  runtime:", _snapshot_texts(current))
            print("  expected:", _texts_dict(before_texts))
        read_page = compile_next_read_page(
            current,
            proposal_region=target,
            cursor=read_cursor,
            proposal_offset=t.get("proposal_view_offset", 0),
        )
        raw_read_page = _require_mapping(
            t.get("read_page"), f"ticks[{idx}].read_page"
        )
        expected_read_page = read_page.to_audit_dict()
        print(f"\n--- tick {idx} read_page diff ---")
        for line in deep_diff(dict(raw_read_page), expected_read_page, "read_page"):
            print(line)

        # Advance cursor for next tick.
        read_cursor = read_page.next_cursor
        # Apply the tick delta to current snapshot (simplified).
        delta_data = t.get("delta", {})
        if delta_data:
            from training.trainer_multitick import _teacher_commit
            current = _teacher_commit(current, t)

    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <checkpoint.pt> <dataset-dir>")
        sys.exit(2)
    sys.exit(diagnose(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])))
