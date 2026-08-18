#!/usr/bin/env python3
"""Build the cleaned, lossless Phase0b exact-v3 continuation curriculum.

The legacy Phase0b sources contain long exact deltas.  This builder preserves
every character of each selected source chain in <=64-character response
targets, prefers word/punctuation boundaries, balances repeated source records,
and assigns whole source chains to one split before emission.  Structured rows
are de-leaked and de-duplicated before splitting.  It makes no native scratch-
region or private-state claim; all supervised characters remain on the
checkpoint-compatible response slice.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.build_phase1a_curriculum import (
    HISTORY_CHARS,
    TARGET_CHARS,
    USER_CHARS,
    SENTENCE_BOUNDARY_CHARS,
    WORD_BOUNDARY_CHARS,
    allocate_splits,
    build_source_chains,
    sha256_file,
)
from substrate import assert_supported_text

DEFAULT_INPUT = ROOT / "datasets" / "recovered" / "phase0b_curriculum_v1"
DEFAULT_OUTPUT = ROOT / "datasets" / "recovered" / "phase0b_exact_curriculum_v3"
MAX_RECORDS_PER_SOURCE = 4
MAX_EXAMPLES_PER_SOURCE = 24

FAMILY_SPECS = {
    "runtime_response_delta_chain_v3": "runtime_response_delta_v1",
    "structured_knowledge_delta_chain_v3": "structured_knowledge_delta_v1",
    "scratchpad_delta_chain_v3": "scratchpad_delta_v1",
}


def exact_legacy_target(row: dict[str, Any], _input_family: str) -> str:
    target = row.get("target_delta") if isinstance(row.get("target_delta"), dict) else {}
    text = "" if target.get("text") is None else str(target.get("text"))
    return assert_supported_text(text)


def split_quotas(total: int) -> dict[str, int]:
    train_count = int(total * 0.8)
    dev_count = int(total * 0.1)
    return {"train": train_count, "dev": dev_count, "test": total - train_count - dev_count}


def cap_emitted_source_groups(
    groups: dict[str, list[dict[str, Any]]],
    max_examples: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Exclude over-cap source groups without truncating any response chain."""
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    eligible: dict[str, list[dict[str, Any]]] = {}
    excluded_groups = 0
    excluded_rows = 0
    for source_key, rows in groups.items():
        if len(rows) > max_examples:
            excluded_groups += 1
            excluded_rows += len(rows)
            continue
        eligible[source_key] = rows
    return eligible, {
        "emitted_source_cap": max_examples,
        "emitted_source_cap_excluded_groups": excluded_groups,
        "emitted_source_cap_excluded_rows": excluded_rows,
        "emitted_source_cap_eligible_groups": len(eligible),
        "emitted_source_cap_eligible_rows": sum(len(rows) for rows in eligible.values()),
    }


def audit_selected_rows(assigned: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Build a deterministic post-selection audit for the persisted manifest."""
    rows = [row for split in ("train", "dev", "test") for row in assigned[split]]
    allowed_modes: collections.Counter[str] = collections.Counter()
    cut_kinds: collections.Counter[str] = collections.Counter()
    source_counts: dict[str, collections.Counter[str]] = {
        split: collections.Counter() for split in ("train", "dev", "test")
    }
    chains: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    silent_clips = 0
    unsupported_substitutions = 0
    for row in rows:
        allowed_modes["/".join(row["allowed_draft_modes"])] += 1
        cut_kinds[str(row["chain"]["cut_kind"])] += 1
        source_counts[row["split"]][row["source"]["record_id"]] += 1
        chains[row["chain"]["chain_id"]].append(row)
        silent_clips += int(row["budget_audit"]["silent_clips"])
        unsupported_substitutions += int(row["budget_audit"]["unsupported_substitutions"])

    broken_chains = 0
    midword_boundaries = 0
    avoidable_midword_boundaries = 0
    for chain_rows in chains.values():
        chain_rows.sort(key=lambda row: row["chain"]["part_index"])
        reconstructed = "".join(row["target_delta"]["text"] for row in chain_rows)
        spans_are_contiguous = all(
            row["source"]["char_start"]
            == (0 if index == 0 else chain_rows[index - 1]["source"]["char_end"])
            for index, row in enumerate(chain_rows)
        )
        if (
            len(chain_rows) != chain_rows[0]["chain"]["part_count"]
            or not spans_are_contiguous
            or chain_rows[-1]["source"]["char_end"] != len(reconstructed)
        ):
            broken_chains += 1
        for index in range(1, len(chain_rows)):
            previous = chain_rows[index - 1]
            current = chain_rows[index]
            cut = int(current["source"]["char_start"])
            if not (reconstructed[cut - 1].isalnum() and reconstructed[cut].isalnum()):
                continue
            midword_boundaries += 1
            chunk_start = int(previous["source"]["char_start"])
            hard_end = min(len(reconstructed), chunk_start + TARGET_CHARS)
            search_start = chunk_start + max(1, math.ceil(TARGET_CHARS / 2))
            sensible_boundary_exists = any(
                reconstructed[position - 1] in WORD_BOUNDARY_CHARS
                for position in range(search_start, hard_end + 1)
            ) or any(
                reconstructed[position - 1] in SENTENCE_BOUNDARY_CHARS
                for position in range(search_start, hard_end + 1)
            )
            if sensible_boundary_exists:
                avoidable_midword_boundaries += 1

    split_source_stats: dict[str, dict[str, int | float]] = {}
    split_sources = {split: set(counts) for split, counts in source_counts.items()}
    for split, counts in source_counts.items():
        values = sorted(counts.values())
        split_source_stats[split] = {
            "rows": sum(values),
            "sources": len(values),
            "min_rows_per_source": min(values),
            "median_rows_per_source": statistics.median(values),
            "mean_rows_per_source": sum(values) / len(values),
            "p95_rows_per_source": values[math.ceil(0.95 * len(values)) - 1],
            "max_rows_per_source": max(values),
        }
    return {
        "rows": len(rows),
        "chains": len(chains),
        "allowed_draft_modes": dict(sorted(allowed_modes.items())),
        "cut_kinds": dict(sorted(cut_kinds.items())),
        "broken_chains": broken_chains,
        "midword_boundaries": midword_boundaries,
        "avoidable_midword_boundaries": avoidable_midword_boundaries,
        "silent_clips": silent_clips,
        "unsupported_substitutions": unsupported_substitutions,
        "split_source_stats": split_source_stats,
        "source_split_overlap": {
            "train_dev": len(split_sources["train"] & split_sources["dev"]),
            "train_test": len(split_sources["train"] & split_sources["test"]),
            "dev_test": len(split_sources["dev"] & split_sources["test"]),
        },
    }


def build_curriculum(input_dir: Path, output_dir: Path, *, per_family: int, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "axon_phase0b_exact_curriculum_manifest_v3",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "phase0b_exact_v3_cleaned_alignment",
        "bootstrap_window": {
            "history_chars": HISTORY_CHARS,
            "user_chars": USER_CHARS,
            "response_chars": TARGET_CHARS,
            "total_chars": HISTORY_CHARS + USER_CHARS + TARGET_CHARS,
            "doctrine": False,
        },
        "supervised_region": "response_draft",
        "native_nonresponse_region_claim": False,
        "private_state_claim": False,
        "requested_examples_per_family": per_family,
        "source_balance": {
            "max_input_records_per_source": MAX_RECORDS_PER_SOURCE,
            "max_emitted_examples_per_source": MAX_EXAMPLES_PER_SOURCE,
            "over_cap_action": "exclude_complete_source_group",
            "truncate_chains": False,
        },
        "chunk_policy": {
            "lossless": True,
            "prefer_word_or_punctuation_boundary": True,
            "hard_cut_only_without_boundary_in_latter_half": True,
        },
        "structured_policy": {
            "visible_target_cleanup": True,
            "visible_context_target_dedup": True,
        },
        "draft_mode_policy": {
            "per_example": True,
            "all_modes": ["copy", "partial", "blank"],
            "blank_excluded_when_masked_or_underdetermined": True,
        },
        "training_sampler_contract": {
            "schema": "axon_weighted_family_sampler_v1",
            "strategy": "weighted_family_then_uniform_example",
            "row_duplication_for_balance": False,
            "required": True,
            "family_weights": {
                "runtime_response_delta_chain_v3": 0.30,
                "structured_knowledge_delta_chain_v3": 0.40,
                "scratchpad_delta_chain_v3": 0.30,
            },
        },
        "families": {},
    }
    for output_family, input_family in FAMILY_SPECS.items():
        input_path = input_dir / f"{input_family}.jsonl"
        structured = input_family == "structured_knowledge_delta_v1"
        groups, stats = build_source_chains(
            input_path,
            input_family,
            output_family,
            target_getter=exact_legacy_target,
            schema="axon_phase0b_delta_v3",
            prefer_boundaries=True,
            max_records_per_source=MAX_RECORDS_PER_SOURCE,
            clean_visible_target=structured,
            deduplicate_visible_target=structured,
            emit_allowed_draft_modes=True,
        )
        groups, emitted_cap_stats = cap_emitted_source_groups(groups, MAX_EXAMPLES_PER_SOURCE)
        stats.update(emitted_cap_stats)
        available_rows = sum(len(rows) for rows in groups.values())
        effective_rows = min(per_family, available_rows)
        quotas = split_quotas(effective_rows)
        assigned = allocate_splits(groups, quotas, seed=seed)
        output_path = output_dir / f"{output_family}.jsonl"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for split in ("train", "dev", "test"):
                for row in assigned[split]:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        manifest["families"][output_family] = {
            "input": str(input_path),
            "input_sha256": sha256_file(input_path),
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "requested_rows": per_family,
            "available_rows": available_rows,
            "effective_rows": effective_rows,
            "quotas": quotas,
            "counts": {split: len(rows) for split, rows in assigned.items()},
            "source_groups": {
                split: len({row["source"]["record_id"] for row in rows})
                for split, rows in assigned.items()
            },
            "build_stats": stats,
            "selected_audit": audit_selected_rows(assigned),
        }
    manifest_path = output_dir / "phase0b_exact_curriculum_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact chained Axon Phase0b curricula")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--per-family", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    per_family = min(args.per_family, 30) if args.smoke else args.per_family
    manifest = build_curriculum(Path(args.input_dir), Path(args.output_dir), per_family=per_family, seed=args.seed)
    print(json.dumps({family: info["counts"] for family, info in manifest["families"].items()}, indent=2))
    print(Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
