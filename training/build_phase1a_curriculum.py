#!/usr/bin/env python3
"""Build the bounded, response-compatible Phase1A curriculum.

Every target is preserved as exact supported characters and explicitly chained
into at-most-64-character response deltas.  Context masking is recorded rather
than hidden.  Source records are assigned wholly to one split before their
chains are emitted, preventing train/dev/test leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from substrate import assert_supported_text


DEFAULT_INPUT = ROOT / "datasets" / "recovered" / "phase0b_curriculum_v1"
DEFAULT_OUTPUT = ROOT / "datasets" / "recovered" / "phase1a_curriculum_v2"
TARGET_CHARS = 64
HISTORY_CHARS = 256
USER_CHARS = 64
ALL_DRAFT_MODES = ("copy", "partial", "blank")
WORD_BOUNDARY_CHARS = frozenset(" \t\r\n\f\v")
SENTENCE_BOUNDARY_CHARS = frozenset(".,;:!?)]}>/\\-_")

FAMILY_SPECS = {
    "runtime_response_chain_v2": "runtime_response_delta_v1",
    "structured_knowledge_response_v2": "structured_knowledge_delta_v1",
    "procedure_next_step_response_v2": "scratchpad_delta_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def exact_text(value: Any) -> str:
    text = "" if value is None else str(value)
    assert_supported_text(text)
    return text


def masked_window(text: str, width: int, *, keep_tail: bool) -> tuple[str, dict[str, int]]:
    if len(text) <= width:
        start, end = 0, len(text)
    elif keep_tail:
        start, end = len(text) - width, len(text)
    else:
        start, end = 0, width
    return text[start:end], {
        "original_chars": len(text),
        "visible_start": start,
        "visible_end": end,
        "masked_chars": len(text) - (end - start),
    }


def response_target(row: dict[str, Any], input_family: str) -> str:
    if input_family == "scratchpad_delta_v1":
        secondary = row.get("secondary_delta")
        if isinstance(secondary, dict) and secondary.get("text"):
            return exact_text(secondary["text"])
    target = row.get("target_delta")
    return exact_text(target.get("text", "") if isinstance(target, dict) else "")


def context_parts(row: dict[str, Any]) -> tuple[str, str]:
    active = row.get("active_field") if isinstance(row.get("active_field"), dict) else {}
    sections: list[str] = []
    for key in ("conversation_history", "structured_knowledge", "scratch"):
        value = exact_text(active.get(key, ""))
        if value:
            sections.append(f"{key}:\n{value}")
    history = "\n".join(sections)
    user_input = exact_text(active.get("user_input", ""))
    return history, user_input


def lossless_target_chunks(
    text: str,
    width: int = TARGET_CHARS,
    *,
    prefer_boundaries: bool = False,
) -> list[tuple[int, int, str, str]]:
    """Return exact, bounded chunks with absolute spans and cut provenance.

    When enabled, boundary preference searches the latter half of each window
    for whitespace or punctuation and cuts immediately after it.  A hard cut
    is used only when that window contains no sensible boundary.  Concatenating
    the returned text always reconstructs the input byte-for-byte in UTF-8.
    """
    if width <= 0:
        raise ValueError("chunk width must be positive")
    chunks: list[tuple[int, int, str, str]] = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + width)
        end = hard_end
        cut_kind = "final" if hard_end == len(text) else "hard"
        if prefer_boundaries and hard_end < len(text):
            search_start = start + max(1, math.ceil(width / 2))
            word_candidates = [
                position
                for position in range(search_start, hard_end + 1)
                if text[position - 1] in WORD_BOUNDARY_CHARS
            ]
            candidates = word_candidates or [
                position
                for position in range(search_start, hard_end + 1)
                if text[position - 1] in SENTENCE_BOUNDARY_CHARS
            ]
            if candidates:
                end = candidates[-1]
                cut_kind = "boundary"
        chunks.append((start, end, text[start:end], cut_kind))
        start = end
    return chunks


def _normalized_visible_target(history: str, user_input: str, target: str) -> str:
    normalize = lambda value: " ".join(value.casefold().split())
    return stable_hash(
        json.dumps(
            [normalize(history), normalize(user_input), normalize(target)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def remove_visible_target(text: str, target: str) -> tuple[str, int]:
    """Remove case-insensitive, token-bounded target leakage from context."""
    needle = target.strip()
    if not text or not needle:
        return text, 0
    left = r"(?<![0-9A-Za-z])" if needle[0].isalnum() else ""
    right = r"(?![0-9A-Za-z])" if needle[-1].isalnum() else ""
    pattern = re.compile(left + re.escape(needle) + right, flags=re.IGNORECASE)
    cleaned, replacements = pattern.subn("", text)
    if not replacements:
        return text, 0
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return cleaned.strip(), replacements


def allowed_draft_modes_for_example(
    *,
    part_index: int,
    visible_history: str,
    visible_user: str,
    history_audit: dict[str, int],
    leakage_replacements: int,
) -> tuple[list[str], list[str]]:
    """Keep blank supervision only where visible evidence can support it."""
    visible_chars = history_audit["visible_end"] - history_audit["visible_start"]
    masked_chars = history_audit["masked_chars"]
    reasons: list[str] = []
    if masked_chars > visible_chars:
        reasons.append("severely_masked_history")
    if part_index > 0 and masked_chars:
        reasons.append("masked_continuation_prefix")
    if not visible_history.strip() and not visible_user.strip():
        reasons.append("no_visible_conditioning_context")
    if leakage_replacements:
        reasons.append("target_leakage_removed")
    allowed = list(ALL_DRAFT_MODES)
    if reasons:
        allowed.remove("blank")
    return allowed, reasons


def build_source_chains(
    input_path: Path,
    input_family: str,
    output_family: str,
    *,
    target_getter: Callable[[dict[str, Any], str], str] = response_target,
    schema: str = "axon_phase1_delta_v2",
    prefer_boundaries: bool = False,
    max_records_per_source: int | None = None,
    clean_visible_target: bool = False,
    deduplicate_visible_target: bool = False,
    emit_allowed_draft_modes: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = {
        "input_rows": 0,
        "empty_targets": 0,
        "unsupported_rows": 0,
        "candidate_records": 0,
        "retained_records": 0,
        "retained_sources": 0,
        "candidate_rows": 0,
        "duplicate_visible_target_records": 0,
        "source_balance_dropped_records": 0,
        "target_leakage_records": 0,
        "target_leakage_replacements": 0,
        "boundary_cuts": 0,
        "hard_cuts": 0,
        "blank_excluded_rows": 0,
    }
    prepared: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_visible_targets: set[str] = set()
    for line_number, row in iter_jsonl(input_path):
        stats["input_rows"] += 1
        try:
            target = target_getter(row, input_family)
            history, user_input = context_parts(row)
        except ValueError:
            stats["unsupported_rows"] += 1
            continue
        if not target:
            stats["empty_targets"] += 1
            continue

        source_key = str(row.get("source") or f"{input_family}:{line_number}")
        leakage_replacements = 0
        if clean_visible_target:
            history, history_replacements = remove_visible_target(history, target)
            user_input, user_replacements = remove_visible_target(user_input, target)
            leakage_replacements = history_replacements + user_replacements
            if leakage_replacements:
                stats["target_leakage_records"] += 1
                stats["target_leakage_replacements"] += leakage_replacements
        visible_target_key = _normalized_visible_target(history, user_input, target)
        if deduplicate_visible_target and visible_target_key in seen_visible_targets:
            stats["duplicate_visible_target_records"] += 1
            continue
        seen_visible_targets.add(visible_target_key)
        prepared[source_key].append(
            {
                "line_number": line_number,
                "target": target,
                "history": history,
                "user_input": user_input,
                "visible_target_key": visible_target_key,
                "leakage_replacements": leakage_replacements,
                "selection_key": stable_hash(
                    f"{output_family}|{source_key}|{line_number}|{stable_hash(target)}"
                ),
            }
        )
        stats["candidate_records"] += 1

    for source_key in sorted(prepared):
        records = prepared[source_key]
        if max_records_per_source is not None:
            if max_records_per_source <= 0:
                raise ValueError("max_records_per_source must be positive")
            records = sorted(records, key=lambda item: item["selection_key"])
            stats["source_balance_dropped_records"] += max(0, len(records) - max_records_per_source)
            records = records[:max_records_per_source]
        records = sorted(records, key=lambda item: item["line_number"])
        stats["retained_records"] += len(records)
        if records:
            stats["retained_sources"] += 1
        for source_rank, record in enumerate(records):
            line_number = int(record["line_number"])
            target = str(record["target"])
            history = str(record["history"])
            user_input = str(record["user_input"])
            leakage_replacements = int(record["leakage_replacements"])
            parts = lossless_target_chunks(
                target,
                TARGET_CHARS,
                prefer_boundaries=prefer_boundaries,
            )
            chain_id = stable_hash(
                f"{output_family}|{source_key}|{line_number}|{stable_hash(target)}"
            )[:24]
            for part_index, (char_start, char_end, part, cut_kind) in enumerate(parts):
                prior = target[:char_start]
                visible_history = history
                if prior:
                    visible_history = (
                        f"{history}\nprevious response:\n{prior}"
                        if history
                        else f"previous response:\n{prior}"
                    )
                visible_history, history_audit = masked_window(
                    visible_history,
                    HISTORY_CHARS,
                    keep_tail=True,
                )
                if part_index:
                    visible_user = f"continue response part {part_index + 1} of {len(parts)}"
                    user_audit = {
                        "original_chars": len(visible_user),
                        "visible_start": 0,
                        "visible_end": len(visible_user),
                        "masked_chars": 0,
                    }
                else:
                    visible_user, user_audit = masked_window(
                        user_input,
                        USER_CHARS,
                        keep_tail=False,
                    )
                assert_supported_text(visible_history)
                assert_supported_text(visible_user)
                assert_supported_text(part)
                allowed_modes, blank_exclusion_reasons = allowed_draft_modes_for_example(
                    part_index=part_index,
                    visible_history=visible_history,
                    visible_user=visible_user,
                    history_audit=history_audit,
                    leakage_replacements=leakage_replacements,
                )
                if cut_kind == "boundary":
                    stats["boundary_cuts"] += 1
                elif cut_kind == "hard":
                    stats["hard_cuts"] += 1
                if "blank" not in allowed_modes:
                    stats["blank_excluded_rows"] += 1
                example_id = stable_hash(f"{chain_id}|{char_start}|{char_end}")[:24]
                output_row = {
                    "schema": schema,
                    "family": output_family,
                    "example_id": example_id,
                    "split": "",
                    "source": {
                        "record_id": source_key,
                        "input_family": input_family,
                        "input_line": line_number,
                        "source_balance_rank": source_rank,
                        "char_start": char_start,
                        "char_end": char_end,
                    },
                    "active_field": {
                        "conversation_history": visible_history,
                        "user_input": visible_user,
                        "structured_knowledge": "",
                        "response_draft": "",
                    },
                    "target_delta": {"region": "response_draft", "op": "replace", "text": part},
                    "chain": {
                        "chain_id": chain_id,
                        "part_index": part_index,
                        "part_count": len(parts),
                        "cut_kind": cut_kind,
                    },
                    "budget_audit": {
                        "silent_clips": 0,
                        "unsupported_substitutions": 0,
                        "history": history_audit,
                        "user_input": user_audit,
                    },
                    "quality_audit": {
                        "visible_target_key": record["visible_target_key"],
                        "target_leakage_replacements": leakage_replacements,
                        "blank_exclusion_reasons": blank_exclusion_reasons,
                    },
                }
                if emit_allowed_draft_modes:
                    output_row["allowed_draft_modes"] = allowed_modes
                groups[source_key].append(output_row)
                stats["candidate_rows"] += 1
    return groups, stats


def allocate_splits(
    groups: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    remaining = sorted(groups, key=lambda key: stable_hash(f"{seed}|{key}"))
    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in quotas}
    for split, quota in quotas.items():
        while len(assigned[split]) < quota:
            need = quota - len(assigned[split])
            chosen_at = next((i for i, key in enumerate(remaining) if len(groups[key]) <= need), None)
            if chosen_at is None:
                raise ValueError(f"cannot fill {split} quota {quota} with complete source chains")
            key = remaining.pop(chosen_at)
            rows = groups[key]
            for row in rows:
                row["split"] = split
            assigned[split].extend(rows)
    return assigned


def build_curriculum(input_dir: Path, output_dir: Path, *, per_family: int, seed: int) -> dict[str, Any]:
    train_count = int(per_family * 0.8)
    dev_count = int(per_family * 0.1)
    quotas = {"train": train_count, "dev": dev_count, "test": per_family - train_count - dev_count}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "axon_phase1a_curriculum_manifest_v2",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "phase1a_bounded_delta_competence",
        "bootstrap_window": {
            "history_chars": HISTORY_CHARS,
            "user_chars": USER_CHARS,
            "response_chars": TARGET_CHARS,
            "total_chars": HISTORY_CHARS + USER_CHARS + TARGET_CHARS,
            "doctrine": False,
        },
        "private_state_claim": False,
        "quotas_per_family": quotas,
        "families": {},
    }
    for output_family, input_family in FAMILY_SPECS.items():
        input_path = input_dir / f"{input_family}.jsonl"
        groups, stats = build_source_chains(input_path, input_family, output_family)
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
            "counts": {split: len(rows) for split, rows in assigned.items()},
            "source_groups": {split: len({row["source"]["record_id"] for row in rows}) for split, rows in assigned.items()},
            "build_stats": stats,
        }
    manifest_path = output_dir / "phase1a_curriculum_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact, source-split Axon Phase1A curricula")
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
