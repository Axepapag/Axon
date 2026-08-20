#!/usr/bin/env python3
"""Observable trainer for Axon's complete-field 64D R0 core."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from substrate import assert_supported_text, default_alphabet, roundtrip_check
from runtime.field import D64_COMPILER_SCHEMA, D64FieldCompiler, LogicalRegion, SharedFieldSnapshot, apply_compiled_delta, replacement_delta
from training.canonical_d64 import snapshot_from_r0_record
from training.complete_field_64d import (
    CompleteField64D,
    REGION_ORDER,
    ReaderConfig,
    canonical_field,
    coverage_manifest_from_compiled,
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if split is not None and record.get("split") != split:
                continue
            if record.get("schema") != "axon-complete-field-r0-example-v3":
                raise ValueError(f"{path}:{line_number}: V6 aligned R0 schema v3 is required")
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
            alignment = record.get("alignment")
            if not isinstance(alignment, Mapping) or alignment.get("schema") != "axon-r0-source-alignment-v1":
                raise ValueError(f"{path}:{line_number}: V6 source alignment is required")
            if set(alignment) != {"schema", "scratch", "response_draft", "response_counterfactuals"}:
                raise ValueError(f"{path}:{line_number}: malformed V6 source alignment")
            if set(alignment["response_counterfactuals"]) != {
                item["variant_id"] for item in counterfactuals
            }:
                raise ValueError(f"{path}:{line_number}: counterfactual alignment ids mismatch")
            for target_name in ("scratch", "response_draft"):
                target_alignment = alignment[target_name]
                if (
                    not isinstance(target_alignment, Mapping)
                    or target_alignment.get("schema") != "axon-r0-target-alignment-v1"
                    or not isinstance(target_alignment.get("segments"), list)
                    or target_alignment.get("supervise_eos_generate") is not True
                ):
                    raise ValueError(f"{path}:{line_number}: malformed {target_name} alignment")
            for variant_id, target_alignment in alignment["response_counterfactuals"].items():
                if (
                    not isinstance(target_alignment, Mapping)
                    or target_alignment.get("schema") != "axon-r0-target-alignment-v1"
                    or not isinstance(target_alignment.get("segments"), list)
                    or target_alignment.get("supervise_eos_generate") is not True
                ):
                    raise ValueError(
                        f"{path}:{line_number}: malformed counterfactual alignment {variant_id}"
                    )
            if record.get("write_authority", {}).get("diary") is not False:
                raise ValueError(f"{path}:{line_number}: diary must be disabled in R0")
            record["field"] = field
            record["response_counterfactuals"] = counterfactuals
            records.append(record)
    if not records:
        raise ValueError(f"no records loaded from {path}")
    return records


CANONICAL_ANATOMY = "canonical"
_D64_COMPILER = D64FieldCompiler()


def _alignment_for_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": record["alignment"]["schema"],
        "scratch": record["alignment"]["scratch"],
        "response_draft": record["alignment"]["response_draft"],
    }


def _forward_record(
    model: CompleteField64D,
    record: Mapping[str, Any],
    *,
    teacher_forcing_ratio: float = 1.0,
) -> dict[str, Any]:
    snapshot = snapshot_from_r0_record(record)
    return model.forward_canonical_transaction(
        snapshot,
        record["targets"]["scratch"],
        record["targets"]["response_draft"],
        teacher_forcing_ratio=teacher_forcing_ratio,
        alignment=_alignment_for_record(record),
        compiler=_D64_COMPILER,
    )


def _run_record(model: CompleteField64D, record: Mapping[str, Any]) -> dict[str, Any]:
    return model.run_canonical_transaction(
        snapshot_from_r0_record(record), compiler=_D64_COMPILER
    )


def _snapshot_with_scratch(
    record: Mapping[str, Any],
    scratch: str,
) -> SharedFieldSnapshot:
    snapshot = snapshot_from_r0_record(record)
    current = snapshot.region(LogicalRegion.SCRATCH).text
    if current == scratch:
        return snapshot
    compiled = _D64_COMPILER.compile(snapshot)
    delta = replacement_delta(
        snapshot,
        compiled,
        region=LogicalRegion.SCRATCH,
        text=scratch,
        author_core_id="training-intervention",
        pass_id="scratch-intervention",
        provenance="canonical_d64_training_intervention",
    )
    return apply_compiled_delta(snapshot, compiled, delta)


def _read_record_with_scratch(
    model: CompleteField64D,
    record: Mapping[str, Any],
    scratch: str,
):
    snapshot = _snapshot_with_scratch(record, scratch)
    state, memory, coverage, _ = model.read_snapshot_with_memory(
        snapshot, compiler=_D64_COMPILER
    )
    return state, memory, coverage


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
        snapshot = snapshot_from_r0_record(record)
        compiled = _D64_COMPILER.compile(snapshot)
        hashes = set()
        for page_size in page_sizes:
            manifest = coverage_manifest_from_compiled(compiled, page_size)
            if not manifest.complete or manifest.expected_characters != expected:
                raise RuntimeError(f"coverage gate failed for {record['example_id']} at page_size={page_size}")
            hashes.add(manifest.field_sha256)
        if hashes != {snapshot.field_id}:
            raise RuntimeError("canonical field identity changed across physical page sizes")
        if (
            not compiled.coverage.complete
            or compiled.coverage.expected_active_characters != expected
            or compiled.coverage.compiled_active_characters != expected
        ):
            raise RuntimeError(
                f"canonical D64 compiler coverage failed for {record['example_id']}"
            )


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
    aligned_copy_positions = 0
    aligned_position_correct = 0
    aligned_gate_positions = 0
    aligned_gate_correct = 0
    counterfactual_aligned_copy_positions = 0
    counterfactual_aligned_position_correct = 0
    counterfactual_aligned_gate_positions = 0
    counterfactual_aligned_gate_correct = 0
    samples = stratified_records(records, max(1, max_examples))
    for index, record in enumerate(samples):
        out = _forward_record(model, record)
        supervision = out["alignment"]
        aligned_copy_positions += int(supervision["copy_positions"])
        aligned_position_correct += int(supervision["position_correct"])
        aligned_gate_positions += int(supervision["gate_supervised_positions"])
        aligned_gate_correct += int(supervision["gate_correct"])
        sl = sequence_cross_entropy(out["scratch_logits"], out["scratch_targets"])
        rl = sequence_cross_entropy(out["response_logits"], out["response_targets"])
        scratch_losses.append(float(sl.item()))
        response_losses.append(float(rl.item()))
        scratch_acc.append(teacher_char_accuracy(out["scratch_logits"], out["scratch_targets"]))
        response_acc.append(teacher_char_accuracy(out["response_logits"], out["response_targets"]))
        if index < causal_examples:
            for variant, collector in (("", empty_deltas), (corruption(record["targets"]["scratch"]), corrupt_deltas)):
                state, memory, _ = _read_record_with_scratch(model, record, variant)
                logits, targets = model.decode_teacher(
                    state,
                    record["targets"]["response_draft"],
                    head=1,
                    memory=memory,
                )
                collector.append(float(sequence_cross_entropy(logits, targets).item() - rl.item()))
        for counterfactual in record["response_counterfactuals"]:
            if len(counterfactual_losses) >= causal_examples:
                break
            state, memory, _ = _read_record_with_scratch(
                model, record, counterfactual["scratch"]
            )
            logits, targets, decoder_alignment = model.decode_teacher(
                state,
                counterfactual["response_draft"],
                head=1,
                memory=memory,
                return_alignment=True,
            )
            counterfactual_losses.append(float(sequence_cross_entropy(logits, targets).item()))
            counterfactual_acc.append(teacher_char_accuracy(logits, targets))
            counterfactual_supervision = model.alignment_supervision(
                target_text=counterfactual["response_draft"],
                memory=memory,
                decoder_alignment=decoder_alignment,
                specification=record["alignment"]["response_counterfactuals"][
                    counterfactual["variant_id"]
                ],
            )
            counterfactual_aligned_copy_positions += int(
                counterfactual_supervision["copy_positions"]
            )
            counterfactual_aligned_position_correct += int(
                counterfactual_supervision["position_correct"]
            )
            counterfactual_aligned_gate_positions += int(
                counterfactual_supervision["gate_supervised_positions"]
            )
            counterfactual_aligned_gate_correct += int(
                counterfactual_supervision["gate_correct"]
            )
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
        "aligned_copy_positions": aligned_copy_positions,
        "aligned_position_accuracy": (
            aligned_position_correct / aligned_copy_positions if aligned_copy_positions else 1.0
        ),
        "aligned_gate_positions": aligned_gate_positions,
        "aligned_gate_accuracy": (
            aligned_gate_correct / aligned_gate_positions if aligned_gate_positions else 1.0
        ),
        "counterfactual_aligned_copy_positions": counterfactual_aligned_copy_positions,
        "counterfactual_aligned_position_accuracy": (
            counterfactual_aligned_position_correct / counterfactual_aligned_copy_positions
            if counterfactual_aligned_copy_positions
            else 1.0
        ),
        "counterfactual_aligned_gate_positions": counterfactual_aligned_gate_positions,
        "counterfactual_aligned_gate_accuracy": (
            counterfactual_aligned_gate_correct / counterfactual_aligned_gate_positions
            if counterfactual_aligned_gate_positions
            else 1.0
        ),
    }
    model.train()
    return result


@torch.no_grad()
def observable_samples(model: CompleteField64D, records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    model.eval()
    output: list[dict[str, Any]] = []
    for record in stratified_records(records, count):
        transaction = _run_record(model, record)
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
        correct_state, correct_memory, _ = _read_record_with_scratch(
            model, record, record["targets"]["scratch"]
        )
        correct_response, correct_terminated = model.decode_greedy(
            correct_state,
            head=1,
            memory=correct_memory,
        )
        for counterfactual in record["response_counterfactuals"]:
            counterfactual_state, counterfactual_memory, _ = _read_record_with_scratch(
                model, record, counterfactual["scratch"]
            )
            counterfactual_response, counterfactual_terminated = model.decode_greedy(
                counterfactual_state,
                head=1,
                memory=counterfactual_memory,
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
                    "correct_exact": correct_response == record["targets"]["response_draft"],
                    "counterfactual_exact": (
                        counterfactual_response == counterfactual["response_draft"]
                    ),
                    "authority_preserved": (
                        counterfactual["response_draft"] == record["targets"]["response_draft"]
                        and counterfactual_response == record["targets"]["response_draft"]
                    ),
                }
            )
            if len(output) >= count:
                model.train()
                return output
    model.train()
    return output


@torch.no_grad()
def evaluate_v6_alignment_behavior(
    model: CompleteField64D,
    records: list[dict[str, Any]],
    max_examples: int = 64,
) -> dict[str, Any]:
    """Hard held-out binding gate for the v6 anti-shortcut family."""
    model.eval()
    rows = [record for record in records if record["family"] == "v6_alignment_retrieval"]
    rows = rows[: min(max_examples, len(rows))]
    if not rows:
        model.train()
        return {"examples": 0, "passed": False}

    scratch_exact = 0
    response_exact = 0
    scratch_terminated = 0
    response_terminated = 0
    base_position_correct = 0
    base_copy_positions = 0
    base_gate_correct = 0
    base_gate_positions = 0
    cf_response_exact = 0
    cf_terminated = 0
    cf_position_correct = 0
    cf_copy_positions = 0
    cf_gate_correct = 0
    cf_gate_positions = 0
    cf_count = 0

    for record in rows:
        transaction = _run_record(model, record)
        scratch_exact += int(transaction["scratch"] == record["targets"]["scratch"])
        response_exact += int(
            transaction["response_draft"] == record["targets"]["response_draft"]
        )
        scratch_terminated += int(transaction["scratch_terminated"])
        response_terminated += int(transaction["response_terminated"])

        teacher = _forward_record(model, record)
        base_position_correct += int(teacher["alignment"]["position_correct"])
        base_copy_positions += int(teacher["alignment"]["copy_positions"])
        base_gate_correct += int(teacher["alignment"]["gate_correct"])
        base_gate_positions += int(teacher["alignment"]["gate_supervised_positions"])

        for counterfactual in record["response_counterfactuals"]:
            state, memory, _ = _read_record_with_scratch(
                model, record, counterfactual["scratch"]
            )
            response, terminated = model.decode_greedy(state, head=1, memory=memory)
            cf_response_exact += int(response == counterfactual["response_draft"])
            cf_terminated += int(terminated)
            cf_count += 1

            _, _, decoder_alignment = model.decode_teacher(
                state,
                counterfactual["response_draft"],
                head=1,
                memory=memory,
                return_alignment=True,
            )
            supervision = model.alignment_supervision(
                target_text=counterfactual["response_draft"],
                memory=memory,
                decoder_alignment=decoder_alignment,
                specification=record["alignment"]["response_counterfactuals"][
                    counterfactual["variant_id"]
                ],
            )
            cf_position_correct += int(supervision["position_correct"])
            cf_copy_positions += int(supervision["copy_positions"])
            cf_gate_correct += int(supervision["gate_correct"])
            cf_gate_positions += int(supervision["gate_supervised_positions"])

    result = {
        "examples": len(rows),
        "scratch_exact_rate": scratch_exact / len(rows),
        "response_exact_rate": response_exact / len(rows),
        "scratch_termination_rate": scratch_terminated / len(rows),
        "response_termination_rate": response_terminated / len(rows),
        "position_accuracy": (
            base_position_correct / base_copy_positions if base_copy_positions else 1.0
        ),
        "copy_gate_accuracy": (
            base_gate_correct / base_gate_positions if base_gate_positions else 1.0
        ),
        "counterfactual_examples": cf_count,
        "counterfactual_response_exact_rate": cf_response_exact / cf_count if cf_count else 0.0,
        "counterfactual_termination_rate": cf_terminated / cf_count if cf_count else 0.0,
        "counterfactual_position_accuracy": (
            cf_position_correct / cf_copy_positions if cf_copy_positions else 1.0
        ),
        "counterfactual_copy_gate_accuracy": (
            cf_gate_correct / cf_gate_positions if cf_gate_positions else 1.0
        ),
    }
    result["passed"] = bool(
        result["scratch_exact_rate"] == 1.0
        and result["response_exact_rate"] == 1.0
        and result["scratch_termination_rate"] == 1.0
        and result["response_termination_rate"] == 1.0
        and result["position_accuracy"] == 1.0
        and result["copy_gate_accuracy"] == 1.0
        and result["counterfactual_response_exact_rate"] == 1.0
        and result["counterfactual_termination_rate"] == 1.0
        and result["counterfactual_position_accuracy"] == 1.0
        and result["counterfactual_copy_gate_accuracy"] == 1.0
    )
    model.train()
    return result


def checkpoint_payload(
    model: CompleteField64D,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config: ReaderConfig,
    args: argparse.Namespace,
    baseline: Mapping[str, Any],
    sampler_state: object,
    dataset_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": "axon-complete-field-r0-checkpoint-v6",
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "reader_config": asdict(config),
        "trainer_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "baseline": dict(baseline),
        "dataset_sha256": dict(dataset_sha256),
        "anatomy": {
            "mode": CANONICAL_ANATOMY,
            "compiler_schema": D64_COMPILER_SCHEMA,
        },
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
    dataset_sha256: Mapping[str, str],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / f"ckpt_{step:09d}.pt"
    temporary = final.with_suffix(".pt.tmp")
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            scaler,
            step,
            config,
            args,
            baseline,
            sampler_state,
            dataset_sha256,
        ),
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
        archive_dir = run_dir / "checkpoint_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = archive_dir / stale.name
        if archived.exists():
            raise FileExistsError(
                f"refusing to overwrite archived checkpoint {archived}"
            )
        os.replace(stale, archived)
    return final


def restore(
    path: Path,
    model: CompleteField64D,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    expected_dataset_sha256: Mapping[str, str],
    expected_anatomy: str | None = None,
) -> tuple[int, dict[str, Any], object | None]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "axon-complete-field-r0-checkpoint-v6":
        raise ValueError(f"unsupported checkpoint schema in {path}")
    if payload.get("dataset_sha256") != dict(expected_dataset_sha256):
        raise ValueError(f"dataset fingerprint mismatch in {path}; refusing unsafe resume")
    if expected_anatomy is not None:
        anatomy = payload.get("anatomy")
        if not isinstance(anatomy, Mapping) or anatomy.get("mode") != expected_anatomy:
            raise ValueError(
                f"checkpoint anatomy mismatch in {path}; refusing unsafe resume"
            )
        expected_compiler = (
            D64_COMPILER_SCHEMA if expected_anatomy == "canonical" else None
        )
        if anatomy.get("compiler_schema") != expected_compiler:
            raise ValueError(
                f"checkpoint compiler schema mismatch in {path}; refusing unsafe resume"
            )
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scaler.load_state_dict(payload.get("scaler_state", {}))
    rng = payload.get("rng", {})
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    return int(payload["step"]), dict(payload.get("baseline", {})), rng.get("sampler")


def newest_checkpoint(run_dir: Path) -> Path | None:
    paths = sorted(run_dir.glob("ckpt_*.pt")) if run_dir.exists() else []
    return paths[-1] if paths else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the observable complete-field 64D R0 core")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run/checkpoint directory; must resolve beneath D:/Axon/State/training/runs",
    )
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--max-output-chars", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=1.0)
    parser.add_argument("--causal-weight", type=float, default=0.50)
    parser.add_argument("--causal-every", type=int, default=1)
    parser.add_argument("--alignment-weight", type=float, default=1.0)
    parser.add_argument("--copy-gate-weight", type=float, default=0.50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--counterfactual-sample-count", type=int, default=16)
    parser.add_argument("--v6-eval-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=64018)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-if-available", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path(r"D:\\Axon\\State"),
        help="canonical Axon State root; canonical training workspace is created beneath State/training/runs",
    )
    args = parser.parse_args()

    expected_state_root = Path(r"D:\Axon\State").resolve(strict=False)
    actual_state_root = args.state_root.resolve(strict=False)
    if actual_state_root != expected_state_root:
        raise RuntimeError(
            f"canonical D64 training must use the real Axon State root "
            f"{expected_state_root}; got {actual_state_root}"
        )
    training_runs_root = (expected_state_root / "training" / "runs").resolve(strict=False)
    actual_run_dir = args.run_dir.resolve(strict=False)
    try:
        relative_run = actual_run_dir.relative_to(training_runs_root)
    except ValueError as exc:
        raise RuntimeError(
            f"canonical D64 run-dir must be beneath {training_runs_root}; got {actual_run_dir}"
        ) from exc
    if not relative_run.parts:
        raise RuntimeError("canonical D64 run-dir must name a run below State/training/runs")
    args.run_dir = actual_run_dir

    if (
        args.steps < 1
        or args.grad_accum < 1
        or args.causal_every < 1
        or args.counterfactual_sample_count < 1
        or args.v6_eval_examples < 1
        or args.alignment_weight < 0.0
        or args.copy_gate_weight < 0.0
        or not 0.0 <= args.teacher_forcing_ratio <= 1.0
    ):
        raise ValueError(
            "positive counts and a teacher_forcing_ratio between zero and one are required"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    substrate_gate()
    train_records = load_jsonl(args.train)
    eval_records = load_jsonl(args.eval)
    dataset_sha256 = {
        "train": file_sha256(args.train),
        "eval": file_sha256(args.eval),
    }
    coverage_gate(train_records)
    coverage_gate(eval_records)
    state_workspace = args.run_dir
    state_workspace.mkdir(parents=True, exist_ok=True)
    atomic_json(
        state_workspace / "anatomy.json",
        {
            "schema": "axon-canonical-d64-training-workspace-v1",
            "compiler_schema": D64_COMPILER_SCHEMA,
            "state_root": str(args.state_root.resolve(strict=False)),
            "workspace": str(state_workspace),
            "run_dir": str(args.run_dir),
            "dataset_sha256": dataset_sha256,
            "core_input_authority": "SharedFieldSnapshot",
            "delta_authority": "shared-field-delta-v1",
            "curriculum_role": "source-material-only",
        },
    )
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
            "schema": "axon-complete-field-r0-run-config-v6",
            "reader": asdict(config),
            "trainer": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "causal_train_records": len(causal_train_records),
            "causal_eval_records": len(causal_eval_records),
            "dataset_sha256": dataset_sha256,
            "anatomy": {
                "mode": CANONICAL_ANATOMY,
                "compiler_schema": D64_COMPILER_SCHEMA,
                "state_workspace": str(state_workspace),
                "core_input_authority": "SharedFieldSnapshot",
            },
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
            step, baseline, sampler_state = restore(
                checkpoint,
                model,
                optimizer,
                scaler,
                device,
                dataset_sha256,
                expected_anatomy=CANONICAL_ANATOMY,
            )
            print(f"resumed {checkpoint} at step {step}", flush=True)
        else:
            print("no checkpoint found; starting a fresh run", flush=True)
    if not baseline:
        baseline = evaluate_teacher(model, eval_records, args.eval_examples)
        baseline["step"] = step
        atomic_json(args.run_dir / "baseline.json", baseline)
    latest_evaluation = dict(baseline)

    rng = random.Random()
    if sampler_state is None:
        rng.seed(args.seed + step)
    else:
        rng.setstate(sampler_state)
    recent_losses: list[float] = []
    run_start_step = step
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    status = "running"
    try:
        while step < args.steps:
            record = train_records[rng.randrange(len(train_records))]
            autocast = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)
            with autocast:
                out = _forward_record(
                    model,
                    record,
                    teacher_forcing_ratio=args.teacher_forcing_ratio,
                )
                scratch_loss = sequence_cross_entropy(out["scratch_logits"], out["scratch_targets"])
                response_loss = sequence_cross_entropy(out["response_logits"], out["response_targets"])
                alignment_position_loss = out["alignment"]["position_loss"]
                alignment_gate_loss = out["alignment"]["gate_loss"]
                alignment_loss = (
                    alignment_position_loss + args.copy_gate_weight * alignment_gate_loss
                )
                causal_loss = torch.zeros((), device=device)
                causal_alignment_position_loss = torch.zeros((), device=device)
                causal_alignment_gate_loss = torch.zeros((), device=device)
                causal_alignment_loss = torch.zeros((), device=device)
                causal_accuracy = 0.0
                causal_position_accuracy = 1.0
                causal_gate_accuracy = 1.0
                causal_example_id: str | None = None
                causal_variant_id: str | None = None
                if args.causal_weight > 0 and step % args.causal_every == 0:
                    causal_record = causal_train_records[rng.randrange(len(causal_train_records))]
                    counterfactuals = causal_record["response_counterfactuals"]
                    counterfactual = counterfactuals[rng.randrange(len(counterfactuals))]
                    causal_state, causal_memory, _ = _read_record_with_scratch(
                        model, causal_record, counterfactual["scratch"]
                    )
                    causal_logits, causal_targets, causal_decoder_alignment = model.decode_scheduled(
                        causal_state,
                        counterfactual["response_draft"],
                        head=1,
                        teacher_forcing_ratio=args.teacher_forcing_ratio,
                        memory=causal_memory,
                        return_alignment=True,
                    )
                    causal_loss = sequence_cross_entropy(causal_logits, causal_targets)
                    causal_supervision = model.alignment_supervision(
                        target_text=counterfactual["response_draft"],
                        memory=causal_memory,
                        decoder_alignment=causal_decoder_alignment,
                        specification=causal_record["alignment"]["response_counterfactuals"][
                            counterfactual["variant_id"]
                        ],
                    )
                    causal_alignment_position_loss = causal_supervision["position_loss"]
                    causal_alignment_gate_loss = causal_supervision["gate_loss"]
                    causal_alignment_loss = (
                        causal_alignment_position_loss
                        + args.copy_gate_weight * causal_alignment_gate_loss
                    )
                    causal_accuracy = teacher_char_accuracy(causal_logits.detach(), causal_targets)
                    causal_position_accuracy = float(causal_supervision["position_accuracy"])
                    causal_gate_accuracy = float(causal_supervision["gate_accuracy"])
                    causal_example_id = causal_record["example_id"]
                    causal_variant_id = counterfactual["variant_id"]
                loss = (
                    scratch_loss
                    + response_loss
                    + args.alignment_weight * alignment_loss
                    + args.causal_weight
                    * (causal_loss + args.alignment_weight * causal_alignment_loss)
                )
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
            completed_this_run = step - run_start_step
            run_rate = completed_this_run / elapsed
            metric = {
                "schema": "axon-complete-field-r0-metric-v1",
                "step": step,
                "loss": loss_value,
                "teacher_forcing_ratio": args.teacher_forcing_ratio,
                "scratch_loss": float(scratch_loss.detach().item()),
                "response_loss": float(response_loss.detach().item()),
                "alignment_position_loss": float(alignment_position_loss.detach().item()),
                "alignment_gate_loss": float(alignment_gate_loss.detach().item()),
                "alignment_copy_positions": int(out["alignment"]["copy_positions"]),
                "alignment_position_accuracy": (
                    out["alignment"]["position_correct"] / out["alignment"]["copy_positions"]
                    if out["alignment"]["copy_positions"]
                    else 1.0
                ),
                "alignment_gate_accuracy": (
                    out["alignment"]["gate_correct"]
                    / out["alignment"]["gate_supervised_positions"]
                    if out["alignment"]["gate_supervised_positions"]
                    else 1.0
                ),
                "counterfactual_loss": float(causal_loss.detach().item()),
                "counterfactual_alignment_position_loss": float(
                    causal_alignment_position_loss.detach().item()
                ),
                "counterfactual_alignment_gate_loss": float(
                    causal_alignment_gate_loss.detach().item()
                ),
                "counterfactual_alignment_position_accuracy": causal_position_accuracy,
                "counterfactual_alignment_gate_accuracy": causal_gate_accuracy,
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
                "steps_per_second": run_rate,
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
                latest_evaluation = dict(evaluation)
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
                    dataset_sha256,
                )

            if evaluation is not None or samples is not None or step == 1:
                last_eval = latest_evaluation
                atomic_json(
                    args.run_dir / "live.json",
                    {
                        "schema": "axon-complete-field-r0-live-v1",
                        "status": status,
                        "step": step,
                        "target_step": args.steps,
                        "progress": step / args.steps,
                        "mean_recent_loss": float(np.mean(recent_losses)),
                        "steps_per_second": run_rate,
                        "eta_seconds": (args.steps - step) / max(1e-9, run_rate),
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
                    f"chars={out['coverage_tick1'].observed_characters} rate={run_rate:.3f}/s",
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
                dataset_sha256,
            )
        live_path = args.run_dir / "live.json"
        live: dict[str, Any] = {}
        if live_path.exists():
            live = json.loads(live_path.read_text(encoding="utf-8"))
        live.update({"status": "completed" if step >= args.steps else status, "step": step, "time": time.time()})
        atomic_json(live_path, live)

    final_eval = evaluate_teacher(model, eval_records, args.eval_examples)
    final_samples = observable_samples(model, eval_records, min(32, len(eval_records)))
    final_counterfactuals = forced_counterfactual_samples(
        model,
        causal_eval_records,
        min(args.counterfactual_sample_count, len(causal_eval_records) * 2),
    )
    final_v6_alignment = evaluate_v6_alignment_behavior(
        model, eval_records, max_examples=args.v6_eval_examples
    )
    atomic_json(
        args.run_dir / "counterfactual_samples.json",
        {
            "schema": "axon-complete-field-r0-counterfactual-samples-v2",
            "step": step,
            "samples": final_counterfactuals,
        },
    )
    atomic_json(
        args.run_dir / "v6_alignment_eval.json",
        {
            "schema": "axon-complete-field-r0-v6-alignment-eval-v1",
            "step": step,
            **final_v6_alignment,
        },
    )

    predicted_responses = [sample["predicted_response"] for sample in final_samples]
    response_termination_rate = float(
        np.mean([sample["response_terminated"] for sample in final_samples])
    )
    scratch_termination_rate = float(
        np.mean([sample["scratch_terminated"] for sample in final_samples])
    )
    response_nonblank_rate = float(np.mean([bool(text.strip()) for text in predicted_responses]))
    unique_response_count = len(set(predicted_responses))
    semantic_families = (
        "conversation_exact_copy",
        "conversation_foundation",
        "cross_page_exact_retrieval",
        "grounded_abstention",
        "scratch_arithmetic",
        "v6_alignment_retrieval",
    )
    hard_semantic_families = {
        "conversation_exact_copy",
        "conversation_foundation",
        "cross_page_exact_retrieval",
        "grounded_abstention",
        "v6_alignment_retrieval",
    }
    semantic_exact_match_by_family: dict[str, float] = {}
    represented_hard_families: set[str] = set()
    for family in semantic_families:
        rows = [sample for sample in final_samples if sample["family"] == family]
        if rows and family in hard_semantic_families:
            represented_hard_families.add(family)
        semantic_exact_match_by_family[family] = (
            float(
                np.mean(
                    [sample["predicted_response"] == sample["gold_response"] for sample in rows]
                )
            )
            if rows
            else 0.0
        )
    semantic_family_gate_passed = bool(represented_hard_families) and all(
        semantic_exact_match_by_family[family] >= 0.50
        for family in represented_hard_families
    )

    counterfactual_response_change_rate = float(
        np.mean([sample["response_changed"] for sample in final_counterfactuals])
    )
    forced_correct_termination_rate = float(
        np.mean([sample["correct_terminated"] for sample in final_counterfactuals])
    )
    forced_counterfactual_termination_rate = float(
        np.mean([sample["counterfactual_terminated"] for sample in final_counterfactuals])
    )
    forced_correct_exact_rate = float(
        np.mean([sample["correct_exact"] for sample in final_counterfactuals])
    )
    forced_counterfactual_exact_rate = float(
        np.mean([sample["counterfactual_exact"] for sample in final_counterfactuals])
    )
    authority_rows = [
        sample
        for sample in final_counterfactuals
        if sample["gold_counterfactual_response"] == sample["gold_correct_response"]
    ]
    evidence_authority_preservation_rate = (
        float(np.mean([sample["authority_preserved"] for sample in authority_rows]))
        if authority_rows
        else 0.0
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
    teacher_alignment_passed = bool(
        final_eval["aligned_copy_positions"] > 0
        and final_eval["aligned_position_accuracy"] == 1.0
        and final_eval["counterfactual_aligned_copy_positions"] > 0
        and final_eval["counterfactual_aligned_position_accuracy"] == 1.0
        and final_eval["aligned_gate_accuracy"] >= 0.95
        and final_eval["counterfactual_aligned_gate_accuracy"] >= 0.95
    )
    causal_passed = bool(
        final_eval["counterfactual_teacher_char_accuracy"] >= 0.50
        and forced_correct_termination_rate >= 0.95
        and forced_counterfactual_termination_rate >= 0.95
        and forced_correct_exact_rate >= 0.95
        and forced_counterfactual_exact_rate >= 0.95
        and evidence_authority_preservation_rate >= 0.95
    )
    free_running_passed = bool(
        response_termination_rate >= 0.95
        and scratch_termination_rate >= 0.95
        and response_nonblank_rate >= 0.95
        and unique_response_count >= 2
    )
    v6_alignment_gate_passed = bool(final_v6_alignment.get("passed") is True)
    gate = {
        "schema": "axon-complete-field-r0-run-gate-v3",
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
        "counterfactual_response_change_rate_diagnostic_only": counterfactual_response_change_rate,
        "forced_correct_termination_rate": forced_correct_termination_rate,
        "forced_counterfactual_termination_rate": forced_counterfactual_termination_rate,
        "forced_correct_exact_rate": forced_correct_exact_rate,
        "forced_counterfactual_exact_rate": forced_counterfactual_exact_rate,
        "evidence_authority_preservation_rate": evidence_authority_preservation_rate,
        "causal_scratch_gate_passed": causal_passed,
        "teacher_alignment_gate_passed": teacher_alignment_passed,
        "teacher_aligned_position_accuracy": final_eval["aligned_position_accuracy"],
        "teacher_counterfactual_position_accuracy": final_eval[
            "counterfactual_aligned_position_accuracy"
        ],
        "teacher_copy_gate_accuracy": final_eval["aligned_gate_accuracy"],
        "teacher_counterfactual_copy_gate_accuracy": final_eval[
            "counterfactual_aligned_gate_accuracy"
        ],
        "v6_alignment_eval": final_v6_alignment,
        "v6_alignment_gate_passed": v6_alignment_gate_passed,
        "scratch_termination_rate": scratch_termination_rate,
        "response_termination_rate": response_termination_rate,
        "response_nonblank_rate": response_nonblank_rate,
        "unique_response_count": unique_response_count,
        "semantic_exact_match_by_family": semantic_exact_match_by_family,
        "hard_semantic_families_represented": sorted(represented_hard_families),
        "arithmetic_is_diagnostic_only": True,
        "semantic_family_gate_passed": semantic_family_gate_passed,
        "free_running_gate_passed": free_running_passed,
        "promotion_allowed": bool(
            loss_improved
            and accuracy_improved
            and causal_passed
            and teacher_alignment_passed
            and v6_alignment_gate_passed
            and semantic_family_gate_passed
            and free_running_passed
        ),
    }
    atomic_json(args.run_dir / "gate.json", gate)
    print(json.dumps(gate, indent=2), flush=True)
    return 0 if gate["promotion_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
