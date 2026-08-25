"""Produce immutable, checkpoint-bound Heart decoder failure evidence.

This command performs no training and never activates a candidate.  It verifies
the Trainer checkpoint lineage and bytes before exposing teacher-forced,
free-running, copy-gate, alignment, and EOS behavior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from runtime.heart import HeartTranslatorDescriptor, HeartTranslatorRole, HeartTranslatorState
from runtime.heart.translation_core import (
    HEART_TRANSLATION_ARCHITECTURE,
    HeartTranslationCore,
    HeartTranslationCoreConfig,
)
from runtime.trainer import CandidateCheckpointRecord, TrainerStateStore
from training.heart_translation import (
    DIALECTS,
    HeartDecoderDiagnosticReport,
    build_heart_decoder_mechanism_curriculum,
    build_heart_translation_curriculum,
    evaluate_heart_decoder_diagnostics,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _checkpoint_record(path: Path) -> CandidateCheckpointRecord:
    value = _read_json(path)
    stated_id = value.pop("checkpoint_id", None)
    value.pop("schema", None)
    record = CandidateCheckpointRecord(**value)
    if stated_id != record.checkpoint_id:
        raise ValueError("checkpoint record identity disagrees with canonical content")
    return record


def diagnose_checkpoint(
    *,
    state_root: Path,
    checkpoint_record_path: Path,
    run_summary_path: Path,
    curriculum_kind: str,
    split: str,
    limit: int | None,
    device: str,
) -> tuple[HeartDecoderDiagnosticReport, Path]:
    record = _checkpoint_record(checkpoint_record_path)
    summary = _read_json(run_summary_path)
    if summary.get("candidate_generation_id") != record.candidate_generation_id:
        raise ValueError("run summary and checkpoint record name different candidate generations")
    architecture = summary.get("architecture_config")
    if not isinstance(architecture, dict):
        raise ValueError("run summary has no architecture_config object")

    semantic_curriculum = build_heart_translation_curriculum()
    if summary.get("curriculum_id") != semantic_curriculum.curriculum_id:
        raise ValueError("run summary curriculum is not the current governed Heart curriculum")
    if curriculum_kind == "semantic":
        curriculum = semantic_curriculum
        split_cases = {
            "train": curriculum.train_cases,
            "heldout": curriculum.heldout_cases,
            "regression": curriculum.regression_cases,
        }[split]
    elif curriculum_kind == "decoder-mechanism":
        curriculum = build_heart_decoder_mechanism_curriculum()
        if split == "regression":
            raise ValueError("decoder mechanism curriculum has no regression split")
        split_cases = {
            "train": curriculum.train_cases,
            "heldout": curriculum.heldout_cases,
        }[split]
    else:
        raise ValueError(f"unsupported curriculum kind {curriculum_kind!r}")
    curriculum.write(state_root)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        split_cases = split_cases[:limit]
    split_label = f"{curriculum_kind}:{split}"
    if limit is not None:
        split_label += f":first-{limit}"

    store = TrainerStateStore.active(state_root=state_root)
    checkpoint = store.load_verified_candidate_checkpoint(record)
    model = HeartTranslationCore(HeartTranslationCoreConfig(**architecture))
    model.load_state_dict(checkpoint["module_state_dict"], strict=True)
    model.to(torch.device(device))
    descriptor = checkpoint.get("descriptor", {})
    if descriptor.get("architecture") != HEART_TRANSLATION_ARCHITECTURE:
        raise ValueError("verified checkpoint does not describe the governed Heart architecture")
    heart_descriptor = HeartTranslatorDescriptor(
        translator_id=record.module_id,
        generation_id=record.candidate_generation_id,
        role=HeartTranslatorRole.TRANSLATOR,
        architecture=HEART_TRANSLATION_ARCHITECTURE,
        d_model=model.cfg.d_model,
        source_dialects=DIALECTS,
        destination_dialects=DIALECTS,
        state=HeartTranslatorState.CANDIDATE,
    )
    descriptor_id = heart_descriptor.descriptor_id

    report = evaluate_heart_decoder_diagnostics(
        model,
        split_cases,
        descriptor_id=descriptor_id,
        checkpoint_id=record.checkpoint_id,
        curriculum_id=curriculum.curriculum_id,
        split=split_label,
    )
    return report, report.write(state_root.resolve(strict=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=Path("State"))
    parser.add_argument("--checkpoint-record", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument(
        "--curriculum",
        choices=("semantic", "decoder-mechanism"),
        default="semantic",
    )
    parser.add_argument("--split", choices=("train", "heldout", "regression"), default="heldout")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report, artifact = diagnose_checkpoint(
        state_root=args.state_root,
        checkpoint_record_path=args.checkpoint_record,
        run_summary_path=args.run_summary,
        curriculum_kind=args.curriculum,
        split=args.split,
        limit=args.limit,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "diagnostic_id": report.diagnostic_id,
                "artifact": str(artifact),
                "case_count": report.case_count,
                "teacher_forced_character_accuracy": report.teacher_forced_character_accuracy,
                "teacher_forced_eos_accuracy": report.teacher_forced_eos_accuracy,
                "teacher_forced_sequence_exact_rate": report.teacher_forced_sequence_exact_rate,
                "greedy_exact_rate": report.greedy_exact_rate,
                "greedy_termination_rate": report.greedy_termination_rate,
                "mean_greedy_correct_prefix_fraction": report.mean_greedy_correct_prefix_fraction,
                "mean_generation_gate": report.mean_generation_gate,
                "mean_copyable_generation_gate": report.mean_copyable_generation_gate,
                "mean_noncopyable_generation_gate": report.mean_noncopyable_generation_gate,
                "mean_target_character_attention_mass": report.mean_target_character_attention_mass,
                "mean_attention_peak": report.mean_attention_peak,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
