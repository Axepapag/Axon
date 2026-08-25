"""Recover deterministic evidence for a preserved Heart generalization checkpoint.

This evaluator performs no optimization, candidate lifecycle transition,
promotion proposal, or activation. It exists so a valid immutable checkpoint
survives a post-training orchestration failure without rewriting history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from runtime.field import canonical_sha256
from runtime.heart.translation_core import HeartTranslationCore, HeartTranslationCoreConfig
from runtime.trainer import EvaluationObservation, TrainerControlPlane, TrainerStateStore
from train_heart_decoder_generalization_smoke import (
    HEART_DECODER_GENERALIZATION_SMOKE_SCHEMA,
    HEART_DECODER_GENERALIZATION_RECOVERY_SCHEMA,
    HEART_DECODER_GENERALIZATION_SUITE,
    _atomic_json,
    _checkpoint_record,
    _fixed_audit_loss,
    _generalization_gate,
    _generalization_metrics,
    _read_json,
)
from training.heart_translation import (
    HeartDecoderGeneralizationObjective,
    build_heart_decoder_generalization_curriculum,
    evaluate_heart_decoder_diagnostics,
    evaluate_heart_decoder_generalization,
)


def _terminal_lifecycle_event(state_root: Path, candidate_generation_id: str) -> dict[str, Any]:
    lifecycle_path = state_root / "training" / "trainer" / "candidate_lifecycle.jsonl"
    terminal: dict[str, Any] | None = None
    with lifecycle_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if event.get("candidate_generation_id") == candidate_generation_id:
                terminal = event
    if terminal is None:
        raise ValueError("candidate has no lifecycle event")
    if terminal.get("status") != "rejected":
        raise ValueError("recovery evaluator only accepts an already-rejected candidate")
    return terminal


def recover_generalization_checkpoint(
    *,
    state_root: Path,
    source_checkpoint_path: Path,
    candidate_checkpoint_path: Path,
    source_summary_path: Path,
    device: str,
) -> tuple[dict[str, Any], Path]:
    root = state_root.resolve(strict=False)
    source_record = _checkpoint_record(source_checkpoint_path)
    candidate_record = _checkpoint_record(candidate_checkpoint_path)
    source_summary = _read_json(source_summary_path)
    if source_summary.get("schema") != HEART_DECODER_GENERALIZATION_SMOKE_SCHEMA:
        raise ValueError("source summary is not a Heart decoder generalization run")
    if source_summary.get("candidate_generation_id") != source_record.candidate_generation_id:
        raise ValueError("source summary and source checkpoint identify different generations")
    if candidate_record.base_generation_id != source_record.candidate_generation_id:
        raise ValueError("candidate checkpoint does not descend from the stated source checkpoint generation")
    if candidate_record.module_id != source_record.module_id:
        raise ValueError("candidate and source checkpoints belong to different modules")
    architecture_value = source_summary.get("architecture_config")
    if not isinstance(architecture_value, dict):
        raise ValueError("source run summary has no architecture_config")

    curriculum = build_heart_decoder_generalization_curriculum()
    if source_summary.get("curriculum_id") != curriculum.curriculum_id:
        raise ValueError("source checkpoint used another generalization curriculum")
    objective = HeartDecoderGeneralizationObjective()
    resolved_device = torch.device(device)
    store = TrainerStateStore.active(state_root=root)

    def load_model(record):
        model = HeartTranslationCore(HeartTranslationCoreConfig(**architecture_value)).to(resolved_device)
        checkpoint = store.load_verified_candidate_checkpoint(record)
        model.load_state_dict(checkpoint["module_state_dict"], strict=True)
        model.eval()
        return model

    source_model = load_model(source_record)
    candidate_model = load_model(candidate_record)
    replay_ids = set(curriculum.replay_case_ids)
    replay_cases = tuple(case for case in curriculum.train_cases if case.case_id in replay_ids)
    novel_cases = tuple(case for case in curriculum.train_cases if case.case_id not in replay_ids)
    audit_novel = tuple(
        min(novel_cases, key=lambda case: abs(len(case.source_text) - target_length))
        for target_length in (32, 128, 257, 448)
    )
    audit_cases = tuple(dict.fromkeys((*audit_novel, replay_cases[0], replay_cases[-1])))

    source_diagnostic, source_evidence = evaluate_heart_decoder_generalization(
        source_model,
        curriculum,
        descriptor_id=source_record.parameter_manifest_id,
        checkpoint_id=source_record.checkpoint_id,
        split="decoder-generalization:heldout:recovery-source",
    )
    source_diagnostic.write(root)
    source_evidence.write(root)
    source_replay = evaluate_heart_decoder_diagnostics(
        source_model,
        replay_cases,
        descriptor_id=source_record.parameter_manifest_id,
        checkpoint_id=source_record.checkpoint_id,
        curriculum_id=curriculum.curriculum_id,
        split="decoder-generalization:replay:recovery-source",
    )
    source_replay.write(root)
    source_audit_loss = _fixed_audit_loss(source_model, audit_cases, objective, resolved_device)

    candidate_diagnostic, candidate_evidence = evaluate_heart_decoder_generalization(
        candidate_model,
        curriculum,
        descriptor_id=candidate_record.parameter_manifest_id,
        checkpoint_id=candidate_record.checkpoint_id,
        split="decoder-generalization:heldout:recovered-candidate",
    )
    candidate_diagnostic.write(root)
    candidate_evidence.write(root)
    candidate_replay = evaluate_heart_decoder_diagnostics(
        candidate_model,
        replay_cases,
        descriptor_id=candidate_record.parameter_manifest_id,
        checkpoint_id=candidate_record.checkpoint_id,
        curriculum_id=curriculum.curriculum_id,
        split="decoder-generalization:replay:recovered-candidate",
    )
    candidate_replay.write(root)
    candidate_audit_loss = _fixed_audit_loss(candidate_model, audit_cases, objective, resolved_device)

    metrics = _generalization_metrics(candidate_diagnostic, candidate_evidence, candidate_replay)
    observation = EvaluationObservation.from_mapping(
        suite_id=HEART_DECODER_GENERALIZATION_SUITE,
        module_id=candidate_record.module_id,
        candidate_generation_id=candidate_record.candidate_generation_id,
        metrics=metrics,
        artifact_id=candidate_evidence.evidence_id,
    )
    control = TrainerControlPlane.active(state_root=root)
    try:
        decision = control.evaluate_gate(
            _generalization_gate(candidate_record.module_id, candidate_record.candidate_generation_id),
            (observation,),
        )
    finally:
        control.close()

    lifecycle = _terminal_lifecycle_event(root, candidate_record.candidate_generation_id)
    run_core = {
        "schema": HEART_DECODER_GENERALIZATION_RECOVERY_SCHEMA,
        "module_id": candidate_record.module_id,
        "source_generation_id": source_record.candidate_generation_id,
        "source_checkpoint_id": source_record.checkpoint_id,
        "candidate_generation_id": candidate_record.candidate_generation_id,
        "candidate_checkpoint_id": candidate_record.checkpoint_id,
        "candidate_checkpoint_step": candidate_record.step,
        "curriculum_id": curriculum.curriculum_id,
        "source_evidence_id": source_evidence.evidence_id,
        "candidate_evidence_id": candidate_evidence.evidence_id,
        "gate_decision_id": decision.decision_id,
        "terminal_lifecycle_event_id": lifecycle["event_id"],
        "candidate_status": "recovered_evaluation_of_rejected_candidate",
        "device": str(resolved_device),
    }
    run_id = canonical_sha256(run_core)
    summary = {
        **run_core,
        "run_id": run_id,
        "architecture_config": architecture_value,
        "audit_case_ids": [case.case_id for case in audit_cases],
        "source_audit_loss": source_audit_loss,
        "candidate_audit_loss": candidate_audit_loss,
        "audit_loss_improved": candidate_audit_loss["total"] < source_audit_loss["total"],
        "source_diagnostic": source_diagnostic.to_canonical_dict(),
        "source_evidence": source_evidence.to_canonical_dict(),
        "source_replay_diagnostic": source_replay.to_canonical_dict(),
        "candidate_diagnostic": candidate_diagnostic.to_canonical_dict(),
        "candidate_evidence": candidate_evidence.to_canonical_dict(),
        "candidate_replay_diagnostic": candidate_replay.to_canonical_dict(),
        "gate_decision": decision.to_canonical_dict(),
        "terminal_lifecycle_event": lifecycle,
        "activation_forbidden": True,
        "lifecycle_rewritten": False,
    }
    path = root / "training" / "heart" / "generalization_recoveries" / run_id / "summary.json"
    _atomic_json(path, summary)
    return summary, path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=Path("State"))
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary, path = recover_generalization_checkpoint(
        state_root=args.state_root,
        source_checkpoint_path=args.source_checkpoint,
        candidate_checkpoint_path=args.candidate_checkpoint,
        source_summary_path=args.source_summary,
        device=args.device,
    )
    print(json.dumps({
        "run_id": summary["run_id"],
        "summary": str(path),
        "candidate_status": summary["candidate_status"],
        "gate_passed": summary["gate_decision"]["passed"],
        "failed_requirements": summary["gate_decision"]["failed_requirements"],
        "source_audit_loss": summary["source_audit_loss"]["total"],
        "candidate_audit_loss": summary["candidate_audit_loss"]["total"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
