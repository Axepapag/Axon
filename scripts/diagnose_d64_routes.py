"""Read-only receipt-decoder diagnosis of an accepted checkpoint and its Soul.

All examination cases are retained. Counterfactuals exist only in RAM and never
advance a Trainer pointer or constitute a promotion/training result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from runtime.field import D64FieldCompiler
from runtime.heart.reasoning_output import ReasoningDecision
from runtime.soul.store import SoulBranch
from runtime.trainer import CandidateCheckpointRecord, TrainerStateStore
from runtime.trainer.step_bundle import AcceptedStepPointer, AcceptedTrainingStepBundle
from substrate import encode_unicode_text
from training.first_form_curriculum import load_first_form_curriculum
from training.foundation_motor_curriculum import foundation_motor_v2_probe
from training.living_reasoning_curriculum import evaluate_living_episode
from training.living_reasoning_d64 import LivingReasoningCoreConfig, LivingReasoningCoreD64


def io_path(path: Path) -> Path:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return Path("\\\\?\\" + resolved)
    return Path(resolved)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(io_path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def tensor_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        data = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(data.dtype), list(data.shape)]).encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def gate_bias_counterfactual(model: LivingReasoningCoreD64, zero_bias: bool):
    before = model.copy_gate.bias.detach().clone()
    try:
        if zero_bias:
            with torch.no_grad():
                model.copy_gate.bias.zero_()
        yield
    finally:
        with torch.no_grad():
            model.copy_gate.bias.copy_(before)


@contextmanager
def stop_at_exact_failure(model, episode):
    """Stop a diagnostic after an irreversible categorical mismatch, not a cap.

    Expected text controls only examination stopping; it never enters the model
    or chooses an emission. Exactness is decided; eventual termination after a
    wrong prefix remains explicitly unmeasured. The exact state is retained.
    """
    advance = model.advance_decoder_execution
    saved_override = model.__dict__.get("advance_decoder_execution")
    targets = {t.phase: tuple(encode_unicode_text(t.payload)) for t in episode.targets}
    stops = []

    def observed_advance(output, state, *, work_units):
        expected = targets[output.phase]
        for _ in range(work_units):
            result = advance(output, state, work_units=1)
            state = result.state
            actual = state.transport_categories
            prefix_wrong = len(actual) > len(expected) or actual != expected[:len(actual)]
            if prefix_wrong or result.complete or result.malformed_reason is not None:
                if prefix_wrong and not result.complete and result.malformed_reason is None:
                    stops.append({"phase": output.phase, "reason": "irreversible_category_mismatch",
                                  "termination_after_mismatch": "not_measured",
                                  "execution_state": state.to_canonical_dict()})
                return result
        return result

    model.advance_decoder_execution = observed_advance
    try:
        yield stops
    finally:
        if saved_override is None:
            del model.advance_decoder_execution
        else:
            model.advance_decoder_execution = saved_override


TRANSCRIPT_FAILURE_MODES = (
    "exact",
    "failed_to_stop",
    "stopped_early",
    "wrong_symbol",
    "emitted_nothing",
)


def classify_transcript_failure(row: dict[str, Any]) -> str:
    """Name how a payload phase failed without guessing at model internals.

    Read the classification through the diagnostic's stop policy: the decoder is
    halted at the first irreversible category mismatch, so `terminated` is the
    model's own verdict only when the payload matched. A phase that emitted the
    whole expected content and then kept going is recorded as an over-long
    payload with `terminated` False, which is exactly the "failed to stop" case.
    """

    predicted = str(row.get("predicted_payload") or "")
    expected = str(row.get("expected_payload") or "")
    terminated = bool(row.get("terminated"))
    if predicted == expected:
        return "exact" if terminated else "failed_to_stop"
    if expected.startswith(predicted):
        return "stopped_early" if terminated else "wrong_symbol"
    if predicted.startswith(expected):
        return "failed_to_stop"
    if not predicted:
        # Unreachable while `expected` is non-empty: the empty string is a prefix
        # of every payload.  Kept because an empty target would land here.
        return "emitted_nothing"
    return "wrong_symbol"


def enrich_case_transcripts(
    transcripts: list[dict[str, Any]],
    *,
    family_by_episode: dict[str, str],
    case_id_by_episode: dict[str, str],
) -> list[dict[str, Any]]:
    """Attribute each transcript to its family and case, and name its failure.

    The transcript sink records only what the decoder did; `episode_id` is a
    hash.  Without this step a forensic table can prove a rate but not which
    action family produced it.
    """

    return [
        {
            **row,
            "family": family_by_episode.get(str(row.get("episode_id") or "")),
            "case_id": case_id_by_episode.get(str(row.get("episode_id") or "")),
            "failure_mode": classify_transcript_failure(row),
        }
        for row in transcripts
    ]


def count_failure_modes(transcripts: list[dict[str, Any]]) -> dict[str, int]:
    counts = {mode: 0 for mode in TRANSCRIPT_FAILURE_MODES}
    for row in transcripts:
        mode = str(row.get("failure_mode") or classify_transcript_failure(row))
        counts[mode] = counts.get(mode, 0) + 1
    return counts


@contextmanager
def reuse_frozen_unroll(model):
    """Reuse one exact read/Soul trajectory across decoder-only probes.

    The gate is not part of the reader or exhale path. Reject different inputs
    rather than accidentally sharing a trajectory across episodes.
    """
    unroll = model.unroll_runtime_phases
    saved_override = model.__dict__.get("unroll_runtime_phases")
    cached = None
    binding = None

    def cached_unroll(**kwargs):
        nonlocal cached, binding
        key = (kwargs["initial_soul"].soul_id, kwargs["canonical"].rail_id,
               kwargs["expected_core_id"], kwargs["parameter_generation"], kwargs["tick_uid"],
               kwargs["first_workspace_text"], kwargs["refined_workspace_text"])
        if binding is not None and binding != key:
            raise ValueError("attempted to reuse a frozen unroll across different inputs")
        if cached is None:
            with torch.no_grad():
                cached = unroll(**kwargs)
            binding = key
        return cached

    model.unroll_runtime_phases = cached_unroll
    try:
        yield
    finally:
        if saved_override is None:
            del model.unroll_runtime_phases
        else:
            model.unroll_runtime_phases = saved_override


def config_from_architecture(architecture: dict[str, Any]) -> LivingReasoningCoreConfig:
    """Rebuild the exact core config a report architecture dict describes.

    Report architecture dicts carry the opt-in EOS route only implicitly via
    emission_routes; without restoring the flag the recomputed architecture
    identity does not round-trip.
    """
    config_kwargs = {
        field.name: architecture[field.name]
        for field in fields(LivingReasoningCoreConfig)
        if field.init and field.name in architecture
    }
    routes = architecture.get("emission_routes", ())
    if "eos_generate_head_route" not in config_kwargs:
        config_kwargs["eos_generate_head_route"] = "generate_head_eos" in routes
    if "termination_head_route" not in config_kwargs:
        config_kwargs["termination_head_route"] = "termination_head" in routes
    return LivingReasoningCoreConfig(**config_kwargs)


def load_evidence(state_root: Path, report_path: Path):
    state_root = io_path(state_root)
    report = read_json(report_path)
    candidate = report["candidate_generation_id"]
    module = report["soul_promotion_plan"]["core_id"]
    store = TrainerStateStore.active(state_root=state_root)
    candidate_root = store.candidates_dir / module / candidate
    pointer = AcceptedStepPointer.from_mapping(read_json(candidate_root / "accepted_steps/pointer.json"))
    bundle = AcceptedTrainingStepBundle.from_mapping(read_json(
        candidate_root / "accepted_steps/bundles" / f"{pointer.current_bundle_id}.json"
    ))
    checkpoint = CandidateCheckpointRecord.from_mapping(
        read_json(
            candidate_root
            / "checkpoint_records"
            / f"{bundle.checkpoint_id}.json"
        )
    )
    checkpoint_id = checkpoint.checkpoint_id
    if checkpoint_id != report["final_checkpoint_id"]:
        raise ValueError("accepted checkpoint or report identity mismatch")
    if (bundle.checkpoint_id, bundle.module_id, bundle.candidate_generation_id) != (
        checkpoint_id, module, candidate
    ) or pointer.current_step != checkpoint.step:
        raise ValueError("checkpoint is not the accepted pointer boundary")
    payload = store.load_verified_candidate_checkpoint(checkpoint)
    architecture = report["architecture"]
    config = config_from_architecture(architecture)
    if config.architecture_id != architecture["architecture_id"] or not config.receipt_continuation:
        raise ValueError("diagnostic requires the exact receipt architecture")
    if payload["descriptor"]["architecture"] != config.architecture_id:
        raise ValueError("checkpoint architecture disagrees with report")
    model = LivingReasoningCoreD64(config)
    model.load_state_dict(payload["module_state_dict"], strict=True)
    model.eval()
    soul = SoulBranch(
        state_root / "training/soul_candidates" / candidate / module / "branch",
        core_id=module, branch_id=candidate,
    ).load_head()
    if soul.soul_id != bundle.after_soul_id or soul.soul_id != report["soul_promotion_plan"]["candidate_soul_id"]:
        raise ValueError("Soul is not the accepted checkpoint companion")
    return model, soul, checkpoint, report


@torch.no_grad()
def position_rows(model, episode, soul) -> list[dict[str, Any]]:
    compiled = D64FieldCompiler().compile(episode.snapshot)
    compiled.verify_roundtrip(episode.snapshot)
    unroll = model.unroll_runtime_phases(
        initial_soul=soul, expected_core_id=soul.core_id,
        parameter_generation=soul.parameter_generation,
        tick_uid=f"evaluation-episode:{episode.episode_id}", canonical=compiled,
        first_workspace_text=episode.first_workspace_text,
        refined_workspace_text=episode.refined_workspace_text,
    )
    result = []
    for output, target in zip(unroll.outputs, episode.targets, strict=True):
        if target.supervision_weight <= 0 or target.decision is not ReasoningDecision.DELTA:
            continue
        mixed, expected, alignment = model.decode_teacher(
            output.reader_state, target.payload, head=1,
            memory=output.complete_memory, return_alignment=True,
        )
        supervision = None if target.payload_alignment is None else model.alignment_supervision(
            target_text=target.payload, memory=output.complete_memory,
            decoder_alignment=alignment, specification=target.payload_alignment,
        )
        mask = torch.ones_like(expected, dtype=torch.bool) if supervision is None else supervision["learned_decision_mask"]
        gates = alignment["generate_gate_logits"][0]
        generated = alignment["generated_logits"][0].argmax(-1)
        mixture = mixed[0].argmax(-1)
        pointers = alignment["position_logits"][0].argmax(-1)
        for position in range(expected.shape[1]):
            is_eos = position == expected.shape[1] - 1
            learned = bool(mask[0, position])
            gate = float(gates[position])
            pointer = int(pointers[position])
            selected = int(generated[position]) if gate >= 0 else int(output.complete_memory.char_indices[0, pointer])
            receipt = output.complete_memory.receipt(pointer)
            label = "eos" if is_eos else (
                "continuation" if not learned else "copy_anchor" if supervision is not None else "unaligned_content"
            )
            result.append({
                "episode_id": episode.episode_id, "phase": output.phase,
                "field_id": output.source_field_id, "rail_id": output.rail_id,
                "position": position, "position_role": label,
                "learned_decision": learned, "expected_category": int(expected[0, position]),
                "gate_logit": gate, "gate_bias": float(model.copy_gate.bias[0]),
                "gate_weight_contribution": gate - float(model.copy_gate.bias[0]),
                "generate_probability": float(torch.sigmoid(gates[position])),
                "selected_route": "generate" if gate >= 0 else "copy",
                "generated_category": int(generated[position]),
                "mixed_category": int(mixture[position]),
                "routed_category": selected,
                "pointer_index": pointer,
                "pointer_unit_index": receipt.address.transport_unit_index,
                "pointer_scalar_position": receipt.address.region_position,
                "complete_field_coverage": bool(output.canonical_coverage.complete),
            })
    return result


def summarize_positions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for role in ("copy_anchor", "eos", "continuation", "unaligned_content"):
        selected = [r for r in rows if r["position_role"] == role]
        if not selected:
            result[role] = {"count": 0}
            continue
        logits = [r["gate_logit"] for r in selected]
        result[role] = {
            "count": len(selected), "gate_logit_min": min(logits),
            "gate_logit_max": max(logits), "gate_logit_mean": sum(logits) / len(logits),
            "generate_route_fraction": sum(x >= 0 for x in logits) / len(logits),
            "mixed_category_accuracy": sum(r["mixed_category"] == r["expected_category"] for r in selected) / len(selected),
            "routed_category_accuracy": sum(r["routed_category"] == r["expected_category"] for r in selected) / len(selected),
            "generated_category_accuracy": sum(r["generated_category"] == r["expected_category"] for r in selected) / len(selected),
        }
    copy = [r["gate_logit"] for r in rows if r["position_role"] == "copy_anchor"]
    eos = [r["gate_logit"] for r in rows if r["position_role"] == "eos"]
    result["perfect_scalar_threshold_exists"] = bool(copy and eos and max(copy) < min(eos))
    result["threshold_note"] = "Descriptive examination statistic only; never used to select a deployed threshold."
    return result


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rates = {
        "typed_emission_exact": ("typed_emission_exact_count", "supervised_phase_count"),
        "payload_transport_exact": ("payload_transport_exact_count", "payload_supervised_phase_count"),
        "teacher_token": ("payload_teacher_forced_token_correct", "payload_teacher_forced_token_count"),
        "teacher_content": ("payload_teacher_forced_content_correct", "payload_teacher_forced_content_count"),
        "teacher_eos": ("payload_teacher_forced_eos_correct", "payload_teacher_forced_eos_count"),
        "copy_gate": ("alignment_copy_gate_correct", "alignment_copy_gate_count"),
        "eos_gate": ("alignment_eos_gate_correct", "alignment_eos_gate_count"),
        "source_position": ("alignment_position_correct", "alignment_position_count"),
    }
    result = {"case_count": len(rows)}
    for name, (numerator, denominator) in rates.items():
        count = sum(r[denominator] for r in rows)
        result[name] = None if count == 0 else sum(r[numerator] for r in rows) / count
    return result


def diagnose(args) -> Path:
    torch.set_num_threads(args.threads)
    model, soul, checkpoint, report = load_evidence(args.state_root, args.run_report)
    model.to(args.device)
    fingerprint = tensor_fingerprint(model)
    curricula = [load_first_form_curriculum(p) for p in args.curriculum_manifest]
    if set(c.manifest_id for c in curricula) != set(report["standard_ffcs_manifest_ids"]):
        raise ValueError("diagnostic manifests do not match the complete recorded examinations")
    artifact = {
        "schema": "axon-d64-route-diagnostic-v1",
        "checkpoint_id": checkpoint.checkpoint_id, "checkpoint_sha256": checkpoint.artifact_sha256,
        "checkpoint_step": checkpoint.step, "parameter_fingerprint": fingerprint,
        "soul_id": soul.soul_id, "architecture_id": model.architecture_id,
        "run_report_sha256": hashlib.sha256(args.run_report.read_bytes()).hexdigest(),
        "device": str(args.device), "variants": {}, "training_performed": False,
        "training_stage": args.training_stage,
        "serving_promotion_claimed": False,
        "free_running_policy": "Stop on exact termination, malformed output, or first irreversible category mismatch; preserve decoder state. Eventual termination after a mismatch is unmeasured. No target guides neural emissions.",
    }
    artifact["variants"] = {variant: [] for variant in ("baseline", "zero_bias")}
    for curriculum in curricula:
        # The transcript sink records the decoder's behaviour only; the family
        # and case identity live on the teaching cases, not on the decode.
        family_by_episode = {
            case.episode.episode_id: case.family for case in curriculum.teaching_cases
        }
        case_id_by_episode = {
            case.episode.episode_id: case.case_id for case in curriculum.teaching_cases
        }
        for split in ("heldout", "regression"):
            episodes = curriculum.teaching_living_curriculum.split(split)
            collected = {variant: ([], [], []) for variant in artifact["variants"]}
            for ordinal, episode in enumerate(episodes, 1):
                with reuse_frozen_unroll(model):
                    for variant, (all_positions, metric_rows, transcripts) in collected.items():
                        with gate_bias_counterfactual(model, variant == "zero_bias"):
                            all_positions.extend(position_rows(model, episode, soul))
                            with stop_at_exact_failure(model, episode) as stops:
                                metrics = evaluate_living_episode(
                                    model, episode, soul, core_id=soul.core_id,
                                    parameter_generation=soul.parameter_generation,
                                    transcript_sink=transcripts,
                                    transcript_sink_cap=None,
                                )
                            metrics["diagnostic_stops"] = stops
                            metric_rows.append(metrics)
                print(json.dumps({"manifest": curriculum.manifest_id,
                                  "split": split, "case": ordinal, "total": len(episodes)}), flush=True)
            for variant, (all_positions, metric_rows, transcripts) in collected.items():
                enriched = enrich_case_transcripts(
                    transcripts,
                    family_by_episode=family_by_episode,
                    case_id_by_episode=case_id_by_episode,
                )
                entry = {
                        "manifest_id": curriculum.manifest_id, "split": split,
                        "complete": len(metric_rows) == len(episodes),
                        "metrics": aggregate_metrics(metric_rows),
                        "motor_probe": foundation_motor_v2_probe(episodes, metric_rows),
                        "position_summary": summarize_positions(all_positions),
                        "case_transcripts": enriched,
                        "case_transcript_failure_modes": count_failure_modes(enriched),
                        "positions": all_positions,
                        "episode_metrics": metric_rows,
                }
                if args.training_stage is not None:
                    entry["motor_probe_stage_scoped"] = foundation_motor_v2_probe(
                        episodes, metric_rows, training_stage=args.training_stage
                    )
                artifact["variants"][variant].append(entry)
                print(json.dumps({"variant": variant, "split": split,
                                  "summary": summarize_positions(all_positions)}), flush=True)
            if tensor_fingerprint(model) != fingerprint:
                raise ValueError("diagnostic failed to restore exact original parameters")
    artifact["original_parameters_restored"] = True
    data = json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    path = args.output_root / f"{digest}.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != data:
            raise ValueError("diagnostic output identity collision")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True, help="Fetched checkpoint State root (read only)")
    parser.add_argument("--run-report", type=Path, required=True)
    parser.add_argument("--curriculum-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--training-stage",
        default=None,
        help=(
            "additionally report the stage gate surface with the payload/EOS rates "
            "narrowed to this stage's eligible_actions; the whole-surface probe "
            "is still recorded unchanged"
        ),
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    print(json.dumps({"artifact": str(diagnose(args))}), flush=True)


if __name__ == "__main__":
    main()
