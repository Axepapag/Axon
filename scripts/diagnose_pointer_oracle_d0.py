"""Run the ratified read-only D0 pointer diagnosis on exact candidate boundaries.

This evaluator loads verified checkpoints and immutable candidate Souls, runs
FIRST-pass heldout fields, and compares the normal decoder with two oracle
interventions.  It never creates an optimizer or writes under canonical State.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from runtime.soul import SOUL_TEMPERATURE_ORDER
from runtime.trainer import CandidateCheckpointRecord, CandidateSoulWorkspace, TrainerStateStore
from training import LivingReasoningCoreConfig, LivingReasoningCoreD64, build_pointer_bootstrap_curriculum
from training.complete_field_64d import D64FieldCompiler
from training.pointer_oracle_diagnostic import (
    aggregate_first_decisions,
    cross_episode_cortex_key_accuracy,
    first_decision_diagnostic,
    leave_one_group_out_centroid_accuracy,
    summarize_vectors,
)


ROOT = Path(__file__).resolve().parent.parent
MODULE_ID = "r64-english-reasoning"
CANDIDATE_GENERATION = "english-candidate-1c991f8c911f79394e91"
BOUNDARIES: Mapping[str, Mapping[str, str | int]] = {
    "step24": {
        "step": 24,
        "bundle_id": "1d77a96509993005e03366066fdd43f1143adb8895ba470b62f1a300aff7aeb5",
        "checkpoint_id": "ef373220713565efa9da7b6c7877ffb5de75ab4522513ce679945102690ced3e",
        "soul_id": "308ff2a2b5de6337e904733a04da001af613faa7faa364a984150b3837753a0a",
    },
    "step25": {
        "step": 25,
        "bundle_id": "ea681656e548c27afb28b5d59418da4313f87d6b0a4402344fa2478b83c3103e",
        "checkpoint_id": "1b0282b026794109a5c574ed211f7c835ab05451e3785b2b7bf71d44d723cfe7",
        "soul_id": "13cf886a5a083200816c4487b12edbeb434500993dc8816ce6a374f0abe8f19e",
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=ROOT / "State")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--boundary", choices=("all", *BOUNDARIES), default="all")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_boundary(
    state_root: Path,
    device: torch.device,
    name: str,
) -> tuple[LivingReasoningCoreD64, Any, CandidateCheckpointRecord]:
    identity = BOUNDARIES[name]
    checkpoint_id = str(identity["checkpoint_id"])
    trainer_root = state_root / "training" / "trainer"
    record_path = (
        trainer_root
        / "candidates"
        / MODULE_ID
        / CANDIDATE_GENERATION
        / "checkpoint_records"
        / f"{checkpoint_id}.json"
    )
    record = CandidateCheckpointRecord.from_mapping(
        json.loads(record_path.read_text(encoding="utf-8"))
    )
    if (
        record.checkpoint_id != checkpoint_id
        or record.step != int(identity["step"])
        or record.module_id != MODULE_ID
        or record.candidate_generation_id != CANDIDATE_GENERATION
    ):
        raise RuntimeError(f"{name} checkpoint record disagrees with the ratified boundary")
    payload = TrainerStateStore(trainer_root).load_verified_candidate_checkpoint(record)
    config = LivingReasoningCoreConfig(
        d_model=64,
        n_heads=1,
        n_layers=2,
        ffn_dim=16_384,
        state_tokens=4,
        page_size=32,
        generate_gate_bias=0.0,
    )
    descriptor = payload["descriptor"]
    if descriptor["architecture"] != config.architecture_id:
        raise RuntimeError("checkpoint architecture differs from the exact D0 model")
    model = LivingReasoningCoreD64(config).to(device)
    model.load_state_dict(payload["module_state_dict"], strict=True)
    model.eval()
    soul_id = str(identity["soul_id"])
    soul = CandidateSoulWorkspace(state_root).branch(
        CANDIDATE_GENERATION,
        MODULE_ID,
    ).load_snapshot(soul_id)
    if (
        soul.soul_id != soul_id
        or soul.architecture_id != model.architecture_id
        or soul.parameter_generation != CANDIDATE_GENERATION
    ):
        raise RuntimeError(f"{name} Soul disagrees with the ratified boundary")
    return model, soul, record


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    hidden = {
        "reader_vector",
        "decoder_fused_vector",
        "pointer_query_vector",
        "pointer_distribution",
        "cortex_keys",
    }
    return {key: value for key, value in row.items() if key not in hidden}


@torch.inference_mode()
def _evaluate_boundary(
    *,
    state_root: Path,
    device: torch.device,
    name: str,
) -> dict[str, Any]:
    model, soul, record = _load_boundary(state_root, device, name)
    heldout = build_pointer_bootstrap_curriculum().split("heldout")
    compiler = D64FieldCompiler()
    intact_rows: list[dict[str, Any]] = []
    reset_rows: list[dict[str, Any]] = []
    soul_rows: list[dict[str, Any]] = []
    intact_keys: list[tuple[str, Mapping[int, torch.Tensor]]] = []
    reset_keys: list[tuple[str, Mapping[int, torch.Tensor]]] = []

    for episode in heldout:
        target = episode.targets[0]
        if target.phase != "first" or target.text_alignment is None:
            raise RuntimeError("D0 heldout episode lacks a FIRST exact-source alignment")
        compiled = compiler.compile(episode.snapshot)
        compiled.verify_roundtrip(episode.snapshot)
        common = {
            "soul": soul,
            "expected_core_id": MODULE_ID,
            "parameter_generation": CANDIDATE_GENERATION,
            "phase": "first",
            "canonical": compiled,
        }
        intact_output = model.forward_surfaces(**common)
        reset_output = model.forward_surfaces(
            **common,
            ablate_temperatures=SOUL_TEMPERATURE_ORDER,
        )
        intact = first_decision_diagnostic(
            model,
            intact_output,
            target_text=target.text,
            alignment_specification=target.text_alignment,
        )
        reset = first_decision_diagnostic(
            model,
            reset_output,
            target_text=target.text,
            alignment_specification=target.text_alignment,
        )
        if intact["expected_cortex_position"] != reset["expected_cortex_position"]:
            raise RuntimeError("Soul ablation changed the compiler-certified address")
        pointer_tv = 0.5 * float(
            (intact["pointer_distribution"] - reset["pointer_distribution"])
            .abs()
            .sum()
            .item()
        )
        query_delta = float(
            (intact["pointer_query_vector"] - reset["pointer_query_vector"])
            .norm()
            .item()
        )
        soul_rows.append(
            {
                "episode_id": episode.episode_id,
                "cortex_position": intact["expected_cortex_position"],
                "pointer_distribution_total_variation": pointer_tv,
                "pointer_query_l2_delta": query_delta,
                "pointer_probability_delta": (
                    intact["pointer_probability"] - reset["pointer_probability"]
                ),
                "normal_target_probability_delta": (
                    intact["normal"]["target_probability"]
                    - reset["normal"]["target_probability"]
                ),
                "intact_pointer_top1_correct": intact["pointer_top1_correct"],
                "reset_pointer_top1_correct": reset["pointer_top1_correct"],
                "intact_decoded_soul_layers": intact["soul_telemetry"]["decoded_soul_layers"],
                "intact_soul_contribution_l2": intact["soul_telemetry"]["soul_contribution_l2"],
                "reset_decoded_soul_layers": reset["soul_telemetry"]["decoded_soul_layers"],
                "reset_soul_contribution_l2": reset["soul_telemetry"]["soul_contribution_l2"],
            }
        )
        intact_rows.append({"episode_id": episode.episode_id, **intact})
        reset_rows.append({"episode_id": episode.episode_id, **reset})
        intact_keys.append((episode.episode_id, intact["cortex_keys"]))
        reset_keys.append((episode.episode_id, reset["cortex_keys"]))

    labels = [int(row["expected_cortex_position"]) for row in intact_rows]
    groups = [str(row["episode_id"]) for row in intact_rows]

    def mean(name: str) -> float:
        return sum(float(row[name]) for row in soul_rows) / len(soul_rows)

    return {
        "boundary": name,
        "step": record.step,
        "bundle_id": BOUNDARIES[name]["bundle_id"],
        "checkpoint_id": record.checkpoint_id,
        "checkpoint_artifact_sha256": record.artifact_sha256,
        "soul_id": soul.soul_id,
        "architecture_id": model.architecture_id,
        "candidate_generation": CANDIDATE_GENERATION,
        "device": str(device),
        "surface": "pointer_bootstrap_native_v1 heldout FIRST only",
        "intact": aggregate_first_decisions(intact_rows),
        "all_soul_layers_ablated": aggregate_first_decisions(reset_rows),
        "representation_geometry": {
            "reader_state": summarize_vectors([row["reader_vector"] for row in intact_rows]),
            "decoder_fused": summarize_vectors([row["decoder_fused_vector"] for row in intact_rows]),
            "pointer_query": summarize_vectors([row["pointer_query_vector"] for row in intact_rows]),
            "pointer_query_address_centroid": leave_one_group_out_centroid_accuracy(
                [row["pointer_query_vector"] for row in intact_rows], labels, groups
            ),
            "cortex_key_cross_episode_address": cross_episode_cortex_key_accuracy(intact_keys),
        },
        "ablated_representation_geometry": {
            "pointer_query": summarize_vectors([row["pointer_query_vector"] for row in reset_rows]),
            "pointer_query_address_centroid": leave_one_group_out_centroid_accuracy(
                [row["pointer_query_vector"] for row in reset_rows], labels, groups
            ),
            "cortex_key_cross_episode_address": cross_episode_cortex_key_accuracy(reset_keys),
        },
        "soul_sensitivity": {
            "interpretation": (
                "Sensitivity to in-memory all-layer ablation is causal input-path evidence only; "
                "it is not a delayed-recall or useful-Soul mastery claim."
            ),
            "pointer_distribution_total_variation_mean": mean(
                "pointer_distribution_total_variation"
            ),
            "pointer_query_l2_delta_mean": mean("pointer_query_l2_delta"),
            "pointer_probability_delta_mean": mean("pointer_probability_delta"),
            "normal_target_probability_delta_mean": mean(
                "normal_target_probability_delta"
            ),
            "intact_soul_contribution_l2_mean": mean("intact_soul_contribution_l2"),
            "reset_soul_contribution_l2_mean": mean("reset_soul_contribution_l2"),
            "rows": soul_rows,
        },
        "rows": [
            {
                "episode_id": intact["episode_id"],
                "intact": _public_row(intact),
                "all_soul_layers_ablated": _public_row(reset),
            }
            for intact, reset in zip(intact_rows, reset_rows, strict=True)
        ],
    }


def main() -> int:
    args = _arguments()
    state_root = args.state_root.resolve()
    device = _device(args.device)
    selected = tuple(BOUNDARIES) if args.boundary == "all" else (args.boundary,)
    report = {
        "schema": "axon-pointer-oracle-d0-report-v1",
        "authority": "read_only_diagnostic_not_mastery",
        "state_root": str(state_root),
        "boundaries": [
            _evaluate_boundary(state_root=state_root, device=device, name=name)
            for name in selected
        ],
    }
    if args.report is not None:
        _atomic_json(args.report.resolve(), report)
    data = json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n"
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(data.encode("utf-8"))
        buffer.flush()
    else:
        sys.stdout.write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
