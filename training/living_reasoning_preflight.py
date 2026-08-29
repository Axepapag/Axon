"""Hard launch firewall for Soul-conditioned living reasoning candidates."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from runtime.field import D64_COMPILER_SCHEMA, D64FieldCompiler, LogicalRegion, SharedFieldSnapshot, canonical_sha256
from runtime.soul import (
    SOUL_TEMPERATURE_ORDER,
    SoulSnapshot,
    SoulTemperature,
    apply_soul_transition,
    empty_soul_layers,
)
from runtime.trainer import (
    CompleteFieldTrainingContract,
    DeclaredTrainingBound,
    ParameterInventory,
    ParameterMutationPlan,
    PreflightEvidenceKind,
    TrainingBoundCategory,
    TrainingPreflightEvidence,
    TrainingPreflightReceipt,
    build_training_preflight_receipt,
)

from .heart_preflight import scan_active_capacity_poison
from .living_reasoning_curriculum import LivingReasoningCurriculum
from .living_reasoning_d64 import LivingReasoningCoreD64

LIVING_REASONING_PREFLIGHT_SCHEMA = "axon-living-reasoning-preflight-evidence-v1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _architecture_capacity(model: LivingReasoningCoreD64) -> dict[str, Any]:
    learned_positions = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Embedding) and ("position" in name.lower() or "page" in name.lower())
    ]
    report = model.architecture_report()
    passed = bool(
        not learned_positions
        and report["d_model"] == 64
        and report["ffn_dim"] > 0
        and report["state_tokens"] > 0
        and report["page_size"] > 0
    )
    return {
        "schema": LIVING_REASONING_PREFLIGHT_SCHEMA,
        "check": PreflightEvidenceKind.ARCHITECTURE_CAPACITY.value,
        "architecture": report,
        "learned_position_modules": learned_positions,
        "source_position_scheme": "dynamic-sinusoidal-absolute-v1",
        "page_semantics": "bounded-compute-unit-complete-sequential-sweep",
        "soul_codec": model.soul_codec.tensor_layout,
        "passed": passed,
    }


def _curriculum_distribution(curriculum: LivingReasoningCurriculum, page_size: int) -> dict[str, Any]:
    tags = {
        split: sorted({tag for episode in curriculum.split(split) for tag in episode.mechanism_tags})
        for split in ("train", "heldout")
    }
    lengths = {
        split: [
            sum(len(region.attended_text) for region in episode.snapshot.regions)
            for episode in curriculum.split(split)
        ]
        for split in ("train", "heldout")
    }
    all_tags = set(tags["train"]) | set(tags["heldout"])
    required = {
        "head",
        "middle",
        "tail",
        "multi_page",
        "unicode",
        "proposal_refinement",
        "field_authority",
        "no_op",
        "abstain",
    }
    passed = required.issubset(all_tags) and all(
        any(length > page_size for length in lengths[split]) for split in lengths
    )
    return {
        "schema": LIVING_REASONING_PREFLIGHT_SCHEMA,
        "check": PreflightEvidenceKind.CURRICULUM_DISTRIBUTION.value,
        "curriculum_id": curriculum.curriculum_id,
        "train_manifest_id": curriculum.train_manifest_id,
        "heldout_manifest_id": curriculum.heldout_manifest_id,
        "episode_counts": {split: len(curriculum.split(split)) for split in lengths},
        "observed_active_character_lengths": lengths,
        "mechanism_tags": tags,
        "required_tags": sorted(required),
        "outcome_quality": sorted({item.outcome_quality for item in curriculum.episodes}),
        "serving_quality_claimed": False,
        "passed": passed,
    }


@torch.no_grad()
def _boundary_coverage(model: LivingReasoningCoreD64) -> dict[str, Any]:
    page = model.living_config.page_size
    lengths = sorted({1, max(1, page - 1), page, page + 1, page * 3 + 1})
    observations = []
    for length in lengths:
        text = ("λa🧠b" * ((length + 3) // 4))[:length]
        snapshot = SharedFieldSnapshot.from_texts({LogicalRegion.USER_INPUT: text})
        compiled = D64FieldCompiler().compile(snapshot)
        compiled.verify_roundtrip(snapshot)
        _state, _memory, manifest = model.read_compiled_with_memory(compiled)
        observations.append(
            {
                "characters": length,
                "transport_units": compiled.coverage.compiled_transport_units,
                "pages": manifest.page_count,
                "complete": manifest.complete,
                "observed_characters": manifest.observed_characters,
            }
        )
    passed = all(
        item["complete"] and item["observed_characters"] == item["characters"]
        for item in observations
    ) and observations[-1]["pages"] > len(tuple(LogicalRegion))
    return {
        "schema": LIVING_REASONING_PREFLIGHT_SCHEMA,
        "check": PreflightEvidenceKind.BOUNDARY_COVERAGE.value,
        "page_size": page,
        "observations": observations,
        "passed": passed,
    }


def _observable(output) -> torch.Tensor:
    return torch.cat(
        (
            output.reader_state.detach().flatten(),
            output.decision_logits.detach().flatten(),
            output.operation_logits.detach().flatten(),
            output.region_logits.detach().flatten(),
        )
    )


@torch.no_grad()
def _counterfactual_dependence(
    model: LivingReasoningCoreD64,
    *,
    core_id: str,
    parameter_generation: str,
) -> dict[str, Any]:
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: "Current canonical field outranks stale private assumptions.",
            LogicalRegion.USER_INPUT: "current truth=GREEN; proposal says RED",
        }
    )
    compiled = D64FieldCompiler().compile(snapshot)
    initial = SoulSnapshot(
        core_id=core_id,
        architecture_id=model.architecture_id,
        parameter_generation=parameter_generation,
        generation=0,
        parent_soul_id=None,
        layers=empty_soul_layers(),
    )
    first = model.forward_surfaces(
        soul=initial,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        phase="first",
        canonical=compiled,
    )
    transition = model.exhale_transition(
        before=initial,
        exhaled_state=first.exhaled_state,
        tick_uid="preflight-tick",
        request_id="preflight-first",
        phase="first",
    )
    correct_soul = apply_soul_transition(initial, transition)
    correct = model.forward_surfaces(
        soul=correct_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        phase="refined",
        canonical=compiled,
        proposal_texts=("brother proposal: RED",),
    )
    zero = model.forward_surfaces(
        soul=correct_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        phase="refined",
        canonical=compiled,
        proposal_texts=("brother proposal: RED",),
        ablate_temperatures=SOUL_TEMPERATURE_ORDER,
    )
    stale = model.forward_surfaces(
        soul=initial,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        phase="refined",
        canonical=compiled,
        proposal_texts=("brother proposal: RED",),
    )
    irrelevant = model.forward_surfaces(
        soul=correct_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        phase="refined",
        canonical=compiled,
        proposal_texts=("brother proposal: RED",),
        ablate_temperatures=(SoulTemperature.DEEP_COLD,),
    )
    changed_snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: "Current canonical field outranks stale private assumptions.",
            LogicalRegion.USER_INPUT: "current truth=BLUE; proposal says RED",
        }
    )
    changed_field = model.forward_surfaces(
        soul=correct_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        phase="refined",
        canonical=D64FieldCompiler().compile(changed_snapshot),
        proposal_texts=("brother proposal: RED",),
    )
    swapped_rejected = False
    try:
        model.forward_surfaces(
            soul=SoulSnapshot(
                core_id="swapped-brother",
                architecture_id=model.architecture_id,
                parameter_generation=parameter_generation,
                generation=correct_soul.generation,
                parent_soul_id=correct_soul.parent_soul_id,
                layers=correct_soul.layers,
            ),
            expected_core_id=core_id,
            parameter_generation=parameter_generation,
            phase="refined",
            canonical=compiled,
        )
    except ValueError:
        swapped_rejected = True
    correct_vector = _observable(correct)
    differences = {
        "zero_soul_l2": float((correct_vector - _observable(zero)).norm().item()),
        "stale_soul_l2": float((correct_vector - _observable(stale)).norm().item()),
        "irrelevant_empty_deep_cold_l2": float(
            (correct_vector - _observable(irrelevant)).norm().item()
        ),
        "current_field_change_l2": float(
            (correct_vector - _observable(changed_field)).norm().item()
        ),
    }
    passed = bool(
        differences["zero_soul_l2"] > 0.0
        and differences["stale_soul_l2"] > 0.0
        and differences["irrelevant_empty_deep_cold_l2"] == 0.0
        and differences["current_field_change_l2"] > 0.0
        and swapped_rejected
    )
    return {
        "schema": LIVING_REASONING_PREFLIGHT_SCHEMA,
        "check": PreflightEvidenceKind.COUNTERFACTUAL_DEPENDENCE.value,
        "mechanism_only_not_learned_behavior": True,
        "differences": differences,
        "swapped_brother_rejected": swapped_rejected,
        "current_field_override_behavior_proven": False,
        "passed": passed,
    }


@torch.no_grad()
def _checkpoint_compatibility(model: LivingReasoningCoreD64) -> dict[str, Any]:
    clone = LivingReasoningCoreD64(model.living_config).to(model.device)
    clone.load_state_dict(model.state_dict(), strict=True)
    names = tuple(model.state_dict())
    exact = all(torch.equal(model.state_dict()[name], clone.state_dict()[name]) for name in names)
    return {
        "schema": LIVING_REASONING_PREFLIGHT_SCHEMA,
        "check": PreflightEvidenceKind.CHECKPOINT_COMPATIBILITY.value,
        "architecture_id": model.architecture_id,
        "tensor_count": len(names),
        "strict_load": True,
        "exact_tensor_roundtrip": exact,
        "passed": exact,
    }


def build_living_reasoning_preflight(
    *,
    model: LivingReasoningCoreD64,
    curriculum: LivingReasoningCurriculum,
    inventory: ParameterInventory,
    plan: ParameterMutationPlan,
    state_root: Path | str,
    repo_root: Path | str,
    batch_size: int = 1,
) -> TrainingPreflightReceipt:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.eval()
    static = scan_active_capacity_poison(repo_root)
    payloads = {
        PreflightEvidenceKind.STATIC_CAPACITY_SCAN: static,
        PreflightEvidenceKind.ARCHITECTURE_CAPACITY: _architecture_capacity(model),
        PreflightEvidenceKind.CURRICULUM_DISTRIBUTION: _curriculum_distribution(
            curriculum, model.living_config.page_size
        ),
        PreflightEvidenceKind.BOUNDARY_COVERAGE: _boundary_coverage(model),
        PreflightEvidenceKind.COUNTERFACTUAL_DEPENDENCE: _counterfactual_dependence(
            model,
            core_id=plan.module_id,
            parameter_generation=plan.base_generation_id,
        ),
        PreflightEvidenceKind.CHECKPOINT_COMPATIBILITY: _checkpoint_compatibility(model),
    }
    evidence_root = Path(state_root).resolve() / "training" / "reasoning" / "preflight_evidence"
    for kind, payload in payloads.items():
        _atomic_json(evidence_root / f"{kind.value}.json", payload)
    evidence = tuple(
        TrainingPreflightEvidence.from_payload(
            kind=kind,
            payload=payload,
            summary=(
                f"Living reasoning {kind.value}: "
                + ("PASS" if payload["passed"] else "FAIL")
            ),
            passed=bool(payload["passed"]),
        )
        for kind, payload in payloads.items()
    )
    contract = CompleteFieldTrainingContract(
        module_id=plan.module_id,
        organ_kind=inventory.module(plan.module_id).descriptor.organ_kind.value,
        architecture=model.architecture_id,
        architecture_config_id=canonical_sha256(model.living_config.to_canonical_dict()),
        compiler_schema_ids=(D64_COMPILER_SCHEMA,),
        source_position_scheme="dynamic-sinusoidal-absolute-v1",
        target_position_scheme="autoregressive-no-learned-position-ceiling-v1",
        declared_bounds=(
            DeclaredTrainingBound(
                name="reasoning_page_characters",
                category=TrainingBoundCategory.PHYSICAL_PROCESSING_UNIT,
                value=model.living_config.page_size,
                source_preserved=True,
                continuation_or_failure="Continue recurrent Soul-conditioned sweep until every eligible character is visited.",
            ),
            DeclaredTrainingBound(
                name="optimizer_steps",
                category=TrainingBoundCategory.OPTIMIZATION_BUDGET,
                value=plan.max_steps,
                source_preserved=True,
                continuation_or_failure="Stop this candidate campaign and retain an exact resumable checkpoint.",
            ),
            DeclaredTrainingBound(
                name="training_batch_size",
                category=TrainingBoundCategory.COMPUTE_BUDGET,
                value=batch_size,
                source_preserved=True,
                continuation_or_failure="Accumulate later whole episodes without truncating any episode.",
            ),
        ),
    )
    return build_training_preflight_receipt(
        contract=contract,
        inventory=inventory,
        plan=plan,
        evidence=evidence,
    )


__all__ = [
    "LIVING_REASONING_PREFLIGHT_SCHEMA",
    "build_living_reasoning_preflight",
]
