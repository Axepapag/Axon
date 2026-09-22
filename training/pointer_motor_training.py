"""Direct canonical-address pointer-motor curriculum and diagnostics.

The pointer motor is deliberately narrower than English reasoning.  A sample
names an exact canonical address, the existing complete-field reader produces
the memory keys, and only the versioned query scaffold is optimized.  The
module keeps the ordinary LivingReasoningEpisode/Soul contracts so every
optimizer step still inhales a private Soul and emits a recoverable transition.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from runtime.field import D64FieldCompiler, LogicalRegion, SharedFieldSnapshot, canonical_sha256
from runtime.soul import SoulSnapshot, SoulTemperature, SoulTransition, apply_soul_transition
from substrate import encode_unicode_text

from .living_reasoning_curriculum import (
    LivingReasoningCurriculum,
    LivingReasoningEpisode,
    LivingReasoningTarget,
)
from .living_reasoning_d64 import (
    D64_SOUL_MEDIA_TYPE,
    D64SoulCodec,
    LivingReasoningCoreConfig,
    LivingReasoningCoreD64,
)

POINTER_MOTOR_CURRICULUM_SCHEMA = "axon-pointer-motor-curriculum-v1"
POINTER_MOTOR_OBJECTIVE_SCHEMA = "axon-pointer-motor-objective-v1"
POINTER_SCAFFOLD_MIGRATION_SCHEMA = "axon-pointer-scaffold-migration-v1"
POINTER_MOTOR_GATE_SCHEMA = "axon-pointer-motor-gate-v1"

POINTER_MOTOR_REQUIRED_TAGS = (
    "english",
    "proposal",
    "refinement",
    "tagged_final",
    "pointer_motor",
    "canonical_address",
    "native_one_cell",
    "persistent_soul",
    "exact_address",
    "unicode",
)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class PointerScaffoldTensorReceipt:
    name: str
    shape: tuple[int, ...]
    dtype: str
    source_sha256: str | None
    target_sha256: str
    disposition: str

    def __post_init__(self) -> None:
        if not self.name or not self.dtype or not self.target_sha256:
            raise ValueError("pointer scaffold tensor receipt fields must be non-empty")
        if self.disposition not in {"copied_exact", "initialized_new"}:
            raise ValueError("unsupported pointer scaffold tensor disposition")
        if self.disposition == "copied_exact" and self.source_sha256 != self.target_sha256:
            raise ValueError("copied pointer scaffold tensor is not byte exact")

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "source_sha256": self.source_sha256,
            "target_sha256": self.target_sha256,
            "disposition": self.disposition,
        }


@dataclass(frozen=True, slots=True)
class PointerScaffoldMigrationReceipt:
    source_architecture_id: str
    target_architecture_id: str
    source_parameter_generation: str
    target_parameter_generation: str
    source_checkpoint_id: str
    source_bundle_id: str
    source_soul_id: str
    copied_tensors: tuple[PointerScaffoldTensorReceipt, ...]
    initialized_tensors: tuple[PointerScaffoldTensorReceipt, ...]
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "source_architecture_id",
            "target_architecture_id",
            "source_parameter_generation",
            "target_parameter_generation",
            "source_checkpoint_id",
            "source_bundle_id",
            "source_soul_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.source_architecture_id == self.target_architecture_id:
            raise ValueError("pointer scaffold migration must create a new architecture identity")
        if self.source_parameter_generation == self.target_parameter_generation:
            raise ValueError("pointer scaffold migration must create a new parameter generation")
        copied = tuple(sorted(self.copied_tensors, key=lambda item: item.name))
        initialized = tuple(sorted(self.initialized_tensors, key=lambda item: item.name))
        if not copied or not initialized:
            raise ValueError("pointer scaffold migration needs copied and initialized tensors")
        if any(item.disposition != "copied_exact" for item in copied):
            raise ValueError("copied_tensors contains a non-copied receipt")
        if any(item.disposition != "initialized_new" for item in initialized):
            raise ValueError("initialized_tensors contains a non-initialized receipt")
        if set(item.name for item in copied) & set(item.name for item in initialized):
            raise ValueError("migration tensor surfaces overlap")
        object.__setattr__(self, "copied_tensors", copied)
        object.__setattr__(self, "initialized_tensors", initialized)
        object.__setattr__(self, "receipt_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": POINTER_SCAFFOLD_MIGRATION_SCHEMA,
            "source_architecture_id": self.source_architecture_id,
            "target_architecture_id": self.target_architecture_id,
            "source_parameter_generation": self.source_parameter_generation,
            "target_parameter_generation": self.target_parameter_generation,
            "source_checkpoint_id": self.source_checkpoint_id,
            "source_bundle_id": self.source_bundle_id,
            "source_soul_id": self.source_soul_id,
            "copied_tensors": [item.to_canonical_dict() for item in self.copied_tensors],
            "initialized_tensors": [item.to_canonical_dict() for item in self.initialized_tensors],
        }
        if include_id:
            value["receipt_id"] = self.receipt_id
        return value


def initialize_pointer_scaffold_from_donor(
    model: LivingReasoningCoreD64,
    donor_state: Mapping[str, torch.Tensor],
    *,
    source_architecture_id: str,
    source_parameter_generation: str,
    target_parameter_generation: str,
    source_checkpoint_id: str,
    source_bundle_id: str,
    source_soul_id: str,
) -> PointerScaffoldMigrationReceipt:
    """Load an old English donor while proving the four new tensors are fresh."""

    if model.living_config.pointer_address_scaffold_version != "query-scaffold-v1":
        raise ValueError("pointer scaffold donor initialization requires query-scaffold-v1")
    target_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    donor = dict(donor_state)
    target_names = set(target_before)
    donor_names = set(donor)
    missing_expected = {
        "pointer_address_query.0.weight",
        "pointer_address_query.0.bias",
        "pointer_address_query.2.weight",
        "pointer_address_query.2.bias",
    }
    if not missing_expected.issubset(target_names):
        raise ValueError("target model does not expose the complete pointer scaffold")
    # The only permitted missing keys are the four newly introduced scaffold
    # tensors; the explicit post-load identity checks below make this a
    # bounded migration rather than a forgiving resume.
    incompatible = model.load_state_dict(donor, strict=bool(0))
    if set(incompatible.unexpected_keys):
        raise ValueError(f"donor contains unexpected tensors: {incompatible.unexpected_keys}")
    if set(incompatible.missing_keys) != missing_expected:
        raise ValueError(
            "donor mismatch is outside the approved pointer scaffold: "
            f"missing={incompatible.missing_keys}"
        )
    current = model.state_dict()
    copied: list[PointerScaffoldTensorReceipt] = []
    for name in sorted(donor_names):
        if name not in target_names:
            raise ValueError(f"donor tensor {name!r} is absent from the target")
        source = donor[name]
        target = current[name]
        if tuple(source.shape) != tuple(target.shape) or source.dtype != target.dtype:
            raise ValueError(f"donor tensor anatomy differs for {name!r}")
        source_hash = _tensor_sha256(source)
        target_hash = _tensor_sha256(target)
        copied.append(
            PointerScaffoldTensorReceipt(
                name=name,
                shape=tuple(source.shape),
                dtype=str(source.dtype),
                source_sha256=source_hash,
                target_sha256=target_hash,
                disposition="copied_exact",
            )
        )
    initialized = [
        PointerScaffoldTensorReceipt(
            name=name,
            shape=tuple(current[name].shape),
            dtype=str(current[name].dtype),
            source_sha256=None,
            target_sha256=_tensor_sha256(current[name]),
            disposition="initialized_new",
        )
        for name in sorted(missing_expected)
    ]
    return PointerScaffoldMigrationReceipt(
        source_architecture_id=source_architecture_id,
        target_architecture_id=model.architecture_id,
        source_parameter_generation=source_parameter_generation,
        target_parameter_generation=target_parameter_generation,
        source_checkpoint_id=source_checkpoint_id,
        source_bundle_id=source_bundle_id,
        source_soul_id=source_soul_id,
        copied_tensors=tuple(copied),
        initialized_tensors=tuple(initialized),
    )


def _cortex_text(seed: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    rotated = alphabet[seed % len(alphabet) :] + alphabet[: seed % len(alphabet)]
    return (rotated * 2)[:68]


def _pointer_episode(*, split: str, position: int, seed: int) -> LivingReasoningEpisode:
    cortex = _cortex_text(seed)
    user = f"#read# cortex {position:02d} | canonical address motor exercise {seed:02d}."
    # Keep the command and every region exact Unicode, but make the Cortex
    # contents independent of the requested address so content cannot answer
    # the pointer question for the model.
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: "Axon pointer motor training.",
            LogicalRegion.USER_INPUT: user,
            LogicalRegion.CORTEX: cortex,
            LogicalRegion.ADVISOR_INPUT: "Use the canonical Cortex address only.",
        },
        source_manifest_ids=(f"pointer-motor:{split}:{seed}",),
    )
    target = "A"
    return LivingReasoningEpisode(
        label=f"pointer-motor-{split}-position-{position:03d}-seed-{seed:03d}",
        split=split,
        snapshot=snapshot,
        first_workspace_text="[pointer-motor-first] canonical address selected",
        refined_workspace_text="[pointer-motor-refined] exact source cell required",
        targets=(
            LivingReasoningTarget(phase="first", text=target),
            LivingReasoningTarget(phase="refined", text=target),
            LivingReasoningTarget(phase="consolidated", text="#responseDraft# A"),
        ),
        mechanism_tags=POINTER_MOTOR_REQUIRED_TAGS,
        target_basis=f"pointer-motor-v1|region=cortex|position={position}",
    )


def build_pointer_motor_curriculum() -> LivingReasoningCurriculum:
    """Train every Cortex address, then retest it with different content."""

    train = tuple(
        _pointer_episode(split="train", position=position, seed=position + 11)
        for position in range(68)
    )
    heldout = tuple(
        _pointer_episode(split="heldout", position=position, seed=position + 137)
        for position in range(68)
    )
    return LivingReasoningCurriculum(episodes=train + heldout)


def pointer_target_address(episode: LivingReasoningEpisode) -> tuple[LogicalRegion, int]:
    prefix = "pointer-motor-v1|region=cortex|position="
    if not episode.target_basis.startswith(prefix):
        raise ValueError("episode is not a pointer-motor target")
    try:
        position = int(episode.target_basis[len(prefix) :])
    except ValueError as exc:
        raise ValueError("pointer-motor target position is invalid") from exc
    if position < 0:
        raise ValueError("pointer-motor target position must be non-negative")
    return LogicalRegion.CORTEX, position


def _source_index(memory, region: LogicalRegion, position: int) -> int:
    matches = [
        index
        for index, receipt in enumerate(memory.receipts)
        if receipt is not None
        and receipt.address.region is region
        and receipt.address.region_position == position
    ]
    if len(matches) != 1:
        raise ValueError(f"canonical address does not resolve to one source cell: {region.value}:{position}")
    return matches[0]


def pointer_motor_episode_objective(
    model: LivingReasoningCoreD64,
    episode: LivingReasoningEpisode,
    soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
) -> tuple[torch.Tensor, tuple[SoulTransition, ...], dict[str, Any]]:
    """One exact-address motor loss with a complete FIRST/REFINED/FINAL chain."""

    region, position = pointer_target_address(episode)
    compiled = D64FieldCompiler().compile(episode.snapshot)
    compiled.verify_roundtrip(episode.snapshot)
    phase_outputs = []
    phase_souls = soul
    for phase, proposals in (
        ("first", ()),
        ("refined", ("[pointer-motor] exact canonical address is authoritative.",)),
        ("consolidated", ("[pointer-motor] exact canonical address is authoritative.",)),
    ):
        output = model.forward_surfaces(
            soul=phase_souls,
            expected_core_id=core_id,
            parameter_generation=parameter_generation,
            phase=phase,
            canonical=compiled,
            proposal_texts=proposals,
        )
        transition = model.exhale_transition(
            before=phase_souls,
            exhaled_state=output.exhaled_state,
            tick_uid=f"pointer-motor:{episode.episode_id}:{phase}",
            request_id=f"pointer-motor:{episode.episode_id}:{phase}",
            phase=phase,
        )
        phase_outputs.append((phase, output, transition))
        phase_souls = apply_soul_transition(phase_souls, transition)
    _phase, output, _first_transition = phase_outputs[0]
    target_index = _source_index(output.complete_memory, region, position)
    logits = model.pointer_motor_logits(
        output.complete_memory,
        region=region,
        position=position,
    )
    target = torch.tensor([target_index], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, target)
    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=-1)
        ordered = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
        margin = float((logits[0, target_index] - ordered[0, 1]).item()) if logits.shape[-1] > 1 else float("inf")
    return loss, tuple(item[2] for item in phase_outputs), {
        "episode_id": episode.episode_id,
        "region": region.value,
        "position": position,
        "target_memory_index": target_index,
        "target_probability": float(probabilities[0, target_index].item()),
        "top1_memory_index": int(logits[0].argmax().item()),
        "top1": int(int(logits[0].argmax().item()) == target_index),
        "margin": margin,
        "loss": float(loss.detach().item()),
    }


@torch.no_grad()
def evaluate_pointer_motor(
    model: LivingReasoningCoreD64,
    episodes: Iterable[LivingReasoningEpisode],
    soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    queries: list[torch.Tensor] = []
    episode_keys: list[tuple[str, dict[int, torch.Tensor]]] = []
    positions: set[int] = set()
    for episode in episodes:
        region, position = pointer_target_address(episode)
        compiled = D64FieldCompiler().compile(episode.snapshot)
        output = model.forward_surfaces(
            soul=soul,
            expected_core_id=core_id,
            parameter_generation=parameter_generation,
            phase="first",
            canonical=compiled,
        )
        target_index = _source_index(output.complete_memory, region, position)
        logits = model.pointer_motor_logits(output.complete_memory, region=region, position=position)
        probabilities = torch.softmax(logits, dim=-1)
        best = int(logits[0].argmax().item())
        sorted_values = torch.sort(logits[0], descending=True).values
        margin = float((logits[0, target_index] - sorted_values[1]).item())
        queries.append(model.canonical_pointer_query(region, position).detach().float().cpu())
        key_vectors = model.position_key(output.complete_memory.states)[0].detach().float().cpu()
        episode_keys.append(
            (
                episode.episode_id,
                {
                    int(receipt.address.region_position): key_vectors[index]
                    for index, receipt in enumerate(output.complete_memory.receipts)
                    if receipt is not None and receipt.address.region is LogicalRegion.CORTEX
                },
            )
        )
        positions.add(position)
        rows.append(
            {
                "episode_id": episode.episode_id,
                "position": position,
                "target_memory_index": target_index,
                "top1_memory_index": best,
                "top1": int(best == target_index),
                "target_probability": float(probabilities[0, target_index].item()),
                "margin": margin,
            }
        )
    if not rows:
        raise ValueError("pointer motor evaluation requires at least one episode")
    query_matrix = torch.stack(queries)
    query_variance = float(query_matrix.var(dim=0, unbiased=False).mean().item())
    top1_rate = sum(row["top1"] for row in rows) / len(rows)
    probability_mean = sum(row["target_probability"] for row in rows) / len(rows)
    margins = [float(row["margin"]) for row in rows]
    return {
        "schema": "axon-pointer-motor-evaluation-v1",
        "count": len(rows),
        "unique_target_positions": len(positions),
        "target_position_coverage_rate": len(positions) / 68.0,
        "exact_top1_rate": top1_rate,
        "target_probability_mean": probability_mean,
        "margin_mean": sum(margins) / len(margins),
        "margin_min": min(margins),
        "query_variance_mean": query_variance,
        "key_cross_episode_address_accuracy": pointer_key_cross_episode_accuracy(episode_keys),
        "rows": rows,
    }


def pointer_key_cross_episode_accuracy(
    episode_keys: Iterable[tuple[str, Mapping[int, torch.Tensor]]],
) -> dict[str, float]:
    """Measure whether the frozen key space is stable across field contexts."""

    items = tuple(episode_keys)
    if len(items) < 2:
        raise ValueError("key geometry requires at least two episodes")
    positions = tuple(sorted(set.intersection(*(set(keys) for _name, keys in items))))
    if not positions:
        raise ValueError("key geometry has no common Cortex positions")
    correct = 0
    total = 0
    for heldout_name, heldout in items:
        centroids = []
        for position in positions:
            samples = [
                keys[position].to(dtype=torch.float64)
                for name, keys in items
                if name != heldout_name
            ]
            centroids.append(F.normalize(torch.stack(samples).mean(dim=0), dim=0))
        centroid_matrix = torch.stack(centroids)
        for position in positions:
            query = F.normalize(heldout[position].to(dtype=torch.float64), dim=0)
            predicted = positions[int((centroid_matrix @ query).argmax().item())]
            correct += int(predicted == position)
            total += 1
    return {
        "eligible_count": float(total),
        "correct_count": float(correct),
        "accuracy": correct / max(1, total),
        "position_count": float(len(positions)),
        "uniform_chance": 1.0 / len(positions),
    }


def decide_pointer_motor_mastery(
    evaluation: Mapping[str, Any],
    *,
    minimum_margin: float = 0.25,
) -> dict[str, Any]:
    """Gate exact selection without requiring pathological softmax saturation."""

    top1 = float(evaluation.get("exact_top1_rate", 0.0))
    coverage = float(evaluation.get("target_position_coverage_rate", 0.0))
    margin = float(evaluation.get("margin_min", float("-inf")))
    variance = float(evaluation.get("query_variance_mean", 0.0))
    passed = bool(
        math.isfinite(top1)
        and math.isfinite(coverage)
        and math.isfinite(margin)
        and math.isfinite(variance)
        and top1 >= 1.0
        and coverage >= 1.0
        and margin >= minimum_margin
        and variance > 0.0
    )
    return {
        "schema": POINTER_MOTOR_GATE_SCHEMA,
        "passed": passed,
        "requirements": {
            "exact_top1_rate": 1.0,
            "target_position_coverage_rate": 1.0,
            "minimum_margin": minimum_margin,
            "query_variance_mean_positive": True,
        },
        "observed": {
            "exact_top1_rate": top1,
            "target_position_coverage_rate": coverage,
            "margin_min": margin,
            "query_variance_mean": variance,
        },
        "assignment_completed": passed,
    }


def rebind_d64_soul_layers(
    source: SoulSnapshot,
    target_config: LivingReasoningCoreConfig,
) -> tuple[Any, ...]:
    """Retag the architecture-owned recurrent payload for a same-shape D64 target."""

    codec = D64SoulCodec(target_config)
    rebound = []
    for layer in source.layers:
        if layer.temperature is SoulTemperature.HOT and layer.payload:
            rebound.append(
                type(layer)(
                    temperature=layer.temperature,
                    payload=layer.payload,
                    media_type=D64_SOUL_MEDIA_TYPE,
                    tensor_layout=codec.tensor_layout,
                )
            )
        else:
            rebound.append(layer)
    return tuple(rebound)


__all__ = [
    "POINTER_MOTOR_CURRICULUM_SCHEMA",
    "POINTER_MOTOR_OBJECTIVE_SCHEMA",
    "POINTER_MOTOR_REQUIRED_TAGS",
    "POINTER_SCAFFOLD_MIGRATION_SCHEMA",
    "PointerScaffoldTensorReceipt",
    "PointerScaffoldMigrationReceipt",
    "initialize_pointer_scaffold_from_donor",
    "build_pointer_motor_curriculum",
    "pointer_target_address",
    "pointer_motor_episode_objective",
    "evaluate_pointer_motor",
    "pointer_key_cross_episode_accuracy",
    "decide_pointer_motor_mastery",
    "rebind_d64_soul_layers",
]
