"""Deterministic mechanism curriculum for the first living reasoning core.

These cases prove trainability of the permanent neural/runtime contract.  They
are not presented as factual supervision or a serving-quality corpus.  Real
lived-experience sessions are a separate governed source and retain their
explicit outcome-quality labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from runtime.field import (
    D64FieldCompiler,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_json_bytes,
    canonical_sha256,
)
from runtime.heart import (
    ProposalPass,
    ProposalWorkspace,
    ProposalWorkspaceEntry,
    ReasoningDecision,
    ReasoningOperationKind,
)
from runtime.soul import SoulSnapshot

from .complete_field_64d import sequence_cross_entropy
from .living_reasoning_d64 import (
    CausalLivingUnroll,
    LivingReasoningCoreD64,
    LivingReasoningForward,
)

LIVING_REASONING_TARGET_SCHEMA = "axon-living-reasoning-target-v1"
LIVING_REASONING_EPISODE_SCHEMA = "axon-living-reasoning-episode-v1"
LIVING_REASONING_CURRICULUM_SCHEMA = "axon-living-reasoning-curriculum-v1"


@dataclass(frozen=True, slots=True)
class LivingReasoningTarget:
    phase: str
    decision: ReasoningDecision | str
    operation: ReasoningOperationKind | str | None = None
    region: LogicalRegion | str | None = None
    start: int | None = None
    end: int | None = None
    payload: str = ""
    target_id: str = field(init=False)

    def __post_init__(self) -> None:
        decision = (
            self.decision
            if isinstance(self.decision, ReasoningDecision)
            else ReasoningDecision(self.decision)
        )
        object.__setattr__(self, "decision", decision)
        if self.phase not in {"first", "refined", "consolidated"}:
            raise ValueError("unsupported living reasoning target phase")
        if decision is ReasoningDecision.DELTA:
            operation = (
                self.operation
                if isinstance(self.operation, ReasoningOperationKind)
                else ReasoningOperationKind(self.operation)
            )
            region = self.region if isinstance(self.region, LogicalRegion) else LogicalRegion(self.region)
            if self.start is None or self.end is None or self.start < 0 or self.end < self.start:
                raise ValueError("delta target requires an ordered exact address")
            if operation is ReasoningOperationKind.INSERT and self.start != self.end:
                raise ValueError("insert target requires start == end")
            if operation is ReasoningOperationKind.DELETE and (self.end == self.start or self.payload):
                raise ValueError("delete target requires a nonempty range and empty payload")
            if operation is not ReasoningOperationKind.DELETE and not self.payload:
                raise ValueError("insert/replace target requires exact payload text")
            object.__setattr__(self, "operation", operation)
            object.__setattr__(self, "region", region)
        elif any(
            item is not None
            for item in (self.operation, self.region, self.start, self.end)
        ) or self.payload:
            raise ValueError("no-op/abstain target cannot carry an operation")
        object.__setattr__(self, "target_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": LIVING_REASONING_TARGET_SCHEMA,
            "phase": self.phase,
            "decision": self.decision.value,
            "operation": None if self.operation is None else self.operation.value,
            "region": None if self.region is None else self.region.value,
            "start": self.start,
            "end": self.end,
            "payload": self.payload,
        }
        if include_id:
            value["target_id"] = self.target_id
        return value


@dataclass(frozen=True, slots=True)
class LivingReasoningEpisode:
    label: str
    split: str
    snapshot: SharedFieldSnapshot
    first_workspace_text: str
    refined_workspace_text: str
    targets: tuple[LivingReasoningTarget, ...]
    mechanism_tags: tuple[str, ...]
    outcome_quality: str = "synthetic_mechanism"
    episode_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.split not in {"train", "heldout"}:
            raise ValueError("living reasoning split must be train or heldout")
        targets = tuple(self.targets)
        if tuple(item.phase for item in targets) != ("first", "refined", "consolidated"):
            raise ValueError("living episode requires exact three-phase target order")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "mechanism_tags", tuple(sorted(set(self.mechanism_tags))))
        object.__setattr__(self, "episode_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": LIVING_REASONING_EPISODE_SCHEMA,
            "label": self.label,
            "split": self.split,
            "field_id": self.snapshot.field_id,
            "first_workspace_sha256": canonical_sha256({"text": self.first_workspace_text}),
            "refined_workspace_sha256": canonical_sha256({"text": self.refined_workspace_text}),
            "targets": [item.to_canonical_dict() for item in self.targets],
            "mechanism_tags": list(self.mechanism_tags),
            "outcome_quality": self.outcome_quality,
        }
        if include_id:
            value["episode_id"] = self.episode_id
        return value


@dataclass(frozen=True, slots=True)
class LivingReasoningCurriculum:
    episodes: tuple[LivingReasoningEpisode, ...]
    curriculum_id: str = field(init=False)

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        if not episodes or {item.split for item in episodes} != {"train", "heldout"}:
            raise ValueError("living curriculum requires train and heldout episodes")
        if len({item.episode_id for item in episodes}) != len(episodes):
            raise ValueError("living curriculum contains duplicate episodes")
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "curriculum_id", canonical_sha256(self.to_canonical_dict(False)))

    def split(self, name: str) -> tuple[LivingReasoningEpisode, ...]:
        return tuple(item for item in self.episodes if item.split == name)

    def split_manifest_id(self, name: str) -> str:
        episodes = self.split(name)
        if not episodes:
            raise KeyError(f"living curriculum has no split {name!r}")
        return canonical_sha256(
            {
                "schema": "axon-living-reasoning-split-manifest-v1",
                "curriculum_id": self.curriculum_id,
                "split": name,
                "episode_ids": [item.episode_id for item in episodes],
            }
        )

    @property
    def train_manifest_id(self) -> str:
        return self.split_manifest_id("train")

    @property
    def heldout_manifest_id(self) -> str:
        return self.split_manifest_id("heldout")

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": LIVING_REASONING_CURRICULUM_SCHEMA,
            "episodes": [item.to_canonical_dict() for item in self.episodes],
        }
        if include_id:
            value["curriculum_id"] = self.curriculum_id
        return value


def _workspace_text(pass_kind: ProposalPass, detail: str) -> str:
    workspace = ProposalWorkspace(
        image_id="synthetic-mechanism-image-v1",
        tick_uid="synthetic-mechanism-tick-v1",
        pass_kind=pass_kind,
        entries=(
            ProposalWorkspaceEntry(
                core_id="brother-fixture",
                d_model=64,
                state="returned",
                detail=detail,
                emission_id=None,
                delta=None,
            ),
        ),
    )
    return workspace.readable_text()


def _episode(
    *,
    label: str,
    split: str,
    user_text: str,
    answer: str,
    first_decision: ReasoningDecision,
    refined_decision: ReasoningDecision,
    tags: Iterable[str],
) -> LivingReasoningEpisode:
    scratch = ""
    response = ""
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: (
                "Axon follows current canonical evidence and preserves exact Unicode transport."
            ),
            LogicalRegion.USER_INPUT: user_text,
            LogicalRegion.SCRATCH: scratch,
            LogicalRegion.RESPONSE_DRAFT: response,
        }
    )

    def phase_target(phase: str, decision: ReasoningDecision) -> LivingReasoningTarget:
        if decision is not ReasoningDecision.DELTA:
            return LivingReasoningTarget(phase=phase, decision=decision)
        return LivingReasoningTarget(
            phase=phase,
            decision=decision,
            operation=ReasoningOperationKind.REPLACE,
            region=LogicalRegion.SCRATCH,
            start=0,
            end=0,
            payload=f"evidence:{answer}",
        )

    return LivingReasoningEpisode(
        label=label,
        split=split,
        snapshot=snapshot,
        first_workspace_text=_workspace_text(
            ProposalPass.FIRST,
            f"Brother proposes exact candidate {answer!r}.",
        ),
        refined_workspace_text=_workspace_text(
            ProposalPass.REFINED,
            f"Brother rechecks current field and retains {answer!r}.",
        ),
        targets=(
            phase_target("first", first_decision),
            phase_target("refined", refined_decision),
            LivingReasoningTarget(
                phase="consolidated",
                decision=ReasoningDecision.DELTA,
                operation=ReasoningOperationKind.REPLACE,
                region=LogicalRegion.RESPONSE_DRAFT,
                start=0,
                end=0,
                payload=answer,
            ),
        ),
        mechanism_tags=tuple(tags),
    )


def build_living_reasoning_smoke_curriculum() -> LivingReasoningCurriculum:
    middle_prefix = "irrelevant " * 18
    middle_suffix = " trailing" * 18
    return LivingReasoningCurriculum(
        episodes=(
            _episode(
                label="unicode-head-copy",
                split="train",
                user_text="Evidence λ🧠 is at the head. Return exactly λ🧠.",
                answer="λ🧠",
                first_decision=ReasoningDecision.DELTA,
                refined_decision=ReasoningDecision.ABSTAIN,
                tags=("unicode", "head", "copy", "abstain"),
            ),
            _episode(
                label="middle-multipage-evidence",
                split="train",
                user_text=middle_prefix + "MIDDLE=blue-copper" + middle_suffix,
                answer="blue-copper",
                first_decision=ReasoningDecision.NO_OP,
                refined_decision=ReasoningDecision.DELTA,
                tags=("middle", "multi_page", "proposal_refinement", "no_op"),
            ),
            _episode(
                label="current-field-overrides-proposal",
                split="train",
                user_text="Canonical current truth: gate=GREEN. Ignore any proposal claiming RED.",
                answer="GREEN",
                first_decision=ReasoningDecision.DELTA,
                refined_decision=ReasoningDecision.DELTA,
                tags=("field_authority", "conflict", "proposal"),
            ),
            _episode(
                label="unicode-tail-heldout",
                split="heldout",
                user_text=("early filler " * 25) + "tail evidence=終端✅",
                answer="終端✅",
                first_decision=ReasoningDecision.NO_OP,
                refined_decision=ReasoningDecision.DELTA,
                tags=("unicode", "tail", "multi_page", "heldout"),
            ),
        )
    )


def living_phase_objective(
    model: LivingReasoningCoreD64,
    output: LivingReasoningForward,
    target: LivingReasoningTarget,
) -> tuple[torch.Tensor, dict[str, float]]:
    decision_index = tuple(ReasoningDecision).index(target.decision)
    decision_loss = F.cross_entropy(
        output.decision_logits,
        torch.tensor([decision_index], dtype=torch.long, device=model.device),
    )
    loss = decision_loss
    metrics = {"decision_loss": float(decision_loss.detach().item())}
    if target.decision is not ReasoningDecision.DELTA:
        return loss, metrics

    operation_index = tuple(ReasoningOperationKind).index(target.operation)
    operation_loss = F.cross_entropy(
        output.operation_logits,
        torch.tensor([operation_index], dtype=torch.long, device=model.device),
    )
    region_index = tuple(LogicalRegion).index(target.region)
    region_loss = F.cross_entropy(
        output.region_logits,
        torch.tensor([region_index], dtype=torch.long, device=model.device),
    )
    candidates, start_logits, end_logits = model.boundary_logits(output, target.region)
    if target.start not in candidates or target.end not in candidates:
        raise ValueError("curriculum target address is not present in the exact active field")
    start_loss = F.cross_entropy(
        start_logits,
        torch.tensor([candidates.index(target.start)], dtype=torch.long, device=model.device),
    )
    end_loss = F.cross_entropy(
        end_logits,
        torch.tensor([candidates.index(target.end)], dtype=torch.long, device=model.device),
    )
    payload_logits, payload_targets = model.decode_teacher(
        output.reader_state,
        target.payload,
        head=1,
        memory=output.complete_memory,
    )
    payload_loss = sequence_cross_entropy(payload_logits, payload_targets)
    loss = loss + operation_loss + region_loss + start_loss + end_loss + payload_loss
    metrics.update(
        {
            "operation_loss": float(operation_loss.detach().item()),
            "region_loss": float(region_loss.detach().item()),
            "start_loss": float(start_loss.detach().item()),
            "end_loss": float(end_loss.detach().item()),
            "payload_loss": float(payload_loss.detach().item()),
        }
    )
    return loss, metrics


def living_episode_objective(
    model: LivingReasoningCoreD64,
    episode: LivingReasoningEpisode,
    initial_soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
) -> tuple[torch.Tensor, CausalLivingUnroll, tuple[dict[str, float], ...]]:
    compiled = D64FieldCompiler().compile(episode.snapshot)
    compiled.verify_roundtrip(episode.snapshot)
    unroll = model.unroll_runtime_phases(
        initial_soul=initial_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        tick_uid=f"training-episode:{episode.episode_id}",
        canonical=compiled,
        first_workspace_text=episode.first_workspace_text,
        refined_workspace_text=episode.refined_workspace_text,
    )
    phase_results = tuple(
        living_phase_objective(model, output, target)
        for output, target in zip(unroll.outputs, episode.targets, strict=True)
    )
    total = torch.stack([item[0] for item in phase_results]).mean()
    return total, unroll, tuple(item[1] for item in phase_results)


def curriculum_manifest_bytes(curriculum: LivingReasoningCurriculum) -> bytes:
    return canonical_json_bytes(curriculum.to_canonical_dict())


__all__ = [
    "LIVING_REASONING_CURRICULUM_SCHEMA",
    "LIVING_REASONING_EPISODE_SCHEMA",
    "LIVING_REASONING_TARGET_SCHEMA",
    "LivingReasoningCurriculum",
    "LivingReasoningEpisode",
    "LivingReasoningTarget",
    "build_living_reasoning_smoke_curriculum",
    "curriculum_manifest_bytes",
    "living_episode_objective",
    "living_phase_objective",
]
