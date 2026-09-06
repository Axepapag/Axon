"""Deterministic mechanism curriculum for the first living reasoning core.

These cases prove trainability of the permanent neural/runtime contract.  They
are not presented as factual supervision or a serving-quality corpus.  Real
lived-experience sessions are a separate governed source and retain their
explicit outcome-quality labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

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
from runtime.soul import SoulSnapshot, SoulTemperature

from .complete_field_64d import sequence_cross_entropy
from .living_reasoning_d64 import (
    CausalLivingUnroll,
    LivingReasoningCoreD64,
    LivingReasoningForward,
)

LIVING_REASONING_TARGET_SCHEMA = "axon-living-reasoning-target-v1"
LIVING_REASONING_EPISODE_SCHEMA = "axon-living-reasoning-episode-v1"
LIVING_REASONING_CURRICULUM_SCHEMA = "axon-living-reasoning-curriculum-v1"
LIVING_OBJECTIVE_COMPONENTS = (
    "decision",
    "operation",
    "region",
    "start",
    "end",
    "payload",
    "alignment_position",
    "alignment_copy_gate",
    "alignment_eos_gate",
)


@dataclass(frozen=True, slots=True)
class LivingReasoningTarget:
    phase: str
    decision: ReasoningDecision | str
    operation: ReasoningOperationKind | str | None = None
    region: LogicalRegion | str | None = None
    start: int | None = None
    end: int | None = None
    payload: str = ""
    payload_alignment: Mapping[str, Any] | None = None
    supervision_weight: float = 1.0
    target_id: str = field(init=False)

    def __post_init__(self) -> None:
        decision = (
            self.decision
            if isinstance(self.decision, ReasoningDecision)
            else ReasoningDecision(self.decision)
        )
        object.__setattr__(self, "decision", decision)
        weight = float(self.supervision_weight)
        if not 0.0 <= weight < float("inf"):
            raise ValueError("supervision_weight must be finite and non-negative")
        object.__setattr__(self, "supervision_weight", weight)
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
        if self.payload_alignment is not None:
            alignment = dict(self.payload_alignment)
            if decision is not ReasoningDecision.DELTA or operation is ReasoningOperationKind.DELETE:
                raise ValueError("payload alignment requires a non-delete delta target")
            if set(alignment) != {"schema", "segments", "supervise_eos_generate"}:
                raise ValueError("payload alignment fields are invalid")
            if alignment["schema"] != "axon-r0-target-alignment-v1":
                raise ValueError("unsupported payload alignment schema")
            if not isinstance(alignment["segments"], list) or not alignment["segments"]:
                raise ValueError("payload alignment requires one or more exact source segments")
            if alignment["supervise_eos_generate"] is not True:
                raise ValueError("payload alignment must supervise EOS as generated")
            object.__setattr__(self, "payload_alignment", alignment)
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
            "supervision_weight": self.supervision_weight,
        }
        if self.payload_alignment is not None:
            value["payload_alignment"] = dict(self.payload_alignment)
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
    source_example_id: str | None = None
    outcome_evidence_ids: tuple[str, ...] = ()
    target_basis: str = "synthetic_mechanism"
    episode_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.split not in {"train", "heldout", "regression"}:
            raise ValueError("living reasoning split must be train, heldout, or regression")
        targets = tuple(self.targets)
        if tuple(item.phase for item in targets) != ("first", "refined", "consolidated"):
            raise ValueError("living episode requires exact three-phase target order")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "mechanism_tags", tuple(sorted(set(self.mechanism_tags))))
        if self.source_example_id is not None and len(self.source_example_id) != 64:
            raise ValueError("source_example_id must be None or a content identity")
        evidence = tuple(sorted(set(map(str, self.outcome_evidence_ids))))
        object.__setattr__(self, "outcome_evidence_ids", evidence)
        if not isinstance(self.target_basis, str) or not self.target_basis:
            raise ValueError("target_basis must be non-empty")
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
            "source_example_id": self.source_example_id,
            "outcome_evidence_ids": list(self.outcome_evidence_ids),
            "target_basis": self.target_basis,
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
        if not episodes or not {"train", "heldout"}.issubset(
            {item.split for item in episodes}
        ):
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
    *,
    component_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights: dict[str, float] | None = None
    if component_weights is not None:
        unknown = set(component_weights) - set(LIVING_OBJECTIVE_COMPONENTS)
        if unknown:
            raise ValueError(f"unknown living objective components: {sorted(unknown)}")
        weights = {}
        for component in LIVING_OBJECTIVE_COMPONENTS:
            value = float(component_weights.get(component, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("living objective component weights must be finite and non-negative")
            weights[component] = value

    def weighted(component: str, value: torch.Tensor) -> torch.Tensor:
        return value if weights is None else value * weights[component]

    decision_index = tuple(ReasoningDecision).index(target.decision)
    decision_loss = F.cross_entropy(
        output.decision_logits,
        torch.tensor([decision_index], dtype=torch.long, device=model.device),
    )
    loss = weighted("decision", decision_loss)
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
    decoded = model.decode_teacher(
        output.reader_state,
        target.payload,
        head=1,
        memory=output.complete_memory,
        return_alignment=target.payload_alignment is not None,
    )
    if target.payload_alignment is None:
        payload_logits, payload_targets = decoded
        alignment_supervision = None
    else:
        payload_logits, payload_targets, decoder_alignment = decoded
        alignment_supervision = model.alignment_supervision(
            target_text=target.payload,
            memory=output.complete_memory,
            decoder_alignment=decoder_alignment,
            specification=target.payload_alignment,
        )
    payload_loss = sequence_cross_entropy(payload_logits, payload_targets)
    loss = (
        loss
        + weighted("operation", operation_loss)
        + weighted("region", region_loss)
        + weighted("start", start_loss)
        + weighted("end", end_loss)
        + weighted("payload", payload_loss)
    )
    metrics.update(
        {
            "operation_loss": float(operation_loss.detach().item()),
            "region_loss": float(region_loss.detach().item()),
            "start_loss": float(start_loss.detach().item()),
            "end_loss": float(end_loss.detach().item()),
            "payload_loss": float(payload_loss.detach().item()),
        }
    )
    if alignment_supervision is not None:
        position_loss = alignment_supervision["position_loss"]
        gate_loss = alignment_supervision["gate_loss"]
        copy_gate_loss = alignment_supervision["copy_gate_loss"]
        eos_gate_loss = alignment_supervision["eos_gate_loss"]
        if weights is None:
            # Preserve the historical objective exactly for every existing
            # manifest/checkpoint. The staged v2 program opts into separated
            # copy and EOS terms through explicit content-addressed weights.
            loss = loss + position_loss + gate_loss
        else:
            loss = (
                loss
                + weighted("alignment_position", position_loss)
                + weighted("alignment_copy_gate", copy_gate_loss)
                + weighted("alignment_eos_gate", eos_gate_loss)
            )
        metrics.update(
            {
                "alignment_position_loss": float(position_loss.detach().item()),
                "alignment_gate_loss": float(gate_loss.detach().item()),
                "alignment_copy_gate_loss": float(copy_gate_loss.detach().item()),
                "alignment_eos_gate_loss": float(eos_gate_loss.detach().item()),
                "alignment_copy_positions": float(alignment_supervision["copy_positions"]),
                "alignment_position_accuracy": float(alignment_supervision["position_accuracy"]),
                "alignment_copy_gate_accuracy": float(
                    alignment_supervision["copy_gate_accuracy"]
                ),
                "alignment_eos_gate_accuracy": float(
                    alignment_supervision["eos_gate_accuracy"]
                ),
                "alignment_gate_accuracy": float(alignment_supervision["gate_accuracy"]),
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
    ablate_temperatures: tuple[SoulTemperature, ...] = (),
    component_weights: Mapping[str, float] | None = None,
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
        ablate_temperatures=ablate_temperatures,
    )
    phase_results = tuple(
        living_phase_objective(
            model,
            output,
            target,
            component_weights=component_weights,
        )
        for output, target in zip(unroll.outputs, episode.targets, strict=True)
    )
    weights = torch.tensor(
        [item.supervision_weight for item in episode.targets],
        dtype=phase_results[0][0].dtype,
        device=model.device,
    )
    if not bool(weights.gt(0).any()):
        raise ValueError("living episode has no supervised phase")
    total = (torch.stack([item[0] for item in phase_results]) * weights).sum() / weights.sum()
    return total, unroll, tuple(item[1] for item in phase_results)


@torch.no_grad()
def evaluate_living_episode(
    model: LivingReasoningCoreD64,
    episode: LivingReasoningEpisode,
    initial_soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
    ablate_temperatures: tuple[SoulTemperature, ...] = (),
    transcript_sink: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure free-running exact typed emissions on one complete episode."""

    compiled = D64FieldCompiler().compile(episode.snapshot)
    unroll = model.unroll_runtime_phases(
        initial_soul=initial_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        tick_uid=f"evaluation-episode:{episode.episode_id}",
        canonical=compiled,
        first_workspace_text=episode.first_workspace_text,
        refined_workspace_text=episode.refined_workspace_text,
        ablate_temperatures=ablate_temperatures,
    )
    region_texts = {state.name: state.text for state in episode.snapshot.regions}
    prompt_text = region_texts.get(LogicalRegion.USER_INPUT, "")
    supervised = 0
    typed_exact = 0
    payload_count = 0
    payload_exact = 0
    payload_token_count = 0
    payload_token_correct = 0
    payload_content_token_count = 0
    payload_content_token_correct = 0
    payload_eos_token_count = 0
    payload_eos_token_correct = 0
    alignment_position_count = 0
    alignment_position_correct = 0
    alignment_copy_gate_count = 0
    alignment_copy_gate_correct = 0
    alignment_eos_gate_count = 0
    alignment_eos_gate_correct = 0
    decision_correct = 0
    operation_count = 0
    operation_correct = 0
    region_count = 0
    region_correct = 0
    start_count = 0
    start_correct = 0
    end_count = 0
    end_correct = 0
    decision_target_counts = [0] * len(ReasoningDecision)
    decision_correct_by_target = [0] * len(ReasoningDecision)
    operation_target_counts = [0] * len(ReasoningOperationKind)
    operation_correct_by_target = [0] * len(ReasoningOperationKind)
    phase_diagnostics: list[dict[str, Any]] = []
    payload_target_counts = torch.zeros(
        model.eos_index + 1,
        dtype=torch.long,
        device=model.device,
    )
    for output, target in zip(unroll.outputs, episode.targets, strict=True):
        if target.supervision_weight <= 0:
            continue
        supervised += 1
        decision = tuple(ReasoningDecision)[
            int(output.decision_logits.argmax(dim=-1).item())
        ]
        decision_index = tuple(ReasoningDecision).index(target.decision)
        decision_is_exact = decision is target.decision
        decision_target_counts[decision_index] += 1
        decision_correct_by_target[decision_index] += int(decision_is_exact)
        decision_correct += int(decision_is_exact)
        exact = decision_is_exact
        diagnostic: dict[str, Any] = {
            "target_decision": target.decision.value,
            "decision_exact": bool(decision_is_exact),
        }
        if target.decision is ReasoningDecision.DELTA:
            payload_count += 1
            operation = tuple(ReasoningOperationKind)[
                int(output.operation_logits.argmax(dim=-1).item())
            ]
            region = tuple(LogicalRegion)[int(output.region_logits.argmax(dim=-1).item())]
            operation_index = tuple(ReasoningOperationKind).index(target.operation)
            operation_is_exact = operation is target.operation
            operation_count += 1
            operation_correct += int(operation_is_exact)
            operation_target_counts[operation_index] += 1
            operation_correct_by_target[operation_index] += int(operation_is_exact)
            region_is_exact = region is target.region
            region_count += 1
            region_correct += int(region_is_exact)

            target_candidates, target_start_logits, target_end_logits = model.boundary_logits(
                output, target.region
            )
            target_start_prediction = target_candidates[
                int(target_start_logits.argmax(dim=-1).item())
            ]
            target_end_prediction = target_candidates[
                int(target_end_logits.argmax(dim=-1).item())
            ]
            start_is_exact = target_start_prediction == target.start
            end_is_exact = target_end_prediction == target.end
            start_count += 1
            start_correct += int(start_is_exact)
            end_count += 1
            end_correct += int(end_is_exact)

            candidates, start_logits, end_logits = model.boundary_logits(output, region)
            start = candidates[int(start_logits.argmax(dim=-1).item())]
            end = candidates[int(end_logits.argmax(dim=-1).item())]
            if operation is ReasoningOperationKind.INSERT:
                end = start
            payload, terminated = model.decode_transport_greedy(output)
            payload_match = terminated and payload == target.payload
            payload_exact += int(payload_match)
            if transcript_sink is not None and payload_count <= 3:
                transcript_sink.append(
                    {
                        "episode_id": episode.episode_id,
                        "prompt": prompt_text,
                        "predicted_payload": payload,
                        "expected_payload": target.payload,
                        "exact_match": bool(payload_match),
                        "terminated": bool(terminated),
                        "operation": (operation.name if operation is not None else None),
                        "region": (region.name if region is not None else None),
                        "decision": (decision.name if decision is not None else None),
                    }
                )
            teacher_decoded = model.decode_teacher(
                output.reader_state,
                target.payload,
                head=1,
                memory=output.complete_memory,
                return_alignment=target.payload_alignment is not None,
            )
            if target.payload_alignment is None:
                teacher_logits, teacher_targets = teacher_decoded
                alignment = None
            else:
                teacher_logits, teacher_targets, decoder_alignment = teacher_decoded
                alignment = model.alignment_supervision(
                    target_text=target.payload,
                    memory=output.complete_memory,
                    decoder_alignment=decoder_alignment,
                    specification=target.payload_alignment,
                )
            teacher_predictions = teacher_logits.argmax(dim=-1)
            payload_token_count += int(teacher_targets.numel())
            payload_token_correct += int(
                teacher_predictions.eq(teacher_targets).sum().item()
            )
            content_targets = teacher_targets[:, :-1]
            content_predictions = teacher_predictions[:, :-1]
            content_count = int(content_targets.numel())
            content_correct = int(content_predictions.eq(content_targets).sum().item())
            payload_content_token_count += content_count
            payload_content_token_correct += content_correct
            eos_count = int(teacher_targets.shape[0])
            eos_correct = int(
                teacher_predictions[:, -1].eq(teacher_targets[:, -1]).sum().item()
            )
            payload_eos_token_count += eos_count
            payload_eos_token_correct += eos_correct
            payload_target_counts += torch.bincount(
                teacher_targets.reshape(-1),
                minlength=model.eos_index + 1,
            )
            if alignment is not None:
                alignment_position_count += int(alignment["copy_positions"])
                alignment_position_correct += int(alignment["position_correct"])
                alignment_copy_gate_count += int(alignment["copy_positions"])
                alignment_copy_gate_correct += int(alignment["copy_gate_correct"])
                alignment_eos_gate_count += int(
                    alignment["eos_gate_supervised_positions"]
                )
                alignment_eos_gate_correct += int(alignment["eos_gate_correct"])
            diagnostic.update(
                {
                    "target_operation": target.operation.value,
                    "operation_exact": bool(operation_is_exact),
                    "region_exact": bool(region_is_exact),
                    "start_exact": bool(start_is_exact),
                    "end_exact": bool(end_is_exact),
                    "payload_content_count": content_count,
                    "payload_content_correct": content_correct,
                    "payload_content_exact": (
                        None if content_count == 0 else content_correct == content_count
                    ),
                    "payload_eos_exact": bool(eos_correct == eos_count),
                    "alignment_position_exact": (
                        None
                        if alignment is None
                        else alignment["position_correct"] == alignment["copy_positions"]
                    ),
                    "alignment_copy_gate_exact": (
                        None
                        if alignment is None
                        else alignment["copy_gate_correct"] == alignment["copy_positions"]
                    ),
                    "alignment_eos_gate_exact": (
                        None
                        if alignment is None
                        else alignment["eos_gate_correct"]
                        == alignment["eos_gate_supervised_positions"]
                    ),
                }
            )
            exact = exact and all(
                (
                    operation is target.operation,
                    region is target.region,
                    start == target.start,
                    end == target.end,
                    payload_match,
                )
            )
        phase_diagnostics.append(diagnostic)
        typed_exact += int(exact)
    coverage = sum(item.canonical_coverage.complete for item in unroll.outputs) / len(
        unroll.outputs
    )
    constant_token_correct = int(payload_target_counts.max().item())
    return {
        "supervised_phase_count": float(supervised),
        "typed_emission_exact_count": float(typed_exact),
        "payload_supervised_phase_count": float(payload_count),
        "payload_transport_exact_count": float(payload_exact),
        "phase_output_count": float(len(unroll.outputs)),
        "complete_field_coverage_count": float(
            sum(item.canonical_coverage.complete for item in unroll.outputs)
        ),
        "typed_emission_exact_rate": typed_exact / max(1, supervised),
        "payload_transport_exact_rate": payload_exact / max(1, payload_count),
        "payload_teacher_forced_token_accuracy": payload_token_correct
        / max(1, payload_token_count),
        "payload_teacher_forced_token_count": payload_token_count,
        "payload_teacher_forced_token_correct": payload_token_correct,
        "payload_teacher_forced_target_counts": payload_target_counts.tolist(),
        "payload_teacher_forced_content_accuracy": payload_content_token_correct
        / max(1, payload_content_token_count),
        "payload_teacher_forced_content_count": payload_content_token_count,
        "payload_teacher_forced_content_correct": payload_content_token_correct,
        "payload_teacher_forced_eos_accuracy": payload_eos_token_correct
        / max(1, payload_eos_token_count),
        "payload_teacher_forced_eos_count": payload_eos_token_count,
        "payload_teacher_forced_eos_correct": payload_eos_token_correct,
        "alignment_position_accuracy": alignment_position_correct
        / max(1, alignment_position_count),
        "alignment_position_count": alignment_position_count,
        "alignment_position_correct": alignment_position_correct,
        "alignment_copy_gate_accuracy": alignment_copy_gate_correct
        / max(1, alignment_copy_gate_count),
        "alignment_copy_gate_count": alignment_copy_gate_count,
        "alignment_copy_gate_correct": alignment_copy_gate_correct,
        "alignment_eos_gate_accuracy": alignment_eos_gate_correct
        / max(1, alignment_eos_gate_count),
        "alignment_eos_gate_count": alignment_eos_gate_count,
        "alignment_eos_gate_correct": alignment_eos_gate_correct,
        "decision_accuracy": decision_correct / max(1, supervised),
        "decision_correct": decision_correct,
        "decision_target_counts": decision_target_counts,
        "decision_correct_by_target": decision_correct_by_target,
        "operation_accuracy": operation_correct / max(1, operation_count),
        "operation_count": operation_count,
        "operation_correct": operation_correct,
        "operation_target_counts": operation_target_counts,
        "operation_correct_by_target": operation_correct_by_target,
        "region_accuracy": region_correct / max(1, region_count),
        "region_count": region_count,
        "region_correct": region_correct,
        "start_accuracy": start_correct / max(1, start_count),
        "start_count": start_count,
        "start_correct": start_correct,
        "end_accuracy": end_correct / max(1, end_count),
        "end_count": end_count,
        "end_correct": end_correct,
        "phase_diagnostics": phase_diagnostics,
        "complete_field_coverage_rate": coverage,
        "constant_typed_emission_exact_floor": 0.0,
        "constant_payload_transport_exact_floor": 0.0,
        "constant_payload_token_accuracy_floor": constant_token_correct
        / max(1, payload_token_count),
        "constant_payload_content_accuracy_floor": (
            0.0
            if payload_content_token_count == 0
            else max(payload_target_counts[:-1]).item()
            / payload_content_token_count
        ),
    }


@torch.no_grad()
def living_source_counterfactuals(
    model: LivingReasoningCoreD64,
    episode: LivingReasoningEpisode,
    initial_soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
) -> dict[str, float]:
    """Prove field, proposals, and private Soul all affect consolidated state."""

    compiled = D64FieldCompiler().compile(episode.snapshot)
    unroll = model.unroll_runtime_phases(
        initial_soul=initial_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        tick_uid=f"counterfactual-episode:{episode.episode_id}",
        canonical=compiled,
        first_workspace_text=episode.first_workspace_text,
        refined_workspace_text=episode.refined_workspace_text,
    )
    consolidated_soul = unroll.souls[2]

    def forward(canonical, proposals, *, ablate=()):
        return model.forward_surfaces(
            soul=consolidated_soul,
            expected_core_id=core_id,
            parameter_generation=parameter_generation,
            phase="consolidated",
            canonical=canonical,
            proposal_texts=proposals,
            ablate_temperatures=ablate,
        )

    proposals = (episode.first_workspace_text, episode.refined_workspace_text)
    baseline = forward(compiled, proposals)
    changed_texts = {state.name: state.text for state in episode.snapshot.regions}
    changed_texts[LogicalRegion.USER_INPUT] += "\n[counterfactual-field-change]"
    changed_snapshot = SharedFieldSnapshot.from_texts(
        changed_texts,
        tick_id=episode.snapshot.tick_id,
        source_manifest_ids=episode.snapshot.source_manifest_ids,
    )
    changed_field = forward(D64FieldCompiler().compile(changed_snapshot), proposals)
    changed_proposals = forward(
        compiled,
        (*proposals[:-1], proposals[-1] + "\n[counterfactual-proposal-change]"),
    )
    ablated_soul = forward(compiled, proposals, ablate=tuple(SoulTemperature))

    def signature(output: LivingReasoningForward) -> torch.Tensor:
        return torch.cat(
            (
                output.reader_state.mean(dim=1),
                output.decision_logits,
                output.operation_logits,
                output.region_logits,
            ),
            dim=-1,
        )

    reference = signature(baseline)
    return {
        "field_counterfactual_l2": float((signature(changed_field) - reference).norm().item()),
        "proposal_counterfactual_l2": float(
            (signature(changed_proposals) - reference).norm().item()
        ),
        "soul_counterfactual_l2": float((signature(ablated_soul) - reference).norm().item()),
    }


@torch.no_grad()
def living_phase_breakdown(
    model: LivingReasoningCoreD64,
    episode: LivingReasoningEpisode,
    initial_soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
) -> dict[str, dict[str, float]]:
    """Per-phase free-running exactness for one episode.

    Returns one row per phase target (``first``, ``refined``,
    ``consolidated``).  A supervised weight of zero still produces a row —
    the emission is measured, it simply never becomes a training target.
    Callers use the FIRST-vs-REFINED rows to measure proposal-board use and
    the CONSOLIDATED row for final-authority exactness.
    """

    compiled = D64FieldCompiler().compile(episode.snapshot)
    unroll = model.unroll_runtime_phases(
        initial_soul=initial_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        tick_uid=f"phase-breakdown:{episode.episode_id}",
        canonical=compiled,
        first_workspace_text=episode.first_workspace_text,
        refined_workspace_text=episode.refined_workspace_text,
    )
    rows: dict[str, dict[str, float]] = {}
    for output, target in zip(unroll.outputs, episode.targets, strict=True):
        decision = tuple(ReasoningDecision)[
            int(output.decision_logits.argmax(dim=-1).item())
        ]
        decision_match = float(decision is target.decision)
        payload_match = 0.0
        if target.decision is ReasoningDecision.DELTA:
            operation = tuple(ReasoningOperationKind)[
                int(output.operation_logits.argmax(dim=-1).item())
            ]
            region = tuple(LogicalRegion)[int(output.region_logits.argmax(dim=-1).item())]
            candidates, start_logits, end_logits = model.boundary_logits(output, region)
            start = candidates[int(start_logits.argmax(dim=-1).item())]
            end = candidates[int(end_logits.argmax(dim=-1).item())]
            if operation is ReasoningOperationKind.INSERT:
                end = start
            payload, terminated = model.decode_transport_greedy(output)
            payload_match = float(terminated and payload == target.payload)
            decision_match = float(
                decision_match
                and operation is target.operation
                and region is target.region
                and start == target.start
                and end == target.end
                and payload_match
            )
        rows[target.phase] = {
            "decision_match": decision_match,
            "payload_match": payload_match,
            "supervision_weight": float(target.supervision_weight),
        }
    return rows


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
    "evaluate_living_episode",
    "living_episode_objective",
    "living_phase_breakdown",
    "living_phase_objective",
    "living_source_counterfactuals",
]
