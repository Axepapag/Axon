"""English-native curriculum, objective, evaluation, and gate for D64 reasoning.

This is the active post-2026-09-19 training contract.  A supervised phase teaches
one exact variable-length Unicode utterance.  FIRST and REFINED targets are free
English reasoning proposals; CONSOLIDATED targets are compact tagged-region
FINAL verdicts.  There is no decision, operation, region, or address classifier
in the objective.

The previous typed DELTA/NO_OP/ABSTAIN curriculum is preserved, by name, in
``legacy_typed_reasoning_curriculum`` for immutable historical evidence only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from runtime.field import D64FieldCompiler, LogicalRegion, SharedFieldSnapshot, canonical_json_bytes, canonical_sha256
from runtime.heart import (
    CoreDescriptor,
    D64ProposalWorkspaceRenderer,
    EnglishProposal,
    FrozenTickImage,
    ParticipantRecord,
    ParticipantState,
    ProposalBoard,
    ProposalPass,
    ProposalWorkspace,
    RailRuntimeView,
    ReasoningPassRequest,
    ReasoningPassResult,
    TechnicalFinalVerdict,
    TickIdentity,
)
from runtime.soul import SoulSnapshot, SoulTemperature, apply_soul_transition
from substrate import encode_unicode_text

from .complete_field_64d import sequence_cross_entropy
from .living_reasoning_d64 import CausalLivingUnroll, LivingReasoningCoreD64, LivingReasoningForward

LIVING_REASONING_TARGET_SCHEMA = "axon-living-reasoning-english-target-v1"
LIVING_REASONING_EPISODE_SCHEMA = "axon-living-reasoning-english-episode-v1"
LIVING_REASONING_CURRICULUM_SCHEMA = "axon-living-reasoning-english-curriculum-v1"
LIVING_REASONING_GATE_SCHEMA = "axon-living-reasoning-english-gate-v1"

LIVING_OBJECTIVE_COMPONENTS = (
    "text",
    "alignment_position",
    "alignment_copy_gate",
    "alignment_eos_gate",
)

ENGLISH_REASONING_GATE_REQUIREMENTS: Mapping[str, float] = {
    "text_exact_rate": 1.0,
    "text_teacher_forced_content_accuracy": 1.0,
    "text_teacher_forced_eos_accuracy": 1.0,
    "complete_field_coverage_rate": 1.0,
    "final_verdict_valid_rate": 1.0,
}


def _require_unicode(value: str, *, name: str, nonempty: bool) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text")
    if nonempty and not value.strip():
        raise ValueError(f"{name} must be nonempty")
    try:
        roundtrip = value.encode("utf-8", errors="strict").decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError(f"{name} must be valid exact Unicode") from exc
    if roundtrip != value:
        raise ValueError(f"{name} failed exact Unicode roundtrip")
    return value


def _validate_final_syntax(text: str) -> None:
    # TechnicalFinalVerdict validates exact Unicode and tag grammar without
    # materializing a mutation; these bindings are deliberately dummy envelope
    # data because the learned target is text only.
    TechnicalFinalVerdict(
        base_field_id="training-syntax-only",
        base_tick_id=0,
        author_core_id="training-syntax-only",
        rail_d_model=64,
        text=text,
    )


@dataclass(frozen=True, slots=True)
class LivingReasoningTarget:
    """One exact public text target for one reasoning phase."""

    phase: str
    text: str
    text_alignment: Mapping[str, Any] | None = None
    supervision_weight: float = 1.0
    target_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.phase not in {"first", "refined", "consolidated"}:
            raise ValueError("unsupported living reasoning target phase")
        _require_unicode(self.text, name=f"{self.phase} target text", nonempty=True)
        if self.phase == "consolidated":
            _validate_final_syntax(self.text)
        weight = float(self.supervision_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("supervision_weight must be finite and non-negative")
        object.__setattr__(self, "supervision_weight", weight)
        if self.text_alignment is not None:
            alignment = dict(self.text_alignment)
            if set(alignment) != {"schema", "segments", "supervise_eos_generate"}:
                raise ValueError("text alignment fields are invalid")
            if alignment["schema"] != "axon-r0-target-alignment-v1":
                raise ValueError("unsupported text alignment schema")
            if not isinstance(alignment["segments"], list) or not alignment["segments"]:
                raise ValueError("text alignment requires one or more exact source segments")
            if alignment["supervise_eos_generate"] is not True:
                raise ValueError("text alignment must supervise EOS as generated")
            object.__setattr__(self, "text_alignment", alignment)
        object.__setattr__(self, "target_id", canonical_sha256(self.to_canonical_dict(False)))

    @property
    def payload(self) -> str:
        """Compatibility spelling for display-only consumers; the contract is text."""
        return self.text

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": LIVING_REASONING_TARGET_SCHEMA,
            "phase": self.phase,
            "text": self.text,
            "supervision_weight": self.supervision_weight,
        }
        if self.text_alignment is not None:
            value["text_alignment"] = dict(self.text_alignment)
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
    target_basis: str = "english_reasoning_v1"
    episode_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("living reasoning episode label must be non-empty")
        if self.split not in {"train", "heldout", "regression"}:
            raise ValueError("living reasoning split must be train, heldout, or regression")
        if not isinstance(self.snapshot, SharedFieldSnapshot):
            raise TypeError("living reasoning snapshot must be SharedFieldSnapshot")
        _require_unicode(self.first_workspace_text, name="first workspace text", nonempty=False)
        _require_unicode(self.refined_workspace_text, name="refined workspace text", nonempty=False)
        targets = tuple(self.targets)
        if tuple(item.phase for item in targets) != ("first", "refined", "consolidated"):
            raise ValueError("living episode requires exact three-phase target order")
        if not any(item.supervision_weight > 0.0 for item in targets):
            raise ValueError("living episode requires at least one supervised English phase")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "mechanism_tags", tuple(sorted(set(map(str, self.mechanism_tags)))))
        evidence = tuple(sorted(set(map(str, self.outcome_evidence_ids))))
        object.__setattr__(self, "outcome_evidence_ids", evidence)
        if self.source_example_id is not None and len(self.source_example_id) != 64:
            raise ValueError("source_example_id must be None or a content identity")
        if not isinstance(self.target_basis, str) or not self.target_basis:
            raise ValueError("target_basis must be non-empty")
        object.__setattr__(self, "episode_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": LIVING_REASONING_EPISODE_SCHEMA,
            "label": self.label,
            "split": self.split,
            "snapshot": self.snapshot.to_dict(),
            "first_workspace_text": self.first_workspace_text,
            "refined_workspace_text": self.refined_workspace_text,
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
        if not episodes or not {"train", "heldout"}.issubset({item.split for item in episodes}):
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
                "schema": "axon-living-reasoning-english-split-manifest-v1",
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
        value: dict[str, Any] = {
            "schema": LIVING_REASONING_CURRICULUM_SCHEMA,
            "public_output_contract": "english-proposal-tagged-final-v1",
            "episodes": [item.to_canonical_dict() for item in self.episodes],
        }
        if include_id:
            value["curriculum_id"] = self.curriculum_id
        return value


def _smoke_episode(
    *,
    label: str,
    split: str,
    user_text: str,
    answer: str,
    first: str,
    refined: str,
) -> LivingReasoningEpisode:
    snapshot = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.IDENTITY: "I am Axon.",
            LogicalRegion.USER_INPUT: user_text,
            LogicalRegion.RESPONSE_DRAFT: "",
            LogicalRegion.SCRATCH: "",
        },
        source_manifest_ids=("english-reasoning-smoke-v1",),
    )
    return LivingReasoningEpisode(
        label=label,
        split=split,
        snapshot=snapshot,
        first_workspace_text=f"[brother-first]\n{first}",
        refined_workspace_text=f"[brother-refined]\n{refined}",
        targets=(
            LivingReasoningTarget(phase="first", text=first),
            LivingReasoningTarget(phase="refined", text=refined),
            LivingReasoningTarget(phase="consolidated", text=f"#responseDraft# {answer}"),
        ),
        mechanism_tags=("english", "proposal", "refinement", "tagged_final", "unicode"),
    )


def build_living_reasoning_smoke_curriculum() -> LivingReasoningCurriculum:
    """Small mechanism proof for English proposal/refinement/FINAL generation."""

    return LivingReasoningCurriculum(
        episodes=(
            _smoke_episode(
                label="english-unicode-train",
                split="train",
                user_text="Return the exact marker λ🧠.",
                answer="λ🧠",
                first="The user asks for the exact marker λ🧠; I should preserve it exactly.",
                refined="I agree that λ🧠 must be preserved exactly and the answer should stay concise.",
            ),
            _smoke_episode(
                label="english-evidence-train",
                split="train",
                user_text="Canonical evidence says gate=GREEN. State the value.",
                answer="GREEN",
                first="The canonical evidence says gate=GREEN, so the response should preserve GREEN.",
                refined="The other proposal is consistent with the field: answer GREEN without adding unsupported detail.",
            ),
            _smoke_episode(
                label="english-unicode-heldout",
                split="heldout",
                user_text="Return the exact tail marker 終端✅.",
                answer="終端✅",
                first="The requested tail marker is 終端✅ and exact Unicode preservation matters.",
                refined="I support preserving 終端✅ exactly; no additional claim is needed.",
            ),
        )
    )


def _component_weights(component_weights: Mapping[str, float] | None) -> dict[str, float] | None:
    if component_weights is None:
        return None
    unknown = set(component_weights) - set(LIVING_OBJECTIVE_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown living objective components: {sorted(unknown)}")
    weights: dict[str, float] = {}
    for component in LIVING_OBJECTIVE_COMPONENTS:
        value = float(component_weights.get(component, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("living objective component weights must be finite and non-negative")
        weights[component] = value
    return weights


def living_phase_objective(
    model: LivingReasoningCoreD64,
    output: LivingReasoningForward,
    target: LivingReasoningTarget,
    *,
    component_weights: Mapping[str, float] | None = None,
    alignment_position_reduction: str = "mean",
    text_eos_weight: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float | None]]:
    """Supervise exactly one variable-length English/tagged-text emission."""

    text_eos_weight = float(text_eos_weight)
    if not math.isfinite(text_eos_weight) or text_eos_weight <= 0.0:
        raise ValueError("text_eos_weight must be finite and positive")
    if output.phase != target.phase:
        raise ValueError("reasoning output phase differs from English target phase")
    weights = _component_weights(component_weights)

    decoded = model.decode_teacher(
        output.reader_state,
        target.text,
        head=1,
        memory=output.complete_memory,
        return_alignment=target.text_alignment is not None,
    )
    if target.text_alignment is None:
        text_logits, text_targets = decoded
        alignment = None
    else:
        text_logits, text_targets, decoder_alignment = decoded
        alignment = model.alignment_supervision(
            target_text=target.text,
            memory=output.complete_memory,
            decoder_alignment=decoder_alignment,
            specification=target.text_alignment,
            position_reduction=alignment_position_reduction,
        )
    learned_mask = None if alignment is None else alignment["learned_decision_mask"]
    text_loss = sequence_cross_entropy(
        text_logits,
        text_targets,
        eos_weight=text_eos_weight,
        token_mask=learned_mask,
    )
    loss = text_loss if weights is None else text_loss * weights["text"]
    metrics: dict[str, float | None] = {
        "text_loss": float(text_loss.detach().item()),
        "target_transport_units": float(len(encode_unicode_text(target.text))),
    }
    if alignment is not None:
        position_loss = alignment["position_loss"]
        copy_gate_loss = alignment["copy_gate_loss"]
        eos_gate_loss = alignment["eos_gate_loss"]
        if weights is None:
            loss = loss + position_loss + alignment["gate_loss"]
        else:
            loss = (
                loss
                + position_loss * weights["alignment_position"]
                + copy_gate_loss * weights["alignment_copy_gate"]
                + eos_gate_loss * weights["alignment_eos_gate"]
            )
        metrics.update(
            {
                "alignment_position_loss": float(position_loss.detach().item()),
                "alignment_copy_gate_loss": float(copy_gate_loss.detach().item()),
                "alignment_eos_gate_loss": float(eos_gate_loss.detach().item()),
                "alignment_position_accuracy": float(alignment["position_accuracy"]),
                "alignment_copy_gate_accuracy": float(alignment["copy_gate_accuracy"]),
                "alignment_eos_gate_accuracy": float(alignment["eos_gate_accuracy"]),
                "deterministic_continuation_positions": float(
                    alignment["deterministic_continuation_positions"]
                ),
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
    alignment_position_reduction: str = "mean",
    text_eos_weight: float = 4.0,
) -> tuple[torch.Tensor, CausalLivingUnroll, tuple[dict[str, float | None], ...]]:
    compiled = D64FieldCompiler().compile(episode.snapshot)
    compiled.verify_roundtrip(episode.snapshot)
    unroll = model.unroll_runtime_phases(
        initial_soul=initial_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        tick_uid=f"english-training-episode:{episode.episode_id}",
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
            alignment_position_reduction=alignment_position_reduction,
            text_eos_weight=text_eos_weight,
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


def constant_baseline_target_key(target: LivingReasoningTarget) -> str:
    """Exact fixed answer a zero-skill text emitter would repeat."""
    return target.text


def constant_baseline_floors(
    text_target_histogram: Mapping[str, int],
    *,
    supervised_phase_count: float,
) -> dict[str, float]:
    strongest = max(text_target_histogram.values(), default=0)
    return {
        "constant_text_exact_count": float(strongest),
        "constant_text_exact_floor": strongest / max(1.0, supervised_phase_count),
    }


def _decode_one_slice(model: LivingReasoningCoreD64, output: LivingReasoningForward) -> tuple[str, bool]:
    return model.decode_transport_greedy(output, work_units=256)


@dataclass(frozen=True, slots=True)
class LivingRuntimePhase:
    """One isolated pass using the active Heart request and emission contracts.

    ``diagnostic_forward`` is a second, no-grad replay of the exact request for
    teacher-forced diagnostics and coverage accounting.  The public output and
    Soul successor always come from ``model.emit`` first, so the pass's control
    behavior is the production decoder behavior rather than a training helper.
    """

    phase: str
    request: ReasoningPassRequest
    before_soul: SoulSnapshot
    after_soul: SoulSnapshot
    state: ParticipantState
    detail: str
    output: EnglishProposal | TechnicalFinalVerdict | None
    diagnostic_forward: LivingReasoningForward | None

    @property
    def emitted_text(self) -> str:
        return "" if self.output is None else self.output.text

    @property
    def terminated(self) -> bool:
        # Active emit rejects unterminated and blank strings before returning.
        return self.output is not None


@dataclass(frozen=True, slots=True)
class LivingRuntimeTick:
    """An isolated, nonpersistent execution of the real three-pass runtime."""

    descriptor: CoreDescriptor
    image: FrozenTickImage
    initial_soul: SoulSnapshot
    final_soul: SoulSnapshot
    phases: tuple[LivingRuntimePhase, ...]
    first_records: tuple[ParticipantRecord, ...]
    refined_records: tuple[ParticipantRecord, ...]
    first_workspace: ProposalWorkspace
    refined_workspace: ProposalWorkspace

    def phase(self, name: str) -> LivingRuntimePhase:
        for item in self.phases:
            if item.phase == name:
                return item
        raise KeyError(name)


def _assert_runtime_output_binding(
    output: EnglishProposal | TechnicalFinalVerdict,
    descriptor: CoreDescriptor,
    image: FrozenTickImage,
    phase: str,
) -> None:
    """Mirror the Heart's output-envelope checks without opening a transaction."""

    identity = image.identity
    if output.author_core_id != descriptor.core_id:
        raise ValueError("reasoning output author differs from the invoked core")
    if output.rail_d_model != descriptor.d_model:
        raise ValueError("reasoning output names the wrong home rail")
    if output.base_field_id != identity.base_field_id or output.base_tick_id != identity.base_tick_id:
        raise ValueError("reasoning output is stale for the frozen tick")
    if phase in {ProposalPass.FIRST.value, ProposalPass.REFINED.value}:
        if not isinstance(output, EnglishProposal) or output.pass_id != phase:
            raise ValueError("FIRST/REFINED output is not bound to its English proposal pass")
    elif not isinstance(output, TechnicalFinalVerdict):
        raise ValueError("CONSOLIDATED output is not a tagged FINAL verdict")


def _assert_runtime_transition_binding(
    request: ReasoningPassRequest,
    result: ReasoningPassResult,
) -> None:
    """Mirror the Heart's private-Soul successor checks before isolated apply."""

    transition = result.soul_transition
    if (
        transition.core_id != request.descriptor.core_id
        or transition.architecture_id != request.descriptor.architecture_id
        or transition.parameter_generation != request.descriptor.parameter_generation
    ):
        raise ValueError("Soul transition belongs to another core generation")
    if transition.before_soul_id != request.soul.soul_id:
        raise ValueError("Soul transition is stale for the inhale snapshot")
    if transition.before_generation != request.soul.generation:
        raise ValueError("Soul transition generation is stale")
    if transition.tick_uid != request.image.identity.tick_uid:
        raise ValueError("Soul transition names the wrong tick")
    if transition.request_id != request.request_id or transition.phase != request.phase:
        raise ValueError("Soul transition names the wrong request or phase")


@torch.no_grad()
def run_living_runtime_tick(
    model: LivingReasoningCoreD64,
    snapshot: SharedFieldSnapshot,
    initial_soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
    tick_sequence: int = 1,
    heartbeat_id: int = 1,
) -> LivingRuntimeTick:
    """Execute FIRST → REFINED → CONSOLIDATED through the production seam.

    This intentionally does not open a Heart transaction or persist a Soul.
    It is an evaluator seam: the field, rail binding, request envelopes,
    production proposal renderer, ``emit`` decoder, failure accounting, and
    transition validation are all the active runtime implementations.

    A failed FIRST does *not* suppress REFINED.  The active Heart instead
    renders the failed accounting row into the proposal workspace, lets the
    participant see that exact board, and preserves the prior Soul.  The same
    rule applies to REFINED before consolidation.
    """

    if not isinstance(model, LivingReasoningCoreD64):
        raise TypeError("runtime tick requires a LivingReasoningCoreD64")
    if not isinstance(snapshot, SharedFieldSnapshot):
        raise TypeError("runtime tick requires a SharedFieldSnapshot")
    if not isinstance(initial_soul, SoulSnapshot):
        raise TypeError("runtime tick requires a SoulSnapshot")
    compiled = D64FieldCompiler().compile(snapshot)
    compiled.verify_roundtrip(snapshot)
    image = FrozenTickImage.from_compiled(
        TickIdentity(
            tick_sequence=tick_sequence,
            heartbeat_id=heartbeat_id,
            base_field_id=snapshot.field_id,
            base_tick_id=snapshot.tick_id,
        ),
        {64: compiled},
    )
    descriptor = CoreDescriptor(
        core_id=core_id,
        d_model=64,
        architecture_id=model.architecture_id,
        parameter_generation=parameter_generation,
    )
    rail = RailRuntimeView(
        binding=image.require_rail(64),
        exact_surface=compiled,
        semantic_surface=None,
    )
    board = ProposalBoard(image, (descriptor,))
    renderer = D64ProposalWorkspaceRenderer()
    soul = initial_soul

    def run_phase(
        phase: str,
        proposal_rails: tuple[Any, ...],
    ) -> LivingRuntimePhase:
        nonlocal soul
        request = ReasoningPassRequest(
            descriptor=descriptor,
            image=image,
            phase=phase,
            rail=rail,
            soul=soul,
            proposal_rails=proposal_rails,
        )
        before = soul
        after = before
        state = ParticipantState.FAILED
        detail = ""
        output: EnglishProposal | TechnicalFinalVerdict | None = None
        try:
            result = model.emit(request)
            if not isinstance(result, ReasoningPassResult):
                raise TypeError("core returned a value other than ReasoningPassResult")
            candidate = result.output
            _assert_runtime_output_binding(candidate, descriptor, image, phase)
            _assert_runtime_transition_binding(request, result)
            # The active Heart validates FINAL materialization before it may
            # commit the final private transition.  Do the same in this
            # isolated evaluator, without mutating canonical state.
            if phase == "consolidated":
                if not isinstance(candidate, TechnicalFinalVerdict):
                    raise ValueError("CONSOLIDATED output is not a tagged FINAL verdict")
                candidate.materialize(snapshot)
            after = apply_soul_transition(before, result.soul_transition)
            soul = after
            output = candidate
            state = ParticipantState.RETURNED
        except TimeoutError as exc:
            detail = f"{type(exc).__name__}: {exc}"
            state = ParticipantState.TIMED_OUT
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            state = ParticipantState.FAILED

        # ``emit`` internally calls forward_request. Replay that exact request
        # only for diagnostics; it has no authority and cannot affect the
        # output, workspace, or Soul lineage recorded above.
        diagnostic_forward: LivingReasoningForward | None = None
        try:
            diagnostic_forward = model.forward_request(request)
        except Exception as exc:
            forward_detail = f"diagnostic {type(exc).__name__}: {exc}"
            detail = forward_detail if not detail else f"{detail}; {forward_detail}"
        return LivingRuntimePhase(
            phase=phase,
            request=request,
            before_soul=before,
            after_soul=after,
            state=state,
            detail=detail,
            output=output,
            diagnostic_forward=diagnostic_forward,
        )

    first = run_phase("first", ())
    if first.state is ParticipantState.RETURNED and isinstance(first.output, EnglishProposal):
        board.submit(first.output)
    elif first.state is ParticipantState.TIMED_OUT:
        board.mark_timed_out(core_id, ProposalPass.FIRST, first.detail)
    else:
        board.mark_failed(core_id, ProposalPass.FIRST, first.detail)
    board.close_first_pass()
    first_records = board.participant_states(ProposalPass.FIRST)
    first_workspace = renderer.render(image, ProposalPass.FIRST, first_records)

    refined = run_phase("refined", (first_workspace.require_rail(64),))
    if refined.state is ParticipantState.RETURNED and isinstance(refined.output, EnglishProposal):
        board.submit(refined.output)
    elif refined.state is ParticipantState.TIMED_OUT:
        board.mark_timed_out(core_id, ProposalPass.REFINED, refined.detail)
    else:
        board.mark_failed(core_id, ProposalPass.REFINED, refined.detail)
    board.close_refinement()
    board.assert_ready_for_consolidation()
    refined_records = board.participant_states(ProposalPass.REFINED)
    refined_workspace = renderer.render(image, ProposalPass.REFINED, refined_records)

    consolidated = run_phase(
        "consolidated",
        (first_workspace.require_rail(64), refined_workspace.require_rail(64)),
    )
    return LivingRuntimeTick(
        descriptor=descriptor,
        image=image,
        initial_soul=initial_soul,
        final_soul=soul,
        phases=(first, refined, consolidated),
        first_records=first_records,
        refined_records=refined_records,
        first_workspace=first_workspace,
        refined_workspace=refined_workspace,
    )


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
    transcript_sink_cap: int | None = 3,
) -> dict[str, Any]:
    """Measure held-out free-running runtime behavior and diagnostics.

    The evaluation path never feeds ``first_workspace_text`` or
    ``refined_workspace_text`` into the Core. Those authored strings are
    training-only credit-assignment context; runtime refinement must attend the
    actual accepted FIRST proposal.
    """

    if ablate_temperatures:
        raise ValueError(
            "free-running runtime evaluation does not support Soul ablation; "
            "use a separate counterfactual diagnostic rather than changing the live emit path"
        )
    runtime_tick = run_living_runtime_tick(
        model,
        episode.snapshot,
        initial_soul,
        core_id=core_id,
        parameter_generation=parameter_generation,
    )
    supervised = 0
    exact = 0
    terminated_count = 0
    token_count = 0
    token_correct = 0
    content_count = 0
    content_correct = 0
    eos_count = 0
    eos_correct = 0
    final_count = 0
    final_valid = 0
    target_counts = torch.zeros(model.eos_index + 1, dtype=torch.long, device=model.device)
    target_histogram: dict[str, int] = {}
    phase_diagnostics: list[dict[str, Any]] = []

    for runtime_phase, target in zip(runtime_tick.phases, episode.targets, strict=True):
        output = runtime_phase.diagnostic_forward
        predicted = runtime_phase.emitted_text
        terminated = runtime_phase.terminated
        final_is_valid: bool | None = None
        if target.phase == "consolidated":
            final_count += 1
            final_is_valid = bool(
                runtime_phase.state is ParticipantState.RETURNED
                and isinstance(runtime_phase.output, TechnicalFinalVerdict)
            )
            final_valid += int(final_is_valid)

        if target.supervision_weight <= 0.0:
            continue
        supervised += 1
        target_histogram[target.text] = target_histogram.get(target.text, 0) + 1
        is_exact = bool(terminated and predicted == target.text)
        exact += int(is_exact)
        terminated_count += int(terminated)

        if output is not None:
            teacher = model.decode_teacher(
                output.reader_state,
                target.text,
                head=1,
                memory=output.complete_memory,
                return_alignment=target.text_alignment is not None,
            )
            if target.text_alignment is None:
                logits, targets = teacher
                learned_mask = torch.ones_like(targets, dtype=torch.bool)
            else:
                logits, targets, decoder_alignment = teacher
                alignment = model.alignment_supervision(
                    target_text=target.text,
                    memory=output.complete_memory,
                    decoder_alignment=decoder_alignment,
                    specification=target.text_alignment,
                )
                learned_mask = alignment["learned_decision_mask"]
            predictions = logits.argmax(dim=-1)
            learned_targets = targets[learned_mask]
            learned_predictions = predictions[learned_mask]
            token_count += int(learned_targets.numel())
            token_correct += int(learned_predictions.eq(learned_targets).sum().item())
            target_counts += torch.bincount(learned_targets.reshape(-1), minlength=model.eos_index + 1)

            content_mask = learned_mask[:, :-1]
            content_targets = targets[:, :-1][content_mask]
            content_predictions = predictions[:, :-1][content_mask]
            content_count += int(content_targets.numel())
            content_correct += int(content_predictions.eq(content_targets).sum().item())
            eos_count += int(targets.shape[0])
            eos_correct += int(predictions[:, -1].eq(targets[:, -1]).sum().item())

        diagnostic = {
            "phase": target.phase,
            "attempted": True,
            "proposal_context": [item.text for item in runtime_phase.request.proposal_rails],
            "expected_text": target.text,
            "predicted_text": predicted,
            "terminated": terminated,
            "proposal_accepted": bool(
                runtime_phase.state is ParticipantState.RETURNED
                and isinstance(runtime_phase.output, EnglishProposal)
            ),
            "text_exact": is_exact,
            "final_verdict_valid": final_is_valid,
            "participant_state": runtime_phase.state.value,
            "before_soul_id": runtime_phase.before_soul.soul_id,
            "after_soul_id": runtime_phase.after_soul.soul_id,
            "before_soul_generation": runtime_phase.before_soul.generation,
            "after_soul_generation": runtime_phase.after_soul.generation,
            "runtime_detail": runtime_phase.detail or None,
        }
        phase_diagnostics.append(diagnostic)
        if transcript_sink is not None and (
            transcript_sink_cap is None or len(transcript_sink) < transcript_sink_cap
        ):
            transcript_sink.append({"episode_id": episode.episode_id, **diagnostic})

    coverage_count = sum(
        item.diagnostic_forward.canonical_coverage.complete
        for item in runtime_tick.phases
        if item.diagnostic_forward is not None
    )
    floors = constant_baseline_floors(
        target_histogram,
        supervised_phase_count=float(supervised),
    )
    constant_token_correct = int(target_counts.max().item()) if token_count else 0
    return {
        "supervised_phase_count": float(supervised),
        "text_exact_count": float(exact),
        "text_exact_rate": exact / max(1, supervised),
        "text_terminated_count": float(terminated_count),
        "text_termination_rate": terminated_count / max(1, supervised),
        "text_teacher_forced_token_accuracy": token_correct / max(1, token_count),
        "text_teacher_forced_token_count": token_count,
        "text_teacher_forced_token_correct": token_correct,
        "text_teacher_forced_target_counts": target_counts.tolist(),
        "text_teacher_forced_content_accuracy": content_correct / max(1, content_count),
        "text_teacher_forced_content_count": content_count,
        "text_teacher_forced_content_correct": content_correct,
        "text_teacher_forced_eos_accuracy": eos_correct / max(1, eos_count),
        "text_teacher_forced_eos_count": eos_count,
        "text_teacher_forced_eos_correct": eos_correct,
        "constant_text_token_accuracy_floor": constant_token_correct / max(1, token_count),
        "constant_text_target_histogram": dict(sorted(target_histogram.items())),
        "final_verdict_count": float(final_count),
        "final_verdict_valid_count": float(final_valid),
        "final_verdict_valid_rate": final_valid / max(1, final_count),
        "phase_output_count": float(sum(item.output is not None for item in runtime_tick.phases)),
        "phase_returned_count": float(
            sum(item.state is ParticipantState.RETURNED for item in runtime_tick.phases)
        ),
        "phase_expected_count": float(len(episode.targets)),
        "complete_field_coverage_count": float(coverage_count),
        "complete_field_coverage_rate": coverage_count / max(1, len(episode.targets)),
        "runtime_image_id": runtime_tick.image.image_id,
        "initial_soul_id": runtime_tick.initial_soul.soul_id,
        "final_soul_id": runtime_tick.final_soul.soul_id,
        "first_workspace_id": runtime_tick.first_workspace.workspace_id,
        "first_workspace_text_sha256": canonical_sha256(
            {"text": runtime_tick.first_workspace.readable_text()}
        ),
        "refined_workspace_id": runtime_tick.refined_workspace.workspace_id,
        "refined_workspace_text_sha256": canonical_sha256(
            {"text": runtime_tick.refined_workspace.readable_text()}
        ),
        "refinement_context": "production_first_workspace",
        "authored_workspace_text_used": False,
        "phase_diagnostics": phase_diagnostics,
        **floors,
    }


def decide_living_reasoning_mastery(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed heldout gate for the English-native learned surface."""

    failures: list[dict[str, Any]] = []
    for metric, threshold in ENGLISH_REASONING_GATE_REQUIREMENTS.items():
        raw_observed = metrics.get(metric)
        observed = (
            None
            if isinstance(raw_observed, bool) or not isinstance(raw_observed, (int, float))
            else float(raw_observed)
        )
        if observed is not None and (not math.isfinite(observed) or not 0.0 <= observed <= 1.0):
            observed = None
        if observed is None or observed < threshold:
            failures.append(
                {
                    "metric": metric,
                    "observed": observed,
                    "required_minimum": threshold,
                }
            )
    raw_exact = metrics.get("text_exact_rate")
    raw_floor = metrics.get("constant_text_exact_floor")
    exact = (
        None
        if isinstance(raw_exact, bool) or not isinstance(raw_exact, (int, float))
        else float(raw_exact)
    )
    if exact is not None and (not math.isfinite(exact) or not 0.0 <= exact <= 1.0):
        exact = None
    floor = (
        None
        if isinstance(raw_floor, bool) or not isinstance(raw_floor, (int, float))
        else float(raw_floor)
    )
    if floor is not None and (not math.isfinite(floor) or not 0.0 <= floor <= 1.0):
        floor = None
    if (
        exact is None
        or floor is None
        or not math.isfinite(exact)
        or not math.isfinite(floor)
        or not 0.0 <= exact <= 1.0
        or not 0.0 <= floor <= 1.0
        or exact <= floor
    ):
        failures.append(
            {
                "metric": "text_exact_rate_vs_constant",
                "observed": exact,
                "required_strictly_greater_than": floor,
            }
        )
    value = {
        "schema": LIVING_REASONING_GATE_SCHEMA,
        "requirements": dict(ENGLISH_REASONING_GATE_REQUIREMENTS),
        "failures": failures,
        "passed": not failures,
    }
    value["gate_id"] = canonical_sha256(value)
    return value


@torch.no_grad()
def living_source_counterfactuals(
    model: LivingReasoningCoreD64,
    episode: LivingReasoningEpisode,
    initial_soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
) -> dict[str, float]:
    """Prove field, proposal-board, and Soul inputs all affect decoder state."""

    compiled = D64FieldCompiler().compile(episode.snapshot)
    unroll = model.unroll_runtime_phases(
        initial_soul=initial_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        tick_uid=f"english-counterfactual:{episode.episode_id}",
        canonical=compiled,
        first_workspace_text=episode.first_workspace_text,
        refined_workspace_text=episode.refined_workspace_text,
    )
    consolidated_soul = unroll.souls[2]

    def forward(canonical: Any, proposals: Sequence[str], *, ablate: tuple[SoulTemperature, ...] = ()) -> LivingReasoningForward:
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
                model._initial_decoder_hidden(output.reader_state, 1).reshape(1, -1),
            ),
            dim=-1,
        )

    reference = signature(baseline)
    return {
        "field_counterfactual_l2": float((signature(changed_field) - reference).norm().item()),
        "proposal_counterfactual_l2": float((signature(changed_proposals) - reference).norm().item()),
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
    compiled = D64FieldCompiler().compile(episode.snapshot)
    unroll = model.unroll_runtime_phases(
        initial_soul=initial_soul,
        expected_core_id=core_id,
        parameter_generation=parameter_generation,
        tick_uid=f"english-phase-breakdown:{episode.episode_id}",
        canonical=compiled,
        first_workspace_text=episode.first_workspace_text,
        refined_workspace_text=episode.refined_workspace_text,
    )
    rows: dict[str, dict[str, float]] = {}
    for output, target in zip(unroll.outputs, episode.targets, strict=True):
        text, terminated = _decode_one_slice(model, output)
        rows[target.phase] = {
            "text_match": float(terminated and text == target.text),
            "terminated": float(terminated),
            "supervision_weight": float(target.supervision_weight),
        }
    return rows


def curriculum_manifest_bytes(curriculum: LivingReasoningCurriculum) -> bytes:
    return canonical_json_bytes(curriculum.to_canonical_dict())


__all__ = [
    "ENGLISH_REASONING_GATE_REQUIREMENTS",
    "LIVING_OBJECTIVE_COMPONENTS",
    "LIVING_REASONING_CURRICULUM_SCHEMA",
    "LIVING_REASONING_EPISODE_SCHEMA",
    "LIVING_REASONING_GATE_SCHEMA",
    "LIVING_REASONING_TARGET_SCHEMA",
    "LivingReasoningCurriculum",
    "LivingReasoningEpisode",
    "LivingReasoningTarget",
    "LivingRuntimePhase",
    "LivingRuntimeTick",
    "build_living_reasoning_smoke_curriculum",
    "constant_baseline_floors",
    "constant_baseline_target_key",
    "curriculum_manifest_bytes",
    "decide_living_reasoning_mastery",
    "evaluate_living_episode",
    "living_episode_objective",
    "living_phase_breakdown",
    "living_phase_objective",
    "living_source_counterfactuals",
    "run_living_runtime_tick",
]
