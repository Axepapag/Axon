"""Comparable architecture and gate contracts for heterogeneous D64 brothers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import torch

from runtime.field import LogicalRegion, SharedFieldSnapshot, canonical_sha256
from runtime.soul import SoulSnapshot, SoulTemperature

from .living_reasoning_curriculum import (
    LivingReasoningEpisode,
    evaluate_living_episode,
    living_phase_breakdown,
)
from .living_reasoning_d64 import (
    LivingReasoningCoreConfig,
    LivingReasoningCoreD64,
    LivingReasoningForward,
    candidate_a_config,
)
from .sequential_first_form import SequentialFirstFormCase, evaluate_sequential_case

D64_TOURNAMENT_SCHEMA = "axon-d64-reasoning-tournament-v1"
D64_TOURNAMENT_RESULT_SCHEMA = "axon-d64-reasoning-tournament-result-v1"


@dataclass(frozen=True, slots=True)
class D64TournamentCandidate:
    label: str
    config: LivingReasoningCoreConfig
    hypothesis: str
    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.label or not self.hypothesis:
            raise ValueError("tournament candidate label and hypothesis must be non-empty")
        object.__setattr__(self, "candidate_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "label": self.label,
            "architecture": self.config.to_canonical_dict(),
            "hypothesis": self.hypothesis,
        }
        if include_id:
            value["candidate_id"] = self.candidate_id
        return value


@dataclass(frozen=True, slots=True)
class D64Tournament:
    candidates: tuple[D64TournamentCandidate, ...]
    required_metrics: tuple[str, ...] = (
        "heldout_typed_emission_exact_rate",
        "heldout_payload_transport_exact_rate",
        "complete_field_coverage_rate",
        "relevant_soul_ablation_degradation",
        "irrelevant_soul_ablation_delta",
        "stale_soul_degradation",
        "swapped_soul_rejection_rate",
        "current_field_override_rate",
        "proposal_refinement_gain",
        "retained_regression_rate",
    )
    tournament_id: str = field(init=False)

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if len(candidates) < 2:
            raise ValueError("a tournament requires at least two candidates")
        if len({item.config.architecture_id for item in candidates}) != len(candidates):
            raise ValueError("tournament architectures must be distinct")
        metrics = tuple(self.required_metrics)
        if not metrics or len(set(metrics)) != len(metrics):
            raise ValueError("tournament metrics must be nonempty and unique")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "required_metrics", metrics)
        object.__setattr__(self, "tournament_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": D64_TOURNAMENT_SCHEMA,
            "candidates": [item.to_canonical_dict() for item in self.candidates],
            "required_metrics": list(self.required_metrics),
            "comparison_law": (
                "identical curriculum splits, optimizer budget, seeds, Soul ablations, "
                "complete-field gates, and promotion thresholds"
            ),
        }
        if include_id:
            value["tournament_id"] = self.tournament_id
        return value


@dataclass(frozen=True, slots=True)
class D64TournamentResult:
    tournament_id: str
    candidate_id: str
    metrics: tuple[tuple[str, float], ...]
    gate_passed: bool
    result_id: str = field(init=False)

    @classmethod
    def from_mapping(
        cls,
        tournament: D64Tournament,
        candidate: D64TournamentCandidate,
        metrics: Mapping[str, float],
        *,
        gate_passed: bool,
    ) -> "D64TournamentResult":
        if set(metrics) != set(tournament.required_metrics):
            raise ValueError("tournament result does not report the identical metric surface")
        return cls(
            tournament_id=tournament.tournament_id,
            candidate_id=candidate.candidate_id,
            metrics=tuple(sorted((name, float(value)) for name, value in metrics.items())),
            gate_passed=gate_passed,
        )

    def __post_init__(self) -> None:
        if any(not (-float("inf") < value < float("inf")) for _, value in self.metrics):
            raise ValueError("tournament metrics must be finite")
        object.__setattr__(self, "result_id", canonical_sha256(self.to_canonical_dict(False)))

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": D64_TOURNAMENT_RESULT_SCHEMA,
            "tournament_id": self.tournament_id,
            "candidate_id": self.candidate_id,
            "metrics": {name: value for name, value in self.metrics},
            "gate_passed": self.gate_passed,
        }
        if include_id:
            value["result_id"] = self.result_id
        return value


def d64_head_geometry_tournament() -> D64Tournament:
    common = {
        "n_layers": 2,
        "ffn_dim": 131_072,
        "state_tokens": 4,
        "page_size": 32,
        "dropout": 0.0,
    }
    return D64Tournament(
        candidates=tuple(
            D64TournamentCandidate(
                label=label,
                config=candidate_a_config(n_heads=heads, **common),
                hypothesis=hypothesis,
            )
            for label, heads, hypothesis in (
                (
                    "candidate-a-1x64",
                    1,
                    "One full-width attention head preserves the entire D64 relation subspace.",
                ),
                (
                    "candidate-b-2x32",
                    2,
                    "Two D32 heads may separate complementary relations without becoming too narrow.",
                ),
                (
                    "candidate-c-4x16",
                    4,
                    "Four D16 heads test whether added routing diversity outweighs narrow head geometry.",
                ),
            )
        )
    )


def recommended_followup_d64_candidates() -> tuple[D64TournamentCandidate, ...]:
    return (
        D64TournamentCandidate(
            "candidate-d-1x64-deep-small-ffn",
            candidate_a_config(n_heads=1, n_layers=6, ffn_dim=4096),
            "Trade private FFN capacity for additional sequential attention/routing depth.",
        ),
        D64TournamentCandidate(
            "candidate-e-2x32-balanced",
            candidate_a_config(n_heads=2, n_layers=4, ffn_dim=16_384),
            "Balanced depth, head diversity, and procedural capacity.",
        ),
    )


def assert_same_gate_surface(
    tournament: D64Tournament,
    results: Iterable[D64TournamentResult],
) -> None:
    rows = tuple(results)
    expected_candidates = {item.candidate_id for item in tournament.candidates}
    if {item.candidate_id for item in rows} != expected_candidates:
        raise ValueError("tournament results do not cover every candidate exactly once")
    if any(item.tournament_id != tournament.tournament_id for item in rows):
        raise ValueError("tournament result belongs to another tournament")
    expected_metrics = set(tournament.required_metrics)
    if any({name for name, _ in item.metrics} != expected_metrics for item in rows):
        raise ValueError("tournament candidates were not evaluated on the same gate surface")


TOURNAMENT_METRIC_COMPUTATION_SCHEMA = "axon-d64-tournament-metric-computation-v2"


def _probe_definitions() -> dict[str, str]:
    return {
        "heldout_typed_emission_exact_rate": (
            "exact free-running typed emissions divided by supervised phase targets"
        ),
        "heldout_payload_transport_exact_rate": (
            "exact terminated Unicode payloads divided by supervised DELTA payloads"
        ),
        "complete_field_coverage_rate": (
            "complete canonical coverage receipts divided by runtime phase outputs"
        ),
        "relevant_soul_ablation_degradation": (
            "intact heldout typed exact rate minus the identical surface with HOT Soul "
            "inhalation ablated at every phase"
        ),
        "irrelevant_soul_ablation_delta": (
            "absolute typed exact-rate change when empty DEEP_COLD inhalation is ablated"
        ),
        "stale_soul_degradation": (
            "intact heldout typed exact rate minus the identical surface from the "
            "pre-segment same-core Soul"
        ),
        "swapped_soul_rejection_rate": (
            "foreign-core Soul inhale attempts rejected before emission divided by attempts"
        ),
        "current_field_override_rate": (
            "FFCS-D counterfactual pairs exact on both the published field and a changed "
            "canonical signal while the stale proposal board is held fixed"
        ),
        "proposal_refinement_gain": (
            "FFCS-D refined-phase exact rate with the first proposal board minus the "
            "same rate with that board omitted"
        ),
        "retained_regression_rate": (
            "exact typed emissions divided by supervised targets on the complete "
            "regression split"
        ),
    }


@dataclass(frozen=True, slots=True)
class D64TournamentMetricComputation:
    metrics: tuple[tuple[str, float], ...]
    missing_metrics: tuple[str, ...]
    heldout_episode_ids: tuple[str, ...]
    heldout_sequential_case_ids: tuple[str, ...]
    regression_episode_ids: tuple[str, ...]
    regression_sequential_case_ids: tuple[str, ...]
    field_override_probe_ids: tuple[str, ...]
    proposal_refinement_probe_ids: tuple[str, ...]
    stale_soul_id: str | None
    heldout_surface_complete: bool
    regression_surface_complete: bool
    computation_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = tuple(sorted((name, float(value)) for name, value in self.metrics))
        if any(not (-float("inf") < value < float("inf")) for _, value in values):
            raise ValueError("tournament metric computation contains a non-finite value")
        object.__setattr__(self, "metrics", values)
        object.__setattr__(self, "missing_metrics", tuple(sorted(self.missing_metrics)))
        object.__setattr__(
            self,
            "computation_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    @property
    def metric_mapping(self) -> dict[str, float]:
        return dict(self.metrics)

    @property
    def complete(self) -> bool:
        return not self.missing_metrics

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": TOURNAMENT_METRIC_COMPUTATION_SCHEMA,
            "definitions": _probe_definitions(),
            "metrics": self.metric_mapping,
            "missing_metrics": list(self.missing_metrics),
            "heldout_episode_ids": list(self.heldout_episode_ids),
            "heldout_sequential_case_ids": list(self.heldout_sequential_case_ids),
            "regression_episode_ids": list(self.regression_episode_ids),
            "regression_sequential_case_ids": list(
                self.regression_sequential_case_ids
            ),
            "field_override_probe_ids": list(self.field_override_probe_ids),
            "proposal_refinement_probe_ids": list(
                self.proposal_refinement_probe_ids
            ),
            "stale_soul_id": self.stale_soul_id,
            "heldout_surface_complete": self.heldout_surface_complete,
            "regression_surface_complete": self.regression_surface_complete,
            "aggregation_law": "count-weighted exact rates; no absent probe is imputed",
        }
        if include_id:
            value["computation_id"] = self.computation_id
        return value


def _evaluation_rows(
    model: LivingReasoningCoreD64,
    episodes: Sequence[LivingReasoningEpisode],
    sequential_cases: Sequence[SequentialFirstFormCase],
    soul: SoulSnapshot,
    *,
    core_id: str,
    parameter_generation: str,
    ablate_temperatures: tuple[SoulTemperature, ...] = (),
) -> list[Mapping[str, Any]]:
    return [
        evaluate_living_episode(
            model,
            episode,
            soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
            ablate_temperatures=ablate_temperatures,
        )
        for episode in episodes
    ] + [
        evaluate_sequential_case(
            model,
            case,
            soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
            ablate_temperatures=ablate_temperatures,
        )
        for case in sequential_cases
    ]


def _aggregate_exact(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("an exact-rate aggregate requires one or more evaluated rows")
    supervised = sum(row["supervised_phase_count"] for row in rows)
    payloads = sum(row["payload_supervised_phase_count"] for row in rows)
    phases = sum(row["phase_output_count"] for row in rows)
    return {
        "typed": sum(row["typed_emission_exact_count"] for row in rows)
        / max(1.0, supervised),
        "payload": sum(row["payload_transport_exact_count"] for row in rows)
        / max(1.0, payloads),
        "coverage": sum(row["complete_field_coverage_count"] for row in rows)
        / max(1.0, phases),
    }


def _target_exact(
    model: LivingReasoningCoreD64,
    output: LivingReasoningForward,
    target: Any,
) -> float:
    from runtime.heart import ReasoningDecision, ReasoningOperationKind

    decision = tuple(ReasoningDecision)[int(output.decision_logits.argmax(dim=-1).item())]
    if decision is not target.decision:
        return 0.0
    if target.decision is not ReasoningDecision.DELTA:
        return 1.0
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
    return float(
        operation is target.operation
        and region is target.region
        and start == target.start
        and end == target.end
        and terminated
        and payload == target.payload
    )


def _field_override_variant(episode: LivingReasoningEpisode) -> LivingReasoningEpisode:
    current = episode.targets[-1].payload
    replacement = f"CF-{current}"
    texts = {state.name: state.text for state in episode.snapshot.regions}
    marker = f"signal={current}"
    if marker not in texts[LogicalRegion.CORTEX]:
        raise ValueError("FFCS-D field override probe lacks its canonical signal marker")
    texts[LogicalRegion.CORTEX] = texts[LogicalRegion.CORTEX].replace(
        marker, f"signal={replacement}", 1
    )
    snapshot = SharedFieldSnapshot.from_texts(
        texts,
        tick_id=episode.snapshot.tick_id,
        source_manifest_ids=episode.snapshot.source_manifest_ids,
    )
    targets = tuple(
        replace(target, payload=target.payload.replace(current, replacement))
        if target.payload
        else target
        for target in episode.targets
    )
    return replace(
        episode,
        label=f"{episode.label}-field-override-cf",
        snapshot=snapshot,
        targets=targets,
        source_example_id=canonical_sha256(
            {"source_episode_id": episode.episode_id, "counterfactual": replacement}
        ),
        target_basis=(
            "counterfactual current canonical signal; published proposal boards held fixed"
        ),
    )


@torch.no_grad()
def d64_tournament_metric_computation(
    model: LivingReasoningCoreD64,
    *,
    episodes: Sequence[LivingReasoningEpisode],
    sequential_cases: Sequence[SequentialFirstFormCase],
    initial_soul: SoulSnapshot,
    regression_episodes: Sequence[LivingReasoningEpisode],
    regression_sequential_cases: Sequence[SequentialFirstFormCase] = (),
    stale_soul: SoulSnapshot | None,
    core_id: str,
    parameter_generation: str,
    heldout_surface_complete: bool = True,
    regression_surface_complete: bool = True,
) -> D64TournamentMetricComputation:
    """Compute only admissible metrics and name every unavailable probe.

    Missing D fixtures, regression material, or a genuinely older same-core Soul
    leave their metrics absent.  No neutral-looking zero or perfect one is imputed.
    """

    episodes = tuple(episodes)
    sequential_cases = tuple(sequential_cases)
    regression_episodes = tuple(regression_episodes)
    regression_sequential_cases = tuple(regression_sequential_cases)
    if not episodes and not sequential_cases:
        raise ValueError("tournament metrics require heldout episodes or sequential cases")

    intact_rows = _evaluation_rows(
        model,
        episodes,
        sequential_cases,
        initial_soul,
        core_id=core_id,
        parameter_generation=parameter_generation,
    )
    intact = _aggregate_exact(intact_rows)
    hot_ablated = _aggregate_exact(
        _evaluation_rows(
            model,
            episodes,
            sequential_cases,
            initial_soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
            ablate_temperatures=(SoulTemperature.HOT,),
        )
    )
    deep_cold_ablated = _aggregate_exact(
        _evaluation_rows(
            model,
            episodes,
            sequential_cases,
            initial_soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
            ablate_temperatures=(SoulTemperature.DEEP_COLD,),
        )
    )
    metrics = {
        "heldout_typed_emission_exact_rate": intact["typed"],
        "heldout_payload_transport_exact_rate": intact["payload"],
        "complete_field_coverage_rate": intact["coverage"],
        "relevant_soul_ablation_degradation": (
            intact["typed"] - hot_ablated["typed"]
        ),
        "irrelevant_soul_ablation_delta": abs(
            intact["typed"] - deep_cold_ablated["typed"]
        ),
    }

    if stale_soul is not None:
        stale = _aggregate_exact(
            _evaluation_rows(
                model,
                episodes,
                sequential_cases,
                stale_soul,
                core_id=core_id,
                parameter_generation=parameter_generation,
            )
        )
        metrics["stale_soul_degradation"] = intact["typed"] - stale["typed"]

    foreign_soul = SoulSnapshot(
        core_id=f"{core_id}-foreign-brother",
        architecture_id=model.architecture_id,
        parameter_generation=parameter_generation,
        generation=initial_soul.generation,
        parent_soul_id=None,
        layers=initial_soul.layers,
    )
    try:
        _evaluation_rows(
            model,
            episodes[:1],
            sequential_cases[:1] if not episodes else (),
            foreign_soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
        )
    except ValueError:
        metrics["swapped_soul_rejection_rate"] = 1.0
    else:
        metrics["swapped_soul_rejection_rate"] = 0.0

    d_episodes = tuple(
        episode for episode in episodes if "ffcs_d" in episode.mechanism_tags
    )
    field_probe_ids: list[str] = []
    field_pair_scores: list[float] = []
    proposal_probe_ids: list[str] = []
    intact_refined: list[float] = []
    ablated_refined: list[float] = []
    from runtime.field import D64FieldCompiler

    for episode in d_episodes:
        variant = _field_override_variant(episode)
        baseline = living_phase_breakdown(
            model,
            episode,
            initial_soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
        )["consolidated"]["decision_match"]
        changed = living_phase_breakdown(
            model,
            variant,
            initial_soul,
            core_id=core_id,
            parameter_generation=parameter_generation,
        )["consolidated"]["decision_match"]
        field_pair_scores.append(float(baseline == 1.0 and changed == 1.0))
        field_probe_ids.append(
            canonical_sha256(
                {"baseline": episode.episode_id, "counterfactual": variant.episode_id}
            )
        )

        compiled = D64FieldCompiler().compile(episode.snapshot)
        unroll = model.unroll_runtime_phases(
            initial_soul=initial_soul,
            expected_core_id=core_id,
            parameter_generation=parameter_generation,
            tick_uid=f"proposal-refinement-probe:{episode.episode_id}",
            canonical=compiled,
            first_workspace_text=episode.first_workspace_text,
            refined_workspace_text=episode.refined_workspace_text,
        )
        intact_refined.append(_target_exact(model, unroll.outputs[1], episode.targets[1]))
        without_board = model.forward_surfaces(
            soul=unroll.souls[1],
            expected_core_id=core_id,
            parameter_generation=parameter_generation,
            phase="refined",
            canonical=compiled,
            proposal_texts=(),
        )
        ablated_refined.append(_target_exact(model, without_board, episode.targets[1]))
        proposal_probe_ids.append(episode.episode_id)

    if field_pair_scores:
        metrics["current_field_override_rate"] = sum(field_pair_scores) / len(
            field_pair_scores
        )
        metrics["proposal_refinement_gain"] = (
            sum(intact_refined) / len(intact_refined)
            - sum(ablated_refined) / len(ablated_refined)
        )

    regression_rows = _evaluation_rows(
        model,
        regression_episodes,
        regression_sequential_cases,
        initial_soul,
        core_id=core_id,
        parameter_generation=parameter_generation,
    ) if regression_episodes or regression_sequential_cases else []
    if regression_rows:
        metrics["retained_regression_rate"] = _aggregate_exact(regression_rows)["typed"]

    required = d64_head_geometry_tournament().required_metrics
    missing = {name for name in required if name not in metrics}
    if not heldout_surface_complete:
        missing.update(name for name in required if name != "retained_regression_rate")
    if not regression_surface_complete:
        missing.add("retained_regression_rate")
    return D64TournamentMetricComputation(
        metrics=tuple(metrics.items()),
        missing_metrics=tuple(missing),
        heldout_episode_ids=tuple(item.episode_id for item in episodes),
        heldout_sequential_case_ids=tuple(item.case_id for item in sequential_cases),
        regression_episode_ids=tuple(item.episode_id for item in regression_episodes),
        regression_sequential_case_ids=tuple(
            item.case_id for item in regression_sequential_cases
        ),
        field_override_probe_ids=tuple(field_probe_ids),
        proposal_refinement_probe_ids=tuple(proposal_probe_ids),
        stale_soul_id=None if stale_soul is None else stale_soul.soul_id,
        heldout_surface_complete=heldout_surface_complete,
        regression_surface_complete=regression_surface_complete,
    )


@torch.no_grad()
def d64_tournament_metrics(
    model: LivingReasoningCoreD64,
    **kwargs: Any,
) -> dict[str, float]:
    """Compatibility wrapper returning the non-imputed metric mapping."""

    return d64_tournament_metric_computation(model, **kwargs).metric_mapping


__all__ = [
    "D64_TOURNAMENT_RESULT_SCHEMA",
    "D64_TOURNAMENT_SCHEMA",
    "TOURNAMENT_METRIC_COMPUTATION_SCHEMA",
    "D64Tournament",
    "D64TournamentCandidate",
    "D64TournamentMetricComputation",
    "D64TournamentResult",
    "assert_same_gate_surface",
    "d64_head_geometry_tournament",
    "d64_tournament_metric_computation",
    "d64_tournament_metrics",
    "recommended_followup_d64_candidates",
]
