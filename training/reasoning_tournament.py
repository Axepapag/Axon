"""Comparable architecture and gate contracts for heterogeneous D64 brothers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from runtime.field import canonical_sha256

from .living_reasoning_d64 import LivingReasoningCoreConfig, candidate_a_config

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


__all__ = [
    "D64_TOURNAMENT_RESULT_SCHEMA",
    "D64_TOURNAMENT_SCHEMA",
    "D64Tournament",
    "D64TournamentCandidate",
    "D64TournamentResult",
    "assert_same_gate_surface",
    "d64_head_geometry_tournament",
    "recommended_followup_d64_candidates",
]
