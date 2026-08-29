"""Evidence-qualified conversion of exact lived ticks into D64 supervision.

The runtime episode loader proves that an episode is reproducible. This layer
answers the different question of what, if anything, that episode may teach.
A successful final response does not silently bless every earlier proposal:
intermediate phases are supervised only when outcome evidence explicitly names
the whole trajectory as its target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from runtime.field import (
    DeleteText,
    FieldDelta,
    InsertText,
    ReplaceText,
    apply_delta,
    canonical_json_bytes,
    canonical_sha256,
    field_delta_from_canonical_dict,
)
from runtime.heart import (
    AuthorityGrant,
    ReasoningDecision,
    ReasoningEmission,
    ReasoningOperationKind,
)
from runtime.trainer import EpisodeOutcomeQuality, LoadedRuntimeEpisode

from .living_reasoning_curriculum import (
    LivingReasoningCurriculum,
    LivingReasoningEpisode,
    LivingReasoningTarget,
)

LIVED_REASONING_COMPILATION_SCHEMA = "axon-lived-reasoning-compilation-v1"


def _workspace_text(value: Mapping[str, Any]) -> str:
    workspace = dict(value)
    return canonical_json_bytes(
        {
            key: workspace[key]
            for key in (
                "schema",
                "image_id",
                "tick_uid",
                "pass_kind",
                "entries",
                "workspace_id",
            )
        }
    ).decode("utf-8")


def _operation_target(
    *,
    phase: str,
    operation: InsertText | DeleteText | ReplaceText,
) -> LivingReasoningTarget:
    if isinstance(operation, InsertText):
        kind = ReasoningOperationKind.INSERT
    elif isinstance(operation, DeleteText):
        kind = ReasoningOperationKind.DELETE
    elif isinstance(operation, ReplaceText):
        kind = ReasoningOperationKind.REPLACE
    else:  # pragma: no cover - FieldDelta already enforces its union.
        raise TypeError("unsupported exact field operation")
    return LivingReasoningTarget(
        phase=phase,
        decision=ReasoningDecision.DELTA,
        operation=kind,
        region=operation.region,
        start=operation.start,
        end=operation.end,
        payload=operation.replacement_text,
    )


def _delta_target(*, phase: str, delta: FieldDelta) -> LivingReasoningTarget:
    if len(delta.operations) != 1:
        raise ValueError("living D64 target contract currently requires exactly one operation")
    return _operation_target(phase=phase, operation=delta.operations[0])


def _emission_target(*, phase: str, emission: ReasoningEmission) -> LivingReasoningTarget:
    if emission.decision is not ReasoningDecision.DELTA:
        return LivingReasoningTarget(phase=phase, decision=emission.decision)
    if len(emission.operations) != 1:
        raise ValueError("living D64 target contract currently requires exactly one operation")
    operation = emission.operations[0]
    return LivingReasoningTarget(
        phase=phase,
        decision=ReasoningDecision.DELTA,
        operation=operation.kind,
        region=operation.region,
        start=operation.start,
        end=operation.end,
        payload=operation.payload.text,
    )


def _unsupervised_target(phase: str) -> LivingReasoningTarget:
    return LivingReasoningTarget(
        phase=phase,
        decision=ReasoningDecision.ABSTAIN,
        supervision_weight=0.0,
    )


def _one_author_emission(
    values: Sequence[Mapping[str, Any]],
    *,
    core_id: str,
    phase: str,
) -> ReasoningEmission:
    matches = tuple(
        ReasoningEmission.from_mapping(value)
        for value in values
        if value.get("author_core_id") == core_id
    )
    if len(matches) != 1 or matches[0].pass_id != phase:
        raise ValueError("whole-trajectory evidence lacks one exact consolidator phase emission")
    return matches[0]


@dataclass(frozen=True, slots=True)
class LivedReasoningCompilation:
    episodes: tuple[LivingReasoningEpisode, ...]
    excluded_counts: tuple[tuple[str, int], ...]
    source_example_ids: tuple[str, ...]
    compilation_id: str = field(init=False)

    def __post_init__(self) -> None:
        episodes = tuple(self.episodes)
        source_ids = tuple(sorted(set(self.source_example_ids)))
        if len(source_ids) != len(self.source_example_ids):
            raise ValueError("lived compilation source examples must be unique")
        if len(episodes) != len(source_ids):
            raise ValueError("every accepted lived episode must bind one source example")
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "source_example_ids", source_ids)
        object.__setattr__(self, "excluded_counts", tuple(sorted(self.excluded_counts)))
        object.__setattr__(
            self,
            "compilation_id",
            canonical_sha256(self.to_canonical_dict(include_id=False)),
        )

    def require_curriculum(self) -> LivingReasoningCurriculum:
        """Return a curriculum only when train and heldout splits both exist."""

        return LivingReasoningCurriculum(self.episodes)

    def to_canonical_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": LIVED_REASONING_COMPILATION_SCHEMA,
            "source_policy": "runtime-faithful-explicit-outcome-evidence-v1",
            "intermediate_policy": "unsupervised-unless-explicit-full-trajectory-v1",
            "source_example_ids": list(self.source_example_ids),
            "excluded_counts": [list(item) for item in self.excluded_counts],
            "episodes": [item.to_canonical_dict() for item in self.episodes],
        }
        if include_id:
            value["compilation_id"] = self.compilation_id
        return value


class EvidenceQualifiedLivedCurriculumCompiler:
    """Fail-closed converter from verified lived episodes to exact targets."""

    @staticmethod
    def _final_delta(episode: LoadedRuntimeEpisode) -> tuple[FieldDelta, str]:
        outcome = episode.outcome_record
        if outcome is None:
            raise ValueError("missing_explicit_outcome")
        payload = outcome.payload
        scope = str(payload.get("target_scope", "final_delta"))
        quality = episode.example.outcome_quality
        if quality is EpisodeOutcomeQuality.CORRECTED:
            if scope != "corrected_delta" or not isinstance(
                payload.get("corrected_source_delta"), Mapping
            ):
                raise ValueError("invalid_corrected_target")
            delta = field_delta_from_canonical_dict(payload["corrected_source_delta"])
            if (
                delta.base_field_id != episode.pre_action_field.field_id
                or delta.base_tick_id != episode.pre_action_field.tick_id
            ):
                raise ValueError("stale_corrected_target")
            apply_delta(
                episode.pre_action_field,
                delta,
                permitted_regions=AuthorityGrant.consolidator().governed_regions,
            )
            return delta, f"corrected_delta:{outcome.record_id}"
        if quality not in {EpisodeOutcomeQuality.SUCCESS, EpisodeOutcomeQuality.ENDORSED}:
            raise ValueError("outcome_quality_not_teachable")
        if scope not in {"final_delta", "full_trajectory"}:
            raise ValueError("invalid_target_scope")
        return episode.source_delta, f"accepted_{scope}:{outcome.record_id}"

    def compile(
        self,
        loaded: Sequence[LoadedRuntimeEpisode],
    ) -> LivedReasoningCompilation:
        episodes: list[LivingReasoningEpisode] = []
        source_ids: list[str] = []
        excluded: dict[str, int] = {}
        seen: set[str] = set()
        for episode in loaded:
            example_id = episode.example.example_id
            if example_id in seen:
                raise ValueError("duplicate loaded runtime example")
            seen.add(example_id)
            try:
                if not episode.example.serving_promotion_eligible:
                    raise ValueError("not_serving_eligible")
                final_delta, basis = self._final_delta(episode)
                outcome = episode.outcome_record
                assert outcome is not None
                scope = str(outcome.payload.get("target_scope", "final_delta"))
                if scope == "full_trajectory":
                    consolidator = str(episode.circulation["consolidator_core_id"])
                    first = _emission_target(
                        phase="first",
                        emission=_one_author_emission(
                            episode.circulation["first_emissions"],
                            core_id=consolidator,
                            phase="first",
                        ),
                    )
                    refined = _emission_target(
                        phase="refined",
                        emission=_one_author_emission(
                            episode.circulation["refined_emissions"],
                            core_id=consolidator,
                            phase="refined",
                        ),
                    )
                else:
                    first = _unsupervised_target("first")
                    refined = _unsupervised_target("refined")
                target = LivingReasoningEpisode(
                    label=f"lived:{episode.example.episode_event_id}",
                    split=episode.example.split,
                    snapshot=episode.pre_action_field,
                    first_workspace_text=_workspace_text(
                        episode.circulation["first_workspace"]
                    ),
                    refined_workspace_text=_workspace_text(
                        episode.circulation["refined_workspace"]
                    ),
                    targets=(
                        first,
                        refined,
                        _delta_target(phase="consolidated", delta=final_delta),
                    ),
                    mechanism_tags=(
                        "lived_experience",
                        "runtime_faithful",
                        "explicit_outcome_evidence",
                        scope,
                    ),
                    outcome_quality=episode.example.outcome_quality.value,
                    source_example_id=example_id,
                    outcome_evidence_ids=episode.example.outcome_evidence_ids,
                    target_basis=basis,
                )
            except (KeyError, TypeError, ValueError) as exc:
                reason = str(exc) or type(exc).__name__
                excluded[reason] = excluded.get(reason, 0) + 1
                continue
            episodes.append(target)
            source_ids.append(example_id)
        return LivedReasoningCompilation(
            episodes=tuple(sorted(episodes, key=lambda item: item.episode_id)),
            excluded_counts=tuple(excluded.items()),
            source_example_ids=tuple(source_ids),
        )


__all__ = [
    "LIVED_REASONING_COMPILATION_SCHEMA",
    "EvidenceQualifiedLivedCurriculumCompiler",
    "LivedReasoningCompilation",
]
