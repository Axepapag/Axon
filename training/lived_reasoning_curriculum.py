"""Evidence-qualified conversion of lived runtime ticks into English supervision.

The runtime episode loader proves reproducibility.  This layer decides what that
experience may teach under the post-2026-09-19 public reasoning contract.  New
circulation-v3 FIRST/REFINED English proposals may be supervised only when
explicit outcome evidence blesses the whole trajectory.  FINAL supervision is
always compact tagged-region text.  Historical typed circulations may contribute
only a mechanically translated FINAL target; their intermediate typed emissions
are never converted into English supervision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from runtime.field import (
    FieldDelta,
    apply_delta,
    canonical_json_bytes,
    canonical_sha256,
    field_delta_from_canonical_dict,
)
from runtime.heart import AuthorityGrant, EnglishProposal, TechnicalFinalVerdict
from runtime.trainer import EpisodeOutcomeQuality, LoadedRuntimeEpisode

from .living_reasoning_curriculum import LivingReasoningCurriculum, LivingReasoningEpisode, LivingReasoningTarget

LIVED_REASONING_COMPILATION_SCHEMA = "axon-lived-english-reasoning-compilation-v1"
_CURRENT_CIRCULATION_SCHEMA = "axon-reasoning-circulation-v3"


def _workspace_text(value: Mapping[str, Any]) -> str:
    workspace = dict(value)
    readable = workspace.get("readable_text")
    if isinstance(readable, str):
        return readable
    # Immutable v2 episodes did not carry the new readable field.  Keep their
    # exact workspace available as context without pretending it was English
    # proposal supervision.
    return canonical_json_bytes(workspace).decode("utf-8")


def _one_author_proposal(
    values: Sequence[Mapping[str, Any]],
    *,
    core_id: str,
    phase: str,
) -> EnglishProposal:
    matches = tuple(
        EnglishProposal.from_mapping(value)
        for value in values
        if value.get("author_core_id") == core_id
    )
    if len(matches) != 1 or matches[0].pass_id != phase:
        raise ValueError("whole-trajectory evidence lacks one exact consolidator English proposal")
    return matches[0]


def _unsupervised_text_target(phase: str, text: str | None = None) -> LivingReasoningTarget:
    placeholder = text or "Historical phase retained as context only; no English supervision is authorized."
    return LivingReasoningTarget(phase=phase, text=placeholder, supervision_weight=0.0)


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
        object.__setattr__(self, "compilation_id", canonical_sha256(self.to_canonical_dict(False)))

    def require_curriculum(self) -> LivingReasoningCurriculum:
        return LivingReasoningCurriculum(self.episodes)

    def to_canonical_dict(self, include_id: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": LIVED_REASONING_COMPILATION_SCHEMA,
            "source_policy": "runtime-faithful-explicit-outcome-evidence-english-v1",
            "intermediate_policy": "english-v3-only-and-unsupervised-unless-full-trajectory-v1",
            "historical_typed_policy": "final-only-mechanical-tagged-translation-v1",
            "source_example_ids": list(self.source_example_ids),
            "excluded_counts": [list(item) for item in self.excluded_counts],
            "episodes": [item.to_canonical_dict() for item in self.episodes],
        }
        if include_id:
            value["compilation_id"] = self.compilation_id
        return value


class EvidenceQualifiedLivedCurriculumCompiler:
    """Fail-closed converter from verified lived episodes to English targets."""

    @staticmethod
    def _outcome_scope(episode: LoadedRuntimeEpisode) -> tuple[str, str]:
        outcome = episode.outcome_record
        if outcome is None:
            raise ValueError("missing_explicit_outcome")
        payload = outcome.payload
        scope = str(payload.get("target_scope", "final_delta"))
        quality = episode.example.outcome_quality
        if quality is EpisodeOutcomeQuality.CORRECTED:
            if scope != "corrected_delta" or not isinstance(payload.get("corrected_source_delta"), Mapping):
                raise ValueError("invalid_corrected_target")
            return scope, f"corrected_delta:{outcome.record_id}"
        if quality not in {EpisodeOutcomeQuality.SUCCESS, EpisodeOutcomeQuality.ENDORSED}:
            raise ValueError("outcome_quality_not_teachable")
        if scope not in {"final_delta", "full_trajectory"}:
            raise ValueError("invalid_target_scope")
        return scope, f"accepted_{scope}:{outcome.record_id}"

    @staticmethod
    def _corrected_delta(episode: LoadedRuntimeEpisode) -> FieldDelta:
        outcome = episode.outcome_record
        assert outcome is not None
        delta = field_delta_from_canonical_dict(outcome.payload["corrected_source_delta"])
        if delta.base_field_id != episode.pre_action_field.field_id or delta.base_tick_id != episode.pre_action_field.tick_id:
            raise ValueError("stale_corrected_target")
        apply_delta(
            episode.pre_action_field,
            delta,
            permitted_regions=AuthorityGrant.consolidator().governed_regions,
        )
        # Historical corrections may use pass_id=corrected.  The public FINAL
        # contract is consolidated, so rebind only the runtime envelope before
        # mechanically rendering the equivalent desired-region text.
        if str(delta.pass_id) != "consolidated":
            delta = FieldDelta(
                base_field_id=delta.base_field_id,
                base_tick_id=delta.base_tick_id,
                author_core_id=delta.author_core_id,
                pass_id="consolidated",
                operations=delta.operations,
                evidence=delta.evidence,
            )
        return delta

    @staticmethod
    def _final_text(episode: LoadedRuntimeEpisode, scope: str) -> str:
        if scope == "corrected_delta":
            delta = EvidenceQualifiedLivedCurriculumCompiler._corrected_delta(episode)
            return TechnicalFinalVerdict.from_delta(episode.pre_action_field, delta, rail_d_model=64).text

        circulation = episode.circulation
        if circulation.get("schema") == _CURRENT_CIRCULATION_SCHEMA:
            verdict = TechnicalFinalVerdict.from_mapping(circulation["consolidator_verdict"])
            # Materialization is a second exact check that the serialized FINAL
            # remains valid against the loaded frozen base.
            verdict.materialize(episode.pre_action_field)
            return verdict.text

        # Immutable v2 evidence remains useful for accepted FINAL behavior only.
        # Convert its already-verified source delta mechanically; do not imitate
        # the old decision/operation/address language.
        delta = episode.source_delta
        if str(delta.pass_id) != "consolidated":
            delta = FieldDelta(
                base_field_id=delta.base_field_id,
                base_tick_id=delta.base_tick_id,
                author_core_id=delta.author_core_id,
                pass_id="consolidated",
                operations=delta.operations,
                evidence=delta.evidence,
            )
        return TechnicalFinalVerdict.from_delta(episode.pre_action_field, delta, rail_d_model=64).text

    @staticmethod
    def _intermediate_targets(
        episode: LoadedRuntimeEpisode,
        *,
        scope: str,
    ) -> tuple[LivingReasoningTarget, LivingReasoningTarget]:
        circulation = episode.circulation
        is_v3 = circulation.get("schema") == _CURRENT_CIRCULATION_SCHEMA
        if is_v3:
            consolidator = str(circulation["consolidator_core_id"])
            first_proposal = _one_author_proposal(
                circulation["first_proposals"],
                core_id=consolidator,
                phase="first",
            )
            refined_proposal = _one_author_proposal(
                circulation["refined_proposals"],
                core_id=consolidator,
                phase="refined",
            )
            if scope == "full_trajectory":
                return (
                    LivingReasoningTarget(phase="first", text=first_proposal.text),
                    LivingReasoningTarget(phase="refined", text=refined_proposal.text),
                )
            return (
                _unsupervised_text_target("first", first_proposal.text),
                _unsupervised_text_target("refined", refined_proposal.text),
            )

        if scope == "full_trajectory":
            raise ValueError("historical_typed_full_trajectory_is_not_english_supervision")
        return (_unsupervised_text_target("first"), _unsupervised_text_target("refined"))

    def compile(self, loaded: Sequence[LoadedRuntimeEpisode]) -> LivedReasoningCompilation:
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
                scope, basis = self._outcome_scope(episode)
                first, refined = self._intermediate_targets(episode, scope=scope)
                final_text = self._final_text(episode, scope)
                target = LivingReasoningEpisode(
                    label=f"lived:{episode.example.episode_event_id}",
                    split=episode.example.split,
                    snapshot=episode.pre_action_field,
                    first_workspace_text=_workspace_text(episode.circulation["first_workspace"]),
                    refined_workspace_text=_workspace_text(episode.circulation["refined_workspace"]),
                    targets=(
                        first,
                        refined,
                        LivingReasoningTarget(phase="consolidated", text=final_text),
                    ),
                    mechanism_tags=(
                        "lived_experience",
                        "runtime_faithful",
                        "explicit_outcome_evidence",
                        "english_reasoning",
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
