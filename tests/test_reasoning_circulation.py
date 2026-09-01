from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from runtime.dormant import DormantExperienceStore
from runtime.field import FieldDelta, LogicalRegion, ReplaceText
from runtime.heart import (
    BeatConfig,
    CategoricalTextFrame,
    CoreDescriptor,
    CoreRegistry,
    ExactProposalWorkspaceRenderer,
    FrozenTickImage,
    HeartHost,
    HeartHostConfig,
    ParticipantRecord,
    ParticipantState,
    ProposalPass,
    RailBinding,
    ReasoningDecision,
    ReasoningEmission,
    ReasoningOperationEmission,
    ReasoningOperationKind,
    ReasoningPassRequest,
    ReasoningPassResult,
    TickIdentity,
)
from runtime.soul import SoulLayer, SoulTemperature, SoulTransition
from runtime.trainer import (
    EpisodeOutcomeQuality,
    RuntimeEpisodeLoader,
    RuntimeEpisodeSessionCompiler,
)
from training import EvidenceQualifiedLivedCurriculumCompiler


@dataclass
class FixtureCorePort:
    core_id: str

    @staticmethod
    def _decision(
        request: ReasoningPassRequest,
        *,
        decision: ReasoningDecision,
        operations=(),
        detail: str = "",
    ) -> ReasoningEmission:
        return ReasoningEmission(
            base_field_id=request.image.identity.base_field_id,
            base_tick_id=request.image.identity.base_tick_id,
            author_core_id=request.descriptor.core_id,
            pass_id=request.phase,
            rail_d_model=request.descriptor.d_model,
            decision=decision,
            operations=tuple(operations),
            detail=(
                None
                if not detail
                else CategoricalTextFrame.from_text(detail, d_model=request.descriptor.d_model)
            ),
        )

    @staticmethod
    def _result(request: ReasoningPassRequest, emission: ReasoningEmission) -> ReasoningPassResult:
        hot = SoulLayer(
            SoulTemperature.HOT,
            f"{request.descriptor.core_id}:{request.phase}:{request.soul.generation + 1}".encode(),
            tensor_layout="fixture-hot-v1",
        )
        return ReasoningPassResult(
            emission=emission,
            soul_transition=SoulTransition(
                core_id=request.descriptor.core_id,
                architecture_id=request.descriptor.architecture_id,
                parameter_generation=request.descriptor.parameter_generation,
                before_soul_id=request.soul.soul_id,
                before_generation=request.soul.generation,
                tick_uid=request.image.identity.tick_uid,
                request_id=request.request_id,
                phase=request.phase,
                updates=(hot,),
            ),
        )

    def emit(self, request: ReasoningPassRequest) -> ReasoningPassResult:
        if request.phase == "consolidated":
            assert len(request.proposal_rails) == 2
            assert all(rail.text for rail in request.proposal_rails)
            current_response = request.rail.exact_surface.region_text(
                LogicalRegion.RESPONSE_DRAFT.value
            )
            return self._result(
                request,
                self._decision(
                    request,
                    decision=ReasoningDecision.DELTA,
                    operations=(
                    ReasoningOperationEmission(
                        kind=ReasoningOperationKind.REPLACE,
                        region=LogicalRegion.RESPONSE_DRAFT,
                        start=0,
                        end=len(current_response),
                        payload=CategoricalTextFrame.from_text(
                            "I received the exact Unicode: λ🧠",
                            d_model=request.descriptor.d_model,
                        ),
                    ),
                    ),
                ),
            )
        if self.core_id == "core-beta" and request.phase == "first":
            return self._result(
                request,
                self._decision(
                    request,
                    decision=ReasoningDecision.DELTA,
                    operations=(
                    ReasoningOperationEmission(
                        kind=ReasoningOperationKind.INSERT,
                        region=LogicalRegion.SCRATCH,
                        start=0,
                        end=0,
                        payload=CategoricalTextFrame.from_text(
                            "candidate evidence",
                            d_model=request.descriptor.d_model,
                        ),
                    ),
                    ),
                ),
            )
        if self.core_id == "core-beta" and request.phase == "refined":
            assert len(request.proposal_rails) == 1
            return self._result(
                request,
                self._decision(
                    request,
                    decision=ReasoningDecision.ABSTAIN,
                    detail="no grounded refinement",
                ),
            )
        return self._result(
            request,
            self._decision(
                request,
                decision=ReasoningDecision.NO_OP,
                detail="no independent edit",
            ),
        )


@dataclass
class MalformedFixturePort(FixtureCorePort):
    def emit(self, request: ReasoningPassRequest) -> ReasoningPassResult:
        return self._result(
            request,
            ReasoningEmission(
                base_field_id=request.image.identity.base_field_id,
                base_tick_id=request.image.identity.base_tick_id,
                author_core_id="not-the-invoked-core",
                pass_id=request.phase,
                rail_d_model=request.descriptor.d_model,
                decision=ReasoningDecision.NO_OP,
            ),
        )


def _host(root: Path, *, beta_port: FixtureCorePort | None = None) -> HeartHost:
    registry = CoreRegistry(
        (
            CoreDescriptor(core_id="core-alpha", d_model=64),
            CoreDescriptor(core_id="core-beta", d_model=64),
        )
    )
    return HeartHost(
        state_root=root,
        beat_config=BeatConfig(recall_items_per_materialization=0),
        host_config=HeartHostConfig(idle_interval_seconds=0.01),
        core_registry=registry,
        reasoning_ports=(FixtureCorePort("core-alpha"), beta_port or FixtureCorePort("core-beta")),
    )


def test_host_runs_both_barriers_finalizes_turn_and_deposits_loadable_episode(tmp_path: Path) -> None:
    with _host(tmp_path) as host:
        admitted = host.submit_user("hello λ🧠")
        assert admitted.admitted
        beat = host.heartbeat()
        assert beat.reasoning_result is not None
        result = beat.reasoning_result
        assert result.consolidator_core_id == "core-alpha"
        assert [item.state for item in result.first_records] == [
            ParticipantState.NO_OP,
            ParticipantState.RETURNED,
        ]
        assert [item.state for item in result.refined_records] == [
            ParticipantState.NO_OP,
            ParticipantState.ABSTAINED,
        ]
        assert result.first_workspace.require_rail(64).text == result.first_workspace.readable_text()
        assert result.refined_workspace.require_rail(64).text == result.refined_workspace.readable_text()
        assert not host.coordinator.tick_in_flight

        field = beat.field
        assert field.region(LogicalRegion.USER_INPUT).text == ""
        assert field.region(LogicalRegion.RESPONSE_DRAFT).text == "I received the exact Unicode: λ🧠"
        history = field.region(LogicalRegion.CONVERSATION_HISTORY).text
        assert "hello λ🧠" in history
        assert "I received the exact Unicode: λ🧠" in history
        assert result.finalization_receipt is not None
        assert [lineage.core_id for lineage in result.soul_lineages] == [
            "core-alpha",
            "core-beta",
        ]
        assert [len(lineage.transition_receipt_ids) for lineage in result.soul_lineages] == [
            3,
            2,
        ]
        assert [host.soul_store.branch(core).load_head().generation for core in ("core-alpha", "core-beta")] == [
            3,
            2,
        ]

        episode_event_id = f"reasoning-{result.result_id}"
        host.record_episode_outcome(
            event_id="outcome-1",
            episode_event_id=episode_event_id,
            outcome_quality="endorsed",
            evidence_ids=("human-endorsement-1",),
            detail="The exact response was accepted for this fixture episode.",
        )

        experience = DormantExperienceStore(tmp_path)
        session = RuntimeEpisodeSessionCompiler(experience).compile()
        assert len(session.examples) == 1
        assert session.examples[0].serving_promotion_eligible
        loaded = RuntimeEpisodeLoader(experience).load(session)
        assert len(loaded) == 1
        assert loaded[0].pre_action_field.region(LogicalRegion.USER_INPUT).text == "hello λ🧠"
        assert loaded[0].successor_field.field_id == field.field_id
        assert loaded[0].record.exact_text == "I received the exact Unicode: λ🧠"
        assert loaded[0].outcome_record is not None
        assert loaded[0].soul_lineages == tuple(
            lineage.to_canonical_dict() for lineage in result.soul_lineages
        )
        compilation = EvidenceQualifiedLivedCurriculumCompiler().compile(loaded)
        assert len(compilation.episodes) == 1
        lived = compilation.episodes[0]
        assert lived.source_example_id == loaded[0].example.example_id
        assert [item.supervision_weight for item in lived.targets] == [0.0, 0.0, 1.0]
        assert lived.targets[-1].payload == "I received the exact Unicode: λ🧠"

        full_outcome = replace(
            loaded[0].outcome_record,
            payload={
                **loaded[0].outcome_record.payload,
                "target_scope": "full_trajectory",
            },
        )
        full_example = replace(
            loaded[0].example,
            outcome_record_id=full_outcome.record_id,
        )
        full_compilation = EvidenceQualifiedLivedCurriculumCompiler().compile(
            (replace(loaded[0], example=full_example, outcome_record=full_outcome),)
        )
        assert [item.supervision_weight for item in full_compilation.episodes[0].targets] == [
            1.0,
            1.0,
            1.0,
        ]

        corrected_delta = FieldDelta(
            base_field_id=loaded[0].pre_action_field.field_id,
            base_tick_id=loaded[0].pre_action_field.tick_id,
            author_core_id="human-correction",
            pass_id="corrected",
            operations=(
                ReplaceText(
                    region=LogicalRegion.RESPONSE_DRAFT,
                    start=0,
                    end=0,
                    text="Corrected λ🧠",
                ),
            ),
            evidence=("human-correction-1",),
        )
        corrected_outcome = replace(
            loaded[0].outcome_record,
            payload={
                **loaded[0].outcome_record.payload,
                "outcome_quality": "corrected",
                "target_scope": "corrected_delta",
                "corrected_source_delta": corrected_delta.to_canonical_dict(),
            },
        )
        corrected_example = replace(
            loaded[0].example,
            outcome_quality=EpisodeOutcomeQuality.CORRECTED,
            outcome_record_id=corrected_outcome.record_id,
        )
        corrected_compilation = EvidenceQualifiedLivedCurriculumCompiler().compile(
            (
                replace(
                    loaded[0],
                    example=corrected_example,
                    outcome_record=corrected_outcome,
                ),
            )
        )
        assert corrected_compilation.episodes[0].targets[-1].payload == "Corrected λ🧠"


def test_whole_conversation_split_is_stable_for_multiple_episode_records(tmp_path: Path) -> None:
    with _host(tmp_path) as host:
        host.submit_user("first")
        first = host.heartbeat()
        assert first.reasoning_result is not None
        host.submit_user("second")
        second = host.heartbeat()
        assert second.reasoning_result is not None

    experience = DormantExperienceStore(tmp_path)
    session = RuntimeEpisodeSessionCompiler(experience).compile()
    assert len(session.examples) == 2
    assert len({item.conversation_id for item in session.examples}) == 1
    assert len({item.split for item in session.examples}) == 1
    assert session.serving_eligible_count == 0


def test_reasoning_autobiography_recovers_commit_before_deposit_crash(tmp_path: Path) -> None:
    host = _host(tmp_path)
    host.start()
    host.submit_user("recover this exact λ🧠 turn")

    def crash_after_commit(*args, **kwargs):
        raise RuntimeError("injected autobiography outage")

    host._deposit_reasoning_episode = crash_after_commit  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="injected autobiography outage"):
        host.heartbeat()
    committed_field_id = host.coordinator.current_field.field_id
    assert tuple(
        (tmp_path / "active" / "heart" / "reasoning_recovery" / "prepared").glob("*.json")
    )
    host.stop()

    with _host(tmp_path) as recovered:
        assert recovered.coordinator.current_field.field_id == committed_field_id
        completed = tuple(
            (
                tmp_path
                / "active"
                / "heart"
                / "reasoning_recovery"
                / "completed"
            ).glob("*.json")
        )
        assert len(completed) == 1

    experience = DormantExperienceStore(tmp_path)
    session = RuntimeEpisodeSessionCompiler(experience).compile()
    loaded = RuntimeEpisodeLoader(experience).load(session)
    assert len(loaded) == 1
    assert loaded[0].record.exact_text == "I received the exact Unicode: λ🧠"


def test_malformed_core_output_is_rejected_but_accounted_at_both_barriers(tmp_path: Path) -> None:
    with _host(tmp_path, beta_port=MalformedFixturePort("core-beta")) as host:
        host.submit_user("continue despite one malformed participant")
        beat = host.heartbeat()
        assert beat.reasoning_result is not None
        assert beat.reasoning_result.first_records[1].state is ParticipantState.FAILED
        assert beat.reasoning_result.refined_records[1].state is ParticipantState.FAILED
        assert "author differs" in beat.reasoning_result.first_records[1].detail
        assert beat.field.region(LogicalRegion.USER_INPUT).text == ""


def test_exact_workspace_renderer_is_width_generic_without_claiming_wider_compilers() -> None:
    identity = TickIdentity(
        tick_sequence=1,
        heartbeat_id=1,
        base_field_id="f" * 64,
        base_tick_id=0,
    )
    view_id = "v" * 64
    image = FrozenTickImage(
        identity=identity,
        view_id=view_id,
        rails=(
            RailBinding(
                d_model=64,
                rail_id="rail-64",
                source_field_id=identity.base_field_id,
                source_tick_id=0,
                view_id=view_id,
            ),
            RailBinding(
                d_model=128,
                rail_id="rail-128",
                source_field_id=identity.base_field_id,
                source_tick_id=0,
                view_id=view_id,
            ),
        ),
    )
    workspace = ExactProposalWorkspaceRenderer().render(
        image,
        ProposalPass.FIRST,
        (
            ParticipantRecord(
                core_id="core-a",
                d_model=64,
                state=ParticipantState.NO_OP,
            ),
            ParticipantRecord(
                core_id="core-b",
                d_model=128,
                state=ParticipantState.ABSTAINED,
                detail="not enough evidence",
            ),
        ),
    )
    assert workspace.require_rail(64).text == workspace.readable_text()
    assert workspace.require_rail(128).text == workspace.readable_text()
    assert len(workspace.require_rail(64).rows[0]) == 4
    assert len(workspace.require_rail(128).rows[0]) == 8
