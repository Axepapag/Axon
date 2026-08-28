from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from runtime.dormant import DormantExperienceStore
from runtime.field import LogicalRegion
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
    TickIdentity,
)
from runtime.trainer import RuntimeEpisodeLoader, RuntimeEpisodeSessionCompiler


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

    def emit(self, request: ReasoningPassRequest) -> ReasoningEmission:
        if request.phase == "consolidated":
            assert len(request.proposal_rails) == 2
            assert all(rail.text for rail in request.proposal_rails)
            current_response = request.rail.exact_surface.region_text(
                LogicalRegion.RESPONSE_DRAFT.value
            )
            return self._decision(
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
            )
        if self.core_id == "core-beta" and request.phase == "first":
            return self._decision(
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
            )
        if self.core_id == "core-beta" and request.phase == "refined":
            assert len(request.proposal_rails) == 1
            return self._decision(
                request,
                decision=ReasoningDecision.ABSTAIN,
                detail="no grounded refinement",
            )
        return self._decision(
            request,
            decision=ReasoningDecision.NO_OP,
            detail="no independent edit",
        )


@dataclass
class MalformedFixturePort(FixtureCorePort):
    def emit(self, request: ReasoningPassRequest) -> ReasoningEmission:
        return ReasoningEmission(
            base_field_id=request.image.identity.base_field_id,
            base_tick_id=request.image.identity.base_tick_id,
            author_core_id="not-the-invoked-core",
            pass_id=request.phase,
            rail_d_model=request.descriptor.d_model,
            decision=ReasoningDecision.NO_OP,
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
        beat_config=BeatConfig(recall_limit=0),
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
