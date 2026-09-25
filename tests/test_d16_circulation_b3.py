from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from runtime.field import LogicalRegion
from runtime.heart import (
    BeatConfig,
    CanonicalSyncEvent,
    CoreDescriptor,
    CoreRegistry,
    D16CoreMirror,
    EnglishProposal,
    FieldDeltaEvent,
    FieldSnapshotEvent,
    HeartHost,
    HeartHostConfig,
    MirrorAck,
    ReasoningPassRequest,
    ReasoningPassResult,
    TechnicalFinalVerdict,
)
from runtime.soul import SoulLayer, SoulTemperature, SoulTransition


@dataclass
class D16FixturePort:
    core_id: str
    core_generation: int = 0
    events: list[str] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mirror = D16CoreMirror(self.core_id, self.core_generation)

    def apply_field_event(
        self,
        event: FieldSnapshotEvent | FieldDeltaEvent | CanonicalSyncEvent,
    ) -> MirrorAck:
        self.events.append(event.kind.value)
        if isinstance(event, FieldSnapshotEvent):
            return self.mirror.apply_snapshot(event)
        return self.mirror.apply_delta(event)

    @staticmethod
    def _result(
        request: ReasoningPassRequest,
        output: EnglishProposal | TechnicalFinalVerdict,
    ) -> ReasoningPassResult:
        hot = SoulLayer(
            SoulTemperature.HOT,
            f"{request.descriptor.core_id}:{request.phase}:{request.soul.generation + 1}".encode(),
            tensor_layout="d16-fixture-hot-v1",
        )
        return ReasoningPassResult(
            output=output,
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
        self.phases.append(request.phase)
        assert request.rail is None
        assert request.d16_binding is not None
        assert request.d16_binding.core_id == self.core_id
        assert self.mirror.identity == request.d16_binding.identity
        assert self.mirror.view is not None

        if request.phase == "first":
            assert request.proposal_frames == ()
            output = EnglishProposal(
                base_field_id=request.image.identity.base_field_id,
                base_tick_id=request.image.identity.base_tick_id,
                author_core_id=self.core_id,
                pass_id="first",
                rail_d_model=request.descriptor.d_model,
                text=f"{self.core_id} read the exact D16 field without a private-width rail.",
            )
            return self._result(request, output)

        if request.phase == "refined":
            assert len(request.proposal_frames) == 1
            assert request.proposal_frames[0].decode().startswith("FIRST PROPOSALS")
            output = EnglishProposal(
                base_field_id=request.image.identity.base_field_id,
                base_tick_id=request.image.identity.base_tick_id,
                author_core_id=self.core_id,
                pass_id="refined",
                rail_d_model=request.descriptor.d_model,
                text=f"{self.core_id} refined after reading the D16 proposal set.",
            )
            return self._result(request, output)

        assert request.phase == "consolidated"
        assert len(request.proposal_frames) == 2
        assert all(frame.decode() for frame in request.proposal_frames)
        user_text = self.mirror.view.region(LogicalRegion.USER_INPUT).text
        verdict = TechnicalFinalVerdict(
            base_field_id=request.image.identity.base_field_id,
            base_tick_id=request.image.identity.base_tick_id,
            author_core_id=self.core_id,
            rail_d_model=request.descriptor.d_model,
            text=f"#responseDraft# D16:{user_text}",
        )
        return self._result(request, verdict)


def test_d512_core_completes_live_circulation_without_d512_rail(tmp_path: Path) -> None:
    d512 = D16FixturePort("core-d512")
    registry = CoreRegistry((CoreDescriptor(core_id="core-d512", d_model=512),))
    with HeartHost(
        state_root=tmp_path,
        beat_config=BeatConfig(recall_items_per_materialization=0),
        host_config=HeartHostConfig(idle_interval_seconds=0.01),
        core_registry=registry,
        reasoning_ports=(d512,),
    ) as host:
        assert host.submit_user("CAT λ").admitted
        beat = host.heartbeat()
        assert beat.reasoning_result is not None
        result = beat.reasoning_result
        assert result.image.rail_for(512) is None
        assert result.image.rail_for(64) is not None
        assert result.consolidator_core_id == "core-d512"
        assert result.first_workspace.require_d16_frame().decode() == result.first_workspace.readable_text()
        assert result.refined_workspace.require_d16_frame().decode() == result.refined_workspace.readable_text()
        assert beat.field.region(LogicalRegion.RESPONSE_DRAFT).text == "D16:CAT λ"
        assert d512.phases == ["first", "refined", "consolidated"]
        assert d512.events[0] == "FIELD_SNAPSHOT"
        assert d512.events[-1] == "CANONICAL_SYNC"
        assert d512.mirror.view is not None
        assert d512.mirror.view.source_field_id == beat.field.field_id
        assert d512.mirror.view.source_tick_id == beat.field.tick_id
