"""End-to-end: a real HeartHost beat drives the ContinuousD16CorePort.

The port-level tests in tests/test_continuous_d16_port.py prove the ingestion
logic. This proves the port is a genuine field participant: a live HeartHost
circulation beat issues the real mirror events, the Core reads real canonical
content across consecutive beats, and its recurrent state carries between them.

This is the rung the ContinuousCoreLab proved in isolation and Axon had never
connected: a trained recurrent chamber reading the live field.

Discovery recorded here (measured, not assumed): during a live beat the
consolidator ROTATES the user's turn out of ``user_input`` and into
``conversation_history`` as a turn frame before the canonical sync. So a Core
joining after consolidation must not expect the raw message in ``user_input``;
it must read the turn frame. The port's mirror is asserted byte-identical to the
beat's own field, which is the real contract.
"""
from __future__ import annotations

from pathlib import Path

from runtime.axon_runtime import ContinuousCoreD512
from runtime.field import LogicalRegion
from runtime.heart import BeatConfig, CoreDescriptor, CoreRegistry, HeartHost, HeartHostConfig
from runtime.heart.continuous_d16_port import ContinuousD16CorePort


def _host(tmp_path: Path, port: ContinuousD16CorePort) -> HeartHost:
    registry = CoreRegistry((CoreDescriptor(core_id=port.core_id, d_model=512),))
    return HeartHost(
        state_root=tmp_path,
        beat_config=BeatConfig(recall_items_per_materialization=0),
        host_config=HeartHostConfig(idle_interval_seconds=0.01),
        core_registry=registry,
        reasoning_ports=(port,),
    )


def _region_text(view, region: LogicalRegion) -> str:
    for item in view.regions:
        if item.region == region:
            return item.text
    return ""


def test_live_beat_initialises_core_state_from_the_real_field(tmp_path: Path) -> None:
    port = ContinuousD16CorePort("live-core", ContinuousCoreD512())
    with _host(tmp_path, port) as host:
        assert host.submit_user("the dog needs a walk").admitted
        beat = host.heartbeat()

        # the Core really participated in circulation
        assert beat.reasoning_result is not None
        assert beat.reasoning_result.consolidator_core_id == "live-core"

        # and it read the real field: its mirror matches the beat's field identity
        assert port.mirror.view is not None
        assert port.mirror.view.source_field_id == beat.field.field_id
        assert port.mirror.view.source_tick_id == beat.field.tick_id

        # THE REAL CONTRACT: the Core's mirror is byte-identical to the beat's field.
        # (The consolidator rotated the user's turn into conversation_history before
        # the canonical sync, so user_input is legitimately empty here.)
        for region in (
            LogicalRegion.CONVERSATION_HISTORY,
            LogicalRegion.RESPONSE_DRAFT,
            LogicalRegion.USER_INPUT,
        ):
            assert _region_text(port.mirror.view, region) == beat.field.region(region).text, (
                f"Core mirror diverged from the canonical field at {region.value}"
            )

        # the user's words survived the rotation, inside the turn frame
        history = _region_text(port.mirror.view, LogicalRegion.CONVERSATION_HISTORY)
        assert "the dog needs a walk" in history, "the user's turn must be readable by the Core"

        # the Core ingested real content and the recurrent state is live
        assert port.stats.cells_ingested > 0
        state = port.recurrent_state
        assert state is not None
        assert state.shape == (1, 512)
        assert float(state.norm().item()) > 0.0


def test_core_state_carries_across_consecutive_live_beats(tmp_path: Path) -> None:
    """Continuity: the same Core instance across beats, state carried by delta."""
    port = ContinuousD16CorePort("ongoing-core", ContinuousCoreD512())
    states = []
    with _host(tmp_path, port) as host:
        for text in ["first message", "second message", "third message"]:
            assert host.submit_user(text).admitted
            beat = host.heartbeat()
            assert beat.reasoning_result is not None
            assert beat.reasoning_result.consolidator_core_id == "ongoing-core"
            state = port.recurrent_state
            assert state is not None
            states.append(state.clone())

    # Three live beats ran; the Core's state moved each time.
    assert len(states) == 3
    for earlier, later in zip(states, states[1:]):
        assert not (earlier == later).all(), "state must advance between live beats"

    # The third turn's words are what the Core can now read.
    history = _region_text(port.mirror.view, LogicalRegion.CONVERSATION_HISTORY)
    assert "third message" in history

    audit = port.audit()
    assert audit["schema"] == "axon-continuous-d16-core-port-v1"
    assert audit["state_initialized"] is True
    assert port.stats.cells_ingested > 0
    # the Core lived through real turn rotation
    assert port.stats.snapshots_applied >= 1
