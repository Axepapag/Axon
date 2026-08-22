from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from runtime.field import (
    CanonicalStateBranch,
    D64FieldCompiler,
    LogicalRegion,
    RegionMaskPolicy,
    RegionState,
    SharedFieldSnapshot,
)
from runtime.heart import (
    BeatConfig,
    BeatCoordinator,
    BeatState,
    CoreDescriptor,
    CoreRegistry,
    HeartTransactionError,
    IngressChannel,
    IngressItem,
)

D64 = 64


def _registry() -> CoreRegistry:
    return CoreRegistry(
        (
            CoreDescriptor(core_id="core-alpha", d_model=D64),
            CoreDescriptor(core_id="core-beta", d_model=D64),
        )
    )


def _branch(tmp_path: Path, branch_id: str = "test") -> CanonicalStateBranch:
    return CanonicalStateBranch(
        tmp_path / branch_id,
        branch_id=branch_id,
        authority_root=tmp_path,
    )


class _FakeDormantBridge:
    """A deterministic bridge that surfaces one fact when queried."""

    def __init__(self, fact: str) -> None:
        self.fact = fact
        self.index_id = "fake-index"
        self.index = self
        self.queries: list[str] = []

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 8,
        min_confidence: float = 0.0,
        include_graph: bool = True,
    ) -> tuple[Any, ...]:
        from types import SimpleNamespace

        self.queries.append(query)
        if not query:
            return ()
        container = SimpleNamespace(
            container_id="container-1",
            text=self.fact,
            normalized_text=self.fact,
            kind="fact",
            source="test",
            provenance="test",
            confidence=1.0,
            status="dormant",
        )
        evidence = SimpleNamespace(
            candidate=SimpleNamespace(container_id="container-1", score=1.0),
            container=container,
            edges=(),
        )
        return (evidence,)

    def surface(
        self,
        base: SharedFieldSnapshot,
        evidence: tuple[Any, ...],
        *,
        replace_existing: bool = True,
    ) -> SharedFieldSnapshot:
        from runtime.field import FieldSpan, RegionState, RegionVisibility

        spans = (
            FieldSpan(
                span_id="dormant:fact:1",
                text=self.fact,
                kind="dormant_fact",
                source="test",
                provenance="test",
            ),
        )
        return SharedFieldSnapshot(
            tick_id=base.tick_id + 1,
            parent_field_id=base.field_id,
            source_manifest_ids=base.source_manifest_ids,
            regions=tuple(
                RegionState(
                    name=LogicalRegion.STRUCTURED_KNOWLEDGE,
                    spans=spans,
                    visibility=RegionVisibility.ATTENDED,
                )
                if region.name is LogicalRegion.STRUCTURED_KNOWLEDGE
                else region
                for region in base.regions
            ),
        )


def test_coordinator_initializes_empty_branch_and_freezes_first_field(
    tmp_path: Path,
) -> None:
    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(branch, _registry())
    assert coordinator.current_field.tick_id == 0
    result = coordinator.beat()
    assert result.state is BeatState.TICK_OPENED
    assert result.tick_image is not None
    assert result.commits == ()
    coordinator.close_tick()

    idle = coordinator.beat()
    assert idle.state is BeatState.IDLE
    assert idle.tick_image is None
    assert idle.commits == ()


def test_coordinator_drains_user_ingress_and_opens_tick(tmp_path: Path) -> None:
    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(branch, _registry())

    item = coordinator.enqueue(IngressChannel.USER, "hello")
    assert item.channel is IngressChannel.USER
    assert item.text == "hello"

    result = coordinator.beat()
    assert result.state is BeatState.TICK_OPENED
    assert result.tick_image is not None
    assert len(result.commits) == 1
    assert coordinator.tick_in_flight

    field = result.field
    assert field.region(LogicalRegion.USER_INPUT).text == "hello"
    assert field.tick_id == 1
    assert result.tick_image.base_field_id == field.field_id
    assert result.tick_image.identity.base_tick_id == field.tick_id


def test_coordinator_queues_ingress_while_tick_is_in_flight(tmp_path: Path) -> None:
    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(branch, _registry())
    coordinator.enqueue(IngressChannel.USER, "first")
    coordinator.beat()

    coordinator.enqueue(IngressChannel.USER, "second")
    in_flight = coordinator.beat()
    assert in_flight.state is BeatState.TICK_IN_FLIGHT
    assert in_flight.tick_image is not None

    coordinator.close_tick()
    assert not coordinator.tick_in_flight

    result = coordinator.beat()
    assert result.state is BeatState.TICK_OPENED
    assert result.field.region(LogicalRegion.USER_INPUT).text == "firstsecond"


def test_coordinator_cannot_close_tick_when_none_is_open(tmp_path: Path) -> None:
    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(branch, _registry())
    with pytest.raises(HeartTransactionError, match="no tick is in flight"):
        coordinator.close_tick()


def test_coordinator_routes_tool_and_advisor_to_owned_regions(tmp_path: Path) -> None:
    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(branch, _registry())
    coordinator.enqueue(IngressChannel.TOOL, "tool output")
    coordinator.enqueue(IngressChannel.ADVISOR, "advisor note")

    result = coordinator.beat()
    assert result.state is BeatState.TICK_OPENED
    assert result.field.region(LogicalRegion.TOOL_RESULTS).text == "tool output"
    assert result.field.region(LogicalRegion.ADVISOR_INPUT).text == "advisor note"
    assert len(result.commits) == 2


def test_coordinator_change_detection_and_force_opens_tick(tmp_path: Path) -> None:
    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(branch, _registry())
    coordinator.enqueue(IngressChannel.USER, "a")
    first = coordinator.beat()
    assert first.state is BeatState.TICK_OPENED
    coordinator.close_tick()

    idle = coordinator.beat()
    assert idle.state is BeatState.IDLE

    forced = coordinator.beat(force=True)
    assert forced.state is BeatState.TICK_OPENED
    assert forced.tick_image is not None


def test_coordinator_applies_per_region_attention_policies(tmp_path: Path) -> None:
    from runtime.field import FieldSpan

    # Seed the branch with two conversation spans.  Attention masks are a
    # derived view, not part of the canonical identity, so the policy lives in
    # BeatConfig and is applied at compile/recall time.
    history = RegionState(
        name=LogicalRegion.CONVERSATION_HISTORY,
        spans=(
            FieldSpan(span_id="h1", text="old line 1\n"),
            FieldSpan(span_id="h2", text="old line 2\n"),
        ),
    )
    branch = _branch(tmp_path)
    branch.initialize(SharedFieldSnapshot(tick_id=0, regions=(history,)))

    config = BeatConfig(
        region_policies={
            LogicalRegion.CONVERSATION_HISTORY: RegionMaskPolicy("last_n_spans", 1),
        }
    )
    coordinator = BeatCoordinator(branch, _registry(), config=config)
    coordinator.enqueue(IngressChannel.USER, "question", provenance="test")

    result = coordinator.beat()
    assert result.state is BeatState.TICK_OPENED

    # The canonical field preserves the full text.
    ch = result.field.region(LogicalRegion.CONVERSATION_HISTORY)
    assert ch.text == "old line 1\nold line 2\n"
    assert ch.attended_text == "old line 1\nold line 2\n"

    # The tick rail applies the derived last-n-spans mask.
    compiled = D64FieldCompiler().compile(
        result.field, region_masks=config.region_policies
    )
    assert compiled.region_text("conversation_history") == "old line 2\n"


def test_coordinator_runs_recall_when_field_changes(tmp_path: Path) -> None:
    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(
        branch,
        _registry(),
        state_root=tmp_path,
    )
    coordinator._bridge = _FakeDormantBridge("Axon is a field compiler organ.")
    coordinator.enqueue(IngressChannel.USER, "What is Axon?")

    result = coordinator.beat()
    assert result.state is BeatState.TICK_OPENED
    assert result.field.region(LogicalRegion.STRUCTURED_KNOWLEDGE).text == "Axon is a field compiler organ."
    assert any(commit.delta.author_core_id == "dormant-valve" for commit in result.commits)


def test_coordinator_skips_recall_when_query_is_empty(tmp_path: Path) -> None:
    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(
        branch,
        _registry(),
        state_root=tmp_path,
    )
    coordinator._bridge = _FakeDormantBridge("ignored")
    # Force a tick with no attended ingress text.
    result = coordinator.beat(force=True)
    assert result.state is BeatState.TICK_OPENED
    assert result.field.region(LogicalRegion.STRUCTURED_KNOWLEDGE).text == ""
    assert not any(
        commit.delta.author_core_id == "dormant-valve" for commit in result.commits
    )


def test_coordinator_recall_is_idempotent_when_knowledge_unchanged(
    tmp_path: Path,
) -> None:
    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(
        branch,
        _registry(),
        state_root=tmp_path,
    )
    coordinator._bridge = _FakeDormantBridge("Axon is a field compiler organ.")
    coordinator.enqueue(IngressChannel.USER, "What is Axon?")
    first = coordinator.beat()
    assert first.state is BeatState.TICK_OPENED
    coordinator.close_tick()

    # The same field with the same surfaced knowledge should not create a
    # duplicate recall commit.
    second = coordinator.beat()
    assert second.state is BeatState.IDLE


def test_attention_masks_are_derived_not_canonical(tmp_path: Path) -> None:
    branch = _branch(tmp_path)
    coordinator1 = BeatCoordinator(
        branch,
        _registry(),
        config=BeatConfig(
            region_policies={
                LogicalRegion.USER_INPUT: RegionMaskPolicy("last_n_spans", 1),
            }
        ),
    )
    field_id_with_mask = coordinator1.current_field.field_id

    coordinator2 = BeatCoordinator(
        branch,
        _registry(),
        config=BeatConfig(
            region_policies={
                LogicalRegion.USER_INPUT: RegionMaskPolicy("none"),
            }
        ),
    )
    assert coordinator2.current_field.field_id == field_id_with_mask


def test_failed_ingress_commit_preserves_later_arrivals(tmp_path: Path) -> None:
    from datetime import datetime, timezone

    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(branch, _registry())
    # Inject an empty item at the front that will fail delta construction.
    coordinator._queue._items.insert(
        0,
        IngressItem(
            channel=IngressChannel.USER,
            text="",
            provenance="test",
            enqueued_at=datetime.now(timezone.utc),
        ),
    )
    coordinator.enqueue(IngressChannel.USER, "valid")

    with pytest.raises(Exception):
        coordinator.beat()

    # The failed item and the unprocessed successor must both remain.
    texts = {item.text for item in coordinator._queue}
    assert texts == {"", "valid"}


def test_coordinator_resynchronizes_to_branch_head_after_recall_failure(
    tmp_path: Path,
) -> None:
    branch = _branch(tmp_path)
    coordinator = BeatCoordinator(
        branch,
        _registry(),
        state_root=tmp_path,
    )
    coordinator.enqueue(IngressChannel.USER, "hello")

    class _FailingBridge(_FakeDormantBridge):
        def retrieve(self, query, **kwargs):
            if query:
                raise RuntimeError("bridge failure")
            return ()

    coordinator._bridge = _FailingBridge("ignored")
    with pytest.raises(RuntimeError, match="bridge failure"):
        coordinator.beat()

    # The ingress commit was durable; coordinator memory must reflect it.
    assert coordinator.current_field.region(LogicalRegion.USER_INPUT).text == "hello"
    assert branch.load_head().region(LogicalRegion.USER_INPUT).text == "hello"

    # A new coordinator loading the same branch must not see a stale base.
    coordinator2 = BeatCoordinator(branch, _registry(), state_root=tmp_path)
    assert coordinator2.current_field.region(LogicalRegion.USER_INPUT).text == "hello"
