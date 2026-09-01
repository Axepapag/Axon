from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from runtime.field import (
    CanonicalStateBranch,
    FieldSpan,
    LogicalRegion,
    RegionState,
    RegionVisibility,
    SharedFieldSnapshot,
)
from runtime.heart import BeatConfig, BeatCoordinator, BeatState, CoreRegistry, IngressChannel


class _OversizeBridge:
    def __init__(self) -> None:
        self.index_id = "oversize-fixture"
        self.index = self

    def retrieve(self, query: str, **kwargs):
        candidate = SimpleNamespace(
            container_id="too-big",
            score=50.0,
            lexical_hits=1,
            graph_hits=0,
            edge_ids=(),
        )
        container = SimpleNamespace(
            container_id="too-big",
            text="x" * 10,
            normalized_text="x" * 10,
            kind="fact",
            source="fixture",
            provenance="fixture",
            confidence=1.0,
            status="dormant",
            record={},
        )
        return (SimpleNamespace(candidate=candidate, container=container, edges=()),)

    def surface(self, base, evidence, *, replace_existing=True):
        assert tuple(item.container.text for item in evidence) == ("x" * 10,)
        cortex = RegionState(
            name=LogicalRegion.CORTEX,
            spans=(
                FieldSpan(
                    span_id="fixture:too-big",
                    text="x" * 10,
                    source="fixture",
                    provenance="fixture",
                    container_refs=("too-big",),
                ),
            ),
            visibility=RegionVisibility.ATTENDED,
        )
        return SharedFieldSnapshot(
            tick_id=base.tick_id + 1,
            parent_field_id=base.field_id,
            regions=tuple(
                cortex if region.name is LogicalRegion.CORTEX else region
                for region in base.regions
            ),
        )


def test_coordinator_enforces_full_item_dormant_budget_without_truncation(tmp_path: Path) -> None:
    branch = CanonicalStateBranch(
        tmp_path / "branch",
        branch_id="budget",
        authority_root=tmp_path,
    )
    coordinator = BeatCoordinator(
        branch,
        CoreRegistry(),
        state_root=tmp_path,
        config=BeatConfig(
            recall_items_per_materialization=2,
            recall_target_chars_per_materialization=5,
        ),
    )
    coordinator._bridge = _OversizeBridge()
    coordinator.enqueue(IngressChannel.USER, "x")
    result = coordinator.beat()
    assert result.state is BeatState.TICK_OPENED
    assert result.field.region(LogicalRegion.USER_INPUT).text == "x"
    assert result.field.region(LogicalRegion.CORTEX).text == "x" * 10
    assert len(result.commits) == 2
