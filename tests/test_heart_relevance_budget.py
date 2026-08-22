from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from runtime.field import CanonicalStateBranch, LogicalRegion
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

    def surface(self, base, evidence, *, replace_existing=True):  # pragma: no cover - must not run
        raise AssertionError("oversize evidence must not reach surfacing")


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
            recall_limit=2,
            recall_max_chars=20,
            recall_max_item_chars=5,
        ),
    )
    coordinator._bridge = _OversizeBridge()
    coordinator.enqueue(IngressChannel.USER, "x")
    result = coordinator.beat()
    assert result.state is BeatState.TICK_OPENED
    assert result.field.region(LogicalRegion.USER_INPUT).text == "x"
    assert result.field.region(LogicalRegion.STRUCTURED_KNOWLEDGE).text == ""
    assert len(result.commits) == 1
