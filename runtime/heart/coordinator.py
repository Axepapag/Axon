"""The heart's beat coordinator: the first living circulation organ.

A ``BeatCoordinator`` owns one canonical state branch, a heart transaction
boundary, and a heartbeat clock.  Each beat drains the ingress queue between
ticks, runs primitive dormant recall when the canonical field changes, applies
per-region attention masks as a derived compile-time view, and freezes the
canonical field as a D64 tick image.  Build B stops at the tick image; Build E
will attach proposal, refinement, and consolidation barriers to the same frozen
image.

Attention masks are intentionally not part of the canonical identity.  The
branch stores and hashes only the ordered spans; changing a mask does not create
a new canonical body.  This preserves the one-body doctrine: there is exactly
one authoritative ``SharedFieldSnapshot``, and rails/tick images are derived
projections of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from runtime.field import (
    CanonicalStateBranch,
    D64FieldCompiler,
    FieldDelta,
    InsertText,
    LogicalRegion,
    RegionMaskPolicy,
    SharedFieldSnapshot,
    replacement_delta,
)
from runtime.dormant import DormantEvidenceBridge, DormantEvidenceIndex

from .authority import AuthorityGrant, IngressChannel, INGRESS_OWNED_REGIONS
from .errors import HeartTransactionError
from .ingress_queue import IngressItem, IngressQueue
from .registry import CoreRegistry
from .tick import FrozenTickImage, HeartbeatClock
from .transaction import HeartCommit, HeartTransactionBoundary


class BeatState(str, Enum):
    """The outcome of one heartbeat cycle."""

    IDLE = "idle"
    TICK_OPENED = "tick_opened"
    TICK_IN_FLIGHT = "tick_in_flight"


@dataclass(frozen=True, slots=True)
class BeatConfig:
    """Tunable knobs for the beat coordinator."""

    recall_limit: int = 8
    recall_min_confidence: float = 0.0
    recall_include_graph: bool = True
    max_query_length: int = 512
    region_policies: Mapping[LogicalRegion, RegionMaskPolicy] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.recall_limit, bool) or not isinstance(self.recall_limit, int):
            raise TypeError("BeatConfig.recall_limit must be an integer")
        if self.recall_limit < 0:
            raise ValueError("BeatConfig.recall_limit must be non-negative")
        if not 0.0 <= float(self.recall_min_confidence) <= 1.0:
            raise ValueError("BeatConfig.recall_min_confidence must be in [0, 1]")
        if not isinstance(self.recall_include_graph, bool):
            raise TypeError("BeatConfig.recall_include_graph must be a boolean")
        if isinstance(self.max_query_length, bool) or not isinstance(self.max_query_length, int):
            raise TypeError("BeatConfig.max_query_length must be an integer")
        if self.max_query_length < 0:
            raise ValueError("BeatConfig.max_query_length must be non-negative")
        policies = self.region_policies
        if policies is not None:
            normalized: dict[LogicalRegion, RegionMaskPolicy] = {}
            for key, value in policies.items():
                region = key if isinstance(key, LogicalRegion) else LogicalRegion(key)
                if not isinstance(value, RegionMaskPolicy):
                    value = RegionMaskPolicy(value["kind"], value.get("limit", 0))
                normalized[region] = value
            object.__setattr__(self, "region_policies", normalized)


@dataclass(frozen=True, slots=True)
class BeatResult:
    """Immutable receipt for one heartbeat cycle."""

    state: BeatState
    field: SharedFieldSnapshot
    tick_image: FrozenTickImage | None
    commits: tuple[HeartCommit, ...]


class BeatCoordinator:
    """Heart-owned ingress/change/recall/freeze coordinator."""

    def __init__(
        self,
        branch: CanonicalStateBranch,
        registry: CoreRegistry,
        *,
        state_root: Path | str | None = None,
        config: BeatConfig | None = None,
        compiler: D64FieldCompiler | None = None,
    ) -> None:
        if not isinstance(branch, CanonicalStateBranch):
            raise TypeError("BeatCoordinator requires a CanonicalStateBranch")
        if not isinstance(registry, CoreRegistry):
            raise TypeError("BeatCoordinator requires a CoreRegistry")
        self._branch = branch
        self._registry = registry
        self._state_root = (
            None if state_root is None else Path(state_root).resolve()
        )
        self._config = config if config is not None else BeatConfig()
        self._compiler = compiler if compiler is not None else D64FieldCompiler()
        self._clock = HeartbeatClock()
        self._boundary = HeartTransactionBoundary()
        self._queue = IngressQueue()
        self._current_field = self._load_field()
        self._last_field_id: str | None = None
        self._open_tick_image: FrozenTickImage | None = None
        self._bridge: DormantEvidenceBridge | None = None

    @property
    def tick_in_flight(self) -> bool:
        return self._open_tick_image is not None

    @property
    def current_field(self) -> SharedFieldSnapshot:
        return self._current_field

    def _load_field(self) -> SharedFieldSnapshot:
        if self._branch.initialized:
            return self._branch.load_head()
        empty = SharedFieldSnapshot.empty(tick_id=0)
        self._branch.initialize(empty)
        return empty

    def _sync_to_branch_head(self) -> None:
        """Reload the canonical field from the branch after any durable commit."""

        self._current_field = self._branch.load_head()

    def enqueue(
        self,
        channel: IngressChannel,
        text: str,
        *,
        provenance: str = "",
    ) -> IngressItem:
        """Queue an external arrival.  Items drain only between ticks."""

        return self._queue.enqueue(channel, text, provenance=provenance)

    def _region_masks(self) -> dict[LogicalRegion, RegionMaskPolicy]:
        """Return the derived attention masks supplied by configuration."""

        policies = self._config.region_policies
        return dict(policies) if policies else {}

    def _attended_text(
        self,
        field: SharedFieldSnapshot,
        region: LogicalRegion,
    ) -> str:
        """Return the attended substring for ``region`` using derived masks."""

        region_state = field.region(region)
        override = self._region_masks().get(region)
        if override is not None:
            return region_state.with_policy(override).attended_text
        return region_state.attended_text

    def _append_delta(
        self,
        base: SharedFieldSnapshot,
        region: LogicalRegion,
        item: IngressItem,
    ) -> FieldDelta:
        """Build an append delta for one ingress item into its runtime-owned region."""

        region_state = base.region(region)
        offset = len(region_state.text)
        return FieldDelta(
            base_field_id=base.field_id,
            base_tick_id=base.tick_id,
            author_core_id=f"ingress:{item.channel.value}",
            pass_id="intake",
            operations=(
                InsertText(
                    region=region,
                    offset=offset,
                    text=item.text,
                    provenance=item.provenance or f"ingress:{item.channel.value}",
                ),
            ),
        )

    def _commit_to_branch(self, commit: HeartCommit) -> SharedFieldSnapshot:
        """Persist one heart commit through the canonical state branch.

        The branch is the durable canonical authority.  After persisting, the
        coordinator's in-memory view is synchronized to the branch head so that
        a later failure cannot leave coordinator memory behind durable state.
        """

        persisted = self._branch.commit(
            commit.delta,
            permitted_regions=commit.grant.governed_regions,
        )
        self._current_field = persisted
        return persisted

    def _drain_and_commit_ingress(
        self,
        field: SharedFieldSnapshot,
    ) -> tuple[SharedFieldSnapshot, tuple[HeartCommit, ...]]:
        """Commit queued arrivals one at a time, acknowledging only on success.

        If one item fails, the queue retains that item and all successors.
        """

        commits: list[HeartCommit] = []
        current = field
        while not self._queue.is_empty:
            item = self._queue.peek(1)[0]
            owned = INGRESS_OWNED_REGIONS[item.channel]
            region = next(iter(owned))
            delta = self._append_delta(current, region, item)
            commit = self._boundary.commit(
                current,
                delta,
                AuthorityGrant.ingress(item.channel),
            )
            self._commit_to_branch(commit)
            self._queue.acknowledge(1)
            commits.append(commit)
            current = commit.successor
        return current, tuple(commits)

    def _extract_recall_query(self, field: SharedFieldSnapshot) -> str:
        """Form a query from the attended ingress regions."""

        parts: list[str] = []
        for region in (
            LogicalRegion.USER_INPUT,
            LogicalRegion.TOOL_RESULTS,
            LogicalRegion.ADVISOR_INPUT,
        ):
            text = self._attended_text(field, region).strip()
            if text:
                parts.append(text)
        query = " ".join(parts)
        if self._config.max_query_length > 0:
            query = query[: self._config.max_query_length]
        return query.strip()

    def _ensure_bridge(self) -> DormantEvidenceBridge:
        """Lazily open the P0 dormant evidence bridge from State/dormant."""

        if self._bridge is None:
            if self._state_root is None:
                raise RuntimeError(
                    "BeatCoordinator has no state_root; cannot open dormant evidence bridge"
                )
            index = DormantEvidenceIndex.open(self._state_root)
            self._bridge = DormantEvidenceBridge(index)
        return self._bridge

    def _run_recall(
        self,
        field: SharedFieldSnapshot,
    ) -> tuple[SharedFieldSnapshot, HeartCommit | None]:
        """Run primitive dormant recall and commit surfaced structured_knowledge."""

        if self._state_root is None:
            return field, None

        query = self._extract_recall_query(field)
        if not query:
            return field, None

        bridge = self._ensure_bridge()
        evidence = bridge.index.retrieve(
            query,
            limit=self._config.recall_limit,
            min_confidence=self._config.recall_min_confidence,
            include_graph=self._config.recall_include_graph,
        )
        if not evidence:
            return field, None

        surfaced = bridge.surface(field, evidence, replace_existing=True)
        new_sk = surfaced.region(LogicalRegion.STRUCTURED_KNOWLEDGE)
        current_sk = field.region(LogicalRegion.STRUCTURED_KNOWLEDGE)
        if new_sk.text == current_sk.text:
            return field, None

        compiled = self._compiler.compile(field, region_masks=self._region_masks())
        delta = replacement_delta(
            field,
            compiled,
            region=LogicalRegion.STRUCTURED_KNOWLEDGE,
            text=new_sk.text,
            author_core_id="dormant-valve",
            pass_id="recall",
            provenance=f"dormant_valve:{bridge.index.index_id}",
        )
        commit = self._boundary.commit(
            field,
            delta,
            AuthorityGrant.dormant_valve(),
        )
        self._commit_to_branch(commit)
        return commit.successor, commit

    def _verify_masked_roundtrip(
        self,
        compiled: Any,
        field: SharedFieldSnapshot,
    ) -> None:
        """Verify the compiled rail matches the derived attended text exactly."""

        for region in compiled.active_texts():
            expected = self._attended_text(field, LogicalRegion(region))
            observed = compiled.region_text(region)
            if observed != expected:
                from runtime.field import IncompleteRailError

                raise IncompleteRailError(
                    f"D64 roundtrip mismatch in {region!r}: "
                    f"expected {len(expected)} chars, observed {len(observed)}"
                )

    def _open_tick(self, field: SharedFieldSnapshot) -> FrozenTickImage:
        """Compile the field with derived masks and freeze one 64D tick image."""

        identity = self._clock.open_tick(field)
        compiled = self._compiler.compile(field, region_masks=self._region_masks())
        self._verify_masked_roundtrip(compiled, field)
        image = FrozenTickImage.from_compiled(identity, {64: compiled})
        self._boundary.note_tick_opened(image)
        return image

    def beat(self, *, force: bool = False) -> BeatResult:
        """Execute one heartbeat cycle.

        If a tick is already in flight, the beat returns immediately without
        mutating the frozen base.  Otherwise it drains ingress, recalls,
        stabilizes, and opens a new tick against the one canonical body.

        On any failure after a durable commit, the coordinator resynchronizes
        to the branch head before re-raising so that coordinator memory and
        durable canonical state cannot diverge.
        """

        if self._open_tick_image is not None:
            return BeatResult(
                state=BeatState.TICK_IN_FLIGHT,
                field=self._current_field,
                tick_image=self._open_tick_image,
                commits=(),
            )

        commits: list[HeartCommit] = []
        field = self._current_field

        try:
            # 1. Between-tick ingress commits.
            field, ingress_commits = self._drain_and_commit_ingress(field)
            commits.extend(ingress_commits)

            # 2. Detect canonical change against the last stabilized field.
            changed = force or field.field_id != self._last_field_id

            # 3. Primitive dormant recall when the active field changed.
            if changed:
                field, recall_commit = self._run_recall(field)
                if recall_commit is not None:
                    commits.append(recall_commit)

            # 4. Stabilize.
            self._last_field_id = field.field_id
            self._current_field = field

            # 5. Open a tick if there is any work to freeze.
            if not force and not commits and not changed:
                return BeatResult(
                    state=BeatState.IDLE,
                    field=field,
                    tick_image=None,
                    commits=(),
                )

            image = self._open_tick(field)
            self._open_tick_image = image
            return BeatResult(
                state=BeatState.TICK_OPENED,
                field=field,
                tick_image=image,
                commits=tuple(commits),
            )
        except Exception:
            # A durable commit may have succeeded before the failure.  Resync
            # so the next beat starts from the authoritative branch head.
            self._sync_to_branch_head()
            raise

    def close_tick(self) -> None:
        """Close the in-flight tick without a consolidator commit.

        Build B uses this test/production boundary because reasoning cores and
        consolidation do not yet exist.  Build E will close the tick via the
        consolidator's heart commit.
        """

        if self._open_tick_image is None:
            raise HeartTransactionError("no tick is in flight")
        self._boundary.note_tick_closed(self._open_tick_image.identity)
        self._open_tick_image = None


__all__ = [
    "BeatConfig",
    "BeatCoordinator",
    "BeatResult",
    "BeatState",
]
