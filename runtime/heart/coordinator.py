"""The heart's beat coordinator: the first living circulation organ.

A ``BeatCoordinator`` owns one canonical state branch, a heart transaction
boundary, and a heartbeat clock.  Each beat drains the ingress queue between
ticks, runs primitive dormant recall when the canonical field changes, applies
per-region attention masks as a derived compile-time view, and freezes the
canonical field as a D64 tick image.  The reasoning circulation attaches its
proposal, refinement, and consolidation barriers to that same frozen image and
returns only through the consolidator transaction path.

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

from runtime.dormant import (
    DormantEvidenceBridge,
    DormantEvidenceGenerationStore,
    DormantRelevanceAuditor,
    DormantRelevancePolicy,
)
from runtime.field import (
    SCHEMA_VERSION,
    CanonicalStateBranch,
    CompiledD64DualSurface,
    D64FieldCompiler,
    D64SemanticSurfaceCompiler,
    FieldDelta,
    InsertText,
    LogicalRegion,
    RegionMaskPolicy,
    SharedFieldSnapshot,
    canonical_sha256,
    replacement_delta,
)

from .authority import INGRESS_OWNED_REGIONS, AuthorityGrant, IngressChannel
from .errors import HeartTransactionError
from .ingress_queue import IngressItem, IngressQueue
from .masks import HeartRegionMaskController
from .registry import CoreRegistry
from .tick import FrozenTickImage, HeartbeatClock, TickIdentity, derive_view_id
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
    recall_candidate_multiplier: int = 4
    recall_max_chars: int = 10_000
    recall_max_item_chars: int = 4_096
    recall_min_relevance: float = 0.0
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
        if (
            isinstance(self.recall_candidate_multiplier, bool)
            or not isinstance(self.recall_candidate_multiplier, int)
            or self.recall_candidate_multiplier < 1
        ):
            raise ValueError("BeatConfig.recall_candidate_multiplier must be a positive integer")
        for name in ("recall_max_chars", "recall_max_item_chars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"BeatConfig.{name} must be a positive integer")
        if not 0.0 <= float(self.recall_min_relevance) <= 1.0:
            raise ValueError("BeatConfig.recall_min_relevance must be in [0, 1]")
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
        semantic_compiler: D64SemanticSurfaceCompiler | None = None,
        mask_controller: HeartRegionMaskController | None = None,
    ) -> None:
        if not isinstance(branch, CanonicalStateBranch):
            raise TypeError("BeatCoordinator requires a CanonicalStateBranch")
        if not isinstance(registry, CoreRegistry):
            raise TypeError("BeatCoordinator requires a CoreRegistry")
        self._branch = branch
        self._registry = registry
        self._state_root = None if state_root is None else Path(state_root).resolve()
        self._config = config if config is not None else BeatConfig()
        self._compiler = compiler if compiler is not None else D64FieldCompiler()
        self._semantic_compiler = semantic_compiler if semantic_compiler is not None else D64SemanticSurfaceCompiler()
        if mask_controller is not None and not isinstance(mask_controller, HeartRegionMaskController):
            raise TypeError("mask_controller must be a HeartRegionMaskController")
        self._mask_controller = mask_controller
        self._clock = HeartbeatClock()
        self._boundary = HeartTransactionBoundary()
        self._queue = IngressQueue()
        self._current_field = self._load_field()
        self._last_field_id: str | None = None
        self._last_view_id: str | None = None
        self._open_tick_image: FrozenTickImage | None = None
        self._open_dual_surface: CompiledD64DualSurface | None = None
        self._bridge: DormantEvidenceBridge | None = None
        self._dormant_generations: DormantEvidenceGenerationStore | None = None
        self._bridge_generation_token: str | None = None

    @property
    def tick_in_flight(self) -> bool:
        return self._open_tick_image is not None

    @property
    def current_field(self) -> SharedFieldSnapshot:
        return self._current_field

    @property
    def open_dual_surface(self) -> CompiledD64DualSurface | None:
        """Current noncanonical exact+semantic D64 projection, if a tick is open."""

        return self._open_dual_surface

    @property
    def open_tick_image(self) -> FrozenTickImage | None:
        """Current immutable tick image, if a reasoning tick is in flight."""

        return self._open_tick_image

    @property
    def boundary(self) -> HeartTransactionBoundary:
        """Return the one Heart transaction boundary owned by this coordinator."""

        return self._boundary

    @property
    def branch(self) -> CanonicalStateBranch:
        """Return the one canonical branch owned by this coordinator."""

        return self._branch

    @property
    def dormant_index_id(self) -> str | None:
        """Identity of the opened dormant index, if recall has touched it."""

        return None if self._bridge is None else self._bridge.index.index_id

    @property
    def current_view_id(self) -> str:
        """Identity of the complete mask policy set used for the next freeze."""

        return derive_view_id(self._region_masks())

    @property
    def mask_view_changed(self) -> bool:
        """Whether mask control moved since the last successfully frozen view."""

        return self.current_view_id != self._last_view_id

    def sync_to_branch_head(self) -> SharedFieldSnapshot:
        """Synchronize coordinator memory to durable canonical HEAD."""

        self._sync_to_branch_head()
        return self._current_field

    def _load_field(self) -> SharedFieldSnapshot:
        if self._branch.initialized:
            field = self._branch.load_head()
            if field.schema_version != SCHEMA_VERSION:
                return self._branch.migrate_to_current_schema()
            return field
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
        """Return one detached complete mask set for this operation."""

        if self._mask_controller is not None:
            return self._mask_controller.policies()
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

    def _commit_to_branch(
        self,
        commit: HeartCommit,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SharedFieldSnapshot:
        """Persist one heart commit through the canonical state branch.

        The branch is the durable canonical authority.  After persisting, the
        coordinator's in-memory view is synchronized to the branch head so that
        a later failure cannot leave coordinator memory behind durable state.
        """

        branch_metadata: dict[str, Any] = {"heart_commit": commit.to_canonical_dict()}
        if metadata:
            branch_metadata.update(dict(metadata))
        persisted = self._branch.commit(
            commit.delta,
            permitted_regions=commit.grant.governed_regions,
            metadata=branch_metadata,
        )
        self._current_field = persisted
        return persisted

    def commit_heart_delta(
        self,
        base: SharedFieldSnapshot,
        delta: FieldDelta,
        grant: AuthorityGrant,
        *,
        valve_provenance: Mapping[str, Any] | None = None,
    ) -> HeartCommit:
        """Validate through the one Heart boundary and persist through its branch."""

        commit = self._boundary.commit(
            base,
            delta,
            grant,
            valve_provenance=valve_provenance,
        )
        persisted = self._commit_to_branch(commit)
        if persisted.field_id != commit.successor.field_id:
            raise HeartTransactionError("canonical branch HEAD disagrees with the Heart commit successor")
        return commit

    def rail_surface(self, d_model: int) -> CompiledD64DualSurface:
        """Return the physical surface for one rail on the in-flight tick.

        The execution interface is width-generic, but D64 is intentionally the
        only compiled physical rail today.  Wider labels fail closed until a
        real compiler and surface type exist.
        """

        if d_model != 64:
            raise HeartTransactionError(
                f"no physical runtime surface is registered for d_model {d_model}"
            )
        if self._open_tick_image is None or self._open_dual_surface is None:
            raise HeartTransactionError("no physical rail surface exists without an in-flight tick")
        self._open_tick_image.require_rail(d_model)
        return self._open_dual_surface

    def commit_consolidator_delta(
        self,
        delta: FieldDelta,
        *,
        tick: TickIdentity,
        metadata: Mapping[str, Any] | None = None,
    ) -> HeartCommit:
        """Commit exactly one final reasoning decision and consume its tick."""

        image = self._open_tick_image
        if image is None:
            raise HeartTransactionError("no reasoning tick is in flight")
        if tick != image.identity:
            raise HeartTransactionError("consolidator tick differs from the frozen tick image")
        base = self._current_field
        if base.field_id != tick.base_field_id or base.tick_id != tick.base_tick_id:
            raise HeartTransactionError("canonical field moved after the reasoning tick froze")
        circulation_metadata = {} if metadata is None else dict(metadata)
        commit = self._boundary.commit(
            base,
            delta,
            AuthorityGrant.consolidator(),
            tick=tick,
            valve_provenance={
                "authority_class": "consolidator",
                "tick_uid": tick.tick_uid,
                **circulation_metadata,
            },
        )
        try:
            persisted = self._commit_to_branch(
                commit,
                metadata={"reasoning_circulation": circulation_metadata},
            )
        except Exception:
            # The transaction boundary has consumed the tick.  Clear its
            # derived surfaces and reload durable truth so recovery can open a
            # fresh tick instead of reusing an ambiguous final decision.
            self._open_tick_image = None
            self._open_dual_surface = None
            self._sync_to_branch_head()
            raise
        if persisted.field_id != commit.successor.field_id:
            raise HeartTransactionError("canonical branch HEAD disagrees with the consolidator successor")
        self._open_tick_image = None
        self._open_dual_surface = None
        return commit

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
        """Open the active verified dormant-index generation, swapping atomically."""

        if self._bridge is not None and not isinstance(self._bridge, DormantEvidenceBridge):
            # Existing deterministic tests inject a bridge double through the
            # private slot. Keep that explicit seam without weakening production
            # generation resolution for real DormantEvidenceBridge instances.
            return self._bridge
        if self._state_root is None:
            raise RuntimeError("BeatCoordinator has no state_root; cannot open dormant evidence bridge")
        if self._dormant_generations is None:
            self._dormant_generations = DormantEvidenceGenerationStore(self._state_root)
        generation_token = self._dormant_generations.active_token()
        if self._bridge is not None and generation_token == self._bridge_generation_token:
            return self._bridge

        # Open and verify the new generation before releasing the previous one.
        # A bad/corrupt candidate therefore cannot evict a healthy reader.
        new_index = self._dormant_generations.open_active(verify_binding=True)
        previous = self._bridge
        self._bridge = DormantEvidenceBridge(new_index)
        self._bridge_generation_token = generation_token
        if previous is not None:
            previous.index.close()
        return self._bridge

    def _run_recall(
        self,
        field: SharedFieldSnapshot,
    ) -> tuple[SharedFieldSnapshot, HeartCommit | None]:
        """Run primitive dormant recall and commit surfaced cortex."""

        if self._state_root is None or self._config.recall_limit <= 0:
            return field, None

        query = self._extract_recall_query(field)
        if not query:
            return field, None

        bridge = self._ensure_bridge()
        candidate_limit = min(
            128,
            max(
                self._config.recall_limit,
                self._config.recall_limit * self._config.recall_candidate_multiplier,
            ),
        )
        evidence = bridge.index.retrieve(
            query,
            limit=candidate_limit,
            min_confidence=self._config.recall_min_confidence,
            include_graph=self._config.recall_include_graph,
        )
        if not evidence:
            return field, None

        auditor = DormantRelevanceAuditor(
            DormantRelevancePolicy(
                max_items=self._config.recall_limit,
                max_chars=self._config.recall_max_chars,
                max_item_chars=self._config.recall_max_item_chars,
                min_score=self._config.recall_min_relevance,
            )
        )
        relevance = auditor.select(query, evidence, field)
        if not relevance.selected:
            return field, None

        container_refs = tuple(sorted(relevance.selected_container_ids))
        edge_refs = tuple(
            sorted(
                {
                    str(edge.edge_id)
                    for item in relevance.selected
                    for edge in item.edges
                    if getattr(edge, "edge_id", None)
                }
            )
        )
        generation_token = self._bridge_generation_token or "unversioned"
        recall_provenance = (
            f"dormant_valve:{bridge.index.index_id}:generation:{generation_token}:relevance:{relevance.decision_id}"
        )

        surfaced = bridge.surface(field, relevance.selected, replace_existing=True)
        new_sk = surfaced.region(LogicalRegion.CORTEX)
        current_sk = field.region(LogicalRegion.CORTEX)
        if new_sk.text == current_sk.text:
            return field, None

        compiled = self._compiler.compile(field, region_masks=self._region_masks())
        delta = replacement_delta(
            field,
            compiled,
            region=LogicalRegion.CORTEX,
            text=new_sk.text,
            author_core_id="dormant-valve",
            pass_id="recall",
            evidence=(*container_refs, *edge_refs),
            provenance=recall_provenance,
            container_refs=container_refs,
            edge_refs=edge_refs,
        )
        recall_item_id = canonical_sha256(
            {
                "valve_id": "dormant_recall",
                "base_field_id": field.field_id,
                "query": query,
                "index_id": bridge.index.index_id,
                "generation_token": generation_token,
                "relevance_decision_id": relevance.decision_id,
                "selected_container_ids": list(relevance.selected_container_ids),
                "selected_edge_ids": list(edge_refs),
            }
        )
        commit = self._boundary.commit(
            field,
            delta,
            AuthorityGrant.dormant_valve(),
            valve_provenance={
                "valve_id": "dormant_recall",
                "valve_version": 1,
                "source_id": "dormant_valve",
                "item_id": recall_item_id,
                "provenance": recall_provenance,
                "index_id": bridge.index.index_id,
                "generation_token": generation_token,
                "relevance_decision_id": relevance.decision_id,
                "selected_container_ids": list(relevance.selected_container_ids),
                "selected_edge_ids": list(edge_refs),
                "fallback_used": relevance.fallback_used,
                "selected_chars": relevance.total_chars,
                "authority_class": "dormant_valve",
                "governed_regions": [LogicalRegion.CORTEX.value],
            },
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
                    f"D64 roundtrip mismatch in {region!r}: expected {len(expected)} chars, observed {len(observed)}"
                )

    def stabilize_recall(
        self,
        field: SharedFieldSnapshot,
    ) -> tuple[SharedFieldSnapshot, HeartCommit | None]:
        """Run the accepted primitive P0 recall through the Heart boundary."""

        return self._run_recall(field)

    def freeze_tick(
        self,
        field: SharedFieldSnapshot,
        identity: TickIdentity,
    ) -> FrozenTickImage:
        """Freeze an already-reserved tick identity as the exact D64 view."""

        if identity.base_field_id != field.field_id or identity.base_tick_id != field.tick_id:
            raise HeartTransactionError("reserved tick identity is stale for the field being frozen")
        masks = self._region_masks()
        compiled = self._compiler.compile(field, region_masks=masks)
        self._verify_masked_roundtrip(compiled, field)
        semantic = self._semantic_compiler.compile(field, compiled)
        dual_surface = CompiledD64DualSurface(exact=compiled, semantic=semantic)
        image = FrozenTickImage.from_compiled(
            identity,
            {64: compiled},
            view_id=derive_view_id(masks),
            semantic_surfaces={64: semantic},
        )
        self._boundary.note_tick_opened(image)
        self._open_tick_image = image
        self._open_dual_surface = dual_surface
        self._last_view_id = image.view_id
        return image

    def _open_tick(self, field: SharedFieldSnapshot) -> FrozenTickImage:
        """Compile the field with derived masks and freeze one 64D tick image."""

        identity = self._clock.open_tick(field)
        return self.freeze_tick(field, identity)

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

            # 2. Detect canonical and derived-view change independently.  A
            # mask move must circulate a new view over the same canonical body,
            # but it is not a reason to run canonical dormant recall.
            field_changed = field.field_id != self._last_field_id
            view_changed = self.mask_view_changed
            changed = force or field_changed or view_changed

            # 3. Primitive dormant recall when the active field changed.
            if force or field_changed:
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

        This remains the explicit null-tick/development boundary.  A reasoning
        tick ends through :meth:`commit_consolidator_delta`, not this method.
        """

        if self._open_tick_image is None:
            raise HeartTransactionError("no tick is in flight")
        self._boundary.note_tick_closed(self._open_tick_image.identity)
        self._open_tick_image = None
        self._open_dual_surface = None


__all__ = [
    "BeatConfig",
    "BeatCoordinator",
    "BeatResult",
    "BeatState",
]
