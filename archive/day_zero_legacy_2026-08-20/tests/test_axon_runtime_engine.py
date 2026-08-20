from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time

import pytest

from runtime.axon_runtime.contracts import (
    CoreIdentity,
    CoreStateManifest,
    StoreContractError,
)
from runtime.axon_runtime.engine import (
    AxonRuntimeEngine,
    CommittedStateRecoveryError,
    IngressSystemUpdateProvider,
    ObservedField,
    PreparedCoreCommit,
    RuntimeEngineError,
    StagedCoreTurn,
)
from runtime.axon_runtime.ingress import IngressEvent, IngressQueue
from runtime.axon_runtime.field_transaction import (
    AppendSpans,
    SystemFieldUpdate,
)
from runtime.axon_runtime.store import RuntimeStore
from runtime.field import (
    FieldSpan,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_sha256,
)
from runtime.multi_tick_refiner import RegionProposal


def _hash(*values: object) -> str:
    return canonical_sha256(list(values))


def _identities() -> tuple[CoreIdentity, ...]:
    values = []
    for index, width in enumerate((64, 128, 64, 128)):
        values.append(
            CoreIdentity(
                core_id=f"core-{index}",
                display_name=f"Core {index}",
                base_checkpoint_path=f"D:/Axon/checkpoints/core-{width}.pt",
                base_checkpoint_sha256=_hash("checkpoint", width),
                model_id=f"model-{width}",
                core_state_sha256=_hash("core-state", width),
                lineage=(f"exact-v4-{width}", "runtime-clone"),
                parent_core_id=None,
                soul_id=f"soul-{index}",
                adapter_namespace=f"adapter-{index}",
            )
        )
    return tuple(values)


def _initial_states(
    identities: tuple[CoreIdentity, ...],
    genesis: SharedFieldSnapshot,
) -> tuple[CoreStateManifest, ...]:
    return tuple(
        CoreStateManifest(
            core_id=identity.core_id,
            soul_id=identity.soul_id,
            model_id=identity.model_id,
            core_state_sha256=identity.core_state_sha256,
            generation=0,
            tick_seq=0,
            committed_field_id=genesis.field_id,
            soul_state_sha256=_hash(identity.core_id, "soul", 0),
            cursor_state_sha256=_hash(identity.core_id, "cursor", 0),
            adapter_set_sha256=_hash(identity.core_id, "adapters", 0),
            rng_state_sha256=_hash(identity.core_id, "rng", 0),
        )
        for identity in identities
    )


@dataclass
class _Private:
    generation: int
    field_id: str


class DeterministicDriver:
    def __init__(
        self,
        identities: tuple[CoreIdentity, ...],
        genesis: SharedFieldSnapshot,
        *,
        tool_text: str | None = None,
        corrupt_manifest: bool = False,
    ) -> None:
        self.private = {
            identity.core_id: _Private(0, genesis.field_id)
            for identity in identities
        }
        self.tool_text = tool_text
        self.corrupt_manifest = corrupt_manifest
        self.stage_counts: dict[int, int] = {}
        self.finalized: list[tuple[int, str, str]] = []
        self.discarded: list[tuple[int, str]] = []
        self.observed_drafts: list[tuple[int, int, str]] = []
        self.recovered: list[str] = []
        self.reconciled_heads: list[
            tuple[int, tuple[tuple[str, int, str], ...]]
        ] = []

    def reconcile_head(self, head) -> None:
        states = {
            state.core_id: state for state in head.core_state_manifests
        }
        assert set(states) == set(self.private)
        for core_id, state in states.items():
            private = self.private[core_id]
            private.generation = state.generation
            private.field_id = state.committed_field_id
        self.reconciled_heads.append(
            (
                head.tick_seq,
                tuple(
                    sorted(
                        (
                            core_id,
                            private.generation,
                            private.field_id,
                        )
                        for core_id, private in self.private.items()
                    )
                ),
            )
        )

    def stage_turn(
        self,
        *,
        core_id: str,
        observation: ObservedField,
        canonical_base_field_id: str,
        tick_seq: int,
        target_region: LogicalRegion,
    ) -> StagedCoreTurn:
        count = self.stage_counts.get(tick_seq, 0)
        self.stage_counts[tick_seq] = count + 1
        prior = observation.snapshot.region(target_region).text
        self.observed_drafts.append((tick_seq, count, prior))
        text = (
            f"proposal-{tick_seq}"
            if count == 0
            else (
                self.tool_text
                if self.tool_text is not None
                else f"final-{tick_seq}"
            )
        )
        return StagedCoreTurn(
            core_id=core_id,
            proposal=RegionProposal(
                target_region=target_region,
                text=text,
                evidence_refs=(f"evidence:{tick_seq}:{count}",),
                provenance=f"stub:{core_id}:{tick_seq}:{count}",
            ),
            observed_field_id=observation.snapshot.field_id,
            observed_tick_id=observation.snapshot.tick_id,
            candidate_token={
                "private_generation": self.private[core_id].generation,
                "canonical_base_field_id": canonical_base_field_id,
                "text": text,
                "tick": tick_seq,
            },
            evidence_ids=(f"candidate:{core_id}:{tick_seq}:{count}",),
        )

    def prepare_commit(
        self,
        *,
        turn: StagedCoreTurn,
        accepted_text: str,
        output_field_id: str,
        runtime_generation: int,
        next_tick_seq: int,
        prior_manifest: CoreStateManifest,
    ) -> PreparedCoreCommit:
        token = dict(turn.candidate_token)
        assert token["private_generation"] == self.private[turn.core_id].generation
        assert token["text"] == accepted_text
        committed_field_id = (
            "forged-output-field"
            if self.corrupt_manifest
            else output_field_id
        )
        manifest = CoreStateManifest(
            core_id=prior_manifest.core_id,
            soul_id=prior_manifest.soul_id,
            model_id=prior_manifest.model_id,
            core_state_sha256=prior_manifest.core_state_sha256,
            generation=runtime_generation,
            tick_seq=next_tick_seq,
            committed_field_id=committed_field_id,
            soul_state_sha256=_hash(
                turn.core_id,
                "soul",
                self.private[turn.core_id].generation + 1,
                accepted_text,
            ),
            cursor_state_sha256=_hash(
                turn.core_id,
                "cursor",
                self.private[turn.core_id].generation + 1,
            ),
            adapter_set_sha256=prior_manifest.adapter_set_sha256,
            rng_state_sha256=prior_manifest.rng_state_sha256,
            parent_manifest_id=prior_manifest.manifest_id,
        )
        return PreparedCoreCommit(
            core_id=turn.core_id,
            manifest=manifest,
            commit_token={
                "next_private_generation": (
                    self.private[turn.core_id].generation + 1
                ),
                "output_field_id": output_field_id,
                "text": accepted_text,
                "tick": token["tick"],
            },
        )

    def finalize_commit(self, prepared: PreparedCoreCommit) -> None:
        token = prepared.commit_token
        private = self.private[prepared.core_id]
        assert token["next_private_generation"] == private.generation + 1
        private.generation = token["next_private_generation"]
        private.field_id = token["output_field_id"]
        self.finalized.append(
            (token["tick"], prepared.core_id, token["text"])
        )

    def discard_turn(self, turn: StagedCoreTurn) -> None:
        self.discarded.append(
            (int(turn.candidate_token["tick"]), turn.core_id)
        )

    def recover_committed(self, prepared: PreparedCoreCommit) -> None:
        self.finalize_commit(prepared)
        self.recovered.append(prepared.core_id)


class OneUserInput:
    def __call__(self, head):
        operations = ()
        if head.tick_seq == 0:
            operations = (
                AppendSpans(
                    LogicalRegion.USER_INPUT,
                    (
                        FieldSpan(
                            span_id="user-0",
                            text="hello",
                            kind="user_input",
                            source="test",
                            provenance="ingress:test:0",
                        ),
                    ),
                ),
            )
        return SystemFieldUpdate(
            base_field_id=head.snapshot.field_id,
            base_tick_id=head.snapshot.tick_id,
            operations=operations,
        )


def _engine(
    tmp_path: Path,
    *,
    driver: DeterministicDriver | None = None,
    gate=None,
):
    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts(
        {
            LogicalRegion.CONVERSATION_HISTORY: "genesis",
            LogicalRegion.RESPONSE_DRAFT: "",
        }
    )
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    active_driver = driver or DeterministicDriver(identities, genesis)
    kwargs = {}
    if gate is not None:
        kwargs["consolidation_gate"] = gate
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=active_driver,
        system_updates=OneUserInput(),
        **kwargs,
    )
    return engine, store, identities, genesis, active_driver


def test_six_ticks_rotate_roles_and_consolidator_alone_controls_output(
    tmp_path: Path,
) -> None:
    engine, store, identities, genesis, driver = _engine(tmp_path)
    try:
        results = engine.run_forever(max_ticks=6)
        assert len(results) == 6
        assert [item.proposer_core_id for item in results] == [
            "core-0",
            "core-1",
            "core-2",
            "core-3",
            "core-0",
            "core-1",
        ]
        assert [item.consolidator_core_id for item in results] == [
            "core-3",
            "core-0",
            "core-1",
            "core-2",
            "core-3",
            "core-0",
        ]
        assert [item.sleeper_core_id for item in results] == [
            "core-2",
            "core-3",
            "core-0",
            "core-1",
            "core-2",
            "core-3",
        ]
        assert all(item.accepted for item in results)
        assert all(
            item.finalized_core_ids == (item.consolidator_core_id,)
            for item in results
        )
        assert driver.observed_drafts[0] == (0, 0, "")
        assert driver.observed_drafts[1] == (0, 1, "proposal-0")
        head = store.recover()
        assert head.tick_seq == 6
        assert head.generation == 6
        assert head.snapshot.tick_id == genesis.tick_id + 6
        assert head.snapshot.region(LogicalRegion.USER_INPUT).text == "hello"
        assert (
            head.snapshot.region(LogicalRegion.RESPONSE_DRAFT).text
            == "final-5"
        )
        assert head.snapshot.field_id == results[-1].output_field_id
        assert store.replay_journal().snapshot == head.snapshot
        assert {
            core_id for _, core_id, _ in driver.finalized
        } == {identity.core_id for identity in identities}
    finally:
        store.close()


def test_rejected_consolidation_advances_without_state_or_effects(
    tmp_path: Path,
) -> None:
    def reject(*_):
        return False, "constitutional gate rejected action"

    engine, store, _, genesis, driver = _engine(tmp_path, gate=reject)
    try:
        result = engine.tick_once()
        assert result.accepted is False
        assert result.finalized_core_ids == ()
        assert result.tool_request_ids == ()
        head = store.recover()
        assert head.tick_seq == 1
        assert head.snapshot.tick_id == genesis.tick_id + 1
        assert head.snapshot.region(LogicalRegion.USER_INPUT).text == "hello"
        assert head.snapshot.region(LogicalRegion.RESPONSE_DRAFT).text == ""
        assert all(private.generation == 0 for private in driver.private.values())
        assert len(driver.discarded) == 2
    finally:
        store.close()


def test_consolidated_marker_intent_is_queued_but_not_executed(
    tmp_path: Path,
) -> None:
    marker = (
        ":::axon-invoke/v1\n"
        '## {"action_id":"read-1","op":"stat","root":"workspace","path":"."}\n'
        ":::end"
    )
    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    driver = DeterministicDriver(identities, genesis, tool_text=marker)
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=driver,
    )
    try:
        result = engine.tick_once()
        assert len(result.tool_request_ids) == 1
        row = store.connection.execute(
            "SELECT status, result_id FROM tool_requests WHERE request_id=?",
            (result.tool_request_ids[0],),
        ).fetchone()
        assert tuple(row) == ("queued", None)
        assert (
            store.recover()
            .snapshot.region(LogicalRegion.RESPONSE_DRAFT)
            .text
            == marker
        )
    finally:
        store.close()


def test_failed_store_boundary_does_not_install_prepared_private_state(
    tmp_path: Path,
) -> None:
    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    driver = DeterministicDriver(
        identities,
        genesis,
        corrupt_manifest=True,
    )
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=driver,
    )
    try:
        with pytest.raises(StoreContractError):
            engine.tick_once()
        head = store.recover()
        assert head.generation == 0
        assert head.snapshot == genesis
        assert all(private.generation == 0 for private in driver.private.values())
        assert driver.finalized == []
        assert len(driver.discarded) == 2
    finally:
        store.close()


def test_post_commit_failure_does_not_retry_or_hide_committed_tick(
    tmp_path: Path,
) -> None:
    errors: list[Exception] = []

    def broken_sleep(*_):
        raise RuntimeError("sleep worker unavailable")

    engine, store, _, _, _ = _engine(tmp_path)
    engine.post_commit_hook = broken_sleep
    engine.on_post_commit_error = errors.append
    try:
        results = engine.run_forever(max_ticks=2)
        assert len(results) == 2
        assert store.recover().tick_seq == 2
        assert len(errors) == 2
        assert isinstance(engine.last_post_commit_error, RuntimeError)
    finally:
        store.close()


def test_finalize_failure_reloads_committed_private_state(tmp_path: Path) -> None:
    class FlakyDriver(DeterministicDriver):
        failed = False

        def finalize_commit(self, prepared: PreparedCoreCommit) -> None:
            if not self.failed:
                self.failed = True
                raise RuntimeError("simulated RAM install failure")
            super().finalize_commit(prepared)

    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    driver = FlakyDriver(identities, genesis)
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=driver,
    )
    try:
        result = engine.tick_once()
        assert result.finalized_core_ids == (result.consolidator_core_id,)
        assert driver.recovered == [result.consolidator_core_id]
        assert store.recover().tick_seq == 1
    finally:
        store.close()


def test_unrecoverable_post_commit_private_state_stops_fail_closed(
    tmp_path: Path,
) -> None:
    class BrokenDriver(DeterministicDriver):
        def finalize_commit(self, prepared: PreparedCoreCommit) -> None:
            raise RuntimeError("install failed")

        def recover_committed(self, prepared: PreparedCoreCommit) -> None:
            raise RuntimeError("reload failed")

    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    driver = BrokenDriver(identities, genesis)
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=driver,
    )
    try:
        with pytest.raises(CommittedStateRecoveryError):
            engine.run_forever(max_ticks=1)
        # The journal remains authoritative even though this process halted.
        assert store.recover().tick_seq == 1
    finally:
        store.close()


def test_ingress_consumption_and_one_tick_user_input_clear_are_atomic(
    tmp_path: Path,
) -> None:
    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    driver = DeterministicDriver(identities, genesis)
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    queue = IngressQueue(store.connection)
    event = queue.enqueue(
        IngressEvent(
            idempotency_key="phone:0",
            source="phone",
            source_sequence=0,
            event_kind="user_input",
            exact_text="exact caf\u00e9 input",
            provenance="phone-call:0",
        )
    )
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=driver,
        system_updates=IngressSystemUpdateProvider(queue),
    )
    try:
        first = engine.tick_once()
        head = store.recover()
        assert head.snapshot.region(LogicalRegion.USER_INPUT).text == event.exact_text
        assert (
            head.snapshot.region(LogicalRegion.CONVERSATION_HISTORY).text
            == event.exact_text
        )
        receipts = queue.receipts()
        assert len(receipts) == 1
        assert receipts[0].event_id == event.event_id
        assert receipts[0].tick_commit_id == first.commit.commit_id
        assert queue.pending() == ()

        engine.tick_once()
        head = store.recover()
        assert head.snapshot.region(LogicalRegion.USER_INPUT).text == ""
        assert (
            head.snapshot.region(LogicalRegion.CONVERSATION_HISTORY).text
            == event.exact_text
        )
    finally:
        store.close()


def test_failed_tick_leaves_ingress_pending_and_unconsumed(tmp_path: Path) -> None:
    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    driver = DeterministicDriver(
        identities,
        genesis,
        corrupt_manifest=True,
    )
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    queue = IngressQueue(store.connection)
    event = queue.enqueue(
        IngressEvent(
            idempotency_key="phone:0",
            source="phone",
            source_sequence=0,
            event_kind="user_input",
            exact_text="do not lose me",
            provenance="phone-call:0",
        )
    )
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=driver,
        system_updates=IngressSystemUpdateProvider(queue),
    )
    try:
        with pytest.raises(StoreContractError):
            engine.tick_once()
        assert queue.pending() == (event,)
        assert queue.receipts() == ()
        assert store.recover().snapshot == genesis
    finally:
        store.close()


def test_projector_cannot_substitute_a_same_tick_sibling(
    tmp_path: Path,
) -> None:
    class SiblingProjector:
        def project(
            self,
            snapshot: SharedFieldSnapshot,
            *,
            ancestor_field_ids: tuple[str, ...],
        ) -> ObservedField:
            sibling = SharedFieldSnapshot.from_texts(
                {LogicalRegion.RESPONSE_DRAFT: "unauthenticated sibling"},
                tick_id=snapshot.tick_id,
                parent_field_id=ancestor_field_ids[-1],
            )
            return ObservedField(
                snapshot=sibling,
                ancestor_field_ids=ancestor_field_ids,
            )

    engine, store, _, genesis, driver = _engine(tmp_path)
    engine.projector = SiblingProjector()
    try:
        with pytest.raises(
            RuntimeEngineError,
            match="authenticated direct projection",
        ):
            engine.tick_once()
        assert store.recover().snapshot == genesis
        assert driver.stage_counts == {}
    finally:
        store.close()


@pytest.mark.parametrize("fault", ("observation", "target"))
def test_staged_turn_must_match_observation_and_target(
    tmp_path: Path,
    fault: str,
) -> None:
    class InvalidTurnDriver(DeterministicDriver):
        invalid_returned = False

        def stage_turn(self, **kwargs) -> StagedCoreTurn:
            turn = super().stage_turn(**kwargs)
            if self.invalid_returned:
                return turn
            self.invalid_returned = True
            if fault == "observation":
                return replace(
                    turn,
                    observed_field_id="stale-observed-field",
                )
            return replace(
                turn,
                proposal=replace(
                    turn.proposal,
                    target_region=LogicalRegion.SCRATCH,
                ),
            )

    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    driver = InvalidTurnDriver(identities, genesis)
    store = RuntimeStore(tmp_path / f"{fault}.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=driver,
    )
    try:
        with pytest.raises(RuntimeEngineError):
            engine.tick_once()
        assert store.recover().snapshot == genesis
        assert driver.finalized == []
        assert len(driver.discarded) == 1
    finally:
        store.close()


def test_prepared_commit_must_belong_to_originating_turn(
    tmp_path: Path,
) -> None:
    class SwappedPreparedDriver(DeterministicDriver):
        def prepare_commit(self, **kwargs) -> PreparedCoreCommit:
            item = super().prepare_commit(**kwargs)
            wrong_core_id = (
                "core-0" if item.core_id != "core-0" else "core-1"
            )
            wrong_manifest = replace(
                item.manifest,
                core_id=wrong_core_id,
            )
            return replace(
                item,
                core_id=wrong_core_id,
                manifest=wrong_manifest,
            )

    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    driver = SwappedPreparedDriver(identities, genesis)
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=driver,
    )
    try:
        with pytest.raises(
            RuntimeEngineError,
            match="originating turn",
        ):
            engine.tick_once()
        assert store.recover().snapshot == genesis
        assert driver.finalized == []
    finally:
        store.close()


def test_duplicate_prepared_core_is_rejected_before_database_commit(
    tmp_path: Path,
) -> None:
    class DuplicatePreparedDriver(DeterministicDriver):
        first_prepared: PreparedCoreCommit | None = None

        def prepare_commit(self, **kwargs) -> PreparedCoreCommit:
            item = super().prepare_commit(**kwargs)
            if self.first_prepared is None:
                self.first_prepared = item
                return item
            return self.first_prepared

    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    driver = DuplicatePreparedDriver(
        identities,
        genesis,
        tool_text="proposal-0",
    )
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=driver,
    )
    try:
        with pytest.raises(RuntimeEngineError, match="duplicate prepared"):
            engine.tick_once()
        assert store.recover().snapshot == genesis
        assert driver.finalized == []
    finally:
        store.close()


def test_only_manifest_ids_returned_by_commit_are_finalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, store, _, _, driver = _engine(tmp_path)
    commit_tick = store.commit_tick

    def commit_with_mismatched_receipt(**kwargs):
        committed = commit_tick(**kwargs)
        return replace(
            committed,
            updated_core_state_manifest_ids=(),
        )

    monkeypatch.setattr(store, "commit_tick", commit_with_mismatched_receipt)
    try:
        with pytest.raises(
            CommittedStateRecoveryError,
            match="different private-state manifest set",
        ):
            engine.tick_once()
        assert store.recover().tick_seq == 1
        assert driver.finalized == []
    finally:
        store.close()


def test_restart_reconciles_private_state_after_commit_boundary_crash(
    tmp_path: Path,
) -> None:
    class CrashAfterCommitDriver(DeterministicDriver):
        def finalize_commit(self, prepared: PreparedCoreCommit) -> None:
            raise KeyboardInterrupt("simulated process loss")

    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    database_path = tmp_path / "runtime.sqlite3"
    crashing_driver = CrashAfterCommitDriver(identities, genesis)
    store = RuntimeStore(database_path)
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=crashing_driver,
    )
    with pytest.raises(KeyboardInterrupt, match="process loss"):
        engine.tick_once()
    committed_head = store.recover()
    assert committed_head.tick_seq == 1
    assert crashing_driver.finalized == []
    store.close()

    restarted_store = RuntimeStore(database_path)
    restarted_driver = DeterministicDriver(identities, genesis)
    restarted_engine = AxonRuntimeEngine(
        store=restarted_store,
        identities=identities,
        driver=restarted_driver,
    )
    try:
        restarted_engine.tick_once()
        reconciled_tick, reconciled_states = (
            restarted_driver.reconciled_heads[0]
        )
        reconciled_by_id = {
            core_id: (generation, field_id)
            for core_id, generation, field_id in reconciled_states
        }
        assert reconciled_tick == committed_head.tick_seq
        assert reconciled_by_id["core-3"] == (
            1,
            committed_head.snapshot.field_id,
        )
        assert restarted_store.recover().tick_seq == 2
    finally:
        restarted_store.close()


def test_post_commit_cleanup_and_error_handler_failures_are_warnings(
    tmp_path: Path,
) -> None:
    class ThrowingDiscardDriver(DeterministicDriver):
        def discard_turn(self, turn: StagedCoreTurn) -> None:
            raise RuntimeError(f"cannot discard {turn.core_id}")

    def broken_hook(*_) -> None:
        raise RuntimeError("sleep hook failed")

    def broken_error_handler(_exc: Exception) -> None:
        raise RuntimeError("error handler failed")

    identities = _identities()
    genesis = SharedFieldSnapshot.from_texts({"response_draft": ""})
    driver = ThrowingDiscardDriver(identities, genesis)
    store = RuntimeStore(tmp_path / "runtime.sqlite3")
    store.initialize(genesis, identities, _initial_states(identities, genesis))
    engine = AxonRuntimeEngine(
        store=store,
        identities=identities,
        driver=driver,
        post_commit_hook=broken_hook,
        on_post_commit_error=broken_error_handler,
    )
    try:
        result = engine.tick_once()
        assert store.recover().tick_seq == 1
        assert len(result.post_commit_warnings) == 3
        assert result.post_commit_warnings[0].startswith("discard_turn ")
        assert result.post_commit_warnings[1].startswith("post_commit_hook:")
        assert result.post_commit_warnings[2].startswith(
            "on_post_commit_error:"
        )
    finally:
        store.close()


def test_run_forever_requires_positive_maximum_backoff(
    tmp_path: Path,
) -> None:
    engine, store, _, _, _ = _engine(tmp_path)
    try:
        with pytest.raises(ValueError, match="must be positive"):
            engine.run_forever(
                max_ticks=0,
                error_backoff_initial_seconds=0,
                error_backoff_max_seconds=0,
            )
        with pytest.raises(ValueError, match="at least 1"):
            engine.run_forever(
                max_ticks=0,
                error_backoff_multiplier=0.5,
            )
        with pytest.raises(ValueError, match="must be positive"):
            engine.run_forever(
                max_ticks=0,
                stop_poll_interval_seconds=0,
            )
    finally:
        store.close()


def test_stop_file_interrupts_error_backoff_promptly(tmp_path: Path) -> None:
    engine, store, _, _, _ = _engine(tmp_path)
    stop_path = tmp_path / "STOP"
    errors: list[Exception] = []

    def fail_tick():
        raise RuntimeError("persistent tick failure")

    engine.tick_once = fail_tick
    timer = threading.Timer(0.05, stop_path.touch)
    started = time.monotonic()
    timer.start()
    try:
        results = engine.run_forever(
            stop_file=stop_path,
            error_backoff_initial_seconds=2.0,
            error_backoff_max_seconds=2.0,
            error_backoff_multiplier=3.0,
            stop_poll_interval_seconds=0.01,
            on_error=errors.append,
        )
        elapsed = time.monotonic() - started
        assert results == ()
        assert len(errors) == 1
        assert elapsed < 0.75
    finally:
        timer.join(timeout=1.0)
        store.close()


def test_continuous_run_does_not_retain_tick_results(
    tmp_path: Path,
) -> None:
    engine, store, _, _, _ = _engine(tmp_path)
    stopper = threading.Event()
    tick_once = engine.tick_once
    completed = 0

    def stop_after_three_ticks():
        nonlocal completed
        result = tick_once()
        completed += 1
        if completed == 3:
            stopper.set()
        return result

    engine.tick_once = stop_after_three_ticks
    try:
        results = engine.run_forever(
            stop_event=stopper,
            max_ticks=None,
        )
        assert completed == 3
        assert store.recover().tick_seq == 3
        assert results == ()
    finally:
        store.close()
