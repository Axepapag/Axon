"""Corruption-heavy tests for the isolated v3 transaction store."""
from __future__ import annotations

import copy
import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from runtime.field import (
    FieldDelta,
    InsertText,
    LogicalRegion,
    SharedFieldSnapshot,
    canonical_sha256,
)
from runtime.axon_runtime.field_transaction import (
    AppendSpans,
    FieldSpan,
    SystemFieldUpdate,
)
from runtime.axon_runtime.v3_contracts import (
    ArtifactRejectionRecord,
    V3ContractError,
)
from runtime.axon_runtime.v3_serde import (
    canonical_json_text,
    parse_json_object,
)
from runtime.axon_runtime.v3_transaction import (
    ActiveProjectionArtifactV3,
    PrivateStateArtifactV3,
    ReadPageArtifactV3,
    SyntheticTickSpec,
    TickTransactionV3,
    V3TransactionError,
    V3TransactionPopulationError,
    compose_synthetic_tick_v3,
    replay_tick_transaction_v3,
    validate_tick_transaction_v3,
)
from runtime.axon_runtime.v3_store import (
    V3AlreadyInitializedError,
    V3ContinuationProof,
    V3JournalIntegrityError,
    V3NotInitializedError,
    V3PopulationMismatchError,
    V3RuntimeHead,
    V3RuntimeStore,
    V3StaleContinuationError,
    V3StoreError,
)


_BINDING = "synthetic-binding-v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_genesis_snapshot() -> SharedFieldSnapshot:
    return SharedFieldSnapshot.from_texts(
        {
            "response_draft": "hello world",
            "scratch": "",
        },
        tick_id=0,
    )


def _make_private_state(
    core_id: str,
    *,
    phase: str = "GENESIS",
    tick_seq: int = 0,
    substep: int = 0,
    soul: str | None = None,
    cursor: str | None = None,
    rng: str | None = None,
    binding: str = _BINDING,
    parent: str | None = None,
    working_field_id: str | None = None,
    board_id: str | None = None,
    delta_id: str | None = None,
) -> PrivateStateArtifactV3:
    return PrivateStateArtifactV3(
        core_id=core_id,
        tick_seq=tick_seq,
        phase=phase,
        substep=substep,
        soul_sha256=soul or _sha(f"{core_id}-soul-{phase}"),
        cursor_state_sha256=cursor or _sha(f"{core_id}-cursor-{phase}"),
        rng_state_sha256=rng or _sha(f"{core_id}-rng-{phase}"),
        model_binding_epoch_id=binding,
        parent_state_leaf_id=parent,
        working_field_id=working_field_id,
        board_id=board_id,
        delta_id=delta_id,
    )


def _make_population(
    *,
    online: tuple[str, ...] = ("axon64-a", "axon128-a", "axon64-b"),
    offline: str = "axon128-b",
) -> tuple[SharedFieldSnapshot, tuple[PrivateStateArtifactV3, ...]]:
    snapshot = _make_genesis_snapshot()
    states = []
    for core_id in (*online, offline):
        states.append(_make_private_state(core_id))
    return snapshot, tuple(states)


def _open_store(tmp_path: Path) -> V3RuntimeStore:
    return V3RuntimeStore(str(tmp_path))


def _init_store(store: V3RuntimeStore) -> V3RuntimeHead:
    snapshot, states = _make_population()
    return store.initialize(snapshot, states, _BINDING)


def _make_spec(
    *,
    accepted: bool = True,
    online: tuple[str, ...] = ("axon64-a", "axon128-a", "axon64-b"),
    offline: str = "axon128-b",
    consolidator: str = "axon64-a",
    completion_order: tuple[str, ...] = (),
    seed: int = 0,
) -> SyntheticTickSpec:
    return SyntheticTickSpec(
        online_core_ids=online,
        offline_core_id=offline,
        consolidator_core_id=consolidator,
        accepted=accepted,
        seed=seed,
        completion_order=completion_order,
        model_binding_epoch_id=_BINDING,
    )


def _commit_transaction(
    store: V3RuntimeStore, *, accepted: bool
) -> V3RuntimeHead:
    head = store.current_head()
    states = tuple(
        _state_from_head(store, head, core_id)
        for core_id, _ in head.core_state_leaf_ids
    )
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        states,
        _make_spec(accepted=accepted),
    )
    proof = store.continuation_proof()
    return store.commit_tick(transaction, proof)


def _load_private_state_from_store(
    store: V3RuntimeStore, leaf_id: str
) -> PrivateStateArtifactV3:
    row = store.connection.execute(
        "SELECT canonical_json FROM journal WHERE event_type='private_state' AND record_id=?",
        (leaf_id,),
    ).fetchone()
    assert row is not None, f"missing private state {leaf_id}"
    item = parse_json_object(row["canonical_json"])
    return PrivateStateArtifactV3(
        core_id=item["core_id"],
        tick_seq=item["tick_seq"],
        phase=item["phase"],
        substep=item["substep"],
        soul_sha256=item["soul_sha256"],
        cursor_state_sha256=item["cursor_state_sha256"],
        rng_state_sha256=item.get("rng_state_sha256"),
        model_binding_epoch_id=item["model_binding_epoch_id"],
        parent_state_leaf_id=item.get("parent_state_leaf_id"),
        working_field_id=item.get("working_field_id"),
        board_id=item.get("board_id"),
        delta_id=item.get("delta_id"),
    )


def _state_from_head(
    store: V3RuntimeStore, head: V3RuntimeHead, core_id: str
) -> PrivateStateArtifactV3:
    leaf_id = dict(head.core_state_leaf_ids)[core_id]
    artifact = _load_private_state_from_store(store, leaf_id)
    if artifact.core_id != core_id:
        raise ValueError(f"loaded leaf {leaf_id} belongs to {artifact.core_id}, not {core_id}")
    return artifact


def _journal_row_count(store: V3RuntimeStore) -> int:
    row = store.connection.execute("SELECT COUNT(*) AS n FROM journal").fetchone()
    return int(row["n"])


def _journal_canonical_bytes(store: V3RuntimeStore) -> bytes:
    rows = store.connection.execute(
        "SELECT canonical_json FROM journal ORDER BY seq"
    ).fetchall()
    return b"\n".join(row["canonical_json"].encode("utf-8") for row in rows)


def _copy_db(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Checkpoint any WAL data into the main file before copying so the copy
    # is byte-for-byte complete without depending on sibling -wal/-shm files.
    conn = sqlite3.connect(str(source))
    conn.execute("PRAGMA wal_checkpoint(FULL)")
    conn.close()
    destination.write_bytes(source.read_bytes())


def _canonical_hash_event(
    previous: str | None,
    event_type: str,
    record_id: str,
    payload: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "previous_event_id": previous,
            "event_type": event_type,
            "record_id": record_id,
            "payload": dict(payload),
        }
    )


def _corrupt_transaction_remove_delta(transaction: TickTransactionV3) -> TickTransactionV3:
    return replace(transaction, deltas=transaction.deltas[:-1])


def _corrupt_read_cycle_incomplete(transaction: TickTransactionV3) -> TickTransactionV3:
    # Drop the last page entirely.  The cycle still claims it, so coverage is
    # incomplete without modifying any derived record ID.
    pages = transaction.read_pages[:-1]
    return replace(transaction, read_pages=pages)


def _corrupt_duplicate_page(transaction: TickTransactionV3) -> TickTransactionV3:
    page = transaction.read_pages[0]
    duplicate = replace(page, page_index=1)
    pages = transaction.read_pages + (duplicate,)
    return replace(transaction, read_pages=pages)


def _corrupt_page_characters(transaction: TickTransactionV3) -> TickTransactionV3:
    page = transaction.read_pages[0]
    if not page.characters:
        new_characters = "x"
    else:
        new_characters = page.characters[:-1] + ("X" if page.characters[-1] != "X" else "Y")
    corrupted = replace(page, characters=new_characters)
    pages = (corrupted,) + transaction.read_pages[1:]
    return replace(transaction, read_pages=pages)


def _corrupt_missing_terminal_page(transaction: TickTransactionV3) -> TickTransactionV3:
    pages = transaction.read_pages[:-1]
    return replace(transaction, read_pages=pages)


def _corrupt_reused_proof_after_board_change(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    # Mutate the board_id stamped on every page of one REFINE cycle.  The
    # derived page IDs change, so the cycle's page_view_hashes no longer match.
    refine_cycle = next(c for c in transaction.read_cycles if c.phase == "REFINE")
    corrupted_pages = tuple(
        replace(page, board_id=transaction.refine_board.board_id)
        if (page.core_id == refine_cycle.core_id and page.phase == refine_cycle.phase)
        else page
        for page in transaction.read_pages
    )
    return replace(transaction, read_pages=corrupted_pages)


def _corrupt_offline_participation(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    passes = list(transaction.initial_passes)
    passes[0] = replace(passes[0], core_id=transaction.plan.offline_core_id)
    return replace(transaction, initial_passes=tuple(passes))


def _corrupt_final_author(transaction: TickTransactionV3) -> TickTransactionV3:
    transitions = list(transaction.soul_transitions)
    for index, transition in enumerate(transitions):
        if transition.phase == "FINAL":
            transitions[index] = replace(
                transition,
                core_id=transaction.plan.online_core_ids[1],
            )
    return replace(transaction, soul_transitions=tuple(transitions))


def _corrupt_transition_parent(transaction: TickTransactionV3) -> TickTransactionV3:
    transitions = list(transaction.soul_transitions)
    for index, transition in enumerate(transitions):
        if transition.phase == "REFINE":
            transitions[index] = replace(
                transition,
                parent_transition_id="soul-transition-" + "0" * 64,
            )
    return replace(transaction, soul_transitions=tuple(transitions))


def _corrupt_delta_base_field(transaction: TickTransactionV3) -> TickTransactionV3:
    from runtime.field import FieldDelta

    deltas = list(transaction.deltas)
    delta = deltas[0]
    new_delta = FieldDelta(
        base_field_id="a" * 64,
        base_tick_id=delta.base_tick_id,
        author_core_id=delta.author_core_id,
        pass_id=delta.pass_id,
        operations=delta.operations,
        evidence=delta.evidence,
    )
    deltas[0] = new_delta
    return replace(transaction, deltas=tuple(deltas))


def _corrupt_commit_output(transaction: TickTransactionV3) -> TickTransactionV3:
    commit = replace(transaction.commit, output_field_id="b" * 64)
    return replace(transaction, commit=commit)


def _corrupt_invocation_ids(transaction: TickTransactionV3) -> TickTransactionV3:
    from runtime.axon_runtime.v3_contracts import ConsolidationRecord

    consolidation = replace(
        transaction.consolidation,
        invocation_request_ids=("invoke-1",),
    )
    return replace(transaction, consolidation=consolidation)


def _corrupt_candidate_donor_core(transaction: TickTransactionV3) -> TickTransactionV3:
    candidates = list(transaction.candidate_private_states)
    candidates[0] = replace(candidates[0], core_id="other-core")
    return replace(transaction, candidate_private_states=tuple(candidates))


def _corrupt_candidate_stale_parent(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    candidates = list(transaction.candidate_private_states)
    candidates[1] = replace(
        candidates[1],
        parent_state_leaf_id="private-state-" + "0" * 64,
    )
    return replace(transaction, candidate_private_states=tuple(candidates))


def _corrupt_candidate_wrong_soul(transaction: TickTransactionV3) -> TickTransactionV3:
    candidates = list(transaction.candidate_private_states)
    candidates[0] = replace(candidates[0], soul_sha256="c" * 64)
    return replace(transaction, candidate_private_states=tuple(candidates))


def _corrupt_candidate_wrong_cursor(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    candidates = list(transaction.candidate_private_states)
    candidates[0] = replace(candidates[0], cursor_state_sha256="d" * 64)
    return replace(transaction, candidate_private_states=tuple(candidates))


def _corrupt_candidate_wrong_rng(transaction: TickTransactionV3) -> TickTransactionV3:
    candidates = list(transaction.candidate_private_states)
    candidates[0] = replace(candidates[0], rng_state_sha256="e" * 64)
    return replace(transaction, candidate_private_states=tuple(candidates))


def _corrupt_candidate_wrong_binding(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    candidates = list(transaction.candidate_private_states)
    candidates[0] = replace(candidates[0], model_binding_epoch_id="other-binding")
    return replace(transaction, candidate_private_states=tuple(candidates))


def _corrupt_candidate_arbitrary_leaf(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    transitions = list(transaction.soul_transitions)
    transitions[0] = replace(
        transitions[0],
        candidate_private_state_id="private-state-" + "0" * 64,
    )
    return replace(transaction, soul_transitions=tuple(transitions))


def _corrupt_false_accepted_noop(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    dispositions = list(transaction.dispositions)
    replaced_disp = None
    for index, disp in enumerate(dispositions):
        if disp.disposition == "COMMITTED":
            replaced_disp = replace(
                disp,
                disposition="ACCEPTED_NOOP",
                reason="claims unchanged state",
            )
            dispositions[index] = replaced_disp
            break
    assert replaced_disp is not None
    new_disp_by_transition = {
        disp.transition_id: disp.disposition_id for disp in dispositions
    }
    commit = replace(
        transaction.commit,
        disposition_ids_by_transition=tuple(
            sorted(new_disp_by_transition.items())
        ),
    )
    return replace(transaction, dispositions=tuple(dispositions), commit=commit)


# ---------------------------------------------------------------------------
# Happy path: accepted synthetic transaction
# ---------------------------------------------------------------------------


def test_accepted_synthetic_transaction_is_deterministic(tmp_path: Path) -> None:
    head, states = _make_population()
    tx1 = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    tx2 = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    assert tx1.commit.commit_id == tx2.commit.commit_id
    assert tx1.plan.plan_id == tx2.plan.plan_id
    assert tx1.projection.projection_id == tx2.projection.projection_id
    replay = validate_tick_transaction_v3(tx1)
    assert replay.commit_id == tx1.commit.commit_id


def test_rejected_transaction_preserves_input_leaves(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(
        head,
        states,
        _make_spec(accepted=False),
    )
    replay = validate_tick_transaction_v3(transaction)
    online = set(transaction.plan.online_core_ids)
    online_input_leaves = {
        state.core_id: state.state_leaf_id
        for state in transaction.input_private_states
        if state.core_id in online
    }
    final_leaves = dict(transaction.commit.final_core_state_leaf_ids)
    assert final_leaves == online_input_leaves
    assert replay.final_snapshot.field_id == transaction.plan.no_core_delta_output_field_id
    # Offline leaf is not in the commit final leaves but is preserved conceptually.
    assert transaction.plan.offline_core_id not in final_leaves


@pytest.mark.parametrize(
    "completion_order",
    [
        (),
        ("axon64-a", "axon128-a", "axon64-b"),
        ("axon64-b", "axon128-a", "axon64-a"),
        ("axon128-a", "axon64-b", "axon64-a"),
    ],
)
def test_completion_order_permutations_are_identical(
    tmp_path: Path, completion_order: tuple[str, ...]
) -> None:
    head, states = _make_population()
    baseline = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    permuted = compose_synthetic_tick_v3(
        head,
        states,
        _make_spec(accepted=True, completion_order=completion_order),
    )
    assert permuted.commit.commit_id == baseline.commit.commit_id
    assert permuted.read_pages == baseline.read_pages
    assert permuted.deltas == baseline.deltas


# ---------------------------------------------------------------------------
# Store initialization and recovery
# ---------------------------------------------------------------------------


def test_store_initialization_and_exact_recovery(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    recovered = store.recover()
    assert recovered == head
    assert recovered == store.current_head()


def test_store_commit_accepted_and_restart_replay(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    head_after_commit = _commit_transaction(store, accepted=True)
    assert head_after_commit.tick_seq == 1
    assert head_after_commit.generation == 1

    store2 = _open_store(tmp_path)
    replayed = store2.replay_journal()
    assert replayed == head_after_commit
    assert store2.recover() == head_after_commit


def test_store_commit_rejected_and_restart_replay(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    head_after_commit = _commit_transaction(store, accepted=False)
    assert head_after_commit.tick_seq == 1
    assert head_after_commit.generation == 1
    original_leaves = dict(head.core_state_leaf_ids)
    assert dict(head_after_commit.core_state_leaf_ids) == original_leaves

    store2 = _open_store(tmp_path)
    assert store2.replay_journal() == head_after_commit
    assert store2.recover() == head_after_commit


def test_two_consecutive_ticks_with_role_change(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)

    tx1 = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    proof1 = store.continuation_proof()
    head1 = store.commit_tick(tx1, proof1)

    states2 = tuple(
        _state_from_head(store, head1, core_id) for core_id, _ in head1.core_state_leaf_ids
    )
    tx2 = compose_synthetic_tick_v3(
        head1.snapshot,
        states2,
        SyntheticTickSpec(
            online_core_ids=("axon64-a", "axon128-a", "axon128-b"),
            offline_core_id="axon64-b",
            consolidator_core_id="axon128-a",
            accepted=True,
            seed=1,
            model_binding_epoch_id=_BINDING,
        ),
    )
    proof2 = store.continuation_proof()
    head2 = store.commit_tick(tx2, proof2)
    assert head2.tick_seq == 2
    assert head2.generation == 2
    assert store.recover() == head2


# ---------------------------------------------------------------------------
# Continuation proof and commit safety
# ---------------------------------------------------------------------------


def test_stale_continuation_proof_rolls_back_writes(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    stale_proof = store.continuation_proof()
    stale = replace(stale_proof, tick_seq=stale_proof.tick_seq + 1)

    rows_before = _journal_row_count(store)
    head_before = store.current_head()
    with pytest.raises(V3StaleContinuationError):
        store.commit_tick(transaction, stale)
    assert _journal_row_count(store) == rows_before
    assert store.current_head() == head_before


def test_failed_commit_leaves_equivalent_journal_and_head(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    corrupted = _corrupt_transaction_remove_delta(transaction)
    proof = store.continuation_proof()

    journal_bytes_before = _journal_canonical_bytes(store)
    head_before = store.current_head()
    with pytest.raises(V3TransactionError):
        store.commit_tick(corrupted, proof)
    assert _journal_canonical_bytes(store) == journal_bytes_before
    assert store.current_head() == head_before


# ---------------------------------------------------------------------------
# Structural corruption rejections
# ---------------------------------------------------------------------------


def test_missing_author_pass_delta_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = replace(transaction, initial_passes=transaction.initial_passes[:-1])
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


def test_incomplete_read_cycle_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_read_cycle_incomplete(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


def test_duplicate_page_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_duplicate_page(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


def test_altered_characters_and_counts_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_page_characters(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


def test_missing_terminal_coverage_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_missing_terminal_page(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


def test_reused_proof_after_board_change_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_reused_proof_after_board_change(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


def test_offline_core_participation_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_offline_participation(transaction)
    with pytest.raises((V3TransactionError, V3ContractError)):
        validate_tick_transaction_v3(corrupted)


def test_wrong_final_author_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_final_author(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


def test_wrong_transition_parent_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_transition_parent(transaction)
    with pytest.raises((V3TransactionError, V3ContractError)):
        validate_tick_transaction_v3(corrupted)


def test_stale_w_base_field_delta_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_delta_base_field(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


def test_field_audit_output_mismatch_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_commit_output(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


def test_invocation_authority_mismatch_rejection(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_invocation_ids(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


@pytest.mark.parametrize(
    "corruptor",
    [
        _corrupt_candidate_donor_core,
        _corrupt_candidate_stale_parent,
        _corrupt_candidate_wrong_soul,
        _corrupt_candidate_wrong_cursor,
        _corrupt_candidate_wrong_rng,
        _corrupt_candidate_wrong_binding,
        _corrupt_candidate_arbitrary_leaf,
        _corrupt_false_accepted_noop,
    ],
)
def test_rejected_candidate_state_installation_rejection(
    tmp_path: Path, corruptor: Any
) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = corruptor(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


# ---------------------------------------------------------------------------
# Journal-level corruption
# ---------------------------------------------------------------------------


def test_payload_hash_previous_link_corruption(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    corrupted_dir = tmp_path / "corrupted_store"
    corrupted_dir.mkdir()
    _copy_db(store.path, corrupted_dir / "runtime-v3.sqlite3")
    conn = sqlite3.connect(str(corrupted_dir / "runtime-v3.sqlite3"))
    conn.row_factory = sqlite3.Row
    # Temporarily drop the append-only trigger so we can simulate offline
    # file-level corruption through SQL.
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_update")
    row = conn.execute(
        "SELECT seq, event_id, previous_event_id, event_type, record_id, canonical_json "
        "FROM journal ORDER BY seq LIMIT 1"
    ).fetchone()
    payload = parse_json_object(row["canonical_json"])
    payload["extra"] = "injected"
    new_text = canonical_json_text(payload)
    conn.execute(
        "UPDATE journal SET canonical_json=? WHERE seq=?",
        (new_text, row["seq"]),
    )
    conn.commit()
    conn.close()
    corrupted_store = V3RuntimeStore(str(corrupted_dir))
    with pytest.raises(V3JournalIntegrityError):
        corrupted_store.verify_journal_chain()


def test_hash_valid_unknown_event_type_corruption(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    corrupted_dir = tmp_path / "corrupted_unknown_store"
    corrupted_dir.mkdir()
    _copy_db(store.path, corrupted_dir / "runtime-v3.sqlite3")
    conn = sqlite3.connect(str(corrupted_dir / "runtime-v3.sqlite3"))
    conn.row_factory = sqlite3.Row
    previous = conn.execute(
        "SELECT event_id FROM journal ORDER BY seq DESC LIMIT 1"
    ).fetchone()["event_id"]
    payload = {"schema": "evil", "value": 1}
    event_id = _canonical_hash_event(previous, "unknown_event", "evil-id", payload)
    conn.execute(
        "INSERT INTO journal(event_id, previous_event_id, event_type, record_id, canonical_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (event_id, previous, "unknown_event", "evil-id", canonical_json_text(payload)),
    )
    conn.commit()
    conn.close()
    corrupted_store = V3RuntimeStore(str(corrupted_dir))
    assert corrupted_store.verify_journal_chain() == event_id
    with pytest.raises(V3JournalIntegrityError):
        corrupted_store.replay_journal()


def test_hash_valid_event_reorder_forward_reference_corruption(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    store.commit_tick(transaction, store.continuation_proof())

    corrupted_dir = tmp_path / "corrupted_reorder_store"
    corrupted_dir.mkdir()
    _copy_db(store.path, corrupted_dir / "runtime-v3.sqlite3")
    conn = sqlite3.connect(str(corrupted_dir / "runtime-v3.sqlite3"))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT seq, event_id, event_type, record_id, canonical_json FROM journal ORDER BY seq"
    ).fetchall()
    assert len(rows) >= 2
    a, b = rows[-2], rows[-1]
    previous = conn.execute(
        "SELECT event_id FROM journal WHERE seq=?", (a["seq"] - 1,)
    ).fetchone()
    prev = previous["event_id"] if previous else None
    sequence = [
        (b["canonical_json"], b["event_type"], b["record_id"]),
        (a["canonical_json"], a["event_type"], a["record_id"]),
    ]
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_delete")
    conn.execute("DELETE FROM journal WHERE seq >= ?", (a["seq"],))
    for text, event_type, record_id in sequence:
        event_id = _canonical_hash_event(
            prev, event_type, record_id, parse_json_object(text)
        )
        conn.execute(
            "INSERT INTO journal(event_id, previous_event_id, event_type, record_id, canonical_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_id, prev, event_type, record_id, text),
        )
        prev = event_id
    conn.commit()
    conn.close()
    corrupted_store = V3RuntimeStore(str(corrupted_dir))
    assert corrupted_store.verify_journal_chain() is not None
    with pytest.raises(V3JournalIntegrityError):
        corrupted_store.replay_journal()


def test_hash_valid_missing_artifact_corruption(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    store.commit_tick(transaction, store.continuation_proof())

    corrupted_dir = tmp_path / "corrupted_missing_store"
    corrupted_dir.mkdir()
    _copy_db(store.path, corrupted_dir / "runtime-v3.sqlite3")
    conn = sqlite3.connect(str(corrupted_dir / "runtime-v3.sqlite3"))
    conn.row_factory = sqlite3.Row
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_delete")
    first_private = transaction.candidate_private_states[0].state_leaf_id
    conn.execute(
        "DELETE FROM journal WHERE event_type='private_state' AND record_id=?",
        (first_private,),
    )
    conn.commit()
    conn.close()
    corrupted_store = V3RuntimeStore(str(corrupted_dir))
    with pytest.raises(V3JournalIntegrityError):
        corrupted_store.replay_journal()


def test_hash_valid_reused_artifact_corruption(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    store.commit_tick(transaction, store.continuation_proof())

    corrupted_dir = tmp_path / "corrupted_reused_store"
    corrupted_dir.mkdir()
    _copy_db(store.path, corrupted_dir / "runtime-v3.sqlite3")
    conn = sqlite3.connect(str(corrupted_dir / "runtime-v3.sqlite3"))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT event_id, previous_event_id, record_id, canonical_json FROM journal "
        "WHERE event_type='private_state' LIMIT 1"
    ).fetchone()
    previous = conn.execute(
        "SELECT event_id FROM journal ORDER BY seq DESC LIMIT 1"
    ).fetchone()["event_id"]
    event_id = _canonical_hash_event(
        previous, "private_state", row["record_id"], parse_json_object(row["canonical_json"])
    )
    conn.execute(
        "INSERT INTO journal(event_id, previous_event_id, event_type, record_id, canonical_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (event_id, previous, "private_state", row["record_id"], row["canonical_json"]),
    )
    conn.commit()
    conn.close()
    corrupted_store = V3RuntimeStore(str(corrupted_dir))
    with pytest.raises(V3JournalIntegrityError):
        corrupted_store.replay_journal()


def test_mutable_head_drift_rejection(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    store.commit_tick(transaction, store.continuation_proof())
    conn = sqlite3.connect(str(store.path))
    conn.execute(
        "UPDATE runtime_head SET field_id='0'*64, head_id='0'*64 WHERE singleton=1"
    )
    conn.commit()
    conn.close()
    with pytest.raises(V3JournalIntegrityError):
        store.recover()


def test_v2_store_root_refusal(tmp_path: Path) -> None:
    (tmp_path / "runtime.sqlite3").touch()
    with pytest.raises(V3StoreError, match="v2 store exists"):
        V3RuntimeStore(str(tmp_path))


def test_append_only_trigger_enforcement(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    conn = sqlite3.connect(str(store.path))
    with pytest.raises(sqlite3.IntegrityError, match="journal is append-only"):
        conn.execute("DELETE FROM journal WHERE seq=1")
    with pytest.raises(sqlite3.IntegrityError, match="journal is append-only"):
        conn.execute("UPDATE journal SET record_id='x' WHERE seq=1")
    conn.close()


# ---------------------------------------------------------------------------
# Artifact rejection and lifecycle
# ---------------------------------------------------------------------------


def test_artifact_rejection_changes_no_semantic_state(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    proof = store.continuation_proof()
    rejection = ArtifactRejectionRecord(
        artifact_kind="raw_pass_payload",
        reason="synthetic rejection",
        source_context={"test": True},
        raw_fingerprint_sha256="a" * 64,
    )
    head_after = store.append_artifact_rejection(rejection, proof)
    assert head_after.tick_seq == head.tick_seq
    assert head_after.snapshot == head.snapshot
    assert head_after.core_state_leaf_ids == head.core_state_leaf_ids
    assert head_after.soul_sha256_by_core == head.soul_sha256_by_core
    assert head_after.last_commit_id == head.last_commit_id
    assert head_after.journal_tip_event_id != head.journal_tip_event_id
    assert head_after.head_id != head.head_id


# ---------------------------------------------------------------------------
# Store plumbing
# ---------------------------------------------------------------------------


def test_context_manager_and_exact_database_filename(tmp_path: Path) -> None:
    with V3RuntimeStore(str(tmp_path)) as store:
        _init_store(store)
        assert store.path == tmp_path / "runtime-v3.sqlite3"
        assert store.path.exists()
    with pytest.raises(sqlite3.ProgrammingError):
        store.connection.execute("SELECT 1")


def test_not_initialized_errors(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    with pytest.raises(V3NotInitializedError):
        store.current_head()
    with pytest.raises(V3NotInitializedError):
        store.replay_journal()


def test_already_initialized_error(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    snapshot, states = _make_population()
    with pytest.raises(V3AlreadyInitializedError):
        store.initialize(snapshot, states, _BINDING)


# ---------------------------------------------------------------------------
# Packet 002-R2 adversarial helpers
# ---------------------------------------------------------------------------


def _corrupt_forged_projection(transaction: TickTransactionV3) -> TickTransactionV3:
    """Forge a projection with a different payload but claim the same source."""
    forged = ActiveProjectionArtifactV3(
        source_working_field_id=transaction.projection.source_working_field_id,
        selected_spans=tuple(
            (region, span_id, text + "X", span_hash)
            for region, span_id, text, span_hash in transaction.projection.selected_spans
        ),
        selected_char_count=transaction.projection.selected_char_count + 1,
        source_char_count=transaction.projection.source_char_count + 1,
    )
    return replace(transaction, projection=forged)


def _corrupt_system_update_id(transaction: TickTransactionV3) -> TickTransactionV3:
    """Mutate the system update so its derived ID no longer matches the plan."""
    from runtime.axon_runtime.field_transaction import FieldTransactionAudit

    new_update = replace(
        transaction.system_update,
        evidence=("evidence-tampered",),
    )
    # Also install the tampered update inside the audit so the audit itself is
    # internally consistent but disagrees with the sealed plan/commit.
    forged_audit = replace(transaction.field_audit, system_update=new_update)
    return replace(
        transaction,
        system_update=new_update,
        field_audit=forged_audit,
    )


def _corrupt_audit_before_field(transaction: TickTransactionV3) -> TickTransactionV3:
    """Replace the audit before snapshot so it no longer matches plan.input_field_id."""
    from runtime.axon_runtime.field_transaction import FieldTransactionAudit

    forged_before = SharedFieldSnapshot.from_texts(
        {"response_draft": "forged"},
        tick_id=transaction.plan.tick_seq,
    )
    # Mutate the frozen/slotted audit in-place so the audit_id still matches the
    # commit but the before snapshot no longer matches the plan input field.
    forged_audit = object.__new__(FieldTransactionAudit)
    for field_name in FieldTransactionAudit.__slots__:
        object.__setattr__(forged_audit, field_name, getattr(transaction.field_audit, field_name))
    object.__setattr__(forged_audit, "before_snapshot", forged_before)
    return replace(transaction, field_audit=forged_audit)


def _corrupt_final_delta_author(transaction: TickTransactionV3) -> TickTransactionV3:
    """Swap the FINAL delta author to a non-consolidator online core."""
    final_delta = transaction.consolidation.final_delta_id
    deltas = list(transaction.deltas)
    for index, delta in enumerate(deltas):
        if delta.delta_id == final_delta:
            deltas[index] = FieldDelta(
                base_field_id=delta.base_field_id,
                base_tick_id=delta.base_tick_id,
                author_core_id=transaction.plan.online_core_ids[1],
                pass_id=delta.pass_id,
                operations=delta.operations,
                evidence=delta.evidence,
            )
    return replace(transaction, deltas=tuple(deltas))


def _corrupt_offline_input_missing(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    online = transaction.plan.online_core_ids
    return replace(
        transaction,
        input_private_states=tuple(
            state for state in transaction.input_private_states if state.core_id in online
        ),
    )


def _corrupt_offline_input_changed(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    """Replace the offline input with a state for a different core identity."""
    offline = transaction.plan.offline_core_id
    changed = PrivateStateArtifactV3(
        core_id="offline-impostor",
        tick_seq=0,
        phase="GENESIS",
        substep=0,
        soul_sha256="a" * 64,
        cursor_state_sha256="b" * 64,
        rng_state_sha256=None,
        model_binding_epoch_id=transaction.plan.model_binding_epoch_id,
        parent_state_leaf_id=None,
        working_field_id=None,
        board_id=None,
        delta_id=None,
    )
    new_inputs = tuple(
        changed if state.core_id == offline else state
        for state in transaction.input_private_states
    )
    return replace(transaction, input_private_states=new_inputs)


def _corrupt_offline_participation_v2(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    """Make the offline core author an INITIAL pass."""
    passes = list(transaction.initial_passes)
    passes[0] = replace(passes[0], core_id=transaction.plan.offline_core_id)
    return replace(transaction, initial_passes=tuple(passes))


def _corrupt_page_characters_keep_length(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    """Same-length character mutation on the first page (no ID recomputation)."""
    page = transaction.read_pages[0]
    if not page.characters:
        new_chars = "x"
    else:
        new_chars = page.characters[:-1] + ("X" if page.characters[-1] != "X" else "Y")
    corrupted = replace(page, characters=new_chars)
    pages = (corrupted,) + transaction.read_pages[1:]
    return replace(transaction, read_pages=pages)


def _recompute_page(page: Any, characters: str) -> Any:
    from runtime.axon_runtime.v3_transaction import _cursor_after

    cursor_before = page.cursor_before_hash
    cursor_after = _cursor_after(cursor_before, characters, page.phase, page.page_index)
    return replace(
        page,
        characters=characters,
        cursor_after_hash=cursor_after,
    )


def _corrupt_same_length_with_recomputed_ids(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    """
    Mutate the FINAL page characters to a same-length string, then recompute the
    final page, cycle, transition, candidate, disposition, consolidation and
    commit IDs so the attack is hash-valid everywhere except the reconstructed
    effective view.
    """
    from runtime.axon_runtime.v3_transaction import _cursor_after
    from runtime.field import canonical_sha256

    final_transition = next(t for t in transaction.soul_transitions if t.phase == "FINAL")
    final_cycle = next(c for c in transaction.read_cycles if c.cycle_id == final_transition.read_cycle_id)
    final_pages = sorted(
        [p for p in transaction.read_pages if p.page_id in final_cycle.page_view_hashes],
        key=lambda p: p.page_index,
    )
    if not final_pages:
        raise RuntimeError("no final pages")
    target_page = final_pages[-1]
    new_chars = "".join(
        "\x00" if ch != "\x00" else "\x01" for ch in target_page.characters
    ) or "x"
    if len(new_chars) != len(target_page.characters):
        new_chars = target_page.characters[:-1] + ("Z" if target_page.characters[-1] != "Z" else "Y")

    # Recompute the mutated page and its cursor chain.
    recomputed_pages: list[Any] = []
    cursor_before = target_page.cursor_before_hash
    for page in final_pages:
        if page.page_index == target_page.page_index:
            chars = new_chars
        else:
            chars = page.characters
        cursor_after = _cursor_after(cursor_before, chars, page.phase, page.page_index)
        recomputed_pages.append(
            replace(
                page,
                characters=chars,
                cursor_before_hash=cursor_before,
                cursor_after_hash=cursor_after,
            )
        )
        cursor_before = cursor_after

    # Build id-to-page mapping.
    old_to_new_page = {
        old.page_id: new for old, new in zip(final_pages, recomputed_pages)
    }
    new_page_ids = tuple(p.page_id for p in recomputed_pages)

    # Recompute cycle.
    new_cycle = replace(
        final_cycle,
        page_view_hashes=new_page_ids,
        page_character_counts=tuple(len(p.characters) for p in recomputed_pages),
        cursor_chain=tuple(p.cursor_after_hash for p in recomputed_pages),
    )

    # Recompute final candidate.
    final_candidate = next(
        c for c in transaction.candidate_private_states
        if c.state_leaf_id == final_transition.candidate_private_state_id
    )
    new_candidate = replace(
        final_candidate,
        cursor_state_sha256=recomputed_pages[-1].cursor_after_hash,
    )

    # Recompute final transition.
    new_final_transition = replace(
        final_transition,
        read_cycle_id=new_cycle.cycle_id,
        candidate_private_state_id=new_candidate.state_leaf_id,
        cursor_state_sha256=new_candidate.cursor_state_sha256,
    )

    # Recompute consolidation.
    new_consolidation = replace(
        transaction.consolidation,
        final_soul_transition_id=new_final_transition.transition_id,
    )

    # Recompute disposition for final transition.
    new_dispositions = []
    for disp in transaction.dispositions:
        if disp.transition_id == final_transition.transition_id:
            new_dispositions.append(
                replace(
                    disp,
                    transition_id=new_final_transition.transition_id,
                )
            )
        else:
            new_dispositions.append(disp)

    # Recompute commit references.
    new_disp_by_transition = {
        disp.transition_id: disp.disposition_id for disp in new_dispositions
    }
    new_final_leaves = []
    for core_id, leaf_id in transaction.commit.final_core_state_leaf_ids:
        if core_id == transaction.plan.consolidator_core_id:
            new_final_leaves.append((core_id, new_candidate.state_leaf_id))
        else:
            new_final_leaves.append((core_id, leaf_id))

    new_commit = replace(
        transaction.commit,
        consolidation_id=new_consolidation.consolidation_id,
        disposition_ids_by_transition=tuple(sorted(new_disp_by_transition.items())),
        final_core_state_leaf_ids=tuple(sorted(new_final_leaves)),
    )

    # Assemble new collections.
    new_read_pages = tuple(
        old_to_new_page.get(p.page_id, p) for p in transaction.read_pages
    )
    new_read_cycles = tuple(
        new_cycle if c.cycle_id == final_cycle.cycle_id else c
        for c in transaction.read_cycles
    )
    new_candidates = tuple(
        new_candidate if c.state_leaf_id == final_candidate.state_leaf_id else c
        for c in transaction.candidate_private_states
    )
    new_transitions = tuple(
        new_final_transition if t.transition_id == final_transition.transition_id else t
        for t in transaction.soul_transitions
    )

    return replace(
        transaction,
        read_pages=new_read_pages,
        read_cycles=new_read_cycles,
        candidate_private_states=new_candidates,
        soul_transitions=new_transitions,
        consolidation=new_consolidation,
        dispositions=tuple(new_dispositions),
        commit=new_commit,
    )


def _corrupt_arbitrary_cursor_relink(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    """
    Replace a page cursor_after_hash with a different valid-looking hash and
    consistently relink the tail so the cursor chain appears valid by linkage
    alone; the recomputation check catches it.
    """
    pages = list(transaction.read_pages)
    if len(pages) < 2:
        raise RuntimeError("need at least two pages")
    arbitrary_hash = "a" * 64
    mutated = replace(pages[0], cursor_after_hash=arbitrary_hash)
    pages[0] = mutated
    # Relink page 1's cursor_before to the arbitrary hash.
    page1 = replace(pages[1], cursor_before_hash=arbitrary_hash)
    pages[1] = page1
    return replace(transaction, read_pages=tuple(pages))


# ---------------------------------------------------------------------------
# Packet 002-R2 adversarial tests
# ---------------------------------------------------------------------------


def test_forged_active_projection_rejected(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    forged = _corrupt_forged_projection(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(forged)

    store = _open_store(tmp_path)
    _init_store(store)
    proof = store.continuation_proof()
    journal_before = _journal_canonical_bytes(store)
    head_before = store.current_head()
    with pytest.raises(V3TransactionError):
        store.commit_tick(forged, proof)
    assert _journal_canonical_bytes(store) == journal_before
    assert store.current_head() == head_before


@pytest.mark.parametrize(
    "corruptor",
    [
        _corrupt_system_update_id,
        _corrupt_audit_before_field,
        _corrupt_final_delta_author,
    ],
)
def test_mismatched_transaction_audit_plan_rejected(
    tmp_path: Path, corruptor: Any
) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = corruptor(transaction)
    with pytest.raises((V3TransactionError, V3ContractError)):
        validate_tick_transaction_v3(corrupted)


def test_seed_seven_refine_final_payloads(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(
        head, states, _make_spec(accepted=True, seed=7)
    )
    online = transaction.plan.online_core_ids
    projection_text = transaction.projection.projection_text

    initial_payloads = []
    for pass_record in transaction.initial_passes:
        delta = next(d for d in transaction.deltas if d.delta_id == pass_record.delta_id)
        initial_payloads.append((pass_record.core_id, delta.operations[0].text))
    refine_payloads = []
    for pass_record in transaction.refine_passes:
        delta = next(d for d in transaction.deltas if d.delta_id == pass_record.delta_id)
        refine_payloads.append((pass_record.core_id, delta.operations[0].text))

    for cycle in transaction.read_cycles:
        if cycle.phase == "INITIAL":
            expected = projection_text
        elif cycle.phase == "REFINE":
            peers = [(c, p) for c, p in initial_payloads if c != cycle.core_id]
            expected = projection_text + "".join(p for _, p in sorted(peers))
        else:  # FINAL
            expected = projection_text + "".join(
                p for _, p in sorted(refine_payloads)
            )
        actual = "".join(
            p.characters
            for p in transaction.read_pages
            if p.page_id in cycle.page_view_hashes
        )
        assert actual == expected, f"{cycle.phase} effective text mismatch"
        if cycle.phase in ("REFINE", "FINAL"):
            assert ":7]" in actual, f"{cycle.phase} missing seed 7 payload"


def test_same_length_page_mutation_recomputed_ids_rejected(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_same_length_with_recomputed_ids(transaction)
    # The corrupted transaction still passes raw linkage and ID checks, but
    # validation independently reconstructs the effective view from deltas and
    # rejects the mutated page text.
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


def test_arbitrary_cursor_after_relinked_tail_rejected(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_arbitrary_cursor_relink(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


@pytest.mark.parametrize(
    "corruptor",
    [
        _corrupt_offline_input_missing,
        _corrupt_offline_input_changed,
        _corrupt_offline_participation_v2,
    ],
)
def test_offline_input_missing_changed_or_participating_rejected(
    tmp_path: Path, corruptor: Any
) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = corruptor(transaction)
    with pytest.raises((V3TransactionError, V3PopulationMismatchError)):
        validate_tick_transaction_v3(corrupted)


def _open_raw_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def test_strict_custom_parser_rejects_extra_key_and_wrong_schema(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    conn = _open_raw_connection(store.path)
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_update")
    row = conn.execute(
        "SELECT seq, record_id, canonical_json FROM journal WHERE event_type='private_state' LIMIT 1"
    ).fetchone()
    payload = parse_json_object(row["canonical_json"])
    payload["extra_malicious_key"] = "evil"
    conn.execute(
        "UPDATE journal SET canonical_json=? WHERE seq=?",
        (canonical_json_text(payload), row["seq"]),
    )
    conn.commit()
    conn.close()
    corrupted = V3RuntimeStore(str(tmp_path))
    with pytest.raises(V3JournalIntegrityError):
        corrupted.replay_journal()


def test_strict_custom_parser_rejects_wrong_schema(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    conn = _open_raw_connection(store.path)
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_update")
    row = conn.execute(
        "SELECT seq, record_id, canonical_json FROM journal WHERE event_type='genesis' LIMIT 1"
    ).fetchone()
    payload = parse_json_object(row["canonical_json"])
    payload["schema"] = "axon-runtime-genesis-evil"
    conn.execute(
        "UPDATE journal SET canonical_json=? WHERE seq=?",
        (canonical_json_text(payload), row["seq"]),
    )
    conn.commit()
    conn.close()
    corrupted = V3RuntimeStore(str(tmp_path))
    with pytest.raises(V3JournalIntegrityError):
        corrupted.replay_journal()


def test_hash_valid_journal_graph_mutation_rejected_by_full_frame_replay(
    tmp_path: Path,
) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    store.commit_tick(transaction, store.continuation_proof())

    corrupted_dir = tmp_path / "corrupted_graph_store"
    corrupted_dir.mkdir()
    _copy_db(store.path, corrupted_dir / "runtime-v3.sqlite3")
    conn = sqlite3.connect(str(corrupted_dir / "runtime-v3.sqlite3"))
    conn.row_factory = sqlite3.Row
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_update")
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_delete")

    # Mutate a read-page payload (change characters) and recompute the event_id,
    # then relink all subsequent events so the outer hash chain remains valid.
    rows = list(conn.execute(
        "SELECT seq, event_id, previous_event_id, event_type, record_id, canonical_json "
        "FROM journal ORDER BY seq"
    ).fetchall())
    target_index = next(
        i for i, row in enumerate(rows) if row["event_type"] == "read_page"
    )
    target = rows[target_index]
    payload = parse_json_object(target["canonical_json"])
    payload["characters"] = payload["characters"][:-1] + "Z"
    new_text = canonical_json_text(payload)
    previous = rows[target_index - 1]["event_id"] if target_index > 0 else None
    new_event_id = _canonical_hash_event(
        previous, target["event_type"], target["record_id"], parse_json_object(new_text)
    )
    conn.execute(
        "UPDATE journal SET event_id=?, canonical_json=? WHERE seq=?",
        (new_event_id, new_text, target["seq"]),
    )
    prev = new_event_id
    for row in rows[target_index + 1 :]:
        payload = parse_json_object(row["canonical_json"])
        event_id = _canonical_hash_event(
            prev, row["event_type"], row["record_id"], payload
        )
        conn.execute(
            "UPDATE journal SET event_id=?, previous_event_id=? WHERE seq=?",
            (event_id, prev, row["seq"]),
        )
        prev = event_id
    conn.commit()
    conn.close()

    corrupted_store = V3RuntimeStore(str(corrupted_dir))
    assert corrupted_store.verify_journal_chain() is not None
    with pytest.raises(V3JournalIntegrityError):
        corrupted_store.replay_journal()


def test_replay_derives_binding_from_journal_and_rejects_mutable_drift(
    tmp_path: Path,
) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    # Mutate the mutable head's binding epoch; journal genesis still carries the
    # original binding, so replay-derived truth diverges and recover rejects.
    conn = sqlite3.connect(str(store.path))
    conn.execute(
        "UPDATE runtime_head SET model_binding_epoch_id='drifted-binding' WHERE singleton=1"
    )
    conn.commit()
    conn.close()
    with pytest.raises(V3JournalIntegrityError):
        store.recover()


def test_accepted_and_rejected_two_tick_cold_replay_deterministic(
    tmp_path: Path,
) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)

    tx1 = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    head1 = store.commit_tick(tx1, store.continuation_proof())

    states2 = tuple(
        _state_from_head(store, head1, core_id) for core_id, _ in head1.core_state_leaf_ids
    )
    tx2 = compose_synthetic_tick_v3(
        head1.snapshot,
        states2,
        _make_spec(accepted=False),
    )
    head2 = store.commit_tick(tx2, store.continuation_proof())

    # Append an artifact-rejection event after the rejected commit.
    rejection = ArtifactRejectionRecord(
        artifact_kind="raw_pass_payload",
        reason="post-reject rejection",
        source_context={"test": True},
        raw_fingerprint_sha256="a" * 64,
    )
    head3 = store.append_artifact_rejection(rejection, store.continuation_proof())

    store2 = _open_store(tmp_path)
    assert store2.replay_journal() == head3
    assert store2.recover() == head3


@pytest.mark.parametrize(
    "corruptor",
    [
        _corrupt_forged_projection,
        _corrupt_system_update_id,
        _corrupt_audit_before_field,
        _corrupt_final_delta_author,
        _corrupt_offline_input_missing,
        _corrupt_offline_input_changed,
        _corrupt_offline_participation_v2,
        _corrupt_page_characters_keep_length,
        _corrupt_arbitrary_cursor_relink,
        _corrupt_same_length_with_recomputed_ids,
    ],
)
def test_every_validation_failure_leaves_journal_and_head_unchanged(
    tmp_path: Path, corruptor: Any
) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    corrupted = corruptor(transaction)
    proof = store.continuation_proof()
    journal_bytes_before = _journal_canonical_bytes(store)
    head_before = store.current_head()
    with pytest.raises((V3TransactionError, V3ContractError, V3JournalIntegrityError)):
        store.commit_tick(corrupted, proof)
    assert _journal_canonical_bytes(store) == journal_bytes_before
    assert store.current_head() == head_before


# ---------------------------------------------------------------------------
# Packet 002-R3 adversarial helpers
# ---------------------------------------------------------------------------


def _corrupt_decorative_system_update(
    transaction: TickTransactionV3,
) -> TickTransactionV3:
    """Replace transaction.system_update with a different content-addressed update
    while leaving audit and plan untouched."""
    forged_update = SystemFieldUpdate(
        base_field_id=transaction.system_update.base_field_id,
        base_tick_id=transaction.system_update.base_tick_id,
        operations=(
            AppendSpans(
                region=LogicalRegion.USER_INPUT,
                spans=(
                    FieldSpan(
                        span_id="user-input-forged",
                        text="user-input-forged",
                        kind="text",
                        source="synthetic",
                        provenance="synthetic-system-update-forged",
                    ),
                ),
            ),
        ),
        evidence=("forged-evidence",),
    )
    assert forged_update.update_id != transaction.plan.system_update_id
    return replace(transaction, system_update=forged_update)


def _corrupt_duplicate_input(transaction: TickTransactionV3) -> TickTransactionV3:
    """Append an identical online input state to input_private_states."""
    inputs = list(transaction.input_private_states)
    online_state = next(s for s in inputs if s.core_id == "axon64-a")
    return replace(transaction, input_private_states=tuple(inputs + [online_state]))


def _corrupt_input_as_list(transaction: TickTransactionV3) -> TickTransactionV3:
    return replace(transaction, input_private_states=list(transaction.input_private_states))


class _PrivateStateSubclass(PrivateStateArtifactV3):
    pass


def _corrupt_input_subclass(transaction: TickTransactionV3) -> TickTransactionV3:
    state = transaction.input_private_states[0]
    subclass_instance = _PrivateStateSubclass(
        core_id=state.core_id,
        tick_seq=state.tick_seq,
        phase=state.phase,
        substep=state.substep,
        soul_sha256=state.soul_sha256,
        cursor_state_sha256=state.cursor_state_sha256,
        rng_state_sha256=state.rng_state_sha256,
        model_binding_epoch_id=state.model_binding_epoch_id,
        parent_state_leaf_id=state.parent_state_leaf_id,
        working_field_id=state.working_field_id,
        board_id=state.board_id,
        delta_id=state.delta_id,
    )
    return replace(
        transaction,
        input_private_states=(subclass_instance,) + transaction.input_private_states[1:],
    )


def _recompute_chain_from_index(
    conn: sqlite3.Connection, start_index: int
) -> None:
    """Recompute event_id/previous_event_id from start_index (1-based seq)."""
    rows = list(
        conn.execute(
            "SELECT seq, event_id, previous_event_id, event_type, record_id, canonical_json "
            "FROM journal ORDER BY seq"
        ).fetchall()
    )
    previous = rows[start_index - 2]["event_id"] if start_index > 1 else None
    for row in rows[start_index - 1 :]:
        payload = parse_json_object(row["canonical_json"])
        event_id = _canonical_hash_event(
            previous, row["event_type"], row["record_id"], payload
        )
        conn.execute(
            "UPDATE journal SET event_id=?, previous_event_id=? WHERE seq=?",
            (event_id, previous, row["seq"]),
        )
        previous = event_id


def _find_event_seq(conn: sqlite3.Connection, event_type: str) -> int:
    row = conn.execute(
        "SELECT seq FROM journal WHERE event_type=? ORDER BY seq LIMIT 1",
        (event_type,),
    ).fetchone()
    assert row is not None
    return int(row["seq"])


def _rewrite_genesis_private_state(
    conn: sqlite3.Connection,
    *,
    target_index: int,
    mutate: Any,
) -> str:
    """Persist a content-addressed genesis-state mutation without changing order."""

    genesis_row = conn.execute(
        "SELECT seq, canonical_json FROM journal WHERE event_type='genesis'"
    ).fetchone()
    assert genesis_row is not None
    genesis_payload = parse_json_object(genesis_row["canonical_json"])
    private_state_ids = list(genesis_payload["private_state_ids"])
    state_rows = list(
        conn.execute(
            "SELECT seq, record_id, canonical_json FROM journal "
            "WHERE event_type='private_state' AND seq < ? ORDER BY seq",
            (genesis_row["seq"],),
        ).fetchall()
    )
    assert [row["record_id"] for row in state_rows] == private_state_ids
    target_row = state_rows[target_index]
    original_payload = parse_json_object(target_row["canonical_json"])
    lower = private_state_ids[target_index - 1] if target_index > 0 else None
    upper = (
        private_state_ids[target_index + 1]
        if target_index + 1 < len(private_state_ids)
        else None
    )

    for nonce in range(10_000):
        candidate = copy.deepcopy(original_payload)
        mutate(candidate)
        candidate["soul_sha256"] = _sha(f"persisted-genesis-rewrite-{nonce}")
        canonical = {
            key: value for key, value in candidate.items() if key != "state_leaf_id"
        }
        new_record_id = f"private-state-{canonical_sha256(canonical)}"
        if new_record_id in private_state_ids and new_record_id != target_row["record_id"]:
            continue
        if lower is not None and new_record_id <= lower:
            continue
        if upper is not None and new_record_id >= upper:
            continue
        candidate["state_leaf_id"] = new_record_id
        conn.execute(
            "UPDATE journal SET record_id=?, canonical_json=? WHERE seq=?",
            (
                new_record_id,
                canonical_json_text(candidate),
                target_row["seq"],
            ),
        )
        private_state_ids[target_index] = new_record_id
        assert private_state_ids == sorted(private_state_ids)
        genesis_payload["private_state_ids"] = private_state_ids
        conn.execute(
            "UPDATE journal SET canonical_json=? WHERE seq=?",
            (canonical_json_text(genesis_payload), genesis_row["seq"]),
        )
        _recompute_chain_from_index(conn, int(target_row["seq"]))
        return new_record_id

    raise AssertionError("could not preserve canonical genesis prelude order")


def _load_row(conn: sqlite3.Connection, seq: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT seq, event_id, previous_event_id, event_type, record_id, canonical_json "
        "FROM journal WHERE seq=?",
        (seq,),
    ).fetchone()
    assert row is not None
    return row


# ---------------------------------------------------------------------------
# Packet 002-R3 adversarial tests
# ---------------------------------------------------------------------------


def test_decorative_system_update_rejected_by_validator(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_decorative_system_update(transaction)
    with pytest.raises(V3TransactionError, match="system_update"):
        validate_tick_transaction_v3(corrupted)


def test_decorative_system_update_rejected_atomically_by_commit_tick(
    tmp_path: Path,
) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    corrupted = _corrupt_decorative_system_update(transaction)
    proof = store.continuation_proof()
    journal_bytes_before = _journal_canonical_bytes(store)
    head_before = store.current_head()
    with pytest.raises(V3TransactionError):
        store.commit_tick(corrupted, proof)
    assert _journal_canonical_bytes(store) == journal_bytes_before
    assert store.current_head() == head_before
    # Cold recovery of the untouched store remains valid.
    assert store.recover() == head_before


def test_duplicate_input_state_rejected(tmp_path: Path) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = _corrupt_duplicate_input(transaction)
    with pytest.raises(V3TransactionPopulationError):
        validate_tick_transaction_v3(corrupted)


@pytest.mark.parametrize(
    "corruptor,label",
    [
        (_corrupt_input_as_list, "list container"),
        (_corrupt_input_subclass, "subclass element"),
    ],
)
def test_exact_tuple_and_element_type_boundaries_rejected(
    tmp_path: Path, corruptor: Any, label: str
) -> None:
    head, states = _make_population()
    transaction = compose_synthetic_tick_v3(head, states, _make_spec(accepted=True))
    corrupted = corruptor(transaction)
    with pytest.raises(V3TransactionError):
        validate_tick_transaction_v3(corrupted)


@pytest.mark.parametrize(
    "event_type,schema_value",
    [
        ("private_state", "axon-private-state-artifact-evil"),
        ("active_projection", "axon-active-projection-artifact-evil"),
        ("read_page", "axon-read-page-artifact-evil"),
        ("field_transaction_audit", "axon-field-transaction-audit-evil"),
    ],
)
def test_hash_valid_wrong_schema_rejected_by_custom_parser(
    tmp_path: Path, event_type: str, schema_value: str
) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    # Commit one tick so every custom artifact parser has at least one record.
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    store.commit_tick(transaction, store.continuation_proof())
    store.close()

    conn = _open_raw_connection(store.path)
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_update")
    row = conn.execute(
        "SELECT seq, record_id, canonical_json FROM journal WHERE event_type=? LIMIT 1",
        (event_type,),
    ).fetchone()
    assert row is not None, f"no {event_type} record found"
    payload = parse_json_object(row["canonical_json"])
    payload["schema"] = schema_value
    conn.execute(
        "UPDATE journal SET canonical_json=? WHERE seq=?",
        (canonical_json_text(payload), row["seq"]),
    )
    conn.commit()
    _recompute_chain_from_index(conn, row["seq"])
    conn.commit()
    conn.close()
    corrupted = V3RuntimeStore(str(tmp_path))
    assert corrupted.verify_journal_chain() is not None
    with pytest.raises(V3JournalIntegrityError):
        corrupted.replay_journal()


def test_hash_valid_duplicate_genesis_reference_rejected(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    conn = _open_raw_connection(store.path)
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_update")
    row = conn.execute(
        "SELECT seq, record_id, canonical_json FROM journal WHERE event_type='genesis' LIMIT 1"
    ).fetchone()
    payload = parse_json_object(row["canonical_json"])
    ids = list(payload["private_state_ids"])
    ids.append(ids[0])
    payload["private_state_ids"] = ids
    conn.execute(
        "UPDATE journal SET canonical_json=? WHERE seq=?",
        (canonical_json_text(payload), row["seq"]),
    )
    conn.commit()
    _recompute_chain_from_index(conn, row["seq"])
    conn.commit()
    conn.close()
    corrupted = V3RuntimeStore(str(tmp_path))
    with pytest.raises(V3JournalIntegrityError):
        corrupted.replay_journal()


def test_hash_valid_duplicate_genesis_core_rejected(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    store.close()

    conn = _open_raw_connection(store.path)
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_update")
    state_rows = list(
        conn.execute(
            "SELECT canonical_json FROM journal "
            "WHERE event_type='private_state' ORDER BY seq"
        ).fetchall()
    )
    duplicate_core_id = parse_json_object(state_rows[0]["canonical_json"])["core_id"]
    _rewrite_genesis_private_state(
        conn,
        target_index=1,
        mutate=lambda payload: payload.__setitem__("core_id", duplicate_core_id),
    )
    conn.commit()
    conn.close()

    corrupted = V3RuntimeStore(str(tmp_path))
    assert corrupted.verify_journal_chain() is not None
    with pytest.raises(V3JournalIntegrityError, match="duplicate genesis core_id"):
        corrupted.replay_journal()


@pytest.mark.parametrize(
    "changes",
    [
        {"phase": "INITIAL"},
        {"substep": 1},
        {"parent_state_leaf_id": "private-state-" + "0" * 64},
        {"working_field_id": "a" * 64},
        {"board_id": "proposal-board-" + "b" * 64},
        {"delta_id": "field-delta-" + "c" * 64},
    ],
)
def test_initialize_rejects_noncanonical_genesis_context(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    store = _open_store(tmp_path)
    snapshot, states = _make_population()
    corrupted = (replace(states[0], **changes),) + states[1:]
    with pytest.raises(V3StoreError, match="genesis private state"):
        store.initialize(snapshot, corrupted, _BINDING)


def test_hash_valid_nonzero_genesis_substep_rejected(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    store.close()

    conn = _open_raw_connection(store.path)
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_update")
    _rewrite_genesis_private_state(
        conn,
        target_index=1,
        mutate=lambda payload: payload.__setitem__("substep", 7),
    )
    conn.commit()
    conn.close()

    corrupted = V3RuntimeStore(str(tmp_path))
    assert corrupted.verify_journal_chain() is not None
    with pytest.raises(V3JournalIntegrityError, match="substep must be 0"):
        corrupted.replay_journal()


def test_orphan_read_page_after_genesis_rejected_by_replay(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    _init_store(store)
    last = store.connection.execute(
        "SELECT event_id FROM journal ORDER BY seq DESC LIMIT 1"
    ).fetchone()["event_id"]
    page = ReadPageArtifactV3(
        core_id="axon64-a",
        tick_seq=0,
        phase="INITIAL",
        working_field_id="a" * 64,
        projection_id="b" * 64,
        board_id=None,
        page_index=0,
        start_offset=0,
        end_offset=0,
        characters="",
        cursor_before_hash="c" * 64,
        cursor_after_hash="d" * 64,
        total_effective_character_count=0,
    )
    payload = page.to_dict()
    event_id = _canonical_hash_event(last, "read_page", page.page_id, payload)
    store.connection.execute(
        "INSERT INTO journal(event_id, previous_event_id, event_type, record_id, canonical_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (event_id, last, "read_page", page.page_id, canonical_json_text(payload)),
    )
    store.connection.commit()
    store.close()

    reopened = _open_store(tmp_path)
    with pytest.raises(V3JournalIntegrityError):
        reopened.replay_journal()


def test_orphan_semantic_artifact_inside_committed_frame_rejected(
    tmp_path: Path,
) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    transaction = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    store.commit_tick(transaction, store.continuation_proof())

    corrupted_dir = tmp_path / "corrupted_orphan_frame_store"
    corrupted_dir.mkdir()
    _copy_db(store.path, corrupted_dir / "runtime-v3.sqlite3")
    conn = sqlite3.connect(str(corrupted_dir / "runtime-v3.sqlite3"))
    conn.row_factory = sqlite3.Row
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_update")
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_delete")

    # Insert an unreferenced read_page immediately before the commit, making it
    # a genuinely hash-valid event inside the transaction frame rather than a
    # trailing artifact after the final commit.
    commit_row = conn.execute(
        "SELECT seq, previous_event_id FROM journal "
        "WHERE event_type='tick_commit' ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    assert commit_row is not None
    commit_seq = int(commit_row["seq"])
    previous = str(commit_row["previous_event_id"])
    conn.execute("UPDATE journal SET seq=seq+1 WHERE seq=?", (commit_seq,))
    orphan = ReadPageArtifactV3(
        core_id="axon64-a",
        tick_seq=1,
        phase="INITIAL",
        working_field_id=transaction.plan.working_field_id,
        projection_id=transaction.projection.projection_id,
        board_id=None,
        page_index=99,
        start_offset=0,
        end_offset=0,
        characters="",
        cursor_before_hash="c" * 64,
        cursor_after_hash="d" * 64,
        total_effective_character_count=0,
    )
    payload = orphan.to_dict()
    event_id = _canonical_hash_event(
        previous, "read_page", orphan.page_id, payload
    )
    conn.execute(
        "INSERT INTO journal(seq, event_id, previous_event_id, event_type, record_id, canonical_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            commit_seq,
            event_id,
            previous,
            "read_page",
            orphan.page_id,
            canonical_json_text(payload),
        ),
    )
    _recompute_chain_from_index(conn, commit_seq)
    conn.commit()
    conn.close()

    corrupted_store = V3RuntimeStore(str(corrupted_dir))
    assert corrupted_store.verify_journal_chain() is not None
    with pytest.raises(V3JournalIntegrityError, match="tick frame contains orphan"):
        corrupted_store.replay_journal()


def _find_last_event_seq(conn: sqlite3.Connection, event_type: str) -> int:
    row = conn.execute(
        "SELECT seq FROM journal WHERE event_type=? ORDER BY seq DESC LIMIT 1",
        (event_type,),
    ).fetchone()
    assert row is not None
    return int(row["seq"])


def _derived_record_id(prefix: str, payload: dict[str, Any]) -> str:
    """Recompute a v3 content-addressed record ID from its canonical dict."""
    return f"{prefix}-{canonical_sha256(payload)}"


def test_second_frame_references_first_frame_artifact_rejected(
    tmp_path: Path,
) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    tx1 = compose_synthetic_tick_v3(
        head.snapshot,
        tuple(_state_from_head(store, head, core_id) for core_id, _ in head.core_state_leaf_ids),
        _make_spec(accepted=True),
    )
    store.commit_tick(tx1, store.continuation_proof())

    head1 = store.current_head()
    states2 = tuple(
        _state_from_head(store, head1, core_id) for core_id, _ in head1.core_state_leaf_ids
    )
    tx2 = compose_synthetic_tick_v3(
        head1.snapshot,
        states2,
        _make_spec(accepted=True, seed=1),
    )
    store.commit_tick(tx2, store.continuation_proof())

    corrupted_dir = tmp_path / "corrupted_cross_frame_store"
    corrupted_dir.mkdir()
    _copy_db(store.path, corrupted_dir / "runtime-v3.sqlite3")
    conn = sqlite3.connect(str(corrupted_dir / "runtime-v3.sqlite3"))
    conn.row_factory = sqlite3.Row
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_update")
    conn.execute("DROP TRIGGER IF EXISTS journal_reject_delete")

    # Replace the second tick's system_update_id reference with the first tick's
    # system_update_id, so the second frame resolves an artifact from the first frame.
    plan_seq = _find_last_event_seq(conn, "tick_plan")
    plan_row = _load_row(conn, plan_seq)
    plan_payload = parse_json_object(plan_row["canonical_json"])
    first_system_update_id = tx1.system_update.update_id
    plan_payload["system_update_id"] = first_system_update_id
    # Recompute plan_id after mutating payload (canonical dict excludes plan_id).
    plan_payload_no_id = {k: v for k, v in plan_payload.items() if k != "plan_id"}
    new_plan_id = _derived_record_id("tick-phase-plan", plan_payload_no_id)
    plan_payload["plan_id"] = new_plan_id
    new_plan_text = canonical_json_text(plan_payload)
    plan_event_id = _canonical_hash_event(
        plan_row["previous_event_id"], "tick_plan", new_plan_id, parse_json_object(new_plan_text)
    )
    conn.execute(
        "UPDATE journal SET event_id=?, record_id=?, canonical_json=? WHERE seq=?",
        (plan_event_id, new_plan_id, new_plan_text, plan_seq),
    )

    # Update the second commit to reference the new plan_id.
    commit_seq = _find_last_event_seq(conn, "tick_commit")
    commit_row = _load_row(conn, commit_seq)
    commit_payload = parse_json_object(commit_row["canonical_json"])
    commit_payload["plan_id"] = new_plan_id
    commit_payload_no_id = {k: v for k, v in commit_payload.items() if k != "commit_id"}
    new_commit_id = _derived_record_id("tick-commit-v3", commit_payload_no_id)
    commit_payload["commit_id"] = new_commit_id
    new_commit_text = canonical_json_text(commit_payload)
    commit_event_id = _canonical_hash_event(
        commit_row["previous_event_id"],
        "tick_commit",
        new_commit_id,
        parse_json_object(new_commit_text),
    )
    conn.execute(
        "UPDATE journal SET event_id=?, record_id=?, canonical_json=? WHERE seq=?",
        (commit_event_id, new_commit_id, new_commit_text, commit_seq),
    )

    # Recompute the chain from the modified plan onward.
    _recompute_chain_from_index(conn, plan_seq)
    conn.commit()
    conn.close()

    corrupted_store = V3RuntimeStore(str(corrupted_dir))
    assert corrupted_store.verify_journal_chain() is not None
    with pytest.raises(V3JournalIntegrityError, match="frame"):
        corrupted_store.replay_journal()


def test_trailing_artifact_rejection_changes_only_tip_identity(tmp_path: Path) -> None:
    store = _open_store(tmp_path)
    head = _init_store(store)
    proof = store.continuation_proof()
    rejection = ArtifactRejectionRecord(
        artifact_kind="raw_pass_payload",
        reason="trailing rejection",
        source_context={"test": True},
        raw_fingerprint_sha256="a" * 64,
    )
    head_after = store.append_artifact_rejection(rejection, proof)
    assert head_after.tick_seq == head.tick_seq
    assert head_after.snapshot == head.snapshot
    assert head_after.core_state_leaf_ids == head.core_state_leaf_ids
    assert head_after.soul_sha256_by_core == head.soul_sha256_by_core
    assert head_after.last_commit_id == head.last_commit_id
    assert head_after.model_binding_epoch_id == head.model_binding_epoch_id
    assert head_after.journal_tip_event_id != head.journal_tip_event_id
    assert head_after.head_id != head.head_id
    # Replay accepts the trailing rejection.
    assert store.replay_journal() == head_after
    assert store.recover() == head_after
