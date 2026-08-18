# Phase B Packet 002: isolated v3 transaction, replay, and store

Implement the second neural-free v3 protocol slice. This packet must produce a
complete synthetic three-pass transaction, validate/replay it independently,
persist every artifact in a separate hash-chained SQLite v3 journal, and reject
both ordinary and hash-valid semantic corruption.

This is a bounded implementation packet. Stop after the required files and
tests pass. Do not start an engine, neural inference, training, Kaggle work,
checkpoint work, tools, advisors, or live-state migration.

## Frozen Packet 001 authority

Read and use the current APIs exactly as implemented in:

- `runtime/axon_runtime/v3_contracts.py`
- `runtime/axon_runtime/v3_serde.py`
- `tests/test_axon_runtime_v3_contracts.py`

At packet launch the approved Packet 001 SHA-256 values are:

- `v3_contracts.py`:
  `4EC4A1632E4EA9002E15034B8CA38E5B0A34E606568029C6334B4E649C4BA08E`
- `v3_serde.py`:
  `43790A72E20F2B562A9CE52542ED976A2B33BEE603A1D079EC7838D4E81842F3`
- `test_axon_runtime_v3_contracts.py`:
  `08DCED04AA8D071634089DDBE724AB40E686FB04B96816ACC3EA3DC1CF40047B`

Verify these hashes before editing. If any differs, stop and report the drift.
Do not edit these three files in this packet.

Packet 001 now seals:

- exact online/offline/consolidator population;
- H, the system update, W, and the deterministic no-core-delta successor;
- per-online-core input state-leaf IDs and input soul hashes;
- model-binding epoch;
- W and the correct parent board in every read cycle;
- page hashes, per-page character counts, cursor hashes, and exact expected
  character coverage;
- S0 -> S1 -> S2 and consolidator S2 -> S3 transition lineage;
- all passes, both boards, consolidation, dispositions, installed leaves,
  field-audit/output IDs, and invocation IDs in the commit.

Do not weaken, duplicate, bypass, or silently reinterpret those invariants.

## Files you may create

Create exactly:

- `runtime/axon_runtime/v3_transaction.py`
- `runtime/axon_runtime/v3_store.py`
- `tests/test_axon_runtime_v3_store.py`

Do not modify any other source or test file. In particular, do not modify:

- Packet 001 files;
- v2 `contracts.py`, `serde.py`, `store.py`, `field_transaction.py`, `engine.py`,
  `bootstrap.py`, configuration, or package exports;
- checkpoints, training code, `D:\00`, or live runtime state.

## Required v3 transaction API

In `v3_transaction.py`, define fail-closed, Torch-free types and functions:

```python
class V3TransactionError(V3ContractError): ...
class V3TransactionReplayError(V3TransactionError): ...

@dataclass(frozen=True, slots=True)
class PrivateStateArtifactV3:
    # Exact neural-free metadata/content leaf. Derive state_leaf_id.
    ...

@dataclass(frozen=True, slots=True)
class ActiveProjectionArtifactV3:
    # Exact ordered projection payload over W. Derive raw projection_id SHA.
    ...

@dataclass(frozen=True, slots=True)
class ReadPageArtifactV3:
    # Exact effective-view page and cursor transition. Derive page_id.
    ...

@dataclass(frozen=True, slots=True)
class SyntheticTickSpec:
    online_core_ids: tuple[str, ...]
    offline_core_id: str
    consolidator_core_id: str
    accepted: bool
    seed: int = 0
    completion_order: tuple[str, ...] = ()
    model_binding_epoch_id: str = "synthetic-binding-v1"

@dataclass(frozen=True, slots=True)
class TickTransactionV3:
    system_update: SystemFieldUpdate
    projection: ActiveProjectionArtifactV3
    plan: TickPhasePlan
    input_private_states: tuple[PrivateStateArtifactV3, ...]
    candidate_private_states: tuple[PrivateStateArtifactV3, ...]
    read_pages: tuple[ReadPageArtifactV3, ...]
    read_cycles: tuple[ReadCycleManifest, ...]
    deltas: tuple[FieldDelta, ...]
    soul_transitions: tuple[SoulTransitionRecord, ...]
    initial_passes: tuple[CorePassRecord, ...]
    initial_board: ProposalBoardManifest
    refine_passes: tuple[CorePassRecord, ...]
    refine_board: ProposalBoardManifest
    consolidation: ConsolidationRecord
    dispositions: tuple[SoulTransitionDispositionRecord, ...]
    field_audit: FieldTransactionAudit
    commit: TickCommitRecordV3

@dataclass(frozen=True, slots=True)
class TickReplayV3:
    final_snapshot: SharedFieldSnapshot
    final_core_state_leaf_ids: tuple[tuple[str, str], ...]
    final_soul_sha256_by_core: tuple[tuple[str, str], ...]
    field_transaction_audit_id: str
    commit_id: str

def compose_synthetic_tick_v3(
    head_snapshot: SharedFieldSnapshot,
    all_private_states: Sequence[PrivateStateArtifactV3],
    spec: SyntheticTickSpec,
) -> TickTransactionV3: ...

def validate_tick_transaction_v3(
    transaction: TickTransactionV3,
) -> TickReplayV3: ...

def replay_tick_transaction_v3(
    transaction: TickTransactionV3,
) -> TickReplayV3: ...
```

You may add small private helpers and strict field-audit serialization helpers
needed by the store. Do not invent neural, scheduler, effect, or model behavior.

`PrivateStateArtifactV3` must content-address the exact neural-free state facts
needed to resolve a leaf: core ID, tick/substep/phase, soul SHA-256,
cursor-state SHA-256, RNG-state SHA-256, model-binding epoch, optional parent
state-leaf ID, and the W/board/delta context that produced a candidate. Genesis
input leaves use an explicit genesis phase/context. A transition's
`candidate_private_state_id` is the derived ID of this artifact. Its soul,
cursor, RNG, core, binding, tick, phase/substep, parent, W, board, and delta
must all match the transition. Do not treat an arbitrary matching string as a
resolved state leaf.

`ReadPageArtifactV3` must content-address the exact characters and cursor
progression of one page plus core/tick/phase/W/projection/parent-board identity,
page index, start/end offsets, cursor-before and cursor-after hashes, and total
effective-view character count. `ReadCycleManifest.page_view_hashes` are the
ordered derived page IDs and its cursor chain is the ordered after-cursor
sequence. This artifact may be synthetic, but its characters and hashes must be
replayable evidence rather than asserted counts.

`ActiveProjectionArtifactV3` must carry the exact platform-independent ordered
projection payload reconstructed from W, including region/span/reference order
and exact projected characters. Its `projection_id` is the raw 64-hex canonical
SHA-256 used by Packet 001. The existing active projection implementation uses
a `projection-<sha>` manifest ID and the older v2 head contract is different:
never put either prefixed ID directly into `TickPhasePlan.projection_id`.
If comparing to an existing active projection, recompute the raw canonical hash
and verify it equals the prefix suffix; never blindly strip the prefix.

### Transaction invariants

1. Require exact public record types, ordered sequences, signed 64-bit integer
   bounds, canonical IDs, and no duplicate/orphan artifacts.
2. The plan/transition/commit tick is `head_snapshot.tick_id`, matching the
   same-tick W convention. Only the field transaction final snapshot and the
   published runtime head advance to `head_snapshot.tick_id + 1`.
3. The plan input field is the audit before snapshot. Its system-update ID and W
   exactly match `apply_system_update`. Its sealed no-core-delta output exactly
   matches `compose_tick_transaction(head, update, None).final_snapshot`.
4. The supplied input private-state artifacts have exactly the population
   `online_core_ids + offline_core_id`, have unique resolved leaf IDs, and all
   match the requested model-binding epoch. The plan seals their online
   leaf-ID and soul-hash subsets.
5. Every online core has one INITIAL and one REFINE cycle/transition/delta/pass.
   The consolidator alone has the FINAL cycle/transition/delta.
6. Every actual `FieldDelta` is authored against W and its W tick, appears
   exactly once in `transaction.deltas`, and has the same author/ID at every
   required reference site. INITIAL/REFINE IDs agree in their pass and soul
   transition; FINAL agrees in its transition and consolidation. No other
   artifact may reference them. The FINAL delta author is the consolidator.
   Reject missing, duplicate, extra, stale, donor, or orphan deltas.
7. Call `validate_tick_artifact_graph` and independently validate every actual
   field/delta/audit binding it intentionally cannot inspect.
8. For `accepted=True`, the field audit applies exactly the FINAL delta and the
   commit output is its final snapshot.
9. For `accepted=False`, the FINAL delta remains staged/audited as an artifact,
   but the field audit applies no core delta. The commit uses the plan-sealed
   no-core-delta successor, all online state leaves/soul roots remain exactly
   the plan inputs, every transition is `REJECTED_NOT_INSTALLED`, and invocation
   IDs are empty.
10. Replay with `replay_field_transaction`; require exact audit canonical
    identity, audit ID, final snapshot, output field ID, and commit ID.
11. This neural-free packet has no typed invocation/outbox artifact, so every
    synthetic and store-valid transaction must have
    `invocation_request_ids == ()` in both consolidation and commit. Reject
    nonempty IDs as unresolved references even though Packet 001 reserves
    final-only authority for a later packet. Keep mismatch/nonempty corruption
    tests; execute nothing.
12. `completion_order`, when supplied, must be an exact permutation of online
    cores. It may simulate completion timing but must not change any canonical
    artifact, tuple order, journal order, or ID. Canonical artifact order is
    phase then semantic online-ring order, with mappings/sets sorted as Packet
    001 requires.
13. Reconstruct every effective view from the exact journaled active projection
    over W. INITIAL sees projected W. REFINE core `i` is parent-bound to the
    complete INITIAL board but its peer-delta character section is
    deterministically `B0 minus i`; its own first attempt remains available
    through its private transition/pass and must not be duplicated into the
    peer section. FINAL sees projected W plus every REFINE payload in the
    complete B1 board. Replay pages in exact index/offset order. Require
    contiguous zero-to-end coverage, cursor-before/after linkage, no
    repeated/missing/extra pages, exact characters, page IDs, selected-span
    fingerprint, counts, and terminal cursor. Do not trust manifest counts or
    `coverage_complete=True`. Bind page cursors into the private-state chain:
    INITIAL first cursor-before equals that core's sealed input-state cursor;
    REFINE first cursor-before equals the same core's INITIAL candidate cursor;
    FINAL first cursor-before equals the consolidator's REFINE candidate cursor;
    and each cycle's terminal page cursor-after equals its transition candidate
    artifact cursor. Reject any otherwise-valid spliced cursor chain.
14. Resolve every input and candidate private-state leaf artifact. For
    `COMMITTED`, `SUPERSEDED`, and `REJECTED_NOT_INSTALLED` candidates, verify
    embedded core identity, soul, cursor/RNG state, model binding, parent leaf,
    phase/substep, W, board, and delta against the current transition. For
    `ACCEPTED_NOOP`, Packet 001 points the candidate ID back to the existing
    sealed input artifact: require exact byte/content identity plus matching
    core/soul/cursor/RNG/binding, exempt that old artifact from current
    tick/phase/W/board/delta fields, and do not journal a duplicate candidate
    artifact. Staged rejected candidates remain journaled but never become head
    leaves.
15. Synthetic state leaves, soul hashes, pages, cursor states, and RNG states
    are deterministic content hashes, not random opaque labels. The offline
    core is unchanged.

## Required isolated v3 store API

In `v3_store.py`, define:

```python
V3_STORE_SCHEMA = "axon-runtime-store-v3"
V3_DATABASE_NAME = "runtime-v3.sqlite3"
V3_MAX_JOURNAL_PAYLOAD_BYTES = 8 * 1024 * 1024

class V3StoreError(V3ContractError): ...
class V3AlreadyInitializedError(V3StoreError): ...
class V3NotInitializedError(V3StoreError): ...
class V3StaleContinuationError(V3StoreError): ...
class V3JournalIntegrityError(V3StoreError): ...

@dataclass(frozen=True, slots=True)
class V3RuntimeHead: ...

@dataclass(frozen=True, slots=True)
class V3ContinuationProof: ...

class V3RuntimeStore:
    def __init__(self, state_root: str | Path) -> None: ...
    def close(self) -> None: ...
    def initialize(
        self,
        genesis_snapshot: SharedFieldSnapshot,
        initial_private_states: Sequence[PrivateStateArtifactV3],
        model_binding_epoch_id: str,
    ) -> V3RuntimeHead: ...
    def current_head(self) -> V3RuntimeHead: ...
    def continuation_proof(self) -> V3ContinuationProof: ...
    def commit_tick(
        self,
        transaction: TickTransactionV3,
        expected_continuation: V3ContinuationProof,
    ) -> V3RuntimeHead: ...
    def replay_journal(self) -> V3RuntimeHead: ...
    def recover(self) -> V3RuntimeHead: ...
    def verify_journal_chain(self) -> str | None: ...
    def append_artifact_rejection(
        self,
        rejection: ArtifactRejectionRecord,
        expected_continuation: V3ContinuationProof,
    ) -> V3RuntimeHead: ...
```

Support context-manager use and expose the owned SQLite connection read-only
for integrity tests, as v2 does.

### Store isolation and schema

1. `V3RuntimeStore(state_root)` owns exactly
   `<state_root>\runtime-v3.sqlite3`.
2. If `<state_root>\runtime.sqlite3` exists, refuse the root before creating or
   opening the v3 database. Never migrate, attach, import, or mix v2 rows.
3. Use a new exact schema marker, append-only hash-chained journal, one mutable
   head projection, `PRAGMA foreign_keys=ON`, `synchronous=FULL`, WAL, and
   `BEGIN IMMEDIATE`.
4. Add triggers that reject journal UPDATE and DELETE. Do not create one typed
   table per record; the journal is the authority and the head is a replayable
   projection.
5. Store canonical UTF-8 JSON only. Use Packet 001 strict parsing at every
   persisted JSON boundary. Call `canonical_json_text` only on an already
   validated exact record serialization. Reject payloads larger than
   `V3_MAX_JOURNAL_PAYLOAD_BYTES` before write and before decode/parse, so one
   enormous string cannot bypass structural node budgets. Recompute record IDs
   and event IDs rather than trusting columns.
6. Event IDs must hash the previous event ID, exact event type, record ID, and
   canonical payload. Enforce exact contiguous sequence and previous links.

### Runtime head and continuation proof

`V3RuntimeHead` must immutably and content-addressably seal at least:

- generation and tick sequence;
- exact `SharedFieldSnapshot` and field ID;
- all-population core-state leaf IDs;
- all-population soul SHA-256 values;
- model-binding epoch;
- last commit ID (optional only at genesis);
- current journal-tip event ID;
- derived head ID.

`V3ContinuationProof` must seal all state needed to reject an ABA/stale writer:
head ID, generation, tick, field ID, complete state/soul maps, model binding,
last commit ID, journal tip, and its own derived proof ID.

All maps are exact, sorted, duplicate-free, nonempty, and have the same complete
population. Values are validated with their correct ID/hash rules. Every head
leaf must resolve to a journaled `PrivateStateArtifactV3` whose embedded core,
soul, binding, and lineage agree with the head.

### Initialization, commit, and recovery

1. Genesis must be tick 0 with no parent. Initialize the complete four-core
   synthetic population in one transaction. Journal each resolved genesis
   private-state artifact first, then append a canonical genesis event that
   references them, before publishing the head.
2. `commit_tick` must start `BEGIN IMMEDIATE`, reconstruct/verify current
   authority, compare the complete expected continuation proof, validate and
   replay the candidate transaction, and only then append.
3. The transaction population union (online plus offline) must exactly match
   the head. Plan input field/state/soul/model binding must exactly match the
   current head. Tick must advance exactly once.
4. Journal every object separately in deterministic backward-reference order:

   - system update;
   - active projection artifact;
   - tick plan;
   - for each INITIAL author: read pages, read cycle, field delta, candidate
     private-state artifact, soul transition, pass;
   - INITIAL board;
   - for each REFINE author: read pages, read cycle, field delta, candidate
     private-state artifact, soul transition, pass;
   - REFINE board;
   - FINAL read pages, read cycle, field delta, candidate private-state
     artifact, soul transition;
   - consolidation;
   - every disposition in deterministic transition-ID order;
   - field transaction audit;
   - tick commit last.

5. Every reference must point backward within the same frame or to the sealed
   prior head. Reject unknown, forward, duplicated, reused, missing, orphaned,
   or reordered artifacts even when an attacker recomputes a valid hash chain.
6. Update the snapshot, online state leaves, online soul hashes, generation,
   tick, last commit, and journal tip atomically. Preserve the offline core
   exactly. On any exception, neither journal nor head changes.
7. `append_artifact_rejection` appends one audited rejection event atomically.
   It changes only the journal tip/head identity; field, tick, generation,
   binding, state leaves, soul hashes, and last commit remain unchanged.
8. The only physical pre-genesis journal prelude is exactly one sorted
   `genesis_private_state` event per complete population member, followed by
   exactly one genesis event referencing all of them; no other pre-genesis
   event is legal. Semantic runtime replay starts at that genesis only after
   validating and consuming the complete prelude, then reconstructs every frame
   through exact deserializers and transaction replay and derives the final
   head without trusting the mutable head row.
9. `recover` requires the replayed head and mutable head projection to be
   exactly identical. Head drift is corruption, not permission to silently
   repair.
10. `verify_journal_chain` verifies both cryptographic chaining and semantic
    frame ordering/reachability. Return the exact tip event ID.

## Required tests

Create corruption-heavy tests in `tests/test_axon_runtime_v3_store.py`. Test at
least:

1. deterministic accepted synthetic 3-online/1-offline transaction;
2. deterministic rejected transaction with unchanged neural leaves and no
   accepted core delta;
3. completion-order permutations produce identical artifacts/commit;
4. initialization and exact recovery;
5. accepted commit and restart/replay;
6. rejected commit and restart/replay;
7. two consecutive ticks, with an allowed change of offline/consolidator roles;
8. stale continuation proof rolls back every journal/head write;
9. missing author/pass/delta rejection;
10. incomplete or wrong-board read-cycle rejection;
    include duplicate pages/cursors, gaps/overlap, altered characters/counts,
    missing terminal coverage, and a reused proof after its board changes;
11. offline-core participation rejection;
12. wrong FINAL author or transition-parent rejection;
13. stale W/base-field delta rejection;
14. field-audit/output mismatch rejection;
15. invocation-authority mismatch rejection;
16. rejected candidate-state installation rejection;
    include donor-core, stale-parent, wrong soul/cursor/RNG/binding, arbitrary
    unresolved leaf, and false `ACCEPTED_NOOP` artifact corruption;
17. ordinary event payload/hash/previous-link corruption;
18. hash-valid unknown event type corruption;
19. hash-valid event reorder/forward-reference corruption;
20. hash-valid missing/orphan/reused artifact corruption;
21. mutable-head drift rejection;
22. v2 `runtime.sqlite3` root refusal without touching that file;
23. append-only trigger enforcement;
24. artifact rejection changes no semantic runtime state;
25. context-manager close and exact database filename;
26. failed commit leaves byte-for-byte-equivalent journal rows and head.

For hash-valid semantic corruption tests, deliberately recompute all downstream
event IDs/previous links after the mutation. A test that only breaks the hash is
not evidence that semantic replay rejects a validly rehashed attack.

Use only pytest temporary directories. Never point tests at `D:\Axon` live
state, checkpoint directories, or `D:\00`.

## Validation commands

Run all of:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_axon_runtime_v3_store.py
python -m pytest -q -p no:cacheprovider tests/test_axon_runtime_v3_contracts.py
python -m pytest -q -p no:cacheprovider tests/test_axon_runtime_contracts.py tests/test_axon_runtime_transaction_serde.py tests/test_axon_runtime_store.py tests/test_axon_runtime_field_transaction.py
python -m compileall -q runtime/axon_runtime/v3_transaction.py runtime/axon_runtime/v3_store.py tests/test_axon_runtime_v3_store.py
git diff --check -- runtime/axon_runtime/v3_transaction.py runtime/axon_runtime/v3_store.py tests/test_axon_runtime_v3_store.py
```

Also re-hash the three frozen Packet 001 files after work and prove they are
unchanged.

## Stop conditions

Stop and report without writing outside the three allowed files if:

- any frozen Packet 001 hash differs at preflight;
- a required invariant conflicts with the frozen contract;
- a test would require live-state, v2-store, checkpoint, training, or network
  mutation;
- deterministic replay cannot reject hash-valid semantic reordering.

When finished, report files, hashes, exact tests, residual limitations, and
confirm that no live runtime, v2 file, checkpoint, training job, Kaggle run,
tool, or advisor was touched.
