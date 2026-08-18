# Phase B Packet 002-R3: journal reachability and exact outer bindings

## Status and purpose

Packet 002-R2 remains quarantined. Codex reran its advertised gates
independently (`112`, `34`, and the separate conversational `53` all passed),
then reproduced four additional fail-open attacks against the R2 files:

```text
VULNERABILITY_DECORATIVE_SYSTEM_UPDATE_ACCEPTED
VULNERABILITY_WRONG_SCHEMA_AND_DUPLICATE_GENESIS_ACCEPTED
VULNERABILITY_ORPHAN_SEMANTIC_ARTIFACT_ACCEPTED
VULNERABILITY duplicate_input ACCEPTED
```

This is one narrow repair packet. Fix these exact trust-boundary failures and
the directly implied same-frame reachability rules, test them adversarially,
update Kimi continuity, and stop. Do not propose or begin Packet 003 engine
integration.

## Frozen authority

Do not edit these Packet 001 files; verify they remain byte-identical:

- `runtime/axon_runtime/v3_contracts.py`
  `4EC4A1632E4EA9002E15034B8CA38E5B0A34E606568029C6334B4E649C4BA08E`
- `runtime/axon_runtime/v3_serde.py`
  `43790A72E20F2B562A9CE52542ED976A2B33BEE603A1D079EC7838D4E81842F3`
- `tests/test_axon_runtime_v3_contracts.py`
  `08DCED04AA8D071634089DDBE724AB40E686FB04B96816ACC3EA3DC1CF40047B`

## Allowed Axon edits

Edit only:

- `runtime/axon_runtime/v3_transaction.py`
- `runtime/axon_runtime/v3_store.py`
- `tests/test_axon_runtime_v3_store.py`

The continuity contract still requires appending the Kimi personal log and
allows compact `D:\Kimmy\State\injection\` source refreshes when status changes.
Never edit `live_bundle.md` directly.

## Mandatory preflight reproductions

Before editing, independently reproduce all four exact failures above.

1. Compose a valid transaction, replace only
   `transaction.system_update` with a different content-addressed
   `SystemFieldUpdate`, leave the audit and plan untouched, and show
   `validate_tick_transaction_v3` accepts it.
2. In a temporary v3 store, change a journaled
   `PrivateStateArtifactV3.schema` to an evil schema, duplicate an existing ID
   in `genesis.private_state_ids`, recompute the complete outer event hash
   chain, and show replay accepts it.
3. Append a valid, content-addressed but unreferenced `read_page` semantic
   event to a temporary initialized journal and show replay accepts it.
4. Append an identical online input state to
   `TickTransactionV3.input_private_states` and show validation accepts the
   duplicate even though the core set still appears complete.

Use temporary directories only.

## Transaction repairs

### Exact outer field binding

- Require `transaction.system_update` to be exactly the system update embedded
  in `field_audit`, including content, canonical identity, and update ID.
- Require both to equal `plan.system_update_id`.
- Reconstruct a fresh `FieldTransactionAudit` from its exact six semantic
  fields and require exact equality/canonical identity with the supplied audit,
  not merely matching final field IDs. Reject a frozen object forged with a
  stale `audit_id`.
- Continue independently recomputing H -> U -> W, the no-core successor,
  active projection, accepted/rejected final transaction, and replay.

### Exact complete-population and typed sequences

- `input_private_states` must be an exact tuple of exactly
  `len(online_core_ids) + 1` exact `PrivateStateArtifactV3` objects.
- Reject duplicate core IDs and duplicate state-leaf IDs before building a
  dictionary. Require the exact online-plus-offline population and one common
  plan binding.
- Apply the same exact tuple/container and exact element-type boundary to
  candidate states, read pages, read cycles, deltas, soul transitions,
  INITIAL passes, REFINE passes, and dispositions. Do not allow a subclass or
  decorative extra element to disappear through dictionary/set conversion.
- Keep all existing R2 graph, view, cursor, seed, field, disposition, and
  offline-preservation checks.

## Strict persisted schema repairs

At every custom persisted parser, require both the exact key set and exact
schema value:

- private state: `axon-private-state-artifact-v3`
- active projection: `axon-active-projection-artifact-v3`
- read page: `axon-read-page-artifact-v3`
- field audit: `axon-field-transaction-audit-v1`
- genesis: `axon-runtime-genesis-v3`

Require exact JSON primitive/container types after decoding. In particular,
genesis `private_state_ids` must be an exact list of unique nonempty strings in
the canonical sorted order. Its physical prelude must contain exactly those
private-state record IDs once each, in that same canonical order; embedded
core IDs must also be unique, and every genesis state must have tick zero,
GENESIS phase/context, and the one journal-derived binding.

Wrong schemas, duplicate genesis references, duplicate genesis cores, and
extra keys must be tested after recomputing the complete event hash chain.
A test that merely leaves a stale event hash is not parser evidence.

## Same-frame reachability and orphan rejection

The hash-chained journal is the authority. For each tick commit:

1. Define the frame lower bound as the preceding commit sequence, or genesis
   sequence for the first tick.
2. Every current-tick semantic artifact referenced by the commit graph must
   have exactly one event with:
   `frame_lower_bound < artifact_seq < commit_seq`.
   Prior-head input private states are the only semantic references allowed
   across the lower boundary.
3. Track every event key consumed while reconstructing the transaction.
4. Within the frame, the set of non-`artifact_rejection` semantic events must
   equal exactly the consumed transaction-event set plus that frame's one
   `tick_commit`. Reject an unreferenced page, cycle, delta, state, transition,
   pass, board, update, projection, plan, consolidation, disposition, audit,
   duplicate, or wrong-frame reuse.
5. `artifact_rejection` remains the only legal semantically inert event between
   frames or after the final commit.
6. After the final commit (or after genesis when there are no commits), reject
   every trailing non-rejection semantic event.

Before validating each reconstructed frame during journal replay, explicitly
require:

- plan input field equals the then-current replayed snapshot;
- plan tick equals the then-current replayed tick;
- plan model binding equals the journal-derived binding;
- reconstructed online-plus-offline inputs exactly equal the then-current head
  leaf/soul population.

Do not derive any authority from mutable `runtime_head`.

It is acceptable for `verify_journal_chain()` to remain the cryptographic
chain check if `replay_journal()` is the semantic reachability gate, but tests
and reports must state that distinction accurately. Do not claim
cross-tick/orphan rejection unless the semantic replay actually proves it.

## Required adversarial tests

Add tests that fail on R2 and pass only after the repair:

1. decorative transaction-only system update rejected by validator;
2. the same attack rejected atomically by `commit_tick`, with journal/head
   unchanged and cold recovery still valid;
3. duplicate input state rejected;
4. non-tuple and subclass element boundaries rejected for representative
   transaction collections;
5. hash-valid wrong schema rejected independently for each of the four custom
   artifact parsers;
6. hash-valid duplicate genesis reference rejected;
7. hash-valid duplicate genesis core rejected;
8. valid content-addressed orphan read page after genesis rejected by semantic
   replay;
9. valid content-addressed orphan semantic artifact inside a committed frame
   rejected after the entire outer chain is recomputed;
10. a second frame that references a first-frame semantic artifact is rejected
    specifically by the frame lower-bound check;
11. a trailing artifact rejection remains accepted and changes only the
    journal tip/head identity;
12. all prior R2 same-length/recomputed-ID page attacks, seed-7 views, two-tick
    accepted/rejected replay, strict size cap, atomic rollback, and v2
    regressions remain green.

For database attacks, drop append-only triggers only in pytest temporary
copies, persist the mutation, and recompute every downstream event ID/link.

## Validation gates

Run exactly:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_axon_runtime_v3_store.py tests/test_axon_runtime_v3_contracts.py --tb=short
python -m pytest -q -p no:cacheprovider tests/test_axon_runtime_contracts.py tests/test_axon_runtime_transaction_serde.py tests/test_axon_runtime_store.py tests/test_axon_runtime_field_transaction.py --tb=short
python -m compileall -q runtime/axon_runtime/v3_transaction.py runtime/axon_runtime/v3_store.py tests/test_axon_runtime_v3_store.py
git diff --check -- runtime/axon_runtime/v3_transaction.py runtime/axon_runtime/v3_store.py tests/test_axon_runtime_v3_store.py
```

Then rerun the four preflight exploit scripts and show that each now rejects
for the intended semantic reason. Re-hash all three frozen Packet 001 files.
Report exact test counts from a summary-producing pytest invocation.

## Stop conditions

If complete same-frame reachability cannot be implemented without changing a
frozen Packet 001 or v2 file, stop and report the blocker. Do not weaken a gate,
accept an ignored orphan, start engine integration, launch training/Kaggle,
touch `D:\Axon\State\axon_runtime`, or promote any checkpoint.

Append a dated Kimi personal-log entry which explicitly supersedes the R2
claim of completion and records Codex's four R2 exploit results, the R3
changes, exact test evidence, residual risks, and the next safe step.
