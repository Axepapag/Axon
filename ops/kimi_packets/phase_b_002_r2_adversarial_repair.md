# Phase B Packet 002-R2: adversarial transaction/store repair

## Role and stopping condition

Act only as a bounded repair engineer for the quarantined Packet 002 candidate.
Do not integrate v3 into the v2 engine/runtime. Repair the isolated neural-free
v3 transaction and journal slice, run every gate below, append the required
Kimi personal-log update, report exact evidence, and stop.

The prior Packet 002 exit code and test count are not acceptance evidence.
Codex independently reproduced two semantic failures:

1. `validate_tick_transaction_v3(replace(tx, projection=forged_projection))`
   succeeds even though the forged projection ID is different from the plan.
2. With `SyntheticTickSpec.seed=7`, REFINE pages contain peer payloads ending in
   `:0]` and do not contain the real `:7]` delta payloads.

Codex also proved the forged projection can pass `commit_tick`, write a mutable
head, and then make cold recovery fail because the plan-referenced projection
was never journaled. Treat all Packet 002 files as quarantined until every gate
in this packet passes.

## Frozen authority

These Packet 001 files must remain byte-identical:

- `runtime/axon_runtime/v3_contracts.py`
  SHA-256 `4EC4A1632E4EA9002E15034B8CA38E5B0A34E606568029C6334B4E649C4BA08E`
- `runtime/axon_runtime/v3_serde.py`
  SHA-256 `43790A72E20F2B562A9CE52542ED976A2B33BEE603A1D079EC7838D4E81842F3`
- `tests/test_axon_runtime_v3_contracts.py`
  SHA-256 `08DCED04AA8D071634089DDBE724AB40E686FB04B96816ACC3EA3DC1CF40047B`

## Allowed edits

Edit only:

- `runtime/axon_runtime/v3_transaction.py`
- `runtime/axon_runtime/v3_store.py`
- `tests/test_axon_runtime_v3_store.py`

Do not modify any other Axon file. In particular, do not touch package exports,
v2 runtime source, training source, live state, checkpoints, Kaggle assets, or
generated Kimi bundles.

## Required transaction repairs

### 1. Seal every outer artifact and field binding

Validation must fail closed unless all of these exact relationships hold:

- The transaction and every typed outer artifact have the exact expected type,
  not a subclass or bool-as-int surrogate.
- `system_update.update_id == plan.system_update_id`.
- `system_update` is exactly the update embedded in `field_audit`.
- `field_audit.before_snapshot` matches the plan input field and tick.
- Independently applying `system_update` to the audit before snapshot produces
  the exact audit working snapshot and `plan.working_field_id`.
- Independently composing the no-core transaction produces exactly
  `plan.no_core_delta_output_field_id`.
- Rebuilding the active projection from the independently derived working
  snapshot produces exactly `transaction.projection`, its ID equals
  `plan.projection_id`, every selected span/hash/text/order is exact, and its
  selected/source character counts are exact.
- Accepted consolidation means the audit contains exactly the plan
  consolidator's declared FINAL delta and final output. Rejected consolidation
  means the audit contains no consolidator delta/author and equals the no-core
  output. The consolidation decision, delta, audit, commit output, and plan
  must not disagree.
- No outer transaction field may be a decorative, unbound object.

### 2. Make read evidence prove the effective view

For each online core and phase, independently reconstruct the expected view:

- INITIAL: exact active projection text.
- REFINE: projection text plus the actual INITIAL peer delta payloads in
  deterministic author order, excluding that core.
- FINAL: projection text plus every actual REFINE delta payload in
  deterministic author order.

The synthetic packet uses exactly one response-draft `InsertText` operation per
delta. Enforce that shape or define and test an equally strict canonical
payload representation. Never regenerate peer text from a seed guess.

Validation must prove:

- Exactly one cycle exists for every required transition and no orphan cycle,
  page, transition, candidate, pass, delta, or disposition exists.
- Page order, offsets, character counts, total length, exact concatenated
  characters, board/projection/working/tick/core/phase bindings, and selected
  span fingerprint are exact.
- Every `cursor_after_hash` is independently recomputed from
  `cursor_before_hash`, exact page characters, phase, and page index.
- Cursor chaining begins at the correct sealed input/candidate state and ends
  at the transition/candidate state.
- A hash-valid fully relinked character mutation is rejected, not merely a
  stale page-ID mutation.

Delete the seed-zero reconstruction path and any unused placeholder grouping.

### 3. Represent the complete population

`TickTransactionV3.input_private_states` must carry the complete sealed
population: exactly three online inputs plus the named offline input for this
synthetic slice. The plan's input leaf/soul maps remain the online set defined
by the frozen Packet 001 contract. Validation and store commit/replay must prove
the union is exact, the offline leaf/soul/binding matches the current head, the
offline core authored no pass/delta/transition/candidate/disposition, and its
head is preserved unchanged.

## Required store/replay repairs

The hash-chained journal is the authority. `replay_journal()` must not recover
state by inspecting only the field audit and final leaf list while ignoring
the v3 graph.

For every commit, reconstruct the complete `TickTransactionV3` from journal
records and the then-current committed input population, and call the strict
transaction validator. Resolve all nested records referenced by the plan,
cycles, passes, boards, transitions, consolidation, dispositions, audit, and
commit. Require every reference to exist exactly once, precede the commit,
belong to the current frame (no cross-tick artifact reuse), and have the right
event type. Derive the next head from the validated replay result.

Also:

- Derive the binding epoch from the journal/genesis private-state authority,
  not from mutable `runtime_head`.
- Require all genesis private states to carry one exact binding matching the
  initialization argument and preserve the exact population.
- Make the custom parsers for genesis, private state, active projection, read
  page, and field audit strict about schema, exact key set, primitive types,
  canonical IDs, duplicate entries, and unknown/trailing fields.
- Enforce the 8 MiB UTF-8 payload cap while verifying/replaying existing rows,
  not only when appending.
- A failed preflight or transaction validation must leave both journal and head
  byte/row-equivalent. The forged-projection transaction must be rejected
  before any journal append.
- Cold recovery after every accepted/rejected commit must reproduce the exact
  head, including an artifact-rejection event after a commit.
- Reject hash-valid semantic corruption even when the attacker recomputes all
  affected artifact IDs and the outer journal hash chain.
- Reject mutable binding/head drift by comparing the complete mutable
  projection to journal-derived truth.

Do not weaken append-only triggers or canonical hashing.

## Required new adversarial tests

Add focused tests that demonstrate at least:

1. forged active projection rejected by validation and atomic `commit_tick`;
2. mismatched transaction system update/audit/plan rejected;
3. seed 7 REFINE and FINAL reads contain exact seed 7 delta payloads;
4. same-length page character mutation with recomputed page/cycle/cursor and
   all dependent candidate/transition/disposition/consolidation/commit IDs is
   rejected because it is not the reconstructed effective view;
5. arbitrary cursor-after hash with a consistently relinked tail is rejected;
6. offline input missing, changed, or participating is rejected;
7. strict custom parser rejects an extra key and wrong schema;
8. hash-valid journal graph mutation/relink is rejected by full frame replay;
9. replay derives binding from journal and rejects mutable binding drift;
10. accepted and rejected two-tick cold replay remains deterministic;
11. every validation failure leaves the journal/head unchanged.

Tests must attack semantic recomputation, not only stale IDs or malformed JSON.

## Validation gates

Run exactly:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_axon_runtime_v3_store.py tests/test_axon_runtime_v3_contracts.py
python -m pytest -q -p no:cacheprovider tests/test_axon_runtime_contracts.py tests/test_axon_runtime_transaction_serde.py tests/test_axon_runtime_store.py tests/test_axon_runtime_field_transaction.py
python -m compileall -q runtime/axon_runtime/v3_transaction.py runtime/axon_runtime/v3_store.py tests/test_axon_runtime_v3_store.py
git diff --check -- runtime/axon_runtime/v3_transaction.py runtime/axon_runtime/v3_store.py tests/test_axon_runtime_v3_store.py
```

Then re-hash all three frozen Packet 001 files and report the exact hashes.
Report individual test counts. Do not report Packet 002-R2 complete if any
semantic requirement above remains untested or any command fails.

## Continuity report

Append a dated `TURN UPDATE` to `D:\Kimmy\kimmy_personal_log.md`. State plainly
that Packet 002-R1 was quarantined by Codex's adversarial checks, list the exact
failure repros, exact files changed, exact tests/results, remaining risks, and
the next recommended packet. Update compact Kimi injection sources only if the
durable status materially changed, refresh them via the watcher if changed,
and never edit `live_bundle.md` directly.
