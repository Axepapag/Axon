# Phase B Packet 002-R4: read-only acceptance audit

## Purpose

Audit the Codex-corrected Packet 002-R3 candidate independently and then stop.
This is a read-only Axon audit. Do not repair, refactor, format, or integrate
anything. If any gate fails or any new fail-open path is found, preserve exact
evidence and report `STATUS: QUARANTINE`; do not edit the candidate.

## Required continuity

Before acting, read:

- `D:\Axon\roundtable\28July.txt`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_28July.md`
- `D:\Axon\codex-turn-state.md`, including the latest R3 Codex correction
- the latest entry in `D:\Kimmy\kimmy_personal_log.md`

The original Kimi R3 completion claim is superseded. Codex independently found:

1. nonzero `GENESIS` substeps passed initialize plus full replay;
2. the alleged inside-frame orphan fixture appended after the final commit;
3. duplicate-genesis-core coverage was not a persisted rehashed attack;
4. `tests\r3_preflight_exploits.py` exceeded the allowed file scope.

Codex corrected those issues and removed the temporary script.

## Immutable candidate pins

Do not edit these or any other Axon files:

- `runtime\axon_runtime\v3_transaction.py`
  `F2C418393345D97245D26D304A62376F948A4CE0438B7FF3A9EBFDF240EB2F72`
- `runtime\axon_runtime\v3_store.py`
  `CC12A9377AF55606D1AFED229A385C409D364891FD3BAFC2988D1A1787A89A5F`
- `tests\test_axon_runtime_v3_store.py`
  `FC15E8A397E851CCF633D4DD75394FB09E4EAEC4E8C28AD35D72B7C4372B8A13`

Frozen Packet 001 pins must remain:

- `runtime\axon_runtime\v3_contracts.py`
  `4EC4A1632E4EA9002E15034B8CA38E5B0A34E606568029C6334B4E649C4BA08E`
- `runtime\axon_runtime\v3_serde.py`
  `43790A72E20F2B562A9CE52542ED976A2B33BEE603A1D079EC7838D4E81842F3`
- `tests\test_axon_runtime_v3_contracts.py`
  `08DCED04AA8D071634089DDBE724AB40E686FB04B96816ACC3EA3DC1CF40047B`

## Absolute prohibitions

- Make no Axon source, test, config, data, state, schedule, or documentation
  edits. Use temporary directories and stdin/one-shot diagnostics only.
- Do not recreate `tests\r3_preflight_exploits.py` or any other helper under
  `D:\Axon`.
- Do not touch `D:\Axon\State\axon_runtime`, protected checkpoints, Kaggle
  assets, training, promotion, tools/advisors, or any live runtime.
- Do not start Packet 003 or propose code for engine integration.
- Do not edit generated `D:\Kimmy\State\injection\live_bundle.md` directly.

The continuity contract still requires one dated append to
`D:\Kimmy\kimmy_personal_log.md`. Compact Kimi injection source updates are
allowed only to record the audit result; refresh the generated bundle through
the watcher.

## Audit work

### 1. Re-hash before and after

Verify all six pins above before testing and again before reporting. Any drift
is an immediate quarantine.

### 2. Inspect, do not infer from test names

Read the actual validator, custom persisted parsers, initialize path, commit
path, replay graph reconstruction, frame reachability logic, and the new
tests. Confirm with line-level evidence that:

- exact transaction container/element boundaries cannot be hidden by
  dictionary or set conversion;
- `transaction.system_update`, audit update, and plan update ID are one exact
  content-addressed object;
- genesis physical prelude, sorted unique references, unique cores, binding,
  tick zero, `GENESIS`, substep zero, and null transition context are enforced
  by semantic replay;
- journal replay derives current field/tick/binding/population from the journal,
  not mutable `runtime_head`;
- all current-frame referenced semantic events fall strictly between the prior
  commit/genesis and current commit;
- every non-rejection event in the frame is consumed exactly once or is the
  one commit;
- trailing non-rejection events fail and trailing artifact rejection remains
  legal;
- a validation failure leaves journal and mutable head unchanged.

### 3. Independently execute temporary attacks

Do not merely call a test function. Build independent one-shot attacks using
temporary state roots. When a database is mutated, drop append-only triggers
only in its temporary copy, recompute content-addressed record IDs where
needed, update all dependent references, and recompute every downstream event
hash/link.

At minimum prove:

1. in-memory initialize rejects each of: wrong genesis phase, nonzero substep,
   non-null parent, working field, board, and delta;
2. a fully hash-valid persisted nonzero genesis substep is rejected by semantic
   replay while `verify_journal_chain()` succeeds;
3. a fully hash-valid persisted duplicate genesis core is rejected by semantic
   replay while the physical prelude remains sorted and exact;
4. a valid content-addressed orphan `read_page` inserted immediately before a
   commit is rejected specifically as an orphan inside that frame after all
   downstream links are recomputed;
5. the four original R2 failures remain closed: decorative outer update,
   hash-valid wrong schema/duplicate genesis reference, orphan after genesis,
   and duplicate input;
6. cross-frame artifact reuse rejects by the strict lower-bound gate;
7. trailing `artifact_rejection` remains replayable and changes only tip/head
   identity, while another trailing semantic artifact fails;
8. changing mutable `runtime_head` cannot change replay truth and is detected
   by recovery.

Use error types/messages to prove the intended semantic gate fired. A stale
event hash, malformed record ID, unsorted prelude, or parser failure is not
evidence for a later frame/reachability gate.

### 4. Required fresh test gates

Run:

```powershell
python -m pytest -p no:cacheprovider tests\test_axon_runtime_v3_store.py tests\test_axon_runtime_v3_contracts.py --tb=short
$runtimeTests = rg --files tests | Where-Object { $_ -match '^tests\\test_axon_runtime_.*\.py$' }
python -m pytest -p no:cacheprovider @runtimeTests --tb=short
python -m pytest -p no:cacheprovider tests\test_conversational_no_gold.py tests\test_conversational_objective.py tests\test_conversational_curriculum.py --tb=short
python -m compileall -q runtime\axon_runtime\v3_transaction.py runtime\axon_runtime\v3_store.py tests\test_axon_runtime_v3_store.py
git diff --check -- runtime/axon_runtime/v3_transaction.py runtime/axon_runtime/v3_store.py tests/test_axon_runtime_v3_store.py
```

Expected current counts are `134`, `281`, and `53`. Count drift is not
automatically failure, but must be explained and the candidate hashes must
still match.

## Acceptance rule

Report `STATUS: ACCEPTANCE_AUDIT_PASS` only if all hashes, line-level
invariants, independent temporary attacks, and test gates pass with no new
trust-boundary concern. Otherwise report `STATUS: QUARANTINE` and include the
smallest exact reproduction. Never repair in this packet.

Your final response and Kimi log entry must contain:

- `STATUS`
- `HASHES`
- `LINE-LEVEL AUDIT`
- `INDEPENDENT ATTACKS`
- `TESTS`
- `FILES CHANGED` (Axon must be `none`)
- `RISKS`
- `NEXT SAFE ACTION`

Stop after reporting. Packet 003 remains out of scope.
