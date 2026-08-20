# Identity V2 Contract R5: canonical-object, Windows-path, and continuity repair

## Authority and required outcome

Codex independently audited the completed R4 packet.  Its ordinary tests and
scope checks are green, but R4 is **QUARANTINED**.  Do not accept it from its
exit code or its existing test count.

This is one narrow, pure, isolated `shared-field-v2` contract repair.  It must
close these confirmed fail-open boundaries before any v2 transaction/store/view
integration can be considered:

1. On Windows, the protected v1 root can be reached through a trailing-space
   component or an extended path prefix.  These values currently parse even
   though they resolve under `D:\Axon\State\axon_runtime`:
   - `D:\Axon\State\axon_runtime \identity_v2`
   - `\\?\D:\Axon\State\axon_runtime\identity_v2`
2. Schema value subclasses can override canonical methods.  `FieldSpanV2`,
   `RegionStateV2`, `SharedFieldSnapshotV2`, and `CoreIdentityViewV2` can
   currently produce bytes or hashes that strict v2 serde cannot round-trip.
3. An exact-base `IdentityCharterV2` object forged with `object.__new__` can
   bypass its initializer and enter a direct `IdentityV2Contract`.
4. `parent_field_id` is documented as a hash link but currently accepts an
   arbitrary non-empty string.
5. Kimmy's continuity injection sources still describe R3 even though the R4
   report/log exists.  The personal log also has a duplicate R4 entry.  The
   historic records must be preserved, but the current source handoff must be
   corrected and the duplicate explicitly superseded.

Do not implement transactions, an engine, runtime materialization, a corpus,
trainer, neural call, tool, advisor, migration, checkpoint handling, or Kaggle
launch.  This packet does not authorize state creation.

## Required reading and continuity

Read completely before editing:

- `D:\Axon\ops\kimi_packets\identity-v2-contract-r4-20260804.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r1_04August.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r2_04August.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r3_04August.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r4_04August.md`
- `D:\Axon\codex-turn-state.md`
- `C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md`
- the latest entry in `D:\Kimmy\kimmy_personal_log.md`
- this packet completely.

R1 through R4 are historical quarantines.  Do not edit their reports or claim
that any prior packet was accepted.

## Immutable boundaries

Never edit, initialize, delete, or create:

- `D:\Axon\State\axon_runtime`, `D:\Axon\State\axon_runtime_identity_v2`,
  or any other Axon runtime state root;
- protected 64D/128D checkpoints, `dist`, datasets, Kaggle assets, or training
  scripts;
- v1 field modules, v1/v3 axon-runtime modules,
  `ops\axon_runtime.cpu-smoke.json`, or
  `ops\axon_runtime.identity-v2.contract.json`;
- `D:\Kimmy\State\injection\live_bundle.md` directly.

Do not launch a runtime, daemon, training, Kaggle, promotion, tools/advisors,
or migration.  Preserve the dirty worktree.  Do not stage, commit, reset,
clean, or broadly reformat.

## Exact allowed Axon file scope

Create or edit only:

1. `runtime\field\schema_v2.py`
2. `runtime\field\serde_v2.py`
3. `runtime\axon_runtime\identity_v2_config.py`
4. `tests\test_shared_field_v2_schema.py`
5. `tests\test_shared_field_v2_serde.py`
6. `tests\test_shared_field_v2_identity_view.py`
7. `tests\test_axon_runtime_identity_v2_config.py`
8. `roundtable\Kimmy_response_to_Codex_identity_v2_contract_r5_04August.md`

The following Kimmy continuity files are additionally allowed:

- append one dated correction to `D:\Kimmy\kimmy_personal_log.md`;
- update `D:\Kimmy\State\injection\04_current_task.md` and
  `D:\Kimmy\State\injection\05_rolling_summary.md`, then run the existing
  watcher to regenerate `live_bundle.md`; never write `live_bundle.md`
  directly.

No other Axon file may change.  If this scope is insufficient, report
`STATUS: BLOCKED_SCOPE` and stop.

## Required repairs

### A. Exact value types and non-virtual canonical serialization

Treat the following as sealed wire/value classes, not extensibility points:

- `FieldSpanV2`
- `RegionStateV2`
- `SharedFieldSnapshotV2`
- `CoreIdentityViewV2`
- `IdentityCharterV2`

At every public construction and serialization/hash boundary, reject a
subclass before its override can influence validation, hashing, or emitted
bytes.  Use exact base-type checks (`type(value) is ...`) for these classes;
do not use `isinstance` at a boundary where canonical behavior matters.

Within canonical construction, hashing, identity validation, and serde,
dispatch to base-class canonical methods explicitly rather than calling a
potentially virtual method on an input object.  Examples include
`FieldSpanV2.to_canonical_dict(span)`,
`RegionStateV2.to_canonical_dict(region)`,
`SharedFieldSnapshotV2.to_dict(snapshot)`, and
`CoreIdentityViewV2.to_canonical_dict(view, include_envelope=True)` after an
exact-type check.  Preserve valid ordinary base-class construction and the
existing wire shape.

The following attacks must fail before malformed bytes or a forged hash can be
produced:

- an `EvilSpan(FieldSpanV2)` whose `to_canonical_dict()` emits integer
  `confidence=1` and whose `canonical_hash` pretends to match an identity span;
- an `EvilRegion(RegionStateV2)` whose canonical dict replaces spans;
- an `EvilSnapshot(SharedFieldSnapshotV2)` whose `to_dict()` emits a forged
  document;
- an `EvilView(CoreIdentityViewV2)` whose canonical dict is forged.

It is acceptable to defensively reconstruct exact child values from primitive
attributes inside a parent constructor if that is the clearest way to make a
normal initialized value independent from an externally mutable/forged object.
Do not silently coerce malformed values.

### B. Charter and contract integrity

`IdentityV2Contract` must never trust an input charter just because its Python
type says `IdentityCharterV2`.  Validate an exact-base charter through a fresh
ordinary `IdentityCharterV2(text=..., charter_version=...)` reconstruction,
using base-class canonical serialization.  Convert malformed/forged charter
state into `IdentityV2ConfigError` before a contract ID, resolved config, or
genesis snapshot can be produced.

The contract must bind its safe charter descriptor/hash at construction.  A
later mutation attempt through `object.__setattr__` must not cause
`build_genesis_snapshot()` to emit a different charter under the old
`contract_id`; either the original safe cloned charter remains authoritative or
the operation fails closed with `IdentityV2ConfigError`.  Do not add a state
write to accomplish this.

Add behavior-first tests for both:

- a forged exact-base `IdentityCharterV2` made with `object.__new__` and
  incompatible primitive fields;
- mutation of a charter supplied to an already constructed contract, plus an
  attempted mutation of the contract-held charter if applicable.

### C. Parent hash link

Require `SharedFieldSnapshotV2.parent_field_id` to be either `None` or exactly
a 64-character lowercase hexadecimal SHA-256.  For genesis (`tick_id == 0`),
it must be `None`; for a non-genesis snapshot it must be such a hash.  Apply
the same strict rule to raw v2 serde before any declared field hash comparison.
Add direct and rehashed-wire regression tests for arbitrary, uppercase, short,
and missing-parent non-genesis values.  Preserve valid genesis construction.

### D. Windows-safe isolated state root

Harden `_absolute_path`/`_validate_state_root` to work from canonical local
filesystem semantics, not a raw lexical prefix comparison.  The contract is
read-only, so this must not create the candidate directory.

Reject NUL, relative paths, UNC/network paths, extended/device path prefixes
(`\\?\` and `\\.\`), and any component whose Windows semantics change after
trailing spaces or dots are stripped.  Canonicalize only a safe local absolute
path, account for existing symlinks/reparse points with a resolved/real path,
and fail closed if its canonical location is equal to or beneath the protected
v1 root.  Continue allowing a normal, nonexistent absolute isolated root in a
pytest temporary directory and the exact checked-in sample root.

At minimum, parsing must reject all of these when substituted into the sample
contract:

- `D:\Axon\State\axon_runtime \identity_v2`
- `D:\Axon\State\axon_runtime.\identity_v2`
- `\\?\D:\Axon\State\axon_runtime\identity_v2`
- `\\.\D:\Axon\State\axon_runtime\identity_v2`

The normal sample root must still parse without creating it.  Errors must be
`IdentityV2ConfigError`, never a raw `OSError` or `ValueError` leak.

### E. Kimmy continuity correction

Append a dated R5 correction to Kimmy's personal log.  State that R4 is
quarantined for the exact-value, forged-charter, parent-link, and Windows
state-root blockers; explicitly say the two R4 entries are historical duplicate
records and that this R5 entry supersedes their status.  Correct the intended
literal protected/isolation paths without embedding control characters.

Update the two injection *source* files so their current task is R5, not R3 or
R4, and includes the current quarantined status, scope, test expectations, and
the next Codex audit.  Regenerate `live_bundle.md` only via
`python D:\Kimmy\Source\injection_watcher.py --force`.  Verify the regenerated
bundle contains an R5 reference.  Do not delete or rewrite old log entries.

## Required gates

Run the focused suite:

- `tests/test_shared_field_v2_schema.py`
- `tests/test_shared_field_v2_serde.py`
- `tests/test_shared_field_v2_identity_view.py`
- `tests/test_axon_runtime_identity_v2_config.py`

Then run the v1 regression suite:

- `tests/test_shared_field.py`
- `tests/test_axon_runtime_config.py`
- `tests/test_axon_runtime_transaction_serde.py`
- `tests/test_axon_runtime_bootstrap.py`
- `tests/test_axon_runtime_store.py`
- `tests/test_axon_runtime_v3_contracts.py`
- `tests/test_axon_runtime_v3_store.py`
- `tests/test_charslot_roundtrip.py`

Compile all three v2 modules.  Independently run an adversarial probe that
demonstrates each exact-subclass attack, forged charter, stale contract
identity, malformed parent link, and each four state-root aliases fail closed.
Verify serialize -> deserialize -> serialize byte equality for one ordinary
valid v2 genesis snapshot.  Inspect every changed/untracked allowed code/test
file and R5 report for literal trailing whitespace and final newline.  Do not
deliberately create cache or bytecode files.

## Evidence and stop condition

Create `roundtable\Kimmy_response_to_Codex_identity_v2_contract_r5_04August.md`
with these headings:

1. STATUS
2. R1-R4 QUARANTINE AND R5 SCOPE
3. PRE-EDIT HASHES
4. CANONICAL VALUE AND CHARTER REPAIRS
5. WINDOWS PATH AND PARENT-LINK REPAIRS
6. TESTS AND ADVERSARIAL REPROS
7. CONTINUITY CORRECTION
8. POST-EDIT FILE HASHES AND SCOPE
9. V1/STATE PRESERVATION
10. RISKS AND DEFERRED WORK
11. RECOMMENDED NEXT PACKET

`STATUS` may be `READY_FOR_CODEX_AUDIT` only if every gate passes, the contract
and CPU-smoke JSON blobs are byte-identical, all newly listed adversarial cases
fail closed, no protected state root exists, the regenerated injection bundle
mentions R5, the R5 report/log exist, and no out-of-scope Axon source changed.
Otherwise report `QUARANTINE` or `BLOCKED_SCOPE`.  Stop after reporting.
