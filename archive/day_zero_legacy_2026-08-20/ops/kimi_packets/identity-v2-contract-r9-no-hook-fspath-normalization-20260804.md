# Identity V2 Contract R9: remove test-only authority hook and normalize hostile PathLike failures

## Authority and outcome

Codex independently audited Kimi R8.  R8 remains **QUARANTINED**, despite
green ordinary tests.  Do not start a CPU smoke, v2 transaction/store/view
integration, runtime, trainer, corpus, checkpoint, Kaggle, tool/advisor, or
migration task from this packet.

This is one narrow configuration-only repair.  Codex reproduced these two
issues directly against the completed R8 source:

1. `IdentityV2Contract.build_genesis_snapshot()` reads a mutable private
   `_trace_hook` after `_verify_bound_authority()` and invokes it.  The R8
   packet allowed this as a regression seam, but this is production test
   plumbing and violates the intended rule that no mutable contract field is
   read after local authority verification.  It also conflicts with Jeff's
   requirement that behaviorally meaningful knobs not be hidden outside the
   versioned contract.
2. `_absolute_path()` catches only `TypeError`, `ValueError`, and
   `RuntimeError` from `os.fspath()`.  A hostile `os.PathLike` whose
   `__fspath__()` raises `OSError` or an ordinary custom `Exception` leaks the
   raw exception through direct `IdentityV2Contract(...)` construction.  The
   public `load_identity_v2_contract(path)` also leaks raw conversion errors
   from its `Path(path)` boundary.  This contradicts the R8 requirement to
   normalize arbitrary conversion failures.

The result must be a side-effect-free v2 identity contract with no production
test hook and no mutable descriptor mapping carried from authority verification
into genesis composition.  Its deterministic race regression must use
test-local `sys.settrace`, and hostile ordinary path-conversion errors must
fail closed as `IdentityV2ConfigError`.  No state root may be created.

## Required reading and continuity

Read completely before editing:

- this packet;
- `D:\Axon\ops\kimi_packets\identity-v2-contract-r8-atomic-authority-canonical-descriptor-20260804.md`;
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r8_04August.md`;
- `D:\Axon\codex-turn-state.md`;
- `C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md`;
- the latest relevant `D:\Kimmy\kimmy_personal_log.md` entry.

R1 through R8 are historical quarantines.  Never rewrite their reports to
imply acceptance.

## Immutable boundaries

Never edit, initialize, delete, or create:

- `D:\Axon\State\axon_runtime`, `D:\Axon\State\axon_runtime_identity_v2`, or
  any other Axon runtime state root;
- protected 64D/128D checkpoints, `dist`, datasets, Kaggle assets, or training
  scripts;
- v1/v3 modules, `runtime\field\schema_v2.py`, `runtime\field\serde_v2.py`,
  `ops\axon_runtime.cpu-smoke.json`, or
  `ops\axon_runtime.identity-v2.contract.json`;
- `D:\Kimmy\State\injection\live_bundle.md` directly.

Do not launch a runtime, daemon, training, Kaggle, promotion, tools/advisors,
or migration.  Preserve the dirty worktree.  Do not stage, commit, reset,
clean, or broadly reformat.

## Exact allowed Axon file scope

Create or edit only:

1. `runtime\axon_runtime\identity_v2_config.py`
2. `tests\test_axon_runtime_identity_v2_config.py`
3. `roundtable\Kimmy_response_to_Codex_identity_v2_contract_r9_04August.md`

The following Kimmy continuity files are additionally allowed:

- append one dated correction to `D:\Kimmy\kimmy_personal_log.md`;
- update `D:\Kimmy\State\injection\04_current_task.md` and
  `D:\Kimmy\State\injection\05_rolling_summary.md`, then run the existing
  watcher to regenerate `live_bundle.md`; never edit `live_bundle.md`
  directly.

No other Axon file may change.  If this scope is insufficient, report
`STATUS: BLOCKED_SCOPE` and stop.

## Required repairs

### A. Remove production test plumbing

- Remove the `Callable` import if it becomes unused.
- Remove the `_trace_hook` dataclass field entirely.
- Remove every production read or invocation of `_trace_hook`.
- `_VerifiedAuthority` must not carry a mutable `dict`, `list`, or a mapping
  backed by such a container into either public API.  Carry canonical
  descriptor text plus the primitive values needed for genesis (at least the
  captured contract ID, base-v1 config ID, and charter text), or an equivalent
  fully immutable primitive-only record.  Recreate a fresh ordinary descriptor
  only for `to_canonical_descriptor()` from the verified canonical local text.
  `build_genesis_snapshot()` must use the captured primitive values rather than
  indexing an externally mutable descriptor mapping.
- After `verified = self._verify_bound_authority()` in
  `build_genesis_snapshot()`, code must use only local values derived from
  `verified`; it must not access any `self.*` field before returning or
  failing.  Do not introduce another callback, environment switch, config key,
  global flag, monkeypatch seam, or alternate mutable test hook.

### B. Full ordinary-exception normalization for direct PathLike conversion

- At the `os.fspath()` conversion boundary in `_absolute_path()`, normalize
  every ordinary `Exception` (including `OSError` and custom exception
  subclasses) to `IdentityV2ConfigError` with the existing safe message.
- Do not catch `BaseException`: `KeyboardInterrupt`, `SystemExit`, and similar
  control-flow exceptions must retain normal Python behavior.
- Preserve ordinary valid `str` and `Path` behavior and the existing
  Windows-safe path restrictions.
- At the public `load_identity_v2_contract(path)` conversion boundary, likewise
  normalize every ordinary exception raised while converting an attacker
  supplied `os.PathLike` to `Path` into `IdentityV2ConfigError`.  Preserve its
  ordinary file-not-found behavior only after a successfully converted ordinary
  local path; do not turn normal missing-file semantics into an unrelated
  contract parse result.

### C. Behavior-first regressions

Add/replace tests proving all of the following:

1. The source has no `_trace_hook` field or production reference, and the
   private verified-authority record does not retain a mutable descriptor
   mapping for later genesis composition.
2. A deterministic test-local `sys.settrace` line hook mutates only
   `contract_id` after `_verify_bound_authority()` returns and before source
   manifest composition.  That invocation either fails closed or returns the
   original captured contract ID and baseline genesis field ID; it must never
   emit the forged ID.  The test must restore the prior trace function even on
   failure and must not depend on a production test seam.
3. A direct `os.PathLike.__fspath__()` that raises `RuntimeError`, `OSError`,
   and a custom ordinary `Exception` each raises `IdentityV2ConfigError`, with
   no raw conversion exception, both through direct contract construction and
   through `load_identity_v2_contract(path)`.  A `KeyboardInterrupt` (or a
   clearly equivalent `BaseException` subclass) must not be swallowed if you
   add this coverage.
4. All R8 adversarial cases remain covered: coherent malformed/noncanonical
   descriptors, duplicate/non-finite JSON, raw and escaped unpaired
   surrogates, text/hash/ID mutation, exact-string checks, ADS rejection,
   sample parsing, descriptor-copy isolation, and v2 genesis/successor byte
   round trips.
5. No test creates `D:\Axon\State\axon_runtime_identity_v2`; temporary test
   roots must remain nonexistent after parse/construct operations.

Do not weaken prior tests merely to make the suite pass.

## Required gates

Run the focused v2 suite:

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

Compile `runtime/axon_runtime/identity_v2_config.py`.  Independently run a
compact adversarial probe for every R9 case plus valid v2 genesis and successor
serialize/deserialise byte round trips.  Verify frozen JSON blobs remain exactly
`59d35b0bb5ad74df166d315272fc306e17df6dcf` and
`7ae72883b2c3e059329cf0d464a61ef796b57843`.  Inspect allowed code/tests/report
for trailing whitespace and final newline.  Do not deliberately create cache or
bytecode files.

## Evidence and stop condition

Create `roundtable\Kimmy_response_to_Codex_identity_v2_contract_r9_04August.md`
with headings:

1. STATUS
2. R1-R8 QUARANTINE AND R9 SCOPE
3. PRE-EDIT HASHES
4. NO-HIDDEN-HOOK REPAIR
5. PATHLIKE EXCEPTION-NORMALIZATION REPAIR
6. TESTS AND ADVERSARIAL REPROS
7. CONTINUITY CORRECTION
8. POST-EDIT FILE HASHES AND SCOPE
9. V1/STATE PRESERVATION
10. RISKS AND DEFERRED WORK
11. RECOMMENDED NEXT PACKET

`STATUS` may be `READY_FOR_CODEX_AUDIT` only if every gate passes, frozen
contract/CPU-smoke blobs are byte-identical, every R9 adversarial case fails at
the required boundary, valid v2 genesis/successor roundtrips pass, no protected
state root exists, regenerated injection bundle mentions R9, report/log exist,
and no out-of-scope Axon source changed.  Otherwise report `QUARANTINE` or
`BLOCKED_SCOPE`.  Stop after reporting.
