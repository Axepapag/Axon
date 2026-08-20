# Identity V2 Serde R10: strict supplied-hash and forged-snapshot boundary

## Authority and outcome

Codex independently accepted the bounded R9 configuration repair on 2026-08-04:
the production hook is absent, primitive-only verified authority survives the
post-verification contract-ID race, and hostile ordinary PathLike conversions
normalize correctly. R9 is accepted **only** as a configuration boundary; it
does not accept R1-R8, the v2 serde boundary, an engine, or a runnable v2
runtime. This packet may begin now, but it must never overlap another
supervised Kimi packet. Until Codex separately accepts R10, the complete v2
identity/config/serde slice remains **QUARANTINED** and may not unlock a CPU
smoke, v2 state, transaction/store/view integration, runtime, trainer, corpus,
checkpoint, Kaggle, tool/advisor, or migration task.

Codex independently reproduced two residual persistence-boundary failures in
the R6/R8 v2 serde source:

1. `serialize_shared_field_v2()` compares attacker-supplied
   `snapshot.field_id` / `snapshot.canonical_hash` with `!=`.  A `str` subclass
   overriding `__ne__` to return `False` bypasses the stated tamper-rejection
   gate.  The function happens to emit freshly reconstructed valid bytes, but
   it accepts a forged in-memory declaration instead of failing closed.
2. An `object.__new__(SharedFieldSnapshotV2)` exact-base object missing
   primitive attributes leaks raw `AttributeError` when serialized, rather
   than normalizing the public serde boundary to `SerdeV2Error`.

The result must be a narrow, side-effect-free hardening of the v2 serializer:
it rejects supplied non-exact/malformed hashes before output, reconstructs only
from captured primitive inputs, and converts malformed exact-base snapshot
attribute failures to `SerdeV2Error`.  No state root may be created.

## Required reading and continuity

Read completely before editing:

- this packet;
- `D:\Axon\ops\kimi_packets\identity-v2-contract-r9-no-hook-fspath-normalization-20260804.md`;
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r8_04August.md`;
- the terminal R9 report if it exists;
- `D:\Axon\codex-turn-state.md`;
- `C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md`;
- the latest relevant `D:\Kimmy\kimmy_personal_log.md` entry;
- `runtime\field\serde_v2.py` and relevant current v2 schema/serde tests.

R1 through R8 remain historical quarantines. R9 is a narrowly accepted
configuration repair, not a broad v2 acceptance. Never rewrite earlier reports
to imply acceptance beyond that limited fact.

## Immutable boundaries

Never edit, initialize, delete, or create:

- `D:\Axon\State\axon_runtime`, `D:\Axon\State\axon_runtime_identity_v2`, or
  any other Axon runtime state root;
- protected 64D/128D checkpoints, `dist`, datasets, Kaggle assets, or training
  scripts;
- `runtime\field\schema_v2.py`, v1/v3 modules,
  `ops\axon_runtime.cpu-smoke.json`, or
  `ops\axon_runtime.identity-v2.contract.json`;
- `D:\Kimmy\State\injection\live_bundle.md` directly.

Do not launch a runtime, daemon, training, Kaggle, promotion, tools/advisors,
or migration.  Preserve the dirty worktree.  Do not stage, commit, reset,
clean, or broadly reformat.

## Exact allowed Axon file scope

Create or edit only:

1. `runtime\field\serde_v2.py`
2. `tests\test_shared_field_v2_serde.py`
3. `roundtable\Kimmy_response_to_Codex_identity_v2_serde_r10_04August.md`

The following Kimmy continuity files are additionally allowed:

- append one dated correction to `D:\Kimmy\kimmy_personal_log.md`;
- update `D:\Kimmy\State\injection\04_current_task.md` and
  `D:\Kimmy\State\injection\05_rolling_summary.md`, then run the existing
  watcher to regenerate `live_bundle.md`; never edit `live_bundle.md`
  directly.

No other Axon file may change.  If scope is insufficient, report
`STATUS: BLOCKED_SCOPE` and stop.

## Required repairs

### A. Strict supplied-hash validation before serialization

- Retain the existing exact-base `SharedFieldSnapshotV2` type boundary.
- Capture all primitive snapshot inputs needed for reconstruction and both
  supplied hashes once inside a guarded public serializer boundary.
- Require `field_id` and `canonical_hash` supplied to the serializer to be
  exact built-in `str` instances, lowercase 64-character hexadecimal SHA-256
  values.  Do not use attacker-overridable equality/inequality methods to make
  this decision.
- Reconstruct a fresh exact base snapshot from captured primitives, and compare
  trusted exact strings to its fresh canonical hashes.  Any mismatch must raise
  `SerdeV2Error` before bytes are emitted.
- Serialize using the exact base canonical method on the fresh reconstruction;
  do not trust an instance virtual override.

### B. Normalize forged exact-base snapshot attribute failures

- Any missing/malformed primitive attribute encountered while serializing an
  exact-base forged object must become `SerdeV2Error` with a stable useful
  message.  Do not leak raw `AttributeError`, `KeyError`, `TypeError`, or
  `ValueError` from the public serializer boundary.
- Preserve the existing ordinary behavior and error class for a wrong subclass
  passed as `snapshot` (the explicit exact-base type error may remain
  `TypeError`).
- Do not change v2 schema semantics, region order, identity charter behavior,
  the v1 serializer, or deserialize wire compatibility.

### C. Behavior-first regressions

Add tests proving:

1. A correct ordinary v2 genesis and non-genesis successor still serialize,
   strict-deserialize, and serialize to byte-identical payloads.
2. An exact-base snapshot whose `field_id`, `canonical_hash`, or both are
   replaced with equality-lying `str` subclasses fails closed as
   `SerdeV2Error`; serialization produces no payload.
3. An exact-base snapshot whose supplied hash is an ordinary wrong 64-hex
   string fails closed as before.
4. An exact-base `object.__new__(SharedFieldSnapshotV2)` missing a required
   primitive such as `regions` fails as `SerdeV2Error`, not raw `AttributeError`.
5. The existing EvilSnapshot subclass remains rejected at the exact-base
   serializer boundary.
6. Existing strict JSON duplicate/nonfinite, identity, schema, and v1
   compatibility tests remain green.  No test creates
   `D:\Axon\State\axon_runtime_identity_v2`.

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

Compile `runtime/field/serde_v2.py` and
`runtime/axon_runtime/identity_v2_config.py`.  Independently run a compact
adversarial probe for every R10 case plus valid v2 genesis and successor byte
round trips.  Verify frozen JSON blobs remain exactly
`59d35b0bb5ad74df166d315272fc306e17df6dcf` and
`7ae72883b2c3e059329cf0d464a61ef796b57843`; verify the R9 config source has
not been edited by this packet.  Inspect allowed code/tests/report for trailing
whitespace and final newline.  Do not deliberately create cache or bytecode
files.

## Evidence and stop condition

Create `roundtable\Kimmy_response_to_Codex_identity_v2_serde_r10_04August.md`
with headings:

1. STATUS
2. R1-R8 QUARANTINE, R9 LIMITED ACCEPTANCE, AND R10 SCOPE
3. PRE-EDIT HASHES
4. STRICT SUPPLIED-HASH REPAIR
5. FORGED-OBJECT ERROR-NORMALIZATION REPAIR
6. TESTS AND ADVERSARIAL REPROS
7. CONTINUITY CORRECTION
8. POST-EDIT FILE HASHES AND SCOPE
9. V1/STATE PRESERVATION
10. RISKS AND DEFERRED WORK
11. RECOMMENDED NEXT PACKET

`STATUS` may be `READY_FOR_CODEX_AUDIT` only if every gate passes, frozen
contract/CPU-smoke blobs are byte-identical, all R10 adversarial cases fail at
the required boundary, valid v2 genesis/successor roundtrips pass, no protected
state root exists, regenerated injection bundle mentions R10, report/log exist,
and no out-of-scope Axon source changed.  Otherwise report `QUARANTINE` or
`BLOCKED_SCOPE`.  Stop after reporting.
