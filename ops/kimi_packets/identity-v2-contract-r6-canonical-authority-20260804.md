# Identity V2 Contract R6: canonical authority and forged-object closure

## Authority and outcome

Codex independently audited completed R5. R5 is **QUARANTINED** despite its
green ordinary suites. Do not promote it and do not start transaction/store,
runtime, trainer, corpus, checkpoint, or Kaggle work.

This is the sole authorized repair of the pure isolated `shared-field-v2`
identity contract. It must close these independently reproduced fail-open cases:

1. A `str` subclass with a lying iterator bypasses the frozen 95-character
   substrate check. It can construct/serialize a NUL charter that strict serde
   later rejects.
2. `object.__new__` forged or `object.__setattr__`-mutated exact-base spans,
   regions, and snapshots serialize malformed states that strict serde rejects.
   `int`/`float` subclasses also bypass primitive canonical checks.
3. `render_core_identity_view_v2` accepts a forged exact-base charter because
   it checks only `type(charter)` and does not reconstruct/validate it.
4. The parser accepts protected-root ADS forms:
   `D:\Axon\State\axon_runtime:identity_v2` and
   `D:\Axon\State\axon_runtime:identity_v2:$DATA`.
5. A normal contract mutated with `object.__setattr__` can emit a changed
   descriptor or, after `resolved_config` replacement, a changed genesis under
   the original `contract_id`.
6. A derived contract can be constructed and dynamically hash an override;
   an `object.__new__` forged exact-base contract can invoke genesis.
7. An empty forged charter leaks `AttributeError` instead of
   `IdentityV2ConfigError`.

R6 may make malformed direct-construction values unrepresentable, but must
preserve ordinary valid v2 genesis and valid non-genesis 64-hex parent behavior.
It remains side-effect-free: no state root may be created.

## Required reading and continuity

Read completely before editing:

- `D:\Axon\ops\kimi_packets\identity-v2-contract-r5-boundary-integrity-20260804.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r1_04August.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r2_04August.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r3_04August.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r4_04August.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r5_04August.md`
- `D:\Axon\codex-turn-state.md`
- `C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md`
- the latest entry in `D:\Kimmy\kimmy_personal_log.md`
- this packet completely.

R1 through R5 are historical quarantines. Do not edit their reports to imply
acceptance.

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
or migration. Preserve the dirty worktree. Do not stage, commit, reset, clean,
or broadly reformat.

## Exact allowed Axon file scope

Create or edit only:

1. `runtime\field\schema_v2.py`
2. `runtime\field\serde_v2.py`
3. `runtime\axon_runtime\identity_v2_config.py`
4. `tests\test_shared_field_v2_schema.py`
5. `tests\test_shared_field_v2_serde.py`
6. `tests\test_shared_field_v2_identity_view.py`
7. `tests\test_axon_runtime_identity_v2_config.py`
8. `roundtable\Kimmy_response_to_Codex_identity_v2_contract_r6_04August.md`

The following Kimmy continuity files are additionally allowed:

- append one dated correction to `D:\Kimmy\kimmy_personal_log.md`;
- update `D:\Kimmy\State\injection\04_current_task.md` and
  `D:\Kimmy\State\injection\05_rolling_summary.md`, then run the existing
  watcher to regenerate `live_bundle.md`; never write `live_bundle.md`
  directly.

No other Axon file may change. If scope is insufficient, report
`STATUS: BLOCKED_SCOPE` and stop.

## Required repairs

### A. Exact canonical primitives and reconstructed value graphs

At every canonical value boundary require exact built-in primitive types, not
`isinstance` subclasses, whenever iteration/methods/numeric behavior can affect
validation or hashing. This includes at least canonical strings, integer tick
IDs, float confidence, and all child/reference strings. Do not silently coerce
them. Normal JSON/direct base primitives must continue to work.

Exact class checks alone do not stop `object.__new__` or mutation. When a
`RegionStateV2` receives a `FieldSpanV2`, reconstruct an ordinary exact span
from primitive attributes and store that validated clone. When a
`SharedFieldSnapshotV2` receives a `RegionStateV2`, reconstruct an ordinary
exact region (and thus exact validated spans) and store the clone. Any
inaccessible attribute, invalid primitive, forged base object, mutation, or
inconsistent child must fail before an ordinary snapshot can be constructed.

At `serialize_shared_field_v2`, do not trust an exact-base snapshot merely
because `type(snapshot) is SharedFieldSnapshotV2`. Reconstruct/revalidate a
fresh ordinary snapshot from primitive fields and compare supplied `field_id`
and `canonical_hash` to the validated result before emitting bytes. Serialize
only that fresh snapshot using explicit base-class dispatch. Invalid/forged/
mutated inputs must fail at serialization, never emit bytes rejected later by
strict v2 serde.

The normal `serialize -> deserialize -> serialize` byte identity must remain
true for valid v2 genesis and a valid non-genesis snapshot.

### B. Charter/view reconstruction

Create one small internal safe charter reconstruction path (or equivalent) for
all public charter consumers. It must require the exact base
`IdentityCharterV2`, safely access primitive fields, reconstruct a fresh normal
charter, and convert `AttributeError`, `TypeError`, or `ValueError` from forged
or mutated state into the appropriate fail-closed public error.

`render_core_identity_view_v2` must use this validated clone, not the caller's
object. A forged `object.__new__(IdentityCharterV2)` with missing/invalid text
must fail before a view or hash is produced.

### C. Windows ADS-safe local root

Extend safe local-root validation to reject any colon in a path component other
than the initial local drive token exactly of the form `X:`. This must reject
Windows alternate data streams while preserving normal local absolute roots,
including a nonexistent `tmp_path/.../axon_runtime_identity_v2` test root.

Both of these must reject with `IdentityV2ConfigError` before resolve/commonpath
logic can consider them safe:

- `D:\Axon\State\axon_runtime:identity_v2`
- `D:\Axon\State\axon_runtime:identity_v2:$DATA`

Keep R5's rejection of trailing-space/dot, extended/device, UNC, relative, and
protected-descendant aliases.

### D. Contract-bound descriptor authority

`IdentityV2Contract` must be a sealed authority rather than a set of currently
mutable public fields.

1. At the very beginning of `__post_init__`, reject a derived class with
   `type(self) is not IdentityV2Contract`. Never dynamically dispatch to an
   overridable method while establishing a contract ID.
2. Build one validated descriptor from safe primitive values and a safe cloned
   charter through a non-virtual helper. Bind canonical descriptor bytes and
   `contract_id` at construction. A public descriptor call must return a fresh
   safe descriptor from this bound authority (or fail closed), not re-render
   mutable fields.
3. `to_canonical_descriptor()` and `build_genesis_snapshot()` must use bound
   descriptor authority and verify it hashes to `contract_id` before emitting.
   They must not use mutable `resolved_config` as authority.
4. Mutation of a supplied charter, contract-held charter, `field_schema`, or
   `resolved_config` through `object.__setattr__` must either leave original
   bound output authoritative or fail closed; never emit altered material under
   the original ID.
5. An `object.__new__(IdentityV2Contract)` object without constructor-bound
   authority must make descriptor/genesis APIs raise `IdentityV2ConfigError`,
   not `AttributeError` and not produce a snapshot.

Hostile Python that replaces every private field and the ID coherently can
impersonate any in-memory object. The required boundary is that ordinary
public-field mutation/forgery cannot make APIs emit contradictory material under
an unchanged bound contract ID.

### E. Required behavior-first adversarial tests

Add tests that fail at construction or serializer/API boundary—not only during
later deserialization—for:

- lying `str` text/reference input containing a NUL;
- `int` subclass tick and `float` subclass negative zero;
- forged exact-base `FieldSpanV2`, `RegionStateV2`, `SharedFieldSnapshotV2`,
  and a normal snapshot mutated with `object.__setattr__`;
- forged base charter to renderer and contract, including missing attributes;
- both ADS strings, R5's four aliases, and normal temporary root;
- derived `IdentityV2Contract` overriding `to_canonical_descriptor`;
- forged base contract with manually populated fields/resolved config;
- mutation of charter, field schema, `resolved_config`, and contract-held
  charter cannot alter emitted descriptor/genesis under old ID.

Tests must distinguish correct fail-closed errors and prove no malformed
canonical bytes are emitted.

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

Compile the three v2 modules. Independently run a compact adversarial probe for
every R6 case. Verify valid genesis and valid successor byte roundtrips.
Inspect changed allowed code/tests and report for trailing whitespace/final
newline. Do not deliberately create cache or bytecode files.

## Evidence and stop condition

Create `roundtable\Kimmy_response_to_Codex_identity_v2_contract_r6_04August.md`
with headings:

1. STATUS
2. R1-R5 QUARANTINE AND R6 SCOPE
3. PRE-EDIT HASHES
4. CANONICAL VALUE GRAPH REPAIRS
5. CHARTER, CONTRACT, AND WINDOWS ROOT REPAIRS
6. TESTS AND ADVERSARIAL REPROS
7. CONTINUITY CORRECTION
8. POST-EDIT FILE HASHES AND SCOPE
9. V1/STATE PRESERVATION
10. RISKS AND DEFERRED WORK
11. RECOMMENDED NEXT PACKET

`STATUS` may be `READY_FOR_CODEX_AUDIT` only if every gate passes, frozen
contract/CPU-smoke blobs are byte-identical, every R6 adversarial case fails at
the required boundary, valid genesis/successor roundtrips pass, no protected
state root exists, regenerated injection bundle mentions R6, report/log exist,
and no out-of-scope Axon source changed. Otherwise report `QUARANTINE` or
`BLOCKED_SCOPE`. Stop after reporting.
