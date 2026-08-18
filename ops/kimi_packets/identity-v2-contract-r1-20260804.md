# Identity V2 Contract R1: isolated schema, strict serde, and config contract

## Authority and outcome

Jeff authorized Codex to proceed with this bounded implementation slice. Build
only the pure, isolated `shared-field-v2` identity contract. This is not an
engine integration, a runtime migration, a trainer change, or a Kaggle task.

The outcome is a tested, non-runnable contract package that can represent a
sealed canonical identity charter, serialize it strictly, derive a bounded
per-core identity envelope deterministically, and parse an isolated v2
descriptor. It must leave every v1 runtime state and v1 source contract valid.

## Required continuity

Before editing, read:

- `D:\Axon\roundtable\KIMMY_PACKET_2026-08-04_IDENTITY_CONFIG_128D_PREFLIGHT.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_config_04August.md`
- `D:\Axon\codex-turn-state.md`
- `C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md`
- the latest `D:\Kimmy\kimmy_personal_log.md` entry
- this packet completely.

The historical 28July working files are absent. Do not recreate them and do
not block on them.

## Immutable boundaries

Never edit, initialize, or delete:

- `D:\Axon\State\axon_runtime` or any live state root;
- protected 64D/128D checkpoints, `dist/`, `datasets/`, `kaggle/`, or any
  training script;
- `runtime/field/schema.py`, `delta.py`, `view.py`, `schedule.py`, or
  `__init__.py`;
- `runtime/axon_runtime/config.py`, `bootstrap.py`, `engine.py`,
  `projection.py`, `serde.py`, `store.py`, `contracts.py`, or any `v3_*` file;
- `ops/axon_runtime.cpu-smoke.json` and the frozen Packet 002 record;
- generated `D:\Kimmy\State\injection\live_bundle.md`.

Do not launch a runtime, training, a daemon, Kaggle, tools/advisors, or a
checkpoint promotion. Preserve the dirty worktree. Do not stage, commit,
reset, clean, or broadly reformat anything.

## Exact allowed Axon file scope

Create or edit only these Axon paths:

1. `runtime/field/schema_v2.py`
2. `runtime/field/serde_v2.py`
3. `runtime/axon_runtime/identity_v2_config.py`
4. `ops/axon_runtime.identity-v2.contract.json`
5. `tests/test_shared_field_v2_schema.py`
6. `tests/test_shared_field_v2_serde.py`
7. `tests/test_shared_field_v2_identity_view.py`
8. `tests/test_axon_runtime_identity_v2_config.py`
9. `roundtable/Kimmy_response_to_Codex_identity_v2_contract_r1_04August.md`

The required dated append to `D:\Kimmy\kimmy_personal_log.md` is allowed.
No other Axon file may change. If this scope is insufficient, stop and report
`STATUS: BLOCKED_SCOPE`; do not widen it yourself.

## Design contract

### A. V2 schema must be independent

`runtime/field/schema_v2.py` must not alter or import mutable symbols from v1
as if they were v2. It may reuse only pure, stable primitives such as
`canonical_json_bytes`, `canonical_sha256`, and the frozen `ALPHABET_SET`.
Implement explicit v2 types and exports with `V2` suffixes so there is no
accidental v1 substitution:

- `FIELD_SCHEMA_V2 = "shared-field-v2"`
- `LogicalRegionV2`, with v1 values in their exact v1 order/IDs `0..9`, then
  `IDENTITY = "identity"` at ID `10`.
- `CANONICAL_REGION_ORDER_V2`, `LOGICAL_REGION_IDS_V2`,
  `CORE_WRITABLE_REGIONS_V2` (`scratch`, `response_draft` only), and a v2
  sealed-region guard. Do not reuse v1 delta classes or expose a v2 delta API.
- `FieldSpanV2`, `RegionStateV2`, and `SharedFieldSnapshotV2` as immutable,
  strict, canonical-hash-bound dataclasses. Snapshot normalization must require
  exactly eleven unique regions in the fixed v2 order. It must reject duplicate
  names, unknown names, and a missing identity region rather than silently
  filling identity.
- `IdentityCharterV2` with exact canonical payload schema
  `axon-identity-charter-v1`, an explicit positive `charter_version`, nonempty
  text, and a deterministic charter hash. Text and every derived envelope must
  contain only characters in the current frozen 95-character `ALPHABET_SET`.
  The charter must be no more than 512 characters. Do not silently normalize,
  strip, replace, or truncate unsupported text.
- The identity region must contain exactly one charter span, be
  `visibility="attended"`, and be `write_policy="sealed"`. Its span ID and
  source-manifest ID must be derived from the charter hash. Every v2 snapshot
  must have that same charter hash bound into `source_manifest_ids`.
- `CoreIdentityViewV2` and a pure deterministic renderer. Inputs include the
  canonical charter, `core_id`, `display_name`, `model_label`, a sorted unique
  role-capability sequence, and the current role. It outputs a compact,
  ASCII/95-character-compatible envelope, plus a canonical view hash. The
  envelope has a caller-supplied positive character budget no greater than
  192; do not silently truncate. Raise if the required envelope cannot fit.
  Same charter/core/role/descriptor inputs must reproduce exactly; changing
  either core ID or role must change the view hash.

Use this canonical charter text exactly in the sample config and tests:

```text
You are one of Axon's many cores, not Axon as a whole.
You attend to the shared field, propose and review evidence-bound deltas, and improve the response draft across ticks.
You assist Jeffrey Glickman as a trusted digital collaborator under authorized policy.
Dexter version 2 is user-supplied history.
Tools, advisors, browsing, and account actions require separate configured policy and authorized instruction.
```

It is identity/policy text only. It must never grant execution authority,
external account authority, or autonomous tool/advisor effects.

### B. Strict V2 serde

Implement `runtime/field/serde_v2.py` with only strict V2 serialization and
deserialization APIs. It must:

- reject a v1 field payload (`schema == "shared-field-v1"`), unknown keys,
  duplicate JSON keys, nonfinite values, duplicate regions, wrong region order,
  missing identity, an identity span count other than one, identity visibility
  other than attended, mutable identity policy, source-manifest/charter hash
  disagreement, and every field/canonical hash mismatch;
- never call v1 `deserialize_shared_field` and never alter v1 serializers;
- round-trip valid v2 values byte-for-byte through canonical JSON.

### C. Contract-only isolated config

Implement `runtime/axon_runtime/identity_v2_config.py` as a strict,
side-effect-free parser. It must neither create directories nor load models nor
open a database. Its exact top-level schema is
`axon-runtime-identity-v2-contract-v1`, and it must require exactly:

```json
{
  "schema": "axon-runtime-identity-v2-contract-v1",
  "base_v1_config_id": "<64 lowercase hex>",
  "field_schema": "shared-field-v2",
  "state_root": "<absolute isolated path>",
  "identity": {
    "charter_schema": "axon-identity-charter-v1",
    "charter_version": "1",
    "charter_text": "<exact text>",
    "charter_hash": "<canonical charter hash>",
    "charter_max_chars": 512,
    "sealed": true,
    "physical_role": "context",
    "envelope_char_budget": 128
  },
  "migration": {
    "authority": "human_authorized",
    "v1_auto_migration": false
  }
}
```

Requirements:

- hash the canonical full descriptor into `contract_id` and provide a canonical
  resolved-config mapping; reject literal secret field names recursively;
- require `state_root` to be absolute, not equal to or underneath
  `D:\Axon\State\axon_runtime`, and to contain `identity_v2` in its final path;
- require the exact `base_v1_config_id` from the current v1 smoke config:
  `f28d3eadd79cf5b2331ec0373d3fb0ba6bd8f6733cb04b884bd00e1afd2e5036`;
- require `sealed is true`, `physical_role == "context"`, and
  `v1_auto_migration is false`; these are explicit contract constraints, not
  switches that activate a live runtime;
- validate the charter against `IdentityCharterV2`, verify its hash, and expose
  a method that builds a v2 genesis snapshot in memory only;
- provide a `load_identity_v2_contract()` helper that only reads/parses one
  JSON file.

Create `ops/axon_runtime.identity-v2.contract.json` exactly under this schema,
using `D:\Axon\State\axon_runtime_identity_v2` only as an inert declaration.
It must not create that directory.

## Required tests

Write focused, behavior-first tests for every condition above. At minimum prove:

1. v2 has eleven ordered regions and preserves v1 names/IDs zero through nine;
2. v2 genesis binds exactly one sealed attended identity charter and correct
   source-manifest/charter hashes;
3. duplicate, missing, unknown, mutable, or masked identity fails closed;
4. v2 serde round-trip succeeds and v1/malformed/duplicate-key/wrong-order/
   tampered-hash payloads fail;
5. sealed guard rejects identity while only scratch/response remain writable;
6. same clone inputs render the same envelope/hash; different core or role
   produces a different view hash; a too-small budget and an unsupported
   character fail rather than truncate/normalize;
7. config parses the sample descriptor, binds its hash/base ID/charter, does
   not materialize a state root, and rejects root collision, v1 auto migration,
   secret keys, wrong charter hash, and unsupported identity properties.

Run exactly these focused gates after implementation:

```powershell
python -B -m pytest -q -p no:cacheprovider `
  tests/test_shared_field_v2_schema.py `
  tests/test_shared_field_v2_serde.py `
  tests/test_shared_field_v2_identity_view.py `
  tests/test_axon_runtime_identity_v2_config.py

python -B -m pytest -q -p no:cacheprovider `
  tests/test_shared_field.py `
  tests/test_axon_runtime_config.py `
  tests/test_axon_runtime_transaction_serde.py `
  tests/test_axon_runtime_bootstrap.py `
  tests/test_axon_runtime_store.py `
  tests/test_axon_runtime_v3_contracts.py `
  tests/test_axon_runtime_v3_store.py `
  tests/test_charslot_roundtrip.py

python -m py_compile `
  runtime/field/schema_v2.py `
  runtime/field/serde_v2.py `
  runtime/axon_runtime/identity_v2_config.py

git diff --check -- runtime/field/schema_v2.py runtime/field/serde_v2.py runtime/axon_runtime/identity_v2_config.py tests/test_shared_field_v2_schema.py tests/test_shared_field_v2_serde.py tests/test_shared_field_v2_identity_view.py tests/test_axon_runtime_identity_v2_config.py
```

## Evidence report and stop condition

Write `roundtable/Kimmy_response_to_Codex_identity_v2_contract_r1_04August.md`
with headings:

1. `STATUS`
2. `SCOPE AND PRE-EDIT HASHES`
3. `IMPLEMENTATION`
4. `IDENTITY CHARTER AND CONTRACT HASHES`
5. `TESTS`
6. `POST-EDIT DIFF AND V1 PRESERVATION`
7. `FILES CHANGED`
8. `RISKS AND DEFERRED WORK`
9. `RECOMMENDED NEXT PACKET`

Append a dated Kimmy log entry with the same core facts. Refresh injection
sources only if needed, never `live_bundle.md` directly.

Report `STATUS: READY_FOR_CODEX_AUDIT` only if every focused/new and regression
gate passes, no out-of-scope file changed, and neither state root exists due to
this work. Otherwise report `STATUS: QUARANTINE` or `STATUS: BLOCKED_SCOPE`.
Stop after reporting. Do not implement the v2 engine, pager/view integration,
store/replay, config-cleanup, training config, public corpus, or Kaggle launch.
