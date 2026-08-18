# Identity V2 Contract R2: strict invariant repair and evidence completion

## Authority and outcome

Codex reviewed the completed R1 worker job independently.  R1 remains
quarantined: its required report and Kimmy log append are absent, and its green
tests missed several fail-open contract defects.  This packet is the only
authorized repair.  It remains a pure, isolated `shared-field-v2` contract
package; it is **not** engine integration, runtime migration, a trainer
change, an external tool action, or a Kaggle task.

The outcome is an internally closed v2 contract: an invalid identity must be
unrepresentable by the public schema, serializable bytes must carry and verify
their hashes, configuration must be canonical and self-validating even when
directly constructed, and every required handoff must exist.  Do not accept a
test pass that merely rejects a malformed payload later than construction.

## Required continuity

Read, completely, before edits:

- `D:\Axon\ops\kimi_packets\identity-v2-contract-r1-20260804.md`
- `D:\Axon\roundtable\KIMMY_PACKET_2026-08-04_IDENTITY_CONFIG_128D_PREFLIGHT.md`
- `D:\Axon\roundtable\Kimmy_response_to_Codex_identity_config_04August.md`
- `D:\Axon\codex-turn-state.md`
- `C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md`
- the latest `D:\Kimmy\kimmy_personal_log.md` entry
- this packet completely.

The R1 status record is durable evidence, but R1 has not been accepted.  Do
not change a historical success into an accepted result.

## Immutable boundaries

Never edit, initialize, or delete:

- `D:\Axon\State\axon_runtime` or any live state root;
- protected 64D/128D checkpoints, `dist/`, `datasets/`, `kaggle/`, or training
  scripts;
- any v1 field module (`schema.py`, `delta.py`, `view.py`, `schedule.py`,
  `__init__.py`), or v1/v3 axon-runtime module;
- `ops/axon_runtime.cpu-smoke.json`, frozen Packet 002, or generated
  `D:\Kimmy\State\injection\live_bundle.md`.

Do not launch a runtime, training, daemon, Kaggle, tools/advisors, a checkpoint
promotion, or a state migration.  Preserve the dirty worktree.  Do not stage,
commit, reset, clean, or broadly reformat anything.

## Exact allowed Axon file scope

Create or edit only these Axon paths:

1. `runtime/field/schema_v2.py`
2. `runtime/field/serde_v2.py`
3. `runtime/axon_runtime/identity_v2_config.py`
4. `tests/test_shared_field_v2_schema.py`
5. `tests/test_shared_field_v2_serde.py`
6. `tests/test_shared_field_v2_identity_view.py`
7. `tests/test_axon_runtime_identity_v2_config.py`
8. `roundtable/Kimmy_response_to_Codex_identity_v2_contract_r1_04August.md`
9. `roundtable/Kimmy_response_to_Codex_identity_v2_contract_r2_04August.md`

The required dated append to `D:\Kimmy\kimmy_personal_log.md` is allowed.
`ops/axon_runtime.identity-v2.contract.json` is already correct and must remain
byte-identical.  Do not edit any other Axon file.  If this scope is
insufficient, stop and report `STATUS: BLOCKED_SCOPE`; do not widen it.

## R1 audit findings to repair

These are reproduced defects, not suggestions:

1. `SharedFieldSnapshotV2` accepts a full 11-region snapshot with an identity
   that is masked, empty, malformed, or whose charter manifest is absent.  It
   can serialize that invalid state.  The identity invariant must be checked in
   the schema constructor, not only in deserialization.
2. Identity span metadata is mutable in practice: a forged `span_id`,
   `provenance`, `confidence`, or refs can survive as long as the text/source
   hash looks valid.  The one sealed identity span must equal the canonical
   charter span in every field snapshot.
3. `charter_version=2` is accepted but cannot round-trip because the wire has
   no version discriminator.  This R2 supports exactly
   `axon-identity-charter-v1` with charter version integer `1`, and config
   accepts raw string exactly `"1"` only.  A future version requires a new
   schema and explicit migration, not permissive parsing.
4. Programmatic malformed spans and metadata are coerced rather than rejected:
   object/list mistakes in spans or refs, boolean confidence, and some bad
   text leak raw `TypeError`/`ValueError` instead of a strict serde failure.
5. `serialize_shared_field_v2` omits `field_id`/`canonical_hash`, while
   `to_dict()` includes them and serde rejects them before hash comparisons.
   The persisted byte contract must include exactly both and verify them.
6. `CoreIdentityViewV2` accepts a mutable capability list, malformed direct
   fields, a stale/mismatched envelope, and delimiter injection.  A frozen
   view must remain self-consistent after construction.
7. `IdentityV2Contract` direct construction bypasses parser invariants;
   uppercase SHA aliases and `"01"`, `"+1"`, or whitespace version aliases
   collapse to canonical IDs; `resolved_config` is mutable; and genesis
   provenance does not bind contract/base-v1 identity.

## Required repair contract

### A. Fail closed in the schema

Implement one strict schema-level validation path invoked by
`SharedFieldSnapshotV2.__post_init__`, and make serialization use that same
validation (directly or by relying on an already validated immutable snapshot).
It must require all eleven canonical regions in order, then require the
identity region to be:

- attended and sealed;
- exactly one `identity_charter` span;
- a valid `IdentityCharterV2` using version **1** only;
- exactly equal to `charter.build_identity_region().spans[0]`, including span
  ID, text, kind, source, provenance, confidence, and empty refs;
- bound to `charter.source_manifest_id` in `source_manifest_ids`.

No public constructor may silently fill, sort, deduplicate, stringify, strip,
or normalize a malformed identity or manifest.  Require strict raw types for
spans/refs/manifests and finite numeric confidence; reject booleans where a
numeric value is intended.  V2 is frozen to the existing 95-character
substrate where the original R1 packet requires it.  Keep the v1 modules
unimported and untouched.

`IdentityCharterV2` itself must reject any `charter_version` other than the
integer `1` for this R2 wire contract.  Do not introduce a partial version-2
format.

### B. Strict, self-consistent per-core identity views

`CoreIdentityViewV2` must be genuinely immutable and valid when directly
constructed.  Convert capability input to an immutable tuple without
reordering it, then require it to be nonempty, sorted, and unique.  Validate
all descriptor fields and the lowercase 64-hex charter hash.  The existing
delimited envelope must reject delimiter injection (`;`, `=`, `,`, and line
breaks) in descriptor values, or use an equally unambiguous deterministic
format.  The direct object must verify that its `envelope` exactly matches the
validated descriptor and that it is within the max 192-character envelope
limit before computing its hash.  The renderer may not silently truncate.

### C. Make serde's persisted bytes explicit and total

Define one unambiguous persisted form:

- `serialize_shared_field_v2(snapshot)` returns canonical JSON bytes of
  `snapshot.to_dict()` **including** both `field_id` and `canonical_hash`.
- `deserialize_shared_field_v2` requires exactly those two declared hashes,
  validates their lowercase 64-hex form, recomputes both from the canonical
  no-hash payload, and raises `SerdeV2Error` on a mismatch.
- Valid `snapshot.to_dict()` must deserialize.  A valid snapshot must
  satisfy `serialize -> deserialize -> serialize` byte-for-byte.

Nested JSON objects must use exact keys and types.  Catch schema-construction
`TypeError`/`ValueError` at the serde boundary and re-raise `SerdeV2Error`
with chaining so malformed external bytes always receive the serde error type.
Do not claim a hash-mismatch test passes merely because an unknown key is
rejected.

### D. Canonical, self-validating config

This revision parses only the existing v1 charter contract:

- all SHA-256 identifiers must be exactly 64 lowercase hex; uppercase aliases
  are rejected;
- `identity.charter_version` must be raw string exactly `"1"`; leading zero,
  plus, whitespace, integer, and every other version fail;
- validate all fields in `IdentityV2Contract.__post_init__` as well as through
  `parse_identity_v2_contract`, so direct construction cannot bypass the
  protected v1 root, base config binding, field schema, sealed/context policy,
  isolated root, envelope budget, charter version, or migration gates;
- expose `resolved_config` as deeply immutable, or return fresh defensive
  copies.  Mutating a caller-visible descriptor must not alter its persisted
  identity;
- `build_genesis_snapshot()` remains in-memory only but must add stable,
  non-secret provenance manifest IDs for its exact `contract_id` and its
  `base_v1_config_id` in addition to the charter manifest.  Two distinct
  contract descriptors must not create indistinguishable provenance.

The checked-in `ops/axon_runtime.identity-v2.contract.json` must not change.
Add a regression test that loads `ops/axon_runtime.cpu-smoke.json` and proves
its actual `config_id` equals `EXPECTED_BASE_V1_CONFIG_ID`.

## Required behavior-first tests

Extend the four allowed test files.  Tests must construct all eleven regions
when testing an invalid identity, so no test passes merely due to missing
other regions.  At minimum prove:

1. Full snapshots reject masked, mutable, empty, forged-metadata, and
   missing-manifest identity regions at schema construction and cannot
   serialize them.
2. Charter version 2 and noncanonical direct versions fail; a valid config
   parses, builds genesis, serializes, deserializes, and byte-round-trips.
3. `snapshot.to_dict()` deserializes; each separately tampered field and
   canonical hash fails because its mismatch is detected; raw malformed nested
   shapes (object spans, invalid refs, bool confidence, NUL/unsupported text)
   consistently yield `SerdeV2Error` at the serde boundary.
4. Direct and rendered views reject mutable/stale/directly forged descriptors
   and delimiter injection; normal views remain deterministic, bounded, and
   frozen.
5. Config rejects direct-constructor bypasses, uppercase hash aliases,
   `"01"`, `"+1"`, whitespace, and version 2; actual v1 smoke config binding
   is verified; resolved config cannot be mutated; and distinct valid
   descriptors produce distinct genesis provenance.

Do not weaken a test's assertion or alter a test merely to match a previous
failure message.  Preserve existing correct coverage and add adversarial,
behavior-first cases.

## Required gates

Run exactly these after implementation:

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

python -B -m py_compile `
  runtime/field/schema_v2.py `
  runtime/field/serde_v2.py `
  runtime/axon_runtime/identity_v2_config.py
```

Because these are untracked files, independently inspect them for trailing
whitespace and scope; do not claim `git diff --check` alone validates them.
Do not deliberately create `.pytest_cache` or `.pyc` files.

## Evidence and continuity are deliverables

Create the missing R1 report at
`roundtable/Kimmy_response_to_Codex_identity_v2_contract_r1_04August.md`.
It must truthfully state `STATUS: QUARANTINE`, that its worker exit was 0 but
the R1 handoff/log were missing and independent audit found contract defects.
It is a corrective historical handoff, not a retroactive acceptance.

Create
`roundtable/Kimmy_response_to_Codex_identity_v2_contract_r2_04August.md` with
these headings:

1. `STATUS`
2. `R1 QUARANTINE AND R2 SCOPE`
3. `PRE-EDIT HASHES`
4. `INVARIANT REPAIRS`
5. `PERSISTED FORMAT AND PROVENANCE`
6. `TESTS`
7. `POST-EDIT FILE HASHES AND SCOPE`
8. `V1/STATE PRESERVATION`
9. `RISKS AND DEFERRED WORK`
10. `RECOMMENDED NEXT PACKET`

Report `STATUS: READY_FOR_CODEX_AUDIT` only if all gates pass, the contract
file remains byte-identical, both reports exist, a dated Kimmy personal-log
append exists, no protected state root exists, and no out-of-scope Axon source
file changed.  Otherwise report `STATUS: QUARANTINE` or
`STATUS: BLOCKED_SCOPE`.  Append a dated Kimmy personal-log entry with the R1
quarantine, R2 files/tests/hashes, and blockers.  Refresh injection sources
only if needed; never edit `live_bundle.md` directly.

Stop after reporting.  Do not start v2 engine/pager/store integration,
runtime config cleanup, trainer configuration, a corpus, training, or Kaggle.
