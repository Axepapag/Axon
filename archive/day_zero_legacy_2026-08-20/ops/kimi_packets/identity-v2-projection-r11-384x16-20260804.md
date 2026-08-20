# Identity V2 Projection R11: sealed 11-region field to 384x16 compatibility pages

## Authority and purpose

Codex independently accepted the narrow R9 identity-contract repair and narrow R10
v2-serde repair. R1 through R8 remain historical quarantines. Those narrow
acceptances do not authorize a v2 state root, engine integration, model
execution, a runtime write, training, Kaggle, promotion, tools/advisors, or a
contract migration.

Implement one pure, in-memory v2 projection boundary only. It must compile a
sealed SharedFieldSnapshotV2 into immutable 384 x 16 pages compatible with the
existing 64D/128D physical checkpoint geometry. Identity must be visible as
physical context, tied to its sealed charter, and raw canonical identity
characters must remain part of an auditable multi-page coverage path. This work
does not call a model or create a state root.

The frozen v2 contract permits identity envelopes of at most 128 characters.
The real descriptor for axon128-a as a consolidator is about 144 characters.
This is a deliberate fail-closed preflight finding. Do not shorten a descriptor,
raise the budget, edit a contract, or load/mutate the v1 config to evade it.
The new compiler must reject it and report the incompatibility.

The two Codex turn-state files are the acceptance authority. Kimmy injection
material may say R10 awaits review; that wording is stale. Treat R9/R10 as
accepted only at their narrow boundaries, not as a runtime/training release.

## Required reading and prechecks

Read fully before editing:

- this packet;
- ops\kimi_packets\identity-v2-contract-r9-no-hook-fspath-normalization-20260804.md;
- ops\kimi_packets\identity-v2-serde-r10-hash-boundary-20260804.md;
- roundtable\Kimmy_response_to_Codex_identity_v2_contract_r9_04August.md;
- roundtable\Kimmy_response_to_Codex_identity_v2_serde_r10_04August.md;
- D:\Axon\codex-turn-state.md;
- C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md;
- the latest relevant D:\Kimmy\kimmy_personal_log.md entry;
- runtime\field\schema_v2.py, runtime\field\serde_v2.py,
  runtime\axon_runtime\identity_v2_config.py, runtime\field\view.py,
  runtime\field\schedule.py, and their relevant tests.

Before work, verify and report:

- D:\Axon\State\kimi_supervisor\active.lock.json is absent;
- R10 job status is terminal SUCCEEDED and its actual stdout hash is
  9C234035FD6BA0FC4E9EB92120E317A61A825E7EC4F7E2F23290138A203F2E86;
- the R10 report exists;
- the unique R11 job directory does not already exist.

If a precheck fails, report QUARANTINE and stop. Do not retry or overlap a
supervised Kimi job.

## Immutable boundaries

Never edit, initialize, delete, or create:

- D:\Axon\State\axon_runtime, D:\Axon\State\axon_runtime_identity_v2, or any
  other Axon runtime state root;
- protected 64D/128D checkpoints, dist, datasets, Kaggle assets, training
  scripts, curricula, or the v1 runtime configuration;
- runtime\field\view.py, runtime\field\schedule.py, runtime\field\__init__.py,
  runtime\field\schema.py, runtime\field\schema_v2.py,
  runtime\field\serde_v2.py, any v1/v3 runtime engine/store/projection module,
  any model/core file, or runtime\axon_runtime\identity_v2_config.py;
- ops\axon_runtime.cpu-smoke.json or
  ops\axon_runtime.identity-v2.contract.json;
- D:\Kimmy\State\injection\live_bundle.md directly.

Do not import, instantiate, or call v1 FieldView, compile_field_view, the v1
scheduler, or a v1 runtime projection. Do not launch a runtime, daemon, CPU
model smoke, training, Kaggle, promotion, tools/advisors, migration, or
checkpoint inspection. Preserve the dirty worktree. Do not stage, commit,
reset, clean, or broadly reformat.

## Exact allowed Axon file scope

Create or edit only:

1. runtime\field\view_v2.py
2. tests\test_shared_field_v2_projection.py
3. roundtable\Kimmy_response_to_Codex_identity_v2_projection_r11_04August.md

The only additionally allowed files are:

- one dated append to D:\Kimmy\kimmy_personal_log.md;
- D:\Kimmy\State\injection\04_current_task.md and
  D:\Kimmy\State\injection\05_rolling_summary.md, followed by
  python D:\Kimmy\Source\injection_watcher.py --force.

Never directly edit generated live_bundle.md. No other Axon file may change.
If scope is insufficient, report STATUS: BLOCKED_SCOPE and stop.

## Required public v2-only module

Create a separate v2 dataclass family, never a v1 adapter. Expose at least:

- V2ProjectionError;
- V2SlotKind with TAG, SPAN, DERIVED_IDENTITY, BLANK, and PADDING;
- immutable V2SlotRef and V2ViewOmission records;
- immutable V2ReadCursor with explicit selected context region and offsets;
- immutable PagedIdentityFieldViewV2 / IdentityAwareFieldViewV2 with
  read-only arrays, source/identity authority metadata, cursor-before/after,
  and deterministic view_hash;
- compile_identity_aware_v2_read_page(...);
- a pure coverage helper/auditor suitable for proving a finite explicit page
  sequence covers every attended raw canonical character exactly once.

Do not accept a caller-supplied CoreIdentityViewV2 as authority. The compiler
takes an exact-base IdentityV2Contract, an exact-base SharedFieldSnapshotV2,
and explicit core_id, display_name, model_label, role_capabilities,
current_role, proposal_region, and V2ReadCursor (or equally explicit offsets).
It must derive the CoreIdentityViewV2 internally from freshly verified contract
primitives. Every policy/cursor input must be explicit, documented, and
included in the output hash. A named, canonical projection-plan descriptor and
its hash may live in this new pure module; do not hide a second scheduling
policy in ambient process state.

## Authority reconstruction and strict binding

At every public boundary, reject subclasses and normalize malformed exact-base
forgeries to V2ProjectionError.

1. Require exact-base IdentityV2Contract and call only its public verified
   descriptor method. Immediately canonicalize/re-hash that fresh descriptor to
   derive local primitive contract_id, charter text/hash, envelope budget, and
   required contract source-manifest ID. After that, do not read the contract
   again.
2. Reconstruct a fresh exact-base IdentityCharterV2 from those captured
   primitives. Reject descriptor inconsistency and any malformed ordinary
   conversion error.
3. Require exact-base SharedFieldSnapshotV2. Capture its primitive fields and
   supplied field_id/canonical_hash once in a guarded block; require both
   supplied hashes to be exact built-in lowercase SHA-256 strings. Reconstruct
   a fresh exact-base SharedFieldSnapshotV2 and require supplied hashes to
   equal fresh values. Missing/malformed fields must become V2ProjectionError,
   never raw AttributeError, KeyError, TypeError, or ValueError.
4. Require the fresh snapshot's identity region to be exactly one sealed,
   attended identity_charter span whose text, span ID, source, provenance, and
   charter source manifest match the contract charter. Also require the
   contract-manifest ID to be present in source_manifest_ids. Independently
   valid-but-different snapshot/contract charters are rejected.
5. Render CoreIdentityViewV2 internally from only the captured charter,
   descriptor budget, and caller descriptors. It must fit intact; no
   truncation, normalization, fallback, or descriptor shortening. Contract
   envelope failure happens before rendering a page.
6. Hash-bind a domain-separated canonical manifest containing at least protocol
   plan/geometry, source field ID and canonical hash, parent/tick,
   contract ID, charter hash, full core descriptor/current role, rendered
   identity view hash and envelope SHA-256, selected regions/cursor before and
   after, all arrays, all refs, and all omissions. Field, charter, contract,
   core, role, descriptor, cursor, or plan changes must alter the page hash or
   fail closed.

## Required page policy

The output retains exactly three physical learned roles:

- context [0:256) with role ID 0;
- user [256:320) with role ID 1;
- proposal [320:384) with role ID 2.

field16 is exactly (384,16), float32, C-contiguous/read-only/little-endian.
Role IDs, logical-region IDs, attention, and write arrays are exactly length
384 and immutable. Logical v2 identity ID 10 is audit metadata only; it is
never used as a fourth learned physical type row. write_mask is false before
320 and true exactly at 320:384. All proposal slots, including zero-vector
blanks, are attended; context/user padding is zero-vector/unattended/
nonwritable.

Use only the frozen 95-character 16D substrate. V2 spans and descriptors are
strictly substrate-representable. If a forged object exposes an unsupported
character, fail closed; never normalize or make an unsupported-substrate
omission.

Use this explicit multi-page policy:

1. Every page begins in context with a tag plus the whole derived, hash-bound
   identity envelope. Envelope characters have DERIVED_IDENTITY refs, logical
   identity ID 10, and an envelope character index. They are not falsely
   labelled as raw charter span characters.
2. The page then renders exactly one explicit context region selected by
   V2ReadCursor, with a tag. It may be identity: on that page, raw charter
   characters are normal SPAN refs and can advance the charter cursor. It may
   not be user_input or the current proposal region. Other attended regions are
   explicitly omitted for this page.
3. Render dedicated tagged user and proposal windows from explicit cursors.
   proposal_region can only be v2 scratch or response_draft and blanks remain
   active proposal positions.
4. Return a deterministic next cursor and provide an explicit, pure cyclic
   page-plan/coverage helper. Across the complete named cycle, all attended raw
   canonical characters, including the 413-character identity charter, must be
   selected exactly once with no duplicate source positions. This is a page
   bridge, not a claim that all 11 regions fit on one charslot page.

For every raw canonical span character on each page exactly one condition must
hold: it has a SPAN ref, or it appears in one contiguous V2ViewOmission with
original region/span/source/provenance/offsets/reason/UTF-8 hash. Use explicit
reasons such as context_region_not_selected, context_cursor_before,
context_capacity, user_cursor_before, user_capacity, proposal_cursor_before,
proposal_capacity, and region_masked. DERIVED_IDENTITY and tags are a separate
emitted-character ledger and never count as raw-character coverage. No silent
loss or double-counted source positions is allowed.

## Required tests

Create behavior-first tests proving all of the following:

1. Valid contract genesis and a valid successor compile deterministically
   without mutating source bytes or materializing a v2 state root.
2. Exact 384x16 geometry, physical 0/1/2 roles only, identity logical ID 10
   metadata, write/attention separation, proposal blanks, and immutable arrays.
3. A selected raw span containing the full frozen 95-character alphabet
   round-trips via the 16D letter bank with exact refs/offsets.
4. The derived identity envelope is full/context-only and distinct from raw
   charter SPAN refs. A cycle includes every raw charter character exactly
   once; long 11-region fixtures have zero missing and duplicate raw source
   positions across their named cycle.
5. Stable inputs remain byte/digest stable; changing field/charter/contract
   authority/core/display/model/capabilities/current role/selected region/
   cursor/plan changes binding or fails closed.
6. User/proposal offsets, masked regions, nonselected regions, and pressure
   produce complete deterministic omission accounting.
7. Every hostile boundary fails closed: subclasses; object.__new__ exact-base
   contract/snapshot missing fields; public-field/trace mutation; wrong or
   equality-lying supplied snapshot hashes; nested forged span/region;
   contract/snapshot charter disagreement; absent charter/contract source
   manifests; masked/writable/multi-span/wrong-source identity; malformed
   descriptor capabilities; identity over budget; invalid proposal/context
   selection; invalid cursors; write-mask escape; a fourth physical role; and
   forged unsupported text.
8. The frozen 128-character budget rejects the fixed real
   axon128-a / Axon 128D A / 128D / consolidator,proposer,sleeper /
   consolidator descriptor without reading or changing the v1 config.
9. The new source has no v1 view/compiler/scheduler/runtime-projection import
   or call, no filesystem/state creation behavior, and no relaxed read-hash
   bypass. Existing v1 APIs remain unchanged.

Do not weaken R9/R10 regressions or edit existing tests to make this pass.

## Required gates

Run focused v2 tests:

- tests/test_shared_field_v2_schema.py
- tests/test_shared_field_v2_serde.py
- tests/test_shared_field_v2_identity_view.py
- tests/test_axon_runtime_identity_v2_config.py
- tests/test_shared_field_v2_projection.py

Then run existing v1/compatibility tests:

- tests/test_shared_field.py
- tests/test_field_view_schedule.py
- tests/test_axon_runtime_config.py
- tests/test_axon_runtime_transaction_serde.py
- tests/test_axon_runtime_bootstrap.py
- tests/test_axon_runtime_store.py
- tests/test_axon_runtime_v3_contracts.py
- tests/test_axon_runtime_v3_store.py
- tests/test_charslot_roundtrip.py

Compile runtime/field/view_v2.py, runtime/field/serde_v2.py, and
runtime/axon_runtime/identity_v2_config.py with python -B. Run independent
compact probes for strict boundaries and all-95 roundtrip. Verify frozen values:

- contract blob: 59d35b0bb5ad74df166d315272fc306e17df6dcf;
- CPU smoke blob: 7ae72883b2c3e059329cf0d464a61ef796b57843;
- R9 source/test: 0ebe483f181c317c5a29a6f6ceed7b61ae5f5e16 and
  3618b612cc4b7c7c169a94f83d836a9fecc78329;
- R10 source/test: 6fd17d21a0fe15b86c2a899d7fac6b0fbca68d68 and
  9f3519bb46602ca4020278aeef37666e068eb448.

Inspect allowed code/tests/report for trailing whitespace and final newline.
Do not deliberately create bytecode/cache files.

## Evidence and stop condition

Create roundtable\Kimmy_response_to_Codex_identity_v2_projection_r11_04August.md
with headings:

1. STATUS
2. ACCEPTED/QUARANTINED BOUNDARIES AND R11 SCOPE
3. PRE-EDIT HASHES AND SUPERVISOR PRECHECK
4. PROJECTION CONTRACT AND EXPLICIT PAGE POLICY
5. AUTHORITY/IDENTITY BINDING
6. OMISSION ACCOUNTING AND 16D ROUND TRIP
7. TESTS AND ADVERSARIAL PROBES
8. CONTINUITY CORRECTION
9. POST-EDIT HASHES AND SCOPE
10. V1/STATE/CHECKPOINT PRESERVATION
11. DEFERRED 128-CHAR ENVELOPE CONFIGURATION INCOMPATIBILITY
12. RISKS AND RECOMMENDED NEXT PACKET

STATUS may be READY_FOR_CODEX_AUDIT only if every gate passes, frozen/R9/R10
hashes remain exact, all attacks fail at the stated boundary, valid pages and
full-cycle coverage pass, no protected state root exists, report/log/injection
updates exist, regenerated injection bundle mentions R11, and no out-of-scope
Axon source changed. Otherwise report QUARANTINE or BLOCKED_SCOPE. Stop after
reporting.

