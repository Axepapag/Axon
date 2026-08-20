# Identity V2 Contract R3: close remaining normalization and type holes

## Authority and outcome

Codex independently audited completed R2. R2 handoff and protected-state preservation are sound, but R2 remains QUARANTINED because public schema, serde, and renderer paths still silently normalize malformed input. This is the only authorized corrective packet. It remains a pure isolated shared-field-v2 contract repair, not engine work, state materialization, training, a tool/advisor action, or a Kaggle launch.

The outcome is a strict public contract: wrong region order, malformed raw wire shapes, noncanonical identity metadata, and forged renderer inputs fail at their earliest boundary. Do not change a test merely to make a prior behavior look intentional.

## Required continuity

Read completely before edits:

- D:\Axon\ops\kimi_packets\identity-v2-contract-r2-20260804.md
- D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r1_04August.md
- D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r2_04August.md
- D:\Axon\roundtable\KIMMY_PACKET_2026-08-04_IDENTITY_CONFIG_128D_PREFLIGHT.md
- D:\Axon\codex-turn-state.md
- C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md
- the latest D:\Kimmy\kimmy_personal_log.md entry
- this packet completely.

R1 and R2 are historical quarantines. Do not alter their reports to imply acceptance. R3 must state why it repaired them.

## Immutable boundaries

Never edit, initialize, or delete:

- D:\Axon\State\axon_runtime, D:\Axon\State\axon_runtime_identity_v2, or any other runtime state root;
- protected 64D/128D checkpoints, dist, datasets, kaggle, or training scripts;
- any v1 field module or v1/v3 axon-runtime module;
- ops\axon_runtime.cpu-smoke.json, ops\axon_runtime.identity-v2.contract.json, frozen Packet 002, or D:\Kimmy\State\injection\live_bundle.md directly.

Do not launch runtime, training, daemon, Kaggle, tools/advisors, promotion, or state migration. Preserve the dirty worktree. Do not stage, commit, reset, clean, or broadly reformat.

## Exact allowed Axon file scope

Create or edit only:

1. runtime\field\schema_v2.py
2. runtime\field\serde_v2.py
3. runtime\axon_runtime\identity_v2_config.py
4. tests\test_shared_field_v2_schema.py
5. tests\test_shared_field_v2_serde.py
6. tests\test_shared_field_v2_identity_view.py
7. tests\test_axon_runtime_identity_v2_config.py
8. roundtable\Kimmy_response_to_Codex_identity_v2_contract_r3_04August.md

The dated append to D:\Kimmy\kimmy_personal_log.md is allowed. Refresh compact injection source files only if durable facts change; never edit live_bundle.md directly. Do not edit any other Axon file. If scope is insufficient, stop and report STATUS: BLOCKED_SCOPE.

## Reproduced R2 blockers

These are actual post-R2 repros and must become regression tests:

1. A valid snapshot with all eleven regions reversed is accepted and silently reordered by SharedFieldSnapshotV2. Public construction must require canonical region order.
2. Identity confidence=1 passes equality against canonical 1.0, producing a different serialized identity span. Identity must equal its canonical charter span byte-for-byte, including canonical float representation.
3. Raw JSON values spans={} or spans="" can become empty nonidentity regions; container_refs="", edge_refs="", and {} can become empty refs. Wire data must require JSON arrays before any tuple conversion.
4. IdentityCharterV2.build_genesis_snapshot(source_manifest_ids="bad") splits that string into one-character IDs. Public builders must reject strings, mappings, generators, and duplicate manifest IDs rather than coercing them.
5. render_core_identity_view_v2 accepts role_capabilities=(1,) through string coercion, silently sorts/deduplicates capabilities, and accepts a duck-typed fake charter. It must require an actual IdentityCharterV2, a raw list/tuple of nonempty strings already sorted and unique, and a current role present in capabilities.
6. parse_identity_v2_contract fails closed but leaks raw TypeError for null, scalar, and array JSON roots. It must consistently raise IdentityV2ConfigError before exact-key checks.

## Required repairs

### A. Schema canonicality and raw types

- SharedFieldSnapshotV2.__post_init__ must require regions to be a list or tuple of exactly eleven RegionStateV2 values in CANONICAL_REGION_ORDER_V2. Reject a permutation; do not construct a reordered replacement. Keep immutable tuple storage after validation.
- RegionStateV2.__post_init__ must require spans to be a list or tuple. Reject string, mapping, generator, and other iterable coercions; then store an immutable tuple.
- FieldSpanV2 must store confidence in one canonical form. Require a finite float, not bool or int, within [0,1].
- Identity validation must compare the whole canonical span byte payload or canonical SHA-256 against charter.build_identity_region().spans[0], not Python numeric equality.
- Require source_manifest_ids to be a list or tuple of nonempty string IDs with no duplicates. Do not sort or deduplicate supplied IDs.
- IdentityCharterV2.build_genesis_snapshot must validate supplied extra manifests before concatenation and must not silently change caller-supplied extras.

### B. Serde boundary strictness

- _field_span_from_dict must require container_refs and edge_refs to be JSON lists before tuple conversion.
- _region_state_from_dict must require spans to be a JSON list before iteration, and every item must be an object.
- Preserve hash checks. Wrap every schema TypeError or ValueError caused by malformed external bytes as SerdeV2Error.
- Tests must recompute declared hashes for malformed normalized candidates so they prove raw-shape rejection rather than stale-hash rejection.

### C. Renderer and config hardening

- render_core_identity_view_v2 must require an IdentityCharterV2 object; reject duck typing.
- Before formatting, require capabilities to be a list or tuple of nonempty strings; preserve supplied order and reject it if not already sorted and unique. Do not use coercion, deduplication, or sorting to repair input. Require current_role in capabilities. Direct CoreIdentityViewV2 must preserve the same invariant.
- At config parser entry, use _mapping(value, "contract") or equivalent before _exact_keys so null, scalar, and arrays consistently raise IdentityV2ConfigError.

## Required adversarial tests

Extend only the four allowed test files. At minimum prove:

1. Reversed full-region input, invalid direct region spans, duplicate manifests, and integer identity confidence fail at schema construction.
2. Builder rejects string, mapping, and generator extra manifests.
3. Serde rejects empty/object/string spans and string/object refs even when declared hashes are recomputed to the candidate payload.
4. Renderer rejects fake charter, non-string, unsorted, or duplicate capabilities, and a current role outside capabilities. Valid capabilities work only when already canonical.
5. null, scalar, and array config roots all raise IdentityV2ConfigError.

Keep R2 successful coverage. Do not weaken assertions.

## Required gates

Run the R2 focused v2 suite over:
- tests/test_shared_field_v2_schema.py
- tests/test_shared_field_v2_serde.py
- tests/test_shared_field_v2_identity_view.py
- tests/test_axon_runtime_identity_v2_config.py

Then run the R2 v1 regression suite over:
- tests/test_shared_field.py
- tests/test_axon_runtime_config.py
- tests/test_axon_runtime_transaction_serde.py
- tests/test_axon_runtime_bootstrap.py
- tests/test_axon_runtime_store.py
- tests/test_axon_runtime_v3_contracts.py
- tests/test_axon_runtime_v3_store.py
- tests/test_charslot_roundtrip.py

Also run python -B -m py_compile on schema_v2.py, serde_v2.py, and identity_v2_config.py. Inspect untracked allowed files for trailing whitespace and final newline; git diff --check alone is insufficient. Do not deliberately create cache or bytecode files.

## Evidence and stop condition

Create roundtable\Kimmy_response_to_Codex_identity_v2_contract_r3_04August.md with headings:

1. STATUS
2. R1/R2 QUARANTINE AND R3 SCOPE
3. PRE-EDIT HASHES
4. NORMALIZATION HOLES CLOSED
5. TESTS AND ADVERSARIAL REPROS
6. POST-EDIT FILE HASHES AND SCOPE
7. V1/STATE PRESERVATION
8. RISKS AND DEFERRED WORK
9. RECOMMENDED NEXT PACKET

Report STATUS: READY_FOR_CODEX_AUDIT only if every test passes, checked contract JSON is byte-identical, R3 report/log exist, no protected state root exists, and no out-of-scope Axon source file changed. Otherwise report STATUS: QUARANTINE or STATUS: BLOCKED_SCOPE. Append a dated Kimmy personal-log entry with R2 quarantine, files/tests/hashes, and blockers. Stop after reporting; do not start transaction, projection, store, config cleanup, corpus, training, or Kaggle work.

