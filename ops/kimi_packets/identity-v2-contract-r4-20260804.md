# Identity V2 Contract R4: final public-boundary closure

## Authority and outcome

Codex independently audited completed R3. R3 preserved state, scope, and all ordinary gates, but it remains QUARANTINED because three public-boundary invariants still fail:

1. SharedFieldSnapshotV2 accepts a generator for regions instead of only a list or tuple.
2. A negative-zero confidence (-0.0) hashes differently from canonical 0.0 while both are accepted.
3. A subclass of IdentityCharterV2 can override canonical descriptor/genesis methods and bypass direct IdentityV2Contract construction checks.

This is the sole authorized repair. It remains a pure isolated shared-field-v2 contract change. It must not implement a transaction, engine, runtime, corpus, trainer, state materialization, external tool call, or Kaggle launch.

## Required continuity

Read completely before edits:

- D:\Axon\ops\kimi_packets\identity-v2-contract-r3-20260804.md
- D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r1_04August.md
- D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r2_04August.md
- D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r3_04August.md
- D:\Axon\codex-turn-state.md
- C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md
- latest D:\Kimmy\kimmy_personal_log.md entry
- this packet completely.

R1, R2, and R3 are historical quarantines. Do not edit their reports to imply acceptance.

## Immutable boundaries

Never edit, initialize, or delete:

- D:\Axon\State\axon_runtime, D:\Axon\State\axon_runtime_identity_v2, or any other state root;
- protected 64D/128D checkpoints, dist, datasets, kaggle, or training scripts;
- v1 field modules, v1/v3 axon-runtime modules, ops\axon_runtime.cpu-smoke.json, or ops\axon_runtime.identity-v2.contract.json;
- D:\Kimmy\State\injection\live_bundle.md directly.

Do not launch any runtime, training, daemon, Kaggle, tools/advisors, promotion, or migration. Preserve the dirty worktree. Do not stage, commit, reset, clean, or broadly reformat.

## Exact allowed Axon file scope

Create or edit only:

1. runtime\field\schema_v2.py
2. runtime\axon_runtime\identity_v2_config.py
3. tests\test_shared_field_v2_schema.py
4. tests\test_axon_runtime_identity_v2_config.py
5. roundtable\Kimmy_response_to_Codex_identity_v2_contract_r4_04August.md

The dated append to D:\Kimmy\kimmy_personal_log.md is allowed. Refresh compact injection source files only if durable facts changed; never edit live_bundle.md directly. No other Axon file may change. If scope is insufficient, report STATUS: BLOCKED_SCOPE and stop.

## Exact repairs

### A. Region input boundary

In SharedFieldSnapshotV2.__post_init__, before any tuple conversion, require regions to be a list or tuple. Reject strings, mappings, generators, and arbitrary iterables. Preserve the existing exact length, RegionStateV2, and canonical order requirements. Add a regression test that a generator yielding an otherwise valid canonical eleven-region snapshot fails.

### B. Canonical confidence encoding

For FieldSpanV2 confidence, keep the existing finite float-only rule and additionally reject signed negative zero. The one canonical zero representation is +0.0. Do not silently normalize -0.0. Add a direct FieldSpanV2 regression test proving -0.0 fails, while +0.0 continues to work.

### C. Exact charter type seal

No subclass may modify charter semantics in this v1 wire contract.

- Make IdentityCharterV2 reject subclass instances, or equivalently make all public consumers require type(charter) is IdentityCharterV2 and independently validate the exact base-class canonical descriptor.
- IdentityV2Contract.__post_init__ must require the exact base IdentityCharterV2 type before it uses to_canonical_dict or build_genesis_snapshot.
- render_core_identity_view_v2 must also require the exact base IdentityCharterV2 type.
- Add a regression test with an IdentityCharterV2 subclass that overrides to_canonical_dict and build_genesis_snapshot. Construction must fail before the override can influence a contract or view.

Use errors that clearly indicate exact base type is required. Do not weaken normal valid charter construction.

## Required gates

Run the focused suite:

- tests/test_shared_field_v2_schema.py
- tests/test_shared_field_v2_serde.py
- tests/test_shared_field_v2_identity_view.py
- tests/test_axon_runtime_identity_v2_config.py

Then run the v1 regression suite:

- tests/test_shared_field.py
- tests/test_axon_runtime_config.py
- tests/test_axon_runtime_transaction_serde.py
- tests/test_axon_runtime_bootstrap.py
- tests/test_axon_runtime_store.py
- tests/test_axon_runtime_v3_contracts.py
- tests/test_axon_runtime_v3_store.py
- tests/test_charslot_roundtrip.py

Also compile all three v2 modules: schema_v2.py, serde_v2.py, identity_v2_config.py. Inspect every changed/untracked allowed code/test file for literal trailing whitespace and final newline. Do not deliberately create cache or bytecode files.

## Evidence and stop condition

Create roundtable\Kimmy_response_to_Codex_identity_v2_contract_r4_04August.md with headings:

1. STATUS
2. R1/R2/R3 QUARANTINE AND R4 SCOPE
3. PRE-EDIT HASHES
4. BOUNDARY AND CANONICALITY REPAIRS
5. TESTS AND ADVERSARIAL REPROS
6. POST-EDIT FILE HASHES AND SCOPE
7. V1/STATE PRESERVATION
8. RISKS AND DEFERRED WORK
9. RECOMMENDED NEXT PACKET

STATUS may be READY_FOR_CODEX_AUDIT only if all gates pass, contract and cpu-smoke JSON blobs are byte-identical, R4 report/log exist, no protected state root exists, no out-of-scope Axon source changed, and the R4 report contains no literal trailing whitespace. Otherwise report QUARANTINE or BLOCKED_SCOPE. Append a dated Kimmy log entry with the R3 quarantine, files/tests/hashes, and blockers. Stop after reporting.

