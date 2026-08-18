# Identity V2 Contract R7: bound-authority integrity and exact configuration text

## Authority and outcome

Codex independently audited completed R6. R6 is **QUARANTINED** despite green
ordinary suites and its self-report. Do not promote R6 and do not start CPU
smoke, transaction/store integration, runtime, trainer, corpus, checkpoint, or
Kaggle work.

This is the sole authorized repair of two independently reproduced
configuration-only fail-opens. Preserve the R6 schema/serde repairs without
altering them:

1. '_verify_bound_authority()' checks only '_bound_descriptor_hash ==
   contract_id'; it does not hash '_bound_descriptor_text'. Changing only that
   text with 'object.__setattr__' makes descriptor/genesis APIs emit forged
   material under the original ID.
2. Direct construction accepts equality-lying 'str' subclasses for
   'base_v1_config_id', 'field_schema', 'physical_role', and
   'migration_authority', because '_sha256' and '_nonempty' use 'isinstance'.

The outcome must be a side-effect-free, exact-base, hash-verified v2 identity
contract. No state root may be created.

## Required reading and continuity

Read completely before editing:

- D:\Axon\ops\kimi_packets\identity-v2-contract-r6-canonical-authority-20260804.md
- D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r6_04August.md
- D:\Axon\codex-turn-state.md
- C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md
- the latest D:\Kimmy\kimmy_personal_log.md entry;
- this packet.

R1 through R6 remain historical quarantines. Never edit their reports to imply
acceptance.

## Immutable boundaries

Never edit, initialize, delete, or create:

- D:\Axon\State\axon_runtime, D:\Axon\State\axon_runtime_identity_v2, or any
  other Axon runtime state root;
- protected 64D/128D checkpoints, dist, datasets, Kaggle assets, or training
  scripts;
- v1/v3 modules, runtime\field\schema_v2.py, runtime\field\serde_v2.py,
  ops\axon_runtime.cpu-smoke.json, or ops\axon_runtime.identity-v2.contract.json;
- D:\Kimmy\State\injection\live_bundle.md directly.

Do not launch a runtime, daemon, training, Kaggle, promotion, tools/advisors,
or migration. Preserve the dirty worktree. Do not stage, commit, reset, clean,
or broadly reformat.

## Exact allowed Axon file scope

Create or edit only:

1. runtime\axon_runtime\identity_v2_config.py
2. tests\test_axon_runtime_identity_v2_config.py
3. roundtable\Kimmy_response_to_Codex_identity_v2_contract_r7_04August.md

The following Kimmy continuity files are additionally allowed:

- append one dated correction to D:\Kimmy\kimmy_personal_log.md;
- update D:\Kimmy\State\injection\04_current_task.md and
  D:\Kimmy\State\injection\05_rolling_summary.md, then run the existing
  watcher to regenerate live_bundle.md; never edit live_bundle.md directly.

No other Axon file may change. If scope is insufficient, report
'STATUS: BLOCKED_SCOPE' and stop.

## Required repairs

### A. Verify the actual bound descriptor bytes before output

At '_verify_bound_authority()':

1. Safely obtain '_bound_descriptor_text', '_bound_descriptor_hash', and
   'contract_id'; absent attributes must normalize to 'IdentityV2ConfigError'.
2. Require all three to be exact built-in 'str' values.
3. Compute sha256(_bound_descriptor_text.encode("utf-8")).hexdigest() and
   require it to equal both '_bound_descriptor_hash' and 'contract_id' before
   parsing or emitting any descriptor/genesis material.
4. Parse only a locally captured verified text. 'to_canonical_descriptor()'
   must return a fresh copy derived from that verified local authority, not
   read 'self._bound_descriptor_text' again after verification.
5. Invalid JSON or noncanonical/malformed bound authority must fail closed as
   'IdentityV2ConfigError', with no raw AttributeError, TypeError, or
   JSONDecodeError.

This packet need not defend against hostile Python that coherently replaces
text, hash, and ID with a chosen matching cryptographic hash. It must reject
all inconsistent one- or two-field tampering, especially a text-only mutation.

### B. Exact text for direct-construction authority fields

Make '_nonempty' and '_sha256' (or their direct equivalent) require exact
built-in 'str' before truthiness, equality, regex, or descriptor construction.
This must cover direct IdentityV2Contract construction for:

- base_v1_config_id;
- field_schema;
- physical_role;
- migration_authority.

Do not coerce subclasses. Normal JSON parsing and ordinary str/Path inputs
must remain valid.

### C. Behavior-first regressions

Add tests proving:

1. Changing only '_bound_descriptor_text' to syntactically valid canonical
   forged JSON leaves the old ID present but makes both descriptor/genesis APIs
   raise IdentityV2ConfigError before material is returned.
2. Changing only '_bound_descriptor_hash' or only 'contract_id' also fails
   closed, without a type/attribute leak.
3. A custom equality-lying 'str' subclass whose literal contents differ from
   the required one is rejected at direct construction for each of the four
   authority-bearing text fields above; no descriptor may be emitted.
4. The sample ID remains exactly
   5460121dc988f40f2462e0aae19f598ae086c7987332ed91da8529f2c8f65146,
   both ADS forms remain rejected, a normal nonexistent temporary v2 root
   parses/builds in memory without creating a directory, and public-field
   mutation resistance remains green.

Use failure assertions at public constructor/API boundaries, never merely later
JSON deserialization.

## Required gates

Run the focused v2 suite:

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

Compile runtime/axon_runtime/identity_v2_config.py. Independently run a compact
adversarial probe for every R7 case plus valid v2 genesis and successor
serialize/deserialise byte roundtrips. Verify frozen JSON blobs remain exactly
59d35b0bb5ad74df166d315272fc306e17df6dcf and
7ae72883b2c3e059329cf0d464a61ef796b57843. Inspect allowed code/tests/report
for trailing whitespace and final newline. Do not deliberately create cache or
bytecode files.

## Evidence and stop condition

Create roundtable\Kimmy_response_to_Codex_identity_v2_contract_r7_04August.md
with headings:

1. STATUS
2. R1-R6 QUARANTINE AND R7 SCOPE
3. PRE-EDIT HASHES
4. BOUND DESCRIPTOR AUTHORITY REPAIR
5. EXACT DIRECT-CONSTRUCTION TEXT REPAIR
6. TESTS AND ADVERSARIAL REPROS
7. CONTINUITY CORRECTION
8. POST-EDIT FILE HASHES AND SCOPE
9. V1/STATE PRESERVATION
10. RISKS AND DEFERRED WORK
11. RECOMMENDED NEXT PACKET

'STATUS' may be READY_FOR_CODEX_AUDIT only if every gate passes, frozen
contract/CPU-smoke blobs are byte-identical, every R7 adversarial case fails at
the required boundary, valid v2 genesis/successor roundtrips pass, no protected
state root exists, regenerated injection bundle mentions R7, report/log exist,
and no out-of-scope Axon source changed. Otherwise report QUARANTINE or
BLOCKED_SCOPE. Stop after reporting.
