# Identity V2 Contract R8: atomic authority capture and canonical descriptor validation

## Authority and outcome

Codex independently audited completed R7. R7 is **QUARANTINED** despite green
ordinary suites. Do not promote R7 and do not start CPU smoke,
transaction/store integration, runtime, trainer, corpus, checkpoint, or Kaggle
work.

This is the sole authorized repair of two independently reproduced R7 failures:

1. build_genesis_snapshot() verifies bound authority, then rereads mutable
   self.contract_id while composing its source manifests. A deterministic
   post-verification one-field mutation makes it return a serializable genesis
   that records a forged zero contract ID.
2. _verify_bound_authority() merely JSON-loads its text after hashing. A
   coherent text/hash/ID triple can return malformed or noncanonical descriptor
   material, emit a forged valid genesis, or leak raw KeyError.

The outcome must be a side-effect-free identity contract where a single
verified local authority record supplies every descriptor/genesis value through
the end of each API call. No state root may be created.

## Required reading and continuity

Read completely before editing:

- D:\Axon\ops\kimi_packets\identity-v2-contract-r7-bound-authority-exact-text-20260804.md
- D:\Axon\roundtable\Kimmy_response_to_Codex_identity_v2_contract_r7_04August.md
- D:\Axon\codex-turn-state.md
- C:\Users\Jeffg\Documents\Codex\.codex\codex-turn-state.md
- the latest D:\Kimmy\kimmy_personal_log.md entry;
- this packet.

R1 through R7 remain historical quarantines. Never rewrite their reports to
imply acceptance.

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
3. roundtable\Kimmy_response_to_Codex_identity_v2_contract_r8_04August.md

The following Kimmy continuity files are additionally allowed:

- append one dated correction to D:\Kimmy\kimmy_personal_log.md;
- update D:\Kimmy\State\injection\04_current_task.md and
  D:\Kimmy\State\injection\05_rolling_summary.md, then run the existing
  watcher to regenerate live_bundle.md; never edit live_bundle.md directly.

No other Axon file may change. If scope is insufficient, report
'STATUS: BLOCKED_SCOPE' and stop.

## Required repairs

### A. Atomic local authority record

Make _verify_bound_authority() return an immutable local authority record
containing, at minimum:

- a fresh fully validated canonical descriptor;
- the captured verified contract ID.

It may be a private frozen dataclass, tuple, or equivalent, but it must not
depend on further reads from self.

to_canonical_descriptor() must return a fresh copy of that local descriptor.
build_genesis_snapshot() must use the captured local contract ID when
constructing all extra source manifests; after verification it must not read
self.contract_id, _bound_descriptor_text, _bound_descriptor_hash, or any other
mutable public/private contract field.

Add a deterministic regression that mutates only contract_id after verification
and before source-manifest composition (a trace hook is acceptable). The API
must either return a result retaining the original captured contract ID and
corresponding baseline field ID, or fail closed as IdentityV2ConfigError. It
must never emit the forged ID or a changed genesis due to that mutation.

### B. Full strict canonical descriptor validation

Bound text must not merely be JSON. Before it is returned or used:

1. Require exact base strings and recompute the SHA-256 as in R7. Normalize
   UTF-8 encoding failures, including an unpaired surrogate in bound text, to
   IdentityV2ConfigError before any descriptor/genesis material is returned.
2. Parse it with the same strict JSON behavior as the public parser:
   duplicate keys and non-finite constants must reject.
3. Validate the complete nested descriptor schema and all current v2
   invariants: exact top-level/identity/migration keys, known schemas,
   base-v1 ID, safe isolated root, charter schema/version/hash/text,
   charter size and substrate, sealed context role, envelope budget,
   migration authority, and migration prohibition.
4. Rebuild a descriptor from only the validated primitive values through a
   non-virtual helper, serialize it with canonical_json_bytes, and require
   byte equality with captured bound text as well as the verified hash/ID.
5. Normalize malformed descriptor shape/type/key/index errors to
   IdentityV2ConfigError. No raw KeyError, TypeError, AttributeError, or
   JSONDecodeError may escape.

Avoid recursive construction: if useful, factor the strict JSON and descriptor
validation now embedded in parse_identity_v2_contract into a private
side-effect-free descriptor-input helper shared by public parsing and authority
verification. Constructing/parsing must remain in-memory only.

Also normalize arbitrary errors raised by a direct-construction PathLike
object's __fspath__ method in _absolute_path() to IdentityV2ConfigError. No
direct public parser/constructor boundary in this packet may leak a raw
RuntimeError from an attacker-controlled conversion method.

### C. Behavior-first regressions

Add tests proving:

1. The exact post-verification ID-mutation hook cannot emit a manifest holding
   the forged ID or alter the baseline genesis field ID.
2. A coherent triple with a valid but malformed JSON object fails closed as
   IdentityV2ConfigError for both descriptor and genesis APIs.
3. A coherent triple with valid pretty-printed/noncanonical JSON fails closed.
4. A coherent triple with a syntactically valid but semantically forged
   descriptor (wrong root/base ID/charter/migration/role or missing keys) fails
   closed, with no raw KeyError.
5. A one-field bound-text mutation containing an unpaired surrogate and a
   direct PathLike whose __fspath__ raises RuntimeError both fail closed as
   IdentityV2ConfigError with no raw conversion/encoding exception.
6. Existing R7 text-only/hash-only/ID-only, non-str, exact direct-construction,
   ADS, sample ID, temporary-root, and public-mutation regressions remain
   green.
7. Valid ordinary sample parsing, descriptor copy isolation, v2 genesis, and
   valid non-genesis serialize/deserialise byte roundtrips remain unchanged.

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
adversarial probe for every R8 case plus valid v2 genesis and successor
serialize/deserialise byte roundtrips. Verify frozen JSON blobs remain exactly
59d35b0bb5ad74df166d315272fc306e17df6dcf and
7ae72883b2c3e059329cf0d464a61ef796b57843. Inspect allowed code/tests/report
for trailing whitespace and final newline. Do not deliberately create cache or
bytecode files.

## Evidence and stop condition

Create roundtable\Kimmy_response_to_Codex_identity_v2_contract_r8_04August.md
with headings:

1. STATUS
2. R1-R7 QUARANTINE AND R8 SCOPE
3. PRE-EDIT HASHES
4. ATOMIC AUTHORITY CAPTURE REPAIR
5. STRICT CANONICAL DESCRIPTOR REPAIR
6. TESTS AND ADVERSARIAL REPROS
7. CONTINUITY CORRECTION
8. POST-EDIT FILE HASHES AND SCOPE
9. V1/STATE PRESERVATION
10. RISKS AND DEFERRED WORK
11. RECOMMENDED NEXT PACKET

'STATUS' may be READY_FOR_CODEX_AUDIT only if every gate passes, frozen
contract/CPU-smoke blobs are byte-identical, every R8 adversarial case fails at
the required boundary, valid v2 genesis/successor roundtrips pass, no protected
state root exists, regenerated injection bundle mentions R8, report/log exist,
and no out-of-scope Axon source changed. Otherwise report QUARANTINE or
BLOCKED_SCOPE. Stop after reporting.
