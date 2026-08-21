# Axon

Axon is an exact-character, stateful AI architecture built around one canonical shared field, one canonical State body, private reasoning cores, auditable dormant memory, and validated typed deltas.

## Start here

1. `AGENTS.md`
2. `docs/WORKING_CONTRACT.md`
3. `docs/SOURCE_OF_TRUTH.md`
4. `docs/DAY_ZERO.md`
5. `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`
6. `roundtable/ENGINEERS_LEDGER.md`

`docs/SOURCE_OF_TRUTH.md` is the architecture master. Root `SOURCE_OF_TRUTH.md` is an exact compatibility mirror and is test-enforced to match.

## Day Zero active anatomy

Canonical field/runtime boundary:

- `runtime/field/schema.py`
- `runtime/field/delta.py`
- `runtime/field/compiler_d64.py`
- `runtime/field/state_branch.py`
- `runtime/axon_runtime/d64_adapter.py`
- `runtime/dormant/evidence_bridge.py`

The dormant evidence bridge keeps exact memory authoritative in `State/dormant/*.jsonl`. Its local SQLite index is disposable lookup metadata only; lexical postings use SHA256 term keys plus integer row references rather than copied text/IDs, and selected stable IDs are dereferenced and hash/provenance-verified from the JSONL before surfacing into `structured_knowledge` and D64.

Build or verify the derived sense from the canonical State root:

```powershell
python scripts/build_dormant_evidence_index.py
python scripts/build_dormant_evidence_index.py --verify-only
```

Canonical D64 training:

- `training/canonical_d64.py`
- `training/complete_field_64d.py`
- `training/train_complete_field_64d.py`

Recovered dormant-memory construction/audit utilities:

- `curator/container_schema.py`
- `curator/dormant_materializer.py`
- `curator/recovered_corpus_builder.py`
- `curator/semantic_layout_machine.py`

Engineering collaboration infrastructure:

- `.agents/agents/`
- `scripts/run_kimi_roundtable.py`

There is intentionally no production neural D64 runtime driver yet. Archived runtimes are not fallback production paths.

## Canonical State

`D:\Axon\State` is the sole living/durable State root. The recovered dormant-memory body remains at `State/dormant/`. Training source/branch/workspace material belongs under `State/training/`.

Pre-Day-Zero council state, legacy souls, historical runs, datasets, checkpoint bundles, and generated distributions are preserved under `State/archive/day_zero_legacy_20260820/`.

## Historical implementations

Superseded tracked runtimes, council code, old cores/souls, 384-slot views, ExactV4/identity-v2 stacks, legacy trainers/curricula, old launchers, policies, review packets, and their dedicated tests are preserved under:

`archive/day_zero_legacy_2026-08-20/`

They are evidence, not active anatomy. Do not import, launch, resume, or restore them without an explicit convener decision and Source-of-Truth reconciliation.

## Verification

Focused Day Zero verification:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_day_zero_hygiene.py tests/test_dormant_evidence_bridge.py tests/test_field_compiler_d64.py tests/test_canonical_state_branch.py tests/test_canonical_d64_training_adapter.py
```

Full active suite:

```powershell
python -m pytest -q -p no:cacheprovider
```

## Next build target

Connect the real `State/dormant` body to canonical D64 through a rebuildable retrieval/index bridge that dereferences exact authoritative records and surfaces provenance-bearing evidence into canonical active state. Do not create a second dormant-memory authority.
