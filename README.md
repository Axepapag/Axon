# Axon

Axon is a clean rebuild of Jeff's stateful, always-on AI architecture.

The current repo canon is simple:

- The frozen `substrate/` alphabet is the bedrock.
- Active shared state is represented as exact 16D character cells, grouped into named regions such as conversation history, user input, response draft, scratch, tool results, and surfaced knowledge.
- Cores attend the active character field through deterministic per-core lifts into their `d_model` lane.
- Cores produce deltas. Runtime code validates and commits deltas into the canonical field.
- Souls are private per-core state. A core inhales before attending, acts over the field, then exhales experience into hot soul rows.
- Dormant knowledge is structured, auditable, and masked until surfaced. Semantic edges are spelled out in English.

The rejected wide-slot adapter lineage has been deleted from active code and docs.

## Folder Map

```text
D:\Axon\
  substrate\      frozen 16D alphabet substrate
  cores\          transformer core and temperature-tiered soul
  training\       charfield trainer and curriculum builders
  curator\        container schema, KG search, recovered dormant builders
  runtime\        tick loop, bus, and round-table runtime
  docs\           source of truth and working contract
  tests\          focused regression tests
```

## Quick Checks

```powershell
python -m pytest -q -p no:cacheprovider
python substrate/substrate.py
python -m pytest -q -p no:cacheprovider tests/test_complete_field_64d.py
python -m pytest -q -p no:cacheprovider tests/test_axon_runtime_bootstrap.py
```

## Current D64 Canonical Path

The deterministic shared D64 adapter is implemented. `runtime/field/compiler_d64.py` compiles one immutable `SharedFieldSnapshot` into reversible region-aligned D64 rows containing four exact 16D lanes plus exact lane provenance. `runtime/axon_runtime/d64_adapter.py` exposes the same compiler contract to runtime, while `training/canonical_d64.py` and `training/complete_field_64d.py` feed the V6 reader from that rail and use real typed `FieldDelta` transactions for scratch/response commits and interventions.

`training/train_complete_field_64d.py` now defaults to the shared canonical D64 anatomy and a workspace under the real `State/training` tree; `--canonical-d64` may be supplied as an explicit assertion, while `--legacy-record-direct` is preserved only for mechanism archaeology. `TRAIN_COMPLETE_FIELD_64D_R0.bat` remains a non-training guard, so this implementation does not silently start another GPU run. The next architecture work is canonical dormant retrieval plus the production D64 neural runtime driver.

The older fixed-window conversational trainer remains preserved under `archive/legacy_fixed_window_trainer_2026-08-18/`; the ExactV4 runtime foundation is likewise regression/recovery evidence, not the active organism. See `docs/CANONICAL_STATE_RECONCILIATION.md`.

## Recovered Dormant Corpus

`curator/recovered_corpus_builder.py` produced the recovered dormant body now persisted under `State/dormant/`. That exact corpus is the canonical dormant-memory authority; historical `D:\00` sources remain read-only provenance, not the runtime memory interface.

## Canon

`docs/SOURCE_OF_TRUTH.md` is authoritative. If code and docs disagree, stop and fix the source of truth before adding more machinery.
