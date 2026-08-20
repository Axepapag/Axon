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

## Current D64 Reconciliation Path

No training launcher is currently authorized to update a model. `training/complete_field_64d.py` remains the V6 mechanism baseline because it proves complete ten-region coverage before decoding, but `training/train_complete_field_64d.py` is now explicitly legacy/direct-record unless invoked with `--legacy-record-direct`. The next active implementation is a D64 adapter that feeds the same canonical `SharedFieldSnapshot`/Field Compiler interface to both runtime and training branches under `State/training`.

The older fixed-window conversational trainer remains preserved under `archive/legacy_fixed_window_trainer_2026-08-18/`; the ExactV4 runtime foundation is likewise regression/recovery evidence, not the active organism. See `docs/CANONICAL_STATE_RECONCILIATION.md`.

## Recovered Dormant Corpus

`curator/recovered_corpus_builder.py` produced the recovered dormant body now persisted under `State/dormant/`. That exact corpus is the canonical dormant-memory authority; historical `D:\00` sources remain read-only provenance, not the runtime memory interface.

## Canon

`docs/SOURCE_OF_TRUTH.md` is authoritative. If code and docs disagree, stop and fix the source of truth before adding more machinery.
