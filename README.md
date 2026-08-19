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
python runtime/tick_loop.py --dry-run --list-checkpoints
```

## Current Training Path

`training/train_complete_field_64d.py` is the active R0 trainer. It trains one shallow 64D core to sweep every active character in all ten regions, retain the encoded page tokens as addressable decoder memory, write scratch, reread the complete committed field, and write a variable-length response draft. Coverage manifests block decoding on gaps. Diary, conversation history, and tool results remain sealed. See `docs/COMPLETE_FIELD_64D_R0_TRAINING.md`.

The prior direct fixed-window conversational trainer is preserved under `archive/legacy_fixed_window_trainer_2026-08-18/` and is not an active launch path.

## Recovered Dormant Corpus

`curator/recovered_corpus_builder.py` converts recovered `D:\00` sources into dormant container records and spelled-out edge records. Generated artifacts live under `datasets/recovered/` and are ignored by Git.

## Canon

`docs/SOURCE_OF_TRUTH.md` is authoritative. If code and docs disagree, stop and fix the source of truth before adding more machinery.
