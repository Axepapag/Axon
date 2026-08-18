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
python training/trainer_slot.py --smoke --threshold charslot
python runtime/tick_loop.py --dry-run --list-checkpoints
```

## Current Training Path

`training/trainer_slot.py` is charfield-only. It trains cores on:

- exact 16D field input,
- copy / partial / blank response draft modes,
- suffix-only scoring for partial completions,
- story-aware context where available,
- soul inhale/exhale every step.

The default small rung is core A: `d_model=64`, 2 layers, 1 head, large FFN.

## Recovered Dormant Corpus

`curator/recovered_corpus_builder.py` converts recovered `D:\00` sources into dormant container records and spelled-out edge records. Generated artifacts live under `datasets/recovered/` and are ignored by Git.

## Canon

`docs/SOURCE_OF_TRUTH.md` is authoritative. If code and docs disagree, stop and fix the source of truth before adding more machinery.
