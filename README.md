# Axon

Axon is a token-free, slot-based, English-bearing substrate agent. The frozen
16D character substrate (`substrate/substrate.py`) is the alphabet. The shared
field is made of 8192D deterministic slots, packed by concatenating frozen 16D
codes. Multiple reasoning cores at 64/128/256D attend over the same slot field
through frozen shared-state adapters, with rotating consolidation and private
temperature-tiered souls.

## Folder map

```
D:\Axon/
├── docs/
│   ├── SOURCE_OF_TRUTH.md          — locked doctrine (authoritative)
│   ├── WORKING_CONTRACT.md         — governs all collaborator agents
│   └── roundtable/                 — resolution, deltas, round-table briefs
├── substrate/                      — frozen 16D character alphabet
├── slots/                          — 8192D slot spec + field contract
├── adapters/                       — frozen shared-state adapters (8192D ↔ d_model)
├── cores/                          — transformer core + temperature-tiered soul
├── training/                       — curriculum builders, future trainers, probes
├── curator/                        — container schema, KG search, dormant builders
├── runtime/                        — collaboration bus
├── tests/                          — full test suite
├── PROVENANCE.md                   — archive source + promoted file ledger
└── pyproject.toml
```

## Archive policy

- `D:\AxonGliksbot` and `D:\axon7` are **read-only archives**. Never modify,
  move, or delete within them. Copy from them only via documented promotion.
- See `PROVENANCE.md` for the archive commit hash and per-file source paths.

## Quick checks

```powershell
# Run all tests (200 tests: promoted + Lane 0 slot integrity)
python -m pytest

# Substrate self-test
python substrate/substrate.py

# Slot spec self-test (10 checks)
python slots/slot_spec.py

# Field contract self-test (8 checks)
python slots/slot_field_contract.py

# Adapter gates (snap-idempotence + separability at 64/128/256D)
python adapters/mint_adapters.py
python adapters/slot_adapter.py --check

# Recovered dormant-state smoke build
python curator/recovered_corpus_builder.py --smoke --out-dir datasets/recovered/dormant_state_smoke

# Recovered curriculum smoke build
python training/build_recovered_curriculum.py --smoke `
  --containers datasets/recovered/dormant_state_smoke/containers.jsonl `
  --registry datasets/recovered/dormant_state_smoke/symbol_registry.jsonl `
  --edges datasets/recovered/dormant_state_smoke/semantic_edges.jsonl `
  --out-dir datasets/recovered/curriculum_smoke
```

## Architecture summary

See `docs/SOURCE_OF_TRUTH.md` for the full doctrine. Key points:

- **Layer 0**: frozen 16D substrate alphabet (63 chars, no learned parameters).
- **Layer 0+**: 8192D slots (4096D text + 2048D edge + 512D control + 1536D reserved).
- **Layer 4**: semantic search surfaces dormant knowledge into active state each tick.
- **Layer 5**: frozen adapters project 8192D → d_model (lossy read), codebook-snap
  proposals → exact substrate (exact write).
- **Layer 6**: cores with private temperature-tiered souls (hot/warm/cold/frozen).
- **Layer 7**: forever tick loop, CPU-resident, matmul-light.
- **Layer 8**: cores produce deltas in 8192D slot field space; consolidator commits.
- **Layer 13**: discrete CE loss, smoke gates, no silent truncation, cf-probe-style proof.

## Recovered Dormant Corpus

`curator/recovered_corpus_builder.py` converts the recovered `D:\00` SQLite/JSON
sources into dormant `Container` records, spelled-out semantic edge records,
layout metadata/groups, and a manifest. Sources are opened read-only. Personal
log entries are excluded unless `--include-personal-log` is passed, and then
they are tagged `diary_only`.

`training/build_recovered_curriculum.py` turns those dormant artifacts into
curriculum families: field surfacing, edge prediction, retrieval QA,
procedure next-step, episodic exhale-filter Tick A/B,
contradiction/alias curation, and diary-only self-reflection.

Generated artifacts live under `datasets/recovered/` and are intentionally
ignored by Git.

## Adapter design (v2)

The adapter has asymmetric read/write:

- **DOWN (read)**: one frozen d_model vector per slot (lossy, deterministic).
  The core sees one summary per slot. Exact text lives in the 8192D field.
- **UP (write)**: codebook-snap every 16D block to nearest substrate code.
  Committed content is exact by construction.
- **Gates**: snap-idempotence (valid slots survive unchanged) and separability
  (distinct slots produce distinct projections).
- **Core-owned output**: trainable write behavior lives inside the core. The
  adapter only performs frozen projection and deterministic substrate snapping.

## Identity

Hermes / glm-5.2:cloud / 2026-07-03
