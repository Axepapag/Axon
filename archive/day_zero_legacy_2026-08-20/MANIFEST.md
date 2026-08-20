# Day Zero Legacy Archive Manifest

Archive date: 2026-08-20
Pre-cleanup Git HEAD: `9116ca7adfa6d28a6ff1b7914fe3925da137c722`
Active authority after cleanup: `docs/SOURCE_OF_TRUTH.md` + `docs/DAY_ZERO.md`

## Purpose

This archive is the single tracked historical namespace for implementation material removed from Axon's active Day Zero surface. Archived files are evidence only. They are not importable fallback runtime/training policy and must not be restored without an explicit convener decision plus Source-of-Truth reconciliation.

## Tracked material moved here

The cleanup moved approximately 260 tracked paths out of active architecture locations. Major groups include:

- former `cores/` and private-soul implementation;
- separate `runtime/council/` body, UI, launcher, configs/tests;
- ExactV4/bootstrap/identity-v2/v3/projection/runtime persistence stacks;
- `runtime/bus/` and `runtime/table/` collaboration/runtime experiments;
- old tick loop, core proposer, multi-tick refiner;
- 384-slot field view/schedule/schema-v2/view-v2 lineages;
- legacy trainers, curriculum builders, soul pilots, checkpoint/resume tools;
- Kaggle/bundle/prototype launch scripts and old root launchers;
- dedicated tests/fixtures tied to archived execution lineages;
- historical ops/Kimi packets, review-only curriculum package, and superseded runtime/training docs;
- prior archive islands and deprecated Codex continuity material, consolidated beneath this namespace;
- the first Kimi repository swarm report, preserved as historical pre-cleanup evidence.

## Local State/artifact archive

Large ignored/local artifacts are preserved separately at:

`D:\Axon\State\archive\day_zero_legacy_20260820\`

Measured after relocation:

| Area | Files | Bytes |
|---|---:|---:|
| canonical `State/dormant` remaining live | 6 | 1,094,878,576 |
| archived council State | 14 | 385,065 |
| archived legacy souls | 2 | 148,848 |
| archived training curricula | 36 | 39,030,916 |
| archived checkpoints | 2 | 203,050,854 |
| archived runs | 514 | 23,391,642,821 |
| archived datasets | 113 | 5,615,635,115 |
| archived generated distributions | 2,261 | 105,718,043,852 |
| archived Kaggle upload staging | 4 | 632,524,729 |
| archived local sensitive/config helpers | 3 | 9,080 |

The local archive was moved on the same `D:` volume; it was not promoted into Git. Credential-looking local helpers were moved without copying their contents into tracked files.

## Dormant-memory boundary

`State/dormant` remains the one canonical recovered memory body. The only pre-Day-Zero council-specific dormant artifact removed from the live directory was `council_field_tails.jsonl`; it is preserved in the local council archive. The remaining live dormant files are:

- `containers.jsonl`
- `semantic_edges.jsonl`
- `kg_cache_50k.jsonl`
- `layout_groups.jsonl`
- `symbol_registry.jsonl`
- `corpus_manifest.json`

No second canonical memory database was created.
