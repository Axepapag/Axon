# State/ — Axon's single canonical State body

`D:\Axon\State` is the one living/durable State root for Axon. Runtime and training may create governed branches or derived views inside this tree, but no component may create a competing Axon state universe.

## Day Zero anatomy

- `active/` — reserved for canonical active branch state. Pre-Day-Zero council field files were archived.
- `dormant/` — authoritative recovered dormant-memory corpus and exact supporting metadata. The recovered containers/semantic edges/layout/provenance body remains in place.
- `souls/` — reserved for private core souls attached to the future canonical D64 runtime. Pre-Day-Zero council/legacy soul files were archived.
- `training/` — canonical training branch/run area. Day Zero begins with no pre-authorized live curriculum/run lineage; historical curricula and runs are archived.
- `archive/` — superseded State lineages, historical runs/datasets/checkpoints/distributions, and reconciliation evidence. Archive material is never live authority.
- `kimi_orchestrator/` — durable supervised Kimi job evidence/locks; engineering infrastructure only.
- `transfer/` — explicit transfer/import-export staging only; never canonical truth by itself.

## Hard boundaries

- There is one dormant-memory authority: the exact recovered corpus under `State/dormant`. Derived indexes/embeddings are rebuildable senses, not memory authority.
- There is one canonical shared-field contract. Small tests may use small canonical branches; they may not substitute a fake/truncated core-facing anatomy.
- There is no active council State lineage after the Day Zero cleanup.
- Historical/pre-Day-Zero artifacts may be inspected for provenance or mechanism recovery but may not be resumed or promoted implicitly.

Day Zero historical State archive:

`State/archive/day_zero_legacy_20260820/`

Earlier reconciliation archive:

`State/archive/pre_canonical_reconciliation_20260820/`

See `docs/DAY_ZERO.md` and `docs/SOURCE_OF_TRUTH.md` for the active code/anatomy contract.
