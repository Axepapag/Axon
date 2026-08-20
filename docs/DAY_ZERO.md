# Axon Day Zero Map

Date: 2026-08-20
Authority: `docs/SOURCE_OF_TRUTH.md`

Day Zero is the clean implementation baseline after removing executable parallel anatomies from the active repository surface.

## One doctrine

`docs/SOURCE_OF_TRUTH.md` is the master architecture text. Root `SOURCE_OF_TRUTH.md` is an exact compatibility mirror. The same exact-mirror rule applies to `docs/WORKING_CONTRACT.md` and root `WORKING_CONTRACT.md`.

## One State body

`D:\Axon\State` is Axon's only living/durable State root.

- `State/dormant/` — authoritative recovered dormant-memory corpus and exact supporting metadata.
- `State/active/` — reserved for canonical active branch state. Pre-Day-Zero council field files were archived.
- `State/souls/` — reserved for future canonical private-core soul state. Pre-Day-Zero soul files were archived.
- `State/training/` — canonical training branch/run area. Day Zero begins with no pre-authorized live curriculum or run lineage.
- `State/archive/` — historical evidence only; never live authority.
- `State/kimi_orchestrator/` and `State/transfer/` — orchestration/transfer infrastructure, not alternate cognitive bodies.

## Active cognitive/runtime code

### Canonical field

- `runtime/field/schema.py`
- `runtime/field/delta.py`
- `runtime/field/compiler_d64.py`
- `runtime/field/state_branch.py`

### Runtime-facing D64 boundary

- `runtime/axon_runtime/d64_adapter.py`

No production neural runtime driver exists yet. That absence is explicit; no archived runtime is an implicit fallback.

### Canonical D64 training path

- `training/canonical_d64.py`
- `training/complete_field_64d.py`
- `training/train_complete_field_64d.py`

The active trainer has no detached-record/legacy-anatomy switch. Curriculum JSON is source material only; the core receives canonical snapshots through the D64 compiler.

### Dormant-memory construction utilities

- `curator/container_schema.py`
- `curator/dormant_materializer.py`
- `curator/recovered_corpus_builder.py`
- `curator/semantic_layout_machine.py`

These are offline exact-corpus utilities. The next runtime memory work is a rebuildable retrieval/index bridge that dereferences exact `State/dormant` records into canonical active state; it must not create a second memory authority.

## Engineering infrastructure kept active

- `scripts/run_kimi_roundtable.py`
- `scripts/run_supervised_kimi_packet.py`
- `.agents/agents/`

These coordinate engineers/tools. They are not Axon's cognitive runtime.

## Archived execution lineages

Tracked pre-Day-Zero implementations live under:

`archive/day_zero_legacy_2026-08-20/`

Ignored/local run artifacts and old State lineages live under:

`State/archive/day_zero_legacy_20260820/`

Do not import, launch, resume, or restore archived architecture without an explicit new decision and Source-of-Truth reconciliation.

## Day Zero build order

1. Build the canonical dormant evidence bridge over the existing `State/dormant` body.
2. Build a non-neural production D64 driver contract: load HEAD -> compile -> freshness/coverage barrier -> typed proposal -> validate/commit -> recompile.
3. Make training episode transactions genuinely branch-backed and prove canonical checkpoint/resume.
4. Attach private souls, multi-core refinement/consolidation, tools/advisors, diary, and situation awareness only through that permanent D64 path.
5. Benchmark realistic complete-field time/memory before increasing model scale.

The rule for every future experiment remains: a smoke can be small; its anatomy cannot be fake.
