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

The active developmental trainer has no detached-record/legacy-anatomy switch. Curriculum JSON is source material only; the core receives canonical snapshots through the D64 compiler. Archived Bible checkpoints remain historical evidence and are not an active initialization path.

### Trainer organ control plane

- `runtime/trainer/contracts.py`
- `runtime/trainer/registry.py`
- `runtime/trainer/authority.py`
- `runtime/trainer/activation.py`
- `runtime/trainer/telemetry.py`
- `runtime/trainer/store.py`
- `runtime/trainer/lifecycle.py`
- `runtime/trainer/execution.py`
- `runtime/trainer/learning.py`
- `runtime/trainer/gates.py`
- `runtime/trainer/lease.py`
- `runtime/trainer/inspection.py`
- `runtime/trainer/host.py`
- `scripts/inspect_trainer.py`

This is now a governed candidate-learning and activation boundary, not merely bookkeeping. One OS-backed Trainer writer owns parameter-state mutation; live registered organs are never optimized in place; authorized learning occurs on isolated candidate clones under immutable content-addressed learning policies governing optimizer hyperparameters, weight decay, constant/warmup-cosine scheduling, gradient accumulation, clipping, gradient/update budgets, and FP32/BF16/FP16 precision. Mid-accumulation checkpoints preserve optimizer, pending gradients, AMP scaler state when present, counters, LR and accumulated-loss telemetry for exact resume. A passed proposal can be activated only by the leased Trainer through stale-inventory checks, exact candidate verification, a durable rollback snapshot, and atomic active-generation pointer publication; rollback and restart hydration are explicit. Unit-scale synthetic candidates prove the mechanism; no real semantic/reasoning training campaign is authorized or running by this milestone.

### Dormant-memory construction utilities

- `curator/container_schema.py`
- `curator/dormant_materializer.py`
- `curator/recovered_corpus_builder.py`
- `curator/semantic_layout_machine.py`

These are offline exact-corpus utilities.

### Canonical dormant evidence bridge

- `runtime/dormant/evidence_bridge.py`

The bridge builds only disposable lookup metadata beneath `State/dormant/.derived/`: manifest/hash binding, byte offsets/lengths, hashes, compact SHA256 lexical postings, filters, and graph neighbor row references. Exact container/edge text, source strings, and provenance remain authoritative only in the existing JSONL corpus. Candidate stable IDs are dereferenced and verified from those exact bytes before evidence is surfaced into canonical `cortex` and compiled through D64. `scripts/build_dormant_evidence_index.py` is the canonical rebuild/verify operator entry point and refuses State/index roots outside the one canonical State tree.

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
