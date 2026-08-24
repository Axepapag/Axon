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
- `runtime/heart/intelligence.py` — governed Heart translator/ensemble/fidelity contracts
- `runtime/heart/translation_core.py` — first permanent 64D/2-layer/4096-FFN non-authoritative Heart translator tissue; no learned Heart translator is serving yet
- `training/heart_translation.py` + `scripts/train_heart_translation_smoke.py` — provenance-bound three-dialect curriculum/evaluation and Trainer-governed bounded Heart candidate training

The dormant evidence bridge keeps exact memory authoritative in `State/dormant/*.jsonl`. Its local SQLite index is disposable lookup metadata only; lexical postings use SHA256 term keys plus integer row references rather than copied text/IDs, and selected stable IDs are dereferenced and hash/provenance-verified from the JSONL before surfacing into `cortex` and D64.

Build or verify the derived sense from the canonical State root:

```powershell
python scripts/build_dormant_evidence_index.py
python scripts/build_dormant_evidence_index.py --verify-only
```

Trainer organ control plane:

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

Current developmental D64 training tissue:

- `training/canonical_d64.py`
- `training/complete_field_64d.py`
- `training/train_complete_field_64d.py`

The Trainer now owns governed **isolated candidate**, activation, and rollback boundaries as well as parameter governance/observability. It never optimizes the registered live organ in place: mutations are grant-scoped and every candidate runs under an immutable content-addressed learning policy governing AdamW/SGD hyperparameters, weight decay, deterministic scheduling, gradient accumulation, clipping/budgets, and FP32/BF16/FP16 precision. Mid-accumulation checkpoints preserve optimizer, pending gradients, AMP scaler state when present, counters, LR and accumulated-loss state for exact resume. Deterministic evaluation gates precede promotion, and only the leased Trainer can atomically advance an active-generation pointer after saving an exact rollback target. Restart hydration and reversible rollback are explicit. Use `python scripts/inspect_trainer.py` for read-only current Trainer status, including the latest learning policy/microstep and active generations. No real core-training campaign is launched by this anatomy. The present 64D model path is a proving width, not a final limit on Axon's heterogeneous semantic/reasoning/trainer ensembles.

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

Train the **Heart translation/conduction ensemble first**. The first permanent 64D Heart translator, synthetic provenance-bound curriculum, heldout/counterfactual harness, and governed smoke path now exist. The first real 12-step CUDA candidate lowered loss but still scored 0.0 grounded roundtrip/aggregate semantic fidelity and was rejected without activation, so no longer Heart campaign is justified yet. Next improve bounded semantic/grounding learning evidence and build exact autobiographical experience capture in parallel so Heart translations/corrections become future lived curriculum. Mature topology still requires at least three accepted serving Heart translators plus one isolated candidate-learning lane. Reasoning and Semantic Cortex training remain parked; `semantic_cortex` stays CLOSED.
