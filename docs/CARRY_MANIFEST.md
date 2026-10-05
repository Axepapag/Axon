# Carry manifest

What was carried into this repository on 2026-10-05, from where, what was edited, and what was left behind.
Sources: `OLD` = `G:\My Drive\Projects\Axon_old\Axon_old` (the old Axon repo, 458 commits, HEAD e901604, Sep 29),
`AXON7` = `G:\My Drive\New folder (2)\axon7`, `V6` = `G:\My Drive\New folder (2)\ashes_v6_history`.
All three sources were **copied from, never modified**. The only change in OLD this session is the Engineer's Ledger
event appended before the rename.

Principle: carry what is clean and proven; where code was woven into rails/D64, do not port it by renaming, rewrite it.

## 1. Carried code

Verified: `python -m pytest` = 144 passed; `substrate.py` and `substrate_1024.py` self-tests exit 0; all 96 frozen
16D vectors (95 characters + `<empty>`) are bit-identical to OLD.

### Byte-identical to OLD (34 files)

`substrate/substrate_1024.py`, `substrate/d1024_lane_bank.npy`, `substrate/v7_reference_bank.npy`,
`runtime/source_of_truth.py` + `configs/source_of_truth/capacity_policy.json`,
`runtime/soul/` (`__init__`, `contracts`, `store` is edited, see below),
`runtime/dormant/` (`__init__`, `evaluation`, `experience`, `generations`, `incremental`, `relevance`),
`runtime/heart/` (`authority`, `autobiography`, `durable_ingress`, `health`, `identity`, `ingress_queue`, `lease`, `turns`, `valve`),
`scripts/append_engineers_ledger_event.py`, and 10 tests (five `test_dormant_*`, `test_heart_{durable_ingress,lease,valve}`,
`test_engineers_ledger_append`, `test_substrate_contract`).

### Edited (and exactly how)

| File | Edit |
|---|---|
| `substrate/substrate.py` | `char_to_slot` and `assert_supported_text` now raise `UnsupportedCharacterError` for anything outside the 95 (and `<empty>`). No vector value changed. |
| `substrate/native.py` (new) | Native 95 API: `native_id`, `native_char`, `native_cell16`, `native_bank`, `encode_ids`, `decode_ids`, `assert_native_text`, `text_to_cells16`. |
| `substrate/__init__.py` | Exports the 16D substrate and the native API. The old byte-transport exports are gone. |
| `runtime/field/schema.py` | `FieldSpan` rejects any text outside the 95, so nothing else can enter the canonical body by any path. Rail wording removed from docstrings. |
| `runtime/field/delta.py` | Gained `replacement_delta` (moved from the D64 compiler; the compiled-rail freshness parameter was dropped). |
| `runtime/field/d16_view.py` | Ported from the 351-token transport to the native 95: one cell per character, `native_id` replaces the `transport_*` fields, `substrate_schema` replaces `transport_schema`. Behaviour (views, addresses, region-granular deltas) unchanged. |
| `runtime/field/state_branch.py`, `runtime/soul/store.py` | Default roots moved from `D:\Axon\State` to the repo's `State/` folder. |
| `runtime/field/__init__.py` | D64 compiler, semantic D64 surface and training view exports removed. |
| `runtime/dormant/evidence_bridge.py` | D64 compile step removed; `query_surface_compile` became `query_surface`; `SurfacedDormantResult` lost its `compiled` field. |
| `runtime/heart/errors.py` | Rail-named errors removed (`StaleRailBindingError`, `RailWidthMismatchError`, `RailMembershipError`). |
| `runtime/heart/masks.py` | One docstring word. |
| `runtime/heart/__init__.py` | New, minimal: exports only what is carried. |
| `curator/import_d00_memories.py` | Loads recovered legacy memories into Dormant. `--source-root` is now required; state root defaults to repo `State/`; `D:\` defaults removed. |

### Tests

Carried unchanged: 10 (list above). Carried and edited: `test_canonical_state_branch` (D64 compile steps replaced by
D16 views), `test_heart_region_masks` (host/registry tests dropped; mask-movement and address tests rewritten on D16
views), `test_dormant_evidence_bridge` and `test_dormant_incremental` (D64 and coordinator parts removed),
`test_private_soul` (trainer-bound candidate test replaced by a Soul-branch fork test),
`test_no_fixed_character_poison` (legacy-module half removed), `test_substrate_1024`. New: `test_native_substrate_law`
(95-only law at every boundary; Heart 16D -> 1024D -> 16D round trip).
Left behind: `test_soul_delayed_recall_probe` (old reasoning circulation) and the roughly 100 D64/rail/trainer tests.

## 2. Carried as record or reference (never imported)

- `history/old_axon/`: the full canonical ledger (470 events), the rolling summary, the Sep 24 `SOURCE_OF_TRUTH`
  (the newer of the two copies in OLD), identity v1, Kaggle and trainer guides, the no-ceilings audit, today's Codex audit.
- `history/axon7/`, `history/ashes_v6/`: audits, field contract, v6 doctrine, v6 notes, the commandments text.
- `reference/axon7/`: `train_menu.py`, `trainer_v2.py`, `field_contract.py`, `validate_dataset.py`, `runtime_eval.py`,
  and the four curriculum generators. Source for the new trainer's UX and bookkeeping; known gaps are listed in `reference/README.md`.
- `reference/ashes_v6/`: `dormant_store.py`, `idle_refinery.py` (the compress-memory-while-idle idea that fits the Soul),
  `pump_soul.py`, `pump_dialogue.py`, `commandments.py`, `axon_guard.py`, `training_bucket.py`, `letter_substrate.py`, `tune_substrate.py`.
- `curricula/legacy_raw/`: 37 raw curriculum files (28.7 MB; git-ignored) with `CURRICULA_AUDIT.md`.

## 3. Left behind (still in place, nothing deleted)

### OLD: rewrite later, using the old file only as reference (rail-shaped)

`runtime/heart/`: `tick.py`, `registry.py`, `transaction.py`, `board.py`, `circulation.py`, `coordinator.py`, `host.py`,
`core_bus.py`, `mirror_coherence.py`, `english_reasoning.py`, `reasoning_recovery.py`, `proposal_workspace.py`,
`intelligence.py`. They hold the transaction state machine, barrier and mirror-coherence ideas the new Heart needs, but are
organised around rails or D64 images (for example `tick.py` is 94 D64/rail hits).
`runtime/axon_runtime/continuous_core_d512.py` is the closest existing GRU and a useful reference for the new core.

### OLD: left behind for good (prohibited by the new direction)

`runtime/field/{compiler_d64,semantic_d64,training_view}.py`, `runtime/heart/{d64_codec,remote_rail,rail_auth,continuous_d16_port,translation_core,reasoning_output}.py`,
`runtime/axon_runtime/d64_adapter.py`, `substrate/unicode_transport.py` (351-token byte transport), `Cortext/` (D64 semantic layer),
and the D64/pointer/trainer scripts.

### OLD: later, if needed

`curator/{container_schema,dormant_materializer,recovered_corpus_builder,semantic_layout_machine}.py`, `controlplane/` (an
API design that was never built), `android/` (a real Kotlin client, 119 files), dormant index scripts in `scripts/`
(`build_dormant_evidence_index`, `maintain_dormant_index`, `evaluate_dormant_relevance`, `verify_dormant_evidence_real_index`),
`ops/windows_vm`, `notebooks/Axon_Cloud_Preflight.ipynb`, `archive/` (Day-Zero legacy evidence, 367 files), `roundtable/` proposals and reviews.
The old git history also still contains the trainer packages Jeff deleted (`git show HEAD:runtime/trainer/...`, `HEAD:training/...`).

### AXON7 and V6: large binaries stay where they are

| Location | Size | Verdict |
|---|---:|---|
| `axon7/checkpoints` | 3,338 MB | old v7 transformer cores (D16/D128/D256); not compatible with the new architecture |
| `axon7/checkpoints_quarantine` | 1,627 MB | audit set with a manifest; its manifest is in `history/axon7/checkpoint_audit_manifest_2026-06-13.*` |
| `axon7/stand_by_cores` | 2,059 MB | smoke and reference snapshots |
| `V6/checkpoints` | 5,514 MB | v6 128D cores, mostly duplicate training snapshots |
| `V6/Core` | 49 MB | `core_seed_128C.pt`, the one v6 core worth keeping for history |
| `axon7/core.py, heads.py, runtime.py, substrate.py` | small | v7 transformer stack; superseded |

### Not found

The Dormant memory corpus (59,875 records, the old README's claim) is not in OLD: its `State/` held only a README, because
the corpus lived on the lost D: drive. Recovered memory databases named `brain.db`, `brain_recovered.db`, and
`AXON_CONVERSATION_MEMORY*.md` exist in `G:\My Drive\New folder (2)\`; they were **not searched or opened** (outside the
three requested locations). They may be the source for `curator/import_d00_memories.py`.
