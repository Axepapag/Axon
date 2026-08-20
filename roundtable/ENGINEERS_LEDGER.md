# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-20T18:10:00-05:00
Current through event: `evt-20260820T231000000000Z-chatgpt-timeout-recovery`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission state

Axon is now at a clean **Day Zero** implementation boundary. Parallel historical execution anatomies have been removed from the active import/launch/test surface and preserved as archive evidence.

The next implementation packet is **P0: canonical dormant evidence retrieval/surfacing over the existing `State/dormant` body**. Do not restore an archived runtime/trainer first and do not create a second dormant-memory authority.

## Binding authority and invariants

- Architecture master: `docs/SOURCE_OF_TRUTH.md`.
- Root `SOURCE_OF_TRUTH.md` is an exact compatibility mirror; tests enforce equality.
- Root/docs Working Contract files are also exact byte mirrors.
- One living/durable State root: `D:\Axon\State`.
- One dormant-memory authority: exact recovered files under `State\dormant`.
- Exact visible text remains grounded in the frozen 16D character substrate.
- The deterministic D64 compiler is authority-free: compile/cartography/coverage/freshness/provenance only.
- Cores/consolidation may reason/propose; typed delta validation and canonical transaction remain the commit boundary.
- A smoke may be small in content/compute; its core-facing anatomy may not be fake or truncated.
- Archived code/state is evidence only and is never an implicit fallback.

## Day Zero publication

Cleanup implementation commit:

- `7db5e37` — `Establish clean Day Zero repository` — pushed to private `origin/main`.

The preceding interrupted Kimi-swarm continuity was first completed and published as:

- `9116ca7` — `Publish first Kimi swarm continuity`.

## Active Day Zero implementation surface

### Canonical field/runtime

- `runtime/field/schema.py`
- `runtime/field/delta.py`
- `runtime/field/compiler_d64.py`
- `runtime/field/state_branch.py`
- `runtime/axon_runtime/d64_adapter.py`

No production neural D64 runtime driver exists yet. That absence is explicit.

### Canonical D64 training

- `training/canonical_d64.py`
- `training/complete_field_64d.py`
- `training/train_complete_field_64d.py`

The trainer is canonical-only:

- no `--legacy-record-direct`;
- no `--canonical-d64` compatibility flag;
- no active detached `forward_transaction` / `run_transaction` path;
- no active `CompleteFieldPager` coverage authority;
- core reads enter through a canonical compiled D64 rail;
- State root must resolve exactly to `D:\Axon\State`;
- run/checkpoint directory must resolve beneath `State\training\runs`.

Day Zero begins with no pre-authorized live curriculum/run lineage.

### Dormant construction/audit utilities kept active

- `curator/container_schema.py`
- `curator/dormant_materializer.py`
- `curator/recovered_corpus_builder.py`
- `curator/semantic_layout_machine.py`

These are offline exact-corpus utilities, not a second cognitive runtime.

### Engineering orchestration kept active

- `.agents/agents/`
- `scripts/run_kimi_roundtable.py`
- `scripts/run_supervised_kimi_packet.py`

Kimi remains supervised workforce, never Axon publication/ledger/commit authority.

## Historical archive boundary

Tracked historical implementation is consolidated at:

`archive/day_zero_legacy_2026-08-20/`

It includes the former council body, old cores/souls, ExactV4/bootstrap/identity-v2/v3 runtime stacks, projections, 384-slot field/view/schedule paths, runtime bus/table, tick/refiner/proposer paths, legacy trainers/curricula/soul pilots, old launchers/scripts, ops/Kaggle/review packages, superseded docs/packets, and their dedicated tests/fixtures.

Local/ignored historical artifacts are preserved at:

`State/archive/day_zero_legacy_20260820/`

Measured preserved history includes:

- runs: 514 files / 23,391,642,821 bytes;
- datasets: 113 / 5,615,635,115 bytes;
- generated distributions: 2,261 / 105,718,043,852 bytes;
- checkpoints: 2 / 203,050,854 bytes;
- training curricula: 36 / 39,030,916 bytes;
- Kaggle staging: 4 / 632,524,729 bytes.

Ignored machine-local sensitive/config helpers were moved without promoting their contents to Git.

## Canonical State body

`State/active` and `State/souls` are currently empty/reserved for future canonical D64-attached state.

`State/training` is the canonical future branch/run area; no old curriculum/run lineage remains live by default.

`State/dormant` remains the recovered authority with six live files totaling **1,094,878,576 bytes**:

- `containers.jsonl`
- `semantic_edges.jsonl`
- `kg_cache_50k.jsonl`
- `layout_groups.jsonl`
- `symbol_registry.jsonl`
- `corpus_manifest.json`

The former 4,411-byte council-tail sidecar was archived with council State. No second canonical dormant DB/store was created.

## Doctrine mirror verification

Root/docs Source of Truth SHA256:

`582CC0BE258B9F744389BB4AF19D9C22D4DCCE6E508F9FDEC345F4B196048114`

Root/docs Working Contract SHA256:

`DC946600ADD64C51EC0AE40BB4D7F4F5E708387DD2A9E384DE268701959FAA31`

Each pair is byte-identical.

## Verification

Final Day Zero evidence:

- kept active Python modules compile with `py_compile`;
- focused hygiene + D64 compiler + canonical branch + training suite: **24 passed**;
- full active pytest: **exit code 0, 166 tests passed by progress count**;
- only existing PyTorch nested-tensor/norm-first warnings appeared;
- trainer rejected a noncanonical State root before data load;
- trainer rejected an external run directory before data load;
- staged `git diff --check` passed;
- source/contract mirror hashes matched;
- no model training, checkpoint promotion, Axon cognitive service start, or dormant corpus duplication occurred.

Non-blocking local caveat: ignored `.pytest_cache` remains at repository root because Windows returned EPERM when move/delete was attempted. It is not tracked or architectural.

## Kimi workforce state

Supervised Kimi orchestration remains available. Current intended usage policy:

- ordinary K2.7 Coding for routine read-only/workhorse triage;
- K3 for difficult architecture/synthesis;
- Highspeed opt-in rather than default;
- verify persisted worker binding before claiming a specific model actually ran.

The first repository swarm report is now historical evidence inside the Day Zero archive.

## Next actions

1. **P0 canonical dormant evidence bridge**: build derived manifest/hash-bound lexical + graph indexes over existing `State/dormant`, retrieve candidate IDs, dereference/verify exact authoritative records, surface provenance-bearing evidence into `structured_knowledge`, then prove D64 exact roundtrip.
2. **P1 non-neural production D64 driver**: canonical HEAD -> compile -> hard fresh/complete barrier -> scripted proposer -> typed delta -> validate/atomic commit -> successor -> recompile.
3. **P2 branch-backed training + exact resume**: make training episode transactions use canonical State branches and prove split/resume/anatomy mismatch refusal before substantive training.
4. Only after the permanent spine exists, port useful soul, multi-core refinement/consolidation, advisor/tool, diary, and situation mechanisms from historical evidence without restoring archived modules wholesale.
5. Benchmark realistic complete-field time/memory before model scale or learned semantic-compiler training.
6. Refresh both engineer ledger and `D:\ChatGPT_State` on every substantive turn.

## Fast orientation

- Day Zero map: `docs/DAY_ZERO.md`
- Architecture authority: `docs/SOURCE_OF_TRUTH.md`
- Canonical field: `runtime/field/schema.py`
- Typed delta: `runtime/field/delta.py`
- D64 compiler: `runtime/field/compiler_d64.py`
- Canonical branch: `runtime/field/state_branch.py`
- Runtime adapter: `runtime/axon_runtime/d64_adapter.py`
- Canonical training adapter: `training/canonical_d64.py`
- D64 model: `training/complete_field_64d.py`
- Trainer: `training/train_complete_field_64d.py`
- Dormant authority: `State/dormant/`
- Tracked legacy archive: `archive/day_zero_legacy_2026-08-20/`
- Local historical archive: `State/archive/day_zero_legacy_20260820/`
- Full active tests: `python -m pytest -q -p no:cacheprovider`

**ChatGPT / GPT-5.6 Sol / 2026-08-20**
