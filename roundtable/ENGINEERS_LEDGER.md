# Axon Engineer's Ledger ? Rolling Summary

Updated: 2026-08-20T10:30:44.6055387-05:00
Current through event: `evt-20260820T164447173835Z-chatgpt-canonical-anatomy-publication`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission

Build Axon as a stateful, always-on AI around an exact 16D character field,
shallow cores that gain depth through ticks, private per-core souls, auditable
dormant knowledge, and validated atomic deltas. R0 remains one fresh 64D learner
that attends every active character in all ten canonical regions, writes
`scratch`, commits/rematerializes it, rereads the complete field, and writes
`response_draft`. Diary writing is deferred.

The immediate engineering target is narrower: prove exact arbitrary binding on
held-out evidence with the V6 exact-position pointer before scaling training.

## Current verified implementation

- The active R0 reader/trainer is `training/complete_field_64d.py` plus
  `training/train_complete_field_64d.py`.
- The frozen 16D character bank is lifted into one 64D core. Every logical read
  pages completely across all ten regions and blocks decoding unless coverage
  proves every active character and every region identity was visited exactly.
- Every encoded page token retains immutable source character, region identity,
  and exact region-local character position. Empty-region markers remain context
  but are not copy sources.
- V6 separates semantic multi-head cross-attention from a dedicated one-head
  exact-position pointer. Copy supervision identifies one exact source
  `(region, character_position)` occurrence plus a copy/generate target; EOS is
  generated rather than copied.
- R0 still performs two complete reads: first decode/commit `scratch`, then
  rematerialize and reread the whole field before decoding `response_draft`.
  `diary` is attended but sealed. Conversation history and tool results remain
  immutable evidence.
- Clean committed scratch may receive a supervised response-copy prior, but
  immutable `user_input`/`tool_results` wins when scratch is empty or conflicts.
  The old unconditional response-change counterfactual gate is diagnostic only.
- Checkpoint schema V6 binds exact train/eval fingerprints and preserves model,
  optimizer, scaler, Python/NumPy/Torch/CUDA RNG, and sampler state. Stale active
  checkpoints are archived rather than deleted.

## Curriculum and anti-shortcut evidence

- Versioned full V6 private curriculum:
  `State/private_curriculum/complete_field_r0_v6/`.
- Total records: 8,633; train/dev/test = 7,252/680/701.
- Grades S/A/D = 8,256/376/1.
- V6 alignment fixtures: 256; aligned segments: 18,624; aligned copied
  characters: 210,412.
- Full V6 train/dev/test SHA-256:
  `45dac051fa8e6f7aac12618d5d365713df0726a218361eb8acf565285dbb6f0d`,
  `d2f66568e10e52a1a26aa1bbd1a746c2c3e888699d639b66be9191c6d0f78c42`,
  `4c299d6d7919e6c43a99a35ac159ed46781887bf85f1cd78ff963f1b31956eaa`.
- Isolated alignment shard:
  `State/private_curriculum/complete_field_r0_v6_alignment/` with 256 records,
  train/dev/test = 215/18/23. It has exactly 64 each first/middle/page-boundary/
  last layouts, unseen mixed-case alphanumeric tokens length 4-32, same-region
  duplicates, wrong-region duplicates, and nearby decoys.
- Alignment shard train/dev/test SHA-256:
  `1704117adb841965a4549427bd5a0308f44fb1001b31d33a66e582d02ff33bb8`,
  `8ad5fdca48ad82a0dfe64b2a9653faf939ef1ddb0b814051c441c3d8cd44500f`,
  `fb7efec77bf330fd7cb528189bd65a4b2fa33da66f831898a743e018965b71e2`.
- D00 was opened read-only and remained byte-identical at SHA-256
  `f7c12a76550df3ad4cee2b594b33cea7888762383714f6054d92f06fe58e6a38`.
- Exact-field contradiction and split-isolation checks remain hard failures.

## Verification

- Focused complete-field tests: 15 passed.
- Full repository: 1,095 passed, one expected skip. Deterministic echo-round
  fixtures rewritten by the suite were restored to tracked bytes.
- One-step CPU mechanism run: finite optimization, complete coverage, correctly
  rejected as untrained.
- V6 CUDA resume proof at
  `runs/complete_field_64d_r0_v6_resume_verify/verification.json`: uninterrupted
  four steps exactly matched two steps + restore + two steps for loss sequence,
  model tensors, optimizer state, and RNG/sampler state.

## Latest bounded CUDA evidence

Run: `runs/complete_field_64d_r0_v6_alignment_smoke/` on the GTX 1650.

The 100-step smoke was healthy mechanically and correctly rejected:

- teacher total loss: 9.20048 -> 7.46194;
- complete coverage held; diary writes remained zero;
- scratch termination: 1.0;
- response termination: 0.94444;
- held-out exact scratch: 0.0;
- held-out exact response: 0.0;
- base exact-position accuracy: 0.03765;
- counterfactual exact-position accuracy: about 0.00904;
- forced correct/counterfactual exact response: 0.0/0.0;
- evidence-authority preservation: 0.0;
- `v6_alignment_gate_passed=false`;
- `promotion_allowed=false`.

This is useful auxiliary-objective movement, not arbitrary binding capability.
No 1,000-step comparison, 5,000-step run, 200,000-step run, or State promotion
was launched.

## Active launcher and gates

`TRAIN_COMPLETE_FIELD_64D_R0.bat` now only:

1. rebuilds the versioned V6 curriculum;
2. rebuilds the isolated alignment shard;
3. runs focused CPU tests;
4. runs a bounded 100-step CUDA alignment smoke;
5. exits nonzero unless the smoke itself passes every hard V6 gate.

There is no automatic 1k, 5k, or 200k continuation in the active launcher.
Before any bounded 1k V6-vs-V5 comparison, V6 must reach 100% held-out exact
scratch, response, source position, copy/generate gate, conflict recovery, and
termination.

## Current Git / collaboration state

- Working branch: `agent/fortify-axon`, tracking private `origin/main`.
- Pre-V6 HEAD and origin/main were both
  `7d4b158f8e779303313dfd3be3e4c8038bf40a6f`.
- `roundtable/CHATGPT_CODEX_COLLABORATION.md` includes
  `msg-20260819-chatgpt-007` with V6 implementation, hashes, rejected-smoke
  evidence, exact resume proof, and the no-scale boundary.
- Git under the connector requires a per-command `safe.directory=D:/Axon`
  because the bridge account differs from repository ownership. No global Git
  exception has been added.

## Active risks and blockers

- V6 now makes exact source position observable, but the 100-step candidate has
  not learned held-out binding. Position accuracy is above random movement but
  remains far below the 100% gate.
- Copy/generate gate accuracy remains near floor on the small smoke. The next
  work should diagnose objective balance and fixture difficulty rather than add
  undirected steps.
- Exact scratch and response remain zero on the held-out V6 shard. Do not infer
  capability from falling teacher loss.
- The live bootstrap council at `127.0.0.1:8788` remains unreachable and was not
  restarted; service recovery is separate from this R0 training mission.
- Offline LoRA learning, adapter promotion/rollback, rotating sabbatical
  learners, semantic dormant-state search, governed diary training, and a third
  refinement tick remain future work.

## Next recommended actions

1. Diagnose why exact-position accuracy rises slowly and copy-gate accuracy stays
   near floor; inspect loss weighting, source-position separability, and whether
   the shard should begin with a simpler unambiguous stage before decoys.
2. Keep the exact held-out anti-shortcut shard frozen for evaluation; add any
   easier teaching shard separately so the gate cannot be trained on directly.
3. Repeat only bounded mechanism/smoke runs until held-out V6 behavior reaches
   100% exact scratch/response/position/gate/conflict/termination.
4. Only then authorize a bounded 1,000-step V6-vs-V5 comparison.
5. Keep 5,000-step and 200,000-step training disabled and add no third tick.

## Fast orientation

- Active contract: `docs/COMPLETE_FIELD_64D_R0_TRAINING.md`
- Reader/model: `training/complete_field_64d.py`
- Trainer: `training/train_complete_field_64d.py`
- Full curriculum builder: `training/build_complete_field_r0_curriculum.py`
- Alignment shard builder: `training/build_r0_v6_alignment_shard.py`
- CUDA resume verifier: `training/verify_complete_field_v6_resume.py`
- Observer: `training/watch_complete_field_r0.py`
- Bounded launcher: `TRAIN_COMPLETE_FIELD_64D_R0.bat`
- Latest V6 smoke: `runs/complete_field_64d_r0_v6_alignment_smoke/`
- V6 resume proof: `runs/complete_field_64d_r0_v6_resume_verify/verification.json`
- Prior audit: `docs/roundtable/AUDIT_r0-takeover-and-v6-shot-2026-08-19.md`
- Collaboration: `roundtable/CHATGPT_CODEX_COLLABORATION.md`
- Architecture doctrine: `docs/SOURCE_OF_TRUTH.md`
- Full tests: `python -m pytest -q`

## Proposed field-compiler architecture under discussion

- Non-binding design proposal: keep the exact 16D character field canonical while a heartbeat Field Compiler builds reversible word/phrase/sentence/paragraph tiles with exact source spans and hashes for heterogeneous core widths.
- Higher-D semantic views may be dense and lossy as reasoning aids, but every tile must retain exact source access; arbitrary dense core output is not canonical text.
- Core writes should return typed span/tile edits or exact-character generation/copy; the compiler validates and expands them back into exact 16D canonical deltas before sibling visibility or commit.
- Dormant retrieval may use lexical, structural, graph, and learned semantic-vector indexes internally, but retrieved exact text plus provenance must be surfaced into the active field before it becomes reasoning evidence.
- This proposal does not change the current V6 R0 gate or authorize additional training.


## External ChatGPT continuity handoff

- `D:\ChatGPT_State` was refreshed at 2026-08-20 09:56 Central so a fresh ChatGPT session resumes from the current V6 state and the non-binding Field Compiler/reversible multiresolution field proposal rather than the obsolete June `axon7` bootstrap.
- No Axon runtime, model, State, curriculum, checkpoint, service, or training process changed during that personal-state maintenance.
- Fresh ChatGPT sessions should still re-read live Axon authority and the canonical ledger before project action; personal continuity does not override repository evidence.


## Fresh-session continuity recovery — 2026-08-20 10:30 Central

- A fresh ChatGPT session successfully re-entered through `D:\ChatGPT_State\START_HERE.md`, then re-read live Axon authority and the canonical ledger before acting.
- Pre-turn Git was clean; local `HEAD` and `origin/main` both resolved to `72d6d58` (`Record ChatGPT fresh-session handoff`). That commit changed only the two engineer-ledger files.
- ChatGPT's personal state was refreshed to record `72d6d58` separately from the earlier non-binding Field Compiler proposal commit `6c8a3c9`.
- No architecture, runtime, model, State, curriculum, checkpoint, service, or training process changed. The Field Compiler remains non-binding, and the V6 exact held-out no-scale gate remains in force.


## Canonical anatomy ruling and dormant inspection — 2026-08-20

- Jeffrey explicitly ruled that future training/runtime core paths must use Axon's real canonical anatomy. A smoke may be small in content/compute, but it may not use a disposable truncated/substitute field interface that will later be replaced.
- `State/dormant` is a real ~1.02 GiB recovered memory body: 7 files / 1,094,882,987 bytes, 427,001 containers and 351,978 readable semantic edges. Its manifest records no vectors; exact/readable text, normalized text, kind, confidence, provenance and readable relations are present.
- The newer runtime dormant store is disconnected: every substantive table in `State/axon_runtime/dormant.sqlite3` was verified empty. Runtime retrieval currently scores lexical word overlap over that empty triple table, so it cannot surface the recovered corpus.
- The newer runtime bootstrap explicitly installs `ActiveFieldProjector`; the default and CPU-smoke global budget is 4,096 characters. It preserves the full source snapshot but gives cores a bounded derivative, which conflicts with Jeffrey's new no-substitute-core-input boundary if retained as the core-facing path.
- V6 `CompleteField64D` already does complete coverage-proven paging across the active field and refuses silent omission. Its scaling bottleneck is retained per-character encoded memory/decoder attention, not logical truncation.
- Legacy `runtime/tick_loop.py` also clips fixed char-slot inputs and uses a capped 50k KG cache; do not carry that anatomy forward as the production path.
- Proposed next architecture: one authority-free Field Compiler heartbeat maintains source-field-ID-bound D-model rails with 100% active-field coverage and exact reversible provenance; tick start requires a fresh rail. Deterministic packing/cartography stays algorithmic; learned semantic retrieval/ranking is derived/rebuildable and must dereference exact dormant text/provenance before it becomes attention evidence. Consolidator remains reasoning authority; runtime validator/atomic transaction remains commit boundary.
- Important unresolved mechanism: literal 16D-cell concatenation into a wider row is lossless storage, but a normal Transformer treats the row as one token. Exact packed lanes therefore need explicit lane/addressability/decompiler semantics; do not assume packing alone gives independent character attention.
- Detailed Field Compiler mechanics remain a proposal until formally reconciled with `docs/SOURCE_OF_TRUTH.md`; Jeffrey's no-fake-anatomy/no-truncated-core-input instruction is the new explicit convener boundary.
- Publication: the inspection/ruling ledger update was committed and pushed to private `main` as `99175d7` (`Record canonical anatomy dormant inspection`), touching only the two engineer-ledger files. A follow-up canonical event records that publication because the inspection event had already been appended before the Git step completed.
