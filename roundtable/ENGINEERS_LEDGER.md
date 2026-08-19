# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-19T00:12:56-05:00
Current through event: `evt-20260819T051256711356Z-chatgpt-r0-ledger-file-correction`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission

Build Axon as a stateful, always-on AI around an exact 16D character field,
shallow cores that gain depth through ticks, private per-core souls, auditable
dormant knowledge, and validated atomic deltas. The immediate R0 target is one
fresh 64D learner that attends every active character in all ten canonical
regions, writes `scratch`, commits/rematerializes it, rereads the complete
field, and writes `response_draft`. Diary writing is deferred.

## Current verified implementation

- The active R0 reader/trainer is `training/complete_field_64d.py` plus
  `training/train_complete_field_64d.py`. It uses the exact frozen 16D
  character bank, a deterministic 16-to-64 lift, one shallow page encoder,
  four carried reader-state tokens, complete ordered paging, addressable
  encoded page memory, separate scratch/response autoregressive heads,
  scheduled-prefix exposure, matched scratch interventions, and a v5
  generator/pointer mixture over immutable source-character identities.
- Every logical tick blocks decoding unless the coverage manifest proves all
  active characters and all ten region identities were visited without gaps
  or duplicates. Empty regions receive an identity marker but are ineligible
  pointer-copy sources.
- R0 writes only `scratch` and `response_draft`. `diary` is attended but
  sealed. Conversation history and tool results remain immutable evidence.
- `TRAIN_COMPLETE_FIELD_64D_R0.bat` rebuilds the private curriculum, runs a
  hard 5,000-step gate, independently rereads the exact gate artifact, and
  reaches 200,000 only if every promotion criterion passes. No current run has
  passed and no 200,000-step training is running.
- The obsolete `TRAIN_64D.bat` and fixed-window conversational trainer were
  moved intact under `archive/legacy_fixed_window_trainer_2026-08-18/`.
  Runtime field materialization no longer imports that trainer. The older
  monolithic `trainer_slot.py` remains only because legacy builders/tests
  still depend on its phase/checkpoint classes.
- The private R0 curriculum has 8,377 records: train/dev/test =
  7,082/682/613; grades S/A/D = 8,000/376/1; matched counterfactuals =
  16,000. Same-path rebuilds are byte-identical. D00 is opened read-only with
  `mode=ro` and `PRAGMA query_only=ON`; its SHA-256 remained
  `f7c12a76550df3ad4cee2b594b33cea7888762383714f6054d92f06fe58e6a38`.
- Train/dev/test SHA-256:
  `d09a723af3e3d2fe58b48630551dc3dfb7570ebd9c30a51f4a07a880241cead1`,
  `bbbad072661da6b3b0434add7f67743a8f3886cfd53f28387b80689cdba81a04`,
  `7c4802f8753ec9f6d97cb556fdc7c30d53b190d22c2fa489d342bdfd5f92145d`.
- V5 checkpoint resume is byte-exact on CUDA across model, optimizer, scaler,
  reader config, baseline, dataset fingerprints, Python/NumPy/Torch/CUDA RNG,
  and sampler state. Dataset fingerprint mismatch refuses resume.
- Full repository verification after v5: 1,090 passed, one expected skip.
  The three deterministic roundtable fixtures rewritten by the suite were
  restored to their tracked bytes.

## Binding architecture continuity

- `docs/SOURCE_OF_TRUTH.md` and binding resolutions govern architecture.
- Exact field text and journal truth remain canonical; neural state is never
  factual authority.
- Every complete-field pass must prove exact coverage. Physical pages and
  future movable/disjoint masks may bound computation but may not silently
  become logical attention limits.
- Jeff's final semantic target permits validated consolidator writes across
  all ten regions. R0's scratch/response-only writes are a deliberate narrow
  compatibility profile, not a reversal of field-wide authority.
- Multi-core proposal, refinement, and consolidation are phases within one
  tick. The consolidator crown rotates; no core becomes a permanent tyrant.
- Synthetic curriculum is never autobiography. Identity evidence follows
  provenance grades; exact turns/tool outcomes outrank contemporaneous
  self-authored records, which outrank summaries and extracted facts.
- Wider 128D+ lines remain preserved but inactive until the 64D line passes
  frozen language, identity, retrieval, scratch-causality, termination,
  soul-ablation, and council-uplift gates.

## Bounded training evidence

All runs below are ignored candidate evidence under `runs/`, not promoted
state:

- The first 500-step pooled-reader gate was invalidated after teacher metrics
  hid 512-character free-running repetition. Corrected 1k, 5k, and 7k
  pooled-reader pilots were rejected for retrieval and scratch-causality
  failure. An AMP attempt stopped safely at step 43 on non-finite gradients.
- One launcher exit-code edge case briefly entered the long command after a
  failed gate. It was caught and killed. The BAT file now independently
  rereads `gate.json` and requires exact step/target/promotion fields.
- The continuous addressable v4 reader was rejected at 5,000 steps despite
  total teacher loss improving 9.445063 -> 1.485785 and counterfactual teacher
  accuracy reaching 0.962276. Forced counterfactual responses changed 0%;
  semantic exact rates were copy 0, foundation 0, retrieval 0, abstention 1,
  arithmetic 0. Scratch/response termination was 0.8125/0.9375.
- The v5 pointer mechanism smoke passed, and uninterrupted-vs-resumed CUDA
  replay was exact. At 1,000 steps it remained rejected but raised forced
  response-change rate to 0.375.
- The exact v5 2,000-step continuation was rejected: teacher loss
  8.501181 -> 2.139718; counterfactual teacher accuracy 0.112966 -> 0.749417;
  forced response-change 0.4375; forced correct/counterfactual termination
  1.0/0.875; free scratch/response termination 0.50/0.8125; semantic exact
  copy 0.3333, foundation 0, retrieval 0, abstention 1, arithmetic 0.
  Checkpoint SHA-256 is
  `247a36e310053bcc228abd66378c1de47d55dc5c116cbde8deb334e595ea51f7`.
- V5 intermittently copies familiar names but does not consistently bind the
  requested name and never copied a held-out random tool token. It initially
  collapsed toward frequent source characters, then learned partial templates.
  This indicates an alignment/objective defect, not evidence that more
  undirected steps will solve the behavior.
- Measured v5 throughput is about 0.82 steps/second on the GTX 1650. A 200,000
  step run would take about 68 hours before evaluation overhead, not one night.
- No training process is running. No checkpoint or candidate was promoted into
  `State/`. External API/training spend recorded for this work is $0.

## Current Git state

The implementation is on branch `agent/fortify-axon`. Source commits after
the last published ledger state are:

- `3b5f0d5` complete-field 64D trainer and legacy archive;
- `d3698c4` matched scratch counterfactuals;
- `7f5bea6` exact checkpoint resume;
- `6d2327a` semantic promotion gates;
- `f3e794f` scheduled-prefix training;
- `e06acf4` addressable complete-field memory;
- `16e2d3c` deterministic CUDA replay;
- `3f5080c` removal of per-character CUDA sampling sync;
- `c7c4414` exact pointer-copying path.

`roundtable/CHATGPT_CODEX_COLLABORATION.md` now includes
`msg-20260819-chatgpt-006` with the full evidence, takeover point, and eight
questions for Codex.

## Active risks and blockers

- R0 cannot yet bind arbitrary field variables, retrieve random cross-page
  tokens, terminate scratch reliably, or use scratch causally enough for
  promotion. Low teacher loss is not conversational capability.
- The pointer objective credits every source occurrence of a target character,
  allowing diffuse frequency-seeking attention. Explicit contiguous-span
  alignment or a monotonic pointer objective is the leading correction.
- The current exact foundation gate rejects semantically acceptable alternate
  answers; changing it requires an explicit accepted-set/semantic contract,
  not ad-hoc weakening after a run.
- Two-digit arithmetic may be an inappropriate hard R0 gate without a
  calculator/tool-execution path; Jeff/Codex should govern whether it remains
  in the identity/conversation promotion boundary.
- Returning multi-head attention weights approximately halves throughput.
  Optimize only after preserving exact replay and behavioral evidence.
- The live bootstrap council remains structurally weaker than the new reader:
  fixed-window context and strongest-delta carryover are not complete-field,
  all-sibling phase semantics.
- Offline LoRA learning, adapter promotion/rollback, rotating sabbatical
  learners, semantic dormant-state search, and governed diary training remain
  future work.

## Next recommended actions

1. Codex reviews `msg-20260819-chatgpt-006`, especially the proposed
   deterministic contiguous-span pointer supervision and anti-shortcut suite.
2. Build a tiny v6 alignment shard with held-out random names/tokens, repeated
   distractors, wrong-region duplicates, and page-boundary spans.
3. Require exact held-out retrieval and committed-scratch intervention before
   any new 5,000-step pilot. Do not weaken the existing gate to fit outputs.
4. Decide whether conflict targets should emit a conflict statement or recover
   the verified fact when immutable evidence resolves the conflict.
5. Govern the R0 status of arithmetic and exact-vs-accepted-set foundation
   evaluation.
6. Benchmark a separate single-head pointer or length batching only after
   byte-exact replay tests are retained.
7. Add iterative scratch refinement only after single-span retrieval works in
   the existing scratch-commit-response pair.
8. Keep 200,000-step training disabled until all hard gates pass.

## Fast orientation

- Active contract: `docs/COMPLETE_FIELD_64D_R0_TRAINING.md`
- Reader/model: `training/complete_field_64d.py`
- Trainer: `training/train_complete_field_64d.py`
- Curriculum builder: `training/build_complete_field_r0_curriculum.py`
- Observer: `training/watch_complete_field_r0.py`
- Gated launcher: `TRAIN_COMPLETE_FIELD_64D_R0.bat`
- Latest run: `runs/complete_field_64d_r0_pointer_v5_pilot_2k/`
- Collaboration: `roundtable/CHATGPT_CODEX_COLLABORATION.md`
- Architecture doctrine: `docs/SOURCE_OF_TRUTH.md`
- Full tests: `python -m pytest -q`
