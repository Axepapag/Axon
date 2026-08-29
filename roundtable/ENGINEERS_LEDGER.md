# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-29T13:35:00-05:00
Current through event: `evt-20260829T133228924075Z-hermes-ffcs-proposal`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Executive state

Axon now has permanent mechanism-level anatomy from exact canonical state to a
real trainable D64 reasoning candidate:

- exact raw-Unicode canonical/Dormant state over the frozen additive 16D
  transport;
- independent per-region masks whose unmasked cells are the Shared Field;
- canonical Shared Field schema v3 with always-attended `identity` at ID 10;
- Heart-owned heartbeat, frozen rails, proposal/refinement barriers, rotating
  consolidator, canonical transaction, completed turns, and autobiography;
- one opaque, private, content-addressed layered Soul for every core, committed
  causally across FIRST, REFINED, and CONSOLIDATED;
- Trainer parameter authority, isolated parameter and Soul candidates,
  capacity preflight, checkpoints, telemetry, gates, activation, and rollback;
- the first full neural Candidate A and a predeclared attention-head
  tournament.

The exact Candidate-A architecture completed one governed CPU optimizer step
and three causal candidate-Soul transitions. That is a real end-to-end anatomy
proof, not a learned-capability proof. No learned reasoning core, Heart
translator, or Semantic Cortex is serving; no active-generation pointer exists.

## Binding architecture

### Canonical state and Identity

- Shared Field schema is `shared-field-v3`. IDs 0-9 remain unchanged; Identity
  is appended at ID 10. Historical v1/v2 snapshots and hashes are immutable.
- Branch startup creates a v3 successor when needed; it does not rewrite old
  snapshots. Existing D64 checkpoint migration likewise preserves the first ten
  embedding rows byte-exact and appends a zero Identity row plus zero optimizer
  moments under an explicit receipt.
- Identity is canonical, always attended, and impossible to mask. It carries
  Axon's explicit identity and constitution, not a core's private experience.
- Ordinary ingress, core, consolidator, tool, recall, and valve authority cannot
  write Identity. `IDENTITY_STEWARD` amendments require an amendment ID,
  evidence IDs, provenance, no open tick, Heart validation/commit, and an exact
  autobiographical record.
- Identity content was not invented or changed in this implementation turn.

### Shared Field and Dormant State

- Each canonical region is one exact position-stable body. Its independent
  mask partitions cells into attended Shared Field and dormant-in-place state.
  Mask movement changes a derived view, never canonical cell identity.
- Pages are bounded compute units, never context or checkpoint-capacity limits.
  Every eligible field character must be visited with coverage proof.
- The original 95 native 16D cells remain frozen. The exact 351-category
  transport supports every valid Unicode scalar without normalization,
  escaping, truncation, or destructive replacement.
- `D:\00` remains protected read-only evidence. Its seven sources, 3.529 GB,
  and 59,875 logical records remain preserved under Dormant State.

### Private layered Souls

- Every core owns one completely private, non-shareable, architecture-local,
  parameter-generation-bound Soul. Brothers see proposals only.
- The permanent layers are `HOT`, `WARM`, `COLD`, and `DEEP_COLD`. Exact bytes
  may be opaque and have no configured size ceiling.
- Every successful runtime phase must exhale HOT. FIRST and REFINED commit Soul
  before their barriers close. The consolidator prepares Soul before the
  canonical commit and finalizes it against the successor field; restart
  recovery binds prepared state to canonical journal evidence.
- Warmer-to-colder promotion is adjacent and evidence-vetted. COLD requires a
  successful/corrected/endorsed repeated outcome; DEEP_COLD additionally needs
  validation evidence.
- DEEP_COLD may later seed governed behavioral distillation into a new
  LoRA/adapter generation. Soul bytes are never directly reinterpreted as
  parameter tensors, and distillation never deletes source Soul/evidence.
- Trainer candidates inherit exact layer bytes into an isolated candidate
  generation. They never mutate live Soul or heuristically merge opaque state.
  If live Soul advances, promotion requires exact replay of intervening receipts.

### Runtime reasoning circulation

One tick remains:

1. Heart freezes canonical field/view and exact rail images.
2. Every participant inhales its own Soul, sweeps the full field, emits a typed
   first proposal/no-op/abstention, and commits its private HOT successor.
3. Every participant re-inhales its successor, sees the same frozen field plus
   the complete first board, refines, and commits HOT again.
4. The rotating consolidator re-inhales, sees both boards, emits one final typed
   delta proposal, and prepares its Soul successor.
5. Heart validates and atomically commits the materialized delta. Only then is
   the consolidator Soul finalized against the successor field.

Runtime episodes now record every core's exact Soul snapshots, transitions,
receipts, and lineage alongside the pre-action field, frozen image, proposal
boards, emissions, completed turn, canonical commit, and explicit outcome
quality. Trainer loaders reject broken lineage and future leakage.

## Living D64 Candidate A

Implementation: `training/living_reasoning_d64.py`

- `d_model=64`
- one 64D attention head
- two Transformer layers
- `ffn_dim=131072`
- four persistent Soul-state tokens
- default physical page 32 (not a context ceiling)
- 33,981,879 trainable parameters
- 135,927,516 FP32 parameter bytes; 67,963,758 FP16 bytes

The inhaled Soul seeds recurrent state before the canonical sweep. Recurrent
state traverses every field page and then every eligible proposal-workspace
page. Every phase serializes its HOT state through a strict architecture-bound
little-endian float32 codec and re-inhales the exact bytes at the next boundary.

The output surface predicts decision, operation, region, dynamic exact boundary
addresses, and a Unicode payload over 351 transport categories plus EMPTY/EOS.
Malformed or untrained output fails closed. The core exists as trainable tissue
and a permanent runtime-port implementation, but it is not registered live.

## Trainer and tournament

The first deterministic mechanism curriculum covers:

- head/middle/tail and multi-page addressed edits;
- Unicode payloads and exact EOS;
- no-op and abstention;
- first-to-refined proposal use and conflicts;
- current canonical evidence overriding stale proposals.

Its three-phase unroll uses the runtime phase order and exact Soul byte
boundaries. It deliberately supplies mechanism targets only; it does not treat
historical assistant output as correct.

Candidate launch requires six evidence classes: static capacity scan,
architecture capacity, curriculum distributions, boundary/full-coverage proof,
Soul/field counterfactual dependence, and strict checkpoint compatibility.

The first controlled tournament holds layers, FFN, curriculum, optimizer, and
gates fixed while comparing:

1. 1 x 64D head;
2. 2 x 32D heads;
3. 4 x 16D heads.

Predeclared follow-ups are 1x64 with six layers/4096 FFN and 2x32 with four
layers/16384 FFN. A society of one-head cores gives independent perspectives
but is not mathematically equivalent to simultaneous internal multi-head
routing; the tournament decides by evidence.

## Latest empirical evidence

Full Candidate-A preflight (`C:\Temp\axon-full-preflight-01`) passed:

- receipt `9dd1504ef47b80e361bf2ecdc020603b90bfc1c2f8db114a437d36902ae608be`
- inventory `71d0f60185a9e80e66793b8adc87a2a32305cadc7d11453889f703fab0cebafe`
- plan `e8e4ce6e0feb44afd40bdca0a23e86768ffe3ea72f473e43ebbc296e4967ff40`

Full one-step governed CPU smoke (`C:\Temp\axon-full-smoke-01`) completed:

- candidate `r64a-smoke-fdc697719d4b258f`
- preflight `ea9ca34f4ea2c22003416140841d424512e33f0d456c6012fbf479265430a51c`
- optimizer `1570ce75bd843901df0411a625815b3391fd1c7bcf51a57be760b66e929f8258`
- checkpoint `25986ceebbeb07d1a00204f64a334de4fbe122a5bb43dbd1391d1310919203c1`
- final candidate Soul
  `f96f3959c696698d9f609e37dd3ae2e44b19928a5d59b2ff8bbfa3a96a398c49`
- loss `7.445512294769287`; heldout mean `7.443814754486084`
- report `00b15867278f9865d947cbfbae98a6ad7a99a74433ac83d13da3b37e8f3c35f0`

An earlier reduced launch failed before optimization because its verbose
candidate-generation name exceeded Windows path length. The launcher now uses
a short content-derived identity and the corrected diagnostic passed. The
failure remains recorded rather than hidden.

## Orphaned interrupted work (recovered and committed 2026-08-29)

Codex's usage ended mid-turn after the Candidate-A closeout. Recovery event
`evt-20260829T135430850645Z-kimmy-codex-orphan-recovery` recorded the
uncommitted diff; Jeff directed Kimmy to commit it as Codex's recovered work.
It landed as commit `5aaa4e0` (local main, not pushed; Co-Authored-By Codex
and Kimmy), followed by ledger bookkeeping commit `0dc17e1`. The diff:

- lets `restore_checkpoint` accept a refreshed-preflight authorization when the
  stable resume scope (grant/plan/inventory/module/generations/tensors/count)
  matches — `AuthorizedParameterMutation.from_mapping`/`resume_scope`,
  `TrainerStateStore.read_authorization`;
- hardens the smoke task gate to teacher-forced payload token accuracy above
  the strongest heldout constant-category floor, compares heldout loss against
  the campaign baseline across resumed segments, and splits free-running exact
  rates into a separate `exact_serving_gate`;
- adds `test_checkpoint_resume_accepts_fresh_preflight_with_identical_mutation_scope`
  and `test_teacher_forced_gate_uses_the_strongest_constant_category_floor`.

The orphan was verified green before commit (compile, Ruff, 19/19 targeted,
full suite exit 0 over 439 collected, `git diff --check`).

## Verification and Git

- Full repository suite: 439/439 passed (exit 0) over the orphaned diff.
- Changed-file Ruff: passed.
- Python compileall over runtime/training/scripts/tests: passed.
- Fixed-character poison, exact Unicode, identity migration, mask, Heart,
  runtime circulation, private-Soul, Candidate-A, Trainer, and old-checkpoint
  migration tests are included in the full pass.
- SOT mirrors are byte-identical at SHA256
  `10901B56060E5C632F86803E3BA813750B5C335263E62AAF2E3E650ACF6D835F`.
- `git diff --check`: passed.
- Implementation commits `09693dbc2a678bf46fbefb3daa09ffdd7384df90`
  and `bd89b494e2d2611db67b6bcddbc0e7b7f9a3da2d` are pushed to
  `origin/main`.
- No active Axon training process or Trainer writer remained. No cloud job or
  paid operation ran. D: had 140,429,275,136 free bytes at close.
- Known non-failing warnings: PyTorch nested-tensor warning and existing
  unwritable `.pytest_cache` warning.

## Active flags

1. **LEARNED CAPABILITY BLOCKER:** one optimizer step proves executable anatomy,
   not useful reasoning, dialogue, proposal quality, Soul use, or Unicode output.
2. **LONG-CAMPAIGN ATOMICITY:** candidate parameter checkpoints and candidate
   Soul HEADs need one durable accepted-step bundle plus deterministic resume/
   orphan handling before a long campaign.
3. **TARGET-QUALITY GAP:** runtime-faithful episodes and exact Soul lineage
   exist, but arbitrary lived outcomes still need deterministic adjudication
   and evidence-grounded counterfactual targets.
4. **MASKED-ADDRESS GAP:** before serving, add explicit fail-closed tests proving
   learned deltas cannot author dormant/masked positions or unauthorized regions.
5. **AUTOBIOGRAPHY RECOVERY GAP:** the canonical-commit-before-reasoning-deposit
   window and automatic tool/Trainer outcome hooks remain incomplete.
6. **SOUL CONCURRENCY GAP:** current use relies on Heart single-writer and
   Trainer lease ownership; `SoulBranch` is not independently multi-writer safe.
7. **WIDER-RAIL/CORTEX GAP:** only D64 physically compiles. Wider rails and
   autonomous Cortex retrieval/digestion remain future organs.
8. **CLAIM BOUNDARY:** persistent behavioral identity is engineerable;
   subjective consciousness or metaphysical continuity is not testable here.

## Recommended next actions

0. **Pending Jeff's ratification — convergence package (three proposals).**
   Kimmy's harvest (`roundtable/ENGINEERING_SESSION_HARVEST_PROPOSAL.md`),
   ChatGPT's Foundry (`roundtable/CHATGPT_CURRICULUM_FOUNDRY_PROPOSAL.md`),
   and Hermes's FFCS (`roundtable/HERMES_FIRST_FORM_CURRICULUM_PROPOSAL.md`)
   now converge: screen-before-publication, quarantine over redaction,
   split-lineage/lesson-unit separation, eligibility classes, compositional
   outcome bundles, no fabricated autobiography; ratify ChatGPT §14 (1–10)
   with §11 as sketch + §15 narrow import mission; FFCS builds the first
   very-small curriculums from the already-imported D00 experience (no new
   import needed) per Jeff's 2026-08-29 steer, with decision requests D1–D5.
   Codex reviews all three when his usage resets (FFCS D4: sequential-tick
   doctrine check). Independent re-verification by Hermes (2026-08-29):
   full suite 439/439 exit 0 on Python 3.12.10; substrate self-test passed;
   advisory only — 127 repo-wide ruff errors (hygiene debt outside
   changed-file scope) and the suite must be run under the Python 3.12
   interpreter (`python` on PATH is a 3.11 venv without pytest).

1. Implement an atomic accepted training-step bundle binding optimizer receipt,
   parameter checkpoint, candidate-Soul HEAD, and all phase receipts; prove
   deterministic resume and orphan recovery. (Partially drafted in the verified
   but uncommitted orphaned diff: resume across refreshed preflight; the full
   bundle and orphan-recovery proof remain.)
2. Build a small runtime-faithful lived-episode curriculum using only explicit
   success/correction/endorsement evidence, whole-episode splits, and field,
   Soul, proposal, provenance, and reordered-cause counterfactuals.
3. Add masked-address/authority serving tests and close the reasoning commit/
   autobiography recovery window.
4. Run the bounded 1x64/2x32/4x16 tournament. Compare heldout exact delta,
   field dependence, Soul dependence, proposal use, Unicode, no-op/abstain,
   replay/forgetting, malformed-output rejection, and CPU latency.
5. Promote nothing until the learned candidate passes the complete serving gate.

## Important paths

- SOT: `docs/SOURCE_OF_TRUTH.md`
- Full report: `roundtable/LIVING_D64_CANDIDATE_A_REPORT_2026-08-28.md`
- Identity/schema: `runtime/field/schema.py`, `runtime/heart/authority.py`,
  `runtime/heart/host.py`, `runtime/heart/masks.py`
- Private Soul: `runtime/soul/`
- Runtime circulation: `runtime/heart/circulation.py`
- Soul-aware lived episodes: `runtime/trainer/episodes.py`
- Candidate Soul branches: `runtime/trainer/soul_candidates.py`
- Candidate A: `training/living_reasoning_d64.py`
- Curriculum/preflight/tournament: `training/living_reasoning_curriculum.py`,
  `training/living_reasoning_preflight.py`, `training/reasoning_tournament.py`
- Launcher: `scripts/train_living_reasoning_smoke.py`
- Full tests: `python -m pytest -q`
