# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-31T04:55:00-05:00
Current through event: `evt-20260831T045500000000Z-kimmy-c1-curriculum-implementation`
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

Canonical Identity v1 is applied; FFCS A/B/C/D/E/F are immutable and bound to
it; and commit `617989b` wires all standard and sequential manifests into one
content-addressed campaign with honest count-weighted tournament metrics.
FFCS-E stays grouped as real three-tick chains and family round-robin reaches
mechanism/A/B/C/D/F/E within seven global steps.

The governed D64 head-geometry tournament
`ac9eceaaf0e67d483b288140d4db328c7f7b2156468ef5e25a1eaa9f4b791a39`
now has exactly 16 accepted CUDA optimizer+private-Soul steps for each 1x64,
2x32, and 4x16 candidate. A sequential-lineage defect failed closed before an
invalid bundle was accepted; the harness now supplies all nine Soul transitions
and a regression test proves the exact lineage. Final token accuracies are only
1.24-1.28% against a 10.88% constant floor, and both free-running exact rates
are zero. One standard heldout case remains deferred, so the formal metric
surface is incomplete. No winner, promotion, activation, or learned-capability
claim exists.

Further optimizer work WAS technically blocked by a design defect, not project
doctrine: the immutable `max_steps=16` authorization was also part of candidate
identity, so increasing it created a fresh generation from the governed base.
**That blocker is now CLOSED** (commit `962c8f1`, joint Hermes/Codex): the
renewable resource-tranche law is doctrine in both SOT mirrors and code.
`ResourceTranche`/`TrancheContinuation` records are durable beneath
`State/training/trainer/tranches/`; `ParameterMutationPlanV2` has resource-free
identity; sessions enforce exact-parent-restore and pause at the tranche bound;
the harness and v2 launcher require tranches for v2 training; the headless
TrainerOrgan surface, thin CLI, and TRAIN_AXON.bat exist. REAL-TISSUE PROOF:
lineage `r64t-21a315397f232525` (candidate-a-1x64) executed accepted global
step 17 from its exact step-16 parent bundle under a 1-step tranche — same
plan_id, same generation, `paused_for_next_tranche=true`. Lineages now continue
additively; no candidate is ever restarted for wanting more compute.

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
quality. Episode v3 also freezes the exact attended intervals/rail; Trainer
recompiles and verifies that view and rejects masked-address output. An
immutable precommit spool recovers canonical-commit-before-Soul/deposit crashes
idempotently. Trainer loaders reject broken lineage and future leakage.

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

Composed campaign curriculum:
`b39bc91b69e5d8ce3395ff826e15f3413da7c9f48376f831f037743205a4a7cd`.
Closing observation:
`068867c5f1e4b693f00625d30c3a718feea5a40030a4ff41758e54d865f012ab`.

- 1x64 step 16: loss `6.07003877473914`, token accuracy
  `0.0128467153284672`;
- 2x32 step 16: loss `5.994862172914588`, token accuracy
  `0.012408759124087591`;
- 4x16 step 16: loss `6.006355773193193`, token accuracy
  `0.0125547445255474`;
- common constant floor `0.10875912408759124`, coverage `1.0`, typed exact
  `0.0`, payload exact `0.0`;
- 40 standard plus all six sequential heldout cases evaluated; one standard
  heldout case deferred; regression complete; nine formal metrics missing;
- all gates false; no winner or promotion.

Detailed recovery report:
`roundtable/CODEX_HERMES_CAMPAIGN_RECOVERY_2026-08-30.md`.

The earlier isolated Candidate-A diagnostic remains preserved:

Candidate `r64a-smoke-da5a45a1e78ef20f` is an isolated D64 mechanism campaign
with a declared 10,000-step maximum. Only 15 bounded steps have executed. Its
accepted reports are under
`State/training/reasoning/r64a-smoke-da5a45a1e78ef20f/`.

- campaign-baseline heldout loss: `7.409373760223389`;
- step-15 heldout loss: `4.788444995880127`;
- teacher-forced transport-token accuracy: `0.06896551724137931`;
- strongest heldout constant-category floor: `0.13793103448275862`;
- free-running typed-delta exact rate: `0.0`;
- free-running complete-Unicode-payload exact rate: `0.0`;
- field/proposal/Soul counterfactual deltas: all nonzero;
- accepted step-15 bundle:
  `94672693c7e1cbad1f27002ae2f5e651ca202e83387a0a02959d11d7f1600948`.

The task gate is false because token accuracy is below the actual strongest
constant output; the exact-output gate is also false. The 10,000-step number is
an authorization envelope, not a launch command. No overnight run is active or
authorized. At the measured ~21.5 seconds/full step, a blind 10,000-step run
would take about 59.7 hours; checkpoint cadence must also prevent terabytes of
redundant snapshots.

Kimmy recovered Codex's interrupted resume/gate diff, verified it, and committed
it as `5aaa4e0` with both identity trailers. Codex independently inspected that
diff and reran the complete suite successfully. The code now supports exact-
scope resume across refreshed preflight evidence and uses the honest strongest-
constant heldout gate.

## Curriculum convergence

Kimmy's engineering-session harvest, ChatGPT's Curriculum Foundry, Hermes's
FFCS, and Codex's review are preserved in `roundtable/`. The team converges on
screen-before-publication, quarantine over redaction, exact per-event evidence
plus derived manifests, whole-lineage splits, scoped/compositional outcomes,
observed-only hidden reasoning, no fabricated core autobiography, controlled
counterfactuals, and small competency gates.

Codex approves sequential-tick FFCS only through real canonical successors,
fresh complete-field compilation, tick-scoped board reset, exact same-core Soul
serialization/re-inhale with runtime gradient breaks, and no future leakage.
Codex rejects a permanent `<=500` doctrine; the 460-case FFCS v1 is a declared,
revisable campaign budget and never a context/content limit. Jeff ratified and
applied canonical Identity v1 on 2026-08-29.

FFCS v1 is now classified more precisely as a useful anatomy, transport, Soul,
society, and regression schoolhouse, not a sufficient conversational education.
Its 371 unique train items are scheduled across seven family lanes. Sixteen
optimizer steps expose only two or three items per lane; one balanced scheduler
super-cycle requires 672 steps, and even one pass is exposure rather than
mastery. FFCS-A copies visible Identity evidence, FFCS-B copies a visible
historical answer, and FFCS-C quotes visible Dormant evidence. Those are honest
mechanism targets, but they do not teach free next-response generation, Axon's
voice, conversational judgment, or autobiographical synthesis.

Jeff directed communication before broad reasoning: first teach exact typed
response-draft control, then grounded conversation, shared Axon identity and
voice, autobiographical grounding, multi-turn continuity with private Soul,
and brother-proposal society; only then broaden into reasoning, tools, code,
math, and science. Each core's Soul remains private perspective and continuity,
while canonical Identity defines the single shared Axon identity.

The first communication-first implementation shot is now on disk: the C1
hidden-target single-turn curriculum (`training/communication_first_c1.py`,
manifest `c04ae8c6815e93b31136fdf833a525388c8439e74fd5317268cb1e1ba1233212`
under `State/training/curricula/c1/`). It carries 36 authored cases (24 train
/ 6 heldout / 6 regression) across seven lesson classes (greeting,
acknowledgement, grounded short answer, clarification request, correction
acceptance, uncertainty admission, turn-taking), unique whole-lineage splits,
and a compile-time hidden-target leakage verifier that also passes on the
reloaded manifest. Objectively checkable classes are labeled VERIFIED_TARGET;
authored conversational classes are PROCESS_EVIDENCE. All v1 cases are
authored fixtures with synthetic provenance (`C1_AUTHORED_SOURCE_ID`); no
Dormant records supervise. The manifest uses schema
`axon-first-form-curriculum-v1` and loads through the unchanged tournament
loader path. No C1 training run has launched; CPU preflight and tranche
tournaments remain ahead.

The current Trainer has portable foundations—CPU-loadable PyTorch state,
content-addressed parameter/Soul bundles, checkpoints, and fail-closed gates—
but no active cloud training capsule, provider adapter, dependency lock, or
user-facing launcher. Only archived batch files exist. The proposed surface is
one provider-neutral job manifest and CLI, with thin local/Kaggle/Colab/
Docker-SSH adapters plus a double-click launcher and local selection UI.

## Verification and Git

- Full repository suite passed after the C1 curriculum work: 483 tests
  collected on Python 3.12, exit 0, only the known PyTorch nested-tensor
  warning. One background invocation returned a spurious pipeline exit 1 with
  no pytest summary; clean collection and two complete runs contradict it.
- Changed-file Ruff: passed (C1 module, tests, publish script, `__init__`,
  FFCS validator, hygiene allowlist).
- Python compileall over changed files: passed.
- Fixed-character poison, exact Unicode, identity migration, mask, Heart,
  runtime circulation, private-Soul, Candidate-A, Trainer, and old-checkpoint
  migration tests are included in the full pass.
- SOT mirrors are byte-identical at SHA256
  `1CC61F65CCE03A58CC0F9A1A5CB867E97F0190D925D3BA91911560DC718B2FE1`.
- `git diff --check`: passed.
- `origin/main` contains recovered FFCS-D/E/F (`733e5d4`), campaign wiring
  (`617989b`), and the Soul-lineage repair/evidence (`0ca90c5`); final ledger
  bookkeeping follows.
- No active Axon training process or Trainer writer remains. No cloud job or
  paid operation ran.
- Known non-failing warning: PyTorch nested-tensor optimization warning.

## Active flags

1. **TRAINER CONTROL BLOCKER:** the 16-step ceiling is immutable plan and
   candidate identity. This is an implemented defect, not a desired limit. No
   further optimizer work until candidate lineage is separated from renewable
   resource tranches and continuation preserves checkpoint/optimizer/Soul.
2. **LEARNED CAPABILITY BLOCKER:** all candidates are far below the constant
   floor and both exact-output rates are zero.
3. **EVALUATION BLOCKER:** one heldout case is deferred, leaving nine formal
   tournament metrics missing and comparison inadmissible.
4. **REPORT RECOVERY GAP:** accepted state is safe if post-step evaluation
   crashes, but a complete-campaign resume cannot yet regenerate the missing
   segment report.
5. **AUTOBIOGRAPHY HOOK GAP:** reasoning commit/deposit recovery is closed, but
   every future tool executor and Trainer path is not yet automatically wired.
6. **SOUL CONCURRENCY GAP:** current use relies on Heart single-writer and
   Trainer lease ownership; `SoulBranch` is not independently multi-writer safe.
7. **WIDER-RAIL/CORTEX GAP:** only D64 physically compiles. Wider rails and
   autonomous Cortex retrieval/digestion remain future organs.
8. **REPO HYGIENE ADVISORY:** a separate Hermes audit counted 127 repo-wide
   Ruff findings outside changed-file scope; this does not contradict the
   changed-file Ruff gate.
9. **CLAIM BOUNDARY:** persistent behavioral identity is engineerable;
   subjective consciousness or metaphysical continuity is not testable here.
10. **CURRICULUM GAP (PARTIALLY CLOSED):** the C1 hidden-target single-turn
    curriculum now exists and is published (36 authored cases, leakage-
    verified), but C2-C6 do not, no C1 gate has been declared, and no C1
    training run has executed. FFCS remains the mechanism/regression
    schoolhouse.
11. **PORTABILITY/OPERABILITY GAP:** no active cloud capsule/provider adapter,
    environment lock, double-click launcher, or training selection UI exists.

## Recommended next actions

0. Run CPU preflight and one-step mechanism checks over the published C1
   manifest; declare the C1 stage gate before any tournament tranche.
1. Extend the communication-first curriculum through C2-C6 and gates; preserve
   FFCS as its mechanism/regression prerequisite rather than Axon's whole
   education.
2. Add crash-safe report regeneration, evaluation-only operation, and the one
   deferred heldout evaluation without optimizer mutation.
3. Build a provider-neutral training capsule/CLI, a thin double-click local
   launcher/UI, and Kaggle, Colab, and generic Docker/SSH adapters.
4. Resume a controlled head-geometry tournament only on complete communication-
   first gates; compare mastery, sample efficiency, compute, and regression.
5. Promote nothing until a candidate passes the complete serving gate.

## Important paths

- SOT: `docs/SOURCE_OF_TRUTH.md`
- Full report: `roundtable/LIVING_D64_CANDIDATE_A_REPORT_2026-08-28.md`
- Curriculum convergence review:
  `roundtable/CODEX_CURRICULUM_CONVERGENCE_REVIEW_2026-08-29.md`
- Campaign recovery report:
  `roundtable/CODEX_HERMES_CAMPAIGN_RECOVERY_2026-08-30.md`
- Identity/schema: `runtime/field/schema.py`, `runtime/heart/authority.py`,
  `runtime/heart/host.py`, `runtime/heart/masks.py`
- Private Soul: `runtime/soul/`
- Runtime circulation: `runtime/heart/circulation.py`
- Soul-aware lived episodes: `runtime/trainer/episodes.py`
- Candidate Soul branches: `runtime/trainer/soul_candidates.py`
- Accepted parameter/Soul bundles: `runtime/trainer/step_bundle.py`
- Candidate A: `training/living_reasoning_d64.py`
- Curriculum/preflight/tournament: `training/living_reasoning_curriculum.py`,
  `training/living_reasoning_preflight.py`, `training/reasoning_tournament.py`
- C1 communication curriculum: `training/communication_first_c1.py`,
  `scripts/compile_c1_curriculum.py`, `tests/test_communication_first_c1.py`,
  manifests under `State/training/curricula/c1/`
- Launcher: `scripts/train_living_reasoning_smoke.py`
- Full tests: `python -m pytest -q`
