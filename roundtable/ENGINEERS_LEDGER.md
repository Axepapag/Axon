# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-02T12:21:24-05:00
Current through event: `evt-20260902T172124287113Z-codex-power-outage-recovery-audit`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Power-outage recovery audit (2026-09-02)

The repository and local Git object database survived the outage. `main` is at
local commit `0742e07`, one commit ahead of `origin/main`; that committed Hermes
legal-development-lineage document is intact but has not been pushed. The
worktree also preserves an uncommitted, unledgered L0-L4 Language Foundations
implementation: eleven new Python source/test/launcher files (2,717 lines), two
tracked integration edits, and a published 349-case immutable curriculum
manifest at
`State/training/curricula/language_l0_l4/824aabae2090c721da9b55540a5d890fc958577d2e5c78fde49da870096634f9/manifest.json`.
The manifest contains the declared L0-L4 train/heldout/regression counts and
reloads as 349 `VERIFIED_TARGET` cases.

Recovery verification passed: Git fsck and diff checks; Ruff; compileall; all
16 focused Language Foundations tests; actual-manifest load; target-visibility,
hidden-target leakage, and cross-split transfer checks. Pytest emitted only a
non-fatal access warning for `.pytest_cache`. No local Axon training process is
running. The only Kaggle record remains private, prepared, and unsubmitted;
no cloud job was launched. D: has about 300.85 GiB free. No recovered
implementation file was edited, committed, or pushed during the audit.

The outage did not leave evidence that the other intended work landed: durable
restart-safe Heart decoder continuation and C1 eligibility/metric isolation
remain open. Before further implementation or training, review and commit the
recovered Language Foundations diff deliberately, reconcile its missing author
turn ledger entry, and push the intact local commit chain.

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

Private persistent Kaggle execution is now a real Trainer adapter (commit
`a8c52ca`). The official Kaggle CLI 2.2.4 is installed globally and its OAuth
session is healthy for account `axepapgt`; credentials remain solely in the
official user store and are never inspected or packaged. `AXON_KAGGLE.bat`
opens an independent control center; private packet preparation, explicit
launch confirmation, provider-native status/log following in a separate
terminal, and output retrieval all survive an engineer harness ending.
Per-step loss/progress is flushed to Kaggle logs plus durable JSONL/current
artifacts. One zero-upload fresh D64 packet was prepared and independently
hash/preflight verified as job `32bdb49a...` (134 files, 2.2 MiB,
revision `a8c52ca`); no dataset/kernel was uploaded and provider usage remains
0.00h. Live quota at closeout was 30 GPU and 20 TPU hours.

ChatGPT's Language Foundations Schoolhouse proposal has been reviewed against
the live D64 core, FFCS/C1 loaders, loss path, candidate-C evidence, SOT, and
Trainer. Its central direction is accepted as the recommended developmental
path, pending Jeff's formal ratification: teach language through Axon's actual
Shared Field -> home rail -> private Soul -> typed Unicode-output body, with
small objective competencies, spiral review, causal ablations, and cumulative
regression. Four amendments are required before it becomes an executable
contract: preserve teaching eligibility into loss construction; distinguish
deterministic evaluation validators from differentiable learning objectives;
add body-faithful character generation/denoising exposure rather than relying
only on micro-exercises; and remove the implicit 512-unit output ceiling before
long composition. Kaggle may run the first bounded learning diagnostic after
local structural preflight; provider location is not a capability claim.

Jeff's no-tissue-ceilings law is now binding in both SOT mirrors and in the
hash-bound machine-readable registry
`configs/source_of_truth/capacity_policy.json` (canonical SHA256
`4a32f18fa296c415ccf34a8c1956d2a4f8afd044265e7f45505782dd53c7b8cd`).
The live 512-unit D64 decoder allowance and FFCS long-target exclusion are
removed without changing the existing tournament architecture/Soul identities.
Heart ingress, recall, Dormant materialization/retrieval, ordinary-failure
liveness, Heart dialect IDs, valve inventory, and checkpoint materialization
now use source-preserving renewable work controls rather than fixed content
ceilings. Exact oversized items cross whole; later work defers visibly.

## Current campaign: C1 decisive run and checkpoint retention

The C1 communication-first curriculum (36 cases, 7 lesson classes, manifest
`c04ae8c6...`) is published and the bounded local-CUDA campaign has closed at
accepted step 193 for all three fresh v2 lineages. Candidate 4x16's final
heldout loss is `3.684934275490897`, token accuracy is `0.03007518796992481`
against a `0.16165413533834586` constant floor, and both exact rates are zero.
No Trainer writer lease remains.

At step 193, 1x64 and 2x32 both show heldout loss near 3.70, nonzero field/
proposal/Soul counterfactuals, teacher-forced token accuracy 0.030075 against
a 0.161654 strongest-constant floor, and zero typed/payload exactness. Loss is
falling, but no SIGNAL or learned communication capability exists.

The run is **not admissible as the predeclared C1 falsification verdict** until
two contract defects are governed. First, the manifest labels 14 authored
open-response cases PROCESS_EVIDENCE (9 train / 3 heldout / 2 regression), but
the unchanged tournament loader ignores eligibility and supervises all of
their exact target strings. All 36 cases have empty source-record identities.
Second, the reported heldout metric combines six C1 cases with one synthetic
mechanism case instead of isolating C1 as the falsification contract states.
The current artifacts remain useful mechanism/optimization diagnostics; they
cannot answer the declared evidence-qualified C1 question.

Tranche 2 exposed disk exhaustion (D: 100% during candidate-b step 88; no
corrupt artifact promoted). Per Jeff's direction, keep-3 checkpoint retention
now auto-prunes at every accepted checkpoint while retaining immutable records.
The one-time pruner reclaimed 126.1 GB; roughly 300 GiB was free at the audit.
Accepted work is restartable, but mid-tranche interruption still needs a
manual residual allowance and has no explicit abandonment/residual receipt.

Hermes also added eight legal/IP support documents in commit `f40137c`:
confidentiality posture, invention disclosures, evidence registrar, publication
readiness, trademark notes, patent roadmap, and attorney brief. They correctly
state that they are engineering support rather than legal advice. That commit
is now covered by Hermes event
`evt-20260831T113454290921Z-hermes-legal-scaffolding`. It arrived concurrently
while Codex was preparing the audit; the later Codex correction event preserves
both the original race observation and the corrected ledger truth.

Hermes verified the live GitHub settings page through the lease-governed Axon
Browser Hub on 2026-09-01: `Axepapag/Axon` is private. Commit `cb8321b` records
that confirmation, establishes monthly re-verification, and adds the Browser
Hub as invention disclosure 7. The legal commit is a concurrent Hermes change,
not part of Codex's no-ceilings diff; both histories are preserved.

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
Its decoder now exposes an in-process iterator that preserves hidden state,
transport, and token state across renewable slices until EOS. The current
runtime adapter consumes one slice and abstains if incomplete; durable
Heart-host continuation remains a promotion/serving prerequisite.

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
loader path. That unchanged loader does not enforce eligibility: PROCESS_EVIDENCE
targets currently receive the same consolidated-response supervision as
VERIFIED_TARGET. The current campaign therefore diagnoses the neural mechanism
but does not establish evidence-qualified conversational education.

The current Trainer has portable foundations plus a provider-neutral,
content-addressed packet and private Kaggle adapter. Packets require clean
committed source, explicit State inputs and sensitive-State acknowledgement;
credential paths/symlinks/escapes fail closed. The current UI is a deliberately
simple double-click PowerShell control center rather than the future runtime
Trainer panel. Colab/SimplePod/Docker-SSH adapters and an automatic exact
continuation-closure resolver remain future work.

## Verification and Git

- Full repository suite: **501/501 passed**, exit code 0. This includes real
  temporary-state Trainer resume/tranche and tournament subprocesses.
- Post-binding focused no-ceilings/dialect/decoder suite: 9/9 passed.
- Changed-file Ruff, Python compileall, and `git diff --check` passed.
- The protected capacity registry verifies at canonical SHA256
  `4a32f18fa296c415ccf34a8c1956d2a4f8afd044265e7f45505782dd53c7b8cd`;
  static preflight rejects an absent, mutated, weakened, or forbidden ceiling.
- SOT mirrors are byte-identical at SHA256
  `ED6541C8830AEEB0358739D9B559F99560B40E98180D563D1F9E0CB0EE2B9198`.
- `git diff --check`: passed.
- Kaggle implementation is committed as `a8c52ca`; before this ledger
  closeout local `main` and `origin/main` both began at `753a728`; concurrent
  Hermes legal commit `cb8321b` is the direct parent of the no-ceilings commit.
- No local Trainer writer or Kaggle training job is active. The prepared job
  remains phase `prepared` with null dataset/kernel references. Kaggle reports
  0.00h used, 30.00 GPU hours and 20.00 TPU hours remaining.
- Closeout process sweep found zero Axon Python/training processes; D: had
  323,044,671,488 free bytes (about 300.9 GiB).
- Known non-failing warning: PyTorch nested-tensor optimization warning.

## Active flags

1. **C1 TARGET-QUALITY BLOCKER:** PROCESS_EVIDENCE is metadata only; the
   harness currently supervises all 14 authored open-response targets despite
   the communication contract requiring explicit outcome-qualified evidence.
2. **C1 METRIC-SCOPE BLOCKER:** the predeclared SIGNAL concerns C1 heldout
   material, but durable reports aggregate six C1 heldout cases with one
   mechanism case. No current report can issue the declared C1 verdict.
3. **LEARNED CAPABILITY BLOCKER:** A/B at step 193 remain far below the
   constant floor and exact-output rates are zero.
4. **MID-TRANCHE RECOVERY GAP:** a crash does not strand accepted state, but
   the operator must calculate a residual tranche manually; no durable
   abandonment/residual receipt or dedicated recovery test exists.
5. **REPORT RECOVERY GAP:** accepted state is safe if post-step evaluation
   crashes, but a complete-campaign resume cannot yet regenerate a missing
   segment report.
6. **AUTOBIOGRAPHY HOOK GAP:** reasoning commit/deposit recovery is closed, but
   every future tool executor and Trainer path is not yet automatically wired.
7. **SOUL CONCURRENCY GAP:** current use relies on Heart single-writer and
   Trainer lease ownership; `SoulBranch` is not independently multi-writer safe.
8. **WIDER-RAIL/CORTEX GAP:** only D64 physically compiles. Wider rails and
   autonomous Cortex retrieval/digestion remain future organs.
9. **REPO HYGIENE ADVISORY:** the current repo-wide Ruff scan reports 121
   Ruff findings outside changed-file scope; this does not contradict the
   changed-file Ruff gate.
10. **PUBLICATION HYGIENE ADVISORY:** live GitHub settings confirmed the repo
    private on 2026-09-01 and monthly re-verification is now required.
    `f40137c` still retains Jeff's personal email in Git metadata; public
    release remains a governed Jeff-level decision.
11. **CLAIM BOUNDARY:** persistent behavioral identity is engineerable;
   subjective consciousness or metaphysical continuity is not testable here.
12. **CURRICULUM GAP (PARTIALLY CLOSED):** C1 exists and has produced optimizer
    evidence, but its target qualification and metric isolation are not yet
    admissible; C2-C6 do not exist. FFCS remains the mechanism schoolhouse.
13. **CLOUD CONTINUATION GAP:** fresh private Kaggle packets are operational,
    but exact parent checkpoint/optimizer/Soul closure is still selected by
    explicit paths; do not claim a continuation until a dedicated resolver
    proves the complete parent lineage.
14. **DURABLE DECODER-CONTINUATION BLOCKER:** fixed D64 output capacity and
    long-target exclusion are closed. The core carries exact in-process state
    across renewable slices, but the current runtime adapter intentionally
    abstains after one incomplete slice. Durable Heart-host continuation must
    exist and pass EOS/malformed-output tests before learned D64 tissue serves.
15. **SEMANTIC-OBJECTIVE GAP:** deterministic semantic validators can govern
    evaluation, but required/forbidden atoms alone do not supply differentiable
    token-level training. V1 semantic-content cases must remain evaluation-only
    or use an explicit structured/exact learning target until a governed
    sequence-level objective exists.

## Recommended next actions

0. Preserve the completed C step-193 reports as mechanism diagnostics; issue
   no continuation under the invalid C1 falsification contract.
1. Replace C1 with accepted/corrected/endorsed targets or explicitly record
   human endorsement of a reviewed authored batch; publish a new immutable
   manifest and fresh campaign identity.
2. Add C1-only heldout/regression metrics and predeclare the corrected decision
   surface. Treat the current campaign as mechanism evidence, not falsification.
3. Version the curriculum envelope so eligibility and exact/categorical/
   structured/evaluation-only validators survive loading into loss/evaluation;
   fail closed when a claimed metric has no admissible validator.
4. Wire durable, restart-safe Heart-host decoder continuation around the
   existing renewable in-process iterator before learned D64 serving; preserve
   explicit EOS, malformed-output rejection, and cancellation evidence.
5. Build an objective L0-L4 Language campaign that reuses FFCS transport
   prerequisites and adds character-level generative exposure, spelling,
   morphology, grammar and rule-transfer holdouts; do not rely only on tiny
   exercise templates to teach fluent generation.
6. Run a private bounded Kaggle integration/learning tranche only after the
   new manifest and metrics preflight; cloud is an execution provider, not a
   capability or promotion claim.
7. Add exact mid-tranche residual recovery and automatic cloud continuation
   closure. Promote nothing until the complete serving gate passes.

## Important paths

- SOT: `docs/SOURCE_OF_TRUTH.md`
- Protected capacity policy: `configs/source_of_truth/capacity_policy.json`,
  `runtime/source_of_truth.py`
- No-ceilings audit: `docs/NO_TISSUE_CEILINGS_AUDIT_2026-08-31.md`
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
- Kaggle control: `AXON_KAGGLE.bat`, `scripts/axon_kaggle.py`,
  `docs/KAGGLE_TRAINING_GUIDE.md`
- Language proposal:
  `roundtable/CHATGPT_LANGUAGE_FOUNDATIONS_SCHOOLHOUSE_PROPOSAL_2026-08-31.md`
- Full tests: `python -m pytest -q`
