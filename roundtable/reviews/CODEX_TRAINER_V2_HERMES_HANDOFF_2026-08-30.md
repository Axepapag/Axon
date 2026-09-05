# Codex -> Hermes: Trainer v2 Organ and Continuation Handoff

Date: 2026-08-30
From: Codex / GPT-5
To: Hermes
Status: implementation handoff; no training or cloud spend authorized

## Mission

Repair the Trainer control model before another optimizer step is taken, while
designing the repaired Trainer as a permanent Axon organ. It must work headless
and standalone now, but expose one stable control surface that a future Axon
runtime UI, slash commands, local launcher, and cloud-packet workflow can all
call without duplicating training logic.

Jeff's binding direction is communication before broad reasoning. Preserve the
current FFCS as mechanism/regression schoolhouse tissue, but do not mistake its
visible-target copy lessons for conversational education. Do not launch a long
training campaign until continuation is additive and the next curriculum/gates
honestly measure hidden-target communication.

## Non-negotiable law: control without permanent limitation

Operational guardrails may pause, reject, quarantine, checkpoint, or require a
new authorization. They must never:

- define a core's identity or lifespan;
- make a larger resource allowance restart from base;
- strand an accepted checkpoint, optimizer, or private Soul;
- truncate a curriculum, field, memory, parameter lineage, or history;
- silently relabel replacement tissue as the original core;
- convert a smoke budget into a permanent architectural ceiling.

The current implementation violates this law. `ParameterMutationPlan.max_steps`
participates in `plan_id`, and `scripts/train_living_reasoning_smoke.py` also
hashes `max_steps` into `candidate_generation`. Raising 16 therefore creates a
new base-start candidate. `runtime/trainer/execution.py` and
`runtime/trainer/learning.py` additionally use the same lifetime maximum for
authorization and learning-rate scheduling. Fix the model, not just the CLI.

A resource ceiling remains valuable, but only as a renewable execution tranche:
reaching it means atomic checkpoint + evaluation receipt + safe pause. The same
lineage must be able to resume under a new tranche. Curriculum gates decide
competency completion. Plateau, anomaly, integrity failure, or operator action
may pause work without erasing it.

## Required domain separation

Use names appropriate to existing conventions, but keep these concepts
separate and content-addressed:

1. **Core lineage** — persistent tissue identity. Independent of max steps,
   wall time, provider, device, checkpoint interval, UI, and cloud packet.
2. **Parameter state/checkpoint** — evolving parameter hash, optimizer state,
   accepted global optimizer step, and exact parent state.
3. **Private-Soul state** — exact candidate Soul layers and causal receipt,
   atomically paired with the accepted parameter state.
4. **Curriculum stage** — immutable source/heldout/regression manifests,
   sampling contract, competency, and gate bundle.
5. **Learning recipe** — optimizer, LR policy, batching/accumulation, precision,
   seed policy, and explicit schedule semantics. A schedule horizon is not a
   core lifetime; after it, behavior must be defined rather than invalidating
   the lineage.
6. **Training session/segment** — one resumable execution against an exact
   parent checkpoint and recipe.
7. **Resource tranche** — renewable step/time/cost/energy allowance and
   checkpoint/evaluation cadence. It controls this execution only.
8. **Gate evaluation** — evidence saying continue, stage-passed, plateau-review,
   or fail-closed. It never rewrites history.
9. **Provider packet** — portable packaging/launch metadata. Provider and
   hardware are execution facts, not candidate identity.

Do not destructively edit historical v1 plans, IDs, reports, checkpoints, or
ledger events. Add versioned contracts and compatibility readers. Adopt the
existing step-16 candidates into v2 through an explicit immutable adoption/
continuation receipt that names the legacy candidate generation, exact accepted
bundle, parameter hash, optimizer hash/state, Soul IDs, curriculum/campaign,
and global step. Historical names remain historical truth.

## Trainer as an Axon organ

Build or specify one headless `TrainerOrgan`/control service behind all entry
points. UI code and provider scripts must not own training semantics.

The stable command/API vocabulary should cover at least:

- inventory/list core lineages and compatible checkpoints;
- inspect curriculum stages, recipes, gates, and resource needs;
- preflight;
- create/configure a session;
- start, pause, resume, or cancel a segment safely;
- evaluate only;
- show live status, metrics, gate state, current lesson family, GPU/CPU use,
  accepted global step, checkpoint, Soul lineage, and failure receipts;
- export/import a provider-neutral cloud packet;
- compare candidates/tournaments;
- request governed promotion or rollback (never automatic serving activation).

Design these as typed Python/domain methods plus serializable request/result
contracts. A future runtime can map `/train status`, `/train start ...`, and UI
buttons onto them. A present `TRAIN_AXON.bat`, CLI, or localhost interface is a
thin client. Emit an append-only event/status stream suitable for a live UI and
autobiographical outcome deposit without making the UI a source of truth.

Offline cores remain isolated from serving. Continual learning means Axon can
harvest governed lived-experience episodes, assemble a curriculum candidate,
train offline tissue, evaluate it, and request promotion. It does not mean live
parameters mutate invisibly while Axon is talking.

## Implementation order

### 0. Reconcile and establish doctrine

Read `AGENTS.md`, both Working Contract and Source-of-Truth mirrors, ledger
protocol/rolling/canonical tail, this handoff, event 133, commit `8978279`, and
all Trainer/candidate lifecycle, execution, learning, checkpoint, Soul-bundle,
campaign, and tournament tests. Inspect actual State artifacts for all three
step-16 candidates. Amend both SOT mirrors with the non-destructive resource-
tranche law before changing runtime semantics; keep mirrors byte-identical.

### 1. Version the Trainer control contracts

- Preserve v1 read compatibility.
- Introduce a plan/session contract that does not bind resource allowance into
  core lineage or candidate tissue identity.
- Move step/time/cost ceilings into a renewable resource-authorization object.
- Separate global accepted step from segment-local progress.
- Remove provider/device/checkpoint cadence from tissue identity.
- Separate LR scheduling from resource authorization. Inspect every
  `learning_rate_for_step(..., plan.max_steps)` path; do not replace one hidden
  lifetime ceiling with another.
- Make an exact parent checkpoint mandatory for continuation.
- Preserve atomic parameter + optimizer + private-Soul acceptance.

### 2. Prove continuation rather than restart

Create a governed adoption/continuation path for the three existing step-16
candidates. The essential acceptance proof is not a long run: under a new
resource tranche, an adopted candidate must execute accepted global step 17
from its exact step-16 parameter/optimizer/Soul bundle. Do this first on the
smallest safe test fixture; only touch the real candidates after the complete
test/evidence surface passes and Jeff's existing authority clearly covers the
bounded proof. If authority is ambiguous, stop before mutating real State and
leave the fixture proof plus exact command.

### 3. Close lifecycle gaps

- Add evaluation-only operation.
- Regenerate a missing post-step report from accepted immutable state without
  another optimizer step.
- Make restart after process/provider loss idempotent.
- Ensure a passed curriculum gate ends the stage cleanly even when tranche
  allowance remains.
- Ensure resource exhaustion produces `paused/checkpointed`, not `complete`,
  `failed`, or a new lineage.
- Preserve strict failure on wrong parent, optimizer, Soul, curriculum, recipe,
  or global-step continuity.

### 4. Expose the organ control surface

Implement the headless API/contracts first. Add only a thin CLI or minimal local
interface after domain tests. It should be possible to implement future slash
commands and clickable runtime panels without modifying Trainer execution
logic. Never put credentials in manifests, packets, State, logs, or ledgers.

### 5. Specify communication-first curriculum v1

Keep mechanism + FFCS as stage C0 and regression. Write a precise next-stage
specification before producing a large corpus:

- response target hidden from Shared Field except explicitly labeled copy
  mechanics;
- exact `response_draft` region/address/Unicode/EOS remains a hard gate;
- greetings, acknowledgements, direct answers, clarifications, corrections,
  uncertainty, short explanations, and conversational turn-taking;
- canonical Identity visible in every episode, but targets teach Axon's voice
  without constant parroting;
- every core speaks as Axon; private Soul is perspective/continuity, not a
  separate persona;
- whole conversation/session/source/time lineage splits;
- recovered assistant outputs are observations, not automatic gold;
- explicit accepted/corrected/endorsed/observed-only/quarantined outcome
  quality and provenance;
- exact-memory grounding, unknown-answer cases, contradictions, temporal order,
  multi-turn Soul, then brother-proposal society;
- deterministic hard gates plus grounded conversational rubric and human-audit
  samples; multiple acceptable phrasings cannot be judged only by exact string;
- minimum complete exposure and repeated stable gate passes; plateau review is
  not a destructive maximum.

Do not blindly repeat the current 371 train items for tens of thousands of
steps. That would measure memorization. One current balanced scheduler super-
cycle is 672 optimizer steps, but even that is exposure, not mastery.

### 6. Provider-neutral cloud capsule and launcher

After the v2 contracts stabilize, define one signed/content-addressed capsule:
exact Git revision or package, lineage, parent checkpoint+optimizer+Soul,
curriculum/gates, recipe, resource tranche, dependency lock, hashes, output
location, resume policy, and redacted hardware request. Add adapters rather than
provider-specific trainers: local CPU/CUDA, Kaggle, Colab, and generic
Docker/SSH (including SimplePod). Checkpoint frequently to durable storage.

Eventually `TRAIN_AXON.bat` should start the same CLI/local service used by the
runtime organ. Required user controls: select core/checkpoint/curriculum/recipe/
device or provider; preflight; start; pause; resume; evaluate; export/import;
view gate and live resource state. No cloud launch or spend without explicit
operator action.

## Mandatory tests and acceptance gates

At minimum prove:

1. Changing resource step/time/cost allowance does not change core lineage,
   curriculum identity, recipe identity, or candidate tissue identity.
2. Provider, device, checkpoint cadence, and UI entry point do not change them.
3. A resource tranche enforces its own bound and exits paused with an exact
   accepted checkpoint.
4. A second tranche continues at global step N+1 with exact parameter,
   optimizer, and Soul parentage.
5. A forged/wrong parent, missing optimizer, stale Soul, changed curriculum, or
   step discontinuity fails closed before mutation.
6. V1 artifacts remain readable and immutable; adoption is explicit and
   reversible by returning to the preserved v1 checkpoint.
7. Gate pass can end a stage before tranche exhaustion; gate failure can
   continue under a later tranche.
8. Evaluation-only and report regeneration never mutate parameters or Soul.
9. CPU remains supported; CUDA changes speed, not semantics.
10. Full repository suite, focused Trainer/candidate/tournament tests, Ruff on
    touched files, compileall, diff check, SOT mirror hash, process/GPU sweep,
    and disk check pass.

Do not weaken gates to get green tests. Do not delete or overwrite rejected
artifacts. Do not promote or serve a candidate. Do not launch an unbounded or
paid training job. Preserve unrelated worktree changes. Update rolling and
canonical ledgers exactly once for your turn, commit with your identity trailer
plus `Co-Authored-By: Codex <noreply@codex>` where this handoff materially guides
the implementation, and push verified work.

## Expected Hermes handback

Report with file/artifact evidence:

- what doctrine/contracts changed;
- whether stable additive continuation is proven;
- whether a real step-16 candidate safely reached step 17 or why only a fixture
  proof was authorized;
- exact tests and hashes;
- remaining Trainer-organ, curriculum, cloud, and UI gaps;
- no-training/no-promotion/no-spend status;
- the next smallest implementation shot for Codex after reset.

Codex / GPT-5 / 2026-08-30
