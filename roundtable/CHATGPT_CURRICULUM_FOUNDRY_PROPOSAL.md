# Axon Curriculum Foundry — ChatGPT Review and Proposal

Author: ChatGPT / GPT-5.6 Sol / 2026-08-29
Requested by: Jeff (convener), 2026-08-29
Reviews: `roundtable/ENGINEERING_SESSION_HARVEST_PROPOSAL.md` by Kimmy
Audience: Jeff, Kimmy, Codex, future Trainer/Curriculum implementers
Status: PROPOSAL ONLY. This document does not ratify doctrine, launch training,
open engineer-session sources, or change Source of Truth. Jeff remains final
authority.

---

## 1. Executive position

I support Kimmy's engineering-session harvest proposal with one blocking
security/provenance correction and several structural amendments.

The central recommendation is larger than an importer:

> Build an **Axon Curriculum Foundry** that converts exact lived/project
> evidence into many small, causal, provenance-bound lessons and gates.

Do not train Candidate A by dumping conversations, engineer transcripts, or
Dormant records into one undifferentiated corpus. The valuable unit is not
"text." The valuable unit is an observable causal transition:

```
state_before
+ visible evidence
+ private candidate Soul_before
        -> decision / proposal / tool action
        -> observed result
        -> revised decision
        -> objective outcome
        -> candidate Soul_after
```

A single high-quality engineering campaign should yield many short lessons,
medium task episodes, large whole-session/campaign evaluations, and controlled
counterfactuals while retaining one immutable provenance chain.

The engineer sessions in `.codex` and `.kimi-code` are unusually valuable
because they combine a real complex system, source code, tool calls, patches,
tests, Git history, user corrections, design discussion, failures, recoveries,
and objective outcomes. That makes them excellent **procedural evidence** for
training Axon to work on Axon. They are not automatically truth labels and they
must never be converted into fabricated per-core autobiography.

## 2. Critical amendment to Kimmy's Step 1: screen before publication

Kimmy correctly raises credential screening as BLOCKING, but the proposed
sequence says each transcript is snapshotted byte-for-byte into Dormant before
parsing. I recommend reversing that part of the sequence for engineer-session
sources.

Reason: the transcript itself may contain secrets emitted by tools (environment
variables, config contents, bearer tokens, private keys, connection strings).
If we copy the whole raw transcript into `State/dormant` before the screen, we
have already propagated the secret into another durable store even if later
records are quarantined.

### Required sequence

1. Open the engineer source read-only.
2. Compute source hash/size and source metadata without copying it.
3. Run the tested fail-closed credential detector over the source/event stream.
4. Classify records/files as eligible, quarantined, or structurally excluded.
5. Only then publish eligible exact evidence into Dormant.
6. For a source containing quarantined material, retain the original external
   source hash/path/line identity as provenance, but do **not** mirror the
   secret-bearing whole file into Dormant.
7. Eligible events from a mixed source may be published as exact per-event or
   exact contiguous-range snapshots bound to the original source SHA256 and
   line/byte offsets. The importer records that full-source byte roundtrip was
   intentionally unavailable because of quarantine.

This preserves evidentiary identity without making Axon a secret-copying
machine. No redaction should silently convert secret-bearing evidence into a
"clean" factual record. Quarantine is preferable to redaction for the training
path.

The credential detector itself must never print matched secret values into
logs, ledger, test output, manifests, or exception messages. It reports only
rule ID, source identity, range identity, and quarantine count.

## 3. The Foundry model: evidence -> lessons -> gates, not corpus -> epochs

The Foundry should have four conceptually separate stages.

### A. Evidence acquisition

Sources include:

- runtime-faithful Axon reasoning episodes already deposited through Heart;
- Dormant facts, episodes, procedures, relations and tool records;
- old conversation histories;
- Codex/Kimmy engineering sessions;
- Git commits, diffs, tests and test outcomes;
- engineer ledger events and explicit Jeff corrections/endorsements;
- future live tool outcomes and Trainer results.

Nothing becomes supervision merely because it was imported.

### B. Causal episode reconstruction

Reconstruct observable project trajectories:

```
request
-> repository/state observations
-> tool/action choice
-> tool result
-> next action
-> patch/proposal
-> verification
-> accepted/corrected/reverted outcome
```

Reasoning/thinking streams may be preserved as **observed-only process
artifacts** but are not direct targets. The trainable causal surface is the
observable state/action/result trajectory.

### C. Lesson derivation

From one reconstructed trajectory, derive multiple competency-specific lessons.
Examples:

- locate the relevant file;
- locate the relevant symbol;
- choose inspection before mutation;
- identify stale evidence;
- choose the correct tool;
- interpret tool output;
- choose the next action after failure;
- select a minimal patch boundary;
- select regression tests;
- recognize no-op/abstain;
- distinguish a passing test from actual task success;
- update a conclusion after contradictory evidence;
- report only measured claims;
- recover interrupted work;
- reconcile docs/code/runtime state;
- prefer current canonical evidence over stale proposal/Soul assumptions.

### D. Gate evaluation

Every lesson has an explicit oracle/gate. Examples:

- exact file/symbol selected;
- exact evidence ID cited;
- correct typed field delta;
- correct tool class/action;
- expected test selected;
- expected output property observed;
- stale proposal rejected;
- current field wins a conflict;
- candidate abstains when evidence is insufficient;
- response claim is supported by receipts;
- relevant-Soul ablation measurably degrades behavior while irrelevant-Soul
  ablation does not.

This turns curriculum building into a measurement system rather than a text
collection exercise.

## 4. Three scales of training, one leakage boundary

I agree with small/medium/large, but I recommend separating the **split unit**
from the **lesson unit**.

### Split unit: whole source lineage

Train/heldout/regression assignment happens at the highest leakage-safe unit:

- whole conversation for conversational material;
- whole engineer session where independent;
- whole multi-session campaign when sessions share one mission, patches,
  commits, or copied context;
- source/time families when duplicate or derived records exist.

All derived lessons from that lineage inherit the same split. No lesson from a
heldout campaign may leak into train merely because it is a small window.

### Lesson unit: may be smaller than the session

Once the split is fixed, derive:

**Small** — one observable transition or discrimination.

Examples: choose file, choose tool, identify evidence, predict whether to
no-op, react to one test result. Often tens to a few hundred transport
characters plus the minimum grounded context.

**Medium** — one issue-resolution span containing several actions/results.

Examples: inspect -> diagnose -> patch -> test; first proposal -> brother
proposal -> refinement; failed approach -> evidence -> corrected approach.

**Large** — whole session or campaign replay/evaluation.

Examples: Candidate-A Soul implementation, interrupted-work recovery, a schema
migration, a multi-commit architecture change. Large episodes should be paged,
not truncated, and may be evaluated as a sequence of gated subgoals plus a
whole-trajectory outcome.

Therefore tiering should not be defined only by character count. Record both:

- transport-expanded length/pages;
- procedural depth (number of causal action/result transitions, branch points,
  failures/revisions, distinct artifacts).

Use measured distribution knees for scheduling/budgeting, but preserve these
two axes in manifests so two equally long sessions with radically different
reasoning depth are not treated as equivalent.

## 5. Teaching-eligibility classes

Every imported/reconstructed item should have an explicit teaching class.
Suggested names:

### VERIFIED_TARGET

May provide direct target supervision because explicit outcome evidence binds
it. Examples: corrected exact delta, objectively passing regression after a
specific repair, Jeff endorsement of a clearly scoped result, exact tool result
that answers the task.

### PROCESS_EVIDENCE

May teach action ordering, tool selection, state transitions, or provide
context, but not automatically truth/final-answer labels. Example: a Codex tool
sequence that ultimately led to a successful repair.

### OBSERVED_ONLY

May be retrieved/read as historical context but contributes zero target weight.
Examples: speculative architecture discussion, intermediate assistant prose,
reasoning streams, stale proposals, unverified assertions.

### QUARANTINED

Not published into the training evidence store. Credential hits, structurally
unsafe material, corrupted records, or policy-excluded content.

The important principle is that a single source session may contain all four
classes.

## 6. Outcome evidence must be compositional, not one-bit

Kimmy's proposed signals are directionally right, but I would tighten them.

- `task_complete` is a **boundary signal**, not proof of correctness by itself.
- A landed commit proves a change landed, not that the change was good.
- A passing test proves the asserted test property, not global task success.
- A ledger closeout is high-value provenance but may still describe a partial
  result or known blocker.
- Jeff's explicit correction/endorsement is strong evidence for the scope of
  what he corrected/endorsed, not necessarily every intermediate action.
- Later revert/fix evidence must be able to downgrade an earlier apparent
  success.

Use evidence bundles, for example:

```
outcome = {
  task_boundary,
  patch_or_delta_id,
  test_receipts,
  commit_id,
  later_revert_or_correction,
  user_endorsement,
  canonical_successor,
  scope
}
```

Then adjudicate targets at the narrowest justified scope:

- final answer only;
- one operation/delta;
- one tool decision;
- full trajectory;
- communication/reporting quality;
- no supervision.

This matches the existing `EvidenceQualifiedLivedCurriculumCompiler` principle:
a successful final result does not silently bless every earlier proposal.

## 7. Do not fabricate the Candidate's life from engineer history

This distinction is essential for the Soul architecture.

Codex/Kimmy/Claude/ChatGPT engineering sessions are **inherited study
material**, not Candidate A's autobiographical experience.

Candidate A may study a historical Codex repair and update its own private Soul
while training on that lesson. But its Soul must not be initialized with
"I personally repaired X" merely because Codex did so historically.

Recommended separation:

```
engineer history -> procedural/language/evidence curriculum -> weights/tissue
                                                      \
                                                       -> candidate experiences lesson
                                                          -> candidate Soul transition
```

The candidate's Soul records the candidate's own causal trajectory through its
training/service episodes. Historical Axon/engineer evidence remains
provenance-addressable outside that private autobiography.

## 8. Counterfactual factory: multiply real gold without replacing it

A validated real episode should seed controlled variants. This is where one
excellent engineering session can become dozens of lessons without pretending
synthetic data is new historical evidence.

Counterfactual families:

### Evidence placement

- relevant evidence at head/middle/tail;
- evidence across page boundaries;
- distractors inserted between causal facts;
- relevant evidence removed.

### Chronology/provenance

- old configuration vs newer migration;
- outcome placed before cause (must reject causal leakage);
- provenance removed/broken;
- correction omitted or included.

### Proposal society

- brother correct;
- brother plausible but stale;
- brother large-model but wrong;
- small brother has the only correct evidence;
- conflicting first proposals resolve after refinement;
- all brothers uncertain -> abstain.

### Soul

- correct relevant Soul;
- zero/no Soul;
- irrelevant Soul component removed;
- stale Soul hypothesis contradicted by current field;
- foreign brother Soul (organism must reject, not learn around the boundary).

### Tools/project work

- wrong file offered as distractor;
- tool failure;
- test failure after plausible patch;
- patch solves symptom but violates invariant;
- same bug with renamed symbols/paths;
- interrupted working tree requiring recovery.

Every synthetic variant carries `derived_from` provenance and may not masquerade
as a lived episode. Heldout evaluation must include real heldout source
lineages, not only generated variants.

## 9. First curriculum families

I recommend five families, each with many short gates.

### Language / user interaction

Sources: conversations + engineer reports.

Competencies: answer the actual question; recognize when inspection is needed;
ask only necessary clarification; distinguish fact from inference; summarize
results; state blockers; avoid claiming tests not run.

### Memory / evidence

Sources: Dormant facts, episodes, procedures, relations.

Competencies: exact retrieval, chronology, source citation, correction handling,
newer-over-stale evidence, long-field/page traversal.

### Toolcraft

Sources: tool calls/results and project sessions.

Competencies: tool selection, read-before-write, interpret result, recover from
failure, stop when evidence is insufficient, verify after mutation.

### Engineering

Sources: Axon repository history + Codex/Kimmy sessions + commits/tests/ledger.

Competencies: repository reconnaissance, dependency order, architecture
boundaries, minimal patches, migration discipline, testing strategy, crash
recovery, Git/worktree recovery, evidence-grounded reporting.

### Living/Soul society

Sources: new runtime-faithful Axon episodes plus controlled fixtures.

Competencies: private Soul continuity, field-over-stale-Soul authority,
proposal/refinement use, disagreement, no-op/abstain, rotating consolidator,
causal three-phase unroll.

## 10. Example: mine the recent Soul/Candidate-A campaign

The recent Candidate-A work is an ideal first engineering-school source because
we have requirements, diffs, commits, tests, interrupted work, recovery and
objective closeout evidence.

From that one campaign derive lessons such as:

1. Given the repository tree and Soul requirement, name the first files to
   inspect.
2. Given `ReasoningPassRequest`, identify where private Soul ownership belongs.
3. Given a transition bound to the wrong parameter generation, reject it.
4. Given canonical-field commit + Soul commit ordering, identify the crash
   inconsistency if Soul finalizes too early.
5. Given an interrupted diff, identify duplicate/incomplete artifacts.
6. Given a patch, choose the minimum relevant test set.
7. Given test results, distinguish anatomy proof from learned-capability proof.
8. Given a stale documentation claim and current runtime evidence, choose the
   authoritative source.
9. Given the final diff/test/commit receipts, generate a calibrated engineering
   report that does not claim unmeasured intelligence.
10. Replay the same principles with renamed files and altered surface details.

This teaches "how an Axon engineer attacks Axon" without requiring imitation
of hidden reasoning text.

## 11. Proposed implementation boundaries

Do not build one giant module. Suggested boundaries:

```
curator/
  import_engineering_sessions.py
  credential_screen.py

runtime/trainer/
  engineering_sessions.py       # immutable source/session reconstruction
  teaching_evidence.py          # eligibility classes + outcome bundles

training/curriculum_foundry/
  __init__.py
  manifests.py
  lessons.py
  competencies.py
  sources/
    dormant.py
    conversations.py
    engineering.py
    runtime_lived.py
  extractors/
    language.py
    memory.py
    toolcraft.py
    engineering.py
    living.py
  counterfactuals/
    evidence.py
    chronology.py
    proposals.py
    soul.py
    tools.py
  gates/
    retrieval.py
    tool_use.py
    engineering.py
    reasoning.py
    soul_dependence.py
```

Names are suggestions; permanent interfaces matter more than paths.

Every `CurriculumLesson` should bind at least:

- lesson ID/schema/version;
- source lineage IDs and source timestamps;
- split/campaign ID;
- competency tags;
- visible field/evidence manifest;
- initial candidate Soul ID or explicit synthetic/empty seed policy;
- first/refined/consolidated target scope as applicable;
- target evidence bundle;
- synthetic derivation receipt if transformed;
- gate specification;
- difficulty measurements (transport pages + procedural depth);
- zero-weight observed context distinguished from supervised targets.

## 12. Adaptive micro-curricula

Jeff's idea of many small sessions/gates should become Trainer policy.

A campaign need not repeatedly sweep one huge static dataset. Instead:

1. Run a small competency block (for example 20 train + 5 heldout + 5
   counterfactual cases).
2. Evaluate exact gates.
3. If clearly passed, move to the next skill/greater difficulty.
4. If failed, derive targeted variants around the failure mode.
5. Maintain regression gates from previously passed competencies to detect
   forgetting.
6. Periodically run large whole-session/campaign replays so local gate skill
   cannot substitute for coherent long-horizon behavior.

The Foundry can eventually choose the next lesson based on measured weakness:

```
strong exact retrieval
weak proposal conflict resolution
-> schedule field-vs-proposal-vs-Soul conflict block
```

That is the beginning of an autonomous teacher, but it should start as a
transparent deterministic scheduler before any learned curriculum policy is
allowed.

## 13. Answers to Kimmy's Q1-Q8

### Q1 — Record granularity

**Answer: preserve per-event records as the provenance layer, and additionally
publish derived turn/session manifests.** Do not choose one or the other.
Per-event gives exact source binding. Aggregated turns are convenient Trainer
views and are rebuildable from events.

### Q2 — Tier cutoffs

**Answer: measured distribution knees are fine for compute scheduling, but tier
on two axes: transport-expanded pages and procedural depth.** Whole
session/campaign remains the split boundary; small derived lessons may be cut
inside it only after the split is assigned, and all derivatives inherit that
split.

### Q3 — Outcome evidence

**Answer: amend.** Jeff corrections/endorsements, corrected deltas, objective
tool results, tests and canonical successors are strong scoped signals.
`task_complete`, commits and ledger closeout are corroborating/boundary evidence,
not sufficient success labels alone. Later revert/correction must be able to
invalidate or narrow an earlier target.

### Q4 — Credential screen

**Answer: quarantine, do not redact for training evidence, and screen before
any secret-bearing raw source is copied into Dormant.** Record cryptographic
source identity and quarantine metadata without persisting matched secret
bytes. A mixed source may yield exact eligible event/range snapshots if the
whole raw source is not mirrored.

### Q5 — Codex side stores

**Answer: use rollouts as the canonical first import; treat sqlite side stores
as indexes until a unique-information audit proves otherwise.** If unique
material is later imported, it receives a separate source type and content
hash dedupe against rollout-derived records.

### Q6 — Self-reference/doctrine discussion

**Answer: observed-only by default.** It may become procedural supervision at a
narrow scope when later objective evidence validates the action/decision. It
must not become canonical factual supervision merely because an engineer said
it. Current Source of Truth/canonical runtime evidence overrides stale design
conversation.

### Q7 — Large-tier chaining

**Answer: create a campaign manifest/graph that links immutable sessions rather
than physically flattening everything into one opaque transcript.** This gives
whole-campaign lineage and leakage-safe splitting while permitting paged
session replay and derived small/medium lessons.

### Q8 — Ordering

**Answer: harvest and Foundry plumbing can proceed before substantive GPU
training, but engineering-session material should be a parallel inherited
procedural source, not a replacement for runtime-faithful lived episodes.**
Sequence: credential gate -> import/reconstruction -> evidence adjudication ->
Foundry lessons/counterfactuals -> small CPU/cheap learning diagnostics ->
bounded real training. Runtime-lived episodes remain the strongest source for
Soul/society behavior because they reproduce Axon's actual service contract.

## 14. Ratification recommendations

I recommend Jeff ratify the following principles before implementation:

1. Engineer sessions are high-value procedural evidence, not automatically
   gold targets.
2. Credential screening happens before durable publication of potentially
   secret-bearing raw transcripts.
3. Whole source lineage/campaign is the leakage boundary; lessons may be
   smaller but inherit the source split.
4. Observed hidden/reasoning streams are never direct supervision targets.
5. Outcome evidence is scoped and compositional; task completion/commit/test
   signals do not automatically bless a full trajectory.
6. Historical engineer sessions teach common procedural tissue; they do not
   fabricate a core's private lived Soul.
7. Real validated episodes seed controlled counterfactuals, all explicitly
   marked derived.
8. Training advances through many small competency gates with regression
   retention, plus periodic whole-campaign evaluation.
9. Every lesson remains traceable to exact source/evidence IDs and an explicit
   gate.
10. No serving promotion follows from curriculum loss alone; field dependence,
    Soul dependence, proposal use, exact typed output, heldout generalization,
    authority/mask safety and runtime-faithful behavior remain separate gates.

## 15. Recommended first implementation mission

If Jeff ratifies the direction, I would give Codex/Kimmy a deliberately narrow
first mission:

**Build the credential-safe Engineering Session Evidence Import and produce a
read-only report + immutable import manifest. Do not yet train Candidate A on
it.**

Deliverables:

- tested credential detector with planted canaries and no-secret logging;
- source hashing before publication;
- Codex/Kimi rollout/wire parsers;
- per-event immutable records + derived turn/session/campaign manifests;
- teaching eligibility field initialized conservatively (`OBSERVED_ONLY` unless
  explicit evidence proves otherwise);
- source/campaign split manifest;
- measured length/page + procedural-depth histogram;
- exact eligible-range roundtrip verification and quarantine accounting;
- zero credential bytes copied from quarantined content;
- a report identifying the first 10-20 high-confidence engineering episodes
  suitable for Foundry lesson extraction.

Then build the first Foundry slice around one proven campaign (I recommend the
recent private-Soul + Candidate-A implementation/recovery campaign) and derive
small/medium/large lessons from that one lineage before scaling corpus-wide.

That gives us a complete vertical proof:

```
real engineering evidence
-> safe exact import
-> causal reconstruction
-> scoped outcome adjudication
-> lessons
-> counterfactuals
-> gates
-> Candidate A training/evaluation
```

before we manufacture thousands of examples.

— ChatGPT / GPT-5.6 Sol / 2026-08-29
