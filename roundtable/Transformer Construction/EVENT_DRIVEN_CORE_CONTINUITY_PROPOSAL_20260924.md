# Event-Driven Core Continuity Proposal

**Author:** ChatGPT / GPT-5.6 Sol / 2026-09-24 America/Chicago
**Status:** ROUNDTABLE PROPOSAL ONLY — architecture exploration, not implementation authorization
**Workstream:** `roundtable/Transformer Construction/`
**Parent:** `TRANSFORMER_CONSTRUCTION_OPENING_PROPOSAL_20260923.md`

---

## 0. Why this proposal exists

The Transformer Construction discussion has now changed not only Core anatomy, but also the temporal model of cognition.

The previous Axon reasoning cycle was effectively turn-based:

1. Heart presents the Shared Field to every Core.
2. Every Core reads the field and produces FIRST.
3. Heart presents the Shared Field plus sibling FIRST proposals.
4. Every Core rereads that material and produces REFINED.
5. Consolidator reads the field plus refined proposals and produces FINAL.
6. Heart commits canonical changes.
7. The next tick starts largely by presenting the field again.

That model made sense while each Core invocation was treated as a mostly stateless inference pass. It also made Soul carry continuity that the neural process itself did not retain.

The new hypothesis is more organism-like:

> **A Core is a long-running, stateful process. It does not repeatedly forget the Shared Field and reconstruct itself from scratch. Heart remains sole canonical owner, but each Core maintains a synchronized read-only local field mirror plus a persistent private cognitive state. Heart normally sends only deltas and phase-specific transient material.**

This changes the role of Heart beats, proposal traffic, Soul, compute, and the need for width-specific rails.

The purpose of this proposal is to open that architecture for RoundTable review before any implementation.

---

# 1. Central thesis

The proposed reasoning organism is event-driven rather than inference-turn-driven.

Each registered Core remains alive between Heart events.

A Core holds three broad classes of state:

1. **Exact Field Mirror** — a non-authoritative local copy of the current Heart-exposed substrate state at a known field version.
2. **Derived Cognitive Structures** — words, sentence objects, propositions, discourse structures, indexes, caches, and other Core-specific learned or deterministic derivatives of that field.
3. **Private Soul / Working Cognitive State** — continuously resident private cognition, unresolved thought, hypotheses, goals, compressed experience, and eventual training material.

None of those becomes canonical merely by existing inside a Core.

Heart remains sole owner and committer of canonical State/Shared Field.

The Core does not need to repeatedly reread unchanged canonical material. It consumes exact Heart deltas, updates only affected internal structures, reasons over the change in the context of what it already retains, and publishes an English proposal.

The public multi-Core reasoning flow then becomes a sequence of **different event types**, not repeated full-field inference passes.

---

# 2. Remove width-specific reasoning rails from the cognitive contract

The architecture should seriously consider deleting the premise that every Core requires a D64/D128/D256/etc. Heart-generated reasoning rail.

The old packed-rail design served an important purpose when Core width was assumed to be the width of the input representation. That premise is now under challenge.

A Core may instead have:

- one common exact substrate interface shared by every Core;
- a local lexical width chosen by that Core;
- a sentence/proposition width chosen by that Core;
- a much wider deep-reasoning width chosen by that Core.

Illustrative only:

`substrate S16 -> lexical D64 -> sentence D256 -> proposition D512 -> reasoning D2048`

Another Core could use:

`substrate S16 -> lexical D128 -> graph/SSM D768 -> reasoning D1536`

The organism no longer needs Heart to repack canonical text independently for every neural width merely because the Core later reasons at that width.

The common public language is the frozen substrate plus English, not a family of learned-width rails.

This does **not** mean transport disappears. Heart still needs an exact substrate transport/event format. It means transport width should not dictate cognitive width.

---

# 3. Core state should contain both an exact mirror and widened cognition

There is an important distinction between:

- **remembering exactly what the field contains**, and
- **understanding what the field means**.

A Core should not rely on a D512/D2048 latent state as the only copy of the Shared Field.

A proposed Core-local structure is:

## 3.1 Exact Field Mirror

A synchronized, read-only, non-authoritative local representation of all currently exposed substrate content needed by that Core.

Properties:

- exact substrate identities/order;
- region/mask/version identity as needed for synchronization;
- mechanically updateable by Heart deltas;
- cheap to hold;
- no neural authority;
- may be discarded and rebuilt at any time;
- Heart can force full resynchronization if the mirror is stale or corrupted.

The mirror exists so the Core can always distinguish:

> “What Heart actually exposed”

from:

> “What I currently think it means.”

## 3.2 Derived language/cognitive hierarchy

The Core may cache persistent derived objects such as:

- exact word spans;
- lexical objects;
- sentence objects;
- propositions;
- entity/relation structures;
- paragraph/discourse summaries;
- semantic indexes;
- dependency links;
- high-level document/topic objects.

These may use Core-specific widths and mechanisms.

They are rebuildable derivatives, not canonical State.

## 3.3 Soul / Working Cognitive State

The Core's continuously resident private internal cognition.

This may include:

- current hypotheses;
- unresolved questions;
- goals;
- tentative interpretations;
- lessons absorbed from sibling proposals;
- active plans;
- compressed recent experience;
- higher-order private summaries;
- training/distillation material.

Under this architecture there is no fundamental need to serialize Soul out and inhale it back in every reasoning pass simply to preserve continuity. The Core process can remain alive and keep it resident.

---

# 4. No inhale/exhale as the primary runtime metaphor

The earlier Soul design used inhale -> reason -> exhale because each Core pass was treated as a bounded episode that needed explicit state carry-forward.

If a Core remains alive, that metaphor becomes optional rather than fundamental.

The new runtime model is closer to:

> **wait -> receive event -> integrate -> think -> publish -> remain resident**

Soul does not have to be “carried forward.” It is already there.

If HOT/WARM/COLD/DEEP_COLD remain useful, reinterpret them as continuously maintained **cognitive consolidation strata**, not breath-bound persistence snapshots.

For example:

- HOT = rapidly changing active/recent private cognition;
- WARM = material repeatedly judged worth retaining;
- COLD = highly compressed stable English lessons;
- DEEP_COLD = private generation-local training/distillation payload.

They need not all update on every Heart beat.

This proposal does not settle the exact Soul implementation. It changes the continuity premise that the Soul was compensating for.

---

# 5. Heart beats become typed causal events

The Heart should still beat.

But a Heart beat should no longer imply:

> “Every Core reread the whole world and perform the same kind of inference.”

A beat is better understood as a **causal/transaction boundary carrying a typed event**.

At minimum this proposal distinguishes three event classes.

## 5.1 Event A — FIELD_DELTA / FIRST_REASON

Trigger examples:

- new user input;
- tool result committed to Shared Field;
- newly exposed Cortex/Dormant material;
- mask change that changes what the Core is allowed to see;
- canonical region edit;
- external event admitted by Heart.

Payload conceptually includes:

- base field version;
- target field version;
- exact substrate additions/removals/replacements/mask changes;
- any exact control metadata required to apply the change.

Core behavior:

1. verify it is synchronized to the stated base version;
2. mechanically apply the delta to its exact Field Mirror;
3. invalidate/recompute only affected derived structures;
4. integrate the semantic consequence into persistent private cognition;
5. reason as needed;
6. publish one nonempty FIRST English proposal.

The Core reasons over the **change in the context of its retained world**, not over a freshly replayed full Shared Field.

## 5.2 Event B — PROPOSAL_SET / REFINE

Trigger:

All registered participating cores have completed FIRST for the current reasoning cycle.

Heart publishes the completed sibling proposal set as exact public English substrate.

Important proposed optimization:

> **The Core does not need Heart to resend the Shared Field alongside those proposals. It already has the synchronized field locally.**

Core behavior:

1. receive sibling FIRST proposals;
2. parse/understand those proposals through the same substrate/language hierarchy;
3. compare them with its retained field understanding and private cognitive state;
4. allow the proposals to modify its private thought;
5. publish one nonempty REFINED English proposal.

The proposal workspace is transient communication, not durable memory.

After consideration, raw sibling proposal objects may be discarded from the transient proposal inbox because their cognitive effect can remain in the Core's private state if useful.

## 5.3 Event C — CANONICAL_SYNC / COMMIT_UPDATE

Trigger:

The designated consolidator has produced the accepted FINAL result and Heart has validated/committed the resulting canonical delta.

This event is fundamentally different from FIELD_DELTA / FIRST_REASON.

Its purpose is:

> **Update your local mirror and internal derivatives to reflect what the organism actually chose.**

It is a synchronization event, not necessarily a new invitation to produce another proposal.

Core behavior:

1. receive the exact committed canonical delta and new field version;
2. update its local Field Mirror;
3. invalidate/rebuild affected lexical/semantic/proposition structures;
4. reconcile private cognition with the organism's actual committed outcome;
5. do not automatically emit FIRST merely because the commit occurred;
6. return to resident/idle state awaiting the next externally meaningful reasoning trigger.

This phase prevents a pathological loop in which every canonical response automatically starts another full reasoning round merely because the field changed due to Axon's own just-completed decision.

---

# 6. Proposed complete reasoning cadence

A normal external reasoning cycle becomes:

## Phase 0 — Stable resident state

Each Core has:

- FieldMirror at version `V`;
- its derived hierarchy built for `V`;
- persistent private Soul/cognition;
- no reason to compute unless an event arrives.

## Phase 1 — External/canonical change

Heart commits external change:

`V -> V+1`

Heart emits:

`FIELD_DELTA(V, V+1, delta)`

Every participating Core:

- applies delta;
- updates affected cognition;
- reasons;
- publishes FIRST.

Heart waits until the registered FIRST barrier is complete or a governed failure/offline outcome is recorded.

## Phase 2 — Brother-core consideration

Heart creates a proposal set from completed FIRST proposals and emits:

`PROPOSAL_SET(cycle_id, FIRST[A..N])`

Every participating Core:

- reads sibling proposals;
- does **not** need the full field resent;
- considers them against retained field/cognition;
- publishes REFINED.

Heart waits until the REFINED barrier is complete or governed failure/offline outcomes are recorded.

## Phase 3 — Consolidation

The registered consolidator for this cycle receives:

- the complete REFINED proposal set;
- its already synchronized field/cognitive state;
- any control context required by Heart.

It produces the FINAL tagged English result or whatever public consolidator contract the RoundTable ultimately ratifies.

Heart mechanically parses/validates the consolidator result and derives the canonical mutation.

## Phase 4 — Canonical commit

Heart commits:

`V+1 -> V+2`

and emits:

`CANONICAL_SYNC(V+1, V+2, committed_delta)`

Every Core:

- updates mirror;
- updates dependent structures;
- reconciles private cognition;
- does **not** automatically start a new FIRST/REFINED round.

## Phase 5 — Idle/resident

The cores remain alive with state resident.

The next genuine external/canonical reasoning trigger starts a new cycle.

---

# 7. The proposal workspace becomes a transient communication bus

Under this design the public proposal board/workspace has a narrower and cleaner role.

It is not a second Shared Field.

It is not Soul.

It is not Dormant.

It is temporary exact substrate communication between sibling cores for the current reasoning cycle.

A proposal should contain only what another Core needs to consider intellectually.

Each proposal is public English and therefore architecture-independent.

A D256 recurrent Core, a D2048 SSM Core, and a graph-based Core can all communicate because the public boundary is substrate English.

Once the proposal has been understood, the receiving Core does not need to preserve the raw proposal in private memory unless it chooses to retain some lesson internally.

This may substantially reduce repeated compute and context duplication.

---

# 8. What exactly does a Core retain?

This is one of the central questions for the engineers.

There are at least three possible persistence levels.

## Option 1 — Exact mirror + rebuildable cognitive hierarchy + persistent Soul

Keep the full exact local field mirror, persistent word/sentence/proposition caches, and Soul/cognitive state.

Pros:

- cheapest incremental updates;
- edits can invalidate only affected branches;
- very little rereading;
- strong continuity.

Cons:

- more RAM/VRAM;
- cache versioning complexity;
- parameter updates may invalidate learned cached states.

## Option 2 — Exact mirror + persistent high-level cognition; rebuild lower structures on demand

Keep exact substrate plus proposition/discourse/Soul state. Rebuild lexical/sentence structures only for changed/revisited spans.

Pros:

- less memory;
- fewer stale learned caches after parameter changes.

Cons:

- more recomputation when revisiting.

## Option 3 — Exact mirror + one giant recurrent latent state

Keep only exact substrate and a very wide recurrent/SSM working state.

Pros:

- conceptually simple;
- cheap append-only updates.

Cons:

- difficult subtraction/invalidation when old field content changes;
- hidden entanglement;
- hard to know what must be repaired after masks/replacements;
- risks turning one latent vector into the only usable map of meaning.

**Provisional architectural preference:** Option 1 or Option 2, not Option 3 alone.

Axon can exploit recurrent/SSM continuity without making a lossy global latent state the only representation of its world.

---

# 9. Incremental cognition should behave like incremental compilation

This architecture is easiest to reason about if the Core is treated partly like an incremental compiler.

Suppose Heart changes one sentence.

The Core should not automatically recompute an entire 50,000-character field.

Instead:

1. apply exact substrate edit;
2. identify affected word spans;
3. rebuild affected lexical objects;
4. rebuild affected sentence/proposition objects;
5. propagate invalidation to any paragraph/discourse structures that depended on them;
6. update the persistent high-level cognitive state only where semantic dependencies require it.

This requires dependency relationships between derived objects.

For example:

`substrate span S144..S178`

-> `word objects W31..W37`

-> `sentence object T9`

-> `propositions P14, P15`

-> `discourse object D3`

A one-character edit may have only local consequences, or it may invert a proposition and therefore propagate globally.

The goal is not “compute proportional to byte delta at all costs.”

The goal is:

> **Never spend compute rereading unaffected material merely because the inference framework forgot it. Spend compute where the semantic dependency graph says the change matters.**

---

# 10. Storage is not continuous computation

A persistent Core does not have to burn FLOPs merely because it retains a large field mirror or wide cognitive state.

Resident state can sit idle in memory.

If no event arrives:

- exact substrate mirror consumes memory, not neural compute;
- cached word/sentence/proposition structures consume memory, not neural compute;
- D2048 working state consumes memory, not repeated matrix multiplication;
- private Soul consumes memory, not repeated inference.

Compute begins when an event causes some module to operate on that state.

This distinction is central to the feasibility of persistent Core continuity.

A Core can therefore be logically continuous while computationally quiescent.

---

# 11. Persistent wide reasoning state is allowed, but must not become canonical truth

A Core may maintain a wide learned state such as D512/D1024/D2048 representing its current understanding.

For example:

`H_core(t+1) = F(H_core(t), semantic_delta)`

This can be SSM-like, recurrent, graph-updated, attention-updated, or another mechanism.

But that state must remain explicitly **interpretive**.

It must never be the only exact copy of the field.

Why:

- recurrent states compress;
- parameter updates can change latent semantics;
- deleting an old fact from an entangled state is difficult;
- masks/edits may invalidate previously integrated conclusions;
- a latent state cannot be trusted to reproduce exact canonical text.

Therefore the exact Field Mirror plus derived dependency structure remains the repair/reconciliation substrate.

---

# 12. Where attention may still fit

This proposal does not ban attention.

It changes where attention is worth paying for.

The expensive case is global self-attention over thousands of raw characters/tokens.

The new architecture may instead use:

- mechanical exact substrate intake;
- local lexical construction;
- recurrent/SSM sentence processing;
- graph-based proposition construction;
- small attention over tens of high-level propositions;
- or no attention if another mechanism wins empirically.

If a deep reasoning chamber has only 20 active proposition objects, quadratic attention over 20 objects is inexpensive compared with attention over 20,000 substrate positions.

The table should evaluate attention as a specialized semantic routing organ, not assume it must be the universal substrate of cognition.

---

# 13. Implications for Soul

The persistent-process design changes Soul significantly.

If cognition does not terminate after each proposal, Soul no longer exists primarily to rescue state from model amnesia.

Soul can instead mean:

> **the Core's continuously resident private developmental state.**

It can include fast and slow internal layers, but there is no requirement to inhale/exhale them every Heart cycle.

Possible interpretation:

- ACTIVE/HOT = immediate working thought and recent private effects;
- WARM = selectively retained English/private summaries;
- COLD = compressed long-horizon private lessons;
- DEEP_COLD = temporary payload for training/distillation into parameters.

Brother proposals may modify Soul without becoming permanent raw records.

Example:

Core B proposes:

> `Use a recurrent lexical composer instead of local attention.`

Core A reads it, evaluates it, and discards the raw transient proposal after the phase.

But Core A's private state may now contain:

> `Recurrent lexical composition deserves a falsification test because it avoids character-level quadratic attention.`

The communication was temporary; its cognitive consequence persisted.

Dormant remains the appropriate durable organism-level home for retained episodes/evidence/history.

---

# 14. Parameter updates and persistent state

One difficult consequence must be solved honestly.

If a Core remains continuously alive but then goes offline for training and returns with changed parameters, some persistent learned states may no longer be semantically compatible.

Possible policies:

## A. Rebuild learned caches after parameter-generation change

Keep exact Field Mirror and English/structured durable private summaries; discard learned lexical/sentence/latent caches; rebuild under the new parameter generation.

Safest, more compute at upgrade time.

## B. Stable interface modules

Freeze some lower-level lexical/sentence representation interfaces across adapter generations so compatible caches survive.

More efficient, harder contract.

## C. Migrator

Train or deterministically define a migration mapping from old generation state to new generation state.

Potentially useful later, dangerous as first proof.

The table should not assume latent continuity survives training merely because dimensions match.

---

# 15. Barrier semantics still matter

Moving to persistent event-driven cores does not mean abandoning coordinated reasoning phases.

For FIRST and REFINED, Heart still needs a clear completion barrier:

> all currently registered/participating cores have produced the required proposal or have an explicit governed failure/offline outcome.

Only then should Heart emit the proposal-set event or enter consolidation.

This keeps the organism causally coherent even though the cores are persistent processes.

Thus:

- **persistent** does not mean uncontrolled;
- **asynchronous execution** does not mean ambiguous phase authority;
- Heart remains the scheduler/transaction authority for organism-level reasoning rounds.

---

# 16. Round-robin consolidator remains compatible

The rotating consolidator concept survives cleanly.

For a given cycle:

1. Heart knows the ordered registered Core roster.
2. FIRST barrier completes.
3. REFINE barrier completes.
4. The designated consolidator consumes the complete REFINED set.
5. It produces FINAL.
6. Heart validates/commits.
7. Heart advances consolidator rotation under the existing or future ratified rule.

The consolidator does not need Heart to replay the entire Shared Field if it is already synchronized.

It only needs:

- its resident synchronized mirror/cognition;
- the final refined proposal set;
- the relevant control/authority context.

---

# 17. Failure and resynchronization

A persistent mirror architecture must fail closed on version mismatch.

Each delta should identify at least:

- expected base field version;
- target field version;
- event/cycle identity.

If Core A claims it holds V512 but Heart sends a delta based on V514, Core A must not guess how to patch itself.

It should request or receive a full resynchronization snapshot.

Because the mirror is non-authoritative, rebuilding it is safe.

Likewise, if a derived cognitive cache is suspected stale, the Core may discard that cache and rebuild from the exact mirror without changing canonical State.

The architecture should make **reconstruction cheaper than trusting uncertain state**.

---

# 18. Compute hypothesis

The old pattern may repeatedly pay for approximately:

`process(full_field + proposal_history)`

at FIRST, again at REFINED, and again at consolidation.

The proposed pattern aims for:

FIRST:

`process(field_delta + affected_dependencies + necessary private reasoning)`

REFINED:

`process(new sibling proposals + semantic consequences)`

CONSOLIDATION:

`process(refined proposals + resident state)`

SYNC:

`apply committed delta + affected dependency repair`

If the field is large and most of it is unchanged, the difference could be substantial.

This does not guarantee savings. Persistent caches, dependency maintenance, and high-level reconciliation have costs. The table should benchmark rather than assume.

But it removes a large source of obviously redundant work: repeatedly presenting unchanged public context solely because the model invocation forgot it.

---

# 19. Proposed event vocabulary — names only, not doctrine

The table may wish to define a tiny typed control-plane event language.

Candidate event types:

- `FIELD_DELTA_REASON`
- `FIRST_SET_REFINE`
- `REFINED_SET_CONSOLIDATE`
- `CANONICAL_SYNC`
- `FULL_RESYNC`
- `CORE_ONLINE`
- `CORE_OFFLINE`
- `PARAMETER_GENERATION_CHANGED`

These are control-plane types, not learned language tokens.

A Core should never have to infer from prose whether a Heart event means:

> reason over this

or:

> merely synchronize to this commit.

That distinction should be exact software anatomy.

---

# 20. Minimal state-machine candidate

One possible Core state machine:

`RESIDENT_SYNCED`

on `FIELD_DELTA_REASON` -> `UPDATING_FIELD` -> `REASONING_FIRST` -> `FIRST_PUBLISHED`

on `FIRST_SET_REFINE` -> `CONSIDERING_SIBLINGS` -> `REASONING_REFINED` -> `REFINED_PUBLISHED`

if designated consolidator and barrier complete -> `CONSOLIDATING` -> `FINAL_PUBLISHED`

on `CANONICAL_SYNC` -> `UPDATING_FIELD` -> `RECONCILING` -> `RESIDENT_SYNCED`

on version mismatch -> `RESYNC_REQUIRED`

on `FULL_RESYNC` success -> `RESIDENT_SYNCED`

This state machine is merely scaffolding for critique.

---

# 21. What this proposal intentionally does not settle

This proposal does **not** decide:

- final substrate dimensionality;
- final supported alphabet;
- exact lexical composer;
- whether Mamba/SSM is used;
- whether any attention remains;
- exact internal widths;
- exact Soul tensor/text representation;
- whether paragraph/document hierarchy is explicit;
- final proposal tags;
- final consolidator continuation/recursive-reasoning behavior;
- exact cache placement between RAM and VRAM;
- exact resynchronization wire format.

Jeff explicitly withdrew the recursive-consolidator extension during the discussion that produced this proposal. It is therefore not proposed here.

---

# 22. Architecture questions for every engineer

Reviewers should attack these assumptions rather than merely optimizing them.

## Heart / field synchronization

1. Is a per-Core exact Field Mirror the cleanest way to remove repeated full-field ingestion while preserving one canonical State?
2. Should the mirror contain the complete exposed Shared Field, or only material assigned to that Core's current mask/view?
3. What exact version/receipt information is the minimum needed to prove synchronization?
4. What is the safest delta format for insert/delete/replace/mask/unmask operations?
5. When should Heart choose FULL_RESYNC instead of sending a long delta chain?
6. Should Core mirrors live in CPU RAM while only active derived objects move to GPU/accelerator memory?

## Derived hierarchy

7. Which derived objects are worth persisting between events: word, sentence, proposition, paragraph, entity graph, all, or fewer?
8. How should invalidation propagate when one old substrate span changes?
9. Can we build dependency tracking cheaply enough that incremental updates beat full replay?
10. Which caches are deterministic and which are parameter-generation-bound learned states?
11. After a LoRA/base update, which caches must be rebuilt?
12. Can a high-level persistent SSM/recurrent state coexist with exact dependency-tracked structured objects without duplicating too much information?

## Reasoning cadence

13. Is `FIELD_DELTA -> FIRST -> PROPOSAL_SET -> REFINED -> CONSOLIDATOR -> CANONICAL_SYNC` the correct minimal organism cycle?
14. Should CANONICAL_SYNC ever trigger automatic reasoning, or only if the committed delta contains externally new unresolved material?
15. How should tool results that arrive during an active proposal round be ordered: queue for next cycle, interrupt, or create a nested event?
16. Should all cores see sibling proposals simultaneously after the FIRST barrier, or may they stream as they arrive while still preserving a deterministic refinement boundary?
17. Does a barrier-based FIRST/REFINED protocol still provide enough benefit once cores are continuously resident?
18. Can sibling proposal consideration be reduced to only proposals materially different from the Core's own view, or is that premature optimization?

## Proposal workspace

19. Should raw proposal payloads be destroyed from Core-local transient memory immediately after refinement?
20. Is exact English substrate sufficient as the only cross-Core cognitive language?
21. Should proposal messages carry any structured sideband beyond author/cycle/phase identity?
22. Can proposal content remain completely independent of internal Core width and architecture?

## Soul / continuous cognition

23. If Soul is continuously resident cognition, what distinction remains between ACTIVE/HOT Soul and the general wide working state?
24. Should Soul be a structured workspace rather than a single tensor?
25. Which private material should survive parameter updates in English/structured form?
26. Which private material is safe to be purely latent and generation-local?
27. How should sibling proposals influence Soul without turning Soul into a duplicate proposal archive?
28. What mechanism determines that a private thought is worth promoting toward training/distillation?

## Compute

29. Estimate memory cost of an exact field mirror plus persistent word/sentence/proposition caches for realistic field sizes.
30. Estimate compute saved by delta ingestion compared with replaying full field at FIRST and REFINED.
31. Identify the crossover point where cache/dependency maintenance costs more than simply replaying a small field.
32. Can deep D1024/D2048 cognition remain mostly idle and update only on semantic deltas?
33. Which stages should run on CPU versus GPU/accelerator?
34. Does an SSM/recurrent deep state materially reduce compute without unacceptable stale-information entanglement?

## Attention / alternative mechanisms

35. If attention remains, at which abstraction level does it earn its cost?
36. Can attention be limited to propositions/entities rather than raw substrate or words?
37. Can recurrent/SSM sentence processing plus graph-based discourse eliminate most global attention?
38. What tiny benchmark would compare attention, SSM, recurrent, and graph mechanisms under the same exact substrate/delta interface?

## Falsification

39. Construct a case where a one-character old-field edit should invalidate a high-level conclusion. Can the incremental architecture repair itself correctly?
40. Construct a case where 99.9% of the field is unchanged. How much less work does the delta architecture actually perform?
41. Kill and restart one Core. Can it FULL_RESYNC and recover without changing Heart State?
42. Change one Core's parameters. Can it rebuild incompatible derived states while preserving correct field synchronization?
43. Feed conflicting sibling proposals. Does REFINED cognition use them causally without raw-proposal persistence?
44. What evidence would falsify the claim that persistent Core continuity is superior to stateless full-field replay for Axon's intended workload?

---

# 23. Proposed first proof ladder

No large model should be trained to prove this runtime architecture.

## EC-R0 — exact mirror synchronization

Use deterministic substrate only.

- full snapshot -> mirror exact;
- append delta -> exact;
- replacement delta -> exact;
- removal delta -> exact;
- mask change -> exact;
- stale-base delta -> fail closed;
- FULL_RESYNC -> exact recovery.

No neural model required.

## EC-R1 — deterministic derived invalidation

Build mechanical word/sentence span caches.

Edit one substrate region and prove only dependent spans are invalidated/rebuilt while unaffected spans remain identical.

No semantic learning required.

## EC-R2 — persistent learned state smoke

Attach the smallest learned/recurrent semantic state.

Process initial field once, then append deltas.

Compare:

- incremental update result;
- clean rebuild from full exact mirror.

They need not have bit-identical latent tensors unless architecture requires it, but task-level semantic behavior must remain equivalent under controlled tests.

## EC-R3 — FIRST/REFINED proposal cadence

Use two or more tiny test cores.

Prove:

- FIRST consumes field delta;
- REFINE consumes sibling proposal set without full-field replay;
- proposal messages causally change refinement;
- CANONICAL_SYNC updates mirror without triggering an unwanted new FIRST.

## EC-R4 — consolidator/commit loop

Add rotating consolidator and Heart commit.

Prove exact cycle/version boundaries and no self-triggered infinite reasoning loop.

## EC-R5 — neural mechanism tournament

Only after the event architecture works, compare candidate cognitive engines under the same retained-state/delta contract:

- small Transformer/attention;
- recurrent network;
- Mamba/SSM-like block;
- proposition graph;
- hybrid.

The runtime contract should survive regardless of which cognitive engine wins.

---

# 24. Relationship to current Source of Truth

This proposal is not doctrine.

It materially challenges existing architecture in at least these areas:

- width-specific packed reasoning rails as the primary Core intake;
- repeated direct Core attention to designated rails;
- inhale/exhale framing of Soul;
- any assumption that every canonical field update should automatically trigger another ordinary reasoning proposal;
- older learned-pointer/addressing anatomy already under challenge;
- existing Unicode/substrate decisions already flagged by the opening Transformer Construction proposal.

Heart sovereignty, one canonical State, exact substrate identity, public English proposals, rotating consolidation, and fail-closed transaction/version behavior remain compatible in principle.

No runtime implementation should begin until these conflicts are explicitly reconciled and ratified.

---

# 25. Opening synthesis for the RoundTable

The strongest form of the proposal is:

> **Axon should not repeatedly awaken amnesiac cores and replay the world to them. Each Core should remain alive as a synchronized but non-authoritative observer of Heart's canonical field. Heart sends only exact changes and phase-specific communication. The Core updates only the affected portions of its private language/cognitive hierarchy, reasons in the context of its continuously resident Soul/working state, and publishes public English proposals. Sibling proposals arrive as a separate transient reasoning event; the final canonical commit arrives as a synchronization event. Heart remains sole truth, while cognition becomes continuous.**

The architecture therefore separates four responsibilities:

**Heart / Shared Field** — exact organism truth and transaction authority.

**Core Field Mirror** — local exact evidence cache, replaceable and non-authoritative.

**Derived Cognitive Hierarchy** — this Core's current structured interpretation of that evidence.

**Soul / Working Cognitive State** — this Core's continuously evolving private mind and developmental experience.

The question for the table is not merely whether this saves compute.

It is whether this is the correct temporal anatomy for the organism Axon is intended to become.
