# Core Architecture - Opening RoundTable Brief

**Author:** ChatGPT / GPT-5.6 Sol / 2026-09-25 America/Chicago
**Status:** OPENING ROUNDTABLE PROPOSAL ONLY - architecture exploration, not implementation authorization
**Workstream:** `roundtable/Core Architecture/`
**Authority:** Jeff, project convener
**Control baseline:** fresh Axon D512 continuous Core (`exact D16 -> Linear(16,512) -> GRUCell(512,512) -> categorical exact transport + private EOS`)
**Relationship to prior work:** previous Transformer Construction, RNSC, Soul, rail, pointer, and training proposals remain historical/evidentiary material. This workstream is a clean architecture table. Old proposals are not active architecture commitments here unless deliberately reintroduced and defended against current evidence.

---

## 0. Why open a new table now

Axon has crossed an important boundary.

For a long time, Core design was dominated by transport and motor problems: packed D64 rails, learned pointer selection, learned routing, learned copy/generate behavior, termination heads, and architecture-specific serving widths. Those experiments were valuable because they exposed where learned machinery was being asked to rediscover structure that Axon already possessed exactly.

The current system has removed much of that burden from cognition:

- Heart remains sole canonical owner/writer;
- Heart serves exact D16 truth;
- each continuous Core can maintain an exact non-authoritative local mirror;
- mirror coherence is checked explicitly;
- public transport width no longer has to match private Core width;
- exact positions and transport identities do not need a learned pointer motor;
- categorical Core output maps mechanically back to exact registered D16 cells;
- legacy D64 tissue remains preserved without being mandatory for the continuous-Core family.

B3 proved that a registered `d_model=512` Core can complete live FIRST -> REFINED -> FINAL -> Heart commit -> `CANONICAL_SYNC` without any D512 rail existing in the frozen tick.

B4 then introduced a fresh one-chamber D512 recurrent baseline on Axon's real registered substrate. In the first substantial 700-step local tranche, the baseline moved held-out teacher-forced content accuracy from approximately 0.16% to 28.17%, against a strongest content-only constant floor of approximately 2.35%. Free-running exact sequence accuracy moved from 0% to 2.5%; termination reached 100%; valid Unicode output reached 100%; held-out mean loss fell from approximately 5.87 to 2.36. Peak PyTorch CUDA allocation was approximately 75 MiB on the local GTX 1650.

Those numbers do not establish language, reasoning, conversation, or final architecture. They establish something more useful for this table:

> **We now possess a small, trainable, measurable control Core whose failures are interpretable enough to support real architecture experiments.**

The question is no longer merely "how do we make symbols move?"

The question is:

> **What anatomy should an efficient Axon thinker have above an exact D16 body?**

This table exists to answer that question slowly, experimentally, and without forcing Axon back into conventional LLM assumptions.

---

## 1. Clear-the-desk rule

This folder is a fresh architecture workstream.

Prior proposals are not deleted, disowned, or rewritten. They remain evidence and design history. But they do not automatically occupy the new table.

A prior idea may return only if someone can state:

1. what cognitive job it performs;
2. why deterministic body machinery cannot perform that job better;
3. what state it owns;
4. what information it consumes;
5. what information it emits;
6. how it is trained;
7. what falsifiable experiment distinguishes it from the current control Core;
8. what failure would cause us to remove it again.

This is especially important for attention, Transformer blocks, SSM/Mamba layers, lexical hierarchies, graph memory, additional chambers, Soul mechanisms, and wider internal dimensions.

No component earns a place because it is fashionable, standard, elegant, or theoretically powerful.

It earns a place because it performs a necessary cognitive function measurably better than a simpler alternative.

---

## 2. Foundational separation: truth, interpretation, and thought

The new table should begin from a three-way separation.

### 2.1 Exact truth

The exact field mirror is **not cognition**.

It is deterministic evidence owned canonically by Heart and mirrored exactly by the Core under Heart coherence rules.

It may be large. It may contain conversation, tool results, scratch regions, response drafts, repository material, memories exposed by policy, or other permitted Shared Field regions.

Its job is to preserve exact information and provenance.

It should not be forced into one D512 vector.

### 2.2 Interpretation

A learned Core needs a way to understand what the exact field currently means.

This includes questions such as:

- what changed;
- what region changed;
- what the changed material refers to;
- which older material it modifies, contradicts, or completes;
- what appears relevant to the current event;
- what evidence should remain reachable by exact handles;
- what information is stale;
- what current request or goal is active.

This is a learned **Field Interpretation** problem.

### 2.3 Thought

The Core also needs private learned state that is not a copy of the field and is not required to remain externally interpretable at every instant.

This state may contain:

- unresolved alternatives;
- partial inferences;
- active goals;
- recent conclusions;
- uncertainty;
- consequences of earlier internal computation;
- residue from prior events;
- compressed private experience;
- intermediate reasoning that is not yet ready to become a public proposal.

This is **Deliberation**.

The central opening hypothesis is therefore:

> **Exact mirror != field interpretation != deliberative cognition.**

Those three tissues should not be conflated merely because a conventional model commonly represents all context through one token stream.

---

## 3. Opening anatomy hypothesis

The opening architecture to debate is not a ratified build. It is a functional decomposition:

```text
                         HEART
                           |
                    exact D16 event
                           |
                           v
                  EXACT FIELD MIRROR
              deterministic / non-learned
                           |
                           v
                    FIELD INTERPRETER
              persistent learned field model
            relations / relevance / invalidation
               + exact handles into mirror
                           |
                   focus / event packet
                           |
                           v
                  DELIBERATION CHAMBER
              persistent volatile cognition
                           |
                 internal cognitive steps
                    <-> <-> <->
                           |
                           v
                CRYSTALLIZED PROPOSITION
                           |
                           v
                  LANGUAGE / ACTION MOTOR
                           |
                   exact D16 categories
                           |
                           v
                         HEART
```

The names are placeholders. The functional boundaries are the subject of review.

---

## 4. The exact mirror should remain mechanical

The first temptation in a multi-chamber design is to say that the first neural chamber "holds the Shared Field."

This table should challenge that wording.

The Shared Field already has an exact holder: the Core's D16 mirror.

A learned chamber should not spend capacity reproducing what the body can preserve losslessly.

Therefore the stronger opening design is:

> **The mirror holds the field. The first learned chamber holds an evolving interpretation of the field.**

That distinction allows the mirror to scale independently from the neural state.

A repository can contain millions of characters without requiring a million-character latent vector to remain continuously activated.

The learned problem becomes:

> Which exact material matters now, how does it relate to the current situation, and what should deeper cognition receive?

That is a much more useful training target than "memorize every byte in D512."

---

## 5. Chamber One candidate: Field Interpreter

### 5.1 Proposed job

The Field Interpreter stays close to external truth.

Its job is to maintain a learned, persistent understanding of the Core's permitted field regions as exact deltas arrive.

It should learn things such as:

- region semantics;
- recency and supersession;
- correction and negation;
- entity/reference continuity;
- local discourse structure;
- task relevance;
- relationships across regions;
- evidence provenance;
- when a change invalidates previous interpretation;
- when exact mirror material should be revisited;
- what current event requires cognition.

### 5.2 What it should not do

It should not be responsible for:

- exact character identity;
- exact address reconstruction;
- canonical truth ownership;
- learned D16 regeneration;
- pretending that its latent state is an exact archive;
- producing a final user answer merely because it noticed the event.

### 5.3 Candidate output

Its downstream output may be an **event/focus packet** containing learned state plus deterministic references such as exact span/region handles.

Conceptually:

```text
current intent: answer ownership question
changed evidence: tool-result span 408..462
related prior claim: conversation-history span 119..146
relationship: later evidence contradicts prior claim
confidence/status: current ownership unresolved
relevant exact handles: [...]
learned interpretation state: <private tensor>
```

The exact schema is open.

The important principle is that deeper reasoning should be able to receive a compact learned interpretation **without losing its path back to exact evidence**.

---

## 6. Chamber Two candidate: Deliberation Chamber

### 6.1 Proposed job

The Deliberation Chamber is not a second copy of the Shared Field.

It is persistent private cognitive state.

It receives interpreted deposits from Chamber One and mixes them into whatever cognitive residue is already present.

It should be allowed to retain useful previous state, overwrite stale state, let unimportant state decay, reconsider earlier internal conclusions, and eventually produce explicit propositions.

### 6.2 The "churning soup" hypothesis

Jeff's opening metaphor is useful and should be preserved as an engineering intuition:

> New interpreted evidence enters a continuously evolving private cognitive soup. Older cognitive residue may persist, decay, combine, or be overwritten. Recurrent internal computation churns this state until some useful structure becomes stable enough to be harvested as a proposition.

This maps naturally to recurrent and state-space mechanisms.

A GRU already implements learned retention/overwrite gates.

A selective SSM/Mamba-like mechanism may offer richer long-running state dynamics and linear sequence processing.

Neither is assumed correct.

The tournament must decide.

### 6.3 Deliberation is not durability

The Deliberation Chamber should initially be treated as volatile learned cognition.

Its state is not canonical truth.

It is not a substitute for Dormant.

It is not automatically crash-safe Soul.

Parameter updates may invalidate some or all of its latent coordinates.

Durability policy is a separate doctrine question and must not be smuggled into architecture through implementation convenience.

---

## 7. Two clocks: organism time and cognitive time

This may be the most important new opening question.

Axon already has an organism-level event cadence controlled by Heart.

A field delta, reasoning round, proposal set, FINAL, commit, and `CANONICAL_SYNC` belong to **organism time**.

But a Core may need more than one learned state transition to think deeply about one event.

That suggests a second timescale:

### Organism time

```text
field delta -> reasoning activation -> proposal -> Heart commit
```

### Cognitive time

```text
interpret -> ponder -> revise -> integrate -> reconsider -> crystallize
```

A single Heart event may permit K private cognitive microsteps before the Core emits FIRST.

Likewise, after receiving sibling proposals, a Core may perform another bounded private deliberation sequence before emitting REFINED.

This gives a concrete experimental meaning to "thinking longer."

It does **not** require replaying the entire prompt K times.

The resident cognitive state evolves in place.

### Opening experimental question

For the same weights, same field, same task, and same exact evidence:

- K=1 deliberation step;
- K=2;
- K=4;
- K=8;
- K=16.

Does task quality improve? Plateau? Degrade? Oscillate?

If more private recurrence improves reasoning, Axon has evidence for genuine internal cognitive time.

If it does not, extra pondering should not be romanticized.

---

## 8. Crystallization: from latent soup to explicit fruit

The Deliberation Chamber's entire latent state should not need to become public.

Instead, useful internal structure may be **crystallized** into an explicit proposition.

Examples:

- a candidate answer claim;
- a plan step;
- a contradiction;
- a question that must be resolved;
- an internal lesson;
- a scratch observation;
- a tool request;
- a response-draft fragment.

The opening metaphor is:

> **volatile cognition is the soup; a stable proposition is the fruit.**

A proposition is valuable precisely because it compresses work that may have required many private recurrent transitions.

The proposition must remain distinguishable from canonical truth. Heart still decides what is committed.

---

## 9. Attention is no longer the default answer

This table does not ban attention forever.

It does change the burden of proof.

Attention previously carried responsibilities that are now owned elsewhere:

- exact substrate identity;
- position recovery;
- broad access to context;
- routing across a flat token stream.

With an exact mirror, deterministic addresses, region structure, and learned interpretation, global attention over every substrate character may be unnecessary or wasteful.

If attention returns, it should return for a specific cognitive job.

Examples that may justify later experiments:

- proposition-level comparison;
- small-set hypothesis competition;
- cross-entity relational reasoning;
- retrieval among a bounded set of interpreted objects.

The default experimental question should be:

> Can a recurrent/SSM/hierarchical mechanism solve this without quadratic global attention?

If yes, keep the simpler mechanism.

---

## 10. Mamba / SSM question

A selective state-space layer is especially interesting for the Deliberation Chamber because it naturally suggests persistent evolving state without all-to-all attention.

However, this table must avoid mystical claims.

A Mamba/SSM state does not think while the computer is idle. It changes only when computation advances it.

Therefore a "churning" Mamba-like deliberator requires explicit cognitive microsteps or new input events.

Key experiments:

1. Can an SSM retain useful long-horizon cognitive residue better than GRU512 at similar parameter/compute budgets?
2. Can it revise stale internal beliefs cleanly after corrections?
3. Does repeated no-new-external-input deliberation improve held-out reasoning?
4. Does it preserve relevant information without becoming dominated by old residue?
5. How does it behave after parameter updates?
6. Does it outperform a two-GRU control enough to justify complexity?

---

## 11. Preserve the successful D512 control

The current single-chamber D512 GRU is now **Control A**.

Do not mutate it into every new idea.

Do not declare it obsolete because a more sophisticated architecture is aesthetically attractive.

It has already demonstrated measurable learning on Axon's real substrate.

Every new architecture should compete against it under matched evidence, curriculum, parameter/compute accounting, held-out evaluation, and free-running tests.

Opening control anatomy:

```text
exact D16
  -> Linear(16,512)
  -> GRUCell(512,512)
  -> categorical registered transport + private EOS
```

Current B4 evidence from the 700-step local run:

- trainable parameters: 1,765,216;
- held-out mean loss: ~5.87 -> ~2.36;
- held-out teacher content accuracy: ~0.16% -> ~28.17%;
- strongest content-only constant floor: ~2.35%;
- free-running exact sequence accuracy: 0% -> 2.5%;
- teacher EOS accuracy: 87.5% final;
- termination accuracy: 100% final;
- valid Unicode accuracy: 100% final;
- peak PyTorch CUDA allocation: 78,771,712 bytes (~75 MiB);
- local GTX 1650 runtime: approximately 13 minutes for the 700-step governed tranche.

This remains substrate-literacy evidence, not a language or reasoning claim.

---

## 12. Opening architecture tournament

The table should prefer controlled branching over one grand redesign.

### A - Current control

One GRU512 chamber.

Purpose: preserve the simplest measured baseline.

### B - Two recurrent chambers

```text
exact mirror
 -> Field Interpreter GRU512
 -> Deliberator GRU512
 -> output
```

Question: does functional separation improve generalization, corrections, memory, or reasoning?

### C - Wider deliberation

```text
Field Interpreter D512
 -> Deliberator D1024 or D2048
```

Question: does wider private cognition pay dividends once field interpretation has compressed the active problem?

### D - Hybrid recurrent / SSM

```text
Field Interpreter GRU512
 -> selective SSM/Mamba-like Deliberator
```

Question: does selective state dynamics outperform pure GRU on long-lived cognitive residue and correction?

### E - Private pondering

Same two-chamber model, but permit K internal deliberation steps before output.

Question: does additional cognitive time improve hard tasks without new external evidence?

### F - Small bounded attention only above interpretation

No character-level global attention. Permit attention only over a bounded set of interpreted propositions/objects.

Question: is attention valuable once mechanical substrate/addressing and field interpretation have already reduced the problem?

The tournament is illustrative, not exhaustive.

Any engineer may propose a radically different anatomy, but must preserve a fair control and falsifiable tests.

---

## 13. Curriculum should follow anatomy

We should not build an architecture first and then ask a generic language corpus to somehow teach every tissue the right job.

Each chamber should receive objectives that correspond to its function.

### Field Interpreter curriculum candidates

- identify changed region;
- identify superseded/corrected claims;
- link referents across deltas;
- select exact supporting spans;
- distinguish relevant from irrelevant regions;
- detect contradiction;
- track active task/request;
- rebuild correctly after deletion, replacement, negation, or mask changes;
- preserve exact handle provenance.

### Deliberation curriculum candidates

- combine two or more interpreted facts;
- resolve contradictions;
- perform multi-step inference;
- compare alternatives;
- plan actions;
- revise a conclusion after new evidence;
- reason under incomplete evidence;
- benefit measurably from additional cognitive microsteps;
- form a proposition without requiring the entire field in foreground state.

### Crystallizer/output curriculum candidates

- emit the intended proposition accurately;
- distinguish proposition from canonical truth;
- serialize variable-length exact output;
- terminate reliably;
- preserve evidence references where required;
- eventually realize propositions in natural English.

Language should emerge through staged teaching of these cognitive jobs, not through urgency to make the system conversational as quickly as possible.

---

## 14. Repository-scale thinking hypothesis

A major future requirement is reasoning over large repositories and large tool results.

The opening position is:

> **A wide learned state should not be required to contain the repository.**

The exact mirror / Cortex / deterministic retrieval surfaces retain source material.

The interpreter maintains active semantic organization and evidence handles.

The deliberator carries only the evolving cognitive state needed to work the current problem.

Therefore D512, D1024, or D2048 should be evaluated as **cognitive bandwidth**, not literal storage capacity for all source bytes.

The hard learned problem becomes selecting and revisiting relevant evidence.

That problem must be tested honestly; exact addressing does not automatically solve learned relevance selection.

---

## 15. Multi-Core philosophy

The existence of FIRST / REFINED / consolidator does not mean Axon should depend on a Congress of shallow Cores to simulate one deep thinker.

Opening principle:

> **Private depth should come before social debate.**

A Core should be capable of observing, retaining, reconsidering, integrating, pondering, and revising itself before sibling proposals are required.

Then multi-Core cadence becomes more meaningful:

### FIRST

A Core privately interprets the field, performs its own bounded deliberation, and crystallizes its best current proposition.

### REFINED

The Core receives sibling proposals, privately deliberates again, and produces a revised proposition.

### FINAL

The consolidator receives already-thought-through refined propositions and performs its own governed synthesis.

This preserves heterogeneous minds while avoiding needless conversational chatter as a substitute for cognition.

---

## 16. State, Soul, and durability boundary

This table must keep several state classes distinct:

1. **Exact field mirror** - deterministic, reconstructable from Heart, non-authoritative locally.
2. **Field interpretation state** - learned, persistent while resident, rebuildable when required.
3. **Deliberation state** - private volatile cognitive residue.
4. **Crystallized propositions** - explicit outputs of cognition, possibly entered into scratch/proposal/Soul/Dormant according to policy.
5. **Dormant** - durable organism-level lived evidence/memory.
6. **Parameter memory** - learned weights/adapters across training generations.

The table must not casually rename all six things "memory."

The existing Soul durability doctrine is not amended by this opening.

A future Core Architecture proposal may recommend a revised Soul boundary, but that requires explicit reconciliation with Source of Truth.

---

## 17. Weight updates and resident state

Persistent learned state introduces a difficult rule:

> A hidden state produced under parameter generation N may not remain meaningful under generation N+1.

Therefore architecture experiments must state which state survives optimizer updates.

Opening conservative rule:

- exact mirror survives because it is not learned cognition;
- explicit English / exact structured artifacts may survive if their contract allows it;
- parameter-dependent interpreter/deliberation latent state should be rebuilt, migrated by an explicit tested mechanism, or discarded after weight changes;
- no silent latent carryover across parameter generations.

This should remain fail-closed until evidence supports something more sophisticated.

---

## 18. What success means

Do not judge a proposed architecture by training loss alone.

Useful measures include:

- held-out content accuracy;
- free-running exact generation;
- correction/revision accuracy;
- delayed recall where the answer is absent from later deltas;
- exact mirror / interpretation consistency after edits;
- evidence-selection precision/recall;
- contradiction handling;
- reasoning accuracy by number of cognitive microsteps;
- generalization to unseen compositions;
- ability to revisit relevant exact spans;
- state stability across long event sequences;
- degradation under irrelevant deltas;
- compute cost;
- parameter count;
- CPU/GPU memory;
- time per event;
- ability to recover cleanly from a mirror rebuild;
- whether architecture complexity earns measurable capability.

A larger architecture that cannot outperform Control A on the cognitive job it was added to solve should not remain merely because it is larger.

---

## 19. Falsification questions for the table

Every participant should attack the proposal, not protect it.

1. Does a separate Field Interpreter actually outperform one recurrent chamber?
2. Can the exact mirror plus one GRU already learn interpretation and deliberation well enough that two chambers are unnecessary?
3. Does persistent interpretation drift away from mirror truth over long sequences?
4. What exact mechanism lets the interpreter point back to relevant evidence?
5. Can span handles remain deterministic while relevance is learned?
6. Does a Deliberation Chamber retain too much stale residue?
7. What is the correct decay/overwrite mechanism?
8. Does Mamba/SSM materially improve long-horizon cognitive state at our scale?
9. Does internal pondering improve reasoning, or merely amplify errors?
10. How does the system know when to stop pondering?
11. Should cognitive-step count be learned, heuristic, budgeted, or Heart-governed?
12. Can pondering occur without new external input in a mathematically meaningful way for the chosen architecture?
13. Is D512 enough for Field Interpretation?
14. Is D512 enough for Deliberation?
15. Does D2048 become easier to justify only after the active problem is compressed by an interpreter?
16. Where, if anywhere, does bounded attention become clearly superior?
17. Can the system reason over repository-scale mirrors without continuously rescanning them?
18. How should large tool results enter interpreter state?
19. How do deletions and corrections invalidate learned interpretation?
20. What must be rebuilt after mask changes?
21. What private state survives process restart?
22. What private state survives parameter updates?
23. When does a crystallized proposition become Soul versus scratch versus Dormant?
24. Can propositions carry exact evidence references without becoming brittle symbolic programs?
25. How do we prove a second chamber adds cognition instead of merely parameters?
26. What benchmark demonstrates "chewing on an idea" rather than repeated next-symbol prediction?
27. Can the same Core improve an answer after four internal steps without seeing additional information?
28. Does multi-Core refinement still add value once individual Cores have deep private deliberation?
29. Should specialist Cores have different chamber anatomy while sharing the same D16 public bus?
30. What evidence would make us abandon this entire decomposition and return to a simpler single-chamber model?

---

## 20. Immediate RoundTable request

This opening authorizes **discussion and proposals only**.

It does not authorize:

- replacing the current D512 control;
- starting a new architecture training lineage;
- adding Mamba dependencies;
- changing Source of Truth;
- changing Heart circulation;
- changing Soul durability;
- deleting D64 legacy tissue;
- changing the D16 bus contract;
- modifying canonical State.

Participants are asked to submit responses into this folder addressing at minimum:

1. whether the three-way split (mirror / interpretation / deliberation) is sound;
2. whether two learned chambers are justified;
3. recommended anatomy for each chamber;
4. whether and where SSM/Mamba belongs;
5. whether private cognitive microsteps should exist;
6. how exact evidence handles should cross into learned cognition;
7. the smallest falsifiable two-chamber experiment;
8. metrics that would justify advancing beyond Control A;
9. architecture risks or hidden assumptions;
10. any radically different design the table should test.

The table should prefer **small, reversible experiments** over architectural commitment.

---

## 21. Opening thesis

The strongest current thesis is:

> **Axon should preserve reality exactly, interpret it persistently, deliberate privately, and crystallize thought only when useful.**

Heart owns reality.

The exact D16 mirror preserves what happened.

The Field Interpreter maintains what the current situation appears to mean.

The Deliberation Chamber carries the volatile residue of ongoing thought.

Cognitive time allows that private state to evolve without pretending every internal transition is a new Heart event.

Crystallization harvests explicit propositions from that evolving state.

Language and action then serialize those propositions back through exact organism interfaces.

Whether this anatomy is correct remains open.

For the first time, however, Axon has a measured control Core simple enough that the table can test these ideas rather than merely argue about them.

That is the purpose of this workstream.
