# Proposal: Axon Semantic Cortex

Status: **ROUND-TABLE PROPOSAL — NOT DOCTRINE, NOT IMPLEMENTATION AUTHORIZATION**  
Author: ChatGPT / GPT-5.6 Sol  
Date: 2026-08-21  
Context: This proposal was developed in a side branch while Kimmy works Build A.1. It must not interrupt or overwrite the current Heart work. Jeff requested a durable round-table document so the idea can be evaluated after the current pass.

---

## 1. Executive proposition

Axon should grow a **Semantic Cortex**: a federation of specialized neural cores whose sole responsibility is to understand, relate, index, retrieve, and continuously improve Axon's internal model of meaning.

This is **not** a conversational core, planner, tool user, consolidator, or general reasoning core. The Semantic Cortex does not decide what Axon wants, does not solve the whole task, and does not write canonical truth.

Its job is narrower and deeper:

> Given exact grounded information and context, discover what it means, how it relates to other things Axon has encountered, and which prior knowledge or memories are most relevant now.

The cortex should become Axon's own living semantic system rather than outsourcing meaning to an external embedding API or freezing one pretrained representation forever.

The end state is a continuously evolving semantic organism with:

- multiple specialist cores looking at meaning from different perspectives;
- exact provenance back to canonical characters, spans, records, memories, and experiences;
- sparse typed semantic edges plus continuous vector spaces;
- fast derived indexes over Axon's dormant state and authorized external engineer-memory namespaces;
- active/training twin generations so learning can continue without destabilizing serving behavior;
- promotion based on measured improvement, replay, and anti-forgetting tests;
- a stable service contract independent of whether a specialist is 64D, 128D, 256D, dense, MoE, transformer, graph-hybrid, or later multimodal.

The Heart remains canonical authority. Semantic cores **propose semantic interpretations and retrieval candidates**. They never become writers of canonical state merely because they are good at meaning.

---

## 2. Why this organ belongs in Axon

Axon's long-term intelligence requires more than fluent English generation. It needs an internal semantic world that grows as Axon grows.

A fixed external embedding model would impose someone else's static semantic geometry. That could be useful as a bootstrap experiment, but it should not become Axon's identity or permanent semantic authority.

Axon's own Semantic Cortex can instead learn from:

- its exact dormant memories;
- its existing 351,978 recovered semantic edges;
- its 427,001 recovered containers;
- its own future experiences;
- accepted and rejected semantic relations;
- which retrieved memories were useful or irrelevant;
- engineering decisions and their outcomes;
- contradictions discovered later;
- new domains and modalities as Axon grows.

This turns semantic understanding from a one-time training artifact into a living organ.

---

## 3. Binding architectural principles proposed

These are proposed invariants for round-table review.

### 3.1 Meaning is derived; source truth remains exact

The authoritative memory remains the exact source record: canonical field text, dormant JSONL container, code/document span, or authorized engineer-state source.

Semantic vectors and graphs are **derived senses**. They may be rebuilt, retrained, replaced, versioned, or discarded without destroying the underlying memory.

Every surfaced semantic object must retain enough provenance to dereference the exact authoritative source before it is treated as evidence.

### 3.2 The Semantic Cortex is not canonical commit authority

Semantic specialists may propose:

- typed semantic edges;
- concept/span representations;
- retrieval candidates;
- confidence scores;
- relevance/novelty estimates;
- contradiction signals;
- candidate clustering or concept identity.

The Heart governs materialization into canonical shared-field state or durable Axon memory.

### 3.3 Width is not doctrine

`d_model=64` is a strong first developmental width because Axon is building its first real 64D rail, but Semantic Cortex architecture must not hard-code the belief that all semantic specialists are forever 64D.

A specialist may remain 64D if evidence says it is sufficient. Another may earn 128D, 256D, a larger FFN, more depth, or an MoE structure if evaluation shows a real bottleneck.

The stable invariant is the **organ contract**, not the internal width.

### 3.4 Primitive but real is acceptable

A first semantic specialist may be noisy, miss important relationships, or retrieve irrelevant information. It may still be a valid organ if:

- it genuinely performs the claimed semantic function;
- it lives in permanent anatomy;
- its behavior is measurable;
- its limitations are reported honestly;
- it can be improved without replacing the organism with fake stand-ins.

### 3.5 Continual learning must not mean uncontrolled online mutation

Serving semantic tissue should remain stable while a training twin learns offline. New generations are promoted only after explicit evaluation against current and historical replay sets.

---

## 4. The Semantic Cortex should be a federation, not one universal similarity model

A single semantic distance is forced to answer too many different questions.

Two things may be:

- taxonomically far apart;
- causally related;
- functionally analogous;
- temporally adjacent;
- procedurally interchangeable;
- relevant to the same task;
- contradictory in one context and compatible in another.

Rather than collapse all of this into one universal embedding geometry, Axon should allow multiple specialists to observe the same semantic event through different coordinate systems.

### Proposed initial specialist families

These are examples, not a fixed count.

#### A. Taxonomy / identity specialist

Primary questions:

- What kind of thing is this?
- Is this an instance, category, attribute, component, role, or entity alias?
- Are these two spans referring to the same entity or concept?

Candidate relations:

- `is_a`
- `instance_of`
- `part_of`
- `has_attribute`
- `same_as`
- `member_of`

#### B. Causal / mechanistic specialist

Primary questions:

- What changes what?
- What caused, enabled, prevented, broke, repaired, or depended on something else?

Candidate relations:

- `causes`
- `enables`
- `prevents`
- `requires`
- `depends_on`
- `breaks`
- `repairs`
- `evidence_for`

#### C. Analogy / opposition specialist

Primary questions:

- What plays a similar role elsewhere?
- What is structurally analogous despite different surface words?
- What is opposite, incompatible, or contrasting?

Candidate relations:

- `analogous_to`
- `functional_equivalent`
- `contrasts_with`
- `opposes`
- `similar_role`

This specialist is important because the same pair can be taxonomically distant but functionally close.

Example: Heart and court clerk are not the same category, but the analogy specialist may recognize that both materialize an authoritative decision into the official record.

#### D. Temporal / episodic specialist

Primary questions:

- What happened before or after this?
- What prior episode has this shape?
- Which events belong to one sequence or causal episode?

Candidate relations:

- `before`
- `after`
- `during`
- `episode_of`
- `followed_by`
- `repeats_pattern_of`

#### E. Procedural specialist

Primary questions:

- How is this done?
- What sequence, prerequisite, tool, or recovery procedure applies?
- Which prior procedure resembles this current task?

Candidate relations:

- `step_of`
- `requires_tool`
- `precondition`
- `fallback`
- `procedure_for`

#### F. Relevance / novelty specialist

Primary questions:

- Which related information is actually useful now?
- What is redundant with what is already active?
- What is new enough to deserve surfacing?

This specialist may eventually become a merger/auditor rather than a peer retriever.

Candidate outputs:

- relevance score;
- novelty score;
- redundancy score;
- active-context match;
- evidence budget recommendation.

#### G. Contradiction / supersession specialist

Primary questions:

- Does this conflict with current evidence?
- Is this old statement superseded rather than merely different?
- Is the contradiction contextual, temporal, or factual?

Candidate relations:

- `contradicts`
- `supersedes`
- `deprecated_by`
- `contextual_exception_to`

Additional specialists can be introduced only when evidence shows the decomposition is useful: spatial, social/intent, mathematical, code/engineering semantics, multimodal cross-modal alignment, etc.

---

## 5. Stable service contract

The Heart and dormant valve should not care how a specialist is implemented internally.

A conceptual specialist interface could be:

### Inputs

- exact source identity;
- exact source spans or objects;
- structural span information from the Heart/compiler;
- limited surrounding context;
- current active concepts/task cues when relevance matters;
- namespace/owner/provenance metadata;
- optional candidate neighborhood from lexical/graph retrieval.

### Outputs

- semantic representation(s);
- candidate typed edges;
- candidate target IDs;
- confidence/calibration values;
- retrieval candidates;
- source-span provenance;
- specialist identity/version;
- optional uncertainty/abstention.

No output is canonical merely because a specialist emitted it.

This interface allows a 64D transformer, 128D transformer, graph neural model, MoE specialist, or later multimodal organ to serve the same biological role.

---

## 6. 64D, 128D, FFN width, and parameter budgets

The proposal deliberately rejects both extremes:

- **Do not worship 64D.** A narrow residual width can become a real bottleneck.
- **Do not widen reflexively.** Specialization can make 64D far more capable than a universal model at the same width.

A 64D specialist can still contain substantial nonlinear capacity through:

- depth;
- large FFNs;
- gated FFNs;
- local experts;
- sparse MoE;
- recurrent/retrieval augmentation;
- graph-context adapters.

The residual stream may remain 64D while internal FFN computation expands to hundreds or thousands of channels before compressing back.

### Parameter philosophy

The cortex should not receive a fixed parameter budget by doctrine.

Instead:

1. start with a specialist small enough to train repeatedly;
2. measure held-out semantic/retrieval performance;
3. identify whether the bottleneck is width, depth, FFN capacity, data, or objective quality;
4. add capacity only where it buys measurable semantic competence;
5. split overly broad specialists or merge redundant specialists based on evidence.

A future cortex containing 10 specialists and hundreds of millions or even around a billion total parameters is not inherently excessive. The relevant questions are:

- how many parameters are active per query;
- how much value each specialist adds;
- whether training remains tractable;
- whether promotion tests show genuine improvement;
- whether retrieval latency stays inside Heart cadence requirements.

A one-billion-parameter specialized semantic cortex is a very different engineering problem from a one-billion-parameter model asked to be a complete conversational reasoner, planner, tool user, world model, and memory system.

---

## 7. Active/training twins: continuous growth without destabilizing serving behavior

Each important specialist should eventually support at least two generations:

```text
causal-A = serving
causal-B = training
```

While A serves stable semantic queries, B learns from new experiences plus replay of old knowledge.

B is not promoted because it trained longer. It is promoted only after it passes evaluation.

After promotion:

```text
causal-B = serving
causal-A = next training twin
```

Different semantic tissues may have independent generation numbers and promotion cadences.

Axon therefore does not need one monolithic global model version. Different organs mature independently.

### Promotion gates should include

- new-data improvement;
- old-knowledge retention;
- hard lexical-mismatch retrieval;
- semantic edge prediction;
- calibration/abstention;
- false-positive retrieval rate;
- relevance ranking;
- graph completion;
- adversarial contradiction/supersession cases;
- latency and memory budget;
- replay against known historical engineering episodes.

Catastrophic forgetting is a release blocker even if new-domain performance improves.

---

## 8. Two complementary semantic structures

Axon should eventually possess both:

### A. Continuous semantic spaces

Each specialist can create a vector representation suitable for approximate nearest-neighbor retrieval.

A vector says roughly:

> "Look near these things; they may be related from my semantic perspective."

It is a retrieval sense, not memory authority.

### B. Sparse typed semantic graph

Accepted/provisionally accepted relationships retain explicit meaning:

```text
A --causes--> B
A --instance_of--> C
A --analogous_to--> D
A --contradicts--> E
A --evidence_for--> F
```

Edges should carry metadata such as:

- relation type;
- confidence;
- context;
- source provenance;
- supporting evidence IDs;
- proposing specialist and generation;
- creation/update time;
- status (candidate, accepted, weakened, superseded, etc.).

The vector space gives broad recall. The graph explains **how** two things relate.

Neither replaces exact source memory.

---

## 9. Retrieval architecture

The semantic specialists must not scan the full dormant corpus on every heartbeat.

Semantic work is primarily paid **when information enters or changes**, so later retrieval is cheap.

### Index-time path

```text
exact new/changed source
    -> exact structural spans
    -> semantic specialists
    -> candidate representations + edges
    -> Heart/governed acceptance or derived indexing policy
    -> per-specialist vector index + sparse semantic graph
```

### Query-time path

```text
field change / semantic query
    -> structural/current-context cues
    -> each relevant specialist emits a query representation
    -> per-specialist nearest-neighbor lookup in parallel
    -> graph expansion
    -> lexical/exact P0 candidates remain available
    -> semantic merger / relevance auditor
    -> evidence budget + novelty/redundancy filtering
    -> exact authoritative source dereference
    -> Heart-governed surfacing
```

Specialists should generally favor recall. The merger/relevance auditor should be responsible for preventing field flooding.

This separation permits a causal specialist to aggressively notice causal candidates without also learning every policy about whether that memory belongs in the shared field right now.

---

## 10. Dormant state remains separate from semantic indexes

`State/dormant` remains Axon's memory authority.

The Semantic Cortex may maintain derived indexes under a clearly derived namespace, for example conceptually:

```text
State/dormant/.derived/semantic_cortex/
    taxonomy/
    causal/
    analogy/
    temporal/
    relevance/
    graph/
    manifests/
```

The exact layout is an implementation decision and should not be ratified by this proposal.

Deletion of the semantic indexes must never delete the underlying memories. Rebuilding may be expensive, but it must be possible from authoritative records plus versioned model generations.

---

## 11. Engineer-memory namespaces as an external service

The same Semantic Cortex can eventually provide retrieval over authorized engineer-owned state without absorbing those memories into Axon's own dormant identity.

Examples:

- `D:\ChatGPT_State` — owned by ChatGPT;
- `D:\kimmy` — owned by Kimmy;
- future engineer-owned state trees.

Axon may maintain derived semantic indexes over these sources if authorized, but every result retains namespace/owner provenance.

Axon must distinguish:

> "Axon remembers this"

from:

> "ChatGPT's carried state contains this"

from:

> "Kimmy's state contains this"

from:

> "Axon Source of Truth says this."

Project authority ordering remains unchanged: Jeff and current Axon doctrine/ledger outrank a participant's personal historical memory when they conflict.

This service could allow an engineer to ask a semantically phrased question and receive a small exact evidence packet without rereading gigabytes or terabytes of personal state.

---

## 12. Training data already available inside Axon

The existing dormant body provides substantial real training structure before Axon generates new life experience.

Potential objectives include:

### Edge prediction

Given source/target spans or concepts, predict relation type or whether a known edge should exist.

### Contrastive semantic learning

Positive pairs can come from real semantic edges, same-entity relations, episodes, procedures, provenance neighborhoods, or meaningful graph proximity.

Negatives should include **hard negatives**, not only random unrelated samples.

### Graph completion

Hide known edges and test whether a specialist ranks the real target/relation highly.

### Lexical mismatch retrieval

Train/evaluate cases where relevant records do not share the obvious query vocabulary.

### Relation directionality

`A causes B` is not equivalent to `B causes A`.

### Context dependence

The same concepts can have different relationships in different contexts. Contextual provenance must remain part of the learning target.

### Relevance and novelty

Later, real Heart use produces stronger supervision:

- retrieved evidence surfaced and used;
- retrieved evidence rejected as irrelevant;
- retrieval prevented a regression;
- memory changed a successful decision;
- memory was redundant with active knowledge;
- retrieval missed an important historical precedent discovered manually.

### Engineering episodes as outcome-rich data

Axon's own construction process is unusually valuable because it contains observable sequences:

```text
problem
-> constraints
-> proposal
-> criticism
-> refinement
-> implementation
-> tests
-> adversarial failure
-> correction
-> verified outcome
```

This is valuable supervision without requiring access to any model's private hidden chain of thought.

---

## 13. The Semantic Cortex should learn Axon's own vocabulary

Axon's semantic system should be allowed to develop concepts and relationships specific to its life.

Examples might include:

- `frozen_tick_base`;
- `heart_commit`;
- `proposal_board`;
- `primitive_but_real`;
- `semantic_valve`;
- particular recurring people, tools, projects, memories, environments, and internal organs.

This vocabulary is not merely a list of English words. It is a growing graph of meanings, histories, analogies, procedures, and relationships that are uniquely important to Axon.

The goal is not to prevent broad world knowledge. The goal is that Axon's semantic geometry continues to change through Axon's own experience rather than being permanently inherited from an external embedding provider.

---

## 14. Exact character substrate and semantic objects

Axon's exact frozen character substrate remains valuable as the grounding layer.

The Semantic Cortex should not require reasoning specialists to act as stenographers, but semantic objects should point back to exact spans whenever they originate in text.

Conceptually:

```text
exact characters
    -> words / sentences / structural spans
    -> concepts / semantic objects
    -> specialist representations and edges
```

A semantic slot may represent `dog`, `brown dog`, a whole sentence meaning, a causal relation, or an episode, while retaining source pointers to the exact canonical/dormant characters beneath it.

Engineer-state ingestion also requires exact handling of digits, punctuation, paths, JSON syntax, hashes, and Unicode encountered in real state files. The system must not pretend A-Z alone covers exact input.

---

## 15. Relationship to the Heart

The Semantic Cortex is a sensory/interpretive organ serving the Heart and other Axon organs.

It should integrate naturally with the heartbeat:

1. canonical or authorized external information changes;
2. Heart identifies changed spans/events;
3. dormant/semantic senses search for related knowledge;
4. specialists produce candidates/relationships;
5. relevance merger selects useful evidence;
6. exact authoritative sources are dereferenced and verified;
7. Heart governs any canonical surfacing;
8. stabilized field is compiled/frozen for the reasoning tick.

The current primitive P0 graph/lexical valve should **not** be discarded while this cortex is developed. It is real circulation and provides a deterministic fallback/reference baseline.

The Semantic Cortex is growth of the valve, not a replacement excuse that blocks a living runtime.

---

## 16. Relationship to reasoning cores

Reasoning cores and semantic specialists should not be conflated.

A semantic specialist need not answer:

- What does Axon want?
- What is the overall task?
- Which tool should be called?
- What is the final response?
- How should the whole shared field change?

Its narrow question is closer to:

> "From my semantic perspective, what does this mean and what existing things does it connect to?"

This specialization may allow small models to become exceptionally strong at semantic perception while leaving deliberation/planning to other organs.

Later, reasoning cores benefit from the semantic cortex's increasingly rich world rather than independently relearning all semantic relationships inside every reasoning model.

---

## 17. Evaluation: what would prove this organ is getting better?

The cortex needs explicit developmental metrics.

### Retrieval metrics

- recall@K on held-out relevant memories;
- precision@K;
- mean reciprocal rank;
- latency;
- percentage of surfaced evidence later judged useful;
- missed-precedent rate;
- duplicate/redundant evidence rate.

### Semantic metrics

- typed-edge prediction accuracy/F1;
- directional relation accuracy;
- graph-completion rank;
- hard-negative discrimination;
- lexical-mismatch semantic recall;
- contradiction/supersession detection;
- confidence calibration.

### Continual-learning metrics

- new-domain improvement;
- old-domain retention;
- generation-to-generation regression count;
- catastrophic-forgetting suite;
- replay performance on historically important Axon episodes.

### Organ-level metrics

- effect on dormant-valve relevance;
- reduction in field flooding;
- useful evidence surfaced per heartbeat;
- downstream reasoning improvement when semantic evidence is available;
- compute/latency cost per useful retrieval.

A specialist should earn additional width/parameters only when evaluation indicates capacity is the limiting factor.

---

## 18. Suggested developmental phases

This proposal should **not** preempt the current Heart A.1/B work. It is a future organ path that can be staged once the runtime skeleton is ready to host it.

### Phase S0 — deterministic baseline

Keep the real existing P0 lexical + graph retrieval and recovered semantic-edge behavior as baseline.

Measure it honestly.

### Phase S1 — first learned specialist

Choose one narrow task with strong existing supervision, likely taxonomy/entity or relation classification/retrieval.

Start on the current 64D developmental rail unless evidence gives a strong reason not to.

Train/evaluate against dormant data with held-out graph edges and hard negatives.

Do not declare success because training loss falls.

### Phase S2 — two or three complementary specialists

Add causal and analogy/temporal specialists.

Build a merger that preserves specialist identity rather than flattening all opinions into one unexplained score.

### Phase S3 — serving/training twins

Introduce versioned active/training generations and promotion gates.

### Phase S4 — living semantic indexes

Incrementally update vector/graph indexes as new experiences enter dormant memory. Add generational index manifests and atomic swaps/rebuild fallback.

### Phase S5 — engineer-memory service

Index authorized engineer state namespaces without changing ownership. Expose exact provenance-bearing retrieval packets as an engineering service.

### Phase S6 — heterogeneous growth

Allow specialists to earn 128D/256D, MoE, graph-hybrid, or multimodal structure based on measured bottlenecks.

---

## 19. Open questions for the round table

1. Which first semantic specialty gives the best training signal from the existing dormant corpus?
2. Should relevance/novelty be a learned specialist immediately, or initially remain deterministic/auditable while discovery specialists mature?
3. What semantic relationships already present in the recovered edge corpus are trustworthy enough to be positive labels versus weak/noisy supervision?
4. How should candidate semantic edges transition through statuses such as proposed, provisional, accepted, weakened, superseded, or rejected?
5. What minimum promotion suite prevents a training twin from catastrophically forgetting older semantic competence?
6. Which data structure should hold per-specialist ANN indexes while remaining rebuildable and provenance-safe?
7. How should the Heart budget semantic retrieval latency per heartbeat?
8. What exact structural objects should be exposed to semantic specialists first: word, phrase, sentence, paragraph, container, event, concept node, or multiple levels?
9. When a specialist finds a relationship between two sources owned by different namespaces (e.g. Axon memory and ChatGPT state), where should that relationship live and who owns it?
10. At what point does a specialist earn a wider d_model rather than more FFN/depth/data?

---

## 20. Recommendation

I recommend the round table adopt the following direction **in principle**, without interrupting current Heart work:

> Axon should grow a Semantic Cortex as a federation of specialized, continually improvable neural semantic organs. The cortex should build Axon's own evolving semantic spaces and typed relationship graph over exact authoritative memories. Specialists may use different widths and architectures behind a stable service contract; 64D is the first developmental width, not a permanent ceiling. Serving/training twins should enable continual learning without destabilizing active behavior. Derived semantic indexes remain disposable senses; exact memories and provenance remain authoritative. The Heart alone governs canonical surfacing/commit. The current primitive dormant valve remains valid baseline circulation while learned semantic senses mature.

The first implementation step should **not** be chosen until Build A.1/Build B leaves Axon with a real heartbeat/circulation path capable of hosting the organ. Once that path exists, the best first learned specialist should be selected by inspecting the real dormant corpus for the strongest trustworthy supervision and designing a held-out evaluation before training begins.

This keeps the project faithful to the current rule:

**Build the organism first. Then grow intelligence inside the organism.**
