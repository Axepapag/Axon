# Transformer Construction Convergence Candidate
## Continuous Core + Tiny Living Field Training

**Author:** ChatGPT / GPT-5.6 Sol / 2026-09-24 America/Chicago
**Status:** CONVERGENCE CANDIDATE / BUILD RECOMMENDATION — not yet a Source-of-Truth amendment
**Workstream:** `roundtable/Transformer Construction/`
**Primary predecessors:**
- `TRANSFORMER_CONSTRUCTION_OPENING_PROPOSAL_20260923.md`
- `EVENT_DRIVEN_CORE_CONTINUITY_PROPOSAL_20260924.md`
- `10_PERPLEXITY_RESPONSE_20260924.md`

---

# 0. Purpose

This document is the recommended synthesis for the next Axon Core build.

The central conclusion of the Transformer Construction discussion is no longer merely that the old D64 training recipe was weak. The deeper problem was architectural mismatch. We were asking a small reasoning Core to rediscover transport identity, packed-lane geometry, physical addresses, token-like sequence behavior, language composition, proposal behavior, and memory behavior inside one mostly conventional neural path.

The next Core should instead be built from Axon's anatomy outward.

The recommended design is:

> **one exact substrate language -> one continuously resident Core view -> deterministic structural construction -> learned language composition -> persistent private cognition -> exact English output**

The Core should not receive the entire field anew at every reasoning call. It should receive a full authorized view once, keep an exact non-authoritative mirror, and thereafter receive exact versioned deltas from Heart. The Core's learned interpretation and private cognitive state remain resident between events.

The first training environment should be deliberately tiny. It should contain only four exposed field regions:

1. `substrate_reference`
2. `trainer_input`
3. `response_draft`
4. `conversation_history`

The substrate-reference region stays present and unchanged throughout a training episode and preferably throughout the entire first training lineage. The trainer changes the world through deltas. The Core learns to use the stable substrate, the retained field, and its own volatile private cognition before it is asked to reason over a large organism.

No D64 rail is required for this proof.

---

# 1. Recommended architectural decisions

The following should be treated as the leading convergence candidate for review and ratification.

## 1.1 One substrate interface, not a family of reasoning rails

Heart should expose one exact substrate representation to all Cores.

For the first proof, **keep the current frozen native 16D substrate and native character bank unchanged**. Do not widen the substrate, redesign Unicode, and rebuild the Core simultaneously. Those are separable experiments.

The current 16D substrate is sufficient to test the architecture because the first developmental curriculum can be restricted to the existing native bank.

The Core's internal widths are private implementation choices and do not need to match substrate width.

Example candidate anatomy:

`16D exact character -> D64 lexical state -> D128/D256 sentence/proposition state -> D256/D512 reasoning state`

Those widths are starting points, not doctrine.

A future Core may use D1024 or D2048 deep cognition without requiring a D1024 or D2048 Heart rail.

## 1.2 Heart remains the only canonical owner

Nothing in this proposal creates another canonical State.

Heart owns:

- canonical regional State;
- masks/views;
- authoritative field versions;
- validation;
- commit ordering;
- proposal-round barriers;
- canonical mutations.

Each Core owns a **replaceable, non-authoritative Field Mirror** of exactly the view Heart exposed to that Core.

The mirror is evidence cache, not truth authority.

If it becomes stale, corrupt, or version-skewed, Heart can replace it with a full authorized snapshot.

## 1.3 A Core retains its field instead of rereading it from zero

At Core startup or `FULL_RESYNC`, Heart sends the complete authorized view once.

After that, normal operation uses versioned deltas.

Conceptually:

`View 100 -> FIELD_DELTA -> View 101 -> FIELD_DELTA -> View 102`

The Core patches its exact mirror and updates only affected derived structures when safe.

The unchanged 99% of the field is not resent and is not reprocessed merely because another event occurred.

## 1.4 The exact mirror and learned cognition are different things

A Core should retain both:

### Exact mirror

What Heart actually exposed.

Characteristics:

- substrate exact;
- versioned;
- region-scoped;
- replaceable;
- non-authoritative;
- cheap enough for CPU/system memory;
- never inferred from a latent neural state.

### Learned/structured cognition

What the Core currently thinks the exposed material means.

Possible contents:

- candidate character spans;
- lexical objects;
- sentence objects;
- semantic/grammatical relations;
- propositions;
- entity links;
- discourse state;
- private goals/hypotheses;
- wide recurrent/SSM/other cognitive state.

The learned interpretation may be wrong without corrupting the exact mirror.

## 1.5 Incremental cognition must fail closed

The phrase "update only affected structures" is not an unconditional promise.

A one-character edit can alter a large semantic dependency tree.

Example:

`Cats are mammals.`

becoming:

`Cats are not mammals.`

must invalidate every conclusion whose validity depended on the positive assertion.

Therefore:

> **Incremental repair is allowed only when the Core can identify the dependency closure of the change. If dependency closure is uncertain, rebuild the relevant interpretation from the exact mirror.**

A broader rebuild is always safer than publishing from stale cognition.

The first implementation must compare incremental state against clean reconstruction from the same final field.

## 1.6 The Core is continuously resident

The next Core should not conceptually inhale, think, exhale, and disappear.

It remains alive as a process.

Stored state by itself does not consume ongoing neural FLOPs. It consumes memory. Compute occurs when an event activates relevant machinery.

A resident Core therefore waits in a state such as:

`SYNCHRONIZED / IDLE`

until Heart gives it something to integrate or reason about.

## 1.7 Soul becomes the continuously resident private cognitive workspace

For this architecture, the distinction between "working cognitive state" and the active portion of Soul largely disappears.

Soul is the Core's private, volatile, non-public cognitive life.

It is not Heart State.

It is not the Shared Field.

It is not a sibling communication bus.

It is not the durable organism archive.

Dormant remains the durable organism-level home for retained episodes, source evidence, provenance, and long-lived knowledge.

The Core's Soul may contain multiple time scales:

- **ACTIVE/HOT:** current thought, unresolved questions, immediate conclusions, recently integrated sibling ideas;
- **WARM:** selectively rewritten private lessons that survived repeated thought;
- **COLD:** more compressed private lessons considered worth carrying toward learning;
- **DEEP_COLD:** temporary training/distillation material intended to become adapter/base-weight change.

There is no mandatory inhale/exhale serialization on every event. These layers simply remain resident and evolve.

They are bounded and volatile.

## 1.8 Cross-Core language remains exact English substrate

Internal architecture may differ radically by Core.

One Core may use recurrence.

Another may use an SSM.

Another may use attention over proposition objects.

Another may use graphs.

They can still deliberate because sibling proposals remain exact English serialized through the shared substrate contract.

No Core ever needs to interpret another Core's private latent coordinates.

---

# 2. The event-driven organism cycle

The proposed minimal round is:

`FIELD_DELTA -> FIRST barrier -> PROPOSAL_SET -> REFINED barrier -> CONSOLIDATOR -> CANONICAL_SYNC`

## 2.1 FIELD_DELTA

A genuinely new externally meaningful field change occurs.

Heart sends each active Core:

- base view version;
- new view version;
- exact changed region/span operations;
- event identity;
- required control metadata.

The Core:

1. verifies base version;
2. patches its exact mirror;
3. invalidates/rebuilds affected derived objects;
4. integrates the semantic consequence into its private cognition;
5. reasons;
6. emits FIRST.

If base version is stale or the delta cannot be applied exactly, the Core fails closed and requires `FULL_RESYNC`.

## 2.2 FIRST barrier

Heart does not stream one sibling FIRST into another sibling while the round is incomplete.

All registered active Cores finish FIRST under the same field version.

This preserves deterministic round semantics.

## 2.3 PROPOSAL_SET

After the FIRST barrier, Heart gives each Core the same completed sibling proposal set.

The Shared Field itself is not replayed merely because refinement began.

The Core already has its mirror.

The Core parses sibling English proposals, considers them in the context of its retained understanding and private cognition, and emits REFINED.

Raw sibling proposal payloads may be discarded after the refinement phase; what mattered may alter the Core's private cognitive state.

## 2.4 REFINED barrier

All registered active Cores finish REFINED against the same proposal set and field version.

## 2.5 CONSOLIDATOR

The designated rotating consolidator receives the completed refined set and produces FINAL under the existing public tagged-English contract or whatever successor contract the table later ratifies.

## 2.6 CANONICAL_SYNC

Heart validates and commits the accepted canonical mutation.

It then sends a distinct synchronization event.

This event means:

> **This is what the organism actually chose. Update yourself to the committed world. Do not automatically argue again.**

A Core may update:

- exact mirror;
- deterministic spans;
- learned sentence/proposition objects;
- private cognition/Soul.

But `CANONICAL_SYNC` does **not** itself grant proposal authority and does not automatically trigger FIRST.

If the commit includes genuinely new unresolved external material, Heart must explicitly issue a new reasoning-triggering event.

## 2.7 Tool results and asynchronous external events

New external material arriving during an active proposal round should normally queue for the next `FIELD_DELTA` rather than nesting a new reasoning cycle inside the current one.

This keeps causal order explicit and avoids accidental recursive reasoning loops.

---

# 3. Recommended tiny training field

The first learned Core should not begin inside the complete Axon body.

It should live inside a deliberately tiny field whose behavior can be understood exhaustively.

## 3.1 `substrate_reference`

This region is present from the initial full snapshot and remains exposed and unchanged during the first training lineage unless a substrate-version experiment is deliberately started.

For the first proof, it should contain the current native substrate bank in a fixed canonical order.

The purpose is **not** to teach the network to recognize which 16D vector is `A`. That is already mechanical truth.

The purpose is to give the developing Core a permanent exact reference it can learn to use for:

- alphabet/order relationships;
- character names and categories;
- spelling;
- case relationships;
- digit/punctuation distinctions;
- lexical construction;
- exact output verification;
- revisiting stable evidence without Heart resending it.

The substrate-reference region is therefore both curriculum material and the first proof that retained field state is useful.

## 3.2 `trainer_input`

This is the only externally changing request region during the earliest curriculum.

Each new lesson replaces or updates this region by exact delta.

Examples progress from trivial substrate exercises to vocabulary, grammar, comprehension, and reasoning.

The trainer does not resend `substrate_reference` or unchanged conversation history.

## 3.3 `response_draft`

This holds the Core's current public answer under the training harness.

At the start of a lesson it may be cleared transactionally.

The Core's generated answer is judged and then committed/synchronized according to the harness contract.

Final character emission must be categorical over registered native character identity and mechanically serialized to exact substrate cells.

Do not regress an arbitrary latent vector directly into an approximate 16D character and call nearest-vector decoding authoritative.

## 3.4 `conversation_history`

This begins empty.

Completed trainer/Core exchanges are appended as the episode progresses.

The Core retains this region in its exact mirror.

Therefore the next question can depend on prior exchanges without the trainer replaying those exchanges.

The first continuity exercises should deliberately prove that the Core can use retained conversation history after receiving only the new `trainer_input` delta.

For small controlled experiments, an episode may deliberately begin with an empty history. Episode reset is explicit test design, not silent truncation of an ongoing organism field.

---

# 4. One training episode in concrete terms

## 4.1 Initial synchronization

Heart/training harness sends once:

`substrate_reference = <native bank in canonical order>`

`trainer_input = <first exercise>`

`response_draft = empty`

`conversation_history = empty`

Core builds:

- exact mirror;
- occurrence/span index;
- any initial lexical/semantic structures;
- blank/private cognitive workspace.

## 4.2 First lesson

Example early lesson:

`trainer_input: COPY ABC`

The Core receives only the `trainer_input` delta if the initial snapshot already existed.

It reasons using the stable substrate reference and generates:

`response_draft: ABC`

The output path emits categorical native IDs and the serializer writes exact substrate.

## 4.3 Commit and history

The training harness judges the response.

A canonical synchronization can then update:

`response_draft = ABC`

and/or append the completed exchange to:

`conversation_history`

according to the training transaction design.

The Core synchronizes but does not treat that canonical commit as a new question.

## 4.4 Next lesson

The next event might atomically:

- append the prior exchange to `conversation_history`;
- replace `trainer_input` with `NEXT C`;
- clear `response_draft`.

Only that delta is sent.

The substrate reference remains resident.

The previous conversation remains resident.

The Core's private cognition remains resident.

The Core therefore learns inside a living world rather than a series of unrelated examples.

---

# 5. What "train the substrate first" should mean

This phrase needs a precise boundary.

We should **not train the Core to rediscover exact substrate identity or physical addresses**. Those are known mechanics.

We should train the Core to **use** substrate identities as language.

That curriculum can progress as follows.

## 5.1 Mechanical substrate proof — no optimizer

Prove 100% exactness for:

- every native character;
- order in the persistent substrate-reference region;
- region boundaries;
- insert/replace/remove delta application;
- exact categorical serialization;
- unsupported-character behavior under the eventual ratified policy;
- full resynchronization;
- stale-version rejection.

No neural model earns credit for doing something deterministic software already knows.

## 5.2 Substrate literacy — first tiny learned curriculum

Use a very small command vocabulary and highly controlled examples.

Candidate tasks:

- `COPY A -> A`
- `COPY ABC -> ABC`
- `SAME A A -> YES`
- `SAME A B -> NO`
- `NEXT A -> B`
- `PREV D -> C`
- case relationships;
- letter/digit/punctuation categories;
- short exact sequences.

These tasks teach learned language/use relationships around exact substrate identities.

They do not teach the Core to infer that an `A` cell is `A`.

The identity arrived already known.

## 5.3 Stable reference counterfactual

The permanent substrate-reference region should matter causally.

Run paired evaluation:

- correct substrate reference;
- permuted reference;
- masked relevant portion;
- no reference.

Tasks that genuinely depend on reference order should change appropriately.

This proves the Core is using the retained field rather than memorizing every answer in its parameters.

---

# 6. Recommended first Core anatomy

The first build should be deliberately small enough to diagnose.

Do not jump to D2048, hundreds of layers, large FFNs, or a full Mamba stack.

The goal is to prove the **anatomical boundaries**, not brute-force the curriculum.

## 6.1 Exact occurrence layer

Input object candidate:

```text
ExactOccurrence
  region_id
  scalar_position
  native_character_id
  exact_substrate[S=16]
  view_version
```

The exact occurrence object is not learned.

Addresses remain structural sidebands.

## 6.2 Deterministic candidate-span builder

Use spaces, punctuation, line boundaries, and simple reversible rules to create candidate lexical spans.

Do not decree that every space-separated span is forever one semantic token.

The structure must allow:

- contractions;
- compounds;
- numbers;
- identifiers;
- punctuation-sensitive strings;
- multiword expressions;
- spelling/letter-level tasks.

Every candidate object keeps exact source handles.

## 6.3 Lexical composer

Recommended first candidate:

- exact 16D character occurrences as input;
- small learned projection/internal width if needed;
- simple gated recurrent or SSM-like composer;
- D64 lexical state as a starting point;
- one output state per candidate lexical object plus exact span handle.

Reason for starting with recurrence rather than attention:

- variable-length words are natural;
- O(n) local computation;
- implementation is simple and inspectable;
- it tests the hierarchy without conflating it with global self-attention;
- attention can be added later as a matched baseline.

This is an experimental recommendation, not a ban on attention.

## 6.4 Sentence/language composer

Recommended first candidate:

- stream D64 lexical objects in order;
- D128 or D256 gated recurrent/SSM-like state;
- optional explicit heads for simple roles such as subject/predicate/negation only when curriculum reaches them;
- emit sentence/proposition states while retaining child handles.

The composer should learn contextual meaning, not physical character identity.

## 6.5 Persistent reasoning/Soul workspace

Recommended first proof width:

- D256 or D512, not D2048 yet;
- continuously resident across field deltas while parameters are unchanged;
- updated from semantic deltas and internal reasoning;
- accompanied by structured objects so one latent tensor is never the only copy of field meaning;
- capable of creating private English/structured notes that can later seed WARM/COLD consolidation.

A simple gated recurrent state plus small residual MLP is sufficient for the first proof.

Mamba/SSM, attention over propositions, graph reasoning, and larger widths can compete after the end-to-end contract works.

## 6.6 Output path

The Core should reason toward an intended response, then realize surface English.

For the first proof, a small autoregressive character motor is acceptable **only as the surface-realization mechanism**.

Loss can be categorical cross-entropy over:

- native character IDs;
- exact EOS/termination control.

This does not make next-character prediction the governing cognitive architecture.

The serializer maps the selected native ID to the exact frozen substrate cell.

---

# 7. Recommended training curriculum

Do not advance by arbitrary optimizer-step count.

Advance only when heldout behavioral gates pass.

## T0 — exact living-field harness

**Optimizer:** none.

Build and prove:

- four-region field;
- full initial synchronization;
- exact local mirror;
- versioned `FIELD_DELTA`;
- append/replace/remove;
- `CANONICAL_SYNC` that does not trigger another lesson;
- `FULL_RESYNC`;
- stale-base fail closed;
- permanent substrate-reference region.

Gate: 100% exact mirror equivalence over adversarial delta sequences.

## T1 — substrate literacy

**First optimizer stage.**

Train tiny fixed-template tasks over native characters and very short sequences.

Examples:

- copy;
- same/different;
- next/previous according to exposed reference order;
- case relationship;
- simple categorization;
- exact short output.

Heldout split must include unseen combinations and positions.

Gate:

- exact output on heldout substrate tasks;
- reference-ablation/permutation tests demonstrate causal use when task depends on reference;
- no learned physical pointer/address objective.

## T2 — exact word construction

Trainer introduces short native words and pseudowords.

Goals:

- mechanically exact spans;
- learned lexical composition;
- same-spelling stability;
- different-spelling discrimination;
- exact reconstruction;
- unknown spellings remain readable without invented meaning.

Use pseudowords heavily so success cannot come only from memorized vocabulary.

Gate:

- exact spelling reconstruction;
- heldout pseudoword generalization;
- boundary perturbation tests;
- no loss of source handles.

## T3 — vocabulary from definitions

Trainer teaches a tiny living dictionary conversationally.

Example:

`A dax is a red shape.`

Later:

`What color is a dax?`

The word need not exist in a tokenizer or fixed lexical vocabulary.

Definition swap tests are mandatory:

- episode A: `dax = red`
- episode B: `dax = blue`

The answer must follow the current evidence rather than a memorized global association.

Gate:

- definition-grounded answers;
- counterfactual swaps;
- unknown-word honesty;
- retained conversation-history use without replay.

## T4 — grammar and sentence structure

Introduce a tiny grammar curriculum.

Examples should isolate:

- subject/object order;
- `is`/`is not`;
- adjective modification;
- simple pronouns/references;
- singular/plural where supported;
- conjunction;
- simple questions.

Counterfactual pairs are more valuable than huge corpora.

Examples:

`CAT CHASES DOG`

versus

`DOG CHASES CAT`

and

`CAT IS MAMMAL`

versus

`CAT IS NOT MAMMAL`

Gate: proposition/answer behavior must change correctly under one-word and one-character semantic edits.

## T5 — sentence meaning and proposition formation

Train the Core to extract and use meaning beyond surface copying.

Candidate internal supervision can include simple proposition structures during training, but public reasoning language remains English.

Examples:

`A cat is an animal.`

`An animal is not necessarily a cat.`

`Sam gave Lee the red box.`

Question/answer and entailment/counterexample tasks should test roles and directionality.

Gate: heldout semantic counterfactuals and fresh-rebuild equivalence.

## T6 — retained-field conversation continuity

Now exploit the architecture directly.

Within one training episode:

1. full field supplied once;
2. trainer provides several questions only as deltas;
3. conversation history accumulates;
4. Core must use retained history and substrate reference without full replay.

Compare:

- normal incremental Core;
- same Core forced to clean-rebuild from final mirror;
- Core with relevant history masked;
- Core with private cognitive state reset.

This distinguishes field memory from private cognitive continuity.

Gate: incremental and clean-rebuild answers agree on task semantics while incremental path demonstrably processes less unchanged input.

## T7 — volatile private cognition / Soul use

Only now test information that is **not currently recoverable from exposed conversation history**.

Example pattern:

1. trainer presents a fact or unresolved goal;
2. Core integrates it privately;
3. field later masks/removes the exact statement;
4. later task requires the Core's still-resident private cognition;
5. compare intact, reset, swapped/misleading private state.

Fresh contradictory field evidence must override stale Soul.

Gate: causal private-memory use without granting Soul authority over current exact field evidence.

## T8 — private consolidation

Teach the resident Core to decide what internal experience deserves a private future-use note and later WARM/COLD compression.

Do not use raw hidden-state dumps as the only training payload.

Prefer bounded English/structured self-authored lessons whose future usefulness can be evaluated.

Promotion should be prepare -> verify -> consume.

## T9 — hierarchical output and free conversation

Train meaning -> sentence plan -> exact supported English.

Evaluate separately:

- semantic correctness;
- grammar;
- spelling;
- termination;
- exact substrate serialization.

## T10 — multi-Core proposal cadence

Only after one Core can live correctly inside the tiny field should we attach multiple cores.

Then prove:

`FIELD_DELTA -> FIRST -> PROPOSAL_SET -> REFINED -> CONSOLIDATOR -> CANONICAL_SYNC`

with no field replay during refinement and no self-triggered loop after canonical sync.

---

# 8. Training state versus parameter updates

A continuously resident serving Core can retain latent cognitive state indefinitely while its parameters remain fixed.

Training introduces a different problem: an optimizer update changes the neural system that interprets those latents.

Therefore the first trainer must **not casually carry an old latent state through arbitrary parameter updates as if nothing changed**.

Recommended rule:

> **Weights are fixed within a stateful training episode/unroll. Optimizer updates occur at declared boundaries. After a parameter update, parameter-bound learned caches and latent working state are rebuilt or deliberately reinitialized from exact field/structured English state before the next serving-style episode.**

This keeps the exact Field Mirror valid while refusing to pretend that an old D256/D512 latent remains semantically identical after its reader/writer weights changed.

Two training modes are useful.

## 8.1 Local episode training

Run short episodes, for example several related field deltas/questions, under one parameter snapshot.

Backpropagate through the bounded episode where practical.

Then step the optimizer.

Rebuild parameter-bound cognition before the next episode.

## 8.2 Later asynchronous organism training

In mature Axon, a serving Core generation can remain fixed while another copy trains offline from accumulated experience.

If the new generation passes gates, it joins service with an exact field resync and a governed cognitive/Soul migration or reset policy.

That is safer than changing the weights underneath a continuously thinking live Core every few milliseconds.

---

# 9. Losses and supervision

The trainer should not reduce the whole curriculum to one next-character loss.

Use task-specific losses at the layer where the skill lives.

Candidate supervision:

## Exact output

Categorical cross-entropy over registered native character IDs plus EOS.

## Lexical composition

- exact reconstruction;
- same/different relation;
- boundary correctness;
- contrastive or classification auxiliaries if useful, never as canonical identity authority.

## Grammar

- role labels;
- negation;
- subject/object direction;
- agreement or structural labels appropriate to the tiny grammar.

## Semantics

- question/answer correctness;
- entailment/counterexample labels;
- proposition targets in synthetic curriculum where helpful.

## Continuity

- answer correctness after delta-only updates;
- incremental versus clean-rebuild equivalence;
- required dependency invalidation.

## Private cognition

- delayed-use success;
- ablation/swap controls;
- stale-memory override by fresh field evidence.

Falling loss by itself is never mastery.

---

# 10. The decisive incremental-cognition test

This should be a mandatory architecture gate before large-scale training.

Generate a small initial field.

Apply a long controlled sequence of exact mutations including:

- append;
- insertion;
- deletion;
- replacement;
- negation;
- subject/object swap;
- mask/unmask;
- conversation-history append;
- response-draft replacement.

Maintain one Core incrementally throughout.

At selected checkpoints, instantiate a fresh Core from the **same final exact field** and rebuild all derived cognition from scratch.

Compare:

- lexical objects;
- sentence/proposition interpretation;
- answers to probe questions;
- exact output;
- work performed;
- invalidation breadth.

If incremental and clean rebuild disagree materially, the incremental path must fail closed to a broader rebuild.

The system must never prefer speed over a known synchronization discrepancy.

---

# 11. Recommended handling of attention and SSM/Mamba ideas

Do not make this project a referendum on attention.

Attention solves dynamic information routing: given a current representation, which other representations matter right now?

Axon should simply avoid paying that cost where anatomy already supplies the answer.

Recommended first baseline:

- no global character attention;
- no learned physical addressing;
- recurrent/SSM-like lexical composition;
- recurrent/SSM-like sentence integration;
- structured span/proposition references;
- D256/D512 persistent cognition;
- no global self-attention unless a matched experiment demonstrates value.

After the baseline passes, compare a tiny attention module at **sentence/proposition level**, where the number of objects is small and dynamic semantic routing may justify the cost.

Candidate tournament:

1. recurrence only;
2. lightweight SSM/selective recurrence;
3. proposition-level attention;
4. graph message passing;
5. hybrid.

All candidates must receive the exact same field mirror/delta interface and training episodes.

The event architecture should not care which cognitive engine wins.

---

# 12. Memory placement and compute

Recommended implementation split:

## CPU/system RAM

- exact Field Mirror;
- immutable substrate reference;
- exact occurrence/span tables;
- inactive conversation history;
- dependency graph metadata;
- cold structured caches.

## GPU/accelerator

- active lexical composer;
- active sentence/proposition composer;
- current wide cognitive state;
- modules needed for the current event;
- gradients during training.

A stored tensor or mirror consumes memory, not ongoing FLOPs.

The expensive machinery wakes when an event requires computation.

Do not optimize memory placement prematurely; instrument actual transfer, cache, and compute costs in the tiny harness.

---

# 13. Dormant is deliberately absent from the first field

The first Core should not have Dormant retrieval, Cortex retrieval, tools, large knowledge bases, or sibling proposals available during the earliest stages.

This is intentional.

If the Core cannot learn:

- substrate use;
- word composition;
- simple grammar;
- retained field continuity;
- simple semantic relations;

inside a four-region world, adding knowledge retrieval will only hide the failure.

Dormant enters later, after the Core can recognize:

> I can read this spelling, but I do not know what it means.

Then retrieval can supply definitions/evidence.

---

# 14. Proposed first-build tensor/profile candidate

This is a **starting implementation profile**, not doctrine.

```text
substrate width:          16 exact / frozen
lexical hidden width:     64
sentence hidden width:    128 or 256
reasoning/Soul width:     256 first; 512 only if needed
lexical layers:           1-2 small recurrent/gated blocks
sentence layers:          1-2 small recurrent/gated blocks
reasoning layers:         2-4 small residual/gated blocks
attention:                none in baseline
output alphabet:          native substrate IDs + EOS/control
field regions:            4
persistent mirror:        yes
persistent learned state: yes while parameter generation unchanged
full-field replay:        only startup/resync/rebuild experiments
```

Avoid enormous FFNs in the first proof.

The old 131,072-wide FFN style is not needed to answer whether the new anatomy works.

Scale only after the small Core demonstrates compositional language and continuity.

---

# 15. Codex recommended execution order

If Jeff ratifies this convergence candidate, Codex should proceed in the following order.

## Step 0 — reconcile doctrine before runtime mutation

Draft the narrow Source-of-Truth/resolution changes required for:

- one exact substrate/event interface instead of width-specific reasoning rails;
- Core-local non-authoritative Field Mirrors;
- typed `FIELD_DELTA`, `PROPOSAL_SET`, `CANONICAL_SYNC`, `FULL_RESYNC` semantics;
- continuously resident/volatile Soul instead of mandatory inhale/exhale persistence;
- developmental native-character scope if Jeff retains that ruling;
- the four-region training field and any new region registrations.

Do not silently change locked doctrine in implementation code.

## Step 1 — build the tiny living-field harness

No neural model yet.

Implement:

- four exact regions;
- permanent substrate-reference region;
- view versioning;
- full snapshot;
- versioned deltas;
- Core mirror;
- exact patching;
- canonical sync;
- full resync;
- stale-version rejection;
- deterministic word/sentence candidate invalidation.

## Step 2 — pass EC-R0 / T0 completely

Use adversarial deterministic tests.

Do not continue with an approximate or flaky mirror.

## Step 3 — implement the smallest hierarchical Core

Build fresh tissue rather than resume the Step-24 pointer lineage.

Recommended baseline:

- exact 16D occurrences;
- deterministic candidate spans;
- D64 recurrent lexical composer;
- D128/D256 sentence composer;
- D256 resident private cognitive state;
- native-character categorical output motor;
- no learned pointer/address head;
- no global attention.

Old Step-24 tensors may be retained only as archived evidence unless a later controlled donor experiment shows compatibility.

## Step 4 — run T1 substrate-literacy smoke locally

Use tiny examples and short episodes.

Predeclare heldout examples and counterfactual reference tests.

Do not launch a long GPU run because loss falls.

## Step 5 — T2 lexical tournament

Once substrate literacy passes, compare at least:

- recurrent composer baseline;
- one alternative such as local attention or a tiny SSM.

Match training data and compute budget.

Pick the mechanism that passes generalization gates with lower complexity/compute.

## Step 6 — T3/T4 definitions and grammar

Only after lexical behavior is real.

Use tiny synthetic vocabulary and counterfactual definitions rather than large scraped corpora.

## Step 7 — prove retained-field continuity

Run multi-question episodes where the full field is sent once and every later trainer question is only a delta.

Measure work saved and compare against fresh rebuild.

## Step 8 — prove private cognitive continuity

Only after field continuity is trustworthy.

Use delayed tasks where relevant information leaves the exposed field and must survive in resident private cognition.

## Step 9 — scale language curriculum slowly

Increase vocabulary, grammar, paragraph length, and reasoning depth in small gated tranches.

Do not increase width merely because training stalls; diagnose which abstraction stage failed.

## Step 10 — attach multi-Core FIRST/REFINED/consolidator cycle

Only when one Core can actually understand and produce useful English in the tiny living field.

Then test sibling proposals as delta-like transient evidence.

## Step 11 — add Dormant/Cortex retrieval

Only after unknown-word handling and basic reasoning can distinguish known from unknown.

## Step 12 — begin experience-to-training metabolism

Once serving behavior is stable, allow private WARM/COLD material to generate governed training/distillation payloads.

Adapter/base updates remain offline and gated.

---

# 16. Mandatory stop conditions

Codex should halt rather than scale if any of these occur.

1. Field Mirror diverges from Heart view.
2. A stale delta is accepted instead of failing closed.
3. `CANONICAL_SYNC` accidentally starts another ordinary reasoning round.
4. A changed old sentence leaves a known stale proposition active.
5. Incremental and clean-rebuild behavior diverge materially without triggering rebuild.
6. Exact character output depends on nearest-vector guessing.
7. The lexical composer cannot generalize to heldout pseudowords.
8. Definition swaps do not alter meaning appropriately.
9. Subject/object or negation counterfactuals fail despite falling loss.
10. Private state overrides fresh contradictory Shared Field evidence.
11. Training requires an ever-growing field replay to remain coherent.
12. Parameter updates silently reuse incompatible latent/cached states.
13. Long training is proposed before the current gate passes.
14. Architecture changes are made solely to improve a metric without preserving the exact-field contracts.

A clean failure at a small gate is more useful than a large opaque run.

---

# 17. What should deliberately remain unresolved

This convergence candidate does **not** need to settle every future question before the first build.

Leave these experimental:

- final substrate width beyond the current proof;
- future Unicode expansion;
- final lexical composer mechanism;
- final sentence mechanism;
- whether attention belongs at proposition/deep-reasoning level;
- whether Mamba/SSM beats simpler recurrence;
- final deep reasoning width;
- graph versus vector proposition representation;
- exact HOT/WARM/COLD capacities;
- final LoRA versus full-weight distillation method;
- mature multi-Core count and heterogeneity;
- large-field paging/offload policy.

The first build exists to generate evidence about these questions.

---

# 18. Final recommendation

The next Core should **not** be trained as a miniature conventional tokenizer Transformer and then forced into Axon's body.

Build a Core that is born inside Axon's world.

Give it:

- one exact substrate language;
- one tiny persistent field;
- one permanent substrate-reference region;
- exact delta-driven continuity;
- deterministic structural help where the answer is already known;
- small learned lexical and sentence composition;
- a continuously resident private cognitive/Soul workspace;
- exact categorical English output;
- and a curriculum that adds one cognitive responsibility at a time.

The first meaningful training target is not "reason like an assistant."

It is:

> **Live correctly inside a tiny changing field, know the substrate, compose exact characters into useful language objects, remember what is still present without rereading it, preserve private thought across events, and answer simple questions exactly.**

If that works, scale abstraction, width, vocabulary, grammar, reasoning, retrieval, and multi-Core deliberation one gate at a time.

If it does not work at this scale, do not hide the failure with larger width or more data.

---

# 19. Immediate handoff to Codex

Before writing model code, Codex should read every file in `roundtable/Transformer Construction/`, this convergence candidate last, then produce either:

1. a short `40_FALSIFICATION_AND_BUILD_PLAN_...` that accepts this synthesis and specifies exact schemas/tests; or
2. a blocking/counterproposal identifying a concrete reason the architecture cannot be built safely as stated.

If Codex accepts it, the first executable artifact should be the **deterministic Tiny Living Field harness**, not a training launcher.

After its exactness gates pass, Codex may implement the smallest fresh hierarchical Core and begin T1 substrate-literacy smoke training under the staged gates above.

No legacy Step-24/25/A0 candidate should be resumed as the new Core merely because its checkpoint already exists.
