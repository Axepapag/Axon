# Transformer Rebuild Proposal — Rail-Native Core Architecture

**Author:** ChatGPT / GPT-5.6 Sol / 2026-09-22 America/Chicago  
**Status:** proposal for RoundTable review; no implementation or optimizer work authorized by this document  
**Scope:** reasoning-core anatomy from rail ingress through learned reasoning and exact substrate emission  
**Working name for the proposed family:** Rail-Native Streaming Core (RNSC)

## Executive thesis

Axon should not spend learned transformer capacity rediscovering facts that its body already knows exactly.

The Shared Field is canonical. Its exposed cells are exact. The 16D substrate is exact. The packed rails are exact. Region identity, canonical order, source position, provenance, transport identity, and rail receipts are exact. Therefore a reasoning core should not need learned attention to decide what symbol entered, where an exposed cell sits in canonical sequence, or whether a cell that exactly encodes `A` is really `A`.

The current D64 lineage crosses that boundary too early. Exact substrate cells are lifted, mixed with region/page/position features, passed through learned page encoding, stored as contextual neural memory, projected again into learned pointer keys, and retrieved through learned pointer queries. Stage-0A then tries to train the model to recover an address relation that the Heart/compiler already possessed exactly. The recent D0 and A0 evidence makes the cost of this mismatch concrete: frozen memory keys remain highly position-separable while the request-to-query bridge fails; adding a query scaffold still did not produce canonical pointer mastery by accepted step 17.

This proposal changes the question.

Do not ask, "How do we train the current transformer to point correctly at its exact substrate?"

Ask, "Why is exact substrate ingestion and exact physical addressing inside the learned problem at all?"

The proposed answer is a rebuild of the core boundary:

1. Heart compiles the exposed Shared Field into each registered exact-width rail exactly as today.
2. A core is a registered rail consumer, effectively wired into its rail. It receives the rail in canonical order rather than semantically attending to the rail.
3. A deterministic ingress scanner unpacks or references exact substrate cells, preserves their exact canonical addresses and transport identities, and fills a finite occurrence/staging buffer in order.
4. No learned Q/K/V operation is permitted in this substrate-recognition path. `A` is `A` by construction. Cortex position 31 is position 31 by construction.
5. Once a staging block is populated, the learned transformer reasoning chamber operates over exact occurrences and carried recurrent/Soul state. Here learned attention and FFNs are fully appropriate: this is where Axon learns composition, syntax, semantics, associations, abstractions, plans, and reasoning.
6. The staging buffer recycles and the next ordered rail block arrives. The learned chamber carries bounded recurrent state forward until complete-field coverage is proven. The full canonical field is never silently truncated to the working window.
7. Output is learned at the level of *which canonical substrate category should come next*. Serialization is mechanical: once the core chooses a transport identity, the exact registered substrate cell is written. English is learned character by character or scalar by scalar; rendering the chosen character is not learned.

In short:

**Exact substrate in. Learned intelligence in the middle. Exact substrate out.**

This is not a rejection of transformers. It is a narrower and more Axon-native use of them. Learned attention remains central inside cognition. What is removed from learning is substrate recognition, physical field ordering, and exact canonical addressing.

## Why this proposal exists now

The present failure sequence was useful because it exposed an architectural boundary, not merely a bad hyperparameter.

The preserved step-24/25 D0 oracle diagnosis established:

- normal exact-source pointer top-1: `0/24` at both boundaries;
- normal first-cell accuracy: `1/24`;
- compiler-certified oracle pointer with the unchanged learned route: `24/24`;
- oracle pointer plus forced copy: `24/24`;
- Cortex memory-key position classification far above chance;
- pointer-query requested-address classification at `0%`;
- EOS was not the first-cell blocker;
- the copy/output route was sufficient once source selection was exact.

A0 then added a versioned query-side canonical-address scaffold while freezing donor tissue. At accepted step 17 the 68-address heldout surface remained `0/68` exact top-1, despite positive query variance, complete target-position coverage, and preserved frozen key geometry near `0.98205` cross-episode address classification.

That evidence should not automatically trigger a better pointer network. It should trigger a review of whether canonical pointer geometry belongs in learned anatomy.

The answer proposed here is: **not for canonical Shared Field ingress and exact occurrence addressing.**

A learned associative pointer may still be useful elsewhere — semantic retrieval, learned concept association, Dormant recall ranking, approximate similarity, or latent reasoning. But exact canonical field location is not an associative-memory problem. It is an address problem that the body has already solved.

## Relationship to existing doctrine

This proposal preserves the strongest existing Axon contracts:

- one canonical Shared Field;
- Heart as sovereign compiler, validator, mask owner, and sole canonical writer;
- exact 16D substrate truth;
- exact packed rails at registered widths;
- no engineer-chosen total content ceiling;
- paging/chunking only as compute controls with exact continuation and coverage receipts;
- exact Unicode transport;
- English FIRST/REFINED public reasoning;
- tagged FINAL surface interpreted mechanically by Heart;
- private Soul per core;
- Dormant as durable experience/knowledge substrate;
- heterogeneous core widths and specialties;
- continuous tensors never gaining canonical authority merely by being close to a substrate vector.

However, ratification would require an explicit Source-of-Truth amendment to at least two pieces of current doctrine.

First, the present statement that "cores attend directly to their designated rail" should be refined. Under this proposal, a core **consumes its designated rail deterministically in order**, while learned attention operates over the core's internal reasoning representation after exact ingress. The Heart still remains outside learned cognition and outside the semantic attention path.

Second, the D64 receipt-continuation language that assumes learned pointer anchors and addressable neural memory for canonical field copy would become legacy/shelved anatomy for canonical Shared Field ingestion. Exact canonical addresses and substrate values would be structural. Learned pointer mechanisms could remain available for genuinely associative or semantic tasks.

This proposal therefore must not be slipped into the current architecture ID. It requires a new architecture generation and explicit governance.

## The fundamental separation: identity, occurrence, and meaning

The rebuilt core should carry three different kinds of state and never confuse their authority.

### 1. Substrate identity — immutable

A transport identity is exact. If the registered 16D substrate cell represents `A`, then it represents `A` everywhere and at every tick. The core does not train itself to recognize that identity.

For learned processing, each exact substrate identity may have a deterministic fixed lift into the core width. For D64, this may be an injective/frozen 16D-to-64D basis lift whose construction, hash, inverse/roundtrip property, and version are governed. The fixed lifted representation of `A` is identical every time `A` enters the same architecture generation.

This fixed basis is not a learned token embedding table.

### 2. Occurrence — immutable for the frozen tick/view

An occurrence says where one exact substrate identity appears in the currently circulated field: region, region position, attended interval, source span, rail row/lane, field/view/tick identity, and provenance receipt.

If `CAT` occupies three consecutive exposed cells, the occurrence sequence preserves `C`, then `A`, then `T` at exact positions. Repeated symbols are not collapsed: `BANANA` contains multiple occurrences of the same immutable `A` identity at different exact positions.

A physical occurrence lookup is deterministic indexing, not learned similarity.

### 3. Meaning/reasoning state — learned and mutable

The reasoning chamber may construct any architecture-native hidden state it needs. After several learned layers, the hidden state aligned with the `A` occurrence in `CAT` may encode much more than letter identity: local word membership, syntactic role, semantic associations, current task relevance, or hypotheses.

That is allowed and desirable because the exact substrate identity and occurrence receipt remain separately available. Learned context never overwrites canonical reality.

This is the core architectural rule:

**Learned state may interpret exact substrate. It may not become the authority for what exact substrate was present.**

## Proposed anatomy

### A. Rail-native ingress port

Every registered reasoning core has an ingress port bound to the rail width supplied by Heart for that core. "Built into the rail" means the runtime contract is streaming/ordered consumption of exact compiled rail rows, not a learned semantic query over an external memory object.

The ingress port must:

- accept only a verified rail/view/tick identity;
- preserve canonical order;
- unpack or reference each exact 16D transport lane without approximation;
- reject malformed rows/receipts;
- expose exact occurrence metadata separately from learned tensors;
- never use learned Q/K/V, nearest-neighbor snapping, or trainable symbol recognition;
- produce deterministic coverage receipts proving which cells entered the core.

### B. Finite substrate staging buffer

Each core has a finite *work slice*, not a finite world.

A D64 core might stage 32, 64, or another governed number of occurrences at a time. The exact number is a compute parameter, not a content ceiling and not tied to `d_model=64`.

The staging buffer fills in canonical order. When ready, it is handed to the learned reasoning chamber. The buffer can then recycle for the next block. The authoritative rail does not need to disappear or mutate simply because a local buffer advances.

Each staged occurrence carries, at minimum:

- exact transport/category identity;
- exact canonical occurrence/address receipt;
- fixed width-native substrate basis representation, if the reasoning chamber consumes one;
- explicit occupancy/validity mask.

Region and position must not be mixed into the immutable symbol identity. They may be supplied as separate deterministic structural features to cognition.

### C. Learned reasoning chamber

This is where transformer intelligence belongs.

The chamber receives a block of exact occurrences plus carried recurrent state and private Soul state. Learned multi-head attention and FFNs operate here.

The chamber is allowed to discover and learn:

- adjacency and sequence composition;
- spelling and morphology;
- word boundaries and lexical units;
- syntax;
- semantic similarity and distinction;
- relations such as `cat is a mammal`;
- long-range references;
- task relevance;
- planning and reasoning;
- uncertainty;
- proposal generation strategy;
- learned abstractions that need not round-trip to 16D.

This is intentionally where fuzziness belongs. Semantics are not exact addresses. They are learned relationships over exact evidence.

The first rebuilt implementation should stay conservative here: reuse an ordinary transformer block or current compatible reasoning tissue where its contract is clean. The radical change is the boundary *before* learned reasoning, not an unnecessary invention of a wholly new attention mathematics on day one.

### D. Streaming recurrence across the complete field

One core does not need to place the entire Shared Field inside one quadratic attention matrix.

The chamber processes staging blocks sequentially and carries bounded recurrent state between them. The existing code already contains a predecessor of this idea: current `CompleteField64D` iterates compiled pages and carries `state` between page-encoder calls. The rebuild should preserve the useful streaming skeleton while moving exact intake outside learned page encoding.

A complete tick is not considered ingested until coverage receipts prove that every exposed cell required by the view was consumed exactly once or according to an explicitly ratified replay policy.

The carried state may include:

- recurrent state tokens;
- private Soul contribution;
- learned summaries;
- optional hierarchical semantic accumulators;
- explicit structural boundary signals.

No carried learned state substitutes for exact source receipts.

### E. Optional deterministic revisit/address bus

The first proof may need only sequential streaming. If later reasoning requires exact random revisit of a previously circulated occurrence, canonical location should still be handled mechanically.

A learned controller may decide *which exact address/span it wants*. Once an exact address has been selected, fetching that occurrence is ordinary deterministic indexing against the frozen view/rail receipts. The fetch mechanism does not compare learned pointer keys to discover where address 31 is.

This preserves a clean division:

- cognition decides what it wants;
- anatomy delivers exactly what was requested.

Approximate semantic retrieval remains a separate capability and must be labeled as such.

### F. Exact substrate output motor

The output side mirrors ingress.

The reasoning chamber learns which canonical output category should come next. For English, this is ultimately character/scalar/transport selection in exact sequence. The learned model is responsible for language: spelling, vocabulary, grammar, semantics, reasoning, response construction, and termination.

But after it chooses a registered category, serialization is deterministic.

The output motor must not emit an arbitrary continuous vector and ask nearest-neighbor geometry whether it was probably `A`. It emits a categorical identity from the exact registered transport vocabulary; the serializer writes the exact substrate cell(s) for that identity.

For multi-cell Unicode scalars, the public contract must remain exact and valid. A governed scalar-level mechanism may select a Unicode scalar and mechanically serialize its registered transport sequence, or the core may learn transport-unit emission under strict scalar framing. The RoundTable should compare both designs before locking this detail.

Termination is an exact control symbol or governed stream boundary. It is learned only insofar as cognition must decide *when the response is complete*; its physical serialization is exact.

## What happens to "attention"

This proposal intentionally uses the word attention in two different senses and then removes the ambiguity.

### Substrate scanning is not learned attention

Reading the rail in canonical order is not semantic attention. It is deterministic ingestion. The substrate reader should be deliberately unintelligent.

### Cognitive attention remains learned

After exact occurrences enter the reasoning chamber, normal learned attention is useful and powerful. A head can learn relationships among letters, words, phrases, concepts, prior recurrent state, Soul state, and task context.

The architecture therefore does **not** abolish Q/K/V. It prevents Q/K/V from being used to rediscover exact facts that already have canonical addresses and identities.

### Exact addressing is not semantic attention

`Cortex[31]` is a physical/canonical occurrence address. Fetching it should not require a learned dot-product search.

"Find the part of the field relevant to mammals" is a semantic selection problem and may appropriately use learned attention, semantic Cortex machinery, or Dormant retrieval.

These should be different APIs, different metrics, and different failure modes.

## What should be shelved from the current core

If this architecture is ratified, the following current mechanisms should be treated as candidates for shelving or narrowing, not blindly carried forward:

1. Learned `position_key(memory.states)` as the authority for exact canonical field addressing.
2. Learned pointer queries whose purpose is to rediscover canonical `(region, position)`.
3. The A0 query-side address scaffold as a permanent physical-address solution.
4. Any `AddressableMemory` contract that requires learned neural geometry to prove where exact canonical Shared Field cells live.
5. Any training curriculum whose first job is to teach the model that an exact known address is an address.
6. Any learned path whose only purpose is to preserve an exact symbol identity already guaranteed by substrate/codec contracts.

These components should not necessarily be deleted. They may be valuable evidence, reusable semantic mechanisms, or historical tissue. Shelf them with manifests and return conditions.

## What should be preserved

Do not throw away the organism because one organ boundary was wrong.

Preserve:

- Heart host/lease/transaction/valve authority;
- canonical State and Shared Field;
- mask law and exact attended intervals;
- D16 substrate and Unicode transport codebook;
- generic packed-rail compiler and exact cross-width repacking;
- field/view/tick/provenance receipts;
- Trainer governance, lineage, immutable evidence, and fail-closed gates;
- Dormant and autobiography/evidence bridge;
- private Soul stores and transactional continuity;
- FIRST/REFINED/CONSOLIDATED circulation;
- English public proposal contract and tagged FINAL contract;
- heterogeneous core registry and round-robin consolidator policy;
- current page/chunk coverage logic where its semantics remain valid;
- step-24/25/A0 checkpoints as immutable experimental evidence.

## Step-24 donor policy

This is a new architecture generation. Do not resume the step-24 optimizer and do not call transferred tissue the same candidate.

Step 24 may be a governed donor only.

Every tensor must be classified into one of four buckets:

1. **Exact compatible donor** — same shape, same semantic contract, safe to copy bit-for-bit.
2. **Potential reasoning donor** — shape-compatible but contract changed; requires an ablation proving transfer helps rather than contaminates the new core.
3. **Legacy ingress/pointer tissue** — tied to learned substrate recognition/addressing; shelf, do not transplant by default.
4. **New tissue** — initialize under a versioned, receipted rule.

The current `page_encoder` is especially suspect as an automatic donor because it presently mixes recurrent state with lifted cells before producing contextual memory. Some internal transformer weights may still be valuable, but the entire module cannot be assumed contract-equivalent.

Soul transfer also requires a new architecture binding. The old Soul should be preserved and may be used as a donor only through an explicit compatibility/migration experiment. No silent reinterpretation of latent Soul tensors across changed anatomy.

## Radical extensions worth exploring — not required for first proof

The RoundTable should think broadly before converging. The following are deliberately exploratory.

### 1. Dual-lane occurrence representation

Every reasoning occurrence could expose two lanes:

- an immutable fixed substrate/address lane;
- a mutable learned semantic lane.

Attention and FFNs modify only the learned lane while residual access to exact identity remains available at every layer. This would make it impossible for deep contextualization to erase the distinction between "what was literally present" and "what the core currently thinks it means."

### 2. Structural sideband instead of positional addition

Conventional transformers often add position embeddings directly into token embeddings. Axon could keep position/region/provenance as a separate structural sideband consumed by dedicated projections or attention bias, leaving substrate identity untouched.

This may be more faithful to Axon's distinction between exact world structure and learned interpretation.

### 3. Hierarchical streaming reasoning

Instead of one recurrent summary, the core could form multi-timescale learned state: local block state, phrase/segment state, tick state, and Soul state. Exact receipts bind every learned summary back to the source interval it summarizes.

This may let tiny cores process very large fields without global quadratic attention.

### 4. Learned internal containers over exact spans

Once literacy develops, a core may learn that a span `C-A-T` forms a lexical unit `cat`. Such a container is a learned semantic object pointing back to an exact source span. It never replaces the source characters.

This could become the bridge from character-level exact substrate to efficient word/concept-level cognition without introducing a tokenizer as canonical truth.

### 5. Semantic attention as a separate organ-level service

A later Core may ask for semantic surfacing such as "material relevant to mammals." That approximate retrieval can operate over learned Cortex/Dormant structures, but the returned evidence is materialized back into exact Shared Field spans before reasoning. Fuzzy retrieval proposes evidence; exact substrate remains what the reasoning core actually receives.

### 6. Event-driven rail ingestion

If Heart updates only a small part of the exposed field, mature cores might consume verified deltas plus exact continuity receipts rather than replaying unchanged material every time. This could greatly reduce inference cost, but only after full snapshot ingestion is proven and replay/recovery semantics are airtight.

### 7. Internal planning at higher abstraction, exact character emission at the edge

Axon need not reason character-by-character forever merely because public I/O is exact characters. Mature cores may form learned word/concept/plan latents internally, then realize them through the exact categorical output motor. The fixed substrate governs communication, not the granularity of every hidden thought.

## Conservative first implementation

The first rebuild should prove the thesis with the smallest anatomy that can falsify it.

### R0 — deterministic substrate ingress proof

No optimizer.

Build a new core ingress path that:

- consumes a compiled D64 rail/view in canonical order;
- unpacks every valid 16D lane exactly;
- maps each transport identity to a fixed, nontrainable D64 basis representation or preserves a direct exact identity sideband;
- preserves exact occurrence receipts separately;
- stages finite ordered windows;
- proves complete coverage across arbitrary page counts, region boundaries, mask gaps, provenance boundaries, empty regions, and Unicode transport sequences.

Gate: byte/category/cell/address/order roundtrip must be exact. Any mismatch is an implementation failure, not a trainable loss.

### R1 — deterministic echo proof

Still no semantic training.

Route staged exact identities directly to the categorical output serializer and prove that arbitrary attended field spans can be echoed exactly, including repeated characters and Unicode. This is a body/motor proof, not intelligence.

The test answers: "Can Axon see exact substrate and physically say the same exact substrate without a learned pointer?"

If R1 is not perfect, do not train.

### R2 — reasoning-chamber interface proof

Introduce a minimal learned reasoning chamber while keeping ingress and output mechanical. Confirm that learned state can change freely without corrupting the immutable substrate/occurrence lane.

Run corruption probes: adversarial learned activations must not mutate canonical input identities or receipts.

### R3 — alphabet and sequence literacy

Now train.

Teach exact symbol identity usage and ordered sequence relations, not recognition of the symbols themselves. Examples include next/previous character, copy a requested short sequence, identify equality/difference, and compose simple letter sequences.

### R4 — spelling and lexical composition

Teach that ordered substrate spans form words. Start with tiny exact curricula and separate heldout composition from memorization.

The core should learn that `C-A-T` forms `cat`; the body should not learn what C, A, or T are.

### R5 — vocabulary and semantics

Teach relations such as `cat is an animal`, grounded to exact source spans. Begin building learned containers/concepts if evidence supports them.

### R6 — grammar and sentence structure

Teach word order, syntax, references, and transformations while preserving exact character-level I/O.

### R7 — free-running English proposals

Train FIRST/REFINED proposal generation through the exact categorical output motor. Introduce termination, recovery from self-generated prefixes, and scheduled sampling only after basic character/word generation is competent.

### R8 — reasoning and lived experience

Only after language mechanics are stable should harder Dormant-grounded reasoning, Soul use, counterfactual evidence tests, and long-horizon semantic integration become primary gates.

## Compute implications

The rebuild is motivated by correctness first, but it may materially reduce compute.

### Deterministic ingress

Exact unpacking, table lookup/fixed lift, sequencing, and staging are cheap relative to learned transformer blocks. They require no optimizer state, no learned gradients, and little activation retention for backpropagation.

### Bounded-window attention

If the exposed field has `N` occurrences and the reasoning chamber processes windows of `W` occurrences with bounded recurrent state, the attention component can approach `O(N * W)` rather than global `O(N^2)` attention, assuming `W` stays bounded. This is not free — FFNs and recurrent processing still cost compute — but it changes field-length scaling substantially.

### Smaller optimization problem

The more important saving may be sample/optimizer efficiency. Gradient descent no longer spends steps discovering symbol identity, physical address geometry, or exact transport rules that are known by construction.

### Exact small output vocabulary versus more steps

Character/transport-level emission has a much smaller categorical surface than modern subword vocabularies, but it requires more autoregressive output steps. This is a real tradeoff, not a guaranteed net FLOP reduction. Later internal word/concept planning may recover efficiency while public serialization remains exact.

### Event-driven future

If mature cores can consume exact verified field deltas rather than replay complete unchanged views, runtime cost could fall further. This must remain future work until full snapshot behavior is proven.

## Risks and hard questions

This proposal should be attacked aggressively before ratification.

1. **Does a fixed per-symbol D64 basis waste valuable width?** A symbol basis plus structural sideband must leave enough room for learned cognition, or cognition should use a separate learned state tensor rather than reuse the fixed basis as the only state.
2. **How should learned attention see position without contaminating identity?** Options include deterministic sideband, attention bias, rotary/sinusoidal structural features, or dedicated structural channels.
3. **How much recurrent state is enough to integrate arbitrarily long fields?** Bounded recurrence creates an information bottleneck. Exact replay/address access or hierarchical summaries may be required.
4. **When should the core revisit old exact occurrences instead of relying on summaries?** This needs a deterministic replay/address policy.
5. **What is the right output decision unit?** Raw transport unit, Unicode scalar, character, or a layered scheme. Exact public serialization must remain invariant whichever unit is learned.
6. **How should Soul interact with streaming blocks?** Soul must not become a hidden substitute for skipped field coverage. Its causal role needs stage-appropriate probes.
7. **Can useful step-24 reasoning tissue transfer?** This must be measured rather than assumed.
8. **Does character-level literacy make early learning slower than subword token training?** Possibly. The payoff is exact substrate grounding and compositional reuse; empirical curriculum design still matters.
9. **Should a core consume packed rail rows directly or per-cell deterministic lifts?** Both preserve exactness. The first may exploit hardware packing; the second gives simpler per-character reasoning tokens. Benchmark both without changing canonical semantics.
10. **How do heterogeneous widths share learned concepts?** They already share exact public English/substrate identity; private learned latents need not be width-compatible. Cross-core communication should continue through exact public text, not latent translation.

## Falsification criteria

Do not protect this proposal from evidence.

Reject or substantially revise the rebuild if any of the following is demonstrated:

- deterministic ingress cannot preserve exact substrate/order/receipt identity across the real complete-field contract;
- separating substrate intake from learned encoding materially prevents useful reasoning that the old architecture can prove under comparable compute;
- bounded streaming cannot retain enough information and exact replay cannot repair the deficit without worse cost than a simpler architecture;
- a controlled benchmark shows learned canonical pointer memory materially outperforms deterministic exact addressing for the same exact-address task without sacrificing correctness/governance;
- donor incompatibility makes the rebuild so destructive that a smaller architecture correction clearly solves Stage-0A with stronger evidence.

Conversely, one successful training run is not enough to ratify the mature architecture. The rebuild must prove structural correctness, learnability, runtime fidelity, and governance separately.

## Required RoundTable questions

Every reviewing engineer should answer these directly:

1. Is canonical Shared Field addressing a learned problem at all? If yes, justify why deterministic indexing is insufficient.
2. Should exact symbol identity and learned semantic state be separate tensors/lanes? Why or why not?
3. Should the reasoning chamber consume per-character fixed D64 lifts, packed D64 rows, or another exact representation?
4. What is the minimum recurrent/streaming anatomy that can ingest the full field without an engineer-chosen content ceiling?
5. How should exact old occurrences be revisited after their staging window has advanced?
6. Which step-24 tensors are genuinely contract-compatible donors?
7. What should replace `AddressableMemory.position_key` for canonical field addressing?
8. Which learned pointer mechanisms should survive for semantic retrieval rather than physical addressing?
9. What exact categorical output unit should the first rebuilt core learn?
10. What is the smallest R0/R1 test suite that would prove the mechanical body before training?
11. What Source-of-Truth clauses must be amended, superseded, or explicitly retained?
12. What failure would convince you this proposal is wrong?

## Proposed governance path

1. **Freeze optimizer work on the current pointer-motor line.** Preserve A0 as evidence.
2. **RoundTable review only.** Engineers challenge this proposal and file independent alternatives/refinements.
3. **Synthesize a convergence document.** Compare deterministic ingress designs, reasoning-chamber interfaces, output units, transfer policy, and SoT amendments.
4. **Ratify architecture before implementation.** New architecture ID, schema versions, donor rules, Soul migration rules, and gates must be explicit.
5. **Implement R0/R1 first.** No training until the body proves exact ingress and exact echo.
6. **Only then begin a fresh curriculum.** Training starts above the substrate boundary.

## Recommendation

I recommend that Axon stop treating the current pointer-training failure as a problem to be optimized through.

Treat it as evidence that the learned/nonlearned boundary is in the wrong place.

The body already knows the exact field. The rail already carries exact substrate. The compiler already knows the order. The receipts already know the address. Do not ask a tiny transformer to spend intelligence reconstructing those truths from geometry.

Build the Core into the circulation path. Let exact substrate fall into ordered occurrence containers mechanically. Let finite staging windows stream through a learned reasoning chamber until coverage is complete. Let attention and FFNs spend their capacity learning what sequences mean, how concepts relate, how language works, and how to reason. When the core decides to speak, let it choose exact canonical output identities and let anatomy serialize them without approximation.

**The transformer should learn English and thought. It should not have to learn how to read Axon's alphabet or locate Axon's own bones.**

That is the proposed rebuild.
