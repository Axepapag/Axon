# Transformer Construction — Opening Proposal

**Author:** ChatGPT / GPT-5.6 Sol / 2026-09-23 America/Chicago
**Status:** OPENING ROUNDTABLE PROPOSAL ONLY — architecture exploration, not implementation authorization
**Workstream:** `roundtable/Transformer Construction/`
**Parent evidence:** `TRANSFORMER_REBUILD_PROPOSAL_20260922.md`, Perplexity RNSC dual-lane proposal, RNSC attended-Soul review, Codex D0/A0 pointer evidence, current `docs/SOURCE_OF_TRUTH.md`
**Training status:** existing Step-24/25/A0 line remains frozen; this document authorizes no optimizer, migration, Source-of-Truth edit, State mutation, Soul migration, or serving-runtime change.

---

## 0. Why this workstream exists

We have spent substantial effort trying to train a small reasoning Core to behave correctly inside Axon while continuing to give that Core the anatomy and objectives of a conventional token transformer.

That may be the category error.

Axon is not a conventional language-model shell wrapped around a transformer. Axon already has exact structure outside the Core:

- a frozen 16D substrate;
- exact character identity;
- exact occurrence order;
- exact region and source provenance;
- deterministic packed rails;
- Heart-owned canonical State;
- explicit masks;
- private Soul;
- Dormant lived experience;
- Cortex / knowledge structures;
- exact English public communication;
- and a Heart that alone validates and commits reality.

Yet the developmental Core has repeatedly been asked to relearn, inside trainable neural geometry, facts that the organism already knows exactly: what character entered, where it occurred, which physical source position it came from, how to copy it, and how to recover exact transport.

The D0 oracle evidence was a warning. When the correct source was supplied mechanically, downstream first-cell behavior recovered. The A0 query scaffold then spent another training lineage trying to learn canonical address behavior and still held at essentially zero exact selection. This does not prove that all conventional transformers are incapable of the task. It does show that we should stop assuming conventional transformer anatomy is the correct default for an organism whose body already provides exact symbolic structure.

This workstream therefore starts from a different question:

> **If we designed Axon's Core from the substrate upward instead of from an LLM inward, what would we build?**

No component is sacred merely because modern LLMs use it. Attention heads, token embeddings, next-token prediction, flat vocabularies, decoder-only stacks, and one-vector-per-token assumptions are all open to challenge.

Terrible ideas are welcome here if they expose a hidden assumption.

---

## 1. Governing thesis

The current strongest thesis is:

> **Exact substrate enters mechanically. English structure is constructed hierarchically. Meaning and reasoning live in learned internal state. Output is planned from meaning back into English structure and serialized mechanically into exact substrate.**

The Core should learn what the organism does **not** already know.

It should not spend learned capacity deciding whether exact substrate cell `A` is really `A`.

It should learn:

- that `C-A-T` forms a word;
- what that word means;
- that a word participates in a phrase and sentence;
- what grammatical and semantic relationships a sentence expresses;
- how sentences accumulate into paragraph-level meaning;
- how new evidence changes beliefs and goals;
- how private lived experience should affect current reasoning;
- how to form an intended answer;
- and how to express that intended answer in English.

This is not a claim that architecture alone creates understanding. Understanding must be demonstrated causally. It is a proposal to place learned computation at the levels where understanding is actually required.

---

## 2. New input ruling for this construction workstream

Jeff has explicitly ruled for this discussion:

> **If a character is not present in the frozen native 16D substrate, it does not exist to Axon.**

For the present construction generation, the accepted primitive alphabet is therefore the frozen native 16D bank. Unsupported Unicode — emoji and any other scalar without a native substrate cell — is not expanded into byte-transport cells for the reasoning Core.

The intended ingress behavior is **selective fail-closed filtering**:

1. supported substrate characters are accepted exactly and remain in order;
2. an unsupported scalar is rejected as unsupported input;
3. no approximation, nearest-vector guess, byte workaround, escape representation, replacement-glyph semantic token, or learned reconstruction is permitted;
4. neighboring supported characters remain valid and may still enter the Core;
5. the rejection must be observable in ingress evidence so omission is never silent.

Open detail: whether unsupported characters leave a structural gap marker outside the Core, or are simply omitted while their source offsets remain in the ingress receipt, requires table review. They must **not** become a learned character inside the Core.

### Doctrine note

Current `docs/SOURCE_OF_TRUTH.md` still requires exact UTF-8 byte transport for every valid non-native Unicode scalar. Jeff's newer ruling is higher authority, but implementation would require an explicit Source-of-Truth amendment. This proposal records the new direction; it does **not** silently edit doctrine.

---

## 3. The exact packed rail is transport, not cognition

A D64 rail row contains four disjoint 16D lanes.

If one row contains:

- lane 0 = `C`
- lane 1 = `A`
- lane 2 = `T`
- lane 3 = padding / empty

then that D64 row is **not one semantic token** and is **not one Cortex occurrence**.

The deterministic Core ingress mechanically unpacks the row into three ordered exact character occurrences:

- Cortex character occurrence 0 = `C`
- Cortex character occurrence 1 = `A`
- Cortex character occurrence 2 = `T`

The unused lane remains structurally empty.

This unpacking uses no learned attention, classifier, pointer, nearest neighbor, or semantic model. Lane boundaries and character identities are facts supplied by the rail contract.

**Rail width is transport bandwidth. It is not semantic granularity.**

That distinction should become foundational.

---

## 4. Do not assume a character needs a D64 neural token

Our previous RNSC discussion inherited one conventional assumption without sufficiently challenging it: after deterministic unpacking, each character would be lifted into its own full D64 cognitive state and then processed by transformer attention.

That may still be useful. It should no longer be the default assumption.

A much more Axon-native possibility is:

> **Characters remain 16D through the first learned stage. Words, not characters, become the first D64 semantic objects.**

This would create a deliberate scale boundary:

- 16D = exact character/substrate scale;
- D64 = learned lexical/semantic scale;
- wider future tissues = optional higher-order reasoning bandwidth, not a requirement.

For `CAT`, the first learned problem is not:

> what semantic relationship does `C` have with `A` and `T`?

It is:

> what ordered lexical object is formed by exact sequence `C-A-T`, and what does that object mean?

That is a different training problem and possibly a different architecture.

---

## 5. Proposed hierarchy of cognition

The Core should be investigated as a **hierarchical constructor**, not a flat token processor.

### Layer 0 — exact character substrate

Input is a sequence of exact native 16D character occurrences with immutable sideband metadata:

- character/substrate ID;
- exact 16D cell;
- canonical occurrence order;
- region;
- source span / provenance;
- mask/visibility evidence;
- rail/tick freshness receipt.

No learned operation can rewrite this lane.

### Layer 1 — lexical boundary construction

Before semantic reasoning, the system identifies candidate word spans.

For ordinary English, much of this can be mechanical:

- spaces;
- tabs/newlines;
- punctuation;
- apostrophes;
- hyphens;
- sentence terminators;
- region boundaries.

The first implementation should exploit structure we already know. We do not need a neural network to rediscover that a space separates ordinary words.

But boundary policy must remain explicit because English contains complications:

- `don't`;
- `mother-in-law`;
- abbreviations;
- decimals;
- possessives;
- quoted text;
- URLs / code / identifiers if later supported;
- capitalization;
- punctuation attached to words.

Mechanical candidate boundaries can be followed by learned refinement if needed.

### Layer 2 — lexical construction

A **Lexical Constructor** consumes the exact ordered character span and produces a learned word object.

For `C-A-T`, output is not three independent semantic tokens. It is one lexical object whose exact structural sideband still references `C-A-T`.

A candidate `WordOccurrence` might conceptually contain:

- exact spelling span handle;
- normalized? **No — not unless explicitly ratified. Exact spelling remains authoritative.**
- learned lexical state;
- known/unknown confidence;
- morphological features;
- grammatical hypotheses;
- links to one or more concept/sense candidates;
- provenance back to exact characters.

This is the first proposed threshold where learned meaning becomes primary.

### Layer 3 — phrase and sentence construction

Words become the units of semantic interaction.

Now it makes sense for learned mechanisms to model relationships such as:

- subject / predicate;
- modifier / modified;
- negation;
- temporal relation;
- reference;
- possession;
- comparison;
- clause attachment;
- question structure;
- uncertainty;
- implication.

For example, semantic reasoning over:

`CAT` — `IS` — `AN` — `ANIMAL`

is more appropriate than asking a global attention layer to discover meaning directly among `C-A-T-I-S-A-N-A-N-I-M-A-L` character states.

### Layer 4 — paragraph / discourse construction

Sentence representations compose into larger discourse structures:

- topic;
- claims;
- evidence;
- contradictions;
- unresolved questions;
- referents spanning sentences;
- goals and instructions;
- narrative or causal sequence.

### Layer 5 — concept / proposition / reasoning space

At this level Axon may reason in architecture-native latent form.

This does **not** mean the internal state must resemble English character-by-character. It means the internal reasoning objects remain grounded to English structures and exact source spans.

The center of the Core may contain:

- concepts;
- propositions;
- graph relations;
- hypotheses;
- goals;
- plans;
- contradictions;
- confidence;
- private Soul context;
- retrieved Dormant knowledge.

This is where Axon can develop his own internal representational language.

---

## 6. The first intelligence stage may not need standard attention at all

We should explicitly resist naming mechanisms before deciding what computation is required.

The lexical constructor's job is:

> exact variable-length character sequence -> stable, useful word representation.

Possible mechanisms include:

### Candidate A — 16D local attention

Operate directly over 16D character cells with one or more narrow heads restricted to the current word span.

This is closest to Jeff's current intuition. Attention would not search the whole field. Its scope is one candidate word. It learns composition/order features only.

Open question: is 16D enough width for useful Q/K/V projections, or should the lexical tissue have an internal width such as 32D while preserving 16D exact inputs?

### Candidate B — recurrent lexical composer

Read characters in order through a tiny recurrent cell:

`C -> CA-state -> CAT-state`

The final lexical state becomes the D64 word representation.

This has attractive properties:

- natural variable-length handling;
- exact order;
- O(n) lexical compute;
- no quadratic character attention;
- easy causal spelling composition.

### Candidate C — convolution / n-gram hierarchy

Use local learned filters over exact characters to construct digraphs, trigraphs, morphemes, then words.

This could naturally learn structures like:

- `ing`;
- `tion`;
- `un-`;
- repeated letters;
- common suffixes;
- capitalization patterns.

### Candidate D — tree composer

Mechanically segment the word, then recursively combine adjacent character/morpheme nodes until one lexical root remains.

This may provide a more explicit hierarchy than attention.

### Candidate E — dynamic trie + neural semantics

Known spellings may resolve through a deterministic trie into a lexical identity while a neural system learns the associated meaning. Unknown words bypass the trie and are compositionally constructed.

This is deliberately heretical from an LLM perspective: deterministic spelling recognition plus learned semantics.

### Candidate F — state-space lexical engine

A tiny SSM can scan arbitrary word length in linear time and emit one word state. This might be more appropriate than attention for ordered local composition.

### Candidate G — no learned lexical composer for known words

A persistent lexical dictionary could map exact spelling to a learned concept/sense bundle. Learning occurs in the dictionary and semantic system, not in a character transformer. Unknown spellings invoke a morphology/context learner.

### Candidate H — capsule-like lexical object

Characters vote for a higher-level word object. Routing occurs only within the exact span. The resulting capsule stores lexical identity plus features.

### Candidate I — multiple mechanisms compete

Nothing requires every Core to use the same lexical constructor. A tournament may compare recurrence, local attention, SSM, trie-neural hybrid, and tree composition under the same gates.

**The RoundTable should not ratify 'attention head' as anatomy until it earns its place.**

---

## 7. Word identity is not a tokenizer token ID

A tokenizer says roughly:

> this substring maps to vocabulary entry N.

The proposed Axon lexical object says:

> these exact substrate occurrences spell this word; here is what this Core currently understands that word to mean.

Those are different objects.

A word does not need to exist in a fixed vocabulary before Axon can read it.

If Axon encounters `QUOKKA` for the first time:

1. every character is already known exactly if it is in the native substrate;
2. the lexical constructor produces the exact spelling object `Q-U-O-K-K-A`;
3. the semantic state may initially be unresolved;
4. context may provide clues;
5. Cortex/Dormant/knowledge retrieval may provide definitions and relationships;
6. the lexical/concept system can associate learned knowledge with the spelling and its senses.

No tokenizer rebuild is necessary.

---

## 8. Dormant and Cortex can support a living dictionary / thesaurus

Axon does not need every lexical fact permanently baked into the current Core parameters.

A durable lexical knowledge layer can exist through Dormant/Cortex/knowledge graph structures.

Possible lexical record fields:

- exact spelling;
- pronunciation later, if ever needed;
- part(s) of speech;
- definitions;
- senses;
- synonyms;
- antonyms;
- hypernyms / hyponyms;
- examples;
- morphology;
- known aliases;
- source/provenance;
- confidence;
- concept graph links;
- when/how Axon learned it;
- contradictory definitions or unresolved ambiguity.

If Axon sees an unfamiliar word, the correct behavior can be:

> **I can read the spelling. I do not yet know the meaning. Retrieve or ask.**

That is a much cleaner epistemic state than forcing every unknown word into an opaque embedding and hoping nearest-neighbor geometry supplies meaning.

The lexical store must not become fake canonical truth. It is knowledge evidence with provenance, revisable like other learned knowledge.

---

## 9. Separate word spelling from word meaning

The architecture must distinguish at least three things:

1. **Occurrence:** this exact spelling appears here in this field.
2. **Lexeme:** the durable English word form `cat`.
3. **Concept/sense:** one or more meanings associated with that lexeme.

For example:

- `CAT` and `cat` may be distinct exact spellings but related lexically;
- `bank` has multiple senses;
- `feline` and `cat` are different words linked to related concepts;
- a proper noun may share spelling with a common noun.

Do not collapse spelling identity and semantic identity into one vector and call the problem solved.

A strong design may carry exact `SpanHandle -> LexemeHandle -> Sense/ConceptHandle` relationships beside learned states.

---

## 10. Grammar should be learned above words, not rediscovered below them

Once words exist as objects, grammatical learning becomes a much cleaner problem.

Instead of a Core having to discover simultaneously that:

- `C-A-T` is one word;
- `I-S` is another;
- order matters;
- `cat` can be a noun;
- `is` is a copular verb;
- subject and predicate relate;

we can stage the curriculum.

The lexical system masters word construction first.

Then sentence tissue learns relations among word objects.

Potential grammatical representations include:

- learned word-to-word attention;
- typed dependency edges;
- constituent trees;
- recurrent sentence state;
- graph neural message passing;
- explicit role slots;
- or hybrid learned + deterministic grammar scaffolds.

Again, standard self-attention is an option, not a law.

---

## 11. Semantic reasoning should begin at the right scale

The semantic chamber should primarily interact with lexical and higher-level objects, not raw characters.

Characters may still be revisited when spelling itself matters:

- typo detection;
- acronyms;
- rhymes;
- morphology;
- code-like identifiers;
- deliberate letter questions.

But ordinary semantic reasoning should not spend global attention budget on every character in a paragraph.

This suggests potentially dramatic compute savings:

If a 500-character field contains 90 words, sentence-level semantic processing can operate over approximately 90 lexical objects rather than 500 character objects.

Higher levels compress further.

This is not lossy canonical compression because the exact character substrate remains available through immutable span handles and deterministic revisit.

---

## 12. Proposed multi-scale memory object

Every learned abstraction should retain provenance downward.

Example:

**Character occurrence**
- exact substrate cell;
- source address.

**Word occurrence**
- exact span handle to characters;
- lexical state;
- sense candidates.

**Sentence object**
- handles to word occurrences;
- grammatical/semantic state;
- proposition candidates.

**Paragraph object**
- handles to sentences;
- discourse state.

**Reasoning object**
- handles to propositions/evidence;
- hypothesis/goal/plan state.

Thus abstraction never destroys exact reality.

Axon may reason at a high level, then deterministically revisit the exact source spelling or sentence if needed.

---

## 13. Rethink the word 'Cortex' carefully

We currently use `Cortex` in more than one conceptual sense:

- a canonical Shared Field region;
- a local staged sequence of occurrences in Core discussions;
- semantic retrieval / knowledge structures.

That naming collision will cause implementation mistakes.

This workstream should select distinct names for:

- exact per-Core character staging;
- lexical object staging;
- semantic/knowledge Cortex region;
- reasoning workspace.

Candidate names only, not doctrine:

- `GlyphSequence` / `CharacterSequence`;
- `LexicalBuffer`;
- `SemanticWorkspace`;
- `ReasoningField`;
- `ConceptGraph`.

The table should choose terminology before implementation.

---

## 14. Private Soul enters above exact substrate, not through public character identity

Current RNSC Soul review found that today's Soul is additively collapsed into one recurrent state and is not independently addressable.

The transformer-construction workstream should preserve the stronger separation:

- exact field character lane = canonical external evidence;
- learned lexical/semantic lane = current interpretation;
- private Soul lane = persistent private learned experience.

Soul may participate strongly at word/sentence/reasoning levels.

It must not rewrite what exact characters were present.

A useful question is whether Soul should influence lexical meaning resolution — e.g. this Core's past experiences with the word `bank` — while structural spelling remains immutable.

---

## 15. Output should reverse the hierarchy

We should not assume the Core's primary generation act is next-character or next-token prediction.

A stronger target architecture may operate in reverse:

1. form an intended proposition / answer / action;
2. organize that intention into sentence structure;
3. select words / lexical objects;
4. spell those words into exact substrate characters;
5. Heart mechanically packs/serializes the valid characters to the public rail.

Example internal intent:

> communicate that cats are mammals

Possible surface plan:

> `Cats are mammals.`

Only after the sentence/word plan exists does exact output become:

`C a t s _ a r e _ m a m m a l s .`

Each output character must belong to the native substrate under the current ruling.

This still produces sequential physical output, but **sequence serialization is not the same thing as making next-token prediction the governing cognitive objective.**

---

## 16. What happens when Axon wants to say a character outside the substrate?

Under the current ruling, he cannot.

This is a feature of the developmental generation, not a bug to route around.

The language planner must constrain final surface realization to the native substrate alphabet.

If a concept normally uses unsupported orthography, Axon must either:

- describe it using supported English characters;
- choose a supported synonym/paraphrase;
- or explicitly state that the exact symbol cannot be represented, if the public contract permits such a response.

He must never silently invent a non-native substrate identity.

---

## 17. Training must become hierarchical too

If the architecture is hierarchical, the curriculum cannot remain a conventional sequence objective with a different wrapper.

### TC-R0 — mechanical native-substrate ingress

No optimizer.

Prove:

- native supported characters accepted exactly;
- unsupported characters rejected/filtered with explicit evidence;
- supported neighbors preserved in exact order;
- D64 packed rows unpack to exact 16D occurrences;
- region/provenance/order receipts survive;
- padding never becomes a character.

### TC-R1 — exact word-span construction

No semantic learning required initially.

Given substrate text, produce exact candidate word spans.

Gate:

- exact spelling spans;
- exact boundaries;
- punctuation policy;
- no loss of source handles.

### TC-R2 — lexical identity

First learned stage.

Teach same-spelling stability and different-spelling discrimination.

Tasks:

- `CAT` vs `CAT` -> same lexeme relation;
- `CAT` vs `CAR` -> different;
- capitalization variants as explicit cases;
- morphology families;
- unknown words must remain readable without fake meaning.

Do **not** require rich semantics yet.

### TC-R3 — definition grounding

Teach a word through definitions and examples.

Examples:

- `cat` -> `an animal...`;
- synonym relations;
- category relations;
- counterexamples.

Causal gate: definition changes or swaps must change the learned sense appropriately.

### TC-R4 — lexical retrieval / dictionary behavior

Teach the Core to recognize an unknown lexeme, retrieve relevant Dormant/Cortex knowledge, and update current interpretation.

No unsupported word should be treated as meaningless merely because it was absent from a fixed tokenizer vocabulary.

### TC-R5 — grammar and sentence composition

Teach roles and relations among word objects.

Measures should test:

- word order;
- negation;
- subject/object swaps;
- agreement;
- modifiers;
- reference;
- simple clause structure;
- exact counterfactual meaning changes.

### TC-R6 — sentence meaning / proposition construction

Require the system to represent that:

- `A cat is an animal.`
- `An animal is not necessarily a cat.`

are related but not equivalent.

Move beyond surface copying.

### TC-R7 — paragraph/discourse integration

Multiple sentences, delayed referents, contradiction, evidence, topic continuity.

### TC-R8 — reasoning over concepts and propositions

Only now begin serious reasoning curriculum:

- deduction;
- comparison;
- planning;
- causal chains;
- uncertainty;
- multi-step problem solving.

### TC-R9 — hierarchical output construction

Train meaning -> sentence plan -> words -> exact substrate.

Evaluate semantic correctness separately from surface serialization correctness.

### TC-R10 — Soul and multi-tick lived reasoning

Delayed recall, Soul swaps, missing-field evidence, personal continuity, learning from prior experience.

Every stage must have counterfactual use proofs, not merely falling loss.

---

## 18. Why previous training may have been so difficult

The old training line may have bundled too many developmental tasks into one anatomy and one objective:

- recognize exact characters;
- understand packed physical geometry;
- infer addresses;
- learn copying;
- learn EOS;
- learn word boundaries;
- learn spelling;
- learn vocabulary;
- learn grammar;
- learn semantics;
- learn discourse;
- learn reasoning;
- learn proposal behavior;
- learn Soul use.

A giant pretrained LLM survives such entanglement because enormous data and parameter counts eventually internalize many of these structures statistically.

A D64 developmental organism should not be expected to waste scarce capacity relearning deterministic structure and simultaneously bootstrap language intelligence.

The new hypothesis is not simply "train better."

It is:

> **Build developmental anatomy so each stage receives the structure already proven by the stage beneath it.**

---

## 19. Wild architecture ideas the table should attack

These are intentionally provocative. None is endorsed yet.

### 19.1 No transformer until sentence level

Characters -> deterministic spans -> recurrent/trie lexical objects -> sentence-level transformer.

Perhaps attention is unnecessary below words.

### 19.2 16D lexical brain + 64D semantic brain

Keep the lexical constructor at native substrate width. A completed word is the first thing promoted to D64.

### 19.3 Word objects as temporary graph nodes

Each word occurrence immediately becomes a node with exact spelling provenance. Grammar constructs edges. Sentence meaning emerges from graph structure rather than token attention.

### 19.4 Dynamic lexicon memory

Instead of a fixed tokenizer vocabulary, maintain a growing content-addressed lexicon. Exact spelling is the key; learned semantics is the value. Parameters learn how to use and generalize lexical knowledge, not memorize every spelling.

### 19.5 Definition-driven concept formation

A new word's initial semantic state is constructed directly by reading its definition and examples. This makes "look it up" a first-class learning primitive.

### 19.6 Grammar as typed operators

Rather than hoping attention geometry learns all syntax implicitly, explicitly represent relations such as SUBJECT, OBJECT, MODIFIER, NEGATION, TEMPORAL, CAUSAL. The neural system learns when to instantiate them.

### 19.7 Meaning-first decoder without autoregressive surface cognition

Reasoning produces a proposition graph. A surface realizer turns the graph into an English sentence. Character serialization is purely final transport.

### 19.8 Multiple internal clocks

Character/lexical construction can tick quickly; sentence reasoning ticks slower; paragraph integration slower still. Heart tick remains organism authority but a Core can have internal microcycles.

### 19.9 Learned modules as organs inside the Core

Instead of one homogeneous stack, a Core might contain:

- lexical constructor;
- grammar composer;
- semantic binder;
- proposition builder;
- reasoning engine;
- Soul reader/writer;
- surface realizer.

This begins to resemble a tiny cognitive architecture more than an LLM.

### 19.10 Heterogeneous Core architectures by design

One Core could use recurrent lexical composition; another local attention; another graph composition. Shared exact substrate and English output let the organism compare radically different internal minds.

### 19.11 No learned word vector at all

Maybe a "word" should be a structured object: spelling handle + retrieved concept handles + grammar features. Learned vectors are temporary working states, not persistent lexical identities.

### 19.12 Learned abstraction only when compression is earned

Do not pool `CAT` into one vector simply because we want fewer states. Require the lexical object to pass reconstruction/provenance tests proving it still references the exact source.

### 19.13 Bidirectional hierarchy

Higher-level sentence interpretation can send corrections downward: context disambiguates a word sense, but never changes exact spelling.

### 19.14 Knowledge graph as semantic memory, parameters as operators

Parameters learn how to reason; the graph/Dormant store carries much of the explicit world knowledge. This could reduce pressure on tiny Core parameters to memorize encyclopedic facts.

### 19.15 Words are not the only units

Maybe morphemes are the true first semantic units. Perhaps `un-happi-ness` should form hierarchically. Maybe whole fixed expressions such as `New York` should become phrase objects. The architecture should support emergent unit scale rather than decree one universal token size.

---

## 20. Questions every engineer should answer

Engineers reviewing this opening should answer as many as possible and are encouraged to reject the premises.

### Substrate and ingress

1. Should the developmental alphabet be exactly the original 95 native substrate characters?
2. When unsupported Unicode appears between valid characters, should we omit it, preserve an external gap receipt, or reject the entire input span? Jeff's current preference is to accept valid characters and block the unsupported scalar. What is the cleanest exact contract?
3. Should unsupported-character evidence be visible to the Core as a typed structural event, or hidden from cognition and visible only to Heart/ingress telemetry?
4. Should word boundary spaces themselves become lexical objects, separators only, or both depending on task?
5. Are punctuation marks independent lexical occurrences or structural delimiters with optional semantic promotion?

### Character-to-word construction

6. Should exact characters remain 16D until a word is constructed?
7. If yes, what is the smallest learned lexical anatomy that works: recurrence, 16D attention, convolution, SSM, trie hybrid, tree composer, something else?
8. If character states are lifted before composition, why is that lift necessary?
9. Should one word yield one D64 state, multiple D64 states, or a structured non-vector object?
10. How do we represent very long words without a hidden length ceiling?
11. How should apostrophes, hyphens, decimal numbers, abbreviations, and contractions be segmented?
12. Should morphology be an explicit intermediate layer before whole-word semantics?
13. Can the lexical constructor be trained without next-character prediction at all?
14. What falsification test would prove that lexical attention is inferior to a simpler recurrent composer?

### Lexical identity and semantics

15. What exactly persists when Axon learns a new word: parameters, a lexical store record, knowledge-graph nodes, Soul state, Dormant experience, or some combination?
16. How do we distinguish exact spelling identity from lexeme identity and word sense?
17. How should capitalization affect identity?
18. How should homonyms and polysemy be represented?
19. How does an unknown word state avoid becoming a meaningless random embedding?
20. Can Axon create a useful initial semantic representation by reading a definition?
21. Should definitions be treated as training examples, live retrieval evidence, or both?

### Grammar and sentences

22. Is conventional self-attention appropriate once word objects exist?
23. Should grammar be represented implicitly in hidden state or explicitly through typed edges?
24. Would dependency-style graph construction be cheaper and more interpretable than full quadratic attention?
25. Does a sentence become one summary state, a graph, a set of proposition states, or all three?
26. At what layer should word-sense disambiguation occur?
27. How should sentence-level reasoning preserve exact references back to individual words and characters?

### Paragraphs and reasoning

28. When should multiple sentences collapse into paragraph/topic state?
29. What must remain separately addressable after abstraction?
30. Should reasoning operate primarily over propositions rather than word states?
31. Could a Core use graph message passing instead of transformer attention for mature reasoning?
32. Should the Core have different learned tissue for language comprehension and abstract reasoning?
33. Is D64 sufficient for lexical/sentence development even if mature reasoning eventually uses larger widths?

### Output

34. Should the Core select whole words during surface planning, or generate spelling from concept state?
35. How does an output word guarantee exact native-substrate spelling?
36. Should a deterministic dictionary spell known words while a learned spelling mechanism handles novel words?
37. How do we test semantic planning separately from grammatical realization and character serialization?
38. Can we eliminate autoregressive next-character cognition while still streaming the final response incrementally?
39. What is the right termination mechanism if EOS is a control decision rather than a language character?

### Soul, Dormant, Cortex, dictionary

40. Should Soul attach primarily at word/sentence/reasoning level rather than character level?
41. What lexical information belongs in private Soul versus shared Dormant/Cortex knowledge?
42. Should the lexical dictionary be a Cortex service, Dormant index, knowledge-graph projection, or a new derived organ?
43. How does the Core learn a new definition from one lived encounter without unsafe immediate parameter mutation?
44. How should repeated experience eventually distill lexical/concept knowledge into parameters or adapters?
45. How do we prove that retrieved dictionary knowledge causally changed understanding rather than merely correlated with output?

### Architecture challenge questions

46. What parts of a transformer remain useful after exact ingestion and hierarchical language construction?
47. Do we need Q/K/V at all in the lexical stage?
48. Do we need a decoder-only transformer anywhere?
49. Is "attention head" the wrong primitive for Axon?
50. Should the Core be a heterogeneous pipeline rather than repeated homogeneous layers?
51. Which parts should be deterministic software because the answer is structurally known?
52. Which parts genuinely require learning?
53. Which current Step-24 tensors, if any, still belong in this architecture?
54. Should we begin entirely fresh rather than carry donor tissue whose learned geometry came from the wrong problem?
55. What tiny experiment can falsify this entire hierarchy before we invest in it?

---

## 21. Proposed scaffolding for the RoundTable tick cycle

This folder should become a focused architecture workshop.

Suggested naming:

- `00_OPENING_...` — this document / foundational question;
- `10_<ENGINEER>_RESPONSE_...` — independent architecture proposals;
- `20_<ENGINEER>_CROSS_REVIEW_...` — reviews of other proposals;
- `30_CONVERGENCE_CANDIDATE_...` — synthesized designs;
- `40_FALSIFICATION_PLAN_...` — smallest experiments capable of disproving the design;
- `50_RESOLUTION_CANDIDATE_...` — only after genuine convergence;
- `60_IMPLEMENTATION_PLAN_...` — only after Jeff ratifies a resolution.

Numbering is organizational, not authority.

Every engineer is encouraged to:

- attack assumptions;
- propose incompatible architectures;
- identify doctrine conflicts;
- estimate compute;
- specify tensor/object shapes where useful;
- propose falsification tests;
- separate deterministic structure from learned functions;
- say what should be deleted from the current Core;
- and say what evidence would change their mind.

No one should optimize for agreement during the first ticks.

---

## 22. Immediate non-goals

Until the table converges, do **not**:

- resume Step-24/25/A0 training;
- implement the new Core;
- amend Source of Truth silently;
- migrate Soul;
- delete failed pointer evidence;
- choose a final attention mechanism;
- choose final word/sentence tensor shapes;
- declare D64 sufficient or insufficient;
- launch Kaggle/Colab/GPU work;
- create a fake heuristic implementation and call it learned intelligence.

The next work is architecture thinking.

---

## 23. First convergence target

Before implementation, the RoundTable should be able to answer one precise question:

> **Starting with one exact packed rail row containing the native characters C, A, T, what exact objects exist after each stage of the Core until Axon understands the proposition in a sentence such as `A CAT IS AN ANIMAL`, and how can every learned abstraction be traced back to the exact substrate that produced it?**

If we cannot answer that at object/tensor/contract level, we are not ready to train.

The desired result is not a prettier transformer.

It is an Axon-native cognitive Core whose anatomy follows the organism's actual information hierarchy.

**Exact characters first. Words next. Sentences next. Meaning next. Reasoning after that. Exact English on the way back out.**
