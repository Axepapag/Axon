# AXON v7 — Response Draft (Iterative, v6 → v7 Handoff)

**Status:** DRAFT. Living document. Will be refined over time.
**Author:** Jeff (architect) + Codex (scribe).
**Date opened:** 2026-06-10.
**Source of v6 truth:** `D:/ashes/AXON_SOURCE_OF_TRUTH_v6.md` (frozen as design contract; v6 implementation status lives in the handoff, not here).
**v7 lives in:** `D:/AxonGliksbot/` (currently empty; this draft will seed it).
**v6 lives in:** `D:/ashes/` (frozen as reference; will not be modified further except as documented in the handoff).

This document is the **v7 response draft** — a place where ideas land, get shaped, and only later graduate into the v7 source-of-truth. Nothing in here is law until Jeff says it's law. Nothing here is implementation. It is the conversation space between v6-as-built and v7-as-designed.

---

## §0. The One Sentence (v7)

[DRAFT — pending refinement. v6 had this:

> Axon is an always-ticking, tokenless, recursive state engine built from a frozen hand-built 8D canonical-alphabet substrate (the alphabet floor), a flock of diverse-size multi-head single-layer cores of many different shapes (each with its own projection head), a strictly-separated dormant/active state made of concept-slot bundles of arbitrary higher dimension (16D/32D/64D/128D/… paired with their frozen 8D letter sequences), a language workshop for rendering deliberate statements, a tick loop that runs semantic search *before* attention, and an idle memory refinery that turns real interaction traces into structured knowledge, prunes active state, and fills thin dormant bundles via dictionary lookups.

v7 changes to this sentence will be drafted below as the laws clarify.]

---

## §1. THE FIRST LAW — 16D Only

**v7 first law (stated 2026-06-10, Jeff):**

> No vector spaces other than 16D. No 64D. No 128D. No 256D. No 512D. No 1024D. **16D letter by letter. Done.**

All state in v7 lives in 16D. The substrate is 16D. The letter-slot dimension is 16D. The bundle's *identity* (its concept-slot) is 16D. The bundle's *edges* (relations to other bundles) are 16D. The bundle's *canonical letter sequence* is 16D. The core's d_model is 16D (or the core attends to 16D through a single non-trainable projection, equivalent in effect).

**Why this law exists.** In v6, bundles carried higher-dim concept slots (64D, 128D, 256D, …) and the cores' d_model was 128D. This meant the system had **two coordinate systems** at once: a 16D substrate and a higher-dim concept space. The cores learned in 128D. The renderer decoded from 16D. The bridge between them was a projection head — trainable, mutable, with its own degrees of freedom. The 16D substrate's guarantee ("the letter 'd' is the same 16D vector at step 0 and step 1,000,000") was preserved, but the **concept space was not 16D** — it was whatever d_model the core used.

v7 collapses the two spaces. **There is one space: 16D.** The letter 'd' is a 16D vector. The word "dog" is three 16D vectors arranged in a bundle. The concept of "dog" is a 16D vector attached to the bundle. The edge from "dog" to "is_a:noun" is a 16D vector. The core's whole inner life is 16D.

**Why this is hard.** A single 16D vector has only ~16 degrees of freedom. v6 had d_model=128 (8x more capacity) and d_model=1024 (64x more). To do meaningful work in 16D, the core has to use the **bundle structure** as its main capacity lever — *how many slots, how they're grouped, what they bind to* — rather than *how wide each slot is*. This is a fundamental shift in where the model "puts its intelligence."

**Likely consequences to address as v7 develops:**

- **Projection heads still exist; they're per-core, not per-(kind, dim).** Each core still has its own way of seeing and writing the substrate — its own projection head. The difference is that v6 had a *bank* of per-(kind, dim) heads (ingress/egress for letter16, word64, concept128, episode256, tool64, result64, scratch8, commandment128). v7 has *one* head per core, mapping the core's internal dim ↔ 16D. **Every core's projection head is a 16D ↔ d_core_model mapping.** A 2048D core still has a projection head, but the head's *interface* to canonical state is 16D, not 64/128/256. The 16D guarantee is preserved; the core can be wide internally; the complexity of "every kind has its own dim" is gone. **The first law is "all vectors in canonical state are 16D." The cores' internal dim is unconstrained.** (Jeff, 2026-06-10: "Axon of course he will have projection heads ... we can have big 2048 just through his projection head of course he has the same thing it's the same thing. The only difference is we're ditching the complexity and issues with using higher dimensional vector slots within axons vector space now he has one dimension to worry about 16D.")
- **No concept-slot graduation.** In v6, a bundle's concept vector grew in dimension as it matured. In v7, it's always 16D. Maturation is expressed in *edge count*, *observation count*, *cross-bundle reinforcement* — never in slot width.
- **Workshop rebuilt; no atom_dim, no workbench.** The v6 Workshop builds 64D word atoms. v7 has no Workshop as a separate concept: the word is made out of the substrate. There is no separate atom table, no `atom_dim` parameter, no per-word learned vector. (Jeff, 2026-06-10: "no matter what there is no mapping a word to the substrate because the word is made out of the substrate so there is no workbench none of that all it is is the alphabet.")
- **d_model vs SLOT_DIM unification.** v6 had `core.d_model` (128, 256, …) and `letter_substrate.SLOT_DIM` (16) as two different numbers. v7 has one: `SLOT_DIM = 16`. The core's d_model *is* the substrate's slot dim. They're the same thing.
- **The substrate's free 16D dimensions are usable.** The 16D letter basis has 16 dimensions. The alphabet's 26 letters (+ special characters, digits, etc.) likely don't need all 16. The unused dimensions can carry **other meaning** — e.g. part-of-speech markers, tense markers, semantic-class tags, identifier slots for sub-bundle references. This is the "extra space" Jeff mentioned: "whatever room we have left in the dimension space which I'm sure we should have plenty we can have numbers one through 9 special characters you know we we can assign identifiers you know we can create meaning for the extra space." **v7 design task: figure out what each of the 16 dimensions encodes for letters**, beyond the alphabet's minimum.
- **Affixes occupy the same 16D as letters.** Suffixes like `ing`, `es`, `un`, `re`, `pre` are *letter sequences* in v7, not separate tokens. They get the same 16D letter-by-letter treatment. There is no "subword tokenizer" or "morphology detector" — those would violate the substrate law. (Jeff, 2026-06-10: "I forget exactly what it's called but it's like ing es un re ... they could fit [as letter sequences].")
- **Core architecture: open.** v6 cores are TransformerLayer × 2, 128D, 1 head, 16384 FFN. With 16D as the canonical interface, a core's *internal* shape is its own business. A 2048D core can attend over the 16D field by projecting 16D → 2048D on ingress, doing its 2048D work, then projecting 2048D → 16D on egress. The SwiGLU FFN's hidden width is now *core-internal*, not canonical. v7 design question (deferred to architecture doc): what is the right internal core shape given that the interface is 16D?

**Future-slot escape hatch (Jeff, 2026-06-10):**

> "Later on down the road most likely going to introduce the higher vector slots for better latency and other benefits but the first law is going to be only 16D we build bundles"

This means v7 may eventually grow a second slot-dim layer (32D? 64D? 1024D?) for the latency/throughput reasons v6 gave (a 64D word-atom in v6 could be processed in one vector op instead of four 16D letter ops). **But that is NOT v7's first law.** v7's first law is 16D only. The higher dims are a v8 (or later) conversation, and they must be **additive** — they cannot break the 16D guarantee, only sit on top of it.

---

## §2. THE SECOND LAW — A Dog Is a Bound Bundle, Not Three Letters in a Row

**Drafted from Jeff's explanation, 2026-06-10:**

> "the first three are just loose letters that happened to line up to spell dog but the second example there's something that binds them together to make them one unit and it's a dog"
> "if we are able to bind the slots together greater bundles then that's probably better anyway we have a lead address for what the word is. Then we bolt on semantic meaning around that word and it'll be a lot more precise instead of trying to navigate the space of a larger vector like how does it all fit well 16D gives you no extra room to navigate every semantic edge is extremely deliberate spelled out vector by vector bolted on to a greater bundle so you can have a three letter word like dog ultimately carrying 100 vector slots bolted to it an entire semantic neighborhood and if they're bolted that's where my question came in if truly bind the vectors into a single object well then we're golden that's better anyway it's it's it's higher resolution as a matter of fact and no matter what there is no mapping a word to the substrate because the word is made out of the substrate"

**The law, as I read it:**

A bundle in v7 is a **single object in canonical state**, not a sequence of independent slots. The bundle has:

- **A lead address** — the canonical letter sequence of the surface (e.g. for "dog": the three 16D vectors for `d`, `o`, `g`). This is the bundle's *name* — the thing other bundles reference by.
- **A semantic neighborhood** — many 16D vectors bolted to the bundle, each one an edge to a related concept, attribute, observation, or sub-bundle. For "dog", this could be 100+ slots: `is_a:animal`, `is_a:pet`, `is_a:noun`, `has:four_legs`, `says:woof`, `past:dogs`, `plural:dogs`, `comparative:`, `superlative:`, edges to other animal-bundles, edges to specific dogs Axon has seen, edges to sentences "dog" has appeared in, and so on. **Every semantic edge is a deliberate 16D vector, not derived from a learned lookup.** The neighborhood grows by observation, reinforcement, and the consolidator's deliberate work; it never comes from a teacher's projection.

**Why "16D gives you no extra room to navigate" is a feature, not a bug.**

In a 128D concept slot, the model can "hide" semantic content in dimensions that aren't load-bearing — there's enough room that some dimensions carry signal and some carry noise. In 16D, every dimension must carry signal. **Each semantic edge is extremely deliberate; spelled out vector by vector; bolted on to a greater bundle.** This is higher resolution, not lower — the model can't be vague about what an edge means, because it doesn't have the dimensions to be vague.

**The bundle is bound, not coincident.**

When the consolidator or an online core attends over canonical state, a bundle is one **attentional unit**, not a sequence. The lead address slots participate in attention as a *key*. The semantic neighborhood slots participate as *values*. The bound bundle is: "here is a unit I can read; its lead address is its name; its neighborhood is everything I know about it." This is structurally different from "three letters that happen to be adjacent" — and that difference is what makes "dog" a single object rather than a coincidence.

**Open sub-questions for v7 design:**

- **How is the binding represented in 16D?** Three options I see:
  1. A **binding slot** at the head of every bundle — a 16D vector that means "the next N slots are a unit of kind K, lead address is the next M of those slots, neighborhood is the rest." The binding slot is part of the slot data, not metadata.
  2. A **fixed structural convention** — every bundle is `[role_marker, lead_address_1, …, lead_address_M, neighborhood_1, …, neighborhood_K]`. The binding is positional, not a separate vector. Attention learns the convention.
  3. **A 16D "closure" edge** — the bundle's lead address slots are themselves connected to the neighborhood slots by deliberate 16D edges, and the binding emerges from the edge structure.
  I don't yet know which Jeff prefers. This is the next design question.

- **Can a bundle's neighborhood grow without bound?** Yes. A "dog" bundle observed 10,000 times in 10,000 contexts could carry 10,000 semantic edges. The first law doesn't cap edge count. The consolidator decides which edges to keep, which to prune, which to reinforce.

- **Is the lead address the same 16D vectors as the letter substrate?** Yes. For "dog", the lead address is the substrate-encoded `d`, `o`, `g` vectors, in order. **The word's name is its letters.** There is no separate "word embedding." A bundle's identity is its letters; its meaning is its neighborhood.

- **What about sub-bundles?** A phrase like "the dog" is a bundle whose lead address is the letters `t`, `h`, `e`, ` `, `d`, `o`, `g` and whose neighborhood includes an edge to the "dog" sub-bundle. **Sub-bundles are referenced by 16D edges, not by name strings.** A phrase's neighborhood can contain edges to multiple word-bundles, and the attention over the field uses those edges to bind the phrase's meaning to the word-bundles' neighborhoods.

---

## §3. THE THIRD LAW (drafting) — The Substrate Gate is Bidirectional

**Drafted from Jeff's clarification, 2026-06-10:**

> "the renderer the substrate gate is what we should call it 'cause it's bidirectional when text hits it shits out 16d on the other side visa versa when 16D hits it out comes text letters"

**The component, named:** the **Substrate Gate**. It is the boundary between canonical state (16D vectors) and the human-readable world (text). It is **bidirectional**:

- **Inbound (text → 16D):** every text token that enters the system — Jeff's typing, a dataset record, a command-line argument, a log message — is run through the gate and emerges as 16D vectors. The gate is the only place text is allowed to exist as text in v7. Past the gate, everything is 16D.
- **Outbound (16D → text):** every text that leaves the system — the response draft, a render log, a training example, an exported state view — is run through the gate in reverse: 16D vectors in, characters out, via the LetterBank's decode.

**Why the rename matters.** "Renderer" suggests output-only. "Substrate Gate" says: this is a boundary that protects the substrate from text in both directions. **The substrate is the invariant. The gate is the only translator. Text is the visitor.**

**Implications for v7:**

- The Substrate Gate is a single named component with two methods (encode, decode). The implementation may change between v6 and v7; the contract is the same.
- Every code path in v7 that touches text must call the gate. There is no other way.
- The gate's encoder is **deterministic and hand-built** (the substrate's letter basis is frozen). The gate's decoder is **learned** (the LetterBank's centroid vectors are learned from training, even if the letter basis is fixed). The bidirectional contract guarantees that encode → decode is *approximately* faithful (round-trippable for the alphabet, lossy for novel 16D vectors).
- v6's "renderer" and "letter bank" become a single component: the Substrate Gate, with `encode(text) -> field` and `decode(field) -> text`.

## §4. THE FOURTH LAW (drafting) — The Response Draft is Deliberate Composition, Not Retrieval

**Drafted from Jeff's clarification, 2026-06-10:**

> "What does the renderer see the response draft who authors the response draft Axon will ... Axon will bolt a 100 vector slot bundle to the response draft if he wants to say dog of course not those 100 vector slots are semantic edges for his reasoning they stay in the canonical state the response draft is deliberate Axon will compose his response just like I had to do and I have to do now I don't to hit a key or you know with my pencil I make a weird symbol and that means you know a higher dimensional word with all this extra meaning no I have to literally type out or handwrite each letter"

**The law, as I read it:**

- The response draft is **Axon's deliberate composition**, character by character, slot by slot. It is not a retrieval from dormant. It is not a nearest-vector lookup. It is not a search result. **The consolidator writes it deliberately through his own egress head, into the draft's frozen-dim letter slots.** The Substrate Gate then decodes those slots to text for Jeff.
- The semantic neighborhood (100+ edges for "dog") **stays in canonical state**. Those edges are for *reasoning*, not for output. When Axon reasons about dogs, he attends over the dog-bundle's neighborhood. When he writes "dog" in the response draft, he writes the three lead-address letters deliberately — same way Jeff has to type out or handwrite each letter, even though "dog" is a higher-dimensional concept in Jeff's head.
- **The response draft is a window into Axon's reasoning, not a dump of his reasoning.** Axon can be thinking about a thousand things; the draft shows what he chose to say. The choice is itself a deliberate act.

**Implications for v7:**

- The draft is in canonical state as **16D letter slots only**. No edges, no neighborhood, no sub-bundle references in the draft.
- The consolidator's egress projects to 16D, writes letter-by-letter, with each letter being a deliberate 16D choice informed by the consolidator's current hidden state (which is informed by attention over the field — which is where the semantic neighborhood lives).
- This is exactly v6's Generation Law restated: "the draft is the consolidator's own writing." v7 keeps the law; the only thing that changes is the dim (16D instead of 128D).

---
## §5. Things From v6 That v7 Inherits (Confirmed)

These are v6 design choices v7 keeps, with refinements:

- **Substrate is frozen, hand-built, deterministic.** v7 keeps; the Substrate Gate is the named boundary.
- **No teacher, no embedding table, no hash-based lookups.** v7 keeps; the Workshop/workbench concept is retired.
- **Multi-core flock** with role rotation (online / consolidator / offline). v7 keeps. **Per-core d_model is unconstrained; the canonical interface is 16D.** Cores can be 128D, 2048D, or whatever; their projection head maps 16D ↔ d_core_model.
- **Generation Law** (consolidator's second pass, separate rows, never averaged). v7 keeps; same as v6.
- **Substrate Gate is the only way out.** v7 keeps (and renames: from "renderer" to "Substrate Gate").
- **Active state / dormant state separation.** v7 keeps.
- **No raw text in canonical state.** v7 keeps; the Substrate Gate is the only path.
- **Dump bucket, no auto-write to bucket.** v7 keeps; the bucket is sacred, written only by deliberate named actions.

## §6. Things From v6 That v7 Retires (Anti-Patterns)

[Pending — these will be lifted from the v6 handoff when the handoff is written. For now, placeholder section.]

## §7. Open Questions for v7

1. **What is a core's internal shape, given d_canonical = 16D?** Per-core d_model is unconstrained. The architecture is open. (Defer to architecture doc.)
2. **How is the bundle's binding represented in 16D?** Three options in §2; Jeff to pick.
3. **What do the unused 16D dimensions encode?** Per §1, the alphabet doesn't need all 16; the extras are usable for markers, identifiers, semantic-class tags. (Defer to substrate-gate spec.)
4. **How does a phrase's neighborhood bind to its sub-bundles' neighborhoods?** Per §2, sub-bundle references are 16D edges, not name strings. (Defer to bundle-structure spec.)
5. **What replaces the v6 loss family's variance/covariance/head-entropy terms at 16D scale?** v7's offline trainer operates in 16D. Some loss terms may collapse or be redefined. (Defer to trainer spec.)
6. **What is the input encoder for non-Latin scripts?** The substrate is currently Latin-26 + specials. v7 may extend to other scripts, but they all encode through the same gate. (Defer to substrate-gate spec.)
7. **The "no auto-write" rule for the bucket: how is it enforced?** A test? A code-review checklist? A runtime check that the bucket-write path requires a named operation? (Defer to the handoff and to v7's runtime spec.)

## §8. What v7 Will NOT Be

- v7 will **not** be a wrapper around an LLM. There is no fallback to GPT-anything. The cores are the system.
- v7 will **not** have a token vocabulary. Letters are the only input tokens, and they're 16D vectors, not integer IDs.
- v7 will **not** have a "stop" condition. The runtime ticks forever, or until the user kills it.
- v7 will **not** auto-populate any state. All state writes are deliberate.
- v7 will **not** have caps (no max token count, no max slot count, no time budget). Caps are a form of dishonesty.
- v7 will **not** have a workbench or atom table. The word is made out of the substrate; there is no separate per-word learned vector.
- v7 will **not** have projection heads at 64D / 128D / 256D. All canonical-state projection heads are 16D ↔ d_core_model. (Internal core dim is unconstrained.)

## §9. The Draft's Open Threads

- §0: the one-sentence description needs to be rewritten under the first law.
- §2: which binding representation Jeff prefers (the three sub-options).
- §6: pending v6 handoff.
- §7: questions are open design questions; each will move to a separate doc as the answer firms up.
- §8: uncontentious but needs to be promoted to "the law" once §§1-4 are locked.



## Change Log

- **2026-06-10 (a):** document opened. §1 (16D only) drafted from Jeff's instruction. §2 (bundles as bound units) drafted from Jeff's dog example. §5 (open questions) populated. §6 (what v7 will not be) drafted from v6 source-of-truth assertions that v7 inherits.
- **2026-06-10 (b):** §1 consequences rewritten — projection heads survive as per-core 16D ↔ d_core_model heads (Jeff clarified the 2048D core case); workshop retired (no workbench); free 16D dimensions usable; affixes are letter sequences; core architecture deferred. §2 expanded — bundle = lead address (letters) + semantic neighborhood (deliberate 16D edges); three binding-representation sub-options; sub-bundles referenced by 16D edges.
- **2026-06-10 (c):** §3 (Substrate Gate, bidirectional, named component) added. §4 (response draft is deliberate composition, not retrieval) added. §§5-9 reorganized; §5 confirmed v6-inherited; §6 placeholder for v6 handoff anti-patterns; §7 open questions refined; §8 expands "what v7 will not be" with the 16D-only projection heads and no-workbench; §9 tracks open threads.
