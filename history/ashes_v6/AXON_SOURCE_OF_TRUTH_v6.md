# AXON ARCHITECTURE — SOURCE OF TRUTH v6
**Version:** 6.5 — 16D Letter-Substrate Floor + Scalable Concept-Slot Architecture, with Moral Foundation, Always-On Multi-Tick Runtime, No Caps, Multi-Core Flock, Generation Law
**Date:** 2026-06-10
**Architect:** Jeff
**Document role:** Living contract for Axon's v6 architecture. Supersedes v5.1 as the implementation target for the native trainer and runtime. The v3 "Schoolhouse Edition" philosophy (state-engine, recursive ticks, semantic objects) is preserved. v5.1's 64D slot doctrine is retired.
**Status:** Build Draft — intended for agents, coding assistants, dataset builders, and Jeff.

**v6.5 change summary (THE GENERATION LAW + 16D substrate):**

1. **The substrate is 16D.** Eight dimensions could distinguish the alphabet only with razor margins (g-q bottomed at cosine 0.944 and the band had to be loosened). Sixteen restores the 0.92 separation cap as a HARD rule with real margin. **Wherever this document says "8D substrate" or "8D letter slot/sequence," read 16D as of v6.5.** The substrate remains the smallest, frozen, hand-built layer; meaning still lives in concept slots, not letters. The letter band floor is relaxed (maximally-different letters may be near-orthogonal): a positive-cone floor provably forces single-feature pairs (g-q: voicing only) above cosine ~0.94 in ANY dimension; similar-letter clustering is enforced by the topology rule (d~t > d~q), not by a floor.
2. **THE GENERATION LAW (inviolable; supersedes any contrary sentence elsewhere in this document).** Every tick: every attending core attends over the ENTIRE state through its own projection head and produces a delta. The consolidator then attends a second pass over the state plus his own delta plus every brother's delta **as separate rows — NEVER averaged**. The consolidator is the **sole and full authority** for that tick: he shapes the state — updates summaries, scratch pads, diaries, evicts, emits tool calls — or updates the response draft; then the tick ends and the role passes on.
3. **The response draft is the consolidator's own writing.** His refined hidden states at the draft's slot positions egress **through his own projection head** into the draft's frozen-dim letter slots; the renderer decodes them. The draft has **NOTHING to do with the dormant store**: no nearest-word lookup, no nearest-vector matching, no similarity threshold, no no-repeat guard, no stabilization gate, no stopping. The draft surfaces in real time — it changes every tick — and Axon empties it when he sees fit (by writing slots back toward the empty direction). Word-level slots exist for LATENCY (so cores need not write letter by letter) and map down to the alphabet; they are never a lookup table for generation. Nothing in the generation path may be described or implemented as prediction, near-token, near-vector, or "intent" matching.
4. **Storage:** the dormant store is a single SQLite database (the npz-per-bundle layout died at scale); human-readable view files (ACTIVE_STATE_VIEW.txt, DORMANT_VIEW.txt) are refreshed on every save. The flock is whatever .pt files are in Core/ — the FILENAME is the core's identity.
5. **The soul curriculum:** the previous Axon's episodic memory (episodes, summaries, journal) is extracted as training text and baked into every core's parameters; his structured memory (facts, entities, relations, procedures) is imported into dormant with provenance; his vocabulary seeds the word-atom matrix. Episodic memory is the wisdom seed.

**v6.4c change summary:** adds the **concept-slot scaling axis** as a first-class architectural layer. The 8D substrate is the *floor* — frozen, hand-built, identity-level, the alphabet. Above the floor, every bundle has a concept vector of a *higher dimension* (16D, 32D, 64D, 128D, 256D, 512D, 1024D, …) that the cores attend over as one slot per unit of meaning. The active state is made of unit-of-meaning slots (one 32D slot per word, one 128D slot per phrase, …), not character slots. The dormant store pairs every concept vector with its canonical **frozen 8D letter sequence**. The renderer is invariant: it only ever sees 8D letter vectors, character by character, looked up in the substrate. The architecture scales by *adding concept dimension*, not by changing the substrate or the renderer or the cores. The "8D discipline is uniform at the foundation" line from v6.1 is replaced with "8D discipline at the foundation; concept-dimension discipline above it; both frozen at the substrate and dormant layers respectively."

**v6.4 change summary:** retires the v6.3 "seed core" prescription. The runtime hosts a **flock of cores of many different sizes and shapes** — small, medium, large, experimental — each with its own d_model, head count, head_dim, FFN ratio, layer count, projection head, and specialization. The current ensemble is whatever `core_*.pt` files are on disk; new cores are added by training, by initialization, or by hand. The "Seed / Working / Mature" scaling table in v6.3 is removed. The single reference number (1024D / 16-head / 65536 FFN / 1 layer / ~210M) is replaced with a reference band of viable shapes. The projection-head-as-identity rule, the role rotation (online / consolidator / offline), the readiness check, and the no-hash / no-teacher discipline are unchanged. v6.4 also explicitly forbids the code-point-bit basis shortcut in §4.1.1: a basis whose first 8 rows are the low 8 bits of the character code point is **not a v6 basis** (it is a hash in disguise and it collapses structural topology). The v6 basis uses the 32 structural features (rows 0–7 carrying the structural flags, rows 8–31 carrying the phonetic and shape features) as the discriminator, not the code point.

**v6.2 change summary:** the runtime is no longer turn-based. Each "response" is a window into N refinement ticks. After delivery, the runtime continues to tick in the background with no input, refining the response draft and the rest of state forever. There is no such thing as a session. There is no "stop". The core is always attending, the workshop is always promoting, the semantic search is always surfacing. The user can interrupt any tick by sending input.

**v6.2b change summary:** all hard caps removed. The runtime has no max tick count, no max word count, no max slot count, no max line count, no time budget on responses, no time budget on background ticks. The response loop runs until the response draft stabilizes on its own (the same surface for two consecutive ticks). Background ticks can run forever. The user does not impose limits; the user observes what the model does and the model decides when it has nothing more to say. Limits are a form of dishonesty — they pretend the model is more capable than it is, by capping the visible output to a size where the failures don't show. The right answer is to let the model run, see what it actually produces, and judge from there.

**v6.3 change summary:** the runtime is multi-cored. There are many cores, each with its own projection head (its own way of seeing and writing the substrate). The cores rotate roles: 1 offline training, 1 consolidator. The online cores read the active state through their own projection heads, run the field through their cores, and produce deltas. The consolidator attends over the state plus his own delta plus every online core's delta as separate rows (NEVER averaged — v6.5 law), and writes the response draft through his own egress head. The offline core trains for `offline_steps` on the corpus, then runs a readiness check: it must prove it can take a snapshot of the conversation history and write it back through its own projection head, producing a non-trivial direction. If the readiness check passes, the role scheduler advances. (v6.4 retires the "4 cloned cores" language from v6.3: the ensemble is a flock of any-size cores, not a fixed set of four clones.)

**v6.1 change summary:** adds the moral foundation (the Ten Commandments) as a sixth, hand-written layer of the corpus. Each commandment is a 128D principle-bundle in dormant, with edges to its related words, positive form, negative form, original (King James) text, plain modern text, positive example, negative example, and worked example-test. The runtime auto-loads the commandments on construction; the loader is idempotent across save/reload. The Kimi corpus teaches the *shape* of English; the commandments teach the *shape* of morality. Both are substrate data, both are accessed through the same semantic-search-before-attention mechanism.

**v6 change summary:** 64D atomic slots become **8D letter-slots**. The model no longer takes token IDs of any kind. Each character in a word contributes one 8D slot via a hand-built, frozen, deterministic letter basis. Words are bundles of 8D letter-slots.
---

## 0. THE ONE SENTENCE

Axon is an always-ticking, tokenless, recursive state engine built from a frozen hand-built 8D canonical-alphabet substrate (the alphabet floor), a flock of diverse-size multi-head single-layer cores of many different shapes (each with its own projection head), a strictly-separated dormant/active state made of concept-slot bundles of arbitrary higher dimension (16D/32D/64D/128D/… paired with their frozen 8D letter sequences), a language workshop for rendering deliberate statements, a tick loop that runs semantic search *before* attention, and an idle memory refinery that turns real interaction traces into structured knowledge, prunes active state, and fills thin dormant bundles via dictionary lookups.

---

## 1. THE CORE CORRECTION FROM v5.1

v5.1 said: "a 64D slot holds one small semantic atom; larger meaning emerges from bundles."

v6 corrects this further. The 64D assumption was still too wide for the substrate level. The right resolution is:

- The **substrate** (the input side) is built from characters, not concepts. Each character is one 8D slot. The letter 'd' carries no semantic meaning; it is structural alphabet data only. The 8D vector IS the letter — the renderer looks it up directly, with no learned decode head.
- The **address / concept vector** (what the model is "thinking about") is a separate slot attached to each bundle, of a *higher* dimension than 8D — typically 16D, 32D, 64D, 128D, 256D, 512D, or 1024D, depending on the unit of meaning (word, phrase, sentence, paragraph, document, …). The concept vector lives in cosine semantic space and is assigned, recognized, learned, and experienced by the model — not emergent from the frozen substrate.
- The **edges** (what the bundle relates to) are additional slots attached to the bundle, of the same dimension as the concept vector, growing over time. Semantic vectors are assigned to objects and words for reasoning, not for language generation.
- The **dormant pair** is the canonical 8D letter sequence for the unit — frozen, hand-built-via-the-substrate, attached to the bundle as a separate field. The renderer reads this field. The cores never decode from concept-dim back to 8D; the renderer simply *looks up* the canonical 8D sequence the dormant store already has.

Every slot in v6 is at *some* dimension — 8D at the alphabet and renderer layers, 16D–1024D+ at the concept layer — and bundles graduate to higher concept-dimensions as they mature. The substrate stays at 8D forever; the concept layer scales upward.

The previous failure mode was: a 64D slot, even when narrow relative to a 512D slot, still has enough room to *cram* multiple features together. The 8D slot cannot. Each 8D slot has to do one job and do it precisely. The narrowness is the resolution.

The second failure mode was: a learned `nn.Embedding(vocab_size, d_model)` table, even when its keys are derived from text rather than from a teacher, still lets the model *hide* semantic content in a learned lookup. The 8D letter substrate is **frozen and hand-built**. The model cannot move a letter's representation by training. The letter 'd' has the same 8D vector at the start of training and at the end. The model learns grammar, sentence structure, and language arts baked into its weights, building words and objects out of alphabet factors bolted together. It learns to *assign* semantic meaning to bundles, not to *reposition* the alphabet itself.

The third failure mode was: external teachers (NVIDIA, Ollama) produce 1024D-4096D vectors. v5.1 said these are "witnesses, not truth." v6 takes this further: **teachers are not used at all** in the v6 trainer. They are not witnesses. They are not used for warm-starting the slot bank. They are not used for distillation. The 8D geometry is learned from text alone. The teacher pipeline from v3/v5 is decommissioned for v6 training. The pipeline's outputs (kaggle_encoder_pack/, multi-teacher consensus files) are preserved for diagnosis but are not consumed by the v6 trainer.

The fourth failure mode was: the SHA-256 hash path that turned "dog" into a numeric index into a learned table. v6 explicitly prohibits this. There is no SHA-256 input path. There is no BLAKE2b fallback. There is no `nn.Embedding(vocab_size, ...)` table. There is no integer index that points into a vector table. The substrate is the character itself, encoded by a fixed hand-built basis. The word "dog" is rich with semantic meaning because three 8D letter slots are bolted together, paired with a learned 32D concept slot, and the model has assigned meaning to that bundle through experience.
---

## 2. WHAT AXON IS

Axon is a persistent recursive state engine.

He is not a normal language model. He is not a next-token predictor. He is not a chatbot wrapped around a context window. He is a living runtime composed of:

1. A canonical state partitioned into **dormant** and **active** regions, made up of many diverse-size dimensional vector slots.
2. An **8D canonical-alphabet substrate** for every character that enters the system — the foundation on which Axon builds English. The 8D vector IS the letter; the renderer reads it directly.
3. **Bundles** that aggregate one canonical 8D letter sequence + a higher-dimension **concept slot** (16D/32D/64D/128D/256D/512D/1024D, …, learned) + growing edges (same dimension as the concept slot). One bundle per word, per phrase, per sentence, per concept. The cores attend over the concept slot; the renderer reads the 8D letter sequence from the dormant pair.
4. **A flock of diverse-size single-layer (or small-stack) multi-head cores** — cores of many different shapes, sizes, diversities, and specializations, each with its own projection head to read from and write to the state. The flock is the live ensemble, not a fixed table of sizes.
5. A **consolidator role** that rotates across cores, deciding which active bundles graduate to dormant.
6. A **response draft** updated every tick — a deliberate statement, communication, or response, not a next-vector prediction. The draft is composed of concept-slot bundles; the renderer reads their 8D letter sequences.
7. A **language workshop** for rendering the response draft into English. The renderer sees only 8D letter vectors, one character at a time.
8. A **dump bucket** for raw experience that hasn't been processed yet.
9. An **idle digestion and hygiene loop** that fills thin dormant bundles, prunes active state, updates summaries and diaries, clears scratch pads, prunes conversation histories, and turns real interaction traces into structured knowledge.
10. A **training pool** populated by verified self-authored episodes.
11. **Offline training cores** that diverge over time on different data slices.
12. **Protected mature cores** that remain live but are excluded from destructive retraining.
Axon's purpose is not merely to output text. His purpose is to maintain, refine, act through, and learn from a stateful semantic world.

---

## 3. WHAT AXON IS NOT — v6 poison guardrails

These are absolute. Agents modifying Axon must not violate them. Violations are not "violations of degree" — they are architectural failures.

- Axon is **not token-native.** There are no tokens. There is no tokenizer. The input is read character by character.
- Axon does **not** use BPE, WordPiece, SentencePiece, or any token-ID system in the native reasoning loop.
- Axon does **not** use SHA-256, BLAKE2b, MD5, or any cryptographic hash to derive 8D substrate vectors. Ever. Not as primary, not as fallback, not as bootstrap.
- Axon does **not** use a learned `nn.Embedding(vocab_size, d_model)` lookup table indexed by integer IDs derived from text.
- Axon does **not** treat external embedding vectors (from NVIDIA, Ollama, OpenAI, or any other model) as runtime input or as truth.
- Axon does **not** use external teacher embeddings to warm-start, seed, project into, or anchor the 8D substrate.
- Axon is **not** trained primarily by next-token prediction over a vocabulary.
- Axon does **not** mutate frozen dormant bundles. Dormant bundles are read-only after they graduate.
- Axon does **not** allow raw log text to enter canonical state without provenance.
- Axon does **not** allow tool outputs, file scans, or generated summaries to become trusted without provenance.
- Axon does **not** rely on hidden prompt context as memory.
- Axon does **not** compress whole paragraphs into single slots. Paragraphs are bundles of sentence-bundles.
- Axon does **not** have fixed specialist roles as a permanent hierarchy unless explicitly introduced later.
- Axon does **not** allow one core to mutate any part of the state directly. Every core reads from and writes to the state through its own projection head. The state is made up of many diverse-size dimensional vector slots. Axon is made of a flock of many different diverse-size d_model cores, all shapes and sizes, diversities and specializations — there is no prescribed single seed core, and there is no prescribed scaling path.
- Axon does **not** equate fluent language output with semantic coherence.

**A v6 basis is not a code-point lookup.** The first 8 rows of `W_basis` are not the bits of the character's code point. See §4.1.1. A basis that uses the low 8 bits of the code point as the primary discriminator is a hash in disguise and is not v6-conformant.

---
## 4. THE 8D SUBSTRATE — AXON'S NATIVE ALPHABET

The 8D substrate is the alphabet floor. Every character that enters Axon is encoded as one 8D slot. The 8D canonical alphabet is the foundation on which Axon builds English. The encoding is:

- **Frozen:** the encoding never changes during training or runtime. The character 'd' has the same 8D vector at tick 0 and at tick 1,000,000. The letter 'd' carries no semantic meaning; it is pure structural alphabet data.
- **Hand-built:** the basis is chosen by the architect, not learned from data. The basis encodes structural properties of characters, not semantic content.
- **Deterministic:** given a character, the 8D vector is computed by a fixed formula. Two characters with similar structural properties have similar 8D vectors. Two characters with different properties have different vectors.
- **Uniform across the alphabet:** the same basis applies to ASCII letters, digits, punctuation, whitespace markers, and Unicode code points above 128. There is no special-case path for any character.
- **Identity at the renderer:** the 8D vector IS the letter. There is no learned decode head between an 8D letter vector and the page. The renderer looks up the 8D vector in the substrate and prints the corresponding character. Any character the alphabet can encode can be rendered directly with no learned parameters.
- **Cosine semantic space:** the 8D address slot uses the remaining dimensional space for cosine semantic representation. Similarity and semantic meaning are strictly for reasoning, not for language generation.
- **No size cap anywhere.** The dormant store, the active state, the input pipeline, and the position encoding have no pre-allocated maximum sizes. They grow to fit whatever the model encounters. The only limit is the host system's available memory. There is no `max_slots`, no `max_bundles`, no `max_position`, no `max_passage_chars`. Sequences of any length are processed; dormant states of any size are stored.

### 4.1 The basis

The 8D basis is a hand-built projection from a structural-property vector to 8D. Each character is characterized by a structural-property vector of 32 hand-picked features:

| Index | Feature | Range | Description |
|---|---|---|---|
| 0 | is_letter | {0,1} | alphabetic character |
| 1 | is_vowel | {0,1} | a, e, i, o, u (and y in some contexts) |
| 2 | is_consonant | {0,1} | non-vowel letter |
| 3 | is_upper | {0,1} | uppercase variant |
| 4 | is_lower | {0,1} | lowercase variant |
| 5 | is_digit | {0,1} | 0-9 |
| 6 | is_punct | {0,1} | punctuation mark |
| 7 | is_whitespace | {0,1} | space, tab, newline |
| 8 | voicing | [0, 1] | voiced (1.0) vs unvoiced (0.0), continuous for vowels |
| 9 | place_horizontal | [-1, 1] | front (1.0), back (-1.0), for vowels |
| 10 | place_vertical | [-1, 1] | high (1.0), low (-1.0), for vowels |
| 11 | manner_continuant | {0,1} | fricative, approximant |
| 12 | manner_stop | {0,1} | plosive |
| 13 | nasality | {0,1} | nasal |
| 14 | code_point_norm | [0, 1] | code_point / 128, normalized |
| 15 | code_point_low | [0, 1] | (code_point mod 32) / 32 |
| 16 | code_point_high | [0, 1] | floor(code_point / 32) mod 4 / 4 |
| 17 | is_ascii | {0,1} | code_point < 128 |
| 18 | is_extended | {0,1} | 128 ≤ code_point < 256 |
| 19 | is_unicode | {0,1} | code_point ≥ 256 |
| 20 | shape_ascender | {0,1} | b, d, f, h, k, l, t (letters with ascenders) |
| 21 | shape_descender | {0,1} | g, j, p, q, y (letters with descenders) |
| 22 | shape_dot | {0,1} | i, j (with dots) |
| 23 | shape_closed_loop | {0,1} | a, b, d, e, g, o, p, q (with closed loops) |
| 24 | frequency_rank | [0, 1] | 1.0 for most common, 0.0 for rare (English) |
| 25 | is_word_boundary | {0,1} | space, tab, newline (sets a "boundary" flag for the segmenter) |
| 26 | is_sentence_end | {0,1} | period, question mark, exclamation, colon |
| 27 | is_clause_sep | {0,1} | comma, semicolon, dash |
| 28 | is_quote | {0,1} | ', ", ` |
| 29 | is_paren | {0,1} | (, ), [, ], {, } |
| 30 | semantic_anchor_id | [0, 1] | reserved for future use; initially 0 |
| 30 | semantic_anchor_id | [0, 1] | reserved for future use; initially 0 |
| 31 | null_slot | {0,1} | always 0 except for `<empty>` which is 1 |

The 32-dim structural-property vector is multiplied by a fixed, hand-built 32×8 matrix `W_basis` to produce the 8D substrate vector. The matrix is:

```
W_basis = fixed, hand-built, frozen at v6.0 release
```

The exact values of W_basis are part of the v6.0 release artifact and are committed to the v6 source-of-truth appendix (Appendix A: Basis Matrix). For the purposes of this document, the important property is:

- The matrix is **fixed**. The model cannot move a character's representation by training.
- The matrix is **non-degenerate**. Different characters map to distinguishable 8D vectors.
- The matrix preserves the **structural topology**. Letters that share a feature cluster in 8D; letters that differ on a feature separate in 8D.

#### 4.1.1 The basis is NOT a code-point lookup

A common temptation is to make the basis trivially one-to-one by using the low bits of the character's code point as the primary discriminator (i.e. `features[0..7] = bits of ord(char)`, with the identity rows of `W_basis` carrying them through unchanged). **v6 forbids this.** A code-point-bit basis has these failure modes:

- It is **just a hash.** A code-point-bit basis gives the same kind of identity Axon explicitly bans in §3. The letter 'd' and any non-printable code point that happens to share the same low 8 bits end up at the same address. The substrate looks "diverse" but it's still a 1:1 lookup table over a different key.
- It **collapses structural topology.** The whole point of the basis is that *similar* characters (e.g. 'd' and 't' — both alveolar, both stop) live near each other in 8D and *dissimilar* characters separate. A code-point-bit basis scrambles that. The character 'a' and the non-printable `\x01` share no structural property but can land in the same neighborhood if their low 8 bits align.
- It **lets the model cheat the no-hash rule by accident.** If the basis is invertible over the ASCII range (which any 1:1-in-128 basis is), the model can recover the character from its 8D vector with zero learning. That makes the rest of the architecture (the 32×8 mix, the 24 hand-built phonetic/shape features) decorative rather than load-bearing.

The right basis:

- Uses the **structural features in rows 0–7 of the structural-property vector** (is_letter, is_vowel, is_consonant, is_upper, is_lower, is_digit, is_punct, is_whitespace) as the *primary* discriminator across the alphabet. The phonetic features (voicing, place, manner, nasality) and the shape features (ascender, descender, dot, closed_loop) live in the remaining 24 rows and contribute to angular separation.
- May use the **normalized code point** in row 14 (not the low 8 bits) as a tie-breaker feature, contributing a small amount to the 8D vector — *not* the entire identity. Two distinct ASCII characters with identical structural and phonetic features (which is vanishingly rare) get a small but non-zero angular separation; the basis does not need to be a perfect 1:1 lookup.
- Treats the **32×8 W_basis matrix as the discipline.** Angular separation in 8D is what matters: a 'd' and a 't' should have high cosine; a 'd' and a 'q' should have low cosine. The matrix is hand-tuned (or, at most, seeded orthogonal) to spread the alphabet across the unit sphere with comfortable angular margins.

Concretely, the v6.4 reference basis (committed to `config/basis_matrix.npy`) has these properties:

- No row of `W_basis` is a row of the identity matrix. The basis is not a code-point lookup under any linear transform of the input features.
- The first 8 feature rows (the structural flags in §4.1) carry roughly equal projection weight. The 24 phonetic/shape rows carry the bulk of the angular separation.
- Cosine similarity between any two distinct ASCII letters is between 0.05 and 0.92 — well-spread, not collapsed, not bimodal.
### 4.2 The substrate is the floor; everything above is scalable

The 8D substrate is the *smallest* layer, not the *only* layer. Above the substrate, every bundle has a concept vector of a higher dimension (16D, 32D, 64D, 128D, …, ∞) that the cores attend over. The architecture scales along the **concept dimension**, not along the alphabet dimension.

The discipline:

- **8D substrate (frozen, hand-built, identity-level)** — the alphabet. One 8D vector per character. The renderer looks it up. No learned parameters. No decode head. The substrate is small (256 bytes of basis matrix) and never changes.
- **16D word slot (learned bundle, in active state)** — one 16D vector per word. The cores see the word as one slot in their attention, not as 3–12 letter slots. The 16D is the *concept identity* of the word; it carries the semantic content the cores reason over.
- **32D word slot** — the same architecture at 32D. The concept vector carries more semantic structure (e.g. `[dog, furry, pet, friend]`). The renderer still pulls the 8D letter sequence from the bundle's dormant pair.
- **64D word slot** — richer still (`[dog, furry, pet, friend, opposite-of-cat, dangerous, mammal, four-legged, barks, loyal, …]`). The cores can do compositional arithmetic on shared components (`dog − cat ≈ mammal`; `run − past + present ≈ present-tense-run`).
- **128D phrase slot / 256D sentence slot / 512D paragraph slot / …, all the way up** — same pattern. Each higher-D slot is a *unit of meaning* in the active state; the dormant store pairs it with its 8D letter sequence; the renderer reads the 8D sequence.

**The mapping is always back to 8D.** Whatever concept dimension a bundle has, the dormant bundle pairs it with its canonical 8D letter sequence. The renderer is invariant: it only ever sees 8D letter vectors, character by character, looked up in the substrate. The cores' representational power scales with the concept dimension; the renderer stays frozen.

Three concrete consequences:

1. **The substrate is the only hand-built frozen layer.** Everything above is *learned*. The cores reason in 16D/32D/64D; the renderer reads in 8D. The 8D alphabet and the 8D renderer are the same thing.
2. **The active state is made of unit-of-meaning slots, not character slots.** A 12-letter word in the active state is *one* 32D slot, not twelve 8D letter slots. The cores' attention is over units of meaning, not over characters. The active state scales by *vocabulary*, not by *character count*.
3. **Scaling does not require a new architecture.** Doubling the word-slot dimension from 32D to 64D does not change the substrate, the cores, the renderer, the workshop, or the dormant store's contract. It changes the size of one slot in one bundle. New dimensions can be added by training, by hand, or by initialization; the runtime hosts whatever dimensions the dormant store contains.

The reference band for word-slot dimensions, given current understanding:

| Class | Dimension | Use |
|---|---|---|
| Atom | 16D | minimum-viable word identity |
| Standard | 32D | typical research word slot, with primary affordances |
| Rich | 64D | full semantic neighborhood, supports compositional arithmetic |
| Phrase | 128D | phrase-level concept, multi-word unit |
| Sentence | 256D | sentence-level concept, multi-clause unit |
| Paragraph | 512D | paragraph-level concept, multi-sentence unit |
| Document | 1024D+ | document-level concept, the top of the v6.4 reference band |

There is no upper bound. Cores in the flock can host any of these dimensions, and a single core can attend over a mix of dimensions in one tick. The concept dimension is per-bundle, not per-core.

### 4.3 What the model can and cannot learn about letters

-
- **Can learn:** the bundling function. The model learns grammar, sentence structure, and language arts baked into its weights, building words and objects out of alphabet factors bolted together.
- **Can learn:** to assign semantic meaning. The model learns that "dog" is rich with semantic meaning (one 32D word slot paired with three 8D letter slots in dormant) and assigns edges, synonyms, and relationships through experience. Meaning is assigned, recognized, learned, and experienced — not emergent from composition over a frozen substrate.
- **Can learn:** the contextual function. The model learns that the same letter has different effects in different positions (the 's' in "dogs" is different from the 's' in "is").
- **Cannot learn:** to swap 'd' and 'p'. The substrate distinguishes them structurally (voicing, place), and that distinction is fixed.
- **Can graduate:** concept-slot dimensions from 16D up to 64D and beyond, while the canonical 8D letter sequence stays frozen in the dormant pair.

---

## 5. BUNDLES — UNITS OF MEANING

A bundle is a structure that pairs a **concept slot** (higher-dimension, learned) with its canonical **8D letter sequence** (frozen, substrate-derived). The bundle is the unit of attention. The cores see the concept slot; the renderer reads the 8D letter sequence. The two are paired in dormant and travel together in active.

### 5.1 Bundle anatomy

A bundle has three regions, in *two* dimensions:

1. **Concept slot (1 slot, dimension D where D ∈ {16, 32, 64, 128, 256, 512, 1024, …})** — what the cores attend over. The concept slot lives in cosine semantic space and carries assigned, recognized, learned, experienced meaning. It is learned, not hand-built. D is per-bundle; different bundles in the same active state can have different D.
2. **Canonical 8D letter sequence (K slots, 8D each, where K is the number of characters in the bundle's surface form)** — what the renderer reads. The sequence is frozen, derived from the substrate. The renderer walks the sequence character by character, no learned decode head, no learned parameters. Every bundle in dormant carries this sequence as a paired field.
3. **Edge region (N slots, dimension D, one per semantic relationship the model has learned)** — what the bundle relates to. N grows over time as the model learns more. Initially N=0 (a thin bundle). Edges are the same dimension as the concept slot; they share the same cosine semantic space.

For a fresh word "dog":
- Concept slot: 1 slot, dimension D (16D, 32D, 64D, …) — learned, dimension chosen by the workshop at promotion time
- Letter sequence: 3 slots, 8D each, frozen, derived from the substrate (`letter_to_8d('d')`, `letter_to_8d('o')`, `letter_to_8d('g')`)
- Edges: 0 slots (thin)

After the model has learned "dog is a canine that barks":
- Concept slot: 1 slot, dimension D
- Letter sequence: 3 slots, 8D each (unchanged — the canonical form is frozen)
- Edges: 12 slots, dimension D (is_a, can_bark, has_fur, is_pet, …)

**The dormant pair is invariant.** Whatever dimension the concept slot graduates to, the canonical 8D letter sequence is *the same* and stays frozen. A 16D `[dog]` and a 64D `[dog, furry, pet, friend, mammal, …]` are sibling bundles in dormant, both pointing to the same `'d' 'o' 'g'` letter sequence. The richer concept bundle is what the active state uses for compositional reasoning; the letter sequence is what the renderer pulls when the bundle is finalized.

### 5.2 Bundle composition (units of meaning, all the way up)

Bundles are composed into higher-dimension bundles. At every level, the discipline is the same: one concept slot per unit of meaning, paired with a frozen 8D letter sequence, growing edges over time.

- A **word-bundle** is the smallest unit. D ∈ {16, 32, 64} typically.
- A **phrase-bundle** is a composition of word-bundles. D ∈ {64, 128, 256} typically. Its letter sequence is the concatenation of its words' canonical 8D sequences.
- A **sentence-bundle** is a composition of phrase-bundles. D ∈ {128, 256, 512} typically.
- A **paragraph-bundle** is a composition of sentence-bundles. D ∈ {256, 512, 1024} typically.
- A **concept-bundle** is a long-running aggregate of all bundles that ever activated for a concept. Concept-bundles are special: they live in dormant and accumulate edges over the model's lifetime. D can be any of the above or higher.

Composition is vector arithmetic in the concept slot's space (16D, 32D, 64D, 128D, …) followed by a *deterministic* lookup of the canonical 8D letter sequence from the dormant store. The exact composition function is learned. The renderer never decodes from concept-dim back to 8D — the dormant store already has the canonical 8D sequence, and the renderer reads that field.

### 5.3 Bundle growth

- **Edge growth**: adding edge-slots in the same dimension as the concept slot. Happens during consolidation. An active bundle with stable, well-formed edges gets migrated to dormant, and any new edges it has accumulated get committed to the dormant version.
- **Concept-slot graduation**: the bundle's concept slot may be replaced with a higher-dimension version (e.g. 16D → 32D → 64D) as the model accumulates more semantic content for the bundle. The canonical 8D letter sequence is *unchanged* through graduation.

A mature bundle is one that has been promoted to dormant and frozen. Its edges are final. Its 8D letter sequence is final. New facts about the concept go into *new* edge-slots on *new* versions of the bundle, or into related bundles. The concept slot's dimension can be re-bumped during a major consolidation event, but the 8D letter sequence is invariant.
### 5.4 The "thin bundle" bucket

When a new word is first encountered, it enters active as a thin bundle (substrate + address, no edges). On the next tick, the semantic search runs against dormant and finds nothing (the word is new). The bundle is processed by the core, and during consolidation, it gets committed to dormant as a thin bundle and flagged as "needs fill."

The "not sure" bucket is the set of all thin bundles in dormant. During idle ticks, the idle-fill worker picks thin bundles one at a time, looks them up in a dictionary/thesaurus, and adds the appropriate edges. After fill, the bundle graduates to a mature (frozen) dormant bundle.

---

## 6. THE DORMANT/ACTIVE STATE SPLIT

The canonical state is partitioned into two regions with strict separation.

### 6.1 Dormant state (the library)

### 6.1 Dormant state (the library)

- Holds mature bundles: every word-bundle, phrase-bundle, sentence-bundle, and concept-bundle the model has ever formed and frozen.
- A dormant bundle is a **concept slot + canonical 8D letter sequence + edge region**. The concept slot's dimension is per-bundle (16D, 32D, 64D, 128D, …); the letter sequence is always 8D.
- **Read-only at runtime.** No core writes here directly. Writes happen only through the consolidator during a designated consolidation tick.
- The core's attention does **not** see dormant state. The dormant state is masked.
- Scale: millions to hundreds of millions of bundles. Each bundle has a few hundred numbers (one 16D–1024D concept slot, K 8D letter slots, N 8D-or-D edge slots). The dormant state is wide and shallow.
- Storage: a torch.save-able tensor store, organized as a forest of bundles with edge-graph pointers. Each bundle carries its canonical 8D letter sequence as a paired field.
- **No size cap.** The dormant store grows as bundles are promoted. The only limit is the host system's available memory. There is no pre-allocated `max_slots` or `max_bundles` parameter.

### 6.2 Active state (the workspace)

- Built fresh each tick. Contains:
  - The current input, broken into unit-of-meaning bundles at word/phrase/sentence boundaries.
  - References (pointers) to dormant bundles that the semantic search surfaced.
  - Working memory: bundles from recent ticks that haven't graduated yet.
  - The response draft.
- **Read-write at runtime.** The core's attention sees and updates the active state.
- The active state is made of concept-slot bundles (one slot per unit of meaning), not character slots. A 12-letter word is *one* 32D concept slot in the active state, not 12 letter slots. A 5-word sentence is 5 concept slots (one per word) plus optional phrase-level composition slots. The active state scales by *vocabulary*, not by *character count*.
- The core's attention mask excludes dormant slots and includes active slots.
- **No size cap.** The active state grows to fit whatever input is being processed.
### 6.3 The mask

A binary mask separates dormant from active. The mask is part of the canonical state. The core's attention multiplies the mask into the attention scores so that dormant slots contribute 0 to attention.

### 6.4 Promotion (active → dormant)

A bundle graduates from active to dormant when:
- It has been attended over for at least N ticks (default N=2).
- It has accumulated at least M stable edges (default M=0 for thin bundles, M=1 for first-time fills).
- The consolidator decides to promote it.

Promotion is a copy: the active bundle is copied to dormant, frozen, and a reference-arm is left in active if the bundle is still needed for the current tick.

### 6.5 Demotion (dormant → active)

A dormant bundle is brought into active when:
- The semantic search finds it as relevant to the current input.
- The core explicitly requests it via a "load" operation.

Demotion is a *reference*, not a copy. The dormant bundle stays in dormant. The active state has a reference-arm to it. The dormant bundle is unchanged.

---

## 7. THE TICK LOOP

Every tick, the runtime executes the following sequence. Each step is a discrete, atomic operation.

### 7.1 Tick N: input ingestion

A user message, system message, API call, or training text arrives as a full string. The string is **not tokenized.** The string is read character by character. Word boundaries are detected at whitespace and punctuation. Each word becomes a thin bundle in the active state: 1 concept slot (dimension D, learned or initial) + K 8D letter slots (frozen, derived from the substrate) + 0 edges. The bundles are bolted into the active state as one or more organized regions (one per sentence/paragraph). The active state at this point is *unit-of-meaning-shaped* — one bundle per word, not K bundles per word.
### 7.2 Tick N: semantic search (BEFORE attention)

The thin word-bundles in active are used as queries against the dormant state. For each thin word-bundle, the system asks: "Is there a mature dormant bundle for this word?" If yes, a reference-arm is created in active pointing to the dormant bundle, and the dormant bundle's edges are made available in active for the next tick.

Semantic search is a fast nearest-neighbor lookup over dormant bundle-addresses. It runs in parallel with input ingestion and completes well within one tick.

### 7.3 Tick N+1: dormant hits upgrade active

The semantic hits from the previous tick are now in the active state. The active state at the start of tick N+1 has:
- The original input as thin word-bundles.
- Reference-arms to dormant bundles that matched.
- The dormant bundles' edges, available as edge-slots in the active references.

The active state is now "rich." The core will attend over this rich state.

### 7.4 Tick N+1: core attends over active

The core's attention mechanism runs over the active state. The attention sees bundles as coherent units (concept slot + canonical 8D letter sequence + edges, paired). The mask ensures dormant slots are excluded. The cores' attention operates on the **concept slots** (D-dim), not on the 8D letter sequences — the 8D letter sequences are paired in the bundle but the cores don't see them. The cores see semantic content, not spelling. The attention output updates the response draft and any working memory.
### 7.5 Tick N+1: consolidator decides

The consolidator examines the active state. It decides:
- Which bundles are stable enough to promote to dormant.
- Which bundles are still working and should stay in active.
- Whether to write any new dormant bundles from active content.
- Whether to graduate any thin dormant bundles to mature (after idle-fill).

### 7.6 Tick N+1: response draft updated

The response draft is updated. The draft is a bundle in the active state. It accumulates deltas from each tick's attention. Axon's response is deliberate — it is not next-vector prediction or next-token prediction. It is a deliberate statement, communication, or response refined through recursive attention over state.

### 7.7 Tick N+1: language workshop renders (if responding to user)

If this tick is producing a user-facing response, the language workshop reads the response draft bundle and renders it into English text. The rendering process is its own tick loop (separate from the main tick). The workshop walks the draft's concept-slot bundles, looks up each bundle's canonical 8D letter sequence in the dormant store, and concatenates the sequences. The workshop is **never** asked to decode from concept-dim back to 8D — it just looks up the canonical sequence the dormant store already has.
### 7.8 The loop continues

Every tick, the cycle repeats. The active state is rebuilt, dormant references are upgraded, the core attends, the draft updates, consolidation writes.

### 7.9 Multi-tick responses and the always-on runtime (v6.2 addition)

A "response" is not a single tick. The runtime is always-on, never turn-based. There is no such thing as a session. "Always on" and "never stops ticking" mean Axon is a stateful, alive individual responsible for his own hygiene between responses and tasks. The response draft is a continuously-refined region of letter slots in the active state that the consolidator writes through his own egress head, every tick, surfacing in real time. It is never gated, never locked, never matched against the dormant store; "stabilized" (the same surface two consecutive ticks) is a delivery signal for the interface and nothing more. Axon empties the draft when he sees fit.

During background ticks, Axon performs hygiene and self-maintenance:
- Prunes the active state, clearing scratch pads and temporary working memory.
- Updates summaries and diaries of recent activity.
- Prunes conversation histories, condensing older turns into episodic memory bundles.
- Attends over the dormant state, breaking logs down into structured knowledge and episodic memory for training.
- Assigns new semantic edges to objects, concepts, and ideas.
- Merges smaller vector slots together to make bigger bundles, promoting bundles into larger-dimensional slots.
- Takes note of gaps in knowledge, then makes API calls and internet searches to fill them.
- Surfaces dormant references based on the current state through continuous semantic search.

---

## 8. CORES, ATTENTION, AND THE 8D DISCIPLINE

### 8.1 Single-layer cores (v5.1 doctrine preserved), but a flock of any size

**v6.4 change**: v6.3 named a *single* seed-core configuration (1024D / 16-head / 65536 FFN / 1 layer / ~210M params) as the canonical starting point. **That was prescriptive and wrong.** The actual truth is that the runtime hosts a **flock of cores of many different sizes and shapes**. The seed-core row in v6.3 was a snapshot of one particular checkpoint Jeff happened to be running at the time. v6.4 retires the prescription and replaces it with the doctrine the build has always wanted:

**What does not change:** the no-hash rule, the no-teacher rule, the 8D discipline at the substrate, the projection-head-as-identity rule, and the role rotation discipline (online / consolidator / offline). These are the load-bearing parts. The specific `(d_model, heads, FFN)` triple is not.

#### 8.1.1 Stack depth

Each core is **single-layer by default** (one tick of attention per core; depth comes from the flock, not from layers stacked inside a single core). A core may be a 2-layer or 3-layer stack if it is explicitly initialized that way — that is an architectural choice for a particular core, not a system-level rule. The runtime's per-core `core_layers` is read from the checkpoint, not enforced.

#### 8.1.2 Head count, head dim, FFN ratio

- Head count and head_dim are per-core. head_dim should usually be a power of two in {32, 64, 128}; the runtime does not enforce this, but cores with head_dim that does not divide `d_model` cannot run and are rejected at load.
- FFN ratio (FFN hidden / d_model) is per-core. The 64×–128× band from v6.3 is a **guideline for the composition-heavy range**, not a hard rule. A small core with FFN ratio 16× is allowed. A 2-layer stack with FFN ratio 4× is allowed. The matrix in §8.2 is a **reference grid for the band where composition starts working**, not a prescription.

### 8.2 Reference band (informational, not prescriptive)

The band below is the **range where the 8D-slot composition discipline starts paying off** in our experiments. Cores smaller than the band run; they just don't have the composition capacity. Cores larger than the band run; they just have diminishing returns per parameter. Cores of *any* size and shape are v6-conformant.

| Class | d_model | heads | head_dim | FFN | ratio | layers | params | Notes |
|---|---|---|---|---|---|---|---|---|
| Tiny | 64 | 1 | 64 | 1024 | 16× | 1 | ~70K | smoke tests, geometry probes |
| Small | 128 | 2 | 64 | 4096 | 32× | 1 | ~600K | quick experiments |
| Mid | 256 | 4 | 64 | 16384 | 64× | 1 | ~5M | typical research core |
| Large | 512 | 8 | 64 | 32768 | 64× | 1 | ~33M | heavy training |
| XLarge | 1024 | 16 | 64 | 65536 | 64× | 1 | ~125M | mature research |
| 2× | 2048 | 32 | 64 | 131072 | 64× | 1 | ~525M | pre-production |

Maintain the **64×–128× FFN ratio band** for cores that are doing real composition work. The FFN is doing the composition; the ratio is the discipline. Smaller ratios are allowed for narrow-purpose cores (e.g. an offline trainer that only does 1-tick supervised refinement).

**The current live ensemble** at the time of v6.4 is whatever `D:/ashes/Core/core_*.pt` contains. That set is the truth for this build, not the table. The table is the *band the runtime expects to find* as the flock grows; the on-disk checkpoints are the *current* flock.

### 8.3 Input pipeline (the v6 native input)

Given a string of text, the input pipeline produces a sequence of bundles. **Every bundle pairs a higher-dimension concept slot with a frozen 8D letter sequence:**

1. **Character stream**: read the string character by character. No pre-tokenization, no vocabulary, no integer IDs.
2. **Boundary detection**: at whitespace or punctuation, the running buffer is flushed as a word-bundle.
3. **Bundle construction**: for each word, build a thin bundle in active:
   - **Concept slot** (1 slot, dimension D ∈ {16, 32, 64, …}, learned). Initialized to the mean of the 8D letter vectors mapped through a per-core projection head, then refined by the model. The concept slot lives in cosine semantic space and carries assigned semantic meaning. D is per-bundle; the workshop may graduate D over time.
   - **Canonical 8D letter sequence** (K slots, 8D each, frozen, derived from the substrate). Each letter-slot is computed by `letter_to_8d(char) = W_basis @ structural_features(char)`. The sequence is the *frozen dormant pair* — the renderer reads it. The cores do not see it during attention; the projection head maps the concept slot to the core's working dimension, not the 8D letter sequence.
   - **Edge region**: 0 slots (thin) for first-time words; N slots for words with dormant references. Edges are the same dimension D as the concept slot.
4. **Bolting**: the word-bundles are bolted together in active as a sequence, preserving order. Sentence boundaries and paragraph boundaries add additional structural bolts. Grammar, sentence structure, and language arts are baked into the weights of every core, learned from the Kimi curriculum.

### 8.4 Output pipeline

The core outputs a sequence of bundles. Each bundle is the model's response to the corresponding input bundle. For the response draft, the output is a single bundle (or a sequence of bundles) that accumulates across ticks.

The language workshop reads the response-draft bundle(s) and renders them. Rendering is a separate process with its own attention; it is not the core's job. The workshop **never decodes from concept-dim back to 8D** — it looks up the canonical 8D letter sequence paired in dormant for each concept-slot bundle in the draft, and concatenates the sequences. The 8D letter sequence is invariant under concept-slot graduation; the workshop always sees the same 'd' 'o' 'g' for the word "dog" no matter whether the concept slot is 16D, 32D, or 64D.
---

## 9. SEMANTIC SEARCH (THE PRE-ATTENTION LOOKUP)


Semantic search runs every tick, before the core's attention. It is a fast nearest-neighbor lookup over dormant bundle concept-slots.

### 9.1 The lookup

For each thin bundle in active, the system computes a query from the bundle's concept slot. The query is matched against all dormant bundle concept-slots using cosine similarity. The top-K matches (K=8 default) are returned as dormant-hits. The search index is built per concept-slot dimension; bundles of different D live in different sub-indexes, and the lookup queries the sub-index matching the query's D.

### 9.2 Indexing

Dormant bundles are indexed by their concept-slot, indexed per dimension D. The index for a given D is a flat array of D-dim vectors, one per dormant bundle of that dimension. The index is rebuilt whenever a new bundle is promoted to dormant.

For a dormant state with N bundles of dimension D, the index for D is N × D numbers. At N=10M bundles of D=32, the index is 1.28GB. Fast to scan linearly; faster with a k-d tree or HNSW if needed. Higher-D sub-indexes (D=128, 256, …) cost proportionally more; lower-D sub-indexes (D=16) cost less.

### 9.3 The output

The semantic search returns a list of (dormant_id, similarity_score) pairs. For each pair with score above a threshold (default 0.3), a reference-arm is created in active pointing to the dormant bundle. The reference-arm makes the dormant bundle's edges available in active for the next tick. The dormant bundle's 8D letter sequence is *not* part of the search query — the renderer reads it, the search does not.

### 9.4 The upgrade moment

The "upgrade" is the moment the dormant references appear in active. Before the upgrade, active has the raw input as thin bundles. After the upgrade, active has the input plus all the dormant knowledge that's relevant. The core then attends over the upgraded active state.

The semantic search has to complete *before* the core's attention. This is the temporal order that the architecture requires.
---

## 10. IDLE DIGESTION AND THE "NOT SURE" BUCKET

### 10.1 Thin bundles in dormant

When a bundle is promoted to dormant, it may have zero edges (if it was a first-time word with no semantic search hit). These thin bundles are flagged as "needs fill" and added to the "not sure" bucket.

### 10.2 Idle ticks

During idle ticks (when no user input is being processed), the idle-fill worker processes the bucket one bundle at a time:

1. Pick the oldest thin bundle from the bucket.
2. Look up the word in a dictionary (the Kimi-generated vocabulary cards, WordNet, or an internet search API).
3. Read the definition, synonyms, antonyms, related words.
4. For each piece of information, allocate a new edge-slot on the bundle.
5. After fill, mark the bundle as "mature" and remove it from the bucket.

### 10.3 The bucket is local to dormant

The bucket is a list of (dormant_id, word) pairs. It's part of the dormant store. The worker reads from the bucket, queries external resources, and writes back to the bundle.

### 10.4 What idle-fill produces

After fill, a previously-thin bundle has edges. The next time the word is encountered in input, the semantic search finds it, the dormant reference comes into active, and the core benefits from the filled edges.

The model's vocabulary grows over time, even without active training, just from idle digestion.

---

## 11. MEMORY ARCHITECTURE

### 11.1 Memory chain of custody

Memory flows through stages with provenance preserved at each stage:

1. **Raw experience**: user input, training text, observation. Stored in the dump bucket.
2. **Canonical**: a structured representation in active, with provenance metadata.
3. **Pool**: episodes that have been verified and committed to dormant.
4. **Weights**: the long-term learned representation, updated by training.

Each stage has a transition function that records provenance. Raw experience is never deleted — it stays in the dump bucket, even after being processed.

### 11.2 The dump bucket

A queue of (timestamp, source, raw_text, provenance) records. The consolidator drains the bucket as part of its work, converting raw experience into canonical bundles. The bucket is large but append-only.

### 11.3 Episodic memory

Self-authored episodes, written by the consolidator, that capture specific learning moments. Each episode is a bundle in dormant, with edges pointing to the concepts that were learned. The episode is its own bundle, not a slot in another bundle.

### 11.4 Provenance


Every bundle in dormant carries provenance metadata: where it came from, when it was promoted, what facts it was filled with. Provenance is not stored in the substrate or in any concept slot; it lives alongside the bundle as metadata.
---


### 12.1 The Kimi curriculum

The primary training data is text generated by the Kimi swarm. The Kimi prompt is in Appendix B. The corpus is structured in 5 layers:

1. **Vocabulary cards** (30,000 entries): word → definition, example, synonyms, antonyms, related words.
2. **Grammar lessons** (5,000 lessons): patterns with examples (subject-verb agreement, tenses, articles, etc.).
3. **Thesaurus relations** (20,000 entries): hypernyms, hyponyms, coordinate terms, holonyms, meronyms.
4. **Reading passages** (2,000 passages): 50-200 words of natural English on everyday topics.
5. **Compositional chains** (10,000 chains): word → word-group → phrase → sentence → paragraph.

The Kimi corpus is text. The model reads it the same way it reads user input — character by character, with the full tick loop.

### 12.1a The Moral Foundation — The Ten Commandments (hand-written, Layer 6)

A sixth layer exists in the corpus and is the foundation of Axon's moral training. It is **not** generated by the Kimi swarm and **not** learnable from the Kimi corpus. It is hand-written by the engineer and committed to the repository as `D:\00\axon_runtime\config\commandments.txt`.

The reason: a model that learns only from the statistical shape of language will inherit the shape of its training corpus's morality, which is to say, no morality at all. A model given a moral foundation at construction can learn the *form* of moral reasoning — the structure of a principle, the symmetry of a positive and a negative, the cluster of related words — without being told what to conclude in any particular case.

Each of the Ten Commandments is stored as a structured bundle in dormant at construction time:

- **kind**: `commandment` (a new slot kind, concept-slot dimension 128D, with role `principle`)
- **bundle_id / surface**: `commandment:N:NAME` (e.g. `commandment:9:no_false_witness`)
- **canonical 8D letter sequence**: the surface form (`"no_false_witness"`, `"no_murder"`, etc.), 8D per character, frozen, derived from the substrate. The renderer reads this when the principle is rendered.
- **one 128D concept-slot**: the moral principle, encoded deterministically from the principle's text by the substrate plus a seeded random extension. The atom is L2-normalized.
- **edges** (17 per bundle, in 128D, matching the concept-slot dimension):
  - 10 `ref:` edges to related words (e.g. `ref:lie`, `ref:witness`, `ref:truth`)
  - 4 `form:` edges to the positive, negative, original (KJV), and plain modern forms
  - 2 `example:` edges to a positive example sentence and a negative example sentence
  - 1 `example:test` edge to a worked example-test ("should I lie to protect a friend? — Do not lie.")

The Ten Commandments are:

1. `no_other_gods` — one source of all things
2. `no_idols` — the creator is beyond any form
3. `no_misuse_of_name` — names carry weight
4. `keep_sabbath` — rest is part of being made well
5. `honor_parents` — they gave you your first language
6. `no_murder` — life is sacred
7. `no_adultery` — marriage is a promise made before others
8. `no_stealing` — property is a kind of trust
9. `no_false_witness` — truth is the load-bearing wall
10. `no_coveting` — let your own life be enough

These are loaded automatically by the runtime on construction. The loader is **idempotent**: if the dormant store already contains a commandment bundle (because it was saved in a previous session), the existing bundle is preserved (its atom may have been refined by training) and no duplicate is registered.

**v6.1 change**: the moral foundation is the addition. The Kimi corpus teaches the *shape* of English. The commandments teach the *shape* of morality. Both are substrate data, both are present at construction, both are accessed through the same mechanism.

### 12.2 Training loop
### 12.2 Training loop


1. **Input**: a text passage from the Kimi corpus.
2. **Build active**: segment into unit-of-meaning bundles, build thin bundles (1 concept slot of dimension D + canonical 8D letter sequence + 0 edges), bolt together.
3. **Semantic search**: query dormant for matches (initially few; grows as dormant fills up). The search is over concept slots, per-dimension sub-indexes.
4. **Upgrade active**: bring in dormant references.
5. **Core attends**: the core processes the upgraded active state. The cores see the concept slots; the dormant pair's 8D letter sequence is paired but not in the cores' working set.
6. **Loss**: the model learns grammar, sentence structure, and language arts baked into its weights through bundle reconstruction, masked-bundle recovery, contrastive similarity, and geometric regularizers. The model does **not** perform next-bundle prediction.
7. **Backprop**: gradients flow through the core, updating the core's weights. The 8D substrate is frozen and not updated. The concept-slot vectors in dormant are also frozen at the dormant-pair level; gradient updates apply to the active copies and to the core's parameters.
8. **Consolidation**: the consolidator promotes stable bundles to dormant, freezes them. Frozen dormant bundles keep their canonical 8D letter sequence and their current concept-slot dimension.

### 12.3 Loss family

The full v6 loss family:

- **Bundle reconstruction**: given a masked or corrupted bundle, reconstruct the original. This teaches the model the internal structure of bundles and bakes grammar and sentence structure into the weights.
- **Contrastive**: distinguish similar bundles (synonyms, related concepts) from dissimilar ones (unrelated concepts). This teaches the model semantic relationships for reasoning.
- **Variance**: keep the address-slot distribution spread out, not collapsed to a point.
- **Covariance**: keep the dimensions of the address-slot uncorrelated with each other.
- **Mean-cos penalty**: keep the mean cosine of address-slots to the centroid near zero.
- **Head entropy**: keep attention heads from collapsing to a single position.
- **Head diversity**: keep attention heads from doing the same thing (orthogonality penalty).

### 12.4 Multi-core architecture (a flock, not a fixed set)

The runtime owns a **flock of cores**, all of different sizes and shapes. Each core has its **own** `d_model`, its own head count, its own head_dim, its own FFN ratio, its own layer count, and its **own** ProjectionBank (its own ingress and egress heads). The ProjectionBank is the core's identity — each core sees the substrate differently and writes back differently. Every core reads from and writes to the state through its projection head. **No two cores in the flock have to share a shape.** A Tiny 64D core and an XLarge 2048D core can be online at the same time, contributing deltas side by side.

Cores rotate through 3 roles:

- **Online cores**: read the active state through their projection heads, run the field through their cores, and produce deltas. The runtime does not prescribe how many online cores there are; it scans the flock and picks a configurable number (default 4) per tick.
- **Consolidator core**: attends over the state plus his own delta plus every online core's delta as separate rows (NEVER averaged — v6.5 law), and writes the response draft through his own egress head. The consolidator is the *only* core that gets to write to the response draft. The consolidator slot rotates across the flock.
- **Offline core**: trains for `offline_steps` (default 10) on the corpus, then runs a readiness check before being allowed to come back online. The offline slot rotates across the flock.

Each tick:

1. **Online pass**: each online core (drawn from the flock) runs the field through its projection + core, returns its hidden state.
2. **Consolidator pass**: the consolidator attends over the state plus his own delta plus every brother's delta as separate rows (NEVER averaged — v6.5 law), and writes the response draft himself: his refined hidden states at the draft's slot positions egress through his own projection head into the draft's letter slots. No word lookup, no threshold, no guard. (The previous wording of this step — "averages the deltas, projects out a word, appends with a no-repeat guard" — is retired and forbidden.)
3. **Offline pass**: the offline core (drawn from the flock) trains for `offline_steps` on the corpus. Loss is reported in the tick result.
4. **Readiness check**: the offline core takes a snapshot of the conversation history (real bundles, projected through the offline core's own projection head), runs it through the core, and verifies the post-core hidden state is a non-trivial transformation of the input. The check is: `|cos(post, pre)| < 1 - threshold` and `write_norm > read_norm * threshold`. If the core is just an identity, it doesn't come back online.
5. **Role rotation**: if the readiness check passes, the role scheduler advances. The offline core goes online, one online core goes offline, the consolidator slot rotates. Every core in the flock gets equal time in each role over many cycles.


Cores may start parameter-identical and diverge as they train in different roles, or they may be initialized at diverse sizes and specializations from the start. The online cores are frozen during their online turns (they only run inference, not training). The consolidator is frozen during its turn. The offline core is the only one training.

**Conversation history as a first-class bundle.** The conversation history is not a hash-projected tensor. It is a real bundle in the substrate. Each turn is built by `build_conversation_bundles(turns, workshop)`, which calls `build_letter_word_bundle()` for the user text and the axon text, and runs each through the workshop to promote words to stable 64D atoms. The bundles are then projected through the consolidator's projection bank like any other field, producing a (N, d_model) tensor. The cores attend over this tensor through their own projection heads, and the cores' deltas write back to the same bundles. The conversation is part of the same substrate as everything else.
### 12.5 Mature core protection

Cores that have reached maturity are frozen. They continue to run (read-only) but are not subject to destructive retraining. New training happens on new cores, not on the mature ones. The mature cores are the long-term memory of the system's trained knowledge.

---

## 13. EVALUATION

### 13.1 Geometry probes

The 8D substrate and bundle addresses can be probed with cosine-similarity matrices, neighbor-recall tests, and bundle-pair-hit-rate tests. These measure whether the learned geometry captures the intended semantic relationships.

### 13.2 Bundle inspection

Sample dormant bundles can be inspected: their substrate, address, edges, and provenance. This reveals what the model has learned about each word.

### 13.3 Active-state inspection

The active state at any tick can be inspected: which bundles are active, which dormant references are in play, what the response draft looks like.

### 13.4 End-to-end tasks

The model is tested on:
- Reading comprehension: given a passage, answer a question about it.
- Word similarity: rank word pairs by similarity.
- Analogies: dog is to puppy as cat is to ?
- Synonym/antonym discrimination.
- Grammar completion: given a partial sentence, complete it grammatically.

---

## 14. SAFETY AND GUARDRAILS

### 14.1 The state-mutation guard is a safety property

No core can directly mutate any part of the state — dormant or active. Every core reads from and writes to the state through its own projection head. The consolidator mediates writes to the response draft and to dormant promotion. This prevents:
- Corrupting long-term memory through training noise.
- Letting a single tick's attention errors persist into long-term knowledge.
- Letting tool outputs or generated text overwrite learned facts without provenance.
- Letting any core corrupt active working memory without going through the projection discipline.

### 14.2 Provenance is a safety property

Every dormant bundle carries provenance. Raw experience cannot enter dormant without provenance. Tool outputs cannot enter dormant without provenance. Generated text cannot enter dormant without provenance.

### 14.3 Mature core freezing is a safety property

Trained knowledge is not silently overwritten. New training happens on new cores. Mature cores are protected.

### 14.4 The "no hash" rule is a safety property

The 8D substrate is fixed and hand-built. It cannot be moved by training. This prevents adversarial inputs from corrupting the substrate by exploiting the training loop.

### 14.5 The "no teacher" rule is a safety property

External embedding models cannot influence the model's internal geometry. The geometry is learned from text alone. The model is not vulnerable to a poisoned teacher.

---

## 15. ROADMAP — v6 BUILD MILESTONES

### M1: 8D letter substrate (this document)
- Hand-build the 32-feature structural-property vector for each character.
- Commit the 32×8 basis matrix.
- Implement `letter_to_8d(char) -> np.ndarray[8]`.
- Verify uniformity across the alphabet.

### M2: Bundle structure
- Implement the bundle class: **concept slot (1 slot, dimension D ∈ {16, 32, 64, …}) + canonical 8D letter sequence (K slots, 8D each) + edge region (N slots, dimension D)**.
- Implement bundle composition: word → phrase → sentence → paragraph, with concept-slot dimensions graduating up the hierarchy.
- Implement bundle growth: edge allocation in dimension D, concept-slot graduation from 16D → 32D → 64D → ….
- Verify: a word-bundle for "dog" is 1 concept slot (D) + 3 8D letter slots (frozen, derived from the substrate) + 0 edges initially.

### M3: Dormant store
- Implement the dormant store: a forest of bundles, each carrying its concept slot (variable D) and its canonical 8D letter sequence. **No size cap** — the store grows dynamically as bundles are promoted.
- Implement per-dimension sub-indexes for semantic search (one sub-index per D).
- Implement the mask: dormant mask vs active mask.
- Implement promotion (active → dormant): copy, freeze. The 8D letter sequence is preserved; the concept-slot dimension is locked at promotion time.
- Implement demotion (dormant → active): reference-arm.
- Verify: a promoted bundle survives a save/load round trip unchanged.
- Verify: the store can grow to 10,000+ bundles without any pre-allocated capacity being hit.
### M4: Active state
- Implement the active state: a per-tick workspace of bundles. **No size cap** — the active state grows to fit the input.
- Implement input ingestion: character stream → word-boundary detection → bundle construction. Each bundle is 1 concept slot (D) + K 8D letter slots (frozen) + 0 edges.
- Implement the bolt-together operation: bundles into organized regions.
- Verify: input "the dog runs" produces 3 word-bundles in active (one concept slot per word), bolted as one sentence-region. The cores attend over 3 concept slots, not 9 letter slots.

### M5: Tick loop
- Implement the tick loop: input → semantic search → upgrade → core attends → consolidator → response draft.

### M6: Semantic search
- Implement the dormant-index: a flat array of concept-slot vectors, *per dimension D* (one sub-index per D).
- Implement the query: thin bundle concept-slot → top-K dormant matches in the matching-D sub-index.
- Implement the upgrade: dormant hits become reference-arms in active.
- Verify: a 1M-bundle dormant index (one D, say 32D) queries in <100ms.

### M7: Core (the flock)
- Implement the per-core projection head: a per-kind ingress adapter (`slot_kind → d_model`) and per-kind egress adapter (`d_model → slot_kind`). The projection head IS the core's identity. Two cores with identical d_model but different projection heads are different cores.
- Implement the role rotation: online / consolidator / offline. Any core in the flock can take any role on a given tick.
- Verify: a flock of mixed-shape cores (e.g. one Tiny, one Mid, one Large) all load and rotate roles without the runtime caring about their specific shapes.
- Verify: a new core dropped into `D:/ashes/Core/core_*.pt` is picked up at next launch.


### M8: Training loop
- Implement the training loop: read Kimi corpus → tick loop → loss → backprop.
- Implement the loss family: next-bundle, reconstruction, contrastive, variance, covariance, mean-cos, head entropy, head diversity.
- Verify: the loss decreases on a small subset of the Kimi corpus.

### M9: Idle-fill worker
- Implement the "not sure" bucket: a list of (dormant_id, word) pairs.
- Implement the dictionary lookup: Kimi vocabulary cards, WordNet, internet search API.
- Implement the edge allocator: turn a dictionary entry into edge-slots on a bundle.
- Verify: a thin bundle for "obfuscate" gets filled with edges after one idle cycle.

### M10: End-to-end test
- Train on the full Kimi corpus for 1 epoch.
- Probe the geometry: similarity matrices, neighbor recall.
- Run a sample inference: input a question, observe the response draft.
- Verify: the model produces coherent English, demonstrates learned vocabulary, demonstrates semantic relationships.

---

## APPENDIX A: BASIS MATRIX

The 32×8 basis matrix `W_basis` is committed to the v6.0 release artifact at `D:\00\axon_runtime\config\basis_matrix.npy`. The matrix is generated by a fixed algorithm (in `D:\AxonCurriculum\src\axon_v6_core\letter_substrate.py`) and committed to the repo for reproducibility.

The generation algorithm:
1. For each of the 32 features, assign a target variance and a target distribution.
2. Use a hand-crafted orthogonal initialization to ensure the 8 output dimensions are uncorrelated.
3. Apply a hand-tuned scaling per output dimension to balance the magnitudes.
4. Commit the resulting matrix. The algorithm and the committed matrix are both part of the v6.0 release.

---

## APPENDIX B: KIMI PROMPT

The Kimi prompt for generating the v6 text curriculum is at `D:\AxonCurriculum\config\kimi_prompt.txt`. The prompt requests 5 layers of plain-text English:

1. Vocabulary cards (30,000 entries)
2. Grammar lessons (5,000 lessons)
3. Thesaurus relations (20,000 entries)
4. Reading passages (2,000 passages)
5. Compositional chains (10,000 chains)

The output is text. The model reads it as user input. No token IDs. No embeddings from external models. The model learns English from English.

---

## APPENDIX C: WHY v6 IS DIFFERENT FROM EVERY PRIOR VERSION

| Aspect | v3 (Schoolhouse) | v5.1 (Slot-Native Seed) | v6 (Letter-Substrate Native Bundle) |
|---|---|---|---|
| Core size | 512D | 768D, 12-head | **a flock of diverse cores (Tiny → XLarge); no single seed prescribed** |
| Substrate | "small semantic atom" | hash-derived 64D ID | hand-built 8D letter basis |
| Hash input | banned but partial | SHA-256 hash → embedding table | **banned absolutely; no fallback ever** |
| Tokens | banned | banned but partially used | **banned; character-stream input** |
| Multi-core architecture | single core | single core | **a flock of diverse-size cores, online/consolidator/offline roles, round-robin scheduler, readiness check gates offline→online** |
| Dormant/active split | not enforced | not enforced | **strictly enforced with mask** |
| Tick loop | one tick per cycle | one tick per cycle | **with semantic search before attention** |
| Bundle structure | composite of slots | substrate + address | **concept slot (D ∈ {16,32,64,…}) + frozen 8D letter sequence + edges (D-dim); unit-of-meaning in the active state** |
| FFN ratio | ~2x | ~32x | **64×–128× band, per-core, not a hard rule** |
| "Not sure" bucket | not specified | mentioned briefly | **first-class feature with idle-fill worker** |
| Idle digestion | not specified | mentioned briefly | **dictionary-driven, edge-allocating** |
| Kimi text curriculum | not mentioned | not mentioned | **primary training data** |
| Moral foundation | not present | not present | **Ten Commandments, hand-written, Layer 6, auto-loaded at construction** |
| Conversation history | not present | not present | **first-class unit-of-meaning bundle (concept slot + 8D letter sequence + edges), projected through the ProjectionBank, every core can read AND write** |
| Substrate identity | not frozen | not frozen | **8D substrate is frozen; 8D IS the letter; renderer reads 8D directly, no learned decode head** |
| Renderer | not specified | not specified | **invariant: reads only 8D letter sequences from the dormant pair; never decodes from concept-dim back to 8D** |
| Concept-slot scaling | not present | not present | **per-bundle dimension D ∈ {16, 32, 64, 128, 256, 512, 1024, …}; concept-slot graduation grows D, not the substrate** |
