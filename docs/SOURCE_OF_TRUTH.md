# Axon — Source of Truth

Date: 2026-07-03
Repo: `D:\Axon`
Archive repos (read-only): `D:\AxonGliksbot`, `D:\axon7`

This file is the source of truth for the intended Axon architecture. The code
is allowed to be behind this document. If the architecture changes, update this
file first or in the same change.

## One Sentence

Axon is a forever-ticking, token-free, slot-based English-bearing substrate
agent whose active shared field is made of surfaced conversation, knowledge,
tools, scratch, diary, awareness, task state, context annotations, response
draft, and control regions; whose semantic core builds readable semantic edges
over structured records; and whose dormant structured knowledge is searched
each tick to surface relevant facts into active state.

## Ground Rules

- `D:\Axon` is the clean active repo.
- `D:\AxonGliksbot` and `D:\axon7` are read-only archives. Never modify, move,
  or delete anything in them.
- Runtime tools are part of Axon's agency.
- Do not add runtime tool lockdowns, approval gates, truncation, sandbox
  behavior, or artificial limits without an explicit design discussion first.
- Training curriculum pacing is allowed. Runtime autonomy limits are not added
  casually.
- The source of truth is the architecture in this document, not whatever
  partial implementation happens to exist today.
- The word "row" is retired repo-wide in docs. The shared field is built from
  fixed-width slots; database or container records are called records, not
  rows.

## Layer 0: Slot Substrate

The bottom floor is the frozen 16D character substrate (`substrate/substrate.py`).
It is the alphabet: every supported character maps to one frozen, hand-authored
16D vector, and every character round-trips exactly through nearest-code
decode. This is the same deterministic letter bridge that served the legacy
recall and morphology trainers. It is not a learned embedding matrix and not a
vocabulary lookup table.

Above the alphabet sits the slot layer. The shared field is made of **regions**.
Each region holds many **slots**. A slot is one fixed-width vector. The
canonical slot width is 8192D everywhere.

### Vocabulary

The active shared field is divided into regions:

- `conversation_history`
- `response_draft`
- `structured_knowledge`
- `tool_results`
- `scratch`
- `diary`
- `awareness`
- `task_state`
- `context_annotations`
- `control`

A region is a contiguous sequence of slots. Paragraphs, documents, and long
messages chain across many slots in region order.

### Slot Layout (8192D, deterministic, one width everywhere)

- **dims 0-4095** — text payload. Up to 256 characters, one frozen 16D substrate
  code per character slot, packed by concatenation.
- **dims 4096-6143** — edge payload. Up to 128 characters of spelled-out edge text,
  same 16D packing.
- **dims 6144-6655** — control block. Kind, length, chain links, status, and
  edge-form flags encoded as substrate characters. This block is
  ensemble-visible and reasoning-relevant.
- **dims 6656-8191** — reserved zeros.

Provenance hashes, timestamps, and source pointers live in the container
**record**, not in slot dimensions. Layout metadata is not ensemble-visible.

### Deterministic Pack and Unpack

All runtime construct/deconstruct is deterministic:

- **pack** = concatenation of frozen 16D substrate codes.
- **unpack** = read slots + nearest-code decode.

`pack -> unpack -> identical text` is a hard substrate gate, lossless by
arithmetic, 100 percent, gated in tests. There is no learned encoder or
decoder in the runtime read/write path.

Long text chains across slots. Each slot carries enough control-block
information to know its kind, length, and the next slot in the chain, so a
region can be traversed without hidden out-of-band state.

### Edge Payload and Overflow Contract (R1)

The slot edge payload (dims 4096-6143, 128 chars) is a bounded **view** of the
container's full edge record, which lives whole in the dormant/container record.
No semantic edge is ever dropped silently.

**Edge label system (R1 corrected position):**

1. Every edge is born as readable English text: edge_type plus target, with
   optional sense, directionality, provenance, confidence, and status.
2. The system does not create semantic-edge aliases. Compact `{AA}`-style edge
   codes force cores to learn a private vocabulary and are rejected.
3. Slot edge-payload pack policy (deterministic): pack full edge text.
4. Edge overflow contract: if edges exceed the payload, the packer must
   (a) chain an edge-continuation slot, or (b) surface a
   prioritized subset by explicit, recorded policy. Either path is
   deterministic, tested, and counted.
5. The context-annotation lane may add short English explanations, never opaque
   semantic-edge symbols.

### Why One Slot Per Semantic Unit

A slot holds one semantic unit, typically one sentence of up to 256 characters.
The rationale: after adapter projection, one slot reaches a core as `d_model`
floats; slot capacity and core `d_model` are co-tuned. Designs that put one
slot per region are rejected because padding scales with the maximum rather
than the typical length, and attention/masking/delta granularity collapses to
a single update.

### Learned Compression Is Reassigned

Learned compression is no longer the field construct/deconstruct mechanism. It
is the mechanism for soul temperature-tier compression passes
(hot -> warm -> cold). The capsule-core trainer machinery (cross-entropy
exact-fill, collapse tripwire, smoke gates, rolling-3 checkpoints) is the
prototype for those passes. The runtime field stays deterministic.

The Capsule Core is therefore not a runtime organ for reading and writing the
shared field. It is a research prototype whose training discipline is reused
for soul compression. Any future learned compression module must still respect
the substrate round-trip gate if it ever materializes text back into the field.

Multiple reasoning cores at different d_model sizes (64, 128, 256, 512, 1024,
future sizes) attend over the same slot field through shared per-d_model
adapters and work together as a single mind. Ensemble, rotating consolidator,
private temperature-tiered souls, deltas, and training contracts remain
unchanged from the prior architecture except where the field width and slot
mechanics touch them.

The response draft remains English. Axon does not write one letter per tick,
and the shared field does not need one slot per letter. A committed response
decodes through deterministic nearest-code unpack before leaving the system.

## Layer 1: Containers

Letters group into words, phrases, concepts, observations, tool results, and
facts through containers.

A container is a structured Python/PyTorch dictionary-style record. Its payload
contains letter sequences, decoded text, semantic edges, provenance, and status.

A container is not a whole-text embedding and is not encoded as one object by
the substrate. The container is the envelope. Inside it, words/letter sequences
and semantic edges remain clearly separate:

- `[word]` / `letters` fields are letter sequences.
- `{edge}` records are typed semantic relationships with spelled-out relation
  text, targets, confidence, provenance, and status.

When a container is materialized into canonical state, participating English
surface text remains recoverable and is grounded in substrate characters. A
slot may carry a phrase, paragraph fragment, tool-result summary, scratch
hypothesis, surfaced fact bundle, or semantic edge bundle, but it must retain
exact source text and/or provenance metadata so it can be inspected and
rendered.

Conceptually, the container is shaped like:

`(<[d][o][g]>{edge: is a -> animal}{edge: has property -> furry}{...})`

Where:

- `(...)` is the container/envelope.
- `<...>` is the word/letter payload region.
- `[d][o][g]` are individual substrate character slots.
- `{edge: ...}` records are attached typed semantic edges in readable English.

Substrate rule:

- The ensemble must be able to see every materialized letter.
- Edge meaning is spelled out as English edge type plus target text.
- Opaque semantic-edge symbols are not part of the active doctrine.
- Slots are English-bearing: grounded in exact text, edges, tool
  output, or structured facts.
- No arbitrary opaque state slots are allowed in the active shared field.

Canonical container fields should include:

- `id`
- `kind`
- `letters`
- `text`
- `normalized_text`
- `spans`
- `edges`
- `symbols` (legacy compatibility only; active no-symbol path keeps it empty)
- `source`
- `source_tick`
- `created_tick`
- `updated_tick`
- `confidence`
- `provenance`
- `status`

The exact schema can evolve, but containers must stay structured and auditable.

## Layer 2: Semantic Edges

The solitary semantic core is responsible for inspecting containers and
creating semantic edges. Its job is to discover relationships and attach
readable edge records to containers.

Semantic edges are permanent and auditable records:

- `label` (full human-readable canonical form)
- `edge_type`
- `target`
- `sense`
- `directionality`
- `description`
- `created_by`
- `created_tick`
- `examples`
- `confidence`
- `status`

The full canonical label is the source of truth. The system does not depend on
opaque edge symbols such as `{AA}` or `{K9}` because every reasoning core can
already read English edge text in the 8192D shared field.

## Layer 3: Canonical State

The canonical state is the shared field. It is always grounded in 8192D slot
English-bearing field space. It is the common state that cores attend over,
either directly when their d_model matches the field width or through shared
state adapters when they use a smaller private d_model.

The canonical state has two major parts:

1. Active state
2. Dormant or masked state

### Growing Regions And Sliding Masks (convener amendment, 2026-07-03)

Every region GROWS CONTINUOUSLY and is NEVER truncated or evicted. Active
versus dormant is not two containers — it is one append-only tape per region
with a SLIDING MASK deciding what the ensemble attends this tick:

- Each new slot appends to its region's tape. Older slots are pushed deeper
  as the tape grows; depth beyond the mask IS the dormant state for that
  region. There is no separate dump bucket — the tapes are the capture.
- The mask is a VIEW, never a mutation. `conversation_history` might be
  masked to the last 10 turns, 100 turns, or 0 turns depending on the
  moment; moving the mask changes what is attended, never what exists.
- Mask windows are per-region policy: defaults in config, adjustable by the
  runtime, the convener, and — because runtime autonomy is doctrine — by
  the cores themselves via a typed delta op (`set_mask{region, window}`).
- Storage form follows depth: masked-in (active) slots are materialized
  8192D vectors; deep tape lives as slot RECORDS (packed text + edges +
  provenance) and is re-materialized on unmask. Growth is unbounded in
  records, bounded in RAM.
- CONSOLIDATION PASSES (personal hygiene): consolidation runs during IDLE
  ticks — this is a core purpose of the forever tick. When Axon is done
  with a task, he has more work to do: the cores mask off most of the
  active state and unmask the deep ends of each region in CHUNKS,
  processing them into semantic edges, episodic memory, and structured
  knowledge — the structuring duty that previously belonged to the dump
  bucket. Chunks may equally be DRAINED to API curation workers (the Track
  1b propose/dispose validator consumes region tails).
- REGION LIFECYCLE (convener refinement, 2026-07-03): three stages, one
  temperature ladder shared with the souls. (1) ACTIVE WINDOW: masked-in,
  materialized 8192D vectors, RAM. (2) UNPROCESSED TAIL: slot records on
  disk in the working state, awaiting hygiene. (3) ARCHIVE: once a chunk is
  processed — edges extracted, episodic memory built, knowledge filed into
  the dormant knowledge base — its raw records are EVICTED from the working
  state into an append-only archive database: still searchable, still
  restorable, full provenance (eviction-to-archive is filing, never
  deletion), but out of RAM and out of the working set. History grows into
  terabytes on disk, never in Axon's working memory.

### Active State

The active state is the masked-in window of every region tape, attended over
by the ensemble each tick. Regions:

- `conversation_history`
- `response_draft`
- `tool_results`
- `scratch`
- `diary`
- `awareness`
- `task_state`
- `context_annotations`
- `control`
- `structured_knowledge`
- delta/draft regions

The `structured_knowledge` region inside the active state is not the whole
memory. It is the relevant slice that has been surfaced for this tick.

### Dormant Masked State

The dormant state is where structured knowledge lives cheaply. It contains
facts, triples, relationships, patterns, prior observations, stable project
knowledge, semantic edge records, and containers not currently active.

Dormant knowledge remains in structured-record form plus optional 8192D slot
snapshots, but it is masked from direct ensemble attention by default. It is
searchable and editable.

### Capture And Structuring (dump bucket superseded, 2026-07-03)

Every input must be captured — and IS captured, by the growing region tapes
themselves (every interaction, tool result, advisor input, user message,
model output, and runtime observation lands in some region). The old
separate raw dump bucket is superseded by the Growing Regions amendment.
Structuring is the consolidation-pass duty: region tails are processed (by
cores in sleep cycles, or drained to API curation workers) into facts,
triples, relationships, patterns, containers, semantic edges, and curation
candidates. Raw experience never remains only raw — but now it also never
leaves its region of origin, so provenance is structural.

### Bootstrap Dormant State

The first dormant state is built deterministically by an
offline importer (`curator/semantic_layout_machine.py`), NOT by waiting for a
trained semantic core. The importer reads recovered memory DBs in streaming
fashion, classifies items, assigns layout metadata, and emits
container_schema-compatible dormant containers, spelled-out semantic edges,
layout groups, and a corpus manifest.

The semantic core later trains as curator and builder from this deterministic
ground truth.

## Layer 4: Semantic Search And Surfacing

Each tick performs semantic search over the dormant masked state. The search
uses spelled-out semantic edges, exact text, slot metadata, and container
relationships. Relevant dormant knowledge is surfaced into the active
state's `structured_knowledge` region.

### Context-Annotation Lane (R6)

A `context_annotations` region is owned by an async worker (API first, trained
later, outside the tick gate). It writes:

- Short English edge/context explanations
- Why-this-surfaced notes
- Source confidence
- Conflict notes
- Request-for-more-context markers

It never overwrites source slots and never blocks the tick.

### Edge-Ablation Gate (R3)

Edges must prove retrieval benefit empirically before they are trusted.
Experiment design:

- Held-out QA set from recovered DBs: one-hop, two-hop, procedure,
  project-memory, and negative-control questions.
- Frozen answer core, slot budget, corpus, and context window.
- Conditions: (1) exact text only; (2) + exact edge lookup; (3) + edge-walk
  1 hop; (4) edge-walk 2 hops; (5) + context-annotation
  capsules.
- Metrics: answer accuracy, recall@{1,3,5}, precision@k, active-slot count,
  latency, sibling-noise rate.
- Pass: edges improve accuracy or recall on held-out questions; improvement
  survives negative controls; active-field noise stays under a fixed budget.
- Run after Lane 2 exists. Never on proxies.

## Layer 5: State Adapters

State adapters map from the 8192D slot shared field into each core's model
dimension and back. They are the bolted barrier between the shared canonical
state and each core's private space.

Adapters belong to the shared state infrastructure. They are not owned by cores
and are not part of a core's identity. There is exactly one adapter per d_model
size, not one per core.

### Adapter Basis

Adapters operate on the frozen 16D substrate basis: a slot's text payload
(dims 0-4095) is a concatenation of frozen 16D character codes.

READ AND WRITE ARE ASYMMETRIC (convener-signed 2026-07-03). The original exact round-trip gate through a
per-slot d_model bottleneck is information-theoretically impossible (256
chars do not survive a frozen linear projection into 64 floats) and is
superseded:

- READ is lossy by design: the frozen adapter projects a whole 8192D slot
  to ONE d_model summary vector (compute law: one attention token per
  slot). Exact text lives in the FIELD, which is the source of truth;
  English is decoded from the field, never from a core's compressed view.
- WRITE is exact by construction: anything committed to the field is
  codebook-snapped (each 16D block to the nearest substrate code).
- Adapter gates (replacing the old round-trip gate, both verified at
  64/128/256D): SNAP-IDEMPOTENCE (validly packed slots survive
  DOWN -> UP -> snap unchanged) and SEPARABILITY (slots differing by one
  character produce distinct projections).

### The Write Path (convener correction, 2026-07-03)

LOCKED LAW (convener): TRAINED PARAMETERS LIVE IN CORES AND NOWHERE ELSE.
Everything between a core and the field — read or write — is FROZEN
ARITHMETIC: minted, deterministic, checkable, weightless. The prior
shared trained decode organ violated this law and is RESCINDED.

The conforming write path (the proven 16D-era pattern, generalized):

- The CORE unrolls its own characters. When a core writes a span, its own
  trained layers emit one d_model vector per character position — trained
  parameters inside the core, where they belong. The voice IS the core.
- Each character vector crosses to the field through a FROZEN per-character
  prototype decode (same minted sign-vector family as the read adapter and
  the legacy rails): nearest-code, deterministic, no weights.
- The deterministic packer and codebook snap commit the result. The field
  stays exact by construction.
- READ is unchanged: one frozen summary vector per slot (compute law).
- COST placement: writing N characters costs N output positions inside the
  writing core, only for spans it edits — never read-side attention, never
  a shared trained module.

Surviving write-path rules that do not involve trained projection:

- COMMIT ONLY DIFFS: the runtime diffs decoded output against the current
  slot and commits only changed positions as typed deltas.
- Length is explicit and countable (declared in the typed delta / control
  block) — truncation is never silent.
- EDGES NEVER PASS THROUGH CHARACTER DECODE: typed edge deltas; the
  deterministic packer and canonical edge records remain authoritative.
- Gates before any long run: read-fidelity probe plus the cf-probe write
  gate — positive exact-fill > 90 percent, zero/swapped/irrelevant-field
  controls < 5 percent.

Kimi's recorded dissent (per-core voice, delta-declared length) is
substantially VINDICATED by this correction and is absorbed into the law
above.

Convener note (2026-07-03, sign-off): souls remain NATIVE d_model thought
vectors (Layer 6 unchanged). Readable personal memory lives in the field's
diary region. Soul auditability is behavioral (cf-probes, bake gates);
hot-to-cold compression distills content before any adapter baking, so
cold-tier readability is not a blocker.

The archive `legacy_8192/state_adapter.py` is a design template only. Its
prototype-key projection and round-trip check pattern are good; its substrate
basis is wrong (it imports the old wide_substrate, not the frozen 16D alphabet).
The active adapter lives in `adapters/slot_adapter.py`.

Target core adapter sizes (first set): 64, 128, 256. Future: 24, 48, 512, 1024.

Adapter artifacts: `adapters/slot8192_adapter_{d_model}d.pt`.

A core checkpoint records its d_model. It does not carry adapter weights.

## Layer 6: Cores And Souls

There is a solitary semantic core role. The semantic core's purpose is to
inspect containers, create readable semantic edges, and keep semantic structure
growing.

There is also an ensemble of model cores at different dimensions: 64D, 128D,
256D, future 512D, 1024D.

Each core attends over the same active canonical state after projection through
the shared-state adapter for that core's d_model.

### Souls

Each core gets its own personal soul.

- Every individual core has a private soul.
- Souls are inhaled at the start of tick work.
- Souls are exhaled/updated after the core has produced its delta.
- Persistent factual knowledge belongs in dormant structured state, not in a
  hidden shared soul.
- Input never enters the soul first.
- The soul write path is separate from the response/delta path.

### Soul Encoding

The soul lives in the core's native d_model space. It is never projected
through the shared-state adapter. It is never encoded as substrate characters.
Each soul slot holds a d_model-dimensional thought.

### Soul Persistence

The soul persists across ticks, across episodes, and across sessions. It is
saved separately from the checkpoint (as soul state) and restored on load.

### soul_v2 Integration

`soul_v2.py` implements the temperature-tiered soul. It is wired into
`core.py` via a new `soul_mode="v2_tiered"` option, config-gated, so existing
checkpoints are unaffected. The exhale path (write to hot soul) is a new code
path, not present in the current core.py.

## Layer 7: Forever Tick Loop

Once Axon is turned on, he ticks forever.

Each tick follows this conceptual loop:

1. Capture new inputs into the raw dump bucket.
2. Structure any ready raw material into containers and dormant knowledge.
3. Perform semantic search over dormant masked state.
4. Surface relevant knowledge into active `structured_knowledge`.
5. Each core inhales its private soul.
6. Each core attends over the active shared field.
7. Each core produces a delta over the state.
8. Each core exhales/updates its private soul.
9. Each core inhales again.
10. Each core attends over the other cores' previous deltas.
11. Each core produces a refined delta.
12. The consolidating core attends over the refined deltas.
13. The consolidating core produces the committed active-state update.
14. The response draft becomes more refined.
15. The consolidator role rotates.
16. The next tick begins.

### CPU Residency Principle

The runtime tick loop must stay matmul-light (adapters + small cores only;
pack/unpack is zero-FLOP concatenation of frozen codes) so the living agent can
tick on CPU. GPUs are for offline training only.

## Layer 8: Deltas And Consolidation

Cores produce deltas. A delta is a proposed replacement of the shared field in
8192D slot field space. It is the core's full proposed field state — not an
additive diff. Each core says "the field should look like THIS." Only the
consolidator's delta actually replaces the shared field.

### Typed Delta Records (R5)

The core may internally produce a proposed 8192D field snapshot (for training
simplicity). The runtime derives explicit typed delta records from each
proposal:

- `update_slot`
- `append_slot`
- `clear_slot`
- `update_response_draft`
- `attach_edge`
- `request_surface_memory`
- `request_tool_call`

The consolidator attends over proposals and commits typed deltas.
Deterministic CPU diffing; auditable state mutation; no added transformer
compute.

### Consolidator Rotation

No permanent tyrant core. The consolidator role rotates round-robin.

## Layer 9: Response Draft

The `response_draft` region is where Axon builds his answer. It evolves in real
time as ticks pass. The ensemble sees the draft, reasons over it, edits it, and
refines it. The draft is not a one-shot completion. It is a living region of the
active state. Axon's final response is the stabilized/committed version of that
draft.

The response-draft commit marker is a state contract separating in-progress
draft from committed output. It is not a runtime autonomy limit.

## Layer 10: Offline Cores And Distillation

There is always at least one offline core. Offline cores train while Axon is
alive. Their main job is not to bake raw facts into weights. Raw facts belong
in dormant structured knowledge.

The job of offline distillation is to bake experience, intuition, and skill
into parameters.

## Layer 11: LoRA And Parameter Growth

Offline cores distill cold layers of their soul and repeated experience into
LoRA adapters. Adapters should represent skills, habits, repeated workflows,
learned anticipations, stable heuristics, and procedural intuition — not
ordinary facts.

## Layer 12: Temperature-Tiered Soul And Adapter Baking

The soul has temperature tiers. Thoughts enter hot, compress through warm and
cold, and the coldest tier distills into LoRA adapters during offline cycles.

### Tier Structure

HOT (many slots, every tick) — active working memory. Categories are the write
template. Every exhale evaluates each category and decides whether to add or
update a hot slot.

WARM (fewer slots, compressed) — multiple hot thoughts compressed into fewer
warm slots. Periodic compression pass.

COLD (fewest slots, dense) — many episodes compressed into compact wisdom
representations. Category-agnostic or meta-categorical. Less frequent
compression.

FROZEN / DISTILLED (in core weights via LoRA) — cold slots distilled into LoRA
adapters during offline cycle. After distillation, source cold slots are
cleared.

### Dynamic Soul: Inflation And Deflation

The soul is not a fixed-size array. It inflates and deflates. The soul tensor
is over-allocated to a max capacity. An active mask tracks which slots are
live. Inactive slots are zeroed and invisible to cross-attention.

INFLATE: triggered when exhale produces a thought that does not fit any
existing hot slot (router confidence below threshold). Find an inactive slot,
write, set active. If none remain, trigger compression.

DEFLATE: triggered by salience decay. Each slot tracks a running average of
attention weight. Low-salience slots are candidates for compression or
eviction.

### Category Map

Hot tier category map v1: `episodic`, `lessons`, `diary`, `awareness`,
`scratch`, `tasks`. Warm and cold tiers may use different schemes or become
category-agnostic.

### Cold-Soul to LoRA Distillation (R4 — Deferred, Gates Hardened)

Sequencing: after core reasoning lanes work. Design notes for when it goes
live:

- Classify cold slots procedural vs episodic via compression lineage +
  cross-slot similarity. Only procedural candidates distill. Unique episodes
  stay in soul/records.
- Bake gates: held-out procedural task improvement with source cold slots
  masked (generalization, not just recall); must not increase false factual
  recall; manifest links training traces, cold-slot lineage, evals, redaction.
- Target: "behave better because I have done this class of thing before," never
  "remember this fact."

## Layer 13: Training Contracts

Recall and morphology training proved useful because the output target was
exact and hard to game. Future trainers should keep that discipline.

### Train As You Live (convener directive, 2026-07-03 — BINDING)

Cores train in exactly the posture they will live in. Every trainer:

1. Places the lesson in the shared field's regions (the trainer is just
   another writer of field slots); unused regions are masked.
2. Requires the answer in the `response_draft` region — the same mouth,
   the same typed-delta commit path as runtime.
3. SOULS BREATHE EVERY STEP FROM DAY ONE (supersedes the nextsteps-0703
   deferral of tiered souls). Every training step: INHALE (read the soul)
   -> attend the field -> produce the draft answer -> EXHALE (write the
   soul). Fully built soul_v2 temperature mechanics from the first step.
   Garbage soul content early is expected and acceptable; the breathing
   must be real. "Soul" may be called PERSONAL MEMORY interchangeably.
4. Input NEVER enters the soul first (unchanged): lessons through the
   field, exhale after the act.

### Locked First Cores (convener, 2026-07-03)

| core | d_model | layers | heads | FFN     | ~params | 3060 fit          |
|------|---------|--------|-------|---------|---------|-------------------|
| A    | 64      | 2      | 1     | 131,072 | ~34M    | easy              |
| B    | 128     | 2      | 1     | 262,144 | ~134M   | easy              |
| C    | 256     | 2      | 2     | 524,288 | ~537M   | needs whole card  |

The huge FFN ratio is deliberate lineage (the proven 64,1,2,16384 pattern
scaled up): the FFN is the pattern/intuition capacity; knowledge stays in
the dormant state. Core C requires the GPU box's resident uvicorn stopped
during training (or an 8-bit optimizer); A and B train beside it.

### Phase 0: Field Pretraining (the volume question, answered)

The "ingest massive data first, patterns emerge" recipe is the LLM recipe;
here it is adapted, not copied: VOLUME COMES FIRST, BUT DELIVERED IN THE
LIVE POSTURE. Phase 0 streams bulk text (curriculum corpus + dormant-state
text) through the same loop as life itself — text packed into field slots,
core inhales, attends, continues/reconstructs into response_draft, exhales
— at high volume and low precision. Pattern recognition at scale, gained
without ever leaving the train-as-you-live contract. Then the precision
curricula (recall, edge prediction, exhale-filter, surfaced-knowledge QA)
tighten the same loop on the same machinery. Pretraining is not a separate
mode; it is the first, easiest, biggest rung of the curriculum.

### Locked Training Rules (R7)

(a) Reconstruction objectives over discrete substrate content must be discrete.
Per-slot cross-entropy over the registered codebook is the primary objective.
Continuous losses are auxiliary only.

(b) Every trainer must pass a short smoke run proving loss falls and the task
metric climbs above the constant-output floor BEFORE any long run is launched.

(c) No silent truncation. Over-length examples are skipped and counted
explicitly.

### Locked Training Interface

1. The trainer writes curriculum input into the shared field, not into the soul.
2. Shared-field regions that are not part of the drill are masked off.
3. The core inhales its current private soul.
4. The core attends over the masked shared field through the shared-state
   adapter.
5. The core produces a delta/response as its outward product.
6. Only after the delta is produced does the core exhale/write to its soul.
7. Soul writes move through the temperature path hot to warm to cold; direct
   curriculum injection into cold or into the soul before field attention is
   invalid.

### Training The Exhale Filter

The target behavior is not "copy input into soul." The target is:

1. Inhale current private soul.
2. Attend over the masked shared field.
3. Produce a delta/response.
4. Observe the result/outcome.
5. Exhale a selective experience trace into hot soul.
6. Use that trace on a later tick only when relevant.

Paired or multi-tick drills (Tick A: lesson in field, supervise delta + exhale;
Tick B: mask input, require recall from soul) with required probes: correct
soul, zero soul, swapped soul, irrelevant soul, field leak check.

## Layer 14: Tutor Hook

The trainer should eventually expose an active tutor hook for inspection and
intervention. Any tutor intervention must be turned into a learning target for
the core's own mechanism and then faded out.

## Layer 15: Corpus And Redaction

Curriculum data needs source tags, category tags, and redaction status before
it is treated as production training material. Secret-bearing operational dumps
are deferred by default.

## Layer 16: Checkpoint And Provenance Contract

Checkpoint claims must come from checkpoint payloads and recorded metrics, not
from filenames alone. Production checkpoints should record full training config,
core geometry, soul geometry, alphabet/substrate version, dataset manifests,
metrics, probe results, core state, soul state, and adapter manifest
references.

## Layer 17: Tool Call Contract

Axon's runtime tools are part of his agency. Do not add runtime caps,
truncation, sandbox behavior, tool lockdowns, or approval gates without
explicit source-of-truth discussion first.

## Layer 18: Agent Collaboration Bus

Axon may run a sidecar collaboration bus (`runtime/bus/`) for external agents.
The bus maintains a shared collaboration board: objectives, tasks, talking
points, questions, direct messages, connected participants.

Locked rules:

- The bus is not an approval gate.
- The bus is not a runtime sandbox.
- The bus is not a tool lockdown surface.
- The bus must not add permission tiers that limit Axon's runtime agency.

## Layer 19: Curation Contract (R2)

API workers (Kimi/Hermes/cloud) staff the curator role. Propose/dispose split:

- Workers PROPOSE: containers, edge candidates, canonical labels,
  confidence, provenance, source pointers.
- A deterministic VALIDATOR commits or rejects. Checklist: schema validity;
  SOURCE POINTER REQUIRED (claims without a source record pointer are
  committed as status=candidate, never as facts); canonical label dedup;
  edge direction/type check; provenance hash;
  confidence/status; payload capacity check; redaction/sensitivity status;
  append-only edge writes; CONTRADICTION GATE (a new edge contradicting a
  higher-confidence existing edge on the same (source, edge_type) is flagged
  for review, not auto-committed).
- Drift detection: every batch stamped `created_by: model:version:date`;
  periodic re-probe of a fixed 100-container held-out sample; Jaccard
  similarity below threshold flags review.
- All accepted AND rejected proposals accumulate as training traces for an
  optional future local curator.
- Cost expectation: bootstrap ~243k items ~= $50-250 API; ongoing trickle
  negligible; the real budget item is validator engineering.

## What Current Code Already Covers

- `substrate/substrate.py`: the frozen 16D character substrate.
- `cores/core.py`: transformer core with legacy soul modes (concat,
  act_reflect, act_reflect_v2).
- `cores/soul_v2.py`: temperature-tiered soul manager (not yet wired into
  core.py).
- `curator/kg_search.py`: knowledge graph search.
- `curator/container_schema.py`: container and edge schema.
- `curator/semantic_layout_machine.py`: deterministic dormant-state importer.
- `curator/recovered_corpus_builder.py`: recovered DB/JSON to dormant
  containers, spelled-out semantic edges, layout groups, and manifest.
- `curator/dormant_materializer.py`: dormant container materializer.
- `training/build_recovered_curriculum.py`: dormant artifacts to curriculum
  families for field surfacing, edge prediction, retrieval
  QA, procedures, episodic exhale filters, curation review, and diary-only
  self-reflection.
- `runtime/bus/`: sidecar HTTP/WebSocket collaboration bus.
- `slots/slot_spec.py`: 8192D slot pack/unpack (Step 0).
- `slots/slot_field_contract.py`: nine-region field contract (Step 0).
- `adapters/slot_adapter.py`: 16D-basis state adapters (Step 0).

Counterfactual probes remain required by Layer 13, but the previous legacy
`training/cf_probe.py` implementation was removed because it depended on the
rescinded `heads` path and the pre-slot projection bank.

## Current Training Reality

As of 2026-06-29, remote morphology training validated the recall-fill path:
128D and 256D cores passed SOUL_IS_READ on the morphology curriculum (float32
required; fp16 NaNs at 256D).

As of 2026-07-03, the capsule-core compression prototype settled the training
rules in Layer 13:

- `capsule_core_v1` reached 46.9 percent exact-fill at 100,000 steps with
  1.23M parameters. This is the current ceiling.
- `capsule_core_v2` collapsed to a padded-MSE constant-output solution, caught
  at 8,000 steps. This is the evidence that continuous reconstruction losses
  cannot be the primary objective for discrete substrate content.
- The planned 300,000-step `capsule_core_v3` long run is cancelled. Any future
  compression work must start from a smoke run satisfying Layer 13(b).

## Next Architecture Contracts To Implement

1. ~~Slot spec: 8192D pack/unpack, region layout, control-block, chains,
   edge overflow contract, exact-reconstruction gate.~~ (Step 0)
2. ~~Slot field contract: nine regions, masking, materialization.~~ (Step 0)
3. ~~State adapter revival: 16D-basis adapters for 64/128/256D.~~ (Step 0)
4. Container schema (promoted, needs slot-era integration).
5. Semantic edge schema (promoted, needs slot-era integration).
6. Layout metadata schema.
7. Active state region contract (slot-era materialization).
8. Dormant masked state storage contract.
9. Semantic search/surfacing protocol.
10. Temperature-tiered soul schema (soul_v2.py exists; wiring into core.py
    pending).
11. Soul write template: per-category exhale evaluation, router, inflate/
    deflate triggers.
12. Soul compression passes: hot->warm and warm->cold attention summarizers.
13. 8192D slot delta format: typed delta records (R5).
14. Consolidator rotation policy.
15. Adapter manifest format and bake gate runner.
16. Checkpoint payload/manifest standard.
17. Corpus manifest, redaction, and promotion pipeline.
18. Response draft commit marker.
19. Tool call syntax contract.
20. Tutor hook interface.
21. Bus-to-runtime surfacing hook.
22. Context-annotation lane (R6).
23. Edge-ablation gate experiment (R3).

## Locked Principles

- The 16D substrate is the frozen alphabet.
- The 8192D slot field is the shared English-bearing state.
- Slots are the unit of the shared field. All runtime pack/unpack is
  deterministic concatenation of frozen 16D codes with exact round-trip.
- The shared field is divided into regions: `conversation_history`,
  `response_draft`, `structured_knowledge`, `tool_results`, `scratch`,
  `diary`, `awareness`, `task_state`, `context_annotations`, and `control`.
- One slot holds one semantic unit, typically one sentence of up to 256
  characters.
- Semantic edges must not be silently truncated. The edge payload is a bounded
  view; the full record lives in dormant state. Overflow chains or surfaces by
  explicit policy.
- Edge labels are born and remain typed English edge text.
- State adapters are per d_model size, shared by all cores of that size, on the
  frozen 16D substrate basis.
- The adapter is the only bridge between 8192D slot shared state and smaller
  d_model cores. The soul never crosses the adapter.
- Semantics come from the semantic core creating readable semantic edges.
- Containers are structured records, not loose blobs.
- Layout metadata is auditable but is not semantic edge meaning.
- The canonical state has active and dormant/masked parts.
- Dormant structured knowledge is cheap, searchable, editable, and not attended
  over directly by default.
- Each tick surfaces relevant dormant knowledge into active structured
  knowledge.
- Each core has a private soul in native d_model space, never projected through
  the adapter.
- The soul has temperature tiers: hot, warm, cold, frozen (distilled into
  LoRA).
- The soul inflates and deflates dynamically based on salience and capacity.
- The ensemble attends over the active state together.
- Cores produce deltas in 8192D slot field space. The runtime derives typed
  delta records (R5).
- The consolidator commits the next active state. Consolidator role rotates.
- The response draft is shaped over time.
- Offline training distills experience into skill/intuition, not raw facts.
- Curation is propose/dispose: API workers propose, deterministic validator
  commits (R2).
- Training reconstruction over discrete substrate content must be discrete
  per-slot cross-entropy; continuous losses are auxiliary only.
- Every trainer must pass a smoke run before any long run.
- No silent truncation in training.
- The runtime tick loop stays matmul-light for CPU residency.
- Production checkpoint claims require payload/manifest evidence.
- Training data requires provenance and redaction status.
- Runtime autonomy is preserved unless explicitly redesigned.

## Open Maybes

- Whether only high-D/stable cores can be consolidators.
- Whether same-dimension shared souls are ever useful later.
- Exact final-response/tool-call commit marker.
- Exact adapter promotion metrics beyond the 90 percent minimum hot-read gate.
- Exact warm and cold tier category schemes, or whether they become fully
  category-agnostic.
- Exact compression cadence and whether it is clock-based or
  salience-triggered.
- Whether the semantic-curator role is staffed by a dedicated trained core or
  by offline workers initially.
- Sequencing of cold-soul-to-LoRA distillation: late, after core reasoning
  lanes work.
- Exact context-annotation surfacing policy and budget.
