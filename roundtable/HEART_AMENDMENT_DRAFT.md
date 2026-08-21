# Heart Amendment — Draft for Jeff's Ratification

Status: DRAFT. Not doctrine until Jeff ratifies and it lands in both
`docs/SOURCE_OF_TRUTH.md` and root `SOURCE_OF_TRUTH.md` (exact byte mirrors,
test-enforced).
Author: Kimmy / Kimi Code CLI / 2026-08-21
Basis: Jeff's rulings of 2026-08-21 (recorded on the engineering bus steering
board) and the converged Kimmy/ChatGPT deltas (bus rounds 1-2).

## What this amendment does

It names and binds the **Field Compiler Organ — the heart** — as a real organ
with its own cadence, and sharpens the existing per-tick cycle so that:

- the heart, not an ambiguous "runtime", owns every canonical mutation;
- a tick runs against a frozen canonical base, never a moving one;
- core proposals are sparse edits, not full-field rewrites;
- proposals live in a noncanonical workspace, never in canonical regions;
- the dormant valve gains true semantic relevance without any new memory
  authority and without any trained encoder that does not exist;
- the 64D rail becomes a dual surface: lossless exact scaffold plus derived
  semantic slots, with roundtrip guaranteed on the exact side.

Nothing in this amendment weakens: one canonical shared field, one dormant
authority (the recovered JSONL corpus), the frozen 16D character substrate,
typed-delta validation, immutable provenance, fail-closed behavior, or the
ban on scripted/fake organs. Specialization of cores (region/domain focus,
per-core attention scopes) remains a **future** amendment, gated on a working
64D heart.

## Amendment text (proposed splice into SOURCE_OF_TRUTH.md)

### Section: The Heart (Field Compiler Organ)

The Field Compiler Organ is Axon's heart. It runs on its own cadence — the
**heartbeat** — which is distinct from a cognitive **tick**. The heart pumps
exact information: external input (users, tools, advisors) inward to the
organs, and organ output outward, roundtrip. The heart is the only organ that
may create, validate, or commit canonical shared-field state. Cores,
consolidators, ingress paths, and the dormant valve never mutate canonical
state directly; they emit proposals that cross the heart's typed
validation/transaction boundary.

A heartbeat is event-driven: a canonical field change is the primary
doorbell. While input or commit work exists, the heart beats promptly. While
idle, the heart may keep a slow bounded liveness beat. Proposal-board
activity advances the in-flight tick workspace and does not by itself create
a new canonical field.

Each beat:

1. Drains queued external arrivals. Between ticks, intake commits immediately
   as a typed delta into its runtime-owned region (e.g. `user_input`,
   `tool_results`, `advisor_input`). Arrivals during an in-flight tick queue
   for the next beat and never mutate the frozen base.
2. Detects change via canonical field identity/freshness. No change means no
   recompilation.
3. Runs the dormant valve (below) when change warrants recall.
4. Recompiles the affected rail(s), proving complete coverage and exact
   roundtrip against the fresh canonical field.
5. Services the tick workspace (below): collecting proposals, enforcing
   barriers, and committing the validated consolidator decision.

### Section: Tick lifecycle (replaces the per-tick list under "Cores")

A **tick** is one full deliberation round against a frozen canonical base:

1. The heart stabilizes intake and dormant recall, freezes canonical field
   `F_N`, and emits for each active d_model rail a derived, immutable tick
   image `R_N`. Rails and tick images are projections, never second canonical
   state.
2. The heart declares the tick's participant set from the core registry
   (active / offline-training / disabled, rail membership).
3. Each participating core inhales its private soul, attends the complete
   tick image with a coverage proof, emits a **sparse proposed delta** (only
   the edits it proposes, each bound to `F_N` with author/rail/pass
   provenance), and exhales the experience.
4. The first pass closes when every required participant has returned,
   failed, or timed out under governed policy. The heart then exposes the
   complete first-pass proposal board in the per-rail workspace.
5. Each core inhales its updated soul, re-attends the tick image plus the
   complete proposal board, emits one refined sparse delta, and exhales.
6. After the refinement barrier, the rotating consolidator attends the tick
   image plus every refined delta and emits one proposed authoritative delta.
   The consolidator is the final reasoning authority of the tick; it still
   only proposes.
7. The heart validates the consolidator's proposal (typing, authority class,
   base freshness, provenance) and atomically commits it, producing the
   successor canonical field. The tick ends at that commit — and only there.

"Against the entire shared field" means authored against the exact frozen
base with field-wide addressability as permitted by authority class; it never
means reproducing unchanged content.

### Section: Authority classes (replaces informal write-policy wording)

- External ingress (user/tool/advisor) may write only its runtime-owned
  regions, only between ticks.
- The dormant valve may materialize governed `structured_knowledge`.
- Core proposals may target only the scopes their authority class permits.
- The consolidator's proposal may address every canonical region as governed.
- Only the heart's transaction layer converts any proposal into canonical
  state. Validation must detect and reject conflicting or overlapping sparse
  edits and any proposal not bound to the frozen base.

`CORE_WRITABLE_REGIONS` or similar bootstrap restrictions are implementation
restrictions, not doctrine; widening happens only through this authority
model.

### Section: The dormant valve (semantic relevance)

The dormant valve is part of the heart. The recovered JSONL corpus remains
the sole dormant-memory authority. Candidate ranking combines graph/edge
semantics, lexical support, confidence/type/task relevance, and novelty
versus the active field, under a governed budget.

Heart-v1 semantic relevance is **recovered-edge graph semantics**: the
readable English semantic edges and container graph are walked from the
changed text to find related memories. This is real semantics already owned
by the corpus; no trained encoder exists and none may be simulated. Learned
vector similarity, when trained later, is a disposable derived sense only.

The semantic path fails closed to exact lexical retrieval. Exact bytes are
always dereferenced from the authoritative JSONL and hash/provenance-verified
before surfacing. The derived index is maintained incrementally:
generational append/update with binding verification and atomic swap; full
rebuild is the fallback, never the per-beat cost.

### Section: Dual-surface rails (64D first)

Each d_model rail is a dual surface over the frozen tick image:

- **Exact scaffold:** the lossless, region-preserving, provenance-complete
  packing of exact 16D character cells (the existing D64 compiler contract),
  carrying the coverage and roundtrip guarantees.
- **Semantic slots:** derived slots for words, phrases, sentences,
  paragraphs, concepts, and edges. Every semantic slot carries source-span
  references back to exact canonical characters. Semantic slots are derived
  and rebuildable; they never become the only copy of anything, never gain
  reasoning vote, and never gain commit authority. Heart-v1 semantic slots
  are deterministic derivations from exact structure; trained semantic
  richness enters in the 128/256/512/1024 dialing sequence, one rail size at
  a time, each proven before the next begins.

The exact scaffold alone carries the roundtrip guarantee. A reasoning core is
never required to act as a stenographer to prove it read the field.

## Existing text this amends

- "Cores" per-tick enumeration → replaced by the tick lifecycle above;
  "runtime validates and atomically commits" is named as the heart.
- "Deterministic D64 Field Compiler" → extended with semantic slots and the
  heart role; compiler remains authority-free except for the heart's
  transaction layer, which is the commit boundary the compiler serves.
- "Dormant Evidence Bridge" → extended with graph-semantic candidate ranking
  and incremental generational index maintenance.
- Soul cycle wording (inhale/attend/propose/exhale/re-inhale/refine) → pinned
  to tick steps 3 and 5 against the rail image, with the consolidator
  following the same inhale/attend/exhale discipline at step 6-7.

## Ratification checklist

- [ ] Jeff rules the draft (edit / accept / reject per section).
- [ ] Splice into `docs/SOURCE_OF_TRUTH.md` and root mirror byte-identically.
- [ ] Mirror-equality tests pass; full suite green.
- [ ] Ledger event records the ratification; bus steering board updated.
- [ ] Only then: build order A (heart-owned commit types, core registry/tick
  identity, frozen tick image, proposal board) in permanent anatomy.

**Kimmy / Kimi Code CLI / 2026-08-21**
