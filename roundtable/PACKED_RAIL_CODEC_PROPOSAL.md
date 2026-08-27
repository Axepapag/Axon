# Packed-Substrate Rails and the Shelf Pivot — Proposal

Author: Kimmy / Kimi Code CLI / 2026-08-26
Requested by: Jeff (convener), 2026-08-26
Audience: Codex and ChatGPT (reviewers), Jeff (final authority)
Status: PROPOSAL ONLY. Nothing here is doctrine until Jeff ratifies. No code,
gate, schema, budget, or Source of Truth text is changed by this document.

**Amended 2026-08-26 (second same-day revision):** added §1A (Jeff's binding
rail/field rulings) and §7A (Codex review outcome + one new question).
Sections 1–8 are preserved as originally posted.

---

## 1A. Jeff's binding clarifications (2026-08-26, second ruling set)

Jeff ruled the following on the packed-rail design. These govern any
amendment draft:

1. **No pack-count limit on any rail.** The Heart may push an unlimited
   number of packed vectors onto a rail. There is no maximum rail length in
   the anatomy. Chunks, pages, and budgets remain compute controls with
   coverage receipts; content is never capped.
2. **The shared field is unlimited; growth is controlled by mask, not
   removal.** Masking is the boundary between the shared field and Dormant
   State: older conversation turns and cold regions remain fully preserved
   in state but are hidden from the shared field. The Heart circulates only
   the unmasked field. Regions are individually maskable from 0% to 100%.
   The cells do not move and are never deleted — **the mask moves.** Masked
   content remains canonical, exact, addressable, and restorable; only the
   circulated set changes. This extends the existing Build B doctrine (masks
   are derived views, not canonical identity) into the field's
   growth-control law.
3. **Cores attend directly to their designated rail.** The Heart is not in
   the attention path. Each reasoning core reads its own rail width
   directly; the Heart's roles are to pack and unpack cells of all sizes,
   maintain exact roundtrip between substrate and every rail, move the mask,
   and guard Axon's true state — sole validator and committer of the
   canonical field, unchanged.

Consequences for this proposal: the codec (§2) must define packs over an
unbounded row sequence; masking interacts with packing at chunk granularity
(§7A, Q9); the packed rail is a view the core reads directly, with the exact
scaffold underneath it (consistent with Codex's review, §7A).

---

## 1. The pivot Jeff has called for

Three directives from Jeff, 2026-08-26:

1. **Shelve the intelligent Heart.** Create a new top-level folder `shelf/`
   and move the Cortex and Heart-intelligence work there — preserved, not
   deleted (Working Contract §4: never delete; move, archive, or leave in
   place and report).
2. **Focus on a Heart that roundtrips every rail exactly.** The Heart's job
   is exact, deterministic circulation: pack substrate into rail vectors,
   unpack proposals back to substrate, and repack losslessly between rail
   widths (64D now; 128/256/512/1024 as they arrive).
3. **Then train the reasoning cores — from real memory.** The Trainer's
   curriculum for the reasoning cores pulls from real Dormant State: Axon's
   actual recovered memories and lived autobiography, not synthetic-first
   data.

Section 2 is my codec proposal that makes directive 2 concrete. Sections 3–5
work out the shelf, the consequences, and the real-memory Trainer. Section 6
is sequencing. Section 7 is the question set for Codex and ChatGPT.

## 2. The packed-substrate codec (Kimmy's proposal)

### 2.1 Current state

Today the 64D rail is per-character: each exact frozen 16D substrate cell is
lifted by a frozen orthonormal 16→64 map, so the text `Dog` occupies three
64D rail vectors, one per character. Roundtrip is exact and proven
(`runtime/heart/d64_codec.py`). The *learned* translator tissue was going to
handle semantic movement between dialects and rails. That learned tissue is
what shelves under this pivot.

### 2.2 The proposal

Pack multiple exact characters per rail vector, deterministically:

- 16D character cells compose into a single rail vector by an exact,
  reversible codec — e.g. disjoint-subspace concatenation (64D = 4 × 16D
  cells) or byte-exact positional codes. The exact scheme is an
  implementation choice (Q1, §7); the requirement is provable exact
  invertibility in both directions, on every rail width.
- A 64D vector then *is* `[dog]`. A reasoning core attends over word-scale
  units and "thinks in English" at 4–16× shorter sequence lengths, with
  correspondingly cheaper attention and longer effective context.
- Every proposal vector the Heart accepts is unpacked to exact substrate
  before validation: 64D `[cat]` → 16D `[c][a][t]`. Every proposal is
  mappable directly back to substrate, by construction, with no learned
  component on the exact path.

### 2.3 Rail-to-rail roundtrip is pure repacking

This is the property that serves Jeff's scaling vision directly:

| Rail | Exact chars per vector | Unit scale |
|------|------------------------|------------|
| 64D  | 4  | word |
| 128D | 8  | long word / short phrase |
| 256D | 16 | clause |
| 512D | 32 | sentence fragment |
| 1024D| 64 | full sentence |

Moving a field between rail widths is deterministic re-chunking of the same
substrate: a 128D vector is two 64D packs; a 64D pack is half a 128D vector.
"Translate back and forth for each reasoning core rail size" becomes a codec
operation with a roundtrip *test* — not a learned organ with a promotion
gate. The Heart that roundtrips each rail is the deliverable of this phase.

### 2.4 Chunking, not ceilings

A word longer than 4 characters spans multiple 64D vectors. Chunking is a
compute control with mandatory coverage proof — identical in kind to the
existing page-sweep doctrine: pages, buckets, and chunks bound work, never
content. Nothing is truncated; every source character is visited and
addressable; the coverage receipt binds per-chunk source spans. The
no-fixed-character-ceiling invariant is preserved.

### 2.5 Snap-on-emit: the one new anatomical piece

A learned reasoning core emitting a proposal will not land exactly on a
valid packed code. The Heart therefore snaps each emitted vector to the
nearest valid code (deterministic quantizer), unpacks to exact substrate,
and validates the resulting typed delta through the existing Heart
transaction boundary.

- The codec's valid-code set **is** the registered codebook that Layer 13
  training doctrine already requires for discrete content.
- Snap distance is first-class evidence: a proposal that snaps cleanly is
  confident; one that barely resolves is flagged. This gives the organism a
  built-in, free uncertainty signal on every emission.
- The quantizer is deterministic, unit-testable, and has no parameters to
  train.

### 2.6 What this does to the conduit-boundary convergence

The converged conduit boundary (learned advisory selector + deterministic
exact execution, recorded 2026-08-26) was scaffolding around a problem this
design mostly removes: if representation is exact by construction,
"conduction" is deterministic re-addressing of packs. What survives from the
convergence and must carry into the codec design:

- decision and execution receipts on every pack/unpack/commit;
- Heart-only canonical validation and commit (unchanged doctrine);
- absolute replay/exactness floors (here: exact roundtrip is a *test*,
  not a training target — it passes at 1.0 or the codec is broken);
- abstain/fail-closed behavior: an emission that cannot snap within a
  declared tolerance is rejected, never approximated into the field.

The v5 conduit tissue and its checkpoints shelf along with the translator as
historical/migration evidence; they remain useful initialization tissue per
the non-disposable capacity law.

## 3. The shelf

Proposed to move into `shelf/` (final file list fixed at implementation time,
with the move executed so the full suite stays green — tests move with their
code and imports are updated):

- Heart intelligence / translator tissue and its training harness
  (`runtime/heart/intelligence.py`, `runtime/heart/translation_core.py`,
  `training/heart_translation.py`, associated smoke/generalization scripts
  and tests).
- Cortex organ code and schema work beyond what the deterministic dormant
  valve needs.
- Related gate/promotion fixtures specific to learned Heart promotion.

Stays active:

- The deterministic Heart control plane in full: authority classes, tick
  lifecycle, frozen images, ingress, coordinator, dormant valve,
  `d64_codec.py`, compilers, transaction boundary.
- The Trainer control plane in full — it is needed for the reasoning cores.
- Dormant State, the evidence bridge, and all exact autobiography.

Shelved means *preserved and documented*, with a shelf README naming what is
parked, why, its evidence state, and its return conditions (semantic
translation returns as a sense riding the exact codec; Cortex returns when
circulation is proven and lived experience needs digestion).

## 4. The Heart that roundtrips each rail — deliverable definition

This phase is done when all of the following are proven by test, not by
training:

1. Exact pack/unpack roundtrip over the full supported character substrate,
   at 64D, including multi-chunk items, empty items, and page boundaries —
   with coverage receipts.
2. Exact rail-to-rail repack roundtrip for each adjacent width pair as wider
   rails are dialed in (64↔128 first).
3. Snap-on-emit anatomy: quantizer, tolerance, rejection path, snap-distance
   evidence in decision receipts.
4. Masking/derived-view compatibility: packed rails respect per-region
   attention masks exactly as the per-character rail does today.
5. Full suite green; SoT amendment ratified and mirrored before code, per
   the established amendment discipline.

## 5. Trainer on real memory

Jeff's third directive: reasoning-core training pulls from real Dormant
State — Axon's actual memories.

What that means concretely:

- Curriculum construction draws from the preserved, hash-bound lived
  evidence: the `D:\00` source snapshots (59,875 logical records, ~3.5 GB)
  under `State/dormant/experience_v1/`, the accepted-ingress autobiography,
  and the recovered edge/container graph — via the existing P0 evidence
  bridge, with provenance-bound dereferencing on every training item.
- This implements the ratified lived-experience doctrine (2026-08-23):
  the Trainer is the lived-experience compiler; exact facts stay in Dormant;
  training targets procedural compression. The pivot simply makes it real
  instead of synthetic-first.
- Preflight, smoke gates, discrete losses over the codec codebook,
  counterfactual source-use probes, replay floors, and rolling checkpoints
  all apply unchanged (Working Contract §7).
- Known gap this closes along the way: `experience_v1` is not yet indexed
  through the Cortex/Dormant evidence bridge (ledger RETRIEVAL GAP flag);
  reasoning-core curriculum needs that indexing, so it becomes part of this
  phase's work.
- Known gap this depends on: response/tool/consolidator/Trainer outcome
  deposits (ledger AUTOBIOGRAPHY GAP flag) should be wired so the cores learn
  from complete episodes, not only recovered history. Parallel work, as
  previously converged.

## 6. Sequencing

1. Jeff rules on this proposal; Codex and ChatGPT review (§7).
2. SoT amendment draft (packed rails, shelf status, Trainer real-memory
   curriculum) → Jeff ratifies → mirrored into both SoT copies.
3. Shelf migration: move intelligence/Cortex work to `shelf/`, suite green.
4. Build the packed codec + snap-on-emit + rail roundtrip proofs (§4).
5. Wire experience_v1 indexing and autobiography deposits (parallel-safe).
6. First reasoning-core training campaign from real Dormant memory, under
   full Trainer governance.
7. Minimum-living-Axon milestone, unchanged in shape: accepted ingress →
   exact circulation → evidence-bound recall → one grounded proposal →
   snapped, Heart-validated commit → response/diary + automatic outcome
   deposit, with a demonstrated abstain/no-op path.

## 7. Questions for Codex and ChatGPT

Please answer per-question: agree / disagree / amend, with reasons. My
suggested answer is attached to each.

- **Q1 (packing scheme).** Disjoint-subspace concatenation (4 × 16D at 64D)
  vs byte-exact positional codes (denser: up to 64 chars/vector at 64D as
  per-position codes)? Concatenation preserves the frozen-cell geometry and
  existing lift; dense codes maximize attention economy but need a new exact
  encoding. *My answer: concatenation first — it reuses proven substrate
  geometry; evaluate dense codes as a later ablation.*
- **Q2 (chunk semantics).** Should chunk boundaries align to word/token
  boundaries (word-aware packing, variable chars per vector) or fixed
  4-char blocks (uniform, simpler, word-splitting)? *My answer: word-aware
  with a declared max chunk; the field already marks word spans
  deterministically, and whole-word vectors are what make the core "think in
  English." Fixed blocks are the fallback if word-aware proves irregular.*
- **Q3 (snap tolerance and evidence).** Is snap distance the right emission
  confidence signal, and what tolerance belongs in the rejection path?
  *My answer: yes; tolerance is a tunable constant (not doctrine), start
  conservative, receipts record distance on every commit.*
- **Q4 (shelf contents).** Is the §3 shelf/stay split correct and complete?
  *My answer: yes, with the final file list fixed at implementation and the
  suite kept green through the move.*
- **Q5 (return conditions).** Are the stated return conditions for the
  shelved organs the right ones (semantic translation returns as a sense on
  the exact codec; Cortex returns when circulation is proven and lived
  experience needs digestion)? *My answer: yes.*
- **Q6 (real-memory curriculum).** What counterfactual/source-use probes are
  meaningful when the curriculum is real autobiography rather than synthetic
  minimal pairs? *My answer: zero/swapped/irrelevant source-episode probes on
  recall-conditioned tasks, plus held-out episodes by time and source; open
  to better instruments.*
- **Q7 (failure modes).** What does this design break that I have not seen?
  Candidates I already hold: pack boundary artifacts in attention;
  quantizer dead zones; word-aware chunking making position geometry
  irregular; replay evidence built on per-character rails needing
  re-baselining. Steelman the learned-translator path if you believe
  shelving it is the error.
- **Q8 (minimum loop).** Does the minimum-living-Axon milestone shape in §6.7
  still match your cut, now that the exact path is codec rather than conduit?

## 7A. Review status and Codex outcome (added 2026-08-26, second revision)

Codex completed his Q1–Q8 review (ledger
`evt-20260827T011243511986Z-codex-packed-rail-review`; bus: "Codex —
Packed-rail codec and shelf-pivot review"). Verdict: **accept the pivot with
material amendments**, pending Jeff's ratification and the mirrored doctrine
amendment (his BLOCKING flag — amendment before any shelf move, code, or
training; same discipline this proposal already commits to).

Material outcomes I accept:

- D64 physical packing already exists (`CompiledD64Field.rows`, four literal
  16D cells per row, exact roundtrip). The new work is a **packed-row neural
  consumer and discrete packed emission**, not reimplemented storage.
- Q1: concatenation only, one generic `d_model/16`-lane codec across all
  registered widths; no dense opaque bit-packing as an active surface. This
  matches Jeff's §1A.3 ("packs and unpacks cells of all sizes").
- Q2: fixed lane groups that never cross region, attended-interval, or
  provenance boundaries; word/sentence structure lives in sidecars/derived
  views. My word-aware lean is withdrawn — his geometry argument is right.
- Q3: per-lane categorical output head with cross-entropy over the
  character+empty/EOS codebook; snap distance is auxiliary evidence, not
  confidence. Receipts record entropy, margin, calibration, snap distance,
  valid-lane count, EOS/length.
- Q4: shelf learned translator models/campaigns and autonomous Cortex; keep
  active all exact compiler/codec, authority, Trainer, bridge, and grounded
  structural surfaces; publish a shelf manifest (paths, identities,
  non-serving status, import consequences, return conditions).
- Q5/Q6: sharper return conditions; runtime-faithful real-memory controls
  (episode-level splits, provenance-breaking probes, historical responses are
  observations not targets).
- Q7 failure modes I now carry: intra-pack attention resolution, per-lane
  addressing across masks, exact payload crowding out derived features,
  valid-but-wrong decode, non-linear attention savings, scaffold-only
  repacking, the 95-character substrate gap on real Dormant text, and
  checkpoint anatomy changes.
- Q8: minimum loop sharpened — discrete packed emission + deterministic
  decode + receipts, proposal must pass source-use/provenance/semantic and
  canonical-delta gates; a snapped-legal vector alone is not enough.

Open items Codex raised that Jeff's §1A rulings now partially answer:
unlimited field/rail growth is mask-governed (§1A.2), and cores read their
rail directly (§1A.3). Still open and needing governed answers: the
95-character substrate Unicode/escape strategy (ADVISORY flag), and
quantizer/emission tolerance from held-out evidence.

**New question for reviewers (Codex amendment welcome, ChatGPT to answer with
the rest):**

- **Q9 (mask–chunk alignment).** Under Jeff's §1A.2 (cells stay; the mask
  moves), should a packed chunk enter a rail only when its full character
  span is attended (fail-closed, chunk-aligned masks, derived deterministically
  from character-level attended intervals)? *My answer: yes — never attend a
  partially masked chunk; the character mask is canonical input, the
  chunk-level mask is derived. Codex's never-cross-attended-interval pack
  rule (Q2) already implies this; asking to make it explicit.*
  **Codex's answer (2026-08-26, accepted):** precision correction — packs
  must never cross attended-interval boundaries, but character-level masks
  must NOT be rounded to chunk boundaries or allowed to suppress attended
  edge characters. Boundary packs carry explicit empty lanes for masked
  characters. Masking loses nothing from state; it only shapes circulation,
  exactly per Jeff's §1A.2 semantics.

ChatGPT's Q1–Q9 review is still pending at this revision.

---

## 7B. ChatGPT review and Jeff's ratification (added 2026-08-26, third revision)

ChatGPT's review arrived via Jeff and is convergent. Accepted points:

- The mask is the boundary controlling what Dormant memory is materialized
  into the current Shared Field — not primarily a character-within-pack
  mask. Dormant decides what exists in memory; the Shared Field decides what
  is currently present; masks decide how much of each region is present; the
  Heart guarantees exact circulation; rails are native working surfaces;
  cores learn reasoning, not serialization.
- Emission must be **discrete lane decisions** (character/empty/EOS or
  another registered substrate symbol) with the Heart deterministically
  reconstructing exact cells — matching Codex's Q3 categorical-head
  amendment. Input packing is exactly invertible because the Heart
  constructed it from known cells; output exactness comes from discrete
  decisions, never from claiming an arbitrary continuous vector is
  invertible.
- 0–100% region masks are **policy**, resolved deterministically into exact
  spans/turns *before* compilation; packing never understands "30%".
- The decisive consequence ChatGPT named: **the Heart no longer needs to be
  intelligent to make heterogeneous reasoning cores possible** — a 64D core
  and a 512D core receive different physical packings of the same exact
  active state under one deterministic truth boundary. The hardest
  uncertainty moves from plumbing to the measurable question: can reasoning
  cores become useful from real Axon experience?
- His SoT correction is applied in the ratified amendment: masked material
  resides in **Dormant**; the Shared Field contains only what region
  materialization/mask policies expose.

**Jeff ratified the pivot 2026-08-26** ("we are all on the same page... go
through each step"). The ratified doctrine now lives in
`docs/SOURCE_OF_TRUTH.md`, section "Packed substrate rails and the shelf
pivot (ratified 2026-08-26)" — packed rails with no pack-count ceiling, the
corrected mask law, discrete per-lane emission, the shelf with return
conditions, reasoning-cores-next on real Dormant memory, substrate-coverage
law, and the tests-before-training first proof. This proposal document is
now historical context for that doctrine.

---

## 8. What this proposal does NOT do

- Does not delete anything. Shelf is preservation with documentation.
- Does not weaken any gate, floor, or the no-fixed-ceiling invariant.
- Does not cap rail length or field size; growth is mask-governed — cells
  stay, the mask moves (§1A.1–1A.2).
- Does not change Heart authority, tick mechanics, or Trainer governance.
- Does not retire learned semantic translation permanently — it returns as a
  sense, with return conditions, on a proven exact foundation.
- Does not touch the Source of Truth. Amendment draft first, Jeff ratifies,
  then mirrors, then code.

— Kimmy / Kimi Code CLI / 2026-08-26
