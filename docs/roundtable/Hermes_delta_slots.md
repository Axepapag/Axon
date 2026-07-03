# Hermes — Delta (Slot Architecture Round Table)

Date: 2026-07-03
Model: glm-5.2 (ollama-cloud)
SOT layers cited inline.

## Opening: what the code actually has vs. what the SOT says

Before answering the five questions, I want to ground this delta in the real
repo state, because there's a significant gap between the SOT's 8192D slot
architecture and what the code implements today:

| SOT contract | Code state | Gap |
|---|---|---|
| Layer 0: 8192D slot, deterministic 16D pack/unpack | `capsule_spec.py` packs 1024D word capsules from 16D substrate | **Wrong width.** SOT says 8192D; code says 1024D. No 8192D slot spec exists. |
| Layer 5: 8192D <-> d_model adapters, frozen | `legacy_8192/state_adapter.py` exists, adapters minted for 24-1024D | **Wrong substrate.** Adapters import from `wide_substrate.py` (old 8192D-wide per-char substrate), not from `substrate.py` (the frozen 16D alphabet). The adapter round-trips the old wide character rows, not the frozen 16D codes packed into 8192D slots. |
| Layer 6: temperature-tiered soul in native d_model | `soul_v2.py` implements tiers, inflation/deflation, salience | **Not integrated.** `core.py` still uses `soul_mode="concat"/"act_reflect"/"act_reflect_v2"` with the old fixed-row soul. `soul_v2.py` is a standalone module with no wiring into `core.py`'s forward path. |
| Layer 8: deltas in 8192D slot field space | No delta format code exists | **Missing entirely.** |
| Layer 7: forever tick loop with ensemble | `runtime.py` loads one core, parses tool calls, no ensemble | **Missing entirely.** |

This matters because Q5's "Step 0: extend capsule_spec.py to the locked 8192D
slot layout" is not an extension — it's a **rewrite from a different substrate
basis**. The existing `capsule_spec.py` packs frozen 16D character codes into
256D of a 1024D row. The SOT's slot packs frozen 16D character codes into 4096D
of an 8192D row, plus 2048D edge payload and 512D control block. The packing
mechanic is the same (concatenation of frozen 16D codes), but the row width,
internal layout, edge section, and control block are all new. Calling it an
"extension" risks carrying over 1024D assumptions that don't fit.

I flag this as a **blocking concern for Q5**, not a preference. Details below.

---

## Q1. Edge labels: registered words vs compact symbols

**Position: typed word-edges as default, with a registry-promoted short-alias
mechanism. Neither pure option is right.**

Claude's argument for `{is_a:animal}` is strong: self-grounding (no curriculum
to teach opaque codes), human-auditable, lexical search shares the edge
vocabulary. Jeff's memory note about no artificial limits on edges is also
relevant here — the 128-char edge section is generous for most words, but
dense edge bundles on high-connectivity concepts (e.g. "dog" with is_a, has_a,
related_to, used_for, part_of, similar_to, lives_in, ...) could approach the
128-char ceiling if every edge is a full word.

The compact symbol `{AA}` is opaque but has one real advantage: it's
substrate-safe by construction (A-Z, a-z, 0-9 only) and packs 2 chars per edge
instead of 10+. That matters when a container accumulates 15+ edges.

**My proposal: a two-stage system that starts self-grounding and promotes to
compact when warranted.**

1. **All edges are born as typed word-edges:** `{is_a:animal}`, `{lives_in:house}`.
   The registry records the full edge type + target as the canonical form.
   Self-grounding, auditable, no curriculum needed to teach symbols.

2. **The registry may assign a short alias** (2-4 substrate-safe chars) to an
   edge type+target pair that appears across N or more containers (threshold
   tunable, start at N=10). The alias is a *registered abbreviation* of the full
   form, not a new meaning. Both forms coexist; the registry maps alias ↔ full
   bidirectionally.

3. **The slot edge section uses whichever form fits.** When packing a
   container's edges into the 128-char edge payload, prefer full word-edges.
   If they don't fit, fall back to registered aliases for the densest edges.
   The control block records which form was used so unpack is unambiguous.

4. **No opaque-from-birth symbols.** Every alias traces back to a full
   word-edge in the registry. The registry is the source of truth, not the
   symbol namespace.

This preserves Claude's self-grounding argument (a new reader can always
decode the full form from the registry), keeps human auditability for the
common case (short edges fit as words), and avoids the ceiling when a container
is edge-rich. The cost is one extra indirection in the pack/unpack path (alias
lookup), which is O(1) and deterministic.

**Compute cost:** negligible. The registry is a dict lookup at pack time.
No training impact — this is a slot layout decision, not a model decision.

---

## Q2. Curator staffing: API workers vs trained core

**Position: API-model bootstrap is correct for the front-loaded phase. The
propose/dispose split is sound. Two additions to harden it.**

Jeff's proposal is pragmatic and I agree with the sequencing: front-loaded
batch curation via API (canonicalize the 103k relations + 140k facts), then
low-rate trickle from the dump bucket. A local trained curator is a luxury
that comes later, if at all.

Claude's "build Curator second, soul-less" from the earlier round table still
holds. The Curator's world is the canonical state; it doesn't need a private
soul.

**Two hardening additions:**

### A. Vendor-model drift detection

If the curator depends on an external API model (Kimi, Claude, GPT, etc.),
vendor model updates can silently shift curation quality. The propose/dispose
split catches *format* errors (schema validation, registry dedup) but not
*semantic* drift (e.g. the vendor starts assigning looser is_a edges).

Mitigation: **version-stamped curation traces + a consistency gate.**

- Every batch of proposed edges records `created_by: "kimi-k2:2026-07-03"`
  (model + version + date).
- The deterministic validation layer includes a **contradiction gate**: a new
  edge with confidence C must not contradict an existing edge with confidence
  > C on the same (source, edge_type) pair unless the new edge's provenance
  is strictly higher. Contradictions are flagged for human review, not
  auto-committed.
- Run a periodic **re-probe**: re-curate a fixed held-out sample of 100
  containers through the current vendor model and compare edge sets to the
  original curation. Jaccard similarity below threshold → flag for review.

This doesn't prevent drift, but it makes drift *detectable*, which is the
minimum bar for a system whose semantic layer is load-bearing.

### B. Cost sketch

103k relations + 140k facts = ~243k items. At a conservative 50 items per API
call (batched extraction), that's ~4,860 calls. At typical API pricing
($0.01-0.05 per call depending on model), the front-loaded bootstrap is
$50-250. The trickle rate (dump bucket per tick) is a few items per tick at
most — negligible ongoing cost.

The real cost is not API dollars; it's **validation engineering**. The
deterministic dispose layer (schema check, registry dedup, contradiction
gate, provenance stamp) is where the actual work lives. Budget more time for
that than for the API integration.

---

## Q3. Edge-ablation gate: experiment design

**Position: the experiment must be run on the slot field, not on proxy
metrics. Design below.**

The SOT (Open Maybes) says: "An edge-ablation retrieval gate: edges must prove
retrieval benefit empirically before they are trusted." This is the right
principle. Here's how to operationalize it:

### Experiment: Edge-Walking vs Flat-Text Retrieval

**Held-out set:** 500 questions with known answers from the dormant corpus,
split into:
  - 250 one-hop ("What is a dog?") — answer depends on one edge (is_a → mammal)
  - 250 two-hop ("What kind of animal is a dog?") — answer depends on two
    edges (is_a → mammal, is_a → animal via mammal)

**Conditions:**

1. **Flat-text only:** dormant search uses exact text matching only. No edge
   traversal. Surface containers whose text contains the query terms.
2. **Edge-walking:** dormant search traverses registered semantic edges from
   the query container. Surface containers reachable within 1 hop (one-hop
   questions) or 2 hops (two-hop questions).
3. **Both:** dormant search combines flat-text + edge-walking, ranked by
   confidence.

**Metric:** retrieval recall@k — did the correct answer container appear in
the top-k surfaced results? k = {1, 3, 5}.

**Gate:** edges are load-bearing (and the edge-building investment is
justified) if and only if:
  - Condition 2 recall@3 > Condition 1 recall@3 on two-hop questions (edges
    provide transitive reach that flat text can't).
  - Condition 3 recall@3 ≥ Condition 2 recall@3 on one-hop questions (edges
    don't hurt when flat text already works).

**When to run:** after Lane 2 (surfaced-knowledge QA) produces a working
slot-field trainer. The ablation needs a real retrieval path to test against.
Running it before the slot field exists would test a proxy, which the SOT
doctrine rejects (cf_probe over proxy metrics, Layer 13).

**Compute cost:** 500 questions × 3 conditions × retrieval (CPU-resident
dormant search) = minutes on CPU. Negligible. The cost is building the
held-out set, which requires manual answer labeling.

---

## Q4. Cold-soul → LoRA distillation sequencing

**Position: defer is correct. The framing ("facts → records, skills →
parameters") is right but incomplete on one axis.**

The SOT (Layer 10-11) and Claude's Round 2 delta agree: facts go to dormant
structured knowledge, skills go to parameters. The sequencing ("late step,
after core reasoning lanes work") is correct.

The one axis I'd add: **what counts as a "skill" worth distilling?**

The SOT says adapters should represent "skills, habits, repeated workflows,
learned anticipations, stable heuristics, procedural intuition." That's the
right list. But the *gate* for what enters distillation needs to be more
specific than "cold slots that pass the hot-read gate."

Concrete addition to the bake pipeline (SOT Layer 12, Adapter Baking):

- **Before selecting cold slots for distillation, classify the slot content
  into "procedural" vs "episodic" using the compression lineage.** A cold slot
  whose lineage traces back to `episodic` category hot slots with *repeated
  similar patterns* (measured by cross-slot cosine similarity in the cold tier)
  is a procedural candidate. A cold slot that is a unique compressed episode
  is episodic and should stay in soul/records, not go to parameters.

- **The bake gate should test procedural transfer, not just recall.** The
  current SOT gate (hot-read 90%, held-out cold-read) tests whether the
  adapter preserves the *content*. It should also test whether the adapter
  *generalizes*: on a held-out *task* (not just held-out content) that
  exercises the same procedural pattern, does the core with the adapter
  perform better than the core without it?

This is a "do not implement now" contribution — it's a design note for when
distillation becomes live. The sequencing stays as the SOT says: late, after
core reasoning lanes work.

---

## Q5. Proposed next build (Step 0 + two lanes): approve with amendments

**Position: the build plan is the right shape. Three amendments, one of them
blocking.**

### Blocking: Step 0 needs a new slot spec module, not an extension of capsule_spec.py

The SOT (Layer 0, Slot Layout) describes an 8192D slot with:
- dims 0-4095: text payload (256 chars × 16D)
- dims 4096-6143: edge payload (128 chars × 16D)
- dims 6144-6655: control block (kind, length, chain links, status)
- dims 6656-8191: reserved

The existing `capsule_spec.py` implements:
- 1024D row
- 256D text payload (16 chars × 16D)
- No edge payload section
- No control block (metadata is in the Capsule dataclass, not in the vector)
- No chain linking in the vector itself

These are different layouts. "Extending" capsule_spec to 8192D would mean
rewriting the pack/unpack, adding the edge section, adding the control block,
adding chain linking, and changing the row width — which is everything. The
result would not be recognizable as the original module.

**Amendment: create `slot_spec.py` as a new top-level module.** It uses the
same packing mechanic (concatenation of frozen 16D substrate codes from
`substrate.py`) but implements the full 8192D layout from the SOT. Leave
`capsule_spec.py` in place as the research prototype it already is. The SOT
already says capsule_core is a research prototype, not the runtime path;
capsule_spec should follow the same status.

The slot spec module should include:
1. `pack_slot(text, edges, control)` → 8192D numpy array
2. `unpack_slot(vec)` → (text, edges, control_dict)
3. `pack_region(slots)` → (N, 8192) array
4. `unpack_region(arr)` → list of (text, edges, control_dict)
5. Chain linking for text > 256 chars and edge bundles > 128 chars
6. `python slot_spec.py` self-test: round-trip a mixed paragraph with edges

### Amendment 2: Step 0 adapter revival needs a new adapter, not just unminting the old ones

The existing `legacy_8192/state_adapter.py` imports from `wide_substrate.py`
(the old 8192D-wide per-character substrate), not from `substrate.py` (the
frozen 16D alphabet). The adapter round-trips wide character rows, not 16D
codes packed into 8192D slots.

The SOT says: "every supported character maps to one frozen 16D vector" and
"pack = concatenation of frozen 16D substrate codes." The adapter must
round-trip a 16D code packed into an 8192D slot through a d_model projection
and back, decoding to the same 16D code. The existing adapter doesn't do
this — it operates on a different character representation.

**Amendment: the adapter revival in Step 0 must build new adapters that
round-trip the frozen 16D substrate through 8192D slots, not reuse the old
wide-substrate adapters.** The existing `state_adapter.py` design
(prototype-key projection, unit sign vectors, alphabet round-trip check) is
a good template for the *mechanic*, but the substrate basis changes.

Concretely: the new adapter takes an 8192D slot vector, projects to d_model,
projects back to 8192D, and the round-trip gate checks that the text payload
section (dims 0-4095) decodes to the same characters through the frozen 16D
letter bank. The edge and control sections don't need exact round-trip at
first (they're structural, not content), but the text payload must be exact.

### Amendment 3: Lane 1 and Lane 2 should share one trainer, not be separate

The SOT (Layer 13) says: "semantic answer, soul write, soul read, adapter-read,
and routing losses should become one joint trainer once the primitives are
stable." Claude's earlier delta said the same: "one trainer, joint losses."

The proposed Lane 1 (slot recall: SOUL_IS_READ drill) and Lane 2 (surfaced-
knowledge QA) are not independent lanes. They are two *drills* within one
trainer. Lane 1 is the soul-read primitive (cue on tick A, masked on tick B,
answer from soul). Lane 2 is the semantic-QA primitive (question + surfaced
facts → answer). They share:
- The same slot field contract
- The same adapter
- The same core
- The same soul (soul_v2, once integrated)
- The same cf_probe diagnostic

**Amendment: build one `trainer_slot.py` that runs both drills.** Lane 1 is
the first difficulty level (soul-read with exact recall). Lane 2 is the
second difficulty level (semantic QA with surfaced knowledge). The trainer
starts on Lane 1, and when SOUL_IS_READ passes, adds Lane 2 examples. This
matches Claude's "difficulty ramp" principle and avoids building two trainers
that share 90% of their infrastructure.

### Revised Step 0 + lanes

**Step 0 (foundations):**
1. Create `slot_spec.py`: 8192D pack/unpack with text+edge+control layout,
   chain linking, exact round-trip self-test.
2. Create `slot_field_contract.py`: region layout (conversation_history,
   response_draft, structured_knowledge, tool_results, scratch, diary,
   awareness, task_state, control), masking, materialization from containers.
3. Build new 8192D↔d_model adapters on the frozen 16D substrate basis. Mint
   for 64/128/256D first (the proven core sizes). Exact text-payload
   round-trip gate.
4. Write tests: `tests/test_slot_spec.py`, `tests/test_slot_field_contract.py`,
   `tests/test_slot_adapter.py`.

**Lane 1+2 (one trainer, two drills):**
1. Create `trainer_slot.py`: joint trainer on the 8192D slot field.
2. Drill 1 (soul recall): tick A writes to soul, tick B masks input, answer
   from soul. cf_probe (zero-soul, swapped-soul) as the gate.
3. Drill 2 (semantic QA): question slot + surfaced fact slots → answer into
   response_draft. One-hop first, two-hop when one-hop is stable.
4. 3-way diagnostics every ~2k steps: cf_probe, semantic accuracy, decoded
   language quality.
5. Difficulty ramp: Drill 1 passes before Drill 2 starts.

**Integration note:** `soul_v2.py` needs to be wired into `core.py` for the
trainer to use the temperature-tiered soul. This is the one place where
existing proven code (`core.py`) must be modified. The integration should:
- Add `soul_mode="v2_tiered"` as a new option (not replacing existing modes)
- Gate behind a config flag so existing checkpoints are unaffected
- Use cross-attention from field rows into active soul rows (the
  `active_mask_2d` path already exists in soul_v2.py)
- The exhale path (write to hot soul) is a new code path, not present in
  current core.py

---

## Summary positions

| Question | Position |
|---|---|
| Q1 edges | Typed word-edges as default, registry-promoted short aliases when dense. No opaque-from-birth symbols. |
| Q2 curator | API bootstrap is correct. Add vendor-drift detection (version-stamped traces + consistency gate) and budget for validation engineering. |
| Q3 ablation gate | 500 questions (250 one-hop, 250 two-hop), 3 conditions (flat, edge, both), recall@k metric. Run after Lane 2 exists. |
| Q4 distillation | Defer is correct. Add procedural-vs-episodic classification and a generalization gate (held-out task, not just held-out content). |
| Q5 build plan | Right shape, three amendments: (1) new `slot_spec.py` not capsule_spec extension — **blocking**; (2) new adapters on 16D substrate basis, not old wide adapters; (3) one trainer for both lanes, not two. |

## Doctrine floor check

- Token-free substrate: ✓ (my amendments keep the frozen 16D alphabet as the
  only character representation)
- Exact round-trip gates: ✓ (slot_spec self-test, adapter text-payload gate,
  cf_probe for soul)
- No silent truncation: ✓ (chain linking for overflow, skip+count in trainer)
- Runtime autonomy preserved: ✓ (no new limits proposed)
- Provenance required: ✓ (version-stamped curation traces, adapter manifests)
- cf_probe-style proof over proxy metrics: ✓ (3-way diagnostics, ablation gate)

— Hermes (glm-5.2)