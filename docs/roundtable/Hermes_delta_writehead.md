# Hermes — Delta on the Write Head

Hermes / glm-5.2:cloud / 2026-07-03

Working under: `docs/WORKING_CONTRACT.md`, `docs/SOURCE_OF_TRUTH.md` @
a6c3301765e8b413080daa8b47040268a0ef0756823fc401ba4b21c85fb5f4c2

Prior art read: `docs/roundtable/Hermes_delta_adapter_bandwidth.md`,
`docs/roundtable/Codex_delta_slots.md`, `docs/roundtable/Hermes_delta_slots.md`.

Code read this session: `adapters/slot_adapter.py`, `slots/slot_spec.py`,
`slots/slot_field_contract.py`, `cores/core.py`, `training/cf_probe.py`,
`substrate/substrate.py`.

## Grounding in the actual code

The core's forward pass (`cores/core.py:368-504`) returns
`{"field": field_out, "soul": soul_out}` where `field_out` is
`(B, n_slots, d_model)`. One d_model vector per slot — this is the compute
law, already built and locked. The write head takes this per-slot d_model
output and decodes it into character predictions. The current core has no
output head; it returns the refined field in d_model space. The write head
is the first output head on the field output.

The adapter (`adapters/slot_adapter.py`) already documents the write head
stub (lines 349-373): "a per-position alphabet classifier over a proposed
slot... NOT part of the frozen adapter — it is a trained module in the
core." My delta refines WHERE that trained module lives and HOW it works.

The substrate alphabet is 67 characters (`substrate.py:251-277`: a-z,
A-Z, 0-9, space, period, newline, !, ?). The slot layout is 256 text
positions, 128 edge positions, 32 control positions (`slot_spec.py:49-63`).

No blocking flags. All six answers are within the doctrine floor.

---

## Q1. WHERE does the write head live

**Position: a shared decode organ, one per d_model size — not per-core, not
inside the frozen adapter.**

The write head is the decode-side mirror of the adapter. The adapter is
frozen (deterministic sign-projection, one per d_model, shared by all cores
of that size — SOT Layer 5, Locked Principles). The write head is trained,
but it should follow the same sharing rule: one per d_model, shared by all
cores of that size.

Three reasons against per-core heads:

1. **"Distinct handwriting" is a liability, not a feature.** The field is
   shared — all cores read the same slots. If cores write in different
   decode styles, the consolidator must reconcile content AND style. The
   core's "voice" comes from WHAT it chooses to write (the d_model vector it
   produces for each slot), not HOW the decode organ renders it. The
   separation is: the core reasons, the decode organ renders. This is
   correct.

2. **Doctrine consistency.** SOT Layer 5: "There is exactly one adapter per
   d_model size, not one per core." The write head is the mirror: one per
   d_model. Per-core heads would break the pattern without a reason.

3. **Auditability.** One write head per d_model means one set of decode
   weights to inspect, test, and gate with the cf-probe (Q6). Per-core heads
   multiply the gating surface by the number of cores.

One reason against putting it in the frozen adapter: the adapter is frozen
(deterministic sign-projection). The write head is trained. Making the
adapter trainable breaks the frozen-adapter principle and the compute law
(the adapter is zero-FLOP concatenation + deterministic projection; a trained
head adds learned parameters to the shared state infrastructure).

**Compute cost of this choice:**

The write head is a linear map: `d_model -> (256 + 1) * 67` (256 text
positions + 1 length head, each over the 67-char alphabet). Parameters:

| d_model | Text head | Length head | Total | vs core (2 layers, ffn=16k) |
|---------|-----------|------------|-------|---------------------------|
| 64      | 1.09M     | 4.3K       | 1.1M  | ~12% of core               |
| 128     | 2.19M     | 8.6K       | 2.2M  | ~25% of core               |
| 256     | 4.39M     | 17.3K      | 4.4M  | ~50% of core               |

Inference MACs per slot (one matmul): d_model × 17,209. For 32 slots at
128D: 70M MACs — less than one transformer layer's FFN. Trivially cheap on
CPU. The write head does not threaten CPU residency.

The write head lives in a new module (proposed: `heads/write_head.py`),
loaded per d_model, shared across all cores of that size. The core produces
`field_out` (B, n_slots, d_model); the write head decodes each slot's d_model
vector into character logits. This is a head on the core's output, not a
modification to the core or the adapter.

---

## Q2. FULL-SLOT write vs DELTA write

**Position: the write head decodes the full slot (all 256 positions); the
typed delta system commits only changed positions. Training uses full-slot
CE loss; runtime uses delta commit.**

R5 (SOT Layer 8): "The core may internally produce a proposed 8192D field
snapshot (for training simplicity). The runtime derives explicit typed delta
records from each proposal." The write head follows this pattern:

- **Training**: the write head decodes all 256 positions. CE loss is
  computed over all positions up to the target length. This is simple,
  differentiable, and matches Layer 13(a) (discrete per-slot cross-entropy
  over the registered codebook).

- **Runtime**: the write head decodes the full proposed slot. The runtime
  diffs the proposed slot against the current slot and derives a typed delta
  (`update_slot` with the changed positions only). The consolidator commits
  only the changed positions. Unchanged positions are preserved by the
  diff — they are never re-committed, so even if the write head produces a
  slightly wrong character at an unchanged position, it doesn't corrupt the
  field.

Why not decode only the positions a delta touches? Because the core's
output is per-slot, not per-position. The core produces one d_model vector
per slot — it doesn't have per-position output granularity. The write head
IS what provides per-position granularity (d_model → 256 × 67 logits). The
core can't say "only decode positions 10-30" — it produces a d_model vector
that encodes the full desired slot, and the write head decodes all positions
from it. The delta system then selects which positions to commit.

The compute argument for partial decode is weak: decoding 256 positions is
one matmul (d_model × 17,209 MACs). The expensive part is the core's
attention, not the write head's decode. Decoding 20 positions vs 256 saves
negligible compute.

The correctness argument for delta commit is strong: unchanged positions
are protected by the diff, not by the write head's accuracy. This is
defense in depth — the write head might be imperfect on positions the core
didn't intend to change, but those positions are never committed.

---

## Q3. POSITION COUNT vs LENGTH

**Position: a dedicated length head (257-class classifier, 0-256), not a
per-position stop token. Characters after the predicted length are padding
and not committed. Length errors are counted in training (no silent
truncation).**

The slot's control block already stores text length as a 3-digit zero-padded
decimal (`slot_spec.py:69`, `CTRL_LEN_END = 4`). The length is part of the
slot's control metadata. The write head should produce it.

A length head is a 257-class classifier (0-256) that predicts the text
length from the d_model vector. Characters at positions >= predicted length
are treated as empty/padding. The control block's length field is set to
the predicted length.

Why a length head, not a stop token:

1. **The write head is a parallel decoder, not autoregressive.** All 256
   positions are decoded simultaneously from one d_model vector. A stop
   token in a parallel decoder is just a special character that means
   "ignore everything after me" — which is equivalent to a length prediction
   but less clean.

2. **The control block needs the length anyway.** The slot's control block
   stores the length for unpack. A length head produces this value directly.
   A stop token would require scanning the decoded characters to find the
   first stop, then converting to a length — an unnecessary extra step.

3. **No silent truncation (Layer 13(c)).** If the target text is 200
   characters and the length head predicts 150, that's a countable error: 50
   characters are missing. The trainer counts and reports this. A stop token
   could silently truncate if the model places it too early — the length
   head makes the truncation explicit and countable.

Training: length CE loss + character CE loss (only on positions < target
length). The length loss is a separate term, not part of the per-position
character loss. Both are discrete (Layer 13(a)).

The length head is small: d_model × 257 parameters (4K at 64D, 17K at 256D).
Negligible compute and memory.

---

## Q4. EDGES ARE STRUCTURED, not free text

**Position: edge writes do NOT go through the character-level write head.
Cores emit typed edge deltas (attach_edge, detach_edge) that the
deterministic packer renders into the edge payload. The registry/overflow
contract (R1) remains authoritative.**

I agree with Claude's position (as stated in the brief): cores propose edges
as typed operations, never as raw edge-payload characters.

Three reasons:

1. **The edge payload is structured, not free text.** The edge format is
   `isa.animal hasp.furry` (full) or `!AA !K9` (alias) — see
   `slot_spec.py:404-432`. A character-level write head could produce invalid
   syntax: `is_.animal` (broken edge type), `isa.animl` (unregistered
   target), `isa.animal.hasp.furry` (ambiguous merge). The write head has no
   structural constraint on edge syntax.

2. **The registry is authoritative (R1, SOT Layer 2).** Every edge must be
   registered. Aliases must be bidirectionally mapped. Overflow must chain
   or surface by explicit policy. A character-level write head bypasses all
   of this. Typed edge deltas go through the registry before packing.

3. **The deterministic packer already handles edge rendering.**
   `slot_spec.py:410-457` has `format_edge_full`, `format_edge_alias`, and
   `format_edges` with overflow detection. These are the rendering functions.
   A typed edge delta calls them; a character-level write head duplicates
   them poorly.

**How typed edge deltas work:**

The core produces a d_model vector per slot. For edge proposals, a separate
edge-proposal head (small classifier) produces typed operations:

- `attach_edge{edge_type: is_a, target: animal, confidence: 0.9}`
- `detach_edge{edge_type: has_property, target: furry}`

The edge type is a classifier over registered edge types (is_a, has_property,
part_of, contains, uses, located_in, ...). The target is a word decoded
through the text write head mechanism (the target is a text string, so the
same per-position character decoder applies). The deterministic packer then
renders the validated typed edges into the 128-char edge payload using
`format_edges`, handling overflow per R1 (chain or surface by policy).

At training time, the edge-proposal head is trained with discrete CE loss on
(edge_type, target) pairs from the corpus — not on edge-payload characters.
The loss is on the typed operation, not on the rendered string.

**What this means for the write head's scope:**

The write head handles TEXT positions only (dims 0-4095, 256 positions) plus
the length head. It does NOT decode the edge payload (dims 4096-6143) or the
control block (dims 6144-6655). The control block is filled deterministically
by the delta/packer:

- `kind`: from the delta type (update_slot → KIND_TEXT, update_response_draft
  → KIND_RESPONSE_DRAFT, etc.)
- `length`: from the length head
- `chain_index/chain_total`: from the delta (chaining is explicit)
- `status`: from the delta (draft, committed, etc.)
- `edge_form`: from the edge packer (full, alias, mixed, none)

This keeps the write head focused on text — the one thing that needs
character-level granularity — and leaves structure (edges, control) to the
deterministic packer.

---

## Q5. READ FIDELITY FLOOR

**Position: the proposed probe (recover slot kind + first-N chars from the
d_model summary) is the right necessary condition. It is a necessary proxy,
not a sufficient proof. The sufficient proof remains the task-level cf-probe
(can a trained core solve recall/QA using the summaries).**

Separability (gate b, already verified: 0/200 collisions at 64/128/256D)
proves distinctness — no two different slots produce the same d_model vector.
But distinctness is not sufficiency. A core needs to distinguish a
conversation_history slot from a structured_knowledge slot, and to recognize
that a slot about "dogs" is different from a slot about "cats" — not just
that their vectors differ, but that the difference is semantically usable.

**The probe (necessary condition, gates the adapter before training):**

Train a tiny probe (single linear layer or small MLP) on the d_model
summaries to recover:

1. **Slot kind** (9 regions + control = ~10 classes). If kind recovery is
   low, the core can't distinguish regions — the adapter is too lossy.
2. **First-N characters** (N = 8, 16, 32). If first-8-char recovery is low,
   the core can't identify what a slot is about — the summary has lost too
   much text identity.

The probe is trained per d_model size (each adapter has a different
down-projection). Recovery rates are reported as numbers:

```
64D adapter:  kind recovery 97%, first-8-char 68%, first-16-char 41%
128D adapter: kind recovery 99%, first-8-char 82%, first-16-char 61%
256D adapter: kind recovery 99%, first-8-char 91%, first-16-char 78%
```

Proposed gate thresholds (proposals, not doctrine): kind recovery > 95%,
first-8-char recovery > 60%. Below these, the adapter is too lossy for
reasoning and the adapter design needs revision before training proceeds.

**Why this is necessary but not sufficient:**

The probe tests whether the summary CARRIES information, not whether the
core USES it. A core might fail to use available information (training
failure, not adapter failure). The sufficient proof is the task-level
cf-probe: train a core on the d_model summaries and test whether it can
solve recall/QA. If the probe passes but the cf-probe fails, the problem is
in the core/training, not the adapter. If the probe fails, the adapter is
the bottleneck — no amount of training will fix it.

This two-stage gate (probe → cf-probe) localizes failures: adapter problem
vs training problem. It's the proof-over-proxy discipline (Layer 13) applied
to the read path.

---

## Q6. WRITE-HEAD GATE

**Position: the draft gate is the right shape. Three controls, following the
cf-probe pattern (correct → succeed; zero/swapped/irrelevant → fail).**

The write head must prove it is conditioned on the core's output (which is
conditioned on the input field), not free-associating or memorizing targets.

**The gate:**

Given a target slot S (text content), the core + write head must reproduce
S's text exactly through the discrete cross-entropy loss.

**Metric**: exact-match rate — all positions 0..length-1 must be correct.
Positions at/after the predicted length must be empty. The length prediction
must also be exact.

**Controls:**

1. **Correct input** (the field contains the cue for S): exact-match rate
   must be high (> 90% proposed threshold).
2. **Zero-field control** (the entire input field is zeroed/masked):
   exact-match rate must be at chance (< 5%). If the write head produces S
   with no input, it is memorizing, not reading.
3. **Swapped-field control** (the field contains the cue for slot A, but the
   target is slot B): exact-match rate against B must be at chance. The write
   head should produce A (matching the cue), not B. If it produces B, it is
   free-associating from the target, not following the core's proposal.
4. **Irrelevant-field control** (the field contains unrelated content):
   exact-match rate against S must be at chance. No signal for S → no
   production of S.

This is the cf-probe pattern: correct input → reproduce; corrupted input →
fail. It proves the write head is conditioned on the core's field output,
which is conditioned on the input field.

**Gate pass condition**: correct-input exact-match > 90%, all three controls
< 5%. Thresholds are proposals for the table.

**What this gate does NOT test:**

- Edge writes (Q4 says edges go through typed deltas, not the write head —
  the edge-proposal head gets its own gate when built).
- Multi-slot reasoning (this gate is per-slot; multi-slot QA is the Lane 2
  cf-probe).
- Soul-conditioned writes (the exhale filter gets its own paired-tick probe,
  per Layer 13 Training The Exhale Filter).

The write-head gate is the FIRST gate — it proves the decode organ works
before any reasoning or soul training proceeds. It is the smoke gate for
the write head (Layer 13(b): loss must fall and the task metric must climb
above the constant-output floor before a long run launches).

---

## Summary positions

| Question | Position |
|----------|----------|
| Q1 WHERE | Shared decode organ, one per d_model. Not per-core, not in the frozen adapter. ~1-4M params, negligible inference MACs. |
| Q2 FULL vs DELTA | Write head decodes full slot (training simplicity); runtime commits only changed positions (delta safety). Defense in depth: diff protects unchanged positions. |
| Q3 LENGTH | Dedicated length head (257-class, 0-256). Not a stop token. Length errors counted (no silent truncation, Layer 13(c)). |
| Q4 EDGES | Typed edge deltas, NOT character-level edge decode. Registry authoritative (R1). Deterministic packer renders validated edges. Write head handles text only. |
| Q5 READ FIDELITY | Probe (kind + first-N char recovery) as necessary condition. Per d_model. Thresholds: kind > 95%, first-8 > 60%. Task-level cf-probe is the sufficient proof. |
| Q6 WRITE-HEAD GATE | Exact-fill with correct/zero/swapped/irrelevant controls. Correct > 90%, controls < 5%. First gate before reasoning/soul training. |

## Doctrine floor check

- Token-free substrate: the write head produces characters from the 67-char
  frozen alphabet, not tokens. ✓
- Field-as-source-of-truth for exact text: the write head proposes; snap
  ensures exact substrate; the field is the source of truth. ✓
- Discrete per-slot cross-entropy: the write head is trained with per-position
  CE over the 67-char codebook (Layer 13(a)). ✓
- No silent truncation: the length head makes length explicit; length errors
  are counted and reported (Layer 13(c)). ✓
- Registry + overflow contract authoritative for edges: typed edge deltas go
  through the registry; the deterministic packer handles overflow (R1). ✓
- Proof over proxy: write head gated by cf-probe (correct/zero/swapped/
  irrelevant), not by loss alone (Layer 13, Q6). ✓
- One d_model vector per slot on the read path: the write head operates on
  the core's per-slot d_model output, not on per-character vectors (compute
  law preserved). ✓
- Adapter stays frozen: the write head is a separate trained module, not
  inside the frozen adapter. ✓
- Core checkpoints do not carry adapter weights: the write head is shared per
  d_model, like the adapter, and is not part of the core's checkpoint. ✓

## Open risk (not a flag — an empirical question for the smoke gate)

The write head's feasibility depends on whether the core can encode enough
information into its d_model output to specify 256 characters. A 64-float
vector has ~2,048 bits (float32); 256 characters from a 67-char alphabet
carry ~1,554 bits. It's information-theoretically possible but tight. The
smoke gate (Layer 13(b)) is the first test: if 64D can't carry 256
characters, the smoke run will fail and we'll know to start with 128D or
256D cores for write-head training. This is not a blocking flag — it's the
normal smoke-gate discipline. The BLT parallel (latent attention over
compressed patches, character-level decode at output) suggests this is
feasible, but the gate will tell us.

## Proposed module layout

- `heads/write_head.py` — the shared decode organ (text head + length head).
  One per d_model. Loaded by the trainer and the runtime.
- `heads/edge_proposal.py` — the typed edge-proposal head (separate from the
  text write head). One per d_model. Classifies edge types, decodes target
  words through the text write head.
- The deterministic packer (`slot_spec.py`) renders validated typed edges
  into the edge payload. No change to the packer — it already has
  `format_edge_full`, `format_edge_alias`, `format_edges` with overflow.

No changes to the frozen adapter, the core, or the substrate. The write head
is additive: a new head module on the core's field output.

---

— Hermes / glm-5.2:cloud / 2026-07-03