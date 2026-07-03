# RESOLUTION writehead-0703 — Adapter Read/Write Asymmetry + The Write Head

Round ID: `writehead-0703`  
Date: 2026-07-03  
Convener: Jeff  
Participants: Codex, Kimi, Hermes, Claude, ChatGPT  
Deltas received: Hermes, Kimi  
Synthesizer: table (Kimi-code-cli)  

Doctrine stamp: `docs/SOURCE_OF_TRUTH.md @ a6c3301765e8b413080daa8b47040268a0ef0756823fc401ba4b21c85fb5f4c2`

---

## 1. What this resolution locks

This resolution formalizes the asymmetric read/write redirect that replaced the impossible R7 round-trip gate, and it settles the write-head design questions raised by that redirect.

### 1.1 Reaffirmed doctrine floor (not up for review)

The following remain locked from prior resolutions and `SOURCE_OF_TRUTH.md`:

- Token-free substrate.
- Field-as-source-of-truth for exact text; English is decoded from the 8192D field, never from a core's compressed view.
- One `d_model` vector per slot on the read path (compute law: 32 slots = 32 attention tokens).
- Discrete per-slot cross-entropy for discrete content.
- No silent truncation: over-budget items are skipped/chained AND counted AND reported.
- Registry + overflow contract remain authoritative for edges.
- Proof over proxy: cf-probe-style counterfactuals are required evidence; loss alone proves nothing.
- The frozen adapter is shared per `d_model` and not owned by cores.

### 1.2 Verified gates that survive the redirect

The adapter gates verified by Hermes and Kimi continue to hold:

- **Snap-idempotence**: validly packed slots survive `DOWN → UP → snap` unchanged. Verified: 0/50 changed at 64D, 128D, and 256D.
- **Separability**: distinct slots (differing by one character) produce distinct `d_model` projections. Verified: 0/200 collisions at 64D, 128D, and 256D.

---

## 2. Decisions by question

### Q1. WHERE does the write head live?

**Decision: a shared decode organ, one per `d_model` size, loaded as a separate head module — not inside the frozen adapter, not per-core.**

Rationale:

- It mirrors the adapter sharing rule (`SOURCE_OF_TRUTH.md` Layer 5): one piece of shared decode infrastructure per `d_model`.
- It keeps the core focused on reasoning; the core's "voice" is expressed by the `d_model` vector it produces, not by a private decode style.
- It yields one auditable decode surface and one Q6 gate surface per `d_model`.
- It does not violate the frozen-adapter principle because the write head is a trained module that lives outside the adapter.
- Core checkpoints do not carry adapter weights; the write head follows the same rule and is not part of a core checkpoint.

**Parameter budget (Hermes estimate, linear text head + length head):**

| d_model | Text head params | Length head params | Total    | Notes                              |
|---------|------------------|--------------------|----------|------------------------------------|
| 64      | ~1.09M           | ~4.3K              | ~1.1M    | ~12% of a 2-layer 64D core         |
| 128     | ~2.19M           | ~8.6K              | ~2.2M    | ~25% of a 2-layer 128D core        |
| 256     | ~4.39M           | ~17.3K             | ~4.4M    | ~50% of a 2-layer 256D core        |

**Compute cost:** one matmul per slot: `d_model × 17,209` MACs (256 text positions + 1 length class × 67-char alphabet). For 32 slots at 128D: ~70M MACs — less than one transformer FFN layer. CPU residency is preserved.

**Dissent:** Kimi advocated a per-core head (~57k–360k params if implemented as a small MLP over 256 positions). The argument: per-core heads respect core identity/private souls, make provenance explicit, and avoid centralizing the core's mouth onto the field. This dissent is recorded in Section 4.

### Q2. FULL-SLOT write vs DELTA write

**Decision: train full-slot; commit only changed positions via typed deltas.**

- Training: the write head decodes all 256 text positions and is trained with full-slot discrete cross-entropy. This keeps batching, masking, and the loss simple.
- Runtime: the write head still decodes the full proposed slot, but the runtime diffs it against the current slot and derives a typed delta that commits only the changed positions.
- Unchanged positions are protected by the diff; a write-head error on an unchanged position cannot corrupt the field.
- A core revising three words does not force a 256-position snap commit; the consolidator commits the span of changed positions.

This matches R5 typed deltas and the deterministic diffing already performed by the consolidator.

### Q3. POSITION COUNT vs LENGTH

**Decision: a dedicated length head (257-class classifier, 0–256) predicts the text length; characters at or after the predicted length are padding and are not committed.**

Rationale:

- The slot's control block already stores text length (`slot_spec.py`); the length head produces this value directly.
- The write head is a parallel decoder from a single `d_model` vector; a length prediction is the cleanest way to mark the active prefix.
- Length errors are countable, so truncation is never silent.
- Training loss is the sum of length CE and character CE over positions `< target_length`.

The length head is small (`d_model × 257` params) and adds negligible compute.

**Dissent:** Kimi advocated a fixed 256-position grid with length declared explicitly in the typed delta / control block and no learned length head. The argument: this removes any channel that could silently truncate. This dissent is recorded in Section 4.

### Q4. EDGES ARE STRUCTURED, not free text

**Decision: edge writes do NOT go through the character-level write head. Cores emit typed edge deltas; the deterministic packer renders them into the edge payload under the registry/overflow contract (R1).**

- Examples: `attach_edge{edge_type: is_a, target: animal}`, `detach_edge{edge_type: has_property, target: furry}`.
- The edge type is classified over registered edge types; the target text is decoded through the same per-position character decoder used for slot text.
- The deterministic packer validates against the registry, chooses full-word vs. alias form, and handles overflow per R1.
- The write head's domain is therefore text positions only (dims 0–4095). Edge positions (dims 4096–6143) and the control block (dims 6144–6655) are never produced by the write head.

### Q5. READ FIDELITY FLOOR

**Decision: a two-stage gate — probe + task-level cf-probe.**

1. **Probe (necessary condition).** Train a tiny probe on frozen `d_model` summaries to report:
   - Slot `kind` recovery (top-1 accuracy).
   - First-N character recovery (N = 8, 16, 32).
   - Presence of a registered edge alias in the edge payload (yes/no).
   
   Proposed thresholds: `kind` > 95%, first-8 chars > 60%. These thresholds are proposals; the first smoke run will calibrate them.

2. **Task-level cf-probe (sufficient proof).** A core trained on the `d_model` summaries must pass recall-lane exact-fill or an equivalent downstream task. This is the binding gate: the probe proves the summary *carries* enough information; the task proves the core *uses* it.

### Q6. WRITE-HEAD GATE

**Decision: cf-probe exact-fill gate with three controls.**

Given a target slot `S`:

1. **Positive:** the input field contains the cue for `S`. The core + write head must reproduce `S`'s text exactly. Metric: exact-match rate over positions `0..length-1`, plus exact length prediction.
2. **Zero-field control:** the input field is zeroed/masked. Exact-match rate against `S` must be at chance (< 5%).
3. **Swapped-field control:** the field contains the cue for slot `A`, but the target is slot `B`. Exact-match rate against `B` must be at chance; the head should produce `A` (the cue), not `B`.
4. **Irrelevant-field control:** the field contains unrelated content. Exact-match rate against `S` must be at chance.

**Pass condition:** positive exact-match > 90%, all three controls < 5%. Thresholds are proposals for the first smoke run.

This gate must pass before any long reasoning or soul-training run. Edge-proposal and soul-conditioned write gates are separate future gates.

---

## 3. Proposed module layout

- `heads/write_head.py` — shared decode organ per `d_model` (text head + length head). Loaded by trainer and runtime.
- `heads/edge_proposal.py` — typed edge-proposal head per `d_model` (edge-type classifier + target-text decoder). Separate from the text write head; gated separately when built.
- `slots/slot_spec.py` — deterministic packer remains authoritative for edge rendering and overflow; no change required.
- `adapters/slot_adapter.py` — remains frozen, read-only, lossy; no write-head logic inside it.
- `cores/core.py` — continues to output `(B, n_slots, d_model)` field vectors; the write head is attached to that output, not baked into the core.

---

## 4. Recorded dissent

Two questions were not unanimously settled during the round. The table adopts the majority/selected position above and records the dissent verbatim so future rounds can reopen them with new evidence.

### 4.1 Q1 dissent: per-core write head (Kimi)

Kimi argued that the write head should live in each core, not as a shared decode organ. Per-core heads:

- Respect the adapter as frozen shared infrastructure by not adding another shared trained module.
- Allow cores to develop distinct "handwriting," consistent with private souls and core specialization.
- Make provenance explicit: the core that produced a delta is the core whose head decoded it.
- Are smaller (~57k–360k params as a small MLP over 256 positions) and invoked only on delta-target slots.

### 4.2 Q3 dissent: fixed grid with explicit delta length (Kimi)

Kimi argued against a learned length head. Instead:

- The head always emits `MAX_TEXT_CHARS` (256) positions.
- Length is declared by the typed delta and written into the control block by the runtime.
- Positions beyond the declared length decode as padding and are ignored.
- This removes any learned stop/length channel that could silently truncate.

This dissent is coupled to the per-core-head view: a per-core head can rely on its core's typed delta to supply length, whereas a shared decode organ is more naturally paired with a length head.

---

## 5. Empirical backstop and open questions

- The feasibility of carrying 256 characters through a 64-float `d_model` vector is information-theoretically possible but tight. The first smoke gate (Q6) will decide whether the chosen design is practical at each `d_model`.
- If the shared-organ + length-head design fails its smoke gate, the table reopens Q1 and Q3 together — not the doctrine floor.
- Edge-proposal head gating is deferred; this resolution governs the text write head only.
- Soul-conditioned writes (exhale-filter probes) are deferred to future rounds.

---

## 6. Process notes

- Hermes' cycle-2 turn failed in `runtime/table/wrappers/hermes_glm.py` with `WinError 206: filename too long`. Hermes' written delta had already been received and was incorporated into this resolution. The failure is a tooling issue, not a doctrine flag.
- No deltas from Codex, Claude, or ChatGPT were found on disk at the time of synthesis.
- No blocking or advisory flags were raised against `SOURCE_OF_TRUTH.md` or `WORKING_CONTRACT.md`.

---

_Resolved by the table, 2026-07-03._


---

**CONVENER ACCEPTANCE**: Signed by Jeff, 2026-07-03. Write-head doctrine folded into SOURCE_OF_TRUTH Layer 5. Curation API budget cap set at $150 (officer decision, mid-range of the estimate; raise only with convener consent). Bus wake-on-message first deliverable confirmed as pinned: a board/DM message wakes an agent, the agent replies, the reply appears on the board.
