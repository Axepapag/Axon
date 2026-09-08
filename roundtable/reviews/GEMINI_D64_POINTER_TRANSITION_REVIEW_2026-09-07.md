# Gemini Review — D64 Pointer Transition Roundtable

Status: **ADVERSARIAL REVIEW — VERDICT: APPROVE WITH NAMED CHANGES + DOCTRINE RESOLUTION**  
Date: 2026-09-08 (UTC) / 2026-09-07 (Local)  
Reviewed proposal: `roundtable/proposals/CODEX_D64_POINTER_TRANSITION_ROUNDTABLE_2026-09-07.md`  
Identity stamp: Gemini / Gemini 3.8 Flash / 2026-09-07  
Assigned surface: semantic and authority boundaries, categorical emission contract, Layer 13 doctrine reconciliation, deterministic-conduit legitimacy, work-slice boundaries, and curriculum/gate falsification.

---

## 0. Independent Verification of Diagnosis and Artifacts

### VERIFIED (Inspected this turn)

1. **Compiler Receipts Retain Transport Adjacency & Counts:**
   - `runtime/field/compiler_d64.py:87-88`: `CanonicalCharAddress` stores `transport_unit_index` and `transport_unit_count`.
   - `compiler_d64.py:447-477`: These are populated per transport unit during character compilation. Contiguity is proven by `_transport_addresses_complete` (`compiler_d64.py:732-752`).
   - Crucially, `compiler_d64.py:459`: `region_position` is identical for all transport units of a multi-cell scalar.

2. **`AddressableMemory` Discards Transport Indices:**
   - `training/complete_field_64d.py:83-91`: `AddressableMemory` stores only `states`, `char_indices`, `region_ids`, and `region_positions`.
   - `complete_field_64d.py:415-438`: All units of a multi-cell scalar receive the exact same `region_position`. Therefore, querying memory purely by `region_position` cannot distinguish unit 0 from unit 1.

3. **Attention Query Formulation Creates an Inherent Continuation Trap:**
   - `complete_field_64d.py:505-516`: `position_logits = query @ key.T / sqrt(d)`.
   - In `training/living_reasoning_d64.py:398-418`, `alignment_supervision` supervises `position_logits` independently at every target transport token offset (`token_offset = 0, 1, ...`).
   - For a multi-byte scalar, the model points to `matches[0]` at step 0. At step 1, the model is fed unit 0 into its GRU, but `matches[0]` and `matches[1]` share the same `region_position` and page-level contextual encoding. The model repeatedly attends back to `matches[0]`.

4. **Five Consecutive Campaigns Conclusively Prove the Failure Ceiling:**
   - As recorded in `roundtable/ENGINEERS_LEDGER.md` (lines 64-74), across FFN256, FFN512, renewal, multi-cell teaching, and fresh corrected-identity shots, regression position accuracy remained pinned at `0.6667` (exactly 8/12 native characters passing, 4/4 continuation cells failing by repeating unit 0).
   - In the Unicode-walk candidate `050a3a97…` (`r64v3-884aaafb15480948`), combined position accuracy reached only `0.6452 / 0.6571`.
   - The model has reliably learned the copy gate (1.0) and the anchor position (1.0), but is architecturally incapable of stable autoregressive pointer incrementation across sub-scalar cells.

5. **EOS Gate Collapse Trigger Identified:**
   - In both fetched bundles (`2c7e912b…` step 60 and `050a3a97…` step 120), `alignment_eos_gate_accuracy` collapsed from 1.0 down to 0.0.
   - In `living_reasoning_d64.py:407-418`, `copy_gate_losses` penalizes `gate_logits` at every single transport unit. Because multi-byte characters multiply the number of copy-supervised positions relative to the single EOS position (`living_reasoning_d64.py:423-426`), the gate gradient is overwhelmed with copy signals, suppressing the EOS decision at sequence termination.

---

## 1. What Precise Invariant Could This Design Violate?

1. **Causality & GRU Hidden-State Parity (Kimi C1 confirmed):**
   In teacher-forced training, ground-truth tokens are stepped through the GRU (`complete_field_64d.py:554-560`). If the runtime conduit emits continuation cells directly from compiler receipts without feeding each emitted token through the GRU, the hidden state at the conclusion of the scalar diverges between training and runtime. Subsequent decisions (the next scalar anchor, generate, or EOS) will execute from a desynchronized latent state.

2. **Layer 13 & Loss Surface Doctrine (FLAG F1 Analysis):**
   SOURCE_OF_TRUTH lines 337-340 and WORKING_CONTRACT §7 (Layer 13) require discrete per-lane categorical decisions trained with per-slot cross-entropy.
   If continuation slots are emitted by a deterministic conduit:
   - Supervising `position_logits` and `copy_gate` on continuation slots forces the core to learn a pointer transition it does not execute at runtime, re-introducing the destructive gradients that collapsed the EOS gate.
   - Omitting loss on continuation slots without doctrine clarification creates ambiguity under Layer 13.
   - *Resolution detailed in Section 5 below.*

3. **Renewable Work-Slice Incomplete Yield Boundary:**
   Under SOURCE_OF_TRUTH lines 717-723 and 801-806, the core must yield at `reasoning.emission_work_slice_transport_units` boundaries without converting slice exhaustion into completion.
   If a work-slice budget expires while a 4-cell scalar is partially emitted (e.g. after cell 1 of 4), `iter_decode_transport` must preserve `PointerState` (including `continuation_pending`, `target_memory_index`, `current_unit_index`, and `total_unit_count`). If `PointerState` is dropped, resumption would either query the unguided decoder (producing corrupt partial scalars) or fail closed to `""`.

4. **Multi-Rail Memory Collisions (Kimi C4 confirmed):**
   `_proposal_compiled` (`living_reasoning_d64.py:521-528`) compiles proposal workspaces into region `ADVISOR_INPUT` starting at position 0. When concatenated via `_join_memory`, `(region_id, region_position)` is non-unique in `complete_memory`. `PointerState` must bind the absolute memory chunk index and compiled `rail_id`.

5. **Copy vs. Generation Architectural Scope:**
   The conduit operates strictly when the learned copy gate chooses COPY. It cannot and must not be used to auto-complete multi-byte scalars emitted by the GENERATE path. Generated non-native characters remain governed by the generator head.

---

## 2. Deterministic Intra-Scalar Completion: Transport or Unauthorized Bypass?

**It is legitimate transport, fully compliant with Axon doctrine.**

1. **The Substrate Doctrine:**
   SOURCE_OF_TRUTH lines 26-30 explicitly defines byte transport:
   > "Canonical text stores raw exact Unicode scalars. A native character maps to its one original 16D cell. Every other valid scalar maps deterministically to its strict UTF-8 bytes, one typed 16D byte-transport cell per byte. This is an additive categorical transport, not normalization, escaping, or a widening of the substrate."
   The division of a Unicode scalar into 2, 3, or 4 cells is a mechanical serialization artifact of the 16D substrate codebook, NOT a cognitive or semantic decision.
2. **Preservation of Core Agency:**
   The learned core continues to decide:
   - Whether to emit a `DELTA`, `NO_OP`, or `ABSTAIN`;
   - The target region and address;
   - The choice between `GENERATE` and `COPY`;
   - The exact source scalar anchor in the Shared Field.
3. **Preservation of Heart Authority:**
   The conduit resides within the core's transport decode loop. It outputs discrete 351-category tokens. The Heart remains the sole authority that receives the `ReasoningEmission`, validates the category stream, verifies UTF-8 completeness and provenance bounds, and commits canonical state.
4. **Reduction of Malformed State Risk:**
   Allowing attention to guess continuation bytes risks generating invalid UTF-8 byte sequences, causing `decode_unicode_tokens` to raise a `ValueError` and discard entire emissions (`living_reasoning_d64.py:748-751`). Deterministic continuation ensures that once a scalar is chosen, its transport representation is physically valid by construction.

---

## 3. What Failures Would the Current Acceptance List Miss?

In addition to Kimi's C1–C7, the acceptance suite must verify:

1. **G1 — Mid-Scalar Work-Slice Preemption & Resume:**
   Configure `work_units = 2` on a test case copying a 4-cell UTF-8 scalar (e.g. emoji `U+1F600`). Verify:
   - Slice 1 yields `("", False)` after emitting cells 0 and 1;
   - Resuming the iterator immediately emits cells 2 and 3 without invoking `self.decoder` or `position_query`;
   - The final decoded text and GRU hidden state exactly match an uninterrupted run.
2. **G2 — Continuation Loss Masking & EOS Gate Protection:**
   Assert that `alignment_supervision` masks out `position_loss` and `copy_gate_loss` on continuation tokens (`token_offset > 0`). Verify on synthetic training steps that `alignment_eos_gate_accuracy` remains at 1.0 instead of collapsing to 0.0.
3. **G3 — Memory Bounds & Cross-Region Contamination:**
   Construct a test where an anchor is artificially placed at the terminal memory cell of a region with `transport_unit_count > 1`. The conduit must fail closed (emit no further units and abort the emission) rather than reading into an adjacent region or beyond tensor memory.
4. **G4 — Receipt/Cell Category Cross-Check:**
   Assert that for each continuation step `u`, `memory.char_indices[anchor + u]` matches the compiler receipt's `token_id`. Any bit-level discrepancy must fail closed immediately.
5. **G5 — Strict Inactivity on Generate:**
   Assert that when `generate_gate` chooses generation, `PointerState.continuation_pending` remains `False`, even if the generated token ID happens to match a multi-byte leading byte.

---

## 4. Is a Fully Learned Shifted-Pointer Alternative Materially Better?

**No. It is strictly inferior on all engineering, efficiency, and safety criteria.**

1. **Unnecessary Architectural Complexity:**
   Learning a distribution shift requires propagating an $M$-dimensional probability vector (where $M$ is memory length) or a learned recurrent offset through the decoder. This inflates compute and memory, violating the compact $O(1)$ scalar requirement for `PointerState`.
2. **Absence of UTF-8 Safety:**
   A learned distribution can place non-zero probability on incorrect memory indices, leading to partial or illegal UTF-8 byte streams that fail Heart validation.
3. **Falsified by Prior Evidence:**
   Five rigorous empirical campaigns spanning parameter scaling, step renewals, and multi-cell curricula failed to learn this shift because the underlying cells share the same canonical `region_position`. Continuing to demand that attention weights act as a hardware shift register is a known dead end.

---

## 5. Resolution to FLAG F1 (Layer 13 & Loss Supervision)

**Finding:**  
Layer 13 ("Discrete content gets discrete losses: per-slot cross-entropy over the registered codebook. Continuous losses are auxiliary only") was instituted to forbid *continuous embedding regressions* (e.g. MSE or nearest-vector guessing) for discrete characters.

**Doctrine Ruling / Recommendation for Jeff and Codex:**
1. **Continuation Slots are Deterministic Transport Operations, Not Independent Learned Decisions:**  
   Once the discrete anchor is chosen via learned cross-entropy over memory positions, the continuation slots are strictly governed by the verified compiler receipt.
2. **Loss Masking on Continuation Positions:**  
   In `alignment_supervision`, `position_loss` and `copy_gate_loss` must ONLY be computed at the anchor position (`token_offset == 0`). Continuation positions (`token_offset > 0`) must be masked out of alignment loss.
3. **Categorical Cross-Entropy Compliance:**  
   In language-modeling cross-entropy, the emitted token at continuation slots is deterministically correct by construction (loss = 0).
4. **Content-Addressed Identity:**  
   This change modifies the loss surface and MUST be bound under a new content-addressed objective program ID (`FOUNDATION_MOTOR_V3_CONDUIT_PROGRAM_ID`) per Kimi C6.

This disposition fully honors the spirit and letter of Layer 13 without forcing contradictory gradients into the attention heads.

---

## 6. Verdict and Required Changes

**VERDICT: APPROVE WITH NAMED CHANGES.**

The proposal is well-founded, empirically justified, and architecturally sound. Implementation should proceed under Codex's ownership subject to the following unified checklist:

### Consolidated Named Changes (Kimi C1–C7 + Gemini G1–G5):

- [ ] **C1 (GRU Stepping Parity):** Conduit must feed each emitted continuation token into the GRU decoder step, asserting exact post-scalar hidden-state and next-step logit equality across teacher, scheduled, and greedy paths.
- [ ] **C2 (Verbatim Receipt Tensors):** Derive receipt tensors directly from `CanonicalCharAddress`, verify against memory indices at read time, and bind `rail_id` / `source_field_id` in `PointerState`.
- [ ] **C3 (Mid-Scalar Anchor Guard):** Anchors with `transport_unit_index > 0` must not activate the conduit; they must follow the fail-closed path and receive appropriate supervisory penalty.
- [ ] **C4 (Proposal Rail Disambiguation):** Test cross-rail collision rejection with proposal rails active in `complete_memory`.
- [ ] **C5 (Physical Row Straddle):** Replace the vacuous page-straddle test with a D64 4-lane row-straddle test.
- [ ] **C6 (Objective Versioning):** Ship the loss-masking update under a new content-addressed objective program ID with regression-change detection.
- [ ] **C7 (EOS Gate Protection):** Require an explicit teacher-forced EOS gate accuracy floor in the acceptance criteria.
- [ ] **G1 (Slice Preemption & Resume):** Persist `PointerState` in `iter_decode_transport` across renewable work-slice yields, and test pause/resume mid-way through a 4-cell scalar.
- [ ] **G2 (Continuation Loss Masking):** Explicitly mask `position_loss` and `copy_gate_loss` on `token_offset > 0` to eliminate gradient interference with the EOS gate.
- [ ] **G3 (Boundary & Region Safety):** Enforce fail-closed bounds checking preventing the conduit from advancing past region or buffer limits.
- [ ] **G4 (Token Receipt Assertion):** Verify `memory.char_indices[pos] == receipt.token_id` before emitting.
- [ ] **G5 (Generate Exclusivity):** Guarantee that `PointerState` remains inactive during generation.

---

## VERIFIED / ATTEMPTED / ASSUMED

- **VERIFIED:**
  - `runtime/field/compiler_d64.py` (lines 87-88, 447-477, 732-752).
  - `training/complete_field_64d.py` (lines 83-91, 415-438, 505-516).
  - `training/living_reasoning_d64.py` (lines 315-430, 711-761).
  - Checkpoint and segment evaluation metrics from jobs `2c7e912b…` and `050a3a97…` (copy gate 1.0, position 0.6667 / 0.6571, EOS collapse to 0.0).
- **ATTEMPTED:**
  - Full CUDA execution of multi-cell continuation training (not attempted; code changes are deliberately withheld until ratification per roundtable rules).
- **ASSUMED:**
  - Heart canonical commit logic (`runtime/heart/coordinator.py`) remains stable and unchanged.

No runtime code, checkpoints, or serving configurations were altered during this turn.

Identity stamp: Gemini / Gemini 3.8 Flash / 2026-09-07
