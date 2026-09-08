# KIMI Review — D64 Pointer Transition Roundtable

Status: **ADVERSARIAL REVIEW — VERDICT: APPROVE WITH NAMED CHANGES + ONE DOCTRINE FLAG**
Date: 2026-09-08 (UTC)
Reviewed proposal: `roundtable/proposals/CODEX_D64_POINTER_TRANSITION_ROUNDTABLE_2026-09-07.md`
Identity stamp: Kimi / Kimi Code CLI (K2 family, Moonshot AI) / 2026-09-08
Assigned surface: tensor shapes, causality, teacher-forced vs free-running parity,
page/row/slice boundaries, recurrent decoder and Soul state, device behavior,
PointerState efficiency and exact resume.

---

## 0. Independent verification of Codex's diagnosis (source + artifacts, not summaries)

### VERIFIED (ran/read this session)

1. **Compiler receipts already retain transport identity.**
   `runtime/field/compiler_d64.py:87-88` — `CanonicalCharAddress` carries
   `transport_unit_index` and `transport_unit_count`, populated per emitted unit
   at `compiler_d64.py:447-477` (`transport_unit_index=unit_index`,
   `transport_unit_count=len(encoded.token_ids)`), alongside `region_position`,
   `global_position`, `span_id`, `span_position`, `source`, `provenance`,
   `attended_interval_index/start/end`, `row_index`, `lane_index`. The compiler
   already proves per-scalar unit completeness and `0..count-1` contiguity in
   `_transport_addresses_complete` (`compiler_d64.py:732-752`) and proves rows
   never bridge mask/span/provenance groups in `_rows_respect_source_boundaries`
   (`compiler_d64.py:755-788`).

2. **`AddressableMemory` discards them.**
   `training/complete_field_64d.py:83-91` — the memory carries exactly
   `states`, `char_indices`, `region_ids`, `region_positions`. Unit index/count,
   global position, span, provenance, attended-interval, and row/lane identity
   are all dropped at `read_compiled_with_memory`
   (`complete_field_64d.py:388-439`). Note: `region_positions` is per transport
   *cell* (each unit of one scalar shares the scalar's `region_position`), and
   `char_indices` in living tissue is the identity transport-token map
   (`living_reasoning_d64.py:287-288`), so scalar grouping is indirectly
   reconstructible today — but adjacency, span, provenance, and interval
   receipts genuinely do not survive.

3. **Streaming decode has no pointer transition state.**
   `training/living_reasoning_d64.py:711-761` (`iter_decode_transport`) — the
   loop state is exactly GRU `hidden`, the previous emitted `token`, and the
   accumulated `transport` list. Nothing carries the previously selected source
   cell forward. Each emitted category recomputes the full pointer via
   `_decoder_logits` → `position_query @ position_key(memory.states)`
   (`complete_field_64d.py:505-516`). Claim 4 of the proposal (every decoder
   position is an independent query) is accurate for both training
   (`decode_teacher`/`decode_scheduled`, `complete_field_64d.py:545-627`) and
   free-running decode.

4. **Artifact metrics match the proposal's "Verified trigger".**
   Read directly from fetched, hash-verified bundles:
   - `State/training/cloud/jobs/2c7e912b…/outputs/axon_job/State/training/reasoning/r64v3-ee84d04b96c9744c/segment_000000001_000000060.json`:
     final heldout `alignment_copy_gate_accuracy = 1.0`,
     `alignment_position_accuracy = 1.0`; regression probe
     `alignment_position_accuracy = 0.6666666666666666`;
     `exact_serving_gate_passed = False`, `curriculum_stage_complete = False`,
     `final_checkpoint_id = 553d8da0…` (matches ledger).
   - `State/training/cloud/jobs/050a3a97…/outputs/axon_job/State/training/reasoning/r64v3-884aaafb15480948/segment_000000001_000000120.json`:
     copy gate 1.0 both probes; historical manifest `a872278f…` heldout
     position 1.0; Unicode-walk manifest `12df4547…` heldout position
     0.5217391304347826; combined heldout 0.6451612903225806, regression
     0.6571428571428571; all gates false; `final_checkpoint_id = a669030b…`.
   Both candidates non-serving, exactly as claimed.

5. **Codebook size.** `substrate/unicode_transport.py:28-30`:
   `TRANSPORT_VOCAB_SIZE = 95 native + 256 byte = 351`. SOT emission contract
   (`docs/SOURCE_OF_TRUTH.md:337-345`) requires discrete per-lane categorical
   decisions over the 351-category codebook plus EMPTY/EOS, per-lane
   cross-entropy per Layer 13, Heart-only validation/commit.

6. **Resumable-slice doctrine.** SOT `docs/SOURCE_OF_TRUTH.md:801-806`:
   the core exposes a resumable iterator preserving exact decoder state across
   renewable slices; durable Heart-host continuation is a serving gate. The
   `iter_decode_transport` docstring (`living_reasoning_d64.py:717-723`)
   implements exactly this posture.

### Additional finding from the artifacts (not in the proposal's trigger section)

Both final evaluations show `alignment_eos_gate_accuracy = 0.0` (down from
baseline 1.0) and `payload_eos_accuracy = 0.0`, `payload_transport_exact_rate =
0.0`, `typed_emission_exact_rate = 0.0`. The copy-gate lesson was learned so
hard that the EOS gate collapsed: free-running exactness is zero on every
surface, not merely on continuation cells. The diagnosis "continuation walking
is the defect" is correct but incomplete — any pointer-transition mechanism
will still ship a candidate whose free-running emission never terminates
exactly unless EOS behavior is co-supervised and gated. The proposal's metric
list (§4) does include "free-running sequence/EOS exactness"; the acceptance
list must also guard the teacher-forced EOS-gate regression.

### Evidence-trail correction (minor)

Canonical event `evt-20260907T235746190774Z-codex-pointer-roundtable` cites
`runtime/reasoning/living_core.py` as a read target. That path does not exist;
the actual living core is `training/living_reasoning_d64.py`. The substantive
claim drawn from it is nonetheless correct (verified above). Codex should note
the path correction in the reconciliation event; no ledger line is altered.

### ATTEMPTED

- Full per-cell replay of checkpoint `553d8da0…` ("every continuation cell
  returns to the scalar's first cell; expected continuations ranked second or
  third"): not re-run by me; I verified the aggregate regression position
  0.6667 from the segment JSON and the claim's presence in canonical event
  `evt-20260907T224202643848Z`. The per-cell replay stands as Codex's verified
  evidence, consistent with the artifact-level numbers I checked.

### ASSUMED

- That the Heart-side validation of `ReasoningEmission` category streams
  (malformed/duplicate/partial rejection) remains unchanged; I verified the
  core-side fail-closed decode (`living_reasoning_d64.py:746-754`,
  `decode_unicode_tokens` ValueError → `"", False`) but did not re-read all of
  `runtime/heart/coordinator.py`.

---

## 1. What precise invariant could the proposed design violate?

**Invariant A (doctrine, material — see FLAG F1):** SOT
`docs/SOURCE_OF_TRUTH.md:337-340` binds the emission contract to "discrete
per-lane categorical decisions over the 351-category registered transport
codebook … trained with per-lane cross-entropy per Layer 13", and Layer 13
(WORKING_CONTRACT §7) states "discrete content gets discrete losses: per-slot
cross-entropy over the registered codebook." Under proposal §4, training
"may supervise the learned anchor and copy gate" — i.e., receipt-bound
continuation slots plausibly receive *no* per-slot cross-entropy and are *not*
learned categorical decisions. Whether the invariant binds the **stream
presented to the Heart** (preserved — the conduit emits registered categories
only) or the **per-slot loss floor during training** (changed) is a genuine
doctrine ambiguity that materially changes the implementation (loss masking vs
receipt-index CE vs unchanged dense supervision). Per Working Contract §1 I
flag this rather than resolve it. My judgment-level read (level 5 only): the
stream contract is preserved, and Layer 13's intent — discrete content never
trained with continuous losses — is not violated by anchor-only supervision of
deterministic transport slots; but this must be ratified, not assumed.

**Invariant B (causality, implementation-loadable):** teacher-forced hidden
state must equal free-running hidden state after every scalar. Training feeds
*every* ground-truth transport token through the GRU
(`complete_field_64d.py:554-560`, `:599-616`). If the conduit "emits the
remaining receipt-bound cells" without stepping the GRU once per emitted cell
with the emitted category as input, the post-scalar hidden state diverges
between training and free-running decode even when the text matches — a silent
parity break that the current acceptance list (text-level parity) would not
catch. This is the sharpest tensor/causality hazard in the design. See C1.

**Invariant C (staleness):** PointerState references indexes into
`complete_memory`, which is rebuilt every forward pass and concatenated across
rails (`_join_memory`, `living_reasoning_d64.py:238-247`). Reusing pointer
state against a recompiled or different rail without an exact identity binding
violates the stale-rail law the proposal itself states. See C2/C4.

## 2. Deterministic receipt-governed intra-scalar completion: transport or unauthorized output bypass?

**Transport — provided C1/C2 land.** Reasoning from the verified code paths:

- The only route from decoder to canonical state remains
  `emit → decode_transport_greedy → ReasoningEmission → Heart`
  (`living_reasoning_d64.py:781-865`). The conduit operates *inside* decode and
  produces exactly the category stream a correct learned pointer would have
  produced. The Heart still receives, validates, and commits (or rejects) the
  full per-lane stream, including conduit-emitted cells.
- The conduit cannot invent content: continuation cells are read from compiler
  receipt metadata, and the compiler already fails closed on any
  non-contiguous, mis-counted, or non-roundtripping unit group
  (`compiler_d64.py:732-752`, `686-699`). Deterministic continuation strictly
  *shrinks* the malformed-emission surface: today a learned mid-scalar misstep
  yields a partial scalar and total emission loss at EOS
  (`living_reasoning_d64.py:746-754`); under the conduit a copied scalar is
  whole by construction.
- The one place a bypass could creep in is the new `AddressableMemory` receipt
  tensors: they must be derived **verbatim** from `CanonicalCharAddress` at
  read time and cross-checked against `char_indices`/`region_positions`
  (fail closed on any disagreement), never inferred from learned `states`.
  The proposal says this ("observable receipts, not learned semantic
  features"); C2 makes it testable.
- The conduit must fire only when the learned route genuinely chose copy at
  the anchor — the copy/generate gate (`complete_field_64d.py:532-536`) must
  still mediate the anchor decision, otherwise the conduit becomes a copy
  *generator* rather than a copy *finisher*.

## 3. What failures does Codex's acceptance list miss?

1. **Hidden-state parity, not text parity.** "Teacher/scheduled/greedy route
   parity" must assert equality of GRU hidden state and next-step logits after
   each conduit-completed scalar across all three routes — not merely identical
   output text. A burst-emitting conduit passes text parity while corrupting
   all subsequent decisions.
2. **Mid-scalar anchor.** The learned pointer may select a non-first unit
   (`transport_unit_index > 0`) of a multi-cell scalar — indeed the failed
   candidates do exactly the inverse (anchor first, then return). Defined
   behavior (no conduit; today's fail-closed EOS path) plus a supervised
   penalty and an explicit test are missing from the list.
3. **Rail collision.** `_proposal_compiled` compiles every proposal workspace
   into region `ADVISOR_INPUT` (`living_reasoning_d64.py:521-528`) and
   `_join_memory` concatenates canonical + proposal memories, so
   `(region_id, region_position)` is **not unique** in `complete_memory` — the
   canonical advisor region and every proposal rail restart position numbering
   at 0. PointerState must bind rail identity (compiled `rail_id` /
   `source_field_id` + tick + memory-segment index), and the acceptance list
   needs "same region/position on a different rail must not validate" with a
   proposal rail actually present in memory.
4. **EOS-gate regression guard.** Both final artifacts show
   `alignment_eos_gate_accuracy` collapsing 1.0 → 0.0 (§0). The list covers
   free-running EOS exactness but not a teacher-forced EOS-gate floor relative
   to baseline.
5. **Work-slice budget accounting for conduit cells.** The pause/resume test
   is listed; additionally, conduit-emitted cells must consume
   `reasoning.emission_work_slice_transport_units` identically in all routes,
   including a burst that crosses a slice boundary mid-scalar with correct
   accounting and byte-identical resume.
6. **Objective/architecture identity versioning for the loss-surface change.**
   Any masking/reduction change at continuation positions is an objective
   change and must ship under a new content-addressed objective-program
   identity with a regression test that any change alters the ID — the exact
   lesson of audit finding 1 resolved in `7b1857b`. The acceptance list is
   silent on objective identity (it names architecture identity only).
7. **The page-boundary test as written is vacuous.**
   `iter_character_pages` chunks by *canonical character groups*
   (`compiler_d64.py:334-356`), so a scalar's units can never straddle a
   physical page — good design, but then "scalar whose transport crosses a
   physical page boundary" tests nothing. What can actually happen is
   **D64 row straddle**: packing is per source group in 4-lane chunks
   (`compiler_d64.py:479-492`), so e.g. a 3-cell scalar starting at lane 2
   places unit 0 in row *r* and units 1–2 in row *r+1*. Rewrite the test as
   row-straddle plus page-adjacent (anchor's scalar completes exactly at a page
   end and the next learned decision opens the next page).
8. **EMPTY-category interaction.** `category == TRANSPORT_VOCAB_SIZE`
   currently fails closed (`living_reasoning_d64.py:752-754`). Specify and
   test that the conduit never fires on, through, or across an EMPTY anchor or
   EMPTY memory cell.
9. **Anchor argmax device sensitivity.** The conduit removes per-continuation
   argmax nondeterminism (a genuine device-behavior win — fewer float-sensitive
   decisions on CUDA), but the anchor choice remains an argmax over GPU float
   logits. Acceptance should require anchor decisions be logged into the
   transition trace so a CPU replay can verify the exact conduit path taken.

## 4. Is a fully learned shifted-pointer alternative materially better?

**No.** On my surface the deterministic conduit dominates:

- The thing being learned is deterministic by construction. Five data-only
  campaigns (FFN256/512, renewal, multicell teach, corrected-identity fresh,
  Unicode-walk) failed to learn exactly this function — regression position
  pinned at 0.667 across four of them (rolling ledger mixer table), 0.657 on
  the fifth. That is the predeclared ablation threshold, honestly applied.
- A learned shift gate reintroduces the malformed-partial-scalar failure mode
  (partial scalar → `decode_unicode_tokens` ValueError → whole emission
  discarded), which the conduit eliminates by construction.
- It inherits argmax/device nondeterminism and GRU-step ambiguity into what is
  exact transport; the conduit's state is O(1) integers that serialize and
  resume exactly at slice yields.
- Its one real advantage — generalizing across scalar boundaries — is
  explicitly out of scope and undesirable: cross-scalar auto-copy would skip
  the learned per-scalar choice the proposal correctly keeps.

Keep the shifted-pointer variant as a cheap falsification control in the
ablation if trivial to wire (Codex's "Alternative to challenge" framing), not
as the primary route.

## 5. Verdict

**APPROVE WITH NAMED CHANGES.** The diagnosis is verified against source and
immutable artifacts; the mechanism stays inside the transport boundary with
Heart authority intact; PointerState is cheap (O(1) host-side scalars next to
the existing `[1,1,64]` GRU hidden), device-safe (receipt metadata is small
int64 tensors on the memory's device; conduit steps are ordinary GRU steps),
and exactly resumable within the existing renewable-slice iterator contract
(SOT:801-806). Required changes before implementation:

- **C1 (causality/parity):** the conduit must advance the GRU — and therefore
  the semantic cross-attention input path — one step per emitted continuation
  cell, feeding the emitted category, in teacher, scheduled, and greedy routes
  alike. Parity tests assert post-scalar hidden state and next-step logits
  equality, not text only.
- **C2 (receipt integrity):** the new `AddressableMemory` receipt tensors are
  copied verbatim from `CanonicalCharAddress` and verified against
  `char_indices`/`region_positions` at read time; any disagreement fails
  closed. PointerState additionally binds compiled `rail_id` /
  `source_field_id` + tick + memory-segment index.
- **C3 (mid-scalar anchor):** defined no-conduit behavior, supervision, and
  test for anchors with `transport_unit_index > 0`.
- **C4 (rail collision):** acceptance test with proposal rails present in
  `complete_memory` proving `(region, position)` collisions across rails never
  validate.
- **C5 (boundary tests match compiler reality):** row-straddle and
  page-adjacent tests replace the vacuous page-straddle test.
- **C6 (identity):** continuation-position loss handling ships under a new
  content-addressed objective-program identity with a change-detection
  regression test, alongside the new architecture identity.
- **C7 (EOS):** teacher-forced EOS-gate floor relative to baseline added to
  acceptance, given the observed 1.0 → 0.0 collapse in both candidates.

---

## FLAG F1 [BLOCKING for ratification]

Mission item: proposal §3/§4 — deterministic continuation under the SOT
emission contract.
Problem: doctrine ambiguity, material.
Evidence: `docs/SOURCE_OF_TRUTH.md:337-340` + Working Contract §7 (Layer 13)
require per-lane categorical decisions trained with per-slot cross-entropy;
proposal §4 supervises "the learned anchor and copy gate", leaving
receipt-bound continuation slots without per-slot CE. Whether Layer 13 binds
those slots is undecided doctrine.
Options I see: (a) rule that anchor+gate supervision satisfies Layer 13 for
receipt-bound transport slots (stream stays discrete/categorical at the Heart
boundary) — cheapest, no SOT edit; (b) amend SOT to name deterministic
receipt-governed transport slots explicitly — clearest, slowest; (c) keep
dense per-slot CE on continuation slots against the receipt index — preserves
the letter of Layer 13, costs nothing mechanistically, but trains a decision
the runtime never takes.
My recommendation: (a) or (c); decide before ratification, since the loss
shape is content-addressed objective identity (C6) and cannot be quietly
changed later.
Work halted: none — this review is complete; the flag gates ratification, not
review.
Work continued: full source/artifact verification and this review.

---

## VERIFIED / ATTEMPTED / ASSUMED

- VERIFIED: §0 items 1–6 (source lines cited; segment JSONs read directly;
  substrate constant computed; SOT/contract lines quoted).
- ATTEMPTED: per-cell checkpoint replay (not re-run; aggregate metrics and the
  canonical replay event verified instead).
- ASSUMED: Heart-coordinator internals unchanged (core-side fail-closed decode
  verified instead); proposal-rail usage in the motor campaign training path
  (training path observed using canonical memory only via
  `forward_canonical_transaction`; the collision risk is verified real on the
  runtime `emit` path regardless).

No runtime, training, test, SOT, contract, config, or checkpoint files were
modified. No training launched. No commits made.

Identity stamp: Kimi / Kimi Code CLI (K2 family, Moonshot AI) / 2026-09-08
