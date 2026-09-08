# D64 Pointer Transition — Codex Resolution Candidate

Status: **FOR JEFF RATIFICATION — NOT YET BINDING — NOT IMPLEMENTED**  
Date: 2026-09-08  
Resolution owner: Codex  
Identity stamp: Codex / GPT-5 family (exact runtime model ID not exposed) / 2026-09-08

## Question resolved by this candidate

Once learned D64 tissue has chosen COPY and selected the exact first transport
cell of a non-native Unicode scalar, may an exact compiler-receipt mechanism
emit the remaining cells of that same scalar?

**Proposed answer: yes.** The remaining cells are deterministic serialization
of an already-selected canonical scalar. They are transport events, not new
semantic choices, learned pointer decisions, or canonical writes. The route is
permitted only under the controls below.

This candidate does not become doctrine until Jeff ratifies it. It authorizes
no code change, training run, checkpoint promotion, or serving activation by
itself.

## Review record

- Original brief: `CODEX_D64_POINTER_TRANSITION_ROUNDTABLE_2026-09-07.md`,
  SHA256 `371beef96fba958bb202e862152335c85f438d31b8e5c41d27ab848719d6e069`.
- Kimi review: `KIMI_D64_POINTER_TRANSITION_REVIEW_2026-09-07.md`,
  SHA256 `def2cb916583468a67ba752c6bed54528881c2577fbff71008e33b0275d77dde`.
- Gemini review: `GEMINI_D64_POINTER_TRANSITION_REVIEW_2026-09-07.md`,
  SHA256 `b1e6aa484b53f95ba54914b2a8547565b6eff19f3346e741362d382b73d90d40`.
- ChatGPT review received from Jeff and preserved verbatim as
  `CHATGPT_D64_POINTER_TRANSITION_REVIEW_2026-09-08.md`, repository-file
  SHA256 `aced38365994c433012cc7d3baa43e95f2bc777f62108ce992c8fbc882cada81`.

All three reviewers approve with named changes. None recommends the fully
learned shifted-pointer route for this bounded ablation.

## Verified implementation seam

1. `CanonicalCharAddress` already retains exact scalar/transport identity,
   including unit index/count, span, source, provenance, interval, row, and
   lane.
2. `AddressableMemory` retains only neural states, token categories, region
   IDs, and scalar region positions. It loses the receipt identity needed to
   prove continuation.
3. The current decoder performs a new learned memory-pointer query at every
   transport position. Its streaming iterator retains recurrent hidden state,
   the previous token, and the output buffer, but no explicit prior pointer or
   continuation state.
4. Canonical memory and derived proposal memories are concatenated. Several
   memory blocks can contain the same `(region_id, region_position)`, so those
   coordinates cannot identify a source cell by themselves.
5. The failed candidates learned COPY and the first source cell while failing
   continuation positions. Both also ended with EOS-gate accuracy zero.

## Correction to one review claim

Gemini correctly found that every aligned transport unit currently receives
copy-gate and position supervision, and correctly identified the absence of
continuation-state anatomy. However, the stronger claim that multi-cell units
mathematically overwhelm one EOS loss is not fully established by the live
objective:

- `copy_gate_loss` is a mean over copied positions;
- the active `copy_alignment` stage gives `alignment_eos_gate` weight `0.0`;
- the multi-cell overlay lowers `alignment_copy_gate` weight from `4.0` to
  `0.25`, while oversampling multi-cell cases.

The defensible diagnosis is therefore: shared gate parameters were trained
toward COPY while EOS was not protected in that stage. Continuation masking is
architecturally correct after the conduit exists, but it does not by itself
prove EOS retention. The new objective must co-supervise and gate EOS during
the same stage.

## Proposed binding clauses

### R1 — Narrow scope

The first mechanism completes only the remaining strict UTF-8 transport cells
of one exact copied canonical scalar. It never advances across scalar,
canonical region, attended interval, provenance span, source span, memory
segment, rail, field, view, tick, or pass boundaries.

### R2 — Three observable emission routes

Every emitted transport category is labeled with exactly one route:

- `learned_generate`;
- `learned_copy_anchor`;
- `deterministic_receipt_continuation`.

A continuation records its anchor identity, scalar receipt identity, unit
index/count, source memory segment, and transition trace. A deterministic
transport event is never reported as a learned pointer decision.

### R3 — Explicit learned anchor decision

The transition-enabled architecture exposes the learned route decision and
the exact learned pointer anchor before combining them into an emitted token.
The conduit may arm only when the learned route chose COPY and the selected
receipt is unit zero of a multi-cell scalar. Matching token content alone may
not arm it. The mechanism lives above the learned probability mixture; it does
not inject forced probability mass into `_decoder_logits`.

### R4 — One validated receipt construction path

`AddressableMemory` gains immutable per-slot receipt identity derived verbatim
from `CanonicalCharAddress`, plus explicit memory-segment identity. Every
constructor and `_join_memory` preserves these receipts. Joining memories
never manufactures adjacency.

Before use, the memory builder proves receipt count equals neural-memory slot
count and cross-checks the exact token category, region/scalar position, unit
index/count, span/source/provenance/interval, row/lane, rail, source field, and
source tick. Any disagreement fails closed.

### R5 — Complete decoder execution state

`PointerState` is one component of a serializable `DecoderExecutionState`.
The complete state includes at least:

- recurrent GRU hidden tensor with dtype/device-independent serialized bytes;
- previous emitted/input transport category;
- pending scalar receipt and next unit index;
- exact field/tick/view/rail/surface/memory-segment/pass/head bindings;
- emitted transport buffer or content-addressed trace position;
- renewable work-slice accounting;
- architecture, parameter-generation, and state-schema identities.

The state supports exact serialize/reload/resume. Same-process generator
retention alone is not durable continuation evidence.

### R6 — One causal step machine

Teacher-forced, scheduled, greedy, local smoke, and runtime execution call the
same transition primitive in the new architecture. Legacy disabled mode may
retain its current vectorized teacher path only if bit-exact compatibility is
proven.

Every continuation cell consumes one decoder work unit and advances the GRU
once with the preceding emitted category, preserving the same recurrent token
history as ordinary autoregression. The continuation step does not run a new
learned route or pointer selection. Optional diagnostic logits are detached
from gradient and cannot affect emission.

### R7 — Fail-closed transition law

The conduit fails the emission visibly if:

- the anchor is not unit zero;
- a receipt is absent, malformed, duplicated, reordered, stale, or mismatched;
- the next receipt is outside the same scalar or source memory segment;
- exact category/receipt comparison fails;
- field, tick, view, rail, surface, pass, head, architecture, parameter
  generation, or memory identity changes;
- continuation encounters EMPTY, EOS, padding, an invalid slot, a mask or
  provenance boundary, or the end of available memory;
- a persisted execution state is substituted against different compiled
  evidence, even if its visible text is identical.

No transition failure is converted into a completed or partially accepted
Unicode output.

### R8 — Layer 13 and objective identity

Layer 13 applies to every **learned discrete decision**. Learned generation
categories, copy/generate route, learned source anchor, EOS, operation,
region, and address remain trained with their registered categorical losses.
No continuous vector regression or nearest-vector substitution gains output
authority.

Receipt continuation cells are not independently learned decisions. Their
training/runtime obligation is exact receipt validation and categorical
transport equality. Gradient-bearing payload, pointer-position, and
copy-gate losses are masked at those continuation events. Detached diagnostic
scores may be reported, but they do not count as learned accuracy.

This is a narrow clarification of the existing categorical law and requires a
Source-of-Truth amendment upon ratification. The loss/route/state changes ship
under new content-addressed architecture, objective-program, state-schema, and
candidate-generation identities. No existing objective or checkpoint is
relabeled.

### R9 — EOS is co-supervised, not deferred

The new copy-conduit stage trains and evaluates the learned EOS category and
generate-route decision in the same examples as copy anchors. EOS cannot be
assigned zero objective weight during this smoke. Advancement requires exact
teacher-forced and free-running EOS/termination on complete heldout and
regression surfaces, plus retention relative to the declared baseline.

### R10 — Metrics remain honest

Reports separate:

- learned route accuracy;
- learned anchor accuracy and changed-source exactness;
- deterministic continuation integrity;
- final categorical stream and Unicode scalar exactness;
- learned EOS and termination exactness;
- complete free-running typed-emission exactness;
- receipt rejection, state-resume, and legacy compatibility evidence.

Mechanical continuation events are excluded from learned pointer-correct and
learned payload-token accuracy. Mechanism success never becomes a semantic,
reasoning, serving, or promotion claim.

### R11 — Compatibility and lineage

The feature is opt-in. Disabled mode is bit-exact with current D64 behavior.
Existing checkpoints remain immutable evidence. Any old-weight initialization
of the new variant creates a new candidate lineage with an explicit migration
receipt naming every copied tensor, new state field, identity, initialization,
and disabled-route equivalence result. Forgiving load is forbidden.

### R12 — Acceptance before any cloud run

The local suite must include:

1. native and 2/3/4-cell copied scalars;
2. repeated identical scalars at different exact source locations;
3. changed-source minimal pairs;
4. real D64 row straddle and page-adjacent scalar completion—the compiler
   keeps one scalar inside one character page, so a page-straddle scalar is not
   a valid test;
5. false physical adjacency and joined-memory seams;
6. canonical/proposal memory coordinate collisions;
7. right categories under wrong receipts and corrupted categories under right
   receipts;
8. direct anchor on unit greater than zero;
9. EMPTY/EOS/padding/bounds/mask/provenance/stale-identity rejection;
10. mid-scalar renewable-slice pause, serialized process-style reload, and
    byte-identical resume;
11. teacher/scheduled/greedy route-trace, post-scalar hidden-state, and
    next-learned-step equivalence when prefixes are identical;
12. deterministic-event metric-exclusion tests;
13. EOS co-supervision and retention tests;
14. objective/architecture/state identity change-detection tests;
15. legacy-disabled bit-exact checkpoint behavior;
16. complete-field coverage and source/proposal/Soul counterfactuals.

Only after the local suite passes may one bounded, non-serving Kaggle ablation
run against both preserved copy-alignment examinations. The smoke must require
exact continuation, source-change, Unicode, EOS, termination, and typed-delta
gates. Falling loss is not acceptance. No wider rail and no long run are
authorized by this resolution.

## Deliberately rejected alternatives

- No `memory_index + 1` continuation without receipt proof.
- No deterministic probability injection inside the learned mixture.
- No teacher-only ground-truth PointerState.
- No cross-scalar automatic copying.
- No fully learned shifted-pointer primary route for this ablation.
- No partial Unicode scalar returned at a work-slice boundary.
- No mechanical continuation counted as learned intelligence.
- No checkpoint rewrite, silent migration, serving activation, or stage
  advancement from mechanism tests.

## Implementation sequence after ratification

1. Amend both Source-of-Truth mirrors with R1–R12's categorical
   learned-decision versus deterministic-transport distinction.
2. Add receipt-preserving memory and complete decoder-state schemas with
   validation/serialization tests.
3. Add the shared causal transition step and opt-in decoder route.
4. Add the new content-addressed objective and EOS co-supervision.
5. Complete all local acceptance and migration/legacy evidence.
6. Run one bounded local mechanism smoke.
7. If and only if local gates pass, launch one bounded private Kaggle ablation.
8. Reconcile exact evidence. Do not serve or advance to `transport_eos` unless
   every existing and new gate passes.

## Exact ratification requested

Jeff may ratify this candidate by stating:

> I ratify the D64 Pointer Transition Resolution Candidate dated 2026-09-08,
> including R1 through R12 and the Layer 13 clarification that deterministic
> receipt-continuation events are categorical transport rather than learned
> decisions.

