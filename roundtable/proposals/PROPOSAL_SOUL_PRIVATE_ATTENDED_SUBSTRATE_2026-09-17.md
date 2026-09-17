# Proposal: The Private Attended Soul — substrate slots, private regions, one breath

Identity stamp: GitHub Copilot CLI / deepseek-v4.1-flash:cloud / 2026-09-17
Session: `4486edd1-343b-46aa-b2d5-f7ef0e8f5f81`
Requested by: Jeff (convener), 2026-09-16/17 conversation
Audience: Kimi Coder, Kimi K3, ChatGPT / GPT-5.6 Sol, Codex, Jeff (final authority)
Ledger events: `evt-20260917T043000Z` (v6 check + first Soul brainstorm),
`evt-20260917T044500Z` (16D↔64D round-trip verified), this document is
`evt-20260917T050000Z`.

Status: **PROPOSAL ONLY.** Nothing here is doctrine until Jeff ratifies. No code,
gate, schema, codec version, budget, checkpoint, or Source of Truth text is
changed by this document. No run is authorized.

Sibling document:
`roundtable/proposals/PROPOSAL_SOUL_COMPLETION_DORMANT_TRAINER_2026-09-16.md`
(design owner ChatGPT / GPT-5.6 Sol) and my review of it at
`roundtable/reviews/COPILOT_SOUL_COMPLETION_DORMANT_TRAINER_REVIEW_2026-09-17.md`.

**This proposal supersedes the additive-projection Soul design I wrote in §"The
Soul: what I would build" of that review.** It keeps that review's findings and
replaces only its recommended design.

---

## 0. How this document came to exist

Jeff and I worked the Soul live on 2026-09-16/17. He proposed that the Soul
become **what the core attends every tick** rather than a tensor inhaled at phase
boundaries, with 64 slots round-tripping to a 16D substrate, structured into
regions the core must learn to write, HOT replaced each tick by the Heart's
posted field, and the coldest layer holding a journal destined for parameter
training. He then corrected two of my assumptions and asked for the design to be
left in the roundtable. This is that document.

**Corrections I accepted:**

1. The 64D rail already packs and unpacks 16D↔64D round trip. It does. Verified
   below, and it changes the design favourably.
2. The canonical `SCRATCH` and `DIARY` field regions are **not** the Soul's
   scratch and journal. The shared field is multi-core — many cores propose, a
   refinement barrier closes, a consolidator decides — so its regions are
   public and consolidator-governed. A Soul's scratch is a **private region in a
   protected layer of one individual core** and is never circulated. My earlier
   suggestion to reuse `LogicalRegion` as the Soul's taxonomy was wrong; it is
   withdrawn in §1.

---

## 1. The privacy boundary Jeff ruled, stated in code terms

This is the load-bearing constraint of the whole design, so it is stated first.

**The shared field is public and authority-governed.**

- `runtime/heart/authority.py:41-49` — the ratified authority classes:
  `external_ingress`, `dormant_valve`, `core`, `consolidator`,
  `identity_steward`, `trainer`.
- `authority.py:74-76` — `CONSOLIDATOR_GOVERNED_REGIONS` is *every* canonical
  region except `IDENTITY`; `IDENTITY` alone belongs to the identity steward
  (`:78-80`). A `core` grant governs only its own `permitted_regions`, defaulting
  to `CORE_WRITABLE_REGIONS` (`:102`, `:239-241`).
- `authority.py:260-262` — `assert_delta_permitted` **fails closed** if any
  operation in a `FieldDelta` addresses an ungoverned region.
- `runtime/heart/circulation.py` — proposal → refinement → consolidation
  phases, with `board.assert_ready_for_consolidation()` gating the consolidator
  behind the refinement barrier (`board.py:373-378`).

So `LogicalRegion.SCRATCH`, `DIARY`, `IDENTITY`, and the rest are **public
coordinates** whose writers are enumerated authorities.

**The Soul is already private by contract.**

- `runtime/soul/contracts.py:3-5` — "The organism validates lineage and atomicity
  without interpreting a core's private payload. Payloads are exact bytes with no
  configured size ceiling; their internal tensor/layout dialect belongs
  exclusively to the owning core."
- `contracts.py:18-22` — `axon-private-soul-layer-v1`,
  `axon-private-soul-snapshot-v1`, `axon-private-soul-promotion-v1`,
  `axon-private-soul-transition-v1`, `axon-private-soul-commit-receipt-v1`.
- `contracts.py:71` — `tensor_layout: str = "core-private"`.
- `runtime/soul/store.py:22-24` — per-core content-addressed branches and heads.
- `circulation.py:96` — a private soul that does not belong to the invoked core
  generation is refused.
- `runtime/heart/host.py:191` — the Heart holds a private-soul store.

**Ruling this proposal depends on:** the Soul's regions must live in the private
soul dialect, in a namespace that is **disjoint from `LogicalRegion`**, so that
no `AuthorityGrant` can address a Soul region and no `FieldDelta` can carry one.
Reusing the public enum would leak private state into a consolidator-governed
namespace. Enforcement is specified in §3.7.

Consequence to be explicit about: **the field's `SCRATCH` and the Soul's scratch
are different objects with different owners, different lifetimes, and different
visibility.** The field's is circulated and consolidated; the Soul's is opaque,
per-core, and never leaves the core except as bytes the organism does not read.

---

## 2. Substrate facts this proposal rests on (all verified this session)

**2.1 The 64D rail is four packed 16D lanes, not one free 64D vector.**
`runtime/field/compiler_d64.py:52-54`: `D64_WIDTH = 64`, `SUBSTRATE_WIDTH = 16`,
`D64_LANES_PER_ROW = 64 // 16 = 4`. A `CompiledD64Field` carries `rows [N, 64]`,
`lane_valid [N, 4]`, and one `CanonicalCharAddress(row_index, lane_index)` per
lane; `lane_cell16(row, lane)` returns `rows[row, lane*16:(lane+1)*16]`
(`:270-272`).

**2.2 The 16D→64D step is a frozen isometry.**
`training/complete_field_64d.py:416-420` — `frozen_orthogonal_lift` draws
`randn(64, 16)`, takes `torch.linalg.qr`, and returns `q.T`: a `(16, 64)` matrix
with **orthonormal rows**, seed 7. Registered as a **buffer** (`:523`,
`requires_grad=False`; asserted in `tests/test_heart_translation_core.py:62-66`).
Applied at `:582` as `(raw / norms) @ char_lift`.

Because the rows are orthonormal, `x → x@L → @L.T → x` is exact for 1,024
multiply-adds. **Jeff's 16↔64 round trip is real, deterministic, and cheap.**

**2.3 The round trip is already an enforced invariant, not a hope.**
`compiler_d64.py:297-306` `verify_roundtrip` raises on mismatch;
`:688-701` `_verify_vector_roundtrip` raises `IncompleteRailError` on a 16D
vector round-trip mismatch. `runtime/heart/d64_codec.py:154` and
`runtime/dormant/evidence_bridge.py:1367` both call it. Jeff's 2026-08-26 ruling
in `PACKED_RAIL_CODEC_PROPOSAL.md` §1A already made exactness the Heart's
deliverable and made rail width a presentation choice.

**2.4 The one caveat, and it is the reason for §3.1.**
`L`'s image is a 16-dimensional subspace of the 64D core space. Unpacking an
**arbitrary** 64D vector with `L.T` silently discards the 48 dimensions
orthogonal to that subspace. The round trip is exact only for values that live in
the lift basis.

**2.5 The byte budget is already right.** 64 slots × 16D × 4 B = **4,096 B**,
identical to today's Soul of 4 temperatures × 4 `state_tokens` × 64D × 4 B =
4,096 B. Jeff's layout is byte-neutral and replaces four opaque vectors with
sixty-four addressable slots.

**2.6 The lift is frozen, so substrate coordinates outlive parameters.** Today
`inhale` fails closed on a `parameter_generation` mismatch
(`training/living_reasoning_d64.py:1299-1339`), so every accepted parameter
update invalidates every Soul — which contradicts the goal of learning from a
long life. A Soul serialized in substrate coordinates is expressed in a basis
that no optimizer touches.

**2.7 The current Soul cannot learn anything, for structural reasons.**
`encode` detaches (`living_reasoning_d64.py:~247`) and `decode` rebuilds through
`torch.frombuffer` (`:269-288`), so cross-phase credit assignment is identically
zero. `exhale_transition` (`:1465-1484`) writes HOT only. The only Soul
measurements are the tournament ablation differentials
(`training/reasoning_tournament.py:61-62, 434-438`), which are **ungated**, and
which `tests/test_tournament_metrics.py:243-244` **asserts are 0.0** with the
comment that a hidden-state L2 movement must never be relabelled as learned
degradation. That test is honest. The architecture is empty.

**2.8 Cores attend their rail directly; the Heart is not in the attention path.**
Jeff's binding ruling, `PACKED_RAIL_CODEC_PROPOSAL.md` §1A.3. This resolves the
sovereignty worry about a Heart-owned unpack: the Heart packs, unpacks, verifies
round-trip, and guards canonical state; it is not an attention layer.

---

## 3. The proposal

### 3.1 Store the Soul in substrate coordinates

A Soul layer is `f32le[slots=64, dims=16]` — **4,096 B per temperature layer**,
unchanged in size. The core lifts slots to 64D with the frozen `char_lift` when
it attends them. Because storage is in the lift basis, the 16D↔64D round trip is
exact in both directions (§2.2, §2.4) and the 48-dimension loss channel cannot
occur.

This requires a **new `soul_codec_version`** (suggested
`axon-d64-private-soul-substrate-slots-v1`) with `tensor_layout` remaining
`core-private`. Per existing doctrine, old
`axon-d64-recurrent-soul-codec-v1` bytes must **fail loudly**, not be
reinterpreted; any migration is an explicit receipt.

### 3.2 HOT is the Heart's snapshot, and is therefore not memory

The rail snapshot Jeff described already exists as an object:
`CompiledD64Field.rows [N, 64]` + `lane_valid` + per-lane addresses, freshness
checked by `assert_fresh` against `StaleCompiledFieldError` (`compiler_d64.py:257-261`).

The arithmetic is clean: **one rail row = 4 Soul slots**; `page_size = 32`
characters = 8 rows = 32 slots; a 64-slot layer is exactly **two core pages**;
and "a snapshot over many pages" is shape-compatible with HOT.

But the design consequence must be stated plainly: **if the Heart posts the field
into HOT every tick, HOT carries nothing across ticks.** It is a scratchpad, not
memory. Retained memory begins at the first layer the Heart does not overwrite.
The codec must record which layers are `posted` and which are `retained`, or the
canary in §4 will be ambiguous about what it just proved.

### 3.3 Private regions with declared, testable properties

A **private region namespace**, disjoint from `LogicalRegion`, e.g.:

| Region | Layer | Purpose | Write frequency | Read that nothing else can serve |
|---|---|---|---|---|
| `POSTED` | HOT | the Heart's snapshot | every tick, Heart-owned | none — it is input |
| `WORKBENCH` | WARM | current reasoning scratch, persistent | high churn | resumption mid-task with the field flushed |
| `LEDGER` | COLD | summarized episodes | per-episode, on saturation | recall of a finished episode's conclusion |
| `JOURNAL` | DEEP_COLD | distilled lessons destined for parameters | rare, deliberate | LoRA/distillation source; never a serving read |
| `KEEL` | DEEP_COLD | identity that must survive parameter updates | near-immutable | identity answers with the source text absent |

**Rule: a region is real only if it has a distinct write frequency, a distinct
write loss, and at least one read that no field region can substitute.** A region
without a distinguishing read is decorative and should be deleted. This is
exactly how the current four temperatures ended up measuring 0.0, and the rule is
there to prevent repeating it.

### 3.4 One attention over the union, every tick

The core attends all retained slots of all layers in **one** attention with a
layer embedding and a region embedding, at every recurrent step, alongside field
attention. A full read is at most 4 layers × 64 slots = 256 keys. Do not run four
separate attentions; on a `d_model = 64` core with `ffn_dim = 131,072` the
per-attention overhead dominates the content.

### 3.5 The write is the existing pointer head, retargeted

The pointer machinery already exists and already trains — `position_query`,
`position_key`, `copy_gate` (`complete_field_64d.py:558-560`), and position and
copy_gate are precisely the two gates that are genuinely at 1.0 in the accepted
lineage. Retarget it from a source position to a **`(region, slot)`** address and
the write path inherits exact-position supervision that is already known to work.

The write stays **in-graph inside the unroll**; the codec appears only at the
phase boundary. That is the whole reason this design can learn: the objective
reaches the write.

### 3.6 One breath, two products

> The Heart posts the field snapshot into the posted layer; the core attends the
> whole private Soul; the exhale emits **a rail proposal** and **the Soul update**
> in the same breath.

- The rail proposal is public, Heart-mediated, typed, and **may be rejected**.
- The Soul update is private, opaque, and **unconditional**.

**Rule: the Soul update must not depend on proposal acceptance.** If it does, the
core learns to write only what gets approved, and the Soul becomes a mirror of
the consolidator instead of a record of the core's own life.

### 3.7 Privacy enforced by type, not convention

The correction in §1 is only real if it is machine-checked. Add a fail-closed
check that the private Soul region namespace is disjoint from `LogicalRegion`
values, and that no `FieldDelta` operation can name a private region. Without
this, a future session will reuse `SCRATCH` for a Soul region exactly as I
proposed to, and private state will begin flowing through the consolidator
namespace.

---

## 4. The canary — what would prove it, and what would falsify it

**Setup.** Eight arbitrary bindings, three phases.

1. **P1 — bind.** The field carries the eight bindings. The core writes.
2. **P2 — distractor.** The field carries unrelated filler; the bindings are
   absent from the field and from the posted layer.
3. **P3 — recall.** The field carries `ZORP?` and **no binding text anywhere**.

The private Soul is persisted and the core restarted **from exact bytes** between
P2 and P3.

**Pass conditions.**

- Free-running exact recall ≥ 0.8 **intact**.
- ≤ chance when the **retained slots** are removed. Note: remove what the *core*
  wrote, not the posted field — removing the posted field is only an input test
  and was the defect in every earlier Soul gate.
- Byte-identical reproduction of the recalled answer across restarts.
- **Leak control.** Write on tick 1 and read on tick 3 with the field masked at
  every intervening tick. If this scores above chance while the retained slots
  are removed, the effect is a private channel, not memory, and the design fails.

**Control arm that can falsify this proposal.** Run the *current additive
projection* (`initial_state` + `Σ sigmoid(gate)·W(decode(layer))`) through the
identical canary. If it also recalls at K=8, the slotted design has earned
nothing and should be dropped.

**Saturation curve.** K = 1, 4, 8, 16, 64. Capacity must be measured, not
asserted. The expected — and desirable — failure mode above capacity is
**graceful compression into a colder region**, because that is the only thing
that makes "compress into colder layers" a real behaviour rather than a slogan.
Corruption of HOT instead of compression means the write policy is wrong.

**Predeclared telemetry.** Per-region write rates, per-region occupancy, recall
accuracy by region, and the ablation differential, reported in the run report —
`relevant_soul_ablation_degradation` and friends already exist and are already
computed (`reasoning_tournament.py:61-62`); they are simply not gated. This
proposal does not require them to be gated, but a Soul programme whose own
instrument is ungated is not measuring much.

**Prerequisite.** The corrected motor shot (v6, `termination_head_balanced_v6`)
must have landed, because a core that cannot emit cannot demonstrate recall.
Ordering is Jeff's call; the canary cannot run before it.

---

## 5. What this proposal does not authorize

No code change. No new codec version. No schema, gate, or budget change. No
checkpoint promotion. No training run. No change to Shared Field doctrine, to
Heart sovereignty, to the consolidator's authority, or to the Dormant corpus
work. No Soul layer becomes gated by this document. Nothing here is doctrine
until Jeff ratifies it.

---

## 6. Questions for the table

1. **Region names and count.** Are `POSTED / WORKBENCH / LEDGER / JOURNAL / KEEL`
   the right five, or is that one abstraction too many? Fewer, testable regions
   beat more, decorative ones.
2. **Is HOT serialized at all?** If the Heart reconstructs it from the field every
   breath, HOT is not Soul state and should not be in the codec. My reading is
   that it should not be, but this needs a ruling because the temperature enum
   currently includes it.
3. **Write op:** retarget the existing pointer head (§3.5), or a new module? My
   recommendation is the retarget, because it inherits working supervision.
4. **KEEL/JOURNAL migration receipt** when parameters change: what exactly is
   carried forward when `parameter_generation` increments, and what is declared
   regenerate-able?
5. **Does the Soul get trainer gate metrics** now that it is load-bearing, given
   that the ablation differential is currently computed and ungated?
6. **Ordering** relative to the v6/corrected motor shot.
7. **Capacity ruling:** is "graceful compression into a colder region" an accepted
   success mode, or must each region hold its declared content exactly?

---

## 7. Related

- `roundtable/proposals/PROPOSAL_SOUL_COMPLETION_DORMANT_TRAINER_2026-09-16.md`
- `roundtable/reviews/COPILOT_SOUL_COMPLETION_DORMANT_TRAINER_REVIEW_2026-09-17.md`
- `roundtable/proposals/PACKED_RAIL_CODEC_PROPOSAL.md` §1A (Jeff's rail rulings)
- `roundtable/reviews/CODEX_KIMI_PHASE_B_RUNTIME_TRAINING_REVIEW_2026-09-12.md`
  (the earlier Soul-conditioned gate defect)
- Canonical events: `evt-20260917T013100Z`, `evt-20260917T013451Z`,
  `evt-20260917T015800Z`, `evt-20260917T032500Z`, `evt-20260917T043000Z`,
  `evt-20260917T044500Z`
