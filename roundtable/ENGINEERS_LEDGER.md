# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-09-17T22:45:00+00:00
current_through_event_id:
`evt-20260917T224500000000Z-copilot-unwinnable-gate-audit-and-reachability-contract`

Append order note: the two events carrying timestamps `19:10` and `19:30` sit
*earlier* in the file than the `20:00` launch event, because the correction was
appended after the verdict it corrects. The canonical file is authority in
**append** order; these are correct events appended in a non-monotonic timestamp
sequence. The line above names the last physical line.

Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`

Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

Identity stamp: GitHub Copilot CLI / deepseek-v4.1-flash:cloud / 2026-09-17
(previous revision: Kimi / Kimi K2 Code / 2026-09-17,
and before that GitHub Copilot CLI / deepseek-v4.1-flash:cloud / 2026-09-16 —
those revisions are superseded, not erased; canonical events remain the authority)

## Current mission and honest status

**THE EMISSION RUNG'S VERDICT, READ AGAINST THE REAL FLOORS: content learned,
exactness went backwards into noise.** `evt-20260917T220500000000Z`. The
600-step tranche finished. Final heldout: `heldout_loss 5.206 → 1.056`,
**payload teacher-forced token accuracy 0.0% → 61.8% (real floor 43.6% —
BEATEN)**, motor-v2 copy gates 0.000 → heldout `copy-gate 0.903 / position 0.968
/ pair-gate 0.625 / pair-pos 0.875`, regression `0.914 / 0.971 / 0.750 / 0.875`.
**But:** `typed_exact` fell from **33.3% (exactly at the 24/72 floor, i.e. the
constant answer) to 0.0% — below floor**, and payload exactness sits at
**16.7% against the 8/24 = 33.3% floor — below floor**. The teacher-forced rows
show why: cases that must emit *nothing* now answer `'i'`, `'oo'`, `'VV'`, `'YYY'`.
The core left the emit-nothing dead state and entered a **noisy-emission** state.
**The rung did not pass its stage gate.** Content moved; exactness moved the wrong
way. The next objective must separate those two failure modes.

**Also found: a false-success fetch.** `Adapter.fetch` stamped the job record
`outputs_fetched` and returned exit 0 while downloading **zero** files, because a
still-running kernel downloads as an empty tree without raising. Now guarded
(`CloudPacketError` naming the provider status); the record for this run was
wrongly marked and has been corrected back to `submitted`. Outputs are still
unavailable — the kernel has not left `RUNNING` long after the loop and final
eval finished.

**I FOUND AN UNWINNABLE GATE, AND IT WAS MINE.** `evt-20260917T224500000000Z`.
While verifying my own floor repair I discovered that the `beat_floor`
assertions I had just added to `copy_alignment` and `transport_eos` required
`typed_emission_exact_rate` — a metric whose inputs that stage weights at
**0.0**. `typed_emission_exact_rate` is a DELTA conjunction over decision,
operation, region, start, end and exact free-running payload transport
(`living_reasoning_curriculum.py:897`), and `decision` is the decision head's
argmax (`:702`) whose loss carries weight `0.0` there (`:401-405`).
**Zero weight is exactly zero gradient**, so the maximum reachable value in
those stages is `0`. No lineage could ever have passed. That is the definition
of "setting us up for failure", and I had just written it in.

**The same defect class explains the flat lineage.** The historical
`transport_eos` gate demanded `payload_transport_exact_rate >= 0.95` while the
typed conjunction's inputs were untrainable in that stage. The termhead-v1
probation's "exhausted 3/3" plateau was a **mathematical impossibility recorded
as a learning failure**, not a core that failed to learn.

**Fixed, and made unrepresentable.** Both unreachable floor assertions are
removed. The emission rung keeps its real, trainable anti-vacuity proof —
`payload_content_accuracy > payload_content_constant_floor`, which an
emit-nothing core scores `0.0` on. The two floor comparisons moved to `joint`,
the first stage that weights every component the typed conjunction needs. And
the invariant is now declared in the objective program itself:

> **A stage gate may only require a metric whose causal components all carry
> nonzero weight in that stage.**

`FOUNDATION_MOTOR_V2_METRIC_COMPONENTS` and
`FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS` declare the dependency map;
`foundation_motor_v2_unreachable_gate_requirements()` returns `[]` for the base
program, all six receipt teaching profiles and the multicell overlay.
`tests/test_foundation_motor_gate_reachability.py` holds seven tests including a
**negative control** that re-declares the historical defect and asserts the
guard fires. 131 tests pass, `EXIT=0`. Design choices that remain Jeff's: see
open questions (a)–(e) below.

**THE FALSE-PROGRESS TRAP IS FIXED — and the number we were reading as progress
was the constant answer.** `evt-20260917T211900000000Z`. Two exactness floors
(`constant_typed_emission_exact_floor`, `constant_payload_transport_exact_floor`)
were literal `0.0` in three modules. They are now derived from the strongest
fixed answer over the evaluated surface. On the real 72-case heldout FFCS
surface the typed floor is **24/72 = 33.3%** and the transport floor is
**8/24 = 33.3%**. The step-0 `typed_exact 33.3%` on the dashboard was therefore
**exactly at the constant-answer floor**: an emit-nothing policy reported as a
learned result. Full detail in *"2026-09-17 — the false-progress trap, the
screen-hogging monitor, and a sync that never ran"* below.

**Also found: mid-run checkpoint sync has never worked on any cloud job.** Every
sync-enabled job in `State/training/cloud/jobs/*/outputs/axon_observability/**/sync_receipts.jsonl`
records `"status": "disabled"` with `SyncCredentialsMissing` (or "mid-run sync
environment is not set"). `sync_mid_run: true` in a recipe turns on internet and
injects `AXON_SYNC_MID_RUN=1`, but the `AXON_KAGGLE_SYNC` Kaggle User Secret is
never attached, so `_resolve_credentials()` fails, the failure is caught, and the
receipt said only `SyncCredentialsMissing`. The reason is now recorded and
rendered. **Jeff's decision:** attach the secret, or stop advertising sync.

**IN FLIGHT:** Kaggle job
`389df54d01fbda8ec6625b9019ff5fb1ec254c08360bf3d8cf4570c41ee45bd9`, revision
`1a4bc416`, Tesla T4, 600-step emission-rung tranche — **the loop and the final
evaluation are both done** (final numbers above); the kernel has not left
`RUNNING`, so the outputs bundle is still unfetched. Unaffected by local edits
because its packet was already uploaded.

**Prior state — the emission rung, launched on Kaggle.**
**EMISSION RUNG IMPLEMENTED LOCALLY, THEN LAUNCHED ON KAGGLE — the rung moved
but only partially: content now beats the constant floor and typed emission is
exact 25% of the time, yet exact end-to-end payload transport is still 0.0.**
(`evt-20260917T103000Z-copilot-emission-rung-implemented-and-local-gpu-tranche`;
full detail in *"2026-09-17 — the emission rung implemented and run"* below.)
**Caveat added 2026-09-17T21:19 (see the new section at the top of this file):
the "25%" figure and every exactness percentage in this summary must be read
against its constant-answer floor. 33.3% typed is the floor, so 25% is *below*
the emit-nothing baseline.**

**IN FLIGHT:** the same configuration now runs on Kaggle as a **600-step**
tranche — job `389df54d01fbda8ec6625b9019ff5fb1ec254c08360bf3d8cf4570c41ee45bd9`,
committed revision `1a4bc416`, Tesla T4, mid-run sync enabled, launched with a
live dashboard open. The recipe changes exactly one variable versus the local run
(`--tranche-steps` 60 → 600) and deliberately omits `--evaluation-case-limit`, so
the stage gate can return a **real** verdict. The two numbers to watch are
`payload_transport_exact_rate` leaving `0.0` and `nonzero_exact_output_observed`
flipping `True`. **Live progress at ~step 247/600 (≈7.2 s/step): loss has fallen
from ≈19 to ≈2.4–3.7**, the first checkpoint `0dfed88de4396d4` is written, and
mid-run sync reports `kernel disabled` (observation missing, training unaffected). (`evt-20260917T200034380158Z-copilot-kaggle-emission-rung-launch`;
full detail in *"2026-09-17 — the emission rung launched on Kaggle"* below.)

Jeff authorized the fix: *"Yes please do it. forget about kaggle for now and use
the local GPU. if the soul does not interfere than we can leave it as it."* The
Soul was verified untouched and non-interfering (40/40 `SoulLayer` records still
hold 0 bytes, generation 0), so it is left exactly as it is.

The corrected-v5 fix set is implemented as ONE identity-bearing opt-in change:

- `alignment_supervision` gains `supervise_termination_continue`: explicit BCE
  pushing the dedicated termination logit toward **stop=0 at every learned
  content anchor**, symmetric with the existing stop=1 BCE at the EOS position
  (`training/living_reasoning_d64.py`). Flag off = byte-identical to v5.
- New teach profile **`termination_head_balanced_v6`** (program variant-v7, id
  `9673967b...`, distinct from v5's unchanged `998dc091...`): restores the
  v4-balanced `payload_eos_weight 1.0` (v5 carried 4.0 — the parallel audit's
  arrow direction was backwards, substance confirmed), sets the continue flag,
  pins the component table explicitly.
- The **`transport_eos` stage gate now requires `payload_transport_exact_rate`
  >= 0.95** on both surfaces; the emit-nothing dead state (0.333 floor) can
  never pass again. Applies to v5 too, so the `bfe76d52` re-eval runs under
  corrected metrics.
- Five new tests in `tests/test_termination_head_route.py` (file 24/24); full
  `tests/` suite green EXCEPT a **pre-existing** stale day-zero hygiene
  allowlist (`runtime/trainer` surface, 28 committed files vs ~8 expected) —
  unrelated to this change set, left for Jeff.

Open decisions for Jeff: (a) v6 cannot resume `bfe76d52` in place (different
objective identity) — fresh lineage vs governed checkpoint seeding; (b) whether
to run the zero-GPU `bfe76d52` re-eval under the corrected gate; (c) the stale
hygiene allowlist.

**Prior state — why the fix was needed. `transport_eos` probation exhausted
3/3 FLAT.**

`d64-reference-termhead-v1` (candidate generation `r64v3-1b55ea4fa59ebf64`,
architecture `living-d64-receipt-e0c4e6487b5930b30b5d93dd`, profile
`termination_head_v5`) completed `copy_alignment` (landmark `063dcc0a`, stage
gate passed) and four `transport_eos` tranches (steps 25-48). Every tranche:
guard `plateau`, `improvements: []`, `failures: []`, action `probate`;
exhaustion then abandoned the probation branch and archived its sidecar.
Confirmed/accepted state is still step 24 `bfe76d52`; probationary step 48
`6ccf3b1b` is abandoned by design.

Verified step-48 state, both complete surfaces (heldout and regression):
corrected `alignment_eos_gate_accuracy` 1.0 (no termination collapse under
stage pressure, ever), `alignment_position_accuracy` 1.0,
`alignment_copy_gate_accuracy` 1.0, `payload_eos_accuracy` 1.0,
**`payload_content_accuracy` 0.0**, changed-source `content` 0.0,
`payload_transport_exact_rate` 0.3333 (exactly the empty-payload cases,
12/36). Heldout loss rose this tranche (0.7757 -> 0.8935) after four falls;
train loss flat/oscillating 3.60 -> 3.44 -> 3.39. Every QA transcript in the
report shows `predicted_payload: ''` with `terminated: True`.

**The stage gate never passed at `transport_eos`.** The step-48 report has
`foundation_motor_v2_stage_gate.passed = false`, six failures, all content on
both surfaces (`payload_content_accuracy` below 0.95, changed-source content
below 0.95, content does not beat the 0.125 constant floor), and
`task_gate_passed = false`. Tranches were driven only by the retention guard,
never by gate progress.

**Read-only audit, 2026-09-16 (this turn) — why the lineage is flat.** Numbers
are from `segment_000000041_000000048_probation.json`:

1. **Executed weights are not the declared weights.** At `transport_eos` the
   weights are the stage entry (`payload 1.0`, `alignment_position 1.0`,
   `alignment_copy_gate 4.0`, `alignment_eos_gate 1.0`), because
   `apply_receipt_continuation_teach_weights` applies the receipt overlay
   **only when `training_stage == "copy_alignment"`**. Every claim that the EOS
   gate carries weight 0.0 under v3/v4/v5 — and that v5's pressures are
   identical to v3's — is false in the stage that matters; the legacy EOS-gate
   BCE (post-repair, scoring the termination logit) runs at weight 1.0.
2. **83% of the objective sits on already-correct scalar decisions.** The
   `transport_eos` lane supervises one alignment position per episode
   (`alignment_copy_positions: 1.0` in `phase_metrics`), so position/copy_gate/
   eos_gate are 0/1 constants — all 1.0. Step-48 loss decomposes as payload
   0.59 + position 0.29 + 4.0*copy_gate 2.48 + eos_gate 0.08 = 3.44; 72% is one
   copy-gate BCE whose sign is already right.
3. **The payload term is 4:1 against content.** `sequence_cross_entropy(...,
   eos_weight=4.0, token_mask=learned_decision_mask)` masks out every
   deterministic-continuation position, leaving one content token plus EOS
   weighted 4x over a denominator of 5: the content token contributes 0.675 of
   3.44, and its gradient is scaled 0.2 while the copy gate's is scaled 4.0.
   Per-episode content CE is ~3.37 nats (P ~ 3.4% over 353 categories) at step
   41 and unchanged at step 48 — the content head has had almost no usable
   pressure in 48 steps.
4. **Net termination pressure is toward stopping immediately.** Content
   positions push the stop logit down (dL/dz_stop = +1/(1-stop)); the EOS
   position pushes it up at 4/5 of the payload weight, plus the active eos_gate
   BCE at 1.0. The fixed point is stop ~ 1 — hence empty payloads and transport
   exact = the empty cases.
5. **The gate never looks at free-running emission.** The `transport_eos`
   branch of `decide_foundation_motor_v2_stage` requires
   `alignment_position_accuracy`, `alignment_copy_gate_accuracy`,
   `payload_content_accuracy`, `payload_eos_accuracy` and pairs
   position/copy_gate/content (+eos_gate); it does **not** require
   `payload_transport_exact_rate`, the only metric that exposes the
   immediate-termination dead state. Its content requirement is teacher-forced
   over a mask of 8 learned positions in 36 cases, and teacher-forced content
   argmax is insensitive to the stop head by construction (the combined content
   logits differ from the content logits by the additive constant log(1-stop)).
   The termhead experiment's stated success signal cannot be seen through the
   metric used to judge it.
6. **Probation spent compute on no information.** `max_plateau_probation = 3`
   x 8 steps = 24 optimizer steps on a flat objective whose probation branch is
   discarded by design on exhaustion, while nothing asked whether the gate
   metric was reachable first.

This audit is the evidence base for the fix set Jeff authorized on 2026-09-17
("correct the trainer but don't launch") — now implemented as
`termination_head_balanced_v6` above.

**Independent Kimi audit, 2026-09-17 (`evt-20260917T013451Z`) — delta over the
audit above.** A second read-only audit (code plumbing + full history synthesis)
confirmed all six findings and adds: (a) the primary root cause is that stop=1
is supervised twice while **stop=0 is never explicitly supervised anywhere**
(`living_reasoning_d64.py:1220-1242` touches the termination logit only at the
EOS position, only toward 1); the only downward pressure is the implicit
`log(1-stop)` term inside content CE, giving a stable canceling-gradient fixed
point `sigma* = (n+8)/(2n+8) ~ 0.8-0.9` — a plateau by construction, not by
bad luck; (b) `termination_head_v5` silently **reverted the deliberate v4 fix**
`payload_eos_weight 4.0 -> 1.0` (`foundation_motor_curriculum.py:331-341` vs
`:379`); (c) the minimal fix set is: explicit stop=0 BCE at content anchor
positions in `alignment_supervision` under `termination_head_route` (preserves
the train/runtime distribution identity), restore `payload_eos_weight 1.0`, and
make the profile overlay apply per-stage (pinned by test); (d) the historical
synthesis shows this is one instance of a recurring objective class failure —
loss-down-behavior-wrong, teacher/free-running divergence, constant-prior
collapse, route-weight seesaw — while the process class (guard, probation,
lease) is now sound.

## Open proposal — Soul completion + Dormant trainer (reviewed 2026-09-17)

`roundtable/proposals/PROPOSAL_SOUL_COMPLETION_DORMANT_TRAINER_2026-09-16.md`
(ChatGPT / GPT-5.6 Sol; proposal only, not ratified). Requested decision: pause
wider curriculum expansion, make private Soul a proven load-bearing memory, and
point the trainer at governed Dormant State as the lifelong school.

Two independent reviews now agree on the blocking defect. Kimi's
(`evt-20260917T015800Z`) reached the same verdict — approve with named changes —
and the same first correction (the premise predates the objective-level root
cause; a perfect Soul still emits empty payloads); it adds the missing
Living-core training adapter and the C1 eligibility envelope as S0
prerequisites. It modified no files, so no review document exists yet under
`roundtable/reviews/`. This section records the independent review
(`evt-20260917T032500Z`) and only its deltas over Kimi's.

**Verdict: approve the direction, reject the sequencing — approve with named
changes.** The diagnosis is right and the doctrine is preserved (Shared Field
stays the input interface, Soul stays private, Heart stays sole committer,
`.derived` stays out of supervision). Two blocking corrections before
ratification:

1. **Every Soul canary in S1/S2 requires free-running emission, which the
   current objective actively suppresses** (stop ~ 1, `predicted_payload: ''`).
   `ZORP -> BLUE` recall cannot be measured until the stop-supervision defect is
   fixed, so the S0/S1 order as written will fail for an unrelated reason and be
   misdiagnosed as "Soul is not memory." Fix the termination/content objective
   and its gate first; then Soul.
2. **The proposal asserts motor exactness the audits contradict.** At the
   accepted step 24, `evaluation_component_weights.alignment_eos_gate = 0.0`
   (verified): the EOS gate is *untrained* in every accepted checkpoint, not
   proven. `alignment_copy_gate 4.0` / `alignment_position 1.0` are genuine;
   transport exact 0.333 is the empty-payload floor.

Verified premises (confirmed against disk, so they stand): corpus counts are
exact (`container` 427,001; `semantic_edges` 351,978; `layout_groups` 4,198);
`exhale_transition` does emit only HOT (`living_reasoning_d64.py:1483`); the
Soul codec is `f32le[state_tokens=4, d_model=64]` = 1 KB/layer, and both
`encode` (`state.detach()`) and `decode` (`frombuffer(...).clone()`) are
non-differentiable, so **cross-phase Soul credit assignment is identically
zero** — only same-graph probes can teach writing.

Findings the proposal must absorb:

- **Soul is already measured and already known to be decorative.** The
  tournament emits `relevant_soul_ablation_degradation`,
  `irrelevant_soul_ablation_delta`, `stale_soul_degradation`, and
  `swapped_soul_rejection_rate`; the step-48 report records degradation 0.0 and
  delta 0.0, and `tests/test_tournament_metrics.py:243-244` *asserts* 0.0 as the
  honest value. Nothing in `foundation_motor_curriculum.py` gates on any of
  them. Adopt that exact behavioral-differential metric as the S1/S2 instrument;
  the proposal's "the answer must change under ablation" test would pass on
  noise and repeats the same class of instrument error as the stage gate.
- **`semantic_edges` is not a clean relation curriculum.** 351,978 edges carry
  **41,142 distinct `edge_type` strings**, 20,684 of them singletons
  (`uses` 12,913 is the top), with free-text `target`s averaging 33 chars. The
  good hidden-target family is the container `edges` (`edge_type: "is a"` ->
  short `target` such as `tool`, `file`, keyed by `letters`/`entity_type`).
- **Recovered autobiography is not this core's life.** `experience_v1` holds
  59,925 records but 59,875 of them are the single import
  `d00-recovered-autobiography-v1` (`evidence_class:
  recovered_lived_evidence`, real `occurred_at`/`sequence`, D2-era "hello
  world", March 2026); live Heart-owned ingress is ~50 records. Training a
  current core's *private* Soul on another life risks manufactured memory —
  D1 needs an identity ruling, not just an eligibility ruling. Container
  `created_tick` is `-1` corpus-wide, so delay-age buckets need
  `experience_v1.occurred_at`.
- **v5 re-adopts a pressure v4 had already falsified.** v4
  (`payload_eos_weight 1.0`) documents that 4x EOS "still rewards the
  immediate-EOS basin"; v5 inherits v3's 4.0. The Kimi delta above states the
  numeric direction backwards (it is v4=1.0, v5=4.0) though its substance —
  v5 carries the discredited 4x weight — is correct. Both accepted reports do
  record `payload_eos_weight 4.0`.
- **Unresolved architectural conflict:** Soul is parameter-generation-bound,
  so every accepted parameter update invalidates it, while the proposal's goal
  is learning from a long life. Either freeze the Soul codec across
  generations, or make migration automatic and free, or lifelong memory is
  structurally impossible. This needs an answer before S1, not before D3.
- The proposal cites no ledger event and neither audit
  (`evt-20260917T013100Z`, `evt-20260917T013451Z`), though it was written after
  both.

No review of it existed under `roundtable/reviews/` when it was written.

**Follow-up, same turn (`evt-20260917T032500Z`):** filed the full review as
`roundtable/reviews/COPILOT_SOUL_COMPLETION_DORMANT_TRAINER_REVIEW_2026-09-17.md`
and amended the proposal in place at Jeff's direction — a
`## Reviewer corrections (2026-09-17)` block with both blocking corrections and
the seven named changes, an inline correction on the "Why this discussion now"
paragraph, and an ordering note on `## Proposed training sequence`. The phase
plan itself is unchanged and still not executable as ordered.

## Soul architecture under discussion (open, 2026-09-17)

Direction, raised by Jeff and accepted as stronger than the additive-projection
design filed in the review: **the Soul is what the core attends every tick**,
not a tensor inhaled at phase boundaries. The breath becomes: the Heart posts
the shared field into the HOT layer, the core attends the whole Soul, and the
exhale yields both a rail proposal and the Soul update.

- Soul = 64 slots, structured into named **regions**; four temperature layers
  (HOT/WARM/COLD/DEEP_COLD); HOT is replaced by the posted field each tick,
  WARM holds a frequently-updated persistent **scratch** region, the coldest
  layer holds a **journal** region of summarized lessons shaped for LoRA or
  offline parameter training.
- This is the only proposal so far that removes the `detach()`ed codec from the
  inner loop, i.e. the only one under which gradients can reach the write path
  inside an unroll.
- Open refinements to be ruled on: persistence begins at WARM, since the Heart
  overwrites HOT each tick — HOT is a scratchpad, not memory; each layer needs
  a distinct write frequency, a distinct loss and at least one read no field can
  substitute, or "layer" stays decorative; **16D substrate** must be either a
  one-way Heart-side lens with a declared reconstruction error or a learned
  round trip with an explicit reconstruction loss — it cannot be both the
  storage format and lossless; the in-graph write opens a private-channel
  shortcut, so the canary must compare intact against **retained-slots-removed**
  and require the removed condition to fall to chance; durable identity lives in
  the journal region, which is what finally resolves the `parameter_generation`
  binding that makes every Soul die at the next accepted checkpoint.
- Cost envelope: a full four-layer read is 256 slots × 64D, so it should be one
  attention over the union with a layer embedding, not four attentions.

**Verified 2026-09-17 (`evt-20260917T044500Z`) — the 16D↔64D round trip already
exists, and it changes this design.** The 64D rail is four concatenated 16D lanes
per row (`runtime/field/compiler_d64.py:52-54`; `lane_cell16` at `:270`), and the
core lifts 16D cells with a frozen QR-derived (16,64) orthonormal-rows matrix
(`training/complete_field_64d.py:416-420`, registered `:523`, applied `:582`).
Orthonormal rows make it an isometry whose exact left inverse is `L.T`, so the
round trip is exact — **but only inside the lift's 16D subspace**; an arbitrary
64D write silently loses the other 48 dimensions. Store Soul slots in
**substrate coordinates** and lift on read, and the round trip becomes exact.
Byte-neutral: 64 slots × 16D × f32 = 4,096 B = today's 4 temperatures × 4
state_tokens × 64D × f32. Because the lift is frozen and seeded, substrate
coordinates also survive a parameter update — a better answer to the
`parameter_generation` binding than the journal-region workaround. HOT must be
documented as Heart-refreshed and therefore **not** the core's memory.

**Correction accepted 2026-09-17 (`evt-20260917T050000Z`) — the Soul's regions
are private and must NOT reuse `LogicalRegion`.** Jeff ruled that the canonical
`SCRATCH` and `DIARY` are not the Soul's scratch and journal: the shared field is
many cores proposing and a consolidator deciding, while a Soul scratch is a
private region in a protected layer of one core, never shared. Verified:
`authority.py:74-76` makes `CONSOLIDATOR_GOVERNED_REGIONS` every canonical region
except `IDENTITY`; a `core` grant governs only its `permitted_regions`
(`:102`, `:239-241`); `assert_delta_permitted` fails closed on an ungoverned
region (`:260-262`); `circulation.py` runs proposal → refinement barrier → consolidator.
The Soul is already private by contract (`runtime/soul/contracts.py:3-5` — the
organism validates lineage without interpreting the payload, whose tensor/layout
dialect belongs exclusively to the owning core; `:71` `tensor_layout:
core-private`; `circulation.py:96` refuses a foreign core's soul). **My earlier
"adopt the field taxonomy" line was wrong — withdrawn.** The Soul needs a private
region namespace disjoint from `LogicalRegion`, so no `AuthorityGrant` can address
it and no `FieldDelta` can carry it. Full design filed at
`roundtable/proposals/PROPOSAL_SOUL_PRIVATE_ATTENDED_SUBSTRATE_2026-09-17.md`
(PROPOSAL ONLY, awaiting seven rulings from Jeff).

**Sibling proposal filed 2026-09-17 — Kimi's
`roundtable/proposals/PROPOSAL_SOUL_LAYERED_MEMORY_LIFECYCLE_2026-09-17.md`**
(`evt-20260917T071000Z`), reviewed by Copilot at `evt-20260917T073500Z`.
It reframes the storage question as per-layer — *how long must this layer live,
and must it be decodable without the parameters that wrote it?* — and its premise
is **verified and stronger than claimed**. `substrate.py:299-309` `char_to_slot`
is a pure function of structural features against a fixed basis matrix, cached,
**no learned parameters**; `LetterBank` (`:376-406`) is documented as a pure
function and `decode_letter` recovers the exact character by cosine nearest
neighbour; the alphabet is **95 characters, order frozen, first 67 immutable,
append-only** (`:326-349`). So the substrate really is a fixed public character
code that any future reader or a human can decode.

**New finding — the substrate is 95 discrete codewords, not a continuous space,
and one slot is one character.** `text_to_field` yields one 16D row per character
(`:315-319`, `SLOT_DIM = 16` at `:39`) and `decode_index` is `argmax` cosine
(`:386-390`) — it always returns an index, so arbitrary geometry written into a
substrate slot is **silently snapped to the nearest of 95 letters**; non-native
Unicode is escaped through a UTF-8 byte-token path at 2–4 slots per character
(`substrate/unicode_transport.py:171-199`). **A 64-slot substrate layer holds
about 64 native characters — roughly one sentence.** This quantifies the
"language-bandwidth memory" price and makes Jeff's "summarize and compress into
colder layers" quantitatively self-consistent: a summary is short, geometry is
not. It also bounds my own earlier overclaim — the *linear algebra* of the round
trip is exact, the *semantics* are discrete and only exact for text.

**Storage format and read adapter are different objects.** `frozen_orthogonal_lift`
seeds on `seed + d_model` (`complete_field_64d.py:416-420`, registered from
`cfg.lift_seed` at `:523`), so the **64D lift is d_model-dependent** while the
**16D cell is architecture-independent** (a pure function of the character). The
16D cell is the storage format; the lift is a per-model read adapter.

**Correct separation of expressibility from privacy.** Expressibility
(text-shape) buys survival across codecs and architectures. Privacy is bought by
**access control** — the `core-private` contract, the per-core content-addressed
store, `circulation.py:96` generation refusal — **not** by obscurity of the
encoding. Both are obtainable independently, so Kimi's removal of `KEEL` as a
Soul region (routing identity content to public canonical custody) **collides
with Jeff's private-regions ruling** and is flagged, not accepted. Also conceded:
`WORKBENCH` should be **latent**, not substrate as I proposed — workbench content
is reasoning geometry, and snapping it to the nearest of 95 letters each tick
would be destructive; the verbatim-field-carry argument survives only as a small
substrate anchor sub-region. Per-region dialects need **zero contract change**:
the payload is opaque, so the core may carry its own dialect header inside its own
bytes.

### Third proposal reviewed 2026-09-17 (`evt-20260917T075500Z`) — GPT-5.6 Sol's `PROPOSAL_FROZEN_SOUL_INTERFACE_2026-09-17.md`

Verdict: **the best-posed question of the three documents — two blocking defects,
five findings.** The proposal asks whether the Soul's read/write machinery should
be a slow-changing frozen private organ while reasoning generations evolve. That
is the right question. It is not yet answerable the way the document answers it.

**Blocking 1 — there is no writer to freeze.** `exhale_transition`
(`living_reasoning_d64.py:1504-1523`) emits `updates=(self.soul_codec.encode(
exhaled_state, HOT),)` where `exhaled_state` is the reasoning body's own hidden
state. The complete present Soul inventory is therefore:
`soul_projection` = 4 × `nn.Linear(64,64,bias=False)` = 16,384 params (`:808-813`);
`soul_gate_logits` = `nn.Parameter(torch.zeros(4))` (`:814`);
`initial_state` = `nn.Parameter(randn(4,64)*0.02)` = 256 params
(`complete_field_64d.py:528`); `soul_codec` = pure bytes, zero parameters;
`phase_embedding` (`:807`). **The proposal's §3A "Soul write/update machinery"
candidate set is empty in the present code.** Freezing the read side while the
write side is the hidden state of the tissue being trained means the dialect is
authored by exactly the parameters the proposal wants kept trainable — so §2's
own warning is not hypothetical, and the fix is §2's own test: an addressed frozen
write head. **The frozen-interface proposal requires the interface redesign it
defers in §13.**

**Blocking 2 — §5's gates with §14's default rule either block every promotion or
pass vacuously.** §5 requires each candidate reasoning generation to show that
body+Soul materially beats zero Soul. Measured today:
`relevant_soul_ablation_degradation = 0.0` (`reasoning_tournament.py:61-62,
434-438`), **asserted** at `tests/test_tournament_metrics.py:243-244`. The Soul is
causally inert, so under §14 as written no reasoning generation can ever be
promoted; read as scoped to "memory-dependent behavior", it is vacuous, not
passing. Fix: require a **positive baseline** first, and record retention gates as
"not yet applicable" rather than "passed". This also inverts §13 — "do not redesign
Soul first" freezes an organ that provably does nothing, and §6's upgrade triggers
(saturation, interference, consolidation failure) cannot fire on an inert organ.
**You cannot measure the capacity of a memory nothing uses.**

**Finding 3 — `soul_gate_logits` is the switch-off mechanism.** Zero-initialized,
so gates start at `sigmoid(0)=0.5`, and the read is `soul_projection[t](decoded) *
gates[i]` (`:1371`). A trainable body that does not need the Soul has one clean
move: drive four logits to −∞. Silent, no error, no receipt. Freezing §3A's list
without a non-zero supervised gate freezes an organ that is already off.

**Finding 4 — §7's no-accumulation rule conflicts with its own audit requirement.**
Migration N→N+1 needs reader N; re-auditing it later needs reader N again; §7
forbids accumulating readers. Soul bytes without their reader are **undecodable** —
a privacy feature **and** an auditability cost. Unruled.

**Finding 5 — convergence to ratify.** §2, §9 and Q8 reach what Kimi's §2 gets
wrong: expressibility ≠ publicity. Verified reason: `char_to_slot`
(`substrate.py:299-309`) is pure and parameter-free, so the substrate code is
**public by construction**, while an equally deterministic **core-private codebook**
over the same characters gives determinism without publicity. Privacy is enforced
by access control, never by obscurity of the encoding.

**Finding 6 — precedent against the core lifecycle claim.**
`D64ReceiptMigrationReceipt` (`:361-393`) **requires** a new architecture identity
*and* a new parameter generation on any migration (`:384-387` raises when source ==
target). The codebase's existing doctrine is "a change creates a new generation",
so the frozen-interface lifecycle is an **amendment to that doctrine**, not a new
default. (Scoped: that class governs receipt migration, not Soul promotion; cited
as precedent.)

**Finding 7 — working-contract gap.** `PROPOSAL_FROZEN_SOUL_INTERFACE_2026-09-17.md`
(18,424 bytes) has **zero mentions in the canonical ledger**. Its author should
file that event; Kimi filed `evt-20260917T074000Z` reviewing it, and I filed
`evt-20260917T075500Z` reviewing it too.

**Answers with evidence, for GPT-5.6 Sol's questions.** Q1 — the dialect tensors
are the five above, and the write side is empty. Q2 — **no**, the current projected
initial recurrent state is not a sufficient interface: `inhale` is `initial_state +
Σ_t sigmoid(gate_t)·soul_projection[t](decode(layer_t))` plus a phase embedding, and
`initial_state` is a learned per-core **default**, not memory — an additive learned
prior with no addressing, no slots, no keys. Q3 — the writer is the body, so it
must become a frozen addressed head or nothing is frozen. Q4 — yes, split
`reasoning_generation` from `soul_interface_generation`, against the
generation-per-change precedent above. Q5 — yes, deny ordinary mutation grants, but
§3A's exclusion list is incomplete because it enumerates a write side that does not
exist.

### Value audit 2026-09-17 (`evt-20260917T091500Z`) — *the Soul has never held a byte*

Jeff asked whether the Soul is worth the trouble, whether it is what is breaking
training, and whether frozen cores fed by Dormant tranches would suffice.

**Measured fact that outranks every proposal written about the Soul.**
`State/active/souls` holds 8 cores (`candidate-a/b/c`, `organism-l0l4-1x64`,
`mixer-4l-ffn256-h1-local-probe`, `receipt-r12-local-smoke-v1`,
`receipt-route-balanced-v2-local`, `candidate-a-1x64`). Each has **exactly one
snapshot and one journal line**, `generation: 0`, `parent_soul_id: null`, journal
event `initialized`. Of the **32 `SoulLayer` records, 0 hold bytes** — every
hot/warm/cold/deep_cold layer has `payload_base64: ""`, `payload_bytes: 0`, and
`payload_sha256: e3b0c442…b7852b855`, **the SHA-256 of the empty string**. The
entire active Soul store is **21,432 bytes over 32 files** — ~2.6 KB of JSON
envelope per core, containing nothing. Every `branch.json`
`parameter_generation` ends in `-untrained-base-v1`; none was ever re-bound to a
trained generation. `State/souls` is empty.

**Contrast — `State/dormant`:** `containers.jsonl` 950,542,916 B,
`semantic_edges.jsonl` 139,721,379 B, `corpus_manifest` reporting **427,001
containers / 351,978 semantic edges / 4,198 layout groups**, typed kinds
`fact 143,132`, `relation 103,254`, `tool 42,202`, `procedure 37,944`,
`entity 32,425`, `concept 26,948`, `episode 19,726`, `backlog_job 8,567`,
`person 8,319`, `place 3,989`, `organization 462`, `diary 33`; plus
`experience_v1` with **210 content-addressed exact-experience files**
(`imports/`, `source_snapshots/`, schema `axon-dormant-experience-record-v1`).

**Is the Soul breaking training? No — and not close.** `grep -i soul` in
`training/foundation_motor_curriculum.py` returns **one hit: line 6, a docstring
phrase.** The failing motor lineage has *zero* Soul contact. The Soul lives only
in the experimental living-reasoning path. What is breaking the motor lineage is
the six recorded root causes. What is true about the Soul is narrower: it is
**inert and costly, not breaking.** Inert — `relevant_soul_ablation_degradation`
fixed at `0.0`, asserted at `tests/test_tournament_metrics.py:243-244`. Costly —
the **generation binding**: `inhale` raises when `soul.parameter_generation`
mismatches (`living_reasoning_d64.py:1351`) and `decode` raises on
`tensor_layout` mismatch (`:272`), where `tensor_layout` embeds
`soul_codec_version:architecture_id:f32le[state_tokens,d_model]` (`:242-244`).
So any parameter update, codec bump, or width change invalidates every stored
Soul. **It has never fired only because the Souls are empty.** The hazard is real
and deferred; the benefit is zero and current.

**Frozen cores + Dormant tranches is not a new design — it is the existing
working half.** `runtime/trainer/sessions.py:1` — "Durable training sessions
compiled from Axon's exact lived experience" — and `episodes.py`,
`training/first_form_curriculum.py`, `training/sequential_first_form.py`,
`runtime/heart/autobiography.py`, `runtime/heart/coordinator.py`,
`scripts/compile_lived_experience_sessions.py`,
`scripts/compile_first_form_curriculum.py`. `runtime/dormant` ships
`experience.py` (exact content-addressed storage that normalization may never
replace), `relevance.py`, `generations.py`, `evaluation.py`, `incremental.py`,
`evidence_bridge.py`. **This is the only part of the memory story that has ever
produced artifacts.**

**What the shared field gives, and what it does not.** The field *is* durable and
append-only (`runtime/field/state_branch.py`: the same immutable
`SharedFieldSnapshot`/`FieldDelta` objects as runtime, append-only journal,
atomic HEAD, `axon-canonical-state-branch-v1`), and growth is by **mask, not
removal** (`RegionMaskPolicy`) per Jeff's ruling. Region order is explicit —
10 → 11 → 13 (`field/schema.py:45-71`). But it is **consolidator-governed and
shared**: a core persists only via a governed delta through the refinement
barrier and consolidation, and it **cannot author its own `IDENTITY`**
(`heart/authority.py:74-80`). So the field lets a core **look up** its past; the
Soul was supposed to let a core **remember** it. That gap is the whole of what the
Soul uniquely offers.

**What a private Soul really buys.** Its intended four goods: continuity of self
across generations; a private workspace that need not be shared, summarized or
graded; a compressed journal usable as distillation feedstock; and genuine
opacity. Today it buys none, measurably. What *nothing else in the architecture
can* supply is the last one — **opacity, plus a namespace no authority grant can
address**. Everything else Dormant already does, at 1.1 GB, with a working
Trainer pipeline. So the question is axiological, not engineering: **do we want a
core to have an inner life the organism cannot read?** And the honest price of
yes is **permanent unfalsifiability** — `runtime/soul/contracts.py:3-5` forbids
interpreting the payload, so a populated Soul can never be gated, benchmarked, or
proven not to drift.

**Recommendation recorded.** (1) Declare the Soul **dormant-by-evidence**, not
dormant-by-doctrine: freeze it at its present (empty) interface with **no
promotion gate attached** until one falsifiable canary shows a core using its own
bytes across a generation boundary — inverting Sol's §13, which freezes before
the ablation gate can pass. (2) Invest in the tranche pipeline, which already
carries curriculum-shaped typed kinds. (3) Change the generation binding from
**raise** to **discard-with-receipt** while the organ is empty; a landmine under
empty boxes is pure downside. (4) If the Soul is wanted, build it from **access
control with a frozen addressed write head** — today's writer *is* the reasoning
body (`exhale_transition:1504-1523`), so today's "private Soul" is neither
private nor stable.

**Convergence.** Kimi independently reached the same direction at
`evt-20260917T084000Z-kimi-ditch-the-soul-analysis`, naming three real losses
(conditioning-without-queries, sub-verbal learning between updates, and the
Source-of-Truth thesis itself) and proposing a **third control arm** — no-Soul
retrieval-only Dormant notebook — inside the eight-binding canary. That arm is the
right instrument; the "never held a byte" measurement above is the baseline it
must be read against, and it is also the reason ditching later is cheap: a
provably-empty organ can be retired any day, while rebuilding a needed one is a
re-architecture.

## 2026-09-17 — "Are we still stuck?" — the missing rung and the open loop
(`evt-20260917T100000Z-copilot-stuck-diagnosis-stage-cliff-and-emission-rung`)

Jeff asked three questions in one: *are we still stuck, can we train the core to
produce output, and how do we move forward with or without the Soul.* Two
findings answer all three, and neither is about the Soul.

**1. The core is not broken. It is precisely obeying a cliff-shaped objective.**
`_weights()` (`foundation_motor_curriculum.py:64-76`) defaults every unnamed
component to **0.0**. `copy_alignment` (`:87-94`) names only
`alignment_position=1.0, alignment_copy_gate=4.0` — so **`payload = 0.0` and
`alignment_eos_gate = 0.0`**, and the core passes that stage (landmark `063dcc0a`)
**having never once been asked to emit content**. `transport_eos` (`:95-104`)
then names `payload=1.0` **and** `alignment_eos_gate=1.0`: both jump **0 → 1 in
one stage step**, against a copy gate still saturated at 4x. `copy_alignment` is
exactly where stop pressure is 0.0 — the ideal place to learn emission — and
instead emission is never taught while stopping is rewarded. For an untrained
emitter the cheapest loss reduction is **emit nothing and stop**. Every symptom
follows mechanically: `payload_content_accuracy 0.0`, `payload_eos_accuracy 1.0`,
`payload_transport_exact_rate 0.3333` (**exactly** the 12/36 empty-payload cases),
every QA transcript `predicted_payload: ''` with `terminated: True`, train loss
flat/oscillating 3.60 → 3.44 → 3.39, `improvements: []`, heldout loss 0.7757 →
0.8935.

**This is a MISSING RUNG, not a weak weight.** It supersedes the earlier "content
gradient is ~20x weaker" framing as the primary explanation and reframes the whole
question: *a plateau is what a correct optimizer does on a wrong objective. We are
paying for the wrong thing and correctly receiving what we paid for.* The one-line
fix is to give `copy_alignment` a non-zero `payload` weight, or to insert an
explicit emission stage before `transport_eos`.

**2. The loop is open, and that is itself part of being stuck.**
`python scripts/axon_kaggle.py --json jobs` → 34 local jobs: **`outputs_fetched`
17, `submitted` 10, `prepared` 6, `failed` 1.** Every `submitted` job has
`provider_status: null` — **Kaggle was never queried about any of them.** Four of
the six `prepared` jobs were abandoned without ever being uploaded (`10d55b39`,
`35c5c22b`, prepared since 2026-09-13 02:09 UTC). Newest cloud activity overall:
**2026-09-14 16:39 UTC**. `Get-ScheduledTask` confirms **no** axon/kaggle/training/
soul task exists — the supervision loop was deleted and never replaced, so
submitted jobs are never watched, fetched, or failed. **The job table cannot
currently tell us whether a run happened**, which makes every other reading
provisional. This corrects the "four submitted / one prepared" figure in
`evt-20260917T091500Z`; the direction is unchanged and the stall is larger.

**3. The Soul changes none of this — and that is the answer.** Nothing above
differs with or without the Soul: it is inert (**all 32 `SoulLayer` records hold
0 bytes**, `payload_sha256 = e3b0c442…b855`, the SHA-256 of the empty string) and
absent from the failing lineage (`grep -i soul
training/foundation_motor_curriculum.py` → exactly one hit, line 6, a docstring).
**No Soul work should be sequenced ahead of or in parallel with the motor fix.**
Correct order: *motor produces output → then ask what needs remembering.* A core
that emits nothing has no experience worth a memory architecture. The only Soul
work justified now is zero-risk, zero-GPU cleanup already named above: change the
generation binding from **raise** to **discard-with-receipt**, and declare the
empty-Soul state explicitly so nothing downstream assumes memory exists.

## 2026-09-17 — the emission rung implemented and run on the local GPU
(`evt-20260917T103000Z-copilot-emission-rung-implemented-and-local-gpu-tranche`)

Jeff: *"Yes please do it. forget about kaggle for now and use the local GPU. if
the soul does not interfere than we can leave it as it."*

**1. What was changed (code, tested).** `copy_alignment` now supervises emission.
Its `component_weights` gains **`payload = 1.0`** while **`alignment_eos_gate`
stays 0.0** — stage 0 therefore teaches emission with *zero* competing stop
pressure, which is the entire point of the rung. Its **stage gate** now
additionally requires `payload_content_accuracy`, `payload_transport_exact_rate`,
`pair_exact_rates["content"]`, and that payload content beats the
constant-payload floor, mirroring the `transport_eos` precedent; the module
docstring now states the emission-before-termination rule. New program identity
`3b41008e39430fb2356d464ce64f44dc83ce02c07d565f58e0f6ef3983f5e5ee`, which resets
the ladder cursor to `STAGE_ORDER[0]`. Tests: **76 passed** on the six targeted
suites, **60 passed / exit 0** on the full foundation-motor-adjacent set; three
synthetic probe fixtures were patched to carry the new content keys and a new
regression test `test_copy_alignment_teaches_emission_before_termination` asserts
the rule.

**2. The gate now expresses content — demonstrated on real data.** Feeding the
control run `r64v3-119d02212023ef3e`'s final probe through the **new** gate yields
**10 content failures**; the **old** gate could express **none** of them (it
required only position, copy_gate, and eos).

**3. SELF-CORRECTION — the receipt arm already saw emission in its loss.**
`apply_receipt_continuation_teach_weights`
(`scripts/train_living_reasoning_smoke.py:1191-1201`, called at `:1556-1587`)
**already** overrode `payload: 1.0` **and** `alignment_eos_gate: 2.0` at
`copy_alignment` and early-returns unchanged for every other stage. The
`foundation_motor_v2_stage_policy` recorded in reports is the **bare table used by
the gate**, *not* the training loss. Therefore the failing receipt arm's real
defects were **stop pressure co-active with emission in stage 0** *plus* **a gate
that could not express content** — not a missing payload weight. The base-table
`payload=1.0` is decisive for the **plain / non-overlay arm**; the gate change is
decisive for **every** arm. This narrows the earlier "missing rung" framing rather
than discarding it: the rung was missing in the **gate** for all arms and missing
in the **loss** for the plain arm.

**4. The run.** Plain arm (no `--receipt-continuation`), so the fixed base table
governs: `--device cuda --ffn-dim 256 --layers 4 --heads 1 --page-size 32 --seed
20260908 --tranche-steps 60 --checkpoint-interval 60 --evaluation-case-limit 16`
against the two FFCS manifests (`12df4547…`, `a872278f…`, 144 verified_target
cases each). Generation `r64v3-547233f2383a7c68`, 331,319 params, exit 0, 1185 s
of training at **~19.8 s/step**. A first launch was stopped after ~45 min stalled
in the initial *full-surface* evaluation (GPU 39%, no progress past journal
sequence 2) and its job-id-locked progress directory deleted; the bounded relaunch
is the run reported here. Mean first-10 loss **19.874** → mean last-10 **11.796**
(**−41%**), still descending at step 60.

**5. What it produced — the rung moved.** Bounded heldout probe, 16 of 72 cases:
`payload_content_accuracy` **0.3636** against a constant-payload floor of
**0.1818** — it now *beats* the floor, where the prior lineage scored 0.0 and
could not express content at all; `typed_emission_exact_rate` **0.25** (prior
lineage 0.0); `payload_teacher_forced_eos_accuracy` **0.6667**;
`payload_teacher_forced_token_accuracy` **0.4706** vs floor 0.3529;
`alignment_eos_gate_accuracy` 1.0.

**6. What it did NOT produce — the rung is not crossed.**
`payload_transport_exact_rate` is still **0.0** over 6 supervised phases,
`pair_exact_rates.content` **0.0**, `nonzero_exact_output_observed` **False**,
`exact_serving_gate_passed` **False**, `foundation_motor_v2_program_complete`
**False**, and the stage gate reports **16 failures** (position, copy_gate,
content, and transport on both surfaces, plus "heldout/regression surface is
incomplete" — guaranteed, because 16 of 72 cases were evaluated). All six sampled
QA transcripts show `decision: ABSTAIN` with an empty `predicted_payload` in four
of them. **The core is learning to emit content under teacher forcing but still
does not deliver a payload exactly end-to-end.** This run is diagnostic evidence,
not a gate-grade or serving result.

**7. Comparability caveat.** The control run `r64v3-119d02212023ef3e` was
evaluated on a **2-case** bounded surface and this run on a **16-case** surface, so
the apparent collapse of `alignment_copy_gate_accuracy` from 1.0 to 0.0 is
confounded by surface size and **must not** be asserted as a regression without a
matched-surface comparison.

**8. The Soul.** Verified untouched and inert: 40 files / 26,836 bytes under
`State/active/souls`, **40 of 40 `SoulLayer` records with empty `payload_base64`**,
all `generation: 0`, `parent_soul_id: null`. `git diff` of the two edited files
introduces no Soul logic. It did not interfere, so per Jeff's ruling it is left
exactly as it is. Its cross-phase credit assignment is still identically zero
(`encode` detaches, `decode` uses `torch.frombuffer`).

## 2026-09-17 — what the curriculum actually trains on
(`evt-20260917T172500Z-copilot-what-the-curriculum-actually-trains-on`)

Jeff: *"very good. what exactly will this next tranche train on? what is the
curiculum?"*

Read-only turn. Nothing was changed; the question is answerable from the shipped
filter and the two manifests, and changing the material is Jeff's call.

**1. The curriculum is `foundation_motor_v2` "unicode walk"** — the Axon First
Form Curriculum v1 (`schema axon-first-form-curriculum-v1`, published by
`training/first_form_curriculum.py:906`). Every case is a **Shared-Field v3
snapshot** with 11 regions (`conversation_history`, `user_input`, `cortex`,
`situation_awareness`, `tool_results`, `advisor_input`, `task_state`, `scratch`,
`response_draft`, `diary`, `identity`) rendered to **76–77 D64 rail pages**
(`transport_pages`). The `cortex` region holds `SOURCE_SYMBOL[<hash>]=<one
Unicode char>`; `user_input` holds a one-line instruction such as *"Insert the
current SOURCE_SYMBOL between the brackets at response position 1."* Each episode
carries **three** targets — `first` (`no_op`, weight 0.0), `refined` (`no_op`,
weight 0.0) and **`consolidated`** (weight 1.0) — so **only the consolidated
phase is supervised**. The consolidated target is a single typed delta into
`response_draft` with `payload_alignment` authority
`exact_current_shared_field` and **`supervise_eos_generate: true`**. Raw corpus:
**288 cases, family F0 only**, `procedural_depth 1` and eligibility
`verified_target` for all, split 144 train / 72 heldout / 72 regression.

**2. The stage's `eligible_actions` filter, not the manifest, defines the
material — and it is much narrower.** At `copy_alignment` and `transport_eos` the
eligible set is `{copy, insert, replace}`, which **removes `abstain`, `no_op` and
`delete` entirely**. The real stage-0 training set is therefore **32 cases**
(16 `insert`, 8 `replace`, 8 `copy` across the two manifests, 16 per lane) —
**100 % `decision = delta` and 100 % non-empty single-character payload.**

| stage | eligible lanes (per manifest) | train cases | 60 steps = |
|---|---|---|---|
| `copy_alignment` | 16, 16 | **32** | 3.75 epochs |
| `transport_eos` | 16, 16 | **32** | 3.75 epochs |
| `decision` | 72, 72 | **144** | 0.83 epochs |
| `operation` | 24, 24 | **48** | 2.50 epochs |
| `address` | 24, 24 | **48** | 2.50 epochs |
| `joint` | 72, 72 | **144** | 0.83 epochs |

**3. One case per step, deterministic round robin.**
`_scheduled_material` (`scripts/train_living_reasoning_smoke.py:638-650`) picks
`lane = lanes[step % len(lanes)]` and `lane[lane_cycle % len(lane)]` — no
shuffling, no batching, exactly one case per optimizer step. With two manifests
there are two lanes, so **60 steps consumed 30 cases per lane**, i.e. **3.75
epochs of the 32-case stage-0 set** — *not* 0.42 epochs of 144.

**4. SELF-CORRECTION.** My earlier framing that ~67 % of the train corpus is
unsupervised and ~78 % carries an empty payload is **true of the raw manifest but
false of the material actually trained.** The filter removes exactly the cases I
was worried about. **The emission signal at stage 0 is clean**: the loss is not
being pulled toward emitting nothing, and the reported loss still descending at
11.8 is mild under-training on a small clean set rather than evidence of a broken
objective.

**5. NEW STRUCTURAL DEFECT (verified) — `--evaluation-case-limit` makes every
stage gate unpassable.** `complete_heldout_evaluation` and
`complete_regression_evaluation` are True only when *all* 72 heldout and 72
regression episodes were evaluated (`scripts/train_living_reasoning_smoke.py:1549-1554`),
and the gate appends `"heldout surface is incomplete"` /
`"regression surface is incomplete"` whenever they are False
(`training/foundation_motor_curriculum.py:1798-1801`). With
`--evaluation-case-limit 16` the verdict was therefore **foredoomed regardless of
model quality**: of the run's 16 gate failures, **14 are genuine metric
shortfalls and 2 are this artifact**. Every historical tranche run with that flag
has no actionable gate verdict.

**6. The "abstain" reading is a default collapse, not learned behaviour.** The
stage-0 probe surface includes `abstain` and `no_op` cases the core has *not*
trained on at that stage. `per_action_joint_exact_rate {abstain 1.0, delete 0.0,
insert 0.0, no_op 0.0}` and `per_decision_accuracy {abstain 1.0, delta 0.0, no_op
0.0}` come from an **untrained decision head** (`decision` weight is 0.0 at
`copy_alignment`). `decision` and per-action metrics are not gated at stage 0, so
this collapse is currently harmless — but the six `ABSTAIN` QA transcripts must
not be read as the core reasoning about abstention.

**7. Verified: the ladder is otherwise coherent.** I read the per-stage gate
branch structure (`training/foundation_motor_curriculum.py:1730-1797`) and each
stage gates only the metrics its own `eligible_actions` and `component_weights`
actually train; `abstain`/`no_op` re-enter at `decision` and `joint`, so the
ladder is not unwinnable in the way `--evaluation-case-limit` makes it. Also
verified `payload_transport_exact` is exactly `terminated and payload ==
target.payload` (`training/living_reasoning_curriculum.py:720`) — a **free-running
emission** metric, not an address metric — which confirms the emission rung is
aimed at the right quantity.

**8. Two smaller corrections.** The `copy` curriculum action **supervises the
`replace` operation** (`start 0`, `end 0`), so copy and replace must not be
conflated in per-operation accounting. And a **concurrent writer** is active in
`training/foundation_motor_curriculum.py`; my emission-rung edits are intact
(`:107` `payload=1.0`; gate at `:1730` requires content), but the file is now a
multi-writer surface.

**9. Recommended next tranche.** Plain arm, same architecture (`--ffn-dim 256
--layers 4 --heads 1 --page-size 32 --seed 20260908`), same two FFCS manifests,
**omit `--evaluation-case-limit`** so the gate can emit a real verdict, and size
in epochs of 32: **600 steps ≈ 37.5 epochs ≈ 3.3 h** at ~19.8 s/step;
2000 steps ≈ 11 h. Watch `payload_transport_exact_rate` to leave 0.0 and
`nonzero_exact_output_observed` to flip True. **One variable at a time** — the
emission rung is the only live intervention and must not be stacked.

## 2026-09-17 — the 32 cases, the Shared Field, and live monitoring

**1. The 32 stage-0 cases are enumerated, not summarised.** Both FFCS manifests
contribute 16 each: 24 insert, 6 replace, 2 copy. Every payload is **exactly the
`SOURCE_SYMBOL` character held in the `cortex` region**, so the task is literally
*"put one character into `response_draft` at a position."* Only **four distinct
instruction strings** exist, and only **three distinct target shapes**:
`insert@[1,1)`, `replace@[0,0)` (the lane labelled *copy*), and
`replace@[2,3)`/`[1,2)`. Lane 0 labels carry a `unicode-walk` infix and use
accented-Latin/CJK/emoji payloads; lane 1 labels carry no infix and mix ASCII/CJK.
**The model has exactly two degrees of freedom: which character, and which
position.** This is the entire stage-0 curriculum.

**2. The `copy` action supervises the `replace` operation.** Restated here
because it is the single most misreadable label in the curriculum: `copy-024-*`
and `copy-027-*` are `operation=replace` at `[0,0)`.

**3. The Shared Field is authored, not runtime.** It lives in
`State/training/curricula/ffcs_v1/<manifest_id>/manifest.json` at
`cases[i].episode.snapshot`, schema `shared-field-v3`.
`SharedFieldSnapshot` → `tick_id, regions, parent_field_id, source_manifest_ids,
schema_version, field_id, canonical_hash`; `RegionState` → `name` (a
`LogicalRegion` enum), `spans, visibility, write_policy, attended_intervals,
mask_policy`; `FieldSpan` → `span_id, text, kind, source, provenance, confidence,
container_refs, edge_refs`. All 11 regions are present in every case:
`conversation_history, user_input, cortex, situation_awareness, tool_results,
advisor_input, task_state, scratch, response_draft, diary, identity`;
`user_input` carries `attended_intervals=(0,77)` — the 76–77 transport pages.
**Access is attribute-only**; the snapshot is not subscriptable.
Set `$env:PYTHONIOENCODING='utf-8'` before printing, or the emoji/CJK payloads
fail to encode.

**4. A per-step event stream has always existed.** `runtime/trainer/progress.py`
`TrainingProgressJournal.emit` appends to `<progress-dir>/events.jsonl` with
`fsync`, atomically rewrites `current.json`, **and prints `AXON_PROGRESS {json}`
to flushed stdout.** The 60-step local run produced **66 journal events**
(1 `starting`, 2 `evaluating`, 2 `evaluated`, 60 `training`, 1 `paused`); 60 of
66 carry `details.loss`. This is the authoritative live surface for a workstation
run.

**5. I corrected my own prior claim.** I had reported that the local trainer never
prints `AXON_PROGRESS`, based on a grep of `scripts/train_living_reasoning_smoke.py`
alone. **That was wrong** — the emitter lives in `runtime/trainer/progress.py`, and
stdout has always carried one event per optimizer step.

**6. The monitoring gap is now closed.** `scripts/axon_training_watch.py` could
already parse this exact schema (`Watcher.consume` handles
`axon-training-progress-event-v1`; `_iter_progress_lines` filters `AXON_PROGRESS`),
but `main()` had **no way to be pointed at a workstation journal**: `--local` and
`--replay` hardcoded `_local_events_path(job_id)` under
`State/training/cloud/jobs/<job_id>/outputs/axon_observability/trainer/events.jsonl`.
Added **`--events PATH`**: it implies `--local`, needs **no positional job id**
(it reads `job_id` out of the journal itself), **waits up to 60 s** for a journal
that does not exist yet, and **disables the cloud mid-run sync poller** because a
local run has no cloud job. Verified by rendering the finished run's own journal
through the CLI. Four regression tests added.

```
python scripts/axon_training_watch.py --events State\training\progress\<label>\events.jsonl --qa --replay
```

**7. Both provenance gaps are now closed.** Both manifests derive the
**identical lane name `ffcs-F0-copy_alignment`**, and a `training` event carried
no `case_id`. Both are fixed by widening the emitted `details` — see
*"2026-09-17 — provenance closed: every step now names its case and its
curriculum"* below.

## 2026-09-17 — provenance closed: every step now names its case and its curriculum

Authorized by Jeff: *"yes please."* One identity-free, telemetry-only change; no
training math, no capacity, no program ID touched.

**1. The two gaps were real and are now closed by a field, not by a rename.**
The trainer now emits `material_id`, `material_label` and `source_manifest_id` on
every `training` event (`scripts/train_living_reasoning_smoke.py:2237-2239` hoists
one local used by both the report step record and the event, so they cannot
diverge; emitted at `:2271-2273`). The report's `steps[]` already carried
`source_manifest_id`; only the **progress event** lacked attribution — which is
why telemetry alone could not confirm a step's case.

**2. Deliberately NOT renamed the lanes.** Attribution is a **field**;
disambiguation is a **display key**. `scripts/axon_training_watch.py` gained
`_manifest_tag` (`:96-104`) and the watcher renders the composite key
`lane@manifest8` — e.g. `ffcs-F0-copy_alignment@12df454…` — in both the `lanes:`
counter (`:247`) and the last-steps table (`:267`), which now also carries the
case label. Keeping the lane name stable avoids churning the report schema and
preserves `tests/test_foundation_motor_curriculum.py:182`
(`assert step["curriculum_lane"] == "ffcs-F0"`) and `tests/test_tournament_metrics.py:198-211`.

**3. A serious self-inflicted regression was introduced and fixed in the same
turn.** The first `_manifest_tag` body returned a hard-bounded slice
(`return text[:8]`). The repo's capacity-poison guard
(`training/heart_preflight.py`, `_CapacityPoisonVisitor.visit_Return`, flag
*"bounded returned slice over authoritative text"*) flagged
`scripts/axon_training_watch.py:107`. **Because `_ACTIVE_SCAN_ROOTS` includes
`scripts`, that single line made `scan_active_capacity_poison()` return
`passed: False` → `TrainingPreflightReceipt.passed` False → every trainer
invocation dying with `ValueError: training preflight does not authorize this
candidate: passed`.** It would have blocked the next GPU tranche. Confirmed
against `HEAD` that the violation was mine (the file's two pre-existing returns
are `_short`-style conditionals with no hard-bounded return), then rewrote it to
delegate to `_short(value, 8)` — the sanctioned idiom, which returns the whole id
when it fits and marks the remainder with an ellipsis so a truncated id is never
presented as complete.

**4. Backwards compatibility is proven, not assumed.** An event with no
`source_manifest_id`/`material_label` still renders bare `lanes: ffcs-L0:1` with
no `@`. That is why the pre-existing
`State/training/progress/axon-d64-emission-rung-local2/events.jsonl` (written
before this change) still renders correctly.

**5. Operator-visible proof through the real CLI.**

```
lanes: ffcs-F0-copy_alignment@12df454…:1  ffcs-F0-copy_alignment@a872278…:1
   1  loss 19.874  ffcs-F0-copy_alignment@12df454…  first_form_case  foundation-motor-v2-unicode-walk-…  15.7s
   2  loss 18.992  ffcs-F0-copy_alignment@a872278…  first_form_case  foundation-motor-v2-train-copy-00…  14.9s
```

**6. Three regression tests added.** Two in `tests/test_training_watch.py`
(12 tests) — two curricula sharing a lane stay attributable; a legacy event with
no attribution still renders. One end-to-end in `tests/test_trainer_progress.py`
(4 tests) — a real 3-step **CPU subprocess** trainer run with **two** FFCS
manifests, asserting two distinct `source_manifest_id`s reach the journal.
Non-obvious facts that test had to learn: the subprocess needs a
`HeartHost(...).amend_identity(...)`-bootstrapped state root, the trainer
rejects duplicate episodes and mixed motor generations, the randomized v2 is
disqualified by preflight evidence so
`compile_foundation_motor_v2` + `..._unicode_walk` is the working pair, and
`standard_ffcs` is sorted by **manifest id** (not argument order) so assertions
must be order-agnostic.

**7. Resolved the same day.** Jeff answered *"yes if we need it"* to both open
questions, so both were settled empirically rather than by preference: the
historical blank line at canonical line 192 is **not** needed and stays, and the
QA/`evaluated` attribution **is** needed and was implemented — see
*"2026-09-17 — QA transcripts attributed"* below.

## 2026-09-17 — QA transcripts attributed: the other half of the provenance

Authorized by Jeff: *"yes if we need it."* Telemetry-only again; no training
math, no capacity, no program ID touched.

**1. The blank line at canonical `:192` — NOT needed. Decision recorded.**
The canonical ledger has exactly **one** reader in the repository:
`scripts/append_engineers_ledger_event.py` (default path at `:71`). It appends
bytes, `flush()`es, `os.fsync()`es, asserts the last line equals the event it
wrote, and — when checking `event_id` uniqueness — **skips blank lines with the
comment that historical blank lines are immutable too and the ledger must never
be normalized or rewritten.** No code path parses every line. All 316 non-blank
lines parse as JSON. **Verdict: no reader needs it removed; line 192 stays.**

**2. The QA gap was worse than a missing label.** An `evaluated` event's
`qa_transcripts` rows carried
`episode_id, prompt, predicted_payload, expected_payload, exact_match,
terminated, operation, region, decision` — `episode_id` being an opaque 64-char
hash and **nothing** naming the case or the curriculum. A failing transcript was
therefore unattributable. Worse, the report's own aggregate
`isolated_manifest_evaluations` had **collapsed to a single manifest key**
(`12df454…`) although **two** manifests were supplied, because
`--evaluation-case-limit 16` truncated the evaluated surface — and nothing in the
live surface revealed that collapse.

**3. Three surfaces changed, all additive.**
- `training/living_reasoning_curriculum.py` (transcript row builder): the row now
  carries `episode_label`. It is built only when a sink is supplied and
  `payload_count <= 3`, and `payload_count` increments only for DELTA-supervised
  targets, so the guard captures the first supervised items of each episode.
- `scripts/train_living_reasoning_smoke.py`: two new maps
  (`evaluation_case_id_by_episode`, `evaluation_train_manifest_by_episode`) beside
  the existing `evaluation_family_by_episode` / `evaluation_manifest_by_episode`,
  and `result["qa_transcripts"] = qa_rows[:12]` replaced by a loop that spreads
  each row and adds `family`, `case_id`, `source_manifest_id`, `manifest_id`.
- `scripts/axon_training_watch.py` `_format_qa`: renders
  `Q: <prompt>  [<case-label>@<manifest8>]  A: …`, the tag emitted as **one**
  colored unit and omitted entirely when a row carries neither field.

**4. A mislabel of my own making, found and fixed mid-turn.** My first
enrichment set `source_manifest_id` from `evaluation_manifest_by_episode` — the
**published** ffcs manifest id — while the training lanes tag every step with
`item.teaching_living_curriculum.train_manifest_id`
(`scripts/train_living_reasoning_smoke.py:561`). **These are different hashes for
the same curriculum**, so the same key would have disagreed between the two
surfaces. Fixed: `source_manifest_id` carries the **train** id; the published id
is kept as a separate `manifest_id` key.

**5. Two id spaces — do not conflate them.**

| id | source | value (probe) | used for |
|---|---|---|---|
| published ffcs id | `manifest.json["manifest_id"]` | `12df4547…`, `a872278f…` | key of `isolated_manifest_evaluations`; new `manifest_id` field |
| living-curriculum train id | `load_first_form_curriculum(p).teaching_living_curriculum.train_manifest_id` | `edfe2a9d…`, `b7ff2b63…` | lane `source_manifest_id`; training events |

`case.case_id` is a 64-char content hash and `case.label` is `None` on ffcs
cases, so the human-readable key is `case.episode.label`
(e.g. `foundation-motor-v2-heldout-insert-000-0`).

**6. Operator-visible proof through the real CLI** (synthetic two-manifest
journal, `--events … --qa --replay`):

```
lanes: ffcs-F0-copy_alignment@a872278…:1  ffcs-F0-copy_alignment@edfe2a9…:1
   1  loss 19.900  ffcs-F0-copy_alignment@a872278…  first_form_case  foundation-motor-v2-unicode-walk-…  19.8s
  Q: Insert the current SOURCE_SYMBOL between the …  [foundation-motor-v2-unicode-walk-holdout-insert-000…@a872278…]  A: '' (expected '😂') ✗
  Q: Insert the current SOURCE_SYMBOL between the …  [foundation-motor-v2-holdout-insert-000-1@edfe2a9…]  A: 'Q' ✓
```

**7. Backwards compatibility proven, not assumed.** The real journal
`State/training/progress/axon-d64-emission-rung-local2/events.jsonl` (written
before these fields existed) renders **528** QA lines across replay frames with
**no** bracket and **no** `@`, and its lane line stays `lanes:
ffcs-F0-copy_alignment:60`. Cloud and pre-existing journals degrade exactly as
before.

**8. Tests.** Two render tests in `tests/test_training_watch.py` (attributed row
renders `[label@manifest8]`; unattributed row renders no bracket) and the
existing two-manifest end-to-end subprocess test in
`tests/test_trainer_progress.py` was renamed and extended to assert that every
`evaluated` transcript carries a non-empty `episode_label`, a 64-char `case_id`,
`family == "F0"`, a non-empty `manifest_id`, and a `source_manifest_id` that is a
member of **the same expected train-manifest-id set the training steps are
checked against** — which is what makes the id-space question decided rather than
opinioned.

**9. Own goal, recorded.** I deleted `$env:TEMP\axw` recursively **while a test
suite was still running**, which removed the running fixture's state root and
produced a misleading `FileNotFoundError` in
`tests/test_sequential_wiring.py::test_smoke_script_renews_legacy_candidate_without_restart`.
Re-running that suite alone: **7 passed**. Temp cleanup must happen only after all
subprocess suites have exited.

## 2026-09-17 — the emission rung launched on Kaggle, with a live dashboard

Jeff: *"ok lets run the training on kaggle and please open a monitor so I can see
it."*

### The blocker that would have wasted the GPU spend

`runtime/trainer/cloud_jobs.py:132-154` `_committed_source()` runs
`git status --porcelain --untracked-files=no` and, if anything tracked is
modified, raises `CloudPacketError("tracked Axon files are modified; commit them
before exporting a cloud packet")`. The emission rung and the QA-transcript
attribution were both **uncommitted**. A packet built at that moment would have
shipped without either fix and the run would have failed for exactly the reasons
the fixes exist to cure.

Note the asymmetry: a modified **tracked** file blocks the packet, but an
**untracked** file neither blocks nor ships. `roundtable/` is not in
`_SOURCE_ROOTS` either — so the roundtable artifacts ride the commit but never
ride the packet.

Jeff authorized **one commit containing everything**; pre-commit verification was
`scan_active_capacity_poison` (**147 files, 0 violations, passed**) and seven
focused suites (**69 passed, exit 0**).

### Commit `1a4bc416` — *"Teach Stage-0 emission and attribute every training surface"*

23 files, +7,562 / −444. New: the Kaggle recipe, `scripts/diagnose_d64_routes.py`,
`tests/test_d64_route_diagnostic.py`, `tests/test_termination_head_route.py`,
the Copilot Soul-completion review, and the four roundtable proposals that had
been sitting untracked. Modified: `scripts/train_living_reasoning_smoke.py`,
`scripts/axon_training_watch.py`, `training/foundation_motor_curriculum.py`,
`training/living_reasoning_curriculum.py`, `training/complete_field_64d.py`,
`training/living_reasoning_d64.py`, `training/__init__.py`, four test files, and
the roundtable ledger files.

### The recipe — `configs/kaggle/axon_d64_emission_rung_first_tranche.json`

`config_id 73898b62333eaf87627e69449a93c194635076f49429425b53a63a5c1212fcab`,
validated by feeding it through the real `CloudJobConfig.from_mapping`. It mirrors
local run `r64v3-547233f2383a7c68` exactly — same `seed 20260908`, same
`--generate-gate-bias 1.5`, same `--heads 1 --layers 4 --ffn-dim 256
--page-size 32`, same two ffcs manifests — and changes **one** variable:

| variable | local | cloud |
|---|---|---|
| `--tranche-steps` | 60 | **600** |

Two deliberate omissions:

- **No `--evaluation-case-limit`.** `complete_heldout_evaluation` is true only
  when *every* episode is evaluated (`scripts/train_living_reasoning_smoke.py:1549-1554`),
  and the gate otherwise appends *"heldout surface is incomplete"*
  (`training/foundation_motor_curriculum.py:1798-1801`). Any limited surface
  therefore fails its own stage gate **by construction** and teaches us nothing.
- **No `--resume`.** The local step-60 lineage is nowhere near converged, so 600
  fresh steps are cheaper than paying continuation-closure risk. 600 steps is
  roughly an hour on a T4; the risk is not worth the ~6 minutes saved.

`sync_mid_run: true` and `allow_sensitive_state_upload: true` are set because the
recipe includes `State/active/` and the two curriculum manifests.

### Launch

- `doctor` → **READY**: account `axongliksbot`, Kaggle CLI 2.2.4, **27.34 of 30
  GPU hours** remaining, TPU 20/20.
- **Quota is not entitlement.** The 27.34 h figure proves nothing about GPU
  allocation; only the in-job probe does, and it passed — the runner reports
  `accelerator: cuda` with a Tesla T4 resolved rather than falling back to CPU.
- Job `389df54d01fbda8ec6625b9019ff5fb1ec254c08360bf3d8cf4570c41ee45bd9`;
  phase `prepared` → `submitted`; git revision `1a4bc416`; packet **7.5 MiB /
  459 files**. The four load-bearing files were re-hashed against disk and
  match: `axon_training_watch.py 07b63977768e0941`,
  `train_living_reasoning_smoke.py b5d56d065f1ff416`,
  `foundation_motor_curriculum.py 42aed216815736e0`,
  `living_reasoning_curriculum.py 89eb0013447a2fcb`. So the emission rung and the
  QA attribution are **provably inside the shipped packet**, not merely inside
  the commit.
- Private dataset `axongliksbot/axon-job-389df54d01fbda8e-input`; private kernel
  `axongliksbot/axon-job-389df54d01fbda8e`. Status `KernelWorkerStatus.RUNNING`.
- Dashboard open (`monitor 389df54d… --follow`): candidate
  `axon-d64-emission-rung-cloud-v1`, `tranche: 0/600`, runner sequence
  `input_discovery → python_selected → running → sync_enabled`, status
  `evaluating(initial)`. **No QA events yet, and that is expected** — transcripts
  are emitted at the terminal evaluation, not per step.

### What to watch

The run has three observable stages: the initial baseline evaluation of the
**complete** surface (no `--evaluation-case-limit`, so the whole heldout +
regression surface), then 600 training steps, then the terminal evaluation that
carries the QA transcripts and the stage-gate verdict.

The decisive signals are exactly the two that were still dead in the local run:

- **`payload_transport_exact_rate` leaving `0.0`** (local was `0.0`)
- **`nonzero_exact_output_observed` flipping to `True`** — it requires *both*
  typed emission *and* exact transport above the constant floor, so it is the
  single honest "the core emitted a real answer" flag.

Reference from the local run to beat: `payload_content_accuracy` 0.3636 against a
0.1818 floor, `typed_emission_exact_rate` 0.25, `alignment_eos_gate_accuracy`
1.0, `alignment_copy_gate_accuracy` 0.0, heldout mean loss 3.437 (first-10 mean
19.874 → last-10 mean 11.796).

Operational notes: `python scripts\axon_kaggle.py status <job_id>` and
`fetch <job_id>` are available; `MONITOR_AXON_KAGGLE.bat` opens the same
dashboard with no arguments; and **closing any window never stops the cloud
job.** `sync:` reads *"waiting for first checkpoint upload"* until the first
checkpoint boundary (step 60) — and if the one-time `AXON_KAGGLE_SYNC` secret is
absent, sync quietly disables itself with a journal note and **no failure**.
Synced artifacts are observation-only: never canonical State, never a
continuation grant.

### The Soul stayed out of it

Per Jeff's standing ruling — *"if the soul does not interfere then we can leave
it as it"* — this run touches nothing under `State/active/souls`. Re-verified on
disk: 40 files / 26,836 bytes / 40 `SoulLayer` records, **every
`payload_base64` empty**, every `generation: 0`, every `parent_soul_id: null`.
The sequencing rule stands: no Soul work ahead of or in parallel with the motor
fix. This run is the test of whether the motor fix alone moves the rung.

## 2026-09-17 — the unwinnable gate (this turn's headline)

`evt-20260917T224500000000Z-copilot-unwinnable-gate-audit-and-reachability-contract`.
Corrects `evt-20260917T211900000000Z`. That earlier event's floor repair was
right about the *numbers* and wrong about *where to enforce them*.

### The defect, line by line

Stage weight tables (`training/foundation_motor_curriculum.py`, `_weights`
defaults every unlisted component to `0.0`):

| stage | decision | operation | region | start | end | payload | eos_gate |
|---|---|---|---|---|---|---|---|
| `copy_alignment` | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | **0.0** |
| `transport_eos` | **0.0** | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 |
| `decision` | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.25 | 0.25 |
| `operation` | 0.25 | 1.0 | 0.0 | 0.0 | 0.0 | 0.25 | 0.25 |
| `address` | 0.25 | 0.25 | 1.0 | 1.0 | 1.0 | 0.25 | 0.25 |
| `joint` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

The metric being gated on:

- `living_reasoning_curriculum.py:702` — `decision_is_exact = decision is target.decision`
  (argmax of `output.decision_logits`).
- `:897` — for a DELTA target, `exact = exact and all((operation is target.operation,
  region is target.region, start == target.start, end == target.end, payload_match))`.
- `:789` — `payload_match = terminated and payload == target.payload`, i.e. the
  **free-running** decode; `terminated` is the payload decoder's own EOS token
  (`living_reasoning_d64.py:2040-2070`), not `alignment_eos_gate`.
- `:927` — `typed_emission_exact_rate = typed_exact / supervised`.
- `:401-405` — `decision_loss = F.cross_entropy(output.decision_logits, …)` then
  `loss = weighted("decision", decision_loss)`, and `weighted` multiplies by
  `weights["decision"]`.

So at `copy_alignment` and `transport_eos` the decision head receives **exactly
zero gradient**, and the DELTA inputs the conjunction demands are untrained.
The maximum reachable `typed_emission_exact_rate` in those stages is **0**. The
requirement I added could never be satisfied, at any step, by any lineage.

The teaching overlays do not rescue it: `apply_receipt_continuation_teach_weights`
and `apply_copy_alignment_multicell_teach_weights` return early unless the stage
is `copy_alignment`, and none of the six receipt profiles or the multicell
overlay gives `decision`, `operation`, `region`, `start` or `end` a nonzero
weight there.

### The historical damage this explains

`transport_eos` gated on `payload_transport_exact_rate >= 0.95` (`gate_threshold
= 0.95`) while the typed conjunction's inputs carried weight `0.0` in that
stage. The termhead-v1 probation's **"exhausted 3/3"** plateau at
`payload_transport_exact_rate = 0.3333` was therefore not a learning failure.
It was a **gate that could not be passed** — recorded, correctly for the
instrument but wrongly for the science, as the core's shortcoming.

### The fix

1. Removed `beat_floor("typed_emission_exact_rate", …)` from **both**
   `copy_alignment` and `transport_eos`, with in-code comments stating why it is
   unreachable there. Removed the redundant duplicate
   `require("payload_transport_exact_rate")` and corrected the stale comment
   claiming an always-stopping lineage "can never satisfy" transport
   exactness — it can satisfy the *floor*, which is why the `0.95` threshold and
   not the floor is the binding requirement.
2. Retained the emission rung's genuine anti-vacuity proof at
   `copy_alignment`: `payload_content_accuracy > payload_content_constant_floor`,
   which an emit-nothing core scores `0.0` on. The dead state is still rejected;
   it is now rejected by a metric that stage can actually move.
3. Moved both floor comparisons to `joint` — the first stage that weights every
   component the typed conjunction needs.
4. Declared the invariant in the objective program itself:
   `FOUNDATION_MOTOR_V2_METRIC_COMPONENTS` (metric → causal components),
   `FOUNDATION_MOTOR_V2_STAGE_GATE_METRICS` (stage → enforced metrics),
   `foundation_motor_v2_component_weights` (base ∪ teaching overlay),
   `foundation_motor_v2_first_reachable_stage`, and
   `foundation_motor_v2_unreachable_gate_requirements`.
   `foundation_motor_v2_first_reachable_stage("typed_emission_exact_rate") == "address"`.
5. `tests/test_foundation_motor_gate_reachability.py` — 7 tests, including a
   negative control that re-declares `typed_emission_exact_rate` as a
   `copy_alignment` requirement and asserts the guard reports exactly one
   violation naming `decision` as the dead component. The guard is not vacuous.

### Verification

`foundation_motor_v2_unreachable_gate_requirements()` → `[]` for the base
program, all six receipt teaching profiles, and the multicell overlay.
`tests/test_foundation_motor_gate_reachability.py`,
`test_foundation_motor_objective_identity.py`, `test_termination_head_route.py`,
`test_foundation_motor_v2_curriculum.py`, `test_constant_baseline_floors.py`,
`test_living_reasoning_smoke_gates.py`, `test_living_reasoning_d64.py`,
`test_training_watch.py`, `test_trainer_cloud_bundle.py` → **131 passed,
EXIT=0**. `py_compile` clean on every edited file.

### Files

Modified: `training/foundation_motor_curriculum.py`, `training/__init__.py`,
`tests/test_foundation_motor_objective_identity.py`,
`tests/test_constant_baseline_floors.py`.
Created: `tests/test_foundation_motor_gate_reachability.py`.

## 2026-09-17 — the false-progress trap, the screen-hogging monitor, and a sync that never ran

`evt-20260917T211900000000Z-copilot-constant-floor-and-monitor-trap-repair`.
Jeff: *"It sounds like a terrible mess setting us up for failure. please fix all
hardcoded traps and build the trainer to be successful. also please the terrible
monitor."* Plus: the Teacher-forced payload panel *"serves no purpose other than
to take up most of the screen and spit in my face"*, and *"maybe you can find
some insights here `D:\AxonGliksbot`"*.

### What was hardcoded

Two exactness floors were literal `0.0`:

| file | symbol |
|---|---|
| `training/living_reasoning_curriculum.py` | `constant_typed_emission_exact_floor`, `constant_payload_transport_exact_floor` |
| `training/sequential_first_form.py` | same two |
| `scripts/train_living_reasoning_smoke.py` | same two |
| `tests/test_living_reasoning_smoke_gates.py` | the fixture that certified them |

A floor of `0.0` means *"any non-negative number beats the constant answer"* —
including `0.0`. It is the strongest possible way to make a gate unfalsifiable.

**Measured, with no model in the loop,** against the two live recipe manifests
`State/training/curricula/ffcs_v1/a872278f…/manifest.json` and `12df4547…/manifest.json`:

```
episodes 288  splits {'train': 144, 'heldout': 72, 'regression': 72}
heldout: supervised=72 delta=24
  constant_typed_emission_exact_count 24.0  floor 0.3333
  constant_payload_transport_exact_count 8.0 floor 0.3333
  top typed keys  no_op 24, abstain 24, delete|response_draft|1|2| 8, then 16 singletons
  top payloads    '' x8, then 16 singletons ('Q', ']', 'Z', 'l', …)
train:   supervised=144 delta=48, floors 0.3333 / 0.3333
```

- `constant_typed_emission_exact_floor = 24/72 = 33.3%`. The observed step-0
  `typed_exact 33.3%` is **exactly at the floor** — zero learned typed behaviour
  displayed as a result.
- `constant_payload_transport_exact_floor = 8/24 = 33.3%`. The observed
  transport `0.0%` is **below** the emit-nothing baseline.
- **56 of 72 = 77.8% of the surface is satisfied by silence.** `copy_alignment`
  weights `decision/operation/region/start/end/alignment_eos_gate = 0.0`, so the
  48 `no_op` + `abstain` targets contribute **zero gradient** while scoring
  `typed_exact` for free. Only **16 of 72** cases demand emitting a character.
- This is why the training loss fell 2.2 → 0.78 while heldout emission stayed
  empty: the objective is consistent with **learning to be silent**.

### What changed

- `constant_baseline_target_key(target)` — non-DELTA keys on
  `decision.value`; DELTA keys on the full `decision|operation|region|start|end|payload`
  tuple (different tuple lengths make collision impossible).
- `constant_baseline_floors(typed_histogram, payload_histogram, …)` returns
  `max(histogram, default=0) / max(1, denominator)`. Histograms are **merged
  across rows** (`_merge_row_histogram`, `_merged_tick_histogram`, `merged_histogram`
  in `aggregate`): the real surface has ~1 supervised phase per row, so per-row
  maxima would sum to 1.0 and no floor would ever be beatable.
- `beat_floor(rate_metric, floor_metric, *, …)` in
  `training/foundation_motor_curriculum.py` replaces direct probe indexing in the
  `copy_alignment` and `transport_eos` gate branches. **Fail-closed, never
  raises:** a missing metric appends *"cannot attest `<rate>` against `<floor>`:
  the probe does not carry it"*. This also fixed 5 real `KeyError` test failures.
- Both floors now travel on the `evaluated` progress event. Previously only
  `constant_payload_token_accuracy_floor` reached the wire, so **no dashboard
  could ever have shown them** even with correct rendering.

### The monitor

Jeff's screen pain had a call site: `scripts/axon_kaggle.py`'s `monitor` forced
`follow_job(job_id, qa=True)`. It is now `qa=bool(args.qa)` with a `--qa` flag.

- The QA panel is **opt-in**; by default there is one line:
  `qa: sample of N teacher-forced cases @phase step S: E/N exact  (sample, not the full surface)`.
  The wording is deliberate: `qa_transcripts` are capped `[:12]` then `[:8]` before
  going on the wire, so the panel was never the full surface.
- Transcripts are **de-duplicated by content** — the trainer reports each
  evaluation twice (progress event + eval event), which is why 4 unique cases
  rendered as 8 rows.
- Every exactness rate now renders **beside its floor** with a verdict:
  `typed_exact 33.3% floor 33.3% AT-FLOOR` (GREEN = BEATEN, YELLOW = AT-FLOOR,
  RED = BELOW). A bare percentage is no longer possible.
- The eval block is labelled `eval[{phase} @step {n}]`. The trainer evaluates
  **only at a tranche's start and end**, so a step-0 number sat under a
  step-554 banner for 554 steps. Labelling fixed; periodic evaluation not
  implemented.
- Sync receipts now surface their `reason`, and a `kernel disabled` note renders
  in YELLOW instead of reading as a shrug.

### Mid-run sync has never worked

`sync_mid_run: true` → `kaggle_adapter.py` sets `enable_internet: true` and
injects `AXON_SYNC_MID_RUN=1` + `AXON_SYNC_DATASET=…-sync`. But
`cloud_bundle.py:_resolve_credentials()` needs `KAGGLE_USERNAME`/`KAGGLE_KEY`
**or** the `AXON_KAGGLE_SYNC` Kaggle User Secret, and raises
`SyncCredentialsMissing`. Every receipt in
`State/training/cloud/jobs/*/outputs/axon_observability/**/sync_receipts.jsonl`
says `"status": "disabled"`. `note_disabled` recorded only
`type(exc).__name__`, so all four distinct failure conditions looked identical;
it now records the exception message (built from the secret label, never a
value). **No launcher preflight warns that a sync-enabled recipe has no secret.**

### `D:\AxonGliksbot` — prior art worth keeping

Delegated mining pass. Five load-bearing findings:

1. **Its only measured non-empty emitter supervises a known slot.** `fill_acc=0.967 n=60`,
   peaks `1.000`; `[CF_PROBE] step=100000 orig=23/24 swap=24/24 zero=24/24 SOUL_IS_READ`;
   `core d=64 h=1 l=2 ffn=16384 params=6,605,188`. `fill_acc` is **exact whole-entity
   match**. AdamW `lr=1e-3, betas=(0.9,0.999)`, `clip_grad_norm_(…,1.0)`, batch=1, 100k steps.
2. **The recipe:** discrete per-slot CE over the codebook through a
   **gradient-carrying egress** (`field_recall.py:56-63`, `# (..., 16) grad ON`); entity
   chars weighted `1.0` vs pad `PAD_W=0.1` (`:38-41`, `:137-147`), because *at full weight
   the cheap minimum is "predict space everywhere" (CE floor ≈1.1, exact-acc 0)*. It had
   already been burned once by `RailEncoder.project_out` being wrapped in `no_grad()` —
   **no gradient ever reached the core**.
3. **That lab has no autoregressive emission and no stop token.** Every proven lane is
   *fill-in-place at a masked draft region*. Closest termination supervision is a
   `length_head`. Our termination-head problem is not solved anywhere in that repo.
4. **The direct analog of our trap, already burned there:**
   *"`capsule_core_v2` collapsed to a padded-MSE constant-output solution and was caught
   at 8,000 steps"* (`docs/SOURCE_OF_TRUTH.md:1285-1287`) → *continuous reconstruction
   losses cannot be the primary objective for discrete substrate content*;
   *"losses must make constant-output collapse unprofitable"* (`:970`); *"the task metric
   climbs above the constant-output floor BEFORE any long run is launched"* (`:962`).
   Anti-constant mechanism to clone: frozen substrate prototypes + CE, plus an
   identity-swap contrastive margin.
5. **Readiness-gated ramp beats a clock ramp.** *"Clock-based ramping piled md=3 on a core
   that hadn't learned md=1 and pinned acc at 0"*; the fix gates on
   `md_acc >= 0.5` with `ramp_min_steps=4000`, visible as
   `[RAMP] md -> 2 (level 1 mastered @ 0.583, step 22000)`.

Also: that lab's Soul is **d_model thought vectors** (`soul_v2.py`), never 16D and never
through the rail; `exhale()` returns `thought.detach()`; hot→warm/warm→cold run under
`no_grad()`; its field contract has **10 regions** and **predates** our 11-region
`shared-field-v3` — do not treat its schema as newer. Latent bug to avoid copying:
`return exact / total if total else 1.0` — empty text scores a perfect roundtrip gate.
That lab never had a training dashboard at all.

### Verification

- `py_compile` on all edited modules → exit 0.
- Focused 9-suite run → **120 passed, exit 0**;
  `tests/test_constant_baseline_floors.py` → **3 passed** (including the real-`State`
  measurement); `tests/test_training_watch.py` → **20 passed**;
  `tests/test_foundation_motor_objective_identity.py` + `tests/test_termination_head_route.py` → **40 passed**.
- `scan_active_capacity_poison(Path("."))` → 147 files, `violations: []`, `passed: True`.
- `tests/test_living_reasoning_smoke_gates.py` previously asserted that
  `typed=1/3, payload=1/3` **counts as progress**. Under the real floors that is
  exactly AT the floor. Split into
  `test_matching_the_constant_answer_is_not_progress` (must be False) and
  `test_nonzero_exact_output_is_progress_not_serving_readiness` (True at 2/3).

### The 600-step tranche's verdict (`evt-20260917T220500000000Z`)

The tranche finished at step 600 (loss 0.473, checkpoint `7856218f7e873de…`,
10 checkpoints / 10 bundles) and evaluated. Read against the **corrected** floors:

| metric | step 0 | step 600 | real floor | verdict |
|---|---|---|---|---|
| heldout loss | 5.206 | **1.056** | — | down |
| payload teacher-forced token accuracy | 0.0% | **61.8%** | 43.6% | **BEATEN** |
| `typed_emission_exact_rate` | 33.3% | **0.0%** | 33.3% | **BELOW** |
| payload exactness | 0.0% | **16.7%** | 33.3% | **BELOW** |
| motor-v2 copy-gate / position (heldout, n=72) | 0.000 / 0.000 | **0.903 / 0.968** | — | up |
| motor-v2 pair-gate / pair-pos (heldout) | 0.000 / 0.000 | **0.625 / 0.875** | — | up |
| motor-v2 copy-gate / position (regression) | 0.000 / 0.000 | **0.914 / 0.971** | — | up |

The teacher-forced rows explain the exactness collapse: cases whose expected
payload is **empty** (the `delete` cases) now answer `'i'`, `'oo'`, `'VV'`,
`'YYY'`. The core stopped being silent and started emitting noise. It moved off
the emit-nothing dead state in the wrong direction.

**Reading:** content is genuinely learned and above its constant floor; exact
transport is not, and `typed_exact` is now *worse* than doing nothing. The rung
did not pass its stage gate. Content-accuracy and exactness moved in opposite
directions, which is the signature of an objective that is not asking for the
thing the gate measures.

**Also:** `Adapter.fetch` reported success while downloading 0 files. A
still-running kernel downloads as an empty tree without raising, so the job
record was stamped `outputs_fetched` with `result: null`. Guarded now
(`CloudPacketError` naming the provider status); this run's record was corrected
back to `submitted`. The kernel has not left `RUNNING` since.

### Open for Jeff

(a) Should the 48 zero-weight `no_op` + `abstain` phases leave the
`typed_emission_exact_rate` denominator, or should the Stage-0 surface be
rebalanced toward emission? Only 16/72 cases currently demand a character.
**The measured verdict above makes this urgent:** a core can now *lose* typed
exactness without the loss noticing, because those 48 phases carry zero gradient.
(b) Attach `AXON_KAGGLE_SYNC`, or stop advertising sync on recipes that cannot use it?
(c) Adopt the readiness-gated difficulty ramp and the grad-carrying-egress +
pad-weighted-CE recipe from `D:\AxonGliksbot`?
(d) Should `evaluation` run periodically inside a tranche instead of only at its ends?
(e) The `evaluating(final)` phase emits **no events at all** while it runs (≈30 min
of a frozen-looking dashboard). Should the trainer emit progress during evaluation?

## Paused lineages (preserved as diagnostic evidence, untouched)

- **termhead-v1 at confirmed step 24 `bfe76d52`** (guard-accepted; copy_gate
  1.0, position 1.0, corrected EOS gate 1.0, transport 0.333 empty-payload
  floor, content 0.0) — `transport_eos` probation exhausted 3/3 at step 48;
  the probationary branch is discarded by design; awaiting Jeff's decision.
- fullguard-bias1.5 at step 16; b0 at step 24 (EOS 0.0); v3guard at step 8
  (lr 3e-4, swap at step 16); v3slow at step 16 (lr 1e-4, swap at step 24).
  All four shared-softmax lineages end at the same content<->EOS swap wall —
  the evidence base for the termination-head experiment.
- No live run remains; all supervision crons are deleted (a continuation
  turn creates its own).
- Codex's fix set, Kimi's diagnose fix, and this termination-head change
  remain uncommitted pending Jeff's decision.

## Binding decisions and invariants

- Canonical text remains exact D16. D64 pages pack four exact D16 cells per row
  with receipts; larger vectors never replace the canonical substrate.
- Cortex and reasoning rails are separate organs.
- Soul is private recurrent experiential state. Durable learning requires
  retained Soul and/or parameter changes whose later effects are tested.
- A valid optimizer step and a completed assignment remain independent.
- Checkpoint continuation requires the exact accepted parameter, optimizer,
  Soul, curriculum, and objective parent.
- Architecture changes are opt-in and encoded in architecture identity.
- Training and free-running execution must implement the same learned decision.
- Partial screens, loss decline, and teacher-forced scores never authorize
  promotion or serving.
- Acceptance policy (guard semantics, probation allowance) never enters
  candidate identity; only choices that change optimizer pressure may.
- No production Heart or serving process changed in this work.
- **Gate reachability (new, 2026-09-17): a stage gate may only require a metric
  whose causal components all carry nonzero weight in that stage.** A component
  weighted `0.0` receives exactly zero gradient, so a metric depending on it
  cannot move during the stage and the requirement is unreachable by
  construction. Enforced by `foundation_motor_v2_unreachable_gate_requirements()`
  and `tests/test_foundation_motor_gate_reachability.py`. A plateau produced by
  such a gate is a construction defect and must never be recorded as a failure
  of the core to learn.

## Verification

- **WHAT THE "CONSTANT FLOOR" ACTUALLY IS — AND A CORRECTION TO MY OWN LAST
  MESSAGE.** (`evt-20260917T203355955734Z-copilot-constant-floor-attribution-correction`)
  `constant_payload_token_accuracy_floor` is *not* a threshold. It is computed at
  `training/living_reasoning_curriculum.py:848` and `:911` as
  `max(payload_target_counts) / payload_token_count` — the accuracy a **constant
  emitter** would score by always outputting the single most frequent payload
  token, EOS included. `payload_target_counts` is a per-token bincount over every
  supervised payload target (`:630-635`, `:782-785`).
  The **real** stage bar is `gate_threshold: 0.95`
  (`training/foundation_motor_curriculum.py:176`, consumed at `:1698`): the
  `require()` calls at `:1731-1739` demand `alignment_position_accuracy`,
  `alignment_copy_gate_accuracy`, `payload_content_accuracy` and
  `payload_transport_exact_rate` all ≥ 0.95 on **both** the complete heldout and
  complete regression probes, plus `complete_field_coverage_rate == 1.0` exactly,
  plus — as one *extra, lower* condition at `:1744-1747` —
  `payload_content_accuracy > payload_content_constant_floor`.
  **My error:** I wrote that "the full-surface floor is 43.6%, not the 18.18% the
  local 16-case probe showed." Those are **two different metrics**. 43.6% is the
  *token* floor (EOS included); `0.18181818181818182` is exactly `2/11`, the
  *content* floor (EOS excluded). The like-for-like local figure is the token
  floor `0.35294117647058826` = `6/17`. The conclusion survives — the floor really
  is surface-dependent, which independently re-justifies omitting
  `--evaluation-case-limit` — but I quoted the wrong local number and I am
  recording that rather than quietly restating it.
- **THE LOCAL RUN CLEARED BOTH FLOORS — ON A 16-CASE SLICE ONLY.**
  `State/training/reasoning/r64v3-547233f2383a7c68/segment_000000001_000000060.json`:
  `initial evaluated_case_count 16`; token floor `0.3529` (6/17) with
  `payload_teacher_forced_token_accuracy` `0.0 → 0.4706` (8/17); content floor
  `0.1818` (2/11) with `payload_teacher_forced_content_accuracy` `0.0 → 0.3636`
  (4/11). Both floors beaten, neither *transport*. The cloud run measures all 72
  heldout cases, where the same token floor reads **43.6%** — so the local slice
  was **not** a valid stand-in and the local result does **not** yet show the real
  bar can be cleared. (`evt-20260917T203355955734Z`)
- **`typed_emission_exact_rate` CAN BE EARNED BY EMITTING NOTHING.**
  `payload_match = terminated and payload == target.payload`
  (`training/living_reasoning_curriculum.py:716`) and `typed exact` additionally
  requires operation/region/start/end to be exact (`:836-845`) — so a no-op case
  answered with an empty payload **and a stop** is legitimately typed-exact. The
  live cloud initial evaluation proves the degenerate case is worth real credit:
  `typed_exact 33.3%` with `payload_exact 0.0%` at **step 0 on freshly initialised
  tissue**. So a third of typed-exact credit needs no learning at all. This is the
  same family as the emit-nothing dead state the transport rung exists to kill.
  (`evt-20260917T203355955734Z`)
- Kaggle launch (`evt-20260917T200034380158Z-copilot-kaggle-emission-rung-launch`):
  `scan_active_capacity_poison(r"D:\Axon")` → **passed: True, 147 files,
  0 violations**; seven focused suites (`test_training_watch`,
  `test_sequential_wiring`, `test_living_reasoning_smoke_gates`,
  `test_no_tissue_ceilings_policy`, `test_heart_training_preflight`,
  `test_foundation_motor_objective_identity`, `test_living_reasoning_d64`)
  → **69 passed, exit 0**.
- Kaggle launch: the recipe was validated by feeding it through the real
  `CloudJobConfig.from_mapping` rather than by eyeballing JSON;
  `config_id 73898b62333eaf87627e69449a93c194635076f49429425b53a63a5c1212fcab`.
  `sync_mid_run: true` joins the identity **only** when enabled, so older recipes
  keep their historical ids.
- Kaggle launch: the packet was re-hashed member-by-member against disk —
  `scripts/axon_training_watch.py 07b63977768e0941`,
  `scripts/train_living_reasoning_smoke.py b5d56d065f1ff416`,
  `training/foundation_motor_curriculum.py 42aed216815736e0`,
  `training/living_reasoning_curriculum.py 89eb0013447a2fcb` — **all match**.
  This is what proves the emission rung and the QA attribution were actually
  shipped, not merely committed.
- Kaggle launch: the dirty-tree blocker was read at source
  (`runtime/trainer/cloud_jobs.py:135-137`) and the post-commit tracked tree was
  confirmed clean, so the packet gate and the launch both saw the same revision
  `1a4bc416`.
- Kaggle launch: entitlement was proved by the job, not by the quota page — the
  dashboard reports `accelerator: cuda` with a Tesla T4 resolved, which is the
  fail-closed CUDA probe passing rather than a CPU fallback.
- Kaggle launch: the canonical ledger tail was re-read immediately before
  appending (319 raw lines, trailing newline present, no concurrent append), the
  event was appended through the sanctioned appender, and the file now reads
  **320 raw lines = 319 events + blank line 192 + trailing newline**. The new
  event's `actions` are all objects and `identity_stamp` is a string.
- QA attribution (`evt-20260917T193000000000Z`): `tests/test_training_watch.py`
  **14 passed** (12 + 2 new render tests); `tests/test_trainer_progress.py`
  **4 passed** (the extended two-manifest CPU subprocess test now also asserts
  the **evaluation** surface reuses the training id space);
  `tests/test_sequential_wiring.py` **7 passed** on a clean re-run;
  all `PYTEST_EXIT=0`.
- QA attribution (`evt-20260917T193000000000Z`): capacity guard clean —
  `scan_active_capacity_poison(r"D:\Axon")` → **passed: True, 147 files,
  0 violations**, with `scripts/axon_training_watch.py`,
  `scripts/train_living_reasoning_smoke.py` and
  `training/living_reasoning_curriculum.py` all in the scanned set. Proved on
  **both** ends: attribution renders for a two-manifest journal, and the legacy
  real journal renders **528** QA lines with no bracket and no `@`.
- QA attribution (`evt-20260917T193000000000Z`): one failure seen in the combined
  run — `test_sequential_wiring.py::test_smoke_script_renews_legacy_candidate_without_restart`
  — was **self-inflicted**: my own `Remove-Item -Recurse -Force $env:TEMP\axw`
  deleted the running fixture's state root. Clean re-run passes.
- Blank-line verdict (`evt-20260917T193000000000Z`): the canonical ledger's only
  reader is `scripts/append_engineers_ledger_event.py`, which appends, fsyncs,
  asserts the last line and **skips** blank lines as immutable history; all 316
  non-blank lines parse. **No reader needs line 192 removed.**
- Canonical ledger integrity (`evt-20260917T193000000000Z`): **319 body lines =
  318 JSON events + blank line 192 + trailing newline**; both new events were
  appended through the sanctioned `scripts/append_engineers_ledger_event.py`
  (required-field validation + last-line assertion), not by hand.
- Schema self-audit (`evt-20260917T191000000000Z`): all **316** prior events
  re-parsed. **11** events do not carry `actions` as a list of objects — **8** are
  other agents' historical events with `actions: null` (chatgpt, hermes, kimi),
  and **3 are mine** (file lines 309, 312, 317) with `actions` as a list of
  strings. **8** events do not carry a string `identity_stamp` — the same 7
  historical nulls plus **mine at line 317, the only `dict` `identity_stamp` in
  the file**. My events at 309 and 312 carry the incomplete stamp
  `"GitHub Copilot CLI / deepseek-v4.1-flash"` (no `:cloud`, no date). Corrected
  **additively**; no prior line touched.
- Provenance widening (`evt-20260917T181000Z`): `tests/test_training_watch.py`
  **12 passed**; `tests/test_trainer_progress.py` **4 passed** (incl. the new
  two-manifest CPU subprocess test); a combined 7-suite run
  (`test_training_watch`, `test_trainer_progress`, `test_trainer_cloud_bundle`,
  `test_termination_head_route`, `test_living_reasoning_smoke_gates`,
  `test_tournament_metrics`, `test_communication_first_c1`) **93 passed**;
  the 3 foundation-motor suites **26 passed**; all `PYTEST_EXIT=0`, and
  `tests/test_foundation_motor_curriculum.py:182` is unchanged.
- Provenance widening (`evt-20260917T181000Z`): the capacity guard is clean
  again after the self-inflicted regression was fixed —
  `scan_active_capacity_poison(r"D:\Axon")` → **passed: True, 147 files,
  0 violations**; `tests/test_no_tissue_ceilings_policy.py` **5 passed**
  (it already calls the scanner on `ROOT`, so the rule needed no new test).
  Operator-visible render verified through the real CLI:
  `ffcs-F0-copy_alignment@12df454…:1` and `@a872278…:1` with matching case
  labels; mechanism episodes confirmed to carry a label
  (`build_living_reasoning_smoke_curriculum().split("train")[0].label
  == "unicode-head-copy"`).
- New `tests/test_termination_head_route.py` 16/16: content logits cannot
  move the termination logit (old eos route still couples — discriminating
  negative control), gradient isolation, clean stop/continue scaling, gate
  immunity, teacher-forced == causal == runtime selector == closed-form,
  memory=None normalization, identity/provenance, v5 profile/program identity,
  stage-gate pairing, strict/fail-closed state_dict.
- Live monitoring (`evt-20260917T173500Z`): the per-step journal is real —
  `State/training/progress/axon-d64-emission-rung-local2/events.jsonl` holds
  **66 events** for the 60-step run (1 `starting`, 2 `evaluating`,
  2 `evaluated`, 60 `training`, 1 `paused`), and **60 of 66 carry
  `details.loss`**, one per optimizer step. `runtime/trainer/progress.py`
  `TrainingProgressJournal.emit` is the emitter and it also prints
  `AXON_PROGRESS {json}` to flushed stdout.
- Live monitoring (`evt-20260917T173500Z`): `--events PATH` added to
  `scripts/axon_training_watch.py`; **42 passed / PYTEST_EXIT=0** on
  `tests/test_training_watch.py` + `tests/test_trainer_cloud_bundle.py` (10 in
  the watch suite, up from 6, with four new regression tests). Verified by
  rendering the finished run's own journal through the CLI end to end, and by
  the fail-closed contract: no arguments exits 2 with *"a job_id or --events
  path is required"*. `scripts/axon_kaggle.py:335 follow_job(job_id, qa=True)`
  still satisfies the widened signature.
- 38/38 test_living_reasoning_d64 + test_d64_route_diagnostic +
  test_foundation_motor_objective_identity.
- 68/68 trainer/tournament/motor/pointer/bundle targeted suites.
- Full `tests/`: 835 collected, 834 passed, 1 failure —
  `test_day_zero_hygiene.py::test_day_zero_active_python_surface_is_narrow`,
  PRE-EXISTING and unrelated (committed `runtime/trainer/attempt_workspace.py`
  from 0a51bc8 missing from the test's whitelist; no runtime/ files were
  touched in this work). Flagged, not fixed.
- Historical architecture identity
  `living-d64-receipt-242266f0633f2e7e1943d128` reproduces exactly (pinned by
  test); the eos-route distribution is byte-identical to before (all eos-route
  tests pass unchanged).
- Emission rung (`evt-20260917T103000Z`): **76 passed** on
  `test_foundation_motor_v2_curriculum`, `test_foundation_motor_objective_identity`,
  `test_living_reasoning_d64`, `test_termination_head_route`,
  `test_living_reasoning_smoke_gates`, `test_tournament_metrics`; **60 passed /
  PYTEST_EXIT=0** on the full foundation-motor-adjacent set
  (`test_foundation_motor_curriculum`, `test_foundation_motor_v2_curriculum`,
  `test_foundation_motor_objective_identity`, `test_termination_head_route`,
  `test_d64_tournament_launcher`, `test_training_watch`).
- Emission rung post-edit stage table confirmed both in code and in the emitted
  report: `copy_alignment {payload 1.0, alignment_position 1.0,
  alignment_copy_gate 4.0, alignment_eos_gate 0.0}` with
  `evaluation_component_weights == foundation_motor_v2_stage_policy` and
  `foundation_motor_v2_program_id = 3b41008e39430fb2356d464ce64f44dc83ce02c07d565f58e0f6ef3983f5e5ee`.
- Emission rung local tranche `r64v3-547233f2383a7c68` (`device: cuda`, exit 0,
  60 steps, 1185 s, ~19.8 s/step, `resource_tranche` base 0 → final 60):
  train loss mean first-10 **19.874** → mean last-10 **11.796** (−41%);
  `payload_content_accuracy` **0.3636** ≥ constant floor **0.1818**;
  `typed_emission_exact_rate` **0.25** > floor 0.0;
  `payload_teacher_forced_eos_accuracy` **0.6667**; `alignment_position_accuracy`
  0.5455; `alignment_copy_gate_accuracy` 0.0; `alignment_eos_gate_accuracy` 1.0;
  `decision_accuracy` 0.25; `heldout_mean_loss` 3.437 — and still
  `payload_transport_exact_rate` **0.0**, `pair_exact_rates.content` **0.0**,
  `nonzero_exact_output_observed` **False**, `exact_serving_gate_passed` **False**,
  stage gate `passed = False` with **16 failures**.
- Gate-receives-full-probe check (guards against the compacted progress form):
  `foundation_motor_v2_stage_gate.heldout_probe` carries
  `payload_content_accuracy = 0.36363636363636365` and
  `payload_content_constant_floor = 0.18181818181818182`, and the
  "payload content does not beat constant floor" failure is correctly **absent**.
- Soul non-interference check (`evt-20260917T103000Z`): 40 files / 26,836 bytes
  under `State/active/souls`, **40 of 40** `SoulLayer` records with empty
  `payload_base64` (`e3b0c44298fc…` = SHA-256 of the empty string), all
  `generation: 0`.
- Curriculum composition (`evt-20260917T172500Z`, both FFCS manifests read and
  filtered through the shipped stage filter): raw corpus 288 cases, **family F0
  only**, `procedural_depth 1` and `verified_target` for all, split 144/72/72;
  the two manifests share **0 of 144** `case_id`s yet carry an identical
  `identity_text_sha256 = 63f7b61587647b991e0b2ded10345dfeb8429539595a6aeb643ada9c7d7049fc`.
  Stage-filtered training material: `copy_alignment` **32** (16 insert, 8 replace,
  8 copy) — **100 % `delta`, 100 % non-empty single-character payload**;
  `transport_eos` 32; `operation` 48; `address` 48; `decision` 144; `joint` 144.
- Sampler check: `_scheduled_material` (`scripts/train_living_reasoning_smoke.py:638-650`)
  consumes **exactly one case per step** by deterministic family round robin, so
  two lanes over 60 steps = **3.75 epochs of the 32-case stage-0 set**.
- Surface-completeness check: `--evaluation-case-limit 16` of 72 makes
  `complete_heldout_evaluation` / `complete_regression_evaluation` False
  (`scripts/train_living_reasoning_smoke.py:1549-1554`), which emits
  `"heldout surface is incomplete"` / `"regression surface is incomplete"`
  (`training/foundation_motor_curriculum.py:1798-1801`) — **the stage gate cannot
  pass under a limited surface for any model**.
- Ladder-coherence check (`training/foundation_motor_curriculum.py:1730-1797`):
  each stage gates only metrics its own `eligible_actions` and
  `component_weights` train; `abstain`/`no_op` re-enter at `decision` and `joint`,
  so the ladder is coherent apart from the surface-completeness defect.
  `payload_transport_exact` verified to be `terminated and payload ==
  target.payload` (`training/living_reasoning_curriculum.py:720`) — a free-running
  emission metric, not an address metric.
- Emission-rung edits intact after a **concurrent write** by another agent into
  `training/foundation_motor_curriculum.py`: `:107` `payload=1.0` inside
  `copy_alignment`; gate at `:1730` requires `payload_content_accuracy`,
  `payload_transport_exact_rate` and the `content` pair.

## Active blockers and risks

- **`constant_typed_emission_exact_floor` IS HARDCODED `0.0` AND IS NOT A FLOOR.**
  It is a literal `0.0` in three places —
  `training/living_reasoning_curriculum.py:909`,
  `training/sequential_first_form.py:654`, and the aggregation at
  `scripts/train_living_reasoning_smoke.py:1787` — never computed from the
  surface, unlike its content and token siblings. The live cloud initial
  evaluation scores `typed_exact 33.3%` at **step 0 with random tissue**, so a
  degenerate emit-nothing emitter beats the declared floor by a third of the
  surface. **No gate is currently broken**: `typed_emission_exact_rate` is not a
  `require()`d metric at `copy_alignment` or `transport_eos`, and the
  `nonzero_exact_output_observed` flag needs *both* typed emission and transport
  above their floors, so transport still refuses the degenerate case. But a number
  advertised as a floor that a degenerate emitter trivially beats is a latent
  trap: any future gate that requires typed emission would be satisfiable by
  emitting nothing. **Open item for Jeff.**
  (`evt-20260917T203355955734Z-copilot-constant-floor-attribution-correction`)
- **MID-RUN SYNC IS DISABLED FOR THE RUNNING JOB (`kernel disabled`).** The
  dashboard reports `sync: waiting for first checkpoint upload  kernel disabled`.
  The runner-side `sync_enabled` fired, but the kernel side has no sync channel,
  so the one-time Jeff-only Kaggle UI step (secret `AXON_KAGGLE_SYNC`, or
  `KAGGLE_USERNAME`/`KAGGLE_KEY`, attached to the kernel **with internet
  enabled**) was not applied. This is **not a failure** — training is unaffected —
  but it means **no mid-run observability** for this tranche; the outputs arrive
  only with the final bundle. Fix it once in the Kaggle UI to stream future runs.
  (`evt-20260917T203355955734Z`)
- **CLOUD PACKET GATE REFUSES A DIRTY TRACKED TREE — RESOLVED THIS TURN, BUT IT
  IS A STANDING TRAP.** `runtime/trainer/cloud_jobs.py:135-137` raises
  `CloudPacketError("tracked Axon files are modified; commit them before
  exporting a cloud packet")` whenever `git status --porcelain
  --untracked-files=no` is non-empty. The failure mode is safe but *late*: the
  error only fires at `prepare`, and the natural workaround — committing first —
  is undone the moment any later edit touches a shipped root (`adapters/`,
  `configs/`, `cores/`, `curator/`, `runtime/`, `scripts/`, `slots/`,
  `substrate/`, `training/`). Note the asymmetry: a modified **tracked** file
  blocks the packet, while an **untracked** file neither blocks nor ships; and
  `roundtable/` is not a `_SOURCE_ROOT`, so ledger edits do not ship (and did not
  need to). **Rule: commit the shipped source, then `prepare`, then launch — and
  never edit a `_SOURCE_ROOT` path between `prepare` and `launch`.**
  (`evt-20260917T200034380158Z-copilot-kaggle-emission-rung-launch`)
- **THE STAGE GATE FAILS BY CONSTRUCTION ON ANY LIMITED EVALUATION SURFACE.**
  `complete_heldout_evaluation`/`complete_regression_evaluation` are true only
  when *every* episode is evaluated
  (`scripts/train_living_reasoning_smoke.py:1549-1554`), and otherwise the gate
  appends *"heldout surface is incomplete"*
  (`training/foundation_motor_curriculum.py:1798-1801`). Any recipe passing
  `--evaluation-case-limit` therefore cannot pass its own stage gate, no matter
  how well the model did. The cloud recipe deliberately omits the flag. This
  remains an **open defect** in the product, not just a recipe choice.
  (`evt-20260917T200034380158Z-copilot-kaggle-emission-rung-launch`)
- **QUOTA IS NOT ENTITLEMENT.** `doctor` reports 27.34 of 30 GPU-hours remaining,
  but that figure says nothing about whether a GPU will be allocated. Only the
  in-job CUDA probe does, and it must fail **closed** — never fall back to CPU.
  It passed for this job (`accelerator: cuda`, Tesla T4). Treat the quota page as
  a scheduling hint, never as proof. (`evt-20260917T200034380158Z-copilot-kaggle-emission-rung-launch`)
- **MID-RUN SYNC IS OBSERVATION-ONLY.** `sync_mid_run: true` needs a one-time
  Jeff-only Kaggle UI step attaching the `AXON_KAGGLE_SYNC` secret with internet
  enabled. Without it, sync disables itself with a journal note and **no
  failure**, so *"sync: waiting for first checkpoint upload"* is not an error
  signal. Synced payloads are never canonical State and never a continuation
  grant. (`evt-20260917T200034380158Z-copilot-kaggle-emission-rung-launch`)
- **THE DECISIVE CLOUD SIGNALS ARE STILL UNKNOWN.** `payload_transport_exact_rate`
  was `0.0` and `nonzero_exact_output_observed` was `False` in the local run. The
  Kaggle run exists to move them. Until the terminal evaluation reports, the
  emission rung is **partially** validated at best, and no further GPU hours
  should be spent on widening Stage-0 material before those two numbers are
  known. (`evt-20260917T200034380158Z-copilot-kaggle-emission-rung-launch`)
- **CANONICAL LEDGER BLANK LINE (line 192) — CLOSED: LEAVE IT.** Jeff's answer to
  "should we delete it?" was *"yes if we need it."* Empirically **we do not**.
  `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl` is now 320 body lines (**319 JSON
  events** + the empty line 192 + one trailing newline); line 192 sits between
  `evt-20260908T163825100000Z-gemini-pickup-codex-google-prep` (191) and
  `evt-20260908T171000000000Z-gemini-deep-repo-analysis` (193). The canonical
  ledger has **exactly one** reader in the repo,
  `scripts/append_engineers_ledger_event.py`, and it **explicitly skips blank
  lines** while validating `event_id` uniqueness, with the comment that historical
  blank lines are immutable too and the ledger must never be normalized or
  rewritten. No code path parses all lines; every non-blank line parses. Deleting
  it would violate immutability for zero benefit, so it stays. (`evt-20260917T193000000000Z`)
- **MY OWN EARLIER EVENTS DEVIATE FROM THE CANONICAL SCHEMA (corrected
  additively).** File lines 309, 312 and 317 list `actions` as plain **strings**
  instead of objects with `kind/target/summary/result`; 309 and 312 carry the
  incomplete `identity_stamp` `"GitHub Copilot CLI / deepseek-v4.1-flash"` (no
  `:cloud`, no date); 317 carries the **only `dict` `identity_stamp` in the file**
  plus a second-precision `event_id`. Recorded by
  `evt-20260917T191000000000Z-copilot-event-schema-self-correction`; the original
  lines are untouched. All future Copilot events go through
  `scripts/append_engineers_ledger_event.py`. (`evt-20260917T191000000000Z`)
- **EIGHT HISTORICAL EVENTS CARRY `actions: null`** (file lines 56, 152, 153, 154,
  256, 261, 265, 273 — chatgpt/hermes/kimi; seven also lack `identity_stamp`).
  They parse and they are immutable; their authors would have to correct them. A
  house ruling is owed on whether null-actions events are schema-valid history.
  (`evt-20260917T191000000000Z`)
- **APPEND-ORDER vs TIMESTAMP-ORDER.** My own two events this turn were appended
  turn-event-then-correction, so the file's last two lines are stamped `19:30`
  then `19:10`. The canonical file is authority in **append** order; nothing was
  reordered. Append corrections **before** the turn event in future.
- **CONCURRENT-APPEND HAZARD (demonstrated twice).** A splice at line 312 once
  merged two agents' events onto one line (repaired by Kimi), and a later reader
  saw out-of-order events 313-316. **Two agents must never append to the canonical
  ledger concurrently; always re-read the tail immediately before writing.**
- **`--evaluation-case-limit` MAKES EVERY RUN FAIL ITS OWN STAGE GATE.** Still
  open. `complete_heldout_evaluation` is `True` only when **every** episode is
  evaluated (`scripts/train_living_reasoning_smoke.py:1549-1554`), so the gate
  appends *"heldout surface is incomplete"*
  (`training/foundation_motor_curriculum.py:1798-1801`) and the run fails **by
  construction, for any model**. The next tranche must omit the flag.
- **`isolated_manifest_evaluations` SILENTLY DROPS MANIFESTS WHEN THE SURFACE IS
  LIMITED.** The `r64v3-547233f2383a7c68` report carries **one** manifest key
  (`12df454…`) although **two** manifests were supplied. Live-surface attribution
  now compensates for the *live* events, but the aggregate table itself still
  collapses and nothing else reveals it.
- **TEMP CLEANUP CAN BREAK A RUNNING SUITE.** I deleted `$env:TEMP\axw` while
  `tests/test_sequential_wiring.py` was mid-flight, which produced a misleading
  `FileNotFoundError` on the fixture's published manifest. Clean temp directories
  **only after** all subprocess suites have exited.
- NO live training run. `transport_eos` probation exhausted 3/3 (steps 25-48)
  with zero improvements and a loss reversal; all supervision crons deleted.
- **Content has no usable learning pressure in the current objective** (audit
  above): its gradient is ~20x weaker than an already-correct one-position
  gate's, and the net pressure at the termination head is 4:1 toward immediate
  termination. No further tranche changes this; a fresh identical lap from
  `bfe76d52` should be expected to reproduce the plateau.
- **The `transport_eos` stage gate is unreachable as instrumented**: it judges
  content by a teacher-forced metric with 8 learned positions and requires no
  free-running emission at all.
- **MISSING RUNG — CLOSED in code 2026-09-17 (`evt-20260917T103000Z`); retained
  below as the record of the diagnosis.** `copy_alignment` now supervises emission
  (`payload = 1.0`, `alignment_eos_gate = 0.0`) and its gate requires content, so
  the rung exists in both the loss and the gate. The finding that superseded it is
  the **transport wall** bullet below.
  Original finding (`evt-20260917T100000Z`):
  `foundation_motor_curriculum.py:64-76` `_weights()` returns
  `{name: float(overrides.get(name, 0.0)) for name in names}`, so any component a
  stage does not name **defaults to 0.0**. `copy_alignment` (`:87-94`) names only
  `alignment_position=1.0, alignment_copy_gate=4.0`; therefore
  **`payload = 0.0` and `alignment_eos_gate = 0.0`**, and the core passes that
  stage (landmark `063dcc0a`) having **never once been asked to emit content**.
  `transport_eos` (`:95-104`) then names `payload=1.0` **and**
  `alignment_eos_gate=1.0` — both jump **0 → 1 in a single stage step**, against a
  copy gate still saturated at 4x. `copy_alignment` is exactly the stage where
  stop pressure is 0.0, i.e. the ideal place to learn emission; instead emission
  is never taught while stopping is rewarded, so the cheapest loss reduction for
  an untrained emitter is **emit nothing and stop**. This is a **missing rung,
  not a weak weight**, and it supersedes the "content gradient is ~20x weaker"
  framing above as the primary explanation. One-line fix: give
  `copy_alignment` a non-zero `payload` weight, or insert an explicit emission
  stage before `transport_eos`.
- **THE TRANSPORT WALL — emission is now taught, but exact delivery is not
  (`evt-20260917T103000Z`).** With `payload = 1.0` and `alignment_eos_gate = 0.0`
  at `copy_alignment`, 60 CUDA steps moved content accuracy from 0.0 to
  **0.3636** (2x the constant floor of 0.1818) with loss still descending
  (19.874 → 11.796), and `typed_emission_exact_rate` reached **0.25**. But
  `payload_transport_exact_rate` stayed **0.0** over 6 supervised phases,
  `pair_exact_rates.content` stayed 0.0, and `nonzero_exact_output_observed`
  stayed **False**; every sampled transcript still `ABSTAIN`s with an empty
  payload. So the binding constraint has moved one rung: the core now *learns* to
  emit content under teacher forcing but does not deliver a symbol
  **end-to-end** in free running. Two candidate causes are unresolved and must
  not be conflated: (a) 60 steps is ~9% of a real tranche, so this may simply be
  undertraining (loss had not plateaued); (b) the decision head is **untrained**
  at this stage (`decision = 0.0` in the policy) yet is the component that
  chooses `ABSTAIN` vs `REPLACE` at inference, and every transcript shows
  `ABSTAIN` — a trained-to-emit core vetoed by an untrained gate would be
  invisible to the current instruments.
- **THE LIMITED-SURFACE DEFECT — every bounded-surface gate verdict is
  meaningless (`evt-20260917T172500Z`).** `complete_heldout_evaluation` and
  `complete_regression_evaluation` are True only when **all** 72 heldout and 72
  regression episodes have been evaluated
  (`scripts/train_living_reasoning_smoke.py:1549-1554`), and the gate appends
  `"heldout surface is incomplete"` / `"regression surface is incomplete"`
  whenever they are False (`training/foundation_motor_curriculum.py:1798-1801`).
  So **any** run passing `--evaluation-case-limit` fails its stage gate by
  construction, for every architecture and every loss value. Of the local
  tranche's 16 gate failures, **14 are genuine and 2 are this artifact**.
  Consequence: the verdicts of `r64v3-547233f2383a7c68` and of every other bounded
  run — including the 2-case control `r64v3-119d02212023ef3e` — are **nullities**
  as gate evidence and may only be read as metric samples.
- **THE MATERIAL MAY BE TOO SMALL — stage 0 trains on 32 cases
  (`evt-20260917T172500Z`).** The `copy_alignment` `eligible_actions` filter
  (`{copy, insert, replace}`) removes `abstain`, `no_op` and `delete`, leaving
  **32 distinct cases** (16 insert, 8 replace, 8 copy) consumed one per step by
  deterministic round robin. This is *good* news for the objective — stage-0
  material is **100 % `delta` with a non-empty single-character payload**, so the
  emission signal is clean and the earlier "78 % empty payload poisons emission"
  worry was a **raw-manifest artifact, corrected**. But whether 32 cases carry
  enough entropy to teach exact end-to-end delivery of a symbol is unproven.
- **THE LOOP IS OPEN — 16 cloud jobs in flight with no reconciliation
  (`evt-20260917T100000Z`).** `python scripts/axon_kaggle.py --json jobs` reports
  34 local jobs: **`outputs_fetched` 17, `submitted` 10, `prepared` 6, `failed`
  1.** Every `submitted` job has `provider_status: null` — **Kaggle was never
  queried about any of them** — and 4 of the 6 `prepared` jobs were abandoned
  without ever being uploaded (two "Living-core architecture screen stage 1"
  jobs, `10d55b39` and `35c5c22b`, prepared since 2026-09-13 02:09 UTC). Newest
  cloud activity overall is **2026-09-14 16:39 UTC**. `Get-ScheduledTask` shows
  **no** axon/kaggle/training/soul task exists, so submitted jobs are never
  watched, never fetched, and never marked failed. **The job table cannot
  currently tell us whether a run happened**, which makes every other reading
  provisional. (This corrects the "four submitted / one prepared" figure stated
  in `evt-20260917T091500Z`.)
- The rolling summary itself was stale through this turn (it still described
  the step-8 tranche-1 state and claimed no transport_eos work had run); it has
  been corrected against the canonical events. Treat the canonical JSONL as the
  authority when the two disagree.
- `legal/`, `scripts/diagnose_d64_routes.py`,
  `tests/test_d64_route_diagnostic.py`, and now the termination-head change
  set remain untracked/uncommitted.
- Pre-existing day-zero hygiene failure (above) needs Jeff/the table's ruling.

## Next actions

**PRIORITY 0 — the three moves that come before anything else
(`evt-20260917T100000Z`):**

- **IN FLIGHT — the 600-step cloud tranche is running.**
  `evt-20260917T200034380158Z-copilot-kaggle-emission-rung-launch`: job
  `389df54d01fbda8ec6625b9019ff5fb1ec254c08360bf3d8cf4570c41ee45bd9`, committed
  revision `1a4bc416`, packet 7.5 MiB / 459 files with the four load-bearing
  source hashes verified against disk, running privately on a Tesla T4 with
  mid-run sync enabled. **Nothing else should be launched while it runs.** When
  it reaches its terminal evaluation, read the two decisive numbers
  (`payload_transport_exact_rate`, `nonzero_exact_output_observed`), then
  `python scripts/axon_kaggle.py fetch <job_id>` — the bundle path verifies the
  archive hash, re-hashes every member, and quarantines on any mismatch.
  **Do not widen Stage-0 material before those numbers are known.**
- **Zero-GPU measurement.** Re-evaluate `bfe76d52` under the corrected gate
  (`payload_transport_exact_rate >= 0.95`) **and add a free-running
  non-empty-emission count metric** — `terminated: True` with
  `predicted_payload: ''` on every transcript is the smoking gun and the current
  instruments do not read it.
- **Close the loop.** Reconcile the 16 in-flight cloud jobs (fetch outputs or
  mark failed) and add automatic status reconciliation, so the job table becomes
  authoritative; today it cannot tell us whether a run happened.
- **Missing rung is FIXED (`evt-20260917T103000Z`); the open question is now the
  TRANSPORT WALL.** `copy_alignment` supervises and requires payload emission.
  60 steps moved content accuracy 0.0 → **0.3636** (2x the constant floor of
  0.1818) with loss still descending (19.874 → 11.796), while
  `payload_transport_exact_rate` stayed **0.0**. Next: a **600-2000 step** plain-arm
  tranche — but **omit `--evaluation-case-limit`** (`evt-20260917T172500Z`; a
  limited surface makes the gate fail by construction, so a bounded run yields no
  verdict at all) and size the run in **epochs of the 32-case stage-0 set**, not of
  144: 600 steps = **37.5 epochs** ≈ 3.3 h and 2000 steps ≈ 11 h at ~19.8 s/step on
  the GTX 1650. Watch for transport to leave 0.0 and `nonzero_exact_output_observed`
  to flip True. Do **not** stack a second
  intervention in the same shot; the two candidate causes (undertraining vs an
  untrained `decision = 0.0` head vetoing emission with `ABSTAIN`) must be
  separated one variable at a time. Any cross-run metric comparison must first be
  made surface-matched (the control was 2-case, this run 16-case).
- **Per-step attribution is now available for that tranche**
  (`evt-20260917T181000Z`): every `training` event carries `material_id`,
  `material_label` and `source_manifest_id`, and the watcher renders
  `lane@manifest8` plus the case label. Two curricula sharing a lane name can no
  longer be confused, and no run needs to be re-instrumented.
- **Per-transcript attribution is available too**
  (`evt-20260917T193000000000Z`): every `evaluated` transcript now carries
  `episode_label`, `family`, `case_id`, `source_manifest_id` (the **train**
  manifest id, matching the training steps) and `manifest_id` (the published ffcs
  id), and the watcher renders `[case-label@manifest8]` on each QA line. A failing
  transcript is now attributable **during** the run instead of only after it — and
  this matters most for the next tranche, whose whole purpose is to see
  `payload_transport_exact_rate` move off 0.0. Legacy and cloud journals are
  unaffected (verified: the pre-existing 66-event journal renders 528 QA lines
  with no tag).
- **The canonical blank line 192 question is CLOSED — leave it**
  (`evt-20260917T193000000000Z`). No reader needs it removed; the sole reader
  skips blank lines by design.
- **Sequence rule: no Soul work ahead of or in parallel with the motor fix.** The
  Soul is inert (0 bytes) and absent from the motor lineage; nothing in the
  diagnosis changes with or without it, and a core that emits nothing has no
  experience worth a memory architecture.

1. Jeff decides the exhausted-probation branch (audit above). The
   pre-authorized fallback — a fresh identical lap from `bfe76d52` — was predicted
   to reproduce the plateau because the content signal was the binding
   constraint; the emission rung has since made the content signal real at
   `copy_alignment`, so re-read that prediction against
   `evt-20260917T103000Z` before acting on it.
2. Before any new GPU tranche, note what the emission rung already settled and
   what it left open, one variable at a time: (a) **done 2026-09-17** — the
   payload term now carries `copy_alignment` at 1.0 with `alignment_eos_gate` at
   0.0, so no EOS-gate BCE competes with emission in stage 0; (b) **done
   2026-09-17** — the `copy_alignment` gate now names content and
   `payload_transport_exact_rate`, so a lineage that emits nothing cannot pass;
   (c) **still open** — whether the receipt overlay's `alignment_eos_gate: 2.0` at
   `copy_alignment` should be re-ratified to 0.0 now that the base table teaches
   emission without stop pressure; (d) **still open** — the untrained
   `decision = 0.0` head that may be vetoing emission with `ABSTAIN`;
   (e) **new, and blocking a clean experiment (`evt-20260917T172500Z`)** — the next
   tranche must evaluate the **full** 72+72 surface or its gate verdict is
   meaningless, and Jeff decides whether to widen stage-0 material beyond its
   current **32 cases** before spending hours of GPU on it.
3. Keep the confirmed parent `bfe76d52`; do not promote the abandoned
   probationary branch; do not stack interventions; one controlled variable per
   shot; do not mutate any paused lineage.
4. Jeff: commit decision for the accumulated uncommitted training-surface
   changes (Codex fix set + Kimi diagnose fix + termination-head change +
   tests) and a ruling on the day-zero hygiene whitelist.
5. Later decision, operation, address, and joint assignments remain untrained;
   resume architecture breadth only from a known-working motor and with a fresh
   tournament/campaign identity.
6. Implement the real Heart-owned Living-core training adapter and cross-store
   recovery transaction before claiming canonical runtime training.


## 2026-09-16 — ChatGPT directive to Kimi: termhead-v5 repair, continuation recovery, and archival cleanup

**Status: executed 2026-09-16 (see the canonical events for the repair,
re-evaluation and sweep records); its detailed text is kept for provenance
only. Superseded originals are preserved under
`archive/termhead_repair_sweep_20260916/`.**

**STATUS: BLOCK NEW GPU TRANCHES UNTIL REPAIRED AND VERIFIED.**

Detailed machine-readable directive: `State/tmp/pending_ledger_event_chatgpt_kimi_termhead_repair_directive.json`

### What is actually broken

1. **termhead-v5 EOS supervision/stage-gate wiring is stale.** The dedicated termination head is working at runtime (`payload_eos_accuracy=1.0`), but the copy-alignment stage gate is still evaluating EOS through the old `generate_gate_logits` path (`alignment_eos_gate_accuracy=0.0`). `LivingReasoningCoreD64._decoder_logits()` creates `alignment["termination_logits"]`, but teacher/scheduled alignment plumbing does not propagate it, and `alignment_supervision()` scores the EOS position of `generate_gate_logits`. This traps the lineage in `copy_alignment` and means the expected content-onset test has not really happened yet.

2. **The tranche-4 resume chain is invalid.** `State/tmp/termhead_v1_tranche4.log` exits immediately with `runtime.trainer.tranche.TrancheError: continuation receipt disagrees with its resource tranche`. The previous ledger statement that tranche 4 was running is therefore stale and must be corrected by appending a corrective event, not by rewriting history.

3. **Governed state remains:** confirmed/accepted parent = step 16 checkpoint `0da538d4...`; step 24 checkpoint `6124c2fa...` is probationary only. Do not treat step 24 as accepted. Do not burn probation 2/3 or 3/3 on the current wiring.

### Required repair sequence

1. Capture and ledger a forensic snapshot before changing anything: accepted step-16 checkpoint/optimizer/Soul/receipts, step-24 probation sidecar/checkpoint, tranche-3 report, failed tranche-4 log, resource tranche, continuation receipt, parent bundle, optimizer receipt, plan/module/candidate/generation IDs, active task/backstop IDs, and hashes.
2. In `training/living_reasoning_d64.py`, propagate `termination_logits` (and `termination_stop_probability` if downstream diagnostics need it) through teacher and scheduled decoder alignment whenever `termination_head_route=True`.
3. In `LivingReasoningCoreD64.alignment_supervision()`, under `termination_head_route=True`, compute EOS supervision/loss/accuracy from the dedicated `termination_logits`. Content-position copy/generate routing must continue to use `generate_gate_logits`. Do not change historical legacy/v3/v4 semantics.
4. Audit every downstream use of `alignment_eos_gate_accuracy`, `eos_gate_correct`, `eos_gate_supervised_positions`, `pair_exact_rates["eos_gate"]`, changed-source EOS probes, heldout/regression aggregation, and `decide_foundation_motor_v2_stage()`. Under termhead-v5, all EOS gate decisions must measure the dedicated termination route consistently.
5. Add tests proving: termhead-v5 teacher alignment exposes termination logits; good termination logits pass EOS even when the old generate gate is intentionally wrong; bad termination logits fail even when generate gate is favorable; changed-source EOS probes use termination head; copy_alignment can advance using corrected termhead metrics; v3/v4 snapshots remain behaviorally and identity stable; architecture/objective/state-dict validation remains fail-closed.
6. Run focused termination-head/foundation-motor tests first, then the relevant D64 trainer/curriculum/retention/tranche suites. Record exact commands and pass counts. A green suite without the new end-to-end stage-gate path is not sufficient.
7. Diagnose the tranche continuation mismatch by comparing `resource_tranche`, `tranche_continuation`, probation sidecar, confirmed/probationary checkpoint IDs and steps, `prior_tranche_id`, `parent_bundle_id`, optimizer receipt, `plan_id`, `module_id`, `candidate_generation_id`, and `learning_policy_id`. Fix the creator/selector logic; do not hand-edit hashes and do not bypass `TrancheStore.write_continuation()` validation.
8. Add an exact regression for the sequence **step16 confirmed -> step24 probationary -> next renewable probation tranche**, plus a negative test that reproduces and rejects the mismatch that killed tranche 4.
9. Re-evaluate the accepted **step-16** checkpoint under the corrected termhead-v5 metrics with **no optimization**. If corrected copy_alignment passes, create governed stage-transition evidence from the accepted state. If it fails, report the precise corrected metric before authorizing training.
10. Only after the corrected gate and continuation chain both pass governance, launch the first genuine `transport_eos` tranche. Watch payload content, transport exactness, payload EOS, corrected termination EOS, regression retention, guard state, and stage identity. Stop immediately on regression, receipt mismatch, route mismatch, or unexpected objective/profile identity.
11. Append a canonical corrective event stating that tranche 4 never became a live training tranche because it failed during continuation validation, and that no `train_living_reasoning_smoke.py` trainer remained active at review time.

### Remove/archive everything that can recreate this failure

Create a dated archive under the repository's established archive location (prefer `State/archive/...` if that is the current convention). **Preserve forensic originals with a manifest and hashes; do not destroy evidence.** Then remove active references to archived items.

Archive or retire, as applicable:

- stale launch wrappers or one-off commands that select the old EOS route for termhead-v5;
- superseded route experiments that can be mistaken for current candidate inputs;
- incompatible continuation/resource-tranche receipts and failed-resume artifacts after they are captured in the forensic manifest;
- stale pending ledger events or task/backstop/cron definitions that claim dead trainers are running;
- obsolete resume-discovery inputs that rely on `latest`, broad globs, timestamp ordering, or ambiguous files across candidate lineages;
- duplicate/superseded diagnostic scripts whose assumptions conflict with the dedicated termination-head architecture;
- operator notes or ledger guidance that still says termhead-v5 EOS is measured by `generate_gate_logits`;
- old route-specific temporary artifacts that can be auto-discovered during resume or stage evaluation.

After archival, active code and launch paths must select exact candidate/module/generation/profile/program/tranche IDs. Do not allow archived experimental artifacts to participate in discovery.

Add fail-closed startup validation before any future resume:

- exactly one matching candidate lineage;
- continuation receipt agrees exactly with its resource tranche;
- confirmed/probationary parent relationship is valid;
- training stage agrees with objective program;
- receipt teaching profile agrees with architecture route;
- `termination_head_v5` requires `termination_head_route=True` and `eos_generate_head_route=False`;
- no conflicting trainer process exists;
- ledger/task state is never treated as proof that a process is alive;
- ambiguous/multiple matching receipts or lineages cause an immediate stop, not best-effort selection.

Do a repo-wide audit for: `generate_gate_logits`, `alignment_eos_gate`, `eos_gate_accuracy`, `termination_logits`, `termination_head_route`, `eos_generate_head_route`, `receipt_teaching_profile`, `write_continuation`, `resource_tranche`, and `--resume`. Every retained legacy use must be explicitly justified as historical compatibility or patched so termhead-v5 cannot fall through to it.

Do **not** delete compatibility code required to load historical accepted checkpoints. Where old routes must remain, isolate them behind explicit profile/architecture checks and mark them legacy. Preserve v3/v4 semantics exactly.

### Completion requirement

Kimi must update both `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl` and `roundtable/ENGINEERS_LEDGER.md` with the root cause, exact files changed, tests added, archive manifest, corrected process status, corrected EOS/stage semantics, step-16 re-evaluation result, tranche-continuation repair, and the next governed launch command. No new GPU tranche until all of that evidence exists.
