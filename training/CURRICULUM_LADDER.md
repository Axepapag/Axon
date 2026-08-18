# Axon curriculum ladder

This ladder applies independently to the 64D and 128D lines. Weights are
never converted between widths. The current `256 history + 64 user + 64
response` layout is a checkpoint-compatible bootstrap window, not the final
field architecture.

## Gate 0: exact substrate and artifact qualification

A run is blocked unless:

- all 95 supported characters round-trip exactly through the frozen 16D
  substrate and `CharSlotFieldBuilder`;
- unsupported input is rejected and silent target clips/substitutions are zero;
- the checkpoint payload step, dimensions, slot count, byte size, and SHA-256
  match its bundle manifest;
- the target step is greater than the verified source step;
- the preceding leg's final checkpoint includes optimizer, RNG, and curriculum
  sampler state (the old phase0 artifacts predate this and therefore start the
  first phase0b leg with fresh AdamW state).

## Phase 0b: baseline alignment

- 64D: protect the completed step-400,000 phase0b artifact.
- 128D: the original exact-v2 200k-to-250k leg and its 250k-to-260k repair are
  unpromoted. The bounded exact-v3 250k-to-252k pilot below also failed its
  continuation gate. There is no active GPU continuation path; the next rung
  is the local exact-v4 full-field/readback redesign.
- Historical exact-v2 data lives under
  `datasets/recovered/phase0b_exact_curriculum_v2`. It remains source evidence,
  not the active continuation curriculum.
- The legacy v1 JSONLs are source material only. They must not be consumed
  directly because most runtime targets exceed the 64-character active view;
  the v2 build preserves selected targets as complete, source-isolated chains
  and records context masking explicitly.
- Soul pair/probe families are excluded. This phase makes no private-memory or
  true non-response-region-write claim; the three-region bridge emits
  supervised characters through `response_draft`.

Promotion requires exact copy retention, no collapse tripwire, and no more
than 0.02 absolute regression in the width's partial/blank held-out metrics.

### Quarantined 128D repair branch

The exact-v2 200k to 250k leg is a valid continuation artifact but is not a
promoted baseline: its final blank-mode character accuracy regressed from
0.367 at the first exact-v2 evaluation to 0.248. Do not launch the normal 300k
leg from it.

A bounded repair may resume the complete 250k optimizer/RNG/sampler payload to
260k with `10% copy / 25% partial / 65% blank` and a 0.25 visible partial
fraction. It remains quarantined unless the machine-readable gate in
`kaggle/repair_gate_128d_250000_260000.json` passes. Failure at 260k stops this
branch; it must not be relabeled as the normal Phase0b leg.

Result on 2026-07-16: kernel version 5 reached step 260,000 with valid final
optimizer, Python/NumPy/Torch/CUDA RNG, and complete sampler state, but failed
the promotion gate. Copy exact was 0.90625, partial suffix exact was 0.203125,
blank character accuracy was 0.234492, and blank top fraction was 0.850858.
Final checkpoint SHA-256 is
`A444499596B293AD9CA16EDF1FD4E7F6B633F3A810276DA96A82734A828426DF`.
It remains quarantined under
`dist/kaggle_output_128D_repair_260k_v5`; launch no later Phase0b leg from the
250k or 260k candidates without a new explicit curriculum resolution.

### Exact-v3 128D pilot

Kimi's post-failure audit found exact-v2's arbitrary mid-word chains, runtime
source concentration, structured duplicates/leakage, biased evaluator, and
resume-LR override bug were launch-blocking. The redesigned curriculum is
`datasets/recovered/phase0b_exact_curriculum_v3`, with lossless boundary-aware
chunks, a 24-row complete-source cap, structured cleanup/deduplication, and
per-example `allowed_draft_modes`. Its manifest SHA-256 is
`B3F46D03F0E0765069D7192F9F1234333F7444CC796B98313BDB76130BB4D719`.

The quarantined pilot resumes the valid-but-unpromoted step-250,000 checkpoint
`C1A58781FA2F6C0600FB962E562B434AA1B8F1491BD558B30EF804010D9A162A`
and targets step 252,000. It restores AdamW moments, overrides effective LR to
`2e-4`, explicitly resets only the incompatible exact-v2 sampler, and samples
runtime/structured/scratch without replacement at `0.30/0.40/0.30`. Draft
weights are `0.25/0.35/0.40`, partial fraction is `0.50`, evaluation is every
250 steps on the fixed 64-case suite
`7653FA3DFC025B8EB28DE2826D5B73FFE5B09939E8278BC8151B51137F35782B`,
and full-state checkpoints are every 500 steps.

Kernel version 6 was launched at
`axongliksbot/axon-phase0b-128d-exact-v3-pilot`. The uploaded bundle is
2,296,825,334 bytes with SHA-256
`C25B5DA3A212D097B35683129D37C03EB42E5B863B5F755A6760B0595400DF6F`.
Passing authorizes only another bounded exact-v3 leg; this pilot is never
directly promotable.

Result on 2026-07-18: the version-6 pilot completed with valid checkpoint,
optimizer, Python/NumPy/Torch/CUDA RNG, sampler, provenance, and bundle
contracts, but `continuable=false`. Blank character accuracy was
`0.241130092 < 0.27`, predicted-top fraction was
`0.752956636 > 0.75`, and predicted unique characters were `23 < 30`.
The result remains quarantined under
`dist/kaggle_output_128D_exact_v3_pilot_252k_v6`; no later leg is authorized
from the 252k output.

## Phase 1A: bounded delta competence

Generated under `datasets/recovered/phase1a_curriculum_v2`:

| Family | Train | Dev | Test |
| --- | ---: | ---: | ---: |
| `runtime_response_chain_v2` | 8,000 | 1,000 | 1,000 |
| `structured_knowledge_response_v2` | 8,000 | 1,000 | 1,000 |
| `procedure_next_step_response_v2` | 8,000 | 1,000 | 1,000 |

Sources are split before exact 64-character chaining. Context outside the
bootstrap window is explicitly masked with visible offsets; it is not silently
discarded. Run one balanced 24,000-step epoch per width with draft sampling
`5% copy / 35% partial / 60% blank`, then compare against that width's frozen
phase0b baseline.

Local loader/loss smoke:

```powershell
python training/trainer_slot.py --mode phase1a --core-cfg 16,1,1,256 --device cpu `
  --phase1a-dir datasets/recovered/phase1a_curriculum_v2 --phase1a-max-examples 96 `
  --history-chars 256 --user-chars 64 --max-response-chars 64 `
  --charslot-mode-weights 0.05,0.35,0.60 --charslot-partial-frac 0.75 `
  --steps 20 --smoke --smoke-steps 20 --eval-every 20 --eval-n 12
```

## Phase 1B: quarantined private-state write/read proof

The local mechanics are now resolved without changing inherited checkpoint
tensors. The authoritative private state is the saved 168-row layout
(`128 hot / 32 warm / 8 cold`). A separate differentiable writer may change
only the hot rows; warm and cold rows are exact copies. The inherited core
writer remains frozen.

The two-tick trainer keeps the graph from a fact-visible Tick A write through a
fact-free Tick B recall. Its strict checkpoint schema permits only the external
writer plus the architecture-derived soul read path (`soul_ingest` and
per-layer soul cross-attention/norm/gate parameters) to train. Every other
inherited tensor and buffer is source-byte-locked, and load reconstructs the
exact `requires_grad` and AdamW mapping.

The immutable correct/zero/swapped/shuffled evaluator lives in
`training/soul_load_bearing.py` and `training/soul_core_adapter.py`. Promotion
still requires mean `CE_zero - CE_correct >= 0.5` and
`CE_swapped - CE_correct >= 0.3`, paired confidence intervals above zero,
donor-following behavior, locked abstention under zero/shuffle, and the
expected fact absent from Tick B. Occupancy alone is never proof.

The production local curriculum is built from exact-grounded facts, relations,
and procedures in `D:\00\axon_semantic_memory.db`, with mandatory support in
`D:\00\axon_episodic_memory.db`. It is local-only and not cloud-exportable.
Old-memory residual tables are quarantined, and the personal log is
manual-only. No Kaggle writer pilot is authorized until its artifact audit,
privacy decision, and the full-field writable readback gate all pass.

## Phase 1C: retention interleave

After Phase 1B is proven, train each width with 75% Phase1A delta examples and
25% proven two-tick private-state pairs. Both delta retention and
counterfactual memory gates must pass.

## Phase 2: canonical field migration (local contract accepted, launch blocked)

The table accepted additive local implementation in
`docs/roundtable/RESOLUTION_full-field-multitick-soul-2026-07-18.md`. The new
ten-region canonical contract preserves exact characters, spans, provenance,
typed deltas, masking, and runtime-owned validation/commit while projecting
through the inherited three physical role IDs.

The inherited output is now an exact four-by-64 transaction: segment zero
replaces the current scratch/draft and the next three segments append. Every
target character is supervised exactly once, and aggregate reconstruction is
gated at 64/128/192/256-character boundaries. One immutable read cursor pages
sealed context plus user input across scratch-to-response proposal switches;
each tick records cursor before/after, view hash, reference hash/counts, cycle
state, and tag-aware proposal-tail offset. Teacher-forced and free-running
paths use the same paging engine, while free-running receives no gold content,
length, or target hash.

This exact-v4 rung is deliberately `launch_eligible=false`: four output ticks
do not yet prove post-commit readback of every character in a completed
256-character writable scratch/draft. A later review/readback transaction must
close that gate before a GPU launch or checkpoint promotion.
