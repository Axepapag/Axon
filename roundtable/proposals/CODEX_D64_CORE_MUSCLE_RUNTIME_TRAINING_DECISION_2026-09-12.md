# D64 Core Muscle and Runtime Training Decision — 2026-09-12

## Decision

Keep the physical D64 rail, the exact 4 x D16 character packing, and the one-head Candidate-A geometry fixed for the next experiment. Do not enlarge the residual stream or replace the frozen character bank before evidence shows that D64 itself is the limiting factor. Add depth candidates while retaining substantial FFN width, and judge them on causal free-running behavior in the real runtime loop.

Candidate A is already unusually wide: 2 layers with a 131,072-unit FFN and 33,982,137 trainable parameters. More FFN width alone is unlikely to be the most efficient next move because every FFN result is projected back into D64. Depth gives the core additional sequential attention and nonlinear transformations before it must emit a delta. The first comparison should therefore include a parameter-matched deeper core and two larger deeper cores.

## What exact packing changes

A compiled D64 rail row contains four exact D16 character cells. This removes subword-token ambiguity at the substrate boundary: the character and its canonical field address are exact, and the same text recompiles deterministically. Words, sentences, and paragraphs remain ordered sequences of these rows; they are not each compressed into one exact semantic vector.

The learned reader still performs continuous attention and nonlinear transformations over D64 features. That part is intentionally learned and therefore approximate. Exact input identity and canonical addressability make this a different training substrate from ordinary tokenizer-driven language models, but they do not eliminate the D64 communication bottleneck inside the transformer. Soul state, attention outputs, residual updates, and layer-to-layer features still pass through 64 values per position/state token.

## Measured candidates

The counts below were measured by instantiating `LivingReasoningCoreD64` from each exact configuration. The Adam figure is a lower-bound accounting of FP32 parameters, gradients, and two optimizer moments (16 bytes per parameter); it excludes activations, allocator overhead, serialized checkpoints, and any mixed-precision master copies.

| Candidate | Layers | FFN | Parameters | FP32 parameters | Adam training floor | Architecture ID |
|---|---:|---:|---:|---:|---:|---|
| Candidate A baseline | 2 | 131,072 | 33,982,137 | 129.63 MiB | 518.53 MiB | `living-d64-675b5ec0f0053cd54c0fbda6` |
| F parameter-matched depth | 4 | 65,536 | 34,016,057 | 129.76 MiB | 519.04 MiB | `living-d64-901a7cc6136e5fe74c898fb9` |
| G full FFN + depth | 4 | 131,072 | 67,832,633 | 258.76 MiB | 1,035.04 MiB | `living-d64-7ff4a3194cfa54f4f9a1dcdc` |
| H deeper parameter-rich | 6 | 65,536 | 50,958,265 | 194.39 MiB | 777.56 MiB | `living-d64-ef74205782fdfbbe5411dd65` |

The parameter-matched A-versus-F comparison is the cleanest first answer to “width or depth?” G and H should proceed only if local and Kaggle peak-memory probes pass with the real optimizer, objective, Soul codec, and response length.

## Required tournament evidence

Every candidate must use the same frozen bank, D64 compiler, canonical views, assignments, holdouts, work-unit policy, optimizer policy, and gates. Report at least:

- free-running exact assignment success, with no teacher target prefix available to the emission path;
- held-out character loss and accuracy as diagnostic evidence, not as conversational competence;
- rail-content, rail-presence, and persisted-Soul counterfactual deltas;
- delayed Soul benefit across later attempts and after a stop/resume boundary;
- degeneration indicators: constant-character loops, EOS failure, repetition, non-finite metrics, zero gradients, and no parameter movement;
- examples per second, peak accelerator memory, checkpoint bytes, and checkpoint transfer time.

A candidate wins only by the governed gate surface. Parameter count or training loss alone cannot authorize replacement of Candidate A.

## Runtime status after this change

The conformance worker now separates teacher-forced supervision from its free-running rail emission. The gate authenticates and recomputes from the exact Soul payload inhaled by the worker. When Heart is present, each committed response advances the assignment to a newly compiled view of the successor field, so the next attempt can attend its own canonical history. An injected `CanonicalStateBranch` persists that successor as durable HEAD.

The production session still trains `CompleteField64D`, not `LivingReasoningCoreD64`. Rejected attempts now have a hash-verified rolling-three model/optimizer recovery workspace and a restart regression test. Kaggle must not launch until the living-core adapter replaces the conformance motor, the multi-store attempt transaction has prepare/finalize/recover semantics, and the live worker transport passes an interruption/replay smoke test. These are correctness gates, not optional refinements.
