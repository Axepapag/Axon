# Axon Engineer's Ledger — Rolling Summary

Updated: 2026-08-17T21:12:05-05:00
Current through event: `evt-20260818T021205879052Z-codex-machine-fortification`
Historical authority: `roundtable/ENGINEERS_LEDGER_CANONICAL.jsonl`
Protocol: `roundtable/ENGINEERS_LEDGER_PROTOCOL.md`

## Current mission

Build Axon as a stateful, always-on AI around an exact 16D character field,
private per-core souls, auditable dormant knowledge, and validated deltas.
The immediate project need is behavioral proof and continuity, not a claim of
general intelligence.

## Current verified state

- The primary durable runtime is `runtime/axon_runtime/`.
- Its configured ring is `axon64-a`, `axon128-a`, `axon64-b`, `axon128-b`,
  backed by pinned 64D step-416000 and 128D step-252000 exact-v4 checkpoints.
- A read-only status check on 2026-08-17 reported generation 4, tick sequence
  4, 38 journal records, a matching four-core ring, no pending ingress, and no
  queued tool effects.
- The complete default pytest suite passed on 2026-08-17: 1,079 tests were
  collected, all executed tests passed, and one was skipped.
- Recorded conversational evidence is weak outside seed prompts: 10/10 seed
  exact matches, garbled held-out responses despite 10/10 being nonempty, and
  1/3 exact multi-turn responses.
- Typed tools and advisors are disabled in the primary CPU-smoke descriptor.
- The Bible/council-mix run finished normally at step 461,500. Across 24
  held-out evaluations, exact match moved from 0.24 to 0.29 and character
  accuracy from about 0.415 to 0.422; many continuation samples still
  collapsed to one character plus blanks.
- Kimmie's ten-core `runtime/council` experiment was reachable locally but
  stopped at tick 3,818. It preserved ten private souls and produced one
  correct memorized response followed by corrupted multi-turn output; most
  latest deltas were garbled.
- Current host: Columbia, Missouri; Windows 10; AMD FX-8350 (4 physical/8
  logical); 15.97 GB RAM; GTX 1650 with 4 GB VRAM and CUDA capability 7.5.
  Local CUDA works and is suitable for inference and bounded 64D smokes.
- The complete modern source tree is preserved on branch
  `agent/fortify-axon` at commit `181749dc1eecb18f4709c85bcd3601ca9d2f3273`.

## Binding continuity rules

- `docs/SOURCE_OF_TRUTH.md` governs architecture.
- `docs/WORKING_CONTRACT.md` governs collaborator conduct.
- Read and follow `roundtable/ENGINEERS_LEDGER_PROTOCOL.md` every turn.
- Every project turn appends one canonical event and refreshes this summary.
- Exact text and journal truth remain canonical; neural state is not factual
  authority.
- Never delete protected material or silently change doctrine.

## Recent completed work

- A clean-slate audit mapped architecture, repository state, runtime status,
  behavioral evidence, tests, artifact footprint, and security boundaries.
- The engineer's ledger system was established with an agent entry-point,
  protocol, rolling summary, and append-only canonical JSONL history.
- ChatGPT's June `D:\ChatGPT_State` continuity was reconciled to current
  `D:\Axon` without erasing the historical axon7/W1 handoffs. A current-state
  layer and v2 machine-readable manifest now carry the July durable-runtime
  work and August council/training evidence forward.
- Axon is now installable as an editable Python project with declared council
  dependencies; secret/artifact ignore rules were hardened; compilation,
  council in-process checks, and the full pytest suite passed.

## Active risks and blockers

- The modern tree is locally committed, but no Git remote is configured and
  GitHub CLI is absent, so it is not yet protected by an off-machine copy.
- Model behavior does not yet demonstrate useful held-out conversation or
  general reasoning.
- The newest training run improved held-out metrics only modestly, while the
  council often amplifies correlated character-level errors.
- `dist/`, `runs/`, and datasets occupy roughly 125 GB and need explicit
  artifact retention and checkpoint-lineage policy.
- Several overlapping execution paths remain: `runtime/axon_runtime/`,
  `runtime/council/`, `runtime/tick_loop.py`, and `runtime/table/`.
- `runtime/table/waker.py` uses string-prefix path containment and can accept a
  sibling path sharing the `docs/roundtable` prefix.
- Multiple legacy and training paths deserialize PyTorch files with
  `weights_only=False`; only the primary pinned-checkpoint path consistently
  verifies the artifact hash first.

## Next recommended actions

1. Install/authenticate GitHub CLI, configure the intended private GitHub
   remote, push `agent/fortify-axon`, and verify a clean clone.
2. Freeze one held-out behavioral suite, then compare step 440,500, step
   461,500, exact-v4 candidates, and single-core versus council inference.
3. Run correct/zero/swapped/shuffled soul ablations and council-vs-single-core
   causal comparisons before another long training run.
4. Diagnose blank-draft alignment, target truncation, data mixture, and
   collapse with bounded smokes.
5. Reconcile runtime branches only after behavioral evidence, then address
   path containment, deserialization, and artifact retention.

## Fast orientation

- Runtime description: `docs/AXON_RUNTIME.md`
- Architecture doctrine: `docs/SOURCE_OF_TRUTH.md`
- Working rules: `docs/WORKING_CONTRACT.md`
- Primary runtime config: `ops/axon_runtime.cpu-smoke.json`
- Behavioral evidence: `conversational_runtime_prototype_results.json`
- Full test command: `python -m pytest -q -p no:cacheprovider`
- Read-only runtime status: `python -m runtime.axon_runtime status`
