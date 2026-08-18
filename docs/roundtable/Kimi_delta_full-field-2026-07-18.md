# Kimi delta: full-field, multi-tick, and load-bearing soul

Review mode: read-only adversarial review with Kimi Code `0.22.1`.

Scope: the accepted
`RESOLUTION_full-field-multitick-soul-2026-07-18.md`, current 64D/128D
checkpoint path, shared-field runtime, curriculum, and soul evaluation.

Kimi's central verdict was correct: do not launch another GPU pilot while the
only executable path is the legacy one-tick three-region trainer. A legitimate
pilot must prove a canonical field transaction, free-running refinement, and
causal soul use on an actual core.

## Blockers Kimi identified

1. No canonical typed shared-field object or real logical regions.
2. No typed delta validator, atomic commit, or deterministic replay.
3. The legacy trainer/evaluator is one-tick and seeds drafts from the gold
   answer, so it cannot prove free-running refinement.
4. The generic soul evaluator had no adapter to a real `AxonCore`.
5. Soul thresholds were weaker than the table resolution.
6. The 128-row core versus 168-row manager contract and discarded/no-grad
   writers make soul-writing claims invalid.
7. Checkpoint inheritance needed strict tensor and provenance proof.
8. The legacy runtime selects checkpoints permissively and is not a safe pilot
   launcher for the new contract.
9. The deleted wide-slot `training/cf_probe.py` path remained misleading.

## Local disposition

- Items 1-2: implemented additively under `runtime/field/`.
- Item 3: free-running commit/rematerialize/refine exists in
  `runtime/multi_tick_refiner.py`; the legacy evaluator remains a control and
  is not accepted as multi-tick evidence.
- Items 4-5: implemented in `training/soul_core_adapter.py` and
  `training/soul_load_bearing.py` with per-condition locked gaps.
- Item 6: still blocks writer promotion. Current work measures read-bearing
  causality only; no occupancy statistic is accepted as proof.
- Item 7: requires a no-tensor-change inheritance manifest before a pilot.
- Item 8: the new path will use explicit checkpoint manifests; the permissive
  legacy runtime is not the launch path.
- Item 9: must be quarantined or replaced by a clear current-path entry point
  before promotion tooling is considered complete.

## Kimi-required pre-pilot evidence

- exact 95-character substrate round-trip,
- canonical region/hash/provenance/omission tests,
- stale/overlap/sealed delta rejection and identical replay hash,
- a real-core free-running second tick using only its own committed draft,
- correct/zero/swapped/shuffled real-core soul evaluation,
- intended-gradient and named-optimizer coverage,
- strict checkpoint tensor inheritance and standard-suite regression,
- fail-closed runtime checkpoint and soul-layout validation.

This review authorizes no launch or promotion. Its purpose is to keep the next
pilot from measuring the old one-tick shortcut under a new name.
